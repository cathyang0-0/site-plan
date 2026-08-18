"""
SitePlan — the Eto options dialog (Rhino 8, Mac + Windows).

One window, three pages swapped in place:
  options page   map (WebView on siteplan_map.html) + native controls
  progress page  per-stage status while the job runs, with Cancel
  preview page   the site aerial with detected tree crowns as live circles
                 (WebView on /api/jobs/{id}/preview, served by the backend)
                 + the size/variance sliders — moved here from the options
                 page so they adjust REAL detected trees in real time.
                 Detection runs once with neutral sizes; the sliders are an
                 export-time transform (backend rescale_placements), so
                 Import renders exactly what the preview shows.

Division of labor (deliberate — keep it when editing):
  siteplan_map.html   everything map: tiles, drawing, address search.
                      Bridge = polling: a UITimer here reads
                      window.getState() (seq-numbered JSON) twice a second;
                      DocumentTitleChanged is NOT used (flaky on Mac Eto).
  siteplan_form.py    all logic that doesn't touch a widget (request
                      assembly, parsing, estimates) — headless-tested.
  this file           widgets, threads, and nothing else.

Threading rules (violating these freezes or crashes Rhino):
  - network I/O ONLY on the worker thread (submit/poll/export),
  - widget updates ONLY on the UI thread, marshalled via
    forms.Application.Instance.Invoke(System.Action(fn)),
  - the DXF import into the document happens NOT here but in
    SitePlan_command.py, after ShowModal returns, on Rhino's main thread.

Rhino 8 CPython Eto gotchas baked into this file (each cost a real run):
  - super().__init__() must be the FIRST line of an Eto subclass __init__
    (Python.NET skips the base ctor otherwise → NullReferenceException).
  - Constructor property-kwargs (forms.Label(Text=...)) are NOT supported
    by this Rhino build's Python.NET → construct bare, set properties after
    (that's what _props() is for).
  - Parent ShowModal to RhinoEtoApp.MainWindowForDocument(sc.doc), never
    RhinoEtoApp.MainWindow (silently shows nothing on Mac).
"""
import json
import os
import sys
import tempfile
import threading
import time

import System
import Eto.Forms as forms
import Eto.Drawing as drawing
import Rhino.UI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import siteplan_form as form
import siteplan_client as spc

# Stage keys as the backend reports them (jobs.py), in pipeline order.
STAGES = ["imagery", "roads", "buildings", "water", "infrastructure",
          "contours", "trees", "land_types", "export"]
GLYPH = {"running": "…", "done": "✓", "skipped": "–", "failed": "✗"}

MAP_URL = System.Uri(
    "file://" + os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "siteplan_map.html"))


def _props(ctrl, **values):
    """Set properties post-construction (see gotcha note in the docstring)."""
    for name, value in values.items():
        setattr(ctrl, name, value)
    return ctrl


class SitePlanDialog(forms.Dialog[bool]):
    """Collects a request, runs the job, returns with .result set:
    {"path": dxf_path, "warnings": [...]} on success, None otherwise.
    .last_bbox always holds the last drawn bbox (for the caller's sticky)."""

    def __init__(self, initial_bbox, units="m", last_job=None):
        super().__init__()          # REQUIRED first — see module docstring
        self.Title = "Site Plan Drafter"
        self.Padding = drawing.Padding(8)
        self.Resizable = True
        self.MinimumSize = drawing.Size(1020, 660)

        self.result = None
        self.last_bbox = dict(initial_bbox) if initial_bbox else None
        self._initial_bbox = self.last_bbox
        self._units = units
        self._bbox = self.last_bbox
        self._area_km2 = form.bbox_area_km2(self._bbox) if self._bbox else 0.0
        self._map_seq = -1
        self._cancel = threading.Event()
        self._started = None          # time.time() when the job starts

        self._job_id = None           # set when a job completes with trees
        self._final_status = None     # its poll status (for warnings)
        self._request = None          # the submitted request body
        # {"job_id", "style"} of the newest tree-detected job. Fed back in
        # by the command (session sticky) so the preview can be reopened
        # WITHOUT regenerating — valid as long as the backend still holds
        # the job (its jobs live in memory for the server's lifetime).
        self.last_job = dict(last_job) if last_job else None

        self._build_options_page()
        self._build_progress_page()
        self._build_preview_page()
        self._root = _props(forms.Panel(), Content=self._options_page)
        self.Content = self._root
        # If the window is closed mid-run (Esc / red button), stop the client
        # wait too — otherwise the worker keeps polling a dead dialog.
        self.Closed += lambda s, e: self._cancel.set()

        # Poll the map's state twice a second (see bridge note in docstring).
        self._timer = forms.UITimer()
        self._timer.Interval = 0.5
        self._timer.Elapsed += self._poll_map
        self._timer.Start()

    # ------------------------------------------------------------- options UI
    def _build_options_page(self):
        self._webview = forms.WebView()
        self._webview.Size = drawing.Size(620, 520)
        self._webview.DocumentLoaded += self._on_map_loaded
        self._webview.Url = MAP_URL

        self._readout = _props(forms.Label(),
                               Text="Draw a box on the map to set the site.")
        self._estimate = _props(forms.Label(), Text="")
        guidance = _props(
            forms.Label(),
            Text=("Draw slightly larger than needed — edge conditions aren't "
                  "perfectly resolved.\nProcessing time grows steeply with area."),
            TextColor=drawing.Colors.Gray)

        left = _props(forms.DynamicLayout(), Spacing=drawing.Size(4, 4))
        left.Add(self._webview, True, True)
        left.Add(self._readout)
        left.Add(self._estimate)
        left.Add(guidance)

        # --- native controls (right column) ---
        self._trees_check = _props(forms.CheckBox(),
                                   Text="Detect trees (the slow stage)",
                                   Checked=True)
        self._trees_check.CheckedChanged += lambda s, e: self._refresh_estimate()

        self._engine = forms.DropDown()
        self._engine.DataStore = ["kmeans", "segmodel", "off"]
        self._engine.SelectedIndex = 0

        self._contours = _props(forms.TextBox(), Text="5ft")
        self._river_width = _props(forms.TextBox(),
                                   Text=str(form.DEFAULT_RIVER_WIDTH_M))

        self._grid = self._make_width_grid()
        # Expander.Header is typed Control, not str — C# converts implicitly
        # (string → Label), this Python.NET does not. Hand it a real Label.
        widths_expander = _props(forms.Expander(),
                                 Header=_props(forms.Label(),
                                               Text="Road widths by type (m)"),
                                 Expanded=False, Content=self._grid)

        right = _props(forms.DynamicLayout(), Spacing=drawing.Size(4, 6))
        right.Add(self._trees_check)
        right.Add(self._labeled("Land-cover engine", self._engine))
        right.Add(self._labeled('Contour interval ("5ft" or meters, 0 = off)',
                                self._contours))
        right.Add(self._labeled("River width (m)", self._river_width))
        right.Add(widths_expander)

        self._last_btn = _props(forms.Button(),
                                Text="Tree preview of last run…",
                                Enabled=bool(self.last_job))
        self._last_btn.Click += self._on_open_last
        right.Add(self._last_btn)
        right.Add(None, False, True)   # spring: push buttons to the bottom

        self._generate_btn = _props(forms.Button(), Text="Generate")
        self._generate_btn.Click += self._on_generate
        close_btn = _props(forms.Button(), Text="Close")
        close_btn.Click += lambda s, e: self.Close(False)
        buttons = _props(forms.DynamicLayout(), Spacing=drawing.Size(6, 0))
        buttons.BeginHorizontal()
        buttons.Add(None, True, False)
        buttons.Add(close_btn)
        buttons.Add(self._generate_btn)
        buttons.EndHorizontal()
        right.Add(buttons)

        # BorderType.None: "None" is a Python keyword and this Python.NET
        # exposes it neither as .None nor via getattr — parse the CLR name.
        right_scroll = _props(forms.Scrollable(), Content=right,
                              Border=System.Enum.Parse(forms.BorderType, "None"))
        right_scroll.Size = drawing.Size(360, -1)

        page = _props(forms.DynamicLayout(), Spacing=drawing.Size(10, 6))
        page.BeginHorizontal()
        page.Add(left, True, True)
        page.Add(right_scroll)
        page.EndHorizontal()
        self._options_page = page
        self._refresh_estimate()

    def _make_slider(self, title, lo, hi, start, text_fn):
        """Eto sliders are int-only; values are tenths (÷10 on read)."""
        slider = _props(forms.Slider(), MinValue=lo, MaxValue=hi, Value=start)
        label = _props(forms.Label(), Text=text_fn(start))
        slider.ValueChanged += (
            lambda s, e: setattr(label, "Text", text_fn(slider.Value)))
        row = _props(forms.DynamicLayout(), Spacing=drawing.Size(4, 0))
        row.Add(_props(forms.Label(), Text=title))
        row.BeginHorizontal()
        row.Add(slider, True, False)
        row.Add(label)
        row.EndHorizontal()
        return slider, row

    def _crown_label_text(self, v):
        return "%.1f×" % (v / 10.0)

    def _variance_label_text(self, v):
        return {0: "uniform"}.get(v, "%.1f× natural" % (v / 10.0))

    def _labeled(self, text, control):
        row = _props(forms.DynamicLayout(), Spacing=drawing.Size(4, 2))
        row.Add(_props(forms.Label(), Text=text))
        row.Add(control)
        return row

    def _make_width_grid(self):
        """Editable class→width grid. GridItem.Values round-trips edits
        without any binding machinery — the one Eto grid pattern that works
        the same on Mac and Windows Rhino."""
        grid = _props(forms.GridView(), ShowHeader=True)
        grid.Size = drawing.Size(-1, 240)
        grid.Columns.Add(_props(forms.GridColumn(), HeaderText="Class",
                                DataCell=forms.TextBoxCell(0),
                                Editable=False, Width=150))
        grid.Columns.Add(_props(forms.GridColumn(), HeaderText="Width (m)",
                                DataCell=forms.TextBoxCell(1),
                                Editable=True, Width=90))
        items = []
        for cls, width in form.ROAD_CLASS_DEFAULTS:
            items.append(_props(forms.GridItem(), Values=[cls, str(width)]))
        grid.DataStore = items
        return grid

    # ------------------------------------------------------------ progress UI
    def _build_progress_page(self):
        self._stage_labels = {}
        rows = _props(forms.DynamicLayout(), Spacing=drawing.Size(6, 4))
        rows.Add(_props(forms.Label(), Text="Generating site plan…",
                        Font=drawing.SystemFonts.Bold()))
        for stage in STAGES:
            label = _props(forms.Label(), Text="    " + stage)
            self._stage_labels[stage] = label
            rows.Add(label)
        self._elapsed = _props(forms.Label(), Text="")
        self._error = _props(forms.Label(), Text="",
                             TextColor=drawing.Colors.Firebrick,
                             Wrap=forms.WrapMode.Word)

        bar = _props(forms.ProgressBar(), Indeterminate=True)

        self._cancel_btn = _props(forms.Button(), Text="Cancel")
        self._cancel_btn.Click += self._on_cancel
        self._back_btn = _props(forms.Button(), Text="Back", Visible=False)
        self._back_btn.Click += self._on_back

        rows.Add(bar)
        rows.Add(self._elapsed)
        rows.Add(self._error, True, True)
        buttons = _props(forms.DynamicLayout(), Spacing=drawing.Size(6, 0))
        buttons.BeginHorizontal()
        buttons.Add(None, True, False)
        buttons.Add(self._back_btn)
        buttons.Add(self._cancel_btn)
        buttons.EndHorizontal()
        rows.Add(buttons)
        self._progress_page = rows

    # ------------------------------------------------------------ preview UI
    def _build_preview_page(self):
        self._preview_web = forms.WebView()
        self._preview_web.Size = drawing.Size(760, 520)
        self._preview_web.DocumentLoaded += (
            lambda s, e: self._push_preview_params())

        self._crown_slider, crown_row = self._make_slider(
            "Tree size (crown scale)", 5, 30, 15, self._crown_label_text)
        self._variance_slider, variance_row = self._make_slider(
            "Tree size variance", 0, 20, 10, self._variance_label_text)
        self._crown_slider.ValueChanged += (
            lambda s, e: self._push_preview_params())
        self._variance_slider.ValueChanged += (
            lambda s, e: self._push_preview_params())

        hint = _props(forms.Label(),
                      Text=("Circles are the detected trees on your site — "
                            "sliders re-render them instantly, and Import "
                            "draws exactly what you see."),
                      TextColor=drawing.Colors.Gray)

        self._import_btn = _props(forms.Button(), Text="Import into Rhino")
        self._import_btn.Click += self._on_import
        preview_back = _props(forms.Button(), Text="Back")
        preview_back.Click += self._on_back

        controls = _props(forms.DynamicLayout(), Spacing=drawing.Size(10, 4))
        controls.BeginHorizontal()
        controls.Add(crown_row, True, False)
        controls.Add(variance_row, True, False)
        controls.EndHorizontal()

        buttons = _props(forms.DynamicLayout(), Spacing=drawing.Size(6, 0))
        buttons.BeginHorizontal()
        buttons.Add(hint, True, False)
        buttons.Add(preview_back)
        buttons.Add(self._import_btn)
        buttons.EndHorizontal()

        page = _props(forms.DynamicLayout(), Spacing=drawing.Size(6, 6))
        page.Add(self._preview_web, True, True)
        page.Add(controls)
        page.Add(buttons)
        self._preview_page = page

    def _push_preview_params(self):
        try:
            self._preview_web.ExecuteScript(
                "setParams(%s, %s)" % (self._crown_slider.Value / 10.0,
                                       self._variance_slider.Value / 10.0))
        except Exception:
            pass   # page still loading; DocumentLoaded will push again

    def _show_preview(self, job_id, status):
        """UI thread. Job finished with trees — open the live preview."""
        self._job_id = job_id
        self._final_status = status
        self._import_btn.Enabled = True
        self._last_btn.Enabled = True   # Back → options can come back here
        self._root.Content = self._preview_page
        self._preview_web.Url = System.Uri(
            "%s/api/jobs/%s/preview" % (spc.DEFAULT_BASE, job_id))

    def _on_open_last(self, sender, e):
        """Reopen the remembered job's preview — no regeneration. Verifies
        the backend still holds the job first (network → worker thread)."""
        if not self.last_job:
            return
        self._last_btn.Enabled = False
        job = dict(self.last_job)

        def check():
            try:
                # Immediate return for a complete job; a short timeout turns
                # "backend restarted / job gone" into a clean error instead
                # of a hang.
                status = spc.poll_job(job["job_id"], timeout=10)
            except spc.SitePlanError as exc:
                message = str(exc)

                def fail():
                    self._last_btn.Enabled = True
                    forms.MessageBox.Show(
                        self, "The last run isn't available anymore "
                        "(backend restarted?). Generate a new plan.\n\n"
                        + message, "SitePlan")
                self._invoke(fail)
                return

            def ok():
                self._timer.Stop()
                self._request = {"style": dict(job["style"])}
                self._show_preview(job["job_id"], status)
            self._invoke(ok)

        threading.Thread(target=check, daemon=True).start()

    def _on_import(self, sender, e):
        """Export with the slider values (seconds — restyle, no re-detect),
        then close; SitePlan_command imports the file on the main thread."""
        self._import_btn.Enabled = False
        style = form.export_style(self._request,
                                  self._crown_slider.Value / 10.0,
                                  self._variance_slider.Value / 10.0)
        threading.Thread(target=self._export_worker, args=(style,),
                         daemon=True).start()

    def _export_worker(self, style):
        try:
            out = os.path.join(tempfile.mkdtemp(prefix="siteplan_"),
                               "site-plan.dxf")
            path = spc.export_dxf(self._job_id, out, style=style)
            self.result = {"path": path,
                           "warnings": (self._final_status or {}).get("warnings")
                           or []}
            self._invoke(lambda: self.Close(True))
        except spc.SitePlanError as exc:
            message = str(exc)

            def report():
                self._import_btn.Enabled = True
                forms.MessageBox.Show(self, message, "SitePlan")
            self._invoke(report)

    # ---------------------------------------------------------- map bridge
    def _on_map_loaded(self, sender, e):
        if self._initial_bbox:
            try:
                self._webview.ExecuteScript(
                    "initFromPython(%s)" % json.dumps(self._initial_bbox))
            except Exception:
                pass   # page not ready; the user can still draw manually

    def _poll_map(self, sender, e):
        try:
            raw = self._webview.ExecuteScript(
                "return window.getState ? window.getState() : ''")
            state = json.loads(raw) if raw else None
        except Exception:
            return    # page still loading, or a transient bridge hiccup
        if not state or not state.get("bbox"):
            return
        if state.get("seq") == self._map_seq:
            return
        self._map_seq = state["seq"]
        self._bbox = state["bbox"]
        self.last_bbox = self._bbox
        self._area_km2 = state.get("area_km2") or form.bbox_area_km2(self._bbox)
        b = self._bbox
        self._readout.Text = "Site: %.4f, %.4f  →  %.4f, %.4f" % (
            b["west"], b["south"], b["east"], b["north"])
        self._refresh_estimate()

    def _refresh_estimate(self):
        if not self._bbox:
            self._estimate.Text = ""
            return
        trees_on = self._trees_check.Checked is True
        minutes = form.estimate_minutes(self._area_km2, trees_on)
        text = "%.2f km²  ·  rough estimate ~%d min" % (self._area_km2,
                                                        round(minutes))
        if self._area_km2 > 4.0:
            text += "   (large site — slow, and may exceed the imagery cap)"
        self._estimate.Text = text

    # ---------------------------------------------------------- generate flow
    def _read_grid_widths(self):
        """{class: width} from the grid; raises ValueError naming a bad row."""
        widths = {}
        for item in self._grid.DataStore:
            cls, raw = item.Values[0], str(item.Values[1]).strip()
            try:
                value = float(raw)
            except ValueError:
                raise ValueError('road width for "%s" is not a number: %r'
                                 % (cls, raw))
            if value <= 0:
                raise ValueError('road width for "%s" must be > 0' % cls)
            widths[str(cls)] = value
        return widths

    def _on_generate(self, sender, e):
        if not self._bbox:
            forms.MessageBox.Show(self, "Draw a bounding box on the map first.",
                                  "SitePlan")
            return
        try:
            road_widths = self._read_grid_widths()
            river = float(self._river_width.Text)
            if river <= 0:
                raise ValueError("river width must be > 0")
        except ValueError as exc:
            forms.MessageBox.Show(self, str(exc), "SitePlan")
            return

        engine = str(self._engine.SelectedValue)
        # No tree sizes here: detection runs NEUTRAL, sizes are chosen on
        # the preview page and applied at export (see module docstring).
        request = form.build_request(
            self._bbox,
            trees=self._trees_check.Checked is True,
            land_engine=engine,
            contour_interval_m=form.parse_interval(self._contours.Text),
            road_class_widths=road_widths,
            river_width_m=river,
            units=self._units,
        )
        self._request = request

        self._timer.Stop()
        self._cancel.clear()
        self._started = time.time()
        self._error.Text = ""
        self._back_btn.Visible = False
        self._cancel_btn.Enabled = True
        for stage in STAGES:
            self._stage_labels[stage].Text = "    %s" % stage
        self._root.Content = self._progress_page

        worker = threading.Thread(target=self._worker, args=(request,),
                                  daemon=True)
        worker.start()

    # ------------------------------------------------------------- job thread
    def _invoke(self, fn):
        forms.Application.Instance.Invoke(System.Action(fn))

    def _worker(self, request):
        """Runs on a daemon thread: submit → poll, then either the tree
        preview (trees on) or straight export → close (trees off).
        Touches widgets only through _invoke."""
        try:
            job_id = spc.submit_job(request["bbox"], layers=request["layers"],
                                    options=request["options"],
                                    style=request["style"])
            status = spc.poll_job(job_id, on_progress=self._on_job_progress)
            if "trees" in request["layers"]:
                self.last_job = {"job_id": job_id,
                                 "style": request["style"]}
                self._invoke(lambda: self._show_preview(job_id, status))
                return
            out = os.path.join(tempfile.mkdtemp(prefix="siteplan_"),
                               "site-plan.dxf")
            path = spc.export_dxf(job_id, out)
            self.result = {"path": path,
                           "warnings": status.get("warnings") or []}
            self._invoke(lambda: self.Close(True))
        except spc.SitePlanError as exc:
            message = str(exc)
            self._invoke(lambda: self._show_error(message))

    def _on_job_progress(self, status):
        """poll_job callback (worker thread). Returning False cancels the
        client wait — the backend job finishes on its own, same as ever."""
        stages = dict(status.get("progress") or {})
        elapsed = time.time() - self._started

        def update():
            for stage, state in stages.items():
                if stage in self._stage_labels:
                    self._stage_labels[stage].Text = "    %s  %s" % (
                        stage, GLYPH.get(state, state))
            self._elapsed.Text = "elapsed %d:%02d" % divmod(int(elapsed), 60)
        self._invoke(update)
        return not self._cancel.is_set()

    def _show_error(self, message):
        self._error.Text = message
        self._cancel_btn.Enabled = False
        self._back_btn.Visible = True

    def _on_cancel(self, sender, e):
        self._cancel.set()
        self._cancel_btn.Enabled = False
        self._error.Text = ("Cancelling… (the backend job keeps running; "
                            "results are cached for a re-run)")

    def _on_back(self, sender, e):
        self._root.Content = self._options_page
        self._timer.Start()


def show(initial_bbox, units="m", last_job=None):
    """Open the dialog modally.
    Returns (result_dict_or_None, last_bbox, last_job) — the caller keeps
    last_bbox and last_job in sc.sticky so the next run remembers them."""
    import scriptcontext as sc
    dlg = SitePlanDialog(initial_bbox, units=units, last_job=last_job)
    # Per developer.rhino3d.com/guides/eto/rhino-specific: RhinoEtoApp
    # .MainWindow "will not work correctly on Mac" — parent to the document
    # window instead (and pass the script's sc.doc, not RhinoDoc.ActiveDoc).
    parent = Rhino.UI.RhinoEtoApp.MainWindowForDocument(sc.doc)
    ok = dlg.ShowModal(parent)
    dlg._timer.Stop()
    return (dlg.result if ok else None), dlg.last_bbox, dlg.last_job
