"""
SitePlan — the Eto options dialog (Rhino 8, Mac + Windows).

One window, two pages swapped in place:
  options page   map (WebView on siteplan_map.html) + native controls
  progress page  per-stage status while the job runs, with Cancel

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


class SitePlanDialog(forms.Dialog[bool]):
    """Collects a request, runs the job, returns with .result set:
    {"path": dxf_path, "warnings": [...]} on success, None otherwise.
    .last_bbox always holds the last drawn bbox (for the caller's sticky)."""

    def __init__(self, initial_bbox, units="m"):
        # REQUIRED first line: Rhino 8's Python.NET only auto-runs the Eto
        # base constructor when __init__ matches a .NET ctor signature; with
        # our extra args it doesn't, the platform handler stays null, and the
        # first property set (Title) throws NullReferenceException.
        super().__init__()
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

        self._build_options_page()
        self._build_progress_page()
        self._root = forms.Panel(Content=self._options_page)
        self.Content = self._root

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

        self._readout = forms.Label(Text="Draw a box on the map to set the site.")
        self._estimate = forms.Label(Text="")
        guidance = forms.Label(
            Text=("Draw slightly larger than needed — edge conditions aren't "
                  "perfectly resolved.\nProcessing time grows steeply with area."),
            TextColor=drawing.Colors.Gray)

        left = forms.DynamicLayout(Spacing=drawing.Size(4, 4))
        left.Add(self._webview, True, True)
        left.Add(self._readout)
        left.Add(self._estimate)
        left.Add(guidance)

        # --- native controls (right column) ---
        self._trees_check = forms.CheckBox(Text="Detect trees (the slow stage)",
                                           Checked=True)
        self._trees_check.CheckedChanged += lambda s, e: self._refresh_estimate()

        self._crown_slider, crown_row = self._make_slider(
            "Tree size (crown scale)", 5, 30, 15, self._crown_label_text)
        self._variance_slider, variance_row = self._make_slider(
            "Tree size variance", 0, 20, 10, self._variance_label_text)

        self._engine = forms.DropDown()
        self._engine.DataStore = ["kmeans", "segmodel", "off"]
        self._engine.SelectedIndex = 0

        self._contours = forms.TextBox(Text="5ft")
        self._river_width = forms.TextBox(Text=str(form.DEFAULT_RIVER_WIDTH_M))

        self._grid = self._make_width_grid()
        widths_expander = forms.Expander(Header="Road widths by type (m)",
                                         Expanded=False, Content=self._grid)

        right = forms.DynamicLayout(Spacing=drawing.Size(4, 6))
        right.Add(self._trees_check)
        right.Add(crown_row)
        right.Add(variance_row)
        right.Add(self._labeled("Land-cover engine", self._engine))
        right.Add(self._labeled('Contour interval ("5ft" or meters, 0 = off)',
                                self._contours))
        right.Add(self._labeled("River width (m)", self._river_width))
        right.Add(widths_expander)
        right.Add(None, False, True)   # spring: push buttons to the bottom

        self._generate_btn = forms.Button(Text="Generate")
        self._generate_btn.Click += self._on_generate
        close_btn = forms.Button(Text="Close")
        close_btn.Click += lambda s, e: self.Close(False)
        buttons = forms.DynamicLayout(Spacing=drawing.Size(6, 0))
        buttons.BeginHorizontal()
        buttons.Add(None, True, False)
        buttons.Add(close_btn)
        buttons.Add(self._generate_btn)
        buttons.EndHorizontal()
        right.Add(buttons)

        # "None" is a Python keyword, so the enum member needs getattr.
        right_scroll = forms.Scrollable(
            Content=right, Border=getattr(forms.BorderType, "None"))
        right_scroll.Size = drawing.Size(360, -1)

        page = forms.DynamicLayout(Spacing=drawing.Size(10, 6))
        page.BeginHorizontal()
        page.Add(left, True, True)
        page.Add(right_scroll)
        page.EndHorizontal()
        self._options_page = page
        self._refresh_estimate()

    def _make_slider(self, title, lo, hi, start, text_fn):
        """Eto sliders are int-only; values are tenths (÷10 on read)."""
        slider = forms.Slider(MinValue=lo, MaxValue=hi, Value=start)
        label = forms.Label(Text=text_fn(start))
        slider.ValueChanged += (
            lambda s, e: setattr(label, "Text", text_fn(slider.Value)))
        row = forms.DynamicLayout(Spacing=drawing.Size(4, 0))
        row.Add(forms.Label(Text=title))
        row.BeginHorizontal()
        row.Add(slider, True)
        row.Add(label)
        row.EndHorizontal()
        return slider, row

    def _crown_label_text(self, v):
        return "%.1f×" % (v / 10.0)

    def _variance_label_text(self, v):
        return {0: "uniform"}.get(v, "%.1f× natural" % (v / 10.0))

    def _labeled(self, text, control):
        row = forms.DynamicLayout(Spacing=drawing.Size(4, 2))
        row.Add(forms.Label(Text=text))
        row.Add(control)
        return row

    def _make_width_grid(self):
        """Editable class→width grid. GridItem.Values round-trips edits
        without any binding machinery — the one Eto grid pattern that works
        the same on Mac and Windows Rhino."""
        grid = forms.GridView(ShowHeader=True)
        grid.Size = drawing.Size(-1, 240)
        col_cls = forms.GridColumn(HeaderText="Class",
                                   DataCell=forms.TextBoxCell(0),
                                   Editable=False, Width=150)
        col_w = forms.GridColumn(HeaderText="Width (m)",
                                 DataCell=forms.TextBoxCell(1),
                                 Editable=True, Width=90)
        grid.Columns.Add(col_cls)
        grid.Columns.Add(col_w)
        grid.DataStore = [forms.GridItem(Values=(cls, str(w)))
                          for cls, w in form.ROAD_CLASS_DEFAULTS]
        return grid

    # ------------------------------------------------------------ progress UI
    def _build_progress_page(self):
        self._stage_labels = {}
        rows = forms.DynamicLayout(Spacing=drawing.Size(6, 4))
        rows.Add(forms.Label(Text="Generating site plan…",
                             Font=drawing.SystemFonts.Bold()))
        for stage in STAGES:
            label = forms.Label(Text="    " + stage)
            self._stage_labels[stage] = label
            rows.Add(label)
        self._elapsed = forms.Label(Text="")
        self._error = forms.Label(Text="", TextColor=drawing.Colors.Firebrick)
        self._error.Wrap = forms.WrapMode.Word

        bar = forms.ProgressBar(Indeterminate=True)

        self._cancel_btn = forms.Button(Text="Cancel")
        self._cancel_btn.Click += self._on_cancel
        self._back_btn = forms.Button(Text="Back")
        self._back_btn.Visible = False
        self._back_btn.Click += self._on_back

        rows.Add(bar)
        rows.Add(self._elapsed)
        rows.Add(self._error, True, True)
        buttons = forms.DynamicLayout(Spacing=drawing.Size(6, 0))
        buttons.BeginHorizontal()
        buttons.Add(None, True, False)
        buttons.Add(self._back_btn)
        buttons.Add(self._cancel_btn)
        buttons.EndHorizontal()
        rows.Add(buttons)
        self._progress_page = rows

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
        request = form.build_request(
            self._bbox,
            trees=self._trees_check.Checked is True,
            land_engine=engine,
            contour_interval_m=form.parse_interval(self._contours.Text),
            crown_size_scale=self._crown_slider.Value / 10.0,
            size_variance=self._variance_slider.Value / 10.0,
            road_class_widths=road_widths,
            river_width_m=river,
            units=self._units,
        )

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
        """Runs on a daemon thread: the whole submit → poll → export chain.
        Touches widgets only through _invoke."""
        try:
            job_id = spc.submit_job(request["bbox"], layers=request["layers"],
                                    options=request["options"],
                                    style=request["style"])
            status = spc.poll_job(job_id, on_progress=self._on_job_progress)
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


def show(initial_bbox, units="m"):
    """Open the dialog modally. Returns (result_dict_or_None, last_bbox)."""
    dlg = SitePlanDialog(initial_bbox, units=units)
    ok = dlg.ShowModal(Rhino.UI.RhinoEtoApp.MainWindow)
    dlg._timer.Stop()
    return (dlg.result if ok else None), dlg.last_bbox
