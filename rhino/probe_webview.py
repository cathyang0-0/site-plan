"""
Throwaway WebView probe — run in Rhino's ScriptEditor ONCE on a new machine
before trusting the full dialog. It verifies, in isolation:

  1. The backend is reachable (it now also SERVES the map page at /api/map,
     so the dialog can't work without it — the real command auto-starts it,
     this probe just tells you).
  2. The map page renders inside Rhino's Eto WebView (USGS imagery tiles).
  3. DocumentLoaded fires (the dialog pushes the sticky bbox there).
  4. ExecuteScript round-trips a value (the dialog's whole bridge).
  5. Nominatim address search works (type something, hit Go).

If a check fails, the label under the map says which — report that text.

NOTE (learned the hard way, applies to every Eto script in this project):
constructor property-kwargs like forms.Label(Text=...) are NOT supported by
this Rhino build's Python.NET — construct bare, then set properties.
"""
import os
import sys

import System
import Eto.Forms as forms
import Eto.Drawing as drawing
import Rhino.UI
import scriptcontext as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.modules.pop("siteplan_client", None)   # ScriptEditor caches imports
import siteplan_client as spc


class Probe(forms.Dialog):
    def __init__(self):
        super().__init__()   # required in Rhino 8 CPython — see siteplan_dialog.py
        self.Title = "SitePlan WebView probe"
        self.ClientSize = drawing.Size(720, 600)

        self.web = forms.WebView()
        self.web.Size = drawing.Size(700, 480)
        self.web.DocumentLoaded += self._loaded

        self.out = forms.Label()
        self.out.Text = "loading the map page from the backend…"

        btn = forms.Button()
        btn.Text = "Run bridge check (do this after tiles show)"
        btn.Click += self._check

        lay = forms.DynamicLayout()
        lay.Spacing = drawing.Size(6, 6)
        lay.Add(self.web, True, True)
        lay.Add(self.out)
        lay.Add(btn)
        self.Content = lay
        self.web.Url = System.Uri(spc.DEFAULT_BASE + "/api/map")

    def _loaded(self, sender, e):
        self.out.Text = ("DocumentLoaded ✓ — do you SEE aerial imagery "
                         "above? Then draw a box and run the bridge check.")

    def _check(self, sender, e):
        try:
            raw = self.web.ExecuteScript(
                "return window.getState ? window.getState() : 'page has no getState'")
            self.out.Text = "ExecuteScript ✓ → " + str(raw)[:160]
        except Exception as exc:
            self.out.Text = "ExecuteScript ✗ → " + str(exc)


if not spc.health():
    forms.MessageBox.Show(
        "The SitePlan backend isn't running — the map page is served by it.\n"
        "Start it first (the real SitePlan command does this automatically):\n"
        "  siteplan-backend        (installed)\n"
        "  cd backend && python -m uvicorn siteplan_backend.main:app  (dev)",
        "SitePlan probe")
else:
    # MainWindowForDocument, not MainWindow — the latter silently fails on
    # Mac (developer.rhino3d.com/guides/eto/rhino-specific). The dialog may
    # open BEHIND the ScriptEditor window; move the editor if you don't see it.
    Probe().ShowModal(Rhino.UI.RhinoEtoApp.MainWindowForDocument(sc.doc))
