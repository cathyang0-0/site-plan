"""
Throwaway WebView probe — run this in Rhino's ScriptEditor ONCE before
trusting the full dialog, to retire the platform risks in isolation:

  1. Does the map page's imagery render inside Rhino's Eto WebView?
     (file:// pages have a null origin; USGS tiles must allow CORS.)
  2. Does DocumentLoaded fire?  (the dialog pushes the sticky bbox there)
  3. Does ExecuteScript round-trip a value?  (the dialog's whole bridge)
  4. Does the Nominatim address search work?  (type something, hit Go)

Interpreting results:
  - Blank map but the label says DocumentLoaded fired → tile CORS is blocked
    from file://. Fallback (one line in siteplan_dialog.py): serve the page
    from the backend instead — add to backend/app/main.py:
        from fastapi.responses import FileResponse
        @app.get("/map")
        def map_page(): return FileResponse("../rhino/siteplan_map.html")
    and point siteplan_dialog.MAP_URL at http://localhost:8000/map.
  - "ExecuteScript ✗" → tell Claude; the bridge needs the DocumentTitle
    fallback wired instead.
"""
import os

import System
import Eto.Forms as forms
import Eto.Drawing as drawing
import Rhino.UI

HERE = os.path.dirname(os.path.abspath(__file__))


class Probe(forms.Dialog):
    def __init__(self):
        super().__init__()   # required in Rhino 8 CPython — see siteplan_dialog.py
        self.Title = "SitePlan WebView probe"
        self.ClientSize = drawing.Size(720, 600)
        self.web = forms.WebView()
        self.web.Size = drawing.Size(700, 480)
        self.web.DocumentLoaded += self._loaded
        self.out = forms.Label(Text="loading siteplan_map.html…")
        btn = forms.Button(Text="Run bridge check (do this after tiles show)")
        btn.Click += self._check
        lay = forms.DynamicLayout(Spacing=drawing.Size(6, 6))
        lay.Add(self.web, True, True)
        lay.Add(self.out)
        lay.Add(btn)
        self.Content = lay
        self.web.Url = System.Uri(
            "file://" + os.path.join(HERE, "siteplan_map.html"))

    def _loaded(self, sender, e):
        self.out.Text = ("DocumentLoaded ✓ — check 1: do you SEE aerial "
                         "imagery above? Then draw a box and run the bridge check.")

    def _check(self, sender, e):
        try:
            raw = self.web.ExecuteScript(
                "return window.getState ? window.getState() : 'page has no getState'")
            self.out.Text = "ExecuteScript ✓ → " + str(raw)[:160]
        except Exception as exc:
            self.out.Text = "ExecuteScript ✗ → " + str(exc)


Probe().ShowModal(Rhino.UI.RhinoEtoApp.MainWindow)
