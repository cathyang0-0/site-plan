# SitePlan — Rhino client

Generates a site plan from the local backend and imports it into the active
Rhino document at real-world scale.

## Run it (Rhino 8)

1. Have the backend available — either way works:
   - **Installed** (end users): `uv tool install "siteplan-backend @
     git+https://github.com/cathyang0-0/site-plan#subdirectory=backend"`,
     once. The command below finds it and **starts it automatically**.
   - **Dev repo**: start it yourself in a terminal:
     ```bash
     cd backend && python -m uvicorn siteplan_backend.main:app --port 8000
     ```
2. In Rhino 8: `ScriptEditor` → open `SitePlan_command.py` → **Run** (▶).
3. The SitePlan dialog opens:
   - **Map (left):** search an address, then draw the site box — the "Draw
     box" button or shift+drag. The aerial you see is the same USGS imagery
     the pipeline processes. Draw slightly larger than needed (edge
     conditions aren't perfectly resolved); the time estimate updates as
     you draw — it grows steeply with area.
   - **Options (right):** trees on/off, land-cover engine
     (kmeans / segmodel / off), contour interval ("5ft" or meters,
     0 = none), river width, and per-class road widths under the expander
     (values are the *prior* — the CV pavement measurement still refines
     each road).
   - **Generate** shows per-stage progress (Cancel stops the wait; the
     backend job finishes and its fetches stay cached).
   - **Tree preview** (when trees are on): after detection, the dialog
     shows your site's aerial with every detected crown drawn as a circle.
     The size and variance sliders live here and re-render the circles
     instantly — detection ran once with neutral sizes, and the sliders
     are an export-time transform, so **Import draws exactly what you
     see** (in seconds, no re-detection). Caveat: dense-stand fill
     density was decided at detection, so extreme sizes can differ
     slightly from a fresh run at that size.
   - **Import** brings the plan into the active document and zooms to it.

The bbox is remembered between runs. The DXF is written natively in the
document's unit, so geometry AND hatch spacings import true.

## First run on a new machine

Run `probe_webview.py` in the ScriptEditor once (backend running) — it
verifies inside Rhino's WebView that the backend-served map page renders its
tiles, the JS bridge round-trips, and the address search works.

## Files

- `siteplan_client.py` — pure-stdlib HTTP client (submit → poll → export).
  Works anywhere Python runs; also a CLI:
  `python siteplan_client.py WEST SOUTH EAST NORTH out.dxf [layers]`
- `siteplan_form.py` — the dialog's logic (request assembly, parsing,
  estimates), pure Python — headless-tested by
  `backend/tests/test_rhino_form.py`.
- `siteplan_dialog.py` — the Eto dialog (widgets + threads only). Its two
  embedded pages (bbox map, tree preview) are served BY THE BACKEND
  (`/api/map`, `/api/jobs/{id}/preview`) from
  `backend/siteplan_backend/static/` — so this folder ships Python only,
  which is all the `.rhp` plugin format needs to carry. Develop the pages
  in a normal browser; the dialog polls `window.getState()` and pushes
  `initFromPython(bbox)` / `setParams(...)`.
- `SitePlan_command.py` — the Rhino-facing command: opens the dialog, then
  imports the DXF on Rhino's main thread. Thin on purpose.
- `probe_webview.py` — one-time platform probe (see above).

## Building the plugin (.rhp / .yak)

This folder IS a ScriptEditor project (`SitePlan.rhproj`): the command lives
in `Commands/SitePlan.py` (filename = command name) and the shared modules
in `Libraries/siteplan_plugin/` (embedded into the plugin at build). Build
headlessly:

```bash
rhino/build.sh
```

Artifacts land in `rhino/build/rh8/`: `SitePlan.rhp` (drag into Rhino to
install), `SitePlan.rui` (toolbar), and a `siteplan-*.yak` package for the
Package Manager / Food4Rhino. (`build.sh` wraps `rhinocode project build`
through a space-free symlink — rhinocode's wrapper mishandles spaces in
paths.) Installed users then just type `SitePlan`; the command auto-starts
the installed backend.
