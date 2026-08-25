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

Run `probe_webview.py` in the ScriptEditor once — it verifies inside Rhino's
WebView that the map tiles render (CORS), the JS bridge round-trips, and the
address search works, and its docstring says what to do if any check fails.

## Files

- `siteplan_client.py` — pure-stdlib HTTP client (submit → poll → export).
  Works anywhere Python runs; also a CLI:
  `python siteplan_client.py WEST SOUTH EAST NORTH out.dxf [layers]`
- `siteplan_form.py` — the dialog's logic (request assembly, parsing,
  estimates), pure Python — headless-tested by
  `backend/tests/test_rhino_form.py`.
- `siteplan_map.html` — the embedded map page (MapLibre + keyless USGS
  tiles + Nominatim search). Develop it in a normal browser; the dialog
  polls `window.getState()` and pushes `initFromPython(bbox)`.
- `siteplan_dialog.py` — the Eto dialog (widgets + threads only).
- `SitePlan_command.py` — the Rhino-facing command: opens the dialog, then
  imports the DXF on Rhino's main thread. Thin on purpose.
- `probe_webview.py` — one-time platform probe (see above).

## Packaging as a real plugin (later)

Rhino 8's ScriptEditor can compile Python scripts into an installable `.rhp`
plugin with its own command name and toolbar button: ScriptEditor →
**Publish** → Rhino Plugin. The command is already shaped for that (a single
`run()` entry point); packaging is deferred until the dialog has been used
in anger for a while.
