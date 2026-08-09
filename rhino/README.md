# SitePlan — Rhino client

Generates a site plan from the local backend and imports it into the active
Rhino document at real-world scale.

## Run it (Rhino 8)

1. Start the backend (once, in a terminal):
   ```bash
   cd backend && python -m uvicorn app.main:app --port 8000
   ```
2. In Rhino 8: `ScriptEditor` → open `SitePlan_command.py` → **Run** (▶).
3. Answer the prompts (bbox, trees on/off, land engine). Progress shows on the
   command line; **Esc** cancels the wait. When done, the plan imports itself
   and zooms to it.

The bbox is remembered between runs. The DXF declares meters, so Rhino
converts into your document units automatically (a ~9 m tree is 9000 mm in an
mm document — that's correct).

## Files

- `siteplan_client.py` — pure-stdlib HTTP client (submit → poll → export).
  Works anywhere Python runs; also a CLI:
  `python siteplan_client.py WEST SOUTH EAST NORTH out.dxf [layers]`
- `SitePlan_command.py` — the Rhino-facing command (prompts + import). Thin on
  purpose: all logic that can live outside Rhino does.

## Packaging as a real plugin (later)

Rhino 8's ScriptEditor can compile Python scripts into an installable `.rhp`
plugin with its own command name and toolbar button: ScriptEditor →
**Publish** → Rhino Plugin. The script is already shaped for that (a single
`run()` entry point); packaging is deferred until the command has been used
in anger for a while.
