"""
SitePlan — Rhino 8 command script.

Generates an architect-style site plan for a lon/lat bounding box and imports
it into the ACTIVE Rhino document, on proper layers, at real-world scale
(the DXF declares meters; Rhino converts to your document units on import).

Run it:  Rhino 8 → ScriptEditor → open this file → ▶ Run
         (the backend must be running: cd backend &&
          python -m uvicorn app.main:app --port 8000)

Everything network-y lives in siteplan_client.py (pure stdlib); this file is
only the Rhino-facing skin: prompts, progress, Esc-to-cancel, and the import.
"""
import os
import sys
import tempfile

import rhinoscriptsyntax as rs
import scriptcontext as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import siteplan_client as spc

STICKY_KEY = "siteplan_last_bbox"


def _ask_bbox():
    """Prompt for WEST,SOUTH,EAST,NORTH (lon/lat). Remembers the last one."""
    default = sc.sticky.get(STICKY_KEY, "-76.5515,42.5305,-76.5415,42.5385")
    raw = rs.GetString("Site bbox lon/lat as WEST,SOUTH,EAST,NORTH", default)
    if raw is None:
        return None
    try:
        w, s, e, n = (float(v.strip()) for v in raw.split(","))
    except ValueError:
        rs.MessageBox("Expected four comma-separated numbers, e.g.\n"
                      "-76.5515,42.5305,-76.5415,42.5385")
        return None
    if not (w < e and s < n):
        rs.MessageBox("Bbox is inside-out: need west < east and south < north.")
        return None
    sc.sticky[STICKY_KEY] = raw
    return {"west": w, "south": s, "east": e, "north": n}


def _ask_options():
    """A few yes/no choices; defaults match the user's usual full plan."""
    trees = rs.GetString("Detect trees? (the slow stage, ~5 min)", "Yes",
                         ["Yes", "No"])
    if trees is None:
        return None, None
    layers = ["roofs", "roads", "land_types"] + (["trees"] if trees == "Yes" else [])
    engine = rs.GetString("Land-cover engine", "kmeans", ["kmeans", "segmodel"])
    if engine is None:
        return None, None
    options = {"land_types_engine": engine, "crown_size_scale": 1.5}
    return layers, options


def _progress(status):
    """Print stage progress to the command line; Esc cancels the wait.
    (Cancelling stops the Rhino side — the backend job finishes on its own.)"""
    stages = status.get("progress") or {}
    running = [k for k, v in stages.items() if v == "running"]
    done = sum(1 for v in stages.values() if v == "done")
    rs.Prompt("SitePlan: {} | {} done | running: {} (Esc to cancel)".format(
        status["status"], done, ", ".join(running) or "-"))
    rs.Sleep(0)  # let the UI breathe
    return not sc.escape_test(False)


def run():
    bbox = _ask_bbox()
    if bbox is None:
        return
    layers, options = _ask_options()
    if layers is None:
        return

    out = os.path.join(tempfile.mkdtemp(prefix="siteplan_"), "site-plan.dxf")
    try:
        job_id = spc.submit_job(bbox, layers=layers, options=options)
        status = spc.poll_job(job_id, on_progress=_progress)
        path = spc.export_dxf(job_id, out)
    except spc.SitePlanError as exc:
        rs.MessageBox(str(exc), title="SitePlan")
        return

    # Non-fatal stage failures (e.g. an Overture layer timed out): the plan
    # imported fine but is missing that layer — tell the user, don't hide it.
    warnings = status.get("warnings") or []
    if warnings:
        rs.MessageBox("Plan generated WITH WARNINGS — some layers are "
                      "missing:\n\n" + "\n".join(warnings) +
                      "\n\nRe-run later to get them (results are cached once "
                      "a fetch succeeds).", title="SitePlan")

    # Import the DXF into the active document. The file declares meters
    # ($INSUNITS=6); Rhino scales it into the document's units automatically.
    rs.Command('_-Import "{}" _Enter'.format(path), echo=False)
    rs.ZoomExtents()
    rs.Prompt("SitePlan: imported {}".format(os.path.basename(path)))


if __name__ == "__main__":
    run()
