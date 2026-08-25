"""
SitePlan — Rhino 8 command script.

Generates an architect-style site plan and imports it into the ACTIVE Rhino
document, on proper layers, at real-world scale (the DXF is written natively
in this document's unit; no conversion on import).

Run it:  Rhino 8 → ScriptEditor → open this file → ▶ Run
         (the backend must be running: cd backend &&
          python -m uvicorn siteplan_backend.main:app --port 8000)

The old rs.GetString prompt chain is gone: options now come from an Eto
dialog with an embedded map (siteplan_dialog.py — bbox by drawing, address
search, sliders, road widths, live progress). This file keeps only what MUST
run on Rhino's main thread with the document: sticky handling, the DXF
import, and the report.
"""
import os
import sys

import rhinoscriptsyntax as rs
import scriptcontext as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ScriptEditor caches imported modules for the whole Rhino session — without
# this, edits to the siteplan_* files silently don't apply on re-run.
# importlib.reload proved unreliable here; evicting from sys.modules before
# the import forces a genuinely fresh read from disk.
for _name in ("siteplan_client", "siteplan_form", "siteplan_dialog"):
    sys.modules.pop(_name, None)
import siteplan_dialog

STICKY_KEY = "siteplan_last_bbox"
LAST_JOB_KEY = "siteplan_last_job"   # {"job_id", "style"} of the last tree run
DEFAULT_BBOX = "-76.5515,42.5305,-76.5415,42.5385"


def _sticky_bbox():
    """The remembered bbox as a dict (the dialog's format)."""
    raw = sc.sticky.get(STICKY_KEY, DEFAULT_BBOX)
    try:
        w, s, e, n = (float(v.strip()) for v in raw.split(","))
        return {"west": w, "south": s, "east": e, "north": n}
    except ValueError:
        return None


def run():
    # Ask the backend to write the DXF NATIVELY in this document's unit —
    # no unit conversion on import, so hatch pattern spacings stay correct.
    doc_units = {2: "mm", 3: "cm", 4: "m", 8: "in", 9: "ft"}.get(
        rs.UnitSystem(), "m")

    result, last_bbox, last_job = siteplan_dialog.show(
        _sticky_bbox(), units=doc_units, last_job=sc.sticky.get(LAST_JOB_KEY))

    if last_bbox:   # remember the bbox even if the run failed/was cancelled
        sc.sticky[STICKY_KEY] = "%(west)s,%(south)s,%(east)s,%(north)s" % last_bbox
    if last_job:    # so "Tree preview of last run" works on the next open
        sc.sticky[LAST_JOB_KEY] = last_job
    if result is None:
        return
    path = result["path"]

    # Non-fatal stage failures (e.g. an Overture layer timed out): the plan
    # imported fine but is missing that layer — tell the user, don't hide it.
    warnings = result["warnings"]
    if warnings:
        rs.MessageBox("Plan generated WITH WARNINGS — some layers are "
                      "missing:\n\n" + "\n".join(warnings) +
                      "\n\nRe-run later to get them (results are cached once "
                      "a fetch succeeds).", title="SitePlan")

    # Import the DXF into the active document. The file is written NATIVELY
    # in this document's unit (we sent it with the request), so no unit
    # conversion happens on import — geometry AND hatch spacing arrive true.
    #
    # IMPORTANT: use RhinoDoc.Import (the same engine as manual File > Import),
    # NOT the scripted '_-Import' macro. Scripted -Import was observed to
    # SCRAMBLE these plans on every run — ROOFS/CONTOURS entities dropped,
    # curves re-assigned to wrong layers, hatches swapped (faking overlap) —
    # while manual imports of the identical files were always perfect. Chasing
    # that difference cost a full day; do not switch back.
    imported = False
    try:
        imported = bool(sc.doc.Import(path))
    except AttributeError:      # very old Rhino: no RhinoDoc.Import
        pass
    if not imported:
        rs.Command('_-Import "{}" _Enter'.format(path), echo=False)
    rs.ZoomExtents()
    rs.Prompt("SitePlan: imported {}".format(os.path.basename(path)))

    # Ground-truth report: count what ACTUALLY landed in this document, per
    # layer. If a count here is non-zero but you can't see that layer's
    # geometry, the problem is display/layer state — not the import. If a
    # count is zero, the import genuinely dropped it. (Added after a long
    # debugging session where the file was repeatedly proven correct while
    # the on-screen result disagreed.)
    report = []
    for name in sorted(rs.LayerNames() or []):
        objs = rs.ObjectsByLayer(name) or []
        if objs:
            report.append("{}: {}".format(name, len(objs)))
    rs.MessageBox(
        "Imported into the ACTIVE document ({}):\n\n{}\n\nDXF file: {}".format(
            os.path.basename(sc.doc.Path or "untitled"),
            "\n".join(report) or "(nothing!)",
            path,
        ),
        title="SitePlan import report",
    )


if __name__ == "__main__":
    run()
