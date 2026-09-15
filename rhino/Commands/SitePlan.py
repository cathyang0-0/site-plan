"""
SitePlan — Rhino 8 command script.

Generates an architect-style site plan and imports it into the ACTIVE Rhino
document, on proper layers, at real-world scale (the DXF is written natively
in this document's unit; no conversion on import).

This file is the `SitePlan` COMMAND of the ScriptEditor project
(../SitePlan.rhproj — the filename becomes the command name when the
plugin is published). Dev run: Rhino 8 → ScriptEditor → open this file →
▶ Run; the command auto-starts an installed backend, or start the dev
one yourself (cd backend && python -m uvicorn siteplan_backend.main:app).

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

# Dev-run support: when this file runs loose from the repo (ScriptEditor ▶,
# not the published plugin), the project's Libraries/ dir isn't on sys.path
# — add it. In the published plugin the library is embedded and importable
# already; the extra path entry simply won't exist and is harmless.
_LIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "Libraries")
if os.path.isdir(_LIB_DIR) and _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

# ScriptEditor caches imported modules for the whole Rhino session — without
# this, edits to the siteplan_plugin files silently don't apply on re-run.
# importlib.reload proved unreliable here; evicting from sys.modules before
# the import forces a genuinely fresh read from disk.
for _name in [m for m in list(sys.modules) if m.startswith("siteplan_plugin")]:
    sys.modules.pop(_name, None)
from siteplan_plugin import dialog as siteplan_dialog
from siteplan_plugin import client as spc

STICKY_KEY = "siteplan_last_bbox"
LAST_JOB_KEY = "siteplan_last_job"   # {"job_id", "style"} of the last tree run
DEFAULT_BBOX = "-76.5515,42.5305,-76.5415,42.5385"


BACKEND_CMD = "siteplan-backend"
INSTALL_CMD = ('uv tool install "siteplan-backend @ '
               'git+https://github.com/cathyang0-0/site-plan'
               '#subdirectory=backend"')


def _find_backend_cmd():
    """Locate the installed `siteplan-backend` command. Rhino's GUI process
    often lacks the user's shell PATH (especially uv's ~/.local/bin), so
    check the known install locations explicitly after which()."""
    import shutil
    exe = shutil.which(BACKEND_CMD)
    if exe:
        return exe
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".local", "bin", BACKEND_CMD),          # uv tool
        os.path.join(home, ".local", "bin", BACKEND_CMD + ".exe"),  # uv, Win
        "/usr/local/bin/" + BACKEND_CMD,
        "/opt/homebrew/bin/" + BACKEND_CMD,
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def _ensure_backend():
    """Backend reachable? If not, try to start the installed one; failing
    that, tell the user exactly how to get it. Returns True when healthy."""
    if spc.health():
        return True

    exe = _find_backend_cmd()
    if exe is None:
        rs.MessageBox(
            "The SitePlan backend isn't running, and no installed copy was "
            "found.\n\n"
            "One-time install (needs uv, from https://docs.astral.sh/uv):\n"
            "  " + INSTALL_CMD + "\n\n"
            "Then run this command again — it will start the backend "
            "automatically.\n\n"
            "(Developing from the repo instead? Start it yourself:\n"
            "  cd backend && python -m uvicorn siteplan_backend.main:app)",
            title="SitePlan")
        return False

    # First start downloads model weights lazily later; the server itself
    # comes up in seconds. Detach so it outlives this Rhino session.
    import subprocess
    import tempfile
    import time
    log_path = os.path.join(tempfile.gettempdir(), "siteplan-backend.log")
    with open(log_path, "ab") as log:
        subprocess.Popen([exe], stdout=log, stderr=log,
                         start_new_session=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        rs.Prompt("SitePlan: starting the local backend…")
        if spc.health(timeout=2.0):
            rs.Prompt("SitePlan: backend ready")
            return True
        rs.Sleep(500)
    rs.MessageBox("Started the backend but it didn't answer within 30 s.\n"
                  "Check its log:\n" + log_path, title="SitePlan")
    return False


def _sticky_bbox():
    """The remembered bbox as a dict (the dialog's format)."""
    raw = sc.sticky.get(STICKY_KEY, DEFAULT_BBOX)
    try:
        w, s, e, n = (float(v.strip()) for v in raw.split(","))
        return {"west": w, "south": s, "east": e, "north": n}
    except ValueError:
        return None


def run():
    if not _ensure_backend():
        return

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
