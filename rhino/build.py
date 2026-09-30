"""
Build the SitePlan plugin: bundle the library INTO the command, then run
`rhinocode project build` on a staged copy of the project.

Why bundling exists: RhinoCode's project-library mechanism embeds our
Libraries/siteplan_plugin sources into the .rhp correctly, but the runtime
never registers them for import in a PUBLISHED plugin — the command dies
with "cannot import name 'dialog' from 'siteplan_plugin' (unknown
location)". This is an open, unanswered problem on the McNeel forum
(discourse.mcneel.com t/222591), so we stopped depending on the mechanism:
at build time each module's source is zlib+base64-embedded into the command
file, which registers them in sys.modules before its own imports run.
Dev workflow is untouched — ScriptEditor ▶ on Commands/SitePlan.py still
imports from Libraries/ off disk (and hot-reloads via the eviction block).

Staging lives in a temp dir with no spaces — which also sidesteps
rhinocode's unquoted-$@ bug with spaces in paths ("Site Plan Drafter").
"""
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
RHINOCODE = "/Applications/Rhino 8.app/Contents/Resources/bin/rhinocode"
LIB = HERE / "Libraries" / "siteplan_plugin"
MODULES = ["__init__", "client", "form", "dialog"]  # dependency order

BUNDLE_PRELUDE = '''\
# ---- GENERATED BUNDLE (rhino/build.py) — do not edit this block ----------
# The siteplan_plugin library, embedded so the published plugin never
# depends on RhinoCode's project-library deployment (unreliable for Python;
# see rhino/build.py). Modules exec with real filenames for tracebacks.
import base64 as _b64, sys as _sys, types as _types, zlib as _zlib

_BUNDLED_SRC = {bundled!r}

def _register_bundle():
    pkg = _types.ModuleType("siteplan_plugin")
    pkg.__package__ = "siteplan_plugin"
    pkg.__path__ = []          # mark as package so submodule imports resolve
    _sys.modules["siteplan_plugin"] = pkg
    for name, blob in _BUNDLED_SRC:
        src = _zlib.decompress(_b64.b64decode(blob)).decode("utf-8")
        full = "siteplan_plugin" if name == "__init__" else \\
               "siteplan_plugin." + name
        mod = pkg if name == "__init__" else _types.ModuleType(full)
        mod.__package__ = "siteplan_plugin"
        code = compile(src, "siteplan_plugin/%s.py" % name, "exec")
        exec(code, mod.__dict__)
        _sys.modules[full] = mod
        if mod is not pkg:
            setattr(pkg, name, mod)

_register_bundle()
# ---- END GENERATED BUNDLE -------------------------------------------------

'''


def make_bundled_command(modules=None) -> str:
    """The published command file: bundle prelude + Commands/SitePlan.py.
    `modules` is overridable so headless tests can exercise the bundle
    chain on the Rhino-free modules (client/form)."""
    bundled = []
    for name in (modules or MODULES):
        src = (LIB / (name + ".py")).read_text(encoding="utf-8")
        blob = base64.b64encode(zlib.compress(src.encode("utf-8"))).decode()
        bundled.append((name, blob))
    command_src = (HERE / "Commands" / "SitePlan.py").read_text(encoding="utf-8")
    return BUNDLE_PRELUDE.format(bundled=bundled) + command_src


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else None

    staging = Path(tempfile.mkdtemp(prefix="siteplan_build_"))
    try:
        (staging / "Commands").mkdir()
        (staging / "Commands" / "SitePlan.py").write_text(
            make_bundled_command(), encoding="utf-8")

        # Project file: same as the repo's, minus the libraries section —
        # the bundle replaces it, and dropping it stops the builder from
        # embedding a second (non-working) copy of the sources.
        proj = json.loads((HERE / "SitePlan.rhproj").read_text())
        proj.pop("libraries", None)
        (staging / "SitePlan.rhproj").write_text(json.dumps(proj, indent=2))

        cmd = [RHINOCODE, "project", "build", str(staging / "SitePlan.rhproj")]
        if version:
            cmd += ["--buildversion", version]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode

        out = HERE / "build" / "rh8"
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(exist_ok=True)
        shutil.copytree(staging / "build" / "rh8", out)
        print("Artifacts in: %s (.rhp, .rui, .yak)" % out)
        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
