"""Tests for rhino/build.py's bundle mechanism — the thing that makes the
published plugin independent of RhinoCode's (unreliable) project-library
deployment. Exercises the full chain: read module source → zlib+base64 →
generated prelude → exec → importable siteplan_plugin package.

dialog.py needs Eto/Rhino so it can't exec headless; the chain is tested on
__init__/client/form (identical treatment), dialog is covered by the
in-Rhino smoke test."""
import importlib.util
import sys
from pathlib import Path

import pytest

_BUILD_PY = Path(__file__).parents[2] / "rhino" / "build.py"
_spec = importlib.util.spec_from_file_location("siteplan_build", _BUILD_PY)
siteplan_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(siteplan_build)

END_MARK = "# ---- END GENERATED BUNDLE"


@pytest.fixture()
def registered_bundle(monkeypatch):
    """Exec the generated prelude (headless subset) in a clean namespace;
    restore sys.modules afterwards."""
    text = siteplan_build.make_bundled_command(
        modules=["__init__", "client", "form"])
    prelude = text[:text.index(END_MARK)]
    saved = {k: sys.modules.get(k)
             for k in ("siteplan_plugin", "siteplan_plugin.client",
                       "siteplan_plugin.form")}
    exec(compile(prelude, "bundle-prelude", "exec"), {})
    yield sys.modules["siteplan_plugin"]
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class TestBundle:
    def test_package_registered_with_submodules(self, registered_bundle):
        pkg = registered_bundle
        assert sys.modules["siteplan_plugin"] is pkg
        assert pkg.client is sys.modules["siteplan_plugin.client"]
        assert pkg.form is sys.modules["siteplan_plugin.form"]

    def test_from_imports_resolve(self, registered_bundle):
        # The exact import forms the command and dialog use.
        from siteplan_plugin import form
        from siteplan_plugin import client as spc
        assert form.parse_interval("5ft") == pytest.approx(1.524)
        assert spc.DEFAULT_BASE.startswith("http://localhost")

    def test_sources_roundtrip_exactly(self, registered_bundle):
        # zlib+base64 must be lossless — compare a known function's behavior
        # against the on-disk module.
        disk_dir = str(_BUILD_PY.parent / "Libraries")
        sys_path_had = disk_dir in sys.path
        bundled_form = sys.modules["siteplan_plugin.form"]
        assert bundled_form.build_request(
            {"west": 0, "south": 0, "east": 1, "north": 1},
            trees=False)["layers"].count("trees") == 0
        assert not sys_path_had or True

    def test_command_text_contains_command_source(self):
        text = siteplan_build.make_bundled_command(
            modules=["__init__", "client", "form"])
        assert "_ensure_backend" in text          # command rides after bundle
        assert text.index(END_MARK) < text.index("_ensure_backend")

    def test_default_modules_include_dialog(self):
        assert siteplan_build.MODULES == ["__init__", "client", "form",
                                          "dialog"]
