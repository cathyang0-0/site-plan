"""Tests for the Rhino client's pure HTTP module (rhino/Libraries/siteplan_plugin/client.py).

The client is stdlib-only by design (runs inside Rhino's CPython); here we
fake urllib's urlopen so no server is needed — same test philosophy as the
rest of the suite: fake the I/O boundary, test the logic around it.
"""
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "rhino" / "Libraries"))

from siteplan_plugin import client as spc


class FakeResponse(io.BytesIO):
    def __init__(self, payload, content_type="application/json"):
        body = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        super().__init__(body)
        self.headers = {"Content-Type": content_type}
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _fake_urlopen(responses, seen):
    """Queue of FakeResponses; records (method, url, body) of each call."""
    def opener(req, timeout=None):
        body = json.loads(req.data) if req.data else None
        seen.append((req.get_method(), req.full_url, body))
        resp = responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp
    return opener


class TestClient:
    def test_submit_sends_bbox_and_returns_id(self, monkeypatch):
        seen = []
        monkeypatch.setattr(spc.urllib.request, "urlopen", _fake_urlopen(
            [FakeResponse({"job_id": "j1", "status": "queued"})], seen))
        job_id = spc.submit_job({"west": -1, "south": -1, "east": 1, "north": 1},
                                layers=["roofs"])
        assert job_id == "j1"
        method, url, body = seen[0]
        assert method == "POST" and url.endswith("/api/jobs")
        assert body["bbox"]["west"] == -1 and body["layers"] == ["roofs"]

    def test_poll_reports_progress_then_returns(self, monkeypatch):
        seen, statuses = [], []
        monkeypatch.setattr(spc.urllib.request, "urlopen", _fake_urlopen([
            FakeResponse({"job_id": "j", "status": "running",
                          "progress": {"imagery": "running"}}),
            FakeResponse({"job_id": "j", "status": "complete", "progress": {}}),
        ], seen))
        final = spc.poll_job("j", interval=0, on_progress=statuses.append)
        assert final["status"] == "complete"
        assert statuses[0]["progress"] == {"imagery": "running"}

    def test_poll_failure_raises_with_backend_error(self, monkeypatch):
        monkeypatch.setattr(spc.urllib.request, "urlopen", _fake_urlopen([
            FakeResponse({"job_id": "j", "status": "failed",
                          "error": "TimeoutError: Overture slow"})], []))
        with pytest.raises(spc.SitePlanError, match="Overture slow"):
            spc.poll_job("j", interval=0)

    def test_poll_cancel_via_callback(self, monkeypatch):
        monkeypatch.setattr(spc.urllib.request, "urlopen", _fake_urlopen([
            FakeResponse({"job_id": "j", "status": "running"})], []))
        with pytest.raises(spc.SitePlanError, match="cancelled"):
            spc.poll_job("j", interval=0, on_progress=lambda s: False)

    def test_export_writes_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(spc.urllib.request, "urlopen", _fake_urlopen([
            FakeResponse(b"DXFBYTES", content_type="application/dxf")], []))
        out = tmp_path / "plan.dxf"
        assert spc.export_dxf("j", str(out)) == str(out)
        assert out.read_bytes() == b"DXFBYTES"

    def test_backend_down_gives_start_hint(self, monkeypatch):
        import urllib.error
        monkeypatch.setattr(spc.urllib.request, "urlopen", _fake_urlopen(
            [urllib.error.URLError("connection refused")], []))
        with pytest.raises(spc.SitePlanError, match="uvicorn siteplan_backend.main:app"):
            spc.submit_job({"west": -1, "south": -1, "east": 1, "north": 1})
