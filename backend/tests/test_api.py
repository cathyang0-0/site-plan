"""API layer tests — job lifecycle over HTTP with the pipeline FAKED.

Same convention as the rest of the suite: the slow/network parts (imagery,
Overture, CV) are replaced, and we test the machinery around them — routes,
job store, status transitions, export. The fake runs instantly, so tests
exercise the real threading path with a short poll loop.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from app.main import app
from app.api import jobs

client = TestClient(app)

BBOX = {"west": -76.5515, "south": 42.5305, "east": -76.5415, "north": 42.5385}


def _wait(job_id, want=("complete", "failed"), timeout=5.0):
    """Poll until the job reaches a terminal state (the fake is fast)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in want:
            return status
        time.sleep(0.02)
    pytest.fail(f"job {job_id} did not reach {want} within {timeout}s")


def _fake_ok(job):
    job.progress["imagery"] = "done"
    job.dxf_path = job.dir / "plan.dxf"
    job.dxf_path.write_text("fake dxf")
    job.geometry = {}  # marks geometry as present


def _fake_boom(job):
    raise RuntimeError("stage exploded")


class TestJobLifecycle:
    def test_submit_poll_complete(self, monkeypatch):
        monkeypatch.setattr(jobs, "_run_pipeline", _fake_ok)
        res = client.post("/api/jobs", json={"bbox": BBOX})
        assert res.status_code == 200
        body = res.json()
        assert body["status"] in ("queued", "running")
        status = _wait(body["job_id"])
        assert status["status"] == "complete"
        assert status["progress"]["imagery"] == "done"
        assert status["error"] is None

    def test_failure_surfaces_error(self, monkeypatch):
        monkeypatch.setattr(jobs, "_run_pipeline", _fake_boom)
        job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["job_id"]
        status = _wait(job_id)
        assert status["status"] == "failed"
        assert "stage exploded" in status["error"]

    def test_export_returns_dxf(self, monkeypatch):
        monkeypatch.setattr(jobs, "_run_pipeline", _fake_ok)
        job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["job_id"]
        _wait(job_id)
        res = client.post(f"/api/jobs/{job_id}/export")
        assert res.status_code == 200
        assert res.content == b"fake dxf"

    def test_export_before_complete_is_409(self, monkeypatch):
        monkeypatch.setattr(jobs, "_run_pipeline", _fake_boom)
        job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["job_id"]
        _wait(job_id)  # failed
        assert client.post(f"/api/jobs/{job_id}/export").status_code == 409


class TestValidation:
    def test_unknown_job_404(self):
        assert client.get("/api/jobs/nope").status_code == 404
        assert client.post("/api/jobs/nope/export").status_code == 404

    def test_oversize_bbox_rejected_up_front(self):
        # A huge bbox would exceed the USGS export limit — reject at submit
        # time, not minutes into a job.
        huge = {"west": -77.0, "south": 42.0, "east": -76.0, "north": 43.0}
        res = client.post("/api/jobs", json={"bbox": huge})
        assert res.status_code == 422
        assert "export limit" in res.json()["detail"]

    def test_width_overrides_accepted(self, monkeypatch):
        # The plugin's road/river width settings ride through validation.
        monkeypatch.setattr(jobs, "_run_pipeline", _fake_ok)
        res = client.post("/api/jobs", json={
            "bbox": BBOX,
            "options": {"road_class_widths": {"residential": 12.0},
                        "river_width_m": 8.0},
        })
        assert res.status_code == 200
        job = jobs.get_job(res.json()["job_id"])
        assert job.request.options.road_class_widths == {"residential": 12.0}
        assert job.request.options.river_width_m == 8.0

    def test_tree_preview_endpoints(self, monkeypatch):
        # Fake a completed job carrying geometry + a site image, then walk
        # the three preview endpoints the Rhino dialog uses.
        def _fake_with_trees(job):
            job.progress["imagery"] = "done"
            (job.dir / "site.png").write_bytes(b"\x89PNG fake")
            job.dxf_path = job.dir / "plan.dxf"
            job.dxf_path.write_text("fake dxf")
            job.geometry = {
                "tree_placements": [
                    {"block_idx": 0, "position": (10.0, 20.0), "scale": 1.0,
                     "rotation": 0.0},
                    {"block_idx": 0, "position": (30.0, 40.0), "scale": 2.0,
                     "rotation": 0.0},
                ],
                "tree_mean_scale": 1.5, "img_size": (800, 600), "scale": 0.3,
            }
        monkeypatch.setattr(jobs, "_run_pipeline", _fake_with_trees)
        res = client.post("/api/jobs", json={"bbox": BBOX})
        job_id = res.json()["job_id"]
        _wait(job_id)

        img = client.get(f"/api/jobs/{job_id}/image")
        assert img.status_code == 200
        assert img.headers["content-type"] == "image/png"

        trees = client.get(f"/api/jobs/{job_id}/trees").json()
        assert len(trees["placements"]) == 2
        assert trees["placements"][1]["r"] == pytest.approx(2.0 * 20.0)
        assert trees["mean_r"] == pytest.approx(1.5 * 20.0)
        assert trees["min_r"] == pytest.approx(1.5 / 0.3)
        assert (trees["img_w"], trees["img_h"]) == (800, 600)

        page = client.get(f"/api/jobs/{job_id}/preview")
        assert page.status_code == 200
        assert "getState" in page.text or "setParams" in page.text

    def test_preview_endpoints_409_before_complete(self, monkeypatch):
        monkeypatch.setattr(jobs, "_run_pipeline",
                            lambda job: time.sleep(30))
        res = client.post("/api/jobs", json={"bbox": BBOX})
        job_id = res.json()["job_id"]
        for path in ("image", "trees", "preview"):
            assert client.get(f"/api/jobs/{job_id}/{path}").status_code == 409

    def test_zero_river_width_rejected(self):
        res = client.post("/api/jobs", json={
            "bbox": BBOX, "options": {"river_width_m": 0}})
        assert res.status_code == 422

    def test_bad_engine_rejected_by_schema(self):
        res = client.post("/api/jobs", json={
            "bbox": BBOX, "options": {"land_types_engine": "magic"}})
        assert res.status_code == 422

    def test_blocks_parse_still_501(self):
        res = client.post("/api/blocks/parse",
                          files={"file": ("t.dxf", b"x", "application/dxf")})
        assert res.status_code == 501

    def test_health(self):
        assert client.get("/health").json() == {"status": "ok"}


class TestImagerySizing:
    def test_cayuga_bbox_fits(self):
        from app.pipeline.imagery import usgs_export_size_px
        w, h = usgs_export_size_px(**BBOX, m_per_px=0.3)
        assert 2000 < w < 4096 and 2000 < h < 4096

    def test_degenerate_bbox_raises(self):
        from app.pipeline.imagery import usgs_export_size_px
        with pytest.raises(ValueError):
            usgs_export_size_px(west=0, south=0, east=0, north=0, m_per_px=0.3)
