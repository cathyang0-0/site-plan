"""Tests for the Overture-outage resilience trio: retry, disk cache, and
warn-and-continue job degradation. Motivated by a real multi-hour Overture
slow spell that failed every job at the 90 s fetch timeout."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline import overture_cache
from app.pipeline.footprints import fetch_with_retry


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(overture_cache, "CACHE_DIR", tmp_path / "cache")


class TestRetry:
    def test_succeeds_after_timeouts(self, monkeypatch):
        calls = []
        import app.pipeline.footprints as fp

        def fake_fetch_with_timeout(fn, timeout_s, what):
            calls.append(what)
            if len(calls) < 3:
                raise TimeoutError("slow")
            return "result"
        monkeypatch.setattr(fp, "fetch_with_timeout", fake_fetch_with_timeout)
        assert fetch_with_retry(lambda: None, "x", attempts=3) == "result"
        assert len(calls) == 3

    def test_raises_after_all_attempts(self, monkeypatch):
        import app.pipeline.footprints as fp
        monkeypatch.setattr(fp, "fetch_with_timeout",
                            lambda fn, t, w: (_ for _ in ()).throw(TimeoutError("dead")))
        with pytest.raises(TimeoutError):
            fetch_with_retry(lambda: None, "x", attempts=2)

    def test_non_timeout_error_not_retried(self, monkeypatch):
        calls = []
        import app.pipeline.footprints as fp

        def fake(fn, t, w):
            calls.append(1)
            raise ValueError("bad bbox")
        monkeypatch.setattr(fp, "fetch_with_timeout", fake)
        with pytest.raises(ValueError):
            fetch_with_retry(lambda: None, "x", attempts=3)
        assert len(calls) == 1  # no loop on non-transient errors


class TestCache:
    def test_roundtrip(self):
        from shapely.geometry import Polygon
        poly = Polygon([(0, 0), (1, 0), (1, 1)])
        overture_cache.put("water", (1, 2, 3, 4), [poly])
        [back] = overture_cache.get("water", (1, 2, 3, 4))
        assert back.equals(poly)

    def test_miss_returns_none(self):
        assert overture_cache.get("water", (9, 9, 9, 9)) is None

    def test_kinds_are_separate(self):
        overture_cache.put("water", (1, 2, 3, 4), "wet")
        assert overture_cache.get("building", (1, 2, 3, 4)) is None

    def test_corrupt_entry_is_a_miss(self):
        overture_cache.put("water", (1, 2, 3, 4), "ok")
        overture_cache._path("water", (1, 2, 3, 4)).write_bytes(b"garbage")
        assert overture_cache.get("water", (1, 2, 3, 4)) is None

    def test_fetch_uses_cache_without_network(self):
        # Prime the cache, then fetch — if it touched the network/overturemaps
        # at all it would fail (nothing is mocked); the cache short-circuits.
        from app.pipeline.water import fetch_water_footprints
        overture_cache.put("water", (-1.0, -1.0, 1.0, 1.0), ["sentinel"])
        assert fetch_water_footprints(-1.0, -1.0, 1.0, 1.0) == ["sentinel"]


class TestDegradation:
    def test_failed_stage_warns_and_continues(self):
        from app.api.jobs import Job, _overture_stage
        from app.models.schemas import JobRequest, BoundingBox
        job = Job(JobRequest(bbox=BoundingBox(west=0, south=0, east=1, north=1)))

        def boom():
            raise TimeoutError("overture dead")
        result = _overture_stage(job, "roads", [], boom)
        assert result == []
        assert job.progress["roads"] == "failed"
        assert "overture dead" in job.warnings[0]
        # And a healthy stage stays clean:
        assert _overture_stage(job, "water", [], lambda: ["poly"]) == ["poly"]
        assert job.progress["water"] == "done"
        assert len(job.warnings) == 1

    def test_warnings_flow_through_status(self):
        from app.api.jobs import Job
        from app.models.schemas import JobRequest, BoundingBox
        job = Job(JobRequest(bbox=BoundingBox(west=0, south=0, east=1, north=1)))
        job.warnings.append("roads: TimeoutError — continuing without")
        status = job.to_status()
        assert status.warnings == ["roads: TimeoutError — continuing without"]
