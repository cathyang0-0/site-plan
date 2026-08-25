"""Tests for the Overture-outage resilience trio: retry, disk cache, and
warn-and-continue job degradation. Motivated by a real multi-hour Overture
slow spell that failed every job at the 90 s fetch timeout."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from siteplan_backend.pipeline import overture_cache
from siteplan_backend.pipeline.footprints import fetch_with_retry


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(overture_cache, "CACHE_DIR", tmp_path / "cache")


class TestRetry:
    def test_succeeds_after_timeouts(self, monkeypatch):
        calls = []
        import siteplan_backend.pipeline.footprints as fp

        def fake_fetch_with_timeout(fn, timeout_s, what):
            calls.append(what)
            if len(calls) < 3:
                raise TimeoutError("slow")
            return "result"
        monkeypatch.setattr(fp, "fetch_with_timeout", fake_fetch_with_timeout)
        assert fetch_with_retry(lambda: None, "x", attempts=3) == "result"
        assert len(calls) == 3

    def test_raises_after_all_attempts(self, monkeypatch):
        import siteplan_backend.pipeline.footprints as fp
        monkeypatch.setattr(fp, "fetch_with_timeout",
                            lambda fn, t, w: (_ for _ in ()).throw(TimeoutError("dead")))
        with pytest.raises(TimeoutError):
            fetch_with_retry(lambda: None, "x", attempts=2)

    def test_non_timeout_error_not_retried(self, monkeypatch):
        calls = []
        import siteplan_backend.pipeline.footprints as fp

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
        # The water cache stores RAW geometries ("water_raw") so river strips
        # can be re-buffered at any width; an areal polygon passes through.
        from shapely.geometry import Polygon
        from siteplan_backend.pipeline.water import fetch_water_footprints
        lake = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        overture_cache.put("water_raw", (-1.0, -1.0, 1.0, 1.0), [lake])
        [out] = fetch_water_footprints(-1.0, -1.0, 1.0, 1.0)
        assert out.equals(lake)


class TestAuthoritativeWater:
    def _land(self):
        from shapely.geometry import Polygon, MultiPolygon
        veg = MultiPolygon([Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])])
        detected_water = MultiPolygon([Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])])
        return [
            {"label": "vegetation", "polygons": veg, "style": {}},
            {"label": "water", "polygons": detected_water, "style": {}},
        ]

    def test_carves_water_out_of_other_covers(self):
        # Overture water overlapping the vegetation must bite it away — no
        # land hatch may overlap the water hatch (the bug seen in Rhino).
        from shapely.geometry import Polygon
        from siteplan_backend.pipeline.landtypes import apply_authoritative_water
        water = [Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])]  # right half of veg
        out = apply_authoritative_water(self._land(), water)
        assert out[0]["label"] == "water"          # authoritative water first
        veg = [lt for lt in out if lt["label"] == "vegetation"][0]
        assert veg["polygons"].area == pytest.approx(50)   # half carved away
        assert not veg["polygons"].intersects(out[0]["polygons"].buffer(-1e-9))

    def test_replaces_detected_water(self):
        from shapely.geometry import Polygon
        from siteplan_backend.pipeline.landtypes import apply_authoritative_water
        water = [Polygon([(50, 50), (60, 50), (60, 60), (50, 60)])]
        out = apply_authoritative_water(self._land(), water)
        waters = [lt for lt in out if lt["label"] == "water"]
        assert len(waters) == 1
        assert waters[0]["polygons"].geoms[0].bounds == (50, 50, 60, 60)

    def test_swallowed_cover_dropped(self):
        from shapely.geometry import Polygon
        from siteplan_backend.pipeline.landtypes import apply_authoritative_water
        water = [Polygon([(-1, -1), (11, -1), (11, 11), (-1, 11)])]  # covers veg fully
        out = apply_authoritative_water(self._land(), water)
        assert [lt["label"] for lt in out] == ["water"]

    def test_no_water_is_noop(self):
        from siteplan_backend.pipeline.landtypes import apply_authoritative_water
        land = self._land()
        assert apply_authoritative_water(land, []) is land


class TestDegradation:
    def test_failed_stage_warns_and_continues(self):
        from siteplan_backend.api.jobs import Job, _overture_stage
        from siteplan_backend.models.schemas import JobRequest, BoundingBox
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
        from siteplan_backend.api.jobs import Job
        from siteplan_backend.models.schemas import JobRequest, BoundingBox
        job = Job(JobRequest(bbox=BoundingBox(west=0, south=0, east=1, north=1)))
        job.warnings.append("roads: TimeoutError — continuing without")
        status = job.to_status()
        assert status.warnings == ["roads: TimeoutError — continuing without"]
