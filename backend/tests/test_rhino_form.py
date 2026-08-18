"""Tests for rhino/siteplan_form.py — the dialog's pure-Python logic.
Same trick as test_rhino_client.py: the rhino/ dir is importable because
the module deliberately has no Eto/Rhino imports."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "rhino"))

import siteplan_form as form


class TestParseInterval:
    def test_feet(self):
        assert form.parse_interval("5ft") == pytest.approx(1.524)
        assert form.parse_interval("5 ft") == pytest.approx(1.524)

    def test_meters_bare_number(self):
        assert form.parse_interval("2") == 2.0
        assert form.parse_interval("1.5") == 1.5

    def test_zero_and_junk_mean_none(self):
        assert form.parse_interval("0") is None
        assert form.parse_interval("-3") is None
        assert form.parse_interval("banana") is None


class TestBboxArea:
    def test_equator_square(self):
        # 0.01 deg square at the equator: (0.01*111.32)^2 ≈ 1.239 km²
        bbox = {"west": 0, "south": 0, "east": 0.01, "north": 0.01}
        assert form.bbox_area_km2(bbox) == pytest.approx(1.239, rel=0.01)

    def test_shrinks_with_latitude(self):
        eq = {"west": 0, "south": 0, "east": 0.01, "north": 0.01}
        north = {"west": 0, "south": 60, "east": 0.01, "north": 60.01}
        assert form.bbox_area_km2(north) < form.bbox_area_km2(eq) * 0.6

    def test_default_site_is_sub_km2(self):
        # The Myers Point default bbox — the anchor for estimate tuning.
        bbox = {"west": -76.5515, "south": 42.5305,
                "east": -76.5415, "north": 42.5385}
        assert 0.4 < form.bbox_area_km2(bbox) < 1.0


class TestEstimateMinutes:
    def test_monotone_in_area(self):
        assert form.estimate_minutes(2.0, True) > form.estimate_minutes(1.0, True)
        assert form.estimate_minutes(2.0, False) > form.estimate_minutes(1.0, False)

    def test_trees_dominate(self):
        assert form.estimate_minutes(1.0, True) > 2 * form.estimate_minutes(1.0, False)


class TestBuildRequest:
    BBOX = {"west": -76.55, "south": 42.53, "east": -76.54, "north": 42.54}

    def test_defaults_full_plan(self):
        req = form.build_request(self.BBOX, contour_interval_m=1.524)
        assert set(req["layers"]) == {"roofs", "roads", "land_types",
                                      "contours", "infrastructure", "trees"}
        assert req["options"]["land_types_engine"] == "kmeans"
        assert req["options"]["crown_size_scale"] == 1.5
        assert req["style"]["contours"]["interval_m"] == pytest.approx(1.524)
        assert req["style"]["units"] == "m"
        # None-valued optionals are omitted, not sent as null.
        assert "size_variance" not in req["options"]
        assert "river_width_m" not in req["options"]

    def test_trees_off_drops_layer(self):
        req = form.build_request(self.BBOX, trees=False)
        assert "trees" not in req["layers"]

    def test_engine_off_drops_land_types(self):
        req = form.build_request(self.BBOX, land_engine="off")
        assert "land_types" not in req["layers"]
        assert "land_types_engine" not in req["options"]

    def test_no_contours_drops_layer(self):
        req = form.build_request(self.BBOX, contour_interval_m=None)
        assert "contours" not in req["layers"]
        assert "contours" not in req["style"]

    def test_widths_ride_through(self):
        widths = dict(form.ROAD_CLASS_DEFAULTS)
        widths["residential"] = 9.0
        req = form.build_request(self.BBOX, road_class_widths=widths,
                                 river_width_m=8.0)
        assert req["options"]["road_class_widths"]["residential"] == 9.0
        # full table sent, not a diff
        assert len(req["options"]["road_class_widths"]) == len(form.ROAD_CLASS_DEFAULTS)
        assert req["options"]["river_width_m"] == 8.0

    def test_matches_backend_schema(self):
        # The assembled body must validate against the real API contract.
        from app.models.schemas import JobRequest
        req = form.build_request(self.BBOX, contour_interval_m=1.524,
                                 road_class_widths=dict(form.ROAD_CLASS_DEFAULTS),
                                 river_width_m=5.0, size_variance=0.3,
                                 units="mm")
        parsed = JobRequest(**req)
        assert parsed.options.road_class_widths["motorway"] == 14.0
        assert parsed.style.units == "mm"


class TestRoadClassDefaults:
    def test_mirrors_backend_table(self):
        # The grid's defaults must equal the backend's priors, or the UI
        # would lie about what "unchanged" means.
        from app.pipeline.roads import CLASS_WIDTH_M
        assert dict(form.ROAD_CLASS_DEFAULTS) == CLASS_WIDTH_M

    def test_river_default_mirrors_backend(self):
        from app.pipeline.water import DEFAULT_RIVER_WIDTH_M
        assert form.DEFAULT_RIVER_WIDTH_M == DEFAULT_RIVER_WIDTH_M
