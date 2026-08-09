"""Tests for water.py — geometry-to-polygon logic (the fetch itself needs
network, so, like test_footprints, we test the pure conversion instead)."""
import sys
from pathlib import Path

import pytest
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, Point,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.water import (
    _water_polygons_from_geom,
    fetch_water_footprints,
    DEFAULT_RIVER_WIDTH_M,
    _M_PER_DEG_LAT,
    ATTRIBUTION,
)

# The half-width the real fetch uses, in degrees.
HALF = (DEFAULT_RIVER_WIDTH_M / 2) / _M_PER_DEG_LAT


class TestWaterPolygonsFromGeom:
    def test_polygon_passes_through_unchanged(self):
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        out = _water_polygons_from_geom(poly, HALF)
        assert len(out) == 1 and out[0] is poly  # lake polygon untouched

    def test_multipolygon_splits_into_parts(self):
        a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        b = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        out = _water_polygons_from_geom(MultiPolygon([a, b]), HALF)
        assert len(out) == 2
        assert all(p.geom_type == "Polygon" for p in out)

    def test_linestring_is_buffered_into_a_strip(self):
        # A river centerline (1 degree long, vertical) becomes a filled strip.
        line = LineString([(0, 0), (0, 1)])
        out = _water_polygons_from_geom(line, HALF)
        assert len(out) == 1
        poly = out[0]
        assert poly.geom_type == "Polygon"
        assert poly.area > 0
        # Width across the line is a half-width on each side.
        minx, _, maxx, _ = poly.bounds
        assert (maxx - minx) == pytest.approx(2 * HALF, rel=0.05)

    def test_multilinestring_each_part_buffered(self):
        mls = MultiLineString([[(0, 0), (0, 1)], [(5, 5), (5, 6)]])
        out = _water_polygons_from_geom(mls, HALF)
        assert len(out) >= 1
        assert all(p.geom_type == "Polygon" and p.area > 0 for p in out)

    def test_wider_half_width_gives_bigger_strip(self):
        line = LineString([(0, 0), (1, 0)])
        thin = _water_polygons_from_geom(line, 0.001)[0]
        wide = _water_polygons_from_geom(line, 0.01)[0]
        assert wide.area > thin.area

    def test_point_yields_nothing(self):
        # Non-areal, non-linear leftovers are dropped, not crashed on.
        assert _water_polygons_from_geom(Point(0, 0), HALF) == []


class TestModuleContract:
    def test_default_river_width_is_positive(self):
        assert DEFAULT_RIVER_WIDTH_M > 0

    def test_attribution_credits_odbl(self):
        # The export stamps this; ODbL requires the credit line.
        assert "ODbL" in ATTRIBUTION
        assert "OpenStreetMap" in ATTRIBUTION

    def test_fetch_is_callable_with_bbox_signature(self):
        # Guard the public contract (4 lon/lat floats) without hitting network.
        import inspect
        params = list(inspect.signature(fetch_water_footprints).parameters)
        assert params == ["west", "south", "east", "north"]
