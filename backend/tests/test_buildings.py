"""Tests for buildings.py — mask_to_polygons and orthogonalize."""
import numpy as np
import pytest
from shapely.geometry import Polygon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.buildings import _mask_to_polygons, _orthogonalize


def _solid_rect_mask(h, w, x0, y0, x1, y1):
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


class TestMaskToPolygons:
    def test_single_rectangle(self):
        mask = _solid_rect_mask(200, 200, 50, 50, 150, 150)
        polys = _mask_to_polygons(mask, mask.shape)
        assert len(polys) == 1
        assert polys[0].area > 0

    def test_two_separated_rects(self):
        mask = np.zeros((300, 600), dtype=np.float32)
        mask[50:150, 50:150] = 1.0   # left rect
        mask[50:150, 400:500] = 1.0  # right rect
        polys = _mask_to_polygons(mask, mask.shape)
        assert len(polys) == 2

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((200, 200), dtype=np.float32)
        polys = _mask_to_polygons(mask, mask.shape)
        assert polys == []

    def test_tiny_blob_filtered_out(self):
        # 5×5 = 25 px < min_area_px=100
        mask = _solid_rect_mask(200, 200, 100, 100, 105, 105)
        polys = _mask_to_polygons(mask, mask.shape)
        assert polys == []

    def test_all_polygons_are_valid(self):
        mask = _solid_rect_mask(400, 400, 50, 50, 350, 350)
        polys = _mask_to_polygons(mask, mask.shape)
        for p in polys:
            assert p.is_valid
            assert not p.is_empty

    def test_returns_shapely_polygons(self):
        mask = _solid_rect_mask(300, 300, 80, 80, 220, 220)
        polys = _mask_to_polygons(mask, mask.shape)
        for p in polys:
            assert isinstance(p, Polygon)

    def test_custom_area_filter(self):
        # 12×12 = 144 px; passes default min=100 but below 200
        mask = _solid_rect_mask(200, 200, 50, 50, 62, 62)
        polys_default = _mask_to_polygons(mask, mask.shape, min_area_px=100)
        polys_strict = _mask_to_polygons(mask, mask.shape, min_area_px=200)
        assert len(polys_default) == 1
        assert len(polys_strict) == 0

    def test_binary_threshold(self):
        # Values at exactly 0.5 should be excluded (mask > 0.5)
        mask = np.full((200, 200), 0.5, dtype=np.float32)
        polys = _mask_to_polygons(mask, mask.shape)
        assert polys == []

        mask2 = np.full((200, 200), 0.6, dtype=np.float32)
        polys2 = _mask_to_polygons(mask2, mask2.shape)
        assert len(polys2) == 1


class TestOrthogonalize:
    def test_returns_polygon(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = _orthogonalize(poly)
        assert isinstance(result, Polygon)

    def test_already_rectangular_unchanged(self):
        # A perfect rectangle — orthogonalize should preserve it (pass-through impl)
        poly = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        result = _orthogonalize(poly)
        assert result.area == pytest.approx(poly.area, rel=1e-6)

    def test_valid_output(self):
        poly = Polygon([(0, 0), (10, 1), (11, 11), (1, 10)])
        result = _orthogonalize(poly)
        assert result.is_valid
        assert not result.is_empty

    def test_custom_threshold(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = _orthogonalize(poly, angle_threshold_deg=5.0)
        assert isinstance(result, Polygon)
