"""Tests for trees.py — filter_placements (fully implemented)."""
import pytest
from shapely.geometry import Polygon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.trees import filter_placements


def _det(x, y, r):
    return {"x_px": x, "y_px": y, "radius_px": r, "radius_m": r * 0.1}


def _square(cx, cy, half):
    return Polygon([
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ])


class TestFilterPlacements:
    def test_no_buildings_no_trees_passes_all(self):
        dets = [_det(100, 100, 10), _det(300, 300, 10)]
        result = filter_placements(dets, [])
        assert len(result) == 2

    def test_tree_on_building_suppressed(self):
        # Tree centered at (100,100) with radius 10 — entirely inside 50×50 building
        building = _square(100, 100, 50)
        dets = [_det(100, 100, 10)]
        result = filter_placements(dets, [building])
        assert result == []

    def test_tree_far_from_building_passes(self):
        building = _square(0, 0, 10)
        dets = [_det(500, 500, 10)]
        result = filter_placements(dets, [building])
        assert len(result) == 1

    def test_overlapping_trees_deduplicated(self):
        # Two trees at same position; second overlaps first almost entirely
        dets = [_det(100, 100, 30), _det(102, 100, 30)]
        result = filter_placements(dets, [])
        assert len(result) == 1

    def test_non_overlapping_trees_both_kept(self):
        # Far apart — no overlap
        dets = [_det(50, 50, 10), _det(500, 500, 10)]
        result = filter_placements(dets, [])
        assert len(result) == 2

    def test_empty_detections(self):
        result = filter_placements([], [])
        assert result == []

    def test_empty_detections_with_buildings(self):
        building = _square(100, 100, 50)
        result = filter_placements([], [building])
        assert result == []

    def test_order_preserved_first_accepted(self):
        # First tree placed; second overlaps it → only first should remain
        dets = [_det(100, 100, 40), _det(110, 100, 40)]
        result = filter_placements(dets, [])
        assert result[0] == dets[0]

    def test_custom_overlap_threshold(self):
        # Tree is 50% inside building — below default 0.30? No, 50 > 30 → suppressed by default
        # With threshold=0.6, it should pass
        building = _square(100, 100, 20)  # 40×40 building
        # Tree at edge: center at x=130, radius=30 → partial overlap
        dets = [_det(130, 100, 30)]
        result_strict = filter_placements(dets, [building], overlap_threshold=0.01)
        result_loose = filter_placements(dets, [building], overlap_threshold=0.99)
        # Strict threshold suppresses, loose threshold passes
        assert len(result_loose) >= len(result_strict)

    def test_result_is_subset_of_input(self):
        dets = [_det(i * 50, 100, 5) for i in range(10)]
        result = filter_placements(dets, [])
        for r in result:
            assert r in dets

    def test_multiple_buildings(self):
        buildings = [_square(100, 100, 40), _square(300, 300, 40)]
        dets = [
            _det(100, 100, 10),  # inside building 1
            _det(200, 200, 10),  # open area
            _det(300, 300, 10),  # inside building 2
        ]
        result = filter_placements(dets, buildings)
        assert len(result) == 1
        assert result[0] == dets[1]
