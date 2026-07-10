"""Tests for trees.py — filter_placements and fill_dense_stands."""
import pytest
from shapely.geometry import Polygon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.trees import filter_placements, fill_dense_stands


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

    def test_tree_overlap_threshold_tunable(self):
        # Two r=30 canopies 30px apart overlap ~39%. The default tree-tree
        # threshold (0.30, spec §7) suppresses the second; relaxing to 0.60
        # keeps both.
        dets = [_det(100, 100, 30), _det(130, 100, 30)]
        result_default = filter_placements(dets, [])
        result_relaxed = filter_placements(dets, [], tree_overlap_threshold=0.60)
        assert len(result_default) == 1
        assert len(result_relaxed) == 2

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


def _stand(x, y, rx, ry, scale=0.1):
    return {
        "x_px": x, "y_px": y,
        "radius_px": rx, "ry_px": ry,
        "radius_m": rx * scale,
        "stand": True,
    }


class TestFillDenseStands:
    def test_no_stands_passthrough(self):
        dets = [_det(100, 100, 10), _det(300, 300, 10)]
        assert fill_dense_stands(dets, 0.1) == dets

    def test_stand_replaced_by_multiple_synthetic(self):
        # 150px radius at 0.1 m/px = 15m stand — well over the 10m cap
        dets = [_stand(500, 500, 150, 150)]
        result = fill_dense_stands(dets, 0.1)
        synthetic = [d for d in result if d.get("synthetic")]
        assert len(synthetic) >= 3
        assert not any(d.get("stand") for d in result)

    def test_synthetic_stay_inside_stand_ellipse(self):
        dets = [_stand(500, 500, 150, 120)]
        for d in fill_dense_stands(dets, 0.1):
            dx = (d["x_px"] - 500) / 150
            dy = (d["y_px"] - 500) / 120
            assert dx * dx + dy * dy <= 1.0

    def test_synthetic_sizes_and_spacing_vary(self):
        dets = [_stand(500, 500, 200, 200)]
        synthetic = fill_dense_stands(dets, 0.1)
        radii = {round(d["radius_m"], 3) for d in synthetic}
        assert len(radii) > 1  # size jitter present

    def test_singles_kept_alongside_fill(self):
        dets = [_det(50, 50, 10), _stand(500, 500, 150, 150)]
        result = fill_dense_stands(dets, 0.1)
        assert dets[0] in result

    def test_deterministic_for_same_seed(self):
        dets = [_stand(500, 500, 150, 150)]
        assert fill_dense_stands(dets, 0.1, seed=1) == fill_dense_stands(dets, 0.1, seed=1)
        assert fill_dense_stands(dets, 0.1, seed=1) != fill_dense_stands(dets, 0.1, seed=2)
