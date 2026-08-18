"""Tests for trees.py — filter_placements and fill_dense_stands."""
import pytest
from shapely.geometry import Polygon

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.trees import (
    filter_placements,
    fill_dense_stands,
    apply_size_transform,
    rescale_placements,
    MIN_CANOPY_RADIUS_M,
)


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

    def test_small_tree_inside_large_suppressed_either_order(self):
        # A small crown fully inside a large one must be suppressed no matter
        # which is processed first (regression: overlap was measured against
        # the incoming canopy, so a big tree containing a small accepted one
        # slipped through -> tree drawn inside a tree).
        small = _det(100, 100, 3)
        big = _det(101, 100, 12)  # engulfs `small`
        assert len(filter_placements([small, big], [])) == 1  # small first
        assert len(filter_placements([big, small], [])) == 1  # big first

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


class TestApplySizeTransform:
    # _det(x, y, r) -> radius_px = r, radius_m = r * 0.1. So radii 100/150/250
    # give radius_m 10/15/25 etc., all comfortably above MIN_CANOPY_RADIUS_M.

    def test_variance_zero_collapses_to_mean(self):
        # radius_m 5, 15, 25 -> mean 15. v=0 -> every crown equals the mean.
        dets = [_det(0, 0, 50), _det(0, 0, 150), _det(0, 0, 250)]
        apply_size_transform(dets, size_variance=0.0)
        assert all(d["radius_m"] == pytest.approx(15.0) for d in dets)
        # radius_px moves with radius_m (shared per-detection scale 0.1)
        assert all(d["radius_px"] == pytest.approx(150.0) for d in dets)

    def test_variance_one_is_unchanged(self):
        dets = [_det(0, 0, 100), _det(0, 0, 150), _det(0, 0, 200)]
        before_m = [d["radius_m"] for d in dets]
        before_px = [d["radius_px"] for d in dets]
        apply_size_transform(dets, size_variance=1.0)
        assert [d["radius_m"] for d in dets] == pytest.approx(before_m)
        assert [d["radius_px"] for d in dets] == pytest.approx(before_px)

    def test_defaults_are_noop(self):
        dets = [_det(0, 0, 100), _det(0, 0, 200)]
        before = [d["radius_m"] for d in dets]
        apply_size_transform(dets)
        assert [d["radius_m"] for d in dets] == pytest.approx(before)

    def test_variance_two_doubles_deviations(self):
        # radius_m 10, 15, 20 -> mean 15, deviations -5, 0, +5.
        # v=2 doubles them: 5, 15, 25 (none hit the floor).
        dets = [_det(0, 0, 100), _det(0, 0, 150), _det(0, 0, 200)]
        apply_size_transform(dets, size_variance=2.0)
        assert [d["radius_m"] for d in dets] == pytest.approx([5.0, 15.0, 25.0])

    def test_floor_clamps_small_crown(self):
        # radius_m 2, 15 -> mean 8.5. v=3 sends the small one to
        # 8.5 + 3*(-6.5) = -11, which must clamp to MIN_CANOPY_RADIUS_M.
        dets = [_det(0, 0, 20), _det(0, 0, 150)]
        apply_size_transform(dets, size_variance=3.0)
        assert dets[0]["radius_m"] == pytest.approx(MIN_CANOPY_RADIUS_M)
        assert dets[1]["radius_m"] == pytest.approx(28.0)

    def test_crown_scale_multiplies_whole_result(self):
        # v=1 (spread unchanged), average x2 -> every radius doubles.
        dets = [_det(0, 0, 100), _det(0, 0, 200)]
        apply_size_transform(dets, crown_size_scale=2.0)
        assert [d["radius_m"] for d in dets] == pytest.approx([20.0, 40.0])

    def test_floor_applies_before_average_scale(self):
        # Floor acts on the post-variance radius, THEN average scales: the
        # clamped crown ends at MIN * crown_size_scale, not MIN.
        dets = [_det(0, 0, 20), _det(0, 0, 150)]  # radius_m 2, 15; mean 8.5
        apply_size_transform(dets, crown_size_scale=2.0, size_variance=3.0)
        assert dets[0]["radius_m"] == pytest.approx(MIN_CANOPY_RADIUS_M * 2.0)

    def test_stands_excluded_and_passed_through(self):
        stand = _stand(0, 0, 300, 300)  # radius_m 30, flagged stand
        dets = [_det(0, 0, 100), _det(0, 0, 200), stand]  # singles mean 15
        apply_size_transform(dets, size_variance=0.0)
        assert dets[0]["radius_m"] == pytest.approx(15.0)
        assert dets[1]["radius_m"] == pytest.approx(15.0)
        # stand untouched: mean computed from singles only, box left as-is
        assert stand["radius_m"] == pytest.approx(30.0)
        assert stand["radius_px"] == pytest.approx(300.0)
        assert stand.get("stand") is True

    def test_empty_list(self):
        assert apply_size_transform([]) == []

    def test_only_stands_passthrough(self):
        dets = [_stand(0, 0, 300, 300)]
        apply_size_transform(dets, size_variance=0.0, crown_size_scale=2.0)
        assert dets[0]["radius_m"] == pytest.approx(30.0)


class TestSuppressOverWater:
    def test_tree_in_water_removed(self):
        from app.pipeline.trees import suppress_over_water
        from shapely.geometry import Polygon
        water = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        dets = [_det(50, 50, 5), _det(500, 500, 5)]  # first inside water
        out = suppress_over_water(dets, [water])
        assert len(out) == 1 and out[0]["x_px"] == 500

    def test_no_water_is_noop(self):
        from app.pipeline.trees import suppress_over_water
        dets = [_det(50, 50, 5)]
        assert suppress_over_water(dets, []) == dets

    def test_empty_detections(self):
        from app.pipeline.trees import suppress_over_water
        from shapely.geometry import Polygon
        assert suppress_over_water([], [Polygon([(0, 0), (1, 0), (1, 1)])]) == []


class TestRescalePlacements:
    """Export-time twin of apply_size_transform (the preview sliders)."""

    def _pl(self, *scales):
        return [{"block_idx": 0, "position": (i, i), "scale": s,
                 "rotation": 0.0} for i, s in enumerate(scales)]

    def test_neutral_is_identity_above_floor(self):
        out = rescale_placements(self._pl(1.0, 2.0), mean_scale=1.5,
                                 crown_size_scale=1.0, size_variance=1.0,
                                 min_scale=0.1)
        assert [p["scale"] for p in out] == [1.0, 2.0]

    def test_variance_zero_collapses_to_mean(self):
        out = rescale_placements(self._pl(1.0, 2.0), mean_scale=1.5,
                                 crown_size_scale=1.0, size_variance=0.0,
                                 min_scale=0.1)
        assert [p["scale"] for p in out] == [1.5, 1.5]

    def test_crown_scale_multiplies(self):
        out = rescale_placements(self._pl(1.0, 2.0), mean_scale=1.5,
                                 crown_size_scale=2.0, size_variance=1.0,
                                 min_scale=0.1)
        assert [p["scale"] for p in out] == [2.0, 4.0]

    def test_floor_applies_before_crown_scale(self):
        # matches apply_size_transform's order: variance → floor → average
        out = rescale_placements(self._pl(0.2), mean_scale=1.0,
                                 crown_size_scale=2.0, size_variance=2.0,
                                 min_scale=0.5)
        # variance: 1.0 + 2*(0.2-1.0) = -0.6 → floored to 0.5 → ×2 = 1.0
        assert out[0]["scale"] == pytest.approx(1.0)

    def test_cache_not_mutated(self):
        src = self._pl(1.0)
        rescale_placements(src, 1.0, 3.0, 0.5, 0.1)
        assert src[0]["scale"] == 1.0   # baseline stays neutral
