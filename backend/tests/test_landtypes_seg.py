"""Tests for landtypes_seg.py — the supervised-segmentation land-type engine.

The model is never loaded here: every test feeds a synthetic OEM class map via
the `classmap=` hook, so CI never downloads weights or touches MPS. We exercise
the collapse (OEM 9-class -> our 4 labels), the mask exclusion, and the reused
polygon pipeline.
"""
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.landtypes_seg import (
    detect_land_types_seg,
    _confident_classmap,
    _reclaim_water,
    _smooth_preserving_frame,
    OEM_TO_OURS,
    OEM_NAMES,
    LABEL_ORDER,
    PAVED_IDS,
    N_CLASSES,
)

# OEM class indices (see OEM_NAMES): 0 unknown, 1 Bareland, 2 Grass, 3 Pavement,
# 4 Road, 5 Tree, 6 Water, 7 Cropland, 8 buildings.


def _image(h, w):
    return Image.fromarray(np.full((h, w, 3), 180, dtype=np.uint8))


def _zeros(h, w):
    return np.zeros((h, w), dtype=np.uint8)


class TestCollapseMapping:
    def test_four_quadrant_classmap_gives_four_labels(self):
        # water | grass(veg) / bareland | pavement(paved)
        h = w = 200
        cm = np.zeros((h, w), dtype=np.uint8)
        cm[:100, :100] = 6   # Water
        cm[:100, 100:] = 2   # Grass -> vegetation
        cm[100:, :100] = 1   # Bareland -> bare earth / farmland
        cm[100:, 100:] = 3   # Pavement -> paved / hardscape
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        assert {r["label"] for r in results} == set(LABEL_ORDER)

    def test_tree_and_grass_merge_into_one_vegetation_region(self):
        # Grass (2) and Tree (5) both collapse to "vegetation" -> a single label.
        h = w = 200
        cm = np.full((h, w), 2, dtype=np.uint8)  # Grass
        cm[:, 100:] = 5                            # Tree
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        veg = [r for r in results if r["label"] == "vegetation"]
        assert len(veg) == 1  # not one per OEM class

    def test_pavement_and_road_merge_into_paved(self):
        h = w = 200
        cm = np.full((h, w), 3, dtype=np.uint8)  # Pavement
        cm[:, 100:] = 4                            # Road
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        assert [r["label"] for r in results] == ["paved / hardscape"]

    def test_buildings_class_is_ignored(self):
        # OEM `buildings` (8) must NOT become a land region — buildings come
        # from Overture and would double-count with the roof layer.
        h = w = 200
        cm = np.full((h, w), 8, dtype=np.uint8)
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        assert results == []

    def test_unknown_class_is_ignored(self):
        h = w = 200
        cm = np.zeros((h, w), dtype=np.uint8)  # all unknown (0)
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        assert results == []

    def test_mapping_covers_expected_classes(self):
        # 1..7 map to our labels; 0 and 8 are intentionally absent.
        assert set(OEM_TO_OURS) == {1, 2, 3, 4, 5, 6, 7}
        assert 0 not in OEM_TO_OURS and 8 not in OEM_TO_OURS
        assert set(OEM_TO_OURS.values()) == set(LABEL_ORDER)


class TestMaskExclusion:
    def test_building_mask_excludes_region(self):
        h = w = 200
        cm = np.full((h, w), 3, dtype=np.uint8)  # all Pavement
        building = np.ones((h, w), dtype=np.uint8)  # exclude everything
        results = detect_land_types_seg(_image(h, w), building, _zeros(h, w), classmap=cm)
        assert results == []

    def test_road_mask_excludes_region(self):
        h = w = 200
        cm = np.full((h, w), 6, dtype=np.uint8)  # all Water
        road = np.ones((h, w), dtype=np.uint8)
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), road, classmap=cm)
        assert results == []

    def test_partial_exclusion_keeps_remainder(self):
        h = w = 200
        cm = np.full((h, w), 6, dtype=np.uint8)  # all Water
        building = _zeros(h, w)
        building[:, :100] = 1  # exclude left half
        results = detect_land_types_seg(_image(h, w), building, _zeros(h, w), classmap=cm)
        assert [r["label"] for r in results] == ["water"]


class TestThinPavementSurvives:
    def test_thin_road_through_vegetation_is_detected(self):
        # A 3px-wide road stripe crossing a vegetation field. This is exactly
        # what k-means averages away; the seg path must keep it (paved morphology
        # skips the open and shrinks the close so thin hardscape survives).
        h = w = 200
        cm = np.full((h, w), 2, dtype=np.uint8)  # Grass everywhere
        cm[99:102, :] = 4                          # thin horizontal Road, ~600 px
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        labels = {r["label"] for r in results}
        assert "paved / hardscape" in labels
        assert "vegetation" in labels


class TestReclaimWater:
    def test_blue_bareland_becomes_water(self):
        # Bareland (1) over a blue-dominant pixel is really open water.
        cm = np.full((4, 4), 1, dtype=np.uint8)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[..., 2] = 200  # blue >> red
        out = _reclaim_water(cm, img, blue_margin=10)
        assert (out == 6).all()

    def test_brown_bareland_stays_bareland(self):
        # Real bareland is warm/brown (R > B) — must NOT be reclaimed.
        cm = np.full((4, 4), 1, dtype=np.uint8)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[..., 0] = 160  # red >> blue
        img[..., 2] = 90
        out = _reclaim_water(cm, img, blue_margin=10)
        assert (out == 1).all()

    def test_only_bareland_is_touched(self):
        # A blue Grass/Tree pixel must not be flipped — reclaim is bareland-only.
        cm = np.array([[2, 5]], dtype=np.uint8)  # Grass, Tree
        img = np.zeros((1, 2, 3), dtype=np.uint8)
        img[..., 2] = 200
        out = _reclaim_water(cm, img, blue_margin=10)
        assert list(out[0]) == [2, 5]

    def test_disabled_is_noop(self):
        cm = np.full((3, 3), 1, dtype=np.uint8)
        img = np.zeros((3, 3, 3), dtype=np.uint8); img[..., 2] = 255
        assert (_reclaim_water(cm, img, blue_margin=0) == 1).all()

    def test_does_not_mutate_input(self):
        cm = np.full((3, 3), 1, dtype=np.uint8)
        img = np.zeros((3, 3, 3), dtype=np.uint8); img[..., 2] = 200
        _reclaim_water(cm, img, blue_margin=10)
        assert (cm == 1).all()  # original untouched


class TestFrameFlush:
    def test_full_frame_region_is_flush_not_rounded(self):
        # A water region filling the whole image must polygonize to (nearly) the
        # frame rectangle: every exterior vertex sits on a frame edge, i.e. the
        # page edges are straight, not rounded inward.
        h = w = 200
        cm = np.full((h, w), 6, dtype=np.uint8)  # all Water
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        assert len(results) == 1
        poly = max(results[0]["polygons"].geoms, key=lambda g: g.area)
        assert poly.bounds == pytest.approx((0, 0, w, h), abs=1.0)
        xs = np.array(poly.exterior.coords)
        on_frame = (np.isclose(xs[:, 0], 0, atol=1) | np.isclose(xs[:, 0], w, atol=1)
                    | np.isclose(xs[:, 1], 0, atol=1) | np.isclose(xs[:, 1], h, atol=1))
        assert on_frame.all()  # no vertex pulled inward (no rounding at the edge)

    def test_smooth_preserving_frame_keeps_frame_corner_square(self):
        # A square filling the frame keeps its 4 corners after smoothing.
        W = H = 100
        sq = MultiPolygon([Polygon([(0, 0), (W, 0), (W, H), (0, H), (0, 0)])])
        out = _smooth_preserving_frame(sq, W, H, iterations=2)
        coords = set(map(tuple, np.round(np.array(out.geoms[0].exterior.coords)).tolist()))
        for corner in [(0, 0), (W, 0), (W, H), (0, H)]:
            assert corner in coords

    def test_interior_region_still_smoothed(self):
        # A blocky interior region (not touching the frame) should be rounded:
        # its smoothed corner no longer sits exactly at the original sharp corner.
        W = H = 300
        blob = MultiPolygon([Polygon([(100, 100), (200, 100), (200, 200), (100, 200),
                                      (100, 100)])])
        out = _smooth_preserving_frame(blob, W, H, iterations=2)
        coords = set(map(tuple, np.round(np.array(out.geoms[0].exterior.coords)).tolist()))
        assert (100, 100) not in coords  # interior corner got cut


def _prob_pixel(weights: dict) -> np.ndarray:
    """Build a length-N_CLASSES softmax-like vector from {class_idx: weight},
    normalized to sum to 1. Unlisted classes get ~0."""
    v = np.full(N_CLASSES, 1e-6, dtype=np.float32)
    for i, w in weights.items():
        v[i] = w
    return v / v.sum()


class TestConfidentClassmap:
    def test_confident_paved_survives(self):
        # Pavement(3) winning at 0.9 is well above the 0.5 gate -> stays paved.
        prob = _prob_pixel({3: 0.9, 6: 0.1})[:, None, None]  # (C,1,1)
        cls = _confident_classmap(prob, paved_min_conf=0.5)
        assert cls[0, 0] == 3

    def test_low_confidence_paved_demoted_to_runner_up(self):
        # Road(4) narrowly wins (0.4) over Water(6) at 0.35 -> below gate, so
        # the pixel becomes Water, not Road.
        prob = _prob_pixel({4: 0.40, 6: 0.35, 2: 0.25})[:, None, None]
        cls = _confident_classmap(prob, paved_min_conf=0.5)
        assert cls[0, 0] == 6  # Water, the best non-paved class

    def test_low_confidence_nonpaved_is_untouched(self):
        # A low-confidence WATER pixel (0.4) must stay water; the gate only
        # touches paved, so we don't poke holes in less-peaked water/veg.
        prob = _prob_pixel({6: 0.40, 2: 0.35, 3: 0.25})[:, None, None]
        cls = _confident_classmap(prob, paved_min_conf=0.5)
        assert cls[0, 0] == 6

    def test_gate_disabled_keeps_raw_argmax(self):
        prob = _prob_pixel({4: 0.40, 6: 0.35})[:, None, None]
        cls = _confident_classmap(prob, paved_min_conf=0.0)
        assert cls[0, 0] == 4  # weak paved kept when gate off

    def test_demotion_never_picks_a_paved_class(self):
        # Even if both paved classes have mass, a demoted pixel lands on a
        # non-paved class.
        prob = _prob_pixel({3: 0.30, 4: 0.30, 5: 0.25, 6: 0.15})[:, None, None]
        cls = _confident_classmap(prob, paved_min_conf=0.5)
        assert cls[0, 0] not in PAVED_IDS

    def test_mixed_field_only_weak_paved_changes(self):
        # Three pixels: confident paved, weak paved, confident water.
        p_conf_paved = _prob_pixel({3: 0.9, 6: 0.1})
        p_weak_paved = _prob_pixel({4: 0.40, 6: 0.35, 2: 0.25})
        p_water = _prob_pixel({6: 0.95, 2: 0.05})
        prob = np.stack([p_conf_paved, p_weak_paved, p_water], axis=1)[:, :, None]  # (C,3,1)
        cls = _confident_classmap(prob, paved_min_conf=0.5)
        assert list(cls[:, 0]) == [3, 6, 6]

    def test_returns_uint8_class_map(self):
        prob = np.random.rand(N_CLASSES, 5, 7).astype(np.float32)
        prob /= prob.sum(axis=0, keepdims=True)
        cls = _confident_classmap(prob, paved_min_conf=0.5)
        assert cls.shape == (5, 7)
        assert cls.dtype == np.uint8


class TestContract:
    def test_result_has_required_keys(self):
        h = w = 200
        cm = np.full((h, w), 6, dtype=np.uint8)
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        assert results
        for r in results:
            assert set(r) >= {"label", "polygons", "thumbnail"}

    def test_polygons_are_multipolygon(self):
        h = w = 200
        cm = np.full((h, w), 6, dtype=np.uint8)
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        for r in results:
            assert isinstance(r["polygons"], MultiPolygon)

    def test_thumbnails_are_pil_images(self):
        h = w = 200
        cm = np.full((h, w), 6, dtype=np.uint8)
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        for r in results:
            assert isinstance(r["thumbnail"], Image.Image)

    def test_polygons_in_full_image_coordinates(self):
        # Regions must land in the image rectangle (no downscale round-trip here).
        h, w = 240, 320
        cm = np.full((h, w), 6, dtype=np.uint8)
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        for r in results:
            minx, miny, maxx, maxy = r["polygons"].bounds
            assert minx >= -1 and miny >= -1
            assert maxx <= w + 1 and maxy <= h + 1

    def test_classmap_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            detect_land_types_seg(_image(200, 200), _zeros(200, 200), _zeros(200, 200),
                                  classmap=np.zeros((100, 100), dtype=np.uint8))

    def test_labels_emitted_in_canonical_order(self):
        # water, vegetation, bare earth, paved — stable order matching k-means.
        h = w = 200
        cm = np.zeros((h, w), dtype=np.uint8)
        cm[:100, :100] = 6   # water
        cm[:100, 100:] = 2   # vegetation
        cm[100:, :100] = 1   # bare earth
        cm[100:, 100:] = 3   # paved
        results = detect_land_types_seg(_image(h, w), _zeros(h, w), _zeros(h, w), classmap=cm)
        emitted = [r["label"] for r in results]
        assert emitted == [lb for lb in LABEL_ORDER if lb in emitted]
