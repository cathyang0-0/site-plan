"""Tests for landtypes_seg.py — the supervised-segmentation land-type engine.

The model is never loaded here: every test feeds a synthetic OEM class map via
the `classmap=` hook, so CI never downloads weights or touches MPS. We exercise
the collapse (OEM 9-class -> our 4 labels), the mask exclusion, and the reused
polygon pipeline.
"""
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import MultiPolygon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.landtypes_seg import (
    detect_land_types_seg,
    OEM_TO_OURS,
    OEM_NAMES,
    LABEL_ORDER,
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
