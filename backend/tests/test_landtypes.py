"""Tests for landtypes.py — mask helpers and feature extraction."""
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.landtypes import (
    _mask_to_multipolygon,
    _make_thumbnail,
    _extract_superpixel_features,
    detect_land_types,
    N_CLUSTERS,
    CLUSTER_LABELS,
)


def _white_image(h=200, w=200):
    return Image.fromarray(np.full((h, w, 3), 200, dtype=np.uint8))


def _solid_mask(h, w, x0, y0, x1, y1, val=255):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = val
    return m


class TestMaskToMultipolygon:
    def test_single_large_region(self):
        mask = _solid_mask(300, 300, 50, 50, 250, 250)
        mp = _mask_to_multipolygon(mask)
        assert not mp.is_empty
        assert mp.geoms

    def test_two_regions(self):
        mask = np.zeros((300, 600), dtype=np.uint8)
        mask[50:200, 50:200] = 255
        mask[50:200, 400:550] = 255
        mp = _mask_to_multipolygon(mask)
        assert len(list(mp.geoms)) == 2

    def test_empty_mask(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mp = _mask_to_multipolygon(mask)
        assert mp.is_empty

    def test_tiny_region_filtered(self):
        # 10×10 = 100 px < min_area_px=500
        mask = _solid_mask(200, 200, 100, 100, 110, 110)
        mp = _mask_to_multipolygon(mask, min_area_px=500)
        assert mp.is_empty

    def test_returns_multipolygon(self):
        mask = _solid_mask(300, 300, 50, 50, 250, 250)
        mp = _mask_to_multipolygon(mask)
        assert isinstance(mp, MultiPolygon)

    def test_all_geoms_valid(self):
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        mask[200:350, 200:350] = 255
        mp = _mask_to_multipolygon(mask)
        for geom in mp.geoms:
            assert geom.is_valid


class TestMakeThumbnail:
    def test_returns_pil_image(self):
        img_np = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        mask = _solid_mask(200, 200, 50, 50, 150, 150)
        thumb = _make_thumbnail(img_np, mask)
        assert isinstance(thumb, Image.Image)

    def test_correct_size(self):
        img_np = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        mask = _solid_mask(300, 300, 50, 50, 250, 250)
        thumb = _make_thumbnail(img_np, mask, size=64)
        assert thumb.size == (64, 64)

    def test_empty_mask_returns_gray(self):
        img_np = np.zeros((200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        thumb = _make_thumbnail(img_np, mask, size=32)
        assert thumb.size == (32, 32)
        # Should be gray placeholder
        arr = np.array(thumb)
        assert arr.mean() > 100  # 200 gray

    def test_custom_size(self):
        img_np = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        mask = _solid_mask(200, 200, 50, 50, 150, 150)
        for size in [32, 64, 128]:
            thumb = _make_thumbnail(img_np, mask, size=size)
            assert thumb.size == (size, size)


class TestExtractSuperpixelFeatures:
    def test_returns_arrays(self):
        img_np = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        from skimage.segmentation import slic
        segments = slic(img_np, n_segments=20, compactness=10, sigma=1, start_label=0)
        exclude = np.zeros((100, 100), dtype=np.uint8)
        features, valid_ids = _extract_superpixel_features(img_np, segments, exclude)
        assert isinstance(features, np.ndarray)
        assert isinstance(valid_ids, list)

    def test_feature_dimension(self):
        # Color-forward vector: [R, G, B, exg, blueness, sat, value, texture] = 8
        img_np = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        from skimage.segmentation import slic
        segments = slic(img_np, n_segments=20, compactness=10, sigma=1, start_label=0)
        exclude = np.zeros((100, 100), dtype=np.uint8)
        features, _ = _extract_superpixel_features(img_np, segments, exclude)
        assert features.shape[1] == 8

    def test_excluded_superpixels_skipped(self):
        img_np = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        from skimage.segmentation import slic
        segments = slic(img_np, n_segments=20, compactness=10, sigma=1, start_label=0)
        # Exclude everything
        exclude = np.ones((100, 100), dtype=np.uint8)
        features, valid_ids = _extract_superpixel_features(img_np, segments, exclude)
        assert len(features) == 0
        assert len(valid_ids) == 0

    def test_valid_ids_match_features(self):
        img_np = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        from skimage.segmentation import slic
        segments = slic(img_np, n_segments=20, compactness=10, sigma=1, start_label=0)
        exclude = np.zeros((100, 100), dtype=np.uint8)
        features, valid_ids = _extract_superpixel_features(img_np, segments, exclude)
        assert len(features) == len(valid_ids)


class TestDetectLandTypes:
    def test_four_color_regions_get_correct_labels(self):
        # Four quadrants: blue water, green vegetation, gray pavement, brown
        # bare earth. Each should cluster out and get its semantic label.
        a = np.zeros((200, 200, 3), dtype=np.uint8)
        a[:100, :100] = (30, 60, 200)    # water (blue)
        a[:100, 100:] = (40, 150, 40)    # vegetation (green)
        a[100:, :100] = (150, 150, 150)  # paved (gray)
        a[100:, 100:] = (150, 110, 60)   # bare earth (brown)
        img = Image.fromarray(a)
        zero = np.zeros((200, 200), dtype=np.uint8)
        results = detect_land_types(img, zero, zero)
        labels = {r["label"] for r in results}
        assert labels == set(CLUSTER_LABELS)  # all four, each mapped once

    def test_drops_empty_clusters(self):
        # A uniform image yields a single ground-cover region, not N_CLUSTERS.
        img = _white_image(200, 200)
        zero = np.zeros((200, 200), dtype=np.uint8)
        results = detect_land_types(img, zero, zero)
        assert 0 < len(results) <= N_CLUSTERS

    def test_result_has_required_keys(self):
        img = _white_image(200, 200)
        building_mask = np.zeros((200, 200), dtype=np.uint8)
        road_mask = np.zeros((200, 200), dtype=np.uint8)
        results = detect_land_types(img, building_mask, road_mask)
        for r in results:
            assert "cluster_id" in r
            assert "label" in r
            assert "polygons" in r
            assert "thumbnail" in r

    def test_all_excluded_returns_empty(self):
        img = _white_image(100, 100)
        # Exclude all pixels
        building_mask = np.ones((100, 100), dtype=np.uint8)
        road_mask = np.zeros((100, 100), dtype=np.uint8)
        results = detect_land_types(img, building_mask, road_mask)
        assert results == []

    def test_thumbnails_are_pil_images(self):
        img = _white_image(200, 200)
        building_mask = np.zeros((200, 200), dtype=np.uint8)
        road_mask = np.zeros((200, 200), dtype=np.uint8)
        results = detect_land_types(img, building_mask, road_mask)
        for r in results:
            assert isinstance(r["thumbnail"], Image.Image)

    def test_polygons_are_multipolygon(self):
        img = _white_image(200, 200)
        building_mask = np.zeros((200, 200), dtype=np.uint8)
        road_mask = np.zeros((200, 200), dtype=np.uint8)
        results = detect_land_types(img, building_mask, road_mask)
        for r in results:
            assert isinstance(r["polygons"], MultiPolygon)
