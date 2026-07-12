"""Tests for roads.py — geo->pixel conversion, class widths, hybrid width
blending, and the CV pavement-width measurement (fetch needs network)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline import roads

BBOX = dict(west=-98.4720, south=29.4820, east=-98.4640, north=29.4880)


class TestRoadsToPixels:
    def test_horizontal_road_maps_across_image(self):
        # A road spanning the full lon range at mid-lat -> horizontal line
        # across the middle of the image.
        line = LineString([(BBOX["west"], 29.4850), (BBOX["east"], 29.4850)])
        out = roads.roads_to_pixels(
            [{"line": line, "class": "residential"}], **BBOX, img_w=1000, img_h=800
        )
        assert len(out) == 1
        xs = [p[0] for p in out[0]["line"].coords]
        assert min(xs) == pytest.approx(0.0) and max(xs) == pytest.approx(1000.0)
        assert out[0]["class"] == "residential"

    def test_road_outside_bbox_dropped(self):
        line = LineString([(-98.50, 29.40), (-98.49, 29.40)])  # south of bbox
        assert roads.roads_to_pixels([{"line": line, "class": "service"}], **BBOX,
                                     img_w=1000, img_h=800) == []

    def test_straddling_road_clipped(self):
        line = LineString([(-98.4730, 29.4850), (-98.4700, 29.4850)])  # crosses west edge
        out = roads.roads_to_pixels([{"line": line, "class": "residential"}], **BBOX,
                                    img_w=1000, img_h=800)
        assert out and min(p[0] for p in out[0]["line"].coords) == pytest.approx(0.0)


class TestClassWidthPrior:
    def test_known_class(self):
        assert roads.CLASS_WIDTH_M["secondary"] > roads.CLASS_WIDTH_M["residential"]
        assert roads.CLASS_WIDTH_M["footway"] < roads.CLASS_WIDTH_M["residential"]

    def test_unknown_class_uses_default(self):
        # assign_widths without CV: prior only, unknown class -> DEFAULT_WIDTH_M
        line = LineString([(0, 0), (100, 0)])
        out = roads.assign_widths([{"line": line, "class": "mystery"}],
                                  image=None, scale_m_per_px=1.0, cv_informed=False)
        assert out[0]["width_px"] == pytest.approx(roads.DEFAULT_WIDTH_M)


class TestBlendWidth:
    def test_low_confidence_falls_back_to_prior(self):
        assert roads.blend_width(10.0, 30.0, confidence=0.1) == 10.0
        assert roads.blend_width(10.0, None, confidence=1.0) == 10.0

    def test_confident_measurement_blends(self):
        # prior 10, measured 20 (within clamp), weight 0.5 -> 15
        w = roads.blend_width(10.0, 20.0, confidence=1.0)
        assert w == pytest.approx(10.0 * (1 - roads.CV_WIDTH_WEIGHT)
                                  + 20.0 * roads.CV_WIDTH_WEIGHT)

    def test_measurement_clamped_to_band(self):
        # measured 100 is far above the 2x prior clamp -> clamped to 2*prior=20,
        # then blended: 0.5*10 + 0.5*20 = 15 (not runaway)
        assert roads.blend_width(10.0, 100.0, confidence=1.0) == pytest.approx(15.0)
        # measured 1 below 0.5x prior clamp -> clamped to 5, blend 0.5*10+0.5*5=7.5
        assert roads.blend_width(10.0, 1.0, confidence=1.0) == pytest.approx(7.5)


class TestMeasureRoadWidth:
    def _road_image(self, band_half=15):
        # green background, horizontal gray pavement band through the middle
        img = np.zeros((120, 200, 3), dtype=np.uint8)
        img[:, :] = (40, 140, 40)          # grass
        img[60 - band_half:60 + band_half, :] = (130, 130, 130)  # pavement
        return img

    def test_measures_band_width(self):
        img = self._road_image(band_half=15)  # 30px wide pavement
        line = LineString([(10, 60), (190, 60)])  # centerline down the band
        measured, conf = roads.measure_road_width_px(img, line, prior_width_px=30)
        assert conf == pytest.approx(1.0)
        assert measured == pytest.approx(30, abs=3)  # ~2*band_half

    def test_centerline_off_pavement_low_confidence(self):
        img = self._road_image(band_half=15)
        line = LineString([(10, 10), (190, 10)])  # up in the grass, no edges nearby
        measured, conf = roads.measure_road_width_px(img, line, prior_width_px=30)
        # uniform grass out to the search limit -> no clean edges -> unreliable
        assert conf < roads.CV_MIN_CONFIDENCE

    def test_wide_vs_narrow_ordering(self):
        wide = self._road_image(band_half=25)
        narrow = self._road_image(band_half=8)
        line = LineString([(10, 60), (190, 60)])
        mw, _ = roads.measure_road_width_px(wide, line, prior_width_px=60)
        mn, _ = roads.measure_road_width_px(narrow, line, prior_width_px=60)
        assert mw > mn
