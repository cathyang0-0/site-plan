"""Tests for footprints.py — geo-to-pixel conversion (fetch needs network)."""
import sys
from pathlib import Path

from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.footprints import footprints_to_pixels

BBOX = dict(west=-98.4720, south=29.4820, east=-98.4640, north=29.4880)


def _geo_square(lon0, lat0, lon1, lat1):
    return Polygon([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)])


class TestFootprintsToPixels:
    def test_corners_map_to_image_corners(self):
        # A polygon spanning the full bbox maps to the full image frame
        poly = _geo_square(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])
        [px] = footprints_to_pixels([poly], **BBOX, img_w=1000, img_h=800)
        assert px.bounds == (0.0, 0.0, 1000.0, 800.0)

    def test_north_maps_to_y_zero(self):
        # A polygon hugging the north edge lands at the TOP of the image (y=0)
        poly = _geo_square(BBOX["west"], 29.4870, BBOX["east"], BBOX["north"])
        [px] = footprints_to_pixels([poly], **BBOX, img_w=1000, img_h=800)
        assert px.bounds[1] == 0.0
        assert px.bounds[3] < 800 / 2

    def test_outside_bbox_dropped(self):
        poly = _geo_square(-98.5, 29.40, -98.49, 29.41)  # south of bbox
        assert footprints_to_pixels([poly], **BBOX, img_w=1000, img_h=800) == []

    def test_straddling_bbox_clipped(self):
        # Extends past the west edge; clipped result stays within the frame
        poly = _geo_square(-98.4730, 29.4840, -98.4710, 29.4850)
        [px] = footprints_to_pixels([poly], **BBOX, img_w=1000, img_h=800)
        assert px.bounds[0] == 0.0
        assert px.area > 0

    def test_empty_input(self):
        assert footprints_to_pixels([], **BBOX, img_w=1000, img_h=800) == []
