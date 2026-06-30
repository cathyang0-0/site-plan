"""Tests for imagery.py — pure math functions."""
import math
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import io

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.imagery import lon_lat_to_tile, tile_to_lon_lat, meters_per_pixel, fetch_aerial_image, ZOOM, TILE_SIZE


class TestLonLatToTile:
    def test_prime_meridian_equator(self):
        x, y = lon_lat_to_tile(0.0, 0.0, zoom=1)
        assert x == 1
        assert y == 1

    def test_antimeridian_west(self):
        x, y = lon_lat_to_tile(-180.0, 0.0, zoom=0)
        assert x == 0
        assert y == 0

    def test_zoom_0_gives_single_tile(self):
        x, y = lon_lat_to_tile(0.0, 0.0, zoom=0)
        assert x == 0
        assert y == 0

    def test_known_new_york(self):
        # NYC approx lon=-74, lat=40.7 at zoom=10
        x, y = lon_lat_to_tile(-74.0, 40.7, zoom=10)
        # Tile x should be around 301, y around 384
        assert 295 <= x <= 310
        assert 378 <= y <= 390

    def test_returns_integers(self):
        x, y = lon_lat_to_tile(10.0, 50.0, zoom=5)
        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_x_increases_eastward(self):
        x1, _ = lon_lat_to_tile(-10.0, 0.0, zoom=5)
        x2, _ = lon_lat_to_tile(10.0, 0.0, zoom=5)
        assert x2 > x1

    def test_y_increases_southward(self):
        _, y1 = lon_lat_to_tile(0.0, 40.0, zoom=5)
        _, y2 = lon_lat_to_tile(0.0, -40.0, zoom=5)
        assert y2 > y1

    def test_higher_zoom_gives_more_tiles(self):
        x5, y5 = lon_lat_to_tile(10.0, 50.0, zoom=5)
        x10, y10 = lon_lat_to_tile(10.0, 50.0, zoom=10)
        # At zoom 10 indices should be roughly 2^5 = 32x larger
        assert x10 > x5
        assert y10 > y5


class TestTileToLonLat:
    def test_origin_tile(self):
        lon, lat = tile_to_lon_lat(0, 0, zoom=0)
        assert lon == pytest.approx(-180.0, abs=1e-6)
        assert lat == pytest.approx(85.05, abs=0.1)

    def test_roundtrip_consistency(self):
        """lon_lat_to_tile → tile_to_lon_lat should give a nearby lon/lat."""
        orig_lon, orig_lat = -74.0, 40.7
        tx, ty = lon_lat_to_tile(orig_lon, orig_lat, zoom=10)
        tile_lon, tile_lat = tile_to_lon_lat(tx, ty, zoom=10)
        # tile corner should be within one tile-width of original point
        assert abs(tile_lon - orig_lon) < 0.5
        assert abs(tile_lat - orig_lat) < 0.5

    def test_returns_floats(self):
        lon, lat = tile_to_lon_lat(5, 3, zoom=4)
        assert isinstance(lon, float)
        assert isinstance(lat, float)

    def test_longitude_range(self):
        for x in range(0, 8):
            lon, _ = tile_to_lon_lat(x, 4, zoom=3)
            assert -180 <= lon <= 180

    def test_latitude_range(self):
        for y in range(0, 8):
            _, lat = tile_to_lon_lat(4, y, zoom=3)
            assert -90 <= lat <= 90


class TestMetersPerPixel:
    def test_equator(self):
        mpp = meters_per_pixel(0.0, zoom=0)
        # At zoom 0, equator: ~156543 m/px
        assert mpp == pytest.approx(156543.0, rel=0.001)

    def test_increases_at_lower_zoom(self):
        mpp_z10 = meters_per_pixel(0.0, zoom=10)
        mpp_z5 = meters_per_pixel(0.0, zoom=5)
        assert mpp_z5 > mpp_z10

    def test_decreases_at_higher_latitude(self):
        mpp_equator = meters_per_pixel(0.0, zoom=10)
        mpp_polar = meters_per_pixel(60.0, zoom=10)
        assert mpp_polar < mpp_equator

    def test_zoom_doubles_resolution(self):
        mpp_z9 = meters_per_pixel(0.0, zoom=9)
        mpp_z10 = meters_per_pixel(0.0, zoom=10)
        assert mpp_z9 == pytest.approx(mpp_z10 * 2, rel=1e-9)

    def test_positive(self):
        assert meters_per_pixel(45.0, zoom=15) > 0


class TestFetchAerialImage:
    def _make_fake_tile(self):
        img = Image.new("RGB", (TILE_SIZE, TILE_SIZE), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    def _patched_fetch(self, west, south, east, north):
        """Run fetch_aerial_image with ZOOM=1 to keep the canvas small."""
        fake_tile_bytes = self._make_fake_tile()

        mock_resp = MagicMock()
        mock_resp.content = fake_tile_bytes
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        import app.pipeline.imagery as imagery_mod
        with patch("app.pipeline.imagery.httpx.Client", return_value=mock_client), \
             patch.object(imagery_mod, "ZOOM", 1):
            return fetch_aerial_image(west=west, south=south, east=east, north=north)

    def test_returns_image_and_scale(self):
        image, scale = self._patched_fetch(-74.0, 40.0, -73.0, 41.0)
        assert isinstance(image, Image.Image)
        assert isinstance(scale, float)
        assert scale > 0

    def test_stitched_image_dimensions(self):
        image, scale = self._patched_fetch(-74.0, 40.0, -73.0, 41.0)
        w, h = image.size
        # Width and height must be multiples of TILE_SIZE
        assert w % TILE_SIZE == 0
        assert h % TILE_SIZE == 0
        assert w >= TILE_SIZE
        assert h >= TILE_SIZE
