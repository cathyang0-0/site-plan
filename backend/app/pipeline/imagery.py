"""
Fetch and stitch Mapbox Satellite tiles for a given bounding box.
Returns a single PIL Image and the pixel-to-meter scale.
"""
import os
import math
import httpx
from PIL import Image
import io

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")
TILE_SIZE = 512
ZOOM = 19


def lon_lat_to_tile(lon: float, lat: float, zoom: int):
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tile_to_lon_lat(x: int, y: int, zoom: int):
    n = 2 ** zoom
    lon = x / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def meters_per_pixel(lat: float, zoom: int) -> float:
    """Earth circumference at this latitude / number of pixels."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def fetch_aerial_image(west: float, south: float, east: float, north: float):
    """
    Fetch Mapbox Satellite tiles covering the bounding box and stitch them.

    Returns:
        image: PIL.Image (RGB)
        scale_m_per_px: float — real-world meters per pixel
        origin_m: tuple (x, y) — top-left corner in meters (local coords)
    """
    # Tile y grows southward, so the north edge maps to the smallest tile y.
    x_min, y_min = lon_lat_to_tile(west, north, ZOOM)
    x_max, y_max = lon_lat_to_tile(east, south, ZOOM)

    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    canvas = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))

    with httpx.Client() as client:
        for tx in range(x_min, x_max + 1):
            for ty in range(y_min, y_max + 1):
                url = (
                    f"https://api.mapbox.com/v4/mapbox.satellite"
                    f"/{ZOOM}/{tx}/{ty}@2x.png"
                    f"?access_token={MAPBOX_TOKEN}"
                )
                resp = client.get(url, timeout=10)
                resp.raise_for_status()
                tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                px = (tx - x_min) * TILE_SIZE
                py = (ty - y_min) * TILE_SIZE
                canvas.paste(tile_img, (px, py))

    center_lat = (north + south) / 2
    scale_m_per_px = meters_per_pixel(center_lat, ZOOM)

    return canvas, scale_m_per_px
