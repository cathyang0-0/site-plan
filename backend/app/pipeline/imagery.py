"""
Aerial imagery for a lon/lat bounding box.

Two sources:
- fetch_aerial_usgs — USGS National Map (public domain, keyless, US only).
  The default: it's what the test imagery has always used, and no token is
  configured in this environment.
- fetch_aerial_image — Mapbox Satellite tiles (global, needs MAPBOX_TOKEN).
"""
import os
import math
import httpx
from PIL import Image
import io

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")
TILE_SIZE = 512
ZOOM = 19

USGS_EXPORT_URL = ("https://basemap.nationalmap.gov/arcgis/rest/services/"
                   "USGSImageryOnly/MapServer/export")
USGS_MAX_EXPORT_PX = 4096  # service-side request limit


def bbox_size_m(west: float, south: float, east: float, north: float) -> tuple[float, float]:
    """Approximate width/height of a lon/lat bbox in meters (site scale)."""
    lat_mid = math.radians((south + north) / 2)
    width_m = (east - west) * 111_320 * math.cos(lat_mid)
    height_m = (north - south) * 111_320
    return width_m, height_m


def usgs_export_size_px(west: float, south: float, east: float, north: float,
                        m_per_px: float) -> tuple[int, int]:
    """Pixel dimensions the USGS export needs for this bbox at this scale.
    Raises ValueError past the service's request limit (pure function so the
    API can validate a request before starting a job)."""
    width_m, height_m = bbox_size_m(west, south, east, north)
    px_w, px_h = round(width_m / m_per_px), round(height_m / m_per_px)
    if max(px_w, px_h) > USGS_MAX_EXPORT_PX:
        raise ValueError(
            f"bbox needs {px_w}x{px_h}px at {m_per_px} m/px which exceeds the "
            f"USGS {USGS_MAX_EXPORT_PX}px export limit; use a smaller bbox or "
            f"coarser scale")
    if min(px_w, px_h) < 1:
        raise ValueError("bbox is empty or degenerate")
    return px_w, px_h


def fetch_aerial_usgs(west: float, south: float, east: float, north: float,
                      m_per_px: float = 0.3):
    """
    Fetch one aerial image spanning exactly the bbox from the USGS National
    Map export endpoint (public domain, no API key; US coverage only).

    Returns (image, scale_m_per_px) — the actual scale of the returned image,
    computed from the bbox width / pixel width.
    """
    px_w, px_h = usgs_export_size_px(west, south, east, north, m_per_px)
    width_m, _ = bbox_size_m(west, south, east, north)
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",
        "imageSR": "3857",
        "size": f"{px_w},{px_h}",
        "format": "png",
        "f": "image",
    }
    resp = httpx.get(USGS_EXPORT_URL, params=params, timeout=120)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return image, width_m / image.width


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
