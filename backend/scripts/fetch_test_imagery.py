"""Fetch test aerial imagery from the USGS National Map (public domain, no
API key) — a stand-in for the Mapbox pipeline while no MAPBOX_TOKEN is set.

Uses the ArcGIS `export` endpoint rather than cached tiles: the tile cache
404s at high zoom in many areas, while export renders any bbox at any
resolution (up to the service's 4096px request limit) in one request.

US coverage only. Prints the m/px scale to pass to poc.py --scale.

Run:
    python scripts/fetch_test_imagery.py WEST SOUTH EAST NORTH M_PER_PX OUT.png
e.g.
    python scripts/fetch_test_imagery.py -98.4720 29.4820 -98.4640 29.4880 0.3 aerial.png
"""
import math
import sys

import httpx

EXPORT_URL = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/export"
MAX_EXPORT_PX = 4096  # service-side request limit


def bbox_size_m(west: float, south: float, east: float, north: float) -> tuple[float, float]:
    lat_mid = math.radians((south + north) / 2)
    width_m = (east - west) * 111_320 * math.cos(lat_mid)
    height_m = (north - south) * 111_320
    return width_m, height_m


def fetch(west, south, east, north, m_per_px, out_path):
    width_m, height_m = bbox_size_m(west, south, east, north)
    px_w = round(width_m / m_per_px)
    px_h = round(height_m / m_per_px)
    if max(px_w, px_h) > MAX_EXPORT_PX:
        sys.exit(f"Requested {px_w}x{px_h}px exceeds the {MAX_EXPORT_PX}px export limit; "
                 f"use a smaller bbox or coarser m_per_px.")

    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",
        "imageSR": "3857",
        "size": f"{px_w},{px_h}",
        "format": "png",
        "f": "image",
    }
    resp = httpx.get(EXPORT_URL, params=params, timeout=120)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"saved {out_path}  {px_w}x{px_h}px  site {width_m:.0f}x{height_m:.0f}m")
    print(f"scale: {width_m / px_w:.4f} m/px  →  poc.py --scale {width_m / px_w:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    w, s, e, n, scale = (float(a) for a in sys.argv[1:6])
    fetch(w, s, e, n, scale, sys.argv[6])
