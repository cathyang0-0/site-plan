"""
Topographic contours — USGS 3DEP elevation grid → traced contour polylines.

The data has NO native interval: USGS serves a seamless elevation GRID (a
height sample per cell — ~10 m nationally, ~1 m where lidar exists), from the
same keyless National Map family as the aerial imagery. Contour lines are
traced from that grid by marching squares at ANY interval the caller picks —
interval is a user choice, not a data property.

Output polylines are in AERIAL-IMAGE PIXEL coordinates (the pipeline's working
frame), ready for export_dxf, which draws them below the land hatches per the
user's spec (near-hairline, very light gray, bottom-most).
"""
import io
import math

import httpx
import numpy as np
from PIL import Image

ELEVATION_URL = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
                 "3DEPElevation/ImageServer/exportImage")
# Elevation needs far less resolution than imagery: a site contour traced from
# ~3 m cells is smooth at plan scale, and requests stay small (<2 MB).
DEM_M_PER_PX = 3.0
DEM_MAX_PX = 4096

# Contours shorter than this (in aerial-image pixels of arc length) are noise
# specks from grid quantization — dropped.
MIN_CONTOUR_LEN_PX = 40.0


def fetch_elevation_usgs(west: float, south: float, east: float, north: float,
                         m_per_px: float = DEM_M_PER_PX) -> np.ndarray:
    """Fetch a float32 elevation grid (meters) spanning exactly the bbox.
    Keyless; US coverage. Returns (H, W) float32, north row first."""
    from siteplan_backend.pipeline.imagery import bbox_size_m
    width_m, height_m = bbox_size_m(west, south, east, north)
    px_w = min(DEM_MAX_PX, max(2, round(width_m / m_per_px)))
    px_h = min(DEM_MAX_PX, max(2, round(height_m / m_per_px)))
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",
        "imageSR": "3857",
        "size": f"{px_w},{px_h}",
        "format": "tiff",
        "pixelType": "F32",
        "f": "image",
    }
    # The export endpoint occasionally returns a truncated TIFF mid-transfer
    # (seen live: "image file is truncated (194 bytes not processed)") — a
    # fresh request succeeds, so retry the fetch+decode as one unit.
    last_exc = None
    for attempt in range(3):
        try:
            resp = httpx.get(ELEVATION_URL, params=params, timeout=120)
            resp.raise_for_status()
            dem = np.asarray(Image.open(io.BytesIO(resp.content)), dtype=np.float32)
            if dem.ndim != 2:
                dem = dem[..., 0]
            return dem
        except Exception as exc:
            last_exc = exc
    raise last_exc


def contour_levels(zmin: float, zmax: float, interval_m: float) -> list[float]:
    """Elevation levels to trace: multiples of the interval within the range
    (a 3.2–9.8 m range at 1.5 m interval → 4.5, 6.0, 7.5, 9.0)."""
    if not (interval_m > 0) or not math.isfinite(zmin) or not math.isfinite(zmax):
        return []
    first = math.ceil(zmin / interval_m) * interval_m
    if first <= zmin:   # a level AT the minimum traces a degenerate edge-hugger
        first += interval_m
    levels = []
    z = first
    while z < zmax:
        levels.append(round(z, 6))
        z += interval_m
    return levels


def _chaikin_open(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin corner-cutting for an OPEN polyline: endpoints stay fixed
    (an open contour ends at the frame edge and must keep touching it)."""
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * (len(pts) - 1), 2), dtype=pts.dtype)
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid[1:-1], pts[-1:]])
    return pts


def trace_contours(dem: np.ndarray, interval_m: float,
                   img_w: int, img_h: int,
                   min_len_px: float = MIN_CONTOUR_LEN_PX) -> list[dict]:
    """Trace contour polylines from an elevation grid, scaled into aerial-image
    pixel coordinates. Returns [{"points": [(x, y), ...], "level": z}, ...].

    Marching squares runs on the DEM grid; (row, col) vertices are scaled by
    the DEM→image size ratio so contours land exactly on the aerial frame.
    Light open-polyline Chaikin smooths the grid steps without moving ends.
    """
    finite = dem[np.isfinite(dem)]
    if finite.size == 0:
        return []
    levels = contour_levels(float(finite.min()), float(finite.max()), interval_m)
    if not levels:
        return []

    # skimage is pure marching squares here, but respect the project-wide
    # OpenMP guard anyway (torch/MPS may be live in this process).
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        from contextlib import nullcontext as threadpool_limits  # type: ignore

    dem_h, dem_w = dem.shape
    sx, sy = img_w / dem_w, img_h / dem_h
    out = []
    with threadpool_limits(1):
        from skimage.measure import find_contours
        for level in levels:
            for rc in find_contours(dem, level):
                xy = np.column_stack([rc[:, 1] * sx, rc[:, 0] * sy])
                seg = np.diff(xy, axis=0)
                if np.hypot(seg[:, 0], seg[:, 1]).sum() < min_len_px:
                    continue
                smooth = _chaikin_open(xy)
                out.append({"points": [tuple(p) for p in smooth],
                            "level": level})
    return out


def build_contours(west: float, south: float, east: float, north: float,
                   interval_m: float, img_w: int, img_h: int) -> list[dict]:
    """Full stage: fetch the elevation grid, trace at the interval, return
    image-pixel polylines for export."""
    dem = fetch_elevation_usgs(west, south, east, north)
    return trace_contours(dem, interval_m, img_w, img_h)
