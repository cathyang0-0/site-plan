"""
Building footprints from Overture Maps (primary building source for
georeferenced sites; the SAM2/U-Net CV path in buildings.py is the
fallback for non-georeferenced input and very recent construction).

Overture Maps Foundation publishes a unified open map dataset (buildings
theme merges OpenStreetMap, Microsoft ML footprints, Google Open
Buildings, and USGS lidar) as cloud-hosted GeoParquet, re-released
regularly. The `overturemaps` package queries it with bbox pushdown:
only the row groups whose bbox metadata intersect the query are read, so
a site-sized query transfers a few hundred KB and takes ~1 second — no
local mirror of the dataset is needed.

License: ODbL. A generated site plan is an ODbL "Produced Work" — the
caller must display attribution (see ATTRIBUTION), but the drawing itself
is not share-alike encumbered.
"""
import shapely
from shapely.geometry import Polygon

ATTRIBUTION = "Building footprints © OpenStreetMap contributors, Overture Maps Foundation (ODbL)"

# Overture is a cloud GeoParquet query with no built-in timeout; a slow or
# unreachable backend can otherwise hang the whole job indefinitely (observed:
# a fetch stuck at 0% CPU for over an hour). A legitimate fetch is ~1-45s, so
# this ceiling catches true hangs without tripping on normal slowness.
OVERTURE_TIMEOUT_S = 90.0


def fetch_with_timeout(fn, timeout_s: float, what: str):
    """Run a blocking network fetch `fn` on a daemon thread and abandon it if
    it exceeds `timeout_s`, raising TimeoutError. The orphaned thread can't be
    force-killed but is daemonized so it never blocks process exit -- fail-fast
    beats hanging forever. Signal-free, so it is safe inside server threads."""
    import threading

    box = {}

    def run():
        try:
            box["result"] = fn()
        except BaseException as exc:  # propagate the real fetch error to caller
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"{what} timed out after {timeout_s:.0f}s (Overture/network slow or unreachable)")
    if "error" in box:
        raise box["error"]
    return box["result"]


# Overture's slow spells recover on a FRESH attempt more often than on a
# longer wait (observed 2026-08: attempt 1 dead at 420 s, attempt 2 fine in
# seconds) — so retry timed-out fetches rather than raising the timeout.
OVERTURE_ATTEMPTS = 3


def fetch_with_retry(fn, what: str, attempts: int = OVERTURE_ATTEMPTS,
                     timeout_s: float = OVERTURE_TIMEOUT_S):
    """fetch_with_timeout, retried on TimeoutError with a fresh connection.
    Non-timeout errors raise immediately (a bad bbox won't get better by
    looping). Each abandoned attempt leaves its daemon thread behind — see
    fetch_with_timeout; acceptable for a handful of retries."""
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return fetch_with_timeout(fn, timeout_s, what)
        except TimeoutError as exc:
            last = exc
            if attempt < attempts:
                print(f"  {what}: attempt {attempt}/{attempts} timed out; retrying...")
    raise last


def fetch_building_footprints(
    west: float, south: float, east: float, north: float
) -> list[Polygon]:
    """
    Fetch building footprint polygons for a lon/lat bounding box.

    Returns polygons in lon/lat (WGS84) coordinates; convert with
    footprints_to_pixels for pipeline use. Multipolygons are split into
    their parts. Successful results are disk-cached per bbox (see
    overture_cache) so repeat runs skip the network entirely; a miss
    retries timed-out fetches (raises TimeoutError only after all attempts).
    """
    from app.pipeline import overture_cache
    bbox = (west, south, east, north)
    cached = overture_cache.get("building", bbox)
    if cached is not None:
        return cached

    def _fetch():
        from overturemaps import core

        reader = core.record_batch_reader("building", (west, south, east, north))
        polygons: list[Polygon] = []
        for batch in reader:
            if batch.num_rows == 0:
                continue
            for wkb in batch.column("geometry").to_pylist():
                if wkb is None:
                    continue
                geom = shapely.from_wkb(wkb)
                if geom.geom_type == "Polygon":
                    polygons.append(geom)
                elif geom.geom_type == "MultiPolygon":
                    polygons.extend(geom.geoms)
        return polygons

    polygons = fetch_with_retry(_fetch, "building footprints fetch")
    overture_cache.put("building", bbox, polygons)
    return polygons


def footprints_to_pixels(
    geo_polygons: list[Polygon],
    west: float, south: float, east: float, north: float,
    img_w: int, img_h: int,
    clip: bool = True,
) -> list[Polygon]:
    """
    Convert lon/lat footprint polygons to pixel coordinates of an image
    that spans exactly (west, south, east, north), y-down.

    Uses a linear lon/lat mapping: over a site-sized extent the difference
    from true Web Mercator is far below one pixel. With clip=True (default),
    footprints straddling the bbox edge are clipped to the image, and any
    entirely outside are dropped.
    """
    from shapely.geometry import box
    from shapely.ops import transform

    frame = box(0, 0, img_w, img_h)
    sx = img_w / (east - west)
    sy = img_h / (north - south)

    def to_px(x, y):
        return ((x - west) * sx, (north - y) * sy)

    result = []
    for poly in geo_polygons:
        px_poly = transform(lambda x, y: to_px(x, y), poly)
        if clip:
            px_poly = px_poly.intersection(frame)
        if px_poly.is_empty:
            continue
        if px_poly.geom_type == "Polygon":
            result.append(px_poly)
        elif px_poly.geom_type == "MultiPolygon":
            result.extend(px_poly.geoms)
    return result
