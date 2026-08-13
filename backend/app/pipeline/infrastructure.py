"""
Infrastructure shapes from Overture Maps — pier decks, bridges, breakwaters,
walls, parking aprons and the like (`infrastructure` type, `base` theme).

Why: buildings and roads on a wharf import fine, but the DECK under them
lives here — without this layer a coastal plan shows structures "floating"
on open water (see docs/data-coverage.md; Stearns Wharf is the canonical
example, present in this data as class pier/pier).

Only shape data is taken: Polygons and LineStrings (exploded from Multi*).
Point features (hydrants, benches, street lamps...) are dropped — they're
street furniture, not plan geometry.

Same infrastructure as the other Overture stages: disk cache per bbox and
retry on timeout (see overture_cache / fetch_with_retry). License: ODbL —
the export stamps attribution.
"""
import shapely
from shapely.geometry import Polygon, LineString

from app.pipeline.footprints import fetch_with_retry

ATTRIBUTION = "Infrastructure © OpenStreetMap contributors, Overture Maps Foundation (ODbL)"

# Subtypes excluded from the plan (user call, 2026-08-13): service networks
# and utility clutter that read as noise at site-plan scale, not built form.
# Everything else (pier, bridge, barrier walls/fences/kerbs, pedestrian
# structures, transit/parking aprons, breakwaters, airport surfaces) is kept.
EXCLUDED_SUBTYPES = frozenset({
    "power",             # overhead lines, pylons, substation gear
    "communication",     # comms lines/towers
    "utility",           # service pipelines/poles
    "manhole",           # covers
    "waste_management",  # bins, recycling pads
    "emergency",         # hydrants, call boxes
})


def subtype_included(subtype) -> bool:
    return subtype not in EXCLUDED_SUBTYPES


def fetch_infrastructure(west: float, south: float, east: float,
                         north: float) -> dict:
    """
    Fetch infrastructure shapes for a lon/lat bbox.

    Returns {"polygons": [Polygon], "lines": [LineString]} in lon/lat
    (EPSG:4326). Cached per bbox; retries timed-out fetches.
    """
    from app.pipeline import overture_cache
    bbox = (west, south, east, north)
    # v2: cache key bumped when EXCLUDED_SUBTYPES filtering was added, so
    # pre-filter cache entries (which contain power lines etc.) never serve.
    cached = overture_cache.get("infrastructure-v2", bbox)
    if cached is not None:
        return cached

    def _fetch():
        from overturemaps import core

        reader = core.record_batch_reader("infrastructure", bbox)
        polygons: list[Polygon] = []
        lines: list[LineString] = []
        for batch in reader:
            if batch.num_rows == 0:
                continue
            subtypes = (batch.column("subtype").to_pylist()
                        if "subtype" in batch.schema.names
                        else [None] * batch.num_rows)
            for wkb, subtype in zip(batch.column("geometry").to_pylist(), subtypes):
                if wkb is None or not subtype_included(subtype):
                    continue
                geom = shapely.from_wkb(wkb)
                for g in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                    if g.geom_type == "Polygon":
                        polygons.append(g)
                    elif g.geom_type == "LineString":
                        lines.append(g)
                    # Points (street furniture) intentionally dropped.
        return {"polygons": polygons, "lines": lines}

    data = fetch_with_retry(_fetch, "infrastructure fetch")
    overture_cache.put("infrastructure-v2", bbox, data)
    return data


def infrastructure_to_pixels(data: dict,
                             west: float, south: float, east: float, north: float,
                             img_w: int, img_h: int) -> dict:
    """Convert lon/lat infrastructure shapes to pixel coordinates of an image
    spanning exactly the bbox (y-down; same linear mapping as
    footprints_to_pixels), clipped to the frame."""
    from shapely.geometry import box
    from shapely.ops import transform

    frame = box(0, 0, img_w, img_h)
    sx = img_w / (east - west)
    sy = img_h / (north - south)

    def to_px(x, y):
        return ((x - west) * sx, (north - y) * sy)

    polygons, lines = [], []
    for poly in data.get("polygons") or []:
        px = transform(lambda x, y: to_px(x, y), poly).intersection(frame)
        for g in (px.geoms if hasattr(px, "geoms") else [px]):
            if g.geom_type == "Polygon" and not g.is_empty:
                polygons.append(g)
    for line in data.get("lines") or []:
        px = transform(lambda x, y: to_px(x, y), line).intersection(frame)
        for g in (px.geoms if hasattr(px, "geoms") else [px]):
            if g.geom_type == "LineString" and not g.is_empty and g.length > 0:
                lines.append(g)
    return {"polygons": polygons, "lines": lines}
