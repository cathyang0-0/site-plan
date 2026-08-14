"""
Infrastructure shapes from Overture Maps — pier decks, bridges, breakwaters,
walls/fences, parking aprons, airfield pavement, ski lifts (`infrastructure`
type, `base` theme).

Per-class treatment (user-specified 2026-08-13, after reviewing rendered
samples of every class):

  class                geometry   treatment
  -------------------  ---------  -------------------------------------------
  pier                 poly+line  draw as-is, 0.30 mm, no offset
  bridge               poly       as-is, road lineweight (0.18)
  bridge               line       OFFSET to bridge width, road lineweight
  breakwater           line       OFFSET, rounded ends, thinner than road
  wall/retaining wall  line       OFFSET to wall thickness, 0.08 mm
  fence                line       OFFSET (smaller), 0.04 mm
  kerb                 line       hairline stroke, no offset
  toilets/artwork/...  poly       dropped if overlapping a building; else 0.10
  aerialway (lifts)    line       very thin stroke, no offset
  runway/taxiway       line       OFFSET like a road, standard widths

"OFFSET" means the centerline is buffered into a real strip whose OUTLINE is
drawn — a wall reads as two edges the wall's thickness apart, never as one
fat stroke pretending to have width.

Same plumbing as the other Overture stages: disk cache per bbox (key v3 —
bumped when per-class data replaced the flat polygon/line split) and retry on
timeout. License: ODbL — the export stamps attribution.
"""
import math

import shapely
from shapely.geometry import Polygon, LineString

from app.pipeline.footprints import fetch_with_retry

ATTRIBUTION = "Infrastructure © OpenStreetMap contributors, Overture Maps Foundation (ODbL)"

# Service networks and utility clutter: not built form, never fetched.
EXCLUDED_SUBTYPES = frozenset({
    "power", "communication", "utility", "manhole",
    "waste_management", "emergency",
})

# Airport features are whitelisted by class: pavement yes, admin/zoning
# boundaries (municipal_airport etc. — a giant lot outline) no.
AIRPORT_CLASSES = frozenset({
    "runway", "taxiway", "taxilane", "stopway", "apron", "helipad",
})

# Real-world strip widths (meters) for the OFFSET classes.
WIDTH_M = {
    "bridge": 8.0,        # a road-carrying deck
    "breakwater": 6.0,    # harbor breakwater crest
    "wall": 0.4,          # masonry/retaining wall thickness
    "fence": 0.15,        # fence line
    "runway": 45.0,       # FAA standard commercial runway
    "taxiway": 15.0,
    "taxilane": 10.0,
    "stopway": 45.0,
}

# Style groups: outline weight (mm) and whether the group's polygons join the
# built-structure clip mask (hatches/contours/roads stop at their boundary).
# Thin barrier ribbons deliberately DON'T clip — a 0.15 m fence slit across a
# road corridor or hatch would be an invisible defect generator.
GROUP_STYLE = {
    "structure":  {"weight_mm": 0.30, "clip": True},   # piers, aprons, parking...
    "bridge":     {"weight_mm": 0.18, "clip": True},
    "airfield":   {"weight_mm": 0.18, "clip": True},
    "breakwater": {"weight_mm": 0.13, "clip": True},
    "micro":      {"weight_mm": 0.10, "clip": True},   # toilets/artwork (if kept)
    "lift":       {"weight_mm": 0.09, "clip": False},
    "wall":       {"weight_mm": 0.08, "clip": False},
    "kerb":       {"weight_mm": 0.05, "clip": False},  # hairline
    "fence":      {"weight_mm": 0.04, "clip": False},
}

_WALL_CLASSES = {"wall", "retaining_wall", "city_wall"}
_FENCE_CLASSES = {"fence", "cable_barrier", "guard_rail", "handrail"}


def subtype_included(subtype) -> bool:
    return subtype not in EXCLUDED_SUBTYPES


def classify(subtype, cls, geom_type):
    """Map an Overture (subtype, class, geometry type) to a (group, width_m)
    pair; width_m is None for stroke-only treatment, a number for classes
    whose centerline must be OFFSET into a strip. Returns None to drop."""
    if not subtype_included(subtype):
        return None
    if subtype == "airport":
        if cls not in AIRPORT_CLASSES:
            return None                       # admin boundaries etc.
        if geom_type == "Polygon":
            return ("airfield", None)
        return ("airfield", WIDTH_M.get(cls, WIDTH_M["taxiway"]))
    if subtype == "bridge":
        return ("bridge", None if geom_type == "Polygon" else WIDTH_M["bridge"])
    if subtype == "water":                    # breakwaters, dams, groynes
        return ("breakwater",
                None if geom_type == "Polygon" else WIDTH_M["breakwater"])
    if subtype == "aerialway":
        return ("structure", None) if geom_type == "Polygon" else ("lift", None)
    if subtype == "barrier":
        if cls == "kerb":
            return ("kerb", None)
        if cls in _WALL_CLASSES:
            return ("wall", None if geom_type == "Polygon" else WIDTH_M["wall"])
        # everything else barrier-ish reads as light fencing
        return ("fence", None if geom_type == "Polygon" else WIDTH_M["fence"])
    if subtype == "pedestrian":
        return ("micro", None)                # small structures; overlap-culled
    # pier, transit, transportation, towers, anything else: built structure
    return ("structure", None)


def fetch_infrastructure(west: float, south: float, east: float,
                         north: float) -> dict:
    """
    Fetch infrastructure shapes for a lon/lat bbox.

    Returns {"features": [{"subtype", "class", "geom"}...]} in lon/lat
    (EPSG:4326), shapes only (Points dropped), excluded subtypes filtered.
    Cached per bbox; retries timed-out fetches.
    """
    from app.pipeline import overture_cache
    bbox = (west, south, east, north)
    cached = overture_cache.get("infrastructure-v3", bbox)
    if cached is not None:
        return cached

    def _fetch():
        from overturemaps import core

        reader = core.record_batch_reader("infrastructure", bbox)
        features = []
        for batch in reader:
            if batch.num_rows == 0:
                continue
            cols = batch.schema.names
            subs = (batch.column("subtype").to_pylist()
                    if "subtype" in cols else [None] * batch.num_rows)
            classes = (batch.column("class").to_pylist()
                       if "class" in cols else [None] * batch.num_rows)
            for wkb, subtype, cls in zip(batch.column("geometry").to_pylist(),
                                         subs, classes):
                if wkb is None or not subtype_included(subtype):
                    continue
                geom = shapely.from_wkb(wkb)
                for g in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                    if g.geom_type in ("Polygon", "LineString"):
                        features.append({"subtype": subtype, "class": cls,
                                         "geom": g})
        return {"features": features}

    data = fetch_with_retry(_fetch, "infrastructure fetch")
    overture_cache.put("infrastructure-v3", bbox, data)
    return data


def infrastructure_to_pixels(data: dict,
                             west: float, south: float, east: float, north: float,
                             img_w: int, img_h: int) -> dict:
    """Convert fetched features to pixel space and sort them into style
    groups, buffering the OFFSET classes into strips at real-world widths.

    Returns {"groups": {name: {"weight_mm", "clip", "polys", "lines"}}} —
    the export draws each group's outlines at its weight; groups flagged
    `clip` join the built-structure mask.
    """
    from shapely.geometry import box
    from shapely.ops import transform

    frame = box(0, 0, img_w, img_h)
    sx = img_w / (east - west)
    sy = img_h / (north - south)
    lat_mid = math.radians((south + north) / 2)
    m_per_px = ((east - west) * 111_320 * math.cos(lat_mid)) / img_w

    def to_px(x, y):
        return ((x - west) * sx, (north - y) * sy)

    groups: dict = {}

    def bucket(name):
        if name not in groups:
            groups[name] = {**GROUP_STYLE[name], "polys": [], "lines": []}
        return groups[name]

    for feat in data.get("features") or []:
        g = feat["geom"]
        decision = classify(feat["subtype"], feat["class"], g.geom_type)
        if decision is None:
            continue
        group, width_m = decision
        px = transform(lambda x, y: to_px(x, y), g)
        if width_m is not None and px.geom_type == "LineString":
            # Buffer the centerline into a strip. Breakwaters get rounded
            # ends (user spec); engineered strips end square.
            cap = 1 if group == "breakwater" else 2   # 1=round, 2=flat
            px = px.buffer((width_m / 2) / m_per_px, cap_style=cap)
        px = px.intersection(frame)
        for part in (px.geoms if hasattr(px, "geoms") else [px]):
            if part.is_empty:
                continue
            if part.geom_type == "Polygon":
                bucket(group)["polys"].append(part)
            elif part.geom_type == "LineString" and part.length > 0:
                bucket(group)["lines"].append(part)
    return {"groups": groups}
