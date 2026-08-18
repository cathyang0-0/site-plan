"""
Form logic for the SitePlan dialog — pure Python, NO Eto/Rhino imports.

Everything the dialog computes or assembles that doesn't touch a widget
lives here, for the same reason siteplan_client.py is pure stdlib: it can
be tested headless with pytest (backend/tests/test_rhino_form.py) while
the Eto file stays a thin skin. If you're adding dialog behavior, ask
"does this need a widget?" — if not, it belongs in this file.
"""
import math

# Mirrors backend roads.CLASS_WIDTH_M (meters). Kept as an ordered list so
# the grid shows a sensible hierarchy (big roads first). Drift from the
# backend table is harmless by construction: the dialog always sends the
# FULL dict, so what the user sees in the grid is exactly what applies.
ROAD_CLASS_DEFAULTS = [
    ("motorway", 14.0), ("trunk", 12.0), ("primary", 11.0),
    ("secondary", 10.0), ("tertiary", 8.0), ("residential", 6.0),
    ("living_street", 5.0), ("unclassified", 6.0), ("service", 4.0),
    ("driveway", 3.5), ("parking_aisle", 4.0), ("alley", 3.0),
    ("track", 3.0), ("pedestrian", 4.0), ("footway", 1.8),
    ("sidewalk", 1.8), ("path", 1.5), ("cycleway", 2.0), ("steps", 1.5),
]

DEFAULT_RIVER_WIDTH_M = 5.0   # mirrors backend water.DEFAULT_RIVER_WIDTH_M

# Rough job-time model (minutes), tuned against real runs of the default
# ~0.66 km2 bbox: ~5-6 min with trees (DeepForest dominates), well under a
# minute without. Both terms scale with area because detection cost scales
# with pixel count. TUNE ME as more sites are timed.
_MINUTES_BASE_TREES, _MINUTES_PER_KM2_TREES = 1.5, 6.0
_MINUTES_BASE, _MINUTES_PER_KM2 = 1.0, 1.5


def parse_interval(raw):
    """'5ft' / '5 ft' → meters; a bare number is meters already.
    Returns None for unparsable or non-positive input (= no contours)."""
    raw = raw.strip().lower()
    feet = raw.endswith("ft")
    if feet:
        raw = raw[:-2].strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value * 0.3048 if feet else value


def bbox_area_km2(bbox):
    """Approximate area of a lon/lat bbox in km². Spherical shortcut: a
    degree of latitude is ~111.32 km everywhere; a degree of longitude
    shrinks by cos(latitude). Plenty accurate at site scale."""
    dlat = bbox["north"] - bbox["south"]
    dlon = bbox["east"] - bbox["west"]
    mid_lat = math.radians((bbox["north"] + bbox["south"]) / 2)
    return (dlat * 111.32) * (dlon * 111.32 * math.cos(mid_lat))


def estimate_minutes(area_km2, trees_on):
    """Rough minutes for a job. Labeled 'rough' in the UI — the point is
    setting expectations (minutes, not seconds; grows with area), not
    precision."""
    if trees_on:
        return _MINUTES_BASE_TREES + _MINUTES_PER_KM2_TREES * area_km2
    return _MINUTES_BASE + _MINUTES_PER_KM2 * area_km2


def build_request(bbox, trees=True, land_engine="kmeans",
                  contour_interval_m=None, crown_size_scale=None,
                  size_variance=None, road_class_widths=None,
                  river_width_m=None, units="m"):
    """Assemble the POST /api/jobs body from dialog values.

    Returns {"bbox", "layers", "options", "style"} ready for
    siteplan_client.submit_job(**request). Encoding rules:
    - trees=False drops the "trees" layer (detection never runs).
    - land_engine "off" drops the "land_types" layer; otherwise it names
      the engine ("kmeans" | "segmodel").
    - contour_interval_m None drops the "contours" layer.
    - crown_size_scale/size_variance None are OMITTED → detection runs
      NEUTRAL. That's the dialog's path since the preview: sizes are chosen
      on the preview page and applied at export (export_style below), so
      one detection serves every slider position.
    - river_width_m None is omitted so the backend default applies.
    """
    layers = ["roofs", "roads", "land_types", "contours", "infrastructure"]
    if trees:
        layers.append("trees")
    options = {}
    if crown_size_scale is not None:
        options["crown_size_scale"] = crown_size_scale

    if land_engine == "off":
        layers.remove("land_types")
    else:
        options["land_types_engine"] = land_engine

    style = {"units": units}
    if contour_interval_m is None:
        layers.remove("contours")
    else:
        style["contours"] = {"interval_m": contour_interval_m}

    if size_variance is not None:
        options["size_variance"] = size_variance
    if road_class_widths:
        options["road_class_widths"] = dict(road_class_widths)
    if river_width_m is not None:
        options["river_width_m"] = river_width_m

    return {"bbox": dict(bbox), "layers": layers,
            "options": options, "style": style}


def export_style(request, crown_size_scale, size_variance):
    """Style body for POST /jobs/{id}/export after the preview: the
    original request's style (units, contours — they must survive the
    re-render) plus the preview sliders as EXPORT-TIME tree sizes.
    The backend re-applies the size transform to its cached neutral
    placements, so the DXF matches the preview exactly."""
    style = dict(request["style"])
    style["trees"] = {"crown_size_scale": crown_size_scale,
                      "size_variance": size_variance}
    return style
