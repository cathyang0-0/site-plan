"""
Road centerlines from Overture Maps (primary road source for georeferenced
sites; a CV segmentation pipeline is the fallback for non-georeferenced
input -- see detect_roads_cv, not yet implemented).

Overture's transportation "segment" theme already provides what the old CV
plan (segment -> skeletonize -> graph -> spline -> fillet) tried to
reconstruct: clean, connected centerlines, one entity per road, correct at
intersections, classified by type. We fetch them per bbox (same GeoParquet
mechanism as footprints.py), convert to pixels, and assign each road a
width.

Width is hybrid -- a class-based prior nudged by a light CV measurement of
the actual pavement in the image (assign_widths / measure_road_width_px),
so unusually wide or narrow roads aren't forced to the class default. The
class prior anchors it and a clamp bounds the CV, so a noisy measurement
can't blow a road up.

License: ODbL (same as footprints); the export stamps attribution.
"""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, box
from shapely.ops import transform

ATTRIBUTION = "Roads © OpenStreetMap contributors, Overture Maps Foundation (ODbL)"

# Typical full pavement width (m) per Overture road class. Values follow
# spec §4b's "recommended by type" guidance (footpath ~1-2, driveway ~3-5,
# road ~5-10). Used as the prior that the CV measurement adjusts.
CLASS_WIDTH_M = {
    "motorway": 14.0, "trunk": 12.0, "primary": 11.0, "secondary": 10.0,
    "tertiary": 8.0, "residential": 6.0, "living_street": 5.0,
    "unclassified": 6.0, "service": 4.0, "driveway": 3.5,
    "parking_aisle": 4.0, "alley": 3.0, "track": 3.0, "pedestrian": 4.0,
    "footway": 1.8, "sidewalk": 1.8, "path": 1.5, "cycleway": 2.0, "steps": 1.5,
}
DEFAULT_WIDTH_M = 5.0

# CV width blending. The measurement pulls the width toward the observed
# pavement, but only CV_WIDTH_WEIGHT of the way and only within CV_CLAMP x
# the class prior -- "slightly CV-informed", with the class prior as anchor.
# Lower CV_WIDTH_WEIGHT -> more class-driven; raise it -> more image-driven.
CV_WIDTH_WEIGHT = 0.5
CV_MIN_CONFIDENCE = 0.5          # need this fraction of clean perpendicular reads
CV_CLAMP = (0.5, 2.0)           # measured allowed within [lo, hi] x prior
CV_COLOR_THRESHOLD = 45.0        # RGB distance from centerline color = pavement edge
CV_SAMPLES_PER_ROAD = 9
# Excess-green (2G-R-B) above this means the centerline pixel is vegetation,
# not pavement (tree overhang, grass median) -- skip that sample so we don't
# "measure" a grass gap. Pavement sits near 0; grass is strongly positive.
CV_VEGETATION_EXG = 40.0


def fetch_road_network(west: float, south: float, east: float, north: float) -> list[dict]:
    """
    Fetch road centerlines for a lon/lat bbox from Overture's segment theme.

    Returns list of {"line": lon/lat LineString, "class": str}, roads only
    (rail and other subtypes dropped). Requires network access; raises
    TimeoutError past OVERTURE_TIMEOUT_S.
    """
    from app.pipeline.footprints import fetch_with_retry
    from app.pipeline import overture_cache

    bbox = (west, south, east, north)
    cached = overture_cache.get("segment", bbox)
    if cached is not None:
        return cached

    def _fetch():
        from overturemaps import core

        reader = core.record_batch_reader("segment", (west, south, east, north))
        roads: list[dict] = []
        for batch in reader:
            if batch.num_rows == 0:
                continue
            geoms = batch.column("geometry").to_pylist()
            subtypes = batch.column("subtype").to_pylist()
            classes = batch.column("class").to_pylist()
            for wkb, subtype, cls in zip(geoms, subtypes, classes):
                if wkb is None or subtype != "road":
                    continue
                geom = shapely.from_wkb(wkb)
                if geom.geom_type == "LineString":
                    roads.append({"line": geom, "class": cls})
                elif geom.geom_type == "MultiLineString":
                    for part in geom.geoms:
                        roads.append({"line": part, "class": cls})
        return roads

    roads = fetch_with_retry(_fetch, "road network fetch")
    overture_cache.put("segment", bbox, roads)
    return roads


def roads_to_pixels(
    roads: list[dict],
    west: float, south: float, east: float, north: float,
    img_w: int, img_h: int,
    clip: bool = True,
) -> list[dict]:
    """
    Convert lon/lat road lines to pixel coordinates of an image spanning
    exactly (west, south, east, north), y-down. Same linear mapping as
    footprints_to_pixels. Roads straddling the bbox edge are clipped to the
    image; parts outside are dropped. `class` is carried through.
    """
    frame = box(0, 0, img_w, img_h)
    sx = img_w / (east - west)
    sy = img_h / (north - south)

    def to_px(x, y):
        return ((x - west) * sx, (north - y) * sy)

    result = []
    for road in roads:
        px_line = transform(lambda x, y: to_px(x, y), road["line"])
        if clip:
            px_line = px_line.intersection(frame)
        for part in _iter_lines(px_line):
            if part.length > 0:
                result.append({"line": part, "class": road["class"]})
    return result


def _iter_lines(geom):
    """Yield LineString parts from a possibly-clipped geometry."""
    if geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            if g.geom_type == "LineString" and not g.is_empty:
                yield g


def assign_widths(
    roads_px: list[dict],
    image,
    scale_m_per_px: float,
    cv_informed: bool = True,
) -> list[dict]:
    """
    Give each pixel-space road a width_px: a class prior, optionally nudged
    by a CV pavement measurement. Returns export-ready
    {"line": LineString(px), "width_px": float, "class": str} dicts.
    """
    img_np = np.asarray(image.convert("RGB")) if cv_informed else None
    out = []
    for road in roads_px:
        prior_px = CLASS_WIDTH_M.get(road["class"], DEFAULT_WIDTH_M) / scale_m_per_px
        if cv_informed:
            measured_px, confidence = measure_road_width_px(img_np, road["line"], prior_px)
            width_px = blend_width(prior_px, measured_px, confidence)
        else:
            width_px = prior_px
        out.append({"line": road["line"], "width_px": width_px, "class": road["class"]})
    return out


def blend_width(prior_px: float, measured_px, confidence: float) -> float:
    """Blend a CV-measured width toward the class prior. Falls back to the
    prior when the measurement is missing or low-confidence; otherwise clamps
    the measurement to CV_CLAMP x prior and mixes by CV_WIDTH_WEIGHT."""
    if measured_px is None or confidence < CV_MIN_CONFIDENCE:
        return prior_px
    lo, hi = CV_CLAMP[0] * prior_px, CV_CLAMP[1] * prior_px
    measured_c = min(max(measured_px, lo), hi)
    return (1 - CV_WIDTH_WEIGHT) * prior_px + CV_WIDTH_WEIGHT * measured_c


def measure_road_width_px(
    img_np: np.ndarray,
    line: LineString,
    prior_width_px: float,
    n_samples: int = CV_SAMPLES_PER_ROAD,
    color_threshold: float = CV_COLOR_THRESHOLD,
):
    """
    Estimate a road's pavement width from the image by sampling perpendicular
    to its centerline.

    At evenly spaced points along the line, step outward on both sides from
    the centerline pixel until the color diverges from it (pavement -> grass/
    roof/etc.); the two half-distances sum to a width sample. Returns
    (median width_px, confidence), where confidence is the fraction of
    samples that found a clean edge on both sides within the search range.
    Samples that run to the search limit (centerline not on pavement, or a
    huge uniform lot) are discarded rather than trusted.
    """
    H, W = img_np.shape[:2]
    length = line.length
    if length < 2:
        return None, 0.0
    max_half = max(prior_width_px * 1.5, 4.0)

    widths = []
    for i in range(n_samples):
        s = (i + 0.5) / n_samples * length
        p = line.interpolate(s)
        pa = line.interpolate(max(s - 1.0, 0.0))
        pb = line.interpolate(min(s + 1.0, length))
        tx, ty = pb.x - pa.x, pb.y - pa.y
        tlen = math.hypot(tx, ty)
        if tlen == 0:
            continue
        nx, ny = -ty / tlen, tx / tlen  # unit perpendicular
        cx, cy = p.x, p.y
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        ref = img_np[int(cy), int(cx)].astype(float)
        # Skip if the centerline pixel is vegetation (tree over the road, grass
        # median): measuring outward from a grass reference finds a meaningless
        # "gap width". Under-canopy roads fall back to the class prior.
        if 2 * ref[1] - ref[0] - ref[2] > CV_VEGETATION_EXG:
            continue
        left = _edge_distance(img_np, cx, cy, nx, ny, max_half, ref, color_threshold)
        right = _edge_distance(img_np, cx, cy, -nx, -ny, max_half, ref, color_threshold)
        if left is None or right is None:
            continue
        widths.append(left + right)

    if not widths:
        return None, 0.0
    return float(np.median(widths)), len(widths) / n_samples


def _edge_distance(img_np, cx, cy, nx, ny, max_dist, ref, threshold, step=1.0):
    """Distance from (cx,cy) along (nx,ny) to where color first diverges from
    `ref` by more than `threshold` (the pavement edge). The image border
    counts as an edge; returns None if no edge is found within max_dist."""
    H, W = img_np.shape[:2]
    d = step
    while d <= max_dist:
        x, y = int(cx + nx * d), int(cy + ny * d)
        if not (0 <= x < W and 0 <= y < H):
            return d
        if np.linalg.norm(img_np[y, x].astype(float) - ref) > threshold:
            return d
        d += step
    return None


def build_roads(
    image,
    west: float, south: float, east: float, north: float,
    scale_m_per_px: float,
    cv_informed: bool = True,
) -> list[dict]:
    """
    Full open-data road pipeline: fetch Overture centerlines for the bbox,
    convert to pixels, and assign hybrid widths. Returns export-ready
    {"line": LineString(px), "width_px": float, "class": str} dicts.
    """
    img_w, img_h = image.size
    geo_roads = fetch_road_network(west, south, east, north)
    roads_px = roads_to_pixels(geo_roads, west, south, east, north, img_w, img_h)
    return assign_widths(roads_px, image, scale_m_per_px, cv_informed=cv_informed)


def detect_roads_cv(image) -> list[dict]:
    """CV fallback for non-georeferenced imagery (spec §4b: DeepLabV3+
    segmentation -> skeletonize -> graph trace -> spline). Not implemented;
    georeferenced sites use build_roads (Overture)."""
    raise NotImplementedError(
        "CV road detection not implemented; use build_roads() for georeferenced sites."
    )
