"""
Land texture detection — unsupervised clustering of ground cover types.

Input:  PIL Image, building masks, road masks
Output: list of dicts {"label": str, "polygons": MultiPolygon, "thumbnail": PIL.Image}

Approach:
  - Mask out buildings + roads
  - Extract texture features (Gabor + color + LBP) per superpixel
  - K-means cluster into k=4 types: water / grass / dirt+farmland / paved
"""
import numpy as np
from PIL import Image
from shapely.affinity import scale as _affine_scale
from shapely.affinity import translate as _affine_translate
from shapely.geometry import MultiPolygon, Polygon, box as _box
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import cv2

# Land-cover clustering is a coarse regional segmentation -- full-resolution
# aerial (multi-megapixel) is wasteful. Detect at reduced size, scale the
# resulting polygons back. Also caps SLIC/k-means cost.
MAX_DETECT_DIM = 1400

# How strongly a cyan/teal color (blue AND green both above red) counts as
# water. Biases blue-green shallows -- where the lakebed tints the water
# green but blue still beats red -- toward water rather than bare earth.
WATER_TEAL_WEIGHT = 0.6

N_CLUSTERS = 4
CLUSTER_LABELS = ["water", "vegetation", "bare earth / farmland", "paved / hardscape"]

# Default hatch per detected type, following spec §5's table. These are the
# starting styles the user would override in the style panel (§6 Step 4 --
# the one step needing active judgment). hatch_scale is an explicit override
# in real-world-meter modelspace (the mm->scale auto-conversion isn't wired
# yet, see dxf.py), tuned to read at typical site scales.
# AutoCAD-style hatch pattern definitions, extracted verbatim from the user's
# hand-tuned reference plan ("hatch reference.dxf", 2026-08) and baked in so the
# export reproduces that exact look in any viewer, independent of each CAD app's
# own pattern library (acad.pat imperial/metric variants differ). The reference
# file shares this pipeline's units (tree block width 12.0 both sides), so these
# values are already calibrated relative to trees/buildings/roads — use with
# scale=1.0, angle=0.0 (rotation/scale are pre-applied in the numbers below).
# Format per line: [angle_deg, (base_x, base_y), (offset_x, offset_y), [dashes]]
# (dash > 0 draw, < 0 gap, 0 dot; empty list = continuous line).
ACAD_PATTERNS = {
    # Water: AR-RROOF at reference scale 10 — long broken horizontal strokes.
    "AR-RROOF": [
        [0.0, (0.0, 0.0), (22.0, 10.0), [150.0, -20.0, 50.0, -10.0]],
        [0.0, (13.3, 5.0), (-10.0, 13.3), [30.0, -3.3, 60.0, -7.5]],
        [0.0, (5.0, 8.5), (52.0, 6.7), [80.0, -14.0, 40.0, -10.0]],
    ],
    # Vegetation: AR-SAND at reference scale 3.5 — irregular dot stipple.
    "AR-SAND": [
        [37.5, (0.0, 0.0), (-0.220477, 6.74388), [0.0, -5.32, 0.0, -5.95, 0.0, -5.6875]],
        [7.5, (0.0, 0.0), (6.19422, 9.87751), [0.0, -2.87, 0.0, -4.795, 0.0, -1.8375]],
        [327.5, (-4.3, 0.0), (10.8995, 0.0198067), [0.0, -1.75, 0.0, -6.3, 0.0, -8.225]],
        [317.5, (-4.3, 0.0), (10.5214, 3.07186), [0.0, -0.875, 0.0, -4.13, 0.0, -4.725]],
    ],
    # Bare earth / farmland: plain 45° lines, 3.81 m apart (LINE at scale 1.2).
    "LINE45": [
        [45.0, (0.0, 0.0), (-2.69408, 2.69408), []],
    ],
    # Paved: AR-CONC at reference scale 1.0 — fine aggregate speckle.
    "AR-CONC": [
        [50.0, (0.0, 0.0), (7.1726, -0.627521), [0.75, -8.25]],
        [355.0, (0.0, 0.0), (-1.38751, 7.52192), [0.6, -6.6]],
        [100.4514, (0.6, -0.0522934), (5.78509, 6.8944), [0.637402, -7.011421]],
        [46.1842, (0.0, 2.0), (10.6724, -1.65519), [1.125, -12.375]],
        [96.6356, (0.9, 1.86207), (9.34662, 9.74119), [0.956103, -10.517138]],
        [351.1842, (0.0, 2.0), (9.34662, 9.74119), [0.9, -9.9]],
        [21.0, (1.0, 1.5), (5.96907, -4.02619), [0.75, -8.25]],
        [326.0, (1.0, 1.5), (2.43315, 7.2515), [0.6, -6.6]],
        [71.4514, (1.5, 1.16448), (8.40222, 3.22531), [0.637402, -7.011421]],
        [37.5, (0.0, 0.0), (0.121599, 3.32894), [0.0, -6.52, 0.0, -6.7, 0.0, -6.625]],
        [7.5, (0.0, 0.0), (2.6307, 3.94412), [0.0, -3.82, 0.0, -6.37, 0.0, -2.525]],
        [327.5, (-2.2, 0.0), (5.33822, -0.225549), [0.0, -2.5, 0.0, -7.8, 0.0, -10.35]],
        [317.5, (-3.2, 0.0), (5.83186, 1.00105), [0.0, -3.25, 0.0, -5.18, 0.0, -7.35]],
    ],
}

# Hatch color is BLACK to match the user's hand-tuned reference plan: their
# hatching reads as thin black texture, kept quiet by the hairline lineweight
# and pattern sparsity, not by a light color. (The earlier light grays were
# near-invisible on screen in Rhino — patterns rendered fine but couldn't be
# seen at hairline weight.)
DEFAULT_HATCH_STYLES = {
    "water":                 {"hatch_type": "acad", "hatch_pattern": "AR-RROOF",
                              "hatch_color": "#000000"},
    "vegetation":            {"hatch_type": "acad", "hatch_pattern": "AR-SAND",
                              "hatch_color": "#000000"},
    "bare earth / farmland": {"hatch_type": "acad", "hatch_pattern": "LINE45",
                              "hatch_color": "#000000"},
    "paved / hardscape":     {"hatch_type": "acad", "hatch_pattern": "AR-CONC",
                              "hatch_color": "#000000"},
}
_FALLBACK_HATCH_STYLE = {"hatch_type": "lines", "hatch_angle_deg": 0.0, "hatch_scale": 3.0,
                         "hatch_color": "#000000"}


def default_hatch_style(label: str) -> dict:
    """Default hatch style dict for a detected land-type label (copy, so the
    caller can mutate per-region without touching the template)."""
    return dict(DEFAULT_HATCH_STYLES.get(label, _FALLBACK_HATCH_STYLE))


def detect_land_types(
    image: Image.Image,
    building_mask: np.ndarray,
    road_mask: np.ndarray,
) -> list[dict]:
    """
    Segment ground cover into up to 4 texture clusters.

    Returns:
        List of dicts per cluster:
          - label: str (semantic, from cluster color/texture)
          - polygons: shapely MultiPolygon (pixel coords, full-image scale)
          - thumbnail: PIL.Image (representative crop)
          - cluster_id: int
    """
    img_full = np.asarray(image.convert("RGB"))
    H, W = img_full.shape[:2]

    # Downscale for detection (coarse segmentation; see MAX_DETECT_DIM). Masks
    # use nearest-neighbour so they stay binary. Polygons are scaled back to
    # full-image coordinates at the end via `upscale`.
    detect_scale = min(1.0, MAX_DETECT_DIM / max(H, W))
    if detect_scale < 1.0:
        dw, dh = int(round(W * detect_scale)), int(round(H * detect_scale))
        img_np = cv2.resize(img_full, (dw, dh), interpolation=cv2.INTER_AREA)
        b_mask = cv2.resize(building_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
        r_mask = cv2.resize(road_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
    else:
        img_np, b_mask, r_mask = img_full, building_mask, road_mask
    h, w = img_np.shape[:2]
    upscale = W / w  # back to full-image pixels (uniform: dw/dh keep aspect)

    exclude = ((b_mask > 0.5) | (r_mask > 0.5)).astype(np.uint8)

    # SLIC + k-means use OpenMP thread pools that DEADLOCK if PyTorch/MPS was
    # initialized earlier in the same process (e.g. tree detection ran first).
    # threadpool_limits(1) serializes them, which avoids the deadlock; the
    # work is small (downscaled + only 500 superpixels) so single-threaded is
    # fine. Verified: reproduces without the limit, runs clean with it.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                      # degrade gracefully
        from contextlib import nullcontext as threadpool_limits  # type: ignore

    with threadpool_limits(1):
        segments = slic(img_np, n_segments=500, compactness=10, sigma=1, start_label=0)
        features, valid_segment_ids = _extract_superpixel_features(img_np, segments, exclude)
        if len(features) == 0:
            return []
        # Standardize so no single channel dominates the distance.
        feat_std = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
        cluster_ids = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(feat_std)

    # k-means cluster ids are arbitrary -- map each to a ground-cover label
    # from the cluster's mean *raw* color/texture (see _assign_semantic_labels).
    labels_by_cluster = _assign_semantic_labels(features, cluster_ids)

    # Build per-cluster pixel mask → polygons
    full_cluster_map = np.full((h, w), -1, dtype=np.int32)
    for seg_id, cluster_id in zip(valid_segment_ids, cluster_ids):
        full_cluster_map[segments == seg_id] = cluster_id

    # Group clusters by their assigned label (independent assignment means
    # several clusters -- e.g. deep water + shallows -- can share one), then
    # build one merged region per label. Merging at the mask level also lets
    # morphology heal seams between adjacent same-label clusters.
    clusters_by_label: dict[str, list[int]] = {}
    for c in range(N_CLUSTERS):
        clusters_by_label.setdefault(labels_by_cluster.get(c, "ground cover"), []).append(c)

    # A region touching the image edge must reach it squarely, not get eroded
    # inward and rounded by the morphology/Chaikin smoothing (the lake read as
    # a rounded blob floating off the bottom edge). Fix: pad the mask with the
    # edge replicated, so an edge-touching region extends into the pad; do all
    # the smoothing there; then clip back to the exact image rectangle -- the
    # frame edges come out straight and flush, interior boundaries stay smooth.
    pad = 16  # >= the morph-close kernel so edge regions extend into the pad
    frame = _box(0, 0, w, h)

    results = []
    for label, cluster_ids_for_label in clusters_by_label.items():
        label_mask = np.isin(full_cluster_map, cluster_ids_for_label).astype(np.uint8) * 255
        padded = cv2.copyMakeBorder(label_mask, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        polygons = _mask_to_multipolygon(padded)  # morph + simplify + chaikin
        if polygons.is_empty:
            continue
        polygons = _affine_translate(polygons, xoff=-pad, yoff=-pad).intersection(frame)
        polygons = _as_multipolygon(polygons)
        if polygons.is_empty:
            continue
        if upscale != 1.0:  # scale polygons back to full-image coordinates
            polygons = _affine_scale(polygons, xfact=upscale, yfact=upscale, origin=(0, 0))
        results.append({
            "cluster_id": cluster_ids_for_label[0],
            "label": label,
            "polygons": polygons,
            "thumbnail": _make_thumbnail(img_np, label_mask),
        })

    return results


def _assign_semantic_labels(features: np.ndarray, cluster_ids: np.ndarray) -> dict:
    """
    Map arbitrary k-means cluster ids to ground-cover labels using each
    cluster's mean color/texture. k-means finds 4 groups but doesn't know
    which is water vs grass vs paved vs bare.

    Each cluster is assigned to its OWN highest-scoring label, independently
    (not a bijection). A real scene isn't guaranteed to contain all four
    types: a lakeshore has deep water AND blue-green shallows -- two
    water-like clusters -- and no true bare earth. Forcing a one-label-per-
    cluster assignment there pushed the shallows onto "bare". Independent
    assignment lets both become water (and same-label clusters are merged by
    the caller). Land clusters never pick water because their water score is
    negative (low blueness, penalties).

    Feature layout (see _extract_superpixel_features): [R, G, B, exg,
    blueness, sat, value, texture].
    """
    means = {c: features[cluster_ids == c].mean(axis=0) for c in np.unique(cluster_ids)}

    def score(f, label):
        R, G, B, exg, blueness, sat, value, texture = f
        teal = min(G, B) - R  # cyan/teal index: both blue AND green above red
        if label == "water":
            # Blue-shifted, dark, smooth. The teal term (WATER_TEAL_WEIGHT)
            # biases blue-green shallows -- where the lakebed greens the water
            # but blue still dominates red -- toward water rather than bare.
            return blueness + WATER_TEAL_WEIGHT * teal - texture * 0.5 - value * 0.3
        if label == "vegetation":       # green dominates, on land
            # Penalize only POSITIVE blueness so blue-green shallows (some
            # green but blue-shifted) don't read as vegetation; brown/red
            # (negative blueness) must not be rewarded here.
            return exg - max(blueness, 0.0)
        if label == "paved / hardscape":  # gray: low saturation, brighter, flat
            return value - sat * 2.0 - abs(exg) - blueness
        if label == "bare earth / farmland":  # warm/brown: R>=B, low green, low blue
            return (R - B) - exg - abs(blueness) * 0.5
        return 0.0

    return {c: max(CLUSTER_LABELS, key=lambda lb: score(means[c], lb)) for c in means}


def _extract_superpixel_features(
    img_np: np.ndarray,
    segments: np.ndarray,
    exclude: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """
    Per-superpixel feature vector, color-forward: ground-cover types separate
    mainly by color (blue water, green vegetation, gray pavement, brown bare
    earth), with texture distinguishing smooth grass from busy forest. The
    earlier Gabor/LBP-heavy vector let texture noise dominate and mixed water
    with forest; here color-derived channels lead and one texture channel
    (local brightness stddev) supports them.

    Layout: [R, G, B, exg, blueness, sat, value, texture] -- kept in sync with
    _assign_semantic_labels.
    """
    f = img_np.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    exg = 2 * G - R - B                       # greenness
    blueness = B - R                          # water is blue-shifted
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat, value = hsv[..., 1], hsv[..., 2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (9, 9))
    texture = np.sqrt(np.maximum(cv2.blur(gray**2, (9, 9)) - mean**2, 0))

    channels = np.stack([R, G, B, exg, blueness, sat, value, texture], axis=-1)

    features, valid_ids = [], []
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        if exclude[mask].mean() > 0.5:        # mostly building/road → skip
            continue
        features.append(channels[mask].mean(axis=0))
        valid_ids.append(seg_id)

    return np.array(features), valid_ids


def _mask_to_multipolygon(
    mask: np.ndarray,
    min_area_px: int = 1500,
    morph_open_px: int = 3,
    morph_close_px: int = 11,
    simplify_tol_px: float = 2.5,
    chaikin_iterations: int = 2,
) -> MultiPolygon:
    """
    Convert a binary cluster mask to a clean MultiPolygon.

    Per-superpixel k-means labelling produces speckled, jagged regions (a
    lawn shatters into islands wherever a cell flips class). This cleans them:
      1. Morphological OPEN (remove speckle) then CLOSE (fill gaps and merge
         neighbouring islands into one region) -- the main de-fragmenter.
      2. Area filter -- drop what survives that is still tiny.
      3. Douglas-Peucker simplify -- collapse the pixel-staircase boundary
         into clean straight-ish edges instead of a jagged step outline.
      4. Chaikin corner-cutting -- round the simplified corners into smooth
         organic curves, so land-type edges read hand-drawn rather than
         faceted (spec §4d "smooth organic curves, not jagged pixel-step").
    """
    if morph_open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_px, morph_open_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if morph_close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_px, morph_close_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        pts = c.squeeze()
        if pts.ndim < 2 or len(pts) < 4:
            continue
        poly = Polygon(pts)
        if simplify_tol_px > 0:
            poly = poly.simplify(simplify_tol_px, preserve_topology=True)
        if chaikin_iterations > 0 and poly.geom_type == "Polygon" and not poly.is_empty:
            poly = Polygon(_chaikin_closed(list(poly.exterior.coords), chaikin_iterations))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if poly.geom_type == "Polygon":
            polys.append(poly)
        elif poly.geom_type == "MultiPolygon":
            polys.extend(g for g in poly.geoms if not g.is_empty)
    return MultiPolygon(polys) if polys else MultiPolygon()


def _as_multipolygon(geom) -> MultiPolygon:
    """Normalize a Polygon/MultiPolygon/GeometryCollection (e.g. from a frame
    intersection) to a MultiPolygon, dropping non-polygonal slivers."""
    if geom.is_empty:
        return MultiPolygon()
    if geom.geom_type == "Polygon":
        return MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    return MultiPolygon([g for g in geom.geoms if g.geom_type == "Polygon" and not g.is_empty])


def _chaikin_closed(coords: list, iterations: int) -> list:
    """
    Chaikin corner-cutting on a closed ring. Each pass replaces every edge
    with two points at 1/4 and 3/4 along it, cutting the corner; repeating
    converges to a smooth quadratic B-spline-like curve. Returns a closed
    ring (last point == first).

    `coords` is a closed ring (findContours/shapely give first==last); the
    duplicate close point is dropped for the wrap-around math and re-added.
    """
    pts = coords[:-1] if len(coords) > 1 and coords[0] == coords[-1] else list(coords)
    if len(pts) < 3:
        return coords
    for _ in range(iterations):
        out = []
        n = len(pts)
        for i in range(n):
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
            out.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            out.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        pts = out
    pts.append(pts[0])  # close the ring
    return pts


def _make_thumbnail(img_np: np.ndarray, mask: np.ndarray, size: int = 64) -> Image.Image:
    """Crop a representative patch from the cluster region."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return Image.new("RGB", (size, size), (200, 200, 200))
    cx, cy = int(xs.mean()), int(ys.mean())
    h, w = img_np.shape[:2]
    x0 = max(0, cx - size // 2)
    y0 = max(0, cy - size // 2)
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    crop = img_np[y0:y1, x0:x1]
    return Image.fromarray(crop).resize((size, size))
