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
from shapely.geometry import MultiPolygon, Polygon
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import cv2

N_CLUSTERS = 4
CLUSTER_LABELS = ["water", "vegetation", "bare earth / farmland", "paved / hardscape"]

# Default hatch per detected type, following spec §5's table. These are the
# starting styles the user would override in the style panel (§6 Step 4 --
# the one step needing active judgment). hatch_scale is an explicit override
# in real-world-meter modelspace (the mm->scale auto-conversion isn't wired
# yet, see dxf.py), tuned to read at typical site scales.
DEFAULT_HATCH_STYLES = {
    "water":                 {"hatch_type": "lines", "hatch_angle_deg": 0.0, "hatch_scale": 3.0},
    "vegetation":            {"hatch_type": "dots", "hatch_scale": 2.0},
    "bare earth / farmland": {"hatch_type": "lines", "hatch_angle_deg": 45.0, "hatch_scale": 4.5},
    "paved / hardscape":     {"hatch_type": "crosshatch", "hatch_angle_deg": 45.0, "hatch_scale": 3.5},
}
_FALLBACK_HATCH_STYLE = {"hatch_type": "lines", "hatch_angle_deg": 0.0, "hatch_scale": 3.0}


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
          - label: str (auto-named by cluster index)
          - polygons: shapely MultiPolygon (pixel coords)
          - thumbnail: PIL.Image (representative crop)
          - cluster_id: int
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # Combined exclusion mask (buildings + roads)
    exclude = ((building_mask > 0.5) | (road_mask > 0.5)).astype(np.uint8)

    # Superpixel segmentation (SLIC) for spatial smoothing
    segments = slic(img_np, n_segments=500, compactness=10, sigma=1, start_label=0)

    # Extract per-superpixel features
    features, valid_segment_ids = _extract_superpixel_features(img_np, segments, exclude)

    if len(features) == 0:
        return []

    # Standardize features so no single channel dominates the distance, then
    # cluster. (Features are already color-forward; see _extract_*.)
    feat_std = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(feat_std)

    # k-means cluster ids are arbitrary -- map each to a ground-cover label
    # from the cluster's mean *raw* color/texture (see _assign_semantic_labels).
    labels_by_cluster = _assign_semantic_labels(features, cluster_ids)

    # Build per-cluster pixel mask → polygons
    full_cluster_map = np.full((h, w), -1, dtype=np.int32)
    for seg_id, cluster_id in zip(valid_segment_ids, cluster_ids):
        full_cluster_map[segments == seg_id] = cluster_id

    results = []
    for c in range(N_CLUSTERS):
        cluster_mask = (full_cluster_map == c).astype(np.uint8) * 255
        polygons = _mask_to_multipolygon(cluster_mask, min_area_px=500)
        if polygons.is_empty:
            continue
        results.append({
            "cluster_id": c,
            "label": labels_by_cluster[c],
            "polygons": polygons,
            "thumbnail": _make_thumbnail(img_np, cluster_mask),
        })

    return results


def _assign_semantic_labels(features: np.ndarray, cluster_ids: np.ndarray) -> dict:
    """
    Map arbitrary k-means cluster ids to ground-cover labels using each
    cluster's mean color/texture. k-means finds 4 groups but doesn't know
    which is water vs grass vs paved vs bare -- this scores every
    (cluster, label) pair and greedily assigns each label to its best
    remaining cluster, so the 4 labels map sensibly regardless of id order.

    Feature layout (see _extract_superpixel_features): [R, G, B, exg,
    blueness, sat, value, texture].
    """
    means = {c: features[cluster_ids == c].mean(axis=0) for c in np.unique(cluster_ids)}

    def score(f, label):
        R, G, B, exg, blueness, sat, value, texture = f
        if label == "water":            # blue-shifted, dark, smooth, not green
            return blueness - exg - texture * 0.5 - value * 0.3
        if label == "vegetation":       # green dominates
            return exg
        if label == "paved / hardscape":  # gray: low saturation, brighter, flat
            return value - sat * 2.0 - abs(exg) - blueness
        if label == "bare earth / farmland":  # warm/brown: R>=G>B, low green, low blue
            return (R - B) - exg - abs(blueness) * 0.5
        return 0.0

    # Greedy: repeatedly take the highest-scoring (cluster,label) pair.
    clusters = list(means)
    labels = list(CLUSTER_LABELS)
    pairs = sorted(
        ((score(means[c], lb), c, lb) for c in clusters for lb in labels),
        reverse=True,
    )
    assigned, used_c, used_lb = {}, set(), set()
    for _, c, lb in pairs:
        if c in used_c or lb in used_lb:
            continue
        assigned[c] = lb
        used_c.add(c); used_lb.add(lb)
    # Any leftover clusters (fewer labels than clusters shouldn't happen with
    # k=4) get a generic name.
    for c in clusters:
        assigned.setdefault(c, "ground cover")
    return assigned


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


def _mask_to_multipolygon(mask: np.ndarray, min_area_px: int = 500) -> MultiPolygon:
    """Convert binary mask to a MultiPolygon, filtering small regions."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        pts = c.squeeze()
        if pts.ndim < 2 or len(pts) < 4:
            continue
        polys.append(Polygon(pts))
    return MultiPolygon(polys) if polys else MultiPolygon()


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
