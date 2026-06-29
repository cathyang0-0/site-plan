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
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from sklearn.cluster import KMeans
import cv2

N_CLUSTERS = 4
CLUSTER_LABELS = ["water", "vegetation", "bare earth / farmland", "paved / hardscape"]


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

    # K-means clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features)

    # Build per-cluster pixel mask → polygons
    full_cluster_map = np.full((h, w), -1, dtype=np.int32)
    for seg_id, cluster_id in zip(valid_segment_ids, cluster_ids):
        full_cluster_map[segments == seg_id] = cluster_id

    results = []
    for c in range(N_CLUSTERS):
        cluster_mask = (full_cluster_map == c).astype(np.uint8) * 255
        polygons = _mask_to_multipolygon(cluster_mask, min_area_px=500)
        thumbnail = _make_thumbnail(img_np, cluster_mask)

        results.append({
            "cluster_id": c,
            "label": CLUSTER_LABELS[c],  # default label; user can rename
            "polygons": polygons,
            "thumbnail": thumbnail,
        })

    return results


def _extract_superpixel_features(
    img_np: np.ndarray,
    segments: np.ndarray,
    exclude: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Compute feature vector per superpixel (color + Gabor + LBP)."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # LBP texture
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

    # Gabor responses at 2 frequencies × 4 orientations
    gabor_responses = []
    for freq in [0.1, 0.3]:
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            real, _ = gabor(gray, frequency=freq, theta=theta)
            gabor_responses.append(real)
    gabor_stack = np.stack(gabor_responses, axis=-1)

    features = []
    valid_ids = []

    for seg_id in np.unique(segments):
        mask = (segments == seg_id)
        # Skip superpixels that are mostly inside buildings/roads
        if exclude[mask].mean() > 0.5:
            continue

        color_feat = img_np[mask].mean(axis=0)           # mean RGB
        lbp_feat = np.array([lbp[mask].mean()])
        gabor_feat = gabor_stack[mask].mean(axis=0)

        feat = np.concatenate([color_feat, lbp_feat, gabor_feat])
        features.append(feat)
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
