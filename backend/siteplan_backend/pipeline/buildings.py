"""
Building / roof outline detection.

Input:  PIL Image (aerial RGB)
Output: list of shapely.Polygon in pixel coordinates
        (caller converts to real-world meters using scale_m_per_px)

Model:  Fine-tuned U-Net (ResNet-50 backbone) on SpaceNet 2 + INRIA dataset.
        Falls back to SAM2 zero-shot if no weights are available.

SAM2 zero-shot notes (spec §4a calls this the quick baseline): SAM2's
automatic mask generator segments *everything* — water, lawns, docks,
boats, tree clumps — so building detection is really mask *classification*.
A mask is kept as a building only if it passes every heuristic in
_is_building_mask. Thresholds were calibrated on the lakeside test image
(marina crop). Expect some false positives on bare-ground patches and
docks; this is a baseline for pipeline validation, to be replaced by a
fine-tuned segmentation model.
"""
import math

import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.validation import make_valid
from PIL import Image
from pathlib import Path

MODEL_WEIGHTS = Path(__file__).parent.parent.parent.parent / "models" / "buildings_unet.pt"

SAM2_CHECKPOINT = "facebook/sam2.1-hiera-small"
# AMG quality thresholds loosened from the natural-photo defaults
# (0.88/0.95): aerial imagery is off-domain for SAM2 and the defaults
# return almost nothing (5 masks vs 80 on the marina test crop).
SAM2_PRED_IOU_THRESH = 0.7
SAM2_STABILITY_THRESH = 0.85
SAM2_POINTS_PER_SIDE = 32
TILE_SIZE = 512           # SAM2 internally works at 1024²; small roofs vanish
TILE_OVERLAP = 128        # if the whole site is downscaled into one frame

# Building-mask classification thresholds, calibrated on labeled masks from
# the lakeside marina crop (7 buildings vs dirt/lawn/dock/water groups).
# What separates buildings there:
#   - rectangularity: buildings 0.79-1.05, dirt patches 0.66-0.75
#   - texture: buildings 20-47, dirt/lawn/water <17 (roofs carry ridge and
#     shadow detail; bare ground and water are smooth)
#   - aspect: paths and docks are long thin strips, buildings are compact
# Deliberately NOT a blueness/water test: water is already rejected by the
# texture floor, and a blueness cap rejects real blue/teal roofs.
MIN_AREA_M2 = 10.0        # spec §4a
MAX_AREA_M2 = 10_000.0    # spec §4a
MIN_RECTANGULARITY = 0.78  # mask area / min-rotated-rect area
MAX_MEAN_EXG = 15.0       # excess green (2G-R-B); vegetation is well above
TEXTURE_RANGE = (18.0, 55.0)  # gray stddev inside mask
MAX_ASPECT = 4.0          # min-rect long/short side; rejects paths and docks

# Default px-area limits used when no scale is provided (legacy callers).
_DEFAULT_MIN_AREA_PX = 100
_DEFAULT_MAX_AREA_PX = 500_000


def detect_buildings(image: Image.Image, scale_m_per_px: float | None = None) -> list[Polygon]:
    """
    Run building segmentation and return regularized roof polygons.

    Args:
        image: PIL RGB image (stitched aerial tiles)
        scale_m_per_px: real-world scale; enables the spec's m² area limits
            (without it, fixed pixel-area defaults apply)

    Returns:
        List of shapely Polygons in pixel coordinates.
    """
    img_np = np.array(image.convert("RGB"))

    if scale_m_per_px:
        min_area_px = MIN_AREA_M2 / scale_m_per_px**2
        max_area_px = MAX_AREA_M2 / scale_m_per_px**2
    else:
        min_area_px = _DEFAULT_MIN_AREA_PX
        max_area_px = _DEFAULT_MAX_AREA_PX

    if MODEL_WEIGHTS.exists():
        mask = _run_unet(img_np)
    else:
        mask = _run_sam2_fallback(img_np, min_area_px, max_area_px)

    polygons = _mask_to_polygons(mask, img_np.shape, min_area_px, max_area_px)
    return polygons


def _run_unet(img_np: np.ndarray) -> np.ndarray:
    """Run fine-tuned U-Net inference. Returns binary mask (H x W, uint8)."""
    # TODO: load model weights, run inference
    raise NotImplementedError("U-Net weights not yet available. Using SAM2 fallback.")


def _run_sam2_fallback(
    img_np: np.ndarray,
    min_area_px: float,
    max_area_px: float,
) -> np.ndarray:
    """
    Zero-shot SAM2: tile the image, run the automatic mask generator per
    tile, keep masks that classify as buildings, and stamp them into one
    binary mask (which also merges duplicates from overlapping tiles).
    """
    # torch/sam2 imports stay function-local: importing torch at module
    # level segfaults the test suite (native-lib init order clash with
    # cv2/shapely when pytest imports everything together).
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amg = SAM2AutomaticMaskGenerator.from_pretrained(
        SAM2_CHECKPOINT,
        device=device,
        points_per_side=SAM2_POINTS_PER_SIDE,
        pred_iou_thresh=SAM2_PRED_IOU_THRESH,
        stability_score_thresh=SAM2_STABILITY_THRESH,
        min_mask_region_area=int(min_area_px),
    )

    h, w = img_np.shape[:2]
    building_mask = np.zeros((h, w), dtype=np.uint8)
    step = TILE_SIZE - TILE_OVERLAP

    for ty in range(0, max(h - TILE_OVERLAP, 1), step):
        for tx in range(0, max(w - TILE_OVERLAP, 1), step):
            y1, x1 = min(ty + TILE_SIZE, h), min(tx + TILE_SIZE, w)
            tile = img_np[ty:y1, tx:x1]
            if tile.shape[0] < 64 or tile.shape[1] < 64:
                continue
            for m in amg.generate(tile):
                seg = m["segmentation"]
                # Skip masks touching a tile edge that is interior to the
                # image: the overlapping neighbor tile sees them whole.
                if _touches_interior_edge(seg, tx, ty, x1, y1, w, h):
                    continue
                if _is_building_mask(tile, seg, m["area"], min_area_px, max_area_px):
                    building_mask[ty:y1, tx:x1][seg] = 1

    return building_mask


def _touches_interior_edge(seg, tx, ty, x1, y1, img_w, img_h) -> bool:
    """True if the mask touches a tile boundary that isn't the image edge
    (meaning the object continues into a neighboring tile, which will see
    it in full thanks to the tile overlap)."""
    return (
        (seg[0, :].any() and ty > 0)
        or (seg[:, 0].any() and tx > 0)
        or (seg[-1, :].any() and y1 < img_h)
        or (seg[:, -1].any() and x1 < img_w)
    )


def _is_building_mask(
    tile: np.ndarray,
    seg: np.ndarray,
    area_px: float,
    min_area_px: float,
    max_area_px: float,
) -> bool:
    """Heuristic classification of a SAM2 mask as building / not-building.
    See module docstring for calibration context."""
    if area_px < min_area_px or area_px > max_area_px:
        return False

    contours, _ = cv2.findContours(
        seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    (_, (rw, rh), _) = cv2.minAreaRect(contour)
    rect_area = rw * rh
    if rect_area <= 0 or area_px / rect_area < MIN_RECTANGULARITY:
        return False
    if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) > MAX_ASPECT:
        return False

    f = tile.astype(np.float32)
    r, g, b = f[..., 0][seg], f[..., 1][seg], f[..., 2][seg]
    if (2 * g - r - b).mean() > MAX_MEAN_EXG:          # vegetation
        return False
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    if not (TEXTURE_RANGE[0] < gray.std() < TEXTURE_RANGE[1]):
        return False

    return True


def suppress_canopy_false_positives(
    polygons: list[Polygon],
    tree_detections: list[dict],
    coverage_threshold: float = 0.6,
) -> list[Polygon]:
    """
    Drop building polygons that are mostly covered by detected tree canopy.

    SAM2 zero-shot's dominant false positive is tree crowns (especially
    autumn/brown foliage), which match roofs on every per-mask heuristic:
    color, texture, compactness. The tree detector knows where crowns are,
    so a "building" whose area is >= coverage_threshold covered by canopy
    circles is reclassified as canopy and removed.

    IMPORTANT: pass RAW tree detections (before filter_placements' building
    -overlap suppression). Using post-suppression trees is circular: a
    false building suppresses the very trees that would expose it.
    """
    if not polygons or not tree_detections:
        return list(polygons)

    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    canopies = [
        Point(d["x_px"], d["y_px"]).buffer(d["radius_px"]) for d in tree_detections
    ]
    index = STRtree(canopies)

    kept = []
    for poly in polygons:
        nearby = index.query(poly)
        if len(nearby):
            covered = unary_union([canopies[i] for i in nearby]).intersection(poly).area
            if covered / poly.area >= coverage_threshold:
                continue
        kept.append(poly)
    return kept


def _mask_to_polygons(
    mask: np.ndarray,
    img_shape: tuple,
    min_area_px: float = _DEFAULT_MIN_AREA_PX,
    max_area_px: float = _DEFAULT_MAX_AREA_PX,
    simplify_tolerance: float = 2.0,
) -> list[Polygon]:
    """
    Convert binary mask to list of regularized shapely Polygons.

    Steps:
    1. Connected components
    2. Area filter
    3. Contour extraction
    4. Douglas-Peucker simplification
    5. Angle snapping (orthogonalization)
    """
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px or area > max_area_px:
            continue

        # Squeeze contour to (N, 2)
        pts = contour.squeeze()
        if pts.ndim < 2 or len(pts) < 4:
            continue

        poly = Polygon(pts)
        poly = make_valid(poly)
        if poly.is_empty:
            continue
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        if poly.geom_type != "Polygon":
            continue

        poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        poly = _orthogonalize(poly)

        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)

    return polygons


def _orthogonalize(poly: Polygon, angle_threshold_deg: float = 10.0) -> Polygon:
    """
    Snap polygon edges to the building's dominant axis (spec §4a step 5).

    Method: find the dominant angle from the minimum rotated bounding
    rectangle's longest edge; every edge whose direction is within
    angle_threshold_deg of parallel or perpendicular to that axis is
    rotated to exactly parallel/perpendicular (about its midpoint);
    consecutive edges sharing a snapped direction merge into one; vertices
    are rebuilt as intersections of consecutive edge lines. Edges outside
    the threshold (intentional diagonals) are left as detected.

    Falls back to the input polygon on any degenerate geometry.
    """
    try:
        coords = list(poly.exterior.coords)[:-1]
        if len(coords) < 3:
            return poly

        rect = poly.minimum_rotated_rectangle
        if rect.geom_type != "Polygon":
            return poly
        rc = list(rect.exterior.coords)
        edges = [
            (rc[i + 1][0] - rc[i][0], rc[i + 1][1] - rc[i][1]) for i in range(4)
        ]
        longest = max(edges, key=lambda e: e[0] ** 2 + e[1] ** 2)
        dominant = math.atan2(longest[1], longest[0])

        threshold = math.radians(angle_threshold_deg)
        # Snap each edge direction; None = leave as-is (diagonal)
        snapped = []
        for i in range(len(coords)):
            p0, p1 = coords[i], coords[(i + 1) % len(coords)]
            angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            rel = (angle - dominant) % (math.pi / 2)
            if rel < threshold:
                snap = angle - rel
            elif (math.pi / 2) - rel < threshold:
                snap = angle + ((math.pi / 2) - rel)
            else:
                snap = angle
            snapped.append((p0, p1, snap))

        # Merge consecutive edges with the same direction, then rebuild
        # vertices from consecutive line intersections.
        merged = []
        for p0, p1, ang in snapped:
            if merged and abs(_angle_diff(merged[-1][2], ang)) < 1e-9:
                merged[-1] = (merged[-1][0], p1, ang)
            else:
                merged.append((p0, p1, ang))
        if len(merged) > 1 and abs(_angle_diff(merged[0][2], merged[-1][2])) < 1e-9:
            merged[0] = (merged[-1][0], merged[0][1], merged[0][2])
            merged.pop()
        if len(merged) < 3:
            return poly

        new_pts = []
        for i in range(len(merged)):
            a = merged[i]
            b = merged[(i + 1) % len(merged)]
            pt = _line_intersection(_edge_line(a), _edge_line(b))
            if pt is None:
                return poly
            new_pts.append(pt)

        result = Polygon(new_pts)
        result = make_valid(result)
        if result.geom_type == "MultiPolygon":
            result = max(result.geoms, key=lambda p: p.area)
        if result.geom_type != "Polygon" or result.is_empty:
            return poly
        # Reject snaps that wildly change the footprint (degenerate rebuild)
        if abs(result.area - poly.area) > 0.35 * poly.area:
            return poly
        return result
    except Exception:
        return poly


def _angle_diff(a: float, b: float) -> float:
    d = (a - b) % math.pi
    return min(d, math.pi - d)


def _edge_line(edge):
    """Return (point, direction) for an edge rotated to its snapped angle
    about its midpoint."""
    (x0, y0), (x1, y1), ang = edge
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    return (mx, my), (math.cos(ang), math.sin(ang))


def _line_intersection(l1, l2):
    (px, py), (dx, dy) = l1
    (qx, qy), (ex, ey) = l2
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return None
    t = ((qx - px) * ey - (qy - py) * ex) / denom
    return (px + t * dx, py + t * dy)
