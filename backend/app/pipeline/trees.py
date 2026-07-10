"""
Tree instance detection using DeepForest, plus a dense-stand fill pass.

Input:  PIL Image (aerial RGB), scale_m_per_px
Output: list of dicts {"x_px", "y_px", "radius_px", "radius_m"}

DeepForest docs: https://deepforest.readthedocs.io
"""
import math
import random

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

MIN_CANOPY_RADIUS_M = 1.5   # canopies smaller than this are enlarged, not dropped
MAX_CANOPY_RADIUS_M = 10.0  # canopies larger than this are capped (likely merged crowns)

# Detection tuning. The pretrained model ships with nms_thresh=0.05, which
# suppresses any two crown boxes overlapping by more than 5% IoU -- in
# continuous forest canopy (where real crowns interlock) that collapses
# whole stands into one or two boxes and leaves them empty on the plan.
# 0.4 lets neighboring crowns coexist while still merging true duplicates.
# Validated visually on the lakeside test image (2026-07-10).
NMS_IOU_THRESHOLD = 0.4     # model-level + cross-patch mosaic NMS
SCORE_THRESHOLD = 0.1       # confidence floor; lowering to 0.05 fills dense
                            # canopy further but adds shoreline/building
                            # false positives -- keep 0.1 as default
PATCH_SIZE = 400            # inference window (px); model's native scale

# Stand fill: the detector cannot resolve individual crowns in dense,
# continuous canopy -- what it emits there is one oversized box spanning
# the merged crown mass (> MAX_CANOPY_RADIUS_M). Those boxes are treated
# as "stands" and filled with synthetic trees. Fill is strictly
# detection-led: only areas the model itself flagged as canopy get
# synthetic trees (an earlier image-mask-based gap fill covered more but
# read as artificial and bled into treeless areas).
STAND_FILL_SPACING_FACTOR = 1.9   # mean center spacing as multiple of fill radius
STAND_FILL_POSITION_JITTER = 0.60  # ± fraction of spacing each tree is displaced;
                                   # high on purpose -- distances between
                                   # neighbors should vary visibly
STAND_FILL_KEEP_PROB = 0.85        # chance a grid slot gets a tree at all;
                                   # the skips open small clearings so the
                                   # fill doesn't read as a uniform carpet
STAND_FILL_MIN_RADIUS_M = 3.0      # fill crowns are forest trees, not shrubs; the
                                   # detected-size distribution skews small (yard
                                   # trees), so floor the fill size at a mature crown


def detect_trees(image: Image.Image, scale_m_per_px: float) -> list[dict]:
    """
    Detect individual tree canopies.

    Returns:
        List of dicts with pixel centroid + radius, and real-world radius in meters.
    """
    try:
        from deepforest import main as deepforest_main
    except ImportError:
        raise ImportError("deepforest not installed. Run: pip install deepforest")

    model = deepforest_main.deepforest()
    # deepforest 2.x: pretrained weights come from HuggingFace (downloads and
    # caches on first run). Replaces the removed 1.x use_release().
    model.load_model("weecology/deepforest-tree")
    # load_model() restores the checkpoint's own thresholds, so these must be
    # set on the torchvision RetinaNet *after* loading to take effect.
    model.model.nms_thresh = NMS_IOU_THRESHOLD
    model.model.score_thresh = SCORE_THRESHOLD

    img_np = np.array(image.convert("RGB"))
    # predict_tile handles arbitrarily large images by windowing at the
    # model's native patch size; predict_image degrades badly beyond ~400px.
    boxes = model.predict_tile(
        image=img_np,
        patch_size=PATCH_SIZE,
        patch_overlap=0.25,
        iou_threshold=NMS_IOU_THRESHOLD,
    )

    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    for _, row in boxes.iterrows():
        x_center = (row["xmin"] + row["xmax"]) / 2
        y_center = (row["ymin"] + row["ymax"]) / 2
        radius_px = (row["xmax"] - row["xmin"]) / 2   # use width as diameter estimate
        radius_m = radius_px * scale_m_per_px

        # An oversized box is several merged crowns in dense canopy, not one
        # tree -- keep it flagged as a "stand" (with both box half-extents,
        # since stands are rarely round) for fill_dense_stands to populate.
        # An undersized box is a real tree that should stay readable.
        if radius_m > MAX_CANOPY_RADIUS_M:
            detections.append({
                "x_px": x_center,
                "y_px": y_center,
                "radius_px": radius_px,
                "ry_px": (row["ymax"] - row["ymin"]) / 2,
                "radius_m": radius_m,
                "stand": True,
            })
            continue
        if radius_m < MIN_CANOPY_RADIUS_M:
            radius_m = MIN_CANOPY_RADIUS_M
            radius_px = radius_m / scale_m_per_px

        detections.append({
            "x_px": x_center,
            "y_px": y_center,
            "radius_px": radius_px,
            "radius_m": radius_m,
        })

    return detections


def fill_dense_stands(
    detections: list[dict],
    scale_m_per_px: float,
    seed: int = 0,
) -> list[dict]:
    """
    Replace oversized "stand" detections with scattered synthetic trees.

    Returns the single-tree detections unchanged, followed by the synthetic
    trees (marked "synthetic": True) so filter_placements gives real
    detections priority where the two overlap.

    Placement: a jittered hex grid clipped to each stand's ellipse.
    Position jitter, random slot skips (which open small clearings), and
    size jitter (0.65–1.35×) are deliberately strong so the fill reads as
    a natural stand, not a pattern; mean spacing (1.9× fill radius)
    leaves crowns touching to lightly interlocking. Fill size follows the
    75th percentile of the detected single crowns (dominated by small
    yard trees, while stands are mature forest) with a floor of
    STAND_FILL_MIN_RADIUS_M.
    """
    rng = random.Random(seed)
    singles = [d for d in detections if not d.get("stand")]
    stands = [d for d in detections if d.get("stand")]
    if not stands:
        return list(detections)

    base_r_m = (
        float(np.percentile([d["radius_m"] for d in singles], 75))
        if singles
        else (MIN_CANOPY_RADIUS_M + MAX_CANOPY_RADIUS_M) / 4
    )
    base_r_m = max(base_r_m, STAND_FILL_MIN_RADIUS_M)
    spacing_px = STAND_FILL_SPACING_FACTOR * base_r_m / scale_m_per_px
    row_step = spacing_px * math.sqrt(3) / 2  # hex rows pack tighter

    synthetic = []
    for stand in stands:
        cx, cy = stand["x_px"], stand["y_px"]
        rx = stand["radius_px"]
        ry = stand.get("ry_px", rx)
        placed = 0
        row = 0
        y = cy - ry + row_step / 2
        while y < cy + ry:
            x = cx - rx + spacing_px / 2 + (spacing_px / 2 if row % 2 else 0)
            while x < cx + rx:
                if rng.random() > STAND_FILL_KEEP_PROB:
                    x += spacing_px
                    continue
                jx = x + rng.uniform(-1, 1) * STAND_FILL_POSITION_JITTER * spacing_px
                jy = y + rng.uniform(-1, 1) * STAND_FILL_POSITION_JITTER * spacing_px
                # keep the jittered point inside the stand's ellipse
                if ((jx - cx) / rx) ** 2 + ((jy - cy) / ry) ** 2 <= 1.0:
                    r_m = base_r_m * rng.uniform(0.65, 1.35)
                    synthetic.append({
                        "x_px": jx,
                        "y_px": jy,
                        "radius_px": r_m / scale_m_per_px,
                        "radius_m": r_m,
                        "synthetic": True,
                    })
                    placed += 1
                x += spacing_px
            y += row_step
            row += 1
        if placed == 0:
            # tiny stand (barely over the size cutoff): one tree at center
            r_m = base_r_m * rng.uniform(0.65, 1.35)
            synthetic.append({
                "x_px": cx,
                "y_px": cy,
                "radius_px": r_m / scale_m_per_px,
                "radius_m": r_m,
                "synthetic": True,
            })

    return singles + synthetic


def filter_placements(
    detections: list[dict],
    building_polygons: list[Polygon],
    overlap_threshold: float = 0.30,
    tree_overlap_threshold: float = 0.30,
) -> list[dict]:
    """
    Filter tree detections by two overlap rules (both 30% by default,
    per spec §7, but independently tunable):
      1. Tree-building overlap > overlap_threshold → suppress
      2. Tree-tree overlap > tree_overlap_threshold → suppress
         (checked against already-accepted trees)

    Both rules use STRtree spatial indexing to keep complexity at O(n log n)
    rather than O(n²). The bounding-box query narrows candidates; precise
    intersection is only computed for nearby shapes.
    """
    from shapely.strtree import STRtree

    building_index = STRtree(building_polygons) if building_polygons else None

    accepted: list[dict] = []
    accepted_canopies: list[Polygon] = []

    for det in detections:
        canopy = Point(det["x_px"], det["y_px"]).buffer(det["radius_px"])

        # Rule 1: building overlap
        if building_index is not None:
            candidates = building_index.query(canopy)
            if any(
                canopy.intersection(building_polygons[i]).area / canopy.area > overlap_threshold
                for i in candidates
            ):
                continue

        # Rule 2: tree-tree overlap against already-accepted canopies
        if accepted_canopies:
            tree_index = STRtree(accepted_canopies)
            candidates = tree_index.query(canopy)
            if any(
                canopy.intersection(accepted_canopies[i]).area / canopy.area > tree_overlap_threshold
                for i in candidates
            ):
                continue

        accepted.append(det)
        accepted_canopies.append(canopy)

    return accepted
