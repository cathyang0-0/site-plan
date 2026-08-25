"""
Plan-assembly helpers shared by the API job runner and scripts/poc.py.

These lived in scripts/poc.py and were borrowed by the API through a
sys.path hack ("TODO: promote into the package"). Packaging for
distribution called the debt: an installed siteplan-backend has no
scripts/ directory, so the shared pieces now live here and poc.py imports
them back.

Heavy imports (cv2, numpy, torch-adjacent) stay function-local — the
module-scope-torch segfault rule applies here like everywhere else.
"""
import math
import random

# Radius the default tree block is drawn at; placements express crown size
# as a multiple of this ("scale"), so exporters and previews share one unit.
DEFAULT_BLOCK_RADIUS_PX = 20.0


def default_tree_block(radius_px: float = DEFAULT_BLOCK_RADIUS_PX,
                       n_pts: int = 32) -> list[list]:
    """
    Default tree block: circle with a "+" in the center.
    Returns a list of curve point-lists: [circle_pts, horizontal_line, vertical_line].
    """
    circle = [
        (radius_px * math.cos(2 * math.pi * i / n_pts),
         radius_px * math.sin(2 * math.pi * i / n_pts))
        for i in range(n_pts + 1)
    ]
    arm = radius_px * 0.3  # "+" arms are 30% of radius
    horizontal = [(-arm, 0), (arm, 0)]
    vertical = [(0, -arm), (0, arm)]
    return [circle, horizontal, vertical]


def real_tree_detections(
    image,
    scale_m_per_px: float,
    stand_fill: bool = True,
    crown_size_scale: float = 1.0,
    size_variance: float = 1.0,
) -> list[dict]:
    """Run real DeepForest detection + dense-stand fill. Returns raw
    detections (pre overlap-filtering) so callers can cross-check other
    modules against them before suppression runs."""
    from siteplan_backend.pipeline.trees import (
        detect_trees, apply_size_transform, fill_dense_stands, MAX_CANOPY_RADIUS_M,
    )

    detections = detect_trees(image, scale_m_per_px)
    n_stands = sum(1 for d in detections if d.get("stand"))
    print(f"  Raw detections: {len(detections)} ({n_stands} dense-stand boxes)")
    # Reshape rendered crown sizes (variance then average) before stand fill,
    # so the synthetic fill picks up the transformed size distribution.
    apply_size_transform(
        detections,
        crown_size_scale=crown_size_scale,
        size_variance=size_variance,
    )
    if stand_fill:
        # fill_dense_stands puts synthetic trees AFTER real detections so
        # filter_placements gives the real ones priority where they overlap.
        detections = fill_dense_stands(detections, scale_m_per_px)
        print(f"  After stand fill: {len(detections)}")
    else:
        # No fill: draw each stand as a single max-size tree so the dense
        # canopy at least isn't blank.
        for d in detections:
            if d.get("stand"):
                d["radius_m"] = MAX_CANOPY_RADIUS_M
                d["radius_px"] = MAX_CANOPY_RADIUS_M / scale_m_per_px
    return detections


def rasterize_polygons(polygons, img_w: int, img_h: int):
    """Fill shapely Polygons into a uint8 mask (buildings -> exclusion mask
    for land-type detection)."""
    import cv2
    import numpy as np
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def rasterize_roads(roads, img_w: int, img_h: int):
    """Draw road centerlines at their width into a uint8 mask (road exclusion
    mask for land-type detection)."""
    import cv2
    import numpy as np
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for road in roads:
        pts = np.array(road["line"].coords, dtype=np.int32)
        thickness = max(1, int(round(road.get("width_px", 4))))
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
    return mask


def detections_to_placements(detections: list[dict], n_blocks: int = 1) -> list[dict]:
    """Convert accepted detections into tree block placements."""
    rng = random.Random(0)
    return [
        {
            "block_idx": rng.randrange(n_blocks),
            "position": (det["x_px"], det["y_px"]),
            "scale": det["radius_px"] / DEFAULT_BLOCK_RADIUS_PX,
            "rotation": rng.uniform(0, 360),
        }
        for det in detections
    ]
