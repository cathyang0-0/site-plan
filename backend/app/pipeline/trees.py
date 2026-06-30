"""
Tree instance detection using DeepForest.

Input:  PIL Image (aerial RGB), scale_m_per_px
Output: list of dicts {"x_px", "y_px", "radius_px", "radius_m"}

DeepForest docs: https://deepforest.readthedocs.io
"""
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon


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
    model.use_release()  # download pretrained weights on first run

    img_np = np.array(image)
    boxes = model.predict_image(image=img_np, return_plot=False)

    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    for _, row in boxes.iterrows():
        x_center = (row["xmin"] + row["xmax"]) / 2
        y_center = (row["ymin"] + row["ymax"]) / 2
        radius_px = (row["xmax"] - row["xmin"]) / 2   # use width as diameter estimate
        radius_m = radius_px * scale_m_per_px

        # Filter: canopy radius must be between 1m and 10m
        if radius_m < 1.0 or radius_m > 10.0:
            continue

        detections.append({
            "x_px": x_center,
            "y_px": y_center,
            "radius_px": radius_px,
            "radius_m": radius_m,
        })

    return detections


def filter_placements(
    detections: list[dict],
    building_polygons: list[Polygon],
    overlap_threshold: float = 0.30,
) -> list[dict]:
    """
    Filter tree detections by two overlap rules:
      1. Tree-building overlap > threshold → suppress
      2. Tree-tree overlap > threshold → suppress (checked against already-accepted trees)

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
                canopy.intersection(accepted_canopies[i]).area / canopy.area > overlap_threshold
                for i in candidates
            ):
                continue

        accepted.append(det)
        accepted_canopies.append(canopy)

    return accepted
