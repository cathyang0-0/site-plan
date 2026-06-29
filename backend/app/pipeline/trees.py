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


def filter_against_buildings(
    detections: list[dict],
    building_polygons: list[Polygon],
    scale_m_per_px: float,
    overlap_threshold: float = 0.30,
) -> list[dict]:
    """
    Remove trees whose canopy overlaps a building footprint by more than
    overlap_threshold (fraction of canopy area).

    Trees with <=30% overlap are kept — they will be visually masked by
    the roof's white fill in the drawing.
    """
    filtered = []
    for det in detections:
        canopy = Point(det["x_px"], det["y_px"]).buffer(det["radius_px"])
        max_overlap = 0.0
        for building in building_polygons:
            intersection_area = canopy.intersection(building).area
            overlap_ratio = intersection_area / canopy.area
            max_overlap = max(max_overlap, overlap_ratio)

        if max_overlap <= overlap_threshold:
            filtered.append(det)

    return filtered
