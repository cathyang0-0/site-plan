"""
Road / path detection.

Input:  PIL Image (aerial RGB)
Output: list of shapely.LineString (centerlines, in pixel coordinates)
        width_px per line (estimated from mask)
"""
import numpy as np
from PIL import Image
from shapely.geometry import LineString
from skimage.morphology import skeletonize
from pathlib import Path

MODEL_WEIGHTS = Path(__file__).parent.parent.parent.parent / "models" / "roads_deeplab.pt"


def detect_roads(image: Image.Image) -> list[dict]:
    """
    Detect roads and return centerlines with estimated widths.

    Returns:
        List of dicts: {"line": LineString (px coords), "width_px": float}
    """
    img_np = np.array(image)

    if MODEL_WEIGHTS.exists():
        mask = _run_deeplab(img_np)
    else:
        raise NotImplementedError("Road model weights not available yet.")

    return _mask_to_centerlines(mask)


def _run_deeplab(img_np: np.ndarray) -> np.ndarray:
    """Run fine-tuned DeepLabV3+ road segmentation."""
    # TODO: load weights and run inference
    raise NotImplementedError


def _mask_to_centerlines(mask: np.ndarray) -> list[dict]:
    """
    Convert binary road mask to smoothed centerlines.

    Steps:
    1. Skeletonize
    2. Build graph (junction detection)
    3. Prune short stubs (< 15px)
    4. Fit cubic spline per edge
    5. Snap endpoints at intersections
    """
    skeleton = skeletonize(mask > 0.5)
    # TODO: implement graph extraction and spline smoothing
    return []
