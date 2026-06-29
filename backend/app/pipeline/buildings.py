"""
Building / roof outline detection.

Input:  PIL Image (aerial RGB)
Output: list of shapely.Polygon in pixel coordinates
        (caller converts to real-world meters using scale_m_per_px)

Model:  Fine-tuned U-Net (ResNet-50 backbone) on SpaceNet 2 + INRIA dataset.
        Falls back to SAM2 zero-shot if no weights are available.
"""
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.validation import make_valid
from PIL import Image
from pathlib import Path

MODEL_WEIGHTS = Path(__file__).parent.parent.parent.parent / "models" / "buildings_unet.pt"


def detect_buildings(image: Image.Image) -> list[Polygon]:
    """
    Run building segmentation and return regularized roof polygons.

    Args:
        image: PIL RGB image (stitched aerial tiles)

    Returns:
        List of shapely Polygons in pixel coordinates.
    """
    img_np = np.array(image)

    if MODEL_WEIGHTS.exists():
        mask = _run_unet(img_np)
    else:
        mask = _run_sam2_fallback(img_np)

    polygons = _mask_to_polygons(mask, img_np.shape)
    return polygons


def _run_unet(img_np: np.ndarray) -> np.ndarray:
    """Run fine-tuned U-Net inference. Returns binary mask (H x W, uint8)."""
    # TODO: load model weights, run inference
    raise NotImplementedError("U-Net weights not yet available. Using SAM2 fallback.")


def _run_sam2_fallback(img_np: np.ndarray) -> np.ndarray:
    """Zero-shot SAM2 with grid prompts, filtered to building-sized blobs."""
    # TODO: implement SAM2 automatic mask generator
    raise NotImplementedError("SAM2 fallback not yet implemented.")


def _mask_to_polygons(
    mask: np.ndarray,
    img_shape: tuple,
    min_area_px: int = 100,
    max_area_px: int = 500_000,
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

        poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        poly = _orthogonalize(poly)

        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)

    return polygons


def _orthogonalize(poly: Polygon, angle_threshold_deg: float = 10.0) -> Polygon:
    """
    Snap polygon edge angles to the nearest 90° increment
    if they are within angle_threshold_deg of it.
    This creates cleaner rectangular building outlines.
    """
    # TODO: implement iterative angle snapping
    # For now return as-is; this is a non-trivial geometric operation
    return poly
