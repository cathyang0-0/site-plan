from pydantic import BaseModel
from typing import Optional, List


class BoundingBox(BaseModel):
    """Geographic bounding box in WGS84 (lon/lat)."""
    west: float
    south: float
    east: float
    north: float


class TreeBlockCalibration(BaseModel):
    block_index: int          # 0, 1, or 2
    reference_diameter_m: float  # real-world canopy diameter this block represents


class LayerStyle(BaseModel):
    visible: bool = True
    color: str = "#000000"      # hex
    line_weight_mm: float = 0.25


class RoofStyle(LayerStyle):
    hatch: bool = False         # white fill is always on; this adds an extra hatch
    hatch_color: str = "#cccccc"
    hatch_angle_deg: float = 45.0
    hatch_spacing_mm: float = 3.0


class LandTypeStyle(LayerStyle):
    hatch_type: str = "lines"   # "none" | "lines" | "crosshatch" | "dots" | "solid"
    hatch_color: str = "#cccccc"
    hatch_angle_deg: float = 45.0
    hatch_spacing_mm: float = 3.0
    outline_only: bool = False  # fallback: skip hatch entity, draw boundary only


class StyleConfig(BaseModel):
    site_boundary: LayerStyle = LayerStyle(color="#000000", line_weight_mm=0.35)
    roofs: RoofStyle = RoofStyle(color="#1a1a1a", line_weight_mm=0.25)
    ridge_lines: LayerStyle = LayerStyle(color="#444444", line_weight_mm=0.10)
    ridge_lines_uncertain: LayerStyle = LayerStyle(color="#aaaaaa", line_weight_mm=0.05)
    roads: LayerStyle = LayerStyle(color="#333333", line_weight_mm=0.18)
    trees: LayerStyle = LayerStyle(color="#555555", line_weight_mm=0.10)
    contours: LayerStyle = LayerStyle(color="#999999", line_weight_mm=0.05)
    land_types: List[LandTypeStyle] = [
        # defaults match detected cluster order: water, vegetation, bare earth, paved
        LandTypeStyle(color="#aaaaaa", line_weight_mm=0.05, hatch_type="lines",
                      hatch_color="#888888", hatch_angle_deg=0.0,  hatch_spacing_mm=2.0),
        LandTypeStyle(color="#aaaaaa", line_weight_mm=0.05, hatch_type="dots",
                      hatch_color="#777777", hatch_angle_deg=0.0,  hatch_spacing_mm=1.5),
        LandTypeStyle(color="#aaaaaa", line_weight_mm=0.05, hatch_type="lines",
                      hatch_color="#888888", hatch_angle_deg=45.0, hatch_spacing_mm=3.0),
        LandTypeStyle(color="#aaaaaa", line_weight_mm=0.05, hatch_type="crosshatch",
                      hatch_color="#666666", hatch_angle_deg=45.0, hatch_spacing_mm=2.5),
    ]


class JobRequest(BaseModel):
    bbox: BoundingBox
    layers: List[str] = ["roofs", "roads", "trees", "land_types"]
    style: StyleConfig = StyleConfig()
    block_calibrations: List[TreeBlockCalibration] = []


class JobStatus(BaseModel):
    job_id: str
    status: str             # "queued" | "running" | "complete" | "failed"
    progress: Optional[dict] = None   # per-module status
    geometry: Optional[dict] = None   # GeoJSON FeatureCollection when complete
    error: Optional[str] = None
