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
    roofs: RoofStyle = RoofStyle()
    roads: LayerStyle = LayerStyle(line_weight_mm=0.18)
    trees: LayerStyle = LayerStyle(line_weight_mm=0.13)
    land_types: List[LandTypeStyle] = []  # one per detected cluster


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
