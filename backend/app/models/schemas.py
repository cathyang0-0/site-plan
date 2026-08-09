"""
API contract (Pydantic models) — the "order form" between any client (web app,
Rhino command) and the pipeline service.

Modernized 2026-08 to describe the pipeline that actually exists (poc.py):
Overture buildings/roads/water, k-means or SegFormer land cover, DeepForest
trees, the hatch-reference export styling. Fields for never-built features
(ridge lines, site boundary, the old lines/dots/crosshatch hatch model) were
removed — the contract must not promise what the kitchen can't cook. Contours
are included ahead of the stage being built (USGS elevation; user-requested).
"""
from pydantic import BaseModel
from typing import Optional, List, Literal


class BoundingBox(BaseModel):
    """Geographic bounding box in WGS84 (lon/lat)."""
    west: float
    south: float
    east: float
    north: float


class DetectOptions(BaseModel):
    """Knobs for the detection stages — mirrors poc.py's real flags."""
    scale_m_per_px: float = 0.3          # imagery resolution to fetch/run at
    # Open-data sources (Overture). Primary when georeferenced; CV falls back.
    overture_buildings: bool = True
    overture_roads: bool = True
    overture_water: bool = True
    # Land-cover engine. k-means is the default; the SegFormer engine is
    # evaluation-only until the OpenEarthMap license question is resolved, so
    # clients must opt in explicitly (same posture as poc.py).
    land_types_engine: Literal["kmeans", "segmodel"] = "kmeans"
    paved_min_conf: Optional[float] = None   # None = engine default
    # Trees (DeepForest).
    crown_size_scale: float = 1.0
    size_variance: Optional[float] = None    # None = pipeline default
    stand_fill: bool = True


class TreeBlockCalibration(BaseModel):
    block_index: int             # 0, 1, or 2
    reference_diameter_m: float  # real-world canopy diameter this block represents


class LayerStyle(BaseModel):
    visible: bool = True
    color: str = "#000000"       # hex
    line_weight_mm: float = 0.25


class LandTypeStyle(LayerStyle):
    """Style override for one land-cover layer.

    `pattern` is an AutoCAD hatch pattern name (e.g. "AR-SAND"). The default
    styling — the user's hatch-reference patterns per label — lives in the
    export pipeline (landtypes.ACAD_PATTERNS); an empty StyleConfig.land_types
    means "use those defaults". Overrides here are per-layer, in label order.
    """
    label: Optional[str] = None      # "water" | "vegetation" | ... (None = positional)
    pattern: Optional[str] = None    # None = pipeline default for this label
    pattern_scale: float = 1.0
    pattern_angle_deg: float = 0.0
    outline_only: bool = False       # skip hatch entity, draw boundary only


class ContourStyle(LayerStyle):
    """Topographic contours (stage pending: USGS elevation data).
    User spec: as close to hairline as DXF allows, bottom-most in draw order —
    below the land hatches in the export's Z staircase. True hairline is a
    Rhino-only special value DXF can't carry, so: thinnest real DXF weight
    (0.05 mm) compensated with a very light gray."""
    color: str = "#dcdcdc"
    line_weight_mm: float = 0.05
    interval_m: float = 1.0


class StyleConfig(BaseModel):
    # Defaults mirror poc.py's architectural hierarchy: roofs heaviest,
    # roads secondary, trees/land texture lightest.
    roofs: LayerStyle = LayerStyle(color="#000000", line_weight_mm=0.40)
    roads: LayerStyle = LayerStyle(color="#333333", line_weight_mm=0.18)
    trees: LayerStyle = LayerStyle(color="#555555", line_weight_mm=0.10)
    # Empty list = the pipeline's hatch-reference defaults (one entry per
    # detected label, in label order, when overriding).
    land_types: List[LandTypeStyle] = []
    contours: ContourStyle = ContourStyle()


class JobRequest(BaseModel):
    bbox: BoundingBox
    layers: List[str] = ["roofs", "roads", "trees", "land_types"]
    options: DetectOptions = DetectOptions()
    style: StyleConfig = StyleConfig()
    block_calibrations: List[TreeBlockCalibration] = []


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "failed"]
    progress: Optional[dict] = None   # per-stage status while running
    geometry: Optional[dict] = None   # GeoJSON FeatureCollection when complete
    error: Optional[str] = None
