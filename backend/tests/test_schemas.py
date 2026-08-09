"""Tests for Pydantic schemas validation (the API contract)."""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import (
    BoundingBox,
    DetectOptions,
    TreeBlockCalibration,
    LayerStyle,
    LandTypeStyle,
    ContourStyle,
    StyleConfig,
    JobRequest,
    JobStatus,
)


class TestBoundingBox:
    def test_valid(self):
        bb = BoundingBox(west=-74.01, south=40.70, east=-74.00, north=40.71)
        assert bb.west == -74.01

    def test_missing_field_raises(self):
        with pytest.raises(Exception):
            BoundingBox(west=-74.01, south=40.70, east=-74.00)  # missing north

    def test_float_fields(self):
        bb = BoundingBox(west=0, south=0, east=1, north=1)
        assert isinstance(bb.west, float)


class TestDetectOptions:
    def test_defaults_mirror_pipeline(self):
        o = DetectOptions()
        assert o.scale_m_per_px == 0.3
        assert o.overture_buildings and o.overture_roads and o.overture_water
        assert o.land_types_engine == "kmeans"   # seg engine is opt-in (license)
        assert o.paved_min_conf is None
        assert o.crown_size_scale == 1.0
        assert o.stand_fill is True

    def test_segmodel_opt_in(self):
        o = DetectOptions(land_types_engine="segmodel", paved_min_conf=0.6)
        assert o.land_types_engine == "segmodel"
        assert o.paved_min_conf == 0.6

    def test_invalid_engine_rejected(self):
        with pytest.raises(Exception):
            DetectOptions(land_types_engine="magic")


class TestLayerStyle:
    def test_defaults(self):
        ls = LayerStyle()
        assert ls.visible is True
        assert ls.color == "#000000"
        assert ls.line_weight_mm == 0.25

    def test_custom_values(self):
        ls = LayerStyle(visible=False, color="#ff0000", line_weight_mm=0.5)
        assert not ls.visible
        assert ls.color == "#ff0000"


class TestLandTypeStyle:
    def test_defaults_defer_to_pipeline(self):
        # None pattern = "use the pipeline's hatch-reference default".
        lt = LandTypeStyle()
        assert lt.label is None
        assert lt.pattern is None
        assert lt.outline_only is False

    def test_pattern_override(self):
        lt = LandTypeStyle(label="vegetation", pattern="AR-SAND",
                           pattern_scale=2.0, pattern_angle_deg=45.0)
        assert lt.pattern == "AR-SAND"
        assert lt.pattern_scale == 2.0

    def test_outline_only(self):
        lt = LandTypeStyle(outline_only=True)
        assert lt.outline_only


class TestContourStyle:
    def test_user_spec_defaults(self):
        # Hairline, light gray; drawn bottom-most (below hatches) in export.
        c = ContourStyle()
        assert c.line_weight_mm == 0.0
        assert c.color == "#c8c8c8"
        assert c.interval_m == 1.0


class TestStyleConfig:
    def test_defaults(self):
        sc = StyleConfig()
        assert sc.roofs.line_weight_mm == 0.40     # heaviest: outline hierarchy
        assert sc.roads.line_weight_mm == 0.18
        assert sc.trees.line_weight_mm == 0.10
        assert sc.land_types == []                 # [] = hatch-reference defaults
        assert isinstance(sc.contours, ContourStyle)

    def test_land_type_overrides(self):
        sc = StyleConfig(land_types=[LandTypeStyle(label="water", pattern="AR-RROOF")])
        assert len(sc.land_types) == 1
        assert sc.land_types[0].pattern == "AR-RROOF"


class TestTreeBlockCalibration:
    def test_valid(self):
        tbc = TreeBlockCalibration(block_index=1, reference_diameter_m=5.0)
        assert tbc.block_index == 1
        assert tbc.reference_diameter_m == 5.0


class TestJobRequest:
    def test_defaults(self):
        req = JobRequest(bbox=BoundingBox(west=-1, south=-1, east=1, north=1))
        assert "roofs" in req.layers and "land_types" in req.layers
        assert isinstance(req.options, DetectOptions)
        assert isinstance(req.style, StyleConfig)
        assert req.block_calibrations == []

    def test_custom_layers(self):
        req = JobRequest(
            bbox=BoundingBox(west=-1, south=-1, east=1, north=1),
            layers=["roofs"],
        )
        assert req.layers == ["roofs"]

    def test_full_request_round_trip(self):
        # A realistic Rhino-client request survives serialize -> parse intact.
        req = JobRequest(
            bbox=BoundingBox(west=-76.5515, south=42.5305, east=-76.5415, north=42.5385),
            options=DetectOptions(land_types_engine="segmodel", crown_size_scale=1.5),
        )
        again = JobRequest.model_validate(req.model_dump())
        assert again == req


class TestJobStatus:
    def test_minimal(self):
        js = JobStatus(job_id="abc123", status="queued")
        assert js.job_id == "abc123"
        assert js.progress is None and js.geometry is None and js.error is None

    def test_with_error(self):
        js = JobStatus(job_id="x", status="failed", error="timeout")
        assert js.error == "timeout"

    def test_with_geometry(self):
        geo = {"type": "FeatureCollection", "features": []}
        js = JobStatus(job_id="y", status="complete", geometry=geo)
        assert js.geometry == geo

    def test_status_values(self):
        for status in ["queued", "running", "complete", "failed"]:
            assert JobStatus(job_id="id", status=status).status == status

    def test_invalid_status_rejected(self):
        # Literal[] turns typo'd statuses into validation errors, not silent bugs.
        with pytest.raises(Exception):
            JobStatus(job_id="id", status="done")
