"""Tests for Pydantic schemas validation."""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import (
    BoundingBox,
    TreeBlockCalibration,
    LayerStyle,
    RoofStyle,
    LandTypeStyle,
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


class TestRoofStyle:
    def test_defaults(self):
        rs = RoofStyle()
        assert rs.hatch is False
        assert rs.hatch_color == "#cccccc"
        assert rs.hatch_angle_deg == 45.0
        assert rs.hatch_spacing_mm == 3.0

    def test_inherits_layer_style(self):
        rs = RoofStyle(visible=False)
        assert not rs.visible

    def test_hatch_enabled(self):
        rs = RoofStyle(hatch=True, hatch_angle_deg=30.0)
        assert rs.hatch
        assert rs.hatch_angle_deg == 30.0


class TestLandTypeStyle:
    def test_defaults(self):
        lt = LandTypeStyle()
        assert lt.hatch_type == "lines"
        assert lt.outline_only is False

    def test_hatch_types(self):
        for ht in ["none", "lines", "crosshatch", "dots", "solid"]:
            lt = LandTypeStyle(hatch_type=ht)
            assert lt.hatch_type == ht

    def test_outline_only(self):
        lt = LandTypeStyle(outline_only=True)
        assert lt.outline_only


class TestStyleConfig:
    def test_defaults(self):
        sc = StyleConfig()
        assert isinstance(sc.roofs, RoofStyle)
        assert sc.roads.line_weight_mm == 0.18
        assert sc.trees.line_weight_mm == 0.10
        assert len(sc.land_types) == 4

    def test_custom_land_types(self):
        sc = StyleConfig(land_types=[LandTypeStyle(), LandTypeStyle(hatch_type="dots")])
        assert len(sc.land_types) == 2


class TestTreeBlockCalibration:
    def test_valid(self):
        tbc = TreeBlockCalibration(block_index=1, reference_diameter_m=5.0)
        assert tbc.block_index == 1
        assert tbc.reference_diameter_m == 5.0

    def test_block_index_zero(self):
        tbc = TreeBlockCalibration(block_index=0, reference_diameter_m=3.0)
        assert tbc.block_index == 0


class TestJobRequest:
    def test_defaults(self):
        req = JobRequest(bbox=BoundingBox(west=-1, south=-1, east=1, north=1))
        assert "roofs" in req.layers
        assert "roads" in req.layers
        assert isinstance(req.style, StyleConfig)
        assert req.block_calibrations == []

    def test_custom_layers(self):
        req = JobRequest(
            bbox=BoundingBox(west=-1, south=-1, east=1, north=1),
            layers=["roofs"],
        )
        assert req.layers == ["roofs"]

    def test_with_calibrations(self):
        cal = TreeBlockCalibration(block_index=0, reference_diameter_m=4.0)
        req = JobRequest(
            bbox=BoundingBox(west=-1, south=-1, east=1, north=1),
            block_calibrations=[cal],
        )
        assert len(req.block_calibrations) == 1


class TestJobStatus:
    def test_minimal(self):
        js = JobStatus(job_id="abc123", status="queued")
        assert js.job_id == "abc123"
        assert js.status == "queued"
        assert js.progress is None
        assert js.geometry is None
        assert js.error is None

    def test_with_error(self):
        js = JobStatus(job_id="x", status="failed", error="timeout")
        assert js.error == "timeout"

    def test_with_geometry(self):
        geo = {"type": "FeatureCollection", "features": []}
        js = JobStatus(job_id="y", status="complete", geometry=geo)
        assert js.geometry == geo

    def test_status_values(self):
        for status in ["queued", "running", "complete", "failed"]:
            js = JobStatus(job_id="id", status=status)
            assert js.status == status
