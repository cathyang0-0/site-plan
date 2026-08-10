"""Unit-native export tests: the DXF must be written entirely in the requested
unit — geometry, hatch pattern spacing, Z staircase, texts — so importers never
unit-convert (conversion loses hatch spacings; the user's mm document showed
patterns as invisible dust)."""
import sys
from pathlib import Path

import ezdxf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from shapely.geometry import Polygon, MultiPolygon
from app.export.dxf import (
    export_dxf, UNIT_FACTOR, UNIT_INSUNITS,
    LAND_HATCH_Z, CONTOUR_Z, _scale_pattern_definition,
)


def _export(tmp_path, units):
    out = tmp_path / f"u_{units}.dxf"
    building = Polygon([(400, 400), (600, 400), (600, 600), (400, 600)])
    land = MultiPolygon([Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])])
    export_dxf(
        output_path=out,
        buildings=[building], roads=[], tree_placements=[],
        tree_block_curves=[[[(0, 0), (10, 0), (10, 10)]]],
        land_types=[{"label": "vegetation", "polygons": land,
                     "style": {"hatch_type": "acad", "hatch_pattern": "AR-SAND",
                               "hatch_color": "#c8c8c8"}}],
        style={}, scale_m_per_px=0.5, origin_px=(500, 500),
        contours=[{"points": [(0, 100), (900, 120)], "level": 3.0}],
        units=units,
    )
    return ezdxf.readfile(str(out))


class TestUnitNativeExport:
    def test_insunits_declares_the_requested_unit(self, tmp_path):
        assert _export(tmp_path, "m").header["$INSUNITS"] == 6
        assert _export(tmp_path, "mm").header["$INSUNITS"] == 4

    def test_geometry_scales_by_unit_factor(self, tmp_path):
        # Building edge at (400-500)*0.5 = -50 m -> -50000 mm.
        for units, want in (("m", -50.0), ("mm", -50000.0), ("ft", -50 / 0.3048)):
            doc = _export(tmp_path, units)
            roof = [e for e in doc.modelspace().query("LWPOLYLINE")
                    if e.dxf.layer == "ROOFS"][0]
            xs = [p[0] for p in roof.get_points("xy")]
            assert min(xs) == pytest.approx(want, rel=1e-6)

    def test_z_staircase_scales_with_units(self, tmp_path):
        doc = _export(tmp_path, "mm")
        msp = doc.modelspace()
        hatch = [h for h in msp.query("HATCH") if h.dxf.layer == "LANDTYPE_1"][0]
        assert hatch.dxf.elevation.z == pytest.approx(LAND_HATCH_Z * 1000)
        contour = [e for e in msp.query("LWPOLYLINE")
                   if e.dxf.layer == "CONTOURS"][0]
        assert contour.dxf.elevation == pytest.approx(CONTOUR_Z * 1000)

    def test_hatch_pattern_spacing_scales(self, tmp_path):
        # The reference pattern's offset vector must be 1000x in a mm export —
        # this is the exact bug the user saw (patterns invisible in an mm doc).
        m = _export(tmp_path, "m")
        mm = _export(tmp_path, "mm")
        def offset_len(doc):
            h = [h for h in doc.modelspace().query("HATCH")
                 if h.dxf.layer == "LANDTYPE_1"][0]
            line = h.pattern.lines[0]
            return (line.offset[0] ** 2 + line.offset[1] ** 2) ** 0.5
        assert offset_len(mm) == pytest.approx(offset_len(m) * 1000, rel=1e-6)

    def test_scale_bar_labels_stay_in_meters(self, tmp_path):
        # Geometry in mm, but the bar states real-world meters.
        doc = _export(tmp_path, "mm")
        texts = [t.dxf.text for t in doc.modelspace().query("TEXT")
                 if t.dxf.layer == "SCALEBAR"]
        assert any(t.endswith(" m") for t in texts)

    def test_unknown_unit_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown units"):
            _export(tmp_path, "furlong")


class TestPatternScaling:
    def test_definition_scales_lengths_not_angles(self):
        d = [[45.0, (1.0, 2.0), (3.0, 4.0), [5.0, -6.0]]]
        [(angle, base, off, dashes)] = _scale_pattern_definition(d, 1000.0)
        assert angle == 45.0
        assert base == (1000.0, 2000.0) and off == (3000.0, 4000.0)
        assert dashes == [5000.0, -6000.0]

    def test_identity_returns_same_object(self):
        d = [[0.0, (0.0, 0.0), (1.0, 0.0), []]]
        assert _scale_pattern_definition(d, 1.0) is d
