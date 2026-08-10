"""Tests for the contour stage — level selection, tracing, scaling, smoothing.
The USGS fetch itself needs network; per suite convention we test the pure
logic on synthetic elevation grids."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.contours import (
    contour_levels, trace_contours, _chaikin_open, MIN_CONTOUR_LEN_PX,
)


class TestLevels:
    def test_levels_snap_to_interval_multiples(self):
        # 3.2–9.8 m at 1.5 m: first multiple above 3.2 is 4.5.
        assert contour_levels(3.2, 9.8, 1.5) == [4.5, 6.0, 7.5, 9.0]

    def test_flat_terrain_has_no_levels(self):
        assert contour_levels(5.0, 5.0, 1.5) == []

    def test_bad_interval_is_empty(self):
        assert contour_levels(0, 10, 0) == []
        assert contour_levels(0, 10, -1) == []


class TestTrace:
    def _ramp(self, h=60, w=80, rise=0.2):
        # Elevation increasing northward: contours are horizontal lines.
        return (np.arange(h, dtype=np.float32)[:, None] * rise) * np.ones((1, w), np.float32)

    def test_ramp_yields_horizontal_contours_scaled_to_image(self):
        dem = self._ramp()                      # 60x80 grid, 0..11.8 m
        out = trace_contours(dem, 1.5, img_w=800, img_h=600, min_len_px=10)
        assert len(out) == len(contour_levels(0.0, 11.8, 1.5))
        for c in out:
            xs = [p[0] for p in c["points"]]
            ys = [p[1] for p in c["points"]]
            assert max(ys) - min(ys) < 1.0      # horizontal (constant y)
            assert max(xs) > 700                # spans the full IMAGE width,
            assert min(xs) < 100                # i.e. scaled 80 grid -> 800 px

    def test_levels_are_recorded(self):
        out = trace_contours(self._ramp(), 3.0, img_w=80, img_h=60, min_len_px=1)
        assert sorted({c["level"] for c in out}) == [3.0, 6.0, 9.0]

    def test_short_speckle_contours_dropped(self):
        # One raised pixel in a flat field: its contour ring is tiny.
        dem = np.zeros((50, 50), np.float32)
        dem[25, 25] = 10.0
        assert trace_contours(dem, 1.5, img_w=100, img_h=100,
                              min_len_px=MIN_CONTOUR_LEN_PX) == []

    def test_all_nan_dem_is_empty(self):
        dem = np.full((10, 10), np.nan, np.float32)
        assert trace_contours(dem, 1.5, img_w=10, img_h=10) == []


class TestChaikinOpen:
    def test_endpoints_fixed(self):
        pts = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
        out = _chaikin_open(pts, iterations=2)
        assert tuple(out[0]) == (0.0, 0.0)
        assert tuple(out[-1]) == (10.0, 10.0)

    def test_corner_gets_cut(self):
        pts = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
        out = _chaikin_open(pts, iterations=2)
        assert not any(tuple(p) == (10.0, 0.0) for p in out)


class TestExportWiring:
    def test_contours_layer_bottom_of_table_and_staircase(self, tmp_path):
        import ezdxf
        from app.export.dxf import export_dxf, CONTOUR_Z, LAND_HATCH_Z
        assert CONTOUR_Z < LAND_HATCH_Z  # below the hatches, per user spec
        out = tmp_path / "c.dxf"
        export_dxf(
            output_path=out, buildings=[], roads=[], tree_placements=[],
            tree_block_curves=[[[(0, 0), (1, 0), (1, 1)]]], land_types=[],
            style={"contours": {"color": "#dcdcdc", "line_weight_mm": 0.05}},
            scale_m_per_px=0.3, origin_px=(50, 50),
            contours=[{"points": [(0, 10), (40, 12), (80, 10)], "level": 4.5}],
        )
        doc = ezdxf.readfile(str(out))
        lines = [e for e in doc.modelspace().query("LWPOLYLINE")
                 if e.dxf.layer == "CONTOURS"]
        assert len(lines) == 1
        assert abs(lines[0].dxf.elevation - CONTOUR_Z) < 1e-9
        assert not lines[0].closed  # open polyline, not a ring
