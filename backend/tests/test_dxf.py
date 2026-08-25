"""Tests for dxf export — focus on the tree-block coordinate convention."""
import sys
from pathlib import Path

import ezdxf
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent.parent))

from siteplan_backend.export.dxf import export_dxf


def _tree_block():
    """One block = list of curves; here a single diamond curve at radius 20px."""
    return [[(20, 0), (0, 20), (-20, 0), (0, -20), (20, 0)]]  # block: [curve]


def _export(tmp_path, placements, scale=0.3, origin=(500, 400)):
    out = tmp_path / "out.dxf"
    export_dxf(
        out,
        buildings=[Polygon([(480, 380), (520, 380), (520, 420), (480, 420)])],
        roads=[],
        tree_placements=placements,
        tree_block_curves=[_tree_block()],
        land_types=[],
        style={"trees": {"color": "#555555"}},
        scale_m_per_px=scale,
        origin_px=origin,
    )
    return ezdxf.readfile(out)


def _block_extent(doc):
    name = next(b.name for b in doc.blocks if b.name.startswith("TREE_"))
    xs, ys = [], []
    for e in doc.blocks.get(name):
        if e.dxftype() == "LWPOLYLINE":
            for p in e.get_points():
                xs.append(p[0])
                ys.append(p[1])
    return min(xs), max(xs), min(ys), max(ys)


class TestTreeBlockCoordinates:
    def test_block_geometry_centered_on_origin(self, tmp_path):
        # Regression: block curves were run through px_to_m (which subtracts
        # the image origin), offsetting the symbol to the site corner. At
        # insert time scale*offset then flung each tree far from its true
        # position, smearing the whole tree layer. The block must be LOCAL
        # geometry centered on (0,0): 20px at scale 0.3 -> +/-6 m.
        doc = _export(tmp_path, [{"block_idx": 0, "position": (500, 400),
                                  "scale": 1.0, "rotation": 0}])
        xmin, xmax, ymin, ymax = _block_extent(doc)
        assert abs(xmin + 6.0) < 1e-6 and abs(xmax - 6.0) < 1e-6
        assert abs(ymin + 6.0) < 1e-6 and abs(ymax - 6.0) < 1e-6

    def test_block_extent_independent_of_image_origin(self, tmp_path):
        # The block symbol must not move when the image origin changes.
        placement = [{"block_idx": 0, "position": (0, 0), "scale": 1.0, "rotation": 0}]
        doc_a = _export(tmp_path, placement, origin=(500, 400))
        doc_b = _export(tmp_path, placement, origin=(50, 50))
        assert _block_extent(doc_a) == _block_extent(doc_b)

    def test_insert_at_requested_position(self, tmp_path):
        # A tree's insertion point maps straight through px_to_m; since the
        # block is origin-centered, the rendered circle center == this point.
        doc = _export(tmp_path, [{"block_idx": 0, "position": (100, 100),
                                  "scale": 1.4, "rotation": 0}], scale=0.3, origin=(500, 400))
        ins = list(doc.modelspace().query("INSERT"))[0]
        # px_to_m: x=(100-500)*0.3=-120, y=-(100-400)*0.3=90
        assert abs(ins.dxf.insert.x - (-120)) < 1e-6
        assert abs(ins.dxf.insert.y - 90) < 1e-6
        assert abs(ins.dxf.xscale - 1.4) < 1e-6

    def test_insert_uniformly_scaled_all_three_axes(self, tmp_path):
        # All three scale axes must be equal, else the instance is non-uniform
        # in 3D and Rhino refuses in-place block editing (default zscale=1.0
        # against a scaled x/y is the trap).
        doc = _export(tmp_path, [{"block_idx": 0, "position": (100, 100),
                                  "scale": 1.4, "rotation": 0}])
        ins = list(doc.modelspace().query("INSERT"))[0]
        assert ins.dxf.xscale == ins.dxf.yscale == ins.dxf.zscale == 1.4


def _export_with_landtype(tmp_path, label="water"):
    """Export one building + one big land-type region styled by label."""
    from shapely.geometry import MultiPolygon
    from siteplan_backend.pipeline.landtypes import default_hatch_style
    out = tmp_path / "out.dxf"
    region = MultiPolygon([Polygon([(0, 0), (1000, 0), (1000, 800), (0, 800)])])
    export_dxf(
        out,
        buildings=[Polygon([(480, 380), (520, 380), (520, 420), (480, 420)])],
        roads=[],
        tree_placements=[],
        tree_block_curves=[_tree_block()],
        land_types=[{"label": label, "polygons": region,
                     "style": default_hatch_style(label)}],
        style={},
        scale_m_per_px=0.3,
        origin_px=(500, 400),
    )
    return ezdxf.readfile(out)


class TestAcadHatchPatterns:
    def test_water_hatch_reproduces_reference_pattern(self, tmp_path):
        # The written HATCH must carry the AR-RROOF definition verbatim
        # (scale/angle identity) — this is what guarantees the exported file
        # matches the hand-tuned reference in any viewer.
        from siteplan_backend.pipeline.landtypes import ACAD_PATTERNS
        doc = _export_with_landtype(tmp_path, "water")
        hatches = [h for h in doc.modelspace().query("HATCH")
                   if h.dxf.layer == "LANDTYPE_1"]
        assert len(hatches) == 1
        h = hatches[0]
        assert h.dxf.pattern_name == "AR-RROOF"
        ref = ACAD_PATTERNS["AR-RROOF"]
        assert len(h.pattern.lines) == len(ref)
        first, (r_angle, r_base, r_offset, r_dashes) = h.pattern.lines[0], ref[0]
        assert abs(first.angle - r_angle) < 1e-6
        assert abs(first.offset.x - r_offset[0]) < 1e-6
        assert abs(first.offset.y - r_offset[1]) < 1e-6
        assert [round(d, 6) for d in first.dash_length_items] == r_dashes

    def test_all_default_styles_are_acad_patterns(self):
        from siteplan_backend.pipeline.landtypes import (
            DEFAULT_HATCH_STYLES, ACAD_PATTERNS, default_hatch_style,
        )
        for label, style in DEFAULT_HATCH_STYLES.items():
            assert style["hatch_type"] == "acad"
            assert style["hatch_pattern"] in ACAD_PATTERNS
            assert default_hatch_style(label)["hatch_pattern"] == style["hatch_pattern"]


class TestLayerStackingAndAnnotations:
    def test_roofs_layer_comes_after_landtypes_in_table(self, tmp_path):
        # Viewers that break coincident-draw ties by layer-table position need
        # ROOFS above every LANDTYPE_* layer for outlines to keep full weight.
        doc = _export_with_landtype(tmp_path)
        names = [layer.dxf.name for layer in doc.layers]
        assert names.index("ROOFS") > names.index("LANDTYPE_1")

    def test_hatch_boundary_aligns_exactly_with_building(self, tmp_path):
        # The land hatch is clipped exactly on the footprint polygon — no
        # retreat margin (geometries must stay perfectly aligned; the outline-
        # weight problem is solved by the Z staircase, not by a gap).
        # Building edge at x=480px, origin 500, scale 0.3 -> exactly -6.0 m.
        doc = _export_with_landtype(tmp_path)
        h = [h for h in doc.modelspace().query("HATCH")
             if h.dxf.layer == "LANDTYPE_1"][0]
        hole_xs = []
        for path in h.paths:
            if hasattr(path, "vertices"):
                xs = [v[0] for v in path.vertices]
                if -30 < min(xs) and max(xs) < 30:   # the building-hole path
                    hole_xs = xs
        assert hole_xs and abs(min(hole_xs) - (-6.0)) < 1e-6

    def test_z_staircase_fills_below_linework(self, tmp_path):
        # DXF can't carry Rhino draw order (BringToFront is Rhino-side, and
        # Rhino ignores SORTENTSTABLE), so hatches are sunk slightly below the
        # z=0 drawing plane; all linework stays on it. Depth-tested viewers
        # then always draw outlines over hatches, with XY untouched.
        from siteplan_backend.export.dxf import LAND_HATCH_Z
        assert LAND_HATCH_Z < 0
        doc = _export_with_landtype(tmp_path)
        msp = doc.modelspace()
        land = [h for h in msp.query("HATCH") if h.dxf.layer == "LANDTYPE_1"][0]
        assert abs(land.dxf.elevation.z - LAND_HATCH_Z) < 1e-9
        outline = [e for e in msp.query("LWPOLYLINE") if e.dxf.layer == "ROOFS"][0]
        assert outline.dxf.elevation == 0.0  # linework stays on the plane

    def test_self_intersecting_boundary_repaired_before_hatching(self, tmp_path):
        # Rhino refuses hatches with self-intersecting boundaries (the SB
        # ocean bug, user-diagnosed). A bowtie ring must arrive as TWO simple
        # hatches, and every exported ring must be simple.
        import ezdxf
        from shapely.geometry import Polygon, MultiPolygon, LineString
        from siteplan_backend.export.dxf import export_dxf
        out = tmp_path / "bowtie.dxf"
        bowtie = Polygon([(0, 0), (100, 100), (100, 0), (0, 100)])  # crossing ring
        export_dxf(
            output_path=out, buildings=[], roads=[], tree_placements=[],
            tree_block_curves=[[[(0, 0), (1, 0), (1, 1)]]],
            land_types=[{"label": "vegetation",
                         "polygons": MultiPolygon([bowtie]),
                         "style": {"hatch_type": "acad",
                                   "hatch_pattern": "AR-SAND"}}],
            style={}, scale_m_per_px=1.0, origin_px=(50, 50),
        )
        doc = ezdxf.readfile(str(out))
        hs = [h for h in doc.modelspace().query("HATCH")
              if h.dxf.layer == "LANDTYPE_1"]
        assert len(hs) == 2  # split into the two simple triangles
        for h in hs:
            for pp in h.paths:
                if hasattr(pp, "vertices"):
                    ring = [(v[0], v[1]) for v in pp.vertices]
                    assert LineString(ring + ring[:1]).is_simple

    def test_hatch_safe_polygons_unit(self):
        from shapely.geometry import Polygon
        from siteplan_backend.export.dxf import _hatch_safe_polygons
        clean = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        assert _hatch_safe_polygons(clean) == [clean]  # untouched, identity
        pinched = Polygon([(0, 0), (2, 0), (1, 1), (2, 2), (0, 2), (1, 1)])
        parts = _hatch_safe_polygons(pinched)  # touches itself at (1,1)
        assert len(parts) == 2
        assert all(p.is_valid for p in parts)

    def test_giant_water_region_split_into_tile_hatches(self, tmp_path):
        # Rhino won't render the pattern of one giant complex hatch (verified
        # by A/B probe on the Santa Barbara ocean); regions larger than
        # HATCH_TILE_M are grid-split into several hatches. At 0.3 m/px the
        # tile is 1000 px, so a 3000x2800 px ocean must become many hatches.
        import ezdxf
        from shapely.geometry import Polygon, MultiPolygon
        from siteplan_backend.export.dxf import export_dxf
        out = tmp_path / "tiles.dxf"
        ocean = MultiPolygon([Polygon([(0, 0), (3000, 0), (3000, 2800), (0, 2800)])])
        export_dxf(
            output_path=out, buildings=[], roads=[], tree_placements=[],
            tree_block_curves=[[[(0, 0), (1, 0), (1, 1)]]],
            land_types=[{"label": "water", "polygons": ocean,
                         "style": {"hatch_type": "acad",
                                   "hatch_pattern": "AR-RROOF"}}],
            style={}, scale_m_per_px=0.3, origin_px=(1500, 1400),
        )
        doc = ezdxf.readfile(str(out))
        hs = [h for h in doc.modelspace().query("HATCH")
              if h.dxf.layer == "LANDTYPE_1"]
        assert len(hs) == 9  # 3x3 grid of 1000px (=300 m) tiles
        for h in hs:
            assert h.dxf.pattern_name == "AR-RROOF"  # same pattern everywhere

    def test_small_region_stays_single_hatch(self, tmp_path):
        doc = _export_with_landtype(tmp_path)  # small test region
        hs = [h for h in doc.modelspace().query("HATCH")
              if h.dxf.layer == "LANDTYPE_1"]
        assert len(hs) == 1

    def test_no_white_roof_fill(self, tmp_path):
        # Rhino renders pure-white objects BLACK on a white background, so a
        # white "mask" fill becomes a black blob swallowing the roof outline
        # (verified by import A/B test). Roof masking must stay geometric
        # (roads/land clipped against footprints), never a white hatch.
        doc = _export_with_landtype(tmp_path)
        msp = doc.modelspace()
        assert [h for h in msp.query("HATCH") if h.dxf.layer == "ROOFS_FILL"] == []
        assert "ROOFS_FILL" not in [l.dxf.name for l in doc.layers]

    def test_scale_bar_is_single_square_wave(self, tmp_path):
        # Reference style: one continuous alternating outline — no closed
        # boxes, no baseline doubled under the raised segments.
        doc = _export_with_landtype(tmp_path)
        bars = [e for e in doc.modelspace().query("LWPOLYLINE")
                if e.dxf.layer == "SCALEBAR"]
        assert len(bars) == 1
        assert not bars[0].closed

    def test_scale_bar_present_bottom_right(self, tmp_path):
        doc = _export_with_landtype(tmp_path)
        msp = doc.modelspace()
        names = [layer.dxf.name for layer in doc.layers]
        assert "SCALEBAR" in names
        texts = [t for t in msp.query("TEXT") if t.dxf.layer == "SCALEBAR"]
        assert texts, "scale bar labels missing"
        # Total label carries the unit and sits at the drawing's right edge,
        # below its bottom edge.
        unit_labels = [t for t in texts if t.dxf.text.endswith(" m")]
        assert len(unit_labels) == 1
        extmin_y = -400 * 0.3  # bottom of the 800px-tall region at scale 0.3
        assert all(t.dxf.insert.y < extmin_y for t in texts)

    def test_insunits_meters(self, tmp_path):
        doc = _export_with_landtype(tmp_path)
        assert doc.header["$INSUNITS"] == 6  # meters — importers scale right
