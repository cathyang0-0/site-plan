"""Tests for dxf export — focus on the tree-block coordinate convention."""
import sys
from pathlib import Path

import ezdxf
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.export.dxf import export_dxf


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
