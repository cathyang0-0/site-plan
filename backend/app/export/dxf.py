"""
DXF export using ezdxf.

Layer drawing order (bottom to top):
  CONTOURS → LANDTYPE_* → ROADS → TREES → ROOFS (white fill) → ROOFS (outline)

All coordinates are in meters, local origin at site bounding box center.
"""
import random
import ezdxf
from ezdxf import colors
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from pathlib import Path


# DXF lineweight values (hundredths of mm)
LW = {
    0.13: 13,
    0.18: 18,
    0.25: 25,
    0.35: 35,
    0.50: 50,
}


def export_dxf(
    output_path: Path,
    buildings: list[Polygon],
    roads: list[dict],
    tree_placements: list[dict],
    tree_block_curves: list[list],   # one list of curve-point-lists per block
    land_types: list[dict],
    style: dict,
    scale_m_per_px: float,
    origin_px: tuple,               # (cx, cy) — pixel coords of local origin
):
    """
    Assemble all geometry into a layered DXF R2018 file.

    Args:
        output_path:        where to write the .dxf file
        buildings:          list of shapely Polygons (pixel coords)
        roads:              list of {"line": LineString (px), "width_px": float}
        tree_placements:    list of {"block_idx", "position" (px), "scale", "rotation"}
        tree_block_curves:  for each block, a list of polyline point-lists
        land_types:         list of {"label", "polygons": MultiPolygon (px), "style": dict}
        style:              full StyleConfig dict
        scale_m_per_px:     conversion factor
        origin_px:          pixel coordinate that maps to (0, 0) in output
    """
    doc = ezdxf.new("R2018", setup=True)
    msp = doc.modelspace()

    def px_to_m(px_coord):
        """Convert pixel (x, y) to local meter coordinates."""
        x = (px_coord[0] - origin_px[0]) * scale_m_per_px
        y = -(px_coord[1] - origin_px[1]) * scale_m_per_px  # flip Y axis
        return (x, y)

    def poly_pts(polygon):
        return [px_to_m(pt) for pt in polygon.exterior.coords]

    # --- Define tree blocks ---
    block_names = []
    for i, curves in enumerate(tree_block_curves):
        block_name = f"TREE_{i}"
        blk = doc.blocks.new(name=block_name)
        for curve_pts in curves:
            pts_m = [px_to_m(pt) for pt in curve_pts]
            blk.add_lwpolyline(pts_m, dxfattribs={"layer": "TREES"})
        block_names.append(block_name)

    # --- Layers ---
    _add_layer(doc, "LANDTYPE", style.get("land_types", [{}])[0].get("color", "#aaaaaa"), 13)
    _add_layer(doc, "ROADS", style.get("roads", {}).get("color", "#333333"), 18)
    _add_layer(doc, "TREES", style.get("trees", {}).get("color", "#333333"), 13)
    _add_layer(doc, "ROOFS_FILL", "#ffffff", 0)
    _add_layer(doc, "ROOFS", style.get("roofs", {}).get("color", "#000000"), 25)

    # --- Land type hatches (drawn first — bottommost) ---
    for lt in land_types:
        lt_style = lt.get("style", {})
        if lt_style.get("outline_only", False):
            _draw_multipolygon_outlines(msp, lt["polygons"], poly_pts, "LANDTYPE")
        else:
            _draw_multipolygon_hatches(msp, lt["polygons"], poly_pts, "LANDTYPE", lt_style)

    # --- Roads ---
    for road in roads:
        line = road["line"]
        pts_m = [px_to_m(pt) for pt in line.coords]
        msp.add_lwpolyline(pts_m, dxfattribs={"layer": "ROADS"})

    # --- Trees (block insertions) ---
    if block_names:
        rng = random.Random(42)  # seeded for reproducibility
        for placement in tree_placements:
            block_name = block_names[placement["block_idx"] % len(block_names)]
            pos_m = px_to_m(placement["position"])
            msp.add_blockref(
                block_name,
                insert=pos_m,
                dxfattribs={
                    "layer": "TREES",
                    "xscale": placement["scale"],
                    "yscale": placement["scale"],
                    "rotation": placement["rotation"],
                },
            )

    # --- Roofs: white fill hatch (masks everything below) ---
    for poly in buildings:
        pts = poly_pts(poly)
        hatch = msp.add_hatch(color=7, dxfattribs={"layer": "ROOFS_FILL"})  # color 7 = white
        hatch.paths.add_polyline_path(pts, is_closed=True)
        hatch.set_pattern_fill("SOLID")

    # --- Roofs: outline on top ---
    for poly in buildings:
        pts = poly_pts(poly)
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "ROOFS"})

    doc.saveas(str(output_path))


def _add_layer(doc, name: str, hex_color: str, lineweight: int):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    layer = doc.layers.new(name=name)
    layer.rgb = (r, g, b)
    layer.lineweight = lineweight


def _draw_multipolygon_outlines(msp, mpoly, poly_pts_fn, layer_name: str):
    if mpoly.is_empty:
        return
    polys = list(mpoly.geoms) if hasattr(mpoly, "geoms") else [mpoly]
    for poly in polys:
        pts = poly_pts_fn(poly)
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer_name})


def _draw_multipolygon_hatches(msp, mpoly, poly_pts_fn, layer_name: str, lt_style: dict):
    if mpoly.is_empty:
        return
    polys = list(mpoly.geoms) if hasattr(mpoly, "geoms") else [mpoly]

    hatch_type = lt_style.get("hatch_type", "lines")
    angle = lt_style.get("hatch_angle_deg", 45.0)
    spacing = lt_style.get("hatch_spacing_mm", 3.0)

    for poly in polys:
        pts = poly_pts_fn(poly)
        hatch = msp.add_hatch(dxfattribs={"layer": layer_name})
        hatch.paths.add_polyline_path(pts, is_closed=True)

        if hatch_type == "solid":
            hatch.set_pattern_fill("SOLID")
        elif hatch_type == "lines":
            hatch.set_pattern_fill("LINE", scale=spacing / 25.4, angle=angle)
        elif hatch_type == "crosshatch":
            hatch.set_pattern_fill("NET", scale=spacing / 25.4, angle=angle)
        elif hatch_type == "dots":
            hatch.set_pattern_fill("DOTS", scale=spacing / 25.4)
        else:
            pass  # outline only — no hatch entity
