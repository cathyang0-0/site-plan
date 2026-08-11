"""
DXF export using ezdxf.

Layer drawing order (bottom to top):
  CONTOURS → LANDTYPE_* → ROADS → TREES → ROOFS (outline; interiors are
  empty paper via geometric clipping — no white fill, Rhino renders white as black)

All coordinates are in meters, local origin at site bounding box center.

Roads with a defined width are drawn as the boundary of their merged,
filleted pavement corridor (see _build_road_network) rather than as
independent per-road offset edges — this makes intersections and
fillets a natural side effect of a single polygon boolean, instead of
needing explicit graph-based junction detection.
"""
import random
import ezdxf
import ezdxf.bbox
from ezdxf import colors
from ezdxf.enums import TextEntityAlignment
from ezdxf.lldxf.const import VALID_DXF_LINEWEIGHTS
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union
from pathlib import Path

# Default lineweights (mm) per layer, matching spec.md §5's architectural
# hierarchy: site boundary > roof outlines > roads > ridge lines/trees >
# land hatches. Used when the caller's style dict doesn't override a value.
DEFAULT_LINE_WEIGHT_MM = {
    "ROOFS": 0.40,
    "ROADS": 0.18,
    "TREES": 0.10,
    "LANDTYPE": 0.05,
}

# spec.md §4b default fillet radius, applied at road/road intersections.
DEFAULT_FILLET_RADIUS_M = 3.0

# Land-type hatch appearance: lightest lineweight and a light gray so the
# hatch reads as background texture and the roof outlines stay dominant.
HATCH_LINEWEIGHT = 5          # 0.05 mm (thinnest valid DXF lineweight)
HATCH_RGB = (200, 200, 200)   # light gray

# Z staircase: force fills below linework WITHOUT touching XY geometry.
# DXF cannot carry Rhino's per-object draw order (BringToFront is a Rhino
# attribute; Rhino ignores DXF's SORTENTSTABLE on import), so coplanar hatches
# and curves z-fight and hatch pattern lines can render over the roof outline
# stroke, visually thinning it. Sinking the land hatches slightly below the
# drawing plane makes every depth-tested viewer draw curves (at z=0) on top
# deterministically. Centimeters of depth are invisible in plan and
# boundaries stay perfectly aligned in XY.
LAND_HATCH_Z = -0.10
# Contours sit at the very bottom of the staircase — under the land hatches —
# per the user's spec (near-hairline light gray, bottom-most draw order).
CONTOUR_Z = -0.15

# Unit-neutral export: modelspace can be written natively in the CLIENT
# DOCUMENT's unit so importers never have to unit-convert (conversion is where
# hatch pattern spacings get lost — Rhino scales geometry but not pattern
# definitions, turning a 0.3 m spacing into 0.3 mm of invisible dust). All
# internal constants are authored in meters and multiplied by UNIT_FACTOR at
# draw time; DXF lineweights are the exception (always defined in mm).
UNIT_FACTOR = {"m": 1.0, "mm": 1000.0, "cm": 100.0,
               "ft": 1 / 0.3048, "in": 1 / 0.0254}
UNIT_INSUNITS = {"in": 1, "ft": 2, "mm": 4, "cm": 5, "m": 6}

# Scale bar: target fraction of the site width; snapped to a nice round length.
SCALE_BAR_FRACTION = 0.15

def _nearest_dxf_lineweight(mm: float) -> int:
    """Snap an arbitrary mm value to the nearest DXF-valid lineweight code
    (hundredths of a mm). DXF only accepts a fixed enum of lineweights
    (see ezdxf.lldxf.const.VALID_DXF_LINEWEIGHTS) -- arbitrary values are
    rejected, so every style mm value must be snapped to this set."""
    hundredths = mm * 100
    valid = [v for v in VALID_DXF_LINEWEIGHTS if v >= 0]
    return min(valid, key=lambda v: abs(v - hundredths))


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
    attribution: str | None = None,  # data-license credit (e.g. ODbL requires
                                     # it for Overture footprints); drawn as a
                                     # small gray note below the site
    contours: list[dict] | None = None,  # [{"points": [(x,y) px], "level": m}]
    units: str = "m",                # output drawing unit (see UNIT_FACTOR)
):
    """
    Assemble all geometry into a layered DXF R2010 file.

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
    if units not in UNIT_FACTOR:
        raise ValueError(f"unknown units {units!r}; expected one of {sorted(UNIT_FACTOR)}")
    u = UNIT_FACTOR[units]           # meters -> output-unit multiplier
    upx = scale_m_per_px * u         # output units per pixel
    # NOTE: pixel-domain math (fillet radius, road widths) keeps using
    # scale_m_per_px — only the px->drawing conversions below use upx.

    # R2010, deliberately not the newest dialect: nothing this exporter
    # writes needs post-R2010 features, and the older dialect maximizes
    # importer compatibility. (An earlier comment blamed Rhino's R2018
    # reader for scrambled imports — that was a confounded experiment; the
    # actual culprit was Rhino's scripted '_-Import' macro, fixed on the
    # Rhino-command side by using RhinoDoc.Import instead.)
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()

    def px_to_m(px_coord):
        """Convert pixel (x, y) to local drawing-unit coordinates."""
        x = (px_coord[0] - origin_px[0]) * upx
        y = -(px_coord[1] - origin_px[1]) * upx  # flip Y axis
        return (x, y)

    def poly_pts(polygon):
        return [px_to_m(pt) for pt in polygon.exterior.coords]

    def poly_rings(polygon):
        """(exterior_pts, [hole_pts, ...]) -- a polygon's interior rings
        matter whenever it's been clipped against a building that sits
        fully inside it (not touching its outer edge): the building only
        shows up as a hole, not a bite out of the exterior boundary."""
        exterior = [px_to_m(pt) for pt in polygon.exterior.coords]
        holes = [[px_to_m(pt) for pt in ring.coords] for ring in polygon.interiors]
        return exterior, holes

    def block_px_to_m(px_coord):
        """Scale a block's local symbol coordinate from pixels to meters.

        A block definition is LOCAL geometry, relative to its own origin --
        it must be scaled but NOT translated by the image origin the way
        px_to_m does. (Running block curves through px_to_m offsets the
        symbol to the site corner; at insert time `scale * that_offset`
        then flings each tree far from its true position, varying per tree
        as the scale varies -- the classic "trees scattered everywhere"
        bug.) The Y sign is flipped to match px_to_m's axis convention so
        the symbol isn't mirrored relative to world geometry.
        """
        return (px_coord[0] * upx, -px_coord[1] * upx)

    # --- Define tree blocks ---
    # Block names carry a per-export random token. Rhino (and some other
    # CAD apps) key block definitions by name across imports into one
    # document: re-importing a regenerated plan whose block is still called
    # "TREE_0" makes Rhino reuse the FIRST "TREE_0" it saw and ignore the
    # new definition -- so a corrected symbol silently renders with the
    # stale one. A unique suffix per export sidesteps that entirely.
    export_token = f"{random.randrange(16**6):06X}"
    block_names = []
    for i, curves in enumerate(tree_block_curves):
        block_name = f"TREE_{i}_{export_token}"
        blk = doc.blocks.new(name=block_name)
        for curve_pts in curves:
            pts_m = [block_px_to_m(pt) for pt in curve_pts]
            blk.add_lwpolyline(pts_m, dxfattribs={"layer": "TREES"})
        block_names.append(block_name)

    # --- Layers ---
    # Each land type gets its OWN layer (LANDTYPE_1, LANDTYPE_2, ...) so the
    # user can restyle or toggle each ground cover independently (created in
    # the land-type loop below, one per detected type). The layer TABLE is
    # built strictly bottom-to-top — land types first, ROOFS last — mirroring
    # the entity order, because some viewers break draw-order ties between
    # coincident objects by layer position; ROOFS being the topmost layer is
    # what keeps roof outlines at full weight over adjacent hatches.
    # Layer lineweights come from the caller's style dict when provided,
    # falling back to the spec.md §5 architectural defaults.
    def _add_upper_layers():
        _add_layer(
            doc, "ROADS",
            style.get("roads", {}).get("color", "#333333"),
            style.get("roads", {}).get("line_weight_mm", DEFAULT_LINE_WEIGHT_MM["ROADS"]),
        )
        _add_layer(
            doc, "TREES",
            style.get("trees", {}).get("color", "#333333"),
            style.get("trees", {}).get("line_weight_mm", DEFAULT_LINE_WEIGHT_MM["TREES"]),
        )
        _add_layer(
            doc, "ROOFS",
            style.get("roofs", {}).get("color", "#000000"),
            style.get("roofs", {}).get("line_weight_mm", DEFAULT_LINE_WEIGHT_MM["ROOFS"]),
        )

    # Roof-fill "masking" (relying on CAD draw order/SORTENTSTABLE to paint
    # the white roof fill over anything underneath) has proven unreliable
    # across viewers and plot modes. Instead, actually remove the parts of
    # roads and land-type regions that fall under a building footprint
    # before drawing them, so there's no overlap left for any viewer to
    # get wrong. Trees are intentionally left to draw order — a tree
    # canopy is allowed up to 30% overlap with a building by design
    # (filter_placements), and clipping a block INSERT isn't meaningful.
    buildings_union = unary_union(buildings) if buildings else None
    buildings_union = (
        buildings_union if buildings_union is not None and not buildings_union.is_empty else None
    )

    # Roads with a defined width are merged into a single pavement corridor
    # polygon per connected network, with concave (inside) corners at
    # intersections rounded into fillets -- see _build_road_network for why
    # this replaces drawing each road's offset edges independently. Computed
    # up front (before land types) so land-type regions can be clipped
    # against the road network too, not just buildings.
    fillet_radius_px = DEFAULT_FILLET_RADIUS_M / scale_m_per_px
    widthed_roads = [r for r in roads if r.get("width_px", 0) > 0]
    zero_width_roads = [r for r in roads if r.get("width_px", 0) <= 0]
    road_network = _build_road_network(widthed_roads, fillet_radius_px, buildings_union) if widthed_roads else None
    road_network = road_network if road_network is not None and not road_network.is_empty else None

    # Anything a land-type region should never be drawn under -- buildings
    # (masked by roof fill) and now roads (pavement, not ground cover).
    # Clipped exactly, boundary-on-boundary; the outline-weight problem is
    # solved by the Z staircase (LAND_HATCH_Z), not by retreating the hatch.
    landtype_clip = unary_union(
        [g for g in (buildings_union, road_network) if g is not None]
    ) if (buildings_union is not None or road_network is not None) else None

    # --- Contours (the very bottom: first layer in the table, lowest Z) ---
    if contours:
        c_style = style.get("contours", {})
        _add_layer(doc, "CONTOURS",
                   c_style.get("color", "#dcdcdc"),
                   c_style.get("line_weight_mm", 0.05))
        for c in contours:
            if len(c["points"]) < 2:
                continue
            geom = LineString(c["points"])
            # Clip out the runs crossing building footprints — roof interiors
            # are empty paper (no white mask fill anymore), so an unclipped
            # contour would draw straight through a building.
            if buildings_union is not None:
                geom = geom.difference(buildings_union)
            parts = geom.geoms if hasattr(geom, "geoms") else [geom]
            for part in parts:
                if part.is_empty or part.geom_type != "LineString":
                    continue
                pts = [px_to_m(p) for p in part.coords]
                msp.add_lwpolyline(pts, dxfattribs={
                    "layer": "CONTOURS",
                    # Bottom of the Z staircase: under land hatches, so every
                    # depth-tested viewer draws hatches and linework over them.
                    "elevation": CONTOUR_Z * u,
                })

    # --- Land type hatches (drawn first — bottommost), one layer per type ---
    for i, lt in enumerate(land_types):
        lt_style = lt.get("style", {})
        layer_name = f"LANDTYPE_{i + 1}"
        _add_layer(
            doc, layer_name,
            lt_style.get("hatch_color", "#c8c8c8"),
            lt_style.get("line_weight_mm", DEFAULT_LINE_WEIGHT_MM["LANDTYPE"]),
        )
        polygons = lt["polygons"]
        if landtype_clip is not None:
            polygons = _clip_polygons(polygons, landtype_clip)
        if lt_style.get("outline_only", False):
            _draw_multipolygon_outlines(msp, polygons, poly_rings, layer_name)
        else:
            _draw_multipolygon_hatches(msp, polygons, poly_rings, layer_name, lt_style, u=u)

    # Upper layers registered only now, so they land ABOVE every LANDTYPE_*
    # layer in the table (see the layer-order comment above).
    _add_upper_layers()

    # --- Roads ---
    if road_network is not None:
        polys = list(road_network.geoms) if hasattr(road_network, "geoms") else [road_network]
        for poly in polys:
            if poly.is_empty:
                continue
            for ring in [poly.exterior, *poly.interiors]:
                # This boundary is already a dense, smooth polygon straight
                # from Shapely's buffer/fillet math -- fitting a SPLINE
                # through a resampling of it (as elsewhere in this file)
                # risks the CAD-side curve interpolation overshooting
                # between sample points, which is exactly what reopened a
                # small gap into a building near the clipped edge here.
                # Drawing the exact computed points as a polyline has no
                # overshoot; a light simplify() only drops redundant
                # near-collinear points on long straight stretches, it
                # doesn't touch the fillet's curved detail.
                simplified = ring.simplify(1.0, preserve_topology=True)
                pts_m = [px_to_m(pt) for pt in simplified.coords]
                msp.add_lwpolyline(pts_m, close=True, dxfattribs={"layer": "ROADS"})

    for road in zero_width_roads:
        line = road["line"]
        segments = _clip_line(line, buildings_union) if buildings_union is not None else [line]
        for segment in segments:
            if segment.length == 0:
                continue
            pts_m = [px_to_m(pt) for pt in _resample_line(segment)]
            msp.add_spline(fit_points=pts_m, dxfattribs={"layer": "ROADS"})

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
                    # zscale must equal x/yscale: with the default zscale=1.0
                    # a tree scaled to e.g. 1.3 is (1.3, 1.3, 1.0) -- uniform
                    # in plan but NON-uniform in 3D, which makes Rhino refuse
                    # in-place block editing. Setting all three equal keeps
                    # the instance uniformly scaled and editable.
                    "zscale": placement["scale"],
                    "rotation": placement["rotation"],
                },
            )

    # NO white roof-fill hatch: Rhino draws pure-white objects BLACK on a
    # white background (adaptive display color), so a white "mask" fill
    # renders as solid black blobs swallowing the roof outlines (verified by
    # A/B test on import). Masking is GEOMETRIC instead — roads and land-type
    # regions are clipped against building footprints above, so roof
    # interiors are genuinely empty paper. Trees may overlap roofs by up to
    # 30% by design; that reads as canopy over the building, as in the
    # user's reference plans.

    # --- Roofs: outline on top ---
    for poly in buildings:
        pts = poly_pts(poly)
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "ROOFS"})

    # --- Margin annotations: attribution (bottom-left) + scale bar (bottom-
    # right), both placed against the drawing extents computed BEFORE either
    # is added so they don't feed back into each other's placement. ---
    extents = ezdxf.bbox.extents(msp, fast=True)

    if attribution and extents.has_data:
        _add_layer(doc, "NOTES", "#999999", 0.05)
        text = msp.add_text(
            attribution,
            height=3.0 * u,  # 3 m; legible at typical context-plan scales
            dxfattribs={"layer": "NOTES"},
        )
        text.set_placement(
            (extents.extmin.x, extents.extmin.y - 6.0 * u),
            align=TextEntityAlignment.TOP_LEFT,
        )

    if extents.has_data:
        _draw_scale_bar(msp, doc, extents, u=u)

    # NO SORTENTSTABLE: draw order is carried entirely by the Z staircase +
    # bottom-to-top entity order. The explicit redraw-order table was
    # belt-and-suspenders that became a liability — a 1000+-entry table was
    # the prime suspect when Rhino's importer dropped whole layers, and no
    # target viewer needs it (Rhino ignores it at best; AutoCAD respects the
    # Z/entity order fine).

    # Declare real-world units: modelspace is meters. Without $INSUNITS a DXF
    # is unitless and importers guess (Rhino defaulted to mm, shrinking the
    # whole site 1000x on import — a tree read as "10 mm"). 6 = meters.
    doc.header["$INSUNITS"] = UNIT_INSUNITS[units]
    doc.header["$MEASUREMENT"] = 1  # metric

    doc.saveas(str(output_path))


def _build_road_network(roads: list[dict], fillet_radius_px: float, buildings_union):
    """
    Merge every road's pavement corridor into one polygon (or one polygon
    per disconnected road group) and round its concave (inside) corners.

    Drawing each road's two offset edges independently, as before, means
    two roads crossing or meeting at a junction produce edges that
    literally cross each other -- there's no shared understanding that
    they're the same intersection. Buffering each centerline out to its
    road width turns every road into a polygon; unioning those polygons
    merges any that touch or cross into one shape, so an intersection is
    just wherever two corridors overlap -- no separate junction-detection
    step needed.

    The corners at that merge are sharp by construction (buffer() with
    flat/round caps doesn't know about fillets). A round "closing"
    morphological operation -- dilate by the fillet radius, then erode by
    the same amount -- fills in concave notches up to that radius,
    which is exactly a fillet at each inside corner, while leaving convex
    (outside) corners untouched. This matches spec.md §4b's fillet
    description without needing explicit per-corner curvature detection.
    """
    corridors = [road["line"].buffer(road["width_px"] / 2, cap_style="flat") for road in roads]
    network = unary_union(corridors)
    network = network.buffer(fillet_radius_px, join_style="round").buffer(
        -fillet_radius_px, join_style="round"
    )
    if buildings_union is not None:
        network = network.difference(buildings_union)
    return network


def _clip_line(line: LineString, clip_against) -> list[LineString]:
    """Remove the portions of `line` that fall inside `clip_against`,
    returning the surviving piece(s) as a list of LineStrings."""
    remainder = line.difference(clip_against)
    if remainder.is_empty:
        return []
    if remainder.geom_type == "LineString":
        return [remainder]
    if remainder.geom_type == "MultiLineString":
        return list(remainder.geoms)
    return []  # degenerate result (e.g. Point) -- nothing meaningful to draw


def _clip_polygons(mpoly, clip_against):
    """Subtract `clip_against` from a Polygon/MultiPolygon, discarding any
    non-polygonal slivers the boolean difference may produce."""
    if mpoly.is_empty:
        return mpoly
    remainder = mpoly.difference(clip_against)
    if remainder.is_empty:
        return remainder
    if remainder.geom_type in ("Polygon", "MultiPolygon"):
        return remainder
    if remainder.geom_type == "GeometryCollection":
        polys = [g for g in remainder.geoms if g.geom_type == "Polygon"]
        return MultiPolygon(polys) if polys else MultiPolygon()
    return MultiPolygon()


def _resample_line(line: LineString, n_samples: int = 30) -> list[tuple]:
    """
    Evenly resample a LineString by arc length.

    Fixes two related artifacts from shapely's offset_curve(): its raw
    output has an uneven, side-dependent point density, which both (a)
    makes one offset edge visually read as a smoother curve than the
    other, and (b) can make the CAD-side spline interpolation through
    sparse/uneven fit points overshoot and cross the paired edge near
    vertices. Resampling at even arc-length intervals removes both.
    """
    length = line.length
    if length == 0:
        return list(line.coords)
    n_samples = max(n_samples, 2)
    return [
        tuple(line.interpolate(i / (n_samples - 1) * length).coords[0])
        for i in range(n_samples)
    ]


def _hex_to_rgb(hex_color: str) -> tuple:
    """'#rrggbb' -> (r, g, b) ints."""
    return (int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16))


def _add_layer(doc, name: str, hex_color: str, line_weight_mm: float, aci: int = 251):
    """
    Register a layer with both a true-color (rgb) and a fixed ACI fallback.

    ACI 7 (the ezdxf/ezdxf.layers default) is the special "swap black/white
    with background" index — fine for a lone outline, but it collides with
    anything else also left at its default, invisibly merging layers in
    viewers that don't resolve true-color. ACI 251 is a real dark gray that
    reads the same regardless of background, so layers stay visually
    distinct even without true-color support.
    """
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    layer = doc.layers.new(name=name)
    layer.rgb = (r, g, b)
    layer.color = aci
    # `layer.lineweight = ...` is NOT a real DXF-backed property on ezdxf's
    # Layer -- it silently creates a throwaway Python attribute instead of
    # setting the DXF lineweight. The actual attribute lives under `.dxf`.
    layer.dxf.lineweight = _nearest_dxf_lineweight(line_weight_mm)


def _draw_multipolygon_outlines(msp, mpoly, poly_rings_fn, layer_name: str):
    if mpoly.is_empty:
        return
    polys = list(mpoly.geoms) if hasattr(mpoly, "geoms") else [mpoly]
    for poly in polys:
        exterior, holes = poly_rings_fn(poly)
        msp.add_lwpolyline(exterior, close=True, dxfattribs={"layer": layer_name})
        # Draw each hole (e.g. a building clipped out of this land-type
        # region) as its own closed loop so it reads visually as excluded.
        for hole in holes:
            msp.add_lwpolyline(hole, close=True, dxfattribs={"layer": layer_name})


def _scale_pattern_definition(definition: list, u: float) -> list:
    """Scale a hand-tuned AutoCAD pattern definition (authored in meters) into
    the output drawing unit: base point, offset vector and dash lengths all
    multiply by `u`; angles are unit-free. Without this, a unit-converting
    importer keeps the numeric spacing (0.3 m -> 0.3 mm) and the pattern
    collapses into invisible dust — the whole point of unit-native export."""
    if u == 1.0:
        return definition
    return [
        [angle, (base[0] * u, base[1] * u), (off[0] * u, off[1] * u),
         [d * u for d in dashes]]
        for angle, base, off, dashes in definition
    ]


def _draw_multipolygon_hatches(msp, mpoly, poly_rings_fn, layer_name: str, lt_style: dict,
                               u: float = 1.0):
    if mpoly.is_empty:
        return
    polys = list(mpoly.geoms) if hasattr(mpoly, "geoms") else [mpoly]

    hatch_type = lt_style.get("hatch_type", "lines")
    angle = lt_style.get("hatch_angle_deg", 45.0)
    spacing = lt_style.get("hatch_spacing_mm", 3.0)
    # ezdxf's hatch `scale` is a multiplier on the pattern's built-in
    # (inch-based) spacing, applied directly in modelspace units. Our
    # modelspace is real-world meters, not inches or mm, so a naive
    # spacing_mm/25.4 conversion produces a scale far too small (e.g.
    # 3mm/25.4 = 0.12) -- at real-world scale that packs the pattern so
    # densely it reads as a solid fill rather than visible hatch lines.
    # Deriving a correct scale needs the eventual print/plot scale, which
    # isn't tracked yet, so an explicit `hatch_scale` override bypasses the
    # broken auto-conversion until that's wired up.
    scale = lt_style.get("hatch_scale", spacing / 25.4) * u

    for poly in polys:
        exterior, holes = poly_rings_fn(poly)
        # Hatch is a light, thin texture that must recede behind the roof
        # outlines: force the thinnest lineweight and an explicit light color
        # (rather than BYLAYER, which some viewers render at a heavy default).
        # Color is per-type (paved is near-white so its dense crosshatch stays
        # the quietest); falls back to the shared light gray.
        hatch = msp.add_hatch(dxfattribs={
            "layer": layer_name,
            "lineweight": HATCH_LINEWEIGHT,
            # Below the drawing plane so linework always wins the depth test
            # (see LAND_HATCH_Z) — XY stays exactly on the region boundary.
            "elevation": (0, 0, LAND_HATCH_Z * u),
        })
        hatch.rgb = _hex_to_rgb(lt_style.get("hatch_color")) if lt_style.get("hatch_color") else HATCH_RGB
        hatch.paths.add_polyline_path(exterior, is_closed=True)
        # Each hole (e.g. a building clipped out) is its own boundary path;
        # hatch_style=NESTED makes the odd-parity island rule exclude it.
        for hole in holes:
            hatch.paths.add_polyline_path(hole, is_closed=True)
        hatch.dxf.hatch_style = 0  # NESTED / odd-even island detection

        if hatch_type == "solid":
            hatch.set_pattern_fill("SOLID")
        elif hatch_type == "acad":
            # Reproduce a hand-tuned AutoCAD pattern exactly: the definition
            # lines (from landtypes.ACAD_PATTERNS) are baked in drawing units
            # with rotation/spacing pre-applied, so scale/angle must stay at
            # identity — any other value would double-transform the pattern.
            from app.pipeline.landtypes import ACAD_PATTERNS
            name = lt_style["hatch_pattern"]
            hatch.set_pattern_fill(
                name, scale=1.0, angle=0.0,
                definition=_scale_pattern_definition(ACAD_PATTERNS[name], u),
            )
        elif hatch_type == "lines":
            hatch.set_pattern_fill("LINE", scale=scale, angle=angle)
        elif hatch_type == "crosshatch":
            hatch.set_pattern_fill("NET", scale=scale, angle=angle)
        elif hatch_type == "dots":
            # NB: use the same `scale` (hatch_scale override) as the other
            # patterns -- the old spacing/25.4 here ignored hatch_scale and
            # packed dots so densely they read as a solid fill.
            hatch.set_pattern_fill("DOTS", scale=scale)
        else:
            pass  # outline only — no hatch entity


def _nice_round(x: float) -> float:
    """Snap x to the nearest 'nice' drawing number (1, 2 or 5 x 10^k)."""
    from math import floor, log10
    if x <= 0:
        return 1.0
    exp = floor(log10(x))
    candidates = [n * 10 ** exp for n in (1, 2, 5, 10)]
    return min(candidates, key=lambda c: abs(c - x))


def _draw_scale_bar(msp, doc, extents, u: float = 1.0):
    """
    Architect-style alternating scale bar at the bottom-right of the drawing.

    Sized to a nice round length in METERS (~SCALE_BAR_FRACTION of the site
    width) so it stays proportionate at any site size, then drawn in the
    output unit (`u` = meters -> drawing-unit factor). Labels always read in
    meters regardless of the drawing unit — the bar states real-world lengths.
    Graphic follows the reference style: one square-wave outline, first half
    subdivided into five units with alternating raised segments, second half a
    single plain run.
    """
    width_m = (extents.extmax.x - extents.extmin.x) / u
    total_m = _nice_round(width_m * SCALE_BAR_FRACTION)
    unit_m = total_m / 10.0                  # five labelled units in first half
    total = total_m * u                      # drawn size, in output units
    unit = unit_m * u
    box_h = 0.04 * total
    text_h = max(1.5 * u, 0.05 * total)

    x1 = extents.extmax.x                    # right-aligned with the drawing
    x0 = x1 - total
    y = extents.extmin.y - 8.0 * u           # just below the site, clear of it

    _add_layer(doc, "SCALEBAR", "#000000", 0.18)
    attribs = {"layer": "SCALEBAR"}

    # One continuous square-wave outline (per the user's reference): the line
    # alternates between the raised top (over units 1, 3 and 5) and the
    # baseline — no baseline runs under a raised segment, no closed boxes.
    up, dn = y + box_h, y
    pts = [(x0, dn), (x0, up)]               # start tick
    for i in (0, 2, 4):
        bx0, bx1 = x0 + i * unit, x0 + (i + 1) * unit
        pts += [(bx0, up), (bx1, up), (bx1, dn)]          # across the top, down
        nxt = x0 + (i + 2) * unit if i < 4 else x1
        pts += [(nxt, dn)]                                # along the baseline
    pts += [(x1, up)]                        # end tick
    msp.add_lwpolyline(pts, dxfattribs=attribs)

    for i in range(6):                       # 0..5 unit labels (in meters)
        label = f"{i * unit_m:g}"
        t = msp.add_text(label, height=text_h, dxfattribs=attribs)
        t.set_placement((x0 + i * unit, y - text_h * 0.5),
                        align=TextEntityAlignment.TOP_CENTER)
    t = msp.add_text(f"{total_m:g} m", height=text_h, dxfattribs=attribs)
    t.set_placement((x1, y - text_h * 0.5), align=TextEntityAlignment.TOP_CENTER)
