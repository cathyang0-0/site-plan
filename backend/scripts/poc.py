"""
Phase 1 POC — Site Plan Drafter

Takes a local aerial image file and produces a DXF with:
  - Roof outlines (white fill + outline)
  - Tree symbols (simple circles, no imported blocks yet)
  - Road centerlines

Run:
    cd backend
    python scripts/poc.py --image path/to/aerial.png --scale 0.15 --output out.dxf

Arguments:
    --image       Path to an aerial image (PNG or JPG)
    --scale       Real-world meters per pixel (default: 0.15 for Mapbox zoom 19)
    --output      Output DXF path (default: output.dxf)
    --real-trees      Run real DeepForest tree detection (first run downloads
                      model weights; CPU inference takes a few minutes)
    --real-buildings  Run real SAM2 zero-shot building detection; when combined
                      with --real-trees, rooftop tree detections are suppressed
                      by the building-overlap rule

Default mode uses placeholder detections so the DXF output structure can be
validated without any models. In real mode, modules without a --real-* flag
stay empty (their placeholders sit at fixed pixel coords that mean nothing on
real imagery).
"""
import argparse
import random
from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon, LineString, MultiPolygon

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.export.dxf import export_dxf
from app.pipeline.trees import filter_placements, suppress_over_water, MIN_CANOPY_RADIUS_M

DEFAULT_BLOCK_RADIUS_PX = 20.0  # matches default_tree_block()'s default radius


def placeholder_buildings(img_w: int, img_h: int) -> list[Polygon]:
    """Return a few fake rectangular buildings for testing DXF output."""
    return [
        Polygon([(200, 200), (400, 200), (400, 350), (200, 350)]),
        Polygon([(500, 150), (700, 150), (700, 280), (500, 280)]),
        Polygon([(300, 500), (550, 500), (550, 680), (300, 680)]),
    ]


def placeholder_roads(img_w: int, img_h: int) -> list[dict]:
    """Return a couple of fake road centerlines."""
    return [
        {"line": LineString([(0, 400), (300, 420), (700, 380), (img_w, 360)]), "width_px": 20},
        {"line": LineString([(400, 0), (410, 300), (420, img_h)]), "width_px": 15},
    ]


def placeholder_trees(
    img_w: int,
    img_h: int,
    buildings: list[Polygon],
    scale_m_per_px: float,
    n_blocks: int = 1,
    candidate_count: int = 500,
) -> list[dict]:
    """
    Scatter dense random tree candidates, then run them through the real
    filter_placements() (building overlap + tree-tree overlap suppression)
    so the surviving set reads as plausibly planted rather than a uniform
    random scatter. Canopy size varies per tree, and scale is derived from
    that size relative to the tree block's reference radius.

    Mirrors detect_trees()'s minimum-canopy rule: candidates smaller than
    MIN_CANOPY_RADIUS_M are enlarged rather than dropped.
    """
    rng = random.Random(0)
    min_radius_px = MIN_CANOPY_RADIUS_M / scale_m_per_px

    candidates = []
    for _ in range(candidate_count):
        x = rng.uniform(0, img_w)
        y = rng.uniform(0, img_h)
        radius_px = rng.uniform(8, 45)  # varied canopy sizes, small to large
        radius_px = max(radius_px, min_radius_px)
        candidates.append({
            "x_px": x,
            "y_px": y,
            "radius_px": radius_px,
            "radius_m": radius_px * scale_m_per_px,
        })

    accepted = filter_placements(candidates, buildings, overlap_threshold=0.30)

    trees = []
    for det in accepted:
        trees.append({
            "block_idx": rng.randrange(n_blocks),
            "position": (det["x_px"], det["y_px"]),
            "scale": det["radius_px"] / DEFAULT_BLOCK_RADIUS_PX,
            "rotation": rng.uniform(0, 360),
        })
    return trees


def placeholder_land_types(img_w: int, img_h: int) -> list[dict]:
    """
    Return a few fake land-type regions to exercise hatch export.

    All three currently share one style -- parallel lines, 45°, scale 1 --
    as a uniform placeholder while the mm-to-hatch-scale conversion (which
    needs the eventual print/plot scale to be correct) isn't wired up yet.
    """
    placeholder_style = {
        "hatch_type": "lines", "hatch_color": "#888888",
        "hatch_angle_deg": 45.0, "hatch_scale": 1,
    }
    return [
        {
            "label": "Vegetation",
            "polygons": MultiPolygon([Polygon([
                (0, 0), (img_w * 0.35, 0), (img_w * 0.35, img_h * 0.25), (0, img_h * 0.25),
            ])]),
            "style": dict(placeholder_style),
        },
        {
            "label": "Paved / hardscape",
            "polygons": MultiPolygon([Polygon([
                (img_w * 0.65, img_h * 0.7), (img_w, img_h * 0.7), (img_w, img_h), (img_w * 0.65, img_h),
            ])]),
            "style": dict(placeholder_style),
        },
        {
            "label": "Bare earth / farmland",
            "polygons": MultiPolygon([Polygon([
                (0, img_h * 0.75), (img_w * 0.3, img_h * 0.75), (img_w * 0.3, img_h), (0, img_h),
            ])]),
            "style": dict(placeholder_style),
        },
    ]


def default_tree_block(radius_px: float = 20.0, n_pts: int = 32) -> list[list]:
    """
    Default tree block: circle with a "+" in the center.
    Returns a list of curve point-lists: [circle_pts, horizontal_line, vertical_line].
    """
    import math
    circle = [
        (radius_px * math.cos(2 * math.pi * i / n_pts),
         radius_px * math.sin(2 * math.pi * i / n_pts))
        for i in range(n_pts + 1)
    ]
    arm = radius_px * 0.3  # "+" arms are 30% of radius
    horizontal = [(-arm, 0), (arm, 0)]
    vertical   = [(0, -arm), (0, arm)]
    return [circle, horizontal, vertical]


def real_tree_detections(
    image: Image.Image,
    scale_m_per_px: float,
    stand_fill: bool = True,
    crown_size_scale: float = 1.0,
    size_variance: float = 1.0,
) -> list[dict]:
    """Run real DeepForest detection + dense-stand fill. Returns raw
    detections (pre overlap-filtering) so callers can cross-check other
    modules against them before suppression runs."""
    from app.pipeline.trees import (
        detect_trees, apply_size_transform, fill_dense_stands, MAX_CANOPY_RADIUS_M,
    )

    detections = detect_trees(image, scale_m_per_px)
    n_stands = sum(1 for d in detections if d.get("stand"))
    print(f"  Raw detections: {len(detections)} ({n_stands} dense-stand boxes)")
    # Reshape rendered crown sizes (variance then average) before stand fill,
    # so the synthetic fill picks up the transformed size distribution.
    apply_size_transform(
        detections,
        crown_size_scale=crown_size_scale,
        size_variance=size_variance,
    )
    if stand_fill:
        # fill_dense_stands puts synthetic trees AFTER real detections so
        # filter_placements gives the real ones priority where they overlap.
        detections = fill_dense_stands(detections, scale_m_per_px)
        print(f"  After stand fill: {len(detections)}")
    else:
        # No fill: draw each stand as a single max-size tree so the dense
        # canopy at least isn't blank.
        for d in detections:
            if d.get("stand"):
                d["radius_m"] = MAX_CANOPY_RADIUS_M
                d["radius_px"] = MAX_CANOPY_RADIUS_M / scale_m_per_px
    return detections


def _rasterize_polygons(polygons, img_w: int, img_h: int):
    """Fill shapely Polygons into a uint8 mask (buildings -> exclusion mask
    for land-type detection)."""
    import cv2
    import numpy as np
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _rasterize_roads(roads, img_w: int, img_h: int):
    """Draw road centerlines at their width into a uint8 mask (road exclusion
    mask for land-type detection)."""
    import cv2
    import numpy as np
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for road in roads:
        pts = np.array(road["line"].coords, dtype=np.int32)
        thickness = max(1, int(round(road.get("width_px", 4))))
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
    return mask


def detections_to_placements(detections: list[dict], n_blocks: int = 1) -> list[dict]:
    """Convert accepted detections into tree block placements."""
    rng = random.Random(0)
    return [
        {
            "block_idx": rng.randrange(n_blocks),
            "position": (det["x_px"], det["y_px"]),
            "scale": det["radius_px"] / DEFAULT_BLOCK_RADIUS_PX,
            "rotation": rng.uniform(0, 360),
        }
        for det in detections
    ]


def main():
    parser = argparse.ArgumentParser(description="Site Plan Drafter — POC")
    parser.add_argument("--image", required=True, help="Path to aerial image (PNG/JPG)")
    parser.add_argument("--scale", type=float, default=0.15, help="Meters per pixel")
    parser.add_argument("--output", default="output.dxf", help="Output DXF path")
    parser.add_argument("--real-trees", action="store_true",
                        help="Run real DeepForest tree detection instead of placeholders")
    parser.add_argument("--no-stand-fill", action="store_true",
                        help="Draw oversized dense-canopy detections as single max-size "
                             "trees instead of filling them with synthetic stands")
    parser.add_argument("--crown-scale", type=float, default=1.0,
                        help="'Average size' multiplier on detected crown size (default "
                             "1.0). Detected size skews small for a context plan; try "
                             "~1.4 for larger, more prominent canopy. Positions are "
                             "unaffected.")
    parser.add_argument("--size-variance", type=float, default=1.0,
                        help="Crown size-variance factor (default 1.0). Scales each "
                             "tree's deviation from the mean crown size: 0 makes every "
                             "crown uniform, 1 keeps the detected spread, >1 exaggerates "
                             "it (big trees bigger, small trees smaller). Positions are "
                             "unaffected.")
    parser.add_argument("--land-types", action="store_true",
                        help="Detect land-cover regions (water/vegetation/bare/paved) and "
                             "hatch them; buildings and roads found in this run are masked "
                             "out. Engine chosen by --land-types-engine")
    parser.add_argument("--land-types-engine", choices=["kmeans", "segmodel"],
                        default="kmeans",
                        help="Land-type engine (default kmeans). 'segmodel' runs a "
                             "pretrained OpenEarthMap SegFormer that catches thin pavement "
                             "k-means misses, but downloads weights on first use and is "
                             "EVALUATION-ONLY (CC BY-NC-SA training data; see "
                             "docs/land-cover-model-scoping.md)")
    parser.add_argument("--real-buildings", action="store_true",
                        help="Run real SAM2 zero-shot building detection (downloads the "
                             "checkpoint on first run; several minutes of inference)")
    parser.add_argument("--footprint-buildings", action="store_true",
                        help="Fetch building footprints from Overture Maps instead of CV "
                             "detection (preferred when the image is georeferenced; "
                             "requires --bbox)")
    parser.add_argument("--footprint-roads", action="store_true",
                        help="Fetch road centerlines from Overture Maps (class-based "
                             "width, nudged by a light CV pavement measurement); "
                             "requires --bbox")
    parser.add_argument("--bbox", type=str, default=None,
                        help="Geographic extent of the image as WEST,SOUTH,EAST,NORTH "
                             "(lon/lat); the image must span exactly this bbox")
    args = parser.parse_args()

    if (args.footprint_buildings or args.footprint_roads) and not args.bbox:
        parser.error("--footprint-buildings/--footprint-roads require --bbox")

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    print(f"  Size: {img_w} x {img_h} px  |  ~{img_w * args.scale:.0f} x {img_h * args.scale:.0f} m")

    origin_px = (img_w // 2, img_h // 2)

    attributions = []
    if (args.real_trees or args.real_buildings or args.footprint_buildings
            or args.footprint_roads or args.land_types):
        buildings = []
        roads = []
        land_types = []
        tree_placements = []
        detections = []
        if args.footprint_roads:
            print("Fetching road centerlines (Overture Maps)...")
            from app.pipeline.roads import build_roads, ATTRIBUTION as ROAD_ATTR
            west, south, east, north = (float(v) for v in args.bbox.split(","))
            roads = build_roads(image, west, south, east, north, args.scale)
            attributions.append(ROAD_ATTR)
            print(f"  Roads fetched: {len(roads)}")
        if args.real_trees:
            print("Running real tree detection (DeepForest)...")
            detections = real_tree_detections(
                image, args.scale,
                stand_fill=not args.no_stand_fill,
                crown_size_scale=args.crown_scale,
                size_variance=args.size_variance,
            )
        if args.footprint_buildings:
            print("Fetching building footprints (Overture Maps)...")
            from app.pipeline.footprints import (
                fetch_building_footprints, footprints_to_pixels, ATTRIBUTION,
            )
            west, south, east, north = (float(v) for v in args.bbox.split(","))
            geo_polys = fetch_building_footprints(west, south, east, north)
            buildings = footprints_to_pixels(
                geo_polys, west, south, east, north, img_w, img_h
            )
            attributions.append(ATTRIBUTION)
            print(f"  Footprints fetched: {len(buildings)}")
        elif args.real_buildings:
            print("Running real building detection (SAM2 zero-shot)...")
            from app.pipeline.buildings import detect_buildings, suppress_canopy_false_positives
            buildings = detect_buildings(image, scale_m_per_px=args.scale)
            print(f"  Buildings detected: {len(buildings)}")
            if detections:
                # Raw tree detections expose SAM2's crown false positives;
                # must happen BEFORE filter_placements suppresses trees
                # against buildings (else false buildings hide their trees).
                # (Footprint buildings are authoritative -- no cross-filter.)
                buildings = suppress_canopy_false_positives(buildings, detections)
                print(f"  Buildings after canopy cross-filter: {len(buildings)}")
        # Land types run BEFORE the tree filter so its water regions can
        # suppress trees the detector hallucinated on the lake surface.
        if args.land_types:
            from app.pipeline.landtypes import default_hatch_style
            b_mask = _rasterize_polygons(buildings, img_w, img_h)
            r_mask = _rasterize_roads(roads, img_w, img_h)
            if args.land_types_engine == "segmodel":
                print("Detecting land-cover types (OpenEarthMap SegFormer; "
                      "downloads weights on first use, eval-only)...")
                from app.pipeline.landtypes_seg import detect_land_types_seg
                detected = detect_land_types_seg(image, b_mask, r_mask)
            else:
                print("Detecting land-cover types (unsupervised clustering)...")
                from app.pipeline.landtypes import detect_land_types
                detected = detect_land_types(image, b_mask, r_mask)
            land_types = [
                {"label": d["label"], "polygons": d["polygons"],
                 "style": default_hatch_style(d["label"])}
                for d in detected
            ]
            print(f"  Land types: {', '.join(d['label'] for d in land_types)}")
        if args.real_trees:
            # >30% building-overlap rule suppresses rooftop tree detections.
            accepted = filter_placements(detections, building_polygons=buildings)
            # Drop trees the detector placed on open water (wave/reflection
            # false positives), using the detected water regions.
            water_polys = [
                g for lt in land_types if lt["label"] == "water"
                for g in (lt["polygons"].geoms if hasattr(lt["polygons"], "geoms")
                          else [lt["polygons"]])
            ]
            if water_polys:
                before = len(accepted)
                accepted = suppress_over_water(accepted, water_polys)
                print(f"  Trees suppressed over water: {before - len(accepted)}")
            tree_placements = detections_to_placements(accepted)
            print(f"  Trees after filtering: {len(tree_placements)}")
    else:
        print("Generating placeholder geometry (real CV models not yet wired up)...")
        buildings = placeholder_buildings(img_w, img_h)
        roads = placeholder_roads(img_w, img_h)
        tree_placements = placeholder_trees(img_w, img_h, buildings, args.scale)
        land_types = placeholder_land_types(img_w, img_h)
    tree_blocks = [default_tree_block()]
    print(f"  Buildings: {len(buildings)}  Roads: {len(roads)}  "
          f"Trees: {len(tree_placements)}  Land types: {len(land_types)}")

    # Line weights match spec.md §5's architectural hierarchy: roof outlines
    # read heaviest, roads readable but secondary, trees/land types are texture.
    style = {
        "roofs": {"color": "#000000", "line_weight_mm": 0.40},
        "roads": {"color": "#333333", "line_weight_mm": 0.18},
        "trees": {"color": "#555555", "line_weight_mm": 0.10},
        "land_types": [{"color": "#aaaaaa", "line_weight_mm": 0.05}],
    }

    output_path = Path(args.output)
    print(f"Exporting DXF → {output_path}")
    export_dxf(
        output_path=output_path,
        buildings=buildings,
        roads=roads,
        tree_placements=tree_placements,
        tree_block_curves=tree_blocks,
        land_types=land_types,
        style=style,
        scale_m_per_px=args.scale,
        origin_px=origin_px,
        attribution="  •  ".join(attributions) if attributions else None,
    )

    print(f"Done. Open {output_path} in Rhino, AutoCAD, or Vectorworks to check the output.")
    print()
    print("Next steps:")
    print("  1. Check layer structure and geometry in your CAD app")
    print("  2. Implement road detection (pipeline/roads.py)")
    print("  3. Wire land-type clustering into real mode (pipeline/landtypes.py)")


if __name__ == "__main__":
    main()
