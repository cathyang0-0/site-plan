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
    --image     Path to an aerial image (PNG or JPG)
    --scale     Real-world meters per pixel (default: 0.15 for Mapbox zoom 19)
    --output    Output DXF path (default: output.dxf)

Note: CV models are not yet trained. This script uses placeholder
      detections so you can validate the DXF output structure first.
      Swap in real detections as each pipeline module is implemented.
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
from app.pipeline.trees import filter_placements, MIN_CANOPY_RADIUS_M

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


def main():
    parser = argparse.ArgumentParser(description="Site Plan Drafter — POC")
    parser.add_argument("--image", required=True, help="Path to aerial image (PNG/JPG)")
    parser.add_argument("--scale", type=float, default=0.15, help="Meters per pixel")
    parser.add_argument("--output", default="output.dxf", help="Output DXF path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    print(f"  Size: {img_w} x {img_h} px  |  ~{img_w * args.scale:.0f} x {img_h * args.scale:.0f} m")

    origin_px = (img_w // 2, img_h // 2)

    print("Generating placeholder geometry (real CV models not yet wired up)...")
    buildings = placeholder_buildings(img_w, img_h)
    roads = placeholder_roads(img_w, img_h)
    tree_placements = placeholder_trees(img_w, img_h, buildings, args.scale)
    tree_blocks = [default_tree_block()]
    land_types = placeholder_land_types(img_w, img_h)
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
    )

    print(f"Done. Open {output_path} in Rhino, AutoCAD, or Vectorworks to check the output.")
    print()
    print("Next steps:")
    print("  1. Check layer structure and geometry in your CAD app")
    print("  2. Implement _run_sam2_fallback() in pipeline/buildings.py")
    print("  3. Replace placeholder_buildings() with real detections")


if __name__ == "__main__":
    main()
