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
from shapely.geometry import Polygon, LineString

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.export.dxf import export_dxf


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


def placeholder_trees(img_w: int, img_h: int, count: int = 30) -> list[dict]:
    """Scatter random trees across the image (avoiding building areas)."""
    rng = random.Random(0)
    trees = []
    for _ in range(count):
        x = rng.uniform(50, img_w - 50)
        y = rng.uniform(50, img_h - 50)
        radius_px = rng.uniform(15, 40)
        trees.append({
            "block_idx": 0,
            "position": (x, y),
            "scale": 1.0,
            "rotation": rng.uniform(0, 360),
        })
    return trees


def simple_circle_block(radius_px: float = 20.0, n_pts: int = 32) -> list[list]:
    """
    A single tree block made of one circle (Style B from spec).
    Returns a list of curve point-lists (one curve = the circle).
    """
    import math
    pts = [
        (radius_px * math.cos(2 * math.pi * i / n_pts),
         radius_px * math.sin(2 * math.pi * i / n_pts))
        for i in range(n_pts + 1)
    ]
    return [pts]


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
    tree_placements = placeholder_trees(img_w, img_h)
    tree_blocks = [simple_circle_block()]

    style = {
        "roofs": {"color": "#000000"},
        "roads": {"color": "#333333"},
        "trees": {"color": "#555555"},
    }

    output_path = Path(args.output)
    print(f"Exporting DXF → {output_path}")
    export_dxf(
        output_path=output_path,
        buildings=buildings,
        roads=roads,
        tree_placements=tree_placements,
        tree_block_curves=tree_blocks,
        land_types=[],
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
