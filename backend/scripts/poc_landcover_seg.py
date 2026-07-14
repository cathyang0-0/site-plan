"""
THROWAWAY POC — supervised land-cover segmentation vs. the current k-means.

NOT wired into the pipeline. See docs/land-cover-model-scoping.md for context.

What it does
------------
1. Downloads a community OpenEarthMap SegFormer checkpoint from Hugging Face
   (self-declared MIT; trained on OpenEarthMap, whose labels are partly
   CC BY-NC-SA 4.0 — evaluation use only; see the scoping doc's license section).
2. Runs TILED inference on test_data/img/cayuga_myers_point.png on MPS.
   We deliberately do NOT downscale to 1400px like landtypes.py — preserving
   resolution is the whole point (thin pavement survives).
3. Saves:
     - <out>_overlay.png   : OEM class colors blended over the aerial
     - <out>_paved.png     : just the Pavement+Road classes (the k-means blind spot)
     - <out>_classes.png   : the raw argmax class map (OEM palette)
4. Optionally (--compare-kmeans) runs the current detect_land_types and rasterizes
   its paved polygons next to the model's paved mask, for an eyeball comparison.

Run
---
    cd backend
    python scripts/poc_landcover_seg.py \
        --image ../test_data/img/cayuga_myers_point.png \
        --out /tmp/oem_seg --compare-kmeans

Checkpoint format is not guaranteed stable across community uploads; this script
introspects the state_dict and tries to build a matching SegformerForSemanticSegmentation.
If it can't infer the encoder variant, it prints the keys and exits so you can pin it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# OpenEarthMap palette + names — authoritative, from open_earth_map/utils.py
# (class_grey_oem / class_rgb_oem). Index 0 = background/unknown.
OEM_NAMES = ["unknown", "Bareland", "Grass", "Pavement", "Road",
             "Tree", "Water", "Cropland", "buildings"]
OEM_RGB = np.array([
    [0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255],
    [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7],
], dtype=np.uint8)

# OEM class idx -> our four site-plan labels (see scoping doc §3.1).
OEM_TO_OURS = {
    1: "bare earth / farmland",  # Bareland
    2: "vegetation",             # Grass / Rangeland
    3: "paved / hardscape",      # Pavement / Developed
    4: "paved / hardscape",      # Road
    5: "vegetation",             # Tree
    6: "water",                  # Water
    7: "bare earth / farmland",  # Cropland / Agriculture
    8: "paved / hardscape",      # buildings (we get these from Overture; here for completeness)
}
PAVED_IDS = (3, 4)  # the thin-feature classes the k-means blurs

HF_REPO = "odil111/segformer-fine-tuned-on-openearthmap"
HF_CKPT = "segformer_sem_seg_2024-05-16--14-40-45/segformer_sem_seg_checkpoint_epoch35.pt"
TILE = 1000        # this checkpoint documents 1000x1000 inputs
OVERLAP = 128
N_CLASSES = 9


def _device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _strip_prefix(sd):
    """Community checkpoints may nest under 'state_dict'/'model' and prefix
    keys with 'model.' or 'module.'. Normalize to bare SegFormer keys."""
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    out = {}
    for key, v in sd.items():
        nk = key
        for p in ("module.", "model."):
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _infer_mit_variant(sd):
    """Guess mit-b0..b5 from the first block's hidden size (patch_embeddings.0)."""
    # SegformerForSemanticSegmentation keys look like:
    # segformer.encoder.patch_embeddings.0.proj.weight  -> [C0, 3, 7, 7]
    for k, v in sd.items():
        if k.endswith("patch_embeddings.0.proj.weight"):
            c0 = v.shape[0]
            return {32: "nvidia/mit-b0", 64: "nvidia/mit-b1"}.get(c0), c0
    return None, None


def load_model():
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    print(f"Downloading checkpoint {HF_REPO}/{HF_CKPT} ...")
    path = hf_hub_download(repo_id=HF_REPO, filename=HF_CKPT)
    print(f"  -> {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = _strip_prefix(raw)

    variant, c0 = _infer_mit_variant(sd)
    if variant is None:
        print("Could not infer SegFormer variant from checkpoint. First 30 keys:")
        for k in list(sd)[:30]:
            print("   ", k, tuple(getattr(sd[k], "shape", ())))
        raise SystemExit("Pin the architecture manually, then re-run.")
    print(f"Inferred encoder ~ {variant} (patch_embeddings.0 width={c0})")

    id2label = {i: OEM_NAMES[i] for i in range(N_CLASSES)}
    cfg = SegformerConfig.from_pretrained(
        variant, num_labels=N_CLASSES, id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    model = SegformerForSemanticSegmentation(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if len(missing) > 20:
        print("   (many missing keys — architecture likely mismatched; sample:)",
              missing[:8])
    model.eval()
    return model


def _tiles(H, W, tile, overlap):
    step = tile - overlap
    ys = list(range(0, max(1, H - overlap), step))
    xs = list(range(0, max(1, W - overlap), step))
    for y in ys:
        for x in xs:
            yield min(y, max(0, H - tile)), min(x, max(0, W - tile))


def segment(model, img_rgb):
    """Tiled inference; accumulate softmax logits, argmax at the end."""
    import torch
    import torch.nn.functional as F

    dev = _device()
    model.to(dev)
    H, W = img_rgb.shape[:2]
    acc = np.zeros((N_CLASSES, H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    with torch.no_grad():
        for (y, x) in _tiles(H, W, TILE, OVERLAP):
            th, tw = min(TILE, H - y), min(TILE, W - x)
            crop = img_rgb[y:y + th, x:x + tw].astype(np.float32) / 255.0
            crop = (crop - mean) / std
            t = torch.from_numpy(crop.transpose(2, 0, 1))[None].to(dev)
            logits = model(pixel_values=t).logits          # [1, C, h/4, w/4]
            logits = F.interpolate(logits, size=(th, tw),
                                   mode="bilinear", align_corners=False)
            prob = F.softmax(logits, dim=1)[0].cpu().numpy()
            acc[:, y:y + th, x:x + tw] += prob
            cnt[y:y + th, x:x + tw] += 1.0
            print(f"  tile ({y},{x}) {th}x{tw} done")

    cnt[cnt == 0] = 1.0
    return (acc / cnt).argmax(axis=0).astype(np.uint8)


def save_outputs(img_rgb, cls, out_prefix):
    cls_rgb = OEM_RGB[cls]
    Image.fromarray(cls_rgb).save(f"{out_prefix}_classes.png")

    overlay = (0.55 * img_rgb + 0.45 * cls_rgb).astype(np.uint8)
    Image.fromarray(overlay).save(f"{out_prefix}_overlay.png")

    paved = np.isin(cls, PAVED_IDS)
    pv = img_rgb.copy()
    pv[paved] = [255, 0, 255]
    Image.fromarray(pv).save(f"{out_prefix}_paved.png")

    frac = {OEM_NAMES[i]: float((cls == i).mean()) for i in range(N_CLASSES)}
    print("Class pixel fractions:",
          {k: round(v, 3) for k, v in frac.items() if v > 0.001})
    print(f"Saved: {out_prefix}_classes.png, _overlay.png, _paved.png")


def compare_kmeans(image, out_prefix):
    """Rasterize the current detect_land_types paved polygons for comparison."""
    import cv2
    from app.pipeline.landtypes import detect_land_types
    W, H = image.size
    zero = np.zeros((H, W), np.uint8)
    print("Running current k-means detect_land_types for comparison ...")
    regions = detect_land_types(image, zero, zero)
    mask = np.zeros((H, W), np.uint8)
    for r in regions:
        if r["label"] != "paved / hardscape":
            continue
        for poly in r["polygons"].geoms:
            pts = np.array(poly.exterior.coords, np.int32)
            cv2.fillPoly(mask, [pts], 1)
    km = np.array(image.convert("RGB"))
    km[mask.astype(bool)] = [255, 0, 255]
    Image.fromarray(km).save(f"{out_prefix}_kmeans_paved.png")
    print(f"Saved: {out_prefix}_kmeans_paved.png  "
          f"(paved regions found: {sum(r['label']=='paved / hardscape' for r in regions)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="../test_data/img/cayuga_myers_point.png")
    ap.add_argument("--out", default="/tmp/oem_seg")
    ap.add_argument("--compare-kmeans", action="store_true")
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    img_rgb = np.asarray(image)
    print(f"Image {args.image}: {img_rgb.shape[1]}x{img_rgb.shape[0]}")

    model = load_model()
    cls = segment(model, img_rgb)
    save_outputs(img_rgb, cls, args.out)

    if args.compare_kmeans:
        compare_kmeans(image, args.out)


if __name__ == "__main__":
    main()
