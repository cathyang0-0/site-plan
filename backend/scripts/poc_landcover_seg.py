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
    """Guess the base variant from the stem width (stage-0 patch embedding).
    Runs AFTER _remap_legacy_segformer, so keys are in the modern
    `segformer.stages.0.patch_embeddings.proj.weight` form. Width only picks the
    config base: b0 is 32, b1..b5 all share 64 and are disambiguated later by
    depths (see _infer_depths), so 64 -> b1 as a base is fine."""
    for k, v in sd.items():
        if k.endswith("stages.0.patch_embeddings.proj.weight"):
            c0 = v.shape[0]
            return {32: "nvidia/mit-b0", 64: "nvidia/mit-b1"}.get(c0), c0
    return None, None


def _infer_decoder_hidden_size(sd):
    """The community checkpoint may use a non-default decoder width. Read it
    straight from the decode head instead of trusting the encoder-variant
    default (b1 defaults to 256; this checkpoint uses 768)."""
    for key in ("decode_head.batch_norm.weight", "decode_head.classifier.weight"):
        if key in sd:
            return int(sd[key].shape[0] if key.endswith("batch_norm.weight")
                       else sd[key].shape[1])
    return None


def _remap_legacy_segformer(sd):
    """This checkpoint was saved with transformers 4.x, which used different
    SegFormer state-dict key names than transformers >=5 (encoder->stages,
    block->blocks, attention.self.{query,key,value}->attention.{q,k,v}_proj,
    etc.). Rename in place so the weights actually load; otherwise
    load_state_dict(strict=False) silently drops everything and the model runs
    on random init (symptom: whole image classified as one class)."""
    import re
    out = {}
    for k, v in sd.items():
        nk = k
        nk = re.sub(r"segformer\.encoder\.patch_embeddings\.(\d+)\.",
                    r"segformer.stages.\1.patch_embeddings.", nk)
        nk = re.sub(r"segformer\.encoder\.block\.(\d+)\.(\d+)\.",
                    r"segformer.stages.\1.blocks.\2.", nk)
        nk = re.sub(r"segformer\.encoder\.layer_norm\.(\d+)\.",
                    r"segformer.stages.\1.layer_norm.", nk)
        nk = nk.replace(".layer_norm_1.", ".layernorm_before.")
        nk = nk.replace(".layer_norm_2.", ".layernorm_after.")
        nk = nk.replace(".attention.self.query.", ".attention.q_proj.")
        nk = nk.replace(".attention.self.key.", ".attention.k_proj.")
        nk = nk.replace(".attention.self.value.", ".attention.v_proj.")
        nk = nk.replace(".attention.output.dense.", ".attention.o_proj.")
        nk = nk.replace(".attention.self.sr.",
                        ".attention.sequence_reduction.sequence_reduction.")
        nk = nk.replace(".attention.self.layer_norm.",
                        ".attention.sequence_reduction.layer_norm.")
        nk = nk.replace(".mlp.dense1.", ".mlp.fc1.")
        nk = nk.replace(".mlp.dense2.", ".mlp.fc2.")
        nk = re.sub(r"decode_head\.linear_c\.(\d+)\.",
                    r"decode_head.linear_projections.\1.", nk)
        out[nk] = v
    return out


def _infer_depths(sd):
    """Count transformer blocks per stage from the (remapped) checkpoint.
    mit-b1..b5 all share widths [64,128,320,512] and differ only in depth, so
    the encoder width alone can't pin the variant — read the depths instead."""
    import re
    depths = {}
    for k in sd:
        m = re.match(r"segformer\.stages\.(\d+)\.blocks\.(\d+)\.", k)
        if m:
            s, b = int(m.group(1)), int(m.group(2))
            depths[s] = max(depths.get(s, 0), b + 1)
    return [depths[i] for i in range(len(depths))] if depths else None


def load_model():
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    print(f"Downloading checkpoint {HF_REPO}/{HF_CKPT} ...")
    path = hf_hub_download(repo_id=HF_REPO, filename=HF_CKPT)
    print(f"  -> {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = _remap_legacy_segformer(_strip_prefix(raw))

    variant, c0 = _infer_mit_variant(sd)
    if variant is None:
        print("Could not infer SegFormer variant from checkpoint. First 30 keys:")
        for k in list(sd)[:30]:
            print("   ", k, tuple(getattr(sd[k], "shape", ())))
        raise SystemExit("Pin the architecture manually, then re-run.")

    dec_hidden = _infer_decoder_hidden_size(sd)
    depths = _infer_depths(sd)
    print(f"Inferred encoder width={c0} (~{variant}), "
          f"depths={depths}, decoder_hidden_size={dec_hidden}")

    id2label = {i: OEM_NAMES[i] for i in range(N_CLASSES)}
    cfg_kwargs = dict(
        num_labels=N_CLASSES, id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    if dec_hidden is not None:
        cfg_kwargs["decoder_hidden_size"] = dec_hidden
    if depths is not None:
        cfg_kwargs["depths"] = depths
    cfg = SegformerConfig.from_pretrained(variant, **cfg_kwargs)
    model = SegformerForSemanticSegmentation(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # A partial load is worse than a hard failure: it runs on random init and
    # produces confident garbage. Refuse to proceed unless the load is clean.
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if real_missing or unexpected:
        print(f"  load_state_dict: {len(real_missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        print("   missing sample:   ", real_missing[:8])
        print("   unexpected sample:", unexpected[:8])
        raise SystemExit("Checkpoint did not map cleanly onto the model — "
                         "aborting rather than run on partially-loaded weights.")
    print("  load_state_dict: clean (0 missing, 0 unexpected)")
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
