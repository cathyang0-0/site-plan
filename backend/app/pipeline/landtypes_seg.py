"""
Land texture detection — SUPERVISED semantic segmentation variant.

Drop-in alternative to `landtypes.detect_land_types` that swaps the unsupervised
k-means for a pretrained OpenEarthMap SegFormer. Same contract in and out; only
the per-pixel classification changes, everything downstream (polygon cleanup,
frame-flush, thumbnails, per-type DXF layers) is reused from `landtypes`.

Why this exists: k-means averages color over ~500 superpixels and downscales to
1400 px, so thin pavement (winding paths / a road through green) is a minority
inside a big vegetation cell and gets erased. A per-pixel model with a learned
`Pavement`/`Road` concept labels each pixel and — run at ~native resolution via
tiling — keeps thin hardscape. See docs/land-cover-model-scoping.md.

⚠️ LICENSE / SHIP GATE: the community checkpoint is trained on OpenEarthMap,
whose label data is partly CC BY-NC-SA 4.0 (NonCommercial). This engine is
EVALUATION-ONLY until the license is resolved (see the scoping doc §1, §8). It
is gated behind an explicit opt-in (engine="segmodel"); k-means stays the
default. Weights are NOT bundled — they download to the HF cache on first use.

Gotchas respected (see docs/HANDOFF.md): `import torch` stays function-local
(module-scope torch segfaults the test suite); all torch inference runs and
frees its tensors before any OpenCV polygonization; the k-means/SLIC OpenMP
offenders are gone from this path.
"""
import numpy as np
from PIL import Image
from shapely.affinity import translate as _affine_translate
from shapely.geometry import Polygon, MultiPolygon, box as _box
import cv2

# Reuse the whole polygon-cleanup + export-facing surface unchanged. Labels are
# identical strings, so hatch styles / thumbnails carry over verbatim.
from app.pipeline.landtypes import (
    _mask_to_multipolygon,
    _as_multipolygon,
    _make_thumbnail,
    default_hatch_style,  # re-exported for callers that style seg regions
    DEFAULT_HATCH_STYLES,  # noqa: F401  (re-export)
)

# OpenEarthMap 8-class palette + names — authoritative, from the repo's own
# open_earth_map/utils.py (class_grey_oem). Index 0 = background/unknown.
OEM_NAMES = ["unknown", "Bareland", "Grass", "Pavement", "Road",
             "Tree", "Water", "Cropland", "buildings"]
N_CLASSES = len(OEM_NAMES)

# OEM class idx -> our four site-plan labels (scoping doc §3.1). `unknown` (0)
# and `buildings` (8) are intentionally dropped: buildings come from Overture
# (emitting them here would double-count with the roof layer), and unknown is
# background. Everything else collapses onto the same four labels k-means uses.
OEM_TO_OURS = {
    1: "bare earth / farmland",  # Bareland
    2: "vegetation",             # Grass / Rangeland
    3: "paved / hardscape",      # Pavement / Developed space
    4: "paved / hardscape",      # Road
    5: "vegetation",             # Tree
    6: "water",                  # Water
    7: "bare earth / farmland",  # Cropland / Agriculture
}
# Emit in the same order as k-means CLUSTER_LABELS for stable, comparable output.
LABEL_ORDER = ["water", "vegetation", "bare earth / farmland", "paved / hardscape"]

# Per-label morphology overrides for _mask_to_multipolygon. The whole point of
# the model is precise thin hardscape, so DON'T undo it in cleanup: shrink the
# 11 px close (which would swallow a 2 px path) and skip the open for paved.
# Other labels keep the k-means defaults (open=3, close=11), which heal the
# fuzzier vegetation/water boundaries.
_MORPH_BY_LABEL = {
    "paved / hardscape": dict(morph_open_px=0, morph_close_px=3, min_area_px=400),
}

# OEM class indices that collapse to "paved / hardscape".
PAVED_IDS = (3, 4)  # Pavement, Road

# Confidence gate for paved pixels. The off-the-shelf model over-predicts paved
# on shoreline/beach and shadowed tree/roof, where Pavement/Road narrowly wins
# with low probability; genuine roads/paths score much higher. So a paved pixel
# is only kept if its softmax probability clears this bar — otherwise it is
# demoted to its best NON-paved class (water/tree/bare underneath), which curbs
# the false positives without touching the confident thin hardscape. Applied
# only to paved (the false-positive-prone class); a global gate would risk
# poking holes in legitimately less-peaked water/vegetation. Set 0.0 to disable.
# 0.6 chosen by eyeballing the Cayuga overlay: it dissolves the shoreline/shadow
# blobs while the real road/path network survives (paved fraction 0.078 -> 0.059).
PAVED_MIN_CONF = 0.6

# The model mislabels featureless deep/open water as land — out-of-distribution
# (trained on populated tiles), so whole empty-water tiles flip class (visible
# as a hard tile seam), variously to Bareland, Grass, Tree or Cropland. That
# both leaves trees unsuppressed over the mislabeled water AND insets the water
# region from the page edge (rounded blob instead of flush). Fix at the source
# by color: open water is strongly blue-shifted (measured B - R ≈ +40 here,
# whether the model called it bareland or grass), while REAL vegetation/bare
# earth is not (land grass B - R ≈ -1, trees ≈ +5, brown bareland negative). So
# any natural-cover pixel that is blue-dominant by this margin is reclaimed to
# Water. The +40 vs ~0 gap makes margin 10 safe for real land. 0 disables.
WATER_RECLAIM_BLUE_MARGIN = 10

_WATER_ID = 6
# Natural-cover classes the model confuses with open water. Reclaimed to Water
# when blue-dominant. Excludes Pavement/Road (gray, never blue — and paved has
# its own confidence gate) and buildings/unknown (dropped downstream anyway).
_RECLAIMABLE_TO_WATER = (1, 2, 5, 7)  # Bareland, Grass, Tree, Cropland

# --- model / inference config (SegFormer-B2 checkpoint, OEM-fine-tuned) --------
HF_REPO = "odil111/segformer-fine-tuned-on-openearthmap"
HF_CKPT = "segformer_sem_seg_2024-05-16--14-40-45/segformer_sem_seg_checkpoint_epoch35.pt"
TILE = 1000        # this checkpoint documents 1000x1000 inputs
OVERLAP = 128
# ImageNet normalization (the checkpoint was trained with it).
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_MODEL = None  # lazily-initialized singleton (weights download on first use)


def detect_land_types_seg(
    image: Image.Image,
    building_mask: np.ndarray,
    road_mask: np.ndarray,
    *,
    paved_min_conf: float = PAVED_MIN_CONF,
    water_reclaim_margin: float = WATER_RECLAIM_BLUE_MARGIN,
    classmap: np.ndarray | None = None,
) -> list[dict]:
    """
    Segment ground cover with a supervised model. Same contract as
    `landtypes.detect_land_types`.

    Args:
        image: full-resolution aerial (PIL).
        building_mask, road_mask: full-image binary masks (Overture-derived),
            same H×W as `image`; their union is excluded from land regions.
        paved_min_conf: confidence gate for paved pixels (see PAVED_MIN_CONF);
            low-confidence paved is demoted to its best non-paved class. Only
            applied on the model path (ignored when `classmap` is supplied).
        water_reclaim_margin: blue-dominance margin for reclaiming mislabeled
            open water (see WATER_RECLAIM_BLUE_MARGIN); 0 disables.
        classmap: TEST/ADVANCED HOOK — a precomputed H×W array of OEM class
            indices. When given, the model is not loaded or run at all (lets
            tests exercise the full polygon pipeline with a synthetic map, and
            never download weights or touch MPS). Normally left None.

    Returns:
        List of dicts per label: {label, polygons (MultiPolygon, full-image px),
        thumbnail (PIL.Image)}.
    """
    img_np = np.asarray(image.convert("RGB"))
    H, W = img_np.shape[:2]

    if classmap is None:
        prob = _segment_tiled(_get_model(), img_np)     # (C, H, W) softmax
        classmap = _confident_classmap(prob, paved_min_conf)
    if classmap.shape != (H, W):
        raise ValueError(f"classmap {classmap.shape} != image {(H, W)}")

    classmap = _reclaim_water(classmap, img_np, water_reclaim_margin)
    return _classmap_to_regions(classmap, img_np, building_mask, road_mask)


def _reclaim_water(classmap: np.ndarray, img_np: np.ndarray,
                   blue_margin: float = WATER_RECLAIM_BLUE_MARGIN) -> np.ndarray:
    """Reclaim open water the model mislabeled as land: a natural-cover pixel
    (see _RECLAIMABLE_TO_WATER) whose source color is blue-dominant
    (B - R > blue_margin) is really water (see WATER_RECLAIM_BLUE_MARGIN).
    Returns a possibly-new class map; the input is not mutated. `blue_margin
    <= 0` is a no-op."""
    if blue_margin <= 0:
        return classmap
    R = img_np[..., 0].astype(np.int16)
    B = img_np[..., 2].astype(np.int16)
    reclaim = np.isin(classmap, _RECLAIMABLE_TO_WATER) & ((B - R) > blue_margin)
    if not reclaim.any():
        return classmap
    classmap = classmap.copy()
    classmap[reclaim] = _WATER_ID
    return classmap


def _confident_classmap(prob: np.ndarray, paved_min_conf: float = PAVED_MIN_CONF) -> np.ndarray:
    """
    Argmax a (C, H, W) softmax volume to an (H, W) class map, but demote
    low-confidence paved pixels to their best non-paved class.

    A paved pixel (argmax in PAVED_IDS) whose winning probability is below
    `paved_min_conf` is reassigned to the highest-scoring class that is NOT
    paved — i.e. whatever the model's second-guess is (water at the shoreline,
    tree in shadow, bare on the beach). Confident paved (real roads/paths) and
    all non-paved pixels are untouched. `paved_min_conf <= 0` disables the gate.
    """
    cls = prob.argmax(axis=0).astype(np.uint8)
    if paved_min_conf <= 0:
        return cls

    conf = prob.max(axis=0)
    weak = np.isin(cls, PAVED_IDS) & (conf < paved_min_conf)
    if weak.any():
        # Work only on the weak pixels (small): drop the paved rows so they
        # can't win, then argmax over the remaining classes. Softmax probs are
        # >= 0, so -1.0 guarantees the paved classes lose.
        sub = prob[:, weak].copy()           # (C, n_weak)
        for pid in PAVED_IDS:
            sub[pid] = -1.0
        cls[weak] = sub.argmax(axis=0).astype(np.uint8)
    return cls


def _classmap_to_regions(
    classmap: np.ndarray,
    img_np: np.ndarray,
    building_mask: np.ndarray,
    road_mask: np.ndarray,
) -> list[dict]:
    """Collapse an OEM class map to our four labels and polygonize each, reusing
    the k-means path's cleanup (morph → area filter → Douglas-Peucker → Chaikin)
    and its pad/frame-flush trick so edge-touching regions stay square."""
    H, W = classmap.shape
    exclude = (building_mask > 0.5) | (road_mask > 0.5)

    pad = 16  # >= the morph-close kernel, so edge regions extend into the pad
    frame = _box(0, 0, W, H)

    # No SLIC/KMeans here, but OpenCV morphology still spins OpenMP threads;
    # keep the same single-thread guard the k-means path uses, in case torch/MPS
    # was initialized earlier in the process (tree detection, or our own model).
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # degrade gracefully
        from contextlib import nullcontext as threadpool_limits  # type: ignore

    results = []
    with threadpool_limits(1):
        for label in LABEL_ORDER:
            oem_ids = [i for i, lb in OEM_TO_OURS.items() if lb == label]
            mask = np.isin(classmap, oem_ids)
            mask[exclude] = False  # buildings/roads come from Overture layers
            if not mask.any():
                continue
            mask_u8 = mask.astype(np.uint8) * 255

            padded = cv2.copyMakeBorder(mask_u8, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            # Build FACETED polygons (chaikin off), clip flush to the frame, THEN
            # smooth with the frame edges protected. Smoothing before clipping
            # rounds the boundary where it should run flush against the page edge
            # (the "rounded blob floating off the edge" bug); this order keeps
            # edge-touching runs straight and only smooths interior corners.
            polygons = _mask_to_multipolygon(
                padded, chaikin_iterations=0, **_MORPH_BY_LABEL.get(label, {}))
            if polygons.is_empty:
                continue
            polygons = _affine_translate(polygons, xoff=-pad, yoff=-pad).intersection(frame)
            polygons = _as_multipolygon(polygons)
            if polygons.is_empty:
                continue
            polygons = _smooth_preserving_frame(polygons, W, H, iterations=2)
            if polygons.is_empty:
                continue
            results.append({
                "label": label,
                "polygons": polygons,
                "thumbnail": _make_thumbnail(img_np, mask_u8),
            })
    return results


def _on_frame(p, W, H, tol=1.0) -> bool:
    """True if point p lies on any of the four frame edges (within tol)."""
    return (abs(p[0]) <= tol or abs(p[0] - W) <= tol
            or abs(p[1]) <= tol or abs(p[1] - H) <= tol)


def _chaikin_frame_protected(coords: list, W: int, H: int,
                             iterations: int, tol: float = 1.0) -> list:
    """Chaikin corner-cutting that leaves the page frame flush. Vertices on a
    frame edge are anchors: they are never moved, and an edge whose BOTH
    endpoints are anchors is kept straight (no cut). Interior corners round
    normally. Input/output are closed rings (first == last)."""
    pts = coords[:-1] if len(coords) > 1 and coords[0] == coords[-1] else list(coords)
    if len(pts) < 3:
        return coords
    for _ in range(iterations):
        out, n = [], len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            aa, ab = _on_frame(a, W, H, tol), _on_frame(b, W, H, tol)
            if aa and ab:                       # frame run: keep straight
                out.append(a)
            elif aa:                            # preserve the frame anchor a
                out.append(a)
                out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
            elif ab:                            # preserve the frame anchor b
                out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
                out.append(b)
            else:                               # interior corner: cut both
                out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
                out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        pts = out
    pts.append(pts[0])
    return pts


def _smooth_preserving_frame(mp: MultiPolygon, W: int, H: int,
                             iterations: int = 2) -> MultiPolygon:
    """Apply frame-protected Chaikin smoothing to every ring of a MultiPolygon,
    so interiors read hand-drawn while any boundary on the page edge stays flush
    and square. buffer(0) repairs any self-touch the smoothing introduces."""
    out = []
    for g in mp.geoms:
        ext = _chaikin_frame_protected(list(g.exterior.coords), W, H, iterations)
        holes = [_chaikin_frame_protected(list(r.coords), W, H, iterations)
                 for r in g.interiors]
        p = Polygon(ext, holes)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            continue
        out.extend([p] if p.geom_type == "Polygon"
                   else [x for x in p.geoms if not x.is_empty])
    return MultiPolygon(out) if out else MultiPolygon()


# ---------------------------------------------------------------------------
# Model loading + tiled inference. All torch imports are function-local (module
# -scope torch segfaults the pytest suite). None of this runs in the unit tests,
# which pass a synthetic `classmap` instead.
# ---------------------------------------------------------------------------

def _get_model():
    """Lazily load + cache the SegFormer. Downloads weights on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _strip_prefix(sd):
    """Unwrap a checkpoint nested under state_dict/model/model_state_dict and
    strip any 'module.'/'model.' key prefixes."""
    for k in ("state_dict", "model_state_dict", "model"):
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


def _remap_legacy_segformer(sd):
    """This checkpoint was saved with transformers 4.x, whose SegFormer
    state-dict key names differ from transformers >=5 (encoder->stages,
    block->blocks, attention.self.{query,key,value}->attention.{q,k,v}_proj,
    output.dense->o_proj, mlp.dense{1,2}->mlp.fc{1,2}, decode_head.linear_c->
    linear_projections). Without this rename, load_state_dict(strict=False)
    silently drops every key and the model runs on random init — which looks
    like success but classifies the whole image as one class."""
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


def _infer_arch(sd):
    """Read the architecture straight from the (remapped) checkpoint rather than
    trusting a variant name: stem width picks b0 vs b1-base, depths (blocks per
    stage) disambiguate b1..b5 which all share widths, and the decode-head width
    is often non-default (this checkpoint uses 768, not b1's 256)."""
    import re
    c0 = next((v.shape[0] for k, v in sd.items()
               if k.endswith("stages.0.patch_embeddings.proj.weight")), None)
    base = {32: "nvidia/mit-b0", 64: "nvidia/mit-b1"}.get(c0)

    depths = {}
    for k in sd:
        m = re.match(r"segformer\.stages\.(\d+)\.blocks\.(\d+)\.", k)
        if m:
            s, b = int(m.group(1)), int(m.group(2))
            depths[s] = max(depths.get(s, 0), b + 1)
    depths = [depths[i] for i in range(len(depths))] if depths else None

    dec_hidden = None
    for key in ("decode_head.batch_norm.weight", "decode_head.classifier.weight"):
        if key in sd:
            dec_hidden = int(sd[key].shape[0] if key.endswith("batch_norm.weight")
                             else sd[key].shape[1])
            break
    return base, depths, dec_hidden


def _load_model():
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    path = hf_hub_download(repo_id=HF_REPO, filename=HF_CKPT)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = _remap_legacy_segformer(_strip_prefix(raw))

    base, depths, dec_hidden = _infer_arch(sd)
    if base is None:
        raise RuntimeError("Could not infer SegFormer architecture from checkpoint")

    id2label = {i: OEM_NAMES[i] for i in range(N_CLASSES)}
    cfg_kwargs = dict(num_labels=N_CLASSES, id2label=id2label,
                      label2id={v: k for k, v in id2label.items()})
    if depths is not None:
        cfg_kwargs["depths"] = depths
    if dec_hidden is not None:
        cfg_kwargs["decoder_hidden_size"] = dec_hidden
    cfg = SegformerConfig.from_pretrained(base, **cfg_kwargs)

    model = SegformerForSemanticSegmentation(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if real_missing or unexpected:
        # A partial load silently produces confident garbage — fail loudly.
        raise RuntimeError(
            f"Checkpoint did not map cleanly ({len(real_missing)} missing, "
            f"{len(unexpected)} unexpected). Refusing partially-loaded weights."
        )
    model.eval()
    return model


def _device():
    import torch
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _tiles(H, W, tile, overlap):
    step = tile - overlap
    ys = list(range(0, max(1, H - overlap), step))
    xs = list(range(0, max(1, W - overlap), step))
    for y in ys:
        for x in xs:
            yield min(y, max(0, H - tile)), min(x, max(0, W - tile))


def _segment_tiled(model, img_rgb: np.ndarray) -> np.ndarray:
    """Tiled inference at ~native resolution (deliberately NOT downscaled to
    1400 px — preserving resolution is what keeps thin pavement). Accumulates
    per-tile softmax in the overlaps and returns the averaged (C, H, W)
    probability volume; the caller argmaxes it (with the paved confidence gate,
    which needs the probabilities, not a pre-argmaxed class map)."""
    import torch
    import torch.nn.functional as F

    dev = _device()
    model.to(dev)
    H, W = img_rgb.shape[:2]
    acc = np.zeros((N_CLASSES, H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for (y, x) in _tiles(H, W, TILE, OVERLAP):
            th, tw = min(TILE, H - y), min(TILE, W - x)
            crop = img_rgb[y:y + th, x:x + tw].astype(np.float32) / 255.0
            crop = (crop - _MEAN) / _STD
            t = torch.from_numpy(crop.transpose(2, 0, 1))[None].to(dev)
            logits = model(pixel_values=t).logits          # [1, C, h/4, w/4]
            logits = F.interpolate(logits, size=(th, tw),
                                   mode="bilinear", align_corners=False)
            prob = F.softmax(logits, dim=1)[0].cpu().numpy()
            acc[:, y:y + th, x:x + tw] += prob
            cnt[y:y + th, x:x + tw] += 1.0
            del t, logits, prob

    cnt[cnt == 0] = 1.0
    return acc / cnt          # (C, H, W) averaged softmax
