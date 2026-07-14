# Land-cover model scoping — replacing k-means land-type detection with pretrained semantic segmentation

**Status:** exploratory scoping. Nothing here is merged into the pipeline.
**Author:** generated 2026-07-14.
**Scope:** evaluate a *pretrained, supervised* semantic-segmentation model for land cover to replace/augment the unsupervised k-means in [`backend/app/pipeline/landtypes.py`](../backend/app/pipeline/landtypes.py).

> **Source-accuracy note.** Class names below are taken from the dataset's own source
> code (authoritative). License and GSD statements are cross-checked across the GitHub
> README, the Zenodo record, and the arXiv/WACV paper. Where a license drives a
> go/no-go decision (commercial use), **re-read the raw `LICENSE` file before shipping** —
> the wording matters and this doc is scoping, not legal advice.

---

## 1. TL;DR / recommendation

**The problem is a good fit for supervised segmentation, and OpenEarthMap is the right model family — but there is a genuine license conflict that blocks *commercial production* use of the off-the-shelf weights.**

| Question | Answer |
|---|---|
| Best technical fit? | **OpenEarthMap (OEM)** — 8-class land cover at **0.25–0.5 m GSD** (our imagery ≈ 0.30 m/px), global domain incl. North America, with **dedicated `Pavement` *and* `Road` classes** that directly target the thin-pavement misses. |
| Off-the-shelf, no training? | **Partly.** The *official* OEM repo ships **no** downloadable weights. Community checkpoints on Hugging Face do (SegFormer, U-Net, DeepLabV3+), and `transformers` (already installed) can run the SegFormer one. |
| Commercial license clean? | **No — this is the gating issue.** OEM *code* is MIT, but OEM *label data* is partly **CC BY-NC-SA 4.0** (NonCommercial + ShareAlike). Weights trained on it inherit an unresolved NC/ShareAlike question. LoveDA is explicitly **commercial-prohibited**. |
| torchgeo as the integration path? | **No for weights.** torchgeo ships pretrained *backbones/SSL encoders*, not pretrained *land-cover segmentation decoders* — using it for land cover means training. |

**Recommendation, in order:**

1. **Prototype and evaluate now with OpenEarthMap** (SegFormer or U-Net checkpoint, run via `transformers`) as an **evaluation-only** engine, behind a flag, keeping k-means as the default fallback. This validates that supervised segmentation crisply captures thin pavement/paths — the whole motivation.
2. **Do not ship OEM/LoveDA off-the-shelf weights in the commercial product** until the license is resolved. Two clean production paths if the POC succeeds: (a) get a commercial-use clarification/license for OEM weights, or (b) **retrain the same architecture on a permissively-licensed land-cover dataset** (reintroduces training, but the code/integration is identical).
3. **Reject LoveDA** for anything but research comparison (NonCommercial *and* China-urban/rural domain shift). **Do not build the integration on torchgeo weights** (backbones only).

Net: the *engineering* is low-risk and reuses almost all of `landtypes.py`. The *legal* status of the pretrained weights is the real decision, and it's a business/legal call, not a technical one.

---

## 2. Why the current k-means misses thin pavement (recap)

`landtypes.py` runs **unsupervised** k-means (k=4) on per-superpixel color+texture means:

- **SLIC ~500 superpixels** for the whole image → a winding path or a road-through-green is a minority of pixels inside a big vegetation superpixel and gets averaged away.
- **Downscale to ≤1400 px** (`MAX_DETECT_DIM`) before segmentation → thin features sub-pixel out.
- **11 px morphological close** in `_mask_to_multipolygon` → surviving thin slivers get erased.
- **No learned class concept** — `_assign_semantic_labels` guesses water/veg/paved/bare from *mean color* of each cluster, so "gray-ish thin thing in a green field" never wins the `paved` score.

A supervised per-pixel model fixes the root cause: it has a learned `Pavement`/`Road` concept and labels each pixel, so a 2-px path is classified as pavement regardless of what surrounds it — **provided we stop throwing away resolution** (tile at ~native res instead of downscaling to 1400, and shrink/skip the 11 px close on the paved class).

---

## 3. Candidate comparison (verified against primary sources)

### 3.1 OpenEarthMap (OEM) — **lead candidate**

- **Classes (authoritative, from the repo's own `open_earth_map/utils.py` → `class_grey_oem`):**

  | idx | class name (repo) | common name (paper) | → our label |
  |----:|---|---|---|
  | 0 | `unknown` | background | (ignore) |
  | 1 | `Bareland` | Bareland | **bare earth / farmland** |
  | 2 | `Grass` | Rangeland | **vegetation** |
  | 3 | `Pavement` | Developed space | **paved / hardscape** |
  | 4 | `Road` | Road | **paved / hardscape** |
  | 5 | `Tree` | Tree | **vegetation** |
  | 6 | `Water` | Water | **water** |
  | 7 | `Cropland` | Agriculture | **bare earth / farmland** (or vegetation) |
  | 8 | `buildings` | Building | (ignore — we get buildings from Overture; use as cross-check) |

  This maps **cleanly** onto our four labels, and crucially it has **separate `Pavement` and `Road` classes** — exactly the thin-hardscape signal k-means lacks.

- **GSD / domain (arXiv 2210.10732 abstract, verbatim):** *"OpenEarthMap consists of 2.2 million segments of 5000 aerial and satellite images covering 97 regions from 44 countries across 6 continents, with manually annotated 8-class land cover labels at a 0.25–0.5m ground sampling distance."* Global coverage incl. North America; **0.25–0.5 m matches our ≈0.30 m/px** → our Cayuga/Myers Point imagery is in-distribution.

- **License (the gating issue):**
  - **Code:** MIT (`bao18/open_earth_map`).
  - **Labels:** dual and partly restrictive. Per the repo README, label data *"are provided under the same license as the original RGB images, which varies with each source dataset,"* and for public-domain / license-unstated source regions the labels are under a **"Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License."** The Zenodo record lists **CC BY 4.0** as the primary license with that **CC BY-NC-SA 4.0** carve-out; the arXiv landing page shows a **CC BY-NC-SA 4.0** icon.
  - **Implication:** the dataset as distributed contains **NonCommercial + ShareAlike** portions. Whether model *weights* trained on NC data are themselves NC is legally unsettled, but for a commercial product this is a **flag, not a green light**.

- **Off-the-shelf weights?** The official repo publishes code + a baseline recipe but **no downloadable weights** in its README. Usable community checkpoints exist on Hugging Face:
  - `odil111/segformer-fine-tuned-on-openearthmap` — SegFormer, self-declared **`license: mit`**, inputs 1000×1000 (and a 650×650 variant). **Runnable with the already-installed `transformers`.**
  - `odil111/unet-fine-tuned-on-openearthmap` — U-Net, self-declared **`license: mit`**, 1024×1024.
  - `Zongrong/deeplabv3_rs_openearthmap` — DeepLabV3+ state_dict, **no license declared**.

  ⚠️ These uploaders declare MIT **on the weights**, but that declaration can't unilaterally override the **training-data** license. Treat the MIT tag as the *author's* position, not a resolution of the OEM NC-SA question.

### 3.2 LoveDA — **reject (commercial + domain)**

- **Classes (7):** Background, Building, Road, Water, Barren, Forest, Agriculture. Maps to our four but **no separate fine "pavement/developed" class** beyond `Road`.
- **GSD:** 0.3 m. **Domain:** three Chinese cities — Nanjing, Changzhou, Wuhan (urban + rural). **Domain shift risk** for US suburban lakeshore.
- **License:** **CC BY-NC-SA 4.0**, and the README is explicit — *"All images and their associated annotations in LoveDA can be used for academic purposes only, but any commercial use is prohibited."* → **hard fail for a commercial product.** Pretrained HRNet weights exist but inherit the NC restriction.

### 3.3 torchgeo — **not a weights source for this task**

- torchgeo's `SemanticSegmentationTask` wires up U-Net/DeepLab/etc. and can load **ImageNet encoder** weights or torchgeo's own **SSL/backbone** pretrained weights (Sentinel/NAIP/SSL4EO, classification & backbones) — **but it does not ship a pretrained land-cover *segmentation decoder***. LoveDA/LandCover.ai are provided as **datasets to train on**, not as ready segmenters. Using torchgeo for land cover ⇒ **training**, which is out of scope now.
- It also adds a **heavy dependency** (torchgeo + its stack) we don't otherwise need.

### 3.4 Lighter route (what we'd actually run)

We do **not** need torchgeo or mmsegmentation. The lightest viable path:

- **SegFormer via `transformers`** (already installed, **Apache-2.0**) + an OEM SegFormer checkpoint → one HF download, standard `SegformerForSemanticSegmentation.from_pretrained`-style load, MPS inference. **This is the POC path.**
- Alternative: `segmentation-models-pytorch` (**MIT**, *not* currently installed) to load a U-Net/DeepLabV3+ state_dict. Adds one small dep. Use only if the SegFormer checkpoint doesn't load cleanly.

---

## 4. Comparison table (against the six required criteria)

| Criterion | OpenEarthMap | LoveDA | torchgeo (as weights source) |
|---|---|---|---|
| **1. License — code** | MIT ✅ | code Apache-ish, dataset separate | MIT ✅ |
| **1. License — weights/data (GATING)** | Data partly **CC BY-NC-SA 4.0** ⚠️ (community weights self-tag MIT, unresolved) | **CC BY-NC-SA 4.0, commercial prohibited** ❌ | Backbone/SSL weights permissive, but **no land-cover segmenter** |
| **2. Off-the-shelf (no training)** | ✅ via community HF checkpoints (official repo: none) | ✅ (HRNet) but NC | ❌ would require training |
| **3. Class mapping to water/veg/paved/bare** | **Excellent** — separate Pavement+Road ✅ | Good, coarser (Road only) | n/a (whatever you train) |
| **4. GSD / domain fit** | **0.25–0.5 m, global incl. N. America** ✅ | 0.3 m but **China urban/rural** ⚠️ | n/a |
| **5. Local runnable (MPS, ~2700×3000, tiled)** | ✅ SegFormer-B0/B1 or U-Net, seconds/tile | ✅ | — |
| **6. Deps / footprint** | **`transformers` (already installed)** ✅ | heavier (mmseg-style) | **torchgeo stack (heavy)** ❌ |

---

## 5. Local runnability plan (Apple Silicon / MPS)

- Test image is ~2700×3000. **Do not downscale to 1400** (that's what erases thin pavement). Instead **tile**: e.g. 1024×1024 tiles with ~128 px overlap, run per tile on MPS, stitch by summing softmax logits in the overlap and taking argmax → seamless class map at near-native resolution.
- SegFormer-B0 (~3.7M enc params) / B1, or U-Net-EfficientNet-B4: **a few tiles × sub-second each on MPS ⇒ well under a minute** for the whole image (rough estimate; confirm in the POC).
- Memory: 1024² tiles keep MPS memory modest; fall back to 512² if needed.

---

## 6. Integration plan (keeps `detect_land_types`'s contract; nothing merged yet)

The **only** thing that changes is per-pixel classification. Everything downstream is reused.

1. **New module `backend/app/pipeline/landtypes_seg.py`** exposing the same contract as `detect_land_types`: `(image, building_mask, road_mask) -> list[{"label", "polygons": MultiPolygon (pixel coords), "thumbnail", ...}]`.
2. **Engine selection.** Add a flag (e.g. `LANDTYPE_ENGINE = "kmeans" | "segmodel"`, or a `poc.py --land-types-engine` arg). **k-means stays the default fallback** for imagery the model handles poorly / when weights are unavailable.
3. **Per-pixel → per-label masks.** Run the model (tiled), argmax to the 9 OEM classes, then **collapse** via the §3.1 map to our four labels. Apply the same `building_mask | road_mask` exclusion the k-means path uses (and optionally cross-check the model's `buildings`/`Road` against Overture).
4. **Reuse the existing polygon pipeline unchanged:** feed each collapsed label mask through **`_mask_to_multipolygon`** (morph → area filter → Douglas-Peucker → Chaikin) and the same pad/`intersection(frame)` frame-flush logic, then the per-type-layer DXF export. **Tune morphology per class**: shrink/skip the 11 px close for `paved` so thin paths survive (this is the payoff — don't undo the model's precision in cleanup).
5. **Thumbnails / hatch styles:** reuse `_make_thumbnail` and `default_hatch_style` as-is (labels are identical strings).

### Concurrency / import gotchas (must respect)

- **OpenMP deadlock:** torch/MPS and sklearn/skimage share OpenMP pools and deadlock if interleaved. The seg path **removes `slic` + `KMeans`** (the main offenders), but `_mask_to_multipolygon` still uses OpenCV. **Order the work: run *all* torch inference first and free the tensors, *then* do polygonization**; keep any residual skimage/sklearn calls wrapped in `threadpool_limits(1)` exactly as today.
- **Keep `import torch` function-local** — a module-level torch import segfaults the test suite. Load the model inside the detect function (or a lazily-initialized singleton), never at import time.
- **Weights provenance:** while license is unresolved, gate the seg engine behind an explicit opt-in flag and **don't** bundle weights in the repo/image; download on first use to a cache. Document the license caveat at the call site.

### Testing

- Add unit tests that exercise `landtypes_seg` with the model **mocked** (feed a synthetic class map) so CI never downloads weights or needs MPS. Keep the existing 137 tests green; the seg path must be import-safe (no module-level torch).

---

## 7. Proof-of-concept (bonus)

See [`backend/scripts/poc_landcover_seg.py`](../backend/scripts/poc_landcover_seg.py) — a **throwaway** script (not wired into the pipeline) that:

- downloads an OEM SegFormer checkpoint from Hugging Face,
- runs tiled inference on `test_data/img/cayuga_myers_point.png` on MPS,
- saves a color-coded segmentation overlay + a paved-class-only mask, so we can eyeball whether it catches the thin pavement the k-means blurs, and
- (optionally) runs the current `detect_land_types` for side-by-side comparison.

> POC status: script is written to be runnable but **downloads third-party weights** (the exact
> thing whose license is in question) — run it only for evaluation. It had not been executed at
> the time this doc was written; treat any inference-time numbers as estimates until run.

---

## 8. Open questions / decisions for a human

1. **Legal:** Is `CC BY-NC-SA 4.0`-derived weight usage acceptable for evaluation only, and what's the bar for production? (Drives whether path 2a "license clarification" or 2b "retrain on permissive data" is the production plan.)
2. **Cropland mapping:** send OEM `Cropland` → our *bare earth / farmland* (keeps farmland semantics) or → *vegetation*? Depends on how sites like Myers Point read.
3. **Permissive retraining dataset** (if we go 2b): candidates to vet for license + US domain (e.g. Chesapeake Land Cover, DeepGlobe) — separate scoping.

---

## Sources

- OpenEarthMap arXiv: https://arxiv.org/abs/2210.10732 · PDF https://arxiv.org/pdf/2210.10732
- OpenEarthMap code (MIT) + class palette (`open_earth_map/utils.py`): https://github.com/bao18/open_earth_map
- OpenEarthMap project site: https://open-earth-map.org/ · Zenodo record: https://zenodo.org/records/7223446
- WACV 2023 paper: https://openaccess.thecvf.com/content/WACV2023/papers/Xia_OpenEarthMap_A_Benchmark_Dataset_for_Global_High-Resolution_Land_Cover_Mapping_WACV_2023_paper.pdf
- LoveDA: https://github.com/Junjue-Wang/LoveDA
- torchgeo models API: https://docs.torchgeo.org/en/stable/api/models.html
- Community OEM checkpoints: https://huggingface.co/odil111/segformer-fine-tuned-on-openearthmap · https://huggingface.co/odil111/unet-fine-tuned-on-openearthmap · https://huggingface.co/Zongrong/deeplabv3_rs_openearthmap
- OEM lightweight models (MIT): https://github.com/cliffbb/oem-lightweight
