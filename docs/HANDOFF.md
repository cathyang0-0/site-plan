# Site Plan Drafter — session handoff

Snapshot for a fresh session. Full design is in [`spec.md`](spec.md); this is
"where things actually stand and what will bite you."

## What this is
Aerial imagery → clean, layered, architect-style **DXF context plan** (buildings,
roads, trees, land-cover hatches). Python backend; the whole pipeline is driven by
[`backend/scripts/poc.py`](../backend/scripts/poc.py) (no API/frontend yet). Real CV
models are wired in; a React frontend and FastAPI/Celery job layer are stubs.

## What's built (all on `master`, 137 backend tests passing)
| Module | File | Approach |
|---|---|---|
| Buildings | `app/pipeline/footprints.py`, `buildings.py` | **Overture footprints** (primary, georeferenced) + **SAM2 zero-shot** fallback + canopy cross-filter |
| Roads | `app/pipeline/roads.py` | **Overture centerlines** + hybrid width (class prior nudged by CV pavement measurement) |
| Trees | `app/pipeline/trees.py` | **DeepForest** + dense-stand fill + crown-size/variance controls + overlap-containment fix + **suppress-over-water** |
| Land types | `app/pipeline/landtypes.py` | **unsupervised k-means** (color-forward features, semantic labels, Chaikin-smoothed polys, per-type layers, blue-green→water bias) |
| DXF export | `app/export/dxf.py` | merged road corridors + fillets, roof white-fill masking, editable/uniquely-named tree blocks, one layer per land type, ODbL attribution |

Open-data (Overture) beat CV decisively for buildings and roads; land cover stays CV
(open data too coarse). Verified end-to-end on two sites: Alamo Heights (suburban) and
**Myers Point / Cayuga Lake, NY** (lakeside, exercises water). Demo output:
`test_data/output/*.dxf`.

## Plugin front end (2026-08-18) — BUILT, needs first in-Rhino run
The Rhino command now opens an **Eto dialog**: embedded map (keyless MapLibre +
USGS tiles + Nominatim search, `rhino/siteplan_map.html`) for drawing the bbox,
native controls for trees/sliders/engine/contours + per-class **road widths and
river width** (new `DetectOptions.road_class_widths` / `river_width_m` — the
water cache now stores RAW geoms, kind `water_raw`, so widths re-buffer), and a
per-stage progress page. See `rhino/README.md`; run `rhino/probe_webview.py`
once in Rhino first (verifies tile CORS from file:// + the ExecuteScript
bridge; fallback documented in its docstring). Bridge = UITimer polling
`window.getState()` — do NOT switch to DocumentTitleChanged (flaky on Mac).

## Not built yet
- **Web frontend** — `frontend/src/components/*.jsx` are still stubs; the plugin
  dialog covers the primary flow. Web-only wants: style panel, tree-block upload.
- **Celery/Redis** — not wired (in-process job threads in `app/api/jobs.py` are
  fine for solo/local use).
- **Land-cover semantic-seg model** — integrated behind `land_types_engine=
  "segmodel"`, license question still open (see below).
- **Manual tree edits** (plot/paint-fill/erase) — spec'd (§6 Step 4.5), no UI.
- Land-type polish: <50 m² region merge, per-class morphology.

## How to run
```bash
# full plan (georeferenced site). NOTE: --bbox needs the = form (leading-dash values)
cd backend && python scripts/poc.py --image ../test_data/img/cayuga_myers_point.png \
  --scale 0.30 --bbox="-76.5515,42.5305,-76.5415,42.5385" \
  --real-trees --footprint-buildings --footprint-roads --land-types \
  --crown-scale 1.5 --output /tmp/plan.dxf
# tests
python -m pytest backend/tests/ -q          # 137 passing
# fetch USGS test imagery (keyless; no MAPBOX_TOKEN configured)
python scripts/fetch_test_imagery.py W S E N M_PER_PX out.png
```
Tree detection on a full image is CPU/MPS-bound (~5 min); roads/buildings/land-types are
seconds. `poc.py` has flags `--real-buildings` (SAM2), `--no-stand-fill`, `--size-variance`.

## Landmines (these have all bitten us)
- **`import torch` at module scope segfaults the test suite** → keep torch imports *function-local*.
- **torch/MPS + scikit-learn/scikit-image OpenMP deadlock** (0% CPU hang) if interleaved in one
  process → the k-means/SLIC work is wrapped in `threadpoolctl.threadpool_limits(1)`; land-type
  detection runs *after* tree detection safely because of this. Any new torch model must respect it.
- **argparse eats leading-dash values** → always `--bbox="-76.5,..."` (equals form).
- **ezdxf's matplotlib backend cannot render *pattern* hatches** (renders blank) — a preview
  limitation, not a data bug; verify hatches in real CAD or by reading entities back.
- **Overture fetches have no built-in timeout** (one hung >1 h) → wrapped in `fetch_with_timeout`.
- Tree blocks must stay origin-centered (scatter bug), uniform 3-axis scale (Rhino editability),
  and uniquely named per export (Rhino re-import collision) — see `dxf.py` comments.
- Land-type detection downscales to 1400 px for speed, which **erases thin pavement** — the core
  motivation for the land-cover model scoping.

## Next step candidates
1. **Land-cover model** — read [`docs/land-cover-model-scoping.md`](land-cover-model-scoping.md).
   Recommendation: prototype **OpenEarthMap** (SegFormer via already-installed `transformers`)
   behind a flag, k-means stays default. **Gating issue is legal**: OEM label data is partly
   CC BY-NC-SA (NonCommercial) — fine for evaluation, needs resolution before commercial ship.
   Throwaway POC: `backend/scripts/poc_landcover_seg.py` (downloads third-party weights; eval only).
2. **Frontend** build-out (map pick → generate → style → download).
3. **API/job layer** (FastAPI routes + Celery).

## Git workflow
Remote `origin` = `github.com/cathyang0-0/site-plan`. Work is committed to local `master`, then
shipped via a **feature branch → PR → merge** (you can't PR `master`→`master`). Two PRs merged so
far (#1 pipeline, #2 land-type layers). Push/PR only when asked. Ignore the incidental
`.obsidian/workspace.json` and `test_data/img/cpp.png` changes — pre-existing noise, not ours.
