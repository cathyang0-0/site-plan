# Site Plan Drafter

Draw a box on a map inside Rhino → get a clean, layered, architect-style
**site plan** (building roofs, roads, trees, land-cover hatches, contours)
imported into your document at real-world scale.

Under the hood: aerial imagery (USGS) + open map data (Overture/OSM) +
computer vision (DeepForest tree detection, land-cover segmentation),
exported as a DXF with editable blocks, real hatch patterns, and one layer
per feature class. A live preview lets you tune tree sizes after detection
and import exactly what you see.

**Two parts:**

| Part | What it is | Where it runs |
|---|---|---|
| `rhino/` | The Rhino 8 plugin (Eto dialog + map picker) | Inside Rhino 8 (Mac/Windows) |
| `backend/` | FastAPI service doing imagery, CV, and DXF export | A local Python process |

## Install

### 1. The backend (one-time)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv tool install "siteplan-backend @ git+https://github.com/cathyang0-0/site-plan#subdirectory=backend"
```

This puts a `siteplan-backend` command on your PATH in its own isolated
environment. Heads-up: the CV stack (PyTorch, DeepForest) is a multi-GB
download, and the first tree detection additionally downloads model weights.

You don't need to start it yourself — the Rhino command finds it and starts
it automatically. To run it manually anyway:

```bash
siteplan-backend
```

### 2. The Rhino plugin

Until the packaged plugin is published: download/clone this repo, then in
Rhino 8 run `ScriptEditor`, open `rhino/SitePlan_command.py`, press ▶.
See [rhino/README.md](rhino/README.md) for the full tour (map picker,
options, tree preview) and a first-run WebView probe.

## Developing

```bash
cd backend
uv venv && uv pip install -r requirements.txt   # dev-pinned versions
python -m pytest tests/ -q                       # 300+ tests, no network needed
python -m uvicorn siteplan_backend.main:app --port 8000
```

The pipeline can also run headless without Rhino — see
`backend/scripts/poc.py --help`, and `spec.md` / `docs/HANDOFF.md` for
design and current status.

## Data sources, licenses, attribution

- **Aerial imagery & elevation:** USGS (public domain). The DXF's imagery
  is fetched per-job; the map picker shows the same tiles.
- **Buildings / roads / water / infrastructure:** Overture Maps /
  OpenStreetMap contributors, **ODbL** — exported DXFs carry the required
  attribution text automatically.
- **Address search:** OSM Nominatim (see their
  [usage policy](https://operations.osmfoundation.org/policies/nominatim/)).
- **Tree detection:** [DeepForest](https://github.com/weecology/DeepForest).
- **Land cover:** default engine is license-clean unsupervised k-means. The
  optional SegFormer engine (`pip extra: segmodel`) uses OpenEarthMap
  weights that are **CC BY-NC-SA (non-commercial)** — it is evaluation-only
  and never enabled by default.
- **This repo's license:** TBD before first release.
