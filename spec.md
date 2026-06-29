# Site Plan Drafter — Technical Specification

**Version:** 0.2  
**Status:** Draft  
**Last updated:** 2026-06-29

---

## 1. Overview

A web-based tool that takes an aerial image of a site (by coordinate selection or image upload) and generates a clean, architect-quality context site plan drawing — trees as individually placed user-imported blocks, buildings as clean closed outlines with white fill masking, roads as smooth detected lines, land textures as user-assignable hatches, and optional contours. The user customizes styles and downloads a layered DXF and/or 3DM file.

### Target output quality
Matches the style of a hand-drafted context diagram: clean geometry, no noise, architect-legible symbology. Not a construction document — a context plan.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (React)                      │
│  Map picker → Generate → Calibrate → Style → Download   │
└───────────────────────┬─────────────────────────────────┘
                        │ REST API (JSON)
┌───────────────────────▼─────────────────────────────────┐
│                  Backend (FastAPI / Python)              │
│                                                         │
│  1. Imagery fetch        (Mapbox Satellite API)         │
│  2. CV pipeline          (PyTorch + fine-tuned models)  │
│  3. Texture clustering   (unsupervised land type det.)  │
│  4. Vector processing    (Shapely + OpenCV)             │
│  5. File export          (ezdxf + rhino3dm)             │
└─────────────────────────────────────────────────────────┘
```

Backend is stateless per request. Jobs queued via Celery + Redis (CV inference takes 10–60s). Output geometry returned as GeoJSON; styling applied client-side for preview, then re-applied at export time.

---

## 3. Input

### Imagery
- **Source:** Mapbox Satellite Static Tiles API (zoom 18–19, ~10–20 cm/px in urban areas)
- **Selection:** User draws a bounding box on a Mapbox GL JS map
- **Tile stitching:** Backend fetches and stitches tiles into a single image for inference

### User-imported tree blocks
- User uploads 1–3 block files (DXF or 3DM format) representing their custom tree symbols
- After upload, a calibration step establishes real-world scale (see section 6b)
- If 2–3 blocks are imported, they are placed randomly across all detected tree positions (uniform random assignment, seeded so the same result is reproducible)

---

## 4. CV Pipeline

Modules run in parallel after imagery is fetched. Each can be independently enabled/disabled per job.

### 4a. Building / Roof Outlines

**Goal:** Clean closed polygons for all building footprints.

**Model recommendation — fine-tuned segmentation:**

Start with a U-Net (ResNet-50 or EfficientNet-B4 backbone) fine-tuned on one of:
- **SpaceNet 2 Building Footprint** dataset — 150k+ building labels across 5 cities, diverse typologies, best general-purpose choice
- **INRIA Aerial Image Labeling** — high-quality labels for dense European/US urban areas, better for suburban contexts like the reference drawing
- **Microsoft Building Footprints** (used as pseudo-labels) — global coverage, noisier but vast quantity

Recommended approach: pretrain on SpaceNet 2, fine-tune on INRIA. Optionally use SAM2 zero-shot as a quick baseline to validate the pipeline before investing in training.

**Post-processing:**
1. Binary mask → connected components
2. Filter by area (min 10 m², max 10,000 m²)
3. `cv2.findContours` → Shapely polygon
4. Douglas-Peucker simplification (tolerance ~0.5 m)
5. Iterative orthogonalization: snap interior angles to nearest 15°, then to nearest 90° if within 10° threshold — preserves angled rooflines while squaring off near-rectangular buildings
6. Output: list of `shapely.Polygon`

**Layer behavior:**
- Roof polygons carry a **white solid fill hatch** and sit on the topmost layer in drawing order
- This creates a visual masking effect: any road, tree, or contour that spatially overlaps a building is occluded by the white fill, mimicking trimmed geometry without actual boolean operations
- The roof outline (perimeter curve) is drawn on top of the fill

**Layer name:** `ROOFS`

### 4b. Roads / Paths

**Goal:** Smooth polylines representing road centerlines or edge pairs, cleanly closed at intersections.

**Model:** Fine-tuned DeepLabV3+ on road segmentation. Training data options:
- **SpaceNet 3 Road Network** — road mask labels with good coverage
- **Massachusetts Roads Dataset** — aerial road labels, well-established benchmark

Road width estimated from the mask's medial axis thickness → used to offset centerline into edge pairs if the user selects double-line mode.

**Post-processing:**
1. Binary road mask → skeletonization (`skimage.morphology.skeletonize`)
2. Skeleton → graph (junction nodes, branch edges)
3. Dead-end pruning (remove stubs shorter than 3 m)
4. Per-edge cubic spline smoothing
5. Intersection closure: snap endpoints within 1 m tolerance

**Layer name:** `ROADS`

### 4c. Tree Instance Detection

**Goal:** Individual canopy centroid + radius for each detected tree.

**Model:** **DeepForest** (Python package) — pretrained on NEON aerial tree crown detection data, directly outputs bounding boxes around individual canopies. Fine-tunable on custom aerial datasets.

Alternative if DeepForest performs poorly on the target imagery style: SAM2 Automatic Mask Generator filtered to round blobs in the 2–15 m diameter range.

**Post-processing:**
1. NMS on overlapping detections (IoU threshold 0.3)
2. Extract centroid and radius from each bounding box
3. Filter: radius must be 1–10 m
4. Output: list of `(x, y, radius)` tuples in real-world coordinates

**Layer name:** `TREES`

### 4d. Land Texture Detection (new)

**Goal:** Identify distinct ground-cover types (farmland, grass, hardscape, gravel, water, etc.) as separate filled regions that the user can assign hatches to or discard.

**Approach — unsupervised texture segmentation:**
1. Mask out building footprints and detected road areas from the aerial image
2. Extract texture features per pixel: Gabor filter bank (4 scales × 6 orientations), color histograms in HSV space, LBP (Local Binary Patterns)
3. Spatial smoothing (superpixel pre-segmentation via SLIC to reduce noise)
4. K-means clustering on texture features, **k=4** (fixed): targets the four meaningful ground cover types on a typical site — water, grass/vegetation, dirt/farmland, paved/hardscape
5. Merge small disconnected regions (< 50 m²) into their largest neighboring cluster
6. Each cluster → a closed `shapely.MultiPolygon`
7. Backend generates a small thumbnail crop representative of each cluster

**UI presentation:**
- After processing, the style panel shows detected land types as a list
- Each entry has: thumbnail swatch, auto-generated label (e.g. "Type A — dark vegetation", "Type B — light gravel"), and controls:
  - Hatch style (none / parallel lines / cross-hatch / dots / solid)
  - Hatch color, angle, scale
  - **Discard** (hide this region entirely)
- The user assigns the same or different hatches to each cluster

**Layer naming:** `LANDTYPE_A`, `LANDTYPE_B`, etc. (renamed by the user in the style panel)

### 4e. Contours (optional, v2)

Source: Copernicus DEM (30m, free globally) or USGS 3DEP (10m, US only).  
Processing: DEM fetch → reproject → Gaussian smooth → contour extraction → clip to site boundary.  
**Layer name:** `CONTOURS`

---

## 5. Layer Schema & Drawing Order

Drawing order determines visual stacking. Lower index = drawn first (bottom).

| Order | Layer | Geometry | Notes |
|---|---|---|---|
| 1 | `CONTOURS` | Polyline | Bottommost — sits under everything |
| 2 | `LANDTYPE_*` | Filled polygon | Ground cover regions |
| 3 | `ROADS` | Polyline | Roads over ground cover |
| 4 | `TREES` | Block instance | Trees over roads |
| 5 | `ROOFS` | Closed polygon + white fill | White fill masks everything below |
| 6 | `SITE_BOUNDARY` | Closed polygon | Optional site perimeter |

The white solid fill on `ROOFS` is the key: it is a filled white polygon drawn between layer 4 and the roof perimeter curve, so trees and roads that pass under a building are visually hidden without being deleted.

---

## 6. User Workflow

### Step 1 — Select area
User navigates the map, draws a bounding box. Approximate area shown in m².

### Step 2 — Import tree blocks (optional)
- Upload 1–3 DXF files, each containing **exploded curves only** (no nested block definitions — a flat set of curve entities representing the tree symbol geometry)
- Keeping imports as raw curves sidesteps complex DXF block parsing; curves are grouped into a block definition at export time
- If skipped, trees are omitted from the output

### Step 3 — Block calibration
For each imported block, the user specifies: **"This block represents a tree with a canopy diameter of ___ meters."**

At placement time, each block is scaled so that its defined diameter matches the detected canopy diameter. For example, if the block was drawn at 1 unit = 1 cm and the user says it represents a 6 m tree, and the detected canopy is 4 m, the block is inserted at scale `4/6 × (unit conversion factor)`.

This calibration is stored per block and persists within the session. If the user imports blocks in future sessions they repeat this step.

### Step 4 — Generate
User clicks **Generate**. Backend runs CV pipeline (~30–60s). Progress shown per module (Roofs ✓, Roads ✓, Trees ✓, Land types ✓).

### Step 5 — Assign land type hatches
For each detected land cluster, user sees a thumbnail and assigns a hatch or discards it. This is the only step requiring active user judgment.

### Step 6 — Style
User adjusts line weights, colors, and hatch parameters per layer. Preview (SVG) updates live.

### Step 7 — Download
User clicks Download → selects DXF and/or 3DM → backend assembles file with correct block insertions and layer structure → returns `.zip`.

---

## 7. Tree Block Placement Logic

### Overlap filtering

Before placement, each detected tree canopy (circle polygon) is checked against all building footprint polygons:

```python
def filter_tree_placements(detections, building_polygons, overlap_threshold=0.30):
    placements = []
    for (x, y, radius_m) in detections:
        canopy = Point(x, y).buffer(radius_m)
        max_overlap = 0.0
        for building in building_polygons:
            intersection_area = canopy.intersection(building).area
            overlap_ratio = intersection_area / canopy.area
            max_overlap = max(max_overlap, overlap_ratio)
        if max_overlap <= overlap_threshold:
            placements.append((x, y, radius_m))
        # else: skip — tree is >30% inside a building, implausible placement
    return placements
```

Trees with ≤30% overlap are placed and visually masked by the roof's white fill. Trees with >30% overlap are suppressed entirely.

### Block assignment and scaling

```python
def assign_blocks(placements, blocks, calibrations):
    result = []
    for i, (x, y, radius_m) in enumerate(placements):
        block_idx = i % len(blocks)
        ref_diameter_m = calibrations[block_idx]
        detected_diameter_m = radius_m * 2
        scale = detected_diameter_m / ref_diameter_m
        result.append({
            "block_idx": block_idx,
            "position": (x, y),
            "scale": scale,
            "rotation": random.uniform(0, 360)  # always randomized
        })
    return result
```

In DXF this becomes an `INSERT` entity per tree. In 3DM this becomes a `rhino3dm.InstanceReference`.

---

## 8. File Export

### 8a. DXF

Library: `ezdxf` (Python, MIT)  
Format: DXF R2018

- Layers created with display color and lineweight
- `ROOFS`: `LWPOLYLINE` (closed) + `HATCH` entity (solid white fill, draw order = "bring to front")
- `ROADS`: `SPLINE` or `LWPOLYLINE`
- `TREES`: `BLOCK` definition (reconstructed from uploaded curves at export time) + `INSERT` entities per placement with `xscale = yscale = scale` and random rotation
- `LANDTYPE_*`: `LWPOLYLINE` + `HATCH` per region
- `CONTOURS`: `LWPOLYLINE`
- All coordinates in meters, local origin at site bounding box center

**Note:** Output is DXF, not binary DWG. DXF R2018 opens natively in AutoCAD, Rhino, Vectorworks, ArchiCAD.

### 8b. 3DM (v2)

Library: `rhino3dm` (Python, v8.17, MIT)

- Layers with display colors
- `InstanceDefinition` per imported tree block
- `InstanceReference` per tree placement with transform matrix encoding position + scale + rotation
- Curves for all polyline/spline geometry
- Hatched areas as planar meshes (generated from polygon + hatch pattern)
- Z=0 for all geometry, 1 unit = 1 meter

---

## 9. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Frontend framework | React + Vite | Standard, fast HMR |
| Map | Mapbox GL JS | Satellite imagery source + selection UX |
| Preview | SVG (client-side from GeoJSON) | Instant style updates without re-inference |
| Backend | FastAPI (Python 3.11) | Async, easy to expose streaming job status |
| Job queue | Celery + Redis | Handles slow CV inference jobs |
| CV inference | PyTorch 2.x | SAM2 + fine-tuned segmentation models |
| Image processing | OpenCV + scikit-image | Mask post-processing, skeletonization |
| Texture features | scikit-image + scikit-learn | Gabor/LBP feature extraction, K-means |
| Vector geometry | Shapely 2.x | Polygon ops, simplification |
| Tree detection | DeepForest | Purpose-built aerial tree crown detection |
| DXF export | ezdxf 1.4 | Full layer + block + hatch support |
| 3DM export | rhino3dm 8.17 | Official McNeel library, no Rhino license needed |
| Geospatial I/O | rasterio + pyproj | Tile stitching, CRS handling |

---

## 10. MVP Scope (v1)

**In scope:**
- Map-based area selection (Mapbox Satellite)
- Layers: Roofs, Roads, Trees, Land types
- Tree block import (1–3 blocks, DXF or 3DM) with calibration step
- Style controls: color, line weight per layer; hatch style per land type; tree block random assignment
- Roof white-fill masking (automatic, always on)
- SVG preview with live style updates
- DXF export with blocks, layers, hatches
- Local coordinate origin (site center = 0,0)
- Processing time target: <60s for a 500×500 m site

**Out of scope for v1:**
- Contours (v2)
- 3DM export (v2)
- Image upload / GeoTIFF input (v2)
- User accounts / saved sessions
- True .dwg binary export

---

## 11. Resolved Decisions

| Decision | Choice |
|---|---|
| Imagery source | Mapbox Satellite — good developer API, sufficient resolution for most sites |
| CV approach | Fine-tuned: SpaceNet 2 + INRIA for buildings; SpaceNet 3 for roads; DeepForest (pretrained) for trees |
| Tree blocks | User-imported (1–3), uploaded as exploded curves, calibrated to real-world size |
| DXF block output | `INSERT` entities referencing a `BLOCK` definition reconstructed at export — not exploded in output |
| Coordinate origin | Local (site bounding box center = 0,0) |
| Road width | Detected from image via medial axis of road mask |
| Land texture | k=4 unsupervised clusters (water / grass / dirt+farmland / paved) |
| Road width | Detected from image (mask width via medial axis) | Avoids manual input for each road |
| Land texture | Unsupervised cluster detection → user assigns hatch per type | Handles farm fields, gravel, grass, hardscape without labeled training data |

---

## 12. Resolved Decisions (continued)

| # | Decision | Choice |
|---|---|---|
| Block rotation | Randomly rotated on placement | More natural distribution; user can adjust post-export in CAD if needed |
| Tree-building overlap | ≤30% overlap → place and let roof fill mask; >30% → suppress | Reflects realistic planting logic; avoids symbols half-buried in buildings |
| Land type cluster count | Fixed k=4 | Water / grass / dirt+farmland / paved — sufficient for most sites, avoids over-segmentation |
| DXF hatch fallback | Yes — "outline only" toggle per land type | Avoids slow hatch rendering on complex polygon boundaries |
| Block import format | Exploded curves only (flat DXF, no nested blocks) | Avoids DXF block parsing complexity; curves are reconstructed into a block at export |
