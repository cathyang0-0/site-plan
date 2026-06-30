# Site Plan Drafter — Technical Specification

**Version:** 0.2  
**Status:** Draft  
**Last updated:** 2026-06-29

---

## 1. Overview

A web-based tool that takes an aerial image of a site (by coordinate selection or image upload) and generates a clean, architect-quality context site plan drawing — trees as individually placed user-imported blocks, buildings as clean closed outlines with white fill masking, roads as smooth detected lines, land textures as user-assignable hatches, and optional contours. The user customizes styles and downloads a layered DXF and/or 3DM file.

### Target output quality
Matches the style of a hand-drafted context drawing: clean geometry, no noise, architect-legible symbology. Not a construction document — a context plan.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (React)                     │
│  Map picker → Generate → Calibrate → Style → Download   │
└───────────────────────┬─────────────────────────────────┘
                        │ REST API (JSON)
┌───────────────────────▼─────────────────────────────────┐
│                  Backend (FastAPI / Python)             │
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

### Tree blocks
- The system ships with a **default tree block**: a circle with a "+" (cross) in the center. It is used when the user skips block import.
- Optionally, the user uploads 1–3 DXF files of custom tree symbols (exploded curves only — see section 6, Step 3). Custom blocks replace the default.
- If 2–3 blocks are imported, they are placed randomly across all detected tree positions (uniform random assignment, seeded so the same result is reproducible).

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

**Goal:** Smooth continuous NURBS curves representing road centerlines or edge pairs. Visually connected road segments must be a single curve entity — no broken lines.

**Model:** Fine-tuned DeepLabV3+ on road segmentation. Training data options:
- **SpaceNet 3 Road Network** — road mask labels with good coverage
- **Massachusetts Roads Dataset** — aerial road labels, well-established benchmark

Road width estimated from the mask's medial axis thickness → used to offset centerline into edge pairs if the user selects double-line mode.

**Post-processing:**
1. Binary road mask → skeletonization (`skimage.morphology.skeletonize`)
2. Skeleton → graph (junction nodes, branch edges)
3. Dead-end pruning (remove stubs shorter than 3 m)
4. **Graph path tracing:** traverse each connected path from endpoint to endpoint (or junction to junction), collecting ordered point sequences — this ensures each continuous visual segment becomes one curve, not many segments
5. Per-path cubic B-spline fitting (`scipy.interpolate.splprep`) — produces smooth NURBS, not piecewise polylines
6. Intersection closure: snap curve endpoints within 1 m tolerance
7. Output: list of `shapely.geometry.LineString` (smooth, sampled from spline at high density for GeoJSON preview; exported as SPLINE entity in DXF)

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
7. **Boundary smoothing:** the outlines of each land type region are fitted with cubic B-splines (same approach as roads) — boundaries should read as smooth organic curves, not jagged pixel-step polygons. Topologically shared boundaries between two adjacent clusters are smoothed once and shared, so there are no gaps or overlaps.
8. Backend generates a small thumbnail crop representative of each cluster

**UI presentation:**
- After processing, the style panel shows detected land types as a list
- Each entry has: thumbnail swatch, auto-generated label (e.g. "Type A — dark vegetation", "Type B — light gravel"), and controls:
  - Hatch style (none / parallel lines / cross-hatch / dots / solid)
  - Hatch color, angle, scale
  - **Discard** (hide this region entirely)
- The user assigns the same or different hatches to each cluster

**Layer naming:** `LANDTYPE_A`, `LANDTYPE_B`, etc. (renamed by the user in the style panel)

### 4e. Roof Ridge Lines

**Goal:** Detect the ridge line (peak line of a pitched roof) within each building footprint. Flat roofs produce no ridge line. Output is one or more line segments per pitched roof, sitting on top of the white roof fill.

**Approach:**
Ridge lines are visible in aerial imagery as linear intensity transitions (one slope lit, the other in shadow) running along the dominant axis of the roof. Two complementary methods are used and their results merged:

1. **Medial axis fallback (always runs):** compute the medial axis of the building polygon — for a simple rectangular building this produces a centerline that approximates the ridge. Trim to the interior of the polygon. Works well for gabled roofs on rectangular buildings.

2. **Edge detection refinement (runs if confidence is low):** within each building bounding box, run Canny edge detection on the grayscale aerial image. Filter detected edges to those that are (a) roughly parallel to the building's dominant axis and (b) within the building polygon. Fit a line through the strongest edge cluster. This captures the actual ridge position more accurately than the medial axis on non-rectangular buildings.

**Post-processing:**
- Clip all ridge line candidates to the building polygon (slightly inset by ~0.3 m to avoid touching the perimeter)
- Fit a cubic B-spline to the ridge point sequence for smooth output
- Minimum ridge length: 2 m (suppress on very small buildings)
- Output: one `LineString` per building that has a detectable ridge

**Confidence scoring:** if the two methods disagree by more than 1 m, flag the building as "uncertain" — the ridge line is still output but on a sub-layer `RIDGELINES_UNCERTAIN` that the user can review and delete manually in CAD.

**Layer name:** `RIDGELINES` (certain) / `RIDGELINES_UNCERTAIN` (review needed)

### 4f. Contours (optional, v2)

Source: Copernicus DEM (30m, free globally) or USGS 3DEP (10m, US only).  
Processing: DEM fetch → reproject → Gaussian smooth → contour extraction → clip to site boundary.  
**Layer name:** `CONTOURS`

---

---

## 5. Layer Schema & Drawing Order

Drawing order determines visual stacking. Lower index = drawn first (bottom).

| Order | Layer                  | Geometry                    | Notes                                                   |
| ----- | ---------------------- | --------------------------- | ------------------------------------------------------- |
| 1     | `CONTOURS`             | Spline curve                | Bottommost — sits under everything                      |
| 2     | `LANDTYPE_*`           | Filled region + spline edge | Ground cover regions; boundaries are smooth curves      |
| 3     | `ROADS`                | Spline curve                | Continuous curves; connected segments = one entity      |
| 4     | `TREES`                | Block instance              | Trees over roads                                        |
| 5     | `ROOFS`                | Closed polygon + white fill | White fill masks everything below                       |
| 6     | `RIDGELINES`           | Spline curve                | Detected pitch ridge lines; drawn over roof fill        |
| 6b    | `RIDGELINES_UNCERTAIN` | Spline curve                | Low-confidence ridges for user review; deletable in CAD |
| 7     | `SITE_BOUNDARY`        | Closed polygon              | Optional site perimeter                                 |

The white solid fill on `ROOFS` is the key: it is a filled white polygon drawn between layer 4 and the roof perimeter curve, so trees and roads that pass under a building are visually hidden without being deleted.

---

## 6. User Workflow

### Step 1 — Select area
User navigates the map, draws a bounding box, or set paper size and scale, then move box. Approximate area shown in m², scale bar on the side.

### Step 2 — Generate
User clicks **Generate**. Backend runs CV pipeline (~30–60s). Progress shown per module (Roofs ✓, Roads ✓, Trees ✓, Land types ✓).

### Step 3 — Import tree blocks (optional)
- Upload 1–3 DXF files, each containing **exploded curves only** (no nested block definitions — a flat set of curve entities representing the tree symbol geometry)
- Keeping imports as raw curves sidesteps complex DXF block parsing; curves are grouped into a block definition at export time
- If skipped, trees are set to default blocks from the output

### Step 3.5 — Block calibration (visual)

For each imported block, a calibration canvas is shown with two overlaid elements:
- The **imported block geometry** (drawn in black)
- A **reference circle** with a known real-world diameter (default: 6 m, adjustable)

The user aligns them by either:
- Dragging and scaling the **reference circle** to match the outer boundary of the block drawing, or
- Dragging and scaling the **block** to match the circle

Once confirmed, the system records: `reference_diameter_m / block_drawn_diameter_px = scale_factor`. At placement time, this factor is applied to match each block instance to the detected canopy size.

The reference circle diameter is editable (e.g., the user may know their symbol represents a 4 m tree, not 6 m) — changing it updates the calibration instantly.

Calibration persists within the session. Repeated across sessions if new blocks are uploaded.

### Step 4 — Assign land type hatches
For each detected land cluster, user sees a color-coded thumbnail and assigns a hatch or discards it. User has the option to make hatch a gradient (offsets a distance from boundary and assigns the same hatch with lower density closer to the boundary, a slider adjusts gradient level - number of offset steps, 0 being non gradient). This is the only step requiring active user judgment.

### Step 5 — Style
User sees a thumbnail preview and adjusts line weights, colors, opacity, and hatch parameters per layer. Preview (SVG) updates live. (can be a simplified standard drawing that has all the components, doesn't ave to match the actual site plan generated) 

### Step 6 — Download
User clicks Download → selects DXF and/or 3DM → backend assembles file with correct block insertions and layer structure → returns `.zip`.

---

## 7. Tree Block Placement Logic

### Overlap filtering

Two overlap rules are enforced:
1. A tree whose canopy overlaps a **building** by >30% is suppressed (implausible planting location)
2. A tree whose canopy overlaps an **already-accepted tree** by >30% is suppressed (prevents dense clumping)

Both checks use a **spatial index** (`shapely.STRtree`) to avoid O(n²) comparisons. The index narrows candidates to only those whose bounding boxes intersect, so the precise overlap check runs only on a small local set — O(n log n) overall.

```python
from shapely.geometry import Point
from shapely.strtree import STRtree

def filter_tree_placements(detections, building_polygons, overlap_threshold=0.30):
    # Build spatial index over buildings
    building_index = STRtree(building_polygons)

    accepted_canopies = []
    accepted_index = None  # rebuilt incrementally

    for (x, y, radius_m) in detections:
        canopy = Point(x, y).buffer(radius_m)

        # Rule 1: building overlap
        candidates = building_index.query(canopy)
        too_close_to_building = any(
            canopy.intersection(building_polygons[i]).area / canopy.area > overlap_threshold
            for i in candidates
        )
        if too_close_to_building:
            continue

        # Rule 2: tree-tree overlap (check against already-accepted canopies)
        if accepted_canopies:
            accepted_index = STRtree(accepted_canopies)
            candidates = accepted_index.query(canopy)
            too_close_to_tree = any(
                canopy.intersection(accepted_canopies[i]).area / canopy.area > overlap_threshold
                for i in candidates
            )
            if too_close_to_tree:
                continue

        accepted_canopies.append(canopy)

    # Reconstruct detections list from accepted canopies
    accepted = []
    for canopy in accepted_canopies:
        c = canopy.centroid
        radius_m = (canopy.area / 3.14159) ** 0.5
        accepted.append((c.x, c.y, radius_m))
    return accepted
```

**Performance note:** rebuilding `STRtree` on every iteration is slightly wasteful but correct. For sites with >500 trees, batch the index rebuild every 50 accepted trees instead of every insertion.

Trees with ≤30% overlap on both rules are placed; those exceeding either threshold are suppressed.

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

## 8. Gradient Hatch Implementation

Gradient hatches simulate a density falloff from a region's boundary inward (or outward). They are implemented as a series of progressively spaced parallel hatch layers, not a single DXF gradient entity (which has poor CAD app support).

**Algorithm:**

```
Given: polygon P, hatch spacing S, gradient steps N, direction (inward/outward)

1. Generate N inward offsets of P at distances: d_i = i * (polygon_inradius / N)
   - d_0 = boundary (outermost, densest hatching)
   - d_N = center (innermost, least dense / no hatch)

2. For each offset ring i:
   spacing_i = S * (1 + i * gradient_factor)
   Draw parallel hatch lines at spacing_i, clipped to the annular region
   between offset ring i and ring i+1

3. Combine all hatch line sets into a single DXF HATCH entity group
```

**User controls:**
- **Gradient steps** (slider, 0–8): 0 = uniform hatch (no gradient), higher = more offset rings, smoother gradient
- **Direction**: inward (dense at boundary, fades toward center) or outward (dense at center, fades to boundary)
- **Base spacing**: the spacing at the densest point

**DXF export note:** each gradient step is a separate `HATCH` entity with a different pattern scale, stacked within the same layer. CAD apps render them correctly; the file remains manageable since hatches are clipped to their annular region.

---

## 9. File Export

### 8a. DXF

Library: `ezdxf` (Python, MIT)  
Format: DXF R2018

- Layers created with display color and lineweight
- `ROOFS`: `LWPOLYLINE` (closed) + `HATCH` entity (solid white fill, draw order = "bring to front")
- `RIDGELINES` / `RIDGELINES_UNCERTAIN`: `SPLINE` — one entity per detected ridge
- `ROADS`: `SPLINE` — one entity per continuous road path; never split into segments at junctions
- `TREES`: `BLOCK` definition (reconstructed from uploaded curves at export time) + `INSERT` entities per placement with `xscale = yscale = scale` and random rotation
- `LANDTYPE_*`: `SPLINE` boundary curves + `HATCH` entities per region (multiple `HATCH` entities for gradient mode, one per gradient step)
- `CONTOURS`: `SPLINE`
- All coordinates in meters, local origin at site bounding box center

**Curve export note:** all curves are exported as DXF `SPLINE` entities (degree 3, with computed knot vectors) rather than `LWPOLYLINE`. This preserves smoothness in CAD apps and allows the user to edit control points after import.

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
| Tree blocks | Default = built-in circle with a "+" in the center. User can optionally replace with 1–3 custom blocks (uploaded as exploded curves, visually calibrated). |
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
