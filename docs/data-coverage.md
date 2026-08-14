# Data coverage — what the plan includes, and what it deliberately doesn't

Every object in an exported plan comes from one of the sources below. Anything
not listed under "Included" is absent from the output even when it exists in
the source data or is visible in the aerial. This page is the honest inventory
of both sides, so a missing object can be classified at a glance: *not
included by design* vs. *bug*.

## Included layers

| Plan layer | Object | Source | Notes |
|---|---|---|---|
| ROOFS | building footprints | Overture `building` | includes buildings on piers (they're `building` features too) |
| ROADS | road corridors | Overture `segment` (subtype `road`) + CV width | rail and non-road segments are filtered out |
| TREES | tree symbols | DeepForest detection on aerial imagery | detection-led; not the OSM/Overture tree points |
| LANDTYPE_* | water / vegetation / bare / paved | k-means or SegFormer on aerial + Overture `water` | Overture water (incl. ocean, lakes, buffered river centerlines) replaces and carves detected water |
| CONTOURS | topographic contours | USGS 3DEP elevation grid | interval user-chosen; clipped under buildings and infrastructure |
| INFRASTRUCTURE | piers, bridges, breakwaters, walls/fences/kerbs, parking aprons, runways, ski lifts | Overture `infrastructure` | per-CLASS treatment (user-reviewed): structure 0.30 / bridges+airfield 0.18 / breakwater 0.13 / micro 0.10 / lifts 0.09 / walls 0.08 / kerbs hairline / fences 0.04; centerline classes (bridge lines, breakwater, wall, fence, runway/taxiway) are OFFSET into real-width strips, never fat strokes; micro structures overlapping a building are dropped; heavy groups clip hatches/contours/roads, thin barriers don't; airport admin boundaries excluded |
| SCALEBAR / NOTES | annotations | generated | scale bar + ODbL attribution |

## Known NOT included (exists in source data, never fetched)

Verified live against the Santa Barbara harbor bbox (2026-08-13): all of the
following are present in Overture's `base`/`infrastructure`/`land_use` themes
for that site and are absent from our output.

- **Parking lots as land_use zones** — `land_use` polygons (the paved
  *surface* may appear via CV land cover, and parking aprons mapped as
  `infrastructure` shapes ARE included; zoning polygons are not).
- **Land-use areas** — parks, sports pitches, marinas, residential/commercial
  zoning polygons (`land_use` theme).
- **Overture's own land cover** (beach/scrub/rock polygons, `land` theme) —
  we detect ground cover from imagery instead.
- **Railways** — filtered out of the road fetch by design.
- **Street furniture** (hydrants, lamps, benches, signs) — `infrastructure`
  POINT features; deliberately excluded from the infrastructure stage.
- **Service networks** — power lines/pylons, communication lines, utility
  pipelines, manholes, waste bins, emergency fixtures (`EXCLUDED_SUBTYPES`
  in infrastructure.py — user call: utility clutter, not built form).
- **Ridge lines / roof geometry beyond the outline** — no data source; was
  removed from the API contract for honesty.

## Adding one of these later

All the Overture items above use the exact fetch pattern of
`app/pipeline/water.py` (bbox query → cache → polygons/buffered lines), so
each is a small, self-contained stage: fetch, convert to pixels, draw on its
own DXF layer slotted into the Z-staircase. Piers/bridges/walls were added exactly this way (`infrastructure.py`,
2026-08-13).
