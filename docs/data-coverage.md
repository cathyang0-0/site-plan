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
| CONTOURS | topographic contours | USGS 3DEP elevation grid | interval user-chosen; clipped under buildings |
| SCALEBAR / NOTES | annotations | generated | scale bar + ODbL attribution |

## Known NOT included (exists in source data, never fetched)

Verified live against the Santa Barbara harbor bbox (2026-08-13): all of the
following are present in Overture's `base`/`infrastructure`/`land_use` themes
for that site and are absent from our output.

- **Piers / wharves / docks** — Overture `infrastructure`, class `pier/pier`
  (Stearns Wharf's deck outline exists there as a named polygon; the marina
  piers too). This is why a plan can show buildings and a road "floating" on
  open water: their supporting pier deck is a layer we don't draw.
- **Breakwaters / seawalls** — `infrastructure`, `water/breakwater`.
- **Bridges** — `infrastructure`, `bridge/bridge` (decks and edges).
- **Walls / fences / barriers** — `infrastructure`, `barrier/*`.
- **Parking lots** — `infrastructure` `transit/parking` and `land_use`
  polygons (paved *surface* may still appear via CV land cover, but not the
  lot boundary as an object).
- **Land-use areas** — parks, sports pitches, marinas, residential/commercial
  zoning polygons (`land_use` theme).
- **Overture's own land cover** (beach/scrub/rock polygons, `land` theme) —
  we detect ground cover from imagery instead.
- **Railways** — filtered out of the road fetch by design.
- **Power lines**, **street furniture** (hydrants, lamps, benches, signs) —
  `infrastructure` points/lines.
- **Ridge lines / roof geometry beyond the outline** — no data source; was
  removed from the API contract for honesty.

## Adding one of these later

All the Overture items above use the exact fetch pattern of
`app/pipeline/water.py` (bbox query → cache → polygons/buffered lines), so
each is a small, self-contained stage: fetch, convert to pixels, draw on its
own DXF layer slotted into the Z-staircase. Piers are the most plan-relevant
candidate (coastal sites read wrong without them).
