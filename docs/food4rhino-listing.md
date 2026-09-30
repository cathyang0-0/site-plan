# Food4Rhino listing copy (draft)

Paste-ready text for the Food4Rhino app page. Screenshots to prepare are
listed at the bottom.

---

**App name:** Site Plan Drafter

**One-liner:** Draw a box on a map — get an architect-style site plan in
your Rhino document.

**Description:**

Site Plan Drafter generates clean, layered context plans from aerial
imagery and open map data, imported into your active document at
real-world scale.

Type `SitePlan`, search an address, drag a rectangle on the aerial map,
and a few minutes later you have:

- **Building roofs** from Overture Maps footprints, white-filled and
  outlined with architectural line weights
- **Roads** as merged, filleted pavement corridors — per-class widths you
  can override, refined by computer-vision measurement of the actual
  pavement
- **Trees** detected from the imagery (DeepForest), placed as editable,
  origin-centered blocks — with a live preview after detection where size
  and variance sliders re-render instantly, and what you see is exactly
  what imports
- **Land-cover hatches** (vegetation, water, sand, paved) with real hatch
  patterns, one layer per type
- **Topographic contours** (USGS elevation) at your interval, hairline
  gray, bottom of the draw order
- **Piers, bridges, breakwaters** and other infrastructure

Everything arrives on proper layers, in your document's units, with
editable geometry — ready to underlay a site model or drawing set. Results
are cached, so restyling (line weights, colors, tree sizes) re-exports in
seconds without re-running detection.

**Requirements:**

- Rhino 8 (tested on macOS; Windows testers welcome — please report!)
- The free open-source backend running locally (one-time install, see
  below) — the plugin starts it automatically
- Internet connection (imagery and map data are fetched per job; US
  coverage for aerial imagery via USGS)

**Backend install (one time):**

Install [uv](https://docs.astral.sh/uv/), then in a terminal:

    uv tool install "siteplan-backend @ git+https://github.com/cathyang0-0/site-plan#subdirectory=backend"

Heads-up: the computer-vision stack is a multi-GB download.

**Links:**

- Source & docs: https://github.com/cathyang0-0/site-plan
- License: MIT (open source)
- Contact: siteplan.drafter@outlook.com

**Data attribution:** Imagery/elevation: USGS (public domain). Buildings,
roads, water, infrastructure: © OpenStreetMap contributors, Overture Maps
Foundation (ODbL) — exported DXFs carry the attribution automatically.
Geocoding: OSM Nominatim.

---

## Screenshots to prepare (she shoots, ~15 min)

1. The dialog with the map picker, a drawn bbox over a real site
2. The tree preview page mid-adjustment (sliders + circles over aerial)
3. A finished plan in Rhino — top view, layers panel visible
4. Close-up of the DXF linework quality (roads + roofs + hatches)

## Fields the form will ask for

- Category: suggest "Plug-in" / Rhino 8 / Mac (+Windows untested)
- Version: 0.1.0 — upload `rhino/build/rh8/siteplan-0.1.0+*-rh8-any.yak`
  (or publish via `yak push` first and link the package)
