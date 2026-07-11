"""
Building footprints from Overture Maps (primary building source for
georeferenced sites; the SAM2/U-Net CV path in buildings.py is the
fallback for non-georeferenced input and very recent construction).

Overture Maps Foundation publishes a unified open map dataset (buildings
theme merges OpenStreetMap, Microsoft ML footprints, Google Open
Buildings, and USGS lidar) as cloud-hosted GeoParquet, re-released
regularly. The `overturemaps` package queries it with bbox pushdown:
only the row groups whose bbox metadata intersect the query are read, so
a site-sized query transfers a few hundred KB and takes ~1 second — no
local mirror of the dataset is needed.

License: ODbL. A generated site plan is an ODbL "Produced Work" — the
caller must display attribution (see ATTRIBUTION), but the drawing itself
is not share-alike encumbered.
"""
import shapely
from shapely.geometry import Polygon

ATTRIBUTION = "Building footprints © OpenStreetMap contributors, Overture Maps Foundation (ODbL)"


def fetch_building_footprints(
    west: float, south: float, east: float, north: float
) -> list[Polygon]:
    """
    Fetch building footprint polygons for a lon/lat bounding box.

    Returns polygons in lon/lat (WGS84) coordinates; convert with
    footprints_to_pixels for pipeline use. Multipolygons are split into
    their parts. Requires network access.
    """
    from overturemaps import core

    reader = core.record_batch_reader("building", (west, south, east, north))
    polygons: list[Polygon] = []
    for batch in reader:
        if batch.num_rows == 0:
            continue
        for wkb in batch.column("geometry").to_pylist():
            if wkb is None:
                continue
            geom = shapely.from_wkb(wkb)
            if geom.geom_type == "Polygon":
                polygons.append(geom)
            elif geom.geom_type == "MultiPolygon":
                polygons.extend(geom.geoms)
    return polygons


def footprints_to_pixels(
    geo_polygons: list[Polygon],
    west: float, south: float, east: float, north: float,
    img_w: int, img_h: int,
    clip: bool = True,
) -> list[Polygon]:
    """
    Convert lon/lat footprint polygons to pixel coordinates of an image
    that spans exactly (west, south, east, north), y-down.

    Uses a linear lon/lat mapping: over a site-sized extent the difference
    from true Web Mercator is far below one pixel. With clip=True (default),
    footprints straddling the bbox edge are clipped to the image, and any
    entirely outside are dropped.
    """
    from shapely.geometry import box
    from shapely.ops import transform

    frame = box(0, 0, img_w, img_h)
    sx = img_w / (east - west)
    sy = img_h / (north - south)

    def to_px(x, y):
        return ((x - west) * sx, (north - y) * sy)

    result = []
    for poly in geo_polygons:
        px_poly = transform(lambda x, y: to_px(x, y), poly)
        if clip:
            px_poly = px_poly.intersection(frame)
        if px_poly.is_empty:
            continue
        if px_poly.geom_type == "Polygon":
            result.append(px_poly)
        elif px_poly.geom_type == "MultiPolygon":
            result.extend(px_poly.geoms)
    return result
