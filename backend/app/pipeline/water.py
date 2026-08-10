"""
Water footprints from Overture Maps (primary water source for
georeferenced sites; the segmentation model + the blue-water reclamation
in landtypes_seg.py is the fallback for non-georeferenced input).

"""
import shapely
from shapely.geometry import Polygon
from app.pipeline.footprints import fetch_with_retry

ATTRIBUTION = "Water © OpenStreetMap contributors, Overture Maps Foundation (ODbL)"
# Rivers/streams arrive as centerLINES (no width). Buffer each into a fillable
# strip this many meters wide. Overture rarely carries a reliable width for
# these, so we use one default (a small stream); a wide river reads fine at this.
DEFAULT_RIVER_WIDTH_M = 5.0
_M_PER_DEG_LAT = 111_320   # meters per degree latitude (near-constant on Earth)

def fetch_water_footprints(west: float, south: float, east: float, north: float) -> list[Polygon]:
    """
    Fetch water footprint polygons for a lon/lat bounding box, return in EPSG:4326 lon/lat coordinates.
    """
    from app.pipeline import overture_cache
    bbox = (west, south, east, north)
    cached = overture_cache.get("water", bbox)
    if cached is not None:
        return cached

    def _fetch():
        from overturemaps import core

        reader = core.record_batch_reader("water", (west, south, east, north))
        half_width_deg = (DEFAULT_RIVER_WIDTH_M / 2) / _M_PER_DEG_LAT
        polygons: list[Polygon] = []
        for batch in reader:
            if batch.num_rows == 0:
                continue
            for wkb in batch.column("geometry").to_pylist():
                if wkb is None:
                    continue
                geom = shapely.from_wkb(wkb)
                polygons.extend(_water_polygons_from_geom(geom, half_width_deg))
        return polygons

    polygons = fetch_with_retry(_fetch, "water footprints fetch")
    overture_cache.put("water", bbox, polygons)
    return polygons


def _water_polygons_from_geom(geom, half_width_deg: float) -> list[Polygon]:
    """Turn one Overture water geometry into fillable polygons.

    Areal water (Polygon / MultiPolygon) passes straight through. A river or
    stream arrives as a CENTERLINE (LineString / MultiLineString) with no width,
    so it is buffered into a strip `half_width_deg` wide on each side; that
    buffer can yield a Polygon or (for a self-touching line) a MultiPolygon, so
    both are unpacked. Anything else (e.g. a stray Point) yields nothing.
    """
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type in ("LineString", "MultiLineString"):
        buffered = geom.buffer(half_width_deg)
        if buffered.geom_type == "Polygon":
            return [buffered]
        if buffered.geom_type == "MultiPolygon":
            return list(buffered.geoms)
    return []