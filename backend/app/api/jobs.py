"""
In-memory job manager — the engine behind the API routes.

Deliberately simple (solo/local use): jobs run on a daemon thread in this
process and live in a dict; no Celery/Redis/broker. The web app and the Rhino
client both speak to this through routes.py: submit → poll → export.

Design notes:
- The runner mirrors scripts/poc.py's stage order (land types BEFORE the tree
  filter so water can suppress lake trees; Overture water replaces detected
  water). poc.py's tree/rasterize helpers are imported rather than duplicated.
- Detection results (buildings/roads/trees/land regions) are kept on the Job
  after completion so /export can re-render a DXF with new styling in seconds
  without re-running the ~6 min detection ("restyle without redetect").
- Heavy imports stay function-local (module-scope torch segfaults the test
  suite — see docs/HANDOFF.md landmines).
"""
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from typing import Optional

from app.models.schemas import JobRequest, JobStatus, StyleConfig

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

_JOBS: dict[str, "Job"] = {}
_JOBS_LOCK = threading.Lock()


class Job:
    def __init__(self, request: JobRequest):
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.status = "queued"          # queued | running | complete | failed
        self.progress: dict = {}        # stage -> "running" | "done" | "skipped"
        self.error: Optional[str] = None
        self.warnings: list[str] = []
        self.dir = Path(tempfile.mkdtemp(prefix=f"siteplan_{self.id}_"))
        self.dxf_path: Optional[Path] = None
        # Detection results cached for restyle-without-redetect (see /export).
        self.geometry: Optional[dict] = None

    def to_status(self) -> JobStatus:
        return JobStatus(job_id=self.id, status=self.status,
                         progress=self.progress or None, error=self.error,
                         warnings=self.warnings or None)


def create_job(request: JobRequest) -> Job:
    job = Job(request)
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    t = threading.Thread(target=_run_job, args=(job,), daemon=True)
    t.start()
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def _run_job(job: Job) -> None:
    try:
        job.status = "running"
        _run_pipeline(job)
        job.status = "complete"
    except Exception as exc:  # surface the real error to the poller
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()


def _poc():
    """Import scripts/poc.py for its shared helpers (tree detection wrapper,
    rasterizers, default tree block). TODO: promote these into app/pipeline so
    the API doesn't reach into scripts/."""
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    import poc
    return poc


def _overture_stage(job: Job, name: str, default, fn):
    """Run an optional open-data stage; on failure, warn and continue with
    `default`. Rationale: Overture has real multi-hour slow spells, and one
    dead layer must not throw away minutes of finished detection — a partial
    plan with a visible warning beats a failed job."""
    job.progress[name] = "running"
    try:
        result = fn()
        job.progress[name] = "done"
        return result
    except Exception as exc:
        job.progress[name] = "failed"
        job.warnings.append(
            f"{name}: {type(exc).__name__}: {exc} — plan generated without this layer")
        return default


def _run_pipeline(job: Job) -> None:
    req = job.request
    bb = req.bbox
    opts = req.options
    layers = set(req.layers)

    def stage(name, state):
        job.progress[name] = state

    # --- Imagery (always needed) ---
    stage("imagery", "running")
    from app.pipeline.imagery import fetch_aerial_usgs
    image, scale = fetch_aerial_usgs(bb.west, bb.south, bb.east, bb.north,
                                     opts.scale_m_per_px)
    img_w, img_h = image.size
    stage("imagery", "done")

    poc = _poc()
    buildings, roads, water_polys, land_types, tree_placements = [], [], [], [], []
    detections = []

    # All Overture stages run BEFORE the ~5 min tree detection so network
    # trouble surfaces in the first minutes, and each degrades to "warn +
    # continue" instead of failing the job (see _overture_stage).

    # --- Overture roads ---
    if "roads" in layers and opts.overture_roads:
        def _roads():
            from app.pipeline.roads import build_roads
            return build_roads(image, bb.west, bb.south, bb.east, bb.north, scale,
                               class_widths=opts.road_class_widths)
        roads = _overture_stage(job, "roads", [], _roads)
    else:
        stage("roads", "skipped")

    # --- Overture buildings ---
    if "roofs" in layers and opts.overture_buildings:
        def _buildings():
            from app.pipeline.footprints import (
                fetch_building_footprints, footprints_to_pixels)
            geo = fetch_building_footprints(bb.west, bb.south, bb.east, bb.north)
            return footprints_to_pixels(geo, bb.west, bb.south, bb.east, bb.north,
                                        img_w, img_h)
        buildings = _overture_stage(job, "buildings", [], _buildings)
    else:
        stage("buildings", "skipped")

    # --- Overture water (feeds both the land layer and tree suppression) ---
    if opts.overture_water and ("land_types" in layers or "trees" in layers):
        def _water():
            from shapely.ops import unary_union
            from app.pipeline.water import fetch_water_footprints
            from app.pipeline.footprints import footprints_to_pixels
            geo = fetch_water_footprints(
                bb.west, bb.south, bb.east, bb.north,
                **({"river_width_m": opts.river_width_m}
                   if opts.river_width_m is not None else {}))
            water_px = footprints_to_pixels(geo, bb.west, bb.south, bb.east,
                                            bb.north, img_w, img_h)
            if not water_px:
                return []
            merged = unary_union(water_px)
            return (list(merged.geoms)
                    if merged.geom_type == "MultiPolygon" else [merged])
        water_polys = _overture_stage(job, "water", [], _water)
    else:
        stage("water", "skipped")

    # --- Overture infrastructure (pier decks, bridges, breakwaters, walls) ---
    infrastructure = {"groups": {}}
    if "infrastructure" in layers and opts.overture_infrastructure:
        def _infra():
            from app.pipeline.infrastructure import (
                fetch_infrastructure, infrastructure_to_pixels)
            geo = fetch_infrastructure(bb.west, bb.south, bb.east, bb.north)
            return infrastructure_to_pixels(geo, bb.west, bb.south, bb.east,
                                            bb.north, img_w, img_h)
        infrastructure = _overture_stage(
            job, "infrastructure", {"groups": {}}, _infra)
    else:
        stage("infrastructure", "skipped")

    # --- Contours (USGS elevation; interval from the style config) ---
    contours = []
    if "contours" in layers and req.style.contours.visible:
        def _contours():
            from app.pipeline.contours import build_contours
            return build_contours(bb.west, bb.south, bb.east, bb.north,
                                  req.style.contours.interval_m, img_w, img_h)
        contours = _overture_stage(job, "contours", [], _contours)
    else:
        stage("contours", "skipped")

    # --- Tree detection (raw) — the slow stage, after all network fetches ---
    if "trees" in layers:
        stage("trees", "running")
        detections = poc.real_tree_detections(
            image, scale,
            stand_fill=opts.stand_fill,
            crown_size_scale=opts.crown_size_scale,
            **({"size_variance": opts.size_variance}
               if opts.size_variance is not None else {}),
        )

    # --- Land types (before the tree filter: water suppresses lake trees) ---
    if "land_types" in layers:
        stage("land_types", "running")
        from app.pipeline.landtypes import default_hatch_style
        b_mask = poc._rasterize_polygons(buildings, img_w, img_h)
        r_mask = poc._rasterize_roads(roads, img_w, img_h)
        if opts.land_types_engine == "segmodel":
            from app.pipeline.landtypes_seg import detect_land_types_seg
            kwargs = ({"paved_min_conf": opts.paved_min_conf}
                      if opts.paved_min_conf is not None else {})
            detected = detect_land_types_seg(image, b_mask, r_mask, **kwargs)
        else:
            from app.pipeline.landtypes import detect_land_types
            detected = detect_land_types(image, b_mask, r_mask)
        land_types = [{"label": d["label"], "polygons": d["polygons"],
                       "style": default_hatch_style(d["label"])}
                      for d in detected]
        # Overture water is authoritative: replaces detected water AND is
        # carved out of every other cover (no land hatch over the water hatch).
        from app.pipeline.landtypes import apply_authoritative_water
        land_types = apply_authoritative_water(land_types, water_polys)
        stage("land_types", "done")
    else:
        stage("land_types", "skipped")

    # --- Tree filtering (needs buildings + water) ---
    if "trees" in layers:
        from app.pipeline.trees import filter_placements, suppress_over_water
        accepted = filter_placements(detections, building_polygons=buildings)
        suppress_water = water_polys or [
            g for lt in land_types if lt["label"] == "water"
            for g in (lt["polygons"].geoms if hasattr(lt["polygons"], "geoms")
                      else [lt["polygons"]])
        ]
        if suppress_water:
            accepted = suppress_over_water(accepted, suppress_water)
        tree_placements = poc.detections_to_placements(accepted)
        stage("trees", "done")
    else:
        stage("trees", "skipped")

    # Data-license credits (ODbL requires attribution for Overture layers).
    attributions = []
    if roads:
        from app.pipeline.roads import ATTRIBUTION as ROAD_ATTR
        attributions.append(ROAD_ATTR)
    if buildings:
        from app.pipeline.footprints import ATTRIBUTION as BLDG_ATTR
        attributions.append(BLDG_ATTR)
    if water_polys:
        from app.pipeline.water import ATTRIBUTION as WATER_ATTR
        attributions.append(WATER_ATTR)
    if infrastructure.get("groups"):
        from app.pipeline.infrastructure import ATTRIBUTION as INFRA_ATTR
        attributions.append(INFRA_ATTR)

    # Cache results for restyle-without-redetect, then export.
    job.geometry = {
        "buildings": buildings, "roads": roads,
        "tree_placements": tree_placements,
        "tree_blocks": [poc.default_tree_block()],
        "land_types": land_types, "contours": contours,
        "infrastructure": infrastructure, "scale": scale,
        "origin_px": (img_w // 2, img_h // 2),
        "attribution": "  ·  ".join(attributions) if attributions else None,
    }
    stage("export", "running")
    job.dxf_path = export_dxf_for(job, req.style)
    stage("export", "done")


def export_dxf_for(job: Job, style: Optional[StyleConfig]) -> Path:
    """Render the job's cached geometry to DXF with the given style. Fast
    (seconds) — this is what lets clients restyle without re-detecting."""
    if job.geometry is None:
        raise RuntimeError("job has no geometry to export")
    from app.export.dxf import export_dxf
    sc = style or StyleConfig()
    g = job.geometry
    style_dict = {
        "roofs": {"color": sc.roofs.color, "line_weight_mm": sc.roofs.line_weight_mm},
        "roads": {"color": sc.roads.color, "line_weight_mm": sc.roads.line_weight_mm},
        "trees": {"color": sc.trees.color, "line_weight_mm": sc.trees.line_weight_mm},
        "land_types": [{"color": lt.color, "line_weight_mm": lt.line_weight_mm}
                       for lt in sc.land_types] or
                      [{"color": "#aaaaaa", "line_weight_mm": 0.05}],
        "contours": {"color": sc.contours.color,
                     "line_weight_mm": sc.contours.line_weight_mm},
        "infrastructure": {"color": sc.infrastructure.color,
                           "line_weight_mm": sc.infrastructure.line_weight_mm},
    }
    out = job.dir / "plan.dxf"
    export_dxf(
        output_path=out,
        buildings=g["buildings"] if sc.roofs.visible else [],
        roads=g["roads"] if sc.roads.visible else [],
        tree_placements=g["tree_placements"] if sc.trees.visible else [],
        tree_block_curves=g["tree_blocks"],
        land_types=g["land_types"],
        contours=(g.get("contours") or None) if sc.contours.visible else None,
        infrastructure=(g.get("infrastructure") or None)
                       if sc.infrastructure.visible else None,
        style=style_dict,
        units=sc.units,
        scale_m_per_px=g["scale"],
        origin_px=g["origin_px"],
        attribution=g["attribution"],
    )
    return out
