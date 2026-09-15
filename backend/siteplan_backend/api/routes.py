"""
API routes — thin HTTP layer over app.api.jobs (the job manager).

The submit → poll → export shape exists because detection takes minutes:
POST /jobs returns a ticket immediately, GET /jobs/{id} is the ticket check,
and export re-renders the cached geometry with new styling in seconds.
"""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from siteplan_backend.api import jobs
from siteplan_backend.models.schemas import JobRequest, JobStatus, StyleConfig
from siteplan_backend.pipeline.imagery import usgs_export_size_px

router = APIRouter()


@router.post("/jobs", response_model=JobStatus)
async def create_job(request: JobRequest):
    """Submit a site-plan generation job. Returns immediately with a job_id
    to poll; detection runs in the background (minutes)."""
    bb = request.bbox
    try:  # fail fast on a bbox the imagery service can't render
        usgs_export_size_px(bb.west, bb.south, bb.east, bb.north,
                            request.options.scale_m_per_px)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    job = jobs.create_job(request)
    return job.to_status()


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Poll job status; `progress` reports per-stage state while running."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return job.to_status()


@router.post("/jobs/{job_id}/export")
async def export_job(job_id: str, style: Optional[StyleConfig] = None):
    """Download the DXF. With a style body, re-renders the cached geometry
    using it (seconds — no re-detection); without, returns the default render."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    if job.status != "complete":
        raise HTTPException(status_code=409,
                            detail=f"job is {job.status}, not complete")
    if style is not None:
        path = jobs.export_dxf_for(job, style)
    else:
        path = job.dxf_path
    if path is None or not path.exists():
        raise HTTPException(status_code=500, detail="export file missing")
    return FileResponse(path, media_type="application/dxf",
                        filename="site-plan.dxf")


def _completed_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    if job.status != "complete":
        raise HTTPException(status_code=409,
                            detail=f"job is {job.status}, not complete")
    return job


@router.get("/jobs/{job_id}/image")
async def job_image(job_id: str):
    """The fetched aerial for this job — backdrop of the tree preview."""
    job = _completed_job(job_id)
    path = job.dir / "site.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="site image not saved")
    return FileResponse(path, media_type="image/png")


@router.get("/jobs/{job_id}/trees")
async def job_trees(job_id: str):
    """Neutral-size tree placements + the constants the preview's live
    resize math needs (see jobs.tree_preview_payload)."""
    return jobs.tree_preview_payload(_completed_job(job_id))


@router.get("/map")
async def map_page():
    """The bbox-picker map page the Rhino dialog embeds. Backend-served for
    the same reason as /preview below — and so the .rhp plugin only has to
    ship Python files (HTML rides in the backend wheel as package data)."""
    page = Path(__file__).resolve().parents[1] / "static" / "siteplan_map.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="map page missing")
    return FileResponse(page, media_type="text/html")


@router.get("/jobs/{job_id}/preview")
async def job_preview(job_id: str):
    """The tree-preview page itself. Served from the backend (not file://)
    so its image/trees fetches are same-origin — no CORS involved."""
    _completed_job(job_id)
    # Package data, not a repo path — an installed wheel has no rhino/ dir.
    page = Path(__file__).resolve().parents[1] / "static" / "siteplan_preview.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="preview page missing")
    return FileResponse(page, media_type="text/html")


@router.post("/blocks/parse")
async def parse_block(file: UploadFile = File(...)):
    """Parse an uploaded DXF of custom tree symbols. Not implemented yet —
    the built-in tree block is used for all placements."""
    raise HTTPException(status_code=501,
                        detail="custom tree blocks not implemented yet")
