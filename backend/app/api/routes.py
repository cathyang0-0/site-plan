from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.models.schemas import JobRequest, JobStatus

router = APIRouter()


@router.post("/jobs", response_model=JobStatus)
async def create_job(request: JobRequest):
    """
    Submit a new site plan generation job.
    Returns a job_id to poll for status.
    """
    # TODO: enqueue Celery task
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Poll job status. When complete, geometry GeoJSON is included."""
    # TODO: query Celery result
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/jobs/{job_id}/export")
async def export_job(job_id: str):
    """
    Trigger file export with current style config.
    Returns a .zip download with DXF (and optionally 3DM).
    """
    # TODO: run export pipeline and return file
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/blocks/parse")
async def parse_block(file: UploadFile = File(...)):
    """
    Parse an uploaded DXF file containing exploded curves.
    Returns the curve geometry as GeoJSON for preview.
    """
    # TODO: parse with ezdxf, return curves
    raise HTTPException(status_code=501, detail="Not implemented yet")
