"""pipeline_job_status — PipelineJobStatus Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

class PipelineJobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | complete | error | cancelled
    stage: Optional[str] = None
    results: dict = {}
    error: Optional[str] = None
    pipeline: str = "default"
    created_at: float = 0.0
    completed_at: Optional[float] = None
