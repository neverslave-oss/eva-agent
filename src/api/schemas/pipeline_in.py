"""pipeline_in — PipelineIn Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

from .pipeline_stage import PipelineStage


class PipelineIn(BaseModel):
    pipeline: str = "default"
    stages: list[PipelineStage]
