"""pipeline_stage — PipelineStage Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

class PipelineStage(BaseModel):
    name: str
    role: str = "custom"
    brief: str
    task: str = "Begin your work."
    tools: bool = False
    workspace: Optional[str] = None
    input_from: Optional[str] = None   # name of previous stage whose output to inject
    output_path: Optional[str] = None  # write stage result to this path
    adapter_path: Optional[str] = None
