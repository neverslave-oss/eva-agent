"""evolution_control_request — EvolutionControlRequest Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

class EvolutionControlRequest(BaseModel):
    action: str          # start | pause | resume | stop | reset
    cap: Optional[int] = None
    reason: Optional[str] = None
