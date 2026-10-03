"""init_request — InitRequest Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

class InitRequest(BaseModel):
    force: bool = False
    description: str = ""
