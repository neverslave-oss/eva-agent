"""fresh_request — FreshRequest Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

class FreshRequest(BaseModel):
    description: str = ""
    keep_ecosystem: bool = True
    dry_run: bool = False
    # confirmation handled via inline buttons
