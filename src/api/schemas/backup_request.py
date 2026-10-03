"""backup_request — BackupRequest Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

class BackupRequest(BaseModel):
    description: str = ""
    full: bool = True
