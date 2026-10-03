"""task_in — TaskIn Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

class TaskIn(BaseModel):
    role: str
    task: str
    adapter_path: Optional[str] = None
