"""named_replica_in — NamedReplicaIn Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

class NamedReplicaIn(BaseModel):
    name: str
    role: str = "custom"
    brief_path: Optional[str] = None
    custom_prompt: Optional[str] = None
    workspace: Optional[str] = None
    tools_enabled: bool = False
    output_path: Optional[str] = None
    input_path: Optional[str] = None
    adapter_path: Optional[str] = None
    slot: Optional[str] = None  # named model slot (e.g. "audio"); None = primary
