"""message_in — MessageIn Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

class MessageIn(BaseModel):
    message: str
    chat_id: str = ""  # optional session key for conversation isolation
