"""new_session_in — NewSessionIn Pydantic schema for the Kernel Evolving API.

One class per file (project rule). Extracted from src/api.py (issue #2).
"""
from pydantic import BaseModel
from typing import Optional

class NewSessionIn(BaseModel):
    """Payload for creating a new conversation/session record."""
    chat_id: str = ""  # optional; a fresh uuid is generated when omitted
    title: str = ""
