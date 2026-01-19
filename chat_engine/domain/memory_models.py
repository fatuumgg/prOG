from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class UserMemoryFact:
    fact_id: str
    key: str
    value: str
    confidence: float
    updated_at: datetime
    source_message_id: str
