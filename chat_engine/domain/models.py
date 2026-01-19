from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


def utcnow() -> datetime:
    """Всегда timezone-aware UTC."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Message:
    id: str
    role: Role
    content: str = ""
    created_at: datetime = field(default_factory=utcnow)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    conversation_id: str
    messages: List[Message] = field(default_factory=list)

    max_context_tokens: int = 800
    reserved_for_reply_tokens: int = 200

    summary: Optional[str] = None

    memory: Dict[str, Any] = field(default_factory=dict)
