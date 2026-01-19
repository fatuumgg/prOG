from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LoadedPage:
    source: str
    text: str
    page: Optional[int] = None


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source: str
    page: Optional[int] = None
    tokens: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
