from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from chat_engine.domain.models import Message


@dataclass(frozen=True)
class MemoryCandidate:
    key: str
    value: str
    confidence: float


@runtime_checkable
class MemoryExtractor(Protocol):
    """Извлекает факты о пользователе из одного сообщения."""

    def extract(self, message: Message) -> list[MemoryCandidate]:
        ...
