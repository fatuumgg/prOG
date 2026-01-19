from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from chat_engine.domain.models import Conversation, Message


@runtime_checkable
class ContextAugmentor(Protocol):
    """Добавляет системный контекст (память/RAG/и т.п.) в draft_context."""

    def augment(self, convo: Conversation, draft_context: Sequence[Message]) -> list[Message]:
        ...
