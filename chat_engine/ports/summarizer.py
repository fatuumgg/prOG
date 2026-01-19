from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from chat_engine.domain.models import Message


@runtime_checkable
class Summarizer(Protocol):
    """Сжимает историю в короткую сводку."""

    def summarize(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        ...
