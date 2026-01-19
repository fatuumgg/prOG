from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from chat_engine.domain.models import Message


@runtime_checkable
class TokenCounter(Protocol):
    """
    Считает токены.
    Конвенция: message.meta["tokens"] может хранить кэш "стоимости сообщения"
    (content + overhead), чтобы count_messages работал быстро.
    """

    def count_text(self, text: str) -> int:
        ...

    def count_messages(self, messages: Sequence[Message]) -> int:
        ...
