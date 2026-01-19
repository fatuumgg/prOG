from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from chat_engine.domain.models import Message
from chat_engine.ports.tokens import TokenCounter


@runtime_checkable
class TruncationStrategy(Protocol):
    """Обрезает историю под лимит входного контекста."""

    def fit(
        self,
        messages: Sequence[Message],
        *,
        counter: TokenCounter,
        max_input_tokens: int,
    ) -> list[Message]:
        ...
