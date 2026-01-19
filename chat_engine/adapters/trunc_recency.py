from __future__ import annotations

from typing import List, Sequence

from chat_engine.domain.models import Message
from chat_engine.ports.tokens import TokenCounter
from chat_engine.ports.truncation import TruncationStrategy


class RecencyTruncation(TruncationStrategy):
    """
    - pinned=True всегда остаются
    - остальные набираем с конца (самые новые), пока влезает
    """

    def fit(
        self,
        messages: Sequence[Message],
        *,
        counter: TokenCounter,
        max_input_tokens: int
    ) -> List[Message]:
        if not messages:
            return []

        pinned: List[Message] = [m for m in messages if m.meta.get("pinned") is True]
        others: List[Message] = [m for m in messages if m.meta.get("pinned") is not True]

        pinned_tokens = counter.count_messages(pinned)
        if pinned_tokens >= max_input_tokens:
            return pinned

        kept_rev: List[Message] = []
        kept_tokens = 0

        for m in reversed(others):
            m_tokens = counter.count_messages([m])
            if pinned_tokens + kept_tokens + m_tokens <= max_input_tokens:
                kept_rev.append(m)
                kept_tokens += m_tokens
            else:
                break

        return pinned + list(reversed(kept_rev))
