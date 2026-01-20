from __future__ import annotations

import math
from collections.abc import Sequence

from chat_engine.domain.models import Message
from chat_engine.ports.tokens import TokenCounter


class ApproxTokenCounter(TokenCounter):
    """
    MVP-оценка:
      - 1 токен ~ 4 символа
      - + небольшой overhead на сообщение
    Конвенция: meta["tokens"] хранит "стоимость сообщения" (content + overhead),
    чтобы count_messages мог суммировать без пересчёта.
    """

    overhead_per_message: int = 4

    def count_text(self, text: str) -> int:
        return max(1, int(math.ceil(len(text or "") / 4)))

    def count_messages(self, messages: Sequence[Message]) -> int:
        total = 0
        for m in messages:
            cached = m.meta.get("tokens")
            if isinstance(cached, int) and cached >= 0:
                total += cached
            else:
                total += self.count_text(m.content or "") + int(self.overhead_per_message)
        return total
