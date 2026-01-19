from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from chat_engine.domain.models import Message
from chat_engine.ports.tokens import TokenCounter


@dataclass
class TiktokenTokenCounter(TokenCounter):
    """
    Практичная версия для проекта:
    - per-message cost: tokens_per_message + tokens(content) + tokens(name if any)
    - НЕ добавляем "final assistant tokens", чтобы count_messages([msg]) был корректен для кэша.
    """
    encoding_name: str = "cl100k_base"
    tokens_per_message: int = 3
    tokens_per_name: int = 1

    def __post_init__(self) -> None:
        import tiktoken
        self._enc = tiktoken.get_encoding(self.encoding_name)

    def count_text(self, text: str) -> int:
        return len(self._enc.encode(text or ""))

    def count_messages(self, messages: Sequence[Message]) -> int:
        total = 0
        for m in messages:
            total += self.tokens_per_message
            total += self.count_text(m.content or "")
            if "name" in m.meta:
                total += self.tokens_per_name
                total += self.count_text(str(m.meta["name"]))
        return total
