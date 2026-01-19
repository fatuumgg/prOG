from __future__ import annotations

from typing import Sequence

from chat_engine.domain.models import Message
from chat_engine.ports.summarizer import Summarizer


class MockSummarizer(Summarizer):
    """
    Детерминированная "суммаризация" без сети:
    - превращает сообщения в пункты
    - грубо режет по длине, чтобы уложиться в max_tokens (1 токен ~ 4 символа)
    """

    def summarize(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        def norm(s: str) -> str:
            return " ".join((s or "").split())

        lines = []
        for m in messages:
            role = "U" if m.role == "user" else ("A" if m.role == "assistant" else m.role.upper())
            snippet = norm(m.content or "")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            lines.append(f"- {role}: {snippet}")

        lines = lines[:15]
        text = "Сводка (авто):\n" + "\n".join(lines) if lines else "Сводка (авто): (пусто)"

        while (len(text) // 4) > max_tokens and len(lines) > 1:
            lines = lines[:-1]
            text = "Сводка (авто):\n" + "\n".join(lines)

        hard_max_chars = max(20, max_tokens * 4)
        if len(text) > hard_max_chars:
            text = text[: hard_max_chars - 3] + "..."

        return text
