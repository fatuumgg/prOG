from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from chat_engine.domain.models import Message
from chat_engine.ports.llm import LLMClient
from chat_engine.ports.summarizer import Summarizer


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class LLMSummarizer(Summarizer):
    llm: LLMClient

    def summarize(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        if not messages:
            return "Сводка: (пусто)"

        lines = []
        for m in messages:
            if m.role == "system":
                continue
            content = (m.content or "").strip()
            if not content:
                continue
            lines.append(f"{m.role.upper()}: {content}")

        transcript = "\n".join(lines) if lines else "(нет текста)"

        prompt = (
            "Суммаризируй диалог кратко (10–15 пунктов). "
            "Сохрани факты, решения, обязательства, предпочтения пользователя. "
            "Без воды.\n\n"
            f"ДИАЛОГ:\n{transcript}\n"
        )

        created_at = messages[-1].created_at if messages else _now_utc()
        resp = self.llm.generate(
            [
                Message(
                    id="sum_sys",
                    role="system",
                    content="You are a precise summarizer. Output concise bullet points in Russian.",
                    created_at=created_at,
                    meta={"pinned": True},
                ),
                Message(id="sum_user", role="user", content=prompt, created_at=created_at, meta={}),
            ],
            max_output_tokens=int(max_tokens),
        )
        return (resp.text or "").strip()
