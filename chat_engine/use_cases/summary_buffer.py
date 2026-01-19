from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Sequence
from uuid import uuid4

from chat_engine.domain.models import Conversation, Message
from chat_engine.ports.summarizer import Summarizer
from chat_engine.ports.tokens import TokenCounter


def _new_id() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_summary(m: Message) -> bool:
    return m.role == "system" and m.meta.get("type") == "summary"


@dataclass(frozen=True)
class SummaryPolicy:
    min_dropped_messages: int = 4
    max_summary_tokens: int = 256
    every_k_messages: Optional[int] = None


class SummaryBuffer:
    def __init__(self, summarizer: Summarizer, counter: TokenCounter, policy: SummaryPolicy):
        self.summarizer = summarizer
        self.counter = counter
        self.policy = policy

    def should_summarize(self, convo: Conversation, dropped_messages: Sequence[Message]) -> bool:
        if not dropped_messages:
            return False

        if len(dropped_messages) >= self.policy.min_dropped_messages:
            return True

        if self.policy.every_k_messages:
            non_system = [m for m in convo.messages if m.role != "system"]
            return (len(non_system) % self.policy.every_k_messages) == 0

        return False

    def apply(self, convo: Conversation, dropped_messages: Sequence[Message]) -> bool:
        dropped = [m for m in dropped_messages if m.role != "system"]
        if not dropped:
            return False

        existing_summary: Optional[Message] = next((m for m in convo.messages if _is_summary(m)), None)

        to_summarize: List[Message] = []
        if existing_summary is not None:
            to_summarize.append(
                Message(
                    id=existing_summary.id,
                    role="system",
                    content="Текущая сводка:\n" + (existing_summary.content or ""),
                    created_at=existing_summary.created_at,
                    meta=existing_summary.meta,
                )
            )
        to_summarize.extend(dropped)

        summary_text = self.summarizer.summarize(to_summarize, max_tokens=self.policy.max_summary_tokens)

        if existing_summary and isinstance(existing_summary.meta.get("summary_of_range"), dict):
            from_id = existing_summary.meta["summary_of_range"].get("from_id") or dropped[0].id
        else:
            from_id = dropped[0].id
        to_id = dropped[-1].id

        replaced_messages = list(dropped)
        replaced_tokens = self.counter.count_messages(replaced_messages)
        replaced_count = len(replaced_messages)

        if existing_summary is not None:
            replaced_count += 1
            old_sum_tokens = existing_summary.meta.get("tokens")
            replaced_tokens += int(old_sum_tokens) if isinstance(old_sum_tokens, int) else self.counter.count_messages([existing_summary])

        new_summary = Message(
            id=_new_id(),
            role="system",
            content=summary_text,
            created_at=_now(),
            meta={
                "type": "summary",
                "pinned": True,
                "summary_of_range": {"from_id": from_id, "to_id": to_id},
                "replaced": {"message_count": replaced_count, "tokens": replaced_tokens},
            },
        )

        new_summary.meta["tokens"] = self.counter.count_messages([new_summary])

        ids_to_remove = {m.id for m in dropped}
        if existing_summary is not None:
            ids_to_remove.add(existing_summary.id)

        filtered: List[Message] = [m for m in convo.messages if m.id not in ids_to_remove]

        insert_at = 0
        for i, m in enumerate(filtered):
            if m.role == "system" and not _is_summary(m):
                insert_at = i + 1

        filtered.insert(insert_at, new_summary)
        convo.messages = filtered
        return True

    @staticmethod
    def compute_dropped(context: Sequence[Message], fitted: Sequence[Message]) -> List[Message]:
        fitted_ids = {m.id for m in fitted}
        return [m for m in context if (m.id not in fitted_ids and m.role != "system")]
