from __future__ import annotations

from dataclasses import dataclass
from typing import List

from chat_engine.domain.models import Conversation, Message
from chat_engine.ports.augment import ContextAugmentor
from chat_engine.ports.memory_store import UserMemoryStore
from chat_engine.ports.tokens import TokenCounter

def _is_memory(m: Message) -> bool:
    return m.role == "system" and m.meta.get("type") == "user_memory"

@dataclass
class MemoryAugmentor(ContextAugmentor):
    store: UserMemoryStore
    counter: TokenCounter
    user_id: str
    max_tokens: int = 180 

    def augment(self, convo: Conversation, draft_context: List[Message]) -> List[Message]:
        base = [m for m in draft_context if not _is_memory(m)]
        facts = self.store.get_facts(self.user_id)
        if not facts:
            return base

        facts.sort(key=lambda f: (float(f.confidence), f.updated_at), reverse=True)

        lines: List[str] = ["Профиль пользователя (память):"]

        likes = [f for f in facts if f.key.startswith("likes:")]
        dislikes = [f for f in facts if f.key.startswith("dislikes:")]
        name = next((f for f in facts if f.key == "name"), None)
        project = next((f for f in facts if f.key == "project.current"), None)
        goal = next((f for f in facts if f.key == "goal.current"), None)

        def add(line: str) -> None:
            lines.append(f"- {line}")

        if name:
            add(f"Имя: {name.value}")
        if project:
            add(f"Проект: {project.value}")
        if goal:
            add(f"Цель: {goal.value}")

        if likes:
            add("Нравится: " + ", ".join(f.value for f in likes[:5]))
        if dislikes:
            add("Не нравится: " + ", ".join(f.value for f in dislikes[:5]))

        if len(lines) == 1:
            for f in facts[:10]:
                add(f"{f.key} = {f.value}")

        content = "\n".join(lines)

        mem_msg = Message(
            id="user_memory",
            role="system",
            content=content,
            created_at=base[0].created_at if base else convo.messages[0].created_at,
            meta={"type": "user_memory", "pinned": True},
        )
        mem_tokens = self.counter.count_messages([mem_msg])
        if mem_tokens > self.max_tokens:
            kept = lines[: min(len(lines), 8)]
            mem_msg = Message(
                id="user_memory",
                role="system",
                content="\n".join(kept),
                created_at=mem_msg.created_at,
                meta={"type": "user_memory", "pinned": True},
            )
            mem_tokens = self.counter.count_messages([mem_msg])

        mem_msg.meta["tokens"] = mem_tokens

        insert_at = 0
        for i, m in enumerate(base):
            if m.meta.get("pinned") is True:
                insert_at = i + 1

        return base[:insert_at] + [mem_msg] + base[insert_at:]
