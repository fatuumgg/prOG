from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from chat_engine.domain.models import Conversation, Message
from chat_engine.ports.repo import ConversationRepo


def _dt_to_iso(dt: datetime) -> str:
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _dt_from_iso(s: str) -> datetime:
    s = (s or "").strip()
    if not s:
        return datetime.now(timezone.utc)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class JsonFileConversationRepo(ConversationRepo):
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, cid: str) -> Path:
        return self.root / f"{cid}.json"

    def load(self, conversation_id: str) -> Conversation:
        p = self._path(conversation_id)
        if not p.exists():
            return Conversation(conversation_id=conversation_id)

        try:
            raw = p.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
        except Exception:
            return Conversation(conversation_id=conversation_id)

        msgs: list[Message] = []
        for m in data.get("messages", []) if isinstance(data, dict) else []:
            if not isinstance(m, dict):
                continue
            msgs.append(
                Message(
                    id=str(m.get("id", "")),
                    role=m.get("role", "user"),
                    content=str(m.get("content", "")),
                    created_at=_dt_from_iso(str(m.get("created_at", ""))),
                    meta=m.get("meta", {}) if isinstance(m.get("meta", {}), dict) else {},
                )
            )

        settings = data.get("settings", {}) if isinstance(data, dict) else {}
        max_ctx = int(settings.get("max_context_tokens", 800) or 800)
        reserve = int(settings.get("reserved_for_reply_tokens", 200) or 200)

        return Conversation(
            conversation_id=str(data.get("conversation_id", conversation_id)) if isinstance(data, dict) else conversation_id,
            messages=msgs,
            max_context_tokens=max_ctx,
            reserved_for_reply_tokens=reserve,
            summary=(data.get("summary") if isinstance(data, dict) else None),
            memory=(data.get("memory", {}) if isinstance(data, dict) and isinstance(data.get("memory", {}), dict) else {}),
        )

    def save(self, convo: Conversation) -> None:
        p = self._path(convo.conversation_id)

        data: dict[str, Any] = {
            "conversation_id": convo.conversation_id,
            "settings": {
                "max_context_tokens": int(convo.max_context_tokens),
                "reserved_for_reply_tokens": int(convo.reserved_for_reply_tokens),
            },
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "created_at": _dt_to_iso(m.created_at),
                    "meta": m.meta,
                }
                for m in convo.messages
            ],
            "summary": convo.summary,
            "memory": convo.memory,
        }

        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_message(self, conversation_id: str, message: Message) -> None:
        convo = self.load(conversation_id)
        convo.messages.append(message)
        self.save(convo)

    def get_messages(self, conversation_id: str, limit: Optional[int] = None) -> list[Message]:
        convo = self.load(conversation_id)
        if limit is None:
            return list(convo.messages)
        return list(convo.messages[-int(limit) :])

    def delete_messages(self, conversation_id: str, message_ids: list[str]) -> None:
        convo = self.load(conversation_id)
        ids = set(message_ids)
        convo.messages = [m for m in convo.messages if m.id not in ids]
        self.save(convo)

    def trim_before(self, conversation_id: str, timestamp: datetime) -> None:
        ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        convo = self.load(conversation_id)
        convo.messages = [m for m in convo.messages if (m.created_at if m.created_at.tzinfo else m.created_at.replace(tzinfo=timezone.utc)) >= ts]
        self.save(convo)
