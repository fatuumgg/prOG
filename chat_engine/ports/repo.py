from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable
from collections.abc import Sequence

from chat_engine.domain.models import Conversation, Message


@runtime_checkable
class ConversationRepo(Protocol):
    """Персистентная история диалогов."""

    def load(self, conversation_id: str) -> Conversation:
        ...

    def save(self, convo: Conversation) -> None:
        ...

    def append_message(self, conversation_id: str, message: Message) -> None:
        ...

    def get_messages(self, conversation_id: str, limit: int | None = None) -> list[Message]:
        ...

    def delete_messages(self, conversation_id: str, message_ids: Sequence[str]) -> None:
        ...

    def trim_before(self, conversation_id: str, timestamp: datetime) -> None:
        ...
