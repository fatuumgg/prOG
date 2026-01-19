from __future__ import annotations

from typing import Protocol, runtime_checkable

from chat_engine.domain.memory_models import UserMemoryFact


@runtime_checkable
class UserMemoryStore(Protocol):
    """Персистентное хранилище фактов о пользователе."""

    def get_facts(self, user_id: str) -> list[UserMemoryFact]:
        ...

    def upsert_fact(self, user_id: str, fact: UserMemoryFact) -> None:
        ...

    def delete_fact(self, user_id: str, fact_id: str) -> bool:
        ...

    def delete_by_key(self, user_id: str, key: str) -> int:
        ...

    def clear(self, user_id: str) -> None:
        ...
