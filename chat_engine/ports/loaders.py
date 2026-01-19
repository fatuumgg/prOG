from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from chat_engine.domain.rag_models import LoadedPage


@runtime_checkable
class DocumentLoader(Protocol):
    """Загрузка документа в набор страниц (или 1 страницу для txt)."""

    def load(self, path: str) -> Sequence[LoadedPage]:
        ...
