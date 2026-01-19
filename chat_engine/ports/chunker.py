from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from chat_engine.domain.rag_models import LoadedPage, DocumentChunk


@runtime_checkable
class Chunker(Protocol):
    """Режет страницы на чанки для индексации."""

    def chunk(self, pages: Sequence[LoadedPage]) -> Sequence[DocumentChunk]:
        ...
