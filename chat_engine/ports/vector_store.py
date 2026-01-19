from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from chat_engine.domain.rag_models import DocumentChunk


@runtime_checkable
class VectorStore(Protocol):
    """Хранилище эмбеддингов чанков документов."""

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[list[float]]) -> None:
        ...

    def search(self, query_vector: list[float], top_k: int) -> list[DocumentChunk]:
        ...

    def count(self) -> int:
        ...

    def delete_by_source(self, source: str) -> int:
        ...
