from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Текст -> эмбеддинги фиксированной размерности."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    @property
    def dim(self) -> int:
        ...
