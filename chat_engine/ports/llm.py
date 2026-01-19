from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypedDict, runtime_checkable
from collections.abc import Sequence

from chat_engine.domain.models import Message


class LLMUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: LLMUsage = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Общий интерфейс к LLM (реальный/мок)."""

    def generate(self, messages: Sequence[Message], *, max_output_tokens: int) -> LLMResponse:
        ...
