from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    max_context_tokens: int
    reserve_output_tokens: int
    safety_margin_tokens: int = 32

    @property
    def max_input_tokens(self) -> int:
        value = self.max_context_tokens - self.reserve_output_tokens - self.safety_margin_tokens
        return max(0, int(value))
