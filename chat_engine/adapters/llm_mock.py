from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from chat_engine.domain.models import Message
from chat_engine.ports.llm import LLMClient, LLMResponse

class EchoMockLLM(LLMClient):
    def generate(self, messages: Sequence[Message], *, max_output_tokens: int) -> LLMResponse:
        last_user: Optional[Message] = next((m for m in reversed(messages) if m.role == "user"), None)
        text = f"[mock] Ответ на: {last_user.content if last_user else ''}"
        return LLMResponse(text=text, usage={"input_tokens": 0, "output_tokens": 0})

@dataclass
class ScriptedMockLLM(LLMClient):
    rules: Dict[str, str]

    def generate(self, messages: Sequence[Message], *, max_output_tokens: int) -> LLMResponse:
        last_user: Optional[Message] = next((m for m in reversed(messages) if m.role == "user"), None)
        prompt = (last_user.content if last_user else "").lower()

        for k, v in self.rules.items():
            if k.lower() in prompt:
                return LLMResponse(text=v, usage={"input_tokens": 0, "output_tokens": 0})

        return LLMResponse(text="[mock] Не знаю что сказать.", usage={"input_tokens": 0, "output_tokens": 0})
