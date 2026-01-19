from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Dict, Any, Optional

import requests

from chat_engine.domain.models import Message
from chat_engine.ports.llm import LLMClient, LLMResponse


def _strip_trailing_slash(url: str) -> str:
    return url[:-1] if url.endswith("/") else url


@dataclass
class OllamaLLMClient(LLMClient):
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.2
    timeout_s: int = 120

    session: Optional[requests.Session] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.base_url = _strip_trailing_slash(self.base_url)
        if self.session is None:
            self.session = requests.Session()

    def generate(self, messages: Sequence[Message], *, max_output_tokens: int) -> LLMResponse:
        assert self.session is not None 

        ollama_msgs = [{"role": m.role, "content": (m.content or "")} for m in messages]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": ollama_msgs,
            "stream": False,
            "options": {
                "temperature": float(self.temperature),
                "num_predict": int(max_output_tokens),
            },
        }

        r = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json() if r.content else {}

        text = ""
        msg = data.get("message")
        if isinstance(msg, dict):
            text = (msg.get("content") or "").strip()

        if not text:
            text = (data.get("response") or data.get("content") or "").strip()

        usage = {
            "input_tokens": int(data.get("prompt_eval_count") or 0),
            "output_tokens": int(data.get("eval_count") or 0),
        }

        return LLMResponse(text=text, usage=usage)
