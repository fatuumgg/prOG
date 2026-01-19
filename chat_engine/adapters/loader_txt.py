from __future__ import annotations

from pathlib import Path
from typing import Sequence

from chat_engine.domain.rag_models import LoadedPage
from chat_engine.ports.loaders import DocumentLoader

class TxtLoader(DocumentLoader):
    def load(self, path: str) -> Sequence[LoadedPage]:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        return [LoadedPage(source=str(p), text=text, page=None)]
