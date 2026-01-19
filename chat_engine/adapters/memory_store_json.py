from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Sequence, Optional

from chat_engine.domain.rag_models import DocumentChunk
from chat_engine.ports.vector_store import VectorStore

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

class JsonVectorStore(VectorStore):
    """
    Простой persisted store:
    файл JSON: { "items": [ {"chunk": {...}, "vector": [...]}, ... ] }
    """
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: List[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._items = []
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._items = data.get("items", [])

    def _save(self) -> None:
        data = {"items": self._items}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def count(self) -> int:
        return len(self._items)
    
    def delete_by_source(self, source: str) -> int:
        before = len(self._items)
        s = (source or "").lower()
        self._items = [it for it in self._items if (it["chunk"]["source"] or "").lower() != s]
        removed = before - len(self._items)
        if removed:
            self._save()
        return removed

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[List[float]]) -> None:
        for ch, v in zip(chunks, vectors):
            self._items.append(
                {
                    "chunk": {
                        "id": ch.id,
                        "text": ch.text,
                        "source": ch.source,
                        "page": ch.page,
                        "tokens": ch.tokens,
                        "meta": ch.meta,
                    },
                    "vector": v,
                }
            )
        self._save()
        

    def search(self, query_vector: List[float], top_k: int) -> List[DocumentChunk]:
        scored: List[tuple[float, dict]] = []
        for it in self._items:
            v = it["vector"]
            score = _dot(query_vector, v) 
            scored.append((score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(0, top_k)]

        out: List[DocumentChunk] = []
        for _, it in top:
            c = it["chunk"]
            out.append(
                DocumentChunk(
                    id=c["id"],
                    text=c["text"],
                    source=c["source"],
                    page=c.get("page"),
                    tokens=int(c.get("tokens", 0)),
                    meta=c.get("meta", {}),
                )
            )
        return out
