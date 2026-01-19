from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from chat_engine.domain.rag_models import DocumentChunk
from chat_engine.ports.vector_store import VectorStore


def _dot(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class JsonVectorStore(VectorStore):
    """
    файл JSON: { "items": [ {"chunk": {...}, "vector": [...]}, ... ] }
    upsert делаем по chunk.id (чтобы не раздувать файл бесконечно)
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._items = []
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
            items = data.get("items", [])
            self._items = items if isinstance(items, list) else []
        except Exception:
            self._items = []

    def _save(self) -> None:
        _atomic_write(self.path, json.dumps({"items": self._items}, ensure_ascii=False, indent=2))

    def count(self) -> int:
        return len(self._items)

    def delete_by_source(self, source: str) -> int:
        before = len(self._items)
        s = (source or "").lower()
        self._items = [
            it for it in self._items
            if (it.get("chunk", {}).get("source", "") or "").lower() != s
        ]
        removed = before - len(self._items)
        if removed:
            self._save()
        return removed

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[List[float]]) -> None:
        # индекс по id
        idx: Dict[str, int] = {}
        for i, it in enumerate(self._items):
            ch = it.get("chunk", {})
            cid = str(ch.get("id", ""))
            if cid:
                idx[cid] = i

        changed = False
        for ch, v in zip(chunks, vectors):
            item = {
                "chunk": {
                    "id": ch.id,
                    "text": ch.text,
                    "source": ch.source,
                    "page": ch.page,
                    "tokens": int(ch.tokens),
                    "meta": ch.meta,
                },
                "vector": list(v),
            }

            if ch.id in idx:
                self._items[idx[ch.id]] = item
            else:
                self._items.append(item)
            changed = True

        if changed:
            self._save()

    def search(self, query_vector: List[float], top_k: int) -> List[DocumentChunk]:
        k = max(0, int(top_k))
        if k == 0 or not self._items:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for it in self._items:
            v = it.get("vector", [])
            if not isinstance(v, list) or not v:
                continue
            score = _dot(query_vector, v)
            scored.append((score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        out: List[DocumentChunk] = []
        for _, it in top:
            c = it.get("chunk", {})
            out.append(
                DocumentChunk(
                    id=str(c.get("id", "")),
                    text=str(c.get("text", "")),
                    source=str(c.get("source", "")),
                    page=c.get("page"),
                    tokens=int(c.get("tokens", 0) or 0),
                    meta=c.get("meta", {}) if isinstance(c.get("meta", {}), dict) else {},
                )
            )
        return out
