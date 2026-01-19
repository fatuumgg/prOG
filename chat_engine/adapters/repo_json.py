from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
from uuid import uuid4

from chat_engine.domain.models import Conversation, Message
from chat_engine.domain.rag_models import DocumentChunk
from chat_engine.ports.augment import ContextAugmentor
from chat_engine.ports.embeddings import Embedder
from chat_engine.ports.tokens import TokenCounter
from chat_engine.ports.vector_store import VectorStore


def _looks_doc_query(text: str) -> bool:
    t = (text or "").lower()
    keys = [
        "документ", "доки", "файл", "pdf", ".pdf", "txt", ".txt",
        "книга", "страница", "раздел", "по документам", "в документе",
        "в файле", "в книге", "согласно документу", "docs/", "./docs/",
    ]
    return any(k in t for k in keys)


def _loc(ch: DocumentChunk) -> str:
    return ch.source + (f":p{ch.page}" if ch.page else "")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass
class RagAugmentor(ContextAugmentor):
    store: VectorStore
    embedder: Embedder
    counter: TokenCounter
    top_k: int = 4
    max_rag_tokens: int = 250
    mode: str = "auto" 

    def augment(self, convo: Conversation, base: List[Message]) -> List[Message]:
        if not base:
            return base

        last_user: Optional[Message] = next((m for m in reversed(base) if m.role == "user"), None)
        if last_user is None:
            return base

        q = last_user.content or ""
        if self.mode == "auto" and not _looks_doc_query(q):
            return base

        try:
            if hasattr(self.store, "count") and self.store.count() <= 0: 
                return base
        except Exception:
            pass

        qv = self.embedder.embed([q])[0]
        hits: List[DocumentChunk] = self.store.search(qv, top_k=self.top_k)
        if not hits:
            return base

        header = (
            "Релевантные фрагменты из документов (используй их при ответе). "
            "Если ответа нет в этих фрагментах — так и скажи.\n\n"
        )

        def make_msg(text: str, chosen_chunks: Sequence[DocumentChunk]) -> Message:
            return Message(
                id=_new_id("rag"),
                role="system",
                content=text,
                created_at=last_user.created_at,
                meta={
                    "type": "retrieved_context",
                    "pinned": False,
                    "chosen": len(chosen_chunks),
                    "sources": [_loc(c) for c in chosen_chunks],
                },
            )

        chosen: List[DocumentChunk] = []
        parts: List[str] = [header]

        for ch in hits:
            piece = f"[{_loc(ch)}]\n{(ch.text or '').strip()}\n\n"
            trial = "".join(parts) + piece
            trial_msg = make_msg(trial, chosen + [ch])
            tok = self.counter.count_messages([trial_msg])
            if tok <= self.max_rag_tokens:
                parts.append(piece)
                chosen.append(ch)
            else:
                break

        if not chosen:
            ch0 = hits[0]
            text = (ch0.text or "").strip()
            if not text:
                return base

            lo, hi = 0, len(text)
            best = ""
            while lo <= hi:
                mid = (lo + hi) // 2
                trial = header + f"[{_loc(ch0)}]\n{text[:mid]}\n"
                trial_msg = make_msg(trial, [ch0])
                tok = self.counter.count_messages([trial_msg])
                if tok <= self.max_rag_tokens:
                    best = trial
                    lo = mid + 1
                else:
                    hi = mid - 1

            if not best:
                return base

            rag_msg = make_msg(best, [ch0])
            rag_msg.meta["tokens"] = self.counter.count_messages([rag_msg])
        else:
            final_text = "".join(parts)
            rag_msg = make_msg(final_text, chosen)
            rag_msg.meta["tokens"] = self.counter.count_messages([rag_msg])

        insert_at = 0
        for i, m in enumerate(base):
            if m.meta.get("pinned") is True:
                insert_at = i + 1

        return base[:insert_at] + [rag_msg] + base[insert_at:]
