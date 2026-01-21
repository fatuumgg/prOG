from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List
from uuid import uuid4

from chat_engine.domain.rag_models import LoadedPage, DocumentChunk
from chat_engine.ports.chunker import Chunker
from chat_engine.ports.tokens import TokenCounter

def _new_id() -> str:
    return uuid4().hex

@dataclass
class TokenChunker(Chunker):
    counter: TokenCounter
    chunk_tokens: int = 800
    overlap_tokens: int = 120

    def chunk(self, pages: Sequence[LoadedPage]) -> Sequence[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        max_chars = max(200, self.chunk_tokens * 4)
        overlap_chars = max(0, min(self.overlap_tokens * 4, max_chars - 1))


        for pg in pages:
            text = (pg.text or "").strip()
            if not text:
                continue

            pos = 0
            n = len(text)

            while pos < n:
                end = min(n, pos + max_chars)

                if end < n:
                    cut = text.rfind(" ", pos, end)
                    if cut != -1 and cut > pos + 50:
                        end = cut

                chunk_text = text[pos:end].strip()
                if chunk_text:
                    tok = self.counter.count_text(chunk_text)
                    chunks.append(
                        DocumentChunk(
                            id=_new_id(),
                            text=chunk_text,
                            source=pg.source,
                            page=pg.page,
                            tokens=tok,
                            meta={},
                        )
                    )

                if end >= n:
                    break

                pos = max(0, end - overlap_chars)

        return chunks
