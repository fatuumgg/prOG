from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from chat_engine.domain.rag_models import DocumentChunk
from chat_engine.ports.chunker import Chunker
from chat_engine.ports.embeddings import Embedder
from chat_engine.ports.loaders import DocumentLoader
from chat_engine.ports.vector_store import VectorStore


@dataclass
class RagIndexer:
    loaders: Dict[str, DocumentLoader]  
    chunker: Chunker
    embedder: Embedder
    store: VectorStore

    def ingest_paths(self, paths: Sequence[str]) -> int:
        all_chunks: List[DocumentChunk] = []

        for p in paths:
            path = str(p)
            ext = Path(path).suffix.lower().lstrip(".")
            loader = self.loaders.get(ext)
            if loader is None:
                raise ValueError(f"No loader for extension .{ext} (path={path})")

            pages = loader.load(path)
            chunks = list(self.chunker.chunk(pages))
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        vectors = self.embedder.embed([c.text for c in all_chunks])

        if len(vectors) != len(all_chunks):
            raise RuntimeError(f"Embedder returned {len(vectors)} vectors for {len(all_chunks)} chunks")

        dim = getattr(self.embedder, "dim", None)
        if isinstance(dim, int) and dim > 0:
            for v in vectors:
                if len(v) != dim:
                    raise RuntimeError(f"Vector dim mismatch: got {len(v)} expected {dim}")

        self.store.upsert(all_chunks, vectors)
        return len(all_chunks)
