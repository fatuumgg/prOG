from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from chat_engine.ports.embeddings import Embedder


@dataclass
class SentenceTransformerEmbedder(Embedder):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vecs]
