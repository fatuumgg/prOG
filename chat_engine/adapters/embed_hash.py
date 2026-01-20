from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import List
from collections.abc import Sequence

from chat_engine.ports.embeddings import Embedder

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return vec
    return [v / norm for v in vec]


@dataclass
class HashingEmbedder(Embedder):
    _dim: int = 256

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            toks = _WORD_RE.findall((text or "").lower())
            vec = [0.0] * self._dim
            for t in toks:
                h = hashlib.md5(t.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self._dim
                sign = 1.0 if (h[4] & 1) == 1 else -1.0
                vec[idx] += sign
            out.append(_l2_normalize(vec))
        return out
