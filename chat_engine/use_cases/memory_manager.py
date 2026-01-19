from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

from chat_engine.domain.memory_models import UserMemoryFact
from chat_engine.ports.memory_extractor import MemoryCandidate
from chat_engine.ports.memory_store import UserMemoryStore


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fact_id(user_id: str, key: str) -> str:
    h = hashlib.sha1(f"{user_id}:{key}".encode("utf-8")).hexdigest()
    return h[:16]


@dataclass
class MemoryUpdateResult:
    upserted: int = 0
    skipped: int = 0


class MemoryManager:
    def __init__(self, store: UserMemoryStore):
        self.store = store

    def apply(
        self,
        user_id: str,
        *,
        source_message_id: str,
        candidates: List[MemoryCandidate],
    ) -> MemoryUpdateResult:
        best: Dict[str, MemoryCandidate] = {}
        for c in candidates:
            cur = best.get(c.key)
            if cur is None or float(c.confidence) >= float(cur.confidence):
                best[c.key] = c

        existing = {f.key: f for f in self.store.get_facts(user_id)}
        res = MemoryUpdateResult()

        for key, c in best.items():
            fid = _fact_id(user_id, key)
            old = existing.get(key)

            if old is None:
                self.store.upsert_fact(
                    user_id,
                    UserMemoryFact(
                        fact_id=fid,
                        key=key,
                        value=c.value,
                        confidence=float(c.confidence),
                        updated_at=_now(),
                        source_message_id=source_message_id,
                    ),
                )
                res.upserted += 1
                continue

            old_val = (old.value or "").strip()
            new_val = (c.value or "").strip()

            if old_val == new_val:
                new_conf = max(float(old.confidence), float(c.confidence))
                self.store.upsert_fact(
                    user_id,
                    UserMemoryFact(
                        fact_id=fid,
                        key=key,
                        value=old.value,
                        confidence=new_conf,
                        updated_at=_now(),
                        source_message_id=source_message_id,
                    ),
                )
                res.upserted += 1
                continue

            if float(c.confidence) >= float(old.confidence):
                self.store.upsert_fact(
                    user_id,
                    UserMemoryFact(
                        fact_id=fid,
                        key=key,
                        value=c.value,
                        confidence=float(c.confidence),
                        updated_at=_now(),
                        source_message_id=source_message_id,
                    ),
                )
                res.upserted += 1
            else:
                res.skipped += 1

        return res
