from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from chat_engine.domain.memory_models import UserMemoryFact
from chat_engine.ports.memory_store import UserMemoryStore


def _dt_to_iso(dt: datetime) -> str:
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _dt_from_iso(s: str) -> datetime:
    s = (s or "").strip()
    if not s:
        return datetime.now(timezone.utc)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class JsonUserMemoryStore(UserMemoryStore):
    """
    Формат:
    {
      "users": {
        "<user_id>": {
          "<fact_id>": { "key": "...", "value": "...", "confidence": 0.7, "updated_at": "...", "source_message_id": "..." }
        }
      }
    }
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = {"users": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {"users": {}}
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {"users": {}}
            if not isinstance(data, dict) or "users" not in data or not isinstance(data["users"], dict):
                self._data = {"users": {}}
            else:
                self._data = data
        except Exception:
            try:
                backup = self.path.with_suffix(self.path.suffix + ".bad")
                self.path.replace(backup)
            except Exception:
                pass
            self._data = {"users": {}}

    def _save(self) -> None:
        text = json.dumps(self._data, ensure_ascii=False, indent=2)
        _atomic_write(self.path, text)

    def get_facts(self, user_id: str) -> List[UserMemoryFact]:
        users = self._data.get("users", {})
        u = users.get(user_id, {}) if isinstance(users, dict) else {}
        out: List[UserMemoryFact] = []

        if not isinstance(u, dict):
            return out

        for fid, f in u.items():
            if not isinstance(f, dict):
                continue
            out.append(
                UserMemoryFact(
                    fact_id=str(fid),
                    key=str(f.get("key", "")),
                    value=str(f.get("value", "")),
                    confidence=float(f.get("confidence", 0.0)),
                    updated_at=_dt_from_iso(str(f.get("updated_at", ""))),
                    source_message_id=str(f.get("source_message_id", "")),
                )
            )
        return out

    def upsert_fact(self, user_id: str, fact: UserMemoryFact) -> None:
        users = self._data.setdefault("users", {})
        if not isinstance(users, dict):
            self._data["users"] = {}
            users = self._data["users"]

        u = users.setdefault(user_id, {})
        if not isinstance(u, dict):
            users[user_id] = {}
            u = users[user_id]

        u[fact.fact_id] = {
            "key": fact.key,
            "value": fact.value,
            "confidence": float(fact.confidence),
            "updated_at": _dt_to_iso(fact.updated_at),
            "source_message_id": fact.source_message_id,
        }
        self._save()

    def delete_fact(self, user_id: str, fact_id: str) -> bool:
        users = self._data.get("users", {})
        if not isinstance(users, dict):
            return False
        u = users.get(user_id, {})
        if not isinstance(u, dict):
            return False

        if fact_id in u:
            del u[fact_id]
            self._save()
            return True
        return False

    def delete_by_key(self, user_id: str, key: str) -> int:
        users = self._data.get("users", {})
        if not isinstance(users, dict):
            return 0
        u = users.get(user_id, {})
        if not isinstance(u, dict):
            return 0

        to_del = [fid for fid, f in u.items() if isinstance(f, dict) and f.get("key") == key]
        for fid in to_del:
            del u[fid]
        if to_del:
            self._save()
        return len(to_del)

    def clear(self, user_id: str) -> None:
        users = self._data.setdefault("users", {})
        if not isinstance(users, dict):
            self._data["users"] = {}
            users = self._data["users"]
        users[user_id] = {}
        self._save()
