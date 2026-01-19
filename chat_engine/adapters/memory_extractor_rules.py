from __future__ import annotations

import re
from typing import List

from chat_engine.domain.models import Message
from chat_engine.ports.memory_extractor import MemoryCandidate, MemoryExtractor

# Рус/англ
RE_NAME_RU = re.compile(r"\bменя\s+зовут\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё \-]{1,40})", re.IGNORECASE)
RE_NAME_EN = re.compile(r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z \-]{1,40})", re.IGNORECASE)

RE_DISLIKE_RU = re.compile(r"\bя\s+не\s+люблю\s+(.{1,80})", re.IGNORECASE)
RE_LIKE_RU = re.compile(r"\bя\s+люблю\s+(.{1,80})", re.IGNORECASE)
RE_DISLIKE_EN = re.compile(r"\bi\s+don'?t\s+like\s+(.{1,80})", re.IGNORECASE)
RE_LIKE_EN = re.compile(r"\bi\s+like\s+(.{1,80})", re.IGNORECASE)

RE_PROJECT_RU = re.compile(r"\bя\s+(?:делаю|работаю\s+над)\s+(.{1,120})", re.IGNORECASE)
RE_GOAL_RU = re.compile(r"\bя\s+хочу\s+(.{1,120})", re.IGNORECASE)

def _clean(val: str) -> str:
    val = " ".join(val.strip().split())
    val = val.rstrip(".!?,;:")
    return val

class RuleBasedMemoryExtractor(MemoryExtractor):
    def extract(self, message: Message) -> List[MemoryCandidate]:
        text = message.content or ""
        out: List[MemoryCandidate] = []

        m = RE_NAME_RU.search(text) or RE_NAME_EN.search(text)
        if m:
            name = _clean(m.group(1))
            if 2 <= len(name) <= 40:
                out.append(MemoryCandidate(key="name", value=name, confidence=0.9))

        for rx, key_prefix, conf in [
            (RE_DISLIKE_RU, "dislikes:", 0.8),
            (RE_DISLIKE_EN, "dislikes:", 0.8),
            (RE_LIKE_RU, "likes:", 0.7),
            (RE_LIKE_EN, "likes:", 0.7),
        ]:
            mm = rx.search(text)
            if mm:
                topic = _clean(mm.group(1))
                if 2 <= len(topic) <= 80:
                    out.append(MemoryCandidate(key=f"{key_prefix}{topic.lower()}", value=topic, confidence=conf))

        pm = RE_PROJECT_RU.search(text)
        if pm:
            proj = _clean(pm.group(1))
            if 2 <= len(proj) <= 120:
                out.append(MemoryCandidate(key="project.current", value=proj, confidence=0.65))

        gm = RE_GOAL_RU.search(text)
        if gm:
            goal = _clean(gm.group(1))
            if 2 <= len(goal) <= 120:
                out.append(MemoryCandidate(key="goal.current", value=goal, confidence=0.6))

        return out
