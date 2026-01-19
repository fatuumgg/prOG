from __future__ import annotations
from typing import List

from chat_engine.domain.models import Conversation, Message
from chat_engine.ports.augment import ContextAugmentor

class NoopAugmentor(ContextAugmentor):
    def augment(self, convo: Conversation, draft_context: List[Message]) -> List[Message]:
        return draft_context
