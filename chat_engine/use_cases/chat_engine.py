from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from chat_engine.domain.models import Conversation, Message
from chat_engine.ports.augment import ContextAugmentor
from chat_engine.ports.llm import LLMClient
from chat_engine.ports.repo import ConversationRepo
from chat_engine.ports.summarizer import Summarizer
from chat_engine.ports.tokens import TokenCounter
from chat_engine.ports.truncation import TruncationStrategy
from chat_engine.use_cases.budget import Budget
from chat_engine.use_cases.memory_manager import MemoryManager
from chat_engine.use_cases.summary_buffer import SummaryBuffer, SummaryPolicy

from chat_engine.ports.memory_extractor import MemoryExtractor
from chat_engine.ports.memory_store import UserMemoryStore


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex


@dataclass
class ChatEngine:
    repo: ConversationRepo
    llm: LLMClient
    counter: TokenCounter
    budget: Budget
    truncation: TruncationStrategy
    augmentors: List[ContextAugmentor] = field(default_factory=list)
    system_prompt: str = "You are a helpful assistant."

    summarizer: Optional[Summarizer] = None
    summary_policy: SummaryPolicy = field(default_factory=SummaryPolicy)

    user_id: str = "default"
    memory_store: UserMemoryStore | None = None
    memory_extractor: MemoryExtractor | None = None

    def _with_cached_tokens(self, msg: Message) -> Message:
        if "tokens" in msg.meta:
            return msg
        tokens = self.counter.count_messages([msg])
        meta = dict(msg.meta)
        meta["tokens"] = tokens
        return Message(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at,
            meta=meta,
        )

    def _ensure_system(self, convo: Conversation) -> None:
        if not convo.messages or convo.messages[0].role != "system":
            sys_msg = Message(
                id=_new_id(),
                role="system",
                content=self.system_prompt,
                created_at=_now_utc(),
                meta={"pinned": True},
            )
            convo.messages.insert(0, self._with_cached_tokens(sys_msg))

    def handle_user_message(self, conversation_id: str, user_text: str) -> str:
        text, _meta = self.handle_user_message_ex(conversation_id, user_text)
        return text

    def handle_user_message_ex(self, conversation_id: str, user_text: str) -> tuple[str, Dict[str, Any]]:
        convo = self.repo.load(conversation_id)
        self._ensure_system(convo)

        meta: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "user_id": self.user_id,
            "budget": {
                "max_context_tokens": self.budget.max_context_tokens,
                "reserve_output_tokens": self.budget.reserve_output_tokens,
                "max_input_tokens": self.budget.max_input_tokens,
            },
            "summary": {"applied": False},
            "rag": {"inserted": False, "info": None},
            "memory": {"upserted": 0, "skipped": 0},
        }

        user_msg = self._with_cached_tokens(
            Message(id=_new_id(), role="user", content=user_text, created_at=_now_utc(), meta={})
        )
        convo.messages.append(user_msg)

        if self.memory_store is not None and self.memory_extractor is not None:
            mgr = MemoryManager(self.memory_store)
            candidates = self.memory_extractor.extract(user_msg)
            res = mgr.apply(self.user_id, source_message_id=user_msg.id, candidates=candidates)
            meta["memory"]["upserted"] = res.upserted
            meta["memory"]["skipped"] = res.skipped

        context = list(convo.messages)
        for aug in self.augmentors:
            context = aug.augment(convo, context)

        meta["context_tokens_before_fit"] = self.counter.count_messages(context)

        fitted = self.truncation.fit(
            context,
            counter=self.counter,
            max_input_tokens=self.budget.max_input_tokens,
        )

        if self.summarizer is not None:
            sb = SummaryBuffer(self.summarizer, self.counter, self.summary_policy)
            for _ in range(2):
                dropped2 = sb.compute_dropped(context, fitted)
                if not sb.should_summarize(convo, dropped2):
                    break

                changed = sb.apply(convo, dropped2)
                if not changed:
                    break

                meta["summary"]["applied"] = True

                context = list(convo.messages)
                for aug in self.augmentors:
                    context = aug.augment(convo, context)

                meta["context_tokens_before_fit"] = self.counter.count_messages(context)

                fitted = self.truncation.fit(
                    context,
                    counter=self.counter,
                    max_input_tokens=self.budget.max_input_tokens,
                )

        meta["context_tokens_after_fit"] = self.counter.count_messages(fitted)

        fitted_ids = {m.id for m in fitted}
        dropped = [m for m in context if (m.id not in fitted_ids and m.meta.get("pinned") is not True)]
        meta["dropped_messages"] = len(dropped)

        rag_msgs = [m for m in fitted if m.role == "system" and m.meta.get("type") == "retrieved_context"]
        if rag_msgs:
            m = rag_msgs[0]
            meta["rag"]["inserted"] = True
            meta["rag"]["info"] = {
                "chosen": m.meta.get("chosen", 0),
                "sources": m.meta.get("sources", []),
                "tokens": m.meta.get("tokens"),
            }

        resp = self.llm.generate(fitted, max_output_tokens=self.budget.reserve_output_tokens)

        assistant_msg = self._with_cached_tokens(
            Message(
                id=_new_id(),
                role="assistant",
                content=resp.text,
                created_at=_now_utc(),
                meta={"usage": resp.usage},
            )
        )
        convo.messages.append(assistant_msg)

        self.repo.save(convo)

        meta["llm_usage"] = resp.usage
        return resp.text, meta
