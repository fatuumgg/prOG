import tempfile
from datetime import datetime, timezone

from chat_engine.domain.models import Message, Conversation
from chat_engine.adapters.tokens_approx import ApproxTokenCounter
from chat_engine.adapters.memory_store_json import JsonUserMemoryStore
from chat_engine.adapters.memory_extractor_rules import RuleBasedMemoryExtractor
from chat_engine.adapters.memory_augmentor import MemoryAugmentor
from chat_engine.use_cases.memory_manager import MemoryManager

def test_memory_extract_store_and_augment():
    with tempfile.TemporaryDirectory() as d:
        store = JsonUserMemoryStore(f"{d}/mem.json")
        extractor = RuleBasedMemoryExtractor()
        mgr = MemoryManager(store)

        msg = Message(
            id="m1",
            role="user",
            content="Меня зовут Аня. Я не люблю шум.",
            created_at=datetime.now(timezone.utc),
            meta={},
        )
        cands = extractor.extract(msg)
        mgr.apply("u1", source_message_id=msg.id, candidates=cands)

        facts = store.get_facts("u1")
        keys = {f.key for f in facts}
        assert "name" in keys
        assert any(f.key.startswith("dislikes:") for f in facts)

        counter = ApproxTokenCounter()
        aug = MemoryAugmentor(store=store, counter=counter, user_id="u1", max_tokens=200)

        convo = Conversation(conversation_id="c1", messages=[
            Message(id="sys", role="system", content="SYS", created_at=datetime.now(timezone.utc), meta={"pinned": True}),
            msg,
        ])
        out = aug.augment(convo, list(convo.messages))
        mem = [m for m in out if m.role == "system" and m.meta.get("type") == "user_memory"]
        assert len(mem) == 1
