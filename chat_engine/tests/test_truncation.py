from datetime import datetime, timezone

from chat_engine.domain.models import Message
from chat_engine.adapters.tokens_approx import ApproxTokenCounter
from chat_engine.adapters.trunc_recency import RecencyTruncation


def test_recency_truncation_keeps_system():
    counter = ApproxTokenCounter()
    trunc = RecencyTruncation()

    msgs = [
        Message(
            id="s",
            role="system",
            content="sys",
            created_at=datetime.now(timezone.utc),
            meta={"pinned": True},  
        ),
        Message(id="u1", role="user", content="x" * 200, created_at=datetime.now(timezone.utc), meta={}),
        Message(id="a1", role="assistant", content="y" * 200, created_at=datetime.now(timezone.utc), meta={}),
        Message(id="u2", role="user", content="z" * 200, created_at=datetime.now(timezone.utc), meta={}),
    ]

    fitted = trunc.fit(msgs, counter=counter, max_input_tokens=30)  

    assert len(fitted) >= 1
    assert fitted[0].role == "system"
    assert fitted[0].meta.get("pinned") is True
