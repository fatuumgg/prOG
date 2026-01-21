import tempfile
from pathlib import Path

from chat_engine.adapters.tokens_approx import ApproxTokenCounter
from chat_engine.adapters.loader_txt import TxtLoader
from chat_engine.adapters.chunker_token import TokenChunker
from chat_engine.adapters.embed_hash import HashingEmbedder
from chat_engine.adapters.vector_store_json import JsonVectorStore
from chat_engine.adapters.rag_augmentor import RagAugmentor
from chat_engine.use_cases.rag_indexer import RagIndexer
from chat_engine.domain.models import Message, Conversation
from datetime import datetime, timezone

def test_rag_index_and_retrieve_topk_and_budget():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        doc = d / "doc.txt"
        doc.write_text("France capital is Paris.\nGermany capital is Berlin.\n", encoding="utf-8")

        counter = ApproxTokenCounter()
        store = JsonVectorStore(str(d / "rag.json"))
        embedder = HashingEmbedder()
        chunker = TokenChunker(counter=counter, chunk_tokens=80, overlap_tokens=10)

        indexer = RagIndexer(
            loaders={"txt": TxtLoader()},
            chunker=chunker,
            embedder=embedder,
            store=store,
        )
        n = indexer.ingest_paths([str(doc)])
        assert n > 0
        assert store.count() > 0

        aug = RagAugmentor(
            store=store,
            embedder=embedder,
            counter=counter,
            top_k=2,
            max_rag_tokens=80,
            mode="always",  
        )


        convo = Conversation(conversation_id="c1", messages=[
            Message(id="sys", role="system", content="SYS", created_at=datetime.now(timezone.utc), meta={"pinned": True}),
            Message(id="u1", role="user", content="What is the capital of France?", created_at=datetime.now(timezone.utc), meta={}),
        ])

        ctx = list(convo.messages)
        out = aug.augment(convo, ctx)

        rc = [m for m in out if m.role == "system" and m.meta.get("type") == "retrieved_context"]
        assert len(rc) == 1

        assert counter.count_messages([rc[0]]) <= aug.max_rag_tokens
