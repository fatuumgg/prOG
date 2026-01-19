from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional, Tuple, Dict, Any

from chat_engine.app.settings import AppSettings

from chat_engine.ports.tokens import TokenCounter

from chat_engine.adapters.repo_json import JsonFileConversationRepo
from chat_engine.adapters.trunc_recency import RecencyTruncation

from chat_engine.adapters.vector_store_json import JsonVectorStore
from chat_engine.adapters.chunker_token import TokenChunker
from chat_engine.adapters.loader_txt import TxtLoader
from chat_engine.adapters.loader_pdf_pypdf import PdfLoaderPyPDF
from chat_engine.adapters.rag_augmentor import RagAugmentor

from chat_engine.adapters.memory_store_json import JsonUserMemoryStore
from chat_engine.adapters.memory_extractor_rules import RuleBasedMemoryExtractor
from chat_engine.adapters.memory_augmentor import MemoryAugmentor

from chat_engine.use_cases.budget import Budget
from chat_engine.use_cases.chat_engine import ChatEngine
from chat_engine.use_cases.summary_buffer import SummaryPolicy
from chat_engine.use_cases.rag_indexer import RagIndexer


@dataclass(frozen=True)
class EngineBundle:
    engine: ChatEngine
    indexer: RagIndexer
    rag_store: JsonVectorStore
    memory_store: JsonUserMemoryStore
    counter: TokenCounter


# -----------------------
# Internal shared cache
# -----------------------
_cache_lock = Lock()
_shared: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _settings_key(s: AppSettings) -> Tuple[Any, ...]:
    e = s.engine
    r = s.rag
    m = s.memory
    return (
        s.data_store_dir,
        m.memory_store_path,
        r.rag_store_path,
        e.tokenizer_backend,
        e.embedder_backend,
        e.llm_backend,
        e.summarizer_backend,
        e.enable_summary,
        e.enable_rag,
        e.rag_mode,
        e.ollama_url,
        e.ollama_model,
        r.sbert_model,
        r.chunk_tokens,
        r.overlap_tokens,
        r.rag_top_k,
        r.rag_max_tokens,
        e.system_prompt,
        e.max_context_tokens,
        e.reserve_output_tokens,
        e.summary_min_dropped,
        e.summary_max_tokens,
    )


def _build_shared(settings: AppSettings) -> Dict[str, Any]:
    if settings.engine.tokenizer_backend == "tiktoken":
        from chat_engine.adapters.tokens_tiktoken import TiktokenTokenCounter
        counter: TokenCounter = TiktokenTokenCounter()
    else:
        from chat_engine.adapters.tokens_approx import ApproxTokenCounter
        counter = ApproxTokenCounter()

    if settings.engine.llm_backend == "ollama":
        from chat_engine.adapters.llm_ollama import OllamaLLMClient
        llm = OllamaLLMClient(
            base_url=settings.engine.ollama_url,
            model=settings.engine.ollama_model,
        )
    else:
        from chat_engine.adapters.llm_mock import EchoMockLLM
        llm = EchoMockLLM()

    if not settings.engine.enable_summary:
        summarizer = None
    else:
        if settings.engine.summarizer_backend == "llm":
            from chat_engine.adapters.summarizer_llm import LLMSummarizer
            summarizer = LLMSummarizer(llm=llm)
        else:
            from chat_engine.adapters.summarizer_mock import MockSummarizer
            summarizer = MockSummarizer()

    summary_policy = SummaryPolicy(
        min_dropped_messages=settings.engine.summary_min_dropped,
        max_summary_tokens=settings.engine.summary_max_tokens,
        every_k_messages=None,
    )

    repo = JsonFileConversationRepo(settings.data_store_dir)

    memory_store = JsonUserMemoryStore(settings.memory.memory_store_path)
    memory_extractor = RuleBasedMemoryExtractor()

    if settings.engine.embedder_backend == "sbert":
        from chat_engine.adapters.embed_sbert import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(model_name=settings.rag.sbert_model)
    else:
        from chat_engine.adapters.embed_hash import HashingEmbedder
        embedder = HashingEmbedder()

    rag_store = JsonVectorStore(settings.rag.rag_store_path)
    chunker = TokenChunker(
        counter=counter,
        chunk_tokens=settings.rag.chunk_tokens,
        overlap_tokens=settings.rag.overlap_tokens,
    )

    loaders = {
        "txt": TxtLoader(),
        "md": TxtLoader(),
        "pdf": PdfLoaderPyPDF(),
    }

    indexer = RagIndexer(loaders=loaders, chunker=chunker, embedder=embedder, store=rag_store)

    rag_aug: Optional[RagAugmentor] = None
    if settings.engine.enable_rag and settings.engine.rag_mode != "off" and rag_store.count() > 0:
        rag_aug = RagAugmentor(
            mode=settings.engine.rag_mode,  
            store=rag_store,
            embedder=embedder,
            counter=counter,
            top_k=settings.rag.rag_top_k,
            max_rag_tokens=settings.rag.rag_max_tokens,
        )

    return {
        "counter": counter,
        "llm": llm,
        "summarizer": summarizer,
        "summary_policy": summary_policy,
        "repo": repo,
        "memory_store": memory_store,
        "memory_extractor": memory_extractor,
        "rag_store": rag_store,
        "indexer": indexer,
        "rag_aug": rag_aug,
    }


def build_bundle(settings: AppSettings, *, user_id: str) -> EngineBundle:
    key = _settings_key(settings)

    with _cache_lock:
        shared = _shared.get(key)
        if shared is None:
            shared = _build_shared(settings)
            _shared[key] = shared

    counter: TokenCounter = shared["counter"]
    repo = shared["repo"]
    llm = shared["llm"]
    summarizer = shared["summarizer"]
    summary_policy = shared["summary_policy"]
    memory_store: JsonUserMemoryStore = shared["memory_store"]
    memory_extractor = shared["memory_extractor"]
    rag_store: JsonVectorStore = shared["rag_store"]
    indexer: RagIndexer = shared["indexer"]
    rag_aug: Optional[RagAugmentor] = shared["rag_aug"]

    mem_aug: Optional[MemoryAugmentor] = None
    if settings.engine.enable_memory:
        mem_aug = MemoryAugmentor(
            store=memory_store,
            counter=counter,
            user_id=user_id,
            max_tokens=settings.engine.memory_max_tokens,
        )

    augmentors = [a for a in [mem_aug, rag_aug] if a is not None]

    engine = ChatEngine(
        repo=repo,
        llm=llm,
        counter=counter,
        budget=Budget(
            max_context_tokens=settings.engine.max_context_tokens,
            reserve_output_tokens=settings.engine.reserve_output_tokens,
        ),
        truncation=RecencyTruncation(),
        augmentors=augmentors,
        system_prompt=settings.engine.system_prompt,
        summarizer=summarizer,
        summary_policy=summary_policy,
        user_id=user_id,
        memory_store=memory_store if settings.engine.enable_memory else None,
        memory_extractor=memory_extractor if settings.engine.enable_memory else None,
    )

    return EngineBundle(
        engine=engine,
        indexer=indexer,
        rag_store=rag_store,
        memory_store=memory_store,
        counter=counter,
    )
