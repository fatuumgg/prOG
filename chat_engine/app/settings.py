from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    val = v.strip().lower()
    return val if val in allowed else default


def _env_rag_mode(name: str, default: str = "auto") -> str:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    val = v.strip().lower()

    if val in ("0", "false", "off", "disable", "disabled", "none"):
        return "off"
    if val in ("auto", "always"):
        return val
    if val in ("1", "true", "yes", "y", "on", "enabled", "enable"):
        return "auto"
    return default


@dataclass(frozen=True)
class EngineSettings:
    system_prompt: str = "You are a helpful assistant. Answer in Russian."
    max_context_tokens: int = 800
    reserve_output_tokens: int = 200

    enable_summary: bool = True
    summary_max_tokens: int = 256
    summary_min_dropped: int = 4

    enable_memory: bool = True
    memory_max_tokens: int = 180

    enable_rag: bool = True
    rag_mode: str = "auto"  # off | auto | always

    llm_backend: str = "mock"          # mock | ollama
    summarizer_backend: str = "mock"   # mock | llm
    tokenizer_backend: str = "approx"  # approx | tiktoken
    embedder_backend: str = "hash"     # hash | sbert

    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"


@dataclass(frozen=True)
class RagSettings:
    rag_store_path: str = "./rag_store.json"
    rag_top_k: int = 4
    rag_max_tokens: int = 250

    chunk_tokens: int = 800
    overlap_tokens: int = 120

    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class MemorySettings:
    memory_store_path: str = "./user_memory.json"


@dataclass(frozen=True)
class AppSettings:
    data_store_dir: str = "./data"
    engine: EngineSettings = EngineSettings()
    rag: RagSettings = RagSettings()
    memory: MemorySettings = MemorySettings()

    @staticmethod
    def from_env() -> "AppSettings":
        eng = EngineSettings(
            system_prompt=_env_str("CE_SYSTEM_PROMPT", EngineSettings.system_prompt),
            max_context_tokens=_env_int("CE_MAX_CONTEXT", EngineSettings.max_context_tokens),
            reserve_output_tokens=_env_int("CE_RESERVE_OUTPUT", EngineSettings.reserve_output_tokens),

            enable_summary=_env_bool("CE_ENABLE_SUMMARY", EngineSettings.enable_summary),
            summary_max_tokens=_env_int("CE_SUMMARY_MAX_TOKENS", EngineSettings.summary_max_tokens),
            summary_min_dropped=_env_int("CE_SUMMARY_MIN_DROPPED", EngineSettings.summary_min_dropped),

            enable_memory=_env_bool("CE_ENABLE_MEMORY", EngineSettings.enable_memory),
            memory_max_tokens=_env_int("CE_MEMORY_MAX_TOKENS", EngineSettings.memory_max_tokens),

            enable_rag=_env_bool("CE_ENABLE_RAG", EngineSettings.enable_rag),
            rag_mode=_env_rag_mode("CE_RAG_MODE", EngineSettings.rag_mode),

            llm_backend=_env_choice("CE_LLM", EngineSettings.llm_backend, {"mock", "ollama"}),
            summarizer_backend=_env_choice("CE_SUMMARIZER", EngineSettings.summarizer_backend, {"mock", "llm"}),
            tokenizer_backend=_env_choice("CE_TOKENIZER", EngineSettings.tokenizer_backend, {"approx", "tiktoken"}),
            embedder_backend=_env_choice("CE_EMBEDDER", EngineSettings.embedder_backend, {"hash", "sbert"}),

            ollama_url=_env_str("CE_OLLAMA_URL", EngineSettings.ollama_url),
            ollama_model=_env_str("CE_OLLAMA_MODEL", EngineSettings.ollama_model),
        )

        rag = RagSettings(
            rag_store_path=_env_str("CE_RAG_STORE", RagSettings.rag_store_path),
            rag_top_k=_env_int("CE_RAG_TOPK", RagSettings.rag_top_k),
            rag_max_tokens=_env_int("CE_RAG_MAX_TOKENS", RagSettings.rag_max_tokens),
            chunk_tokens=_env_int("CE_CHUNK_TOKENS", RagSettings.chunk_tokens),
            overlap_tokens=_env_int("CE_OVERLAP_TOKENS", RagSettings.overlap_tokens),
            sbert_model=_env_str("CE_SBERT_MODEL", RagSettings.sbert_model),
        )

        mem = MemorySettings(
            memory_store_path=_env_str("CE_MEMORY_STORE", MemorySettings.memory_store_path),
        )

        return AppSettings(
            data_store_dir=_env_str("CE_DATA_STORE", "./data"),
            engine=eng,
            rag=rag,
            memory=mem,
        )
