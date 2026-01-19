from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from chat_engine.app.settings import AppSettings
from chat_engine.app.wiring import build_bundle

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("chat_engine")

app = FastAPI(title="chat_engine")
settings = AppSettings.from_env()

UPLOADS_DIR = Path("./uploads")
ALLOWED_EXTS = {".txt", ".md", ".pdf"}


class ChatRequest(BaseModel):
    conversation_id: str
    user_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    answer: str
    meta: Dict[str, Any]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    bundle = build_bundle(settings, user_id=req.user_id)
    answer, meta = bundle.engine.handle_user_message_ex(req.conversation_id, req.message)
    log.info(json.dumps({"event": "chat", **meta}, ensure_ascii=False))
    return ChatResponse(answer=answer, meta=meta)


@app.get("/conversations/{cid}")
def get_conversation(cid: str):
    bundle = build_bundle(settings, user_id="default")
    convo = bundle.engine.repo.load(cid)

    def m2d(m):
        return {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
            "meta": m.meta,
        }

    return {
        "conversation_id": convo.conversation_id,
        "messages": [m2d(m) for m in convo.messages],
        "settings": {
            "max_context_tokens": convo.max_context_tokens,
            "reserved_for_reply_tokens": convo.reserved_for_reply_tokens,
        },
    }


@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    bundle = build_bundle(settings, user_id=user_id)
    facts = bundle.memory_store.get_facts(user_id)
    facts.sort(key=lambda f: (float(f.confidence), f.updated_at), reverse=True)
    return [
        {
            "fact_id": f.fact_id,
            "key": f.key,
            "value": f.value,
            "confidence": f.confidence,
            "updated_at": f.updated_at.isoformat(),
            "source_message_id": f.source_message_id,
        }
        for f in facts
    ]


@app.delete("/memory/{user_id}/{fact_id}")
def delete_fact(user_id: str, fact_id: str):
    bundle = build_bundle(settings, user_id=user_id)
    ok = bundle.memory_store.delete_fact(user_id, fact_id)
    return {"deleted": ok}


@app.post("/documents/upload")
async def upload_document(
    user_id: str = Query(default="default"),
    replace: bool = Query(default=True),
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dst = UPLOADS_DIR / safe_name

    try:
        data = await file.read()
        dst.write_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    bundle = build_bundle(settings, user_id=user_id)

    if replace:
        bundle.rag_store.delete_by_source(str(dst))

    try:
        n = bundle.indexer.ingest_paths([str(dst)])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"stored_as": str(dst), "ingested_chunks": n, "store_size": bundle.rag_store.count()}
