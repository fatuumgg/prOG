from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from chat_engine.app.settings import AppSettings
from chat_engine.app.wiring import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", required=True)
    parser.add_argument("--uid", default="default")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--store", default=None, help="Override data store dir")
    parser.add_argument("--rag-store", default=None, help="Override rag store path")
    parser.add_argument("--memory-store", default=None, help="Override memory store path")
    parser.add_argument("--max-context", type=int, default=None)
    parser.add_argument("--reserve-output", type=int, default=None)
    parser.add_argument("--rag-max-tokens", type=int, default=None)
    parser.add_argument("--rag-top-k", type=int, default=None)

    parser.add_argument("--ingest", nargs="+", default=None, help="Paths to ingest (txt/md/pdf)")
    parser.add_argument("--ingest-replace", action="store_true", help="Delete old chunks by source before ingest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = AppSettings.from_env()

    if args.store is not None:
        settings = replace(settings, data_store_dir=args.store)
    if args.rag_store is not None:
        settings = replace(settings, rag=replace(settings.rag, rag_store_path=args.rag_store))
    if args.memory_store is not None:
        settings = replace(settings, memory=replace(settings.memory, memory_store_path=args.memory_store))

    eng = settings.engine
    rag = settings.rag
    if args.max_context is not None:
        eng = replace(eng, max_context_tokens=args.max_context)
    if args.reserve_output is not None:
        eng = replace(eng, reserve_output_tokens=args.reserve_output)
    if args.rag_max_tokens is not None:
        rag = replace(rag, rag_max_tokens=args.rag_max_tokens)
    if args.rag_top_k is not None:
        rag = replace(rag, rag_top_k=args.rag_top_k)
    settings = replace(settings, engine=eng, rag=rag)

    bundle = build_bundle(settings, user_id=args.uid)
    engine = bundle.engine
    mem_store = bundle.memory_store

    if args.ingest is not None:
        if args.ingest_replace:
            for p in args.ingest:
                bundle.rag_store.delete_by_source(str(Path(p)))
        n = bundle.indexer.ingest_paths(args.ingest)
        print(f"Ingested chunks: {n}. Store size: {bundle.rag_store.count()}")
        return

    print(f"Conversation: {args.cid}")
    print("Type /exit to quit.")
    print("Memory: /memory | /forget <fact_id> | /forget-key <key> | /forget-all\n")

    while True:
        user_text = input("you> ").strip()
        if not user_text:
            continue
        if user_text == "/exit":
            break

        if user_text == "/memory":
            facts = mem_store.get_facts(args.uid)
            if not facts:
                print("bot> (memory empty)\n")
            else:
                facts.sort(key=lambda f: (float(f.confidence), f.updated_at), reverse=True)
                print("bot> memory:")
                for f in facts:
                    print(f"  {f.fact_id} | {f.key} = {f.value} (conf={f.confidence:.2f})")
                print()
            continue

        if user_text.startswith("/forget "):
            fid = user_text.split(" ", 1)[1].strip()
            ok = mem_store.delete_fact(args.uid, fid)
            print(f"bot> forget: {'ok' if ok else 'not found'}\n")
            continue

        if user_text.startswith("/forget-key "):
            key = user_text.split(" ", 1)[1].strip()
            n = mem_store.delete_by_key(args.uid, key)
            print(f"bot> forget-key removed: {n}\n")
            continue

        if user_text == "/forget-all":
            mem_store.clear(args.uid)
            print("bot> memory cleared\n")
            continue

        answer, meta = engine.handle_user_message_ex(args.cid, user_text)
        print(f"bot> {answer}\n")

        if args.debug:
            print("debug> " + json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
