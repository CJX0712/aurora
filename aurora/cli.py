"""Command-line entry points for Aurora.

  python -m aurora.cli serve   # start the HTTP server (loads models in-process)
  python -m aurora.cli chat    # interactive REPL
  python -m aurora.cli ingest  # ingest a folder/file into the knowledge base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AgentConfig, load_config
from .llm import LocalBackend
from .memory import Memory
from .rag import KnowledgeBase
from .tools import Toolbox


def _build(cfg: AgentConfig):
    backend = LocalBackend(cfg.model_config)
    kb = KnowledgeBase(cfg, backend)
    toolbox = Toolbox.default(cfg.workspace, cfg.knowledge_dir, kb.retrieve)
    return backend, kb, toolbox


def cmd_serve(args) -> None:
    from .server import build_app
    import uvicorn

    cfg = load_config()
    cfg.port = args.port or cfg.port
    cfg.host = args.host or cfg.host
    app = build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


def cmd_chat(args) -> None:
    cfg = load_config()
    backend, kb, toolbox = _build(cfg)
    from .agent import Agent

    agent = Agent(cfg, backend, toolbox)
    print("Aurora local agent. Type 'exit' to quit.\n")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in ("exit", "quit"):
            break
        if prompt.startswith("/ingest "):
            n = kb.ingest_path(prompt[8:].strip())
            print(f"[ingested {n} chunks]")
            continue
        res = agent.run(prompt)
        print("\naurora> " + res.answer + "\n")


def cmd_ingest(args) -> None:
    cfg = load_config()
    backend, kb, _ = _build(cfg)
    target = args.path or str(cfg.knowledge_dir)
    n = kb.ingest_path(target)
    print(f"ingested {n} chunks from {target}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="aurora", description="Local autonomous agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start HTTP server")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(func=cmd_serve)

    c = sub.add_parser("chat", help="interactive REPL")
    c.set_defaults(func=cmd_chat)

    i = sub.add_parser("ingest", help="ingest knowledge")
    i.add_argument("path", nargs="?", default=None)
    i.set_defaults(func=cmd_ingest)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
