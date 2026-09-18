"""E2E check against a running Aurora server: streaming chat + RAG retrieval.

Usage: python3.11 e2e_server.py [base_url]
"""

import json
import sys
import time

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7680"


def main() -> int:
    print("== health ==", flush=True)
    h = requests.get(f"{BASE}/api/health", timeout=30).json()
    print("   ", h, flush=True)

    print("== ingest ==", flush=True)
    ing = requests.post(f"{BASE}/api/knowledge", json={"path": "/tmp/kbdocs"}, timeout=180).json()
    print("   ", ing, flush=True)

    print("== streaming chat (RAG question) ==", flush=True)
    url = f"{BASE}/api/chat"
    body = {"prompt": "请检索知识库，告诉我 Aurora 的默认服务端口是多少？", "stream": True}
    steps, answer = [], None
    t0 = time.time()
    with requests.post(url, json=body, stream=True, timeout=900) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            payload = json.loads(raw[6:])
            if payload.get("type") == "done":
                answer = payload.get("answer")
                print(f"   [done] tool_calls={payload.get('tool_calls')}", flush=True)
            else:
                steps.append(payload)
                kind = payload.get("type")
                text = (payload.get("text") or "")[:110].replace("\n", " ")
                print(f"   [{kind}] tool={payload.get('tool')} ok={payload.get('ok')} :: {text}", flush=True)

    print(f"\n== RESULT ({time.time()-t0:.0f}s) ==", flush=True)
    print("   steps:", len(steps), flush=True)
    print("   answer:", answer, flush=True)

    ok = bool(answer) and steps and answer is not None and "7680" in (answer or "")
    print("\nE2E_SERVER:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
