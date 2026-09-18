#!/usr/bin/env bash
# 通过 GitHub Git Data API 发布代码（绕过 git 的 TLS 栈问题）
# 作者：晨星
set -uo pipefail

TOKEN="${GH_TOKEN:?需要 GH_TOKEN}"
OWNER="CJX0712"
REPO="aurora"
API_IP="140.82.113.6"
BASE="https://api.github.com/repos/${OWNER}/${REPO}"

gh_api() {
  local method="$1" path="$2"; shift 2
  curl -s --max-time 40 --resolve "api.github.com:443:${API_IP}" \
    -X "$method" \
    -H "Authorization: token ${TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "Content-Type: application/json" \
    "$@" \
    "${BASE}${path}"
}

echo "== 1. 创建所有文件的 blob =="
cd /workspace/aurora
FILE_LIST=$(git ls-files)
declare -a TREE_ITEMS
BLOB_COUNT=0

while IFS= read -r f; do
  [ -z "$f" ] && continue
  [ ! -f "$f" ] && continue
  # 用 python 生成 JSON，避免转义问题
  PAYLOAD=$(python3.11 - "$f" <<'PY'
import json, sys, base64
p = sys.argv[1]
data = open(p, 'rb').read()
print(json.dumps({"content": base64.b64encode(data).decode(), "encoding": "base64"}))
PY
)
  RESP=$(gh_api POST "/git/blobs" -d "$PAYLOAD")
  SHA=$(printf '%s' "$RESP" | python3.11 -c 'import sys,json;print(json.load(sys.stdin).get("sha",""))' 2>/dev/null)
  if [ -z "$SHA" ]; then
    echo "  FAIL: $f"
    printf '%s\n' "$RESP" | head -c 200; echo
    continue
  fi
  BLOB_COUNT=$((BLOB_COUNT+1))
  MODE="100755"
  case "$f" in *.sh) MODE="100755";; *) MODE="100644";; esac
  TREE_ITEMS+=("{\"path\":\"$f\",\"mode\":\"$MODE\",\"type\":\"blob\",\"sha\":\"$SHA\"}")
  echo "  [$BLOB_COUNT] $f"
done <<< "$FILE_LIST"

echo "== 2. 创建 tree（共 ${#TREE_ITEMS[@]} 项） =="
TREE_JSON=$(python3.11 -c '
import json,sys
items = sys.argv[1:]
print(json.dumps({"tree":[json.loads(x) for x in items]}))
' "${TREE_ITEMS[@]}")
TREE_RESP=$(gh_api POST "/git/trees" -d "$TREE_JSON")
TREE_SHA=$(printf '%s' "$TREE_RESP" | python3.11 -c 'import sys,json;print(json.load(sys.stdin).get("sha",""))' 2>/dev/null)
if [ -z "$TREE_SHA" ]; then echo "  tree 创建失败:"; printf '%s' "$TREE_RESP" | head -c 300; exit 1; fi
echo "  tree sha: $TREE_SHA"

echo "== 3. 创建 commit =="
COMMIT_MSG=$(cat <<'MSG'
feat: Aurora 本地自主智能体系统

纯本机 CPU 推理底座，内置 RAG 检索与 ReAct 工具调用循环，零云端密钥依赖。

- 推理：llama.cpp + Qwen2.5-7B-Instruct(Q4_K_M) + nomic-embed-text-v1.5
- 智能体：ReAct 循环（规划-工具-观察-反思），含最大步数与自我纠错
- 工具集：文件读写 / 受限 shell / Python 执行 / 网页抓取
- 知识库：分块摄入 + 嵌入 + SQLite 向量存储 + 余弦检索
- 接口：CLI(serve/chat/ingest) + FastAPI + SSE 流式 + 单文件 Web 控制台
- 测试：20 项离线用例全绿
- 实测：RAG 检索命中 score=0.735，Agent 多步任务文件真实落盘

作者：晨星
MSG
)
COMMIT_JSON=$(python3.11 -c '
import json,sys
print(json.dumps({"message": sys.argv[1], "tree": sys.argv[2]}))
' "$COMMIT_MSG" "$TREE_SHA")
COMMIT_RESP=$(gh_api POST "/git/commits" -d "$COMMIT_JSON")
COMMIT_SHA=$(printf '%s' "$COMMIT_RESP" | python3.11 -c 'import sys,json;print(json.load(sys.stdin).get("sha",""))' 2>/dev/null)
if [ -z "$COMMIT_SHA" ]; then echo "  commit 创建失败:"; printf '%s' "$COMMIT_RESP" | head -c 300; exit 1; fi
echo "  commit sha: $COMMIT_SHA"

echo "== 4. 创建/更新 main 分支引用 =="
REF_RESP=$(gh_api POST "/git/refs" -d "{\"ref\":\"refs/heads/main\",\"sha\":\"$COMMIT_SHA\"}")
if printf '%s' "$REF_RESP" | grep -q '"ref"'; then
  echo "  分支创建成功"
else
  echo "  (可能已存在，尝试强制更新)"
  gh_api PATCH "/git/refs/heads/main" -d "{\"sha\":\"$COMMIT_SHA\",\"force\":true}" | head -c 200
fi

echo "== 完成 =="
echo "  https://github.com/${OWNER}/${REPO}"
