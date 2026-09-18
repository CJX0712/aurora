#!/usr/bin/env bash
# 一键发布 Aurora 到 GitHub（晨星）
# 用法：
#   1) 先登录：  gh auth login        （或 export GH_TOKEN=ghp_xxx）
#   2) 再执行：  bash publish.sh [owner/repo]
# 默认仓库名：aurora
set -euo pipefail

REPO="${1:-aurora}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "== 1. 检查 GitHub 鉴权 =="
if ! gh auth status >/dev/null 2>&1; then
  echo "未登录 GitHub。请先执行其一："
  echo "  gh auth login"
  echo "  export GH_TOKEN=ghp_xxxxxxxx"
  exit 1
fi
OWNER="$(gh api user --jq .login)"
echo "已登录：$OWNER"

echo "== 2. 确认本地提交 =="
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "chore: update"
fi
git log --oneline | head -3

echo "== 3. 创建远端仓库并推送 =="
if gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
  echo "仓库已存在：$OWNER/$REPO"
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$OWNER/$REPO.git"
  git push -u origin HEAD:main
else
  gh repo create "$REPO" --public --source=. --remote=origin --push \
    --description "Aurora - 本地自主智能体系统：纯本机 CPU 推理，内置 RAG 与工具调用循环" \
    --license MIT
fi

echo "== 4. 完成 =="
gh repo view "$OWNER/$REPO" --json url --jq .url
