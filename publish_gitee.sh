#!/usr/bin/env bash
# 一键发布 Aurora 到 Gitee（码云）— 作者：晨星
#
# 用法：
#   export GITEE_TOKEN=<你的私人令牌>     # 需勾选 projects 权限
#   bash publish_gitee.sh [仓库名]        # 默认 aurora
#
# 已验证：本沙箱可直连 gitee.com（HTTP 200）
set -euo pipefail

REPO="${1:-aurora}"
TOKEN="${GITEE_TOKEN:-}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ -z "$TOKEN" ]; then
  echo "ERROR: 未设置 GITEE_TOKEN"
  echo "请先执行：export GITEE_TOKEN=<你的 Gitee 私人令牌>"
  exit 1
fi

echo "== 1. 校验令牌 / 获取当前用户 =="
USER_JSON="$(curl -s --max-time 30 "https://gitee.com/api/v5/user?access_token=${TOKEN}")"
LOGIN="$(printf '%s' "$USER_JSON" | python3.11 -c 'import sys,json;print(json.load(sys.stdin).get("login",""))' 2>/dev/null || true)"
if [ -z "$LOGIN" ]; then
  echo "令牌校验失败，Gitee 返回："
  printf '%s\n' "$USER_JSON" | head -5
  exit 1
fi
echo "已登录 Gitee 账号：$LOGIN"

echo "== 2. 创建仓库（已存在则跳过） =="
CREATE="$(curl -s --max-time 40 -X POST "https://gitee.com/api/v5/user/repos" \
  -d "access_token=${TOKEN}" \
  -d "name=${REPO}" \
  -d "description=Aurora - 本地自主智能体系统：纯本机 CPU 推理，内置 RAG 与工具调用循环" \
  -d "private=false" \
  -d "auto_init=false")"
if printf '%s' "$CREATE" | grep -q '"full_name"'; then
  echo "仓库创建成功：${LOGIN}/${REPO}"
else
  echo "创建返回（可能已存在，继续尝试推送）："
  printf '%s\n' "$CREATE" | head -3
fi

echo "== 3. 配置远端并推送 =="
REMOTE_URL="https://${LOGIN}:${TOKEN}@gitee.com/${LOGIN}/${REPO}.git"
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "chore: update"
fi
git branch -M main 2>/dev/null || true
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi
git push -u origin main

echo "== 4. 完成 =="
echo "仓库地址：https://gitee.com/${LOGIN}/${REPO}"
