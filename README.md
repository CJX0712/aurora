# Aurora

<p align="center">
  <a href="https://github.com/CJX0712/aurora/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/aurora/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/aurora/releases"><img src="https://img.shields.io/github/v/release/CJX0712/aurora?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/aurora/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/aurora" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

**本地自主智能体系统** — 纯本机 CPU 运行，内置 RAG 检索与工具调用循环，零云端密钥依赖。

作者：**晨星**

---

## 这是什么

Aurora 是一个完全离线、可自持运行的 AI 智能体系统。它把当前最成熟的开源技术成果（llama.cpp 推理引擎 + Qwen2.5 开源权重 + 向量检索）组装成一个可用的工程产品，而不是从零训练模型：

- **推理层**：`llama.cpp`（`llama-cpp-python`），纯 CPU 推理，无 GPU 依赖
- **模型层**：Qwen2.5-7B-Instruct（Q4_K_M 量化）+ nomic-embed-text-v1.5 嵌入
- **智能体层**：ReAct 循环（规划 → 工具选择 → 执行 → 观察 → 反思），支持流式输出、上下文窗口管理、最大步数保护与自我纠错
- **工具层**：文件读写、受限 shell 执行、Python 执行、网页抓取
- **知识层**：本地嵌入 + SQLite 向量存储 + 余弦检索 + 分块摄入
- **接口层**：CLI 交互 + 单文件 Web 控制台（内联 CSS/JS，零外部依赖）

## 安装

```bash
pip install llama-cpp-python   # 首次需编译，建议限制并行度避免 OOM
pip install fastapi uvicorn numpy pypdf
```

> 编译提示：32 核以上机器建议 `CMAKE_BUILD_PARALLEL_LEVEL=8`，否则并行编译可能触发 OOM，
> 进而产生损坏的 `.so` 导致链接阶段 `Input/output error`。

## 准备模型

```bash
mkdir -p ~/.aurora/models && cd ~/.aurora/models

# 聊天模型（分片，llama.cpp 需按 -00001-of-00002 命名加载分片1，自动拼接）
curl -L -O https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/master/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
curl -L -O https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/master/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf

# 嵌入模型
curl -L -O https://modelscope.cn/models/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/master/nomic-embed-text-v1.5.Q8_0.gguf
```

## 运行

```bash
# CLI
python -m aurora.cli

# Web 控制台（默认 http://127.0.0.1:7680）
python -m aurora.server
```

## 配置

全部通过环境变量或 `.env` 配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AURORA_CHAT_MODEL` | `~/.aurora/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | 聊天模型路径（分片1） |
| `AURORA_EMBED_MODEL` | `~/.aurora/models/nomic-embed-text-v1.5.Q8_0.gguf` | 嵌入模型路径 |
| `AURORA_THREADS` | CPU 核数 | 推理线程数 |
| `AURORA_GPU_LAYERS` | `0` | 卸载到 GPU 的层数（0 = 纯 CPU） |
| `AURORA_CHAT_CTX` | `8192` | 聊天上下文长度 |
| `AURORA_MAX_STEPS` | `12` | 单任务最大推理步数 |
| `AURORA_PORT` | `7680` | 服务端口 |
| `AURORA_CHAT_API_BASE` | 空 | 设为远端 OpenAI 兼容地址可切换到 HTTP 后端 |

## 测试

```bash
pytest tests/ -q          # 离线单测，不依赖模型权重
python smoke.py           # 真模型端到端冒烟
```

## 架构

```
aurora/
├── config.py      配置加载（环境变量 / .env）
├── types.py       核心数据结构与协议
├── llm.py         LLM 后端（本机 llama.cpp / 远端 HTTP）
├── tools.py       工具集（文件 / shell / python / web）
├── rag.py         知识库（分块 + 嵌入 + SQLite 向量检索）
├── memory.py      对话记忆与上下文窗口管理
├── agent.py       ReAct 智能体循环
├── server.py      FastAPI 服务 + Web 控制台托管
└── cli.py         命令行交互
```

## 许可

MIT
