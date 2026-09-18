# Aurora 架构与工程决策记录

> 作者：晨星 · 版本 0.1.0

## 一、定位

Aurora 是一个**完全离线、纯本机 CPU 运行**的自主智能体系统。设计原则是"复用全网顶尖成果 + 工程化组装"，而非从零训练模型——在无 GPU 的硬件条件下，这是唯一可行且务实的技术路线。

## 二、分层架构

```
┌─────────────────────────────────────────────┐
│  接口层  CLI(serve/chat/ingest) · FastAPI · Web 控制台  │
├─────────────────────────────────────────────┤
│  智能体层  Agent(ReAct 循环) · Memory(上下文窗口)      │
├─────────────────────────────────────────────┤
│  能力层  Toolbox(文件/shell/python/web) · KnowledgeBase(RAG) │
├─────────────────────────────────────────────┤
│  模型层  LocalBackend(chat + embed) · HTTPBackend(可选)  │
├─────────────────────────────────────────────┤
│  推理层  llama.cpp (llama-cpp-python) — 纯 CPU       │
└─────────────────────────────────────────────┘
```

**依赖方向单向向下**：接口层依赖智能体层，智能体层依赖能力层与模型层协议，模型层依赖推理层。任何向上依赖都是违规。

## 三、关键协议（可替换点）

系统只依赖两个极小的协议，保证可测试与可替换：

| 协议 | 方法 | 本机实现 | 远端实现 |
|------|------|----------|----------|
| `ChatModel` | `chat(messages, tools) -> ChatResponse` | `LocalBackend`（llama.cpp 进程内） | `_HttpBackend`（OpenAI 兼容） |
| `Embedder` | `embed(texts) -> List[List[float]]` | `LocalBackend.embed`（nomic） | `_HttpBackend.embed` |

测试用 `ScriptedChat` / `FakeEmbedder` 实现同一协议，因此**整个 agent 循环可离线验证**，无需加载 5GB 权重。

## 四、模型选型（ADR）

| 决策 | 选择 | 理由 |
|------|------|------|
| ADR-001 推理引擎 | llama.cpp | 纯 CPU 性能最优的开源方案，无需 GPU、无 Python 训练框架重依赖 |
| ADR-002 聊天模型 | Qwen2.5-7B-Instruct Q4_K_M | 中文能力强、7B 在 32 核 CPU 上响应可接受、Q4_K_M 在质量与体积间平衡 |
| ADR-003 嵌入模型 | nomic-embed-text-v1.5 | 支持非对称检索（query/document 前缀），768 维，139MB 轻量 |
| ADR-004 向量存储 | SQLite + numpy 余弦 | 零外部服务依赖，单文件部署，万级片段性能足够 |
| ADR-005 分片加载 | 保留 `-00001-of-00002` 命名 | llama.cpp 要求按分片命名加载，直接 `cat` 拼接会被判定 `invalid split file name` |

## 五、踩坑记录（pitfalls）

> 全部为实际发生并已修复的问题，可复用于同类项目。

| 坑 | 根因 | 修法 |
|----|------|------|
| `cc1plus Killed signal` + `ld: Input/output error` | 32 核全开并行编译触发 OOM，被杀进程留下损坏 `.so` | `CMAKE_BUILD_PARALLEL_LEVEL=8` 限制并行度 |
| hf-mirror 下载得到 15 字节 "Entry not found" | 该镜像的 GGUF 分片重定向到被墙的 `xethub.hf.co` | 改用 ModelScope 镜像，链路实测直返 200 |
| `invalid split file name` | HF 分片 GGUF 带分片引用，不能 `cat` 合成单文件 | 按原始边界切回分片并保留规范命名 |
| Agent 工具调用数 = 0，文件未生成 | Qwen2.5 的 GGUF 聊天模板不输出结构化 `tool_calls`，而是把调用写成 `<tool_call>{...}</tool_call>` 文本 | 增加文本工具调用兜底解析 `_extract_text_tool_calls`（支持多重调用、双花括号容错、未知工具名过滤） |
| POST 端点报 `Field required: body` | 在函数闭包内定义的 Pydantic 模型无法被 FastAPI 解析 ForwardRef | 改用显式 `Body(...)` 绑定原始 dict |
| 流式端点 `Internal Server Error` | FastAPI 尝试 JSON 序列化自定义 ASGI 可调用对象而非 Response | 直接返回 `starlette.responses.StreamingResponse` |

## 六、安全边界

- **文件工具**：路径限制在工作区根目录内，防止目录穿越
- **shell 工具**：命令白名单 + 超时限制，非交互执行
- **python 工具**：子进程隔离执行，超时保护
- **网络**：默认不开外网，`web_fetch` 为显式工具调用
- **数据**：全部本地磁盘，无任何云端上报

## 七、性能基线（实测）

| 指标 | 数值 | 环境 |
|------|------|------|
| 模型加载 | 3.3 秒 | chat + embed 双模型，32 vCPU |
| 单次生成 | 秒级 | 短回答（`AURORA_OK`） |
| Agent 多步任务 | 约 2 分钟 | 2 次工具调用 + 最终答复，7B CPU 推理 |

## 八、后续演进方向

1. **量化升级**：Q4_K_M → Q5_K_M 提升质量（代价是内存与延迟）
2. **混合后端**：配置 `AURORA_CHAT_API_BASE` 即可切到云端模型，本机小模型兜底
3. **工具扩展**：按需增加工具，协议不变
4. **多会话隔离**：当前为单会话，可扩展 session 管理
