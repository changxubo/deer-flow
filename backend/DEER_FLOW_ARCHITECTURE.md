# DeerFlow Backend 架构分析

## 整体架构：双层分离式设计

```
┌─────────────────────────────────────────────────────┐
│                    App Layer (app/)                  │
│  ┌───────────────────┐  ┌─────────────────────────┐ │
│  │  Gateway (FastAPI) │  │  Channels (IM集成)       │ │
│  │  13个Router        │  │  Feishu/Slack/Telegram  │ │
│  │  Services + Deps   │  │  MessageBus Pub/Sub     │ │
│  └────────┬──────────┘  └──────────┬──────────────┘ │
├───────────┼────────────────────────┼────────────────┤
│           ▼            Harness Layer (deerflow/)     │
│  ┌────────────────────────────────────────────────┐  │
│  │  Agent System    │  Runtime     │  Config      │  │
│  │  ├─ LeadAgent    │  ├─ Runs     │  ├─ AppConfig│  │
│  │  ├─ 14+ 中间件   │  ├─ Stream   │  ├─ 20+模块  │  │
│  │  ├─ Memory       │  ├─ Store    │  └─ YAML+env │  │
│  │  └─ Checkpointer │  └─ Bridge   │              │  │
│  ├──────────────────┼──────────────┼──────────────┤  │
│  │  Tools           │  Subagents   │  Models      │  │
│  │  ├─ 7 Builtins   │  ├─ Executor │  ├─ Factory  │  │
│  │  ├─ Community     │  └─ Registry │  └─ Patched  │  │
│  │  └─ MCP          │              │    Providers  │  │
│  ├──────────────────┼──────────────┼──────────────┤  │
│  │  Sandbox         │  Guardrails  │  Skills      │  │
│  │  ├─ Provider     │  ├─ Provider │  ├─ Loader   │  │
│  │  └─ Local/AIO    │  └─ Allowlist│  └─ SKILL.md │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**核心架构规则**：`app` 可以导入 `deerflow`，但 `deerflow` 绝不导入 `app`。此边界由 `tests/test_harness_boundary.py` 在 CI 中强制执行。

---

## 目录结构

```
backend/
  app/                              # 应用层（不可独立发布，import: app.*）
    gateway/                        # FastAPI Gateway API
      app.py                       # FastAPI 应用工厂
      config.py                    # Gateway 配置（host, port, CORS）
      deps.py                      # 依赖注入（单例访问器）
      services.py                  # 服务层（Run 生命周期, SSE 格式化）
      routers/                     # 路由模块（13个Router）
    channels/                       # IM 平台集成（飞书, Slack, Telegram）
  packages/
    harness/                        # 可发布框架包（deerflow-harness）
      deerflow/                     # 核心 Harness（import: deerflow.*）
        agents/                     # Agent 系统
          lead_agent/               # 主 Agent 工厂 + Prompt
          middlewares/              # 14+ 中间件组件
          memory/                   # 记忆提取、队列、Prompt
          checkpointer/            # Checkpoint 持久化提供者
        config/                     # 20+ 配置模块
        models/                     # LLM 模型工厂 + 修补提供者
        tools/                      # 工具系统
          builtins/                 # 内置工具（7个）
        community/                  # 社区/第三方工具集成
        sandbox/                    # 沙箱执行系统
          local/                    # 本地文件系统提供者
        subagents/                  # 子 Agent 委托系统
          builtins/                 # 内置子 Agent
        runtime/                    # Run 管理、流式传输、Store
          runs/                     # RunManager, RunRecord, Schemas
          store/                    # 持久化 Store 提供者
          stream_bridge/            # SSE 流式抽象
        mcp/                        # Model Context Protocol 集成
        skills/                     # Skills 发现、加载、解析
        guardrails/                 # 工具调用前授权
        reflection/                 # 动态模块/类加载
        uploads/                    # 文件上传管理
        utils/                      # 工具库（网络、可读性、文件转换）
        client.py                   # 嵌入式 Python 客户端（DeerFlowClient）
  tests/                            # ~80 个测试文件
  config.yaml                       # 主应用配置
  langgraph.json                    # LangGraph Server 入口
  pyproject.toml                    # 项目元数据 + 依赖
```

---

## 运行时进程

项目包含两个独立的运行时进程：

| 进程 | 端口 | 入口文件 | 职责 |
|------|------|----------|------|
| **LangGraph Server** | 2024 | `langgraph.json` → `deerflow.agents:make_lead_agent` | Agent 图执行引擎 |
| **Gateway API** | 8001 | `app/gateway/app.py` → `create_app()` | REST API + SSE 推流 |

另有嵌入式客户端 `DeerFlowClient`（`deerflow/client.py`），可直接在进程内调用，无需 HTTP。

---

## 核心设计模式

### 1. Factory Pattern（工厂模式）

全局贯穿的核心模式，用于根据配置动态创建对象。

| 工厂 | 文件 | 职责 |
|------|------|------|
| Application Factory | `app/gateway/app.py` → `create_app()` | 构建 FastAPI 实例，注册全部路由 |
| Agent Factory | `agents/lead_agent/agent.py` → `make_lead_agent()` | 根据运行时配置动态创建 Agent |
| Model Factory | `models/factory.py` → `create_chat_model()` | 通过反射从配置实例化 LLM |
| Checkpointer Factory | `agents/checkpointer/async_provider.py` → `make_checkpointer()` | 创建 Memory/SQLite/Postgres Checkpointer |
| Store Factory | `runtime/store/provider.py` → `make_store()` | 创建持久化后端 |
| Stream Bridge Factory | `runtime/stream_bridge/` → `make_stream_bridge()` | 创建流式桥接 |

### 2. Provider Pattern（提供者模式）— Strategy + Abstract Factory

可插拔的实现切换，是系统可扩展性的关键。

**SandboxProvider**（`sandbox/sandbox_provider.py`）：
- 抽象生命周期：`acquire()` → `get()` → `release()`
- 实现：`LocalSandboxProvider`（本地文件系统）、`AioSandboxProvider`（Docker）

**GuardrailProvider**（`guardrails/provider.py`）：
- 基于 Protocol 的结构化类型（非继承）
- 接口：`evaluate()` / `aevaluate()`
- 实现：`AllowlistProvider`、OAP Policy Provider

**Checkpointer & Store Provider**：
- 三种统一后端：InMemory / AsyncSQLite / AsyncPostgres
- 由配置文件中的 `type` 字段决定使用哪种实现

### 3. Middleware / Chain of Responsibility（中间件 / 职责链）

Agent 中间件系统是本项目**最核心的设计**，14+ 个中间件按严格顺序组成处理链：

```
ThreadData → Uploads → Sandbox → DanglingToolCall → Guardrail
→ ToolErrorHandling → Summarization → Todo → TokenUsage
→ Title → Memory → ViewImage → DeferredToolFilter
→ SubagentLimit → LoopDetection → Clarification
```

每个中间件实现 `AgentMiddleware` 基类，提供 4 个生命周期钩子：

| 钩子 | 时机 | 用途 |
|------|------|------|
| `before_agent()` | Agent 调用前 | 初始化状态、注入上下文 |
| `after_model()` | LLM 响应后 | 修改/检查模型输出 |
| `wrap_tool_call()` | 工具调用时 | 拦截、授权、包装工具执行 |
| `after_agent()` | Agent 回合后 | 清理资源、收集指标 |

通过 `@Next` / `@Prev` 装饰器支持在锚点中间件的相对位置插入新中间件。

### 4. Singleton + 智能缓存

全局状态管理采用带自动失效的单例模式：

```python
# 典型模式
_instance: T | None = None

def get_xxx() -> T:       # 懒加载获取
def reset_xxx() -> None:  # 显式重置
def set_xxx(v: T) -> None # 测试注入
```

- `get_app_config()` — 基于文件 mtime 自动重载
- `get_sandbox_provider()` — 带显式 reset/shutdown
- `get_store()` — 带 Context Manager 生命周期

### 5. Observer / Pub-Sub（观察者 / 发布订阅）

IM 通道系统使用异步发布/订阅中枢：

```
Channel (Feishu/Slack/Telegram)
    │ InboundMessage
    ▼
MessageBus (队列)
    │
    ▼
ChannelManager._dispatch_loop() (消费者)
    │ OutboundMessage
    ▼
Channel.send() (回调)
```

- `Channel` 抽象基类定义生命周期：`start` / `stop` / `send`
- 子类实现平台特定行为（飞书、Slack、Telegram）

### 6. Reflection / Service Locator（反射 / 服务定位）

`reflection/resolvers.py` 提供从字符串路径动态加载类和实例的能力：

```python
# 从字符串路径解析变量
resolve_variable("deerflow.community.tavily.tools:web_search_tool")

# 从字符串路径解析并验证类层次
resolve_class("langchain_openai:ChatOpenAI", BaseChatModel)
```

配置文件 `config.yaml` 中所有 `use:` 字段均通过此机制解析，实现零代码扩展。

### 7. Declarative Feature Flags（声明式特性标志）

`RuntimeFeatures`（`agents/features.py`）提供声明式中间件组合：

```python
@dataclass
class RuntimeFeatures:
    sandbox: bool | AgentMiddleware = True       # True=默认实现
    memory: bool | AgentMiddleware = False       # False=禁用
    summarization: Literal[False] | AgentMiddleware = False
    todo: bool | AgentMiddleware = False
    token_usage: bool | AgentMiddleware = False
    ...
```

每个字段接受三种值：
- `True` — 使用默认中间件实现
- `False` — 禁用该特性
- `AgentMiddleware` 实例 — 使用自定义替换

### 8. Bridge Pattern（桥接模式）

`StreamBridge`（`runtime/stream_bridge/base.py`）解耦 Agent Worker（生产者）与 SSE 端点（消费者）：

```
Agent Worker  ──produce──▶  StreamBridge  ──consume──▶  SSE Endpoint
   (异步)                    (缓冲/路由)                  (HTTP 响应)
```

### 9. Template Method Pattern（模板方法模式）

- `Channel` 基类定义骨架算法（消息接收 → 分发 → 响应），子类实现平台特定步骤
- `Sandbox` 基类定义抽象操作接口，具体实现由 Local/AIO 提供
- `AgentMiddleware` 定义钩子点，子类覆盖特定钩子

### 10. Command Pattern（命令模式）

- `SubagentExecutor` 将任务执行封装为对象，拥有 `execute()` / `execute_async()` 方法
- `RunRecord` 封装 Run 状态，包含 `abort_event`、`task`、状态转换逻辑

---

## API 层结构

Gateway API 使用 FastAPI，包含 13 个路由模块：

| 路由文件 | 前缀 | 描述 |
|----------|------|------|
| `routers/models.py` | `/api/models` | AI 模型列表/详情 |
| `routers/mcp.py` | `/api/mcp` | MCP 服务器配置 CRUD |
| `routers/memory.py` | `/api/memory` | 记忆 CRUD、事实管理、导入导出 |
| `routers/skills.py` | `/api/skills` | Skills 列表、启用/禁用、安装 |
| `routers/artifacts.py` | `/api/threads/{id}/artifacts` | Agent 输出文件服务 |
| `routers/uploads.py` | `/api/threads/{id}/uploads` | 文件上传/列表/删除 |
| `routers/threads.py` | `/api/threads` | 线程完整 CRUD、状态、历史 |
| `routers/agents.py` | `/api/agents` | 自定义 Agent 管理 |
| `routers/suggestions.py` | `/api/threads/{id}/suggestions` | 后续问题生成 |
| `routers/channels.py` | `/api/channels` | IM 通道管理 |
| `routers/assistants_compat.py` | — | LangGraph Assistants API 兼容层 |
| `routers/thread_runs.py` | `/api/threads/{id}/runs` | 线程级 Run 生命周期 |
| `routers/runs.py` | `/api/runs` | 无状态 Run 生命周期 |

**依赖注入**：`deps.py` 将单例存储在 `app.state` 上（stream_bridge、checkpointer、store、run_manager），提供 getter 函数，依赖不可用时返回 HTTP 503。

**服务层**：`services.py` 集中处理 Run 生命周期逻辑（创建 Run、SSE 格式化、消费 StreamBridge 事件）。路由层为薄 HTTP 处理器，委托给服务层。

---

## 配置管理

### 配置加载优先级

```
显式路径 > DEER_FLOW_CONFIG_PATH 环境变量 > ./config.yaml > ../config.yaml
```

### 核心特性

- **Pydantic BaseModel 驱动**：20+ 配置子模块，每个 YAML section 加载到独立 Pydantic 模型
- **环境变量解析**：值以 `$` 开头时自动通过 `os.getenv()` 递归解析
- **mtime 自动重载**：`get_app_config()` 检查文件修改时间，变更后自动刷新
- **配置版本化**：`config_version` 字段与 `config.example.yaml` 比对，升级时发出警告
- **扩展配置**：`extensions_config.json`（MCP 服务器 + Skills 状态），同样支持 mtime 缓存

---

## 数据层

项目**不使用传统 ORM**，采用 LangGraph 原生持久化方案：

### 状态持久化

| 组件 | 后端选项 | 用途 |
|------|----------|------|
| Checkpointer | InMemory / AsyncSQLite / AsyncPostgres | Agent 对话状态快照 |
| Store | InMemory / SQLite / Postgres | 线程元数据存储 |

### 文件存储

| 文件 | 用途 | 写入策略 |
|------|------|----------|
| `memory.json` | 记忆事实 | 原子写入（temp + rename） |
| `channels_store.json` | 通道-线程映射 | JSON 文件 |
| `extensions_config.json` | MCP + Skills 状态 | JSON 文件 |
| `.deer-flow/threads/{id}/` | 线程工作区、上传、输出 | 目录结构 |

### 线程状态定义

```python
class ThreadState(AgentState):
    sandbox: SandboxState | None
    thread_data: ThreadDataState | None
    title: str | None
    artifacts: Annotated[list[str], merge_artifacts]
    todos: list | None
    uploaded_files: list[dict] | None
    viewed_images: Annotated[dict[str, ViewedImageData], merge_viewed_images]
```

---

## 第三方集成

### LLM 提供者（通过反射/工厂抽象）

| 提供者 | 类路径 | 说明 |
|--------|--------|------|
| OpenAI / OpenRouter | `langchain_openai:ChatOpenAI` | 通用 OpenAI 兼容 API |
| DeepSeek | `deerflow.models.patched_deepseek:PatchedChatDeepSeek` | 定制包装 |
| OpenAI (Patched) | `deerflow.models.patched_openai:PatchedChatOpenAI` | 定制包装 |
| Minimax | `deerflow.models.patched_minimax:PatchedChatMinimax` | 定制包装 |
| Claude | `deerflow.models.claude_provider` | Claude 专用提供者 |
| Codex | `deerflow.models.openai_codex_provider:CodexChatModel` | Codex Responses API |

所有提供者通过 `config.yaml` 中的 `use:` 字段 + `resolve_class()` 动态加载。

### 社区工具

| 模块 | 集成服务 | 功能 |
|------|----------|------|
| `community/tavily/` | Tavily API | 网页搜索与抓取 |
| `community/jina_ai/` | Jina Reader API | 网页抓取 + 可读性提取 |
| `community/firecrawl/` | Firecrawl API | 网页爬取 |
| `community/image_search/` | DuckDuckGo | 图片搜索 |
| `community/ddg_search/` | DuckDuckGo | 通用搜索 |
| `community/infoquest/` | 自定义接口 | 搜索客户端 |
| `community/aio_sandbox/` | Docker | 容器化沙箱执行 |

### MCP（Model Context Protocol）

- 使用 `langchain-mcp-adapters` 进行多服务器管理
- 支持 stdio / SSE / HTTP 三种传输协议
- 带 mtime 缓存和 OAuth 认证支持

### IM 通道

| 通道 | SDK | 抽象 |
|------|-----|------|
| 飞书 | `lark-oapi` | `Channel` 基类 + `MessageBus` |
| Slack | `slack-sdk` | `Channel` 基类 + `MessageBus` |
| Telegram | `python-telegram-bot` | `Channel` 基类 + `MessageBus` |

---

## 架构特点总结

1. **严格的分层边界** — Harness 包可独立发布，App 层为部署适配，CI 强制边界检查
2. **中间件驱动** — 通过 14+ 可组合中间件实现横切关注点，支持声明式启用/禁用/替换
3. **反射 + 声明式配置** — 所有核心组件通过 YAML 配置 + 字符串路径动态装配，实现零代码扩展
4. **Provider 抽象** — 关键基础设施（沙箱、存储、护栏、检查点）均可插拔替换
5. **事件驱动的 IM 集成** — Pub/Sub 模式解耦多平台通道
6. **双进程架构** — LangGraph Server 负责 Agent 执行，Gateway API 负责 HTTP 服务，职责清晰
