# 作业一：LLM 统一模型调用服务（LLM Gateway）— 设计方案

## 1. 需求概述

构建一个 **生产级的 LLM Gateway（大模型网关）**，作为 Agent 与模型供应商之间的统一中间层。
不仅屏蔽不同厂商 API 的差异，还要解决五个核心问题：
**模型白名单管控、主备自动切换、结构化输出校验、Prompt 模板版本管理、调用审计与成本治理**。

### 核心目标

| 目标 | 说明 |
|------|------|
| **统一接口** | 无论底层是 DeepSeek、OpenAI 还是其他兼容供应商，调用方式完全一致 |
| **密钥隔离** | API Key 只存在于 Gateway 进程，Agent 不接触任何供应商密钥 |
| **主备自动切换** | 主模型超时/异常后自动 fallback 到备用模型，对调用方透明 |
| **结构化输出** | 支持 JSON Schema 约束模型输出格式，并做二次校验 |
| **Prompt 模板管理** | 模板由 Gateway 统一管理，调用方只能选择版本和传变量 |
| **调用审计** | 每次调用记录 request_id、token 用量、延迟、成本，支持成本治理 |
| **全链路校验** | 入口用 Pydantic 校验请求，出口用 jsonschema 校验模型输出 |

## 2. 技术选型

| 项目 | 选择 | 说明 |
|------|------|------|
| 语言 | Python >= 3.14 | 项目要求 |
| 数据校验 | `pydantic >= 2.0` | 请求/响应全链路校验，`extra="forbid"` 防注入 |
| 异步框架 | `asyncio` + `httpx` | 异步调用上游 API |
| Web 框架 | `fastapi` | REST API 暴露 |
| 结构化校验 | `jsonschema` | 模型输出的 JSON Schema 二次校验 |
| OpenAI SDK | `openai >= 1.0.0` | 兼容所有 OpenAI 协议的供应商 |
| 包管理 | pyproject.toml + pip | 标准方式 |

## 3. 整体架构

```
Agent / 客户端
    │  HTTP (POST /v1/llm, POST /v1/llm/stream)
    ▼
┌──────────────────────────────────────────┐
│             LLM Gateway                  │
│                                          │
│  ┌──────────┐  ┌───────────────────────┐ │
│  │ 请求校验  │→│ 模型白名单 + 配置解析  │ │
│  │(Pydantic)│  │ (ModelConfig 映射)    │ │
│  └──────────┘  └───────────┬───────────┘ │
│                             ▼             │
│  ┌──────────────────────────────────────┐ │
│  │   按模型限流 (429 rate_limited)      │ │
│  │   滑动窗口 + asyncio.Lock           │ │
│  └──────────────┬───────────────────────┘ │
│                 ▼                         │
│  ┌──────────────────────────────────────┐ │
│  │   供应商适配器工厂 (create_provider) │ │
│  │   ├─ OpenAICompatibleProvider       │ │
│  │   └─ AnthropicProvider              │ │
│  │   complete() / stream()              │ │
│  └──────────────┬───────────────────────┘ │
│                 ▼                         │
│  ┌──────────────────────────────────────┐ │
│  │  主备切换 + 重试 (call_with_fallback) │ │
│  │  主模型 → 重试1次 → 备用模型          │ │
│  └──────────────┬───────────────────────┘ │
│                 ▼                         │
│  ┌──────────────────────────────────────┐ │
│  │  结构化输出校验 (jsonschema)         │ │
│  │  + TTFT 度量 + 审计记录 (CallTrace)  │ │
│  └──────────────────────────────────────┘ │
└──────────────────────────────────────────┘
    │
    ▼
DeepSeek / OpenAI / Anthropic / 其他兼容供应商
```

### 分层说明

| 层次 | 文件 | 职责 |
|------|------|------|
| **协议层** | `../models.py` | Pydantic 请求/响应模型，`extra="forbid"` 防注入 |
| **配置层** | `../config.py` | 模型白名单 `MODEL_CONFIGS`、Prompt 模板注册表、价格表、限流配置 |
| **适配器层** | `../provider.py` | `BaseProvider` 抽象 + `OpenAICompatibleProvider` + `AnthropicProvider`，工厂函数 `create_provider()` |
| **网关核心** | `../gateway.py` | 主备切换、重试、结构化校验、审计记录、TTFT 度量、按模型限流 |
| **API 层** | `../app.py` | FastAPI REST 端点，429 限流响应映射 |
| **测试** | `../test_gateway.py` | Mock 测试（62 项） + 集成测试 |

## 4. 目录结构

```
1-1-assignment/
├── __init__.py
├── models.py              # Pydantic 请求/响应模型
├── config.py              # 模型白名单、Prompt 模板、价格表
├── provider.py            # OpenAI / Anthropic 双协议适配器
├── gateway.py             # 核心：主备切换、重试、校验、审计
├── app.py                 # FastAPI REST API（统一入口）
├── test_gateway.py        # Mock 测试（62 项）
├── test_gateway_e2e.py    # 集成测试（真实 API）
└── doc/                   # 设计文档
    ├── 01-design.md
    └── README.md
```

## 5. 核心设计

### 5.1 统一请求/响应协议 (`../models.py`)

使用 Pydantic `BaseModel`，全部启用 `extra="forbid"`，拒绝未定义字段。

#### 请求 — `LLMRequest`

```python
class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str                              # 平台模型名（如 "general-primary"）
    messages: list[Message]                 # 消息列表
    stream: bool = False                    # 是否流式
    response_schema: dict | None = None     # 结构化输出 JSON Schema
    response_format: dict | None = None     # 兼容 OpenAI 风格的 response_format
    timeout_seconds: float = 30             # 超时时间（全局预算）
    max_tokens: int | None = None            # 最大输出 Token 数
    prompt: PromptSelection | None = None   # Prompt 模板选择
```

**组合校验**（`model_validator`）：
- `stream=True` 与 `response_schema` 不能同时使用（流式无法做 JSON 校验）
- `response_format.type` 只允许 `json_schema` / `json_object`（不支持 `text`），未知 type 直接报错
- `json_schema` 模式必须包含 `schema` 字段，缺失则报错
- `response_format` 归一化后合并到 `response_schema`，`response_schema` 优先不被覆盖
- **优先级说明**：当 `response_format` 和 `response_schema` 同时传入时，`response_format.type` 决定 provider 的行为模式（json_object / json_schema），`response_schema` 提供 Schema 内容。validator 层只做单向保护：显式 `response_schema` 不被 `response_format.json_schema.schema` 覆盖

#### 响应 — `LLMResponse`

```python
class LLMResponse(BaseModel):
    request_id: str         # UUID 唯一请求标识
    model: str              # 实际使用的供应商模型名
    content: str            # 模型原文
    parsed: dict | None     # 结构化输出的解析结果
    usage: Usage            # Token 用量（input/output/total）
    latency_ms: int         # 请求延迟（毫秒）
    attempts: int           # 尝试次数（含重试）
```

### 5.2 模型白名单与配置 (`../config.py`)

调用方只能使用白名单中的**平台模型名**，实际供应商信息对外不可见：

```python
MODEL_CONFIGS = {
    "general-primary": ModelConfig(
        provider_model="deepseek-chat",          # 实际供应商模型
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",          # 从环境变量读取
        supports_structured_output=True,
        structured_output_mode="json_object",     # json_schema 或 json_object
    ),
    "general-backup": ModelConfig(
        provider_model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_schema",
    ),
}
```

**密钥隔离**：Agent 不需要保存任何供应商 API Key，密钥只在 Gateway 进程内通过环境变量读取。

### 5.3 供应商适配器 (`../provider.py`)

采用 **BaseProvider 抽象基类 + 多协议适配器** 架构，通过 `create_provider()` 工厂函数按 `ModelConfig.protocol` 自动选择适配器：

```
BaseProvider (ABC)
├── OpenAICompatibleProvider   (protocol="openai")
└── AnthropicProvider           (protocol="anthropic")
```

#### OpenAICompatibleProvider

统一两个核心接口：

##### `complete()` — 非流式

- 支持两种结构化输出模式（通过 `_negotiate_response_format()` 共享协商逻辑，complete / stream 共用）：
  - `json_schema`：使用 `response_format.type = "json_schema"` + `strict=True`
  - `json_object`：使用 `response_format.type = "json_object"` + 在 system prompt 中注入 Schema
- 密钥通过 `config.api_key_env` 从环境变量读取，缺失时抛出 `gateway_misconfigured` 错误

##### `stream()` — 流式

- 逐块读取上游响应，yield `delta.content`
- **流开始后不再切换备用模型**，避免文本重复或断裂
- **流首内容前的临时失败纳入指数退避重试**，与非流式保持一致
- **全局时间预算**：`timeout_seconds` 跨整个 fallback 链共享，单次调用传 `min(timeout, remaining)` 真正封顶
- **双参数能力协商**：`response_format`（行为类型） + `response_schema`（Schema 内容）同时传给 provider，provider 内部按 mode 归一化，非流式和流式行为一致
- Anthropic 流式同样支持 Schema 注入到 system prompt

#### AnthropicProvider

适配 Anthropic Messages API，与 OpenAI 协议的关键差异：

| 差异点 | OpenAI 协议 | Anthropic 协议 |
|---------|------------|----------------|
| system 消息 | 在 messages 列表中 | 顶层 `system` 参数 |
| max_tokens | 可选 | **必传**（默认 4096） |
| 结构化输出 | 原生 json_schema/json_object | 无原生支持，通过 system prompt 注入 |
| 流式事件 | `choices[0].delta.content` | `event.type == "text"` + `get_final_message()` |
| SDK | `openai.AsyncOpenAI` | `anthropic.AsyncAnthropic` |

**工厂函数**（带缓存，复用 SDK 连接池）：

```python
_provider_cache: dict[ModelConfig, BaseProvider] = {}

def create_provider(config: ModelConfig) -> BaseProvider:
    cached = _provider_cache.get(config)
    if cached is not None:
        return cached
    if config.protocol == "anthropic":
        provider = AnthropicProvider(config)
    else:
        provider = OpenAICompatibleProvider(config)
    _provider_cache[config] = provider
    return provider
```

### 5.4 主备切换与重试 (`../gateway.py` — `call_with_fallback`)

```
for model_name in [requested_model, "general-backup"]:
    for retry_number in range(2):       # 每个模型最多重试 1 次
        try:
            provider.complete(config, messages, ...)
            → 成功 → 返回 LLMResponse
        except retryable_exception:
            await asyncio.sleep(_retry_delay(attempt))  # 指数退避 + jitter，上限 5s
        except non_retryable:
            → 直接抛出（请求不合法）
→ 全部失败 → 抛出 "model_unavailable"
```

**重试策略**：指数退避 `min(0.1 * 2^attempt, 5.0)` + 随机 jitter，避免所有重试同时打向上游。

**可重试异常**：`APIConnectionError`、`APITimeoutError`、`RateLimitError`、`TimeoutError`、`ConnectionError`（覆盖 OpenAI 和 Anthropic 两种协议）

**不可重试异常**：`GatewayError`（请求不合法）、`openai.AuthenticationError` / `openai.PermissionDeniedError`、`anthropic.AuthenticationError` / `anthropic.PermissionDeniedError`（认证失败，双协议对称）、其他未知异常

### 5.5 流式代理 (`stream_with_fallback`)

```
# app 层先调用 validate_model_chain() + validate_rate_limit()，确保错误返回 JSON
deadline = start_time + timeout_seconds  # 全局时间预算
for model_name in [requested_model, "general-backup"]:
    for attempt in range(MAX_RETRIES_PER_MODEL):   # 流首前可重试
        remaining = deadline - now
        if remaining <= 0: break                    # 预算耗尽，切换模型
        async for pr in provider.stream(response_format=..., response_schema=..., timeout=min(timeout, remaining)):
            if pr.text_delta:
                emitted = True
                yield SSE: content.delta
            if pr.is_usage_chunk:
                stream_usage = Usage(...)
        → 成功 → yield response.completed + return
    except retryable:
        if emitted: break    # 已发送内容，不切换
        await _retry_delay(attempt)  # 指数退避
    → 尝试备用模型
```

**关键**：
- 流开始后出错不再切换模型，因为已向客户端发送部分内容，切换会导致文本断裂
- 流首内容前的临时失败纳入指数退避重试，与非流式 `call_with_fallback` 保持一致
- **全局时间预算**：`timeout_seconds` 跨 fallback 链共享，单次调用传 `min(timeout, deadline - now)` 真正封顶，两条路径语义统一
- 流式调用同样记录 `CallTrace` 审计（含 token 用量、成本、延迟）
- `stream_interrupted` 审计补上已收集的 usage，成本统计不为 0
- `response.completed` 事件携带 `usage` 信息
- 流式端点在返回 `StreamingResponse` 之前调用 `validate_model_chain()` 校验白名单（确保 422 JSON）和 `validate_rate_limit()` 预检限流（确保 429 JSON），避免 SSE 流内嵌 error 事件
- **结构化输出能力协商**：`response_format`（携带类型指令） + `response_schema`（携带 Schema 内容）同时传给 provider，provider 内部按 `structured_output_mode` 归一化，两条路径行为一致

### 5.6 结构化输出校验

模型返回后，如果指定了 `response_schema`：

1. 尝试 `json.loads()` 解析模型输出
2. 用 `jsonschema.validate()` 校验是否符合 Schema
3. 校验通过 → `LLMResponse.parsed` 填充解析结果
4. 校验失败 → 抛出 `structured_output_invalid` 错误

**供应商不支持结构化输出时的兜底**：若 `supports_structured_output=False` 但调用方传了 `response_schema`，不会静默降级，而是在 system prompt 中注入 Schema 文本，引导模型输出 JSON，之后仍走 `jsonschema.validate()` 二次校验。

### 5.7 Prompt 模板管理 (`../config.py`)

```python
PROMPT_TEMPLATES = {
    ("knowledge_decision", "v1"): PromptTemplate(
        system_template="你是${product_name}的知识库决策器，"
                        "根据用户问题判断应该从哪个知识库获取答案..."
    ),
    ("summarize", "v1"): PromptTemplate(
        system_template="请对以下内容进行摘要，控制在${max_words}字以内..."
    ),
}
```

- 调用方只能选择已注册的模板 + 传变量，**不能提交模板正文**
- 使用 `string.Template` 渲染，变量缺失时返回 `missing_prompt_variable` 错误
- 模板由 Gateway 统一管理，支持版本化迭代

### 5.8 调用审计 (`CallTrace`)

每次调用自动记录审计信息（**非流式和流式均记录**）：

| 字段 | 内容 |
|------|------|
| `request_id` | UUID 唯一标识 |
| `requested_model` / `actual_model` | 请求模型 / 实际模型（fallback 后可能不同） |
| `prompt_name` / `prompt_version` | 使用的 Prompt 模板 |
| `input_tokens` / `output_tokens` | Token 用量 |
| `cost_usd` | 按 `PRICE_PER_MILLION` 自动计算的成本 |
| `latency_ms` / `attempts` | 延迟和重试次数 |
| `ttft_ms` | 流式首 Token 时间（毫秒），非流式为 None |
| `status` / `error_code` | 成功/失败状态 |

**默认不保存模型回答和 Prompt 文本**，只保留元数据，保护隐私。

价格配置示例：

```python
PRICE_PER_MILLION = {
    "deepseek-chat":       {"input": 0.27, "output": 1.10},
    "gpt-4o-mini":         {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}
```

### 5.9 TTFT 度量 (`ttft_ms`)

**Time to First Token**：从请求发出到收到第一个非空文本 delta 的时间（毫秒）。

- 仅在流式调用中有意义，非流式调用 `ttft_ms = None`
- 在 `stream_with_fallback()` 中记录首个 `pr.text_delta` 非空时的时间戳
- 写入 `CallTrace.ttft_ms`（审计）和 `response.completed` SSE 事件
- 流中断场景（`stream_interrupted`）同样记录 ttft_ms

### 5.10 按模型限流 (`_ModelRateLimiter`)

线程安全的滑动窗口限流器，每个平台模型独立计数：

| 特性 | 说明 |
|------|------|
| **粒度** | 按平台模型名（如 `general-primary`）独立限流 |
| **算法** | 60 秒滑动窗口，记录请求时间戳 |
| **线程安全** | 每个模型独立的 `asyncio.Lock`，并发安全 |
| **配置** | `ModelConfig.rate_limit_rpm`（0 表示不限制） |
| **超限响应** | 非流式：在 try 内抛出 `GatewayError("rate_limited")` → fallback 下一模型；流式：app 层 `validate_rate_limit()` pre-flight → **429 JSON** |

限流在 provider 调用之前执行，避免重试消耗配额。

## 6. REST API (`../app.py`)

| 端点 | 方法 | 功能 |
|------|------|------|
| `/v1/llm` | POST | 统一入口（`stream=false` 非流式，`stream=true` 流式 SSE） |
| `/v1/llm/stream` | POST | 流式 SSE 调用（兼容别名） |
| `/v1/traces` | GET | 查询调用审计记录 |

> **OpenAPI 双 content type**：`/v1/llm` 的 200 响应在 OpenAPI schema 中显式声明 `application/json`（`$ref` LLMResponse）和 `text/event-stream`（string），确保 Swagger UI 可见两种返回格式。注意 `content` 内层只能用 `{"schema": ...}`，不能用 `{"model": ...}`。

### 请求示例

**非流式调用：**

```bash
curl -X POST http://localhost:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general-primary",
    "messages": [{"role": "user", "content": "你好"}],
    "timeout_seconds": 30
  }'
```

**结构化输出：**

```bash
curl -X POST http://localhost:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general-primary",
    "messages": [{"role": "user", "content": "分析这句话的情感：今天天气真好"}],
    "response_schema": {
      "type": "object",
      "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"}
      },
      "required": ["sentiment", "confidence"]
    }
  }'
```

**Prompt 模板调用：**

```bash
curl -X POST http://localhost:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general-primary",
    "messages": [{"role": "user", "content": "如何退款？"}],
    "prompt": {"name": "knowledge_decision", "version": "v1", "variables": {"product_name": "智能客服"}}
  }'
```

**流式调用（统一入口 stream=true）：**

```bash
curl -X POST http://localhost:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general-primary",
    "messages": [{"role": "user", "content": "讲一个故事"}],
    "stream": true
  }'
```

**使用 response_format（兼容 OpenAI 风格）：**

```bash
curl -X POST http://localhost:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general-primary",
    "messages": [{"role": "user", "content": "分析情感：今天天气真好"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "sentiment",
        "schema": {
          "type": "object",
          "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"}
          },
          "required": ["sentiment", "confidence"]
        }
      }
    }
  }'
```

## 7. 使用方式

### 7.1 安装依赖

```bash
pip install -e .
```

### 7.2 配置环境变量

```bash
export DEEPSEEK_API_KEY="sk-xxx"
export OPENAI_API_KEY="sk-xxx"
```

### 7.3 启动服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 8. 测试体系

### Mock 测试 (`../test_gateway.py`)

使用 `FakeProvider` 模拟供应商行为，不依赖真实 API：

| 测试场景 | 验证点 |
|----------|--------|
| 正常调用 | 返回正确内容 + 平台模型名 + 审计记录 |
| 结构化输出 | JSON 解析 + Schema 校验 |
| 主模型连续超时 | 自动 fallback 到备用模型（指数退避） |
| 流式输出 | SSE 事件格式 + request_id + usage + 审计 |
| 非法请求 | 422 + 错误码 |
| system 消息合并 | 模板 system + 原有 system 合并而非丢弃 |
| 流式白名单校验 | `validate_model_chain` 在 endpoint 层拦截 |

### 集成测试 (`../test_gateway_e2e.py`)

使用真实 API 端到端验证：

| 测试场景 | 验证点 |
|----------|--------|
| 非流式 + Prompt 模板 | 模板渲染 + 正确返回 |
| 结构化输出 | Schema 校验通过 |
| 流式 SSE | 逐块输出 + 审计记录 |
| 未知字段 | 422 拒绝 |
| Prompt 变量缺失 | `missing_prompt_variable` |
| 模型白名单外（非流式） | `model_not_found` |
| 模型白名单外（流式） | `validate_model_chain` 拦截 |
| 主模型不可达 | fallback 到备用模型 |
| 审计完整性 | trace 包含所有必要字段 |

## 9. 扩展指南

### 新增供应商

只需在 `../config.py` 的 `MODEL_CONFIGS` 中添加一条配置：

```python
"qwen-primary": ModelConfig(
    provider_model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key_env="QWEN_API_KEY",
    supports_structured_output=True,
    structured_output_mode="json_object",
),
```

因为使用 OpenAI 兼容协议，无需新增适配器代码。

### 新增 Anthropic 协议供应商

在 `../config.py` 的 `MODEL_CONFIGS` 中添加配置，并设置 `protocol="anthropic"`：

```python
"anthropic-claude": ModelConfig(
    provider_model="claude-sonnet-4-6",
    base_url="https://api.anthropic.com",
    api_key_env="ANTHROPIC_API_KEY",
    protocol="anthropic",
),
```

工厂函数 `create_provider()` 会自动选择 `AnthropicProvider`。

### 新增 Prompt 模板

在 `../config.py` 的 `PROMPT_TEMPLATES` 中注册：

```python
("translate", "v1"): PromptTemplate(
    system_template="你是一个专业翻译，将内容从${source_lang}翻译为${target_lang}..."
),
```

## 10. 设计亮点总结

| # | 亮点 | 说明 |
|---|------|------|
| 1 | **密钥隔离** | API Key 只存在于 Gateway 进程，Agent 通过 Gateway 间接访问模型 |
| 2 | **模型白名单** | 调用方只能用平台模型名，不能直接指定任意供应商模型 |
| 3 | **主备自动切换** | 主模型重试 1 次后自动切备用模型，对调用方透明 |
| 4 | **流式安全** | 流开始后不切换模型，避免文本断裂 |
| 5 | **Prompt 版本管理** | 模板由 Gateway 统一管理，调用方只能选择版本和传变量 |
| 6 | **成本治理** | 每次调用（含流式）自动计算成本并记录审计 |
| 7 | **全链路校验** | 入口 Pydantic 校验请求，出口 jsonschema 校验模型输出 |
| 8 | **`extra="forbid"`** | 所有 Pydantic 模型禁止未知字段，防止注入攻击 |
| 9 | **流式 422 错误处理** | `validate_model_chain` 在 `StreamingResponse` 前校验，确保错误返回 422 JSON |
| 10 | **指数退避重试** | `min(0.1 * 2^attempt, 5.0)` + jitter + 上限保护 |
| 11 | **类型安全适配器** | `ProviderResponse` dataclass 替代裸 dict，`is_usage_chunk` 显式标志 |
| 12 | **`LLMResponse.model` 回填平台名** | 响应返回 `general-primary`，供应商名只保留在审计层 |
| 13 | **双协议适配器** | BaseProvider 抽象 + OpenAI/Anthropic 双协议，工厂函数自动选择 |
| 14 | **TTFT 度量** | 流式首个非空 delta 时间记录，写入审计和 SSE 事件 |
| 15 | **按模型限流 429** | 线程安全滑动窗口，每模型独立锁，非流式走 fallback、流式 pre-flight 429 JSON |
| 16 | **`GatewayError.http_status` 契约** | 异常携带 HTTP 状态码，`_DEFAULT_STATUS` 映射 + 可显式覆盖，测试钉住防漂移 |
| 17 | **`GatewayError` 选择性 fallback** | `rate_limited` 切换模型；`structured_output_invalid` 等立即抛出，不浪费 fallback 链 |
| 18 | **`would_accept()` 公共只读查询** | 限流器暴露公共方法，app 层不直接访问内部字段 |

## 11. 已知限制（Limitations）

以下议题在当前作业阶段可接受，但生产部署前需解决：

| # | 限制 | 说明 |
|---|------|------|
| 1 | **`timeout_seconds` 是全局预算** | 非流式和流式均使用 deadline 封顶，单次调用传 `min(timeout, remaining)`。最坏总时长 ≤ `timeout_seconds`，不再是 2×2×30s |
| 2 | **审计存储为进程内 list** | 多 worker 下不共享，无分页、无鉴权、无 TTL。生产应替换为数据库 + 访问控制 |
| 3 | **`Message` 不支持 tool/function calling** | 当前 `role` 只有 system/user/assistant，`content` 是 `str`。Agent 场景需要扩展支持 |
| 4 | **重试 jitter 不可复现** | `random.uniform` 使测试等待时间非确定性。可改为 `random.seed` 或在测试中 patch |
| 5 | **Anthropic 无原生结构化输出** | Anthropic 协议不支持 `json_schema`/`json_object` 模式，通过 system prompt 注入 Schema 文本引导，可靠性低于原生支持 |
| 6 | **限流器为进程内存储** | `_ModelRateLimiter` 的窗口数据在进程重启后丢失，多 worker 下不共享。生产应替换为 Redis 等分布式限流 |
