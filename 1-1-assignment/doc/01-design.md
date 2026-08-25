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
│  │   供应商适配器 (OpenAICompatible)    │ │
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
│  │  + 审计记录 (CallTrace)              │ │
│  └──────────────────────────────────────┘ │
└──────────────────────────────────────────┘
    │
    ▼
DeepSeek / OpenAI / 其他兼容供应商
```

### 分层说明

| 层次 | 文件 | 职责 |
|------|------|------|
| **协议层** | `../models.py` | Pydantic 请求/响应模型，`extra="forbid"` 防注入 |
| **配置层** | `../config.py` | 模型白名单 `MODEL_CONFIGS`、Prompt 模板注册表、价格表 |
| **适配器层** | `../provider.py` | `OpenAICompatibleProvider`，统一 `complete()` / `stream()` |
| **网关核心** | `../gateway.py` | 主备切换、重试、结构化校验、审计记录 |
| **API 层** | `../app.py` | FastAPI REST 端点 |
| **测试** | `../test_gateway.py` | Mock 测试 + 集成测试 |

## 4. 目录结构

```
1-1-assignment/
├── __init__.py
├── models.py              # Pydantic 请求/响应模型
├── config.py              # 模型白名单、Prompt 模板、价格表
├── provider.py            # OpenAI 兼容供应商适配器
├── gateway.py             # 核心：主备切换、重试、校验、审计
├── app.py                 # FastAPI REST API
├── test_gateway.py        # Mock 测试
├── test_gateway_e2e.py    # 集成测试（真实 API）
├── requirements.txt       # 依赖声明
└── README.md              # 使用文档
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
    timeout_seconds: float = 30             # 超时时间
    prompt: PromptSelection | None = None   # Prompt 模板选择
```

**组合校验**（`model_validator`）：`stream=True` 与 `response_schema` 不能同时使用（流式无法做 JSON 校验）。

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

实现 `OpenAICompatibleProvider`，统一两个核心接口：

#### `complete()` — 非流式

- 支持两种结构化输出模式：
  - `json_schema`：使用 `response_format.type = "json_schema"` + `strict=True`
  - `json_object`：使用 `response_format.type = "json_object"` + 在 system prompt 中注入 Schema
- 密钥通过 `config.api_key_env` 从环境变量读取，缺失时抛出 `gateway_misconfigured` 错误

#### `stream()` — 流式

- 逐块读取上游响应，yield `delta.content`
- **流开始后不再切换备用模型**，避免文本重复或断裂

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

**可重试异常**：`APIConnectionError`、`APITimeoutError`、`RateLimitError`、`TimeoutError`、`ConnectionError`

**不可重试异常**：`GatewayError`（请求不合法）、其他未知异常

### 5.5 流式代理 (`stream_with_fallback`)

```
# app 层先调用 validate_model_chain()，确保 model_not_found 返回 422 JSON
for model_name in [requested_model, "general-backup"]:
    async for pr in provider.stream(...):
        if pr.text_delta:
            emitted = True
            yield SSE: {"type": "content.delta", "delta": pr.text_delta, "request_id": ...}
        if pr.is_usage_chunk:    # 显式标志，不依赖 token 数推断
            stream_usage = Usage(...)
    # 流正常结束：记录 CallTrace 审计
    yield SSE: {"type": "response.completed", "model": model_name, "usage": {...}}
    return
except:
    if emitted or not retryable:
        break    # 已发送部分内容，不再切换
    → 尝试备用模型
```

**关键**：
- 流开始后出错不再切换模型，因为已向客户端发送部分内容，切换会导致文本断裂
- 流式调用同样记录 `CallTrace` 审计（含 token 用量、成本、延迟）
- `response.completed` 事件携带 `usage` 信息
- 流式端点在返回 `StreamingResponse` 之前调用 `validate_model_chain()` 校验白名单，确保错误被 `exception_handler` 捕获为 422 JSON

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
| `status` / `error_code` | 成功/失败状态 |

**默认不保存模型回答和 Prompt 文本**，只保留元数据，保护隐私。

价格配置示例：

```python
PRICE_PER_MILLION = {
    "deepseek-chat":       {"input": 0.27, "output": 1.10},
    "gpt-4o-mini":         {"input": 0.15, "output": 0.60},
}
```

## 6. REST API (`../app.py`)

| 端点 | 方法 | 功能 |
|------|------|------|
| `/v1/llm` | POST | 非流式调用（含结构化输出） |
| `/v1/llm/stream` | POST | 流式 SSE 调用 |
| `/v1/traces` | GET | 查询调用审计记录 |

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

**流式调用：**

```bash
curl -X POST http://localhost:8000/v1/llm/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general-primary",
    "messages": [{"role": "user", "content": "讲一个故事"}],
    "stream": true
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

## 11. 已知限制（Limitations）

以下议题在当前作业阶段可接受，但生产部署前需解决：

| # | 限制 | 说明 |
|---|------|------|
| 1 | **`timeout_seconds` 是单次语义** | 当前原样传给 provider，fallback 链最坏 2×2×30s=120s，超过调用方理解的 30s。生产级应用改为总预算递减 |
| 2 | **审计存储为进程内 list** | 多 worker 下不共享，无分页、无鉴权、无 TTL。生产应替换为数据库 + 访问控制 |
| 3 | **`Message` 不支持 tool/function calling** | 当前 `role` 只有 system/user/assistant，`content` 是 `str`。Agent 场景需要扩展支持 |
| 4 | **重试 jitter 不可复现** | `random.uniform` 使测试等待时间非确定性。可改为 `random.seed` 或在测试中 patch |
