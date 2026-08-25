# LLM Gateway — 统一模型调用服务

生产级的 LLM 网关，作为 Agent 与模型供应商之间的统一中间层。

## 核心特性

- **模型白名单管控** — 调用方只能使用平台模型名，不能直接指定供应商模型
- **密钥隔离** — API Key 只存在于 Gateway 进程，Agent 不接触任何供应商密钥
- **主备自动切换** — 主模型超时/异常后自动 fallback 到备用模型
- **结构化输出** — 支持 JSON Schema 约束 + 二次校验
- **Prompt 模板版本管理** — 模板由 Gateway 统一管理
- **调用审计与成本治理** — 每次调用记录 token 用量、延迟、成本

## 快速开始

### 1. 安装依赖

```bash
# 在项目根目录执行
pip install -e .
```

### 2. 配置环境变量

在 `1-1-assignment/` 目录下创建 `.env` 文件（已在 `.gitignore` 中屏蔽，不会提交）：

```bash
# 必填：API 密钥
DEEPSEEK_API_KEY=sk-xxx

# 可选：覆盖默认 API 地址
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 可选：覆盖默认模型名
DEEPSEEK_MODEL=deepseek-chat

# 备用模型配置（可选）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

> 服务启动时通过 `python-dotenv` 自动加载 `.env` 文件，无需手动 `export`。

### 3. 启动服务

```bash
cd 1-1-assignment
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

启动成功后访问：
- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc

## API 使用

### 非流式调用

```bash
curl -X POST http://localhost:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general-primary",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 结构化输出

```bash
curl -X POST http://localhost:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general-primary",
    "messages": [{"role": "user", "content": "分析情感：今天天气真好"}],
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

### 流式调用

```bash
curl -X POST http://localhost:8000/v1/llm/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general-primary",
    "messages": [{"role": "user", "content": "讲一个故事"}],
    "stream": true
  }'
```

### Prompt 模板调用

```bash
curl -X POST http://localhost:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general-primary",
    "messages": [{"role": "user", "content": "如何退款？"}],
    "prompt": {
      "name": "knowledge_decision",
      "version": "v1",
      "variables": {"product_name": "智能客服"}
    }
  }'
```

### 查询审计记录

```bash
curl http://localhost:8000/v1/traces
```

## 运行测试

```bash
# Mock 测试（不需要 API Key）
pytest test_gateway.py -v

# 集成测试（需要 DEEPSEEK_API_KEY）
pytest test_gateway_e2e.py -v
```

## 扩展

### 新增供应商

在 `../config.py` 的 `MODEL_CONFIGS` 中添加配置即可（OpenAI 兼容协议）：

```python
"qwen-primary": ModelConfig(
    provider_model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key_env="QWEN_API_KEY",
    supports_structured_output=True,
    structured_output_mode="json_object",
),
```

### 新增 Prompt 模板

在 `../config.py` 的 `PROMPT_TEMPLATES` 中注册：

```python
("translate", "v1"): PromptTemplate(
    system_template="将内容从${source_lang}翻译为${target_lang}",
),
```
