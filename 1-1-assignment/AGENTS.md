# 作业 1-1：LLM Gateway 专属规范

> 本文件只适用于 `1-1-assignment/`。**仓库级通用规则见根目录 `AGENTS.md`，先读那份。**
> 开工前两份都要读：根目录讲"怎么做事"，本文件讲"这个作业的具体事实与坑"。

---

## 0. 本作业命令与数字

```bash
uv sync                                                       # 首次或依赖变更后
uv run python -m pytest 1-1-assignment/test_gateway.py -q     # 单元测试（当前 62 项）
uv run python -m pytest 1-1-assignment/test_gateway_e2e.py --collect-only -q   # e2e 只收集，需真实 API Key
```

> ⚠️ 必须在**仓库根目录**执行（`1-1-assignment/` 内的相对导入要求如此），不要 `cd` 进去跑。

**本作业未安装** ruff / mypy。**测试数量一变，同步更新此处 + 根 `AGENTS.md` + `doc/01-design.md`。**

---

## 1. 本作业的文件地图

| 文件 | 职责 |
|------|------|
| `models.py` | Pydantic 请求/响应/审计模型，全部 `extra="forbid"` |
| `config.py` | 模型白名单 `MODEL_CONFIGS`、Prompt 模板、价格表 |
| `provider.py` | `BaseProvider` + OpenAI / Anthropic 双协议适配器 |
| `gateway.py` | 主备切换、重试、超时预算、结构化校验、审计、限流 |
| `app.py` | FastAPI 端点（`/v1/llm` 统一入口、`/v1/llm/stream` 别名、`/v1/traces`）|
| `doc/01-design.md`、`doc/README.md` | 设计与使用文档，**随代码同步** |

**主模型 `general-primary` 的 `structured_output_mode` 是 `json_object`**（不是 `json_schema`）。
涉及结构化输出的改动，一定要拿它验证——历史上两次翻车都出在这条路径。

---

## 2. 本作业踩过的坑（详细记录，不要重复踩）

### 2.1 参数透传：同一个 bug 修了两次

- **第一次**：`gateway.py` 给 provider 传 `response_schema=request.response_schema`，
  而流式请求的 `request.response_schema` **恒为 `None`**（`models.py` 的 validator 禁止 stream + schema）。
  `provider.py` 里那段能力协商成了**死代码**——只有直接构造的测试能命中，真实 HTTP 请求走不到。

- **第二次（修第一次时引入）**：把参数改名却仍只传 `response_schema`，问题原封不动。
  同时 `json_object` 模式**两条路径全失效**（`response_schema is None` 时既不传
  `response_format`、也不做 system 注入），而它恰恰是主模型的配置模式。

**现状（正确形态）**：`gateway.py` 同时传 `response_format`（行为类型）+ `response_schema`（Schema 内容），
由 `provider.py` 的 `_negotiate_response_format()` 统一归一化。**不要退回单参数。**

### 2.2 能力协商分叉

`AnthropicProvider.stream()` 曾接了 `response_schema` 却完全不用（连 system 注入都没做），
而 `complete()` 做了。修法是抽成共享方法 `_inject_schema_to_system_text()`，**不是抄一遍**。

### 2.3 把错误行为写进测试

`test_gateway.py` 里曾有 `assert req.response_schema is None`，让"`json_object` 约束失效"
看起来像预期行为。**该测试已删除并重写为 `TestStreamStructuredRegression`——不要再写回去。**

### 2.4 OpenAPI 写错导致 `/docs` 空白

曾把 `{"model": LLMResponse}` 写进 `responses[200]["content"]`（FastAPI 不认这种嵌套），
`GET /openapi.json` 直接 **500**、Swagger UI 空白，而当时 61 项测试**全绿**——没有一条碰过 OpenAPI。
守卫测试 `test_openapi_json_generates` 已补。**正确写法**（`content` 里只能放 `schema`）：

```python
responses={200: {"description": "...", "content": {
    "application/json": {"schema": {"$ref": "#/components/schemas/LLMResponse"}},
    "text/event-stream": {"schema": {"type": "string"}},
}}}
```
（别用 `LLMResponse.model_json_schema()`——会把 `$defs` 内联进去，产生大量冗余。）

---

## 3. 本作业的语义契约与实测参考值

改 `gateway.py` 的重试 / 预算 / 审计逻辑后，逐项实测：

| 契约 | 期望 | 实测参考值 |
|------|------|-----------|
| 超时预算递减且 ≤ `timeout_seconds` | 递减序列 | `[0.3, 0.132, 0.111]` |
| `attempts` == provider 真实调用次数 | 严格相等 | 三种预算下分别 4/3/1 |
| 流中断的 `CallTrace` 带已收集 usage | tokens、cost 非零 | `tokens=10/5, cost>0` |
| 统一端点 pre-flight 错误 | 白名单 422 / 限流 429 / schema+stream 422，**均为 `application/json`** | 不能退化成 SSE 内嵌 error |

**两条路径（`call_with_fallback` / `stream_with_fallback`）的超时预算必须对称**——
历史上只有流式做了 deadline，非流式没有。

---

## 4. 本作业端点契约

| 端点 | 行为 |
|------|------|
| `POST /v1/llm` | `stream=false` → `LLMResponse` JSON；`stream=true` → `text/event-stream` |
| `POST /v1/llm/stream` | 兼容别名，**不要求**请求带 `stream` 字段 |
| `GET /v1/traces` | 审计记录 |

**行为收窄（有意为之，需保持文档同步）**：
- `stream=true` + `json_schema` → **422 拒绝**（流式无法做 JSON 校验）
- `response_format` 仅支持 `json_schema` / `json_object`，**`{"type":"text"}` 被拒绝**

---

## 5. 本作业提交前 Checklist

- [ ] `uv run python -m pytest 1-1-assignment/test_gateway.py -q` 全绿（当前 62 项）
- [ ] `uv run python -m pytest 1-1-assignment/test_gateway_e2e.py --collect-only -q` 能正常收集
- [ ] 新参数已实测到 provider 真实 kwargs，非 `None`（**流式 + 非流式都跑了**）
- [ ] `complete()` / `stream()` 行为一致，且协商逻辑是**共享实现**（见 2.2）
- [ ] 新增断言钉的是正确行为；HTTP 层有覆盖；`test_openapi_json_generates` 仍通过
- [ ] 第 3 节四项语义契约已实测
- [ ] `doc/01-design.md`（端点表、`LLMRequest` 块、测试项数、已知限制）+ `doc/README.md`（curl 示例）已同步
- [ ] 本文件与根 `AGENTS.md` 的数字/遗留项已核对，无过期描述
- [ ] `git diff` 自查：无残留调试 `print` / 注释掉的代码

---

## 6. 本作业已知遗留

- **审计存储为进程内 list、限流器为进程内窗口**——多 worker 下不共享，生产需替换为
  数据库 / Redis。详见 `doc/01-design.md` 第 11 节。
- **`Message` 不支持 tool/function calling**（`role` 仅 system/user/assistant，`content` 是 `str`），
  Agent 场景需扩展。详见 `doc/01-design.md` 第 11 节。
- **重试 jitter 不可复现**（`random.uniform`），测试等待时间非确定性。
- **`response_format` 与显式 `response_schema` 同时传入时的优先级**未在 `doc/01-design.md`
  写明（行为已确定且有测试覆盖，仅缺文档说明）。

> 已修复、**不要**再当遗留处理：e2e mock 签名漂移 ✅、`/v1/llm` 的 OpenAPI 不体现
> `text/event-stream` ✅、`response_format: {"type":"text"}` 未写进文档 ✅。
