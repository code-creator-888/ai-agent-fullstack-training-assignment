# 作业二：为治理框架增加"转账"工具 — 设计方案

## 1. 需求概述

在课程 2-4 的**工具治理框架**（ToolRuntime：参数校验 → 权限三态 → 审批 → 受控执行 → 脱敏审计）基础上，
新增一个 **转账（`transfer`）工具**，把"高风险资金动作"压进确定的工程边界：

**该成的成（审批后正常转账），不该成的在副作用发生之前停下来（越权、余额不足、伪造身份、
超时后的盲目重放），每一步决策留下可审计的证据。**

### 1.1 作业目标

| 目标 | 说明 |
|------|------|
| **验证框架可扩展性** | 新增工具 = 只新增参数模型 + 策略 + handler + 预检，**框架零逻辑改动**（仅一处文案参数化，见 §5.1） |
| **写操作超时治理** | 大额转账故意超时 → 返回 `TIMEOUT_UNKNOWN`（结果未知），**禁止自动重试**，走状态查询恢复 |
| **审批绑定参数** | 转账必须审批；审批与"转出账户 + 转入账户 + 金额 + 事由"逐字段绑定，改金额重入即失效 |
| **身份防伪** | 账户归属由 `ExecutionContext.tenant_id` 判定；参数携带 `user_id` / `approved` 直接 `INVALID_ARGUMENT` |
| **副作用可验收** | `TRANSFER_EXECUTIONS`（handler 真实进入次数）与写入数、审计数严格对账 |
| **审计与脱敏** | 两阶段审计（decision / execution）+ 参数只记键名 + 邮箱脱敏 |

### 1.2 转账工具的治理要求（策略表）

| 维度 | 要求 | 依据 |
|------|------|------|
| 副作用分类 | `Effect.WRITE`（资金变动，不可逆） | 治理第一阶段：工具分级 |
| 风险等级 | `Risk.HIGH` | 资金操作 |
| RBAC 权限 | `transfer:execute` | 白名单 + RBAC 双查 |
| 人工审批 | **必须**（`requires_approval=True`），绑定全部业务参数 | 高风险写操作 |
| 自动重试 | **禁止**（`max_retries=0`，`idempotent=False`） | 非幂等写不盲重试 |
| 超时预算 | `timeout_seconds=2.0`，写超时返回 `TIMEOUT_UNKNOWN` | 结果未知 ≠ 失败 |
| 执行边界 | 框架锁串行化 handler；余额预检在锁外，handler 内**锁后二次校验**为实际防线 | check-then-act 竞态（真实 I/O 下可达，见 §5.9） |
| 数据边界 | 限定**同租户内**账户互转；跨租户目标一律拒绝 | 租户隔离 |

### 1.3 超时模拟规则（作业指定）

> **超时模拟：如果 `amount > 40000`，执行 `await asyncio.sleep(3.0)`（故意制造超时）。**

落地为两个模块级常量（放在类定义之外，遵守仓库 Pydantic 常量规范）：

```python
# transfer.py — 模块级常量
TRANSFER_SLOWDOWN_THRESHOLD = Decimal("40000")   # 大于此值触发模拟延迟
TRANSFER_SLOWDOWN_SECONDS = 3.0                  # 延迟 3.0s > timeout_seconds(2.0)，必然超时
```

**数字设计的配合关系**（为什么阈值 40000 有效）：

| 参数 | 取值 | 作用 |
|------|------|------|
| 模拟账户 `acc_1001` 余额 | `50000.00` | 覆盖两条路径：35000（成功）/ 45000（超时）均通过余额预检 |
| 超时阈值 | `40000` | 35000 ≤ 40000 不超时；45000 > 40000 触发 sleep |
| `timeout_seconds` | `2.0` | 小于 sleep 3.0 → 大额路径**必然**被 `asyncio.timeout` 取消 |
| 边界语义 | 严格大于 | `amount == 40000.00` **不**触发超时（测试钉住此边界） |

> 若阈值仍为原始的 80000，演示"超时但预检通过"需要 8 万以上余额的账户；
> 调整为 40000 后，5 万余额的账户即可同时覆盖成功与超时两条路径，演示数据更收敛。

## 2. 技术选型

| 项目 | 选择 | 说明 |
|------|------|------|
| 语言 | Python >= 3.14 | 项目要求 |
| 参数校验 | `pydantic >= 2.0` | `extra="forbid"` 防注入；金额用 `Decimal`（资金精度） |
| 异步执行 | `asyncio.timeout`（3.11+） | 取消式超时，写操作结果归一化为 `TIMEOUT_UNKNOWN` |
| 并发控制 | `asyncio.Lock`（按资源键） | 同一转出账户写操作串行 |
| 测试 | `pytest` + `pytest-asyncio` | `@pytest.mark.asyncio` 风格，与作业 1-1 一致 |
| Web 框架 | **无** | 本作业对外接口是 `ToolRuntime.invoke()`（治理框架公共入口），无 HTTP 层 |

> 测试命令沿用仓库约定：`uv run python -m pytest 1-2-assignment/test_transfer.py -q`，
> 在**仓库根目录**执行（相对导入要求）。

## 3. 整体架构

### 3.1 治理链路总览

```
Agent / 模型（提出候选动作：transfer Tool Call，参数不可信）
    │
    ▼
ToolRuntime.invoke() ←—— 唯一执行入口（demo / 测试 / 将来接 CLI 或 LLM 全部走这里）
    │
    ├─ before_tool_call ─── 门禁链（固定顺序，不可交换）
    │    ① 参数校验：TransferArgs.model_validate()，extra="forbid" 拒绝伪造身份
    │    ② 硬拒绝：denied / enabled
    │    ③ plan 模式：WRITE 被 PLAN_MODE_DENIED
    │    ④ 白名单 + RBAC：allowed_tools + transfer:execute
    │    ⑤ 业务预检：账户归属（租户）→ 状态 active → from ≠ to → 余额充足
    │    ⑥ 审批核销：ApprovalStore.consume()，四重绑定 + 一次性 + TTL
    │
    ├─ _execute ─── 受控执行
    │    写操作按 canonical_target 加锁串行（锁仅覆盖 handler，precheck 在锁外）
    │    asyncio.timeout(2.0) 包裹 handler
    │    handler 内：amount > 40000 → asyncio.sleep(3.0)（故意制造超时）
    │    非幂等写：TransientToolError 不重试（框架强制，retries 恒 0）
    │
    └─ after_tool_call ─── 结果保护
         脱敏（敏感键 + 邮箱正则）→ 两阶段审计（argument_keys / duration_ms / 错误码）
         → ToolResult（保留 tool_call_id，标记 untrusted）
         写超时 → TIMEOUT_UNKNOWN："写操作结果未知，请按业务 ID 查询或转人工"
```

### 3.2 门禁固定顺序（与课程基线一致，不可交换）

```
参数校验 → 硬拒绝(deny) → plan 模式 → 白名单 → RBAC → 业务预检 → 审批核销 → allow
```

- **预检先于审批核销**：余额不足等明显非法请求在消耗审批凭证之前被拒
- **审批一次性**：任何执行期失败（超时 / `TOOL_ERROR` / `TEMPORARY_UNAVAILABLE`）都会烧掉审批
  （`consume` 在执行前，`:406`），不只是超时——重放需要**重新审批**，纵深防重放

### 3.3 分层说明

| 层次 | 文件 | 职责 |
|------|------|------|
| **治理框架** | `../governance.py` | ToolRuntime / PermissionEngine / ApprovalStore / AuditSink / 脱敏与投影——从课程 2-4 基线迁移，**仅 `consume()` 文案参数化一处改动**（见 §5.1） |
| **工具层** | `../transfer.py` | 转账域：参数契约、模拟账务数据、预检、handler（含超时模拟）、策略与注册 |
| **演示层** | `../demo.py` | 治理轨迹演示 + 副作用统计输出 |
| **测试** | `../test_transfer.py` | 全部打 `ToolRuntime.invoke()` 公共入口，用副作用证据验收 |

## 4. 目录结构

```
1-2-assignment/
├── __init__.py
├── AGENTS.md            # 作业专属规范（命令与测试数、文件地图、本作业的坑）
├── governance.py        # 治理框架核心（从课程 2-4 tool_governance_demo.py 迁移）
├── transfer.py          # 转账工具：参数契约 / 模拟账务 / 预检 / handler / 策略注册
├── demo.py              # 演示入口：十条治理轨迹 + 副作用统计
├── test_transfer.py     # 测试（覆盖框架门禁 + 转账工具全部治理分支）
└── doc/
    ├── 02-design.md     # 本设计方案
    └── README.md        # 使用说明（运行命令、轨迹解读、注意事项）
```

> 迁移边界：课程基线中的订单域工具（`get_order` / `create_refund` / `run_shell` 及 `ORDERS` / `REFUNDS`）
> **不迁移**——本作业聚焦账户域；框架组件（Runtime / 审批 / 审计 / 脱敏 / 投影）全量保留。

## 5. 核心设计

### 5.1 治理框架基线 (`../governance.py`)

从课程 2-4 `tool_governance_demo.py` 迁移以下组件（保持行为不变）：

| 组件 | 职责 |
|------|------|
| `StrictArgs` | 所有工具参数模型的共同基类：`extra="forbid"` + `str_strip_whitespace` |
| `Effect` / `Risk` / `PermissionDecision` / `PermissionMode` | 副作用、风险、决策三态、执行模式枚举 |
| `ToolCall` / `ExecutionContext` / `ToolPolicy` / `ToolDefinition` / `PreparedCall` / `ToolResult` | 核心对象模型 |
| `PolicyDenied` / `TransientToolError` | 结构化拒绝异常 / 瞬态故障标记 |
| `_stable_value` / `_approval_digest` | 参数规范化与审批摘要（sha256 绑定工具名 + 参数） |
| `Approval` / `ApprovalStore` | 一次性审批凭证：身份、租户、工具、参数摘要四重绑定 + TTL |
| `PermissionEngine` | 白名单 + RBAC 双查 |
| `AuditSink` / `argument_keys` / `redact` / `project_for_model` | 审计、脱敏、模型投影 |
| `ToolRuntime` | `invoke()`（三段编排：门禁 → 受控执行 → 结果保护）/ `invoke_batch()`（批量调度）/ `model_tools()`（运行时工具发现），异常归一化 |
| `SENSITIVE_KEY` / `MODEL_MAX_DEPTH` 等 | 脱敏正则与模型投影边界常量 |

`ToolPolicy` 除 §5.6 展示的核心字段外，还包含 `denied`（硬拒绝开关）、`requires_confirmation`（用户确认）、`enabled`（工具启停）三个策略字段，迁移时一并保留。

`invoke()` 的异常归一化（转账超时治理的关键，框架已实现、转账自动继承）：

```python
except TimeoutError:
    code = "TIMEOUT_UNKNOWN" if prepared.tool.policy.effect is Effect.WRITE else "TIMEOUT"
    message = "写操作结果未知，请按业务 ID 查询或转人工" if code == "TIMEOUT_UNKNOWN" else "查询超时，可稍后重试"
```

**作业验证点**：新增转账工具对 `governance.py` 仅做一处例外改动——`ApprovalStore.consume()` 的
错误文案参数化（见下文 §5.1 补充说明）。除此之外若需改动框架，说明框架抽象有缺口，
应优先修框架而不是在工具里绕过。

> **迁移唯一允许的框架改动**：基线 `:253` 中 `consume()` 抛出
> `PolicyDenied("APPROVAL_REQUIRED", "请确认本次退款的订单、金额和原因")`，
> 文案硬编码了 demo 的退款领域。迁移时将其参数化（如接受 `message` 参数或改为通用文案
> `"请确认本次操作的参数"`）。这是"零框架改动"主张的唯一妥协点，在 §10 已知限制中记录。

### 5.2 转账参数契约 (`../transfer.py` — `TransferArgs`)

```python
class TransferArgs(StrictArgs):
    from_account: str = Field(pattern=r"^acc_[0-9]{4}$")   # 转出账户
    to_account: str = Field(pattern=r"^acc_[0-9]{4}$")     # 转入账户
    amount: Decimal = Field(gt=0, max_digits=10, decimal_places=2)  # 金额，正数、两位小数
    reason: str = Field(min_length=4, max_length=200)      # 转账事由
```

- **身份字段零暴露**：`user_id` / `tenant_id` / `approved` 不在参数模型中，
  `extra="forbid"` 使任何注入尝试得到 `INVALID_ARGUMENT`（含结构化字段错误路径）
- **金额用 `Decimal`**：资金精度；与阈值常量 `Decimal("40000")` 同类型比较，无浮点误差。
  **内部账务全程 `Decimal`**，仅在序列化结果（`TRANSFERS[...]`、`ToolResult.content`）时转 `float`
  ——与基线 `create_refund`（`:599`）一致，handler 返回值中 `float()` 仅用于 JSON 安全传输
- **账户格式受限**：`acc_` 前缀 + 4 位数字，防止路径遍历类脏输入进入业务层

配套只读工具参数（对账与脱敏演示）：

```python
class GetAccountArgs(StrictArgs):
    account_id: str = Field(pattern=r"^acc_[0-9]{4}$")
```

### 5.3 模拟账务数据

```python
# 账户按 (租户, 账户号) 隔离，归属由 ctx.tenant_id 决定，模型参数无权声明
ACCOUNTS: dict[tuple[str, str], dict[str, Any]] = {
    ("tenant_a", "acc_1001"): {"balance": Decimal("50000.00"), "status": "active", "owner_email": "alice@example.com"},
    ("tenant_a", "acc_1002"): {"balance": Decimal("1200.00"), "status": "active", "owner_email": "bob@example.com"},
    ("tenant_b", "acc_2001"): {"balance": Decimal("80000.00"), "status": "active", "owner_email": "carol@example.com"},
}
TRANSFERS: dict[str, dict[str, Any]] = {}   # 已完成的转账（按 transfer_id 索引）
TRANSFER_EXECUTIONS = 0                      # handler 真实进入次数（副作用计量）
```

**余额设计的闭环验证**：

| 演示路径 | 金额 | 预检（余额） | 阈值判断 | 结果 |
|---------|------|------------|---------|------|
| 正常转账 | 35000 | 35000 ≤ 50000 ✓ | 35000 ≤ 40000 不触发 | 成功，余额 50000→15000 |
| 大额转账 | 45000 | 45000 ≤ 50000 ✓（能进 handler） | 45000 > 40000 触发 sleep | `TIMEOUT_UNKNOWN`，余额不变 |
| 余额不足 | 40000（自 acc_1002） | 40000 > 36200 ✗ | 未到 handler | `BUSINESS_RULE_DENIED` |

### 5.4 业务预检查 (`check_transfer`)

预检在审批核销**之前**执行（框架门禁顺序 ⑤→⑥）：

```python
async def check_transfer(args: StrictArgs, ctx: ExecutionContext) -> None:
    assert isinstance(args, TransferArgs)
    from_key = (ctx.tenant_id, args.from_account)
    to_key = (ctx.tenant_id, args.to_account)

    # 租户归属：转出与转入账户都必须属于当前租户（跨租户一律拒绝）
    if args.from_account == args.to_account:
        raise PolicyDenied("BUSINESS_RULE_DENIED", "转出与转入账户不能相同")
    if ACCOUNTS.get(from_key) is None or ACCOUNTS.get(to_key) is None:
        # 与"不存在"同响应，不泄露其他租户账户的存在性（防账户枚举）
        raise PolicyDenied("BUSINESS_RULE_DENIED", "账户不存在或不属于当前租户")
    if ACCOUNTS[from_key]["status"] != "active" or ACCOUNTS[to_key]["status"] != "active":
        raise PolicyDenied("BUSINESS_RULE_DENIED", "账户状态不可用")
    if args.amount > ACCOUNTS[from_key]["balance"]:
        raise PolicyDenied("BUSINESS_RULE_DENIED", "余额不足")
```

- **越权判定只用 `ctx.tenant_id`**：`from_account` 属于其他租户（如 tenant_b 的 `acc_2001`）时，
  与"账户不存在"返回同一错误——既完成租户隔离，又不给探测者信息
- 预检失败 = `PolicyDenied` → handler **零调用**，无副作用
- **覆盖说明**：本作业三个模拟账户均为 `active`、未构造冻结账户——`status != "active"`
  分支已实现但无覆盖（与 deny/enabled 同属未启用分支，删掉该行 25 项测试仍绿，变异实测）；
  接真实账户体系时需补冻结/止付用例

### 5.5 转账处理器与超时模拟 (`transfer` handler)

```python
async def transfer(args: StrictArgs, ctx: ExecutionContext) -> dict[str, Any]:
    global TRANSFER_EXECUTIONS
    assert isinstance(args, TransferArgs)
    TRANSFER_EXECUTIONS += 1                          # handler 进入即计数（见下方说明）

    if args.amount > TRANSFER_SLOWDOWN_THRESHOLD:     # amount > 40000
        await asyncio.sleep(TRANSFER_SLOWDOWN_SECONDS)  # 故意制造超时（作业指定规则）

    from_key = (ctx.tenant_id, args.from_account)
    to_key = (ctx.tenant_id, args.to_account)

    # 锁内余额二次校验：precheck 在锁外，handler 引入挂起点（真实 I/O）后这里是唯一防线（见 §5.9）
    if args.amount > ACCOUNTS[from_key]["balance"]:
        raise PolicyDenied("BUSINESS_RULE_DENIED", "余额不足（锁内二次校验）")

    transfer_id = f"txn_{9001 + len(TRANSFERS):04d}"
    ACCOUNTS[from_key]["balance"] -= args.amount      # 扣款
    ACCOUNTS[to_key]["balance"] += args.amount        # 入账
    TRANSFERS[transfer_id] = {
        "transfer_id": transfer_id,
        "idempotency_key": ctx.idempotency_key,
        "tenant_id": ctx.tenant_id,
        "from_account": args.from_account,
        "to_account": args.to_account,
        "amount": float(args.amount),
        "status": "completed",
    }
    return {
        "transfer_id": transfer_id,
        "status": "completed",
        "from_balance": float(ACCOUNTS[from_key]["balance"]),
        "to_balance": float(ACCOUNTS[to_key]["balance"]),
    }
```

> **自增必须在函数最顶端，不能下移到 sleep 或二检之后**：
> - **不能放到 sleep 之后**：`asyncio.sleep` 被超时取消时协程直接终止，自增将**永不执行**，
>   超时路径计数为 0，直接推翻测试 #3 的 `TRANSFER_EXECUTIONS == 1`。
> - **不能放到二检之后**：根 `AGENTS.md` 第 4 节要求"数 fake 被调用的次数必须与记录的计数
>   严格相等"。被锁内二检拒绝的调用**确实进入了 handler**，计数必须包含它，
>   否则计数与真实 handler 调用次数不符。
>
> 因此语义是 **"handler 真实进入次数"**（与 §5.11 的字段名一致）：
> 门禁拒绝（precheck / 审批 / RBAC，handler 未进入）**不计入**；
> 超时、锁内二检拒绝（handler 已进入、副作用未落地）**计入**。
> 副作用是否落地由 `len(TRANSFERS)` 单独回答，两个计量口径分工明确。

**超时路径的执行时序**：

```
before_tool_call 通过（预检 45000 ≤ 50000，审批核销成功）
  → _execute: asyncio.timeout(2.0) 包裹 handler
  → handler: TRANSFER_EXECUTIONS += 1 → amount > 40000 → sleep(3.0)
  → 2.0s 到点，asyncio.timeout 取消 handler 协程（扣款、写入 TRANSFERS 均未发生）
  → invoke 捕获 TimeoutError → WRITE → TIMEOUT_UNKNOWN
```

- **计数语义**：`TRANSFER_EXECUTIONS` 在 **handler 进入时**自增——
  超时取消也计 1 次（handler 确实被进入过），但 `len(TRANSFERS)` 为 0（副作用未落地）。
  验收断言：`TRANSFER_EXECUTIONS == 1`（恰好一次，无自动重试）。完整口径见 §5.11
- 模拟环境中"超时 = 未执行"（sleep 被取消）。真实部署里下游可能已扣款——
  这正是 `TIMEOUT_UNKNOWN`（结果**未知**）而不是 `FAILED` 的语义依据（见 5.8）

### 5.6 治理策略与工具注册

```python
# 只读对账工具
ToolDefinition(
    name="get_account",
    description="查询账户余额与状态。",
    parameters_model=GetAccountArgs,
    policy=ToolPolicy(
        effect=Effect.READ,
        risk=Risk.LOW,
        permission="account:read",
        timeout_seconds=1,
        max_retries=2,          # 读操作瞬时失败可有限重试
        idempotent=True,
    ),
    handler=get_account,
    canonical_target=lambda args, ctx: f"{ctx.tenant_id}:{args.account_id}",
),

# 转账工具（本作业核心）
ToolDefinition(
    name="transfer",
    description="同租户内账户间转账。",
    parameters_model=TransferArgs,
    policy=ToolPolicy(
        effect=Effect.WRITE,          # 资金变动
        risk=Risk.HIGH,
        permission="transfer:execute",
        requires_approval=True,       # 必须审批，绑定全部参数
        timeout_seconds=2.0,          # < sleep 3.0，大额路径必然超时
        max_retries=0,                # 声明层：不重试
        idempotent=False,             # 非幂等写
    ),
    handler=transfer,
    precheck=check_transfer,
    canonical_target=lambda args, ctx: f"{ctx.tenant_id}:{args.from_account}",  # 写锁粒度
),
```

**`get_account` 处理器与租户隔离**：`get_account` 通过 `ctx.tenant_id` 限定查询范围，
不存在或跨租户账户返回同一错误（防枚举），handler 实现如下：

```python
async def get_account(args: StrictArgs, ctx: ExecutionContext) -> dict[str, Any]:
    assert isinstance(args, GetAccountArgs)
    account = ACCOUNTS.get((ctx.tenant_id, args.account_id))
    if account is None:
        raise PolicyDenied("BUSINESS_RULE_DENIED", "账户不存在或不属于当前租户")
    return {
        "account_id": args.account_id,
        "balance": float(account["balance"]),
        "status": account["status"],
        "owner_email": account["owner_email"],   # 返回后经 redact 脱敏为 ***@***
    }
```

`get_account` 无 `precheck`（租户隔离由 handler 内部 `(ctx.tenant_id, account_id)` 复合键查询实现）。

> **超时预算不覆盖等锁时间**：`asyncio.timeout` 在 `_execute_once()` 内（`:416`），
> 而锁获取在 `_execute()`（`:440`）。`timeout_seconds=2.0` 仅约束 handler 本体，
> 等待锁的时间不在预算内。本作业中因超时 handler 在 2s 内释放锁，影响被掩盖；
> 高并发下理论上存在锁饥饿风险（见 §10 已知限制 #8）。

**重试的双保险设计**：`max_retries=0` 只是策略声明；框架的恢复逻辑是强制条件——

```python
# governance.py（框架既有逻辑，迁移不改）
retries = policy.max_retries if policy.effect is Effect.READ or policy.idempotent else 0
```

即使工具策略误配 `max_retries=3`，WRITE + `idempotent=False` 的重试数仍被框架压回 0。
**决策逻辑只有一份（共享实现），不依赖各工具自觉。**

### 5.7 审批绑定与一次性消费

转账审批复用框架 `ApprovalStore`，无新增审批代码：

```
第一次调用（无审批）        → APPROVAL_REQUIRED，handler 零调用
  ↓
可信界面展示：转出 acc_1001 → 转入 acc_1002，金额 35000，事由 "…"
  ↓
用户确认 → approvals.approve() 签发凭证
           digest = sha256("transfer:" + 规范化参数(from/to/amount/reason))
  ↓
原参数 + approval_id 重入     → consume() 四重校验通过 → 执行 → 凭证标记 used
```

`consume()` 的校验（框架既有）：未使用、未过期（TTL）、`user_id` / `tenant_id` / `tool_name` /
参数摘要**全部匹配**。由此派生的转账验收点：

| 攻击 | 结果 | 测试 |
|------|------|------|
| 审批 35000 后改金额为 36000 重入 | 摘要不匹配 → `APPROVAL_REQUIRED`，handler 零调用 | 场景 6 |
| 同一审批重放第二次转账 | 凭证已 used → `APPROVAL_REQUIRED`，副作用恰为 1 | 场景 7 |
| 换个用户使用他人的 approval_id | `user_id` 不匹配 → `APPROVAL_REQUIRED`，handler 零调用 | 场景 21 |
| **超时后原参数直接重放** | 审批已在超时那次被消耗 → `APPROVAL_REQUIRED`，`executions` 仍为 1 | 场景 22 |
| 审批签发给别的租户的上下文使用 | `tenant_id` 不匹配 → `APPROVAL_REQUIRED`，handler 零调用 | 场景 23 |
| 签给其他工具的审批用于 transfer | `tool_name` 不匹配（且 digest 亦不匹配，双重拦截）→ `APPROVAL_REQUIRED` | 场景 24 |
| 过期审批（TTL 耗尽）使用 | `expires_at` 已过 → `APPROVAL_REQUIRED`，handler 零调用 | 场景 25 |

> **变异自检**（删校验分支跑测试）：`user_id` / `tenant_id` / TTL 三条删除后对应用例变红；
> `tool_name` 独立校验删除后仍绿——因 digest 以工具名为输入、两层独立拦截同一威胁
> （框架有意纵深，删任一层另一层兜底）。deny/enabled 硬拒绝分支本作业未启用
> （两工具均为默认值），无覆盖也无需覆盖。

最后一条是审批治理的纵深：`consume` 在执行前（`:406`），
**任何**执行期失败（超时 / `TOOL_ERROR` / `TEMPORARY_UNAVAILABLE`）都会烧掉审批，
不只是超时——模型无法拿旧审批自动重放同一笔转账。

### 5.8 超时语义与状态恢复（`TIMEOUT_UNKNOWN`）

课程语义：非幂等写超时后**禁止直接重放**，恢复动作是**查询真实状态**：

| 错误码 | Agent Loop 动作 | 是否自动重试 |
|--------|----------------|-------------|
| `TIMEOUT_UNKNOWN` | 调 `get_account` 对账 → 确认扣款是否发生 → 再决策 | **否**（禁止直接重放） |
| `APPROVAL_REQUIRED` | 暂停 Loop，进入可信确认界面 | 否 |
| `BUSINESS_RULE_DENIED` | 重新规划或转人工 | 否 |
| `INVALID_ARGUMENT` | 字段错误回填模型，修正参数后**新** Tool Call | 不用原参数重试 |

本作业的恢复演示（demo 轨迹 call_03 → call_04）：

```
transfer 45000 → TIMEOUT_UNKNOWN（结果未知）
  ↓
get_account(acc_1001) → 余额仍为 50000.00（未扣款）
  ↓
语义等价于"对账确认未执行" → 可安全重新发起（需新审批 + 新参数）
```

> 模拟环境中下游与 Runtime 同进程，sleep 被取消即转账未发生，查询立即得到确定答案；
> 真实部署中账户系统独立，"查余额"应升级为按**幂等键 / 交易号**查询转账单状态（见已知限制 #2）。

### 5.9 写锁串行与余额校验（含竞态可达性实测）

框架 `_execute()` 对写操作按 `canonical_target` 加锁：

```python
key = prepared.tool.canonical_target(prepared.args, ctx)   # "tenant_a:acc_1001"
lock = self._locks.setdefault(key, asyncio.Lock())
async with lock:
    return await self._execute_with_recovery(prepared, ctx)
```

> **⚠ 框架锁仅覆盖 handler，precheck 在锁外**：基线中 `before_tool_call`（`:401-402`）
> 执行 precheck，`_execute`（`:440-442`）才加锁。因此**锁不能保护"检查余额 → 扣款"的原子性**，
> 这正是 check-then-act 竞态的形状。

**竞态可达性分两种情况**（均已实测，见下表；复刻本方案 handler 形态与 §1.3 常量，
基线 `ToolRuntime` 原样引入，`asyncio.gather` 并发两笔）：

| 并发场景 | 结果 | 余额 | `TRANSFER_EXECUTIONS` |
|---------|------|------|----------------------|
| 两笔 35000（无挂起点） | 第一笔成功，第二笔 **precheck** 拒绝 | 15000（正确） | 1 |
| 两笔 45000（慢速 sleep，必超时） | 两笔均 `TIMEOUT_UNKNOWN` | 50000（未变） | 2 |
| 两笔 30000 + **扣款前** `sleep(0.05)`（真实 I/O 位置） | 第一笔成功，第二笔被**锁内二检**拒绝 | 20000（正确） | 2 |
| 两笔 30000 + 扣款**后** `sleep(0.05)`（对照组） | 第一笔成功，第二笔 **precheck** 拒绝 | 20000（正确） | 1 |

**解读**：

- **当前模拟实现下不可达**：handler 唯一挂起点是 `amount > 40000` 的慢速 `sleep`，
  且必然被 `timeout_seconds=2.0` 取消——副作用不落地。无注入时第一笔在第二笔的
  precheck 之前**完整跑完**，第二笔 precheck 看到的是已扣款余额（行 1）。
- **handler 出现非超时挂起点即可达**（行 3，扣款前注入——真实支付调用的等价形态）：
  第一笔持锁挂起期间，第二笔的 precheck 读到**旧余额放行**；等锁、进入 handler 后
  被锁内二检拦截。**接真实支付渠道后（`await` 调下游）此形态必然出现，二检是实际防线**，
  不是理论冗余。
- **注入位置影响结果**（行 3 vs 行 3b）：挂起点在扣款后时，precheck 能读到新余额自行拦截。
  真实 I/O 的挂起点在调用下游时（扣款前），因此行 3 才是有效结论；写实测数据必须声明注入位置。

**因此 `transfer` handler 内保留锁后余额二次校验**——它在真实 I/O 场景下被实测验证为实际防线：

```python
# precheck 在锁外；handler 引入挂起点（真实 I/O）后，下面的校验是唯一防线（实测行 3）
if args.amount > ACCOUNTS[from_key]["balance"]:
    raise PolicyDenied("BUSINESS_RULE_DENIED", "余额不足（锁内二次校验）")
```

**分工与代价**：
- precheck（锁外）：**早期拒绝**，减少审批消耗和审计噪声（无挂起点时也由它拦截）
- handler 重校验（锁内）：**实际防线**——handler 引入非超时挂起点（真实 I/O）后由它拦截（实测行 3）

> **这道防线不是"共享实现"，是刻意的防御性冗余**：判断逻辑确实出现了两份
> （违反根 `AGENTS.md` 第 2 节的默认要求）。接受它的理由是 precheck 与 handler 分处
> 锁外/锁内、**本就无法共用一次判断**；正确的根治手段是把 precheck 挪进锁内
> （需改框架）或使用数据库行锁（见 §10 已知限制 #8）。
>
> **实现约束**：二次校验必须写在 `TRANSFER_EXECUTIONS += 1` **之后**（自增置于函数顶端，
> 见 §5.5）——二检拒绝的调用已进入 handler，按 §5.11 计数语义**应当计入**。

**二次校验被拒时的审计形态**：`PolicyDenied` 在 `_execute` 内抛出，由 `invoke()` 的
`except PolicyDenied` 归一化为 `BUSINESS_RULE_DENIED`，但走的是 **execution 阶段审计**
（`ok=False`）而非 decision 阶段 deny——即"已授权但执行期业务失败"。
这与 precheck 被拒（decision 阶段 deny）在审计上形态不同，是刻意保留的区分。

### 5.10 审计与脱敏

复用框架两阶段审计 + 脱敏，转账零新增代码：

| 阶段 | 记录内容 | 转账示例 |
|------|---------|---------|
| decision（authorized） | 决策结果 + 主体 | `transfer` / allow / u_100 / tenant_a |
| execution（after_tool_call） | ok / error_code / duration_ms / argument_keys | `TIMEOUT_UNKNOWN` / ≈2000ms / `[amount, from_account, reason, to_account]` |
| 拒绝路径 | 同样写审计（decision=deny + 错误码） | `APPROVAL_REQUIRED` / `INVALID_ARGUMENT` 等 |

- **审计只记参数键名**（`argument_keys`），不记金额数值——审计自身不成泄漏源
- **模型视图脱敏**：`get_account` 返回的 `owner_email` 被邮箱正则脱敏为 `***@***`
- `ToolResult.to_model_payload()` 标记 `untrusted: true`——工具结果是不可信数据，不是控制指令
- 超时路径的 `duration_ms` ≈ 2000（timeout 上限），测试断言其落在合理区间

### 5.11 副作用计量

demo 与测试共享三个计量出口，结尾对账：

```python
{
    "transfer_executions": TRANSFER_EXECUTIONS,   # handler 真实进入次数（含超时那次）
    "completed_transfers": len(TRANSFERS),         # 真正落账的转账数
    "audit_records": len(audit.records),           # 审计记录数
}
```

**验收纪律**（仓库规范）：计数值必须与真实 handler 调用次数严格相等。按**拒绝发生在哪一层**区分：

| 拒绝/失败层 | 是否进入 handler | 计入 `TRANSFER_EXECUTIONS` | 计入 `completed_transfers` |
|------------|----------------|--------------------------|--------------------------|
| 门禁拒绝（参数 / 白名单 / RBAC / precheck / 审批） | 否 | **否** | 否 |
| 锁内二检拒绝 | 是 | **是** | 否 |
| 超时取消 | 是 | **是** | 否 |
| 成功 | 是 | **是** | **是** |

即：**`TRANSFER_EXECUTIONS` 回答"handler 被进入几次"，`completed_transfers` 回答"副作用落地几次"**，
两者不可互相替代。测试应对每次拒绝同时断言这两个值，以钉住"拒绝路径零副作用"。

## 6. 演示轨迹 (`../demo.py`)

| # | 场景 | 预期结果 | 治理点 |
|---|------|---------|--------|
| call_01 | 查询 `acc_1001` 余额 | OK，`owner_email` 脱敏为 `***@***` | 读操作 + 脱敏 |
| call_02 | 转账 45000（未审批） | `APPROVAL_REQUIRED`，handler 零调用 | 审批前置 |
| — | *`approvals.approve()` 签发 call_03 参数摘要绑定凭证* | — | 审批签发（call_02 到 call_03 之间） |
| call_03 | 转账 45000（携带 approval_id） | `TIMEOUT_UNKNOWN`，`TRANSFER_EXECUTIONS`=1，余额不变 | **超时模拟 + 不重试** |
| call_04 | 再查 `acc_1001` 余额 | 50000.00 未变 → 对账确认转账未执行 | 超时恢复路径 |
| call_05 | 转账 35000（新审批） | OK：`txn_9001`，双向余额变动（50000→15000 / 1200→36200） | 审批绑定 + 受控执行 |
| call_06 | 查询 `acc_1002` 余额 | 36200.00（到账确认） | 审计闭环 |
| call_07 | 参数携带 `user_id` / `approved` | `INVALID_ARGUMENT`（结构化字段错误） | `extra="forbid"` 防伪造 |
| call_08 | `from_account=acc_2001`（tenant_b） | `BUSINESS_RULE_DENIED` | 租户归属 + 防枚举 |
| call_09 | `acc_1002` 转 40000（余额 36200 不足） | `BUSINESS_RULE_DENIED` | 预检（先于审批消耗） |
| call_10 | plan 模式发起转账 | `PLAN_MODE_DENIED` | 执行层强制只读 |
| 尾声 | 副作用统计 | `transfer_executions=2`（1 超时 + 1 成功）/ `completed_transfers=1` / 审计记录数 | 计量对账 |

> 注意 call_03（大额超时）必须发生在 call_05（成功转账）**之前**：
> call_05 成功后 `acc_1001` 余额为 15000，45000 将无法通过余额预检，到不了超时模拟分支。

## 7. 测试体系 (`../test_transfer.py`)

全部测试打 **`ToolRuntime.invoke()` 公共入口**（治理框架的对外接口），
用副作用证据验收——不直接调 handler、不只测 Pydantic 校验（仓库规范：覆盖度落在对外接口层）。

### 测试场景

| # | 场景 | 断言要点（含副作用证据） |
|---|------|------------------------|
| 1 | 未审批转账被拒 | `APPROVAL_REQUIRED`；`TRANSFER_EXECUTIONS == 0`；`TRANSFERS == {}`（handler 零调用） |
| 2 | 审批后正常转账 | `ok is True`；双向余额变动正确；返回 `transfer_id`；审计含 execution 记录 |
| 3 | **大额超时不重试** | amount=45000 → `TIMEOUT_UNKNOWN`；`TRANSFER_EXECUTIONS == 1`（恰好一次）；`TRANSFERS` 无记录；余额不变；`duration_ms` ≥ 1900 |
| 4 | 阈值边界 | amount=40000.00（严格不大于）→ 不触发超时，转账成功 |
| 5 | 非幂等写瞬态失败不重试 | 注入抛 `TransientToolError` 的 handler → `TEMPORARY_UNAVAILABLE`，fake 调用次数 == 1 |
| 6 | 审批绑定参数 | 审批 35000 后以 36000 重入 → `APPROVAL_REQUIRED`；handler 零调用 |
| 7 | 审批一次性 | 同一审批重放 → `APPROVAL_REQUIRED`；成功副作用恰为 1 |
| 8 | 伪造身份字段 | 参数带 `user_id` / `approved` → `INVALID_ARGUMENT`；handler 零调用 |
| 9 | 越权转出（跨租户 from） | `BUSINESS_RULE_DENIED`；handler 零调用 |
| 10 | 越权转入（跨租户 to） | `BUSINESS_RULE_DENIED`（与不存在账户同响应） |
| 11 | 余额不足 | `BUSINESS_RULE_DENIED`；handler 零调用 |
| 12 | plan 模式拒绝写操作 | `PLAN_MODE_DENIED`；handler 零调用 |
| 13 | RBAC 拒绝 | 权限集缺 `transfer:execute` → `PERMISSION_DENIED`；handler 零调用 |
| 14 | 超时后对账恢复 | 超时后 `get_account` 余额未变（`TIMEOUT_UNKNOWN` ≠ 失败，恢复靠查询） |
| 15 | 审计与脱敏 | 两阶段审计均落记录；`argument_keys == [amount, from_account, reason, to_account]`；邮箱脱敏 |
| 16 | 工具发现不泄露治理字段 | `to_model_tool()` 输出仅含 `name` / `description` / `parameters`，无 handler / 权限 / 审批信息 |
| 17 | `model_tools()` 发现期白名单过滤 | `transfer` 不在 `allowed_tools` 时，`runtime.model_tools(ctx)` 不返回它；旧 ToolCall 重放仍被 `TOOL_NOT_ALLOWED` 拒绝 |
| 18 | 并发双花防护（**注入挂起点，钉住写锁与锁内二检**） | 外包一层**扣款前** `sleep(0.05)`（真实 I/O 等价形态，复用 shipped handler 不复制逻辑）打开竞态窗口，`gather` 两笔 30000：**恰一笔成功**、`TRANSFER_EXECUTIONS == 2`、余额 20000；`max_active == 1` 钉**写锁串行化**；第二笔审计含 `authorized` + execution 阶段 `BUSINESS_RULE_DENIED`（decision=allow）——证明拦截来自**锁内二检**而非锁外 precheck。变异自检：删二检或删写锁本用例必红 |
| 19 | `get_account` 跨租户读被拒 | tenant_a 的 ctx 查询 `acc_2001`（tenant_b）→ `BUSINESS_RULE_DENIED`（与不存在账户同响应） |
| 20 | 自转账被拒（`from == to`） | 带审批以隔离变量 → `BUSINESS_RULE_DENIED`，handler 零调用。变异自检：校验改 `if False` 本用例必红 |
| 21 | 盗用他人审批（`user_id` 绑定） | 签发给 u_100、u_999 盗用同一 approval_id 发起（其余条件均合法，precheck 放行）→ `APPROVAL_REQUIRED`，handler 零调用。变异自检：删绑定必红 |
| 22 | **超时烧审批**（原参数重放被拒） | 大额超时后同一 approval_id + 原参数重放 → `APPROVAL_REQUIRED`；`executions` 仍为 1，余额不变 |
| 23 | 审批绑定租户 | tenant_a 签发、tenant_b 上下文使用（补 tenant_b 转入方使 precheck 放行）→ `APPROVAL_REQUIRED`，handler 零调用。变异自检：删绑定必红 |
| 24 | 审批绑定工具名 | 签发给 get_account 的审批用于 transfer → `APPROVAL_REQUIRED`（独立校验 + digest 双重拦截） |
| 25 | 过期审批被拒（TTL） | `ttl_seconds=0` 签发并等待后使用 → `APPROVAL_REQUIRED`，handler 零调用。变异自检：删 TTL 校验必红 |

### 断言纪律（仓库规范）

- **钉正确行为**：`TIMEOUT_UNKNOWN` 是写超时的**正确**语义（结果未知，禁止重放），
  不是待修复的 bug——断言它，同时断言 handler 恰好一次（不盲重试）
- **计数与调用严格相等**：`TRANSFER_EXECUTIONS` 必须等于 handler 真实执行次数，
  拒绝路径不得虚增计数
- **护栏要有杀手用例（变异自检）**：写锁、锁内二检、`from == to` 等护栏代码被删除/禁用时，
  #18 / #20 必须变红。只断结果不断路径的用例不构成覆盖——旧版 #18 只断"恰一笔成功"，
  把 `gather` 改回顺序 `await` 结果不变（串行路径同样满足该断言），删二检/删锁均全绿，
  属于"绿灯通过却没测到任何并发性质"的假覆盖；真测竞态须注入挂起点（见 #18）
- 每个测试用独立 fixture 重置状态：
  - `ACCOUNTS` / `TRANSFERS` 必须**深拷贝**（handler 原地 mutate `balance`，浅拷贝会让后续用例的初始值已被前一个用例污染——§6 的演示数字 50000 / 1200 / 36200 全靠初始值成立）
  - `TRANSFER_EXECUTIONS` 是模块级 `global`，测试用 `monkeypatch.setattr(transfer, "TRANSFER_EXECUTIONS", 0)` 重置
  - `ApprovalStore` / `AuditSink` 每次新建实例

## 8. 扩展指南

### 新增治理工具（框架可扩展性的证明）

只需四件套，对 `governance.py` 无额外改动（consume 文案参数化已在迁移时完成）：

```python
class CloseAccountArgs(StrictArgs):        # ① 参数契约
    account_id: str = Field(pattern=r"^acc_[0-9]{4}$")

async def check_close(args, ctx): ...      # ② 业务预检（归属 / 状态机）

async def close_account(args, ctx): ...    # ③ handler

ToolDefinition(                            # ④ 策略 + 注册
    name="close_account",
    description="注销账户。",
    parameters_model=CloseAccountArgs,
    policy=ToolPolicy(effect=Effect.WRITE, risk=Risk.HIGH, permission="account:close",
                      requires_approval=True, timeout_seconds=2.0, max_retries=0),
    handler=close_account,
    precheck=check_close,
    canonical_target=lambda args, ctx: f"{ctx.tenant_id}:{args.account_id}",
)
```

### 接真实支付渠道

- handler 的 `asyncio.sleep` 换成真实支付 API 调用；`idempotency_key` 透传给渠道防渠道侧重复入账
- 超时后按幂等键查询渠道转账单状态（`TIMEOUT_UNKNOWN` 的对账升级）
- 账户余额换为数据库行级锁 / 乐观锁，`asyncio.Lock` 仅保留进程内语义

### 金额分级审批

在 `check_transfer` 或审批签发层增加金额阶梯（如 > 20000 需双人审批），
审批摘要已绑定金额，分级只需改签发策略，框架无需变更。

## 9. 设计亮点总结

| # | 亮点 | 说明 |
|---|------|------|
| 1 | **最小框架改动新增工具** | 转账 = 参数模型 + 策略 + 预检 + handler 四件套，仅 `ApprovalStore.consume()` 文案参数化一处例外 |
| 2 | **超时模拟可预期** | 阈值 40000 / sleep 3.0 / timeout 2.0 三数字闭环，大额路径必然超时且能通过预检 |
| 3 | **`TIMEOUT_UNKNOWN` 语义** | 写超时 = 结果未知 ≠ 失败；恢复靠查询（call_04 对账），禁止直接重放 |
| 4 | **重试双保险** | 策略声明 `max_retries=0` + 框架强制 `WRITE ∧ ¬idempotent → retries=0`，共享实现不靠自觉 |
| 5 | **超时烧审批** | 审批在执行前核销，超时后旧审批失效，重放需重新确认——纵深防重放 |
| 6 | **审批参数绑定** | sha256 摘要绑定 from/to/amount/reason，改一个字段重入即拒 |
| 7 | **租户归属防枚举** | 跨租户与不存在账户同响应 `BUSINESS_RULE_DENIED`，不泄露账户存在性 |
| 8 | **锁内外双重余额校验** | 框架锁仅覆盖 handler（precheck 在锁外）；handler 内锁后二检在真实 I/O 场景（扣款前挂起）下实测拦截双花，是实际防线而非理论冗余（§5.9） |
| 9 | **身份零暴露** | `extra="forbid"` + 归属只看 `ctx.tenant_id`，模型参数无法伪造身份 |
| 10 | **审计自防泄漏** | 只记 `argument_keys` 不记金额值；邮箱脱敏；两阶段审计含拒绝路径 |
| 11 | **副作用严格计量** | `TRANSFER_EXECUTIONS`（真实进入）与 `completed_transfers`（落账）分离，超时路径计入前者不计入后者 |
| 12 | **测试打公共入口** | 全部经 `ToolRuntime.invoke()` 验收，副作用证据（handler 零调用 / 恰好一次）钉住治理行为 |

## 10. 已知限制（Limitations）

以下议题在当前作业阶段可接受，生产部署前需解决：

| # | 限制 | 说明 |
|---|------|------|
| 1 | **存储全部进程内** | `ACCOUNTS` / `TRANSFERS` / 审批 / 审计均为进程内数据结构，重启丢失、多 worker 不共享。生产应替换为数据库 + 访问控制 |
| 2 | **超时模拟 = 未执行** | 模拟环境中 sleep 被取消即转账未发生，`get_account` 查余额即可对账；真实渠道超时后状态未知，需按幂等键 / 交易号查询转账单，`get_account` 不足以对账 |
| 3 | **转账限定同租户** | 跨租户 / 外部账户转账被治理边界拒绝；真实跨行转账需反洗钱、收款方校验等独立通道 |
| 4 | **`get_account` 无余额快照** | 并发转账进行中读取的余额可能是中间态；生产应提供交易流水视图而非单一余额 |
| 5 | **审批 TTL 固定 300 秒** | `ApprovalStore.approve()` 的 ttl 参数未按工具风险分级；大额转账可要求更短 TTL |
| 6 | **无 HTTP / CLI 层** | 对外接口是 `ToolRuntime.invoke()`；接入 Agent Loop（如 DeepSeek Function Calling）是后续作业主题 |
| 7 | **写锁仅进程内** | `asyncio.Lock` 只保护单进程并发；分布式部署需数据库行锁或分布式锁 |
| 8 | **框架锁与预检分离** | precheck 在 `before_tool_call`（锁外），handler 在 `_execute`（锁内）。当前模拟实现无挂起点、竞态不可达；但 handler 引入任何非超时挂起点（接真实支付渠道后必然）竞态即可达，此时 precheck 读旧余额失效，只能依赖 handler 内二检（已实测拦截）或改框架把 precheck 纳入锁 |
| 9 | **审批文案与 demo 域耦合** | `ApprovalStore.consume()` 错误消息硬编码"请确认本次退款的订单、金额和原因"（基线 `:253`），迁移时需参数化——这是"最小框架改动"主张的唯一例外 |
| 10 | **超时预算不涵盖等锁时间** | `asyncio.timeout` 在 `_execute_once` 内，锁获取在 `_execute`。等锁时间不受 `timeout_seconds` 约束，高并发下可能出现锁饥饿 |
