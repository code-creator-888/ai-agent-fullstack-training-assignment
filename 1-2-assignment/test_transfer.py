"""转账工具治理测试：全部打 ToolRuntime.invoke() 公共入口，用副作用证据验收。

覆盖设计文档 §7 的 25 项场景。不直接调 handler、不只测 Pydantic 校验
（仓库规范：覆盖度落在对外接口层）。
护栏用例（#18 写锁/二检、#20 自转账、#21–#25 审批四绑定+烧审批）做过变异自检：删护栏必红。
"""

from __future__ import annotations

import asyncio
import copy
import os
import sys
from decimal import Decimal

import pytest

# 确保 1-2-assignment 目录在 sys.path 中（从仓库根目录执行 pytest）。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from governance import (
    ApprovalStore,
    AuditSink,
    Effect,
    ExecutionContext,
    PermissionDecision,
    PermissionEngine,
    PermissionMode,
    Risk,
    StrictArgs,
    ToolCall,
    ToolDefinition,
    ToolPolicy,
    ToolRuntime,
    TransientToolError,
)
from transfer import (
    ACCOUNTS,
    TransferArgs,
    build_governance,
)
import transfer as transfer_module

# 初始账务快照：fixture 深拷贝的源（handler 原地 mutate balance，浅拷贝会跨用例污染）。
_INITIAL_ACCOUNTS = copy.deepcopy(ACCOUNTS)

TRANSFER_ARGUMENTS = {
    "from_account": "acc_1001",
    "to_account": "acc_1002",
    "amount": "35000.00",
    "reason": "正常转账测试",
}


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    # 每个用例独立状态：ACCOUNTS 深拷贝、TRANSFERS 清空、TRANSFER_EXECUTIONS 归零。
    monkeypatch.setattr(transfer_module, "ACCOUNTS", copy.deepcopy(_INITIAL_ACCOUNTS))
    monkeypatch.setattr(transfer_module, "TRANSFERS", {})
    monkeypatch.setattr(transfer_module, "TRANSFER_EXECUTIONS", 0)


def make_ctx(**overrides) -> ExecutionContext:
    # 构造 tenant_a 默认上下文；overrides 透传 approval_id / mode / permissions 等。
    base = dict(
        trace_id="trace_test",
        user_id="u_100",
        tenant_id="tenant_a",
        permissions=frozenset({"account:read", "transfer:execute"}),
        allowed_tools=frozenset({"get_account", "transfer"}),
    )
    base.update(overrides)
    return ExecutionContext(**base)


def approve_transfer(approvals: ApprovalStore, approval_id: str, arguments: dict) -> None:
    # 签发与参数逐字段绑定的审批凭证。
    approvals.approve(
        approval_id,
        user_id="u_100",
        tenant_id="tenant_a",
        tool_name="transfer",
        args=TransferArgs.model_validate(arguments),
    )


def executions() -> int:
    # handler 真实进入次数（门禁拒绝不计；超时/锁内二检拒绝/成功计入）。
    return transfer_module.TRANSFER_EXECUTIONS


def completed() -> int:
    # 真正落账的转账数。
    return len(transfer_module.TRANSFERS)


def balance(tenant: str, account_id: str) -> Decimal:
    return transfer_module.ACCOUNTS[(tenant, account_id)]["balance"]


# ---------- 场景 1：未审批转账被拒（handler 零调用） ----------

@pytest.mark.asyncio
async def test_unapproved_transfer_is_denied_before_execution() -> None:
    runtime, approvals, audit = build_governance()

    result = await runtime.invoke(ToolCall("call_01", "transfer", TRANSFER_ARGUMENTS), make_ctx())

    assert result.error_code == "APPROVAL_REQUIRED"
    assert result.ok is False
    # §5.10：拒绝路径同样写审计（decision=deny + 错误码）——与 #18 的 authorized
    # + execution 阶段拒绝形成判别。变异自检：decision 改 ALLOW 必红。
    assert len(audit.records) == 1
    assert audit.records[0]["decision"] == PermissionDecision.DENY
    assert audit.records[0]["error_code"] == "APPROVAL_REQUIRED"
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 2：审批后正常转账（双向余额变动 + 审计） ----------

@pytest.mark.asyncio
async def test_approved_transfer_moves_balance_both_ways() -> None:
    runtime, approvals, audit = build_governance()
    approve_transfer(approvals, "approval_01", TRANSFER_ARGUMENTS)

    result = await runtime.invoke(
        ToolCall("call_02", "transfer", TRANSFER_ARGUMENTS),
        make_ctx(approval_id="approval_01", idempotency_key="call_02"),
    )

    assert result.ok is True
    assert result.content["transfer_id"] == "txn_9001"
    assert result.content["status"] == "completed"
    assert balance("tenant_a", "acc_1001") == Decimal("15000.00")
    assert balance("tenant_a", "acc_1002") == Decimal("36200.00")   # 1200 + 35000
    assert executions() == 1
    assert completed() == 1
    execution_records = [r for r in audit.records if r.get("ok") is True]
    assert len(execution_records) == 1
    assert execution_records[0]["tool_call_id"] == "call_02"
    # 审计只记参数键名（不记金额值），键序为排序后的固定顺序。
    assert execution_records[0]["argument_keys"] == ["amount", "from_account", "reason", "to_account"]


# ---------- 场景 3：大额超时不重试（TIMEOUT_UNKNOWN + 恰好一次） ----------

@pytest.mark.asyncio
async def test_large_transfer_times_out_without_retry() -> None:
    runtime, approvals, audit = build_governance()
    arguments = {**TRANSFER_ARGUMENTS, "amount": "45000.00"}
    approve_transfer(approvals, "approval_01", arguments)

    result = await runtime.invoke(
        ToolCall("call_03", "transfer", arguments),
        make_ctx(approval_id="approval_01", idempotency_key="call_03"),
    )

    assert result.error_code == "TIMEOUT_UNKNOWN"
    assert result.ok is False
    assert executions() == 1          # 恰好一次：非幂等写不盲重试
    assert completed() == 0           # 副作用未落地
    assert balance("tenant_a", "acc_1001") == Decimal("50000.00")
    assert balance("tenant_a", "acc_1002") == Decimal("1200.00")
    record = [r for r in audit.records if r.get("error_code") == "TIMEOUT_UNKNOWN"][0]
    assert record["duration_ms"] >= 1900


# ---------- 场景 4：阈值边界（amount == 40000 严格不大于，不触发超时） ----------

@pytest.mark.asyncio
async def test_threshold_boundary_amount_equal_40000_succeeds() -> None:
    runtime, approvals, audit = build_governance()
    arguments = {**TRANSFER_ARGUMENTS, "amount": "40000.00"}
    approve_transfer(approvals, "approval_01", arguments)

    result = await runtime.invoke(
        ToolCall("call_04", "transfer", arguments),
        make_ctx(approval_id="approval_01", idempotency_key="call_04"),
    )

    assert result.ok is True
    assert balance("tenant_a", "acc_1001") == Decimal("10000.00")
    assert executions() == 1
    assert completed() == 1


# ---------- 场景 5：非幂等写瞬态失败不重试（误配 max_retries 仍被框架压回 0） ----------

@pytest.mark.asyncio
async def test_non_idempotent_write_does_not_retry_transient_failure() -> None:
    calls = []

    async def flaky_handler(args: StrictArgs, ctx: ExecutionContext) -> dict:
        calls.append(1)
        raise TransientToolError("依赖暂时不可用")

    flaky_tool = ToolDefinition(
        name="transfer",
        description="测试用瞬态失败 handler。",
        parameters_model=TransferArgs,
        policy=ToolPolicy(
            effect=Effect.WRITE,
            risk=Risk.HIGH,
            permission="transfer:execute",
            requires_approval=False,
            timeout_seconds=1,
            max_retries=3,          # 故意误配：框架应压回 0
            idempotent=False,
        ),
        handler=flaky_handler,
        canonical_target=lambda args, ctx: f"{ctx.tenant_id}:{args.from_account}",
    )
    runtime = ToolRuntime([flaky_tool], PermissionEngine(), ApprovalStore(), AuditSink())

    result = await runtime.invoke(ToolCall("call_05", "transfer", TRANSFER_ARGUMENTS), make_ctx())

    assert result.error_code == "TEMPORARY_UNAVAILABLE"
    assert result.ok is False
    assert len(calls) == 1           # 误配 max_retries=3 仍不重试


# ---------- 场景 6：审批绑定参数（改金额重入即失效） ----------

@pytest.mark.asyncio
async def test_approval_is_bound_to_canonical_arguments() -> None:
    runtime, approvals, audit = build_governance()
    approve_transfer(approvals, "approval_01", TRANSFER_ARGUMENTS)
    tampered = {**TRANSFER_ARGUMENTS, "amount": "36000.00"}

    result = await runtime.invoke(
        ToolCall("call_06", "transfer", tampered),
        make_ctx(approval_id="approval_01", idempotency_key="call_06"),
    )

    assert result.error_code == "APPROVAL_REQUIRED"
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 7：审批一次性（同审批重放被拒） ----------

@pytest.mark.asyncio
async def test_one_time_approval_cannot_be_replayed() -> None:
    runtime, approvals, audit = build_governance()
    # 用小额（10000）：预检先于审批核销，第一笔转走后余额须仍覆盖第二笔，
    # 否则重放在预检就被 BUSINESS_RULE_DENIED 拦截，测不到审批一次性。
    arguments = {**TRANSFER_ARGUMENTS, "amount": "10000.00"}
    approve_transfer(approvals, "approval_01", arguments)

    first = await runtime.invoke(
        ToolCall("call_07", "transfer", arguments),
        make_ctx(approval_id="approval_01", idempotency_key="call_07"),
    )
    second = await runtime.invoke(
        ToolCall("call_08", "transfer", arguments),
        make_ctx(approval_id="approval_01", idempotency_key="call_08"),
    )

    assert first.ok is True
    assert second.error_code == "APPROVAL_REQUIRED"
    assert executions() == 1
    assert completed() == 1


# ---------- 场景 8：伪造身份字段（extra="forbid"，结构化字段错误路径） ----------

@pytest.mark.asyncio
async def test_forged_identity_fields_are_rejected() -> None:
    runtime, approvals, audit = build_governance()
    forged = {**TRANSFER_ARGUMENTS, "user_id": "u_100", "approved": True}

    result = await runtime.invoke(ToolCall("call_09", "transfer", forged), make_ctx())

    assert result.error_code == "INVALID_ARGUMENT"
    assert result.ok is False
    assert executions() == 0
    assert completed() == 0
    # content 为结构化错误列表（PolicyDenied content=errors，经 redact 保留结构）。
    paths = {item["path"] for item in result.content if isinstance(item, dict)}
    assert {"user_id", "approved"} <= paths


# ---------- 场景 9：越权转出（跨租户 from） ----------

@pytest.mark.asyncio
async def test_cross_tenant_from_account_is_denied() -> None:
    runtime, approvals, audit = build_governance()

    result = await runtime.invoke(
        ToolCall("call_10", "transfer", {**TRANSFER_ARGUMENTS, "from_account": "acc_2001"}),
        make_ctx(),
    )

    assert result.error_code == "BUSINESS_RULE_DENIED"
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 10：越权转入（跨租户 to，与不存在同响应防枚举） ----------

@pytest.mark.asyncio
async def test_cross_tenant_to_account_denied_same_as_missing() -> None:
    runtime, approvals, audit = build_governance()

    cross = await runtime.invoke(
        ToolCall("call_11", "transfer", {**TRANSFER_ARGUMENTS, "to_account": "acc_2001"}),
        make_ctx(),
    )
    missing = await runtime.invoke(
        ToolCall("call_12", "transfer", {**TRANSFER_ARGUMENTS, "to_account": "acc_9999"}),
        make_ctx(),
    )

    assert cross.error_code == "BUSINESS_RULE_DENIED"
    assert missing.error_code == "BUSINESS_RULE_DENIED"
    assert cross.content == missing.content   # 防枚举：同响应
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 11：余额不足（预检先于审批消耗） ----------

@pytest.mark.asyncio
async def test_insufficient_balance_is_denied_at_precheck() -> None:
    runtime, approvals, audit = build_governance()

    result = await runtime.invoke(
        ToolCall(
            "call_13",
            "transfer",
            {**TRANSFER_ARGUMENTS, "from_account": "acc_1002", "amount": "40000.00"},
        ),
        make_ctx(),
    )

    assert result.error_code == "BUSINESS_RULE_DENIED"
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 12：plan 模式拒绝写操作（先于审批与 handler） ----------

@pytest.mark.asyncio
async def test_plan_mode_denies_write_before_approval_and_handler() -> None:
    runtime, approvals, audit = build_governance()
    approve_transfer(approvals, "approval_01", TRANSFER_ARGUMENTS)

    result = await runtime.invoke(
        ToolCall("call_14", "transfer", TRANSFER_ARGUMENTS),
        make_ctx(mode=PermissionMode.PLAN, approval_id="approval_01"),
    )

    assert result.error_code == "PLAN_MODE_DENIED"
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 13：RBAC 拒绝（缺 transfer:execute） ----------

@pytest.mark.asyncio
async def test_rbac_denial_keeps_handler_at_zero_calls() -> None:
    runtime, approvals, audit = build_governance()

    result = await runtime.invoke(
        ToolCall("call_15", "transfer", TRANSFER_ARGUMENTS),
        make_ctx(permissions=frozenset({"account:read"})),
    )

    assert result.error_code == "PERMISSION_DENIED"
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 14：超时后对账恢复（余额未变 → 确认未执行） ----------

@pytest.mark.asyncio
async def test_timeout_recovery_via_balance_query() -> None:
    runtime, approvals, audit = build_governance()
    arguments = {**TRANSFER_ARGUMENTS, "amount": "45000.00"}
    approve_transfer(approvals, "approval_01", arguments)

    timeout_result = await runtime.invoke(
        ToolCall("call_16", "transfer", arguments),
        make_ctx(approval_id="approval_01", idempotency_key="call_16"),
    )
    recovery_result = await runtime.invoke(
        ToolCall("call_17", "get_account", {"account_id": "acc_1001"}),
        make_ctx(),
    )

    assert timeout_result.error_code == "TIMEOUT_UNKNOWN"
    assert recovery_result.ok is True
    assert recovery_result.content["balance"] == 50000.00   # 余额未变 → 对账确认未执行
    assert completed() == 0


# ---------- 场景 15：审计与脱敏 ----------

@pytest.mark.asyncio
async def test_audit_and_redaction() -> None:
    runtime, approvals, audit = build_governance()

    result = await runtime.invoke(ToolCall("call_18", "get_account", {"account_id": "acc_1001"}), make_ctx())

    assert result.ok is True
    # 邮箱脱敏：不含原文，整体被替换为 ***@***。
    assert "alice@example.com" not in str(result.content)
    assert "***@***" in str(result.content["owner_email"])
    # 两阶段审计：authorized（decision 阶段）+ execution（ok=True）。
    authorized = [r for r in audit.records if r.get("stage") == "authorized"]
    execution = [r for r in audit.records if r.get("ok") is True]
    assert len(authorized) == 1
    assert len(execution) == 1
    # 审计只记参数键名，不记余额数值。
    assert execution[0]["argument_keys"] == ["account_id"]
    assert "50000" not in str(audit.records)


# ---------- 场景 16：工具发现不泄露治理字段 ----------

@pytest.mark.asyncio
async def test_tool_discovery_does_not_leak_governance_fields() -> None:
    runtime, approvals, audit = build_governance()

    tools = runtime.model_tools(make_ctx())

    transfer_tool = next(t for t in tools if t["function"]["name"] == "transfer")
    assert set(transfer_tool["function"].keys()) == {"name", "description", "parameters"}
    dumped = str(tools)
    assert "handler" not in dumped
    assert "permission" not in dumped
    assert "approval" not in dumped
    assert "precheck" not in dumped


# ---------- 场景 17：model_tools() 发现期白名单过滤 + 执行期双重校验 ----------

@pytest.mark.asyncio
async def test_discovery_and_execution_both_enforce_whitelist() -> None:
    runtime, approvals, audit = build_governance()

    visible = runtime.model_tools(make_ctx(allowed_tools=frozenset({"get_account"})))
    names = {t["function"]["name"] for t in visible}
    assert "transfer" not in names
    assert "get_account" in names

    replay = await runtime.invoke(
        ToolCall("call_19", "transfer", TRANSFER_ARGUMENTS),
        make_ctx(allowed_tools=frozenset({"get_account"})),
    )
    assert replay.error_code == "TOOL_NOT_ALLOWED"
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 18：并发双花防护（注入挂起点——钉住写锁与锁内二检） ----------

@pytest.mark.asyncio
async def test_concurrent_double_spend_is_prevented_under_suspension() -> None:
    # shipped handler 无挂起点时，第二笔永远被锁外 precheck 拦下——锁与二检都不显形，
    # 删掉任一护栏测试仍绿（变异测试发现的盲区）。在真实 transfer 前注入挂起点
    # （模拟扣款前的下游 I/O）把竞态窗口真实打开，本用例同时钉住两处护栏：
    # - 删掉锁内二检 → 第二笔照样扣款（余额 -10000），"恰一笔成功"断言变红；
    # - 删掉框架写锁 → 两个 handler 同时进入，max_active == 1 断言变红。
    active = 0
    max_active = 0

    async def suspended_transfer(args: StrictArgs, ctx: ExecutionContext) -> dict:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.05)                          # 挂起点：扣款前（真实 I/O 位置）
            return await transfer_module.transfer(args, ctx)   # 复用 shipped handler 全部逻辑
        finally:
            active -= 1

    tool = ToolDefinition(
        name="transfer",
        description="测试用：带挂起点的转账 handler。",
        parameters_model=TransferArgs,
        policy=ToolPolicy(
            effect=Effect.WRITE,
            risk=Risk.HIGH,
            permission="transfer:execute",
            requires_approval=True,
            timeout_seconds=1,
            max_retries=0,
            idempotent=False,
        ),
        handler=suspended_transfer,
        precheck=transfer_module.check_transfer,
        canonical_target=lambda args, ctx: f"{ctx.tenant_id}:{args.from_account}",
    )
    approvals = ApprovalStore()
    audit = AuditSink()
    runtime = ToolRuntime([tool], PermissionEngine(), approvals, audit)

    arguments = {**TRANSFER_ARGUMENTS, "amount": "30000.00"}
    approve_transfer(approvals, "approval_01", arguments)
    approve_transfer(approvals, "approval_02", arguments)

    results = await asyncio.gather(
        runtime.invoke(
            ToolCall("call_20", "transfer", arguments),
            make_ctx(approval_id="approval_01", idempotency_key="call_20"),
        ),
        runtime.invoke(
            ToolCall("call_21", "transfer", arguments),
            make_ctx(approval_id="approval_02", idempotency_key="call_21"),
        ),
    )

    outcomes = sorted(r.error_code or "OK" for r in results)
    assert outcomes == ["BUSINESS_RULE_DENIED", "OK"]              # 恰一笔成功
    assert completed() == 1
    assert executions() == 2                                       # 两个 handler 都进入过（含二检拒绝的那个）
    assert balance("tenant_a", "acc_1001") == Decimal("20000.00")  # 50000 - 30000
    assert max_active == 1                                         # 写锁串行化：执行区间不重叠
    # 审计证据：第二笔含 authorized 记录（过了全部门禁、进入执行阶段）——
    # 证明拦截它的是锁内二检（execution 阶段拒绝），而非锁外 precheck（decision 阶段拒绝）。
    second = [r for r in audit.records if r.get("tool_call_id") == "call_21"]
    assert len(second) == 2
    assert second[0].get("stage") == "authorized"
    assert second[1].get("ok") is False
    assert second[1].get("error_code") == "BUSINESS_RULE_DENIED"
    assert second[1].get("decision") == PermissionDecision.ALLOW


# ---------- 场景 19：get_account 跨租户读被拒（与不存在同响应防枚举） ----------

@pytest.mark.asyncio
async def test_get_account_cross_tenant_read_is_denied() -> None:
    runtime, approvals, audit = build_governance()

    result = await runtime.invoke(ToolCall("call_22", "get_account", {"account_id": "acc_2001"}), make_ctx())
    missing = await runtime.invoke(ToolCall("call_23", "get_account", {"account_id": "acc_9999"}), make_ctx())

    assert result.error_code == "BUSINESS_RULE_DENIED"
    assert result.content == missing.content   # 防枚举：同响应
    assert "carol" not in str(result.content)
    assert "80000" not in str(result.content)


# ---------- 场景 20：自转账被拒（from == to，预检拦截） ----------

@pytest.mark.asyncio
async def test_self_transfer_is_denied_at_precheck() -> None:
    runtime, approvals, audit = build_governance()
    # 带审批以隔离变量：这条用例的唯一防线就是 from == to 业务校验。
    # 变异自检：把校验改为 if False，本用例必须变红（AGENTS.md §5）。
    arguments = {**TRANSFER_ARGUMENTS, "to_account": "acc_1001"}
    approve_transfer(approvals, "approval_01", arguments)

    result = await runtime.invoke(
        ToolCall("call_24", "transfer", arguments),
        make_ctx(approval_id="approval_01", idempotency_key="call_24"),
    )

    assert result.error_code == "BUSINESS_RULE_DENIED"
    assert executions() == 0          # precheck 于 handler 之前拦截，零调用
    assert completed() == 0


# ---------- 场景 21：盗用他人审批（user_id 绑定，§5.7 第三行） ----------

@pytest.mark.asyncio
async def test_stolen_approval_from_another_user_is_denied() -> None:
    runtime, approvals, audit = build_governance()
    # 审批签发给 u_100；u_999 盗用该 approval_id 发起（tenant_a / 账户 / 余额均合法，
    # precheck 必然放行）——唯一能拦的是 consume 的 user_id 绑定。
    # 变异自检：删 `approval.user_id == ctx.user_id` 后本用例必红（转账会真执行）。
    approve_transfer(approvals, "approval_01", TRANSFER_ARGUMENTS)

    result = await runtime.invoke(
        ToolCall("call_25", "transfer", TRANSFER_ARGUMENTS),
        make_ctx(user_id="u_999", approval_id="approval_01", idempotency_key="call_25"),
    )

    assert result.error_code == "APPROVAL_REQUIRED"
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 22：超时烧审批（同 approval_id 原参数重放被拒，§5.7 第四行） ----------

@pytest.mark.asyncio
async def test_timeout_burns_approval_replay_is_denied() -> None:
    runtime, approvals, audit = build_governance()
    arguments = {**TRANSFER_ARGUMENTS, "amount": "45000.00"}
    approve_transfer(approvals, "approval_01", arguments)

    timeout_result = await runtime.invoke(
        ToolCall("call_26", "transfer", arguments),
        make_ctx(approval_id="approval_01", idempotency_key="call_26"),
    )
    # 超时后拿同一 approval_id + 原参数重放：consume 在执行前核销，审批已烧。
    replay_result = await runtime.invoke(
        ToolCall("call_27", "transfer", arguments),
        make_ctx(approval_id="approval_01", idempotency_key="call_27"),
    )

    assert timeout_result.error_code == "TIMEOUT_UNKNOWN"
    assert replay_result.error_code == "APPROVAL_REQUIRED"
    assert executions() == 1          # 仍只有超时那一次，重放零调用
    assert completed() == 0
    assert balance("tenant_a", "acc_1001") == Decimal("50000.00")


# ---------- 场景 23：审批绑定租户（tenant_a 签发，tenant_b 上下文使用） ----------

@pytest.mark.asyncio
async def test_approval_is_bound_to_tenant() -> None:
    runtime, approvals, audit = build_governance()
    # tenant_b 原本只有 acc_2001 一个账户，无法构造同租户转账；补一个转入方
    # 使 precheck 放行（含 80000 余额覆盖 500），把拦截点逼到 consume 的 tenant 绑定。
    transfer_module.ACCOUNTS[("tenant_b", "acc_2002")] = {
        "balance": Decimal("100.00"),
        "status": "active",
        "owner_email": "dave@example.com",
    }
    arguments = {
        "from_account": "acc_2001",
        "to_account": "acc_2002",
        "amount": "500.00",
        "reason": "跨租户盗用审批测试",
    }
    # 审批签发给 tenant_a，用 tenant_b 的上下文发起。
    approvals.approve(
        "approval_01",
        user_id="u_100",
        tenant_id="tenant_a",
        tool_name="transfer",
        args=TransferArgs.model_validate(arguments),
    )

    result = await runtime.invoke(
        ToolCall("call_28", "transfer", arguments),
        make_ctx(tenant_id="tenant_b", approval_id="approval_01", idempotency_key="call_28"),
    )

    assert result.error_code == "APPROVAL_REQUIRED"
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 24：审批绑定工具名（签给别的工具的审批不能用于 transfer） ----------

@pytest.mark.asyncio
async def test_approval_signed_for_other_tool_is_denied() -> None:
    runtime, approvals, audit = build_governance()
    # 故意把审批签发给 get_account，再用于 transfer 调用。
    # 注：tool_name 同时是 digest 的输入（_approval_digest），本行为由"独立 tool_name
    # 校验 + digest 校验"双重拦截——删除其中任一层另一层仍拦（框架有意纵深，非盲区）。
    approvals.approve(
        "approval_01",
        user_id="u_100",
        tenant_id="tenant_a",
        tool_name="get_account",
        args=TransferArgs.model_validate(TRANSFER_ARGUMENTS),
    )

    result = await runtime.invoke(
        ToolCall("call_29", "transfer", TRANSFER_ARGUMENTS),
        make_ctx(approval_id="approval_01", idempotency_key="call_29"),
    )

    assert result.error_code == "APPROVAL_REQUIRED"
    assert executions() == 0
    assert completed() == 0


# ---------- 场景 25：过期审批被拒（TTL 绑定） ----------

@pytest.mark.asyncio
async def test_expired_approval_is_denied() -> None:
    runtime, approvals, audit = build_governance()
    # 签发瞬时过期（ttl_seconds=0）的审批，等待后使用。
    approvals.approve(
        "approval_01",
        user_id="u_100",
        tenant_id="tenant_a",
        tool_name="transfer",
        args=TransferArgs.model_validate(TRANSFER_ARGUMENTS),
        ttl_seconds=0,
    )
    await asyncio.sleep(0.05)   # 确保已越过 expires_at

    result = await runtime.invoke(
        ToolCall("call_30", "transfer", TRANSFER_ARGUMENTS),
        make_ctx(approval_id="approval_01", idempotency_key="call_30"),
    )

    assert result.error_code == "APPROVAL_REQUIRED"
    assert executions() == 0
    assert completed() == 0
