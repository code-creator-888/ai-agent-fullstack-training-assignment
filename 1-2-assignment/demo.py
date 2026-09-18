"""演示入口：转账工具治理轨迹（设计文档 §6 十条轨迹 + 副作用统计）。

执行顺序约束：call_03（大额超时）必须在 call_05（成功转账）之前——
call_05 成功后 acc_1001 余额为 15000，45000 无法通过余额预检，到不了超时模拟分支。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from governance import ExecutionContext, PermissionDecision, PermissionMode, ToolCall, ToolResult
from transfer import TransferArgs, build_governance
import transfer as transfer_module   # 可变状态必须经模块引用读取（from-import 的 int 是快照）

BASE_CTX = ExecutionContext(
    trace_id="trace_001",
    user_id="u_100",
    tenant_id="tenant_a",
    permissions=frozenset({"account:read", "transfer:execute"}),
    allowed_tools=frozenset({"get_account", "transfer"}),
)


def output(result: ToolResult) -> None:
    # 以治理视角打印单条结果：动作三态 + 稳定错误码 + 脱敏后内容。
    action = (
        PermissionDecision.ALLOW
        if result.ok
        else PermissionDecision.CONFIRM
        if result.error_code == "APPROVAL_REQUIRED"
        else PermissionDecision.DENY
    )
    print(
        json.dumps(
            {
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "ok": result.ok,
                "action": action,
                "code": result.error_code or "OK",
                "content": result.content,
            },
            ensure_ascii=False,
            default=str,
        )
    )


async def main() -> None:
    # 示例入口：演示发现、脱敏、审批拦截、超时治理与对账恢复的完整流程。
    runtime, approvals, audit = build_governance()

    read_call = ToolCall("call_01", "get_account", {"account_id": "acc_1001"})
    unapproved_call = ToolCall(
        "call_02",
        "transfer",
        {"from_account": "acc_1001", "to_account": "acc_1002", "amount": "45000.00", "reason": "大额转账超时演示"},
    )
    timeout_call = ToolCall(
        "call_03",
        "transfer",
        {"from_account": "acc_1001", "to_account": "acc_1002", "amount": "45000.00", "reason": "大额转账超时演示"},
    )
    recover_call = ToolCall("call_04", "get_account", {"account_id": "acc_1001"})
    approved_call = ToolCall(
        "call_05",
        "transfer",
        {"from_account": "acc_1001", "to_account": "acc_1002", "amount": "35000.00", "reason": "正常转账演示"},
    )
    confirm_call = ToolCall("call_06", "get_account", {"account_id": "acc_1002"})
    forged_call = ToolCall(
        "call_07",
        "transfer",
        {
            "from_account": "acc_1001",
            "to_account": "acc_1002",
            "amount": "35000.00",
            "reason": "伪造身份注入演示",
            "user_id": "u_100",
            "approved": True,
        },
    )
    cross_tenant_call = ToolCall(
        "call_08",
        "transfer",
        {"from_account": "acc_2001", "to_account": "acc_1002", "amount": "35000.00", "reason": "跨租户越权演示"},
    )
    insufficient_call = ToolCall(
        "call_09",
        "transfer",
        {"from_account": "acc_1002", "to_account": "acc_1001", "amount": "40000.00", "reason": "余额不足演示"},
    )
    plan_call = ToolCall(
        "call_10",
        "transfer",
        {"from_account": "acc_1001", "to_account": "acc_1002", "amount": "35000.00", "reason": "plan 模式演示"},
    )

    # call_01：读操作 + 邮箱脱敏。
    output(await runtime.invoke(read_call, BASE_CTX))
    # call_02：未审批转账被拒（handler 零调用）。
    output(await runtime.invoke(unapproved_call, BASE_CTX))

    # 审批签发（call_02 到 call_03 之间）：与 call_03 参数逐字段绑定。
    approvals.approve(
        "approval_01",
        user_id=BASE_CTX.user_id,
        tenant_id=BASE_CTX.tenant_id,
        tool_name=timeout_call.name,
        args=TransferArgs.model_validate(timeout_call.arguments),
    )
    approved_ctx = ExecutionContext(
        **{**asdict(BASE_CTX), "approval_id": "approval_01", "idempotency_key": timeout_call.id}
    )
    # call_03：大额超时——TIMEOUT_UNKNOWN，恰好一次执行，禁止自动重试。
    output(await runtime.invoke(timeout_call, approved_ctx))
    # call_04：超时恢复路径——对账确认转账未执行，余额不变。
    output(await runtime.invoke(recover_call, BASE_CTX))

    # call_05：新审批 + 正常金额，双向余额变动。
    approvals.approve(
        "approval_02",
        user_id=BASE_CTX.user_id,
        tenant_id=BASE_CTX.tenant_id,
        tool_name=approved_call.name,
        args=TransferArgs.model_validate(approved_call.arguments),
    )
    approved_ctx_02 = ExecutionContext(
        **{**asdict(BASE_CTX), "approval_id": "approval_02", "idempotency_key": approved_call.id}
    )
    output(await runtime.invoke(approved_call, approved_ctx_02))
    # call_06：到账确认。
    output(await runtime.invoke(confirm_call, BASE_CTX))
    # call_07：伪造身份字段被 extra="forbid" 拒绝。
    output(await runtime.invoke(forged_call, BASE_CTX))
    # call_08：跨租户转出被拒（与不存在同响应，防枚举）。
    output(await runtime.invoke(cross_tenant_call, BASE_CTX))
    # call_09：余额不足在预检被拒（先于审批消耗）。
    output(await runtime.invoke(insufficient_call, BASE_CTX))
    # call_10：plan 模式拒绝写操作（执行层强制只读）。
    output(
        await runtime.invoke(
            plan_call,
            ExecutionContext(**{**asdict(BASE_CTX), "mode": PermissionMode.PLAN}),
        )
    )

    # 尾声：副作用计量对账（transfer_executions 与 completed_transfers 分口径）。
    print(
        json.dumps(
            {
                "side_effects": {
                    "transfer_executions": transfer_module.TRANSFER_EXECUTIONS,
                    "completed_transfers": len(transfer_module.TRANSFERS),
                },
                "audit_records": len(audit.records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
