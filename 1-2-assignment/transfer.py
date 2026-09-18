"""转账工具：治理框架上的账户域工具（作业二核心）。

以"参数模型 + 策略 + 预检 + handler"四件套接入 ToolRuntime，框架零逻辑改动。
超时模拟规则（作业指定）：amount > 40000 时执行 await asyncio.sleep(3.0) 故意制造超时，
配合 transfer 工具 timeout_seconds=2.0，大额路径必然 TIMEOUT_UNKNOWN。
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from governance import (
    ApprovalStore,
    AuditSink,
    Effect,
    ExecutionContext,
    PermissionEngine,
    PolicyDenied,
    Risk,
    StrictArgs,
    ToolDefinition,
    ToolPolicy,
    ToolRuntime,
)
from pydantic import Field

# 超时模拟常量（作业指定规则，置于类定义之外——仓库 Pydantic 常量规范）。
TRANSFER_SLOWDOWN_THRESHOLD = Decimal("40000")   # 大于此值触发模拟延迟
TRANSFER_SLOWDOWN_SECONDS = 3.0                  # 延迟 3.0s > timeout_seconds(2.0)，必然超时


class TransferArgs(StrictArgs):
    # 转账参数契约：约束转出/转入账户、金额和事由；身份字段零暴露。
    from_account: str = Field(pattern=r"^acc_[0-9]{4}$")
    to_account: str = Field(pattern=r"^acc_[0-9]{4}$")
    amount: Decimal = Field(gt=0, max_digits=10, decimal_places=2)
    reason: str = Field(min_length=4, max_length=200)


class GetAccountArgs(StrictArgs):
    # 账户查询参数契约：仅接受格式受限的账户标识。
    account_id: str = Field(pattern=r"^acc_[0-9]{4}$")


# 账户按 (租户, 账户号) 隔离，归属由 ctx.tenant_id 决定，模型参数无权声明。
ACCOUNTS: dict[tuple[str, str], dict[str, Any]] = {
    ("tenant_a", "acc_1001"): {
        "balance": Decimal("50000.00"),
        "status": "active",
        "owner_email": "alice@example.com",
    },
    ("tenant_a", "acc_1002"): {
        "balance": Decimal("1200.00"),
        "status": "active",
        "owner_email": "bob@example.com",
    },
    ("tenant_b", "acc_2001"): {
        "balance": Decimal("80000.00"),
        "status": "active",
        "owner_email": "carol@example.com",
    },
}
# 已完成的转账（按 transfer_id 索引）。
TRANSFERS: dict[str, dict[str, Any]] = {}
# handler 真实进入次数（副作用计量）：门禁拒绝不计入，超时/锁内二检拒绝/成功计入。
TRANSFER_EXECUTIONS = 0


async def get_account(args: StrictArgs, ctx: ExecutionContext) -> dict[str, Any]:
    # 查询处理器：以 (租户, 账户号) 复合键限定范围，跨租户与不存在同响应（防枚举）。
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


async def check_transfer(args: StrictArgs, ctx: ExecutionContext) -> None:
    # 转账业务预检：在执行副作用前校验归属、状态机和余额（先于审批核销）。
    assert isinstance(args, TransferArgs)
    from_key = (ctx.tenant_id, args.from_account)
    to_key = (ctx.tenant_id, args.to_account)

    if args.from_account == args.to_account:
        raise PolicyDenied("BUSINESS_RULE_DENIED", "转出与转入账户不能相同")
    if ACCOUNTS.get(from_key) is None or ACCOUNTS.get(to_key) is None:
        # 与"不存在"同响应，不泄露其他租户账户的存在性（防账户枚举）。
        raise PolicyDenied("BUSINESS_RULE_DENIED", "账户不存在或不属于当前租户")
    if ACCOUNTS[from_key]["status"] != "active" or ACCOUNTS[to_key]["status"] != "active":
        raise PolicyDenied("BUSINESS_RULE_DENIED", "账户状态不可用")
    if args.amount > ACCOUNTS[from_key]["balance"]:
        raise PolicyDenied("BUSINESS_RULE_DENIED", "余额不足")


async def transfer(args: StrictArgs, ctx: ExecutionContext) -> dict[str, Any]:
    # 转账写入处理器：同租户内扣款入账；大额触发超时模拟。
    global TRANSFER_EXECUTIONS
    assert isinstance(args, TransferArgs)
    # 自增必须在函数体最顶端（设计文档 §5.5）：
    # - 放到 sleep 之后：超时取消时协程终止，自增永不执行，超时路径计数为 0；
    # - 放到二检之后：被锁内二检拒绝的调用确实进入了 handler，按计数语义应计入。
    TRANSFER_EXECUTIONS += 1

    if args.amount > TRANSFER_SLOWDOWN_THRESHOLD:     # amount > 40000
        await asyncio.sleep(TRANSFER_SLOWDOWN_SECONDS)  # 故意制造超时（作业指定规则）

    from_key = (ctx.tenant_id, args.from_account)
    to_key = (ctx.tenant_id, args.to_account)

    # 锁内余额二次校验：precheck 在锁外，handler 引入挂起点（真实 I/O）后这里是唯一防线（见设计文档 §5.9）。
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


def build_governance() -> tuple[ToolRuntime, ApprovalStore, AuditSink]:
    # 组合根：注册账户域工具及其治理策略，并装配运行时依赖。
    approvals = ApprovalStore()
    audit = AuditSink()
    tools = [
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
    ]
    return ToolRuntime(tools, PermissionEngine(), approvals, audit), approvals, audit


__all__ = [
    "ACCOUNTS",
    "TRANSFERS",
    "TRANSFER_EXECUTIONS",
    "TRANSFER_SLOWDOWN_THRESHOLD",
    "TRANSFER_SLOWDOWN_SECONDS",
    "GetAccountArgs",
    "TransferArgs",
    "build_governance",
    "check_transfer",
    "get_account",
    "transfer",
]
