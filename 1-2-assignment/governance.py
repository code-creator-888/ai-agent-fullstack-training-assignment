"""工具治理框架：从课程 2-4 tool_governance_demo.py 迁移。

与课程基线的差异（设计文档 §5.1，唯一允许的改动）：
- ApprovalStore.consume() 的错误文案参数化——基线硬编码"退款"文案，与账户域不符。

迁移边界：课程基线中的订单域工具（get_order / create_refund / run_shell 及 ORDERS / REFUNDS）
不迁移；框架组件（Runtime / 审批 / 审计 / 脱敏 / 投影）全量保留。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StrictArgs(BaseModel):
    # 所有工具参数的共同边界：拒绝未声明字段，并清理字符串首尾空白。
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Effect(StrEnum):
    # 工具副作用分类：用于区分读写执行、并发控制和重试边界。
    READ = "read"
    WRITE = "write"
    SHELL = "shell"


class Risk(StrEnum):
    # 工具风险分级：为审批与审计策略提供风险依据。
    LOW = "low"
    HIGH = "high"


class PermissionDecision(StrEnum):
    # 授权决策结果：表达允许、拒绝和需人工确认三种治理状态。
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


class PermissionMode(StrEnum):
    # 执行模式只能由服务端设置，模型参数无权变更。
    DEFAULT = "default"
    PLAN = "plan"
    BYPASS_PERMISSIONS = "bypassPermissions"
    DONT_ASK = "dontAsk"


@dataclass(frozen=True)
class ToolCall:
    # Harness 传入的原始调用请求，承载调用标识、工具名和模型生成的参数。
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionContext:
    # 每次调用携带的可信身份与授权上下文，不接受模型自行声明权限。
    trace_id: str
    user_id: str
    tenant_id: str
    permissions: frozenset[str]
    allowed_tools: frozenset[str]
    approval_id: str | None = None
    idempotency_key: str | None = None
    mode: PermissionMode = PermissionMode.DEFAULT


@dataclass(frozen=True)
class ToolPolicy:
    # 工具治理策略：集中声明副作用、风险、权限、审批、超时与重试规则。
    effect: Effect
    risk: Risk
    permission: str
    denied: bool = False
    requires_confirmation: bool = False
    requires_approval: bool = False
    timeout_seconds: float = 3.0
    max_retries: int = 0
    idempotent: bool = False
    enabled: bool = True


Handler = Callable[[StrictArgs, ExecutionContext], Awaitable[dict[str, Any]]]
Precheck = Callable[[StrictArgs, ExecutionContext], Awaitable[None]]
CanonicalTarget = Callable[[StrictArgs, ExecutionContext], str]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    # 工具注册元数据：将模型契约、治理策略和实际处理器绑定为受控能力。
    name: str
    description: str
    parameters_model: type[StrictArgs]
    policy: ToolPolicy
    handler: Handler
    canonical_target: CanonicalTarget
    precheck: Precheck | None = None

    def to_model_tool(self) -> dict[str, Any]:
        # 仅导出公开描述和参数 Schema，供模型发现可调用的受控工具。
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_model.model_json_schema(),
            },
        }


@dataclass(frozen=True)
class PreparedCall:
    # 已通过调用前治理的内部对象，携带解析后的参数与授权结论。
    call: ToolCall
    tool: ToolDefinition
    args: StrictArgs
    decision: PermissionDecision


@dataclass(frozen=True)
class ToolResult:
    # 返回给 Harness 的安全执行结果，统一表达成功内容或标准化错误码。
    tool_call_id: str
    tool_name: str
    ok: bool
    content: Any
    error_code: str | None = None

    def to_model_payload(self) -> dict[str, Any]:
        # 工具结果是供模型参考的不可信数据，不能成为新的控制指令。
        return {
            "source": "tool",
            "tool_name": self.tool_name,
            "untrusted": True,
            "ok": self.ok,
            "error_code": self.error_code,
            "content": project_for_model(self.content),
        }


class PolicyDenied(Exception):
    # 策略拒绝异常：中断不满足白名单、参数、权限、预检或审批条件的调用。
    def __init__(self, code: str, message: str, content: Any | None = None):
        super().__init__(message)
        self.code = code
        self.content = content if content is not None else {"message": message}


class TransientToolError(Exception):
    # 瞬态依赖故障标记：驱动符合幂等边界的退避重试。
    pass


def _stable_value(value: StrictArgs | Mapping[str, Any] | Any) -> Any:
    # 递归规范化参数，消除映射顺序和 Decimal 表示差异以生成稳定摘要。
    if isinstance(value, BaseModel):
        return _stable_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _stable_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    return value


def _approval_digest(tool_name: str, arguments: StrictArgs | Mapping[str, Any]) -> str:
    # 将工具名和规范化参数绑定；审批不能复用于不同金额、订单或原因。
    canonical = json.dumps(
        _stable_value(arguments),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()


@dataclass
class Approval:
    # 一次性审批凭证实体：绑定身份、租户、目标工具、参数摘要和有效期。
    approval_id: str
    user_id: str
    tenant_id: str
    tool_name: str
    digest: str
    expires_at: float
    used: bool = False


APPROVAL_REQUIRED_MESSAGE = "请确认本次操作的参数"


class ApprovalStore:
    # 审批凭证存储与核销组件，确保高风险操作使用匹配且未使用的授权。
    def __init__(self) -> None:
        # 以审批编号索引一次性凭证。
        self._items: dict[str, Approval] = {}

    def approve(
        self,
        approval_id: str,
        *,
        user_id: str,
        tenant_id: str,
        tool_name: str,
        args: StrictArgs,
        ttl_seconds: int = 300,
    ) -> None:
        # 签发与调用参数绑定、带有效期的一次性审批凭证。
        self._items[approval_id] = Approval(
            approval_id=approval_id,
            user_id=user_id,
            tenant_id=tenant_id,
            tool_name=tool_name,
            digest=_approval_digest(tool_name, args),
            expires_at=time.time() + ttl_seconds,
        )

    def consume(self, approval_id: str | None, prepared: PreparedCall, ctx: ExecutionContext) -> None:
        # 审批必须未使用、未过期，且与当前用户、租户、工具和参数完全一致。
        approval = self._items.get(approval_id or "")
        valid = (
            approval is not None
            and not approval.used
            and approval.expires_at >= time.time()
            and approval.user_id == ctx.user_id
            and approval.tenant_id == ctx.tenant_id
            and approval.tool_name == prepared.tool.name
            and approval.digest == _approval_digest(prepared.tool.name, prepared.args)
        )
        if not valid:
            # 基线此处硬编码"请确认本次退款的订单、金额和原因"；本作业迁移为通用文案（§5.1 唯一框架改动）。
            raise PolicyDenied("APPROVAL_REQUIRED", APPROVAL_REQUIRED_MESSAGE)
        approval.used = True


class PermissionEngine:
    # 调用前授权决策组件：按白名单、业务权限和审批要求判定调用状态。
    def decide(self, tool: ToolDefinition, ctx: ExecutionContext) -> PermissionDecision:
        # 白名单和 RBAC 是确认或 bypass 都不可跳过的硬边界。
        if tool.name not in ctx.allowed_tools:
            return PermissionDecision.DENY
        if tool.policy.permission not in ctx.permissions:
            return PermissionDecision.DENY
        return PermissionDecision.ALLOW


@dataclass
class AuditSink:
    # 审计输出端口：记录调用链路、主体、风险、结果与耗时等可观测信息。
    records: list[dict[str, Any]] = field(default_factory=list)

    async def write(self, record: Mapping[str, Any]) -> None:
        # 持久化单条已脱敏的工具调用审计记录。
        self.records.append(dict(record))


# 识别令牌、密钥、密码和授权信息等敏感字段的匹配规则。
SENSITIVE_KEY = re.compile(
    r"token|secret|password|authorization|api_?key|credential|cookie|session|private_?key|access_?key",
    re.I,
)
MODEL_MAX_DEPTH = 8
MODEL_MAX_ITEMS = 50
MODEL_MAX_STRING_LENGTH = 2_000


def argument_keys(arguments: Any) -> list[str]:
    # 审计只保留参数键名；畸形请求也不能因记录审计再次触发内部异常。
    return sorted(str(key) for key in arguments) if isinstance(arguments, Mapping) else []


def redact(value: Any) -> Any:
    # 审计和返回模型前均做递归脱敏，避免令牌、密码和邮箱泄露。
    if isinstance(value, Mapping):
        return {
            key: "***" if SENSITIVE_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        return re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "***@***", value)
    return value


def project_for_model(value: Any, *, depth: int = 0) -> Any:
    # 对不可信工具结果脱敏并限制规模，防止其挤占模型上下文或伪装为控制信息。
    if depth >= MODEL_MAX_DEPTH:
        return "[内容嵌套过深，已截断]"
    if isinstance(value, Mapping):
        items = list(value.items())
        projected = {
            str(key): project_for_model(item, depth=depth + 1)
            for key, item in items[:MODEL_MAX_ITEMS]
        }
        if len(items) > MODEL_MAX_ITEMS:
            projected["_truncated"] = "[字段过多，已截断]"
        return redact(projected)
    if isinstance(value, (list, tuple)):
        projected = [project_for_model(item, depth=depth + 1) for item in value[:MODEL_MAX_ITEMS]]
        if len(value) > MODEL_MAX_ITEMS:
            projected.append("[项目过多，已截断]")
        return redact(projected)
    if isinstance(value, str):
        text = value[:MODEL_MAX_STRING_LENGTH]
        if len(value) > MODEL_MAX_STRING_LENGTH:
            text += "[文本过长，已截断]"
        return redact(text)
    return redact(value)


class ToolRuntime:
    # 工具治理运行时：编排发现、前置门禁、受控执行、结果保护与审计闭环。

    def __init__(
        self,
        tools: Sequence[ToolDefinition],
        permissions: PermissionEngine,
        approvals: ApprovalStore,
        audit: AuditSink,
    ) -> None:
        # 装配工具注册表、授权、审批、审计依赖及按资源维度隔离的写锁。
        names = [tool.name for tool in tools]
        if not all(isinstance(name, str) and name for name in names) or len(names) != len(set(names)):
            raise ValueError("工具名称必须非空且唯一")
        for tool in tools:
            if not issubclass(tool.parameters_model, StrictArgs):
                raise ValueError(f"工具 {tool.name} 的参数模型必须继承 StrictArgs")
            if tool.policy.timeout_seconds <= 0 or tool.policy.max_retries < 0:
                raise ValueError(f"工具 {tool.name} 的超时或重试配置无效")
        self._tools = {tool.name: tool for tool in tools}
        self.permissions = permissions
        self.approvals = approvals
        self.audit = audit
        self._locks: dict[str, asyncio.Lock] = {}

    def model_tools(self, ctx: ExecutionContext) -> list[dict[str, Any]]:
        # 只向模型暴露当前上下文下可调用的工具，避免能力泄露。
        return [
            tool.to_model_tool()
            for tool in self._tools.values()
            if self.permissions.decide(tool, ctx) is not PermissionDecision.DENY
        ]

    async def before_tool_call(self, call: ToolCall, ctx: ExecutionContext) -> PreparedCall:
        # 门禁顺序：参数 -> 硬拒绝 -> plan -> 白名单/RBAC -> 预检 -> 审批 -> bypass -> allow。
        if not isinstance(call.id, str) or not call.id or not isinstance(call.name, str) or not call.name:
            raise PolicyDenied("INVALID_ARGUMENT", "工具调用标识和名称必须为非空字符串")
        if not isinstance(call.arguments, Mapping):
            raise PolicyDenied("INVALID_ARGUMENT", "工具参数必须为对象")
        if not isinstance(ctx.mode, PermissionMode):
            raise PolicyDenied("INVALID_CONTEXT", "执行模式无效")
        tool = self._tools.get(call.name)
        if tool is None:
            raise PolicyDenied("TOOL_NOT_ALLOWED", f"工具 {call.name} 不在本轮白名单")

        try:
            args = tool.parameters_model.model_validate(call.arguments)
        except ValidationError as exc:
            errors = [
                {"path": ".".join(str(item) for item in item["loc"]), "message": item["msg"]}
                for item in exc.errors()
            ]
            raise PolicyDenied("INVALID_ARGUMENT", "参数校验失败", errors) from exc

        if tool.policy.denied or not tool.policy.enabled:
            raise PolicyDenied("DENY_RULE", "命中 deny 规则")
        if ctx.mode is PermissionMode.PLAN and tool.policy.effect in {Effect.WRITE, Effect.SHELL}:
            raise PolicyDenied("PLAN_MODE_DENIED", "plan 模式禁止写操作和 Shell")

        decision = self.permissions.decide(tool, ctx)
        if decision is PermissionDecision.DENY:
            if tool.name not in ctx.allowed_tools:
                raise PolicyDenied("TOOL_NOT_ALLOWED", f"工具 {call.name} 不在本轮白名单")
            raise PolicyDenied("PERMISSION_DENIED", f"缺少权限 {tool.policy.permission}")

        prepared = PreparedCall(call=call, tool=tool, args=args, decision=decision)
        if tool.precheck:
            await tool.precheck(args, ctx)
        if tool.policy.requires_approval:
            if ctx.mode is PermissionMode.DONT_ASK:
                raise PolicyDenied("APPROVAL_REQUIRED", "当前模式无法请求强制审批")
            self.approvals.consume(ctx.approval_id, prepared, ctx)
        if tool.policy.requires_confirmation:
            if ctx.mode is PermissionMode.DONT_ASK:
                raise PolicyDenied("CONFIRMATION_REQUIRED", "当前模式无法请求用户确认")
            if ctx.mode is not PermissionMode.BYPASS_PERMISSIONS:
                raise PolicyDenied("CONFIRMATION_REQUIRED", "该操作需要用户确认")
        return prepared

    async def _execute_once(self, prepared: PreparedCall, ctx: ExecutionContext) -> dict[str, Any]:
        # 在工具策略规定的时限内执行一次实际处理器调用。
        async with asyncio.timeout(prepared.tool.policy.timeout_seconds):
            return await prepared.tool.handler(prepared.args, ctx)

    async def _execute_with_recovery(self, prepared: PreparedCall, ctx: ExecutionContext) -> dict[str, Any]:
        # 只重试读操作或显式幂等的写操作，避免重复产生不可逆副作用。
        policy = prepared.tool.policy
        retries = policy.max_retries if policy.effect is Effect.READ or policy.idempotent else 0
        for attempt in range(retries + 1):
            try:
                return await self._execute_once(prepared, ctx)
            except TransientToolError:
                if attempt == retries:
                    raise
                await asyncio.sleep(0.05 * (2**attempt))
        raise AssertionError("unreachable")

    async def _execute(self, prepared: PreparedCall, ctx: ExecutionContext) -> dict[str, Any]:
        # 按读写语义调度：读操作直接执行，写操作按业务资源串行化。
        policy = prepared.tool.policy
        if policy.effect is Effect.READ:
            return await self._execute_with_recovery(prepared, ctx)

        key = prepared.tool.canonical_target(prepared.args, ctx)
        # 同一业务目标的写操作串行执行，防止并发请求造成重复或竞争。
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            return await self._execute_with_recovery(prepared, ctx)

    async def after_tool_call(
        self,
        prepared: PreparedCall,
        ctx: ExecutionContext,
        *,
        ok: bool,
        content: Mapping[str, Any],
        error_code: str | None,
        started_at: float,
    ) -> ToolResult:
        # 对执行结果脱敏、写入结构化审计，并封装为安全的 Harness 返回结果。
        safe_content = redact(content)
        await self.audit.write(
            {
                "trace_id": ctx.trace_id,
                "tool_call_id": prepared.call.id,
                "tool_name": prepared.tool.name,
                "user_id": ctx.user_id,
                "tenant_id": ctx.tenant_id,
                "decision": prepared.decision,
                "effect": prepared.tool.policy.effect,
                "risk": prepared.tool.policy.risk,
                "ok": ok,
                "error_code": error_code,
                "duration_ms": round((time.monotonic() - started_at) * 1000, 2),
                "argument_keys": argument_keys(prepared.call.arguments),
            }
        )
        return ToolResult(
            tool_call_id=prepared.call.id,
            tool_name=prepared.tool.name,
            ok=ok,
            content=safe_content,
            error_code=error_code,
        )

    async def invoke(self, call: ToolCall, ctx: ExecutionContext) -> ToolResult:
        # 单次调用总编排：前置治理 -> 受控执行 -> 异常归一化 -> 结果保护与审计。
        started_at = time.monotonic()
        try:
            prepared = await self.before_tool_call(call, ctx)
        except PolicyDenied as exc:
            await self.audit.write(
                {
                    "trace_id": ctx.trace_id,
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                    "user_id": ctx.user_id,
                    "tenant_id": ctx.tenant_id,
                    "decision": PermissionDecision.DENY,
                    "ok": False,
                    "error_code": exc.code,
                    "duration_ms": round((time.monotonic() - started_at) * 1000, 2),
                    "argument_keys": argument_keys(call.arguments),
                }
            )
            return ToolResult(call.id, call.name, False, redact(exc.content), exc.code)

        await self.audit.write(
            {
                "trace_id": ctx.trace_id,
                "tool_call_id": call.id,
                "tool_name": call.name,
                "user_id": ctx.user_id,
                "tenant_id": ctx.tenant_id,
                "decision": prepared.decision,
                "stage": "authorized",
                "duration_ms": round((time.monotonic() - started_at) * 1000, 2),
                "argument_keys": argument_keys(call.arguments),
            }
        )

        try:
            content = await self._execute(prepared, ctx)
            return await self.after_tool_call(
                prepared,
                ctx,
                ok=True,
                content=content,
                error_code=None,
                started_at=started_at,
            )
        except PolicyDenied as exc:
            code, message = exc.code, str(exc)
        except TimeoutError:
            code = "TIMEOUT_UNKNOWN" if prepared.tool.policy.effect is Effect.WRITE else "TIMEOUT"
            message = "写操作结果未知，请按业务 ID 查询或转人工" if code == "TIMEOUT_UNKNOWN" else "查询超时，可稍后重试"
        except TransientToolError:
            code, message = "TEMPORARY_UNAVAILABLE", "依赖暂时不可用，请稍后重试"
        except Exception:
            code, message = "TOOL_ERROR", "工具执行失败，请联系人工处理"

        return await self.after_tool_call(
            prepared,
            ctx,
            ok=False,
            content={"message": message},
            error_code=code,
            started_at=started_at,
        )

    async def invoke_batch(self, calls: Sequence[ToolCall], ctx: ExecutionContext) -> list[ToolResult]:
        # 批量调用调度器：纯读并发执行；含写操作时保持模型给出的因果顺序。
        # 纯读批次可并发；混入写操作时整批串行，保留模型给出的因果顺序。
        has_write = any(
            (tool := self._tools.get(call.name)) is not None and tool.policy.effect is Effect.WRITE
            for call in calls
        )
        if has_write:
            return [await self.invoke(call, ctx) for call in calls]
        return list(await asyncio.gather(*(self.invoke(call, ctx) for call in calls)))
