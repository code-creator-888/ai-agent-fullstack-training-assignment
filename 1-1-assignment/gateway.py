"""LLM Gateway 核心：主备切换、重试、结构化校验、调用审计、限流。

这是整个网关的中枢，协调 provider、config、models 各层。
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from collections import deque
from typing import AsyncIterator

import jsonschema

from config import (
    FALLBACK_MODEL,
    MODEL_CONFIGS,
    calculate_cost,
    render_prompt,
)
from models import (
    CallTrace,
    GatewayError,
    LLMRequest,
    LLMResponse,
    Message,
    ProviderResponse,
    Role,
    Usage,
    new_request_id,
)
from provider import (
    NON_RETRYABLE_PROVIDER_EXCEPTIONS,
    RETRYABLE_EXCEPTIONS,
    create_provider,
)


# ──────────────────────────────────────────────
# 审计存储（进程内，生产环境应替换为数据库）
# ──────────────────────────────────────────────

_traces: list[CallTrace] = []


def get_traces() -> list[CallTrace]:
    """获取所有审计记录。"""
    return list(_traces)


def _record_trace(trace: CallTrace) -> None:
    """记录一条审计。"""
    _traces.append(trace)


# ──────────────────────────────────────────────
# 按模型限流（线程安全滑动窗口）
# ──────────────────────────────────────────────

class _ModelRateLimiter:
    """线程安全的按模型滑动窗口限流器。

    使用 threading.Lock 保护懒初始化（跨 event loop 安全），
    asyncio.Lock 保护每模型的窗口操作。
    """

    def __init__(self) -> None:
        self._init_lock = threading.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._windows: dict[str, deque[float]] = {}

    async def acquire(self, model: str, rpm: int) -> None:
        """尝试获取一次请求许可。

        Args:
            model: 平台模型名
            rpm: 每分钟请求上限（0 表示不限制）

        Raises:
            GatewayError: 超过速率限制时抛出 rate_limited
        """
        if rpm <= 0:
            return
        # threading.Lock 保护懒初始化，跨 event loop 安全
        if model not in self._locks:
            with self._init_lock:
                if model not in self._locks:
                    self._locks[model] = asyncio.Lock()
                    self._windows[model] = deque()
        async with self._locks[model]:
            now = time.monotonic()
            cutoff = now - 60.0
            window = self._windows[model]
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= rpm:
                raise GatewayError(
                    "rate_limited",
                    f"模型 '{model}' 已达速率限制 ({rpm} req/min)",
                )
            window.append(now)

    def would_accept(self, model: str, rpm: int) -> bool:
        """只读检查模型是否可接受请求（不消费配额）。

        与 acquire() 使用相同的滑动窗口逻辑，但不修改窗口内容。
        供 app 层 pre-flight 预检使用。

        Returns:
            True 表示未超限（或 rpm<=0 不限制），False 表示已超限
        """
        if rpm <= 0:
            return True
        now = time.monotonic()
        cutoff = now - 60.0
        window = self._windows.get(model, deque())
        active = sum(1 for t in window if t >= cutoff)
        return active < rpm


_rate_limiter = _ModelRateLimiter()  # 模块级单例


# ──────────────────────────────────────────────
# 模型链校验（供流式端点在返回 StreamingResponse 前调用）
# ──────────────────────────────────────────────

def validate_model_chain(requested_model: str) -> None:
    """校验请求模型及 fallback 链中的所有模型是否在白名单中。

    在 stream endpoint 层调用，确保错误在返回 StreamingResponse 之前抛出，
    从而被 @app.exception_handler(GatewayError) 正确捕获为 422 JSON。
    """
    chain = _build_fallback_chain(requested_model)
    for model in chain:
        if model not in MODEL_CONFIGS:
            raise GatewayError("model_not_found", f"模型 '{model}' 不在白名单中")


def validate_rate_limit(requested_model: str) -> None:
    """在返回 StreamingResponse 之前预检限流。

    与 validate_model_chain 同属 app 层 pre-flight 检查，
    确保 rate_limited 能被 exception_handler 捕获为 429 JSON
    （而非 SSE 流内嵌 error 事件）。
    检查 fallback 链中至少有一个模型未超限。
    使用 would_accept() 公共方法，不直接访问限流器内部字段。
    """
    chain = _build_fallback_chain(requested_model)
    last_error: GatewayError | None = None
    for model in chain:
        if model not in MODEL_CONFIGS:
            continue
        config = MODEL_CONFIGS[model]
        if _rate_limiter.would_accept(model, config.rate_limit_rpm):
            return  # 至少有一个模型未超限
        # 记录此模型的限流错误（用于最终抛出）
        if config.rate_limit_rpm > 0:
            last_error = GatewayError(
                "rate_limited",
                f"模型 '{model}' 已达速率限制 ({config.rate_limit_rpm} req/min)",
            )
    if last_error:
        raise last_error


# ──────────────────────────────────────────────
# Prompt 模板渲染
# ──────────────────────────────────────────────

def _apply_prompt(request: LLMRequest) -> list[Message]:
    """如果请求指定了 Prompt 模板，渲染并注入 system 消息。

    合并策略：模板渲染的 system 消息放在最前面，
    用户原有的 system 消息追加在后面（而非丢弃）。
    """
    if request.prompt is None:
        return request.messages

    system_content = render_prompt(
        name=request.prompt.name,
        version=request.prompt.version,
        variables=request.prompt.variables,
    )

    template_msg = Message(role=Role.SYSTEM, content=system_content)
    existing_system = [m for m in request.messages if m.role == Role.SYSTEM]
    non_system = [m for m in request.messages if m.role != Role.SYSTEM]
    return [template_msg] + existing_system + non_system


# ──────────────────────────────────────────────
# 结构化输出校验
# ──────────────────────────────────────────────

def _validate_structured_output(content: str, schema: dict) -> dict:
    """校验模型输出是否符合 JSON Schema。

    Returns:
        解析后的 dict

    Raises:
        GatewayError: JSON 解析失败或 Schema 校验失败
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        raise GatewayError(
            "structured_output_invalid",
            f"模型输出不是合法的 JSON: {e}",
        ) from e

    try:
        jsonschema.validate(instance=parsed, schema=schema)
    except jsonschema.ValidationError as e:
        raise GatewayError(
            "structured_output_invalid",
            f"模型输出不符合 Schema: {e.message}",
        ) from e

    return parsed


# ──────────────────────────────────────────────
# 主备切换 + 重试
# ──────────────────────────────────────────────


def _build_fallback_chain(requested_model: str) -> list[str]:
    """构建 fallback 链：[请求模型, 备用模型]，去重。"""
    chain = [requested_model]
    if requested_model != FALLBACK_MODEL and FALLBACK_MODEL in MODEL_CONFIGS:
        chain.append(FALLBACK_MODEL)
    return chain


MAX_RETRIES_PER_MODEL = 2  # 每个模型最多尝试 2 次（1 次初始 + 1 次重试）
MAX_BACKOFF_SECONDS = 5.0  # 重试退避上限，防止 attempt 增大后延迟爆炸


def _retry_delay(attempt: int) -> float:
    """指数退避 + jitter + 上限保护。"""
    base = min(0.1 * (2 ** attempt), MAX_BACKOFF_SECONDS)
    jitter = random.uniform(0, base * 0.5)
    return base + jitter


async def call_with_fallback(request: LLMRequest) -> LLMResponse:
    """非流式调用，支持主备自动切换与重试。

    流程：
    1. 按 fallback 链依次尝试每个模型
    2. 限流检查在 try 块内，超限视同可重试异常（切换到下一个模型）
    3. 每个模型最多重试 1 次（指数退避）
    4. 可重试异常等待后重试，不可重试异常直接抛出
    5. 全部失败抛出 model_unavailable

    全局时间预算：整个 fallback 链共享 timeout_seconds，
    单次 provider 调用传 min(timeout_seconds, deadline - now) 真正封顶。
    """
    request_id = new_request_id()
    start_time = time.monotonic()
    # 全局时间预算：整个 fallback 链共享 timeout_seconds
    deadline = start_time + request.timeout_seconds
    messages = _apply_prompt(request)
    model_chain = _build_fallback_chain(request.model)
    total_attempts = 0
    last_error: Exception | None = None

    prompt_name = request.prompt.name if request.prompt else None
    prompt_version = request.prompt.version if request.prompt else None

    for platform_model in model_chain:
        if platform_model not in MODEL_CONFIGS:
            raise GatewayError("model_not_found", f"模型 '{platform_model}' 不在白名单中")

        config = MODEL_CONFIGS[platform_model]
        provider = create_provider(config)

        for attempt in range(MAX_RETRIES_PER_MODEL):
            # 全局预算检查：剩余时间封顶单次调用
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_error = TimeoutError("全局时间预算已耗尽")
                break

            total_attempts += 1
            try:
                # 限流在 try 内：超限 → GatewayError("rate_limited")
                # → 被下方 except GatewayError 捕获 → continue 切换模型
                await _rate_limiter.acquire(platform_model, config.rate_limit_rpm)

                raw: ProviderResponse = await provider.complete(
                    messages=messages,
                    response_format=request.response_format,
                    response_schema=request.response_schema,
                    timeout_seconds=min(request.timeout_seconds, remaining),
                    max_tokens=request.max_tokens,
                )

                latency_ms = int((time.monotonic() - start_time) * 1000)
                usage = Usage(
                    input_tokens=raw.input_tokens,
                    output_tokens=raw.output_tokens,
                    total_tokens=raw.total_tokens,
                )

                parsed = None
                if request.response_schema is not None:
                    parsed = _validate_structured_output(
                        raw.content, request.response_schema
                    )

                cost = calculate_cost(
                    config.provider_model, usage.input_tokens, usage.output_tokens
                )
                trace = CallTrace(
                    request_id=request_id,
                    requested_model=request.model,
                    actual_model=config.provider_model,
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    attempts=total_attempts,
                    status="success",
                )
                _record_trace(trace)

                return LLMResponse(
                    request_id=request_id,
                    model=platform_model,
                    content=raw.content,
                    parsed=parsed,
                    usage=usage,
                    latency_ms=latency_ms,
                    attempts=total_attempts,
                )

            except RETRYABLE_EXCEPTIONS as e:
                last_error = e
                if attempt < MAX_RETRIES_PER_MODEL - 1:
                    await asyncio.sleep(_retry_delay(attempt))
                continue  # 尝试重试或切换模型

            except NON_RETRYABLE_PROVIDER_EXCEPTIONS as e:
                raise GatewayError("provider_auth_error", str(e)) from e

            except GatewayError as e:
                # rate_limited 可切换到下一个模型；其他错误（如
                # structured_output_invalid / gateway_misconfigured）
                # 不应 fallback — 同一模型重试不会改变结果
                if e.error_code in ("rate_limited",):
                    last_error = e
                    continue
                raise

            except Exception as e:
                raise GatewayError("unknown_error", str(e)) from e

    # 全部失败
    latency_ms = int((time.monotonic() - start_time) * 1000)
    trace = CallTrace(
        request_id=request_id,
        requested_model=request.model,
        actual_model="",
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
        attempts=total_attempts,
        status="error",
        error_code="model_unavailable",
    )
    _record_trace(trace)
    raise GatewayError(
        "model_unavailable",
        f"所有模型均不可用，最后错误: {last_error}",
    )


# ──────────────────────────────────────────────
# 流式代理（含审计）
# ──────────────────────────────────────────────

async def stream_with_fallback(
    request: LLMRequest,
) -> AsyncIterator[dict]:
    """流式调用，支持主备切换（流开始后不切换）+ 指数退避重试 + 全局时间预算 + 审计。

    yield 的 dict 格式：
    - {"type": "content.delta", "delta": "..."}   文本块
    - {"type": "response.completed", "model": "...", "usage": {...}} 完成事件
    - {"type": "error", "error_code": "...", "message": "..."} 错误事件

    流首内容前的临时失败（RETRYABLE_EXCEPTIONS）纳入指数退避重试，
    与 call_with_fallback 保持一致的重试策略。全局时间预算由
    request.timeout_seconds 控制，跨 fallback 链共享。

    限流预检在 app 层 validate_rate_limit() 完成（429 JSON），
    此处仅做兜底 acquire（确保计数准确）。
    """
    request_id = new_request_id()
    start_time = time.monotonic()
    # 全局时间预算：整个 fallback 链共享 timeout_seconds
    deadline = start_time + request.timeout_seconds
    messages = _apply_prompt(request)
    model_chain = _build_fallback_chain(request.model)
    total_attempts = 0
    last_error: Exception | None = None

    prompt_name = request.prompt.name if request.prompt else None
    prompt_version = request.prompt.version if request.prompt else None

    for platform_model in model_chain:
        if platform_model not in MODEL_CONFIGS:
            yield {
                "type": "error",
                "error_code": "model_not_found",
                "message": f"模型 '{platform_model}' 不在白名单中",
                "request_id": request_id,
            }
            return

        config = MODEL_CONFIGS[platform_model]
        provider = create_provider(config)
        emitted = False
        stream_usage: Usage | None = None
        ttft_ms: int | None = None

        # 限流计数（app 层已预检，此处仅确保计数）
        try:
            await _rate_limiter.acquire(platform_model, config.rate_limit_rpm)
        except GatewayError:
            # 预检已通过但并发导致超限 → 尝试下一个模型
            continue

        # 每个模型的指数退避重试（仅流首内容前有效）
        for attempt in range(MAX_RETRIES_PER_MODEL):
            # 全局预算检查：剩余时间封顶单次调用
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_error = TimeoutError("全局时间预算已耗尽")
                break  # 切换到下一个模型

            total_attempts += 1

            try:
                async for pr in provider.stream(
                    messages=messages,
                    response_format=request.response_format,
                    response_schema=request.response_schema,
                    timeout_seconds=min(request.timeout_seconds, remaining),
                    max_tokens=request.max_tokens,
                ):
                    if pr.text_delta:
                        if ttft_ms is None:
                            ttft_ms = int((time.monotonic() - start_time) * 1000)
                        emitted = True
                        yield {
                            "type": "content.delta",
                            "delta": pr.text_delta,
                            "request_id": request_id,
                        }
                    if pr.is_usage_chunk:
                        stream_usage = Usage(
                            input_tokens=pr.input_tokens,
                            output_tokens=pr.output_tokens,
                            total_tokens=pr.total_tokens,
                        )

                # 流正常结束
                latency_ms = int((time.monotonic() - start_time) * 1000)
                usage = stream_usage or Usage()
                cost = calculate_cost(
                    config.provider_model, usage.input_tokens, usage.output_tokens
                )
                trace = CallTrace(
                    request_id=request_id,
                    requested_model=request.model,
                    actual_model=config.provider_model,
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    attempts=total_attempts,
                    status="success",
                )
                _record_trace(trace)

                yield {
                    "type": "response.completed",
                    "model": platform_model,
                    "request_id": request_id,
                    "usage": usage.model_dump(),
                    "ttft_ms": ttft_ms,
                }
                return

            except RETRYABLE_EXCEPTIONS as e:
                last_error = e
                if emitted:
                    # 流已开始，不能切换模型（避免文本断裂）
                    latency_ms = int((time.monotonic() - start_time) * 1000)
                    # 补上已收集的 usage（流中断场景下成本统计不为 0）
                    interrupted_usage = stream_usage or Usage()
                    cost = calculate_cost(
                        config.provider_model,
                        interrupted_usage.input_tokens,
                        interrupted_usage.output_tokens,
                    )
                    trace = CallTrace(
                        request_id=request_id,
                        requested_model=request.model,
                        actual_model=config.provider_model,
                        prompt_name=prompt_name,
                        prompt_version=prompt_version,
                        input_tokens=interrupted_usage.input_tokens,
                        output_tokens=interrupted_usage.output_tokens,
                        cost_usd=cost,
                        latency_ms=latency_ms,
                        ttft_ms=ttft_ms,
                        attempts=total_attempts,
                        status="error",
                        error_code="stream_interrupted",
                    )
                    _record_trace(trace)
                    yield {
                        "type": "error",
                        "error_code": "stream_interrupted",
                        "message": str(e),
                        "request_id": request_id,
                    }
                    return
                # 流未开始：指数退避后重试（或切换到下一模型）
                if attempt < MAX_RETRIES_PER_MODEL - 1:
                    if time.monotonic() >= deadline:
                        break  # 预算耗尽，切换模型
                    await asyncio.sleep(_retry_delay(attempt))
                continue

            except NON_RETRYABLE_PROVIDER_EXCEPTIONS as e:
                yield {
                    "type": "error",
                    "error_code": "provider_auth_error",
                    "message": str(e),
                    "request_id": request_id,
                }
                return

            except GatewayError as e:
                yield {
                    "type": "error",
                    "error_code": e.error_code,
                    "message": e.message,
                    "request_id": request_id,
                }
                return

            except Exception as e:
                yield {
                    "type": "error",
                    "error_code": "unknown_error",
                    "message": str(e),
                    "request_id": request_id,
                }
                return

    # 全部模型失败
    latency_ms = int((time.monotonic() - start_time) * 1000)
    trace = CallTrace(
        request_id=request_id,
        requested_model=request.model,
        actual_model="",
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
        attempts=total_attempts,
        status="error",
        error_code="model_unavailable",
    )
    _record_trace(trace)
    yield {
        "type": "error",
        "error_code": "model_unavailable",
        "message": f"所有模型均不可用，最后错误: {last_error}",
        "request_id": request_id,
    }
