"""LLM Gateway 核心：主备切换、重试、结构化校验、调用审计。

这是整个网关的中枢，协调 provider、config、models 各层。
"""

from __future__ import annotations

import asyncio
import json
import random
import time
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
from provider import RETRYABLE_EXCEPTIONS, OpenAICompatibleProvider


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

    # 模板渲染的 system 消息放在最前面
    template_msg = Message(role=Role.SYSTEM, content=system_content)
    # 保留用户原有的 system 消息，追加在后面
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
    2. 每个模型最多重试 1 次（指数退避）
    3. 可重试异常等待后重试，不可重试异常直接抛出
    4. 全部失败抛出 model_unavailable
    """
    request_id = new_request_id()
    start_time = time.monotonic()
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
        provider = OpenAICompatibleProvider(config)

        for attempt in range(MAX_RETRIES_PER_MODEL):
            total_attempts += 1
            try:
                raw: ProviderResponse = await provider.complete(
                    messages=messages,
                    response_schema=request.response_schema,
                    timeout_seconds=request.timeout_seconds,
                )

                latency_ms = int((time.monotonic() - start_time) * 1000)
                usage = Usage(
                    input_tokens=raw.input_tokens,
                    output_tokens=raw.output_tokens,
                    total_tokens=raw.total_tokens,
                )

                # 结构化输出校验
                parsed = None
                if request.response_schema is not None:
                    parsed = _validate_structured_output(
                        raw.content, request.response_schema
                    )

                # 审计记录
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

                # 返回平台模型名，不暴露供应商名
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

            except GatewayError:
                raise  # 请求不合法，直接抛出

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
    """流式调用，支持主备切换（流开始后不切换）+ 审计。

    yield 的 dict 格式：
    - {"type": "content.delta", "delta": "..."}   文本块
    - {"type": "response.completed", "model": "...", "usage": {...}} 完成事件
    - {"type": "error", "error_code": "...", "message": "..."} 错误事件
    """
    request_id = new_request_id()
    start_time = time.monotonic()
    messages = _apply_prompt(request)
    model_chain = _build_fallback_chain(request.model)

    prompt_name = request.prompt.name if request.prompt else None
    prompt_version = request.prompt.version if request.prompt else None

    for platform_model in model_chain:
        if platform_model not in MODEL_CONFIGS:
            # 正常流程不会走到这里（app 层 validate_model_chain 已拦截）。
            # 仅当绕过 app 层直接调用 stream_with_fallback 时才会触发。
            yield {
                "type": "error",
                "error_code": "model_not_found",
                "message": f"模型 '{platform_model}' 不在白名单中",
                "request_id": request_id,
            }
            return

        config = MODEL_CONFIGS[platform_model]
        provider = OpenAICompatibleProvider(config)
        emitted = False
        stream_usage: Usage | None = None

        try:
            async for pr in provider.stream(
                messages=messages,
                timeout_seconds=request.timeout_seconds,
            ):
                # 文本块：转发给客户端
                if pr.text_delta:
                    emitted = True
                    yield {
                        "type": "content.delta",
                        "delta": pr.text_delta,
                        "request_id": request_id,
                    }
                # usage chunk：通过显式标志识别，不依赖 token 数推断
                if pr.is_usage_chunk:
                    stream_usage = Usage(
                        input_tokens=pr.input_tokens,
                        output_tokens=pr.output_tokens,
                        total_tokens=pr.total_tokens,
                    )

            # 流正常结束：记录审计
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
                attempts=1,
                status="success",
            )
            _record_trace(trace)

            yield {
                "type": "response.completed",
                "model": platform_model,
                "request_id": request_id,
                "usage": usage.model_dump(),
            }
            return

        except RETRYABLE_EXCEPTIONS as e:
            if emitted:
                # 已发送部分内容，不再切换，直接报错
                latency_ms = int((time.monotonic() - start_time) * 1000)
                trace = CallTrace(
                    request_id=request_id,
                    requested_model=request.model,
                    actual_model=config.provider_model,
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    latency_ms=latency_ms,
                    attempts=1,
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
            # 流未开始，尝试下一个模型
            continue

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
        attempts=0,
        status="error",
        error_code="model_unavailable",
    )
    _record_trace(trace)
    yield {
        "type": "error",
        "error_code": "model_unavailable",
        "message": "所有模型均不可用",
        "request_id": request_id,
    }
