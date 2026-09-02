"""FastAPI REST API：LLM Gateway 的 HTTP 入口。

端点：
  POST /v1/llm         — 非流式调用（含结构化输出）
  POST /v1/llm/stream   — 流式 SSE 调用
  GET  /v1/traces        — 查询调用审计记录
"""

from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # 自动读取 .env 文件到 os.environ

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway import (
    call_with_fallback,
    get_traces,
    stream_with_fallback,
    validate_model_chain,
    validate_rate_limit,
)
from models import GatewayError, LLMRequest, LLMResponse

app = FastAPI(title="LLM Gateway", version="1.0.0")


# ──────────────────────────────────────────────
# 异常处理
# ──────────────────────────────────────────────

@app.exception_handler(GatewayError)
async def gateway_error_handler(_request: Request, exc: GatewayError) -> JSONResponse:
    """统一的 Gateway 业务异常响应。

    HTTP 状态码由 GatewayError.http_status 控制：
    - rate_limited → 429
    - provider_auth_error → 401
    - 其他业务异常 → 422
    """
    return JSONResponse(
        status_code=exc.http_status,
        content={"error_code": exc.error_code, "message": exc.message},
    )


# ──────────────────────────────────────────────
# REST 端点
# ──────────────────────────────────────────────

@app.post("/v1/llm", response_model=LLMResponse)
async def chat(request: LLMRequest) -> LLMResponse:
    """非流式调用（含结构化输出）。"""
    return await call_with_fallback(request)


@app.post("/v1/llm/stream")
async def chat_stream(request: LLMRequest) -> StreamingResponse:
    """流式 SSE 调用。

    在返回 StreamingResponse 之前执行 pre-flight 检查：
    1. validate_model_chain() — 白名单校验 → 422 JSON
    2. validate_rate_limit() — 限流预检 → 429 JSON
    """
    validate_model_chain(request.model)
    validate_rate_limit(request.model)

    async def event_generator():
        async for event in stream_with_fallback(request):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.get("/v1/traces")
async def traces() -> list[dict[str, Any]]:
    """查询调用审计记录。"""
    return [t.model_dump() for t in get_traces()]
