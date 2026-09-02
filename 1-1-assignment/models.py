"""Pydantic 请求/响应/审计模型。

全链路校验：入口用 Pydantic 校验请求，出口用 jsonschema 校验模型输出。
所有模型启用 extra="forbid"，拒绝未定义字段，防止注入攻击。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator


# ──────────────────────────────────────────────
# 基础模型
# ──────────────────────────────────────────────

class Role(str, Enum):
    """消息角色枚举。"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """统一的消息结构，屏蔽各供应商消息格式差异。"""
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str


class Usage(BaseModel):
    """Token 用量统计。"""
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


# ──────────────────────────────────────────────
# Prompt 模板选择
# ──────────────────────────────────────────────

class PromptSelection(BaseModel):
    """调用方选择的 Prompt 模板（只能选择已注册模板 + 传变量）。"""
    model_config = ConfigDict(extra="forbid")

    name: str               # 模板名称
    version: str            # 模板版本
    variables: dict[str, str] = {}  # 模板变量


# ──────────────────────────────────────────────
# 请求 / 响应
# ──────────────────────────────────────────────

class LLMRequest(BaseModel):
    """统一的 LLM 调用请求。"""
    model_config = ConfigDict(extra="forbid")

    model: str                              # 平台模型名（如 "general-primary"）
    messages: list[Message]                 # 消息列表
    stream: bool = False                    # 是否流式
    response_schema: Optional[dict] = None  # 结构化输出 JSON Schema
    timeout_seconds: float = 30             # 超时时间
    max_tokens: Optional[int] = None         # 最大输出 Token 数（None 使用模型配置默认值）
    prompt: Optional[PromptSelection] = None  # Prompt 模板选择

    @model_validator(mode="after")
    def _check_stream_and_schema(self) -> "LLMRequest":
        """stream 与 response_schema 不能同时使用（流式无法做 JSON 校验）。"""
        if self.stream and self.response_schema is not None:
            raise ValueError(
                "stream=True 与 response_schema 不能同时使用"
            )
        return self


class LLMResponse(BaseModel):
    """统一的 LLM 调用响应。"""
    model_config = ConfigDict(extra="forbid")

    request_id: str                 # UUID 唯一请求标识
    model: str                      # 平台模型名（如 "general-primary"，不暴露供应商名）
    content: str                    # 模型原文
    parsed: Optional[dict] = None   # 结构化输出的解析结果
    usage: Usage                    # Token 用量
    latency_ms: int                 # 请求延迟（毫秒）
    attempts: int                   # 尝试次数（含重试）
    ttft_ms: Optional[int] = None   # 流式首 Token 时间（毫秒），非流式为 None


# ──────────────────────────────────────────────
# 调用审计
# ──────────────────────────────────────────────

class CallTrace(BaseModel):
    """每次调用的审计记录。

    默认不保存模型回答和 Prompt 文本，只保留元数据，保护隐私。
    """
    model_config = ConfigDict(extra="forbid")

    request_id: str                         # UUID 唯一标识
    requested_model: str                    # 请求的平台模型名
    actual_model: str                       # 实际使用的供应商模型名（fallback 后可能不同）
    prompt_name: Optional[str] = None       # 使用的 Prompt 模板名
    prompt_version: Optional[str] = None    # Prompt 模板版本
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0                   # 按 PRICE_PER_MILLION 自动计算
    latency_ms: int = 0
    ttft_ms: Optional[int] = None           # 流式首 Token 时间（毫秒），非流式为 None
    attempts: int = 0
    status: str = "success"                 # success / error
    error_code: Optional[str] = None        # 错误码


# ──────────────────────────────────────────────
# 自定义异常
# ──────────────────────────────────────────────

class GatewayError(Exception):
    """网关业务异常基类。

    http_status 字段控制 REST 层返回的 HTTP 状态码，
    默认 422（业务异常），限流为 429，认证失败为 401。
    """

    _DEFAULT_STATUS: dict[str, int] = {
        "rate_limited": 429,
        "provider_auth_error": 401,
    }

    def __init__(self, error_code: str, message: str, *, http_status: int | None = None):
        self.error_code = error_code
        self.message = message
        self.http_status = http_status or self._DEFAULT_STATUS.get(error_code, 422)
        super().__init__(f"[{error_code}] {message}")


def new_request_id() -> str:
    """生成唯一的请求 ID。"""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class ProviderResponse:
    """供应商适配器返回的类型安全结构（替代裸 dict）。"""
    content: str = ""
    text_delta: str = ""        # 流式文本块
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    is_usage_chunk: bool = False  # 显式标志：区分 usage chunk 和普通文本 chunk
