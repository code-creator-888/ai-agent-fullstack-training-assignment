"""供应商适配器层：BaseProvider 抽象 + OpenAI / Anthropic 双协议。

统一 complete() / stream() 两个接口，通过 create_provider() 工厂
按 ModelConfig.protocol 自动选择适配器。
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator

import anthropic
import openai as _openai_pkg
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

from config import PROTOCOL_DEFAULT_MAX_TOKENS, ModelConfig
from models import GatewayError, Message, ProviderResponse


# ──────────────────────────────────────────────
# 异常分类（双协议对称）
# ──────────────────────────────────────────────

RETRYABLE_EXCEPTIONS = (
    # OpenAI 协议
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    _openai_pkg.InternalServerError,
    # Anthropic 协议
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    # 通用
    TimeoutError,
    ConnectionError,
)

NON_RETRYABLE_PROVIDER_EXCEPTIONS = (
    _openai_pkg.AuthenticationError,
    _openai_pkg.PermissionDeniedError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
)


# ──────────────────────────────────────────────
# 抽象基类
# ──────────────────────────────────────────────

class BaseProvider(ABC):
    """供应商适配器抽象基类。

    所有具体适配器必须实现 complete() 和 stream() 两个异步接口。
    公共的环境变量解析逻辑上提到此层，避免子类重复。
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def _resolve_model(self) -> str:
        """解析实际使用的供应商模型名，优先从环境变量读取。"""
        if self.config.provider_model_env:
            return (
                os.getenv(self.config.provider_model_env, "")
                or self.config.provider_model
            )
        return self.config.provider_model

    def _resolve_base_url(self) -> str:
        """解析 base_url，优先从环境变量读取。"""
        return (
            os.getenv(self.config.base_url_env, "")
            if self.config.base_url_env
            else ""
        ) or self.config.base_url

    def _resolve_api_key(self) -> str:
        """从环境变量读取 API 密钥，缺失时抛出 gateway_misconfigured。"""
        api_key = os.getenv(self.config.api_key_env, "")
        if not api_key:
            raise GatewayError(
                "gateway_misconfigured",
                f"环境变量 {self.config.api_key_env} 未设置或为空",
            )
        return api_key

    def _resolve_max_tokens(self, request_max_tokens: int | None = None) -> int:
        """解析 max_tokens：请求级 > 模型配置 > 协议默认。"""
        if request_max_tokens and request_max_tokens > 0:
            return request_max_tokens
        if self.config.max_tokens > 0:
            return self.config.max_tokens
        return PROTOCOL_DEFAULT_MAX_TOKENS.get(self.config.protocol, 4096)

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        response_format: dict | None = None,
        response_schema: dict | None = None,
        timeout_seconds: float = 30,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """非流式调用，返回 ProviderResponse。"""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        response_format: dict | None = None,
        response_schema: dict | None = None,
        timeout_seconds: float = 30,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderResponse]:
        """流式调用，逐块 yield ProviderResponse。

        Args:
            response_format: 原始响应格式指令（json_object / json_schema），
                             携带行为类型信息，流式路径的核心调度依据。
            response_schema: 从 response_format 提取或显式设置的 JSON Schema，
                             仅在 json_schema 模式或非流式路径下有值。
        """
        ...


# ──────────────────────────────────────────────
# OpenAI 兼容协议适配器
# ──────────────────────────────────────────────

class OpenAICompatibleProvider(BaseProvider):
    """OpenAI 兼容协议的供应商适配器。"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """懒加载 AsyncOpenAI 客户端。"""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._resolve_api_key(),
                base_url=self._resolve_base_url(),
            )
        return self._client

    async def complete(
        self,
        messages: list[Message],
        response_format: dict | None = None,
        response_schema: dict | None = None,
        timeout_seconds: float = 30,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """非流式调用，返回类型安全的 ProviderResponse。

        双参数驱动（与 stream() 一致）：
        - response_format 携带行为类型（json_object / json_schema）
        - response_schema 携带实际的 Schema 内容

        支持两种结构化输出模式：
        - json_schema: response_format.type = "json_schema" + strict=True
        - json_object: response_format.type = "json_object" + 在 system 中注入 Schema
        - 供应商不支持时：回退为在 system 中注入 Schema（不静默降级）
        """
        client = self._get_client()
        api_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        kwargs: dict = {
            "model": self._resolve_model(),
            "messages": api_messages,
            "timeout": timeout_seconds,
        }

        # max_tokens（仅在调用方指定时传递）
        resolved_max = self._resolve_max_tokens(max_tokens)
        if max_tokens and max_tokens > 0:
            kwargs["max_tokens"] = resolved_max

        # 结构化输出处理（双参数驱动，complete / stream 共用）
        self._negotiate_response_format(kwargs, api_messages, response_format, response_schema)

        response = await client.chat.completions.create(**kwargs)

        usage = response.usage
        return ProviderResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    async def stream(
        self,
        messages: list[Message],
        response_format: dict | None = None,
        response_schema: dict | None = None,
        timeout_seconds: float = 30,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderResponse]:
        """流式调用，复用 complete() 的能力协商逻辑。

        双参数驱动：
        - response_format 携带行为类型（json_object / json_schema）
        - response_schema 携带实际的 Schema 内容

        模式协商（与 complete() 一致）：
        - json_object: 传 response_format.type=json_object + 有 schema 时 system 注入
        - json_schema: 按 config.structured_output_mode 归一化
        - 不支持时: system 注入 Schema 降级
        """
        client = self._get_client()
        api_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        kwargs: dict = {
            "model": self._resolve_model(),
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "timeout": timeout_seconds,
        }

        # 结构化输出处理（双参数驱动，complete / stream 共用）
        self._negotiate_response_format(kwargs, api_messages, response_format, response_schema)

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield ProviderResponse(
                    text_delta=chunk.choices[0].delta.content,
                    model=chunk.model or "",
                )
            if chunk.usage:
                yield ProviderResponse(
                    text_delta="",
                    model=chunk.model or "",
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                    is_usage_chunk=True,
                )

    @staticmethod
    def _inject_schema_to_system(messages: list[dict], schema_text: str) -> None:
        """在 system 消息末尾注入 Schema 说明。"""
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] += schema_text
                return
        messages.insert(0, {"role": "system", "content": schema_text.strip()})

    def _negotiate_response_format(
        self,
        kwargs: dict,
        messages: list[dict],
        response_format: dict | None,
        response_schema: dict | None,
    ) -> None:
        """结构化输出能力协商（complete / stream 共用）。

        双参数驱动：
        - response_format 携带行为类型（json_object / json_schema）
        - response_schema 携带实际的 Schema 内容

        根据 config.structured_output_mode 和 config.supports_structured_output
        决定传给上游的 response_format 以及是否在 system 中注入 Schema。
        """
        fmt_type = response_format.get("type") if response_format else None

        if fmt_type == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
            if response_schema is not None:
                schema_text = (
                    f"\n\n你必须返回符合以下 JSON Schema 的 JSON 对象：\n"
                    f"{json.dumps(response_schema, ensure_ascii=False)}"
                )
                self._inject_schema_to_system(messages, schema_text)
        elif fmt_type == "json_schema" or response_schema is not None:
            if self.config.supports_structured_output:
                if self.config.structured_output_mode == "json_schema" and response_schema:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "strict": True,
                            "schema": response_schema,
                        },
                    }
                else:
                    kwargs["response_format"] = {"type": "json_object"}
                    if response_schema is not None:
                        schema_text = (
                            f"\n\n你必须返回符合以下 JSON Schema 的 JSON 对象：\n"
                            f"{json.dumps(response_schema, ensure_ascii=False)}"
                        )
                        self._inject_schema_to_system(messages, schema_text)
            else:
                if response_schema is not None:
                    schema_text = (
                        f"\n\n你必须返回符合以下 JSON Schema 的 JSON 对象：\n"
                        f"{json.dumps(response_schema, ensure_ascii=False)}"
                    )
                    self._inject_schema_to_system(messages, schema_text)


# ──────────────────────────────────────────────
# Anthropic Messages 协议适配器
# ──────────────────────────────────────────────

class AnthropicProvider(BaseProvider):
    """Anthropic Messages 协议的供应商适配器。

    与 OpenAI 协议的关键差异：
    - system 消息是顶层参数（不在 messages 列表中）
    - max_tokens 为必传参数
    - 无原生结构化输出模式，通过 system prompt 注入 Schema
    - content 响应是 ContentBlock 列表，需按 type 过滤拼接
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client: anthropic.AsyncAnthropic | None = None

    @staticmethod
    def _inject_schema_to_system_text(
        system_text: str,
        response_format: dict | None,
        response_schema: dict | None,
    ) -> str:
        """结构化输出：注入 Schema / JSON 提示到 system_text（complete / stream 共用）。

        Anthropic 无原生结构化输出模式，通过在 system prompt 中注入文本引导模型。
        双参数驱动：response_schema 有值时注入完整 Schema，仅有 response_format 时注入 JSON 提示。
        """
        if response_schema is not None:
            schema_text = (
                f"\n\n你必须返回符合以下 JSON Schema 的 JSON 对象：\n"
                f"{json.dumps(response_schema, ensure_ascii=False)}"
            )
            return (system_text + schema_text) if system_text else schema_text.strip()
        if response_format is not None and response_format.get("type") in (
            "json_schema", "json_object",
        ):
            hint = "\n\n你必须返回一个 JSON 对象。"
            return (system_text + hint) if system_text else hint.strip()
        return system_text

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """懒加载 AsyncAnthropic 客户端。"""
        if self._client is None:
            base_url = self._resolve_base_url()
            self._client = anthropic.AsyncAnthropic(
                api_key=self._resolve_api_key(),
                base_url=base_url if base_url else None,
            )
        return self._client

    @staticmethod
    def _split_system_messages(
        messages: list[Message],
    ) -> tuple[str, list[dict]]:
        """从消息列表中提取 system 消息，返回 (system_text, api_messages)。

        Anthropic 协议要求 system 作为顶层参数，不在 messages 列表中。
        多条 system 消息用换行拼接。
        """
        system_parts = [m.content for m in messages if m.role.value == "system"]
        api_messages = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role.value != "system"
        ]
        system_text = "\n".join(system_parts)
        return system_text, api_messages

    @staticmethod
    def _extract_text_from_blocks(blocks: list) -> str:
        """从 Anthropic ContentBlock 列表中按类型拼接所有 text 块。

        跳过 thinking / tool_use 等非文本块，避免 AttributeError。
        空响应（refusal）返回空字符串而非崩溃。
        """
        parts: list[str] = []
        for block in blocks:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "")
                if text:
                    parts.append(text)
        return "".join(parts)

    async def complete(
        self,
        messages: list[Message],
        response_format: dict | None = None,
        response_schema: dict | None = None,
        timeout_seconds: float = 30,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """非流式调用 Anthropic Messages API。

        Anthropic 无原生结构化输出模式，通过在 system prompt 中注入
        Schema 文本引导模型返回 JSON，之后由 gateway 层做二次校验。
        """
        client = self._get_client()
        system_text, api_messages = self._split_system_messages(messages)

        # 结构化输出：注入 Schema 到 system prompt（complete / stream 共用）
        system_text = self._inject_schema_to_system_text(
            system_text, response_format, response_schema,
        )

        resolved_max = self._resolve_max_tokens(max_tokens)
        kwargs: dict = {
            "model": self._resolve_model(),
            "messages": api_messages,
            "max_tokens": resolved_max,
            "timeout": timeout_seconds,
        }
        if system_text:
            kwargs["system"] = system_text

        response = await client.messages.create(**kwargs)

        # 按类型拼接所有 text 块（跳过 thinking/tool_use 等）
        content = self._extract_text_from_blocks(response.content)

        return ProviderResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

    async def stream(
        self,
        messages: list[Message],
        response_format: dict | None = None,
        response_schema: dict | None = None,
        timeout_seconds: float = 30,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderResponse]:
        """流式调用 Anthropic Messages API。

        使用 client.messages.stream() 高级接口：
        - event.type == "text"  → text_delta
        - 流结束后通过 get_final_message() 获取 usage
        - 结构化输出：与 complete() 一致，注入 Schema 到 system prompt
        """
        client = self._get_client()
        system_text, api_messages = self._split_system_messages(messages)

        # 结构化输出：注入 Schema 到 system prompt（complete / stream 共用）
        system_text = self._inject_schema_to_system_text(
            system_text, response_format, response_schema,
        )

        resolved_max = self._resolve_max_tokens(max_tokens)
        kwargs: dict = {
            "model": self._resolve_model(),
            "messages": api_messages,
            "max_tokens": resolved_max,
            "timeout": timeout_seconds,
        }
        if system_text:
            kwargs["system"] = system_text

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "text":
                    yield ProviderResponse(
                        text_delta=event.text,
                        model="",
                    )

            final = await stream.get_final_message()
            yield ProviderResponse(
                text_delta="",
                model=final.model,
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                total_tokens=final.usage.input_tokens + final.usage.output_tokens,
                is_usage_chunk=True,
            )


# ──────────────────────────────────────────────
# 工厂函数（带缓存，避免每请求新建 SDK 客户端）
# ──────────────────────────────────────────────

_provider_cache: dict[ModelConfig, BaseProvider] = {}


def create_provider(config: ModelConfig) -> BaseProvider:
    """根据 ModelConfig.protocol 创建对应的供应商适配器。

    使用 ModelConfig（frozen dataclass，可哈希）作为缓存键，
    复用 SDK 客户端及其连接池。
    """
    cached = _provider_cache.get(config)
    if cached is not None:
        return cached
    if config.protocol == "anthropic":
        provider = AnthropicProvider(config)
    else:
        provider = OpenAICompatibleProvider(config)
    _provider_cache[config] = provider
    return provider
