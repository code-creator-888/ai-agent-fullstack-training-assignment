"""OpenAI 兼容供应商适配器。

统一 complete() / stream() 两个接口，兼容所有 OpenAI 协议的供应商
（DeepSeek、OpenAI、通义千问等）。
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

from config import ModelConfig
from models import GatewayError, Message, ProviderResponse


# 可重试的异常
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    TimeoutError,
    ConnectionError,
)


class OpenAICompatibleProvider:
    """OpenAI 兼容协议的供应商适配器。"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI | None:
        """懒加载 AsyncOpenAI 客户端。密钥和 API 地址均从环境变量读取。"""
        if self._client is None:
            api_key = os.getenv(self.config.api_key_env, "")
            if not api_key:
                raise GatewayError(
                    "gateway_misconfigured",
                    f"环境变量 {self.config.api_key_env} 未设置或为空",
                )
            # base_url 优先从环境变量读取，未设置则回退到配置中的默认值
            base_url = (
                os.getenv(self.config.base_url_env, "")
                if self.config.base_url_env
                else ""
            ) or self.config.base_url
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        return self._client

    def _resolve_model(self) -> str:
        """解析实际使用的供应商模型名，优先从环境变量读取。"""
        if self.config.provider_model_env:
            return (
                os.getenv(self.config.provider_model_env, "")
                or self.config.provider_model
            )
        return self.config.provider_model

    async def complete(
        self,
        messages: list[Message],
        response_schema: dict | None = None,
        timeout_seconds: float = 30,
    ) -> ProviderResponse:
        """非流式调用，返回类型安全的 ProviderResponse。

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

        # 结构化输出处理
        if response_schema is not None:
            if self.config.supports_structured_output:
                if self.config.structured_output_mode == "json_schema":
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "strict": True,
                            "schema": response_schema,
                        },
                    }
                elif self.config.structured_output_mode == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}
                    schema_text = (
                        f"\n\n你必须返回符合以下 JSON Schema 的 JSON 对象：\n"
                        f"{json.dumps(response_schema, ensure_ascii=False)}"
                    )
                    self._inject_schema_to_system(api_messages, schema_text)
            else:
                # 供应商不支持结构化输出：回退为在 system 中注入 Schema 文本
                schema_text = (
                    f"\n\n你必须返回符合以下 JSON Schema 的 JSON 对象：\n"
                    f"{json.dumps(response_schema, ensure_ascii=False)}"
                )
                self._inject_schema_to_system(api_messages, schema_text)

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
        timeout_seconds: float = 30,
    ) -> AsyncIterator[ProviderResponse]:
        """流式调用，逐块 yield ProviderResponse。

        文本块通过 text_delta 传递；最后一个元素携带 usage 信息
        （text_delta 为空字符串，仅用于审计）。
        """
        client = self._get_client()
        api_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        stream = await client.chat.completions.create(
            model=self._resolve_model(),
            messages=api_messages,
            stream=True,
            stream_options={"include_usage": True},
            timeout=timeout_seconds,
        )

        async for chunk in stream:
            # 文本内容块
            if chunk.choices and chunk.choices[0].delta.content:
                yield ProviderResponse(
                    text_delta=chunk.choices[0].delta.content,
                    model=chunk.model or "",
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                )
            # 最后一个 chunk 携带 usage 信息（显式标志）
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
        # 没有 system 消息，插入一个
        messages.insert(0, {"role": "system", "content": schema_text.strip()})
