"""Mock 测试：使用 FakeProvider 模拟供应商行为，不依赖真实 API。

测试场景：
- 正常调用：返回正确内容 + 审计记录
- 结构化输出：JSON 解析 + Schema 校验
- 主模型连续超时：自动 fallback 到备用模型
- 流式输出：SSE 事件格式正确 + 审计记录
- 非法请求：422 + 错误码
- 模型白名单校验（含流式）
- LLMResponse.model 返回平台名
- _apply_prompt 合并 system 消息
- validate_model_chain 校验
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 确保 1-1-assignment 目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIGS, ModelConfig, render_prompt, calculate_cost
from models import (
    CallTrace,
    GatewayError,
    LLMRequest,
    LLMResponse,
    Message,
    PromptSelection,
    ProviderResponse,
    Role,
    Usage,
)
from provider import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    create_provider,
)


# ──────────────────────────────────────────────
# 辅助工具
# ──────────────────────────────────────────────

def make_request(
    model: str = "general-primary",
    content: str = "你好",
    stream: bool = False,
    response_schema: dict | None = None,
    prompt: PromptSelection | None = None,
    messages: list[Message] | None = None,
) -> LLMRequest:
    """快速构造一个测试请求。"""
    if messages is None:
        messages = [Message(role=Role.USER, content=content)]
    return LLMRequest(
        model=model,
        messages=messages,
        stream=stream,
        response_schema=response_schema,
        prompt=prompt,
    )


def make_fake_response(
    content: str = "你好！我是AI助手。",
    model: str = "deepseek-chat",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> ProviderResponse:
    """构造模拟的供应商响应（ProviderResponse）。"""
    return ProviderResponse(
        content=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


# ──────────────────────────────────────────────
# 测试：模型校验
# ──────────────────────────────────────────────

class TestModelValidation:
    """Pydantic 请求校验测试。"""

    def test_extra_field_forbidden(self):
        """未知字段应被拒绝。"""
        with pytest.raises(Exception):
            LLMRequest(
                model="general-primary",
                messages=[Message(role=Role.USER, content="hi")],
                unknown_field="hack",
            )

    def test_stream_and_schema_conflict(self):
        """stream=True 与 response_schema 不能同时使用。"""
        with pytest.raises(Exception):
            LLMRequest(
                model="general-primary",
                messages=[Message(role=Role.USER, content="hi")],
                stream=True,
                response_schema={"type": "object"},
            )

    def test_valid_request(self):
        """合法请求应通过校验。"""
        req = make_request()
        assert req.model == "general-primary"
        assert req.stream is False

    def test_invalid_role(self):
        """无效角色应被拒绝。"""
        with pytest.raises(Exception):
            Message(role="invalid_role", content="hi")


# ──────────────────────────────────────────────
# 测试：Prompt 模板
# ──────────────────────────────────────────────

class TestPromptTemplate:
    """Prompt 模板渲染测试。"""

    def test_render_success(self):
        """正常渲染。"""
        result = render_prompt("knowledge_decision", "v1", {"product_name": "智能客服"})
        assert "智能客服" in result

    def test_missing_variable(self):
        """变量缺失应抛错。"""
        with pytest.raises(KeyError, match="missing_prompt_variable"):
            render_prompt("knowledge_decision", "v1", {})

    def test_template_not_found(self):
        """模板不存在应抛错。"""
        with pytest.raises(KeyError, match="prompt_not_found"):
            render_prompt("nonexistent", "v1", {})


# ──────────────────────────────────────────────
# 测试：成本计算
# ──────────────────────────────────────────────

class TestCostCalculation:
    """成本计算测试。"""

    def test_known_model(self):
        """已知模型的成本计算。"""
        cost = calculate_cost("deepseek-chat", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.27 + 1.10, abs=0.01)

    def test_unknown_model(self):
        """未知模型成本为 0。"""
        cost = calculate_cost("unknown-model", 1_000_000, 1_000_000)
        assert cost == 0.0


# ──────────────────────────────────────────────
# 测试：模型白名单
# ──────────────────────────────────────────────

class TestModelWhitelist:
    """模型白名单测试。"""

    def test_valid_model(self):
        """白名单内的模型应该存在。"""
        assert "general-primary" in MODEL_CONFIGS
        assert "general-backup" in MODEL_CONFIGS

    def test_invalid_model(self):
        """白名单外的模型不应存在。"""
        assert "gpt-4-turbo" not in MODEL_CONFIGS

    def test_validate_model_chain_rejects_invalid(self):
        """validate_model_chain 应拒绝白名单外的模型。"""
        from gateway import validate_model_chain

        with pytest.raises(GatewayError, match="model_not_found"):
            validate_model_chain("nonexistent-model")

    def test_validate_model_chain_accepts_valid(self):
        """validate_model_chain 应接受白名单内的模型。"""
        from gateway import validate_model_chain

        validate_model_chain("general-primary")  # 不应抛出


# ──────────────────────────────────────────────
# 测试：_apply_prompt 合并 system 消息
# ──────────────────────────────────────────────

class TestApplyPrompt:
    """_apply_prompt system 消息合并测试。"""

    def test_no_prompt_returns_unchanged(self):
        """无 prompt 模板时消息不变。"""
        from gateway import _apply_prompt

        msgs = [
            Message(role=Role.SYSTEM, content="原始 system"),
            Message(role=Role.USER, content="你好"),
        ]
        req = make_request(messages=msgs)
        result = _apply_prompt(req)
        assert len(result) == 2
        assert result[0].content == "原始 system"

    def test_prompt_merges_system(self):
        """模板 system + 原有 system 应合并（不丢弃）。"""
        from gateway import _apply_prompt

        msgs = [
            Message(role=Role.SYSTEM, content="原有 system 消息"),
            Message(role=Role.USER, content="如何退款？"),
        ]
        prompt = PromptSelection(
            name="knowledge_decision",
            version="v1",
            variables={"product_name": "测试产品"},
        )
        req = make_request(messages=msgs, prompt=prompt)
        result = _apply_prompt(req)

        system_msgs = [m for m in result if m.role == Role.SYSTEM]
        assert len(system_msgs) == 2  # 模板 + 原有
        assert "测试产品" in system_msgs[0].content  # 模板渲染在前
        assert "原有 system 消息" in system_msgs[1].content  # 原有保留在后


# ──────────────────────────────────────────────
# 测试：Gateway 核心逻辑（使用 Mock）
# ──────────────────────────────────────────────

class TestGatewayLogic:
    """使用 Mock 测试 Gateway 核心逻辑。"""

    @pytest.mark.asyncio
    async def test_normal_call(self):
        """正常调用应返回正确内容、平台模型名和审计记录。"""
        from gateway import call_with_fallback, _traces

        _traces.clear()
        fake_response = make_fake_response()

        with patch(
            "provider.OpenAICompatibleProvider.complete",
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            req = make_request()
            resp = await call_with_fallback(req)

            assert resp.content == "你好！我是AI助手。"
            # 修复验证：返回平台名而非供应商名
            assert resp.model == "general-primary"
            assert resp.attempts >= 1
            assert resp.request_id

            # 审计记录
            assert len(_traces) == 1
            trace = _traces[0]
            assert trace.status == "success"
            assert trace.actual_model == "deepseek-chat"  # 审计保留供应商名
            assert trace.requested_model == "general-primary"
            assert trace.cost_usd > 0

    @pytest.mark.asyncio
    async def test_structured_output_valid(self):
        """结构化输出校验通过。"""
        from gateway import call_with_fallback

        schema = {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["sentiment", "confidence"],
        }
        valid_json = json.dumps({"sentiment": "positive", "confidence": 0.95})
        fake_response = make_fake_response(content=valid_json)

        with patch(
            "provider.OpenAICompatibleProvider.complete",
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            req = make_request(response_schema=schema)
            resp = await call_with_fallback(req)

            assert resp.parsed is not None
            assert resp.parsed["sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_structured_output_invalid(self):
        """结构化输出校验失败。"""
        from gateway import call_with_fallback

        schema = {
            "type": "object",
            "properties": {"sentiment": {"type": "string"}},
            "required": ["sentiment"],
        }
        fake_response = make_fake_response(content="这不是JSON")

        with patch(
            "provider.OpenAICompatibleProvider.complete",
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            req = make_request(response_schema=schema)
            with pytest.raises(GatewayError, match="structured_output_invalid"):
                await call_with_fallback(req)

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(self):
        """主模型连续超时 → 自动 fallback 到备用模型。"""
        from gateway import call_with_fallback, _traces
        from openai import APITimeoutError

        _traces.clear()
        call_count = 0
        fake_response = make_fake_response(
            content="备用模型的回复",
            model="gpt-4o-mini",
        )

        async def mock_complete(messages, response_schema=None, timeout_seconds=30, max_tokens=None):
            nonlocal call_count
            call_count += 1
            # 前 2 次调用（主模型）超时，后面（备用模型）正常返回
            if call_count <= 2:
                raise APITimeoutError(request=None)
            return fake_response

        with patch(
            "provider.OpenAICompatibleProvider.complete",
            side_effect=mock_complete,
        ):
            req = make_request()
            resp = await call_with_fallback(req)

            assert resp.content == "备用模型的回复"
            # 返回平台名（备用模型的平台名）
            assert resp.model == "general-backup"
            assert resp.attempts == 3  # 主模型2次 + 备用模型1次

    @pytest.mark.asyncio
    async def test_model_not_found(self):
        """白名单外的模型应返回 model_not_found。"""
        from gateway import call_with_fallback

        req = make_request(model="nonexistent-model")
        with pytest.raises(GatewayError, match="model_not_found"):
            await call_with_fallback(req)

    @pytest.mark.asyncio
    async def test_all_models_unavailable(self):
        """所有模型都不可用应返回 model_unavailable。"""
        from gateway import call_with_fallback, _traces
        from openai import APIConnectionError

        _traces.clear()

        with patch(
            "provider.OpenAICompatibleProvider.complete",
            new_callable=AsyncMock,
            side_effect=APIConnectionError(request=None),
        ):
            req = make_request()
            with pytest.raises(GatewayError, match="model_unavailable"):
                await call_with_fallback(req)

            # 应该有 error 审计记录
            error_traces = [t for t in _traces if t.status == "error"]
            assert len(error_traces) >= 1

    @pytest.mark.asyncio
    async def test_prompt_template_integration(self):
        """Prompt 模板渲染 + 调用：验证 system 消息合并。"""
        from gateway import call_with_fallback

        fake_response = make_fake_response()
        prompt = PromptSelection(
            name="knowledge_decision",
            version="v1",
            variables={"product_name": "测试产品"},
        )

        with patch(
            "provider.OpenAICompatibleProvider.complete",
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock:
            req = make_request(prompt=prompt)
            resp = await call_with_fallback(req)

            # 验证消息中包含渲染后的 system 消息
            call_args = mock.call_args
            messages = call_args.kwargs.get(
                "messages", call_args.args[0] if call_args.args else []
            )
            system_msgs = [m for m in messages if m.role == Role.SYSTEM]
            assert len(system_msgs) >= 1
            assert "测试产品" in system_msgs[0].content


# ──────────────────────────────────────────────
# 测试：流式输出 + 审计
# ──────────────────────────────────────────────

class TestStreaming:
    """流式输出 + 审计测试。"""

    @pytest.mark.asyncio
    async def test_stream_normal(self):
        """正常流式输出应产生正确的 SSE 事件 + 审计记录。"""
        from gateway import stream_with_fallback, _traces

        _traces.clear()

        async def fake_stream(messages, timeout_seconds=30, max_tokens=None):
            # 模拟文本块
            for chunk in ["你好", "！", "我是", "AI", "助手。"]:
                yield ProviderResponse(
                    text_delta=chunk, model="deepseek-chat",
                )
            # 模拟 usage 信息（显式 is_usage_chunk）
            yield ProviderResponse(
                text_delta="", model="deepseek-chat",
                input_tokens=10, output_tokens=20, total_tokens=30,
                is_usage_chunk=True,
            )

        with patch(
            "provider.OpenAICompatibleProvider.stream",
            side_effect=fake_stream,
        ):
            req = make_request()
            events = []
            async for event in stream_with_fallback(req):
                events.append(event)

            deltas = [e for e in events if e["type"] == "content.delta"]
            completed = [e for e in events if e["type"] == "response.completed"]

            assert len(deltas) == 5
            assert "".join(e["delta"] for e in deltas) == "你好！我是AI助手。"
            assert len(completed) == 1
            # completed 事件应包含 usage
            assert "usage" in completed[0]

            # 流式审计记录
            assert len(_traces) == 1
            trace = _traces[0]
            assert trace.status == "success"
            assert trace.input_tokens == 10
            assert trace.output_tokens == 20

    @pytest.mark.asyncio
    async def test_stream_model_not_found_via_validate(self):
        """validate_model_chain 应在 endpoint 层拦截白名单外模型。"""
        from gateway import validate_model_chain

        with pytest.raises(GatewayError, match="model_not_found"):
            validate_model_chain("nonexistent-model")


# ──────────────────────────────────────────────
# 测试：Anthropic 协议适配器
# ──────────────────────────────────────────────

class TestAnthropicProvider:
    """Anthropic Messages 协议 Mock 测试。"""

    @pytest.mark.asyncio
    async def test_anthropic_complete(self):
        """非流式调用：验证 system 提取、max_tokens 传参、响应解析。"""
        config = MODEL_CONFIGS["anthropic-claude"]
        provider = AnthropicProvider(config)

        # Mock Anthropic 响应对象
        fake_msg = type("FakeMsg", (), {
            "content": [type("FakeBlock", (), {"type": "text", "text": "Hello from Claude"})()],
            "model": "claude-sonnet-4-6",
            "usage": type("FakeUsage", (), {
                "input_tokens": 15, "output_tokens": 25,
            })(),
        })()

        fake_client = AsyncMock()
        fake_client.messages.create = AsyncMock(return_value=fake_msg)
        provider._client = fake_client

        messages = [
            Message(role=Role.SYSTEM, content="你是助手"),
            Message(role=Role.USER, content="你好"),
        ]
        result = await provider.complete(messages)

        assert result.content == "Hello from Claude"
        assert result.input_tokens == 15
        assert result.output_tokens == 25
        assert result.total_tokens == 40

        # 验证 system 被提取为顶层参数，不在 messages 中
        call_kwargs = fake_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "你是助手"
        assert all(m["role"] != "system" for m in call_kwargs["messages"])
        assert call_kwargs["max_tokens"] == 4096  # PROTOCOL_DEFAULT_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_anthropic_stream(self):
        """流式调用：验证 text 事件 → text_delta 映射 + usage 收集。"""
        config = MODEL_CONFIGS["anthropic-claude"]
        provider = AnthropicProvider(config)

        # 构造 Mock 流事件
        text_event_1 = type("E", (), {"type": "text", "text": "你好"})()
        text_event_2 = type("E", (), {"type": "text", "text": "世界"})()

        final_msg = type("FakeMsg", (), {
            "model": "claude-sonnet-4-6",
            "usage": type("FakeUsage", (), {
                "input_tokens": 8, "output_tokens": 4,
            })(),
        })()

        async def mock_events():
            yield text_event_1
            yield text_event_2

        class FakeStream:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def __aiter__(self):
                return mock_events()
            async def get_final_message(self):
                return final_msg

        fake_client = AsyncMock()
        fake_client.messages.stream = MagicMock(return_value=FakeStream())
        provider._client = fake_client

        messages = [Message(role=Role.USER, content="说两个字")]
        results = []
        async for pr in provider.stream(messages):
            results.append(pr)

        # 2 个 text 事件 + 1 个 usage chunk
        assert len(results) == 3
        assert results[0].text_delta == "你好"
        assert results[1].text_delta == "世界"
        assert results[2].is_usage_chunk is True
        assert results[2].input_tokens == 8
        assert results[2].output_tokens == 4

    @pytest.mark.asyncio
    async def test_anthropic_schema_injection(self):
        """结构化输出：Schema 应注入 system 参数（Anthropic 无原生 JSON mode）。"""
        config = MODEL_CONFIGS["anthropic-claude"]
        provider = AnthropicProvider(config)

        fake_msg = type("FakeMsg", (), {
            "content": [type("FakeBlock", (), {"type": "text", "text": '{"ok": true}'})()],
            "model": "claude-sonnet-4-6",
            "usage": type("FakeUsage", (), {
                "input_tokens": 10, "output_tokens": 5,
            })(),
        })()

        fake_client = AsyncMock()
        fake_client.messages.create = AsyncMock(return_value=fake_msg)
        provider._client = fake_client

        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        messages = [Message(role=Role.USER, content="test")]
        await provider.complete(messages, response_schema=schema)

        call_kwargs = fake_client.messages.create.call_args.kwargs
        # 无原始 system 消息时，应自动创建并注入 Schema
        assert "system" in call_kwargs
        assert "JSON Schema" in call_kwargs["system"]

    def test_create_provider_factory(self):
        """工厂函数应根据 protocol 返回正确的适配器类型。"""
        openai_config = ModelConfig(
            provider_model="test", base_url="http://test", api_key_env="K",
        )
        anthropic_config = ModelConfig(
            provider_model="test", base_url="http://test", api_key_env="K",
            protocol="anthropic",
        )

        p1 = create_provider(openai_config)
        p2 = create_provider(anthropic_config)

        assert isinstance(p1, OpenAICompatibleProvider)
        assert isinstance(p2, AnthropicProvider)


# ──────────────────────────────────────────────
# 测试：TTFT 度量
# ──────────────────────────────────────────────

class TestTTFT:
    """TTFT（首 Token 时间）度量测试。"""

    @pytest.mark.asyncio
    async def test_stream_ttft_recorded(self):
        """流式调用应记录 ttft_ms > 0。"""
        from gateway import stream_with_fallback, _traces

        _traces.clear()

        async def fake_stream(messages, timeout_seconds=30, max_tokens=None):
            await asyncio.sleep(0.05)  # 模拟首 Token 延迟
            yield ProviderResponse(text_delta="你好", model="deepseek-chat")
            yield ProviderResponse(text_delta="世界", model="deepseek-chat")
            yield ProviderResponse(
                text_delta="", model="deepseek-chat",
                input_tokens=5, output_tokens=4, total_tokens=9,
                is_usage_chunk=True,
            )

        with patch(
            "provider.OpenAICompatibleProvider.stream",
            side_effect=fake_stream,
        ):
            req = make_request()
            async for _ in stream_with_fallback(req):
                pass

            assert len(_traces) == 1
            assert _traces[0].ttft_ms is not None
            assert _traces[0].ttft_ms > 0

    @pytest.mark.asyncio
    async def test_stream_ttft_in_completed_event(self):
        """response.completed SSE 事件应包含 ttft_ms。"""
        from gateway import stream_with_fallback

        async def fake_stream(messages, timeout_seconds=30, max_tokens=None):
            yield ProviderResponse(text_delta="A", model="m")
            yield ProviderResponse(
                text_delta="", model="m",
                input_tokens=1, output_tokens=1, total_tokens=2,
                is_usage_chunk=True,
            )

        with patch(
            "provider.OpenAICompatibleProvider.stream",
            side_effect=fake_stream,
        ):
            req = make_request()
            events = []
            async for event in stream_with_fallback(req):
                events.append(event)

            completed = [e for e in events if e["type"] == "response.completed"]
            assert len(completed) == 1
            assert "ttft_ms" in completed[0]
            assert completed[0]["ttft_ms"] is not None

    @pytest.mark.asyncio
    async def test_non_stream_ttft_is_none(self):
        """非流式调用的 ttft_ms 应为 None。"""
        from gateway import call_with_fallback

        fake_response = make_fake_response()
        with patch(
            "provider.OpenAICompatibleProvider.complete",
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            req = make_request()
            resp = await call_with_fallback(req)
            assert resp.ttft_ms is None


# ──────────────────────────────────────────────
# 测试：按模型限流 429
# ──────────────────────────────────────────────

class TestRateLimiting:
    """线程安全按模型限流测试。"""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """rpm=2 时第 3 次调用应抛出 rate_limited。"""
        from gateway import _ModelRateLimiter

        limiter = _ModelRateLimiter()
        await limiter.acquire("test-rl-model", 2)
        await limiter.acquire("test-rl-model", 2)

        with pytest.raises(GatewayError, match="rate_limited"):
            await limiter.acquire("test-rl-model", 2)

    @pytest.mark.asyncio
    async def test_rate_limit_per_model_isolation(self):
        """模型 A 触发限流不影响模型 B。"""
        from gateway import _ModelRateLimiter

        limiter = _ModelRateLimiter()
        await limiter.acquire("model-A", 1)

        with pytest.raises(GatewayError, match="rate_limited"):
            await limiter.acquire("model-A", 1)

        # 模型 B 不受影响
        await limiter.acquire("model-B", 1)  # 不应抛出

    @pytest.mark.asyncio
    async def test_rate_limit_sliding_window(self):
        """窗口滑过后新请求应可通过。"""
        from gateway import _ModelRateLimiter

        limiter = _ModelRateLimiter()
        await limiter.acquire("test-sw-model", 1)

        with pytest.raises(GatewayError, match="rate_limited"):
            await limiter.acquire("test-sw-model", 1)

        # 模拟时间前进 61 秒，窗口外记录应被清除
        original_now = time.monotonic
        time.monotonic = lambda: original_now() + 61.0
        try:
            await limiter.acquire("test-sw-model", 1)  # 不应抛出
        finally:
            time.monotonic = original_now

    @pytest.mark.asyncio
    async def test_rate_limit_zero_means_unlimited(self):
        """rpm=0 表示不限制。"""
        from gateway import _ModelRateLimiter

        limiter = _ModelRateLimiter()
        for _ in range(100):
            await limiter.acquire("test-unlimited", 0)


# ──────────────────────────────────────────────
# 测试：模型隔离
# ──────────────────────────────────────────────

class TestModelIsolation:
    """验证不同协议模型使用正确的适配器。"""

    def test_openai_model_uses_openai_provider(self):
        """OpenAI 协议模型应使用 OpenAICompatibleProvider。"""
        config = MODEL_CONFIGS["general-primary"]
        provider = create_provider(config)
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_anthropic_model_uses_anthropic_provider(self):
        """Anthropic 协议模型应使用 AnthropicProvider。"""
        config = MODEL_CONFIGS["anthropic-claude"]
        provider = create_provider(config)
        assert isinstance(provider, AnthropicProvider)


# ──────────────────────────────────────────────
# 测试：HTTP 状态码契约
# ──────────────────────────────────────────────

class TestHTTPStatusContract:
    """HTTP 状态码映射契约测试。

    确保 GatewayError.http_status 映射稳定，
    防止重构翻转映射条件或删掉 pre-flight 调用时测试能捕获。
    """

    def test_gateway_error_rate_limited_429(self):
        """rate_limited 错误码应映射为 HTTP 429。"""
        exc = GatewayError("rate_limited", "test")
        assert exc.http_status == 429

    def test_gateway_error_auth_401(self):
        """provider_auth_error 应映射为 HTTP 401。"""
        exc = GatewayError("provider_auth_error", "test")
        assert exc.http_status == 401

    def test_gateway_error_default_422(self):
        """未注册错误码应默认 HTTP 422。"""
        exc = GatewayError("structured_output_invalid", "test")
        assert exc.http_status == 422

    def test_gateway_error_custom_status(self):
        """显式 http_status 应覆盖默认映射。"""
        exc = GatewayError("custom_error", "test", http_status=503)
        assert exc.http_status == 503

    @pytest.mark.asyncio
    async def test_validate_rate_limit_raises_429(self):
        """validate_rate_limit 在所有模型超限时抛出 rate_limited (429)。"""
        from gateway import validate_rate_limit, _ModelRateLimiter
        import gateway as gw

        # 临时给 fallback 链中所有模型设置限流，避免 backup 的 rpm=0 无条件放行
        rl_config_primary = ModelConfig(
            provider_model="test-p", base_url="http://test", api_key_env="K",
            rate_limit_rpm=1,
        )
        rl_config_backup = ModelConfig(
            provider_model="test-b", base_url="http://test", api_key_env="K",
            rate_limit_rpm=1,
        )
        original_primary = MODEL_CONFIGS["general-primary"]
        original_backup = MODEL_CONFIGS["general-backup"]
        MODEL_CONFIGS["general-primary"] = rl_config_primary
        MODEL_CONFIGS["general-backup"] = rl_config_backup
        # 临时替换限流器实例避免污染全局状态
        original_limiter = gw._rate_limiter
        gw._rate_limiter = _ModelRateLimiter()
        try:
            # 填满两个模型的配额
            await gw._rate_limiter.acquire("general-primary", 1)
            await gw._rate_limiter.acquire("general-backup", 1)
            with pytest.raises(GatewayError) as exc_info:
                validate_rate_limit("general-primary")
            assert exc_info.value.error_code == "rate_limited"
            assert exc_info.value.http_status == 429
        finally:
            MODEL_CONFIGS["general-primary"] = original_primary
            MODEL_CONFIGS["general-backup"] = original_backup
            gw._rate_limiter = original_limiter

    @pytest.mark.asyncio
    async def test_validate_rate_limit_passes_when_under_limit(self):
        """validate_rate_limit 未超限时应正常通过。"""
        from gateway import validate_rate_limit
        validate_rate_limit("general-primary")  # rpm=0 → 不限制

    @pytest.mark.asyncio
    async def test_streaming_provider_auth_error_event(self):
        """流式调用认证失败应产生 provider_auth_error SSE 事件。"""
        from gateway import stream_with_fallback
        from openai import AuthenticationError

        auth_error = AuthenticationError(
            message="Invalid API key",
            response=MagicMock(),
            body=None,
        )

        # AsyncMock 无法正确通过 async for 传播异常，
        # 需要用 async generator 函数 + yield 哨兵
        async def fake_auth_error_stream(*args, **kwargs):
            raise auth_error
            yield  # pragma: no cover — 使其成为 async generator

        with patch(
            "provider.OpenAICompatibleProvider.stream",
            side_effect=fake_auth_error_stream,
        ):
            req = make_request()
            events = []
            async for event in stream_with_fallback(req):
                events.append(event)

            error_events = [e for e in events if e["type"] == "error"]
            assert len(error_events) == 1
            assert error_events[0]["error_code"] == "provider_auth_error"

    @pytest.mark.asyncio
    async def test_would_accept_does_not_consume_quota(self):
        """would_accept 只读检查，不应修改窗口内容。"""
        from gateway import _ModelRateLimiter

        limiter = _ModelRateLimiter()
        await limiter.acquire("wa-test", 2)
        # would_accept 多次调用不应消费配额
        assert limiter.would_accept("wa-test", 2) is True
        assert limiter.would_accept("wa-test", 2) is True
        # 再 acquire 一次，还有 1 个配额
        await limiter.acquire("wa-test", 2)
        # 现在满了
        assert limiter.would_accept("wa-test", 2) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
