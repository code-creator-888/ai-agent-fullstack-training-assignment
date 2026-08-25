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

import json
import sys
import os
from unittest.mock import AsyncMock, patch

import pytest

# 确保 1-1-assignment 目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIGS, render_prompt, calculate_cost
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

        async def mock_complete(messages, response_schema=None, timeout_seconds=30):
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

        async def fake_stream(messages, timeout_seconds=30):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
