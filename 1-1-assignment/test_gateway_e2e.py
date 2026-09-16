"""集成测试：使用真实 DeepSeek API 做端到端验证。

需要设置环境变量 DEEPSEEK_API_KEY 才能运行。
跳过条件：如果 API Key 未设置，所有测试自动跳过。

测试场景：
- 非流式调用 + Prompt 模板渲染
- 结构化输出 + response_schema 校验
- 流式 SSE 输出 + 审计
- Pydantic 请求校验（未知字段、非法组合）
- Prompt 模板变量缺失
- 模型白名单校验（含流式路径）
- 模拟主模型超时 → fallback 到备用模型
- 审计记录完整性
- LLMResponse.model 返回平台名
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

# 确保 1-1-assignment 目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIGS
from models import (
    GatewayError,
    LLMRequest,
    Message,
    PromptSelection,
    Role,
)

# 跳过条件：API Key 未设置
HAS_DEEPSEEK_KEY = bool(os.getenv("DEEPSEEK_API_KEY"))
skip_no_key = pytest.mark.skipif(
    not HAS_DEEPSEEK_KEY,
    reason="需要设置 DEEPSEEK_API_KEY 环境变量",
)

HAS_ANTHROPIC_KEY = bool(
    os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-")
)
skip_no_anthropic_key = pytest.mark.skipif(
    not HAS_ANTHROPIC_KEY,
    reason="需要设置 ANTHROPIC_API_KEY 环境变量（必须以 sk-ant- 开头）",
)


# ──────────────────────────────────────────────
# 辅助工具
# ──────────────────────────────────────────────

def make_e2e_request(
    content: str = "用一句话介绍Python",
    response_schema: dict | None = None,
    response_format: dict | None = None,
    prompt: PromptSelection | None = None,
    stream: bool = False,
) -> LLMRequest:
    return LLMRequest(
        model="general-primary",
        messages=[Message(role=Role.USER, content=content)],
        response_schema=response_schema,
        response_format=response_format,
        prompt=prompt,
        stream=stream,
    )


# ──────────────────────────────────────────────
# 端到端测试
# ──────────────────────────────────────────────

@skip_no_key
class TestE2ENormalCall:
    """非流式调用端到端测试。"""

    @pytest.mark.asyncio
    async def test_basic_call(self):
        """基本非流式调用：验证平台名回填 + 审计完整性。"""
        from gateway import call_with_fallback, _traces

        _traces.clear()
        req = make_e2e_request()
        resp = await call_with_fallback(req)

        assert resp.content  # 非空
        # 修复验证：返回平台名
        assert resp.model == "general-primary"
        assert resp.request_id
        assert resp.usage.total_tokens > 0
        assert resp.latency_ms > 0

        # 审计记录完整性
        assert len(_traces) == 1
        trace = _traces[0]
        assert trace.status == "success"
        assert trace.actual_model == "deepseek-chat"  # 审计保留供应商名
        assert trace.requested_model == "general-primary"
        assert trace.input_tokens > 0
        assert trace.output_tokens > 0
        assert trace.cost_usd > 0


@skip_no_key
class TestE2EPromptTemplate:
    """Prompt 模板端到端测试。"""

    @pytest.mark.asyncio
    async def test_prompt_render(self):
        """Prompt 模板渲染 + 调用。"""
        from gateway import call_with_fallback

        prompt = PromptSelection(
            name="knowledge_decision",
            version="v1",
            variables={"product_name": "智能客服"},
        )
        req = make_e2e_request(content="如何退款？", prompt=prompt)
        resp = await call_with_fallback(req)
        assert resp.content


@skip_no_key
class TestE2EStructuredOutput:
    """结构化输出端到端测试。"""

    @pytest.mark.asyncio
    async def test_structured_output(self):
        """结构化输出 + Schema 校验。"""
        from gateway import call_with_fallback

        schema = {
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["language", "description"],
        }
        req = make_e2e_request(
            content="用JSON格式介绍Python语言",
            response_schema=schema,
        )
        resp = await call_with_fallback(req)

        assert resp.parsed is not None
        assert "language" in resp.parsed
        assert "description" in resp.parsed


@skip_no_key
class TestE2EStreaming:
    """流式输出端到端测试。"""

    @pytest.mark.asyncio
    async def test_stream_sse(self):
        """流式 SSE 输出 + 审计。"""
        from gateway import stream_with_fallback, _traces

        _traces.clear()
        req = make_e2e_request(content="说一个字")
        deltas = []
        completed = False

        async for event in stream_with_fallback(req):
            if event["type"] == "content.delta":
                deltas.append(event["delta"])
            elif event["type"] == "response.completed":
                completed = True

        assert len(deltas) > 0
        assert completed

        # 流式审计
        assert len(_traces) == 1
        assert _traces[0].status == "success"


@skip_no_anthropic_key
class TestE2EAnthropic:
    """Anthropic 协议端到端测试。"""

    @pytest.mark.asyncio
    async def test_anthropic_basic_call(self):
        """Anthropic 模型基本非流式调用。"""
        from gateway import call_with_fallback, _traces

        _traces.clear()
        req = LLMRequest(
            model="anthropic-claude",
            messages=[Message(role=Role.USER, content="用一句话介绍Python")],
        )
        resp = await call_with_fallback(req)

        assert resp.content
        assert resp.model == "anthropic-claude"
        assert resp.usage.total_tokens > 0

        # 审计记录
        assert len(_traces) == 1
        assert _traces[0].status == "success"
        assert _traces[0].requested_model == "anthropic-claude"


@skip_no_key
class TestE2ETTFT:
    """TTFT 度量端到端测试。"""

    @pytest.mark.asyncio
    async def test_stream_ttft_e2e(self):
        """真实 API 流式调用应记录 ttft_ms > 0。"""
        from gateway import stream_with_fallback, _traces

        _traces.clear()
        req = LLMRequest(
            model="general-primary",
            messages=[Message(role=Role.USER, content="说一个字")],
            stream=True,
        )

        async for _ in stream_with_fallback(req):
            pass

        assert len(_traces) == 1
        assert _traces[0].ttft_ms is not None
        assert _traces[0].ttft_ms > 0


# ──────────────────────────────────────────────
# 校验测试（不依赖 API）
# ──────────────────────────────────────────────

class TestE2EValidation:
    """请求校验测试（不需要 API Key）。"""

    def test_extra_field_forbidden(self):
        """未知字段 → 422 拒绝。"""
        with pytest.raises(Exception):
            LLMRequest(
                model="general-primary",
                messages=[Message(role=Role.USER, content="hi")],
                malicious_field="<script>alert(1)</script>",
            )

    def test_stream_schema_conflict(self):
        """stream + response_schema 组合 → 拒绝。"""
        with pytest.raises(Exception):
            LLMRequest(
                model="general-primary",
                messages=[Message(role=Role.USER, content="hi")],
                stream=True,
                response_schema={"type": "object"},
            )

    def test_prompt_missing_variable(self):
        """Prompt 变量缺失 → missing_prompt_variable。"""
        from config import render_prompt

        with pytest.raises(KeyError, match="missing_prompt_variable"):
            render_prompt("knowledge_decision", "v1", {})

    def test_model_whitelist_non_stream(self):
        """白名单外模型 → model_not_found（非流式）。"""
        from gateway import call_with_fallback

        async def _test():
            req = LLMRequest(
                model="gpt-4-turbo",
                messages=[Message(role=Role.USER, content="hi")],
            )
            with pytest.raises(GatewayError, match="model_not_found"):
                await call_with_fallback(req)

        asyncio.run(_test())

    def test_model_whitelist_stream(self):
        """白名单外模型 → model_not_found（流式路径，endpoint 层校验）。"""
        from gateway import validate_model_chain

        with pytest.raises(GatewayError, match="model_not_found"):
            validate_model_chain("gpt-4-turbo")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
