"""模型白名单、Prompt 模板注册表、价格表。

调用方只能使用白名单中的平台模型名，实际供应商信息对外不可见。
密钥只在 Gateway 进程内通过环境变量读取，实现密钥隔离。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Template


# ──────────────────────────────────────────────
# 模型配置
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """单个平台模型的配置。"""
    provider_model: str             # 实际供应商模型名（作为默认值，可被环境变量覆盖）
    base_url: str                   # 供应商 API 地址（作为默认值，可被环境变量覆盖）
    api_key_env: str                # 密钥环境变量名
    base_url_env: str = ""          # API 地址环境变量名（设置后覆盖 base_url）
    provider_model_env: str = ""    # 模型名环境变量名（设置后覆盖 provider_model）
    protocol: str = "openai"        # 上游协议类型："openai" | "anthropic"
    supports_structured_output: bool = False
    structured_output_mode: str = ""  # "json_schema" | "json_object"
    rate_limit_rpm: int = 0         # 每分钟请求上限（0 表示不限制）
    max_tokens: int = 0             # 最大输出 Token 数（0 表示使用协议默认值）


# 模型白名单：调用方只能使用这些平台模型名
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "general-primary": ModelConfig(
        provider_model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        provider_model_env="DEEPSEEK_MODEL",
        supports_structured_output=True,
        structured_output_mode="json_object",
    ),
    "general-backup": ModelConfig(
        provider_model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        provider_model_env="OPENAI_MODEL",
        supports_structured_output=True,
        structured_output_mode="json_schema",
    ),
    "anthropic-claude": ModelConfig(
        provider_model="claude-sonnet-4-6",
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        base_url_env="ANTHROPIC_BASE_URL",
        provider_model_env="ANTHROPIC_MODEL",
        protocol="anthropic",
        supports_structured_output=False,
    ),
}

# fallback 链：主模型失败后自动切换到备用模型
FALLBACK_MODEL = "general-backup"


# ──────────────────────────────────────────────
# Prompt 模板
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class PromptTemplate:
    """Prompt 模板定义，使用 string.Template 语法 (${variable})。"""
    system_template: str


# Prompt 模板注册表：(name, version) -> PromptTemplate
PROMPT_TEMPLATES: dict[tuple[str, str], PromptTemplate] = {
    ("knowledge_decision", "v1"): PromptTemplate(
        system_template=(
            "你是${product_name}的知识库决策器，"
            "根据用户问题判断应该从哪个知识库获取答案。"
            "只返回 JSON 格式的结果。"
        ),
    ),
    ("summarize", "v1"): PromptTemplate(
        system_template=(
            "请对以下内容进行摘要，控制在${max_words}字以内。"
            "保留关键信息，去除冗余内容。"
        ),
    ),
}


def render_prompt(name: str, version: str, variables: dict[str, str]) -> str:
    """渲染 Prompt 模板。

    Args:
        name: 模板名称
        version: 模板版本
        variables: 模板变量

    Returns:
        渲染后的 system prompt

    Raises:
        KeyError: 模板不存在
        KeyError: 变量缺失
    """
    key = (name, version)
    if key not in PROMPT_TEMPLATES:
        raise KeyError(f"prompt_not_found: {name}@{version}")

    template = PROMPT_TEMPLATES[key]
    try:
        return Template(template.system_template).substitute(variables)
    except KeyError as e:
        raise KeyError(f"missing_prompt_variable: {e}") from e


# ──────────────────────────────────────────────
# 价格表（每百万 Token 的美元价格）
# ──────────────────────────────────────────────

# 各协议 max_tokens 默认值（当 ModelConfig.max_tokens == 0 时使用）
PROTOCOL_DEFAULT_MAX_TOKENS: dict[str, int] = {
    "openai": 4096,
    "anthropic": 4096,
}

PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    "deepseek-chat":  {"input": 0.27, "output": 1.10},
    "gpt-4o-mini":    {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


def calculate_cost(provider_model: str, input_tokens: int, output_tokens: int) -> float:
    """计算调用成本（美元）。"""
    prices = PRICE_PER_MILLION.get(provider_model, {"input": 0.0, "output": 0.0})
    cost = (
        input_tokens * prices["input"] / 1_000_000
        + output_tokens * prices["output"] / 1_000_000
    )
    return round(cost, 8)
