"""维护供模型选择界面展示的内置模型信息，无需网络即可查询。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from paicli.llm.model_profiles import DEFAULT_MODEL_PROFILES


@dataclass(frozen=True, slots=True)
class KnownModel:
    """模型选择界面使用的一项模型元信息。"""

    provider: str
    """Provider slug, e.g. ``"deepseek"``, ``"openai"``."""

    model: str
    """Full model identifier as passed to the API, e.g. ``"deepseek-v4-flash"``."""

    context_window: int
    """Maximum context window in tokens."""

    pricing: Literal["builtin", "unknown"] = "unknown"
    """Whether built-in price data is available for this model."""


# ---------------------------------------------------------------------------
# 内置模型展示列表
# ---------------------------------------------------------------------------
# 与模型工厂的上下文窗口、服务地址配置
# 以及价格模块中的默认档案保持一致。

_BUILTIN_MODELS: list[KnownModel] = [
    *(
        KnownModel(
            provider=profile.provider,
            model=profile.model,
            context_window=profile.context_window,
            pricing="builtin" if profile.provider == "deepseek" else "unknown",
        )
        for profile in DEFAULT_MODEL_PROFILES
    ),
    # 通用 OpenAI 兼容服务
    KnownModel(
        provider="openai",
        model="gpt-4o",
        context_window=128_000,
        pricing="unknown",
    ),
    KnownModel(
        provider="openai",
        model="gpt-4o-mini",
        context_window=128_000,
        pricing="unknown",
    ),
    # Kimi / Moonshot 模型
    KnownModel(
        provider="kimi",
        model="kimi-think",
        context_window=128_000,
        pricing="unknown",
    ),
    KnownModel(
        provider="moonshot",
        model="moonshot-v1-8k",
        context_window=8_000,
        pricing="unknown",
    ),
    KnownModel(
        provider="moonshot",
        model="moonshot-v1-32k",
        context_window=32_000,
        pricing="unknown",
    ),
    KnownModel(
        provider="moonshot",
        model="moonshot-v1-128k",
        context_window=128_000,
        pricing="unknown",
    ),
    # Step 模型
    KnownModel(
        provider="step",
        model="step-2-16k",
        context_window=16_000,
        pricing="unknown",
    ),
]


def get_available_models() -> list[KnownModel]:
    """返回内置模型展示列表，无需 API 凭据或网络访问。"""
    return list(_BUILTIN_MODELS)


def get_models_by_provider(provider: str) -> list[KnownModel]:
    """按服务标识过滤内置模型列表，忽略标识大小写。"""
    provider_lower = provider.lower()
    return [m for m in _BUILTIN_MODELS if m.provider.lower() == provider_lower]
