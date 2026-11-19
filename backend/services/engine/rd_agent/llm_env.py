"""因子挖掘 LLM 环境配置 — 统一打通 RD-Agent litellm 后端。

RD-Agent 因子挖掘用 litellm 作为 LLM 后端（rdagent/oai/backend/litellm.py）：
  - LiteLLMSettings env_prefix="LITELLM_" → 读 LITELLM_CHAT_MODEL / LITELLM_OPENAI_API_KEY / LITELLM_OPENAI_API_BASE
  - litellm.completion() 内部还从标准 OPENAI_API_KEY / OPENAI_API_BASE 读 HTTP 认证

但 launcher / rd_loop_wrapper / market_adapters 只设置无前缀变量（OPENAI_API_KEY 等），
且 DEEPSEEK_API_KEY 未接入，导致因子挖掘子进程无法调用任何 LLM（factor 表 0 条）。

本模块统一构建两套变量：
  LITELLM_*（RD-Agent settings 对象用）+ OPENAI_*（litellm HTTP 层用）
优先 DeepSeek（DEEPSEEK_API_KEY），回退现有 AI_IDE/讯飞 MaaS 配置。
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

# docker-compose 占位符，视为未配置
_PLACEHOLDER_KEYS = {"your-deepseek-api-key", "mock-api-key", "mock-api-key-not-configured"}
_OLLAMA_QWEN35_MODEL = "qwen3.5:9b-q4_k_m"


def _uses_local_ollama_qwen(env: dict[str, Any]) -> bool:
    model = (
        env.get("LITELLM_CHAT_MODEL", "").strip()
        or env.get("CHAT_MODEL", "").strip()
    ).lower()
    if model.startswith("openai/"):
        model = model[len("openai/") :]
    if model != _OLLAMA_QWEN35_MODEL:
        return False

    base = (
        env.get("LITELLM_OPENAI_API_BASE", "").strip()
        or env.get("OPENAI_BASE_URL", "").strip()
        or env.get("OPENAI_API_BASE", "").strip()
    )
    try:
        parsed = urlsplit(base)
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname == "host.docker.internal"
        and parsed.port == 11435
        and parsed.path.rstrip("/") == "/v1"
    )


def _allow_local_ollama_reasoning_param(
    env: dict[str, Any], completion_kwargs: dict[str, Any]
) -> bool:
    """Allow LiteLLM to pass the Qwen thinking switch to local Ollama."""
    if (
        env.get("LITELLM_REASONING_EFFORT") != "none"
        or not _uses_local_ollama_qwen(env)
        or completion_kwargs.get("reasoning_effort") != "none"
    ):
        return False

    configured_model = (
        env.get("LITELLM_CHAT_MODEL", "").strip()
        or env.get("CHAT_MODEL", "").strip()
    ).lower()
    request_model = str(completion_kwargs.get("model", "")).strip().lower()
    if configured_model.startswith("openai/"):
        configured_model = configured_model[len("openai/") :]
    if request_model.startswith("openai/"):
        request_model = request_model[len("openai/") :]
    if request_model != configured_model:
        return False

    allowed = completion_kwargs.get("allowed_openai_params")
    if allowed is None:
        allowed_params: list[str] = []
    elif isinstance(allowed, str):
        allowed_params = [allowed]
    else:
        allowed_params = list(allowed)
    if "reasoning_effort" not in allowed_params:
        allowed_params.append("reasoning_effort")
    completion_kwargs["allowed_openai_params"] = allowed_params
    return True


def _allow_local_ollama_response_schema(env: dict[str, Any], model: Any) -> bool:
    """Allow JSON mode only for the configured local Qwen/Ollama model."""
    if not _uses_local_ollama_qwen(env):
        return False

    configured_model = (
        env.get("LITELLM_CHAT_MODEL", "").strip()
        or env.get("CHAT_MODEL", "").strip()
    ).lower()
    request_model = str(model or "").strip().lower()
    if configured_model.startswith("openai/"):
        configured_model = configured_model[len("openai/") :]
    if request_model.startswith("openai/"):
        request_model = request_model[len("openai/") :]
    return request_model == configured_model


def allow_rdagent_local_ollama_response_schema(env: dict[str, Any]) -> bool:
    """Keep LiteLLM from stripping Ollama's supported OpenAI JSON mode."""
    if not _uses_local_ollama_qwen(env):
        return False

    from rdagent.oai.backend import litellm as rdagent_litellm

    if not getattr(
        rdagent_litellm, "_qm_local_qwen_response_schema_passthrough", False
    ):
        original_supports_response_schema = (
            rdagent_litellm.supports_response_schema
        )
        configured_env = dict(env)

        def supports_response_schema_with_local_qwen(
            model: Any, *args: Any, **kwargs: Any
        ) -> bool:
            if _allow_local_ollama_response_schema(configured_env, model):
                return True
            return original_supports_response_schema(model, *args, **kwargs)

        rdagent_litellm.supports_response_schema = (
            supports_response_schema_with_local_qwen
        )
        rdagent_litellm._qm_local_qwen_response_schema_passthrough = True
    return True


def allow_rdagent_none_reasoning_effort(env: dict[str, Any]) -> bool:
    """Enable the local Qwen thinking switch in this RD-Agent process."""
    if (
        env.get("LITELLM_REASONING_EFFORT") != "none"
        or not _uses_local_ollama_qwen(env)
    ):
        return False

    from typing import Literal

    from rdagent.oai.llm_conf import LLMSettings

    field = LLMSettings.model_fields.get("reasoning_effort")
    if field is None:
        raise RuntimeError("RD-Agent has no reasoning_effort setting to extend")

    field.annotation = Literal["low", "medium", "high", "none"] | None
    if not LLMSettings.model_rebuild(force=True):
        raise RuntimeError("Could not rebuild RD-Agent reasoning_effort setting")

    from rdagent.oai.backend import litellm as rdagent_litellm

    if not rdagent_litellm.LiteLLMSettings.model_rebuild(force=True):
        raise RuntimeError("Could not rebuild LiteLLM reasoning_effort setting")
    rdagent_litellm.LITELLM_SETTINGS = rdagent_litellm.LiteLLMSettings()

    if not getattr(
        rdagent_litellm, "_qm_local_qwen_reasoning_passthrough", False
    ):
        original_completion = rdagent_litellm.completion

        def completion_with_local_qwen_reasoning(*args: Any, **kwargs: Any):
            call_kwargs = dict(kwargs)
            _allow_local_ollama_reasoning_param(dict(os.environ), call_kwargs)
            return original_completion(*args, **call_kwargs)

        rdagent_litellm.completion = completion_with_local_qwen_reasoning
        rdagent_litellm._qm_local_qwen_reasoning_passthrough = True
    return True


def _is_placeholder(key: str) -> bool:
    k = (key or "").strip().lower()
    return (not k) or any(p in k for p in _PLACEHOLDER_KEYS) or k.startswith("sk-在此")


def build_llm_env(env: dict[str, Any]) -> dict[str, Any]:
    """在现有 env 基础上补齐 litellm 需要的 LITELLM_ + OPENAI_ 变量。

    API key 优先级：DEEPSEEK_API_KEY（非占位） > LITELLM_OPENAI_API_KEY > AI_IDE_LLM_API_KEY > OPENAI_API_KEY
    模型优先级：   DEEPSEEK（若 deepseek key 生效） > CHAT_MODEL
    """
    deepseek_key = (
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        or env.get("DEEPSEEK_API_KEY", "").strip()
    )
    if _is_placeholder(deepseek_key):
        deepseek_key = ""

    deepseek_base = (
        os.getenv("DEEPSEEK_BASE_URL", "").strip()
        or env.get("DEEPSEEK_BASE_URL", "").strip()
        or "https://api.deepseek.com/v1"
    )
    # DeepSeek base URL 必须带 /v1 后缀，否则 /chat/completions 会 404
    if deepseek_base and not deepseek_base.rstrip("/").endswith("/v1"):
        deepseek_base = deepseek_base.rstrip("/") + "/v1"

    # ---- 决定最终 key / base / model ----
    litellm_key = (
        env.get("LITELLM_OPENAI_API_KEY", "").strip()
        or deepseek_key
        or os.getenv("AI_IDE_LLM_API_KEY", "").strip()
        or os.getenv("AI_IDE_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if _is_placeholder(litellm_key):
        litellm_key = ""

    # Base URL：deepseek 优先
    litellm_base = (
        env.get("LITELLM_OPENAI_API_BASE", "").strip()
        or (deepseek_base if deepseek_key else "")
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("OPENAI_API_BASE", "").strip()
    )

    # Chat model：deepseek 优先
    if deepseek_key:
        deepseek_model = (
            env.get("DEEPSEEK_MODEL", "").strip()
            or os.getenv("DEEPSEEK_MODEL", "").strip()
            or "deepseek-chat"
        )
        if not deepseek_model.startswith(("deepseek/", "openai/")):
            deepseek_model = f"deepseek/{deepseek_model}"
        chat_model = deepseek_model
    else:
        chat_model = (
            env.get("CHAT_MODEL", "").strip()
            or os.getenv("CHAT_MODEL", "").strip()
            or "gpt-4-turbo"
        )
        if chat_model and not chat_model.startswith(("openai/", "azure/", "anthropic/", "huggingface/", "deepseek/")):
            chat_model = f"openai/{chat_model}"

    # ---- 写入 LITELLM_（RD-Agent settings）----
    if litellm_key:
        env["LITELLM_OPENAI_API_KEY"] = litellm_key
    if litellm_base:
        env["LITELLM_OPENAI_API_BASE"] = litellm_base
    if chat_model:
        env["LITELLM_CHAT_MODEL"] = chat_model
    effort = str(env.get("LITELLM_REASONING_EFFORT", "")).strip()
    if _uses_local_ollama_qwen(env) and not effort:
        env["LITELLM_REASONING_EFFORT"] = "none"

    # ---- 写入标准 OPENAI_（litellm HTTP 层 + 兼容旧路径）----
    # 当用 deepseek 时，必须覆盖 OPENAI_API_KEY/BASE，否则 litellm 会用讯飞的 key 打 deepseek
    if deepseek_key:
        env["OPENAI_API_KEY"] = deepseek_key
        env["OPENAI_BASE_URL"] = deepseek_base
        env["CHAT_MODEL"] = chat_model
    elif litellm_key and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = litellm_key
        if litellm_base and not env.get("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = litellm_base

    return env
