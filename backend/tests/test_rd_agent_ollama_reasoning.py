"""Keep Ollama-specific reasoning settings scoped to the local Qwen research call."""

from __future__ import annotations

import pytest

from backend.services.engine.rd_agent.llm_env import (
    _allow_local_ollama_reasoning_param,
    _allow_local_ollama_response_schema,
    allow_rdagent_local_ollama_response_schema,
    build_llm_env,
)


_UNRELATED_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "AI_IDE_LLM_API_KEY",
    "AI_IDE_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "CHAT_MODEL",
    "LITELLM_CHAT_MODEL",
    "LITELLM_OPENAI_API_BASE",
    "LITELLM_OPENAI_API_KEY",
    "LITELLM_REASONING_EFFORT",
)


@pytest.fixture
def clean_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _UNRELATED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_local_qwen_ollama_gets_none_reasoning_effort(
    clean_llm_environment: None,
) -> None:
    env = {
        "CHAT_MODEL": "qwen3.5:9b-q4_K_M",
        "OPENAI_BASE_URL": "http://host.docker.internal:11435/v1",
        "OPENAI_API_KEY": "ollama",
    }

    build_llm_env(env)

    assert env["LITELLM_CHAT_MODEL"] == "openai/qwen3.5:9b-q4_K_M"
    assert env["LITELLM_REASONING_EFFORT"] == "none"


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("qwen3.5:9b-q4_K_M", "https://api.example.com/v1"),
        ("qwen3:8b", "http://host.docker.internal:11435/v1"),
    ],
)
def test_reasoning_effort_none_is_not_set_for_other_routes(
    clean_llm_environment: None, model: str, base_url: str
) -> None:
    env = {
        "CHAT_MODEL": model,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": "ollama",
    }

    build_llm_env(env)

    assert "LITELLM_REASONING_EFFORT" not in env


def test_explicit_reasoning_effort_is_preserved(
    clean_llm_environment: None,
) -> None:
    env = {
        "CHAT_MODEL": "qwen3.5:9b-q4_K_M",
        "OPENAI_BASE_URL": "http://host.docker.internal:11435/v1",
        "OPENAI_API_KEY": "ollama",
        "LITELLM_REASONING_EFFORT": "medium",
    }

    build_llm_env(env)

    assert env["LITELLM_REASONING_EFFORT"] == "medium"


def test_local_qwen_reasoning_effort_is_allowed_through_litellm(
    clean_llm_environment: None,
) -> None:
    env = {
        "CHAT_MODEL": "qwen3.5:9b-q4_K_M",
        "OPENAI_BASE_URL": "http://host.docker.internal:11435/v1",
        "OPENAI_API_KEY": "ollama",
    }
    build_llm_env(env)
    completion_kwargs = {
        "model": env["LITELLM_CHAT_MODEL"],
        "reasoning_effort": "none",
    }

    assert _allow_local_ollama_reasoning_param(env, completion_kwargs)
    assert completion_kwargs["allowed_openai_params"] == ["reasoning_effort"]


def test_local_qwen_preserves_other_allowed_openai_params(
    clean_llm_environment: None,
) -> None:
    env = {
        "CHAT_MODEL": "qwen3.5:9b-q4_K_M",
        "OPENAI_BASE_URL": "http://host.docker.internal:11435/v1",
        "OPENAI_API_KEY": "ollama",
    }
    build_llm_env(env)
    completion_kwargs = {
        "model": env["LITELLM_CHAT_MODEL"],
        "reasoning_effort": "none",
        "allowed_openai_params": ["seed"],
    }

    assert _allow_local_ollama_reasoning_param(env, completion_kwargs)
    assert completion_kwargs["allowed_openai_params"] == [
        "seed",
        "reasoning_effort",
    ]


def test_local_qwen_response_schema_is_scoped_to_configured_model(
    clean_llm_environment: None,
) -> None:
    env = {
        "CHAT_MODEL": "qwen3.5:9b-q4_K_M",
        "OPENAI_BASE_URL": "http://host.docker.internal:11435/v1",
        "OPENAI_API_KEY": "ollama",
    }
    build_llm_env(env)

    assert _allow_local_ollama_response_schema(
        env, "openai/qwen3.5:9b-q4_K_M"
    )
    assert not _allow_local_ollama_response_schema(env, "openai/another-model")


def test_local_qwen_response_schema_capability_reaches_litellm(
    clean_llm_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rdagent.oai.backend import litellm as rdagent_litellm

    env = {
        "CHAT_MODEL": "qwen3.5:9b-q4_K_M",
        "OPENAI_BASE_URL": "http://host.docker.internal:11435/v1",
        "OPENAI_API_KEY": "ollama",
    }
    build_llm_env(env)
    monkeypatch.setattr(
        rdagent_litellm, "supports_response_schema", lambda model: False
    )
    monkeypatch.setattr(
        rdagent_litellm,
        "_qm_local_qwen_response_schema_passthrough",
        False,
        raising=False,
    )

    assert allow_rdagent_local_ollama_response_schema(env)
    assert rdagent_litellm.supports_response_schema(
        "openai/qwen3.5:9b-q4_K_M"
    )
    assert not rdagent_litellm.supports_response_schema("openai/another-model")


def test_reasoning_param_is_not_allowed_for_other_models(
    clean_llm_environment: None,
) -> None:
    env = {
        "CHAT_MODEL": "qwen3:8b",
        "OPENAI_BASE_URL": "http://host.docker.internal:11435/v1",
        "OPENAI_API_KEY": "ollama",
        "LITELLM_REASONING_EFFORT": "none",
    }
    completion_kwargs = {"model": "openai/qwen3:8b", "reasoning_effort": "none"}

    assert not _allow_local_ollama_reasoning_param(env, completion_kwargs)
    assert "allowed_openai_params" not in completion_kwargs
