"""Keep Ollama-specific reasoning settings scoped to the local Qwen research call."""

from __future__ import annotations

import pytest

from backend.services.engine.rd_agent.llm_env import build_llm_env


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
