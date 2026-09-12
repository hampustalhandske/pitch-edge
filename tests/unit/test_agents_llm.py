"""LLM factory: builds a ChatOllama or ChatGroq pointed at settings, without ever calling out.

Every test sets `PITCH_EDGE_LLM_PROVIDER` explicitly rather than relying on "ollama" being the
default — a developer's own `.env` may set it to "groq" for daily use, and these tests must pass
regardless of that.
"""

from __future__ import annotations

import pytest

from pitch_edge.agents.llm import GROQ_DEEP_MODEL, GROQ_FAST_MODEL, get_llm, get_llm_for

pytestmark = pytest.mark.unit


def test_get_llm_defaults_to_ollama_with_settings(monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("PITCH_EDGE_LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setenv("PITCH_EDGE_LOCAL_LLM_BASE_URL", "http://example:11434")
    llm = get_llm()
    assert llm.model == "qwen2.5:7b-instruct"
    assert llm.base_url == "http://example:11434"
    assert llm.temperature == 0


def test_get_llm_ollama_explicit_override(monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_LLM_PROVIDER", "ollama")
    llm = get_llm(model="llama3.1:8b", base_url="http://localhost:11434")
    assert llm.model == "llama3.1:8b"


def test_get_llm_ollama_applies_a_request_timeout(monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("PITCH_EDGE_LOCAL_LLM_TIMEOUT_S", "5")
    llm = get_llm()
    assert llm.client_kwargs == {"timeout": 5.0}


def test_get_llm_ollama_timeout_defaults_to_sixty_seconds(monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("PITCH_EDGE_LOCAL_LLM_TIMEOUT_S", raising=False)
    llm = get_llm()
    assert llm.client_kwargs == {"timeout": 60.0}


def test_get_llm_switches_to_groq_via_provider_setting(monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_CLOUD_API_KEY", "test-key")
    llm = get_llm()
    assert type(llm).__name__ == "ChatGroq"
    assert llm.model_name == "openai/gpt-oss-20b"


def test_get_llm_groq_explicit_model_override(monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_CLOUD_API_KEY", "test-key")
    llm = get_llm(model="openai/gpt-oss-120b")
    assert llm.model_name == "openai/gpt-oss-120b"


def test_get_llm_for_fast_role_on_groq_uses_fast_model_with_deep_fallback(monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_CLOUD_API_KEY", "test-key")
    llm = get_llm_for("fast")
    assert type(llm).__name__ == "RunnableWithFallbacks"
    assert llm.runnable.model_name == GROQ_FAST_MODEL
    assert llm.fallbacks[0].model_name == GROQ_DEEP_MODEL


def test_get_llm_for_deep_role_on_groq_uses_deep_model_with_fast_fallback(monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_CLOUD_API_KEY", "test-key")
    llm = get_llm_for("deep")
    assert llm.runnable.model_name == GROQ_DEEP_MODEL
    assert llm.fallbacks[0].model_name == GROQ_FAST_MODEL


def test_get_llm_for_on_ollama_never_falls_back_to_a_second_model(monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_LLM_PROVIDER", "ollama")
    llm = get_llm_for("fast")
    assert type(llm).__name__ == "ChatOllama"
