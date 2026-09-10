"""Local LLM factory: builds a ChatOllama pointed at settings, without ever calling Ollama."""

from __future__ import annotations

import pytest

from pitch_edge.agents.llm import get_local_llm

pytestmark = pytest.mark.unit


def test_get_local_llm_uses_settings_defaults(monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setenv("PITCH_EDGE_LOCAL_LLM_BASE_URL", "http://example:11434")
    llm = get_local_llm()
    assert llm.model == "qwen2.5:7b-instruct"
    assert llm.base_url == "http://example:11434"
    assert llm.temperature == 0


def test_get_local_llm_explicit_override():
    llm = get_local_llm(model="llama3.1:8b", base_url="http://localhost:11434")
    assert llm.model == "llama3.1:8b"
