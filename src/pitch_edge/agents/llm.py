"""Local LLM factory for the agentic signals pipeline.

Uses Ollama so the reviewer/selection agents never need a paid API key — the
project's non-negotiable is that the LLM only ever *explains* retrieved evidence, and running it
locally keeps that evidence-grounding loop free to run per-fixture. Requires `ollama serve` running
with a model pulled (`ollama pull llama3.1:8b`); that is an operational prerequisite for whoever
runs `agentic-signals`, not something this code can install. Every agent-node test uses a fake chat
model instead (e.g. `langchain_core.language_models.fake_chat_models.FakeListChatModel`) so
`uv run pytest` never needs Ollama running.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from pitch_edge.config import get_settings


def get_local_llm(model: str | None = None, base_url: str | None = None) -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        model=model or settings.local_llm_model,
        base_url=base_url or settings.local_llm_base_url,
        temperature=0,
    )
