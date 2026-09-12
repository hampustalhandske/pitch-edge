"""LLM factory for the `ask` agent — local Ollama by default, Groq's free tier as an alternative.

Provider is chosen by `Settings.llm_provider` (env `PITCH_EDGE_LLM_PROVIDER`, "ollama" or "groq"), so
every call site (`rag/query_parser.py`, `agents/qa_context.py`, `agents/qa_graph.py`,
`agents/tool_reviewer.py`, `rag/generate.py`) gets a swap-in-compatible `BaseChatModel` without its
own code changing — both `ChatOllama` and `ChatGroq` support `.with_structured_output()` and
`bind_tools`/`create_react_agent` the same way.

Groq's free tier (see console.groq.com/docs/rate-limits) is a real recurring daily quota, not a
one-time trial credit like Hugging Face's free tier currently is — 1,000 requests/day and 200,000
tokens/day per model, with a tighter 8,000-tokens-per-minute ceiling that a bursty run can hit.
Structured JSON-schema output and tool calling are natively supported, unlike most of Hugging
Face's free serverless models today.

On Groq, `get_llm_for` assigns a different model per pipeline role and wraps it with the *other*
Groq model as an automatic fallback (`Runnable.with_fallbacks`) — real multi-model resilience, never
a fallback to the local model:
  - "fast" (`parse_intent`, the squad-value news hint): `GROQ_FAST_MODEL` primary, confirmed
    reliable and fast (0.25-0.6s) for short structured-extraction calls; `GROQ_DEEP_MODEL` as
    fallback if it errors.
  - "deep" (`judge`, the tool-calling reasoning step): `GROQ_DEEP_MODEL` primary — more capable
    for synthesizing several fixtures' evidence — with `GROQ_FAST_MODEL` as fallback.

Every call site still wraps its LLM call in its own try/except that degrades to a deterministic
fallback (a template answer, or "I can't understand that"), so even both Groq models failing (or a
timeout) behaves like the provider being unreachable, never a hang or a crash.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from pitch_edge.config import get_settings

GROQ_FAST_MODEL = "openai/gpt-oss-20b"
GROQ_DEEP_MODEL = "openai/gpt-oss-120b"


def get_llm(model: str | None = None, base_url: str | None = None) -> BaseChatModel:
    settings = get_settings()
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        from pydantic import SecretStr

        return ChatGroq(
            model=model or settings.groq_model,
            api_key=SecretStr(settings.groq_api_key) if settings.groq_api_key else None,
            temperature=0,
            timeout=settings.local_llm_timeout_s,
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model or settings.local_llm_model,
        base_url=base_url or settings.local_llm_base_url,
        temperature=0,
        client_kwargs={"timeout": settings.local_llm_timeout_s},
    )


def get_llm_for(role: str) -> BaseChatModel | Runnable:
    """`role`: "fast" (quick classification/extraction) or "deep" (tool-calling reasoning over
    several fixtures). On Groq this returns the role's primary model with the other Groq model
    wired in as an automatic fallback; on Ollama (a single local model, nothing to route between)
    it's just `get_llm()`."""
    settings = get_settings()
    if settings.llm_provider != "groq":
        return get_llm()
    primary_model, fallback_model = (
        (GROQ_FAST_MODEL, GROQ_DEEP_MODEL) if role == "fast" else (GROQ_DEEP_MODEL, GROQ_FAST_MODEL)
    )
    return get_llm(model=primary_model).with_fallbacks([get_llm(model=fallback_model)])
