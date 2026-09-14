"""Grounded answer generation — an LLM explains; the quant models decide.

Contract:
* Only the retrieved documents may be used; every number must appear verbatim in a source and
  every claim must carry a `[doc_id]` citation. `verify_citations` checks both mechanically,
  so a hallucinated figure is flagged (`Answer.verified == False`) rather than trusted.
* The LLM never produces a probability. Nothing it writes is a betting instruction.
* Backend precedence is local-first: the local Ollama model (`agents/llm.py`, same one the
  agentic-signals selection/reviewer agents use) is tried first — free, no key needed. Claude
  Fable 5.1 is used only if a local attempt fails/isn't reachable and `ANTHROPIC_API_KEY` is set
  (Fable specifics per the Anthropic SDK guidance: thinking is always on, so no `thinking`
  parameter is sent; server-side refusal fallbacks are enabled by default; a `refusal` stop
  reason falls through too). Neither reachable → deterministic template with the same grounded
  structure, so the feature works fully offline and in CI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pitch_edge.config import get_settings
from pitch_edge.rag.documents import Document

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the explanation layer of PITCH-EDGE, a football analytics and market-edge research system.

Rules:
- Answer ONLY from the retrieved documents. If they do not contain the answer, say so plainly.
- Quote every number exactly as it appears in a document and cite the document id in square brackets, e.g. [pred:gbdt:fd_E0_...]. Every sentence that states a fact ends with at least one citation.
- Never invent, round differently, or adjust probabilities, odds or statistics. The quantitative models produce the probabilities; you explain them and never offer your own estimate.
- Use the field's vocabulary correctly (xG, xT, PPDA, Dixon-Coles, no-vig, CLV, fractional Kelly).
- Nothing you write is a betting instruction; signals are for manual review only.
- Format: a short direct answer, then bullet points for drivers and caveats. Plain prose, no headings."""

SCOUTING_PROMPT = (
    "Write a concise scouting-style report on {subject} using only the documents. Structure: Summary; Strengths; "
    "Weaknesses; Statistical profile (quote xG/shots/passing figures with citations); Market context if present; Caveats."
)

_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?:\s?%)?(?![\w])")
_CITE_RE = re.compile(r"\[([A-Za-z0-9_:\-./ ]+?)\]")


@dataclass
class Answer:
    text: str
    citations: list[str]
    backend: str  # "claude" | "template"
    documents: list[Document]
    verified: bool = True
    unverified_numbers: list[str] = field(default_factory=list)
    model: str | None = None


def _classify_number(text: str, start: int, end: int) -> str | None:
    """Best-effort metric type for a number at `text[start:end]`, from nearby context — used only
    to sanity-range-check it (a percentage-shaped number > 100, or "odds" of 0.6, is wrong however
    the LLM sourced it). None if no type is inferable; such numbers get only the grounding check."""
    after = text[end : end + 2]
    before = text[max(0, start - 20) : start].lower()
    if after.startswith("%"):
        return "probability"
    if "odds" in before or "odds" in text[end : end + 15].lower():
        return "odds"
    if "edge" in before:
        return "edge"
    return None


def _in_range(kind: str, value: float) -> bool:
    if kind == "probability":
        return 0.0 <= value <= 100.0
    if kind == "odds":
        return value > 1.0
    if kind == "edge":
        return -100.0 <= value <= 100.0
    return True


def verify_citations(text: str, docs: list[Document]) -> tuple[bool, list[str], list[str]]:
    """Return (all numbers grounded?, list of cited doc ids that exist, numbers not found in any
    source OR out of range for their inferred metric type).

    Two independent checks, both must pass:
    1. Grounding (as before): the digit string appears somewhere in the retrieved source text.
    2. Range sanity: a number whose nearby context marks it as a probability/edge/odds (a "%"
       sign, or the word "odds"/"edge" close by) must fall in that metric's valid range — this
       catches a hallucinated "150% probability" or "0.4 odds" even if that exact digit string
       happens to appear elsewhere in an unrelated source document."""
    doc_ids = {d.doc_id for d in docs}
    cited = [c for c in _CITE_RE.findall(text) if c in doc_ids]
    source_text = " ".join(d.text for d in docs)
    source_numbers = set(_NUMBER_RE.findall(source_text))
    source_numbers |= {n.replace(",", ".") for n in source_numbers} | {n.replace(".", ",") for n in source_numbers}

    answer_only = _CITE_RE.sub("", text)
    missing: set[str] = set()
    for m in _NUMBER_RE.finditer(answer_only):
        n = m.group(1)
        ungrounded = n not in source_numbers and n.rstrip("0").rstrip(".") not in source_numbers
        kind = _classify_number(answer_only, m.start(), m.end())
        out_of_range = False
        if kind:
            try:
                out_of_range = not _in_range(kind, float(n.replace(",", ".")))
            except ValueError:
                out_of_range = False
        if ungrounded or out_of_range:
            missing.add(n)
    return (not missing), cited, sorted(missing)


class GroundedGenerator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        local_model: str | None = None,
        force_template: bool = False,
    ):
        settings = get_settings()
        self.model = model or settings.anthropic_model
        self._client = None
        self._local_llm = None
        if not force_template:
            try:
                from pitch_edge.agents.llm import get_llm

                self._local_llm = get_llm(model=local_model)
            except Exception as exc:  # noqa: BLE001
                logger.info("Local LLM unavailable (%s); will try Claude/template", exc)
            if api_key or settings.anthropic_api_key:
                try:
                    import anthropic

                    self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
                except Exception as exc:  # noqa: BLE001
                    logger.info("Anthropic client unavailable (%s)", exc)
                    self._client = None

    @property
    def backend(self) -> str:
        if self._local_llm is not None:
            return "local"
        if self._client is not None:
            return "claude"
        return "template"

    def answer(self, question: str, docs: list[Document]) -> Answer:
        if not docs:
            return Answer(
                "No relevant documents were retrieved for that question, so I can't answer it from the system's data.",
                [],
                "template",
                [],
                verified=True,
            )
        if self._local_llm is not None:
            try:
                return self._answer_local(question, docs)
            except Exception as exc:  # noqa: BLE001 - Ollama unreachable must never break the dashboard
                logger.info("Local LLM generation failed (%s); trying Claude/template", exc)
        if self._client is not None:
            try:
                return self._answer_claude(question, docs)
            except Exception as exc:  # noqa: BLE001 - never break the dashboard on an API error
                logger.warning("Claude generation failed (%s); using template", exc)
        return self._answer_template(question, docs)

    # ------------------------------------------------------------------ local
    def _answer_local(self, question: str, docs: list[Document]) -> Answer:
        from langchain_core.messages import HumanMessage, SystemMessage

        assert self._local_llm is not None
        context = "\n\n".join(f"[{d.doc_id}] ({d.metadata.get('type', 'doc')}) {d.text}" for d in docs)
        response = self._local_llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Retrieved documents:\n\n{context}\n\nQuestion: {question}"),
            ]
        )
        text = str(response.content).strip()
        ok, cited, missing = verify_citations(text, docs)
        if missing:
            text += f"\n\n_Note: the figure(s) {', '.join(missing)} could not be matched to a retrieved document and should not be relied on._"
        local_model_name = getattr(self._local_llm, "model", None) or getattr(self._local_llm, "model_name", None)
        return Answer(text, cited, "local", docs, verified=ok, unverified_numbers=missing, model=local_model_name)

    # ----------------------------------------------------------------- claude
    def _answer_claude(self, question: str, docs: list[Document]) -> Answer:
        assert self._client is not None
        context = "\n\n".join(f"[{d.doc_id}] ({d.metadata.get('type', 'doc')}) {d.text}" for d in docs)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 2000,
            "output_config": {"effort": "medium"},
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": f"Retrieved documents:\n\n{context}\n\nQuestion: {question}"}],
        }
        if self.model.startswith("claude-fable"):
            # Fable: thinking is always on (no `thinking` param); opt into server-side refusal fallbacks.
            response = self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            response = self._client.messages.create(thinking={"type": "adaptive"}, **kwargs)
        if response.stop_reason == "refusal":
            logger.info("Claude declined (%s); using template", getattr(response, "stop_details", None))
            return self._answer_template(question, docs)
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
        ok, cited, missing = verify_citations(text, docs)
        if missing:
            text += f"\n\n_Note: the figure(s) {', '.join(missing)} could not be matched to a retrieved document and should not be relied on._"
        return Answer(
            text,
            cited,
            "claude",
            docs,
            verified=ok,
            unverified_numbers=missing,
            model=getattr(response, "model", self.model),
        )

    # --------------------------------------------------------------- template
    def _answer_template(self, question: str, docs: list[Document]) -> Answer:
        groups = {
            "prediction": ("Model view", 3),
            "match": ("Relevant matches", 4),
            "statsbomb": ("Event-data summary", 2),
            "news": ("News context", 3),
            "player": ("Player profile", 3),
        }
        lines = [f'Grounded summary for: "{question}" (template generator — no LLM configured).']
        for kind, (title, n) in groups.items():
            items = [d for d in docs if d.metadata.get("type") == kind][:n]
            if items:
                lines.append(f"\n{title}:")
                lines += [f"- {d.text[:260]} [{d.doc_id}]" for d in items]
        lines.append(
            "\nAll figures above are quoted verbatim from the system's own records; nothing here is a betting instruction."
        )
        text = "\n".join(lines)
        return Answer(text, [d.doc_id for d in docs], "template", docs, verified=True)


def scouting_report(generator: GroundedGenerator, subject: str, docs: list[Document]) -> Answer:
    return generator.answer(SCOUTING_PROMPT.format(subject=subject), docs)
