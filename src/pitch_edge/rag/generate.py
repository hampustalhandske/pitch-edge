"""Grounded answer generation — Claude Fable 5.1 explains; the quant models decide.

Contract:
* Only the retrieved documents may be used; every number must appear verbatim in a source and
  every claim must carry a `[doc_id]` citation. `verify_citations` checks both mechanically,
  so a hallucinated figure is flagged (`Answer.verified == False`) rather than trusted.
* The LLM never produces a probability. Nothing it writes is a betting instruction.
* Fable 5.1 specifics (per the Anthropic SDK guidance): thinking is always on, so no
  `thinking` parameter is sent; server-side refusal fallbacks are enabled by default; a
  `refusal` stop reason falls back to the deterministic template.
* No credential → deterministic template with the same grounded structure, so the feature
  works offline and in CI.
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


def verify_citations(text: str, docs: list[Document]) -> tuple[bool, list[str], list[str]]:
    """Return (all numbers grounded?, list of cited doc ids that exist, numbers not found in any source)."""
    doc_ids = {d.doc_id for d in docs}
    cited = [c for c in _CITE_RE.findall(text) if c in doc_ids]
    source_text = " ".join(d.text for d in docs)
    source_numbers = set(_NUMBER_RE.findall(source_text))
    source_numbers |= {n.replace(",", ".") for n in source_numbers} | {n.replace(".", ",") for n in source_numbers}
    answer_numbers = _NUMBER_RE.findall(_CITE_RE.sub("", text))
    missing = sorted(
        {n for n in answer_numbers if n not in source_numbers and n.rstrip("0").rstrip(".") not in source_numbers}
    )
    return (not missing), cited, missing


class GroundedGenerator:
    def __init__(self, api_key: str | None = None, model: str | None = None, force_template: bool = False):
        settings = get_settings()
        self.model = model or settings.anthropic_model
        self._client = None
        if not force_template and (api_key or settings.anthropic_api_key):
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            except Exception as exc:  # noqa: BLE001
                logger.info("Anthropic client unavailable (%s); template generator active", exc)
                self._client = None

    @property
    def backend(self) -> str:
        return "claude" if self._client is not None else "template"

    def answer(self, question: str, docs: list[Document]) -> Answer:
        if not docs:
            return Answer(
                "No relevant documents were retrieved for that question, so I can't answer it from the system's data.",
                [],
                self.backend,
                [],
                verified=True,
            )
        if self._client is not None:
            try:
                return self._answer_claude(question, docs)
            except Exception as exc:  # noqa: BLE001 - never break the dashboard on an API error
                logger.warning("Claude generation failed (%s); using template", exc)
        return self._answer_template(question, docs)

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
