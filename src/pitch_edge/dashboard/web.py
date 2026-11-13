"""PITCH-EDGE — the dashboard.

One page: a question box and an answer, and under it a compact read-only "Evidence" section — no
charts, no approval flow. The question goes straight through the same `ask` LangGraph the CLI uses
(`agents/qa_graph.py`); the Evidence section shows, per research family, the best rung on the
`research/STANDARDS.md` ladder and its champion, the display floor (the minimum rung for a number to
be shown as an edge), and the paper ledger's forward record, all read through
`research/summary.py::load_status` (short-lived read-only connections, cached until either ledger
file changes). This module never trains a model, never fetches odds, and never computes a number
itself.

Built with NiceGUI — a Python-native UI framework on top of FastAPI + Vue/Quasar. Chosen over a
separate JS frontend because the "Python-only" platform value in CLAUDE.md is real: no API
contract worth its own service, no client-side build step, and no second language for a single
maintainer to keep in sync.
"""

from __future__ import annotations

import os

from nicegui import run, ui

from pitch_edge.dashboard.theme import INK, SERIES, STATUS, install_template
from pitch_edge.research.summary import meets_floor, rung_rank

install_template()

PAGE_BG = "#0d0d0d"
SURFACE = "#161615"
BORDER = "rgba(255,255,255,0.08)"


# ============================================================================= small helpers
def card(*, title: str | None = None) -> ui.card:
    c = (
        ui.card()
        .classes("w-full")
        .style(f"background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; box-shadow:none;")
    )
    with c:
        if title:
            ui.label(title).classes("text-sm font-medium").style(f"color:{INK['secondary']}; letter-spacing:.01em;")
    return c


def banner(text: str, kind: str = "note") -> None:
    border = STATUS["warning"] if kind == "warn" else SERIES[0]
    bg = "#241f13" if kind == "warn" else SURFACE
    ui.html(
        f'<div style="border-left:3px solid {border}; background:{bg}; color:{INK["secondary"]}; '
        f'padding:10px 14px; border-radius:8px; font-size:.88rem; line-height:1.5;">{text}</div>'
    ).classes("w-full")


def render_data_notes(notes: list[str]) -> None:
    """One warning banner line per league whose warehouse data ends well before the as-of date."""
    for line in notes:
        banner(line, "warn")


def section_title(text: str, caption: str = "") -> None:
    ui.label(text).style("font-size:1.15rem; font-weight:600; letter-spacing:-.01em;")
    if caption:
        ui.label(caption).style(f"color:{INK['muted']}; font-size:.85rem; margin-top:-6px;")


# ============================================================================= ask the system
def build_ask() -> None:
    section_title("Ask the system")
    banner(
        "Three standardized questions only: <b>'top N bets'</b>, <b>'&lt;team&gt; vs &lt;team&gt;'</b>, or "
        "<b>'what's on &lt;day&gt;'</b>. Anything else gets a plain 'I can't understand that'. Every question "
        "is answered as of an explicit past date (there is no live odds feed, so there is no genuine 'now' to "
        "answer as of instead) — grounded in real backtest evidence and real market odds; the LLM only "
        "explains and never computes a probability or an edge itself.",
    )
    examples = ["top 4 bets", "Liverpool vs Arsenal", "what's on 2024-03-16"]
    with card():
        q = ui.input("Question", placeholder=examples[0]).classes("w-full")
        as_of_input = ui.input("As of (YYYY-MM-DD, optional)").classes("w-64")
        ui.label("Try: " + " · ".join(examples)).style(f"color:{INK['muted']}; font-size:.8rem;")
        ask_btn = ui.button("Ask", icon="search").props("color=primary")
        result = ui.column().classes("w-full gap-2")

    async def on_ask() -> None:
        question = q.value.strip()
        if not question:
            return
        result.clear()
        with result:
            ui.spinner(size="lg")
            ui.label("thinking…").style(f"color:{INK['muted']};")
        ask_btn.disable()
        try:
            state = await run.io_bound(_ask_sync, question, as_of_input.value.strip() or None)  # type: ignore[misc]
        finally:
            ask_btn.enable()
        assert state is not None
        result.clear()
        with result:
            ui.label(f"as of {state.as_of.date()}").style(f"color:{INK['muted']}; font-size:.78rem;")
            if state.message:
                ui.markdown(state.message.replace("_", "\\_"))
                render_data_notes(state.data_notes)
                return
            if state.answer is None:
                ui.label("No answer produced.").style(f"color:{INK['muted']};")
                return
            ui.markdown(state.answer.overview.replace("_", "\\_"))
            render_data_notes(state.data_notes)
            badge = (
                "all figures matched a source"
                if state.answer.citations_grounded
                else f"unmatched figures: {', '.join(state.answer.ungrounded_numbers)}"
            )
            ui.label(f"backend={state.answer.backend} · {badge}").style(f"color:{INK['muted']}; font-size:.78rem;")
            for note in state.answer.notes:
                with ui.column().classes("gap-0"):
                    ui.label(f"{note.home_team} vs {note.away_team}".strip(" vs")).style(
                        f"color:{INK['secondary']}; font-size:.85rem; font-weight:600;"
                    )
                    ui.label(note.take).style(f"color:{INK['muted']}; font-size:.85rem;")
            if state.context_docs:
                with ui.expansion("Retrieved context").classes("w-full"):
                    seen: set[str] = set()
                    for docs in state.context_docs.values():
                        for d in docs:
                            if d["doc_id"] in seen:
                                continue
                            seen.add(d["doc_id"])
                            ui.label(f"{d['doc_id']}: {d['text']}").style(f"color:{INK['muted']}; font-size:.8rem;")

    ask_btn.on_click(on_ask)
    q.on("keydown.enter", on_ask)


# ============================================================================= evidence
def rung_colour(rung: str, floor: str) -> str:
    """STATUS colour for a rung: good at/above the display floor, warning from L2 up, serious for
    L0/L1, critical when not even L0 holds."""
    if meets_floor(rung, floor):
        return STATUS["good"]
    if rung_rank(rung) >= rung_rank("L2"):
        return STATUS["warning"]
    if rung_rank(rung) >= rung_rank("L0"):
        return STATUS["serious"]
    return STATUS["critical"]


def _rung_pill(rung: str, floor: str) -> None:
    colour = rung_colour(rung, floor)
    ui.label(rung).style(
        f"border:1px solid {colour}; color:{colour}; border-radius:999px; padding:0 8px; font-size:.75rem;"
    )


def _paper_line(s: dict) -> str:
    parts = [f"decided {s['n_decided']} · settled {s['n_settled']}"]
    ci = s.get("clv_ci95")
    if ci:
        parts.append(f"mean log-CLV {s['mean_clv']:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    else:
        parts.append("not enough settled bets for a CLV CI")
    if s.get("n_exploratory"):
        parts.append(f"exploratory {s['n_exploratory']}")
    if s.get("n_void"):
        parts.append(f"void {s['n_void']}")
    return " · ".join(parts)


def render_evidence(status: dict) -> None:
    """Render a `research.summary.StatusSnapshot.to_dict()` payload: the display floor, one row per
    research family (best rung, champion) and one row per paper strategy (forward record)."""
    research, paper = status.get("research", {}), status.get("paper", {})
    floor = research.get("display_floor", "L4")
    if research.get("validated_edge_exists"):
        banner(f"Display floor {floor}: only strategies at or above it are shown as edges.")
    else:
        banner(
            f"No validated edge: no strategy holds the display floor {floor}. Anything the ask agent shows "
            "is a model-vs-market disagreement, not a validated edge.",
            "warn",
        )
    muted = f"color:{INK['muted']}; font-size:.82rem;"
    with card(title=f"Research ladder · display floor {floor}"):
        families = research.get("families", [])
        if not families:
            ui.label("no hypotheses registered yet").style(muted)
        for fam in families:
            with ui.row().classes("items-center gap-2"):
                _rung_pill(fam["best_rung"], floor)
                ui.label(fam["family"]).style(f"color:{INK['secondary']}; font-size:.85rem; font-weight:600;")
                champion = fam.get("champion")
                ui.label(f"champion {champion['id']}" if champion else "no champion").style(muted)
                ui.label(f"{len(fam.get('hypotheses', []))} hypotheses").style(muted)
    with card(title="Paper ledger · forward record (evaluation window, lake closes)"):
        if not paper.get("available"):
            ui.label(paper.get("reason") or "paper ledger unavailable").style(muted)
        elif not paper.get("strategies"):
            ui.label("no paper decisions yet").style(muted)
        for s in paper.get("strategies", []):
            with ui.column().classes("gap-0"):
                name = (
                    s["strategy_id"]
                    if s.get("counts_as_evidence")
                    else f"{s['strategy_id']} · unregistered, not evidence"
                )
                ui.label(name).style(f"color:{INK['secondary']}; font-size:.85rem; font-weight:600;")
                ui.label(_paper_line(s)).style(muted)


def _evidence_status() -> dict:
    from pitch_edge.research.summary import load_status

    return load_status().to_dict()


def build_evidence() -> None:
    section_title("Evidence", "Read-only: the research evidence ladder and the paper ledger's forward record.")
    container = ui.column().classes("w-full gap-2")
    with container:
        ui.spinner(size="md")

    async def fill() -> None:
        try:
            status = await run.io_bound(_evidence_status)
        except Exception as exc:  # noqa: BLE001 - an unreadable ledger must not break the page
            container.clear()
            with container:
                banner(f"Evidence status unavailable: <code>{type(exc).__name__}</code>", "warn")
            return
        container.clear()
        with container:
            render_evidence(status or {})

    ui.timer(0.05, fill, once=True)


_ASK_CACHE: dict = {}


def _ask_graph():
    if "graph" not in _ASK_CACHE:
        from pitch_edge.agents.qa_graph import build_qa_graph
        from pitch_edge.config import get_settings
        from pitch_edge.data.storage import Warehouse
        from pitch_edge.models import default_models
        from pitch_edge.pipeline import build_rag_index, load_feature_frame

        settings = get_settings()
        settings.ensure_dirs()
        # Kept open (not a `with` block) for the graph's lifetime: PolymarketOddsProvider needs a
        # live connection each time the cached graph is invoked, not just at build time.
        wh = Warehouse(settings.db_path, read_only=True)
        features = load_feature_frame(wh)
        index = build_rag_index(wh)
        slice_path = settings.backtest_dir / "main" / "slice_evidence.csv"
        import pandas as pd

        slice_evidence = pd.read_csv(slice_path) if slice_path.exists() else pd.DataFrame()
        models = default_models()
        _ASK_CACHE["wh"] = wh
        _ASK_CACHE["graph"] = build_qa_graph(features, slice_evidence, index, models, wh=wh)
    return _ASK_CACHE["graph"]


def _ask_sync(question: str, as_of: str | None):
    import pandas as pd

    from pitch_edge.agents.qa_graph import ask_question
    from pitch_edge.backtest.as_of import default_as_of
    from pitch_edge.config import get_settings
    from pitch_edge.data.storage import Warehouse

    if as_of:
        cutoff = pd.Timestamp(as_of)
    else:
        settings = get_settings()
        with Warehouse(settings.db_path, read_only=True) as wh:
            cutoff = default_as_of(wh)
    return ask_question(_ask_graph(), question, cutoff)


# ============================================================================= app shell
@ui.page("/")
def index() -> None:
    ui.add_head_html(
        f"<style>"
        f"body {{ background:{PAGE_BG} !important; }}"
        f".q-page {{ background:{PAGE_BG}; }}"
        f"::-webkit-scrollbar {{ width:10px; height:10px; }}"
        f"::-webkit-scrollbar-thumb {{ background:#333; border-radius:6px; }}"
        f"</style>"
    )
    ui.dark_mode().enable()
    ui.colors(primary=SERIES[0])

    with (
        ui.header()
        .classes("items-center justify-between")
        .style(f"background:{PAGE_BG}; border-bottom:1px solid {BORDER}; padding:10px 20px;")
    ):
        with ui.row().classes("items-center gap-3"):
            ui.label("PITCH-EDGE").style("font-size:1.25rem; font-weight:700; letter-spacing:-.01em;")
            for text in ("no auto-betting", "answers, not bets"):
                ui.label(text).style(
                    "background:#17304f; color:#86b6ef; border-radius:999px; padding:2px 10px; font-size:.72rem;"
                )

    with ui.column().classes("w-full gap-3").style("max-width:1000px; margin:0 auto; padding:20px;"):
        for build in (build_ask, build_evidence):
            try:
                build()
            except Exception as exc:  # noqa: BLE001 - never blank-page the app on a render error
                banner(f"This view failed to render: <code>{exc}</code>", "warn")


def main() -> None:
    port = int(os.environ.get("PITCH_EDGE_DASHBOARD_PORT", "8501"))
    ui.run(title="PITCH-EDGE", port=port, reload=False, show=False, favicon="⚽", dark=True)


if __name__ in {"__main__", "__mp_main__"}:
    # NiceGUI's browser-less test harness (`nicegui.testing.User`) loads this file with
    # `runpy.run_path(..., run_name="__main__")` and relies on this guard calling `ui.run()` to
    # register routes/config; `ui.run()` itself detects the simulation and returns before binding
    # a real socket (see `helpers.is_user_simulation()`), so this is safe under both `pitch-edge
    # serve` and the test harness.
    main()
