"""The NiceGUI dashboard must render, on an empty warehouse and on a populated one, without
throwing. No real browser: NiceGUI's `user` fixture drives the app over an in-process ASGI
transport (httpx), so this stays as fast and network-free as the rest of the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from nicegui.testing import User

WEB_FILE = str(Path(__file__).resolve().parents[2] / "src" / "pitch_edge" / "dashboard" / "web.py")

pytestmark = [pytest.mark.integration, pytest.mark.anyio, pytest.mark.nicegui_main_file(WEB_FILE)]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _restore_main_module():
    """NiceGUI's `user` fixture loads `web.py` via `runpy.run_path(..., run_name="__main__")` to
    re-run its module-level `@ui.page` registration on a clean slate for every test. `runpy` is
    supposed to restore the real `sys.modules['__main__']` afterwards, but in this suite it can
    leave it missing, which later breaks unrelated code that inspects the call stack (e.g.
    librosa's lazy loader in `tests/unit/test_alt_data.py`). Belt-and-braces: snapshot and put it
    back ourselves so no other test in the session sees a corrupted `__main__`."""
    original = sys.modules.get("__main__")
    yield
    if original is not None:
        sys.modules["__main__"] = original


async def test_dashboard_renders_on_empty_warehouse(user: User) -> None:
    await user.open("/")
    await user.should_see("PITCH-EDGE")
    await user.should_see("Ask the system")
    await user.should_see("top N bets")


async def test_ask_flow_degrades_gracefully_without_a_local_llm(user: User, synthetic_league_matches) -> None:
    """No Ollama is running in this test environment, so `parse_intent` can never structure a
    question — the UI must still respond with the standard "can't understand" message rather
    than crashing, even with real backtest-eligible data (real Pinnacle odds) in the warehouse."""
    from pitch_edge.config import get_settings
    from pitch_edge.data.ingest import store_matches
    from pitch_edge.data.storage import Warehouse
    from pitch_edge.pipeline import load_feature_frame, persist_features

    settings = get_settings()
    settings.ensure_dirs()
    with Warehouse(settings.db_path) as wh:
        store_matches(wh, synthetic_league_matches)
        persist_features(wh, load_feature_frame(wh, leagues=["SYN"]))

    await user.open("/")
    user.find("Question").type("top 4 bets")
    user.find("Ask").click()
    await user.should_see("as of", retries=30)
    await user.should_see("I can't understand")


# --------------------------------------------------------------------------- evidence section


def _write_registration(directory: Path, hypothesis_id: str, family: str) -> Path:
    import yaml

    doc = {
        "id": hypothesis_id,
        "family": family,
        "registered_at": "2026-10-20T21:00:00Z",
        "author": "test",
        "question": "Does the test strategy beat the de-vigged close?",
        "hypothesis": {"h1": "clv > 0", "h0": "clv = 0", "direction": "positive", "disconfirming": "CI includes 0"},
        "decision_time": "kickoff - 24h",
        "universe": "synthetic",
        "window": {"exploratory": ["2020-01-01", "2021-12-31"], "evaluation": ["2022-01-01", "2023-12-31"]},
        "benchmark": "synthetic close",
        "primary_metric": "clv_log",
        "costs": "none",
        "min_n": 300,
        "literature_ceiling": "+2% CLV",
    }
    path = directory / f"{hypothesis_id}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _paper_decision(store, strategy_id: str, hypothesis_id: str, clv: float | None) -> None:
    from datetime import UTC, datetime, timedelta

    from pitch_edge.paper.store import Decision, Quote, Settlement

    decided = datetime(2026, 10, 1, 12, tzinfo=UTC)
    q = Quote(
        source="odds_api:pinnacle",
        source_event_id=f"e-{strategy_id}",
        home="Home FC",
        away="Away FC",
        league="EPL",
        kickoff_at=decided + timedelta(hours=6),
        selection="home",
        captured_at=decided,
        decimal_odds=2.0,
    )
    store.add_quotes([q])
    fk = store.quotes().query("source_event_id == @q.source_event_id")["fixture_key"].iloc[0]
    d = Decision(
        strategy_id=strategy_id,
        hypothesis_id=hypothesis_id,
        decided_at=decided,
        fixture_key=fk,
        kickoff_at=q.kickoff_at,
        venue="odds_api:pinnacle",
        selection="home",
        price_taken=2.0,
        paper_stake_units=1.0,
        model_prob=0.5,
        inputs_quote_ids=[q.quote_id],
        code_version="sha",
    )
    store.append_decision(d)
    if clv is not None:
        store.append_settlement(
            Settlement(
                decision_id=d.decision_id,
                closing_price=2.0,
                fair_close_odds=2.0,
                closing_source="lake:fd_matches.betfair_ex_close",
                result="home",
                clv_log=clv,
                pnl=1.0,
                settled_at=decided + timedelta(days=1),
            )
        )


async def test_evidence_section_renders_with_no_ledgers(user: User) -> None:
    await user.open("/")
    await user.should_see("Evidence")
    await user.should_see("display floor L4", retries=30)
    await user.should_see("No validated edge")
    await user.should_see("no hypotheses registered yet")
    await user.should_see("no paper ledger yet")


async def test_evidence_section_renders_families_and_paper_status(user: User, tmp_path) -> None:
    from pitch_edge.config import get_settings
    from pitch_edge.paper.store import PaperStore
    from pitch_edge.research.ledger import Ledger
    from pitch_edge.research.registry import load

    data_dir = get_settings().data_dir
    ledger = Ledger(data_dir / "research" / "ledger.duckdb")
    ledger.register(load(_write_registration(tmp_path, "dash-fam-idea-v1", "dash-fam")))
    with PaperStore(data_dir / "paper" / "paper.duckdb") as store:
        _paper_decision(store, "dash-control", "control", 0.01)

    await user.open("/")
    await user.should_see("dash-fam", retries=30)
    await user.should_see("champion dash-fam-idea-v1")
    await user.should_see("L0")
    await user.should_see("dash-control")
    await user.should_see("decided 1 · settled 1")
    await user.should_see("not enough settled bets")
    await user.should_see("unregistered, not evidence")
