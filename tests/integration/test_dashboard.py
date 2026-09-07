"""The NiceGUI dashboard must render, on an empty warehouse and on a populated one, without
throwing — and the human-approval write path must produce exactly the `paper_trades` row shape
the LangGraph pipeline expects. No real browser: NiceGUI's `user` fixture drives the app over an
in-process ASGI transport (httpx), so this stays as fast and network-free as the rest of the suite.
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
    for view in ("Overview", "Data universe", "Backtest & calibration", "Suggestions & approval", "Ask the system"):
        await user.should_see(view)
    # Overview falls back to a "no data" note rather than raising when nothing has been ingested.
    await user.should_see("No data yet", retries=20)


async def test_dashboard_renders_with_backtest_and_artifacts(user: User, synthetic_league_matches) -> None:
    from pitch_edge.artifacts import build_all_artifacts
    from pitch_edge.backtest.engine import WalkForwardConfig
    from pitch_edge.config import get_settings
    from pitch_edge.data.ingest import store_matches
    from pitch_edge.data.storage import Warehouse
    from pitch_edge.models import DixonColesMatchModel
    from pitch_edge.pipeline import load_feature_frame, persist_features, run_backtests

    settings = get_settings()
    settings.ensure_dirs()
    with Warehouse(settings.db_path) as wh:
        store_matches(wh, synthetic_league_matches)
        f = load_feature_frame(wh, leagues=["SYN"])
        persist_features(wh, f)
        run_backtests(
            f, [DixonColesMatchModel()], WalkForwardConfig(min_train_matches=300, retrain_every_days=120), wh=wh
        )
        build_all_artifacts(wh, f)

    await user.open("/")
    await user.should_see("Best model vs closing price", retries=20)
    await user.should_see("Matches")


async def test_approval_writes_paper_trades_with_expected_columns(user: User, tmp_path) -> None:
    """Mirrors the shape the retired Streamlit tab wrote: one row per proposal, `approved_paper`
    for the chosen ones and `rejected` for the rest, keyed by `trade_id`, stamped with
    `approved_by`/`approved_at`/`run_id`. Nothing here can reach a bookmaker — it only appends to
    the local `paper_trades` warehouse table."""
    import json

    from pitch_edge.config import get_settings
    from pitch_edge.data.storage import Warehouse

    settings = get_settings()
    settings.ensure_dirs()
    proposal = {
        "match_id": "syn_test_match",
        "date": "2030-01-01",
        "home_team": "Home FC",
        "away_team": "Away FC",
        "outcome": "home",
        "model_probability": 0.6,
        "market_probability": 0.5,
        "edge": 0.1,
        "decimal_odds": 2.0,
        "stake": 10.0,
        "bookmaker": "PS",
        "model_name": "gbdt",
        "rationale": "test fixture",
    }
    (settings.artifacts_dir / "pending_signals.json").write_text(
        json.dumps({"thread_id": "thread-test", "proposals": [proposal]})
    )

    await user.open("/")
    user.find("Suggestions & approval").click()
    await user.should_see("Home FC vs Away FC", retries=20)
    user.find("Your name (required)").type("Test Approver")
    user.find("Home FC vs Away FC").click()
    user.find("Record decisions (paper only)").click()
    await user.should_see("Recorded 1 approved", retries=20)

    with Warehouse(settings.db_path, read_only=True) as wh:
        trades = wh.read("paper_trades")
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["trade_id"] == "thread-test:syn_test_match|home"
    assert row["status"] == "approved_paper"
    assert row["approved_by"] == "Test Approver"
    assert row["run_id"] == "thread-test"
    assert row["approved_at"]
