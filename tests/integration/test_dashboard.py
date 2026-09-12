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
