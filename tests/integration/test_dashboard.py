"""The Streamlit app must render without exceptions on an empty warehouse and on a populated one."""

from __future__ import annotations

from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[2] / "src" / "pitch_edge" / "dashboard" / "app.py")

pytestmark = pytest.mark.integration


def _run_app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=300).run()


def test_dashboard_renders_on_empty_warehouse(tmp_path, monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_DATA_DIR", str(tmp_path / "data"))
    at = _run_app()
    assert not at.exception
    assert len(at.tabs) == 10
    assert any("No data yet" in i.value for i in at.info)


def test_dashboard_renders_with_backtest_and_artifacts(tmp_path, monkeypatch, synthetic_league_matches):
    monkeypatch.setenv("PITCH_EDGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PITCH_EDGE_REPORTS_DIR", str(tmp_path / "reports"))
    from pitch_edge.artifacts import build_all_artifacts
    from pitch_edge.backtest.engine import WalkForwardConfig
    from pitch_edge.config import get_settings
    from pitch_edge.data.ingest import store_matches
    from pitch_edge.data.storage import Warehouse
    from pitch_edge.models import DixonColesMatchModel
    from pitch_edge.pipeline import load_feature_frame, persist_features, run_backtests

    s = get_settings()
    s.ensure_dirs()
    with Warehouse(s.db_path) as wh:
        store_matches(wh, synthetic_league_matches)
        f = load_feature_frame(wh, leagues=["SYN"])
        persist_features(wh, f)
        run_backtests(
            f, [DixonColesMatchModel()], WalkForwardConfig(min_train_matches=300, retrain_every_days=120), wh=wh
        )
        build_all_artifacts(wh, f)
    at = _run_app()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Matches" in labels and "Divisions" in labels and "Best model vs closing price" in labels
    assert not at.error
