"""`pitch-edge agentic-signals` and `pitch-edge predict`: end-to-end CLI wiring, offline (tfidf
index, no network fixtures, no LLM needed for `agentic-signals` — it's fully deterministic;
`predict` falls back to its plain-text template since Ollama isn't running in tests)."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

import pitch_edge.rag.index as index_mod
from pitch_edge.cli import app
from pitch_edge.data.ingest import store_matches
from pitch_edge.data.storage import Warehouse
from pitch_edge.pipeline import load_feature_frame, persist_features

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture(autouse=True)
def _offline_agentic_signals(monkeypatch, synthetic_league_matches, tmp_path):
    monkeypatch.setattr("pitch_edge.pipeline.DEFAULT_BACKTEST_LEAGUES", ["SYN"])

    orig_init = index_mod.VectorIndex.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["force_tfidf"] = True
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(index_mod.VectorIndex, "__init__", patched_init)

    upcoming = synthetic_league_matches.sort_values("date").tail(4)[["home_team", "away_team"]].copy()
    upcoming["match_id"] = [f"future_{i}" for i in range(len(upcoming))]
    upcoming["date"] = pd.Timestamp.now().normalize() + pd.Timedelta(days=3)
    upcoming["league_code"] = "SYN"
    upcoming["lineup_source"] = "provisional"
    for col in ("elo_home", "elo_away", "elo_diff", "elo_exp_home"):
        upcoming[col] = 1500.0 if "elo_home" in col or "elo_away" in col else 0.0
    monkeypatch.setattr("pitch_edge.pipeline.upcoming_fixture_frame", lambda wh, f, leagues=None: upcoming)

    # Real backtest evidence for the SYN league so the router has something to pick. Settings.data_dir
    # is redirected to tmp_path/data by the autouse `_isolated_data_dir` fixture, so backtest_dir
    # resolves to tmp_path/data/backtest — write the evidence there, not under tmp_path directly.
    by_league_dir = tmp_path / "data" / "backtest" / "main"
    by_league_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"model": "gbdt", "league_code": "SYN", "n": 300, "log_loss": 0.6, "market_log_loss": 0.65, "edge_bits": 0.05}]
    ).to_csv(by_league_dir / "by_league.csv", index=False)


def _seed_db(tmp_path, synthetic_league_matches):
    db_path = tmp_path / "data" / "pitch_edge.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with Warehouse(db_path) as wh:
        store_matches(wh, synthetic_league_matches)
        f = load_feature_frame(wh, leagues=["SYN"], min_date="2018-01-01")
        persist_features(wh, f)


def test_agentic_signals_runs_end_to_end_with_zero_keys(tmp_path, synthetic_league_matches, monkeypatch):
    _seed_db(tmp_path, synthetic_league_matches)
    result = runner.invoke(app, ["agentic-signals", "--min-date", "2018-01-01"])
    assert result.exit_code == 0, result.output
    assert "HUMAN APPROVAL GATE" in result.output


def test_predict_falls_back_to_template_without_ollama(tmp_path, synthetic_league_matches, monkeypatch):
    _seed_db(tmp_path, synthetic_league_matches)
    monkeypatch.setattr(
        "pitch_edge.agents.explainer.get_local_llm",
        lambda model=None: (_ for _ in ()).throw(RuntimeError("ollama not running")),
    )
    result = runner.invoke(app, ["predict", "--min-date", "2018-01-01"])
    assert result.exit_code == 0, result.output
    assert "HUMAN APPROVAL GATE" in result.output
    assert "backend=template" in result.output
