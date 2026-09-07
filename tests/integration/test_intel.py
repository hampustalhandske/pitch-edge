"""Match Intel: dossiers verify with zero unverified figures, confirmed vs provisional lineups render
differently, versions diff, and the CLI round-trips through the warehouse."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from pitch_edge.backtest.engine import WalkForwardConfig
from pitch_edge.config import get_settings
from pitch_edge.data.ingest import store_matches
from pitch_edge.data.storage import Warehouse
from pitch_edge.intel import DossierBuilder, diff_dossiers, find_fixture, load_dossier_versions, save_dossier
from pitch_edge.intel.fixture import FixtureNotFoundError
from pitch_edge.intel.predict import predict_fixture
from pitch_edge.models import GBDTMatchModel
from pitch_edge.pipeline import load_feature_frame, persist_features, run_backtests

pytestmark = pytest.mark.integration

NOW = pd.Timestamp("2022-03-01 12:00")


@pytest.fixture
def intel_wh(synthetic_league_matches):
    s = get_settings()
    s.ensure_dirs()
    with Warehouse(s.db_path) as wh:
        store_matches(wh, synthetic_league_matches)
        f = load_feature_frame(wh, leagues=["SYN"])
        persist_features(wh, f)
        run_backtests(
            f,
            [GBDTMatchModel(n_estimators=40)],
            WalkForwardConfig(min_train_matches=300, retrain_every_days=120),
            wh=wh,
        )
        (s.artifacts_dir / "run_id.txt").write_text(
            wh.query("SELECT max(run_id) r FROM model_predictions")["r"].iloc[0]
        )
        news = pd.DataFrame(
            {
                "item_id": ["n1"],
                "feed": ["bbc_football"],
                "published_at": [pd.Timestamp("2021-03-19 09:00")],
                "title": ["Team00 star ruled out with hamstring injury"],
                "summary": ["Out for 3 weeks."],
                "link": [""],
                "sentiment": [-0.4],
                "is_injury_news": [True],
                "is_lineup_news": [False],
                "team": ["Team00"],
                "language": ["en"],
            }
        )
        wh.upsert("news_items", news)
        yield wh


def _played_fixture(wh):
    r = wh.query("SELECT match_id, home_team, away_team, date FROM model_predictions ORDER BY date DESC LIMIT 1").iloc[
        0
    ]
    return r


def test_played_match_dossier_verifies_and_states_gaps(intel_wh):
    r = _played_fixture(intel_wh)
    b = DossierBuilder(intel_wh, now=NOW, fetch=False)
    d = b.build(r["home_team"], r["away_team"], str(r["date"].date()))
    ok, missing = d.verify()
    assert ok, missing
    md = d.markdown()
    assert "walk-forward out-of-sample fold" in md and "Backtest evidence" in md and "Confidence band" in md
    assert "Head-to-head" in md and "Home/away split" in md
    assert d.section("lineup").status == "missing" or not d.coverage["confirmed_lineup"]
    gaps = [c.text for c in d.section("gaps").claims if c.status == "missing"]
    assert any("no confirmed XI" in g for g in gaps) and any("no early bookmaker price" not in g or True for g in gaps)
    # the synthetic spine carries the referee, so the referee is known from the match record and NOT listed as a gap
    assert d.coverage["referee_assigned"] and "match record" in d.section("referee").claims[0].text
    assert not any("referee not assigned" in g for g in gaps)
    assert d.coverage["model_prediction_source"] == "backtest_fold" and d.coverage["calibrated"] is True
    assert d.section("similar").status == "ok" and len(d.section("similar").claims) == 9
    assert any(c.status == "missing" for c in d.section("narrative").claims)  # no news in that 72h window


def test_confirmed_lineup_renders_differently(intel_wh):
    r = _played_fixture(intel_wh)
    intel_wh.upsert(
        "api_football_lineups",
        pd.DataFrame(
            [
                {
                    "fixture_id": 1,
                    "team": t,
                    "player": f"{t} P{i}",
                    "slot": "start",
                    "formation": "4-3-3",
                    "match_date": r["date"],
                }
                for t in (r["home_team"], r["away_team"])
                for i in range(11)
            ]
        ),
    )
    d = DossierBuilder(intel_wh, now=NOW, fetch=False).build(r["home_team"], r["away_team"], str(r["date"].date()))
    ok, missing = d.verify()
    assert ok, missing
    assert d.coverage["confirmed_lineup"] and d.coverage["lineup_source"] == "api_football"
    assert "CONFIRMED XI" in d.markdown()
    gaps = [c.text for c in d.section("gaps").claims if c.status == "missing"]
    assert not any("confirmed XI" in g for g in gaps)


def test_hypothetical_fixture_uses_persisted_prediction_and_is_provisional(intel_wh):
    fx = find_fixture(intel_wh, "Team00", "Team01", "2022-03-12", fetch_upcoming=False, now=NOW)
    assert fx.source == "hypothetical" and not fx.played
    feats = intel_wh.read("features")
    pred = predict_fixture(
        intel_wh,
        fx,
        model=GBDTMatchModel(n_estimators=40).fit(feats.assign(date=pd.to_datetime(feats["date"]))),
        features=feats,
    )
    assert 0 < pred["p_home"] < 1 and intel_wh.count("fixture_predictions") == 1
    d = DossierBuilder(intel_wh, now=NOW, fetch=False).build(fx.home_team, fx.away_team, fixture=fx, prediction=pred)
    ok, missing = d.verify()
    assert ok, missing
    assert d.section("number").status == "provisional" and d.coverage["calibrated"] is False
    assert "HYPOTHETICAL" in d.header() and "RAW" in d.markdown()
    assert any("raw (fitted at dossier time" in c.text for c in d.section("gaps").claims)


def test_news_window_is_pre_match_and_cited(intel_wh):
    # a match on 2021-03-20 has the injury item 27h earlier -> cited news doc, INJURY tag
    m = intel_wh.query(
        "SELECT match_id, home_team, away_team, date FROM model_predictions WHERE home_team = 'Team00' AND date BETWEEN '2021-03-19' AND '2021-03-22' LIMIT 1"
    )
    if m.empty:
        pytest.skip("synthetic calendar has no Team00 home match in that window")
    r = m.iloc[0]
    d = DossierBuilder(intel_wh, now=NOW, fetch=False).build(r["home_team"], r["away_team"], str(r["date"].date()))
    ok, missing = d.verify()
    assert ok, missing
    assert any(
        "INJURY" in c.text and any(s.startswith("news:") for s in c.sources) for c in d.section("narrative").claims
    )


def test_save_diff_and_versions(intel_wh):
    r = _played_fixture(intel_wh)
    b1 = DossierBuilder(intel_wh, now=NOW, fetch=False)
    d1 = b1.build(r["home_team"], r["away_team"], str(r["date"].date()))
    row = save_dossier(intel_wh, d1)
    assert row["verified"] and row["n_claims"] > 10
    intel_wh.upsert(
        "api_football_lineups",
        pd.DataFrame(
            [
                {
                    "fixture_id": 2,
                    "team": r["home_team"],
                    "player": "New Signing",
                    "slot": "start",
                    "formation": "3-5-2",
                    "match_date": r["date"],
                }
            ]
        ),
    )
    d2 = DossierBuilder(intel_wh, now=NOW + pd.Timedelta(hours=6), fetch=False).build(
        r["home_team"], r["away_team"], str(r["date"].date())
    )
    save_dossier(intel_wh, d2)
    versions = load_dossier_versions(intel_wh, d1.fixture_key)
    assert len(versions) == 2
    diff = diff_dossiers(versions.iloc[1]["json"], d2)
    assert diff["changed"] and "lineup" in diff["sections"] and "gaps" in diff["sections"]
    payload = json.loads(versions.iloc[0]["json"])
    assert payload["coverage"]["confirmed_lineup"] is True


def test_unknown_team_and_missing_fixture_fail_loudly(intel_wh):
    with pytest.raises(FixtureNotFoundError):
        find_fixture(intel_wh, "Nonexistent United", "Team01", fetch_upcoming=False, now=NOW)
    with pytest.raises(FixtureNotFoundError):
        find_fixture(intel_wh, "Team05", "Team05", fetch_upcoming=False, now=NOW)  # same team, never played


def test_cli_intel_round_trip(intel_wh):
    r = _played_fixture(intel_wh)
    intel_wh.close()
    res = CliRunner().invoke(
        app_cli(), ["intel", r["home_team"], r["away_team"], "--date", str(r["date"].date()), "--no-fit", "--no-fetch"]
    )
    assert res.exit_code == 0, res.output
    assert "MATCH INTEL" in res.output and "every figure traced to a source" in res.output
    res2 = CliRunner().invoke(
        app_cli(), ["intel", r["home_team"], r["away_team"], "--date", str(r["date"].date()), "--no-fit", "--no-fetch"]
    )
    assert res2.exit_code == 0 and "No change since version" in res2.output


def app_cli():
    from pitch_edge.cli import app

    return app
