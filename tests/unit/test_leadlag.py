"""Cross-venue lead-lag: question parsing, snapshot alignment, Granger/xcorr verdicts, honest 'insufficient'."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_edge.odds.leadlag import (
    align_snapshots,
    cross_venue_lead_lag,
    fixture_key_from_question,
    granger_p_value,
    lead_lag_test,
    outcome_from_row,
)

pytestmark = pytest.mark.unit

KALSHI_Q = "Djurgarden wins — If Djurgarden wins the Kalmar vs Djurgarden professional Allsvenskan soccer game originally scheduled for Sep 7, 2026 af"


def test_fixture_key_parsing_kalshi_and_polymarket():
    key, _ = fixture_key_from_question(KALSHI_Q)
    assert key == "kalmar|djurgarden|2026-09-07"
    assert outcome_from_row(KALSHI_Q, "Djurgarden", key) == "away"
    assert outcome_from_row("Tie is the result — If Tie ...", "Tie", key) == "draw"
    key2, _ = fixture_key_from_question("Arsenal vs Chelsea: winner? (2026-09-13)")
    assert key2 == "arsenal|chelsea|2026-09-13"
    assert fixture_key_from_question("Will Jon Ossoff win the 2028 nomination?")[0] == ""


def _snaps(n_steps: int, lag: int = 1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 0.01, n_steps)) + 0.5
    rows = []
    ts0 = pd.Timestamp("2026-09-01 00:00")
    for t in range(n_steps):
        pa = float(np.clip(base[t], 0.05, 0.95))
        pb = float(np.clip(base[max(0, t - lag)] + rng.normal(0, 0.001), 0.05, 0.95))
        for venue, p in (("polymarket", pa), ("kalshi", pb)):
            rows += [
                {
                    "venue": venue,
                    "question": "Alpha vs Beta (2026-09-20)",
                    "outcome": "Alpha",
                    "probability": p,
                    "snapshot_ts": ts0 + pd.Timedelta(hours=t),
                },
                {
                    "venue": venue,
                    "question": "Alpha vs Beta (2026-09-20)",
                    "outcome": "Draw",
                    "probability": 0.25,
                    "snapshot_ts": ts0 + pd.Timedelta(hours=t),
                },
                {
                    "venue": venue,
                    "question": "Alpha vs Beta (2026-09-20)",
                    "outcome": "Beta",
                    "probability": 1 - p - 0.25,
                    "snapshot_ts": ts0 + pd.Timedelta(hours=t),
                },
            ]
    return pd.DataFrame(rows)


def test_align_snapshots_normalises_three_way_books():
    long = align_snapshots(_snaps(3))
    tot = long.groupby(["venue", "ts"])["prob"].sum()
    assert np.allclose(tot, 1.0) and set(long["outcome"]) == {"home", "draw", "away"}


def test_lead_lag_detects_leader_and_reports_insufficient():
    table = cross_venue_lead_lag(_snaps(120, lag=1), min_obs=24)
    home = table[(table["outcome"] == "home") & (table["fixture_key"] != "ALL")].iloc[0]
    assert home["verdict"] == "a_leads" and home["best_lag"] == 1 and home["p_a_leads_b"] < 0.05
    assert "polymarket leads" in table.iloc[-1]["verdict"]
    small = cross_venue_lead_lag(_snaps(5), min_obs=24)
    assert small.iloc[-1]["verdict"].startswith("insufficient")


def test_independent_walks_show_no_lead():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-01-01", periods=200, freq="h")
    a = pd.Series(np.cumsum(rng.normal(0, 0.01, 200)) + 0.5, index=idx)
    b = pd.Series(np.cumsum(rng.normal(0, 0.01, 200)) + 0.5, index=idx)
    r = lead_lag_test(a, b, max_lag=3, min_obs=24)
    assert r["verdict"] in {"no_lead"} and r["n_obs"] == 199
    assert granger_p_value(np.diff(b.to_numpy()), np.diff(a.to_numpy()), 3) > 0.05
