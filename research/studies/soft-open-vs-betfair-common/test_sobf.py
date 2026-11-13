"""Tests for sobf: windows and exclusion, power de-vig, CLV, per_decision frame, stability, selection helper."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sobf  # noqa: E402


def _dates(*d: str) -> pd.DataFrame:
    return pd.DataFrame({"match_date": pd.to_datetime(list(d))})


def test_evaluation_frame_applies_window_and_exclusion_inclusively():
    f = _dates(
        "2025-06-30", "2025-07-01", "2026-04-16", "2026-04-17", "2026-05-01", "2026-05-04", "2026-05-05",
        "2026-10-19", "2026-10-20",
    )  # fmt: skip
    kept = sobf.evaluation_frame(f)["match_date"].dt.strftime("%Y-%m-%d").tolist()
    assert kept == ["2025-07-01", "2026-04-16", "2026-05-05", "2026-10-19"]


def test_in_window_is_inclusive():
    f = _dates("2024-07-31", "2024-08-01", "2025-06-30", "2025-07-01")
    assert len(sobf.in_window(f, "2024-08-01", "2025-06-30")) == 2


def test_season_start_boundary():
    s = sobf.season_start(pd.Series(pd.to_datetime(["2013-06-30", "2013-07-01", "2014-01-05"])))
    assert s.tolist() == [2012, 2013, 2013]


def test_devig_returns_probabilities_summing_to_one_and_nan_for_bad_rows():
    odds = np.array([[1.50, 4.20, 7.00], [2.0, 2.0, 2.0], [1.5, np.nan, 7.0], [1.0, 3.0, 5.0]])
    p = sobf.devig_power_rows(odds)
    assert np.allclose(p[0].sum(), 1.0) and np.allclose(p[1].sum(), 1.0)
    assert (p[0] < 1).all() and p[0, 0] > p[0, 2]  # probabilities, not odds
    assert np.isnan(p[2]).all() and np.isnan(p[3]).all()
    # power method removes margin: de-vigged prob below raw implied prob for the favourite
    assert p[0, 0] < 1 / 1.5


def test_devig_handles_underround_max_odds():
    p = sobf.devig_power_rows(np.array([[3.2, 3.6, 2.6]]))  # booksum < 1
    assert np.isclose(p.sum(), 1.0)


def test_clv_known_value_and_sign():
    assert np.isclose(sobf.clv_log(np.array([2.2]), np.array([0.5]))[0], np.log(1.1))
    assert sobf.clv_log(np.array([1.9]), np.array([0.5]))[0] < 0


def test_per_decision_shape_and_pooling():
    n = 150
    bets = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2025-08-01"] * n),
            "division": ["E0"] * 120 + ["X1"] * 30,
            "home_team": [f"h{i}" for i in range(n)],
            "away_team": [f"a{i}" for i in range(n)],
            "clv": np.linspace(-0.01, 0.02, n),
        }
    )
    pdf = sobf.per_decision(bets)
    assert set(pdf.columns) == {"decision_id", "block", "value", "league"}
    assert pdf["decision_id"].is_unique and set(pdf["league"]) == {"E0", "OTHER"}
    assert pdf["block"].iloc[0] == "2025-08-01"


def test_calendar_thirds_and_stability_flags():
    dates = pd.Series(pd.to_datetime(["2025-07-02", "2025-12-01", "2026-06-01", "2026-09-01"]))
    assert sobf.calendar_thirds(dates, "2025-07-01", "2026-10-19").tolist() == [1, 1, 3, 3]
    rng = np.random.default_rng(0)
    d = pd.to_datetime("2025-07-01") + pd.to_timedelta(rng.integers(0, 470, 600), unit="D")
    bets = pd.DataFrame(
        {
            "match_date": d,
            "division": rng.choice(["E0", "D1"], 600),
            "clv": rng.normal(0.01, 0.02, 600),
        }
    )
    s = sobf.stability(bets, "2025-07-01", "2026-10-19")
    assert set(s["thirds"]) == {"1", "2", "3"} and s["negative_thirds"] == 0
    assert not s["disconfirm_two_of_three_thirds_negative"]
    assert set(s["leave_one_division_out"]) == {"E0", "D1"}
    assert s["disconfirm_lodo_ci_includes_zero_for"] == []


def test_assert_whitelist_rejects_close_and_unknown():
    sobf.assert_whitelist(["a_open"], ["a_open", "b_open"])
    with pytest.raises(AssertionError):
        sobf.assert_whitelist(["a_open", "a_close"], ["a_open", "a_close"])
    with pytest.raises(AssertionError):
        sobf.assert_whitelist(["c_open"], ["a_open"])


def test_pick_applies_min_bets_and_key():
    rows = [{"n_bets": 50, "v": 9}, {"n_bets": 120, "v": 1}, {"n_bets": 200, "v": 2}]
    assert sobf.pick(rows, lambda r: (r["v"],), 100)["v"] == 1
    assert sobf.pick(rows, lambda r: (r["v"],), 1000) is None
