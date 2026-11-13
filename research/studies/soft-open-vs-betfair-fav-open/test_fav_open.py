"""H1 tests on synthetic frames: bet rule, data-error cap, benchmark CLV, whitelist, selection tie-break/fallback."""

import fav_open as H
import numpy as np
import pandas as pd
import pytest
import sobf


def _row(bf, mx, close, date="2024-09-07", div="E0", home="h", away="a", res="H"):
    d = {"division": div, "match_date": pd.Timestamp(date), "home_team": home, "away_team": away, "ft_result": res}
    for o, b, m, c in zip(sobf.OUTCOMES, bf, mx, close, strict=True):
        d[f"betfair_ex_{o}_open"], d[f"max_{o}_open"], d[f"betfair_ex_{o}_close"] = b, m, c
    return d


def _frame(rows):
    return pd.DataFrame(rows)[H.PULL_COLS]


def test_bet_rule_edge_pmin_theta_and_at_most_one_bet():
    # fair-ish favourite 1.40 at Betfair, soft max 1.50 -> clear positive edge on the favourite only
    f = _frame([_row((1.40, 4.8, 9.0), (1.50, 5.2, 9.5), (1.38, 4.9, 9.2))])
    f2, dec, pc = H.universe(f)
    assert dec["fav"].iloc[0] == 0 and dec["eligible"].iloc[0]
    p_fav = sobf.devig_power_rows(np.array([[1.40, 4.8, 9.0]]))[0, 0]
    assert np.isclose(dec["edge"].iloc[0], np.log(1.50 * p_fav))
    b = H.bets_for(f2, dec, pc, theta=0.0, p_min=0.65)
    assert len(b) == 1 and b["outcome"].iloc[0] == "home" and b["odds"].iloc[0] == 1.50
    assert len(H.bets_for(f2, dec, pc, theta=0.0, p_min=0.99)) == 0  # p_min binds
    assert len(H.bets_for(f2, dec, pc, theta=0.5, p_min=0.65)) == 0  # theta binds


def test_edge_exactly_not_above_theta_is_not_bet():
    # max equals the Betfair open price: edge = log(price * p_fav) < 0 because of the Betfair margin
    f = _frame([_row((1.40, 4.8, 9.0), (1.40, 4.8, 9.0), (1.40, 4.8, 9.0))])
    f2, dec, pc = H.universe(f)
    assert dec["edge"].iloc[0] < 0
    assert len(H.bets_for(f2, dec, pc, theta=0.0, p_min=0.65)) == 0


def test_underdog_never_bet_even_with_large_edge():
    f = _frame([_row((1.40, 4.8, 9.0), (1.40, 4.8, 12.0), (1.40, 4.8, 9.0))])
    f2, dec, pc = H.universe(f)
    b = H.bets_for(f2, dec, pc, theta=0.0, p_min=0.5)
    assert (b["outcome"] == "home").all()


def test_data_error_cap_makes_match_ineligible():
    f = _frame([_row((1.40, 4.8, 9.0), (2.20, 5.0, 9.0), (1.40, 4.8, 9.0))])  # 2.20 / 1.40 = 1.57 > 1.5
    f2, dec, pc = H.universe(f)
    assert not dec["eligible"].iloc[0]
    assert len(H.bets_for(f2, dec, pc, theta=0.0, p_min=0.5)) == 0
    at_cap = _frame([_row((2.0, 3.5, 4.0), (3.0, 3.5, 4.0), (2.0, 3.5, 4.0))])  # ratio exactly 1.5 is allowed
    assert H.universe(at_cap)[1]["eligible"].iloc[0]


def test_universe_drops_missing_close_or_open():
    rows = [
        _row((1.4, 4.8, 9.0), (1.5, 5.0, 9.5), (np.nan, 4.8, 9.0)),
        _row((1.4, 4.8, 9.0), (1.5, 5.0, 9.5), (1.4, 4.8, 9.0)),
    ]
    f2, dec, pc = H.universe(_frame(rows))
    assert len(f2) == 1 and pc.shape == (1, 3)


def test_clv_is_against_devigged_betfair_close_probability():
    f = _frame([_row((1.40, 4.8, 9.0), (1.50, 5.2, 9.5), (1.30, 5.5, 11.0))])
    f2, dec, pc = H.universe(f)
    b = H.bets_for(f2, dec, pc, theta=0.0, p_min=0.65)
    p_close = sobf.devig_power_rows(np.array([[1.30, 5.5, 11.0]]))[0, 0]
    assert np.isclose(b["clv"].iloc[0], np.log(1.50 * p_close))
    assert b["clv"].iloc[0] > 0  # the close shortened


def test_decide_rejects_close_columns_and_needs_only_open_columns():
    f = _frame([_row((1.40, 4.8, 9.0), (1.50, 5.2, 9.5), (1.38, 4.9, 9.2))])
    H.decide(f[H.OPEN_COLS])  # works with open columns alone: no close column is read
    with pytest.raises(AssertionError):
        H.decide(f[H.OPEN_COLS + ["betfair_ex_home_close"]])
    with pytest.raises(AssertionError):
        H.decide(f[H.OPEN_COLS + ["ft_result"]])


def test_candidate_grid_is_the_six_registered():
    assert [(c["theta"], c["p_min"]) for c in H.candidates()] == [
        (0.0, 0.65), (0.0, 0.70), (0.0, 0.75), (0.01, 0.65), (0.01, 0.70), (0.01, 0.75),
    ]  # fmt: skip


def _train_with(n_by_group: dict[str, int], clv_gap: dict[str, float]) -> pd.DataFrame:
    """Synthetic exploratory frame: groups of favourites whose max price sets the CLV and the edge."""
    rows, k = [], 0
    for g, n in n_by_group.items():
        for _ in range(n):
            k += 1
            day = pd.Timestamp("2024-09-01") + pd.Timedelta(days=k % 200)
            mx = 1.40 * clv_gap[g]
            rows.append(_row((1.40, 4.8, 9.0), (mx, 5.0, 9.0), (1.40, 4.8, 9.0), date=day, home=f"h{k}", away=f"a{k}"))
    return _frame(rows)


def test_select_picks_highest_mean_clv_among_eligible():
    # all matches are the same favourite (p_fav ~0.69); the higher max price gives a higher edge and CLV
    tr = _train_with({"x": 150}, {"x": 1.07})
    sel, rows = H.select(tr)
    assert len(rows) == 6 and not sel["fallback"]
    best = max(r["mean_clv"] for r in rows if r["eligible"])
    chosen = [r for r in rows if (r["theta"], r["p_min"]) == (sel["theta"], sel["p_min"])][0]
    assert chosen["eligible"] and np.isclose(chosen["mean_clv"], best)


def test_select_tie_break_prefers_lower_theta_then_lower_p_min():
    # identical bet sets (same favourite, large edge) -> every eligible candidate ties on mean CLV
    tr = _train_with({"x": 150}, {"x": 1.10})
    sel, rows = H.select(tr)
    tied = [r for r in rows if r["eligible"]]
    assert len({round(r["mean_clv"], 12) for r in tied}) == 1
    assert (sel["theta"], sel["p_min"]) == (
        min(r["theta"] for r in tied),
        min(r["p_min"] for r in tied if r["theta"] == 0.0),
    )


def test_select_key_orders_theta_before_p_min():
    a = {"mean_clv": 0.01, "theta": 0.01, "p_min": 0.65}
    b = {"mean_clv": 0.01, "theta": 0.0, "p_min": 0.75}
    c = {"mean_clv": 0.01, "theta": 0.0, "p_min": 0.65}
    assert sorted([a, b, c], key=H.select_key) == [c, b, a]


def test_select_fallback_when_no_candidate_has_100_bets():
    tr = _train_with({"x": 40}, {"x": 1.08})
    sel, rows = H.select(tr)
    assert sel["fallback"] and (sel["theta"], sel["p_min"]) == (0.0, 0.70)
    assert not any(r["eligible"] for r in rows)


def test_select_ignores_rows_outside_the_exploratory_window():
    tr = _train_with({"x": 150}, {"x": 1.07})
    late = tr.copy()
    late["match_date"] = pd.Timestamp("2026-01-10")  # evaluation-window rows must never be read
    late["home_team"] = "late" + late["home_team"]
    late["betfair_ex_home_close"] = 1.0 + 1e-3  # would wreck CLV if used
    sel_a, _ = H.select(tr)
    sel_b, rows_b = H.select(pd.concat([tr, late], ignore_index=True))
    assert sel_a["theta"] == sel_b["theta"] and sel_a["p_min"] == sel_b["p_min"]
    assert max(r["n_bets"] for r in rows_b) <= 150
