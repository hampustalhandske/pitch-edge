"""H4 tests on synthetic OddsPapi paths: path holding, invalid quotes, liveness, size, float tolerance, the PIT
assertion and no lookahead, one bet per match, the Pinnacle close screen and its power de-vig, frozen-decision ordering,
commission, selection, tournament ids, and pull.py / run.py end to end on a fake lake client (empty chunks, lake builds,
fetch_status, sha verification, the order of count, selection and score)."""

import json
import sys
from pathlib import Path

import exchange_lag as H
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pull as P  # noqa: E402
import run as R  # noqa: E402

MIN = H.MINUTE_NS
K = H.day_ns("2026-08-01") + 15 * 60 * MIN  # kickoff 15:00 UTC


def rows(spec, k=K, fid="f1"):
    """Path rows of fixture `fid` (kickoff `k`) from (book, outcome, minutes before kickoff, price[, active[,
    stake_limit]]) tuples."""
    out = []
    for r in spec:
        book, o, m, price = r[:4]
        active = r[4] if len(r) > 4 else True
        size = r[5] if len(r) > 5 else 100.0
        out.append(
            {"fixture_id": fid, "bookmaker": book, "outcome": o, "observed_ns": k - int(m * MIN), "price": price,
             "active": active, "stake_limit": size}
        )  # fmt: skip
    return pd.DataFrame(out, columns=H.PATH_COLS)


def base(home_pin_now=1.90, home_bf=2.10, extra=(), home_pin_ref=2.00, k=K, fid="f1"):
    """Pinnacle home `home_pin_ref` from k-180 to k-30, then `home_pin_now`; Betfair home flat; a Betfair draw tick at
    k-40."""
    spec = [
        ("pinnacle", "home", 180, home_pin_ref), ("pinnacle", "draw", 180, 3.50), ("pinnacle", "away", 180, 4.00),
        ("pinnacle", "home", 30, home_pin_now),
        ("betfair-ex", "home", 180, home_bf), ("betfair-ex", "draw", 180, 3.60), ("betfair-ex", "away", 180, 4.20),
        ("betfair-ex", "draw", 40, 3.65),
    ]  # fmt: skip
    return rows(spec + list(extra), k=k, fid=fid)


def points(df, kickoff=K):
    return H.decision_points("f1", kickoff, H.build_paths(df))


def bets(df, s=0.03, b=0.0):
    return H.decide(points(df), s, b)[0]


# -------------------------------------------------------------------------------- decision rule


def test_basic_decision_holds_stale_rows_and_takes_the_betfair_price_at_t():
    b = bets(base())
    assert len(b) == 1
    r = b.iloc[0]
    assert r["outcome"] == "home" and r["t_ns"] == K - 30 * MIN and r["odds"] == 2.10
    assert r["pin_ref"] == 2.00 and r["bf_ref"] == 2.10  # rows from k-180 hold to t - 60 (path holding)
    assert r["shortening"] == pytest.approx(0.05) and r["bf_change"] == 0.0


def test_no_row_stamped_by_t_minus_60_means_no_decision():
    df = base()
    df.loc[(df["bookmaker"] == "pinnacle") & (df["outcome"] == "home") & (df["observed_ns"] == K - 180 * MIN),
           "observed_ns"] = K - 50 * MIN  # fmt: skip
    assert bets(df).empty  # pinnacle home has no row at or before t - 60 for any t in the hour


def test_threshold_is_inclusive_at_exactly_s():
    assert len(bets(base(home_pin_now=1.94), s=0.03)) == 1  # 1 - 1.94 / 2.00 = 0.03
    assert bets(base(home_pin_now=1.95), s=0.03).empty


def test_betfair_move_beyond_b_blocks_and_b_is_relative_absolute():
    moved = base(extra=[("betfair-ex", "home", 50, 2.12)])  # +0.95% since t - 60
    assert bets(moved, b=0.0).empty
    assert len(bets(moved, b=0.01)) == 1
    down = base(extra=[("betfair-ex", "home", 50, 2.07)])  # -1.4%: |change| > 0.01
    assert bets(down, b=0.01).empty


def test_liveness_requires_a_betfair_row_in_the_last_hour():
    df = base()
    df = df[~((df["bookmaker"] == "betfair-ex") & (df["observed_ns"] == K - 40 * MIN))]
    assert bets(df).empty  # betfair silent for 180 min: not live
    late = base().copy()
    late.loc[late["observed_ns"] == K - 40 * MIN, "observed_ns"] = K - 20 * MIN  # tick after k-30
    b = bets(late)
    assert len(b) == 1 and b.iloc[0]["t_ns"] == K - 20 * MIN  # first live t is the tick itself


def test_liveness_ignores_invalid_price_rows():
    df = base()
    df.loc[df["observed_ns"] == K - 40 * MIN, "price"] = 0.0
    assert bets(df).empty


def test_inactive_betfair_quote_blocks():
    df = base(extra=[("betfair-ex", "home", 70, 2.10, False)])
    assert bets(df).empty
    reopened = base(extra=[("betfair-ex", "home", 70, 2.10, False), ("betfair-ex", "home", 10, 2.10, True)])
    b = bets(reopened)
    assert len(b) == 1 and b.iloc[0]["t_ns"] == K - 10 * MIN


@pytest.mark.parametrize(("size", "n"), [(24.99, 0), (25.0, 1), (500.0, 1), (np.nan, 0)])
def test_offered_size_check_uses_stake_limit_in_force_at_t(size, n):
    df = base()
    m = (df["bookmaker"] == "betfair-ex") & (df["outcome"] == "home")
    df.loc[m, "stake_limit"] = size
    assert len(bets(df)) == n


def test_invalid_price_row_means_no_valid_quote_until_the_next_valid_row():
    assert H.MIN_VALID_PRICE == 1.0
    zero = base(extra=[("betfair-ex", "home", 35, 0.0)])  # in force at t = k-30
    assert bets(zero).empty
    restored = base(extra=[("betfair-ex", "home", 35, 0.0), ("betfair-ex", "home", 25, 2.10)])
    b = bets(restored)
    assert len(b) == 1 and b.iloc[0]["t_ns"] == K - 25 * MIN and b.iloc[0]["odds"] == 2.10
    one = base(extra=[("betfair-ex", "home", 35, 1.0)])  # exactly 1.0 is not a quote either
    assert bets(one).empty


def test_invalid_reference_price_blocks():
    # pinnacle home 0.0 in force at t - 60 for t in [k-30, k-20): first valid reference at t = k-20 (ref row k-80)
    df = base(extra=[("pinnacle", "home", 95, 0.0), ("pinnacle", "home", 80, 2.00), ("pinnacle", "home", 20, 1.90)])
    b = bets(df)
    assert len(b) == 1 and b.iloc[0]["t_ns"] == K - 20 * MIN and b.iloc[0]["pin_ref"] == 2.00


def test_pinnacle_move_before_the_hour_does_not_open_a_decision_after_it():
    df = base()
    df.loc[df["observed_ns"] == K - 30 * MIN, "observed_ns"] = K - 70 * MIN  # shortened 70 min before
    b = bets(df)
    # at t = k-40 (betfair tick) the reference at k-100 is 2.00, the price 1.90: still a decision
    assert len(b) == 1 and b.iloc[0]["t_ns"] == K - 40 * MIN


def test_decisions_only_before_kickoff_and_within_the_hour():
    df = base()
    df.loc[df["observed_ns"] == K - 30 * MIN, "observed_ns"] = K  # the move is stamped at kickoff
    assert bets(df).empty


def test_outcome_codes_101_102_103_map_to_home_draw_away():
    df = base()
    df["outcome"] = df["outcome"].map({"home": "101", "draw": "102", "away": "103"})
    assert bets(df).iloc[0]["outcome"] == "home"
    with pytest.raises(ValueError):
        H.build_paths(df.assign(outcome="1"))


def test_float_tolerance_on_s_and_b_is_exercised(monkeypatch):
    df = base(home_pin_ref=2.50, home_pin_now=2.45)  # exactly 2%, but 1 - 2.45 / 2.50 = 0.0199999... in float
    assert 1 - 2.45 / 2.50 < 0.02
    assert len(bets(df, s=0.02)) == 1
    bf = base(home_bf=1.01, extra=[("betfair-ex", "home", 50, 1.0201)])  # exactly +1%, float 0.010000000000000009
    assert abs(1.0201 / 1.01 - 1) > 0.01
    assert len(bets(bf, b=0.01)) == 1
    monkeypatch.setattr(H, "EPS", 0.0)
    assert bets(df, s=0.02).empty and bets(bf, b=0.01).empty


def test_liveness_window_is_inclusive_at_exactly_t_minus_60():
    df = base()
    df.loc[df["observed_ns"] == K - 40 * MIN, "observed_ns"] = K - 90 * MIN  # decision t = k-30, t - 60 = k-90
    b = bets(df)
    assert len(b) == 1 and b.iloc[0]["t_ns"] == K - 30 * MIN and b.iloc[0]["live_ts"] == K - 90 * MIN
    df.loc[df["observed_ns"] == K - 90 * MIN, "observed_ns"] = K - 90 * MIN - 1
    assert bets(df).empty


def test_candidate_window_starts_at_exactly_kickoff_minus_60():
    df = base(extra=[("betfair-ex", "away", 70, 4.20)])  # a betfair tick so t = k-60 is live
    df.loc[df["observed_ns"] == K - 30 * MIN, "observed_ns"] = K - 60 * MIN
    b = bets(df)
    assert len(b) == 1 and b.iloc[0]["t_ns"] == K - 60 * MIN
    df.loc[df["observed_ns"] == K - 60 * MIN, "observed_ns"] = K - 60 * MIN - 1  # just outside: next stamp k-40
    b = bets(df)
    assert len(b) == 1 and b.iloc[0]["t_ns"] == K - 40 * MIN


def test_no_later_retry_for_an_outcome_whose_decision_failed_size():
    df = base()
    df.loc[(df["bookmaker"] == "betfair-ex") & (df["outcome"] == "home"), "stake_limit"] = 5.0
    df = pd.concat([df, rows([("betfair-ex", "home", 10, 2.10, True, 500.0)])], ignore_index=True)
    later = H.decide(points(df), 0.03, 0.0)
    assert later[0].empty and later[1]["outcome_decisions_size_fail"] == 1
    assert len(points(df)[lambda p: p["t_ns"] == K - 10 * MIN]) == 1  # k-10 would qualify on its own


# ---------------------------------------------------------------------------- point in time


def test_assert_pit_raises_on_a_point_stamped_after_t():
    p = points(base())
    assert len(p)
    H.assert_pit(p)
    for col in ("bf_ts_t", "pin_ts_t", "live_ts", "pin_ts_ref", "bf_ts_ref"):
        bad = p.copy()
        bad[col] = bad["t_ns"] - H.LOOKBACK_NS + 1 if col.endswith("_ref") else bad["t_ns"] + 1
        with pytest.raises(AssertionError, match="point-in-time"):
            H.assert_pit(bad)
    early = p.copy()
    early["live_ts"] = early["t_ns"] - H.LOOKBACK_NS - 1
    with pytest.raises(AssertionError, match="point-in-time"):
        H.assert_pit(early)


def test_no_lookahead_future_rows_do_not_change_a_decision():
    rng = np.random.default_rng(3)
    produced = 0
    for _ in range(40):
        spec = []
        for book in ("pinnacle", "betfair-ex"):
            for o, p0 in zip(H.OUTCOMES, (2.0, 3.4, 4.0), strict=True):
                spec.append((book, o, 200, p0))
                mins = np.sort(rng.choice(np.arange(1, 190), size=8, replace=False))[::-1]
                p = p0
                for m in mins:
                    p = max(1.01, p * float(np.exp(rng.normal(0, 0.03))))
                    spec.append((book, o, int(m), round(p, 2), bool(rng.random() > 0.05), float(rng.integers(0, 80))))
        df = rows(spec)
        for s, b in H.CANDIDATES:
            full = H.decide(points(df), s, b)[0]
            produced += len(full)
            for _, r in full.iterrows():
                cut = df[df["observed_ns"] <= r["t_ns"]]
                again = H.decide(points(cut), s, b)[0]
                assert len(again) == 1 and again.iloc[0]["t_ns"] == r["t_ns"] and again.iloc[0]["j"] == r["j"]
                assert again.iloc[0]["odds"] == r["odds"]
    assert produced >= 20  # the property was checked on real decisions, not vacuously


# ---------------------------------------------------------------------------- one bet per match


def test_same_t_tie_goes_to_the_largest_pinnacle_shortening():
    df = base(extra=[("pinnacle", "away", 30, 3.70)])  # away shortens 7.5% at the same t as home's 5%
    b = bets(df)
    assert len(b) == 1 and b.iloc[0]["outcome"] == "away"


def test_earliest_decision_wins_over_a_larger_later_shortening():
    df = base(extra=[("pinnacle", "away", 20, 3.40)])  # away 15% but at k-20, home 5% at k-30
    b = bets(df)
    assert len(b) == 1 and b.iloc[0]["outcome"] == "home"


def test_an_outcome_failing_size_leaves_the_next_outcome_with_a_bet():
    df = base(extra=[("pinnacle", "away", 20, 3.40)])
    df.loc[(df["bookmaker"] == "betfair-ex") & (df["outcome"] == "home"), "stake_limit"] = 5.0
    b, funnel = H.decide(points(df), 0.03, 0.0)
    assert len(b) == 1 and b.iloc[0]["outcome"] == "away"
    assert funnel["outcome_decisions"] == 2 and funnel["outcome_decisions_size_fail"] == 1


# --------------------------------------------------------------------------------- close screen


def close_paths(spec):
    return H.build_paths(rows(spec)[lambda d: d["observed_ns"] <= K])


def _power_devig(odds):
    """Independent power de-vig: bisection on k so that sum((1/o)^k) = 1."""
    q = 1.0 / np.asarray(odds)
    lo, hi = 1.0, 3.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (q**mid).sum() > 1 else (lo, mid)
    return q ** ((lo + hi) / 2)


def test_close_screen_valid_and_power_devigged():
    odds = (1.9, 3.6, 4.4)
    c = H.close_screen(K, close_paths([("pinnacle", o, 10, p) for o, p in zip(H.OUTCOMES, odds, strict=True)]))
    assert c["close_valid"] and c["reason"] == ""
    want = _power_devig(odds)
    got = np.array([c[f"p_{o}"] for o in H.OUTCOMES])
    assert got == pytest.approx(want, abs=1e-9)
    proportional = (1 / np.array(odds)) / (1 / np.array(odds)).sum()
    assert abs(got[0] - proportional[0]) > 1e-3  # the check would catch a proportional de-vig


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        ([("pinnacle", "home", 10, 1.9), ("pinnacle", "draw", 10, 3.6)], "missing_outcome"),
        (
            [("pinnacle", "home", 10, 1.9), ("pinnacle", "draw", 10, 3.6), ("pinnacle", "away", 10, 4.4, False)],
            "inactive",
        ),
        (
            [("pinnacle", "home", 10, 0.0), ("pinnacle", "draw", 10, 3.6), ("pinnacle", "away", 10, 4.4)],
            "invalid_price",
        ),
        ([("pinnacle", o, 16, p) for o, p in zip(H.OUTCOMES, (1.9, 3.6, 4.4), strict=True)], "stale"),
    ],
)
def test_close_screen_failures(spec, reason):
    c = H.close_screen(K, close_paths(spec))
    assert not c["close_valid"] and c["reason"] == reason


def test_close_screen_age_is_the_most_recent_of_the_three_and_kickoff_rows_count():
    spec = [("pinnacle", "home", 0, 1.9), ("pinnacle", "draw", 300, 3.6), ("pinnacle", "away", 300, 4.4)]
    c = H.close_screen(K, close_paths(spec))
    assert c["close_valid"] and c["stale_any_outcome"]  # strict per-outcome reading kept as a diagnostic
    assert H.close_screen(K, {})["reason"] == "no_rows"


def test_close_screen_reads_pinnacle_only():
    with pytest.raises(AssertionError, match="pinnacle only"):
        H.close_screen(K, H.build_paths(base()))


def test_close_needs_frozen_decisions_and_detects_later_edits():
    b = bets(base())
    with pytest.raises(AssertionError, match="frozen"):
        H.close_table(b, [])  # type: ignore[arg-type]
    fz = H.freeze(b)
    pin = pd.concat([base(), rows([("pinnacle", "draw", 5, 3.6)])])[lambda d: d["bookmaker"] == "pinnacle"]
    closes = H.close_table(fz, [("f1", K, pin)])
    kept, drops = H.attach_close(fz, closes)
    assert len(kept) == 1 and drops == {}
    fz.frame.loc[0, "odds"] = 9.0
    with pytest.raises(AssertionError, match="changed after"):
        H.attach_close(fz, closes)


def test_failing_close_drops_the_bet_and_counts_it():
    fz = H.freeze(bets(base()))
    stale = rows([("pinnacle", o, 120, p) for o, p in zip(H.OUTCOMES, (1.9, 3.6, 4.4), strict=True)])
    kept, drops = H.attach_close(fz, H.close_table(fz, [("f1", K, stale)]))
    assert kept.empty and drops == {"stale": 1}
    kept, drops = H.attach_close(fz, H.close_table(fz, []))
    assert kept.empty and drops == {"no_rows": 1}


# ------------------------------------------------------------------------------- CLV, selection


def test_commission_and_clv():
    assert H.effective_odds(np.array([2.0]), 0.02)[0] == pytest.approx(1.98)
    assert H.effective_odds(np.array([3.0]), 0.05)[0] == pytest.approx(2.90)
    assert H.clv_log(np.array([2.0]), np.array([0.5]), 0.0)[0] == pytest.approx(0.0)
    assert H.clv_log(np.array([2.0]), np.array([0.5]), 0.02)[0] == pytest.approx(np.log(0.99))
    fz = H.freeze(bets(base()))
    pin = rows([("pinnacle", o, 5, p) for o, p in zip(H.OUTCOMES, (1.9, 3.6, 4.4), strict=True)])
    kept, _ = H.attach_close(fz, H.close_table(fz, [("f1", K, pin)]))
    r = kept.iloc[0]
    assert r["clv"] == pytest.approx(np.log((1 + 1.10 * 0.98) * r["p_home"]))
    assert r["clv_gross"] > r["clv"] > r["clv_c05"]


def _cands(vals):
    return [{"s": s, "b": b, "n_bets": n, "mean_clv": m} for (s, b), (n, m) in zip(H.CANDIDATES, vals, strict=True)]


def test_selection_best_mean_among_eligible():
    sel = H.select(_cands([(150, 0.01), (150, 0.02), (99, 0.09), (120, 0.015), (100, 0.005), (50, 0.2)]))
    assert (sel["s"], sel["b"], sel["fallback"]) == (0.02, 0.01, False)


def test_selection_tie_break_higher_s_then_lower_b():
    sel = H.select(_cands([(150, 0.01)] * 6))
    assert (sel["s"], sel["b"]) == (0.05, 0.0)


def test_selection_fallback_when_no_candidate_has_100_bets():
    sel = H.select(_cands([(99, 0.5)] * 6))
    assert (sel["s"], sel["b"], sel["fallback"]) == (0.03, 0.0, True)


def test_tournament_is_the_raw_id_and_leave_one_out_is_not_pooled():
    fx = pd.DataFrame({"fixture_id": ["a", "b", "c", "d"], "tournament_id": [7, 8, 8, 9],
                       "tournament_name": ["Championship", "Championship", "Championship", "Premiership"],
                       "division": ["E1", "SC1", "SC1", "SC0"]})  # fmt: skip
    tt = H.tournament_table(fx)
    days = [H.day_ns(d) for d in ("2026-06-10", "2026-07-20", "2026-08-30", "2026-10-01")]
    bets_ = pd.DataFrame({"fixture_id": ["a", "b", "c", "d"], "kickoff_ns": days, "clv": [0.01, -0.02, 0.03, 0.0]})
    sc = H.scored_frame(bets_, tt)
    assert list(sc["division"]) == ["7", "8", "8", "9"]
    assert sorted(H.per_decision(sc)["league"].unique()) == ["7", "8", "9"]  # n < 100 each, still not pooled
    st = H.stability(sc, *H.EVALUATION)
    assert set(st["per_tournament"]) == set(st["leave_one_tournament_out"]) == {"7", "8", "9"}
    assert st["per_tournament"]["7"]["name"] == st["per_tournament"]["8"]["name"] == "Championship"
    assert (st["per_tournament"]["7"]["division"], st["per_tournament"]["8"]["division"]) == ("E1", "SC1")
    assert st["leave_one_tournament_out"]["8"]["n"] == 2
    assert set(st["pooled_diagnostic"]) == {"OTHER"}  # the STANDARDS pooled table, diagnostic only


# ------------------------------------------------------------------------------- windows, outages


def test_chunks_cover_the_kickoff_range_contiguously_and_never_straddle_the_windows():
    c = H.chunks()
    assert c[0][0] == "2026-01-13" and c[-1][1] == "2026-10-20" and len(c) == 41
    assert all(a[1] == b[0] for a, b in zip(c, c[1:], strict=False))
    ex, ev = H.chunks("exploratory"), H.chunks("evaluation")
    assert ex + ev == c and ex[-1][1] == "2026-06-01" and ev[0][0] == "2026-06-01"


def test_window_of_boundaries():
    ks = [H.day_ns(d) + off for d, off in (("2026-05-31", 23 * 60 * MIN), ("2026-06-01", 0), ("2026-10-19", 1),
                                            ("2026-10-20", 0), ("2026-01-12", 0))]  # fmt: skip
    assert list(H.window_of(np.array(ks))) == ["exploratory", "evaluation", "evaluation", "", ""]


def test_outage_days_lists_silent_days_per_book():
    obs = np.array([H.day_ns("2026-06-01"), H.day_ns("2026-06-03")])
    counts = H.daily_counts(np.r_[obs, obs[:1]], np.array(["pinnacle", "pinnacle", "betfair-ex"]))
    out = H.outage_days(counts, pd.Series(["2026-06-02"]), "2026-06-01", "2026-06-03")
    assert {(o["day"], o["bookmaker"]) for o in out} == {
        ("2026-06-02", "pinnacle"), ("2026-06-02", "betfair-ex"), ("2026-06-03", "betfair-ex")
    }  # fmt: skip
    assert next(o for o in out if o["day"] == "2026-06-02")["n_kickoffs"] == 1


# ------------------------------------------------------------------ pull.py and run.py on a fake lake

CLOSE_A = (1.9, 3.6, 4.4)
CLOSE_B = (1.94, 5.0, 7.0)  # home unchanged, draw and away lengthen: a higher fair home probability, no new shortening


def _fixture(fid, k, kind, tournament):
    """Rows of one synthetic fixture. A: Pinnacle home -5% at k-30, Betfair flat (a bet for every candidate).
    B: Pinnacle home -3% at k-30, Betfair home +0.48% at k-50 (a bet only with b = 0.01 and s <= 0.03)."""
    if kind == "A":
        d, close = base(k=k, fid=fid), CLOSE_A
    else:
        d, close = base(home_pin_now=1.94, extra=[("betfair-ex", "home", 50, 2.11)], k=k, fid=fid), CLOSE_B
    c = rows([("pinnacle", o, 5, p) for o, p in zip(H.OUTCOMES, close, strict=True)], k=k, fid=fid)
    fx = {"fixture_id": fid, "tournament_id": tournament, "tournament_name": "Championship", "division": f"D{tournament}",
          "season_id": 1, "kickoff_utc": pd.Timestamp(k, unit="ns", tz="UTC"), "home_team": f"h{fid}",
          "away_team": f"a{fid}", "status_name": "Finished"}  # fmt: skip
    return pd.concat([d, c], ignore_index=True).assign(kickoff_ns=k), fx


def _world(shift_kickoff_of=None, unread=None):
    """Exploratory: 120 A and 40 B fixtures over March 2026; evaluation: 4 A fixtures on 2026-08-01 (f0 also has a
    0.0 betfair row). No rows anywhere else, so most chunks come back empty from the lake."""
    frames, fixtures = [], []
    for i in range(160):
        k = H.day_ns("2026-03-02") + (i % 20) * 24 * 60 * MIN + 15 * 60 * MIN
        d, f = _fixture(f"x{i}", k, "A" if i < 120 else "B", 11 + i % 2)
        frames.append(d)
        fixtures.append(f)
    for i in range(4):
        d, f = _fixture(f"f{i}", K, "A", 21)
        if i == 0:
            d = pd.concat([d, rows([("betfair-ex", "away", 100, 0.0)], fid="f0").assign(kickoff_ns=K)])
        frames.append(d)
        fixtures.append(f)
    paths = pd.concat(frames, ignore_index=True)
    fx = pd.DataFrame(fixtures)
    if shift_kickoff_of:
        fx.loc[fx["fixture_id"] == shift_kickoff_of, "kickoff_utc"] += pd.Timedelta(hours=1)
    st = pd.DataFrame({"fixture_id": fx["fixture_id"], "status": "ok"})
    if unread:
        st = st[st["fixture_id"] != unread]
    return paths, fx, st


class FakeLake:
    """Answers the pull's SQL like the lake client: Arrow batches (none for an empty result) and a `version`."""

    def __init__(self, paths, fx, st, versions=("build1",)):
        self.versions, self.queries, self.version = list(versions), 0, None
        ts = pa.timestamp("us", tz="UTC")
        self.tables = {
            H.fixtures_sql(): pa.Table.from_pandas(fx, preserve_index=False),
            H.fetch_status_sql(): pa.table({"fixture_id": st["fixture_id"].tolist(), "status": st["status"].tolist(),
                                            "pinnacle_rows": [1] * len(st), "betfair_ex_rows": [1] * len(st),
                                            "first_observed_at": pa.array([0] * len(st), ts),
                                            "last_observed_at": pa.array([0] * len(st), ts),
                                            "fetched_at": pa.array([0] * len(st), ts)}),
        }  # fmt: skip
        for lo, hi in H.chunks():
            sub = paths[(paths["kickoff_ns"] >= H.day_ns(lo)) & (paths["kickoff_ns"] < H.day_ns(hi))]
            sub = sub.sort_values(["fixture_id", "bookmaker", "outcome", "observed_ns"])
            self.tables[H.paths_sql(lo, hi)] = pa.table(
                {
                    "fixture_id": pa.array(sub["fixture_id"].astype(str).tolist(), pa.string()),
                    "bookmaker": pa.array(sub["bookmaker"].tolist(), pa.string()),
                    "outcome": pa.array(sub["outcome"].tolist(), pa.string()),
                    "observed_at": pa.array((sub["observed_ns"].to_numpy() // 1000).tolist(), ts),
                    "price": pa.array(sub["price"].tolist(), pa.float64()),
                    "stake_limit": pa.array(sub["stake_limit"].tolist(), pa.float64()),
                    "active": pa.array(sub["active"].tolist(), pa.bool_()),
                    "kickoff_utc": pa.array((sub["kickoff_ns"].to_numpy() // 1000).tolist(), ts),
                    "tournament_id": pa.array([1] * len(sub), pa.int64()),
                }
            )

    def sql_batches(self, sql):
        self.version = self.versions[min(self.queries, len(self.versions) - 1)]
        self.queries += 1
        yield from self.tables[sql].to_batches()  # an empty result yields no batch at all


def _pull(out, **kw):
    lake_kw = {k: kw.pop(k) for k in list(kw) if k == "versions"}
    lake = FakeLake(*_world(**kw), **lake_kw)
    return P.pull(lake, out), lake


class _Ledger:
    def __init__(self, n_amend, n_trials=0):
        self.n, self.t = n_amend, n_trials

    def reviews(self, hypothesis_id=None):
        return pd.DataFrame({"kind": ["amendment"] * self.n})

    def trials(self, hypothesis_id=None):
        return pd.DataFrame({"trial_id": range(self.t)})


def test_read_paths_on_a_zero_row_column_less_chunk_gives_a_typed_empty_frame(tmp_path):
    p = tmp_path / "empty.parquet"
    pq.write_table(pa.table({}), p)
    df = H.read_paths(p, H.PATH_COLS)
    assert df.empty and list(df.columns) == H.PATH_COLS + ["kickoff_ns"]
    assert df["observed_ns"].dtype == np.int64 and df["price"].dtype == np.float64 and df["active"].dtype == bool
    assert H.conform(pa.table({}), H.PATH_SCHEMA).schema == H.PATH_SCHEMA


def test_pull_writes_schema_typed_empty_chunks_and_a_verified_snapshot(tmp_path):
    out = tmp_path / "out"
    snap, lake = _pull(out)
    assert snap["complete"] and snap["fetch_status_unread"] == 0 and snap["fetch_status_counts"] == {"ok": 164}
    assert snap["rows_by_window_book"]["evaluation|betfair-ex"]["invalid_price"] == 1
    empty = out / "paths" / f"{H.chunk_stem('2026-06-01', '2026-06-08')}.parquet"
    assert pq.read_schema(empty).names == H.PATH_SCHEMA.names and pq.ParquetFile(empty).metadata.num_rows == 0
    assert H.read_paths(empty, H.PATH_COLS).empty
    label = H.snapshot_label(out)
    again, lake2 = _pull(out)
    assert lake2.queries == 0 and H.snapshot_label(out) == label  # a finished pull is final


def test_pull_refuses_an_unversioned_lake_and_a_build_change(tmp_path):
    with pytest.raises(SystemExit, match="X-Lakehouse-Version"):
        _pull(tmp_path / "a", versions=(None,))
    with pytest.raises(SystemExit, match="changed mid-pull"):
        _pull(tmp_path / "b", versions=("b1", "b1", "b2"))
    assert not (tmp_path / "b" / "snapshot.json").exists()


def test_pull_refuses_fixtures_the_lake_has_not_read(tmp_path):
    with pytest.raises(SystemExit, match="not read by the lake"):
        _pull(tmp_path / "out", unread="x5")
    assert not (tmp_path / "out" / "snapshot.json").exists()


@pytest.mark.parametrize(
    ("edit", "msg"),
    [
        ({"complete": False}, "incomplete"),
        ({"consistent_lake_version": False}, "mixes lake builds"),
        ({"lake_version": "unversioned"}, "no lake build"),
        ({"fetch_status_unread": 3}, "not read by the lake"),
    ],
)
def test_snapshot_flags_are_refused(tmp_path, edit, msg):
    out = tmp_path / "out"
    snap, _ = _pull(out)
    (out / "snapshot.json").write_text(json.dumps({**snap, **edit}))
    with pytest.raises(SystemExit, match=msg):
        H.snapshot_label(out)


def test_a_changed_chunk_file_is_refused(tmp_path):
    out = tmp_path / "out"
    _pull(out)
    f = H.chunk_files(out, "evaluation")[0]
    pq.write_table(H.PATH_SCHEMA.empty_table().replace_schema_metadata({"x": "1"}), f)
    with pytest.raises(SystemExit, match="changed since the pull"):
        H.snapshot_label(out)


def test_kickoff_mismatch_between_rows_and_fixtures_aborts(tmp_path):
    out = tmp_path / "out"
    _pull(out, shift_kickoff_of="f2")
    with pytest.raises(SystemExit, match="kickoff_utc"):
        R.do_count(out, tmp_path / "trials.csv")


def test_run_count_then_select_in_order_with_real_train_bets(tmp_path, monkeypatch):
    out, trials = tmp_path / "out", tmp_path / "trials.csv"
    _pull(out)
    full = _Ledger(R.REQUIRED_AMENDMENTS)
    with pytest.raises(SystemExit, match="count-test"):
        R.do_select(out, trials, ledger=full)
    called = []
    monkeypatch.setattr(H, "close_screen", lambda *a, **k: called.append(1))
    rep = R.do_count(out, trials)
    assert not called  # the count never reaches the close screen
    ev, ex = rep["windows"]["evaluation"], rep["windows"]["exploratory"]
    assert ev["universe"]["both"] == 4 and ev["invalid_price_rows"]["betfair-ex"] == 1
    assert all(ev["candidates"][R._cand(s, b)]["bets"] == 4 for s, b in H.CANDIDATES)
    assert ex["candidates"][R._cand(0.03, 0.01)]["bets"] == 160 and ex["candidates"][R._cand(0.05, 0.01)]["bets"] == 120
    assert {o["day"] for o in rep["outages"]["evaluation"]} >= {"2026-06-01"}
    with pytest.raises(SystemExit, match="already recorded"):
        R.do_count(out, trials)
    monkeypatch.undo()
    with pytest.raises(SystemExit, match="amendments"):
        R.do_select(out, trials, ledger=_Ledger(R.REQUIRED_AMENDMENTS - 1))
    read = []
    real = H.read_paths
    monkeypatch.setattr(H, "read_paths", lambda path, *a, **k: read.append(path.name) or real(path, *a, **k))
    sel = R.do_select(out, trials, ledger=full)
    monkeypatch.undo()
    assert read and set(read) <= {p.name for p in H.chunk_files(out, "exploratory")}  # train chunks only
    assert not sel["fallback"] and (sel["s"], sel["b"]) == (0.03, 0.01)  # best mean; 0.02 ties, higher s wins
    by = {(c["s"], c["b"]): c for c in sel["candidates"]}
    assert by[(0.02, 0.01)]["n_bets"] == by[(0.03, 0.01)]["n_bets"] == 160 and by[(0.05, 0.0)]["n_bets"] == 120
    assert by[(0.03, 0.01)]["mean_clv"] > by[(0.03, 0.0)]["mean_clv"]
    t = pd.read_csv(trials)
    assert (t["kind"] == "count").sum() == 12 and (t["kind"] == "exploratory").sum() == 6
    assert R.do_select(out, trials, ledger=full)["selected_at"] == sel["selected_at"]  # not re-run on the same pull
    snap = json.loads((out / "snapshot.json").read_text())
    (out / "snapshot.json").write_text(json.dumps({**snap, "pulled_at": "2027-01-01T00:00:00Z"}))
    R.do_count(out, trials)
    with pytest.raises(SystemExit, match="does not belong to"):
        R.do_select(out, trials, ledger=full)


def test_score_refuses_without_selection_a_second_run_and_a_changed_pull(tmp_path):
    out, trials = tmp_path / "out", tmp_path / "trials.csv"
    _pull(out)
    full = _Ledger(R.REQUIRED_AMENDMENTS)
    with pytest.raises(SystemExit, match="select-only"):
        R.do_score(out, ledger=full)
    R.do_count(out, trials)
    R.do_select(out, trials, ledger=full)
    with pytest.raises(SystemExit, match="already holds a counted trial"):
        R.do_score(out, ledger=_Ledger(R.REQUIRED_AMENDMENTS, n_trials=1))
    snap = json.loads((out / "snapshot.json").read_text())
    (out / "snapshot.json").write_text(json.dumps({**snap, "pulled_at": "2027-01-01T00:00:00Z"}))
    with pytest.raises(SystemExit, match="different pull"):
        R.do_score(out, ledger=full)
