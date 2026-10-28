"""H2 tests on synthetic frames: zones, mathematical stakes, point-in-time standings, the bet rule, the count guard."""

import late_stakes as H
import numpy as np
import pandas as pd
import pytest
import sobf

T = 20
ROUNDS = 2 * (T - 1)


def _league(n_teams=T, div="E0", season="2020-2021", start="2020-09-12", drop_last=0):
    """Double round robin by the circle method, weekly rounds; the lower-numbered team always wins (2-0)."""
    teams = list(range(n_teams))
    rows = []
    arr = teams[:]
    legs = []
    for _ in range(n_teams - 1):
        legs.append([(arr[i], arr[n_teams - 1 - i]) for i in range(n_teams // 2)])
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]
    rounds = legs + [[(b, a) for a, b in r] for r in legs]
    for r, pairs in enumerate(rounds):
        d = pd.Timestamp(start) + pd.Timedelta(days=7 * r)
        for h, a in pairs:
            hg, ag = (2, 0) if h < a else (0, 2)
            rows.append(
                {"division": div, "season": season, "match_date": d, "home_team": f"t{h:02d}", "away_team": f"t{a:02d}",
                 "ft_home_goals": hg, "ft_away_goals": ag, "round": r + 1}
            )  # fmt: skip
    df = pd.DataFrame(rows)
    return df[df["round"] <= ROUNDS - drop_last].reset_index(drop=True)


def _opens(n, ref="pinnacle", ref_odds=(2.0, 3.4, 4.2), max_odds=(2.2, 3.6, 4.6)):
    d = {}
    for o, r, m in zip(sobf.OUTCOMES, ref_odds, max_odds, strict=True):
        d[f"{ref}_{o}_open"], d[f"max_{o}_open"] = [r] * n, [m] * n
    return pd.DataFrame(d)


def _count_frame(lg, **kw):
    return pd.concat([lg[H.ID_COLS + H.RESULT_COLS], _opens(len(lg), **kw)], axis=1)


def test_zone_of_fixed_boundaries_and_relegation_precedence():
    r = np.arange(1, 21)
    z = H.zone_of(r, 20)
    assert list(z[:7]) == [0, 1, 1, 1, 2, 2, 3] and z[16] == 3 and list(z[17:]) == [4, 4, 4]
    assert H.zone_of(np.array([7]), 9)[0] == 4  # bottom 3 of 9 teams wins over any other zone


def test_at_stake_clinched_leader_and_dead_rubber_have_nothing_at_stake():
    pts = np.array([90, 60, 30, 20.0] + [10.0] * 16)
    left = np.full(T, 1.0)
    s = H.at_stake(pts, left)
    assert not s[0]  # champion with an unreachable lead: best = worst = 1


def test_at_stake_title_race_and_survival_fight():
    pts = np.array([70, 69.0] + [40.0 - 2 * i for i in range(2, 20)])
    left = np.full(T, 3.0)
    s = H.at_stake(pts, left)
    assert s[0] and s[1]  # both can finish 1st or 2nd+: title at stake
    # a team inside the relegation zone but within reach of safety has something at stake
    pts2 = np.array([60 - 2.0 * i for i in range(20)])
    s2 = H.at_stake(pts2, np.full(T, 3.0))
    assert s2[16] and s2[17]


def test_at_stake_mathematically_relegated_and_mid_table_safe():
    pts = np.array([60 - 3.0 * i for i in range(19)] + [0.0])
    s = H.at_stake(pts, np.full(T, 2.0))
    assert not s[19]  # 0 points, 6 attainable: cannot escape the bottom 3
    assert not s[9]  # mid-table, every reachable rank is in the middle zone


def test_complete_league_flags_last_five_rounds_only():
    lg = _league()
    ctx = H.stakes_context(lg[H.ID_COLS + H.RESULT_COLS])
    assert ctx["complete"].all() and (ctx["n_teams"] == T).all()
    assert set(lg.loc[ctx["last5"], "round"]) == {34, 35, 36, 37, 38}
    assert ctx.loc[ctx["last5"], ["home_left", "away_left"]].max().max() == 5
    assert ctx.loc[ctx["last5"], ["home_left", "away_left"]].min().min() == 1


def test_incomplete_division_season_is_never_last5():
    lg = _league(drop_last=3)  # season cut short: teams have fewer than 2*(T-1) matches
    ctx = H.stakes_context(lg[H.ID_COLS + H.RESULT_COLS])
    assert not ctx["complete"].any() and not ctx["last5"].any() and ctx["left_ok"].any()


def test_missing_goal_makes_division_season_incomplete():
    lg = _league()
    lg.loc[3, "ft_home_goals"] = np.nan
    ctx = H.stakes_context(lg[H.ID_COLS + H.RESULT_COLS])
    assert not ctx["complete"].any()


def test_standings_use_only_results_before_the_decision_day():
    lg = _league()
    base = H.stakes_context(lg[H.ID_COLS + H.RESULT_COLS])
    cut = lg["match_date"].sort_values().unique()[35]  # round 36
    scrambled = lg.copy()
    on_or_after = scrambled["match_date"] >= cut
    scrambled.loc[on_or_after, "ft_home_goals"] = np.arange(on_or_after.sum()) % 5  # arbitrary results
    scrambled.loc[on_or_after, "ft_away_goals"] = (np.arange(on_or_after.sum()) * 3) % 4
    alt = H.stakes_context(scrambled[H.ID_COLS + H.RESULT_COLS])
    before = lg["match_date"] <= cut  # the cut-day matches themselves must not move either
    pd.testing.assert_frame_equal(base[before], alt[before])


def test_own_result_never_changes_own_context():
    lg = _league()
    i = int(lg.index[lg["round"] == 36][0])
    flipped = lg.copy()
    flipped.loc[i, ["ft_home_goals", "ft_away_goals"]] = [9, 0]
    a = H.stakes_context(lg[H.ID_COLS + H.RESULT_COLS]).loc[i]
    b = H.stakes_context(flipped[H.ID_COLS + H.RESULT_COLS]).loc[i]
    assert a.equals(b)


def test_context_matches_independent_standings_in_round_34():
    lg = _league()
    ctx = H.stakes_context(lg[H.ID_COLS + H.RESULT_COLS])
    r34 = lg[lg["round"] == 34]
    prior = lg[lg["round"] < 34]
    pts = {f"t{i:02d}": 0 for i in range(T)}
    for _, m in prior.iterrows():
        h, a = m["ft_home_goals"], m["ft_away_goals"]
        pts[m["home_team"]] += 3 if h > a else (1 if h == a else 0)
        pts[m["away_team"]] += 3 if a > h else (1 if h == a else 0)
    names = sorted(pts)
    flags = dict(zip(names, H.at_stake(np.array([pts[n] for n in names], float), np.full(T, 5.0)), strict=True))
    for i, m in r34.iterrows():
        assert ctx.loc[i, "home_stake"] == flags[m["home_team"]] and ctx.loc[i, "away_stake"] == flags[m["away_team"]]


def test_title_tie_on_points_still_counts_as_at_stake_and_a_gap_beyond_it_does_not():
    base = [20.0 - i for i in range(2, 20)]
    left = np.ones(T)
    assert H.at_stake(np.array([30.0, 27.0] + base), left)[0]  # a rival can only draw level: goal difference decides
    assert not H.at_stake(np.array([31.0, 27.0] + base), left)[0]  # one more point and the title is won


def test_count_rejects_close_and_result_columns():
    lg = _league()
    f = _count_frame(lg)
    for bad in ("pinnacle_home_close", "ft_result"):
        with pytest.raises(AssertionError):
            H.count_bets(f.assign(**{bad: 1.5}))


def test_decide_rejects_close_columns():
    lg = _league()
    ctx = H.stakes_context(lg[H.ID_COLS + H.RESULT_COLS])
    o = _opens(len(lg)).assign(pinnacle_home_close=2.0)
    with pytest.raises(AssertionError):
        H.decide(o, ctx, "pinnacle")


def test_bets_only_when_exactly_one_side_at_stake_and_side_is_the_one_at_stake():
    lg = _league()
    f = _count_frame(lg)
    ctx = H.stakes_context(lg[H.ID_COLS + H.RESULT_COLS])
    dec = H.decide(f[H.open_cols("pinnacle")], ctx, "pinnacle")
    one = ctx["last5"] & (ctx["home_stake"] ^ ctx["away_stake"])
    assert dec["asym"].equals(one)
    assert not dec.loc[~ctx["last5"], "eligible"].any()
    away_only = one & ctx["away_stake"]
    assert (dec.loc[away_only, "side"] == 2).all() and (dec.loc[one & ctx["home_stake"], "side"] == 0).all()
    p = sobf.devig_power_rows(np.array([[2.0, 3.4, 4.2]]))[0]
    h = dec.loc[one & ctx["home_stake"]]
    if len(h):
        assert np.isclose(h["edge"].iloc[0], np.log(2.2 * p[0]))
    a = dec.loc[away_only]
    if len(a):
        assert np.isclose(a["edge"].iloc[0], np.log(4.6 * p[2]))


def test_count_thetas_are_nested_and_theta_binds():
    lg = _league()
    c = H.count_bets(_count_frame(lg))
    b = c["bets"]
    assert b["0.0"] >= b["0.02"] >= b["0.05"]
    assert b["0.0"] == c["n_eligible"] > 0
    big = H.count_bets(_count_frame(lg, max_odds=(2.0, 3.4, 4.2)))  # no edge over the de-vigged open
    assert big["bets"]["0.0"] == 0


def test_data_error_cap_removes_match_and_at_cap_is_allowed():
    lg = _league()
    bad = H.count_bets(_count_frame(lg, max_odds=(3.2, 3.4, 6.4)))  # 3.2/2.0 = 1.6 > 1.5, 6.4/4.2 = 1.52
    assert bad["n_eligible"] == 0 and bad["n_asymmetric"] > 0
    ok = H.count_bets(_count_frame(lg, ref_odds=(2.0, 3.4, 4.0), max_odds=(3.0, 3.4, 6.0)))
    assert ok["n_eligible"] == ok["n_asymmetric_valid_opens"] > 0


def test_missing_open_quote_makes_match_ineligible():
    lg = _league()
    f = _count_frame(lg)
    f.loc[:, "max_home_open"] = np.nan
    f.loc[:, "max_away_open"] = np.nan
    c = H.count_bets(f)
    assert c["n_eligible"] == 0 and c["n_asymmetric"] > 0 and c["n_asymmetric_valid_opens"] == 0


def test_literal_reading_counts_incomplete_seasons_the_strict_one_drops():
    lg = _league(drop_last=3)
    c = H.count_bets(_count_frame(lg))
    assert c["n_last5"] == 0 and c["literal_no_completeness_requirement"]["n_last5"] > 0
    assert c["n_division_seasons_incomplete"] == 1


def _train(n_leagues, ref_close=(2.0, 3.4, 4.2), max_odds=(2.8, 3.6, 5.6)):
    frames = []
    for k in range(n_leagues):
        lg = _league(div=f"D{k}", season="2015-2016", start="2015-09-12")
        f = pd.concat([lg[H.ID_COLS + H.RESULT_COLS], _opens(len(lg), max_odds=max_odds)], axis=1)
        f["ft_result"] = np.where(lg["ft_home_goals"] > lg["ft_away_goals"], "H", "A")
        for o, c in zip(sobf.OUTCOMES, ref_close, strict=True):
            f[f"pinnacle_{o}_close"] = c
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_select_fallback_when_fewer_than_300_bets():
    sel, rows = H.select(_train(1))
    assert sel["fallback"] and sel["theta"] == 0.0 and not any(r["eligible"] for r in rows)


def test_select_picks_an_eligible_theta_and_breaks_ties_to_lower():
    tr = _train(25)
    sel, rows = H.select(tr)
    assert not sel["fallback"] and sel["theta"] == 0.0  # identical bet sets for every theta: tie -> lowest
    assert len({round(r["mean_clv"], 12) for r in rows if r["eligible"]}) == 1


def test_select_key_orders_theta_on_ties():
    a, b = {"mean_clv": 0.01, "theta": 0.02}, {"mean_clv": 0.01, "theta": 0.0}
    assert sorted([a, b], key=H.select_key) == [b, a]


def test_select_ignores_rows_outside_the_exploratory_window():
    tr = _train(25)
    late = tr.copy()
    late["match_date"] = late["match_date"] + pd.Timedelta(days=3650)
    late["division"] = "L" + late["division"]
    late["pinnacle_home_close"] = 1.001
    _, rows_a = H.select(tr)
    _, rows_b = H.select(pd.concat([tr, late], ignore_index=True))
    assert [r["n_bets"] for r in rows_a] == [r["n_bets"] for r in rows_b]


def test_pull_queries_read_the_registered_windows():
    q = H.pull_queries()
    assert "2012-07-01" in q["train"] and "2024-06-30" in q["train"] and "pinnacle_home_open" in q["train"]
    assert "2025-07-01" in q["eval"] and "betfair_ex_home_open" in q["eval"]
    assert "pinnacle_home_close" in q["train"] and "ft_result" in q["train"]
    assert not any(c.endswith("_close") or c == "ft_result" for c in H.count_columns())


def test_decision_day_is_latest_tuesday_or_friday_on_or_before():
    days = pd.to_datetime(
        ["2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08", "2024-03-09", "2024-03-10", "2024-03-11", "2024-03-12"]
    )  # Tue .. Tue
    got = pd.DatetimeIndex(H.decision_day(days.to_numpy(dtype="datetime64[D]"))).strftime("%m-%d").tolist()
    assert got == ["03-05", "03-05", "03-05", "03-08", "03-08", "03-08", "03-08", "03-12"]


def _two_results_league(match_day, result_day):
    """Season where team A beat B on `result_day`, then A hosts C on `match_day`; other teams are filler."""
    rows = [
        {"division": "X", "season": "s", "match_date": pd.Timestamp(result_day), "home_team": "A", "away_team": "B",
         "ft_home_goals": 1, "ft_away_goals": 0},
        {"division": "X", "season": "s", "match_date": pd.Timestamp(match_day), "home_team": "A", "away_team": "C",
         "ft_home_goals": 0, "ft_away_goals": 0},
    ]  # fmt: skip
    return pd.DataFrame(rows)


def _left_seen(match_day, result_day):
    ctx = H.stakes_context(_two_results_league(match_day, result_day))
    return ctx["home_left"].iloc[1]  # matches left for A at the later match, from results it may see


def test_sunday_match_does_not_see_preceding_saturday_results():
    full = 2 * (3 - 1)  # three teams: season length 4
    assert _left_seen("2024-03-10", "2024-03-09") == full  # Saturday result hidden (snapshot was Friday)
    assert _left_seen("2024-03-10", "2024-03-08") == full  # Friday result is on the snapshot day: hidden too
    assert _left_seen("2024-03-10", "2024-03-07") == full - 1  # Thursday result is before the snapshot: visible


def test_wednesday_match_does_not_see_tuesday_results():
    full = 2 * (3 - 1)
    assert _left_seen("2024-03-13", "2024-03-12") == full  # Tuesday result hidden
    assert _left_seen("2024-03-13", "2024-03-11") == full - 1  # Monday result is before the Tuesday snapshot
