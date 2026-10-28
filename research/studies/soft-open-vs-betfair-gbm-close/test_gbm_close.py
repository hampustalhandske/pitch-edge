"""H3 tests on synthetic frames: no sharp book in the model inputs, bet rule, folds, selection tie-break/fallback."""

import gbm_close as H
import numpy as np
import pandas as pd
import pytest
import sobf


def _synthetic(n=600, seed=0, start="2012-08-01", span_days=365 * 5):
    rng = np.random.default_rng(seed)
    date = pd.Timestamp(start) + pd.to_timedelta(np.sort(rng.integers(0, span_days, n)), unit="D")
    p = rng.dirichlet([4, 2.5, 3], n)
    p = np.clip(p, 0.04, None)
    p /= p.sum(axis=1, keepdims=True)

    def odds(prob, margin, noise):
        q = prob * (1 + margin) * np.exp(rng.normal(0, noise, prob.shape))
        return 1.0 / q

    d = {"division": rng.choice(["E0", "D1"], n), "match_date": date, "home_team": [f"h{i}" for i in range(n)],
         "away_team": [f"a{i}" for i in range(n)]}  # fmt: skip
    avg, mx, b365 = odds(p, 0.05, 0.01), odds(p, 0.0, 0.03), odds(p, 0.06, 0.02)
    for name, arr in (("avg", avg), ("max", mx), ("bet365", b365)):
        for i, o in enumerate(sobf.OUTCOMES):
            d[f"{name}_{o}_open"] = arr[:, i]
    close = odds(p, 0.02, 0.005)
    for i, o in enumerate(sobf.OUTCOMES):
        d[f"pinnacle_{o}_close"] = close[:, i]
        d[f"betfair_ex_{o}_close"] = close[:, i]
    return pd.DataFrame(d)


def test_feature_matrix_has_the_registered_28_columns_and_no_sharp_or_close_column():
    f = _synthetic(50)
    x = H.build_features(f[H.SOURCE_COLS], f["match_date"])
    assert list(x.columns) == H.FEATURE_NAMES and x.shape == (50, 28)
    assert not any(s in c for c in x.columns for s in ("pinnacle", "betfair", "close", "ft_"))
    assert np.isfinite(x.to_numpy()).all()


def test_build_features_refuses_pinnacle_betfair_or_close_input():
    f = _synthetic(20)
    for extra in ("pinnacle_home_close", "betfair_ex_home_close", "pinnacle_home_open"):
        f[extra] = 2.0
        with pytest.raises(AssertionError):
            H.build_features(f[H.SOURCE_COLS + [extra]], f["match_date"])
    with pytest.raises(AssertionError):
        H.assert_features(H.FEATURE_NAMES + ["pinnacle_home_close"])
    with pytest.raises(AssertionError):
        H.assert_features([c for c in H.FEATURE_NAMES if c != "era_2019"])


def test_pull_sql_has_no_sharp_book_in_features_and_files_do_not_share_sharp_columns():
    q = H.pull_queries()
    assert "betfair" not in q["train"] and "pinnacle" not in q["eval"]
    assert all(not c.startswith(("pinnacle", "betfair")) for c in H.SOURCE_COLS)


def test_changing_sharp_columns_does_not_change_model_inputs_or_predictions():
    f = _synthetic(400)
    g = f.copy()
    for o in sobf.OUTCOMES:
        g[f"pinnacle_{o}_close"] = 3.0
        g[f"betfair_ex_{o}_close"] = 3.0
    xf = H.build_features(f[H.SOURCE_COLS], f["match_date"])
    xg = H.build_features(g[H.SOURCE_COLS], g["match_date"])
    pd.testing.assert_frame_equal(xf, xg)


def test_era_flag_switches_on_2019_07_01():
    f = _synthetic(10)
    f["match_date"] = pd.to_datetime(["2019-06-30", "2019-07-01"] * 5)
    assert H.build_features(f[H.SOURCE_COLS], f["match_date"])["era_2019"].tolist() == [0.0, 1.0] * 5


def test_hyperparameters_are_the_registered_tiny_setting():
    m = H.new_model()
    assert (m.max_iter, m.learning_rate, m.max_leaf_nodes, m.min_samples_leaf, m.l2_regularization) == (
        150,
        0.03,
        7,
        1000,
        10.0,
    )
    assert m.early_stopping is False and m.random_state == 20261028


def test_decide_edge_cap_argmax_and_ties():
    mx = np.array([[2.0, 3.5, 4.0], [3.0, 3.0, 3.0], [2.0, 3.0, 4.0]])
    avg = np.array([[1.9, 3.4, 2.4], [2.9, 2.9, 2.9], [1.9, 2.9, 3.9]])  # row 0 outcome 2: 4.0/2.4 > 1.5
    q = np.array([[0.5, 0.3, 0.2], [1 / 3, 1 / 3, 1 / 3], [0.3, 0.3, 0.45]])
    j, e, price = H.decide(mx, avg, q)
    assert j[0] == 1 and np.isclose(e[0], np.log(3.5 * 0.3)) and price[0] == 3.5  # outcome 2 excluded
    assert j[1] == 0  # exact tie -> first outcome
    assert j[2] == 2 and np.isclose(e[2], np.log(4.0 * 0.45))
    j2, e2, _ = H.decide(mx[:1], np.array([[1.0, 1.0, 1.0]]), q[:1])  # every outcome a data error
    assert np.isinf(e2[0]) and e2[0] < 0


def test_bets_at_strict_theta_and_clv_vs_close():
    f = _synthetic(5)
    f[[f"max_{o}_open" for o in sobf.OUTCOMES]] = np.array([[2.0, 3.5, 4.0]] * 5)
    f[[f"avg_{o}_open" for o in sobf.OUTCOMES]] = np.array([[1.9, 3.4, 3.9]] * 5)
    q = np.array([[0.5, 0.3, 0.2]] * 5)
    pc = np.array([[0.52, 0.28, 0.20]] * 5)
    # best edge is max 2.0 * 0.5 -> log(1.0) = 0 on outcome 0 / 3.5*.3=1.05 on outcome 1
    j, e, _ = H.decide(sobf.triple(f, "max", "open"), sobf.triple(f, "avg", "open"), q)
    theta = float(e[0])
    assert len(H.bets_at(f, q, pc, theta)) == 0  # edge > theta is strict
    b = H.bets_at(f, q, pc, theta - 1e-9)
    assert len(b) == 5 and np.allclose(b["clv"], np.log(3.5 * 0.28))


def test_walk_forward_uses_only_earlier_seasons():
    f = _synthetic(900, span_days=365 * 4)
    f["season"] = sobf.season_start(f["match_date"])
    x = H.build_features(f[H.SOURCE_COLS], f["match_date"]).to_numpy()
    y = H.logit(sobf.devig_power_rows(sobf.triple(f, "pinnacle", "close")))
    season = f["season"].to_numpy()
    keep = dict(H.HGB_PARAMS)
    H.HGB_PARAMS.update(min_samples_leaf=20, max_iter=5)
    try:
        q = H.walk_forward(x, y, season)
        first = season == season.min()
        assert np.isnan(q[first]).all() and np.isfinite(q[~first]).all()
        assert np.allclose(q[~first].sum(axis=1), 1.0)
        # the last season's predictions do not change if its own targets are scrambled (labels never enter the fold)
        y2 = y.copy()
        last = season == season.max()
        y2[last] = y2[last][::-1] + 5.0
        q2 = H.walk_forward(x, y2, season)
        assert np.allclose(q[last], q2[last])
    finally:
        H.HGB_PARAMS.update(keep)


def test_select_key_prefers_lower_theta_on_ties():
    a = {"total_clv": 12.3, "theta": 0.02}
    b = {"total_clv": 12.3, "theta": 0.01}
    c = {"total_clv": 12.4, "theta": 0.03}
    assert sorted([a, b, c], key=H.select_key) == [c, b, a]


def test_select_fallback_and_tie_with_stubbed_walk_forward(monkeypatch):
    f = _synthetic(300, span_days=365 * 3)

    def fake_wf(x, y, season):
        q = np.full((len(x), 3), np.nan)
        later = season > season.min()
        q[later] = np.array([0.5, 0.3, 0.2])
        return q

    monkeypatch.setattr(H, "walk_forward", fake_wf)
    sel, rows = H.select(f)
    assert sel["fallback"] and sel["theta"] == 0.02 and not any(r["eligible"] for r in rows)  # < 1000 bets
    monkeypatch.setattr(H, "MIN_OOF_BETS", 1)
    sel, rows = H.select(f)
    assert not sel["fallback"] and [r["theta"] for r in rows] == [0.01, 0.02, 0.03]
    best = max(rows, key=lambda r: round(r["total_clv"], 9))
    assert sel["theta"] == min(r["theta"] for r in rows if round(r["total_clv"], 9) == round(best["total_clv"], 9))


def test_select_reads_only_the_train_window():
    f = _synthetic(200, span_days=365 * 2)
    late = f.copy()
    late["match_date"] = pd.Timestamp("2025-09-01")
    assert len(sobf.in_window(pd.concat([f, late]), *H.TRAIN_WINDOW)) == len(f)
    t, x, pc, season = H.train_universe(pd.concat([f, late], ignore_index=True))
    assert len(t) == len(f) and x.shape[1] == 28


def test_eval_universe_requires_betfair_close_and_soft_opens_only():
    f = _synthetic(30)
    f.loc[f.index[0], "betfair_ex_home_close"] = np.nan
    f.loc[f.index[1], "max_draw_open"] = 1.0
    e, x, pc = H.eval_universe(f)
    assert len(e) == 28 and x.shape == (28, 28) and np.allclose(pc.sum(axis=1), 1.0)
