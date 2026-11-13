"""Snapshot-timing tests for R6: is VC's (and Betfair's) 'informativeness' information, or a later capture time?

Football-data has no per-book timestamps, so each test uses a quantity whose behaviour differs between
  (I) simultaneous capture, VC holds information Pinnacle prices later, and
  (L) VC's open captured after Pinnacle's open (VC's deviation contains part of Pinnacle's own later move).

T1  Aggregator bracket. The aggregator Max is shown (R2-Q3 audit) to be captured with Pinnacle's open. If VC is in the
    aggregator set and captured simultaneously, VC_open <= Max_open always. Under (L), VC_open > Max_open appears, and
    more often where the price later lengthened (Pinnacle's odds rose). P(book > Max) by book, and VC's by move bin.
T2  Horizon. Under (L) the captured part is Pinnacle's move inside a fixed lag after the snapshot, so its covariance
    with the move should be largest where Pinnacle moves fastest after the snapshot: same-day games (kickoff hours
    after the Fri/Tue snapshot). VC slopes by snapshot->kickoff hours (kickoff_time, 2019+), and by weekday.
T3  Pinnacle-is-best subsets: VC's partial slope where Pinnacle's open is >= every named book's open on that outcome.
T4  2024/25 Betfair: BF (sportsbook) and BFE (exchange) partial slopes jointly; common residual component.
T5  P(Max == PS_open) and P(Max < PS_open) by subsequent Pinnacle move, evaluation window (re-check of R2-Q3 test).

Usage: OMP_NUM_THREADS=1 arch -arm64 uv run python research/audits/r6/timing.py
"""

from __future__ import annotations

import json

import alib as A
import numpy as np
import pandas as pd

N_BOOT = 1000


def coef_ci(X: np.ndarray, y: np.ndarray, blocks: np.ndarray, j: int = 1, seed: int = 11) -> list[float]:
    """OLS coefficient j (X includes the intercept) with an exact match-day block-bootstrap 95% CI."""
    codes, u = pd.factorize(pd.Series(blocks))
    nb, p = len(u), X.shape[1]
    xx = np.zeros((nb, p, p))
    xy = np.zeros((nb, p))
    np.add.at(xx, codes, X[:, :, None] * X[:, None, :])
    np.add.at(xy, codes, X * y[:, None])
    point = float(np.linalg.solve(xx.sum(0), xy.sum(0))[j])
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(N_BOOT)]).astype(float)
    b = np.linalg.solve(np.einsum("bk,kij->bij", w, xx), (w @ xy)[:, :, None])[:, j, 0]
    return [point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(y))]


def long(d: pd.DataFrame, r: dict) -> pd.DataFrame:
    rows = []
    for k, o in enumerate("HDA"):
        f = pd.DataFrame(
            {
                "mid": d.mid,
                "date": d.match_date.dt.strftime("%Y-%m-%d"),
                "ss": d.ss,
                "division": d.division,
                "o": o,
                "y": r["move"][:, k],
                "g_avg": r["g_avg"][:, k],
                "kick": d.kickoff_time,
                "wd": d.match_date.dt.weekday,
                "ps": r["ps"][:, k],
                "psc": A.odds3(d, "PSC")[:, k],
                "mx": A.odds3(d, "mx")[:, k],
            }
        )
        for b in A.CORE + A.LATE:
            f["x_" + b] = r["dev_" + b][:, k]
            f["odds_" + b] = A.odds3(d, b)[:, k]
            f["self_" + b] = -r["selfmv_" + b][:, k]
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def slopes(f: pd.DataFrame, b: str) -> dict:
    g = f[np.isfinite(f["x_" + b]) & (f["x_" + b].abs() <= 0.5) & np.isfinite(f.y) & np.isfinite(f.g_avg)]
    if len(g) < 300:
        return {"n": len(g)}
    one = np.ones(len(g))
    x, y, ga = g["x_" + b].to_numpy(), g.y.to_numpy(), g.g_avg.to_numpy()
    return {
        "n": len(g),
        "marginal": coef_ci(np.column_stack([one, x]), y, g.date.to_numpy()),
        "partial": coef_ci(np.column_stack([one, x, ga]), y, g.date.to_numpy()),
        "cov_xy_e4": float(np.cov(x, y)[0, 1] * 1e4),
        "var_y_e4": float(np.var(y) * 1e4),
        "var_x_e4": float(np.var(x) * 1e4),
    }


def main() -> None:
    d = A.universe(A.raw())
    r = A.core_frame(d)
    f = long(d, r)
    ev = f[f.ss.between(2019, 2024)].copy()
    res: dict = {}

    # T1 aggregator bracket
    t1 = {}
    for b in A.CORE + ("PS",):
        col = "ps" if b == "PS" else "odds_" + b
        for era, g in (("2012-18", f[f.ss <= 2018]), ("2019-24", ev)):
            m = np.isfinite(g[col]) & np.isfinite(g.mx)
            t1.setdefault(b, {})[era] = {
                "p_above_max": float((g[col][m] > g.mx[m] + 1e-9).mean()),
                "p_eq_max": float((np.abs(g[col][m] - g.mx[m]) < 1e-9).mean()),
                "n": int(m.sum()),
            }
    # VC above max by subsequent Pinnacle odds move (log PSC/PS > 0: price lengthened)
    by = {}
    for era, g in (("2012-18", f[f.ss <= 2018]), ("2019-24", ev)):
        m = np.isfinite(g.odds_VC) & np.isfinite(g.mx) & np.isfinite(g.psc)
        g = g[m]
        mv = np.log(g.psc / g.ps)
        bins = pd.cut(mv, [-9, -0.10, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.10, 9])
        t = pd.DataFrame(
            {
                "bin": bins.astype(str),
                "above": g.odds_VC > g.mx + 1e-9,
                "eq": (g.odds_VC - g.mx).abs() < 1e-9,
                "vc_gt_ps": g.odds_VC > g.ps + 1e-9,
            }
        )
        by[era] = (
            t.groupby("bin")
            .agg(
                n=("above", "size"),
                p_vc_above_max=("above", "mean"),
                p_vc_eq_max=("eq", "mean"),
                p_vc_gt_ps=("vc_gt_ps", "mean"),
            )
            .round(4)
            .to_dict(orient="index")
        )
    t1["VC_by_subsequent_pinnacle_move"] = by
    res["T1_bracket"] = t1
    print(json.dumps(t1, indent=1))

    # T5 PS == Max by subsequent move, eval
    m = np.isfinite(ev.mx) & np.isfinite(ev.psc)
    g = ev[m]
    mv = np.log(g.psc / g.ps)
    t = pd.DataFrame(
        {
            "bin": pd.cut(mv, [-9, -0.10, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.10, 9]).astype(str),
            "eq": (g.mx - g.ps).abs() < 1e-9,
            "lt": g.mx < g.ps - 1e-9,
        }
    )
    res["T5_ps_vs_max"] = (
        t.groupby("bin")
        .agg(n=("eq", "size"), p_eq=("eq", "mean"), p_max_lt_ps=("lt", "mean"))
        .round(4)
        .to_dict(orient="index")
    )

    # T2 horizon: hours from snapshot (Fri/Tue 16:00 UK assumed) to kickoff (UK local), eval rows with kickoff time
    md = pd.to_datetime(ev.date)
    back = ev.wd.map({0: 3, 1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2})
    snap = md - pd.to_timedelta(back, unit="D") + pd.Timedelta(hours=16)
    ko = (
        md
        + pd.to_timedelta(ev.kick.fillna("").str[:2].replace("", np.nan).astype(float), unit="h")
        + pd.to_timedelta(ev.kick.fillna("").str[3:5].replace("", np.nan).astype(float), unit="m")
    )
    ev["hrs"] = (ko - snap).dt.total_seconds() / 3600
    ev["hbin"] = pd.cut(
        ev.hrs, [-24, 8, 30, 54, 100], labels=["same-day (<8h)", "next day (8-30h)", "2 days (30-54h)", "3 days (54h+)"]
    )
    t2 = {}
    for hb, g in ev.groupby("hbin", observed=True):
        t2[str(hb)] = {b: slopes(g, b) for b in ("VC", "B365", "BW", "IW", "WH", "BF", "BFE", "1XB")}
        t2[str(hb)]["n_rows"] = len(g)
    res["T2_by_hours"] = t2
    wdn = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    res["T2_by_weekday_VC"] = {wdn[w]: slopes(g, "VC") for w, g in ev.groupby("wd")}
    # same, exploratory window (no kickoff times; weekday only) as a replication
    res["T2_by_weekday_VC_2012_18"] = {wdn[w]: slopes(g, "VC") for w, g in f[f.ss <= 2018].groupby("wd")}
    for k, v in t2.items():
        print(
            k,
            v["n_rows"],
            {
                b: (round(v[b]["marginal"][0], 3), round(v[b]["partial"][0], 3), v[b]["n"])
                for b in v
                if b != "n_rows" and "marginal" in v[b]
            },
        )
    for k, v in res["T2_by_weekday_VC"].items():
        print(
            "VC eval",
            k,
            v["n"],
            [round(x, 3) for x in v["marginal"][:3]],
            [round(x, 3) for x in v["partial"][:3]],
            round(v["cov_xy_e4"], 3),
            round(v["var_y_e4"], 2),
        )
    for k, v in res["T2_by_weekday_VC_2012_18"].items():
        print(
            "VC expl",
            k,
            v["n"],
            [round(x, 3) for x in v["marginal"][:3]],
            [round(x, 3) for x in v["partial"][:3]],
            round(v["cov_xy_e4"], 3),
            round(v["var_y_e4"], 2),
        )

    # T3 Pinnacle best named price on the outcome vs not
    named = np.nanmax(np.column_stack([ev["odds_" + b] for b in A.CORE]), axis=1)
    ev["pin_best"] = ev.ps >= named - 1e-9
    ev["vc_eq_max"] = (ev.odds_VC - ev.mx).abs() < 1e-9
    res["T3"] = {
        "pin_best_named": slopes(ev[ev.pin_best], "VC"),
        "pin_not_best": slopes(ev[~ev.pin_best], "VC"),
        "pin_eq_max": slopes(ev[(ev.ps - ev.mx).abs() < 1e-9], "VC"),
        "pin_below_max": slopes(ev[ev.ps < ev.mx - 1e-9], "VC"),
        "vc_below_max": slopes(ev[ev.odds_VC < ev.mx - 1e-9], "VC"),
    }
    for k, v in res["T3"].items():
        print("T3", k, v["n"], [round(x, 3) for x in v["marginal"][:3]], [round(x, 3) for x in v["partial"][:3]])

    # T4 2024/25 Betfair SB vs exchange
    g = ev[(ev.ss == 2024)]
    g = g[
        np.isfinite(g.x_BF) & np.isfinite(g.x_BFE) & (g.x_BF.abs() <= 0.5) & (g.x_BFE.abs() <= 0.5) & np.isfinite(g.y)
    ]
    one = np.ones(len(g))
    X = np.column_stack([one, g.x_BF, g.x_BFE, g.g_avg])

    def rx(v):
        return v - np.column_stack([one, g.g_avg]) @ np.linalg.lstsq(np.column_stack([one, g.g_avg]), v, rcond=None)[0]

    res["T4_betfair_2024"] = {
        "n": len(g),
        "joint_BF": coef_ci(X, g.y.to_numpy(), g.date.to_numpy(), 1),
        "joint_BFE": coef_ci(X, g.y.to_numpy(), g.date.to_numpy(), 2),
        "corr_resid_BF_BFE": float(np.corrcoef(rx(g.x_BF.to_numpy()), rx(g.x_BFE.to_numpy()))[0, 1]),
        "corr_raw_BF_BFE": float(np.corrcoef(g.x_BF, g.x_BFE)[0, 1]),
    }
    # joint with B365 / 1XB too
    g2 = g[np.isfinite(g.x_1XB) & np.isfinite(g.x_B365) & (g.x_1XB.abs() <= 0.5) & (g.x_B365.abs() <= 0.5)]
    X2 = np.column_stack([np.ones(len(g2)), g2.x_BF, g2.x_BFE, g2.x_B365, g2.x_1XB, g2.g_avg])
    res["T4_betfair_2024"]["joint5"] = {
        nm: coef_ci(X2, g2.y.to_numpy(), g2.date.to_numpy(), j)
        for j, nm in enumerate(["c", "BF", "BFE", "B365", "1XB", "g_avg"])
        if j
    }
    print(json.dumps(res["T4_betfair_2024"], indent=1))
    (A.REPO / "research/audits/r6/out/timing.json").write_text(json.dumps(res, indent=1, default=str))
    ev[["mid", "o", "hrs", "wd"]].to_parquet(A.OUT / "hours.parquet")


if __name__ == "__main__":
    main()
