"""Q1 — calibration of the de-vigged sharp close (bookmaker-line-structure-close-calibration-v1).

Measures, on the fixed fd_matches snapshot, the favourite-longshot slope and draw bias of the de-vigged
Pinnacle close (2012/13-2024/25) and Betfair Exchange close (2024-08+), per division with BH across
divisions, and compares multiplicative / power / Shin de-vig by paired log loss. Writes trial rows to
`trials.csv` beside this file and tables to data/research/r2/q1/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r2-common"))
import r2lib as L  # noqa: E402

REG = "bookmaker-line-structure-close-calibration-v1"
HERE = Path(__file__).resolve().parent
OUT = L.DATA / "q1"
WINDOWS = {
    "exploratory": ("2012-07-01", "2019-06-30"),
    "evaluation": ("2019-07-01", "2025-06-30"),
}
BF_WINDOW = ("post_segment_betfair", "2024-08-01", "2026-10-19")
BENCH_COLS = {
    "pinnacle_close": [f"pinnacle_{o}_close" for o in L.OUTCOMES],
    "betfair_ex_close": [f"betfair_ex_{o}_close" for o in L.OUTCOMES],
}
MIN_N = 300


def logistic_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Point estimate of b in logit P(y=1) = a + b logit(p) (Newton-Raphson)."""
    x = np.column_stack([np.ones(len(p)), L.logit(p)])
    beta = np.array([0.0, 1.0])
    for _ in range(25):
        mu = 1 / (1 + np.exp(-x @ beta))
        grad = x.T @ (y - mu)
        hess = (x * (mu * (1 - mu))[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[1])


def calib_frames(sub: pd.DataFrame, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Stack home and away outcome rows: (p, y, match-day) for the FLB slope regression."""
    y = sub[["y_home", "y_away"]].to_numpy()
    pp = p[:, [0, 2]]
    day = pd.concat([sub["match_date"], sub["match_date"]], ignore_index=True)
    return pp.T.reshape(-1), y.T.reshape(-1), day


def flb(sub: pd.DataFrame, p: np.ndarray) -> dict:
    """OLS calibration slope minus 1 (home+away rows) with match-day bootstrap CI."""
    pp, yy, day = calib_frames(sub, p)
    o = L.boot_ols(yy, [pp], day)
    b = o["coef"][1]
    return {
        "n": len(sub),
        "mean": b - 1,
        "ci_lo": o["ci_lo"][1] - 1,
        "ci_hi": o["ci_hi"][1] - 1,
        "se": o["se"][1],
        "z": (b - 1) / o["se"][1],
        "intercept": o["coef"][0],
    }


def draw_bias(sub: pd.DataFrame, p: np.ndarray) -> dict:
    return L.boot_mean(sub["y_draw"].to_numpy() - p[:, 1], sub["match_date"])


def logloss(sub: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    y = sub[["y_home", "y_draw", "y_away"]].to_numpy()
    return -np.log(np.clip((p * y).sum(axis=1), 1e-12, 1))


def ece(sub: pd.DataFrame, p: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    """Expected calibration error over all three outcome rows, 10 equal-width bins; also the reliability table."""
    y = sub[["y_home", "y_draw", "y_away"]].to_numpy().reshape(-1)
    pp = p.reshape(-1)
    b = np.minimum((pp * bins).astype(int), bins - 1)
    t = (
        pd.DataFrame({"bin": b, "p": pp, "y": y})
        .groupby("bin")
        .agg(n=("y", "size"), p_mean=("p", "mean"), y_mean=("y", "mean"))
    )
    t["gap"] = t["y_mean"] - t["p_mean"]
    return float((t["n"] * t["gap"].abs()).sum() / t["n"].sum()), t.reset_index()


def eligible(d: pd.DataFrame, bench: str) -> pd.DataFrame:
    cols = BENCH_COLS[bench]
    ok = d[cols].notna().all(axis=1) & (d[cols] > 1).all(axis=1) & d["y_home"].notna()
    return d[ok].copy()


def run_block(led: L.Ledger, sub: pd.DataFrame, bench: str, wname: str, results: dict, lo: str, hi: str) -> None:
    cols = BENCH_COLS[bench]
    probs = {m: L.devig_cols(sub, cols, m) for m in L.DEVIG}
    blk = results.setdefault(f"{bench}|{wname}", {"n": len(sub), "days": int(sub["match_date"].nunique())})
    overround = (1 / sub[cols]).sum(axis=1)
    blk["overround_median"] = float(overround.median())
    lls = {}
    for m, p in probs.items():
        f = flb(sub, p)
        dbias = draw_bias(sub, p)
        lls[m] = logloss(sub, p)
        e, rel = ece(sub, p)
        rel.to_csv(OUT / f"reliability_{bench}_{wname}_{m}.csv", index=False)
        pp, yy, _ = calib_frames(sub, p)
        blk[m] = {
            "flb_slope_minus_1": f,
            "draw_bias": dbias,
            "logloss": float(lls[m].mean()),
            "ece": e,
            "logistic_slope": logistic_slope(yy, pp),
        }
        cfg = {"bench": bench, "devig": m, "division": "ALL", "window": [lo, hi]}
        led.add({**cfg, "test": "flb_slope"}, f, "ols_slope_minus_1", wname, "pooled", "two-sided")
        led.add({**cfg, "test": "draw_bias"}, dbias, "draw_bias", wname, "pooled", "two-sided")
    for a, b_ in (("power", "multiplicative"), ("shin", "multiplicative"), ("shin", "power")):
        r = L.boot_mean(lls[a] - lls[b_], sub["match_date"])
        blk[f"ll_{a}_minus_{b_}"] = r
        led.add(
            {"bench": bench, "test": "devig_pair", "pair": f"{a}-{b_}", "window": [lo, hi]},
            r,
            "mean_logloss_diff_nats",
            wname,
            "devig",
            "two-sided",
        )

    # per division, power de-vig only
    p = probs["power"]
    sub = sub.assign(_i=np.arange(len(sub)))
    counts = sub["division"].value_counts()
    sub["div_g"] = np.where(sub["division"].map(counts) >= MIN_N, sub["division"], "OTHER")
    rows = []
    for div, g in sub.groupby("div_g"):
        pi = p[g["_i"].to_numpy()]
        f = flb(g, pi)
        dbias = draw_bias(g, pi)
        cfg = {"bench": bench, "devig": "power", "division": div, "window": [lo, hi]}
        led.add({**cfg, "test": "flb_slope"}, f, "ols_slope_minus_1", wname, "per_division", "two-sided")
        led.add({**cfg, "test": "draw_bias"}, dbias, "draw_bias", wname, "per_division", "two-sided")
        rows.append(
            {
                "division": div,
                "n": len(g),
                "flb": f["mean"],
                "flb_lo": f["ci_lo"],
                "flb_hi": f["ci_hi"],
                "flb_p": L.p_value(f["z"], "two-sided"),
                "draw": dbias["mean"],
                "draw_lo": dbias["ci_lo"],
                "draw_hi": dbias["ci_hi"],
                "draw_p": L.p_value(dbias["z"], "two-sided"),
                "draw_rate": float(g["y_draw"].mean()),
                "p_draw_mean": float(pi[:, 1].mean()),
            }
        )
    t = pd.DataFrame(rows)
    t["flb_q"], t["flb_reject"] = L.bh(t["flb_p"].to_numpy())
    t["draw_q"], t["draw_reject"] = L.bh(t["draw_p"].to_numpy())
    t.to_csv(OUT / f"per_division_{bench}_{wname}.csv", index=False)
    blk["per_division"] = {
        "n_divisions": len(t),
        "flb_reject_bh": int(t["flb_reject"].sum()),
        "flb_positive_and_reject": int(((t["flb"] > 0) & t["flb_reject"]).sum()),
        "flb_negative_and_reject": int(((t["flb"] < 0) & t["flb_reject"]).sum()),
        "draw_reject_bh": int(t["draw_reject"].sum()),
        "draw_positive_and_reject": int(((t["draw"] > 0) & t["draw_reject"]).sum()),
        "draw_negative_and_reject": int(((t["draw"] < 0) & t["draw_reject"]).sum()),
    }

    # stability: chronological thirds, power de-vig
    th = L.thirds(sub["match_date"], lo, hi)
    blk["thirds"] = {}
    for k in (1, 2, 3):
        mask = (th == k).to_numpy()
        if mask.sum() < MIN_N:
            continue
        g = sub[mask]
        blk["thirds"][k] = {
            "n": int(mask.sum()),
            "flb": flb(g, p[mask]),
            "draw": draw_bias(g, p[mask]),
            "ll_power_minus_mult": L.boot_mean(lls["power"][mask] - lls["multiplicative"][mask], g["match_date"]),
            "ll_shin_minus_power": L.boot_mean(lls["shin"][mask] - lls["power"][mask], g["match_date"]),
        }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = L.load()
    led = L.Ledger(HERE, REG)
    results: dict = {"snapshot": L.SNAPSHOT_AT, "git_sha": led.sha, "registration_sha256": led.reg_sha}
    pin = eligible(d, "pinnacle_close")
    for wname, (lo, hi) in WINDOWS.items():
        run_block(led, L.window(pin, lo, hi), "pinnacle_close", wname, results, lo, hi)
    wname, lo, hi = BF_WINDOW
    bf = L.window(eligible(d, "betfair_ex_close"), lo, hi)
    run_block(led, bf, "betfair_ex_close", wname, results, lo, hi)
    # paired: Betfair close vs Pinnacle close on matches that have both (power de-vig)
    both = bf[bf[BENCH_COLS["pinnacle_close"]].notna().all(axis=1) & (bf[BENCH_COLS["pinnacle_close"]] > 1).all(axis=1)]
    llb = logloss(both, L.devig_cols(both, BENCH_COLS["betfair_ex_close"], "power"))
    llp = logloss(both, L.devig_cols(both, BENCH_COLS["pinnacle_close"], "power"))
    r = L.boot_mean(llb - llp, both["match_date"])
    results["betfair_minus_pinnacle_logloss_overlap"] = r
    led.add(
        {"test": "bench_pair", "pair": "betfair_ex_close-pinnacle_close", "devig": "power", "window": [lo, hi]},
        r,
        "mean_logloss_diff_nats",
        wname,
        "benchmark_compare",
        "two-sided",
    )
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=_np))
    print(json.dumps(results, indent=1, default=_np))


def _np(o: object) -> object:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
