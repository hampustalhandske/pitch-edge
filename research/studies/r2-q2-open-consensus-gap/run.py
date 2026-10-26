"""Q2 — is the Pinnacle open->close 1X2 move predictable from the open-time consensus gap?

Registration: research/registry/bookmaker-line-structure-open-consensus-gap-v1.yaml.
"Open" is football-data.co.uk's pre-closing snapshot (Friday afternoon for weekend games, Tuesday afternoon
for midweek games). Features use only `*_open` columns (asserted); closes and results are targets.
Writes trial rows to `trials.csv` beside this file and outputs to data/research/r2/q2/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r2-common"))
import harness as H  # noqa: E402
import r2lib as L  # noqa: E402
import strat as S  # noqa: E402

REG = "bookmaker-line-structure-open-consensus-gap-v1"
HERE = Path(__file__).resolve().parent
OUT = L.DATA / "q2"
WINDOWS = {"exploratory": ("2012-07-01", "2019-06-30"), "evaluation": ("2019-07-01", "2025-06-30")}
POST = ("post_segment", "2025-07-01", "2026-10-19")
THETAS = [0.00, 0.01, 0.02, 0.03, 0.05]
CAP = 1.5

PIN_O = [f"pinnacle_{o}_open" for o in L.OUTCOMES]
AVG_O = [f"avg_{o}_open" for o in L.OUTCOMES]
MAX_O = [f"max_{o}_open" for o in L.OUTCOMES]
PIN_C = [f"pinnacle_{o}_close" for o in L.OUTCOMES]
BF_C = [f"betfair_ex_{o}_close" for o in L.OUTCOMES]
Y = [f"y_{o}" for o in L.OUTCOMES]


def features(f: pd.DataFrame) -> dict[str, np.ndarray]:
    """Decision-time quantities from the pre-closing snapshot only."""
    L.assert_open_only(PIN_O + AVG_O + MAX_O)
    p_pin = L.devig_cols(f, PIN_O, "power")
    p_avg = L.devig_cols(f, AVG_O, "power")
    ratio = np.minimum(f[MAX_O].to_numpy() / f[PIN_O].to_numpy(), CAP)
    return {
        "p_pin": p_pin,
        "p_avg": p_avg,
        "gap": p_avg - p_pin,
        "g_avg": L.logit(p_avg) - L.logit(p_pin),
        "g_max": np.log(ratio),
    }


def eligible(d: pd.DataFrame, bench_cols: list[str]) -> pd.DataFrame:
    cols = PIN_O + AVG_O + MAX_O + bench_cols
    ok = d["main"] & d[cols].notna().all(axis=1) & (d[cols] > 1).all(axis=1) & d["y_home"].notna()
    return d[ok].reset_index(drop=True)


def bets_for(f: pd.DataFrame, feat: dict, p_close: np.ndarray, theta: float) -> pd.DataFrame:
    """One bet per match on the outcome with the largest consensus-minus-Pinnacle gap, if gap > theta."""
    j = np.argmax(feat["gap"], axis=1)
    rows = np.arange(len(f))
    g = feat["gap"][rows, j]
    take = g > theta
    odds = f[PIN_O].to_numpy()[rows, j]
    y = f[Y].to_numpy()[rows, j]
    pc = p_close[rows, j]
    b = pd.DataFrame(
        {
            "match_date": f["match_date"],
            "division": f["division"],
            "odds": odds,
            "p_close": pc,
            "y": y,
            "gap": g,
            "outcome": np.array(L.OUTCOMES)[j],
        }
    )
    return b[take]


def stacked_moves(f: pd.DataFrame, feat: dict, p_close: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], pd.Series]:
    """One row per (match, outcome): move = logit(p_close) - logit(p_pin_open); regressors g_avg, g_max."""
    move = (L.logit(p_close) - L.logit(feat["p_pin"])).T.reshape(-1)
    xs = [feat["g_avg"].T.reshape(-1), feat["g_max"].T.reshape(-1)]
    day = pd.concat([f["match_date"]] * 3, ignore_index=True)
    return move, xs, day


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = L.load()
    led = L.Ledger(HERE, REG)
    res: dict = {"snapshot": L.SNAPSHOT_AT, "git_sha": led.sha, "registration_sha256": led.reg_sha}
    base = eligible(d, PIN_C)
    fitted = None
    pbo_inputs = None
    for wname, (lo, hi) in WINDOWS.items():
        f = L.window(base, lo, hi).reset_index(drop=True)
        feat = features(f)
        pc = L.devig_cols(f, PIN_C, "power")
        blk = res.setdefault(wname, {"n_matches": len(f), "days": int(f["match_date"].nunique())})
        # reference: every outcome backed at Pinnacle open
        allclv = np.log(f[PIN_O].to_numpy() * pc).reshape(-1)
        blk["reference_all_outcomes_clv"] = float(allclv.mean())
        blk["gap_quantiles"] = np.quantile(feat["gap"].max(axis=1), [0.1, 0.25, 0.5, 0.75, 0.9]).round(4).tolist()
        blk["share_draw_is_max_gap"] = float((np.argmax(feat["gap"], axis=1) == 1).mean())
        # predictability: in-window OLS (descriptive) and OOS R^2 of the exploratory fit
        move, xs, day = stacked_moves(f, feat, pc)
        ols = L.boot_ols(move, xs, day)
        blk["ols"] = {"coef": ols["coef"], "ci_lo": ols["ci_lo"], "ci_hi": ols["ci_hi"], "n_rows": ols["n"]}
        for i, name in ((1, "g_avg"), (2, "g_max")):
            led.add(
                {"test": f"ols_slope_{name}", "window": [lo, hi]},
                {
                    "n": len(f),
                    "mean": ols["coef"][i],
                    "ci_lo": ols["ci_lo"][i],
                    "ci_hi": ols["ci_hi"][i],
                    "z": ols["coef"][i] / ols["se"][i],
                },
                f"ols_slope_{name}",
                wname,
                "predictability",
                "two-sided",
            )
        if wname == "exploratory":
            fitted = ols["coef"]
        else:
            yhat = fitted[0] + fitted[1] * xs[0] + fitted[2] * xs[1]
            r2 = L.boot_oos_r2(move, yhat, day)
            blk["oos_r2"] = r2
            led.add(
                {"test": "oos_r2", "fit": [WINDOWS["exploratory"]], "window": [lo, hi]},
                r2,
                "oos_r2",
                wname,
                "predictability",
                "positive",
            )
        # strategy grid
        blk["strategy"] = {}
        clv_sets = []
        for th in THETAS:
            b = bets_for(f, feat, pc, th)
            r = S.score(b, lo, hi)
            r["outcome_mix"] = b["outcome"].value_counts(normalize=True).round(3).to_dict()
            blk["strategy"][th] = r
            S.ledger_strategy(led, {"theta": th, "bench": "pinnacle_close_power", "window": [lo, hi]}, r, wname, "grid")
            clv_sets.append(pd.Series(np.log(b["odds"] * b["p_close"]), index=b.index))
        if wname == "evaluation":
            pbo_inputs = (f["match_date"], clv_sets)
            for m in ("shin", "multiplicative"):
                pcm = L.devig_cols(f, PIN_C, m)
                r = S.score(bets_for(f, feat, pcm, 0.02), lo, hi)
                blk["strategy"][f"0.02_{m}_close"] = r
                S.ledger_strategy(
                    led, {"theta": 0.02, "bench": f"pinnacle_close_{m}", "window": [lo, hi]}, r, wname, "sensitivity"
                )
    res["evaluation"]["pbo_cscv"] = L.pbo_cscv(S.clv_matrix(*pbo_inputs))

    # post segment: benchmark Betfair Exchange close (Pinnacle close largely missing from 2025/26)
    wname, lo, hi = POST
    f = L.window(eligible(d, BF_C), lo, hi).reset_index(drop=True)
    feat = features(f)
    pc = L.devig_cols(f, BF_C, "power")
    blk = res.setdefault(wname, {"n_matches": len(f), "days": int(f["match_date"].nunique())})
    blk["reference_all_outcomes_clv"] = float(np.log(f[PIN_O].to_numpy() * pc).mean()) if len(f) else None
    blk["strategy"] = {}
    for th in THETAS:
        r = S.score(bets_for(f, feat, pc, th), lo, hi)
        blk["strategy"][th] = r
        S.ledger_strategy(led, {"theta": th, "bench": "betfair_ex_close_power", "window": [lo, hi]}, r, wname, "post")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=_np))
    print(json.dumps(res, indent=1, default=_np))


def main_harness(record: bool) -> None:
    """Score each registered grid point (theta) in the evaluation window as per-decision CLV through
    `pitch_edge.research.run_trial` (see r2-common/harness.py); writes nothing to `trials.csv`."""
    lo, hi = WINDOWS["evaluation"]
    f = L.window(eligible(L.load(), PIN_C), lo, hi).reset_index(drop=True)
    feat = features(f)
    pc = L.devig_cols(f, PIN_C, "power")
    files = [Path(__file__).resolve(), *(HERE.parent / "r2-common" / n for n in ("r2lib.py", "strat.py", "harness.py"))]
    H.run_grid(
        REG,
        L.REPO,
        lambda cfg: (bets_for(f, feat, pc, float(cfg["theta"])), f),
        lambda cfg: lambda oc: oc.get("theta") == float(cfg["theta"]) and oc.get("bench") == "pinnacle_close_power",
        files,
        f"fd.parquet@{L.SNAPSHOT_AT}",
        record,
    )


def _np(o: object) -> object:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


if __name__ == "__main__":
    if "--harness" in sys.argv:
        main_harness(record="--record" in sys.argv)
    else:
        main()
