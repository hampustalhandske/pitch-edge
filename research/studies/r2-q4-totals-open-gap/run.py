"""Q4 — the Q2 consensus-gap question for over/under 2.5 goals, plus an Asian-handicap feasibility check.

Registration: research/registry/bookmaker-line-structure-totals-open-gap-v1.yaml.
Decision at football-data.co.uk's pre-closing snapshot; only `*_open` columns feed the decision (asserted).
Asian handicap CLV exists only where the close is quoted at the same line as the open, which selects on a
hindsight quantity, so AH is capped at L1 and reported as feasibility. Writes trial rows to `trials.csv`
beside this file and outputs to data/research/r2/q4/.
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

REG = "bookmaker-line-structure-totals-open-gap-v1"
HERE = Path(__file__).resolve().parent
OUT = L.DATA / "q4"
WINDOWS = {
    "exploratory": ("2019-07-01", "2021-06-30"),
    "evaluation": ("2021-07-01", "2025-06-30"),
    "post_segment": ("2025-07-01", "2026-10-19"),
}
THETAS = [0.00, 0.01, 0.02, 0.03, 0.05]
AH_THETAS = [0.00, 0.02]

MARKETS = {
    "ou25": {
        "pin_o": ["pinnacle_over25_open", "pinnacle_under25_open"],
        "avg_o": ["avg_over25_open", "avg_under25_open"],
        "pin_c": ["pinnacle_over25_close", "pinnacle_under25_close"],
        "sides": ("over", "under"),
    },
    "ah": {
        "pin_o": ["pinnacle_ah_home_open", "pinnacle_ah_away_open"],
        "avg_o": ["avg_ah_home_open", "avg_ah_away_open"],
        "pin_c": ["pinnacle_ah_home_close", "pinnacle_ah_away_close"],
        "sides": ("home", "away"),
    },
}


def eligible(d: pd.DataFrame, mk: str) -> pd.DataFrame:
    m = MARKETS[mk]
    c = m["pin_o"] + m["avg_o"] + m["pin_c"]
    ok = d["main"] & d[c].notna().all(axis=1) & (d[c] > 1).all(axis=1)
    if mk == "ou25":
        ok &= d["y_over25"].notna()
    else:
        # hindsight selection, registered: the close must be quoted at the open line to compare prices
        ok &= d["ah_line_open"].notna() & (d["ah_line_open"] == d["ah_line_close"])
    return d[ok].reset_index(drop=True)


def bets_for(f: pd.DataFrame, mk: str, theta: float) -> tuple[pd.DataFrame, dict]:
    """Back side 0 (over / home) if consensus-minus-Pinnacle gap > theta, side 1 if gap < -theta."""
    m = MARKETS[mk]
    L.assert_open_only(m["pin_o"] + m["avg_o"])
    p_pin = L.devig_cols(f, m["pin_o"], "power")
    p_avg = L.devig_cols(f, m["avg_o"], "power")
    pc = L.devig_cols(f, m["pin_c"], "power")
    gap = p_avg[:, 0] - p_pin[:, 0]
    side = np.where(gap > 0, 0, 1)
    take = np.abs(gap) > theta
    rows = np.arange(len(f))
    if mk == "ou25":
        y0 = f["y_over25"].to_numpy()
        y = np.where(side == 0, y0, 1 - y0)
    else:
        y = np.full(len(f), np.nan)  # AH settlement (pushes, quarter lines) not modelled: CLV only
    b = pd.DataFrame(
        {
            "match_date": f["match_date"],
            "division": f["division"],
            "odds": f[m["pin_o"]].to_numpy()[rows, side],
            "p_close": pc[rows, side],
            "y": y,
            "side": np.array(m["sides"])[side],
        }
    )
    move = L.logit(pc[:, 0]) - L.logit(p_pin[:, 0])
    g = L.logit(p_avg[:, 0]) - L.logit(p_pin[:, 0])
    return b[take], {"move": move, "g": g}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = L.load()
    led = L.Ledger(HERE, REG)
    res: dict = {"snapshot": L.SNAPSHOT_AT, "git_sha": led.sha, "registration_sha256": led.reg_sha}
    for mk, thetas in (("ou25", THETAS), ("ah", AH_THETAS)):
        base = eligible(d, mk)
        fitted = None
        pbo_inputs = None
        for wname, (lo, hi) in WINDOWS.items():
            f = L.window(base, lo, hi).reset_index(drop=True)
            blk = res.setdefault(mk, {}).setdefault(
                wname, {"n_matches": len(f), "days": int(f["match_date"].nunique())}
            )
            if len(f) == 0:
                continue
            label = (
                "ah_feasibility_hindsight_selected" if mk == "ah" else ("post" if wname == "post_segment" else "grid")
            )
            _, mv = bets_for(f, mk, 0.0)
            ols = L.boot_ols(mv["move"], [mv["g"]], f["match_date"])
            blk["ols"] = {"coef": ols["coef"], "ci_lo": ols["ci_lo"], "ci_hi": ols["ci_hi"]}
            if mk == "ou25" and wname != "post_segment":
                led.add(
                    {"market": mk, "test": "ols_slope_g", "window": [lo, hi]},
                    {
                        "n": len(f),
                        "mean": ols["coef"][1],
                        "ci_lo": ols["ci_lo"][1],
                        "ci_hi": ols["ci_hi"][1],
                        "z": ols["coef"][1] / ols["se"][1],
                    },
                    "ols_slope_g",
                    wname,
                    "predictability",
                    "two-sided",
                )
            if wname == "exploratory":
                fitted = ols["coef"]
            elif mk == "ou25" and wname == "evaluation":
                r2 = L.boot_oos_r2(mv["move"], fitted[0] + fitted[1] * mv["g"], f["match_date"])
                blk["oos_r2"] = r2
                led.add(
                    {"market": mk, "test": "oos_r2", "window": [lo, hi]},
                    r2,
                    "oos_r2",
                    wname,
                    "predictability",
                    "positive",
                )
            blk["strategy"] = {}
            sets = []
            for th in thetas:
                b, _ = bets_for(f, mk, th)
                r = S.score(b, lo, hi)
                r["side_mix"] = b["side"].value_counts(normalize=True).round(3).to_dict()
                blk["strategy"][th] = r
                S.ledger_strategy(
                    led,
                    {"market": mk, "theta": th, "bench": "pinnacle_close_power", "window": [lo, hi]},
                    r,
                    wname,
                    label,
                )
                sets.append(pd.Series(np.log(b["odds"] * b["p_close"]), index=b.index))
            if mk == "ou25" and wname == "evaluation":
                pbo_inputs = (f["match_date"], sets)
        if pbo_inputs is not None:
            res[mk]["evaluation"]["pbo_cscv"] = L.pbo_cscv(S.clv_matrix(*pbo_inputs))
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=_np))
    print(json.dumps(res, indent=1, default=_np))


def main_harness(record: bool) -> None:
    """Score each registered grid point (theta, O/U 2.5) in the evaluation window as per-decision CLV through
    `pitch_edge.research.run_trial` (see r2-common/harness.py); writes nothing to `trials.csv`."""
    lo, hi = WINDOWS["evaluation"]
    f = L.window(eligible(L.load(), "ou25"), lo, hi).reset_index(drop=True)
    files = [Path(__file__).resolve(), *(HERE.parent / "r2-common" / n for n in ("r2lib.py", "strat.py", "harness.py"))]
    H.run_grid(
        REG,
        L.REPO,
        lambda cfg: (bets_for(f, "ou25", float(cfg["theta"]))[0], f),
        lambda cfg: (
            lambda oc: (
                oc.get("market") == "ou25"
                and oc.get("theta") == float(cfg["theta"])
                and oc.get("bench") == "pinnacle_close_power"
                and "test" not in oc
            )
        ),
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
