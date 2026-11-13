"""Post-hoc diagnostics for the R2 studies (labelled post-hoc; not part of any registered grid).

1. Pinnacle open/close overround by season (no outcomes) - explains the level of the "bet at open" CLV.
2. Reference CLV of backing every outcome at max_open / bet365_open / pinnacle_open (no selection), so Q3's
   selected CLV can be read against what the price source gives unconditionally.
3. Q2 confound check: does the consensus gap still predict the Pinnacle move once the Pinnacle open probability
   level and the outcome (home/draw/away) are controlled for? Recorded in Q2's ledger as `post_hoc_diagnostic`.
Writes data/research/r2/diagnostics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2lib as L  # noqa: E402

Q2_REG = "bookmaker-line-structure-open-consensus-gap-v1"
Q2_DIR = L.REPO / "research" / "studies" / "r2-q2-open-consensus-gap"
EVAL = ("2019-07-01", "2025-06-30")
EXPL = ("2012-07-01", "2019-06-30")


def cols(p: str, k: str) -> list[str]:
    return [f"{p}_{o}_{k}" for o in L.OUTCOMES]


def main() -> None:
    d = L.load()
    out: dict = {}
    m = d[d["main"]].copy()
    for k in ("open", "close"):
        m[f"or_{k}"] = (1 / m[cols("pinnacle", k)]).sum(axis=1)
    out["pinnacle_overround_by_season"] = (
        m.groupby("season")[["or_open", "or_close"]].median().round(4).dropna().to_dict(orient="index")
    )

    base = d[
        d["main"]
        & d[
            cols("pinnacle", "open")
            + cols("pinnacle", "close")
            + cols("max", "open")
            + cols("bet365", "open")
            + cols("avg", "open")
        ]
        .gt(1)
        .all(axis=1)
        & d["y_home"].notna()
    ]
    out["reference_clv_all_outcomes"] = {}
    for wname, (lo, hi) in (("exploratory", EXPL), ("evaluation", EVAL)):
        f = L.window(base, lo, hi).reset_index(drop=True)
        pc = L.devig_cols(f, cols("pinnacle", "close"), "power")
        po = L.devig_cols(f, cols("pinnacle", "open"), "power")
        ref = {}
        for src in ("pinnacle", "max", "bet365", "avg"):
            clv = np.log(f[cols(src, "open")].to_numpy() * pc)
            ref[src] = {
                "mean_all_outcomes": float(clv.mean()),
                "mean_by_outcome": dict(zip(L.OUTCOMES, clv.mean(axis=0).round(5).tolist(), strict=True)),
            }
        # Jensen term: mean log(p_close/p_open) of Pinnacle de-vigged probabilities (no margin involved)
        ref["pinnacle_fair_drift_log"] = float(np.log(pc / po).mean())
        ref["n_matches"] = len(f)
        out["reference_clv_all_outcomes"][wname] = ref

        # Q2 confound check
        p_avg = L.devig_cols(f, cols("avg", "open"), "power")
        move = (L.logit(pc) - L.logit(po)).T.reshape(-1)
        g_avg = (L.logit(p_avg) - L.logit(po)).T.reshape(-1)
        g_max = np.log(
            np.minimum(f[cols("max", "open")].to_numpy() / f[cols("pinnacle", "open")].to_numpy(), 1.5)
        ).T.reshape(-1)
        lvl = L.logit(po).T.reshape(-1)
        is_draw = np.repeat([0.0, 1.0, 0.0], len(f))
        is_away = np.repeat([0.0, 0.0, 1.0], len(f))
        day = pd.concat([f["match_date"]] * 3, ignore_index=True)
        o = L.boot_ols(move, [g_avg, g_max, lvl, is_draw, is_away], day)
        names = ["const", "g_avg", "g_max", "logit_p_pin_open", "is_draw", "is_away"]
        out.setdefault("q2_controlled_ols", {})[wname] = {
            n: {"coef": float(o["coef"][i]), "ci_lo": float(o["ci_lo"][i]), "ci_hi": float(o["ci_hi"][i])}
            for i, n in enumerate(names)
        }
        led = L.Ledger(Q2_DIR, Q2_REG)
        led.add(
            {"test": "ols_slope_g_avg_controlled", "controls": names[2:], "window": [lo, hi]},
            {
                "n": len(f),
                "mean": o["coef"][1],
                "ci_lo": o["ci_lo"][1],
                "ci_hi": o["ci_hi"][1],
                "z": o["coef"][1] / o["se"][1],
            },
            "ols_slope_g_avg",
            wname,
            "post_hoc_diagnostic",
            "two-sided",
        )
    (L.DATA / "diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
