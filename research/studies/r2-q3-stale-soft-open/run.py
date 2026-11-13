"""Q3 — stale soft lines: does the best open price beat the Pinnacle-open fair price, and then the close?

Registration: research/registry/bookmaker-line-structure-stale-soft-open-v1.yaml.
Decision at football-data.co.uk's pre-closing snapshot; the fair reference is the power-de-vigged Pinnacle
price at that same snapshot (Betfair Exchange open in the 2025-07+ post segment). Only `*_open` columns feed
the decision (asserted). Writes trial rows to `trials.csv` beside this file and outputs to data/research/r2/q3/.
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

REG = "bookmaker-line-structure-stale-soft-open-v1"
HERE = Path(__file__).resolve().parent
OUT = L.DATA / "q3"
WINDOWS = {"exploratory": ("2012-07-01", "2019-06-30"), "evaluation": ("2019-07-01", "2025-06-30")}
POST = ("post_segment", "2025-07-01", "2026-10-19")
THETAS = [0.00, 0.02, 0.05]
SOURCES = {"max_open": "max", "bet365_open": "bet365"}
CAP = 1.5
Y = [f"y_{o}" for o in L.OUTCOMES]


def cols(prefix: str, kind: str) -> list[str]:
    return [f"{prefix}_{o}_{kind}" for o in L.OUTCOMES]


def eligible(d: pd.DataFrame, ref: str, bench: str, src: str) -> pd.DataFrame:
    c = cols(ref, "open") + cols(bench, "close") + cols(src, "open")
    ok = d["main"] & d[c].notna().all(axis=1) & (d[c] > 1).all(axis=1) & d["y_home"].notna()
    return d[ok].reset_index(drop=True)


def decide(f: pd.DataFrame, ref: str, src: str, cap: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per match: chosen outcome index, its edge log(price * p_ref_open) and its price; ineligible -> -inf."""
    ref_o, src_o = cols(ref, "open"), cols(src, "open")
    L.assert_open_only(ref_o + src_o)
    p_ref = L.devig_cols(f, ref_o, "power")
    price = f[src_o].to_numpy()
    edge = np.log(price * p_ref)
    if cap is not None:
        edge = np.where(price / f[ref_o].to_numpy() > cap, -np.inf, edge)
    j = np.argmax(edge, axis=1)
    rows = np.arange(len(f))
    return j, edge[rows, j], price[rows, j]


def bets_for(f: pd.DataFrame, ref: str, bench: str, src: str, theta: float, cap: float | None) -> pd.DataFrame:
    j, e, price = decide(f, ref, src, cap)
    rows = np.arange(len(f))
    pc = L.devig_cols(f, cols(bench, "close"), "power")[rows, j]
    b = pd.DataFrame(
        {
            "match_date": f["match_date"],
            "division": f["division"],
            "odds": price,
            "p_close": pc,
            "y": f[Y].to_numpy()[rows, j],
            "edge_open": e,
            "outcome": np.array(L.OUTCOMES)[j],
        }
    )
    return b[e > theta]


def describe(f: pd.DataFrame, ref: str, bench: str, src: str, cap: float | None) -> dict:
    """How often any outcome's soft price beats the open fair price, and how often it also beats the close fair."""
    ref_o, src_o = cols(ref, "open"), cols(src, "open")
    p_ref = L.devig_cols(f, ref_o, "power")
    pc = L.devig_cols(f, cols(bench, "close"), "power")
    price = f[src_o].to_numpy()
    valid = np.ones_like(price, dtype=bool) if cap is None else price / f[ref_o].to_numpy() <= cap
    beats_open = (price * p_ref > 1) & valid
    beats_both = beats_open & (price * pc > 1)
    return {
        "n_matches": len(f),
        "share_matches_any_beats_open_fair": float(beats_open.any(axis=1).mean()),
        "share_outcomes_beats_open_fair": float(beats_open.mean()),
        "share_of_those_also_beating_close_fair": float(beats_both.sum() / max(beats_open.sum(), 1)),
        "share_outcomes_capped_as_error": float((~valid).mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = L.load()
    led = L.Ledger(HERE, REG)
    res: dict = {"snapshot": L.SNAPSHOT_AT, "git_sha": led.sha, "registration_sha256": led.reg_sha}
    for wname, (lo, hi) in WINDOWS.items():
        blk = res.setdefault(wname, {})
        for sname, src in SOURCES.items():
            f = L.window(eligible(d, "pinnacle", "pinnacle", src), lo, hi).reset_index(drop=True)
            blk[sname] = {"describe": describe(f, "pinnacle", "pinnacle", src, CAP), "strategy": {}}
            for th in THETAS:
                b = bets_for(f, "pinnacle", "pinnacle", src, th, CAP)
                r = S.score(b, lo, hi)
                r["outcome_mix"] = b["outcome"].value_counts(normalize=True).round(3).to_dict()
                r["mean_edge_open"] = float(b["edge_open"].mean()) if len(b) else None
                blk[sname]["strategy"][th] = r
                S.ledger_strategy(
                    led,
                    {
                        "source": sname,
                        "theta": th,
                        "cap": CAP,
                        "ref": "pinnacle_open_power",
                        "bench": "pinnacle_close_power",
                        "window": [lo, hi],
                    },
                    r,
                    wname,
                    "grid",
                )
        if wname == "evaluation":
            # PBO over the 6-point grid on the matches where both price sources exist (aligned rows)
            both = eligible(d, "pinnacle", "pinnacle", "max")
            both = both[
                both[cols("bet365", "open")].notna().all(axis=1) & (both[cols("bet365", "open")] > 1).all(axis=1)
            ]
            fb = L.window(both, lo, hi).reset_index(drop=True)
            sets = []
            for src in SOURCES.values():
                for th in THETAS:
                    b = bets_for(fb, "pinnacle", "pinnacle", src, th, CAP)
                    sets.append(pd.Series(np.log(b["odds"] * b["p_close"]), index=b.index))
            res[wname]["pbo_cscv"] = L.pbo_cscv(S.clv_matrix(fb["match_date"], sets))
            f = L.window(eligible(d, "pinnacle", "pinnacle", "max"), lo, hi).reset_index(drop=True)
            for cap in (None, 1.25):
                r = S.score(bets_for(f, "pinnacle", "pinnacle", "max", 0.02, cap), lo, hi)
                blk["max_open"]["strategy"][f"0.02_cap_{cap}"] = r
                S.ledger_strategy(
                    led,
                    {
                        "source": "max_open",
                        "theta": 0.02,
                        "cap": cap,
                        "ref": "pinnacle_open_power",
                        "bench": "pinnacle_close_power",
                        "window": [lo, hi],
                    },
                    r,
                    wname,
                    "sensitivity",
                )
    wname, lo, hi = POST
    blk = res.setdefault(wname, {})
    for sname, src in SOURCES.items():
        f = L.window(eligible(d, "betfair_ex", "betfair_ex", src), lo, hi).reset_index(drop=True)
        blk[sname] = {"describe": describe(f, "betfair_ex", "betfair_ex", src, CAP), "strategy": {}}
        for th in THETAS:
            b = bets_for(f, "betfair_ex", "betfair_ex", src, th, CAP)
            r = S.score(b, lo, hi)
            r["mean_edge_open"] = float(b["edge_open"].mean()) if len(b) else None
            blk[sname]["strategy"][th] = r
            S.ledger_strategy(
                led,
                {
                    "source": sname,
                    "theta": th,
                    "cap": CAP,
                    "ref": "betfair_ex_open_power",
                    "bench": "betfair_ex_close_power",
                    "window": [lo, hi],
                },
                r,
                wname,
                "post",
            )
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=_np))
    print(json.dumps(res, indent=1, default=_np))


def main_harness(record: bool) -> None:
    """Score each registered grid point (price_source x theta) in the evaluation window as per-decision CLV
    through `pitch_edge.research.run_trial` (see r2-common/harness.py); writes nothing to `trials.csv`."""
    lo, hi = WINDOWS["evaluation"]
    d = L.load()
    frames = {
        s: L.window(eligible(d, "pinnacle", "pinnacle", src), lo, hi).reset_index(drop=True)
        for s, src in SOURCES.items()
    }
    files = [Path(__file__).resolve(), *(HERE.parent / "r2-common" / n for n in ("r2lib.py", "strat.py", "harness.py"))]

    def make(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        f = frames[cfg["price_source"]]
        return bets_for(f, "pinnacle", "pinnacle", SOURCES[cfg["price_source"]], float(cfg["theta"]), CAP), f

    H.run_grid(
        REG,
        L.REPO,
        make,
        lambda cfg: (
            lambda oc: (
                oc.get("source") == cfg["price_source"]
                and oc.get("theta") == float(cfg["theta"])
                and oc.get("cap") == CAP
                and oc.get("bench") == "pinnacle_close_power"
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
