"""H1 `soft-open-vs-betfair-fav-open-v1`: back the best soft open price on the Betfair-open favourite.

Decision (football-data pre-closing snapshot, 1-3 days before kickoff), from `*_open` columns only:
  p_bf_open = power de-vig of betfair_ex_{home,draw,away}_open; fav = argmax p_bf_open;
  edge = log(max_fav_open * p_bf_open[fav]); bet 1 unit at max_fav_open if p_bf_open[fav] >= p_min and edge > theta.
A quote with max_fav_open / betfair_ex_fav_open > 1.5 is a data error and the match is not eligible.
Benchmark: power de-vig of betfair_ex_{home,draw,away}_close; CLV = log(odds_taken * p_close[fav]).
Selection of (theta, p_min) uses the exploratory window only; see `select`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "soft-open-vs-betfair-common"))
import sobf  # noqa: E402

REG_ID = "soft-open-vs-betfair-fav-open-v1"
STUDY = "soft-open-vs-betfair-fav-open"
OUT = sobf.REPO / "data" / "research" / STUDY
HERE = Path(__file__).resolve().parent

EXPLORATORY = ("2024-08-01", "2025-06-30")
THETAS = (0.0, 0.01)
P_MINS = (0.65, 0.70, 0.75)
MIN_TRAIN_BETS = 100
FALLBACK = {"theta": 0.0, "p_min": 0.70}
CAP = 1.5

OPEN_COLS = [f"{b}_{o}_open" for b in ("betfair_ex", "max") for o in sobf.OUTCOMES]
CLOSE_COLS = [f"betfair_ex_{o}_close" for o in sobf.OUTCOMES]
ID_COLS = ["division", "match_date", "home_team", "away_team"]
PULL_COLS = ID_COLS + ["ft_result"] + OPEN_COLS + CLOSE_COLS
CODE = {"home": "H", "draw": "D", "away": "A"}


def pull_queries() -> dict[str, str]:
    """SQL for the two parquet files: `train` (exploratory window) and `eval` (evaluation window)."""
    base = f"SELECT {', '.join(PULL_COLS)} FROM fd_matches WHERE {sobf.MAIN_LEAGUE_SQL}"
    return {
        "train": f"{base} AND match_date BETWEEN DATE '{EXPLORATORY[0]}' AND DATE '{EXPLORATORY[1]}'",
        "eval": f"{base} AND match_date BETWEEN DATE '{sobf.EVAL_WINDOW[0]}' AND DATE '{sobf.EVAL_WINDOW[1]}'",
    }


def decide(open_cols: pd.DataFrame) -> pd.DataFrame:
    """Per match, from open columns only: valid, fav (0/1/2), p_fav, price (max_fav_open), edge, ratio, eligible.

    Raises AssertionError if `open_cols` holds anything but the whitelisted open columns. `valid` means all six open
    quotes are > 1; `eligible` additionally requires max_fav_open / betfair_ex_fav_open <= 1.5. Ties in p_bf_open
    go to the first of home, draw, away.
    """
    sobf.assert_whitelist(open_cols.columns, OPEN_COLS)
    bf = sobf.triple(open_cols, "betfair_ex", "open")
    mx = sobf.triple(open_cols, "max", "open")
    valid = sobf.odds_ok(bf) & sobf.odds_ok(mx)
    p = np.full(bf.shape, np.nan)
    p[valid] = sobf.devig_power_rows(bf[valid])
    fav = np.zeros(len(bf), dtype=int)
    fav[valid] = np.argmax(p[valid], axis=1)
    rows = np.arange(len(bf))
    p_fav, price, bf_fav = p[rows, fav], mx[rows, fav], bf[rows, fav]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = price / bf_fav
        edge = np.log(price * p_fav)
    return pd.DataFrame(
        {
            "valid": valid,
            "fav": fav,
            "p_fav": p_fav,
            "price": price,
            "edge": edge,
            "ratio": ratio,
            "eligible": valid & ~(ratio > CAP),
        },
        index=open_cols.index,
    )


def universe(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Matches with every open and close quote present, their decisions and the (n, 3) close probabilities.

    Returns (frame restricted to rows whose six open and three close quotes are valid, decisions, p_close).
    Decisions are computed from `f[OPEN_COLS]` only; the close columns are touched only to build the benchmark.
    """
    close = sobf.triple(f, "betfair_ex", "close")
    dec = decide(f[OPEN_COLS])
    keep = (dec["valid"].to_numpy()) & sobf.odds_ok(close)
    f2, dec2 = f[keep], dec[keep]
    return f2, dec2, sobf.devig_power_rows(close[keep])


def bets_for(f: pd.DataFrame, dec: pd.DataFrame, p_close: np.ndarray, theta: float, p_min: float) -> pd.DataFrame:
    """Bets of one (theta, p_min): eligible matches with p_fav >= p_min and edge > theta; at most one per match."""
    take = dec["eligible"].to_numpy() & (dec["p_fav"].to_numpy() >= p_min) & (dec["edge"].to_numpy() > theta)
    return sobf.bets_frame(f, dec["fav"].to_numpy(), dec["price"].to_numpy(), p_close, take)


def candidates() -> list[dict[str, float]]:
    """The six registered candidates in grid order (theta outer, p_min inner)."""
    return [{"theta": t, "p_min": m} for t in THETAS for m in P_MINS]


def select_key(r: dict[str, Any]) -> tuple[float, float, float]:
    """Sort key: highest mean CLV (rounded to 12 decimals against float noise), then lower theta, then lower p_min."""
    return (-round(float(r["mean_clv"]), 12), r["theta"], r["p_min"])


def select(train: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose (theta, p_min) on the exploratory window only.

    `train` is filtered to the exploratory window here as a guard. Each candidate's train bets, mean CLV and CI are
    returned as rows (for `trials.csv`); the selected one is the eligible candidate (>= 100 train bets) with the
    highest mean CLV, ties to lower theta then lower p_min; with none eligible the fallback (0.0, 0.70).
    Returns (selection dict, candidate rows).
    """
    t = sobf.in_window(train, *EXPLORATORY)
    f, dec, pc = universe(t)
    n_elig = int(dec["eligible"].sum())
    rows = []
    for c in candidates():
        b = bets_for(f, dec, pc, c["theta"], c["p_min"])
        s = sobf.boot(b["clv"].to_numpy(), b["match_date"].dt.strftime("%Y-%m-%d"))
        rows.append(
            {
                **c,
                "n_bets": int(s["n"]),
                "eligible": bool(s["n"] >= MIN_TRAIN_BETS),
                "mean_clv": s["mean"] if s["n"] else float("nan"),
                "total_clv": float(b["clv"].sum()),
                "ci_lo": s["lo"],
                "ci_hi": s["hi"],
                "share_eligible_matches_bet": float(s["n"] / n_elig) if n_elig else float("nan"),
            }
        )
    best = sobf.pick(rows, select_key, MIN_TRAIN_BETS)
    if best is None:
        chosen, fallback = dict(FALLBACK), True
    else:
        chosen, fallback = {"theta": best["theta"], "p_min": best["p_min"]}, False
    sel = {**chosen, "fallback": fallback, "n_train_eligible_matches": n_elig, "candidates": rows}
    return sel, rows
