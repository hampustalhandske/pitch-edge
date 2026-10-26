"""Build the edgelab `prediction_market` datasets for the R2-Q3 stale-soft-open strategies.

One binary market per match: the outcome with the largest edge log(max price * p_pin_open), when positive, after
the 1.5x data-error cap (a function of the open quotes only). The Yes ask is 1/price of the price source (`max_open` or
`bet365_open`); bid and mid follow from a fixed relative spread. The model probability `pe_model_p` is the
power-de-vigged Pinnacle open. Each market has two snapshots with the same quote, the open hour and the next hour,
so an order filled one bar after its decision fills at the open quote (edgelab accepts only a quote less than one bar
old, and requires a gap-free hourly series between a market's first and last quote, so a closing snapshot would force
an hourly series). See the strategy's notes.md for every modelling choice.

Usage: python build_dataset.py <max|bet365|max-nopersist>
Writes data/research/r2/edgelab/r2q3-<variant>/ (edgelab parquet dataset) and pe_bets.parquet beside it
(pitch-edge's own bets on the same eligible matches, for reconciliation).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parents[1] / "studies"
sys.path.insert(0, str(STUDIES / "r2-q3-stale-soft-open"))
sys.path.insert(0, str(STUDIES / "r2-common"))
sys.path.insert(0, str(STUDIES / "r1-common"))
import common as R1  # noqa: E402  (r1-common: open_known_at)
import r2lib as L  # noqa: E402
import run as Q3  # noqa: E402  (the study's own eligibility, decision and cap code)

from pitch_edge.edgelab import export_prediction_market_dataset  # noqa: E402

WINDOW = ("2012-07-01", "2025-06-30")
SPREAD = 0.02  # relative full spread: bid = ask * (1 - SPREAD); mid = ask * (1 - SPREAD / 2)
SIZE = 1.0e7  # shares quoted on each side: effectively unlimited depth (stake limits are not modelled)
RESOLVE_AFTER = pd.Timedelta(hours=3)
MIN_LEAD = pd.Timedelta(hours=1, minutes=30)
PERSIST = pd.Timedelta(hours=1)
SIDES = ("home", "draw", "away")
TYPE = {"home": "win_home", "draw": "draw", "away": "win_away"}
THETA = 0.02
OUT_ROOT = L.REPO / "data" / "research" / "r2" / "edgelab"
VARIANTS = {"max": ("max", True), "bet365": ("bet365", True), "max-nopersist": ("max", False)}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def kickoff_utc(f: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """UTC kickoff per match. Recorded `kickoff_time` (UK local) when present; else 15:00 on Sat/Sun and 19:45
    on Mon-Fri, UK local. Returns (kickoff, recorded_flag)."""
    rec = f["kickoff_time"].notna()
    dow = f["match_date"].dt.dayofweek
    default = pd.Series(np.where(dow >= 5, "15:00", "19:45"), index=f.index)
    hhmm = f["kickoff_time"].where(rec, default)
    local = pd.to_datetime(f["match_date"].dt.strftime("%Y-%m-%d") + " " + hhmm)
    ko = local.dt.tz_localize(R1.LONDON, ambiguous=False, nonexistent="shift_forward").dt.tz_convert("UTC")
    return ko, rec


def book(ask: np.ndarray, mid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bid, ask) with ask clipped below 1 and bid = 2 mid - ask kept inside (0, ask]."""
    ask = np.minimum(ask, 0.9999)
    bid = np.clip(2 * mid - ask, 1e-4, ask)
    return bid, ask


def main(variant: str) -> None:
    src, persist = VARIANTS[variant]
    d = L.load()
    f = L.window(Q3.eligible(d, "pinnacle", "pinnacle", src), *WINDOW).reset_index(drop=True)
    n_elig = len(f)
    f["ko"], f["ko_recorded"] = kickoff_utc(f)
    f["t0"] = f["ko"].map(R1.open_known_at)
    f["lead"] = f["ko"] - f["t0"]
    keep = f["lead"] >= MIN_LEAD
    n_short = int((~keep).sum())
    f = f[keep].reset_index(drop=True)
    f["key"] = (
        f["division"] + "-" + f["match_date"].dt.strftime("%Y%m%d") + "-" + f["home_team"].map(slug) + "-"
        + f["away_team"].map(slug)
    )  # fmt: skip
    assert f["key"].is_unique, f["key"][f["key"].duplicated()].tolist()[:5]

    ref_o, src_o, cl_c = Q3.cols("pinnacle", "open"), Q3.cols(src, "open"), Q3.cols("pinnacle", "close")
    pin_odds = f[ref_o].to_numpy()
    odds = f[src_o].to_numpy()
    p_open = L.devig_cols(f, ref_o, "power")
    p_close = L.devig_cols(f, cl_c, "power")
    capped = odds / pin_odds > Q3.CAP  # data-error quotes: the outcome has no market (study: edge = -inf)
    y = f[[f"y_{o}" for o in SIDES]].to_numpy()

    # pitch-edge's own decisions on exactly these matches (study rule, theta = 0.02)
    j, e, price = Q3.decide(f, "pinnacle", src, Q3.CAP)
    rows = np.arange(len(f))
    take = e > THETA
    pe = pd.DataFrame(
        {
            "key": f["key"],
            "match_date": f["match_date"],
            "division": f["division"],
            "outcome": np.array(SIDES)[j],
            "odds": price,
            "p_open": p_open[rows, j],
            "p_close": p_close[rows, j],
            "y": y[rows, j],
            "edge_open": e,
            "ko_recorded": f["ko_recorded"],
        }
    )[take]
    pe["market"] = pe["key"] + ":" + pe["outcome"].map(TYPE)

    snaps, res, feats = [], [], []
    lo_half = SPREAD / 2
    n_markets_possible = int((~capped).sum())
    for k, side in enumerate(SIDES):
        ok = (j == k) & (e > 0.0)  # one market per match: its largest-edge outcome, if the edge is positive
        g = f[ok]
        market = (g["key"] + ":" + TYPE[side]).to_numpy()
        t0, ko = g["t0"], g["ko"]
        ask0 = 1.0 / odds[ok, k]
        bid0, ask0 = book(ask0, ask0 * (1 - lo_half))
        # quote one bar after the open: the same quote when it persists, else Pinnacle's own open price
        ask1 = ask0 if persist else 1.0 / pin_odds[ok, k]
        bid1, ask1 = book(ask1, ask1 * (1 - lo_half)) if not persist else (bid0, ask0)
        for ts, b, a in ((t0, bid0, ask0), (t0 + PERSIST, bid1, ask1)):
            snaps.append(pd.DataFrame({"market": market, "ts": ts.to_numpy(), "bid": b, "ask": a, "size": SIZE}))
        res.append(
            pd.DataFrame(
                {
                    "market": market,
                    "resolved_at": (ko + RESOLVE_AFTER).to_numpy(),
                    "outcome": y[ok, k].astype(int),
                    "resolution_source": "football-data ft_result",
                }
            )
        )
        feats.append(
            pd.DataFrame(
                {
                    "market": market,
                    "ts": t0.to_numpy(),
                    "known_at": t0.to_numpy(),
                    "model_p": p_open[ok, k],
                    "kickoff_ts": ko.to_numpy(),
                }
            )
        )
    out = OUT_ROOT / f"r2q3-{variant}"
    rep = export_prediction_market_dataset(
        pd.concat(snaps, ignore_index=True),
        pd.concat(res, ignore_index=True),
        pd.concat(feats, ignore_index=True),
        out,
        source="synthetic",
        include_market_type=False,
    )
    pe.to_parquet(OUT_ROOT / f"r2q3-{variant}-pe_bets.parquet", index=False)
    n_lead = f["lead"].dt.total_seconds().div(3600)
    summary = {
        "variant": variant,
        "source": src,
        "persist": persist,
        "window": WINDOW,
        "eligible_matches": n_elig,
        "dropped_lead_lt_1h30": n_short,
        "matches": len(f),
        "ko_recorded_share": float(f["ko_recorded"].mean()),
        "outcomes_capped_data_error": int(capped.sum()),
        "outcome_markets_before_selection": n_markets_possible,
        "candidate_markets_exported": int((e > 0.0).sum()),
        "pe_bets_theta_0.02": int(take.sum()),
        "max_lead_h": float(n_lead.max()),
        "export": {k: str(v) for k, v in rep.__dict__.items()},
    }
    (OUT_ROOT / f"r2q3-{variant}-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
