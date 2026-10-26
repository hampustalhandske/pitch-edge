"""r5-xv-arb-v1: paper cross-venue arbitrage between Kalshi and Polymarket soccer 1X2, one-hour execution delay.

At each hour H (ko - 48h <= H, H + 1h <= ko) for each event, using only quotes known at H:
  cost of buying Yes on Kalshi  = ask_K + 0.07 a(1-a)        No on Kalshi  = (1 - bid_K) + fee
  cost of buying Yes on PM      = ask_P + 0.05 a(1-a)        No on PM      = (1 - bid_P) + fee
  cross:  for one outcome, Yes on one venue + No on the other -> pays exactly 1; edge = 1 - cost
  dutch:  Yes on all three outcomes, each on its cheaper venue -> pays exactly 1; edge = 1 - cost;
          or No on all three, each on its cheaper venue -> pays exactly 2; edge = 2 - cost
The best combination of each type at H is a signal when its edge > theta. It is executed with the SAME legs
at the next hour's quotes (H + 1h, still before kickoff); its paper P&L per unit payout is the edge
recomputed at H + 1h (it can be negative: the combination still pays the fixed amount, but it may have cost
more than that). A signal whose legs have no quote at H + 1h is counted as unfilled and not scored.

Primary pricing: the true Polymarket book (bid/ask in force at H; book coverage 2026-04-09 -> 2026-06-10 and
2026-09-29). Secondary (`proxy`): Polymarket ask/bid = last trade +/- max(0.5c, half the bar's high-low),
fresh trade required in the hour, over the whole window — optimistic or pessimistic depending on where the
trade printed, an approximation only.

Usage: arch -arm64 uv run python research/studies/r5-xv-arb-v1/run.py {exploratory|evaluation}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r5-common"))

from harness import record  # noqa: E402
from partb import CODE_FILES, R5, SIDES, panel, snapshot_tag, universe  # noqa: E402
from prices import kalshi_fee, pm_fee  # noqa: E402

REG_ID = "r5-xv-arb-v1"
THETAS = [0.0, 0.01, 0.02]
TYPES = ["cross", "dutch"]


def wide(pan: pd.DataFrame, pricing: str) -> pd.DataFrame:
    """Per (event, H): bid/ask on each venue for each side, at H and at H + 1h (suffix _n)."""
    p = pan.copy()
    if pricing == "book":
        p["pb"], p["pa"] = p.b_bid, p.b_ask
    else:
        hs = np.maximum(0.005, (p.hi - p.lo) / 2)
        p["pb"], p["pa"] = (p.p_fresh - hs).clip(lower=0.001), (p.p_fresh + hs).clip(upper=0.999)
    p = p.rename(columns={"bid": "kb", "ask": "ka"})
    cols = ["kb", "ka", "pb", "pa"]
    w = p.pivot_table(index=["event_ticker", "H"], columns="side", values=cols, aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reindex(columns=[f"{c}_{s}" for c in cols for s in SIDES])
    meta = p.groupby(["event_ticker", "H"]).agg(
        ko=("ko", "first"), match_day=("match_day", "first"), series=("series", "first")
    )
    w = w.join(meta).reset_index()
    nxt = w[["event_ticker", "H"] + [f"{c}_{s}" for c in cols for s in SIDES]].copy()
    nxt["H"] = nxt.H - pd.Timedelta(hours=1)
    w = w.merge(nxt, on=["event_ticker", "H"], how="left", suffixes=("", "_n"))
    w = w[w.H + pd.Timedelta(hours=1) <= w.ko]
    return w.reset_index(drop=True)


def leg_costs(w: pd.DataFrame, suf: str) -> dict[str, np.ndarray]:
    """All-in cost of each single leg: yes/no x venue x side."""
    out = {}
    for s in SIDES:
        ka, kb = w[f"ka_{s}{suf}"].to_numpy(float), w[f"kb_{s}{suf}"].to_numpy(float)
        pa, pb = w[f"pa_{s}{suf}"].to_numpy(float), w[f"pb_{s}{suf}"].to_numpy(float)
        out[f"yes_K_{s}"] = ka + kalshi_fee(ka)
        out[f"no_K_{s}"] = (1 - kb) + kalshi_fee(1 - kb)
        out[f"yes_P_{s}"] = pa + pm_fee(pa)
        out[f"no_P_{s}"] = (1 - pb) + pm_fee(1 - pb)
    return out


def combos() -> dict[str, list[tuple[float, list[str]]]]:
    """Every riskless combination: type -> [(payout, legs)]."""
    cross = []
    for s in SIDES:
        cross.append((1.0, [f"yes_K_{s}", f"no_P_{s}"]))
        cross.append((1.0, [f"yes_P_{s}", f"no_K_{s}"]))
    dutch = []
    for kind, payout in (("yes", 1.0), ("no", 2.0)):
        for venues in np.array(np.meshgrid(*[["K", "P"]] * 3)).T.reshape(-1, 3):
            dutch.append((payout, [f"{kind}_{v}_{s}" for v, s in zip(venues, SIDES, strict=True)]))
    return {"cross": cross, "dutch": dutch}


def signals(w: pd.DataFrame, typ: str) -> pd.DataFrame:
    """Best combination of `typ` per (event, H): its edge at H and the edge of the same legs at H + 1h."""
    now, nxt = leg_costs(w, ""), leg_costs(w, "_n")
    best_edge = np.full(len(w), -np.inf)
    best_next = np.full(len(w), np.nan)
    best_name = np.array([""] * len(w), dtype=object)
    for payout, legs in combos()[typ]:
        e = payout - sum(now[leg] for leg in legs)
        en = payout - sum(nxt[leg] for leg in legs)
        e = np.where(np.isfinite(e), e, -np.inf)
        better = e > best_edge
        best_edge = np.where(better, e, best_edge)
        best_next = np.where(better, en, best_next)
        best_name = np.where(better, "+".join(legs), best_name)
    if typ == "dutch":
        # Both venues must quote every outcome at H, so a dutch signal is a choice between venues and never a
        # single-venue arb that exists only because the other venue was unpriced.
        quoted = np.isfinite(w[[f"{c}_{s}" for c in ("kb", "ka", "pb", "pa") for s in SIDES]].to_numpy(float)).all(
            axis=1
        )
        best_edge = np.where(quoted, best_edge, -np.inf)
    out = w[["event_ticker", "H", "match_day", "series"]].copy()
    out["edge"], out["pnl"], out["legs"] = best_edge, best_next, best_name
    out["cross_venue"] = [("_K_" in n) and ("_P_" in n) for n in best_name]
    return out


def main(window: str) -> None:
    u, drops = universe()
    u = u[u.window == window]
    pan = panel(u)
    detail: dict = {"window": window, "events": len(u), "trials": {}}
    for pricing in ("book", "proxy"):
        w = wide(pan, pricing)
        detail[f"{pricing}_event_hours_priced"] = int(
            np.isfinite(w[[f"pa_{s}" for s in SIDES]].to_numpy()).all(axis=1).sum()
        )
        for typ in TYPES:
            sg = signals(w, typ)
            sg = sg[np.isfinite(sg.edge)]
            detail[f"{pricing}_{typ}_edge_quantiles"] = (
                {str(q): float(sg.edge.quantile(q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)} if len(sg) else None
            )
            for th in THETAS:
                s = sg[sg.edge > th]
                filled = s[np.isfinite(s.pnl)].reset_index(drop=True)
                key = f"{pricing}|{typ}|{th}"
                d = {
                    "signals": len(s),
                    "filled": len(filled),
                    "events": int(filled.event_ticker.nunique()),
                    "days": int(filled.match_day.nunique()),
                    "mean_edge_at_signal": float(s.edge.mean()) if len(s) else None,
                    "mean_pnl": float(filled.pnl.mean()) if len(filled) else None,
                    "total_pnl_per_unit": float(filled.pnl.sum()) if len(filled) else 0.0,
                    "share_pnl_positive": float((filled.pnl > 0).mean()) if len(filled) else None,
                    "share_cross_venue": float(filled.cross_venue.mean()) if len(filled) else None,
                    "by_series": filled.groupby("series")
                    .pnl.agg(["size", "mean"])
                    .sort_values("size", ascending=False)
                    .head(10)
                    .to_dict(orient="index")
                    if len(filled)
                    else {},
                }
                detail["trials"][key] = d
                if window != "evaluation" or len(filled) < 2:
                    continue
                cfg = {"type": typ, "theta": th}
                metric = "pnl_per_signal_book" if pricing == "book" else "pnl_per_signal_proxy"
                v = filled.pnl.to_numpy()
                thirds = np.array_split(np.sort(filled.match_day.unique()), 3)
                vc = filled.series.value_counts()
                extra = {
                    "clean": True,
                    "return_like": True,
                    "within_ceiling": bool(np.mean(v) <= 0.05),
                    "subperiod_estimates": [
                        float(v[filled.match_day.isin(t).to_numpy()].mean()) if filled.match_day.isin(t).any() else None
                        for t in thirds
                    ],
                    "leave_one_league_out": {
                        sname: float(v[(filled.series != sname).to_numpy()].mean())
                        for sname in vc[vc >= 20].index
                        if (filled.series != sname).any()
                    },
                    "describe": d,
                    "pricing": pricing,
                }
                frame = pd.DataFrame(
                    {
                        "decision_id": filled.event_ticker + "|" + filled.H.dt.strftime("%Y-%m-%dT%H"),
                        "block": filled.match_day,
                        "value": v,
                    }
                )
                record(REG_ID, cfg, frame, metric, snapshot_tag(), extra, CODE_FILES + [Path(__file__)])
    out = R5 / REG_ID
    out.mkdir(parents=True, exist_ok=True)
    (out / f"detail_{window}.json").write_text(json.dumps(detail, indent=2, default=str))
    print(json.dumps({k: v for k, v in detail.items() if k != "trials"}, indent=1, default=str))
    print(
        pd.DataFrame(detail["trials"])
        .T[
            [
                "signals",
                "filled",
                "events",
                "mean_edge_at_signal",
                "mean_pnl",
                "total_pnl_per_unit",
                "share_cross_venue",
            ]
        ]
        .to_string()
    )


if __name__ == "__main__":
    main(sys.argv[1])
