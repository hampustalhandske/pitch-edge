"""Join rates and missingness for R4: how much of the Polymarket 1X2 universe each crowd source reaches, and whether
the joined matches differ from the rest by league or Polymarket liquidity. Reads data/research/r4/, writes
`coverage.json` and prints it. No target (price move) is computed here.

Liquidity proxy: Polymarket trades on the three outcome markets in the 24h before kickoff - 1h (bars known by
then), from `pm_bars.parquet` (available for kickoffs 2026-01-05 -> 2026-10-20).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from panel import DATA, NS_H, SIDES, Bars, Bluesky  # noqa: E402

REDDIT_FROM, REDDIT_TO = pd.Timestamp("2025-08-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")
BSKY_FROM, BSKY_TO = pd.Timestamp("2026-01-20", tz="UTC"), pd.Timestamp("2026-10-20", tz="UTC")


def liquidity(g: pd.DataFrame, bars: Bars) -> pd.Series:
    out = []
    for r in g.itertuples(index=False):
        t = pd.Timestamp(r.ko).value - NS_H
        out.append(sum(bars.trades_between(getattr(r, f"cond_{s}"), t - 24 * NS_H, t) for s in SIDES))
    return pd.Series(out, index=g.index)


def by_league(g: pd.DataFrame, flag: str, top: int = 12) -> dict:
    t = g.groupby("league_code")[flag].agg(["size", "sum"]).sort_values("size", ascending=False)
    t["rate"] = (t["sum"] / t["size"]).round(3)
    return {
        k: {"n": int(v["size"]), "joined": int(v["sum"]), "rate": float(v["rate"])} for k, v in t.head(top).iterrows()
    }


def liq_compare(g: pd.DataFrame, flag: str) -> dict:
    out = {}
    for lab, s in (("joined", g[g[flag]]), ("not_joined", g[~g[flag]])):
        out[lab] = {
            "n": len(s),
            "median_trades": float(s.liq.median()),
            "share_zero_trades": float((s.liq == 0).mean()),
        }
    # within-league comparison: median log-liquidity gap joined - not joined, averaged over leagues with both
    gaps = []
    for _, s in g.groupby("league_code"):
        a, b = s[s[flag]], s[~s[flag]]
        if len(a) >= 5 and len(b) >= 5:
            gaps.append(np.log1p(a.liq).median() - np.log1p(b.liq).median())
    out["within_league_median_log1p_gap"] = float(np.mean(gaps)) if gaps else None
    out["leagues_compared"] = len(gaps)
    return out


def main() -> None:
    groups = pd.read_parquet(DATA / "pm_groups.parquet")
    links = pd.read_parquet(DATA / "reddit_links.parquet")
    pre = pd.read_parquet(DATA / "reddit_preko.parquet")
    bars = Bars(pd.read_parquet(DATA / "pm_bars.parquet"))
    bsky = Bluesky(pd.read_parquet(DATA / "bsky_hourly.parquet"), pd.read_parquet(DATA / "bluesky_queries.parquet"))
    rep: dict = {}

    g = groups[(groups.ko >= REDDIT_FROM) & (groups.ko < REDDIT_TO)].copy()
    mt = links[links.kind == "match"]
    g["reddit"] = g.group_id.isin(set(mt.group_id))
    pm = pre[pre.kind == "match"]
    rep["reddit"] = {
        "pm_groups": len(g),
        "linked_match_threads": int(g.reddit.sum()),
        "join_rate": round(float(g.reddit.mean()), 4),
        "by_league": by_league(g, "reddit"),
        "thread_created_minus_ko_h_quantiles": mt.created_minus_ko_h.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95])
        .round(3)
        .to_dict(),
        "share_threads_created_within_1h_of_ko": round(float((mt.created_minus_ko_h.abs() <= 1).mean()), 4),
        "share_threads_created_before_ko_minus_1h": round(float((mt.created_minus_ko_h < -1).mean()), 4),
        "share_threads_with_comment_before": {
            k: round(float((pm[c] > 0).mean()), 4)
            for k, c in (("ko-24h", "lt_24"), ("ko-6h", "lt_6"), ("ko-1h", "lt_1"), ("ko", "lt_0"))
        },
        "comments_before": {
            k: int(pm[c].sum()) for k, c in (("ko-24h", "lt_24"), ("ko-6h", "lt_6"), ("ko-1h", "lt_1"), ("ko", "lt_0"))
        },
        "pre_match_threads_linked": int((pre.kind == "pre-match").sum()),
    }
    g2 = g[g.ko >= pd.Timestamp("2026-01-05", tz="UTC")].copy()
    g2["liq"] = liquidity(g2, bars)
    rep["reddit"]["liquidity_2026"] = liq_compare(g2, "reddit")

    b = groups[(groups.ko >= BSKY_FROM) & (groups.ko < BSKY_TO)].copy()
    b["both"] = b.home_club_id.isin(bsky.clubs) & b.away_club_id.isin(bsky.clubs)
    b["one"] = b.home_club_id.isin(bsky.clubs) | b.away_club_id.isin(bsky.clubs)
    b["liq"] = liquidity(b, bars)
    rep["bluesky"] = {
        "pm_groups": len(b),
        "both_sides_covered": int(b.both.sum()),
        "one_or_both_sides_covered": int(b.one.sum()),
        "join_rate_both": round(float(b.both.mean()), 4),
        "join_rate_any": round(float(b.one.mean()), 4),
        "by_league_both": by_league(b, "both"),
        "liquidity_both": liq_compare(b, "both"),
        "clubs_with_active_query": len(bsky.clubs),
    }
    (DATA / "coverage.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
