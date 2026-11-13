"""Build the R4 decision panel: one row per (Polymarket 1X2 match group, decision hour H), with point-in-time
Polymarket and Bluesky features at H and the bounce-robust Polymarket move after H as the target.

Inputs (data/research/r4/, written by pull_meta.py, linking.py and pull_data.py): pm_groups, pm_bars, bsky_hourly,
bluesky_queries. Output: `panel.parquet` (written by `main`) and the drop log `panel_report.json`.

Decision hours. H = ko - h for h = 24, 23, ..., 2 (ko = the markets' common `close_at`, on the hour or not; H
is ko minus whole hours). h = 1 is excluded because the target starts at the bar after H, which for h = 1 is
already the last pre-kickoff bar.

Polymarket prices (per side s in home/draw/away, Yes-equivalent; `r5-common/prices.pm_hourly`: a bar with
hour_start = t - 1h is known at t; Yes bars win over No-complement bars in the same hour):
  p_H        close of the last bar known by H (known_at <= H), at most 24h old           [feature, known at H]
  p_H6       close of the last bar known by H - 6h, at most 24h old                       [feature, known at H]
  trades24   number of trades in bars known in (H - 24h, H]                               [feature, known at H]
  p_0        close of the FIRST bar with hour_start >= H and known_at <= min(H + 3h, ko)  [target start]
  p_end      close of the last bar with known_at <= ko, at most 3h old at ko              [target end]
The target starts at the first trade after H rather than at p_H: last-trade prices carry bid-ask bounce
(R1: sigma ~ 0.7 cents), and a move measured from p_H shares p_H's bounce with every feature built from p_H
(the own-recent-move baseline would then predict the move mechanically, R1's Roll trap). The bounce in p_0 is
independent of everything known at H.

Bluesky (per club, from the club's active search queries; a post in hour bucket hr is known at hr + 1h):
  n_L        posts in buckets [H - L, H - 1h]                         (L = 6 or 24)
  rate       posts per hour over the club's baseline, buckets [max(2026-01-05, H - 24h - 28d), H - 24h); the
             baseline must span >= 14 days, else the side's Bluesky features are missing
  abn_L      log(1 + n_L) - log(1 + rate * L)                          (abnormal attention)
  sent_L     (pos_L - neg_L) / (pos_L + neg_L + 2)                     (smoothed lexicon net sentiment)

Every feature is asserted known at or before H in code (`_assert_pit`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r5-common"))

from prices import pm_hourly  # noqa: E402  (r5-common)

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "research" / "r4"
SIDES = ("home", "draw", "away")
HOURS_BEFORE = list(range(24, 1, -1))
BSKY_FROM = pd.Timestamp("2026-01-05", tz="UTC")
BSKY_TO = pd.Timestamp("2026-10-20", tz="UTC")
PANEL_FROM = pd.Timestamp("2026-01-20", tz="UTC")  # 14 days of Bluesky baseline + 24h before the earliest H
PANEL_TO = pd.Timestamp("2026-10-20", tz="UTC")
LOOKBACKS = (6, 24)
BASELINE_DAYS = 28
MIN_BASELINE_DAYS = 14
MAX_STALE_H = 24
START_WITHIN_H = 3
END_MAX_STALE_H = 3
NS_H = 3_600_000_000_000


def _ns(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(ts)
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64)


def _assert_pit(known_ns: float, decision_ns: int) -> None:
    if np.isfinite(known_ns):
        assert known_ns <= decision_ns, "feature used before it was known"


class Bars:
    """Per-condition arrays (known_at ns, hour_start ns, close, n_trades) sorted by known_at."""

    def __init__(self, bars: pd.DataFrame):
        h = pm_hourly(bars)
        self.s: dict[str, tuple[np.ndarray, ...]] = {}
        for cid, g in h.sort_values(["condition_id", "H"]).groupby("condition_id", sort=False):
            k = _ns(g.H)
            tr = g.n_trades.fillna(0).to_numpy(float)
            self.s[cid] = (k, k - NS_H, g.p.to_numpy(float), np.concatenate([[0.0], np.cumsum(tr)]))

    def last_known(self, cid: str, t: int, max_stale_h: float) -> tuple[float, float]:
        """(price, known_at ns) of the last bar known by t, NaN when none within max_stale_h."""
        s = self.s.get(cid)
        if s is None:
            return np.nan, np.nan
        i = np.searchsorted(s[0], t, side="right") - 1
        if i < 0 or (t - s[0][i]) > max_stale_h * NS_H:
            return np.nan, np.nan
        return float(s[2][i]), float(s[0][i])

    def trades_between(self, cid: str, lo: int, hi: int) -> float:
        """Trades in bars known in (lo, hi]."""
        s = self.s.get(cid)
        if s is None:
            return 0.0
        a = np.searchsorted(s[0], lo, side="right")
        b = np.searchsorted(s[0], hi, side="right")
        return float(s[3][b] - s[3][a])

    def first_from(self, cid: str, hour_start_min: int, known_max: int) -> float:
        """Close of the first bar with hour_start >= hour_start_min and known_at <= known_max."""
        s = self.s.get(cid)
        if s is None:
            return np.nan
        i = np.searchsorted(s[1], hour_start_min, side="left")
        if i >= len(s[0]) or s[0][i] > known_max:
            return np.nan
        return float(s[2][i])


class Bluesky:
    """Dense hourly club series (posts, positive, negative) with prefix sums, hours from BSKY_FROM."""

    def __init__(self, hourly: pd.DataFrame, queries: pd.DataFrame):
        q = queries[(queries.status == "active") & (queries.kind == "club")].copy()
        q["club_id"] = q.club_ids.map(lambda s: int(json.loads(s)[0]))
        hourly = hourly.merge(q[["query_id", "club_id"]], on="query_id", how="inner")
        hr = pd.to_datetime(hourly.hr, utc=True)
        hourly = hourly[(hr >= BSKY_FROM) & (hr < BSKY_TO)].copy()
        hourly["i"] = ((_ns(pd.to_datetime(hourly.hr, utc=True)) - BSKY_FROM.value) // NS_H).astype(int)
        self.n_hours = int((BSKY_TO - BSKY_FROM) / pd.Timedelta(hours=1))
        self.clubs = set(q.club_id)
        self.cum: dict[int, np.ndarray] = {}
        for cid, g in hourly.groupby("club_id"):
            arr = np.zeros((self.n_hours, 3))
            agg = g.groupby("i")[["n", "n_pos", "n_neg"]].sum()
            arr[agg.index.to_numpy()] = agg.to_numpy(float)
            self.cum[int(cid)] = np.vstack([np.zeros((1, 3)), np.cumsum(arr, axis=0)])
        for cid in self.clubs - set(self.cum):
            self.cum[int(cid)] = np.zeros((self.n_hours + 1, 3))

    def window(self, club: int, lo_ns: int, hi_ns: int) -> tuple[np.ndarray, int]:
        """Sums over buckets with bucket start in [lo, hi) (all known by hi), clipped to the coverage; hours used."""
        c = self.cum[club]
        a = int(np.clip((lo_ns - BSKY_FROM.value) // NS_H, 0, self.n_hours))
        b = int(np.clip((hi_ns - BSKY_FROM.value) // NS_H, 0, self.n_hours))
        return c[b] - c[a], b - a

    def features(self, club: float, h_ns: int) -> dict[str, float]:
        out: dict[str, float] = {}
        if not (np.isfinite(club) and int(club) in self.clubs):
            return out
        club = int(club)
        base, nh = self.window(club, h_ns - (24 + 24 * BASELINE_DAYS) * NS_H, h_ns - 24 * NS_H)
        if nh < 24 * MIN_BASELINE_DAYS:
            return out
        rate = base[0] / nh
        out["rate"] = rate
        for L in LOOKBACKS:
            w, nh_l = self.window(club, h_ns - L * NS_H, h_ns)
            assert nh_l == L
            n, pos, neg = w
            out[f"n_{L}"] = n
            out[f"abn_{L}"] = np.log1p(n) - np.log1p(rate * L)
            out[f"sent_{L}"] = (pos - neg) / (pos + neg + 2)
        return out


def build(groups: pd.DataFrame, bars: Bars, bsky: Bluesky) -> tuple[pd.DataFrame, dict]:
    g = groups[(groups.ko >= PANEL_FROM) & (groups.ko < PANEL_TO)].copy()
    rep: dict = {"groups_in_window": len(g)}
    g["bsky_home"] = g.home_club_id.isin(bsky.clubs)
    g["bsky_away"] = g.away_club_id.isin(bsky.clubs)
    rep["groups_bsky_both_sides"] = int((g.bsky_home & g.bsky_away).sum())
    rep["groups_bsky_any_side"] = int((g.bsky_home | g.bsky_away).sum())
    rows = []
    for r in g.itertuples(index=False):
        ko = pd.Timestamp(r.ko).value
        conds = {s: getattr(r, f"cond_{s}") for s in SIDES}
        end = {s: bars.last_known(conds[s], ko, END_MAX_STALE_H)[0] for s in SIDES}
        for hb in HOURS_BEFORE:
            H = ko - hb * NS_H
            rec = {
                "group_id": r.group_id,
                "league_code": r.league_code,
                "ko": r.ko,
                "h_before": hb,
                "H": pd.Timestamp(H, tz="UTC"),
            }
            for s in SIDES:
                c = conds[s]
                p, kn = bars.last_known(c, H, MAX_STALE_H)
                p6, kn6 = bars.last_known(c, H - 6 * NS_H, MAX_STALE_H)
                _assert_pit(kn, H)
                _assert_pit(kn6, H)
                rec[f"p_H_{s}"] = p
                rec[f"p_H6_{s}"] = p6
                rec[f"stale_{s}"] = (H - kn) / NS_H if np.isfinite(kn) else np.nan
                rec[f"trades24_{s}"] = bars.trades_between(c, H - 24 * NS_H, H)
                rec[f"p_0_{s}"] = bars.first_from(c, H, min(H + START_WITHIN_H * NS_H, ko))
                rec[f"p_end_{s}"] = end[s]
            for side, club in (("home", r.home_club_id), ("away", r.away_club_id)):
                for k, v in bsky.features(club, H).items():
                    rec[f"bs_{k}_{side}"] = v
            rows.append(rec)
    panel = pd.DataFrame(rows)
    rep["rows"] = len(panel)
    return panel, rep


def main() -> None:
    groups = pd.read_parquet(DATA / "pm_groups.parquet")
    bars = Bars(pd.read_parquet(DATA / "pm_bars.parquet"))
    bsky = Bluesky(pd.read_parquet(DATA / "bsky_hourly.parquet"), pd.read_parquet(DATA / "bluesky_queries.parquet"))
    panel, rep = build(groups, bars, bsky)
    panel.to_parquet(DATA / "panel.parquet")
    (DATA / "panel_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
