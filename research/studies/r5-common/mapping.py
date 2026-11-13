"""Map Kalshi soccer 1X2 events to Polymarket 1X2 match groups and, through them, to football-data matches.

Reads the stage-1 metadata in data/research/r5/ and writes `kalshi_pm_map.parquet` (one row per matched
Kalshi event) plus `mapping_report.json` (join rates, drop reasons, league map). No price, result or
resolution column is read: the mapping uses only names, dates and tickers.

Kalshi side. An event (`KX<LEAGUE>GAME-<YYMONDD><HOME><AWAY>`) has three markets: `-TIE` and one per club.
`rules_primary` reads "If <X> wins the <Home> vs <Away> professional <League> soccer game originally
scheduled for <Mon D, YYYY> ...", which gives the home/away names, the scheduled (local) date and, per
market, which club it pays on. A club market is assigned home/away by matching <X> to the two names, with
the ticker codes (`<HOME><AWAY>` = home code + away code) as a tie-break.

Polymarket side. A match group is the (league_code, match_date, home_name, away_name) tuple of
`pm_soccer_match_map` with exactly one win_home, draw and win_away market; its kickoff is the markets'
common `close_at` (R1 verified close_at equals the football-data kickoff to the minute for 96.7% of matches).

Join. Candidate pairs are PM groups whose kickoff UTC date is within one day of the Kalshi scheduled date.
Names are compared with `pitch_edge.paper.fixtures.canonical_team` / `team_similarity`. A pair is
`both` when both clubs score >= 0.85 and `one_strong` when one club scores >= 0.95 and the other >= 0.5
(a club plays at most once a day). Pass 1 learns which PM league codes each Kalshi series maps to from
`both` pairs (codes with >= 3 such pairs and >= 10% of the series' pairs); pass 2 keeps only candidates in
those leagues and accepts the best-scoring candidate when it is unique (second best at least 0.1 lower).
Pairs whose orientation is reversed (Kalshi home = PM away) are kept and flagged; the side mapping
follows the club names.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_edge.paper.fixtures import canonical_team, team_similarity

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "research" / "r5"

RULE_RE = re.compile(
    r"^If (?P<x>.+?) (?:wins|is the result of) the (?P<home>.+?) vs (?P<away>.+?) professional .*?"
    r"originally scheduled for (?P<date>[A-Z][a-z]{2} \d{1,2}, \d{4})"
)
BOTH, STRONG, WEAK = 0.85, 0.95, 0.5
UNIQUE_MARGIN = 0.10


def kalshi_events(km: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per Kalshi event with home/away names, scheduled date and the ticker per side."""
    km = km.copy()
    parsed = km.rules_primary.fillna("").str.extract(RULE_RE)
    km = km.join(parsed)
    km["suffix"] = km.ticker.str.split("-").str[-1]
    km["evcode"] = km.event_ticker.str.split("-").str[-1].str[7:]
    drops: dict[str, int] = {"kalshi_events": int(km.event_ticker.nunique())}
    bad = km.groupby("event_ticker").home.apply(lambda s: s.isna().any())
    drops["drop_rules_unparsed"] = int(bad.sum())
    km = km[~km.event_ticker.isin(bad[bad].index)]

    def side(r: pd.Series) -> str | None:
        if r.suffix == "TIE":
            return "draw"
        x = canonical_team(r.x)
        sh, sa = team_similarity(x, canonical_team(r.home)), team_similarity(x, canonical_team(r.away))
        if sh > sa:
            return "home"
        if sa > sh:
            return "away"
        if r.evcode.startswith(r.suffix) and not r.evcode.endswith(r.suffix):
            return "home"
        if r.evcode.endswith(r.suffix) and not r.evcode.startswith(r.suffix):
            return "away"
        return None

    km["side"] = km.apply(side, axis=1)
    ok = km.groupby("event_ticker").side.apply(lambda s: sorted(s.dropna()) == ["away", "draw", "home"])
    drops["drop_side_ambiguous"] = int((~ok).sum())
    km = km[km.event_ticker.isin(ok[ok].index)]
    wide = km.pivot(index="event_ticker", columns="side", values="ticker").rename(columns=lambda s: f"tk_{s}")
    meta = km.groupby("event_ticker").agg(
        series=("series_ticker", "first"),
        k_home=("home", "first"),
        k_away=("away", "first"),
        k_date=("date", "first"),
        k_close_time=("close_time", "min"),
        k_occurrence=("occurrence_datetime", "min"),
        k_open_time=("open_time", "min"),
        k_volume=("volume", "sum"),
    )
    ev = meta.join(wide).reset_index()
    ev["k_date"] = pd.to_datetime(ev.k_date, format="%b %d, %Y")
    ev["kh"] = ev.k_home.map(canonical_team)
    ev["ka"] = ev.k_away.map(canonical_team)
    drops["kalshi_events_parsed"] = len(ev)
    return ev, drops


def pm_groups(pm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per Polymarket 1X2 match group with exactly one market per outcome."""
    pm = pm.copy()
    key = ["league_code", "match_date", "home_name", "away_name"]
    pm = pm.dropna(subset=["home_name", "away_name", "match_date", "outcome_side"])
    per = pm.groupby(key + ["outcome_side"]).condition_id.nunique().unstack()
    good = per.reindex(columns=["home", "draw", "away"]).eq(1).all(axis=1)
    drops = {"pm_groups": len(per), "drop_pm_not_one_market_per_side": int((~good).sum())}
    pm = pm.set_index(key).loc[good[good].index].reset_index()
    wide = pm.pivot_table(index=key, columns="outcome_side", values="condition_id", aggfunc="first").rename(
        columns=lambda s: f"cond_{s}"
    )
    meta = pm.groupby(key).agg(
        pm_ko=("close_at", "min"),
        pm_ko_max=("close_at", "max"),
        pm_opens=("opens_at", "min"),
        fd_match_key=("fd_match_key", "first"),
        event_slug=("event_slug", "first"),
    )
    g = meta.join(wide).reset_index()
    g["ph"] = g.home_name.map(canonical_team)
    g["pa"] = g.away_name.map(canonical_team)
    g["pm_date"] = g.pm_ko.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    drops["pm_groups_kept"] = len(g)
    return g, drops


def candidates(ev: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Every (Kalshi event, PM group) pair within one day, with both orientations scored."""
    rows = []
    by_date = dict(g.groupby("pm_date"))
    for e in ev.itertuples(index=False):
        for off in (-1, 0, 1):
            sub = by_date.get(e.k_date + pd.Timedelta(days=off))
            if sub is None:
                continue
            for p in sub.itertuples(index=False):
                sh, sa = team_similarity(e.kh, p.ph), team_similarity(e.ka, p.pa)
                rh, ra = team_similarity(e.kh, p.pa), team_similarity(e.ka, p.ph)
                straight, rev = min(sh, sa), min(rh, ra)
                strong = max(sh, sa) >= STRONG and min(sh, sa) >= WEAK
                strong_r = max(rh, ra) >= STRONG and min(rh, ra) >= WEAK
                if not (straight >= BOTH or rev >= BOTH or strong or strong_r):
                    continue
                reversed_ = (rev > straight) and (rev >= BOTH or strong_r)
                lo, hi = (min(rh, ra), max(rh, ra)) if reversed_ else (min(sh, sa), max(sh, sa))
                rows.append(
                    {
                        "event_ticker": e.event_ticker,
                        "series": e.series,
                        "pm_key": (p.league_code, p.match_date, p.home_name, p.away_name),
                        "league_code": p.league_code,
                        "sim_lo": lo,
                        "sim_hi": hi,
                        "score": (lo + hi) / 2,
                        "how": "both" if lo >= BOTH else "one_strong",
                        "reversed": reversed_,
                        "day_offset": off,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    km = pd.read_parquet(DATA / "kalshi_markets.parquet")
    pm = pd.read_parquet(DATA / "pm_1x2.parquet")
    ev, d1 = kalshi_events(km)
    g, d2 = pm_groups(pm)
    cand = candidates(ev, g)
    both = cand[cand.how == "both"]
    lm = both.groupby(["series", "league_code"]).size().rename("n").reset_index()
    lm["share"] = lm.n / lm.groupby("series").n.transform("sum")
    lm = lm[(lm.n >= 3) & (lm.share >= 0.10)]
    league_map = lm.groupby("series").league_code.apply(list).to_dict()
    cand = cand[[lc in league_map.get(s, []) for s, lc in zip(cand.series, cand.league_code, strict=True)]]
    cand = cand.sort_values(["event_ticker", "score"], ascending=[True, False])
    top = cand.groupby("event_ticker").head(2)
    second = top.groupby("event_ticker").score.nth(1)
    best = top.groupby("event_ticker").head(1).set_index("event_ticker")
    best["second"] = second.reindex(best.index) if len(second) else np.nan
    second_score = top.groupby("event_ticker").score.apply(lambda s: s.iloc[1] if len(s) > 1 else np.nan)
    best["second"] = second_score.reindex(best.index)
    ambiguous = best.second.notna() & (best.score - best.second < UNIQUE_MARGIN)
    best = best[~ambiguous]
    # one PM group per Kalshi event and vice versa
    dup_pm = best.pm_key.duplicated(keep=False)
    best = best[~dup_pm]
    gi = g.set_index(["league_code", "match_date", "home_name", "away_name"])
    rows = []
    evi = ev.set_index("event_ticker")
    for et, b in best.iterrows():
        p = gi.loc[b.pm_key]
        e = evi.loc[et]
        rec = {
            "event_ticker": et,
            "series": e.series,
            "k_home": e.k_home,
            "k_away": e.k_away,
            "k_date": e.k_date,
            "k_occurrence": e.k_occurrence,
            "k_close_time": e.k_close_time,
            "k_volume": e.k_volume,
            "pm_league": b.pm_key[0],
            "pm_home": b.pm_key[2],
            "pm_away": b.pm_key[3],
            "pm_ko": p.pm_ko,
            "pm_ko_max": p.pm_ko_max,
            "fd_match_key": p.fd_match_key,
            "match_how": b.how,
            "reversed": bool(b.reversed),
            "sim_lo": b.sim_lo,
            "score": b.score,
        }
        # Side mapping follows club names: Kalshi home is PM away when reversed.
        rec["tk_home"], rec["tk_draw"], rec["tk_away"] = e.tk_home, e.tk_draw, e.tk_away
        if b.reversed:
            rec["cond_home"], rec["cond_away"] = p.cond_away, p.cond_home
        else:
            rec["cond_home"], rec["cond_away"] = p.cond_home, p.cond_away
        rec["cond_draw"] = p.cond_draw
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "kalshi_pm_map.parquet")
    report = {
        **d1,
        **d2,
        "candidate_pairs": len(cand),
        "events_with_candidate": int(cand.event_ticker.nunique()),
        "drop_ambiguous_second_best": int(ambiguous.sum()),
        "drop_pm_group_claimed_twice": int(dup_pm.sum()),
        "matched_events": len(out),
        "matched_with_fd": int(out.fd_match_key.notna().sum()),
        "matched_how": out.match_how.value_counts().to_dict(),
        "reversed": int(out.reversed.sum()),
        "league_map": league_map,
        "series_without_league": sorted(set(ev.series) - set(league_map)),
    }
    (DATA / "mapping_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("league_map",)}, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())


def verified_universe() -> tuple[pd.DataFrame, dict]:
    """Matched events whose kickoff (`ko` = Polymarket close_at) is corroborated, with the drop counts.

    Rules, applied in order: drop reversed-orientation pairs (the reversal test misfires on same-city
    derbies); drop when the PM markets disagree on close_at; when Kalshi `occurrence_datetime` exists it must
    equal ko + 3h within 30 min, otherwise Kalshi `close_time` (set when the game ends) must fall 1.75-3.5h
    after ko; when the match is also keyed to football-data, its UK-local kickoff must equal ko within 30
    min. The close_time check selects on the game having been played as scheduled (postponements dropped),
    which is a hindsight selection on the universe, reported as a threat; it is what keeps in-play hours out.
    """
    m = pd.read_parquet(DATA / "kalshi_pm_map.parquet")
    drops = {"matched_events": len(m)}
    keep = ~m.reversed
    drops["drop_reversed"] = int((~keep).sum())
    k2 = m.pm_ko == m.pm_ko_max
    drops["drop_pm_close_at_disagree"] = int((keep & ~k2).sum())
    keep &= k2
    occ_h = (m.k_occurrence - m.pm_ko).dt.total_seconds() / 3600
    close_h = (m.k_close_time - m.pm_ko).dt.total_seconds() / 3600
    ok_ko = np.where(m.k_occurrence.notna(), (occ_h - 3).abs() <= 0.5, close_h.between(1.75, 3.5))
    drops["drop_kickoff_not_corroborated_by_kalshi"] = int((keep & ~ok_ko).sum())
    keep &= ok_ko
    fd = pd.read_parquet(DATA / "fd.parquet").set_index("fd_match_key")
    x = m.join(fd[["match_date", "kickoff_time"]], on="fd_match_key")
    ko_fd = pd.to_datetime(x.match_date.astype(str) + " " + x.kickoff_time.astype(str), errors="coerce")
    ko_fd = ko_fd.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    fd_bad = m.fd_match_key.notna() & ~((m.pm_ko - ko_fd).dt.total_seconds().abs() <= 1800)
    drops["drop_fd_kickoff_disagrees"] = int((keep & fd_bad).sum())
    keep &= ~fd_bad
    u = m[keep].copy()
    u["ko"] = u.pm_ko
    u["match_day"] = u.ko.dt.tz_convert("UTC").dt.strftime("%Y-%m-%d")
    drops["kept"] = len(u)
    return u.reset_index(drop=True), drops
