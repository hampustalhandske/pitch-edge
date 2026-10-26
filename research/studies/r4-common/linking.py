"""Link r/soccer match threads and Bluesky club queries to Polymarket 1X2 match groups (R4).

Reads the stage-1 metadata in data/research/r4/ and writes:
  pm_groups.parquet       one row per Polymarket 1X2 match group with exactly one market per side and a common
                          `close_at` (= kickoff `ko`, verified against football-data to the minute for 96.7% of
                          mapped matches in R1), with both club ids and the condition per side
  reddit_links.parquet    one row per (PM group, Reddit match thread) link
  link_report.json        join rates and drop reasons

No price, result, resolution or Reddit score column is read.

Polymarket group. `(league_code, match_date, home_name, away_name)` from `pm_soccer_match_map`, the same grouping
as `r5-common/mapping.py::pm_groups`.

Reddit thread. A post whose title starts with "Match Thread" (any case, optional brackets; "Pre-Match",
"Post-Match" and "Serious" threads are separate kinds, kept with `kind`). The title body before the first "|"
is split on " vs " / " vs. " / " v " into home and away names. A thread is linked to a PM group when its
`created_utc` is in [ko - 36h, ko + 3h] and both names match the PM clubs: a name matches a club when its
normalised alias (`club_alias.alias_key`: lower-case, ASCII letters and digits only) belongs to that club_id, or
else when `team_similarity(canonical_team(name), canonical_team(pm_name))` >= 0.85 (both clubs) or one club is
an exact alias match and the other >= 0.6. When several PM groups qualify for one thread the best-scoring one is
kept if it is unique; when several threads of one kind qualify for one PM group the earliest is kept and the
count is recorded (duplicate threads are rare and usually a deleted repost).
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_edge.paper.fixtures import canonical_team, team_similarity

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "research" / "r4"

TITLE_RE = re.compile(
    r"^\s*[\[\(]?\s*(?P<kind>(?:serious\s+)?(?:pre[\s-]*|post[\s-]*)?match\s+thread)\s*[\]\)]?\s*[:\-|]?\s*(?P<body>.*)$",
    re.IGNORECASE,
)
VS_RE = re.compile(r"\s+(?:vs\.?|v\.?|-)\s+", re.IGNORECASE)
WINDOW_BEFORE_H = 36
WINDOW_AFTER_H = 3
STRONG, WEAK = 0.85, 0.6


def alias_key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse_thread(title: str) -> dict | None:
    """Kind ('match', 'pre-match', 'post-match', with an optional 'serious ' prefix), home and away names."""
    m = TITLE_RE.match(title or "")
    if not m:
        return None
    kind = re.sub(r"[\s-]+", " ", m.group("kind").lower()).replace(" thread", "").strip()
    kind = kind.replace("pre match", "pre-match").replace("post match", "post-match")
    body = m.group("body").split("|")[0]
    body = re.sub(r"\[.*?\]|\(.*?\)", " ", body).strip()
    parts = VS_RE.split(body, maxsplit=1)
    if len(parts) != 2:
        return {"kind": kind, "t_home": None, "t_away": None}
    home, away = (p.strip(" .:-") for p in parts)
    # post-match titles carry a score: "Arsenal 2-1 Chelsea" -> strip trailing/leading digits
    home = re.sub(r"\s*\d+\s*$", "", home)
    away = re.sub(r"^\s*\d+\s*", "", away)
    return {"kind": kind, "t_home": home or None, "t_away": away or None}


def pm_groups(pm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
        ko=("close_at", "min"),
        ko_max=("close_at", "max"),
        opens_at=("opens_at", "min"),
        home_club_id=("home_club_id", "first"),
        away_club_id=("away_club_id", "first"),
        fd_match_key=("fd_match_key", "first"),
        event_slug=("event_slug", "first"),
    )
    g = meta.join(wide).reset_index()
    bad = g.ko != g.ko_max
    drops["drop_close_at_disagree"] = int(bad.sum())
    g = g[~bad].drop(columns="ko_max")
    g["group_id"] = g.league_code + "|" + g.match_date.astype(str) + "|" + g.home_name + "|" + g.away_name
    drops["pm_groups_kept"] = len(g)
    return g.reset_index(drop=True), drops


def _name_score(name: str, club_id: float, pm_name: str, alias_clubs: dict[str, set[int]]) -> float:
    if name is None:
        return 0.0
    ids = alias_clubs.get(alias_key(name), set())
    if np.isfinite(club_id) and int(club_id) in ids:
        return 1.0
    return team_similarity(canonical_team(name), canonical_team(pm_name))


def link_reddit(g: pd.DataFrame, posts: pd.DataFrame, alias: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parsed = posts.title.map(parse_thread)
    th = posts[parsed.notna()].copy()
    th = th.join(pd.DataFrame(list(parsed.dropna()), index=th.index))
    rep = {"reddit_posts": len(posts), "threads_parsed": len(th), "threads_by_kind": th.kind.value_counts().to_dict()}
    th = th[th.t_home.notna() & th.t_away.notna()]
    rep["threads_with_two_names"] = len(th)
    alias_clubs: dict[str, set[int]] = {}
    for r in alias.itertuples(index=False):
        alias_clubs.setdefault(r.alias_key, set()).add(int(r.club_id))
    g = g.sort_values("ko").reset_index(drop=True)
    ko_ns = g.ko.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64)
    rows = []
    for t in th.itertuples(index=False):
        c = pd.Timestamp(t.created_utc).value
        lo = np.searchsorted(ko_ns, c - WINDOW_AFTER_H * 3.6e12, side="left")
        hi = np.searchsorted(ko_ns, c + WINDOW_BEFORE_H * 3.6e12, side="right")
        best = []
        for j in range(lo, hi):
            p = g.iloc[j]
            sh = _name_score(t.t_home, p.home_club_id, p.home_name, alias_clubs)
            sa = _name_score(t.t_away, p.away_club_id, p.away_name, alias_clubs)
            ok = (min(sh, sa) >= STRONG) or (max(sh, sa) >= 1.0 and min(sh, sa) >= WEAK)
            if ok:
                best.append((sh + sa, j, sh, sa))
        if not best:
            continue
        best.sort(reverse=True)
        if len(best) > 1 and best[0][0] - best[1][0] < 0.1:
            continue
        s, j, sh, sa = best[0]
        rows.append(
            {
                "group_id": g.group_id[j],
                "post_id": t.post_id,
                "kind": t.kind,
                "created_utc": t.created_utc,
                "title": t.title,
                "sim_home": sh,
                "sim_away": sa,
                "created_minus_ko_h": (c - ko_ns[j]) / 3.6e12,
            }
        )
    links = pd.DataFrame(rows)
    n_dupe = links.groupby(["group_id", "kind"]).post_id.transform("count")
    links["n_same_kind"] = n_dupe
    links = links.sort_values("created_utc").drop_duplicates(["group_id", "kind"], keep="first")
    rep["links"] = len(links)
    rep["links_by_kind"] = links.kind.value_counts().to_dict()
    return links.reset_index(drop=True), rep


def main() -> None:
    pm = pd.read_parquet(DATA / "pm_1x2.parquet")
    posts = pd.read_parquet(DATA / "reddit_posts.parquet")
    alias = pd.read_parquet(DATA / "club_alias.parquet")
    g, d1 = pm_groups(pm)
    g.to_parquet(DATA / "pm_groups.parquet")
    links, d2 = link_reddit(g, posts, alias)
    links.to_parquet(DATA / "reddit_links.parquet")
    rep = {**d1, **d2}
    (DATA / "link_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
