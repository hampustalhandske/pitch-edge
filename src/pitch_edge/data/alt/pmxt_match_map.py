"""Join Polymarket (PMXT archive) markets onto this project's own `matches`/`match_id`.

Polymarket structures a 3-way soccer match as one **event** (`event_title`, e.g. "Arsenal vs
Chelsea") bundling 3 binary **markets** (one `condition_id` each), whose `group_item_title` names
the specific outcome — the two team names, plus one literal "Draw" market. Scoped to match-winner
(1X2) markets only: this project's models only ever produce home/draw/away probabilities, so those
are the only markets a model-vs-market edge can be computed against. A market whose
`group_item_title` doesn't resolve to {home team, away team, draw} is logged as unmapped and
dropped, not guessed at.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from pitch_edge.data.storage import Warehouse
from pitch_edge.data.teams import TeamNameResolver, normalise

logger = logging.getLogger(__name__)

_DRAW_WORDS = {"draw", "tie", "x"}
# "Arsenal vs Chelsea", "Arsenal v Chelsea", "Arsenal - Chelsea"
_EVENT_TITLE_RE = re.compile(r"^\s*(.+?)\s+(?:vs\.?|v\.?|-)\s+(.+?)\s*$", re.IGNORECASE)


def _split_event_title(title: str) -> tuple[str, str] | None:
    if not isinstance(title, str):
        return None
    m = _EVENT_TITLE_RE.match(title)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def _outcome_side(group_item_title: str, home: str, away: str, resolver: TeamNameResolver) -> str | None:
    if not isinstance(group_item_title, str):
        return None
    # Real data confirms both bare "Draw" and "Draw (Team A vs. Team B)" forms — strip a trailing
    # parenthetical before checking, so the latter isn't missed and silently dropped as unmapped.
    bare = re.sub(r"\s*\([^)]*\)\s*$", "", group_item_title).strip()
    if normalise(bare) in _DRAW_WORDS:
        return "draw"
    resolved = resolver.resolve(group_item_title)
    if resolved == home:
        return "home"
    if resolved == away:
        return "away"
    return None


def build_pmxt_match_map(wh: Warehouse) -> pd.DataFrame:
    """Recompute the full `condition_id -> match_id, outcome_side` mapping from whatever
    `dim_soccer_markets`/`matches` currently hold. Cheap and idempotent (full `Warehouse.replace()`,
    like `espn_team_map`) — safe to re-run any time either table grows, no network calls."""
    markets = wh.read("dim_soccer_markets")
    if markets.empty or not wh.table_exists("matches"):
        return pd.DataFrame()

    matches = wh.query("SELECT match_id, date, home_team, away_team FROM matches")
    if matches.empty:
        return pd.DataFrame()

    canonical = set(matches["home_team"]) | set(matches["away_team"])
    resolver = TeamNameResolver(canonical)

    # Build a fixture_key -> [(match_id, date)] index, tolerating the +/-1 day gap between a
    # market's real `end_date` and the stored match `date` (same tolerance espn_fixtures_mapped uses).
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    by_pair: dict[tuple[str, str], list[tuple[str, pd.Timestamp]]] = {}
    for row in matches.itertuples(index=False):
        by_pair.setdefault((normalise(row.home_team), normalise(row.away_team)), []).append((row.match_id, row.date))

    # Group markets by event so all 3 outcome legs of one fixture share one home/away resolution.
    rows: list[dict] = []
    unmapped = 0
    for event_title, group in markets.groupby("event_title", dropna=False):
        split = _split_event_title(str(event_title))
        if split is None:
            unmapped += len(group)
            continue
        home_raw, away_raw = split
        home = resolver.resolve(home_raw)
        away = resolver.resolve(away_raw)
        if not home or not away:
            unmapped += len(group)
            continue
        candidates = by_pair.get((normalise(home), normalise(away)), [])
        end_dates = pd.to_datetime(group["end_date"], errors="coerce", utc=True).dt.tz_localize(None)
        for condition_id, group_item_title, end_date in zip(
            group["condition_id"], group["group_item_title"], end_dates, strict=True
        ):
            side = _outcome_side(group_item_title, home, away, resolver)
            if side is None:
                unmapped += 1
                continue
            match_id = None
            match_date = None
            if pd.notna(end_date):
                best = min(candidates, key=lambda c: abs((c[1] - end_date).days), default=None)
                if best is not None and abs((best[1] - end_date).days) <= 1:
                    match_id, match_date = best
            if match_id is None:
                unmapped += 1
                continue
            rows.append(
                {
                    "condition_id": condition_id,
                    "match_id": match_id,
                    "outcome_side": side,
                    "home_team": home,
                    "away_team": away,
                    "match_date": match_date,
                }
            )

    logger.info("build_pmxt_match_map: %d mapped, %d unmapped/dropped", len(rows), unmapped)
    df = pd.DataFrame(rows, columns=["condition_id", "match_id", "outcome_side", "home_team", "away_team", "match_date"])
    wh.replace("pmxt_match_map", df)
    return df
