"""Information-asymmetry scanner: news/RSS ingestion + sentiment velocity.

Real, ToS-clean text feeds: public RSS from major outlets plus any
local-language beat-journalist/club feeds you add to `FEEDS` (RSS is the one
channel that is explicitly published for machine consumption). Bluesky's
public AppView is attempted opportunistically but returns 403 unauthenticated
from many networks, so it degrades to a no-op rather than failing the run.

Sentiment is scored with VADER (lexicon-based, no model download, fully
deterministic) and items are entity-linked to canonical team names via
`TeamNameResolver`. `sentiment_velocity` turns the item stream into a
per-team rate-of-change feature — the "market hasn't re-priced yet" signal.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient
from pitch_edge.data.teams import TeamNameResolver

logger = logging.getLogger(__name__)

FEEDS: dict[str, str] = {
    "bbc_football": "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "guardian_football": "https://www.theguardian.com/football/rss",
    "sky_football": "https://www.skysports.com/rss/12040",
    "espn_soccer": "https://www.espn.com/espn/rss/soccer/news",
    # Sweden — local-language feeds for the under-covered-market thesis
    "svt_sport": "https://www.svt.se/sport/rss.xml",
    "sportbladet": "https://rss.aftonbladet.se/rss2/small/pages/sections/sportbladet/",
    "expressen_sport": "https://feeds.expressen.se/sport/",
    "dn_sport": "https://www.dn.se/rss/sport/",
    "gp_sport": "https://www.gp.se/rss/sport",
}

SWEDISH_INJURY_TERMS = ("skada", "skadad", "skadan", "missar", "borta i", "opererad", "sjukskriven", "avstängd", "avstängning", "bänkad")
SWEDISH_LINEUP_TERMS = ("startelva", "startelvan", "elvan", "laguppställning", "petad", "vilas", "tillbaka i truppen", "truppen")

BLUESKY_SEARCH = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"

INJURY_TERMS = ("injur", "out for", "ruled out", "doubt", "sidelined", "knock", "hamstring", "suspended", "fitness")
LINEUP_TERMS = ("lineup", "line-up", "starting xi", "team news", "confirmed team", "benched", "dropped", "rested")


class NewsScanner:
    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "news"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=1.0)
        self._vader = SentimentIntensityAnalyzer()

    def fetch_feed(self, name: str, url: str) -> pd.DataFrame:
        try:
            raw = self.http.get_bytes(url, cache=False)
        except Exception as exc:
            logger.warning("feed %s failed: %s", name, exc)
            return pd.DataFrame()
        parsed = feedparser.parse(raw)
        rows = []
        for e in parsed.entries:
            published = e.get("published_parsed") or e.get("updated_parsed")
            ts = datetime(*published[:6]).replace(tzinfo=UTC) if published else datetime.now(UTC)
            rows.append(
                {
                    "item_id": e.get("id") or e.get("link"),
                    "feed": name,
                    "published_at": pd.Timestamp(ts).tz_convert(None),
                    "title": e.get("title", ""),
                    "summary": _strip_html(e.get("summary", "")),
                    "link": e.get("link"),
                }
            )
        return pd.DataFrame(rows)

    def fetch_all_feeds(self, feeds: dict[str, str] | None = None) -> pd.DataFrame:
        frames = [self.fetch_feed(n, u) for n, u in (feeds or FEEDS).items()]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["item_id", "feed", "published_at", "title", "summary", "link"])
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset="item_id")

    def fetch_bluesky(self, query: str, limit: int = 25) -> pd.DataFrame:
        """Opportunistic; returns empty on 403/timeouts instead of raising."""
        try:
            payload = self.http.get_json(BLUESKY_SEARCH, params={"q": query, "limit": limit}, cache=False)
        except Exception as exc:
            logger.info("Bluesky search unavailable (%s) — skipping", exc)
            return pd.DataFrame()
        rows = []
        for p in payload.get("posts", []):
            rec = p.get("record", {})
            rows.append(
                {
                    "item_id": p.get("uri"),
                    "feed": "bluesky",
                    "published_at": pd.to_datetime(rec.get("createdAt"), errors="coerce", utc=True).tz_convert(None),
                    "title": rec.get("text", "")[:120],
                    "summary": rec.get("text", ""),
                    "link": p.get("uri"),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ enrichment
    def score(self, items: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
        if items.empty:
            return items.assign(sentiment=pd.Series(dtype=float), team=pd.Series(dtype=str))
        resolver = TeamNameResolver(teams)
        out = items.copy()
        text = (out["title"].fillna("") + ". " + out["summary"].fillna("")).str.strip()
        out["sentiment"] = text.map(lambda t: self._vader.polarity_scores(t)["compound"])
        lowered = text.str.lower()
        out["is_injury_news"] = lowered.map(lambda t: any(k in t for k in INJURY_TERMS + SWEDISH_INJURY_TERMS))
        out["is_lineup_news"] = lowered.map(lambda t: any(k in t for k in LINEUP_TERMS + SWEDISH_LINEUP_TERMS))
        out["language"] = out["feed"].map(lambda f: "sv" if str(f) in ("svt_sport", "sportbladet", "expressen_sport", "dn_sport", "gp_sport") else "en")
        out["team"] = text.map(lambda t: _link_team(t, teams, resolver))
        return out


def _link_team(text: str, teams: list[str], resolver: TeamNameResolver) -> str | None:
    lowered = text.lower()
    for team in sorted(teams, key=len, reverse=True):
        if team.lower() in lowered:
            return team
    for token in text.replace(",", " ").split():
        if token[:1].isupper() and len(token) > 3:
            hit = resolver.resolve(token, cutoff=0.95)
            if hit:
                return hit
    return None


def _strip_html(text: str) -> str:
    import re

    return re.sub(r"<[^>]+>", " ", text or "").strip()


def sentiment_velocity(scored: pd.DataFrame, window_hours: int = 48) -> pd.DataFrame:
    """Per team: rolling mean sentiment, item volume, and the change vs the prior window."""
    if scored.empty or "team" not in scored:
        return pd.DataFrame(columns=["team", "as_of", "sent_mean", "sent_prev", "sent_velocity", "n_items"])
    df = scored.dropna(subset=["team"]).sort_values("published_at")
    rows = []
    for team, grp in df.groupby("team"):
        as_of = grp["published_at"].max()
        cur = grp[grp["published_at"] > as_of - pd.Timedelta(hours=window_hours)]
        prev = grp[
            (grp["published_at"] <= as_of - pd.Timedelta(hours=window_hours))
            & (grp["published_at"] > as_of - pd.Timedelta(hours=2 * window_hours))
        ]
        cur_mean = cur["sentiment"].mean() if len(cur) else float("nan")
        prev_mean = prev["sentiment"].mean() if len(prev) else float("nan")
        rows.append(
            {
                "team": team, "as_of": as_of, "sent_mean": cur_mean, "sent_prev": prev_mean,
                "sent_velocity": (cur_mean - prev_mean) if len(prev) else 0.0, "n_items": int(len(cur)),
                "injury_items": int(cur.get("is_injury_news", pd.Series(dtype=bool)).sum()),
            }
        )
    return pd.DataFrame(rows)
