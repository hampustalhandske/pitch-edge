"""Referee-assignment lag: when was the referee announced, and what did the price do around it?

Most top leagues publish the appointed officials 2-7 days before kickoff — after the early line
has formed, often before the market re-prices for the referee's tendency. This connector polls
public appointment pages (robots-checked, honest UA, cached), and records the *first time we saw*
each (fixture, referee) pair as the announcement timestamp. Stored for future analysis of whether
strong-tendency referees move the no-vig price.

Honesty notes:
* The announcement timestamp is "first observed by our poller", which is an upper bound on the
  true announcement time; the poll cadence (hourly scheduler) bounds the error.
* Sources are configurable. The Premier League page is the one shipped by default; it is a
  JavaScript-rendered app, so the HTML parser may find nothing on a given day — that is logged in
  `pipeline_runs`, not hidden. Other federations can be added as (name, url) pairs without code.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient
from pitch_edge.data.teams import TeamNameResolver, normalise

logger = logging.getLogger(__name__)

ANNOUNCEMENT_SOURCES: dict[str, dict[str, str]] = {
    "premier_league": {"url": "https://www.premierleague.com/referees/appointments", "league_code": "E0"},
}

_STOP_WORD = r"(?!(?:Referee|Referees|Assistant|Assistants|Fourth|VAR|AVAR|Official|Officials|Kick|KO)\b)"
_FIXTURE_RE = re.compile(
    rf"([A-Z][A-Za-z'&.\-]+(?: {_STOP_WORD}[A-Za-z'&.\-]+){{0,4}}?)\s+(?:v|vs\.?|V)\s+"
    rf"([A-Z][A-Za-z'&.\-]+(?: {_STOP_WORD}[A-Z][A-Za-z'&.\-]+){{0,4}})(?=\s*(?:[,.;:(—–-]|Referee|Assistant|Fourth|VAR|$))"
)
_REF_RE = re.compile(r"Referee\s*[:\-–]?\s*((?:[A-Z][a-zA-Z'\-]+)(?:\s+[A-Z][a-zA-Z'\-]+(?![a-zA-Z'\-]*:)){0,3})")
_REF_STOP = {"Assistants", "Assistant", "VAR", "Fourth", "Official", "Officials", "AVAR"}
_DATE_RE = re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b")


def fixture_key(home: str, away: str, date: str | pd.Timestamp | None) -> str:
    d = pd.Timestamp(date).strftime("%Y-%m-%d") if date is not None and not pd.isna(pd.Timestamp(date)) else ""
    return f"{normalise(home)}|{normalise(away)}|{d}"


def parse_assignments(text: str, teams: list[str] | None = None, window: int = 400) -> pd.DataFrame:
    """Find 'Home v Away … Referee: Name' patterns in page text. Team names are resolved to the spine's
    canonical names when `teams` is given; unresolved pairs are kept raw so nothing is silently dropped."""
    plain = re.sub(r"<[^>]+>", " ", text or "")
    plain = re.sub(r"\s+", " ", plain)
    resolver = TeamNameResolver(teams or [])
    rows = []
    for m in _FIXTURE_RE.finditer(plain):
        tail = plain[m.end() : m.end() + window]
        ref = _REF_RE.search(tail)
        if not ref:
            continue
        home_raw, away_raw = m.group(1).strip(), m.group(2).strip()
        name_words = list(ref.group(1).split())
        while name_words and name_words[-1] in _REF_STOP:
            name_words.pop()
        if len(name_words) < 2:
            continue
        home = _resolve_side(home_raw, resolver) if teams else home_raw
        away = _resolve_side(away_raw, resolver) if teams else away_raw
        dm = _DATE_RE.search(plain[max(0, m.start() - 200) : m.end() + window])
        date = pd.Timestamp(f"{dm.group(2)} {dm.group(1)} {dm.group(3)}") if dm else pd.NaT
        rows.append({"home_team": home, "away_team": away, "match_date": date, "referee": " ".join(name_words)})
    df = pd.DataFrame(rows, columns=["home_team", "away_team", "match_date", "referee"])
    if df.empty:
        return df
    df = df.drop_duplicates(["home_team", "away_team", "referee"])
    df["fixture_key"] = [fixture_key(h, a, d) for h, a, d in zip(df["home_team"], df["away_team"], df["match_date"], strict=True)]
    return df.reset_index(drop=True)


def _resolve_side(raw: str, resolver: TeamNameResolver) -> str:
    """Resolve a captured team string; if the full capture fails (a stray preceding name), try trailing words."""
    words = raw.split()
    for n in range(len(words), 0, -1):
        hit = resolver.resolve(" ".join(words[-n:]))
        if hit:
            return hit
    return raw


class RefereeAnnouncementSource:
    name = "referee_announcements"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None,
                 sources: dict[str, dict[str, str]] | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "referee_announcements"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=2.0)
        self.sources = sources if sources is not None else ANNOUNCEMENT_SOURCES

    def poll(self, teams: list[str] | None = None, now: datetime | None = None) -> pd.DataFrame:
        """Fetch every configured page (never cached — we need *when* it changed) and parse assignments."""
        seen_at = pd.Timestamp(now or datetime.now(UTC)).tz_localize(None) if (now is None or now.tzinfo is None) else pd.Timestamp(now).tz_convert(None)
        frames = []
        for name, cfg in self.sources.items():
            try:
                html = self.http.get_text(cfg["url"], cache=False, check_robots=True)
            except Exception as exc:  # noqa: BLE001 - one page must not block the others
                logger.info("referee source %s unavailable: %s", name, exc)
                continue
            df = parse_assignments(html, teams)
            if df.empty:
                logger.info("referee source %s: no assignments parsed (JS-rendered page?)", name)
                continue
            frames.append(df.assign(source=name, league_code=cfg.get("league_code"), announced_at=seen_at))
        if not frames:
            return pd.DataFrame(columns=["fixture_key", "home_team", "away_team", "match_date", "referee", "source", "league_code", "announced_at"])
        return pd.concat(frames, ignore_index=True)



