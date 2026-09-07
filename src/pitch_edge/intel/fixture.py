"""Resolve 'Brighton vs Forest [on 2026-09-13]' to a concrete fixture.

Order of preference: a played match in the warehouse on that date → an upcoming fixture from
openfootball (a *targeted* on-demand fetch of just that league-season file, through the cached HTTP
client) → a hypothetical fixture on the requested date, flagged as such so the dossier never
pretends the fixture is scheduled.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import pandas as pd

from pitch_edge.data.storage import Warehouse
from pitch_edge.data.teams import TeamNameResolver, normalise

logger = logging.getLogger(__name__)

OPENFOOTBALL_BY_LEAGUE = {
    "E0": "en.1",
    "E1": "en.2",
    "D1": "de.1",
    "D2": "de.2",
    "SP1": "es.1",
    "SP2": "es.2",
    "I1": "it.1",
    "I2": "it.2",
    "F1": "fr.1",
    "F2": "fr.2",
    "N1": "nl.1",
    "P1": "pt.1",
    "B1": "be.1",
    "T1": "tr.1",
    "SC0": "sco.1",
    "G1": "gr.1",
}


@dataclass
class Fixture:
    home_team: str
    away_team: str
    date: pd.Timestamp | None
    league_code: str | None
    match_id: str | None
    source: str  # "warehouse_match" | "openfootball_upcoming" | "hypothetical"
    played: bool
    kickoff_time: str | None = None
    home_goals: float | None = None
    away_goals: float | None = None

    @property
    def fixture_key(self) -> str:
        d = self.date.strftime("%Y-%m-%d") if self.date is not None else ""
        return f"{normalise(self.home_team)}|{normalise(self.away_team)}|{d}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date"] = str(self.date.date()) if self.date is not None else None
        d["fixture_key"] = self.fixture_key
        return d


class FixtureNotFoundError(LookupError):
    pass


def resolve_teams(wh: Warehouse, home: str, away: str) -> tuple[str, str]:
    teams = (
        wh.query("SELECT DISTINCT home_team AS t FROM matches UNION SELECT DISTINCT away_team FROM matches")[
            "t"
        ].tolist()
        if wh.table_exists("matches")
        else []
    )
    resolver = TeamNameResolver(teams)
    h, a = resolver.resolve(home, cutoff=0.75), resolver.resolve(away, cutoff=0.75)
    missing = [n for n, r in ((home, h), (away, a)) if r is None]
    if missing:
        raise FixtureNotFoundError(f"unknown team(s): {missing} — names must resolve to a club in the warehouse")
    return str(h), str(a)


def _team_league(wh: Warehouse, team: str) -> str | None:
    df = wh.query(
        "SELECT league_code FROM matches WHERE home_team = ? OR away_team = ? ORDER BY date DESC LIMIT 1", [team, team]
    )
    return None if df.empty else str(df["league_code"].iloc[0])


def find_fixture(
    wh: Warehouse,
    home: str,
    away: str,
    date: str | pd.Timestamp | None = None,
    fetch_upcoming: bool = True,
    now: datetime | None = None,
) -> Fixture:
    home, away = resolve_teams(wh, home, away)
    want = pd.Timestamp(date).normalize() if date is not None else None
    today = (
        pd.Timestamp(now or datetime.now(UTC)).tz_localize(None).normalize()
        if (now is None or now.tzinfo)
        else pd.Timestamp(now).normalize()
    )
    # 1. played match in the warehouse
    sql = "SELECT * FROM matches WHERE home_team = ? AND away_team = ?"
    params: list = [home, away]
    if want is not None:
        sql += " AND CAST(date AS DATE) = ?"
        params.append(want.date())
    played = wh.query(sql + " ORDER BY date DESC LIMIT 1", params)
    if not played.empty and (want is not None or not fetch_upcoming):
        r = played.iloc[0]
        return Fixture(
            home,
            away,
            pd.Timestamp(r["date"]).normalize(),
            str(r.get("league_code")),
            str(r["match_id"]),
            "warehouse_match",
            True,
            r.get("kickoff_time"),
            float(r["home_goals"]),
            float(r["away_goals"]),
        )
    # 2. upcoming fixture from openfootball (targeted fetch)
    league = _team_league(wh, home)
    if fetch_upcoming and league in OPENFOOTBALL_BY_LEAGUE:
        from pitch_edge.data.sources.openfootball import OpenFootballSource

        year = today.year - (1 if today.month < 7 else 0)
        try:
            fx = OpenFootballSource().fetch_upcoming_fixtures(OPENFOOTBALL_BY_LEAGUE[league], year)
        except Exception as exc:  # noqa: BLE001
            logger.warning("openfootball upcoming %s: %s", league, exc)
            fx = pd.DataFrame()
        if not fx.empty:
            resolver = TeamNameResolver([home, away])
            fx = fx.assign(
                h=fx["home_team"].map(lambda n: resolver.resolve(str(n))),
                a=fx["away_team"].map(lambda n: resolver.resolve(str(n))),
            )
            hit = fx[(fx["h"] == home) & (fx["a"] == away)]
            if want is not None:
                hit = hit[pd.to_datetime(hit["date"]).dt.normalize() == want]
            hit = hit[pd.to_datetime(hit["date"]) >= today - pd.Timedelta(days=1)].sort_values("date")
            if not hit.empty:
                r = hit.iloc[0]
                return Fixture(
                    home,
                    away,
                    pd.Timestamp(r["date"]).normalize(),
                    league,
                    str(r["match_id"]),
                    "openfootball_upcoming",
                    False,
                    r.get("kickoff_time"),
                )
    if not played.empty:  # no upcoming meeting found: most recent played one
        r = played.iloc[0]
        return Fixture(
            home,
            away,
            pd.Timestamp(r["date"]).normalize(),
            str(r.get("league_code")),
            str(r["match_id"]),
            "warehouse_match",
            True,
            r.get("kickoff_time"),
            float(r["home_goals"]),
            float(r["away_goals"]),
        )
    # 3. hypothetical
    if want is None:
        raise FixtureNotFoundError(
            f"no scheduled or played {home} vs {away} found — pass --date to build a hypothetical dossier"
        )
    return Fixture(home, away, want, league, None, "hypothetical", False)
