"""Canonical fixture identity across quote sources.

Each source names leagues and clubs its own way (the-odds-api `soccer_epl` / "Nottingham Forest",
Polymarket `epl` / "Nottingham Forest FC", Kalshi `KXEPLGAME` / "Nottingham Forest"). A fixture key is
`<LEAGUE>|<UTC kickoff date>|<home>|<away>` built from a canonical league code and the club names run
through `PAPER_ALIASES` and `data/teams.py` (alias tables + accent/suffix-free normalisation).

`match_candidate` decides whether a source event is an existing fixture when the keys differ: same
canonical league, kickoffs within 3 hours, both clubs similar (>= 0.6) and at least one close (>= 0.85),
where a token shared by several clubs ("paris" in Paris FC and Paris SG) does not count as containment.
Every candidate is scored and the link is refused when more than one passes.
"""

from __future__ import annotations

import difflib
from datetime import datetime, timedelta

from pitch_edge.data.teams import TeamNameResolver, normalise

# Per source, raw league identifier -> canonical league code.
LEAGUE_CODES: dict[str, dict[str, str]] = {
    "odds_api": {
        "soccer_epl": "EPL",
        "soccer_efl_champ": "CHAMPIONSHIP",
        "soccer_spain_la_liga": "LALIGA",
        "soccer_italy_serie_a": "SERIEA",
        "soccer_germany_bundesliga": "BUNDESLIGA",
        "soccer_france_ligue_one": "LIGUE1",
        "soccer_netherlands_eredivisie": "EREDIVISIE",
        "soccer_uefa_champs_league": "UCL",
        "soccer_uefa_europa_league": "UEL",
    },
    "polymarket": {
        "epl": "EPL",
        "elc": "CHAMPIONSHIP",
        "lal": "LALIGA",
        "sea": "SERIEA",
        "bun": "BUNDESLIGA",
        "fl1": "LIGUE1",
        "ere": "EREDIVISIE",
        "ucl": "UCL",
        "uel": "UEL",
    },
    "kalshi": {
        "KXEPLGAME": "EPL",
        "KXEFLCHAMPIONSHIPGAME": "CHAMPIONSHIP",
        "KXLALIGAGAME": "LALIGA",
        "KXSERIEAGAME": "SERIEA",
        "KXBUNDESLIGAGAME": "BUNDESLIGA",
        "KXLIGUE1GAME": "LIGUE1",
        "KXEREDIVISIEGAME": "EREDIVISIE",
        "KXUCLGAME": "UCL",
        "KXUELGAME": "UEL",
    },
}

# Canonical league -> football-data.co.uk division code (the lake's `fd_matches.division`).
FD_DIVISIONS: dict[str, str] = {
    "EPL": "E0",
    "CHAMPIONSHIP": "E1",
    "LALIGA": "SP1",
    "SERIEA": "I1",
    "BUNDESLIGA": "D1",
    "LIGUE1": "F1",
    "EREDIVISIE": "N1",
}

_RESOLVER = TeamNameResolver()


def source_family(source: str) -> str:
    """`odds_api:pinnacle` -> `odds_api`; `polymarket` -> `polymarket`."""
    return source.split(":", 1)[0]


def canonical_league(family: str, raw: str) -> str:
    """Canonical league code, or `<family>:<raw>` when the league is not in `LEAGUE_CODES`."""
    return LEAGUE_CODES.get(family, {}).get(raw, f"{family}:{raw}")


# Spellings seen in live quotes that `data/teams.py` does not resolve: normalised source name -> club key.
PAPER_ALIASES: dict[str, str] = {
    "stade brestois 29": "brest",
    "stade brestois": "brest",
    "stade rennais 1901": "rennes",
    "nijmegen": "nec",
    "nec nijmegen": "nec",
    "espanyol barcelona": "espanol",
    "alkmaar": "az alkmaar",
    "enschede": "twente",
}


def canonical_team(name: str) -> str:
    """Normalised club key: the paper alias table first, then the `data/teams.py` alias target, else the
    normalised name itself."""
    key = normalise(name)
    if key in PAPER_ALIASES:
        return PAPER_ALIASES[key]
    resolved = _RESOLVER.resolve(name)
    return normalise(resolved or name)


def fixture_key(league: str, kickoff_at: datetime, home: str, away: str) -> str:
    """`<league>|<UTC date>|<home key>|<away key>`."""
    return f"{league}|{kickoff_at.date().isoformat()}|{canonical_team(home)}|{canonical_team(away)}"


def ambiguous_tokens(club_keys: list[str] | set[str]) -> set[str]:
    """Tokens that occur in two or more distinct club keys ("paris" in "paris" and "paris sg")."""
    counts: dict[str, int] = {}
    for key in set(club_keys):
        for tok in set(key.split()):
            counts[tok] = counts.get(tok, 0) + 1
    return {t for t, n in counts.items() if n > 1}


def team_similarity(a: str, b: str, ambiguous_tokens: set[str] | frozenset[str] = frozenset()) -> float:
    """Similarity of two canonical team keys in [0, 1]: max of the difflib ratio and token containment
    (shared tokens over the shorter key's tokens). Tokens in `ambiguous_tokens` (shared by several clubs)
    never count towards containment."""
    if a == b:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    shared = (ta & tb) - set(ambiguous_tokens)
    containment = len(shared) / min(len(ta), len(tb)) if ta and tb else 0.0
    return max(ratio, containment)


STRONG_MATCH = 0.85
WEAK_MATCH = 0.6


def fixture_score(
    home_a: str,
    away_a: str,
    home_b: str,
    away_b: str,
    ambiguous: set[str] | frozenset[str] = frozenset(),
) -> float | None:
    """Score of two club-key pairs as the same match, or None when they fail the rule: both clubs at least
    `WEAK_MATCH` and at least one at least `STRONG_MATCH` (an exact canonical match scores 1)."""
    sims = (team_similarity(home_a, home_b, ambiguous), team_similarity(away_a, away_b, ambiguous))
    if min(sims) >= WEAK_MATCH and max(sims) >= STRONG_MATCH:
        return sum(sims)
    return None


def match_candidate(
    league: str,
    kickoff_at: datetime,
    home_key: str,
    away_key: str,
    candidates: list[tuple[str, str, datetime, str, str]],
    max_kickoff_gap: timedelta = timedelta(hours=3),
    ambiguous: set[str] | frozenset[str] | None = None,
) -> str | None:
    """The one candidate `(fixture_key, league, kickoff_at, home_key, away_key)` that is the same match, or
    None when none passes or more than one does (an ambiguous link is refused rather than guessed).

    `ambiguous` defaults to the tokens shared by several clubs among the candidates and this event."""
    if ambiguous is None:
        clubs = {home_key, away_key} | {c[3] for c in candidates} | {c[4] for c in candidates}
        ambiguous = ambiguous_tokens(clubs)
    passing = [
        c[0]
        for c in candidates
        if c[1] == league
        and abs(c[2] - kickoff_at) <= max_kickoff_gap
        and fixture_score(home_key, away_key, c[3], c[4], ambiguous) is not None
    ]
    return passing[0] if len(passing) == 1 else None


def same_fixture(
    league_a: str,
    kickoff_a: datetime,
    home_a: str,
    away_a: str,
    league_b: str,
    kickoff_b: datetime,
    home_b: str,
    away_b: str,
    max_kickoff_gap: timedelta = timedelta(hours=3),
    ambiguous: set[str] | frozenset[str] = frozenset(),
) -> bool:
    """Whether two events (with canonical team keys) pass the same-match rule of `fixture_score`."""
    if league_a != league_b or abs(kickoff_a - kickoff_b) > max_kickoff_gap:
        return False
    return fixture_score(home_a, away_a, home_b, away_b, ambiguous) is not None
