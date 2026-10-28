"""Settling through `LakeCloseSource` (one lake query per window) equals the one-query-per-decision lookup.

A set of decisions shaped like the paper loop's recorded ones is settled twice against one fake lake that
answers from fixed frames and records every query: once through the current `LakeCloseSource` and once through
`PerDecisionLakeCloseSource`, a reference that asks the lake afresh for every decision. The settlements, flags
and counts must be identical; only the number of lake queries may differ.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from pitch_edge.data.teams import TeamNameResolver
from pitch_edge.paper.fixtures import FD_DIVISIONS, canonical_team, team_similarity
from pitch_edge.paper.settle import _FD_RESULT, FdClose, LakeCloseSource, settle
from pitch_edge.paper.store import Decision, PaperStore, Quote

pytestmark = pytest.mark.unit

SEL = ("home", "draw", "away")
KO1 = datetime(2026, 10, 24, 14, 0, tzinfo=UTC)
KO1_LATE = datetime(2026, 10, 24, 16, 30, tzinfo=UTC)
KO2 = datetime(2026, 10, 25, 15, 0, tzinfo=UTC)
KO_OLD = datetime(2026, 10, 15, 14, 0, tzinfo=UTC)  # past the 4-day grace
NOW = datetime(2026, 10, 25, 20, 0, tzinfo=UTC)
PIN = "odds_api:pinnacle"


# --------------------------------------------------------------------------- reference: one query per decision
class PerDecisionLakeCloseSource:
    """The lookup as it was before windows were shared: every call goes to the lake, nothing is remembered."""

    def __init__(self, lake: Any):
        self.lake = lake

    def fd_match(self, league: str, kickoff_at: datetime, home: str, away: str) -> FdClose | None:
        division = FD_DIVISIONS.get(league)
        if division is None:
            return None
        day = kickoff_at.date()
        df = self.lake.sql(
            "SELECT match_date, home_team, away_team, ft_result, betfair_ex_home_close, betfair_ex_draw_close, "
            "betfair_ex_away_close FROM fd_matches WHERE division = ? AND match_date BETWEEN ? AND ?",
            [division, (day - timedelta(days=1)).isoformat(), (day + timedelta(days=1)).isoformat()],
        )
        if df.empty:
            return None
        resolver = TeamNameResolver(set(df["home_team"]) | set(df["away_team"]))
        h, a = resolver.resolve(home), resolver.resolve(away)
        rows = df[(df["home_team"] == h) & (df["away_team"] == a)] if h and a else df.iloc[0:0]
        if rows.empty:
            hk, ak = canonical_team(home), canonical_team(away)
            sims = df.apply(
                lambda r: min(
                    team_similarity(hk, canonical_team(r["home_team"])),
                    team_similarity(ak, canonical_team(r["away_team"])),
                ),
                axis=1,
            )
            if sims.max() < 0.75:
                return None
            rows = df.loc[[sims.idxmax()]]
        r = rows.iloc[0]
        closes = [r["betfair_ex_home_close"], r["betfair_ex_draw_close"], r["betfair_ex_away_close"]]
        betfair = tuple(float(c) for c in closes) if all(pd.notna(c) and float(c) > 1.0 for c in closes) else None
        result = _FD_RESULT.get(str(r["ft_result"])) if pd.notna(r["ft_result"]) else None
        return FdClose(betfair_close=betfair, result=result)  # type: ignore[arg-type]

    def pm_last_bars(self, asset_ids: list[str], before: datetime) -> dict[str, float]:
        if not asset_ids:
            return {}
        marks = ", ".join("?" for _ in asset_ids)
        df = self.lake.sql(
            f"SELECT asset_id, arg_max(close, hour_start) AS close FROM pm_soccer_bars_1h "
            f"WHERE asset_id IN ({marks}) AND hour_start >= CAST(? AS TIMESTAMPTZ) "
            f"AND hour_start + INTERVAL 1 HOUR <= CAST(? AS TIMESTAMPTZ) GROUP BY asset_id",
            [*asset_ids, (before - timedelta(days=14)).isoformat(), before.isoformat()],
        )
        return {str(r.asset_id): float(r.close) for r in df.itertuples() if pd.notna(r.close)}


# --------------------------------------------------------------------------- the fake lake
FD_COLS = [
    "match_date",
    "home_team",
    "away_team",
    "ft_result",
    "betfair_ex_home_close",
    "betfair_ex_draw_close",
    "betfair_ex_away_close",
]
NAN = math.nan
# (division, match_date, home, away, ft_result, home close, draw close, away close)
FD_ROWS = [
    ("E0", "2026-10-24", "Arsenal", "Everton", "H", 1.8, 3.9, 5.0),
    ("E0", "2026-10-24", "Chelsea", "Brighton", "D", 2.2, 3.5, 3.4),
    ("E0", "2026-10-24", "Liverpool", "Fulham", "H", NAN, NAN, NAN),  # played, no Betfair close published
    ("E0", "2026-10-24", "Man City", "Leeds", "A", 1.5, 2.0, 2.5),  # closes imply a 1.17 book: rejected
    ("E0", "2026-10-24", "Tottenham", "Burnley", "H", 1.9, 3.6, 4.2),
    ("E0", "2026-10-25", "West Ham", "Southampton", "A", 3.0, 3.4, 2.4),
    ("SP1", "2026-10-24", "Barcelona", "Sevilla", "H", 1.4, 5.0, 8.0),
    ("SP1", "2026-10-24", "Real Madrid", "Valencia", "H", 1.3, 5.5, 9.0),
]
PM_BARS = {
    "tok-chelsea-home": 0.42,
    "tok-chelsea-draw": 0.30,
    "tok-chelsea-away": 0.30,
    "tok-westham-home": 0.33,
    "tok-westham-draw": 0.29,
    "tok-westham-away": 0.40,
    "tok-liverpool-home": 0.55,  # only one of the three tokens has a bar
}
FAILING_DIVISION = "SP1"


class RecordingLake:
    """Answers `fd_matches` and `pm_soccer_bars_1h` queries from fixed frames; SP1 windows raise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fd = pd.DataFrame(FD_ROWS, columns=["division", *FD_COLS])
        self.fd["match_date"] = pd.to_datetime(self.fd["match_date"])

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        params = list(params or [])
        self.calls.append((query, tuple(params)))
        if "FROM fd_matches" in query:
            division, lo, hi = params
            if division == FAILING_DIVISION:
                raise RuntimeError("lake: 503 service unavailable")
            m = (self.fd["division"] == division) & self.fd["match_date"].between(pd.Timestamp(lo), pd.Timestamp(hi))
            return self.fd.loc[m, FD_COLS].reset_index(drop=True)
        assert "FROM pm_soccer_bars_1h" in query
        assets = params[:-2]
        return pd.DataFrame(
            [{"asset_id": a, "close": PM_BARS[a]} for a in assets if a in PM_BARS], columns=["asset_id", "close"]
        )

    def fd_windows(self) -> list[tuple[str, date]]:
        out = []
        for q, p in self.calls:
            if "FROM fd_matches" in q:
                out.append((p[0], date.fromisoformat(p[1]) + timedelta(days=1)))
        return out

    def pm_sets(self) -> list[tuple[tuple[str, ...], str]]:
        return [(tuple(sorted(p[:-2])), p[-1]) for q, p in self.calls if "FROM pm_soccer_bars_1h" in q]


# --------------------------------------------------------------------------- recorded-decision scenario
def _quotes(eid, home, away, league, ko, decided, *, odds=None, pm_tokens=None, late_ko=None, late=True):
    """Pinnacle (or Polymarket, with `pm_tokens`) quotes at decision time plus a late Pinnacle snapshot."""
    out = []
    if pm_tokens:
        for s, a in zip(SEL, (0.44, 0.28, 0.26), strict=True):
            out.append(
                Quote(
                    source="polymarket",
                    source_event_id=f"pm-{eid}",
                    home=home,
                    away=away,
                    league=league,
                    kickoff_at=ko,
                    selection=s,
                    captured_at=decided - timedelta(minutes=5),
                    bid=a - 0.02,
                    ask=a,
                    size=50,
                    instrument_id=pm_tokens[s],
                )
            )
    for s, o in zip(SEL, odds or (2.1, 3.4, 3.9), strict=True):
        out.append(
            Quote(
                source=PIN,
                source_event_id=eid,
                home=home,
                away=away,
                league=league,
                kickoff_at=ko,
                selection=s,
                captured_at=decided - timedelta(hours=1),
                decimal_odds=o,
            )
        )
    if late:
        for s, o in zip(SEL, (2.0, 3.5, 4.1), strict=True):
            out.append(
                Quote(
                    source=PIN,
                    source_event_id=eid,
                    home=home,
                    away=away,
                    league=league,
                    kickoff_at=late_ko or ko,
                    selection=s,
                    captured_at=ko - timedelta(hours=2),
                    decimal_odds=o,
                )
            )
    return out


def _tokens(name):
    return {s: f"tok-{name}-{s}" for s in SEL}


# eid, home, away, league, kickoff, late_ko (moved), has_late_snapshot
EVENTS = [
    ("arsenal", "Arsenal", "Everton", "EPL", KO1, None, True),
    ("chelsea", "Chelsea", "Brighton", "EPL", KO1, None, True),
    ("liverpool", "Liverpool", "Fulham", "EPL", KO1_LATE, None, True),
    ("mancity", "Man City", "Leeds", "EPL", KO1_LATE, None, True),
    ("newcastle", "Newcastle", "Brentford", "EPL", KO1_LATE, None, True),  # no lake row at all
    ("spurs", "Tottenham", "Burnley", "EPL", KO1, KO1 + timedelta(days=1), True),  # postponed
    ("westham", "West Ham", "Southampton", "EPL", KO2, None, True),
    ("brentford-old", "Brentford", "Wolves", "EPL", KO_OLD, None, True),  # past grace, no lake row
    ("barca", "Barcelona", "Sevilla", "LALIGA", KO1, None, True),  # lake query raises
    ("madrid", "Real Madrid", "Valencia", "LALIGA", KO1_LATE, None, True),  # same failing window
    ("ajax", "Ajax", "Benfica", "UCL", KO1, None, False),  # league not in football-data
]
# strategy, event, venue, selection, price
DECISIONS = [
    ("s-pin", "arsenal", PIN, "home", 2.3),
    ("s-pin", "chelsea", PIN, "draw", 3.8),
    ("s-pin", "liverpool", PIN, "home", 2.0),
    ("s-pin", "mancity", PIN, "away", 3.1),
    ("s-pin", "newcastle", PIN, "away", 3.0),
    ("s-pin", "spurs", PIN, "home", 2.2),
    ("s-pin", "westham", PIN, "away", 2.9),
    ("s-pin", "brentford-old", PIN, "home", 2.5),
    ("s-pin", "barca", PIN, "home", 1.5),
    ("s-pin", "madrid", PIN, "away", 9.5),
    ("s-pin", "ajax", PIN, "draw", 3.7),
    # the same Polymarket token sets read by several strategies
    ("pm-a", "chelsea", "polymarket", "home", 2.5),
    ("pm-b", "chelsea", "polymarket", "draw", 3.6),
    ("pm-c", "chelsea", "polymarket", "away", 3.5),
    ("pm-a", "westham", "polymarket", "away", 2.6),
    ("pm-b", "westham", "polymarket", "home", 3.2),
    ("pm-a", "liverpool", "polymarket", "home", 1.9),  # single-token bar
    ("pm-b", "liverpool", "polymarket", "draw", 4.0),  # no bar for its token
    ("pm-a", "newcastle", "polymarket", "home", 2.4),  # no bars at all
]
PM_EVENTS = {"chelsea", "westham", "liverpool", "newcastle"}


def _build(store: PaperStore) -> None:
    by_eid = {e[0]: e for e in EVENTS}
    for eid, home, away, league, ko, late_ko, late in EVENTS:
        decided = ko - timedelta(days=3)
        store.add_quotes(
            _quotes(
                eid,
                home,
                away,
                league,
                ko,
                decided,
                pm_tokens=_tokens(eid.split("-")[0]) if eid in PM_EVENTS else None,
                late_ko=late_ko,
                late=late,
            )
        )
    q = store.quotes()
    for strategy, eid, venue, selection, price in DECISIONS:
        _, _, _, _, ko, _, _ = by_eid[eid]
        decided = ko - timedelta(days=3)
        rows = q[(q["source_event_id"] == (f"pm-{eid}" if venue == "polymarket" else eid)) & (q["source"] == venue)]
        rows = rows[rows["captured_at"] <= pd.Timestamp(decided)]
        store.append_decision(
            Decision(
                strategy_id=strategy,
                hypothesis_id="control",
                decided_at=decided,
                fixture_key=rows["fixture_key"].iloc[0],
                kickoff_at=ko,
                venue=venue,
                selection=selection,
                price_taken=price,
                paper_stake_units=1.0,
                model_prob=0.45,
                inputs_quote_ids=list(rows["quote_id"]),
                code_version="sha",
                decision_id=f"{strategy}:{eid}",
            )
        )


def _run(tmp_path, name, make_source):
    store = PaperStore(tmp_path / f"{name}.duckdb")
    _build(store)
    lake = RecordingLake()
    counts = settle(store, make_source(lake), now=NOW)
    out = {
        "counts": counts,
        "final": store.settlements().sort_values("decision_id").reset_index(drop=True),
        "provisional": store.provisional_settlements().sort_values("decision_id").reset_index(drop=True),
        "void": store.voided_decision_ids(),
        "unsettled": sorted(store.unsettled_decisions()["decision_id"]),
    }
    store.close()
    return out, lake


def _outcome_by_decision(r):
    out = dict.fromkeys(r["unsettled"], "waiting")
    out.update(dict.fromkeys(r["final"]["decision_id"], "final"))
    out.update({d: "provisional" for d in r["provisional"]["decision_id"] if d not in out or out[d] == "waiting"})
    out.update(dict.fromkeys(r["void"], "void"))
    return out


@pytest.fixture
def both(tmp_path):
    new, new_lake = _run(tmp_path, "new", LakeCloseSource)
    old, old_lake = _run(tmp_path, "old", PerDecisionLakeCloseSource)
    return new, new_lake, old, old_lake


def test_scenario_exercises_every_outcome(both):
    new, *_ = both
    outcomes = set(_outcome_by_decision(new).values())
    assert {"final", "provisional", "void", "waiting"} <= outcomes
    assert new["final"]["closing_price"].notna().any()
    assert new["final"]["clv_log"].notna().any()
    assert (new["final"]["closing_source"] == "lake:pm_soccer_bars_1h(single-token)").any()
    assert new["provisional"]["closing_source"].str.startswith("own_snapshot").any()


def test_shared_windows_settle_identically_to_one_query_per_decision(both):
    new, _, old, _ = both
    assert new["counts"] == old["counts"]
    assert _outcome_by_decision(new) == _outcome_by_decision(old)
    assert new["void"] == old["void"]
    assert new["unsettled"] == old["unsettled"]
    # close odds, fair close, sharp close, CLV, CLV vs sharp, result and P&L, decision by decision
    pd.testing.assert_frame_equal(new["final"], old["final"], check_exact=True)
    pd.testing.assert_frame_equal(new["provisional"], old["provisional"], check_exact=True)


def test_shared_windows_are_each_queried_once(both):
    _, new_lake, _, old_lake = both
    fd_new, pm_new = new_lake.fd_windows(), new_lake.pm_sets()
    assert max(Counter(fd_new).values()) == 1
    assert max(Counter(pm_new).values()) == 1
    # one failing window (SP1, 2026-10-24) hit by two decisions costs one failed query
    assert fd_new.count((FAILING_DIVISION, date(2026, 10, 24))) == 1
    assert old_lake.fd_windows().count((FAILING_DIVISION, date(2026, 10, 24))) == 2
    # the same windows and token sets are asked as before, just not repeatedly
    assert set(fd_new) == set(old_lake.fd_windows())
    assert set(pm_new) == set(old_lake.pm_sets())
    assert len(new_lake.calls) < len(old_lake.calls)
    assert set(new_lake.calls) == set(old_lake.calls)
