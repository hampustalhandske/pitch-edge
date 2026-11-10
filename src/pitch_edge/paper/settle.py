"""Settle paper decisions: closing price, closing-line value, result and paper P&L.

Closing source, in order of preference:

1. Bookmaker / exchange decisions: the Betfair Exchange close (`betfair_ex_*_close`) for the match in the
   lake's `fd_matches` (football-data.co.uk), power de-vigged across home/draw/away. For competitions
   football-data does not cover (`ODDSPAPI_DIVISIONS`: UCL, UEL) the benchmark is instead the Pinnacle 1X2
   close in the lake's OddsPapi tables, source `lake:oddspapi.pinnacle_close` (see `PinnacleClose`); it is a
   different benchmark from the Betfair close and is reported separately.
2. Polymarket decisions: the last hourly bar known before kickoff (`hour_start + 1h <= kickoff_at`) for
   each of the fixture's three Yes tokens in the lake's `pm_soccer_bars_1h`, power-normalised when all
   three are present. A bar that ended more than `PM_MAX_BAR_AGE` before kickoff is stale and is not a close,
   and neither is a token whose newest hour holds two bars with different closes.
   A Polymarket decision with no valid close (no token, no bar, only a stale bar, two different bars at the
   newest hour, an implausible book) gets a `no_benchmark` flag with the reason once `grace` has passed, and
   no fallback price is written for it: our own snapshot is not the market close. It is still asked for until
   `final_horizon`, and a valid bar arriving by then settles it. The Betfair close is still recorded alongside as `sharp_fair_close_odds`.
3. Fallback for bookmaker / exchange decisions, only once `grace` has passed since kickoff without (1): our own last quote snapshot
   captured after the decision and before kickoff (Pinnacle, then Betfair UK/EU, then Smarkets, then the
   decision's venue), labelled `own_snapshot:<source>@<captured_at>`, or `none` when no such snapshot
   exists. Fallbacks are written to `provisional_settlements`, never as final settlements; a lake close
   arriving later (up to `final_horizon` after kickoff) is written as the final settlement.

Closing sets whose implied probabilities do not sum to [0.95, 1.05] are rejected as data errors. A
decision whose fixture's kickoff moved by more than 3 hours (postponed) is flagged `void` instead.

The result comes from `fd_matches.ft_result` when the league is covered. Inside the grace period a
decision waits until both a lake close and a result exist (leagues with no result source, such as UCL, wait
for the close only and settle with `result` and `pnl` NULL); after it, whatever is known is written (a
missing result leaves `result` and `pnl` NULL). Settlements are written once and never changed. An OddsPapi-
benchmarked bookmaker decision that has no valid Pinnacle close once `grace` has passed gets a `no_benchmark`
flag whose reason says why (no fixture, ambiguous fixture, missing/inactive/stale close, two different Pinnacle
rows at the latest observed_at, implausible book sum).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Protocol

import pandas as pd

from pitch_edge.data.teams import TeamNameResolver
from pitch_edge.paper.fixtures import (
    FD_DIVISIONS,
    ODDSPAPI_DIVISIONS,
    ambiguous_tokens,
    canonical_team,
    fixture_score,
    oddspapi_team,
    team_similarity,
)
from pitch_edge.paper.pricing import clv_log, paper_pnl, power_devig_odds, power_normalise_probs
from pitch_edge.paper.store import SELECTIONS, PaperStore, Settlement

logger = logging.getLogger(__name__)

_FD_RESULT = {"H": "home", "D": "draw", "A": "away"}
_EXCHANGE_SOURCES = ("polymarket", "kalshi")
_OWN_PREFERENCE = ("odds_api:pinnacle", "odds_api:betfair_ex_uk", "odds_api:betfair_ex_eu", "odds_api:smarkets")


@dataclass(frozen=True)
class FdClose:
    """A football-data.co.uk match: Betfair Exchange closing odds (home, draw, away) if published, and result."""

    betfair_close: tuple[float, float, float] | None
    result: str | None


ODDSPAPI_KICKOFF_TOLERANCE = timedelta(hours=3)
ODDSPAPI_MAX_CLOSE_AGE = timedelta(minutes=15)
PINNACLE_CLOSE_SOURCE = "lake:oddspapi.pinnacle_close"
# Longest gap between the end of a token's newest complete hourly bar and kickoff for the bar to count as the
# Polymarket close. Set from the bar-age distribution of mapped 1X2 markets (2025-08..2026-10, n=8,438 with a
# bar): 90.6% are under 3 h, and beyond 3 h the remainder is a long tail of tokens that stopped trading.
PM_MAX_BAR_AGE = timedelta(hours=3)
_OUTCOME_CODES = {"home": 0, "draw": 1, "away": 2}


@dataclass(frozen=True)
class PinnacleClose:
    """The OddsPapi Pinnacle 1X2 close of one fixture: decimal odds (home, draw, away), or `reason` saying why
    there is none (no fixture matched, an outcome missing or inactive, two rows at the latest observed_at that
    differ in price or active, the last row too old). `retry` marks a missing fixture, which is not yet a
    verdict that no benchmark exists."""

    odds: tuple[float, float, float] | None
    reason: str | None = None
    retry: bool = False  # True when the fixture is not listed yet: a later lake refresh may supply it


@dataclass(frozen=True)
class PmBar:
    """A token's newest complete hourly bar before a cut-off: its close price and when the bar ended.

    `conflict` is True when the lake holds two bars with different closes for that hour; `close` is then NaN
    and the bar is not a close."""

    close: float
    ends_at: datetime
    conflict: bool = False


class CloseSource(Protocol):
    def fd_match(self, league: str, kickoff_at: datetime, home: str, away: str) -> FdClose | None: ...

    def pinnacle_close(self, league: str, kickoff_at: datetime, home: str, away: str) -> PinnacleClose: ...

    def pm_last_bars(self, asset_ids: list[str], before: datetime) -> dict[str, PmBar]: ...


class LakeLike(Protocol):
    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame: ...


class LakeCloseSource:
    """`CloseSource` backed by the lakehouse API (`pitch_edge.lake.client.Lake`).

    Build one per settle call. Every lake answer is kept for the life of the instance: the fd_matches rows
    of one (division, kickoff date) window and the bars of one (tokens, kickoff) set are queried once however
    many decisions share them, and a failed query is re-raised from memory, so a lake outage costs one failed
    query per window instead of one per decision.
    """

    def __init__(self, lake: LakeLike):
        self.lake = lake
        self._answers: dict[tuple[Any, ...], pd.DataFrame | Exception] = {}

    def _sql_once(self, key: tuple[Any, ...], query: str, params: list[Any]) -> pd.DataFrame:
        """`lake.sql(query, params)`, answered from memory when `key` was asked before."""
        if key not in self._answers:
            try:
                self._answers[key] = self.lake.sql(query, params)
            except Exception as exc:
                self._answers[key] = exc
        answer = self._answers[key]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def _fd_window(self, division: str, day: date) -> pd.DataFrame:
        """fd_matches rows of `division` with match_date within a day of `day`, ordered by match_date, home_team
        and away_team (whatever order the lake returned them in)."""
        df = self._sql_once(
            ("fd", division, day),
            "SELECT match_date, home_team, away_team, ft_result, betfair_ex_home_close, betfair_ex_draw_close, "
            "betfair_ex_away_close FROM fd_matches WHERE division = ? AND match_date BETWEEN ? AND ?",
            [division, (day - timedelta(days=1)).isoformat(), (day + timedelta(days=1)).isoformat()],
        )
        return df.sort_values(["match_date", "home_team", "away_team"], kind="stable").reset_index(drop=True)

    def fd_match(self, league: str, kickoff_at: datetime, home: str, away: str) -> FdClose | None:
        """The fd_matches row for this fixture (match_date within a day of the UTC kickoff date), or None.

        The row is the one whose resolved team names equal the fixture's; failing that, the one most similar
        to it (at least 0.75). Several rows for one pair of names take the earliest match_date; two different
        pairs equally most similar are ambiguous and give None."""
        division = FD_DIVISIONS.get(league)
        if division is None:
            return None
        df = self._fd_window(division, kickoff_at.date())
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
            best = df[sims == sims.max()]
            if len(best[["home_team", "away_team"]].drop_duplicates()) > 1:
                logger.warning("fd_matches: %d fixtures equally similar to %s v %s; no match", len(best), home, away)
                return None
            rows = best
        r = rows.iloc[0]
        closes = [r["betfair_ex_home_close"], r["betfair_ex_draw_close"], r["betfair_ex_away_close"]]
        betfair = tuple(float(c) for c in closes) if all(pd.notna(c) and float(c) > 1.0 for c in closes) else None
        result = _FD_RESULT.get(str(r["ft_result"])) if pd.notna(r["ft_result"]) else None
        return FdClose(betfair_close=betfair, result=result)  # type: ignore[arg-type]

    def _oddspapi_window(self, division: str, day: date) -> pd.DataFrame:
        """OddsPapi fixtures of `division` kicking off from the day before `day` to the day after it, with the
        distinct Pinnacle 1X2 (price, active) rows at the latest observed_at at or before kickoff per fixture and
        outcome: one row per outcome when that latest row is unique, more when rows at that time disagree, and a
        single row with a NULL outcome for a fixture without any. Rows are ordered by fixture, outcome, price and
        active."""
        return self._sql_once(
            ("oddspapi", division, day),
            "WITH fx AS (SELECT fixture_id, home_team, away_team, kickoff_utc FROM oddspapi_fixtures "
            "WHERE division = ? AND kickoff_utc >= CAST(? AS TIMESTAMPTZ) AND kickoff_utc < CAST(? AS TIMESTAMPTZ)), "
            "cl AS (SELECT DISTINCT o.fixture_id, o.outcome, o.price, o.active, o.observed_at "
            "FROM oddspapi_odds o JOIN fx ON o.fixture_id = fx.fixture_id "
            "WHERE o.bookmaker = 'pinnacle' AND o.market = '1x2' AND o.observed_at <= fx.kickoff_utc "
            "AND o.outcome IN ('home', 'draw', 'away') "
            "QUALIFY o.observed_at = max(o.observed_at) OVER (PARTITION BY o.fixture_id, o.outcome)) "
            "SELECT fx.fixture_id, fx.home_team, fx.away_team, fx.kickoff_utc, cl.outcome, cl.price, cl.active, "
            "cl.observed_at FROM fx LEFT JOIN cl ON cl.fixture_id = fx.fixture_id "
            "ORDER BY fx.fixture_id, cl.outcome, cl.price, cl.active",
            [division, (day - timedelta(days=1)).isoformat(), (day + timedelta(days=2)).isoformat()],
        )

    def pinnacle_close(self, league: str, kickoff_at: datetime, home: str, away: str) -> PinnacleClose:
        """The OddsPapi Pinnacle 1X2 close for a fixture of a competition in `ODDSPAPI_DIVISIONS`.

        The fixture is the one OddsPapi fixture of the competition kicking off within
        `ODDSPAPI_KICKOFF_TOLERANCE` of `kickoff_at` whose clubs pass `fixture_score` (canonical team keys);
        none or several passing means no close. The close is the latest `pinnacle` `1x2` row per outcome with
        `observed_at <= kickoff_utc`; price, active and observed_at all come from that one row, and two rows at
        the latest observed_at that differ in price or active refuse the close. It is valid only when all three
        outcomes exist with price > 1 and `active`, and the latest of the three rows is at most
        `ODDSPAPI_MAX_CLOSE_AGE` before kickoff.
        One lake query per (competition, kickoff date).
        """
        division = ODDSPAPI_DIVISIONS.get(league)
        if division is None:
            return PinnacleClose(None, f"{league} is not an OddsPapi competition")
        df = self._oddspapi_window(division, kickoff_at.date())
        fixtures = df.drop_duplicates("fixture_id")
        if fixtures.empty:
            return PinnacleClose(None, f"no OddsPapi {division} fixtures near {kickoff_at.date()}", retry=True)
        keys = {
            str(r.fixture_id): (oddspapi_team(r.home_team), oddspapi_team(r.away_team)) for r in fixtures.itertuples()
        }
        hk, ak = oddspapi_team(home), oddspapi_team(away)
        ambiguous = ambiguous_tokens({hk, ak} | {k for pair in keys.values() for k in pair})
        passing = [
            str(r.fixture_id)
            for r in fixtures.itertuples()
            if abs(r.kickoff_utc.to_pydatetime() - kickoff_at) <= ODDSPAPI_KICKOFF_TOLERANCE
            and fixture_score(hk, ak, *keys[str(r.fixture_id)], ambiguous) is not None
        ]
        if not passing:
            return PinnacleClose(None, f"no OddsPapi {division} fixture matches {home} v {away}", retry=True)
        if len(passing) > 1:
            return PinnacleClose(None, f"{len(passing)} OddsPapi {division} fixtures match {home} v {away}")
        rows = df[(df["fixture_id"].astype(str) == passing[0]) & df["outcome"].notna()]
        kickoff = fixtures[fixtures["fixture_id"].astype(str) == passing[0]]["kickoff_utc"].iloc[0].to_pydatetime()
        if rows.empty:
            return PinnacleClose(None, "OddsPapi lists the fixture but holds no Pinnacle 1x2 rows for it", retry=True)
        per_outcome = rows["outcome"].astype(str).value_counts()
        tied = [o for o in _OUTCOME_CODES if per_outcome.get(o, 0) > 1]
        if tied:
            return PinnacleClose(
                None, f"two Pinnacle rows at the latest observed_at differ in price or active for {'/'.join(tied)}"
            )
        by_outcome = {str(r.outcome): r for r in rows.itertuples()}
        missing = [o for o in _OUTCOME_CODES if o not in by_outcome or pd.isna(by_outcome[o].price)]
        if missing:
            return PinnacleClose(None, f"Pinnacle 1x2 close before kickoff lacks {'/'.join(missing)}")
        inactive = [o for o in _OUTCOME_CODES if not bool(by_outcome[o].active) or float(by_outcome[o].price) <= 1.0]
        if inactive:
            return PinnacleClose(
                None, f"Pinnacle 1x2 latest row is inactive or has price <= 1 for {'/'.join(inactive)}"
            )
        latest = max(by_outcome[o].observed_at.to_pydatetime() for o in _OUTCOME_CODES)
        age = kickoff - latest
        if age > ODDSPAPI_MAX_CLOSE_AGE:
            return PinnacleClose(
                None, f"Pinnacle 1x2 close is stale: latest row {age.total_seconds() / 60:.0f} min before kickoff"
            )
        return PinnacleClose(tuple(float(by_outcome[o].price) for o in _OUTCOME_CODES))  # type: ignore[arg-type]

    def pm_last_bars(self, asset_ids: list[str], before: datetime) -> dict[str, PmBar]:
        """The last hourly bar with a close per token that was complete (hour_start + 1h) by `before`, looking
        back 14 days: its close and its end time. Tokens without such a bar are absent. A token whose newest
        hour holds bars with different closes gets a `PmBar` with `conflict` set and a NaN close."""
        if not asset_ids:
            return {}
        marks = ", ".join("?" for _ in asset_ids)
        df = self._sql_once(
            ("pm", tuple(sorted(asset_ids)), before),
            f"SELECT DISTINCT asset_id, close, CAST(hour_start + INTERVAL 1 HOUR AS VARCHAR) AS ends_at "
            f"FROM pm_soccer_bars_1h WHERE asset_id IN ({marks}) AND close IS NOT NULL "
            f"AND hour_start >= CAST(? AS TIMESTAMPTZ) AND hour_start + INTERVAL 1 HOUR <= CAST(? AS TIMESTAMPTZ) "
            f"QUALIFY hour_start = max(hour_start) OVER (PARTITION BY asset_id) ORDER BY asset_id, close",
            [*asset_ids, (before - timedelta(days=14)).isoformat(), before.isoformat()],
        )
        df = df[df["close"].notna()]
        bars: dict[str, PmBar] = {}
        for asset_id, g in df.groupby(df["asset_id"].astype(str), sort=True):
            ends_at = pd.Timestamp(g["ends_at"].iloc[0]).tz_convert("UTC").to_pydatetime()
            closes = g["close"].astype(float).unique()
            bars[str(asset_id)] = (
                PmBar(float(closes[0]), ends_at) if len(closes) == 1 else PmBar(math.nan, ends_at, conflict=True)
            )
        return bars


@dataclass(frozen=True)
class _Close:
    closing_price: float
    fair_close_odds: float
    source: str


BOOKSUM_MIN, BOOKSUM_MAX = 0.95, 1.05
KICKOFF_MOVE_VOID = timedelta(hours=3)


def plausible_book(probs: list[float]) -> bool:
    """A closing 1X2 set is usable only when its implied probabilities sum to [BOOKSUM_MIN, BOOKSUM_MAX]."""
    return BOOKSUM_MIN <= sum(probs) <= BOOKSUM_MAX


def _own_snapshot_close(
    store: PaperStore, fixture: str, kickoff: datetime, selection: str, venue: str, decided_at: datetime
) -> _Close | None:
    """Our last quote set captured strictly after `decided_at` and before `kickoff` (never the snapshot the
    decision itself saw), in `_OWN_PREFERENCE` order then the decision's venue; None when there is none.

    When two events of one source map to the fixture and quote a selection at the same capture time, the quote
    of the smallest `source_event_id` (then `quote_id`) is used, the rule of `PaperStore.latest_quotes`."""
    quotes = store.quotes_for_fixture(fixture, before=kickoff)
    quotes = quotes[quotes["captured_at"] > pd.Timestamp(decided_at)]
    if quotes.empty:
        return None
    for src in (*_OWN_PREFERENCE, venue):
        q = quotes[quotes["source"] == src]
        if q.empty:
            continue
        q = q.sort_values(["captured_at", "selection", "source_event_id", "quote_id"], kind="stable").drop_duplicates(
            ["captured_at", "selection"]
        )
        if src.startswith(_EXCHANGE_SOURCES):
            latest = q.groupby("selection").tail(1).set_index("selection")
            if not set(SELECTIONS) <= set(latest.index):
                continue
            mids = []
            for s in SELECTIONS:
                bid, ask, last = latest.loc[s, ["bid", "ask", "last"]]
                mid = (bid + ask) / 2 if pd.notna(bid) and pd.notna(ask) else last
                mids.append(float(mid) if pd.notna(mid) else math.nan)
            if not all(0.0 < m < 1.0 for m in mids):
                continue
            fair = dict(zip(SELECTIONS, power_normalise_probs(mids), strict=True))
            raw = dict(zip(SELECTIONS, mids, strict=True))
            stamp = latest["captured_at"].max()
            return _Close(1.0 / raw[selection], 1.0 / fair[selection], f"own_snapshot:{src}@{stamp.isoformat()}")
        for stamp, g in sorted(q.groupby("captured_at"), key=lambda kv: kv[0], reverse=True):
            g = g.set_index("selection")
            if set(SELECTIONS) <= set(g.index) and g.loc[list(SELECTIONS), "decimal_odds"].gt(1.0).all():
                odds = [float(g.loc[s, "decimal_odds"]) for s in SELECTIONS]
                fair = dict(zip(SELECTIONS, power_devig_odds(odds), strict=True))
                return _Close(
                    float(g.loc[selection, "decimal_odds"]),
                    1.0 / fair[selection],
                    f"own_snapshot:{src}@{stamp.isoformat()}",
                )
    return None


def _pm_bar_close(
    store: PaperStore, source: CloseSource, fixture: str, kickoff: datetime, selection: str
) -> tuple[_Close | None, str | None]:
    """The Polymarket close of `selection` from the fixture's Yes-token bars, and the reason it was refused.

    Exactly one of the two is None. Bars that ended more than `PM_MAX_BAR_AGE` before kickoff, and bars in
    `conflict` (two different closes at the newest hour), are dropped: on the selection's own token either
    refuses the close, on another token it only drops the three-way normalisation. The token of a selection is
    the instrument of its latest Polymarket quote, the largest instrument id among quotes captured at that same
    time."""
    quotes = store.quotes_for_fixture(fixture, before=kickoff)
    pm = quotes[(quotes["source"] == "polymarket") & quotes["instrument_id"].notna()]
    if pm.empty or selection not in set(pm["selection"]):
        return None, f"no Polymarket token recorded for {selection}"
    tokens = (
        pm.sort_values(["captured_at", "instrument_id"], kind="stable")
        .groupby("selection")["instrument_id"]
        .last()
        .to_dict()
    )
    bars = source.pm_last_bars(sorted(tokens.values()), kickoff)
    own = bars.get(tokens[selection])
    if own is None:
        return None, f"no Polymarket {selection} bar in the 14 days before kickoff"
    if own.conflict:
        return (
            None,
            f"two Polymarket {selection} bars at the newest hour (ending {own.ends_at.isoformat()}) "
            "have different closes",
        )
    if kickoff - own.ends_at > PM_MAX_BAR_AGE:
        age_h = (kickoff - own.ends_at).total_seconds() / 3600
        limit_h = PM_MAX_BAR_AGE.total_seconds() / 3600
        return (
            None,
            f"Polymarket close is stale: newest {selection} bar ended {age_h:.1f} h before kickoff (limit {limit_h:.0f} h)",
        )
    prices = {
        s: bars[t].close if t in bars and not bars[t].conflict and kickoff - bars[t].ends_at <= PM_MAX_BAR_AGE else None
        for s, t in tokens.items()
    }
    p = prices.get(selection)
    if p is None or not 0.0 < p < 1.0:
        return None, f"Polymarket {selection} bar close {p} is outside (0, 1)"
    if set(SELECTIONS) <= set(prices) and all(v is not None and 0.0 < v < 1.0 for v in prices.values()):
        if not plausible_book([float(prices[s]) for s in SELECTIONS]):  # type: ignore[arg-type]
            total = sum(prices.values())  # type: ignore[arg-type]
            logger.warning("pm bars for %s sum to %.3f; rejected as a close", fixture, total)
            return None, f"Polymarket bars sum to {total:.3f}, outside the plausible book range"
        fair = dict(zip(SELECTIONS, power_normalise_probs([float(prices[s]) for s in SELECTIONS]), strict=True))  # type: ignore[arg-type]
        return _Close(1.0 / p, 1.0 / fair[selection], "lake:pm_soccer_bars_1h"), None
    return _Close(1.0 / p, 1.0 / p, "lake:pm_soccer_bars_1h(single-token)"), None


def _kickoff_moved(store: PaperStore, fixture: str, kickoff: datetime, now: datetime) -> datetime | None:
    """The newest kickoff any source lists for `fixture`, when it differs from `kickoff` by more than
    `KICKOFF_MOVE_VOID` (postponed or rescheduled); else None. The newest kickoff is the one of the latest
    capture; when quotes of that capture list different kickoffs, the latest of them."""
    quotes = store.quotes_for_fixture(fixture, before=now)
    if quotes.empty:
        return None
    latest = quotes.sort_values(["captured_at", "kickoff_at"], kind="stable")["kickoff_at"].iloc[-1].to_pydatetime()
    return latest if abs(latest - kickoff) > KICKOFF_MOVE_VOID else None


def _pinnacle_sharp(pin: PinnacleClose, selection: str, fixture: str) -> tuple[_Close | None, str | None]:
    """The de-vigged `_Close` of `selection` from an OddsPapi Pinnacle close, or (None, why there is none)."""
    if pin.odds is None:
        return None, pin.reason or "no Pinnacle close"
    if not plausible_book([1.0 / o for o in pin.odds]):
        total = sum(1.0 / o for o in pin.odds)
        logger.warning("Pinnacle close %s for %s fails the book-sum check", pin.odds, fixture)
        return None, f"Pinnacle close book sum {total:.3f} outside [{BOOKSUM_MIN}, {BOOKSUM_MAX}]"
    fair = dict(zip(SELECTIONS, power_devig_odds(pin.odds), strict=True))
    raw = dict(zip(SELECTIONS, pin.odds, strict=True))
    return _Close(raw[selection], 1.0 / fair[selection], PINNACLE_CLOSE_SOURCE), None


def settle(
    store: PaperStore,
    source: CloseSource,
    now: datetime,
    *,
    min_delay: timedelta = timedelta(hours=3),
    grace: timedelta = timedelta(days=4),
    final_horizon: timedelta = timedelta(days=30),
) -> dict[str, int]:
    """Settle every decision without a final settlement whose kickoff was at least `min_delay` before `now`.

    - A decision whose fixture's kickoff moved by more than 3 h is flagged `void` and never settled.
    - Final settlement (table `settlements`) only from a lake-sourced close whose book sum is plausible;
      inside `grace` it also waits for the result.
    - After `grace` with no lake close: one provisional row (our own post-decision snapshot, or `none`);
      the lake keeps being asked until `final_horizon` after kickoff, and a final row then supersedes it.
    - An OddsPapi-benchmarked bookmaker decision is flagged `no_benchmark` (with the reason) when its fixture
      is listed but its Pinnacle close is invalid, or when `final_horizon` passes without a valid close; a
      fixture OddsPapi does not list yet is retried on later calls, never flagged before the horizon.
    - A Polymarket decision without a valid bar close (none, or the newest older than `PM_MAX_BAR_AGE` at
      kickoff) is flagged `no_benchmark` with the reason once `grace` has passed and never gets a provisional
      row; it is still asked for until `final_horizon`, and a valid bar then settles it. A failed lake lookup is
      not a verdict: the decision waits.

    Returns {"settled": final rows written, "waiting": decisions still inside `grace`}.
    """
    fixtures = store.fixtures().set_index("fixture_key")
    provisional = set(store.provisional_settlements()["decision_id"])
    flags = store.decision_flags()
    flagged = set(flags.loc[flags["flag"] == "no_benchmark", "decision_id"])
    counts = {"settled": 0, "waiting": 0}
    n_provisional = 0
    for d in store.unsettled_decisions().itertuples():
        kickoff = d.kickoff_at.to_pydatetime()
        decided = d.decided_at.to_pydatetime()
        if kickoff + min_delay > now:
            continue
        moved = _kickoff_moved(store, d.fixture_key, kickoff, now)
        if moved is not None:
            store.flag_decision(
                d.decision_id,
                "void",
                f"kickoff moved from {kickoff.isoformat()} to {moved.isoformat()}",
                flagged_at=now,
            )
            continue
        past_horizon = now - kickoff > final_horizon
        fx = fixtures.loc[d.fixture_key] if d.fixture_key in fixtures.index else None
        fd: FdClose | None = None
        league = str(fx["league"]) if fx is not None else None
        oddspapi = league is not None and league not in FD_DIVISIONS and league in ODDSPAPI_DIVISIONS
        no_close_reason: str | None = None
        retry_later = False
        if fx is not None and not past_horizon and not oddspapi:
            try:
                fd = source.fd_match(str(fx["league"]), kickoff, str(fx["home"]), str(fx["away"]))
            except Exception as exc:  # the lake may be down; the decision simply waits
                logger.warning("fd_matches lookup for %s failed: %s", d.fixture_key, exc)
        sharp: _Close | None = None
        if fx is not None and not past_horizon and oddspapi:
            try:
                pin = source.pinnacle_close(str(fx["league"]), kickoff, str(fx["home"]), str(fx["away"]))
            except Exception as exc:
                logger.warning("oddspapi lookup for %s failed: %s", d.fixture_key, exc)
                no_close_reason, retry_later = f"OddsPapi lookup failed: {exc}", True
            else:
                sharp, no_close_reason = _pinnacle_sharp(pin, d.selection, d.fixture_key)
                retry_later = pin.retry
        if fd is not None and fd.betfair_close is not None:
            if plausible_book([1.0 / o for o in fd.betfair_close]):
                fair = dict(zip(SELECTIONS, power_devig_odds(fd.betfair_close), strict=True))
                sharp = _Close(
                    dict(zip(SELECTIONS, fd.betfair_close, strict=True))[d.selection],
                    1.0 / fair[d.selection],
                    "lake:fd_matches.betfair_ex_close",
                )
            else:
                logger.warning("Betfair close %s for %s fails the book-sum check", fd.betfair_close, d.fixture_key)
        primary: _Close | None = None
        pm_refusal: str | None = None
        if past_horizon:
            primary = None
        elif d.venue == "polymarket":
            try:
                primary, pm_refusal = _pm_bar_close(store, source, d.fixture_key, kickoff, d.selection)
            except Exception as exc:
                logger.warning("pm_soccer_bars_1h lookup for %s failed: %s", d.fixture_key, exc)
        else:
            primary = sharp
        result = fd.result if fd is not None else None
        in_grace = now - kickoff < grace
        has_result_source = league in FD_DIVISIONS
        if primary is not None and (result is not None or not in_grace or not has_result_source):
            won = None if result is None else result == d.selection
            store.append_settlement(_settlement(d, primary, sharp, result, won, now))
            counts["settled"] += 1
            continue
        if in_grace:
            counts["waiting"] += 1
            continue
        if d.venue == "polymarket":
            reason = pm_refusal or ("no valid Polymarket close within the final horizon" if past_horizon else None)
            if reason is not None and d.decision_id not in flagged:
                store.flag_decision(d.decision_id, "no_benchmark", reason, flagged_at=now)
                flagged.add(d.decision_id)
            continue
        if oddspapi and past_horizon and d.venue != "polymarket" and d.decision_id not in flagged:
            store.flag_decision(
                d.decision_id, "no_benchmark", "no valid Pinnacle close within the final horizon", flagged_at=now
            )
            flagged.add(d.decision_id)
        if d.decision_id not in provisional:
            fallback = _own_snapshot_close(store, d.fixture_key, kickoff, d.selection, d.venue, decided)
            won = None if result is None else result == d.selection
            store.append_provisional_settlement(_settlement(d, fallback, sharp, result, won, now))
            n_provisional += 1
            if (
                oddspapi
                and d.venue != "polymarket"
                and no_close_reason is not None
                and not retry_later
                and d.decision_id not in flagged
            ):
                store.flag_decision(d.decision_id, "no_benchmark", no_close_reason, flagged_at=now)
                flagged.add(d.decision_id)
    if n_provisional:
        logger.info("%d provisional settlements written (no lake close yet)", n_provisional)
    return counts


def _settlement(
    d: Any, close: _Close | None, sharp: _Close | None, result: str | None, won: bool | None, now: datetime
) -> Settlement:
    return Settlement(
        decision_id=d.decision_id,
        closing_price=close.closing_price if close else None,
        fair_close_odds=close.fair_close_odds if close else None,
        closing_source=close.source if close else "none",
        result=result,
        clv_log=clv_log(d.price_taken, close.fair_close_odds) if close else None,
        pnl=paper_pnl(d.paper_stake_units, d.price_taken, won),
        settled_at=now,
        sharp_fair_close_odds=sharp.fair_close_odds if sharp else None,
        clv_vs_sharp_log=clv_log(d.price_taken, sharp.fair_close_odds) if sharp else None,
        sharp_source=sharp.source if sharp else None,
    )
