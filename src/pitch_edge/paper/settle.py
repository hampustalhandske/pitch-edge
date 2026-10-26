"""Settle paper decisions: closing price, closing-line value, result and paper P&L.

Closing source, in order of preference:

1. Bookmaker / exchange decisions: the Betfair Exchange close (`betfair_ex_*_close`) for the match in the
   lake's `fd_matches` (football-data.co.uk), power de-vigged across home/draw/away.
2. Polymarket decisions: the last hourly bar known before kickoff (`hour_start + 1h <= kickoff_at`) for
   each of the fixture's three Yes tokens in the lake's `pm_soccer_bars_1h`, power-normalised when all
   three are present. The Betfair close is still recorded alongside as `sharp_fair_close_odds`.
3. Fallback, only once `grace` has passed since kickoff without (1)/(2): our own last quote snapshot
   captured after the decision and before kickoff (Pinnacle, then Betfair UK/EU, then Smarkets, then the
   decision's venue), labelled `own_snapshot:<source>@<captured_at>`, or `none` when no such snapshot
   exists. Fallbacks are written to `provisional_settlements`, never as final settlements; a lake close
   arriving later (up to `final_horizon` after kickoff) is written as the final settlement.

Closing sets whose implied probabilities do not sum to [0.95, 1.05] are rejected as data errors. A
decision whose fixture's kickoff moved by more than 3 hours (postponed) is flagged `void` instead.

The result comes from `fd_matches.ft_result` when the league is covered. Inside the grace period a
decision waits until both a lake close and a result exist; after it, whatever is known is written (a
missing result leaves `result` and `pnl` NULL). Settlements are written once and never changed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

import pandas as pd

from pitch_edge.data.teams import TeamNameResolver
from pitch_edge.paper.fixtures import FD_DIVISIONS, canonical_team, team_similarity
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


class CloseSource(Protocol):
    def fd_match(self, league: str, kickoff_at: datetime, home: str, away: str) -> FdClose | None: ...

    def pm_last_bars(self, asset_ids: list[str], before: datetime) -> dict[str, float]: ...


class LakeLike(Protocol):
    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame: ...


class LakeCloseSource:
    """`CloseSource` backed by the lakehouse API (`pitch_edge.lake.client.Lake`)."""

    def __init__(self, lake: LakeLike):
        self.lake = lake

    def fd_match(self, league: str, kickoff_at: datetime, home: str, away: str) -> FdClose | None:
        """The fd_matches row for this fixture (match_date within a day of the UTC kickoff date), or None."""
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
        """Close of the last hourly bar per token that was complete (hour_start + 1h) by `before`."""
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
    decision itself saw), in `_OWN_PREFERENCE` order then the decision's venue; None when there is none."""
    quotes = store.quotes_for_fixture(fixture, before=kickoff)
    quotes = quotes[quotes["captured_at"] > pd.Timestamp(decided_at)]
    if quotes.empty:
        return None
    for src in (*_OWN_PREFERENCE, venue):
        q = quotes[quotes["source"] == src]
        if q.empty:
            continue
        if src.startswith(_EXCHANGE_SOURCES):
            latest = q.sort_values("captured_at").groupby("selection").tail(1).set_index("selection")
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
) -> _Close | None:
    quotes = store.quotes_for_fixture(fixture, before=kickoff)
    pm = quotes[(quotes["source"] == "polymarket") & quotes["instrument_id"].notna()]
    if pm.empty:
        return None
    tokens = pm.sort_values("captured_at").groupby("selection")["instrument_id"].last().to_dict()
    bars = source.pm_last_bars(sorted(tokens.values()), kickoff)
    prices = {s: bars.get(t) for s, t in tokens.items()}
    p = prices.get(selection)
    if p is None or not 0.0 < p < 1.0:
        return None
    if set(SELECTIONS) <= set(prices) and all(v is not None and 0.0 < v < 1.0 for v in prices.values()):
        if not plausible_book([float(prices[s]) for s in SELECTIONS]):  # type: ignore[arg-type]
            logger.warning("pm bars for %s sum to %.3f; rejected as a close", fixture, sum(prices.values()))  # type: ignore[arg-type]
            return None
        fair = dict(zip(SELECTIONS, power_normalise_probs([float(prices[s]) for s in SELECTIONS]), strict=True))  # type: ignore[arg-type]
        return _Close(1.0 / p, 1.0 / fair[selection], "lake:pm_soccer_bars_1h")
    return _Close(1.0 / p, 1.0 / p, "lake:pm_soccer_bars_1h(single-token)")


def _kickoff_moved(store: PaperStore, fixture: str, kickoff: datetime, now: datetime) -> datetime | None:
    """The newest kickoff any source lists for `fixture`, when it differs from `kickoff` by more than
    `KICKOFF_MOVE_VOID` (postponed or rescheduled); else None."""
    quotes = store.quotes_for_fixture(fixture, before=now)
    if quotes.empty:
        return None
    latest = quotes.sort_values("captured_at")["kickoff_at"].iloc[-1].to_pydatetime()
    return latest if abs(latest - kickoff) > KICKOFF_MOVE_VOID else None


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

    Returns {"settled": final rows written, "waiting": decisions still inside `grace`}.
    """
    fixtures = store.fixtures().set_index("fixture_key")
    provisional = set(store.provisional_settlements()["decision_id"])
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
        if fx is not None and not past_horizon:
            try:
                fd = source.fd_match(str(fx["league"]), kickoff, str(fx["home"]), str(fx["away"]))
            except Exception as exc:  # the lake may be down; the decision simply waits
                logger.warning("fd_matches lookup for %s failed: %s", d.fixture_key, exc)
        sharp: _Close | None = None
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
        if past_horizon:
            primary = None
        elif d.venue == "polymarket":
            try:
                primary = _pm_bar_close(store, source, d.fixture_key, kickoff, d.selection)
            except Exception as exc:
                logger.warning("pm_soccer_bars_1h lookup for %s failed: %s", d.fixture_key, exc)
        else:
            primary = sharp
        result = fd.result if fd is not None else None
        in_grace = now - kickoff < grace
        if primary is not None and (result is not None or not in_grace):
            won = None if result is None else result == d.selection
            store.append_settlement(_settlement(d, primary, sharp, result, won, now))
            counts["settled"] += 1
            continue
        if in_grace:
            counts["waiting"] += 1
            continue
        if d.decision_id not in provisional:
            fallback = _own_snapshot_close(store, d.fixture_key, kickoff, d.selection, d.venue, decided)
            won = None if result is None else result == d.selection
            store.append_provisional_settlement(_settlement(d, fallback, sharp, result, won, now))
            n_provisional += 1
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
    )
