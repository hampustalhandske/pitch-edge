"""Read-only quote collectors for the paper ledger.

Every request is a public market-data GET through `CachedHttpClient.get_live` (honest User-Agent,
per-host throttle, retries, circuit breaker, and no disk cache: a live price served from a cache file
would carry the wrong capture time). No collector calls an authenticated or order endpoint.

- `OddsApiCollector`: the-odds-api `/v4/sports/{league}/odds`, h2h, every book in regions uk,eu (soft books
  plus Pinnacle, Betfair Exchange UK/EU and Smarkets in the same response; 2 credits per league per call,
  1 with a `bookmakers` filter). The cost of each call is measured (x-requests-last, cross-checked against
  the x-requests-used delta) and the league list is trimmed so the projected monthly spend stays under
  `max_monthly_credits`; the projection is logged on every run. Budget guard: the last reported
  `x-requests-remaining` is kept in the store's `collect_log`; no call is made while it is below
  `budget_floor`, and a run stops as soon as a response reports it below the floor. Cadence guard:
  a league is fetched at most once per `min_hours_between_calls`.
- `PolymarketCollector`: Gamma `/events/keyset?tag_slug=soccer` for upcoming match events with a full
  1X2 set (three `moneyline` Yes/No markets), then CLOB `/book` for each Yes token's best bid/ask.
- `KalshiCollector`: `/events?series_ticker=...&with_nested_markets=true` for soccer game series;
  kickoff from the event's milestone `start_date` (the market's `occurrence_datetime` is kickoff + 3h
  in every case checked, and is the fallback).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pitch_edge.data.http import CachedHttpClient, LiveHttpError
from pitch_edge.paper.fixtures import canonical_league, canonical_team, team_similarity
from pitch_edge.paper.store import PaperStore, Quote, utc_now

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB = "https://clob.polymarket.com"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

DEFAULT_ODDS_API_LEAGUES = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_netherlands_eredivisie",
]
# Books treated as the sharp reference / exchanges; every other book the-odds-api returns is a soft book.
SHARP_BOOKMAKERS = ["pinnacle", "betfair_ex_uk", "betfair_ex_eu", "smarkets"]
EXCHANGE_BOOKMAKERS = ["betfair_ex_uk", "betfair_ex_eu", "betfair_ex_au", "smarkets", "matchbook"]
DEFAULT_KALSHI_SERIES = [
    "KXEPLGAME",
    "KXLALIGAGAME",
    "KXSERIEAGAME",
    "KXBUNDESLIGAGAME",
    "KXLIGUE1GAME",
    "KXUCLGAME",
    "KXEREDIVISIEGAME",
    "KXUELGAME",
    "KXEFLCHAMPIONSHIPGAME",
]
KALSHI_OCCURRENCE_AFTER_KICKOFF = timedelta(hours=3)

Clock = Callable[[], datetime]


def parse_utc(text: str) -> datetime:
    """ISO-8601 timestamp ('Z', '+00', '+00:00' or naive meaning UTC) -> tz-aware UTC datetime."""
    t = text.strip().replace("Z", "+00:00")
    if len(t) >= 3 and t[-3] in "+-" and t[-3:].lstrip("+-").isdigit():
        t = t + ":00"
    dt = datetime.fromisoformat(t)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _float_or_none(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


StoreLike = PaperStore | str | Path


@contextmanager
def open_store(store: StoreLike) -> Iterator[PaperStore]:
    """Yield `store` itself when it is an open `PaperStore`, else open the ledger file at that path for
    the duration of the block only, so no write lock is held while a collector waits on the network."""
    if isinstance(store, PaperStore):
        yield store
        return
    with PaperStore(store) as opened:
        yield opened


class OddsApiCollector:
    """Bookmaker 1X2 prices (soft books plus Pinnacle / Betfair Exchange / Smarkets) from the-odds-api,
    under a credit budget.

    Per league: cadence, error-backoff and budget checks; one GET; the call (status, credits remaining,
    used, cost) is recorded immediately after the response and before parsing, so a parse failure can
    never cause the credits to be spent twice (`parse_error` counts toward the cadence); the league's
    quotes are written as soon as they are parsed. 401/403 stops the run (`refused_auth`).
    `store` may be an open `PaperStore` or the ledger path (opened only around each write).
    """

    family = "odds_api"

    def __init__(
        self,
        http: CachedHttpClient,
        api_key: str,
        store: StoreLike,
        *,
        leagues: list[str] | None = None,
        bookmakers: list[str] | None = None,
        regions: str = "uk,eu",
        budget_floor: int = 60,
        min_hours_between_calls: float = 23.5,
        max_monthly_credits: int = 450,
        error_backoff_h: float = 1.0,
        now: Clock = utc_now,
    ):
        if not api_key:
            raise ValueError("ODDS_API_KEY is not set")
        self.http = http
        self._api_key = api_key
        self.store = store
        self.leagues = list(leagues or DEFAULT_ODDS_API_LEAGUES)
        self.bookmakers = list(bookmakers) if bookmakers else None
        self.regions = regions
        self.budget_floor = budget_floor
        self.min_hours_between_calls = min_hours_between_calls
        self.max_monthly_credits = max_monthly_credits
        self.error_backoff_h = error_backoff_h
        self.now = now
        # urllib3 logs full request URLs (including the apiKey query value) at DEBUG.
        pool_logger = logging.getLogger("urllib3.connectionpool")
        if pool_logger.getEffectiveLevel() < logging.INFO:
            pool_logger.setLevel(logging.INFO)

    def _cost_per_call(self) -> int:
        with open_store(self.store) as st:
            return st.cost_per_call(self.family)

    def projected_monthly_credits(self, n_leagues: int) -> float:
        """Credits a month of daily runs over `n_leagues` would spend: leagues x measured cost per call x
        calls per day (24 / cadence hours) x 30."""
        calls_per_day = 24.0 / max(self.min_hours_between_calls, 1e-9)
        return n_leagues * self._cost_per_call() * calls_per_day * 30

    def planned_leagues(self) -> list[str]:
        """The configured leagues, in order, trimmed so the projected monthly spend fits `max_monthly_credits`."""
        n = len(self.leagues)
        while n > 0 and self.projected_monthly_credits(n) > self.max_monthly_credits:
            n -= 1
        return self.leagues[:n]

    def _skip_reason(self, st: PaperStore, league: str, now: datetime, force: bool) -> tuple[str, str] | None:
        """(status, note) when `league` must not be called now, else None."""
        last = st.last_ok_call(self.family, league)
        if not force and last is not None and now - last < timedelta(hours=self.min_hours_between_calls):
            return "skipped_cadence", f"last ok call {last.isoformat()}"
        n_err, last_err = st.consecutive_errors(self.family, league)
        if n_err and last_err is not None:
            wait = timedelta(hours=min(self.error_backoff_h * 2 ** (n_err - 1), 24.0))
            if now - last_err < wait:
                return "skipped_backoff", f"{n_err} consecutive errors; next try after {(last_err + wait).isoformat()}"
        return None

    def collect(self, force: bool = False) -> list[Quote]:
        """Fetch every due planned league (cadence ignored if `force`; budget and backoff never are).
        Returns every quote fetched; each league's quotes are already stored when this returns."""
        quotes: list[Quote] = []
        planned = self.planned_leagues()
        cost_per_call = self._cost_per_call()
        logger.info(
            "the-odds-api projected monthly spend: %.0f credits for %d leagues (cap %d, cost %d/call, cadence %.1f h)",
            self.projected_monthly_credits(len(planned)),
            len(planned),
            self.max_monthly_credits,
            cost_per_call,
            self.min_hours_between_calls,
        )
        if len(planned) < len(self.leagues):
            logger.warning(
                "the-odds-api: %d leagues would project %.0f credits/month > cap %d; dropping %s",
                len(self.leagues),
                self.projected_monthly_credits(len(self.leagues)),
                self.max_monthly_credits,
                self.leagues[len(planned) :],
            )
        for league in planned:
            now = self.now()
            with open_store(self.store) as st:
                skip = self._skip_reason(st, league, now, force)
                if skip is not None:
                    st.record_call(self.family, league, status=skip[0], called_at=now, note=skip[1])
                    continue
                remaining = st.budget(self.family)
                if remaining is not None and remaining < self.budget_floor:
                    logger.warning(
                        "the-odds-api budget guard: %s credits remaining < floor %s; not calling %s",
                        remaining,
                        self.budget_floor,
                        league,
                    )
                    st.record_call(
                        self.family,
                        league,
                        status="refused_budget",
                        called_at=now,
                        remaining=remaining,
                        note=f"floor {self.budget_floor}",
                    )
                    break
                used_before = st.last_used(self.family)
            try:
                resp = self.http.get_live(
                    f"{ODDS_API_BASE}/sports/{league}/odds",
                    params={
                        "apiKey": self._api_key,
                        "regions": self.regions,
                        "markets": "h2h",
                        "oddsFormat": "decimal",
                        "dateFormat": "iso",
                        **({"bookmakers": ",".join(self.bookmakers)} if self.bookmakers else {}),
                    },
                    redact=("apiKey",),
                )
            except LiveHttpError as exc:
                auth = exc.status_code in (401, 403)
                logger.warning("the-odds-api %s failed: %s", league, exc)
                with open_store(self.store) as st:
                    st.record_call(
                        self.family, league, status="refused_auth" if auth else "error", called_at=now, note=str(exc)
                    )
                if auth:
                    logger.warning("the-odds-api refused the key (HTTP %s); stopping this run", exc.status_code)
                    break
                continue
            remaining_now = _int_header(resp.headers, "x-requests-remaining")
            used = _int_header(resp.headers, "x-requests-used")
            cost_header = _int_header(resp.headers, "x-requests-last")
            cost_delta = used - used_before if used is not None and used_before is not None else None
            if cost_header is not None and cost_delta is not None and cost_header != cost_delta:
                logger.warning(
                    "the-odds-api %s: x-requests-last %s != used delta %s (another client may share the key)",
                    league,
                    cost_header,
                    cost_delta,
                )
            cost = cost_header if cost_header is not None else cost_delta
            with open_store(self.store) as st:
                st.record_call(
                    self.family,
                    league,
                    status="ok",
                    called_at=now,
                    remaining=remaining_now,
                    used=used,
                    note=f"cost {cost} (header {cost_header}, used delta {cost_delta})",
                    cost=cost,
                )
            try:
                league_quotes = self.parse(resp.json(), league, now)
            except Exception as exc:  # malformed payload: credits are spent, so this counts toward cadence
                logger.warning("the-odds-api %s: could not parse the response: %s", league, exc)
                with open_store(self.store) as st:
                    st.record_call(self.family, league, status="parse_error", called_at=now, note=repr(exc)[:500])
                continue
            with open_store(self.store) as st:
                added = st.add_quotes(league_quotes)
            quotes.extend(league_quotes)
            logger.info(
                "the-odds-api %s: %d quotes (%d new), cost %s, %s credits remaining",
                league,
                len(league_quotes),
                added,
                cost,
                remaining_now,
            )
            if remaining_now is not None and remaining_now < self.budget_floor:
                logger.warning(
                    "the-odds-api budget guard: %s credits remaining < floor %s; stopping this run",
                    remaining_now,
                    self.budget_floor,
                )
                break
        return quotes

    def parse(self, events: list[dict[str, Any]], league: str, captured_at: datetime) -> list[Quote]:
        """the-odds-api events -> one quote per (bookmaker, selection) for events not yet kicked off."""
        out: list[Quote] = []
        canon = canonical_league(self.family, league)
        for ev in events:
            kickoff = parse_utc(ev["commence_time"])
            if kickoff <= captured_at:
                continue
            home, away = ev["home_team"], ev["away_team"]
            for book in ev.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        sel = "home" if name == home else "away" if name == away else "draw" if name == "Draw" else None
                        if sel is None:
                            continue
                        raw = {
                            "event_id": ev["id"],
                            "sport_key": ev.get("sport_key"),
                            "bookmaker": book["key"],
                            "last_update": market.get("last_update") or book.get("last_update"),
                            "outcome": outcome,
                        }
                        out.append(
                            Quote(
                                source=f"{self.family}:{book['key']}",
                                source_event_id=ev["id"],
                                home=home,
                                away=away,
                                league=canon,
                                kickoff_at=kickoff,
                                selection=sel,
                                captured_at=captured_at,
                                decimal_odds=_float_or_none(outcome.get("price")),
                                raw_json=json.dumps(raw),
                            )
                        )
        return out


def _int_header(headers: Any, name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


class PolymarketCollector:
    """Best bid/ask for the Yes token of each 1X2 outcome of upcoming Polymarket soccer match events."""

    source = "polymarket"

    def __init__(self, http: CachedHttpClient, *, horizon_days: float = 8.0, max_pages: int = 40, now: Clock = utc_now):
        self.http = http
        self.horizon = timedelta(days=horizon_days)
        self.max_pages = max_pages
        self.now = now

    def _events(self, now: datetime) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(self.max_pages):
            params: dict[str, Any] = {
                "tag_slug": "soccer",
                "closed": "false",
                "limit": 100,
                "end_date_min": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_date_max": (now + self.horizon).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            if cursor:
                params["after_cursor"] = cursor
            page = self.http.get_live(f"{POLYMARKET_GAMMA}/events/keyset", params=params).json()
            events.extend(page.get("events") or [])
            cursor = page.get("next_cursor")
            if not cursor:
                break
        return events

    def collect(self) -> list[Quote]:
        now = self.now()
        quotes: list[Quote] = []
        for ev in self._events(now):
            moneyline = [m for m in ev.get("markets", []) if m.get("sportsMarketType") == "moneyline"]
            if len(moneyline) != 3 or not ev.get("startTime"):
                continue
            kickoff = parse_utc(ev["startTime"])
            if kickoff <= now or kickoff > now + self.horizon:
                continue
            home, away = _pm_teams(ev)
            if not home or not away:
                continue
            league = canonical_league(self.source, (ev.get("sport") or {}).get("sport", "unknown"))
            for m in moneyline:
                sel = _pm_selection(m, home, away)
                token = _pm_yes_token(m)
                if sel is None or token is None:
                    continue
                try:
                    book = self.http.get_live(f"{POLYMARKET_CLOB}/book", params={"token_id": token}).json()
                except LiveHttpError as exc:
                    logger.warning("polymarket book %s (%s) failed: %s", token, ev.get("title"), exc)
                    continue
                captured = self.now()
                bids = [(float(b["price"]), float(b["size"])) for b in book.get("bids", [])]
                asks = [(float(a["price"]), float(a["size"])) for a in book.get("asks", [])]
                best_bid = max(bids, default=None)
                best_ask = min(asks, default=None)
                raw = {
                    "event_id": ev.get("id"),
                    "slug": ev.get("slug"),
                    "title": ev.get("title"),
                    "condition_id": m.get("conditionId"),
                    "token_id": token,
                    "question": m.get("question"),
                    "fees_enabled": m.get("feesEnabled"),
                    "book_timestamp": book.get("timestamp"),
                    "bids_top": sorted(bids, reverse=True)[:5],
                    "asks_top": sorted(asks)[:5],
                }
                quotes.append(
                    Quote(
                        source=self.source,
                        source_event_id=str(ev["id"]),
                        home=home,
                        away=away,
                        league=league,
                        kickoff_at=kickoff,
                        selection=sel,
                        captured_at=captured,
                        bid=best_bid[0] if best_bid else None,
                        ask=best_ask[0] if best_ask else None,
                        last=_float_or_none(book.get("last_trade_price")),
                        size=best_ask[1] if best_ask else None,
                        instrument_id=token,
                        raw_json=json.dumps(raw),
                    )
                )
        return quotes


def _pm_teams(ev: dict[str, Any]) -> tuple[str | None, str | None]:
    teams = {t.get("ordering"): t.get("name") for t in ev.get("teams") or []}
    if teams.get("home") and teams.get("away"):
        return teams["home"], teams["away"]
    title = str(ev.get("title", ""))
    if " vs. " in title:
        home, away = title.split(" vs. ", 1)
        return home.strip(), away.strip()
    return None, None


def _pm_selection(market: dict[str, Any], home: str, away: str) -> str | None:
    title = str(market.get("groupItemTitle") or "")
    if title.lower().startswith("draw"):
        return "draw"
    if title == home:
        return "home"
    if title == away:
        return "away"
    key = canonical_team(title)
    sh, sa = team_similarity(key, canonical_team(home)), team_similarity(key, canonical_team(away))
    if max(sh, sa) < 0.85 or sh == sa:
        return None
    return "home" if sh > sa else "away"


def _pm_yes_token(market: dict[str, Any]) -> str | None:
    try:
        outcomes = json.loads(market.get("outcomes") or "[]")
        tokens = json.loads(market.get("clobTokenIds") or "[]")
    except (TypeError, json.JSONDecodeError):
        return None
    if "Yes" not in outcomes or len(tokens) != len(outcomes):
        return None
    return str(tokens[outcomes.index("Yes")])


class KalshiCollector:
    """Yes bid/ask for each 1X2 outcome of open Kalshi soccer game events."""

    source = "kalshi"

    def __init__(
        self,
        http: CachedHttpClient,
        store: StoreLike,
        *,
        series: list[str] | None = None,
        horizon_days: float = 8.0,
        max_pages: int = 20,
        now: Clock = utc_now,
    ):
        self.http = http
        self.store = store
        self.series = list(series or DEFAULT_KALSHI_SERIES)
        self.horizon = timedelta(days=horizon_days)
        self.max_pages = max_pages
        self.now = now

    def _events(self, series: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        cursor = ""
        for _ in range(self.max_pages):
            params: dict[str, Any] = {
                "series_ticker": series,
                "status": "open",
                "with_nested_markets": "true",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            page = self.http.get_live(f"{KALSHI_BASE}/events", params=params).json()
            events.extend(page.get("events") or [])
            cursor = page.get("cursor") or ""
            if not cursor:
                break
        return events

    def _kickoff(self, ev: dict[str, Any], known: dict[str, datetime]) -> datetime | None:
        ticker = ev["event_ticker"]
        if ticker in known:
            return known[ticker]
        try:
            ms = self.http.get_live(
                f"{KALSHI_BASE}/milestones", params={"related_event_ticker": ticker, "limit": 10}
            ).json()
            for m in ms.get("milestones") or []:
                if (m.get("details") or {}).get("main_game_event_ticker") == ticker and m.get("start_date"):
                    return parse_utc(m["start_date"])
        except LiveHttpError as exc:
            logger.warning("kalshi milestone lookup for %s failed: %s", ticker, exc)
        occurrences = [m.get("occurrence_datetime") for m in ev.get("markets", []) if m.get("occurrence_datetime")]
        if not occurrences:
            return None
        logger.info("kalshi %s: no milestone, kickoff taken as occurrence_datetime - 3h", ticker)
        return parse_utc(min(occurrences)) - KALSHI_OCCURRENCE_AFTER_KICKOFF

    def collect(self) -> list[Quote]:
        now = self.now()
        with open_store(self.store) as st:  # read once, then release the ledger before the network calls
            known = st.known_kickoffs(self.source)
        quotes: list[Quote] = []
        for series in self.series:
            try:
                events = self._events(series)
            except LiveHttpError as exc:
                logger.warning("kalshi series %s failed: %s", series, exc)
                continue
            league = canonical_league(self.source, series)
            for ev in events:
                title = str(ev.get("title", ""))
                markets = ev.get("markets") or []
                if " vs " not in title or len(markets) != 3:
                    continue
                home, away = (s.strip() for s in title.split(" vs ", 1))
                kickoff = self._kickoff(ev, known)
                if kickoff is None or kickoff <= now or kickoff > now + self.horizon:
                    continue
                known[ev["event_ticker"]] = kickoff
                captured = self.now()
                for m in markets:
                    sel = _kalshi_selection(str(m.get("yes_sub_title") or ""), home, away)
                    if sel is None:
                        continue
                    bid = _float_or_none(m.get("yes_bid_dollars"))
                    ask = _float_or_none(m.get("yes_ask_dollars"))
                    last = _float_or_none(m.get("last_price_dollars"))
                    raw = {
                        k: m.get(k)
                        for k in (
                            "ticker",
                            "yes_sub_title",
                            "yes_bid_dollars",
                            "yes_ask_dollars",
                            "yes_bid_size_fp",
                            "yes_ask_size_fp",
                            "last_price_dollars",
                            "occurrence_datetime",
                            "status",
                            "volume_fp",
                            "open_interest_fp",
                        )
                    }
                    raw["event_ticker"] = ev["event_ticker"]
                    quotes.append(
                        Quote(
                            source=self.source,
                            source_event_id=ev["event_ticker"],
                            home=home,
                            away=away,
                            league=league,
                            kickoff_at=kickoff,
                            selection=sel,
                            captured_at=captured,
                            bid=bid if bid and bid > 0 else None,
                            ask=ask if ask and ask < 1 else None,
                            last=last if last and last > 0 else None,
                            size=_float_or_none(m.get("yes_ask_size_fp")),
                            instrument_id=m.get("ticker"),
                            raw_json=json.dumps(raw),
                        )
                    )
        return quotes


def _kalshi_selection(sub: str, home: str, away: str) -> str | None:
    if sub.lower() in ("tie", "draw"):
        return "draw"
    key = canonical_team(sub)
    sh, sa = team_similarity(key, canonical_team(home)), team_similarity(key, canonical_team(away))
    if max(sh, sa) < 0.6 or sh == sa:
        return None
    return "home" if sh > sa else "away"
