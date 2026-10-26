"""Paper collectors: live quotes bypass the disk cache, the-odds-api budget/cadence guard, key hygiene,
and parsing of the-odds-api / Polymarket / Kalshi payloads (shapes copied from real responses)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest
import requests
import responses

from pitch_edge.data.http import CachedHttpClient, LiveHttpError
from pitch_edge.paper.collectors import (
    KALSHI_BASE,
    ODDS_API_BASE,
    POLYMARKET_CLOB,
    POLYMARKET_GAMMA,
    KalshiCollector,
    OddsApiCollector,
    PolymarketCollector,
)
from pitch_edge.paper.store import PaperStore

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 20, 18, 0, tzinfo=UTC)
KEY = "SECRETKEY123"


@pytest.fixture
def http(tmp_path):
    return CachedHttpClient(tmp_path / "cache", min_interval_s=0, max_retries=2)


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "paper.duckdb")
    yield s
    s.close()


# ---------------------------------------------------------------- live fetch never touches the disk cache
@responses.activate
def test_get_live_never_reads_or_writes_the_disk_cache(http, tmp_path):
    responses.add(responses.GET, "https://example.test/live", json={"v": 1}, headers={"x-a": "1"})
    responses.add(responses.GET, "https://example.test/live", json={"v": 2}, headers={"x-a": "2"})
    r1 = http.get_live("https://example.test/live")
    r2 = http.get_live("https://example.test/live")
    assert (r1.json()["v"], r2.json()["v"]) == (1, 2)
    assert r2.headers["x-a"] == "2"
    assert len(responses.calls) == 2
    assert list((tmp_path / "cache").iterdir()) == []


@responses.activate
def test_get_live_redacts_secrets_in_errors(http):
    responses.add(responses.GET, "https://example.test/odds", status=401, json={"message": "bad key"})
    with pytest.raises(LiveHttpError) as exc:
        http.get_live("https://example.test/odds", params={"apiKey": KEY}, redact=("apiKey",))
    assert KEY not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__
    assert exc.value.status_code == 401
    assert len(responses.calls) == 1  # 4xx is not retried


@responses.activate
def test_get_live_retries_5xx(http):
    responses.add(responses.GET, "https://example.test/x", status=503)
    responses.add(responses.GET, "https://example.test/x", json={"ok": True})
    assert http.get_live("https://example.test/x").json() == {"ok": True}


# ---------------------------------------------------------------- the-odds-api
def _odds_event(event_id="e1", home="Ipswich Town", away="Nottingham Forest", kickoff="2026-10-23T19:00:00Z"):
    def book(key, prices):
        return {
            "key": key,
            "title": key,
            "last_update": "2026-10-20T17:50:00Z",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2026-10-20T17:50:00Z",
                    "outcomes": [
                        {"name": home, "price": prices[0]},
                        {"name": away, "price": prices[2]},
                        {"name": "Draw", "price": prices[1]},
                    ],
                }
            ],
        }

    return {
        "id": event_id,
        "sport_key": "soccer_epl",
        "sport_title": "EPL",
        "commence_time": kickoff,
        "home_team": home,
        "away_team": away,
        "bookmakers": [book("pinnacle", (3.14, 2.34, 3.48)), book("betfair_ex_uk", (3.25, 2.40, 3.55))],
    }


def _odds_url(sport):
    return f"{ODDS_API_BASE}/sports/{sport}/odds"


def _collector(http, store, **kw):
    kw.setdefault("leagues", ["soccer_epl"])
    return OddsApiCollector(http, KEY, store, now=lambda: NOW, **kw)


@responses.activate
def test_odds_api_parses_h2h_for_each_bookmaker_and_records_budget(http, store):
    started = _odds_event("e0", kickoff="2026-10-20T17:00:00Z")
    responses.add(
        responses.GET,
        _odds_url("soccer_epl"),
        json=[_odds_event(), started],
        headers={"x-requests-remaining": "494", "x-requests-used": "6", "x-requests-last": "1"},
    )
    quotes = _collector(http, store).collect()
    assert len(quotes) == 6  # 2 books x 3 selections; the started event is skipped
    by = {(q.source, q.selection): q for q in quotes}
    assert by[("odds_api:pinnacle", "home")].decimal_odds == 3.14
    assert by[("odds_api:pinnacle", "draw")].decimal_odds == 2.34
    assert by[("odds_api:betfair_ex_uk", "away")].decimal_odds == 3.55
    q = by[("odds_api:pinnacle", "home")]
    assert q.league == "EPL" and q.kickoff_at == datetime(2026, 10, 23, 19, 0, tzinfo=UTC) and q.captured_at == NOW
    assert json.loads(q.raw_json)["last_update"] == "2026-10-20T17:50:00Z"
    sent = responses.calls[0].request.url
    assert "bookmakers=" not in sent and "markets=h2h" in sent  # all books in the regions: soft + sharp
    assert "regions=uk%2Ceu" in sent
    assert store.budget("odds_api") == 494
    assert store.last_ok_call("odds_api", "soccer_epl") == NOW


@responses.activate
def test_odds_api_refuses_to_call_below_budget_floor(http, store, caplog):
    store.record_call("odds_api", "soccer_epl", status="ok", called_at=NOW - timedelta(days=2), remaining=59)
    caplog.set_level(logging.WARNING)
    assert _collector(http, store, budget_floor=60).collect() == []
    assert len(responses.calls) == 0
    log = store.collect_log()
    assert log.iloc[-1]["status"] == "refused_budget"
    assert "budget" in caplog.text.lower()


@responses.activate
def test_odds_api_stops_mid_run_once_budget_falls_below_floor(http, store):
    responses.add(
        responses.GET,
        _odds_url("soccer_epl"),
        json=[_odds_event()],
        headers={"x-requests-remaining": "60", "x-requests-used": "440"},
    )
    responses.add(
        responses.GET,
        _odds_url("soccer_spain_la_liga"),
        json=[],
        headers={"x-requests-remaining": "59", "x-requests-used": "441"},
    )
    responses.add(
        responses.GET,
        _odds_url("soccer_italy_serie_a"),
        json=[],
        headers={"x-requests-remaining": "58", "x-requests-used": "442"},
    )
    c = _collector(http, store, leagues=["soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a"], budget_floor=60)
    c.collect()
    assert [call.request.url.split("/sports/")[1].split("/")[0] for call in responses.calls] == [
        "soccer_epl",
        "soccer_spain_la_liga",
    ]


@responses.activate
def test_odds_api_calls_each_league_at_most_once_per_cadence_window(http, store):
    store.record_call("odds_api", "soccer_epl", status="ok", called_at=NOW - timedelta(hours=2), remaining=400)
    assert _collector(http, store, min_hours_between_calls=20).collect() == []
    assert len(responses.calls) == 0
    assert store.collect_log().iloc[-1]["status"] == "skipped_cadence"


@responses.activate
def test_odds_api_never_logs_or_stores_the_key(http, store, caplog):
    responses.add(responses.GET, _odds_url("soccer_epl"), status=401, json={"message": "quota"})
    caplog.set_level(logging.DEBUG)
    assert _collector(http, store).collect() == []
    assert KEY not in caplog.text
    assert KEY not in store.collect_log().to_json()
    assert store.collect_log().iloc[-1]["status"] == "refused_auth"


def test_odds_api_requires_a_key(http, store):
    with pytest.raises(ValueError):
        OddsApiCollector(http, "", store)


# ---------------------------------------------------------------- Polymarket
def _pm_market(title, token_yes, bid, ask, sport_type="moneyline"):
    return {
        "question": f"Will {title} win?",
        "conditionId": f"0x{token_yes}",
        "groupItemTitle": title,
        "sportsMarketType": sport_type,
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": json.dumps([token_yes, token_yes + "0"]),
        "bestBid": bid,
        "bestAsk": ask,
        "feesEnabled": True,
        "gameStartTime": "2026-10-21 18:00:00+00",
    }


def _pm_event(eid="ev9", start="2026-10-21T18:00:00Z", sport="lal"):
    return {
        "id": eid,
        "title": "Levante UD vs. Athletic Club",
        "startTime": start,
        "sport": {"sport": sport},
        "teams": [{"name": "Levante UD", "ordering": "home"}, {"name": "Athletic Club", "ordering": "away"}],
        "markets": [
            _pm_market("Levante UD", "111", 0.30, 0.32),
            _pm_market("Draw (Levante UD vs. Athletic Club)", "222", 0.27, 0.29),
            _pm_market("Athletic Club", "333", 0.40, 0.42),
        ],
    }


def _book(token, bids, asks, last="0.31"):
    return {
        "asset_id": token,
        "timestamp": "1792519203872",
        "last_trade_price": last,
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
    }


@responses.activate
def test_polymarket_collects_1x2_best_bid_ask_from_the_clob_book(http):
    side_event = {
        "id": "ev10",
        "title": "Levante UD vs. Athletic Club - Exact Score",
        "startTime": "2026-10-21T18:00:00Z",
        "sport": {"sport": "lal"},
        "teams": [],
        "markets": [_pm_market("1-0", "999", 0.1, 0.12, "exact")],
    }
    started = _pm_event("ev11", start="2026-10-20T17:00:00Z")
    responses.add(
        responses.GET,
        f"{POLYMARKET_GAMMA}/events/keyset",
        json={"events": [_pm_event(), side_event, started], "next_cursor": None},
    )
    books = {
        "111": _book("111", [(0.29, 50), (0.30, 120)], [(0.33, 10), (0.32, 75)]),
        "222": _book("222", [(0.27, 40)], [(0.29, 60)]),
        "333": _book("333", [(0.40, 10)], [(0.43, 5), (0.42, 30)]),
    }
    for token, body in books.items():
        responses.add(responses.GET, f"{POLYMARKET_CLOB}/book?token_id={token}", json=body)
    quotes = PolymarketCollector(http, now=lambda: NOW).collect()
    by = {q.selection: q for q in quotes}
    assert set(by) == {"home", "draw", "away"} and len(quotes) == 3
    assert (by["home"].bid, by["home"].ask, by["home"].size) == (0.30, 0.32, 75.0)
    assert by["home"].last == 0.31
    assert by["away"].ask == 0.42 and by["away"].instrument_id == "333"
    assert by["draw"].league == "LALIGA" and by["draw"].source == "polymarket"
    assert by["home"].home == "Levante UD" and by["home"].kickoff_at == datetime(2026, 10, 21, 18, tzinfo=UTC)
    assert by["home"].decimal_odds is None
    gamma_calls = [c for c in responses.calls if "gamma" in c.request.url]
    assert "tag_slug=soccer" in gamma_calls[0].request.url and "closed=false" in gamma_calls[0].request.url


@responses.activate
def test_polymarket_follows_the_keyset_cursor(http):
    responses.add(responses.GET, f"{POLYMARKET_GAMMA}/events/keyset", json={"events": [], "next_cursor": "abc"})
    responses.add(responses.GET, f"{POLYMARKET_GAMMA}/events/keyset", json={"events": [], "next_cursor": None})
    assert PolymarketCollector(http, now=lambda: NOW).collect() == []
    assert "after_cursor=abc" in responses.calls[1].request.url


# ---------------------------------------------------------------- Kalshi
def _k_market(ticker, sub, bid, ask):
    return {
        "ticker": ticker,
        "yes_sub_title": sub,
        "yes_bid_dollars": f"{bid:.4f}",
        "yes_ask_dollars": f"{ask:.4f}",
        "yes_ask_size_fp": "5545.00",
        "last_price_dollars": "0.0000",
        "occurrence_datetime": "2026-10-23T22:00:00Z",
        "status": "active",
    }


def _k_event(ticker="KXEPLGAME-26OCT23IPSNFO"):
    return {
        "event_ticker": ticker,
        "series_ticker": "KXEPLGAME",
        "title": "Ipswich Town vs Nottingham Forest",
        "markets": [
            _k_market(f"{ticker}-IPS", "Ipswich Town", 0.30, 0.31),
            _k_market(f"{ticker}-TIE", "Tie", 0.27, 0.28),
            _k_market(f"{ticker}-NFO", "Nottingham Forest", 0.41, 0.43),
        ],
    }


@responses.activate
def test_kalshi_collects_1x2_with_kickoff_from_the_milestone(http, store):
    responses.add(responses.GET, f"{KALSHI_BASE}/events", json={"events": [_k_event()], "cursor": ""})
    responses.add(
        responses.GET,
        f"{KALSHI_BASE}/milestones",
        json={
            "milestones": [
                {"start_date": "2026-10-23T19:00:00Z", "details": {"main_game_event_ticker": "KXEPLGAME-26OCT23IPSNFO"}}
            ]
        },
    )
    quotes = KalshiCollector(http, store, series=["KXEPLGAME"], now=lambda: NOW).collect()
    by = {q.selection: q for q in quotes}
    assert set(by) == {"home", "draw", "away"}
    assert (by["home"].bid, by["home"].ask) == (0.30, 0.31)
    assert by["away"].ask == 0.43 and by["draw"].instrument_id.endswith("-TIE")
    assert by["home"].kickoff_at == datetime(2026, 10, 23, 19, tzinfo=UTC)
    assert by["home"].league == "EPL" and by["home"].home == "Ipswich Town" and by["home"].source == "kalshi"
    assert by["home"].last is None  # Kalshi reports 0.0000 when nothing has traded


@responses.activate
def test_kalshi_kickoff_falls_back_to_occurrence_minus_three_hours(http, store):
    responses.add(responses.GET, f"{KALSHI_BASE}/events", json={"events": [_k_event()], "cursor": ""})
    responses.add(responses.GET, f"{KALSHI_BASE}/milestones", json={"milestones": []})
    quotes = KalshiCollector(http, store, series=["KXEPLGAME"], now=lambda: NOW).collect()
    assert quotes[0].kickoff_at == datetime(2026, 10, 23, 19, tzinfo=UTC)


@responses.activate
def test_kalshi_reuses_known_kickoffs_without_a_milestone_call(http, store):
    responses.add(responses.GET, f"{KALSHI_BASE}/events", json={"events": [_k_event()], "cursor": ""})
    responses.add(
        responses.GET,
        f"{KALSHI_BASE}/milestones",
        json={
            "milestones": [
                {"start_date": "2026-10-23T19:00:00Z", "details": {"main_game_event_ticker": "KXEPLGAME-26OCT23IPSNFO"}}
            ]
        },
    )
    c = KalshiCollector(http, store, series=["KXEPLGAME"], now=lambda: NOW)
    store.add_quotes(c.collect())
    responses.calls.reset()
    c.collect()
    assert not [call for call in responses.calls if "milestones" in call.request.url]


# ---------------------------------------------------------------- soft-book calls: cost and monthly spend
@responses.activate
def test_odds_api_bookmaker_filter_is_optional(http, store):
    responses.add(responses.GET, _odds_url("soccer_epl"), json=[], headers={"x-requests-remaining": "400"})
    _collector(http, store, bookmakers=["pinnacle", "smarkets"]).collect()
    assert "bookmakers=pinnacle%2Csmarkets" in responses.calls[0].request.url


@responses.activate
def test_odds_api_measures_cost_from_the_used_delta_and_header(http, store):
    store.record_call(
        "odds_api", "soccer_spain_la_liga", status="ok", called_at=NOW - timedelta(days=1), remaining=488, used=12
    )
    responses.add(
        responses.GET,
        _odds_url("soccer_epl"),
        json=[],
        headers={"x-requests-remaining": "486", "x-requests-used": "14", "x-requests-last": "2"},
    )
    _collector(http, store).collect()
    row = store.collect_log().query("scope == 'soccer_epl' and status == 'ok'").iloc[-1]
    assert row["cost"] == 2
    assert store.cost_per_call("odds_api") == 2


def test_odds_api_cost_per_call_defaults_to_two_before_any_measurement(store):
    assert store.cost_per_call("odds_api", default=2) == 2


@responses.activate
def test_odds_api_trims_leagues_to_the_monthly_credit_cap_and_logs_the_projection(http, store, caplog):
    leagues = ["soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a"]
    for lg in leagues:
        responses.add(
            responses.GET, _odds_url(lg), json=[], headers={"x-requests-remaining": "400", "x-requests-last": "2"}
        )
    caplog.set_level(logging.INFO)
    # 2 credits x 30 days x (24 / 24 h cadence) = 60 per league per month; cap 130 fits two leagues
    c = _collector(http, store, leagues=leagues, max_monthly_credits=130, min_hours_between_calls=24)
    assert c.planned_leagues() == leagues[:2]
    assert c.projected_monthly_credits(len(leagues)) == pytest.approx(180)
    c.collect()
    called = [call.request.url.split("/sports/")[1].split("/")[0] for call in responses.calls]
    assert called == leagues[:2]
    assert "projected monthly spend" in caplog.text.lower()


# ---------------------------------------------------------------- review fixes: C2, I5, I7
@responses.activate
def test_odds_api_records_cost_before_parsing_and_a_parse_error_counts_toward_cadence(http, store):
    responses.add(
        responses.GET,
        _odds_url("soccer_epl"),
        json=[{"broken": True}],
        headers={"x-requests-remaining": "480", "x-requests-used": "20", "x-requests-last": "2"},
    )
    assert _collector(http, store).collect() == []
    log = store.collect_log().query("scope == 'soccer_epl'")
    assert list(log["status"]) == ["ok", "parse_error"]
    assert log.iloc[0]["cost"] == 2 and log.iloc[0]["remaining"] == 480
    responses.calls.reset()
    _collector(http, store).collect()  # same day: the credits were spent, so no second call
    assert len(responses.calls) == 0


@responses.activate
def test_odds_api_persists_each_league_as_soon_as_it_is_parsed(http, store):
    responses.add(responses.GET, _odds_url("soccer_epl"), json=[_odds_event()], headers={"x-requests-remaining": "400"})
    responses.add(responses.GET, _odds_url("soccer_spain_la_liga"), body=requests.exceptions.ConnectionError("boom"))
    c = _collector(http, store, leagues=["soccer_epl", "soccer_spain_la_liga"])
    c.collect()
    assert len(store.quotes()) == 6


@responses.activate
@pytest.mark.parametrize("status", [401, 403])
def test_odds_api_stops_the_run_on_auth_refusal(http, store, status):
    responses.add(responses.GET, _odds_url("soccer_epl"), status=status, json={"message": "no"})
    responses.add(responses.GET, _odds_url("soccer_spain_la_liga"), json=[])
    _collector(http, store, leagues=["soccer_epl", "soccer_spain_la_liga"]).collect()
    assert len(responses.calls) == 1
    assert store.collect_log().iloc[-1]["status"] == "refused_auth"


@responses.activate
def test_odds_api_backs_off_exponentially_after_errors(http, store):
    for hours_ago in (5, 3):
        store.record_call("odds_api", "soccer_epl", status="error", called_at=NOW - timedelta(hours=hours_ago))
    # two consecutive errors: wait 1h * 2**(2-1) = 2h after the last one (3h ago) -> retry allowed
    responses.add(responses.GET, _odds_url("soccer_epl"), json=[], headers={"x-requests-remaining": "400"})
    _collector(http, store, error_backoff_h=1.0).collect()
    assert len(responses.calls) == 1
    store.record_call("odds_api", "soccer_spain_la_liga", status="error", called_at=NOW - timedelta(hours=4))
    store.record_call("odds_api", "soccer_spain_la_liga", status="error", called_at=NOW - timedelta(hours=3))
    store.record_call("odds_api", "soccer_spain_la_liga", status="error", called_at=NOW - timedelta(hours=1))
    responses.calls.reset()
    _collector(http, store, leagues=["soccer_spain_la_liga"], error_backoff_h=1.0).collect()  # needs 4h
    assert len(responses.calls) == 0
    assert store.collect_log().iloc[-1]["status"] == "skipped_backoff"


@responses.activate
def test_odds_api_does_not_hold_the_ledger_open_during_the_http_call(http, tmp_path):
    import duckdb

    path = tmp_path / "ledger.duckdb"
    PaperStore(path).close()
    seen = {}

    def callback(request):
        con = duckdb.connect(str(path), read_only=True)  # fails if a read-write connection is open
        seen["n"] = con.execute("SELECT count(*) FROM collect_log").fetchone()[0]
        con.close()
        return (200, {"x-requests-remaining": "400"}, json.dumps([_odds_event()]))

    responses.add_callback(responses.GET, _odds_url("soccer_epl"), callback=callback)
    quotes = OddsApiCollector(http, KEY, path, leagues=["soccer_epl"], now=lambda: NOW).collect()
    assert len(quotes) == 6 and "n" in seen
    with PaperStore(path) as s:
        assert len(s.quotes()) == 6
