"""Every one-row-per-group pick resolves ties by a total order (PE-36).

Each test builds the same tied rows twice, inserted in opposite orders, and requires the same pick in both,
naming the row the documented tie-break chooses. Nothing touches the network or the real lake: lake queries
run against an in-memory DuckDB that holds the tables the query reads.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from pitch_edge import pipeline
from pitch_edge.agents.qa_data import resolve_fixture
from pitch_edge.backtest.report import _best_strategy
from pitch_edge.data.alt.venues import VenueGeocoder
from pitch_edge.data.sources.transfermarkt_open import TransfermarktOpenSource
from pitch_edge.data.storage import Warehouse
from pitch_edge.features.context import club_id_map, transfermarkt_context
from pitch_edge.features.live_lineups import _latest_value, _usual_xi_values
from pitch_edge.features.squad_value import squad_value_context
from pitch_edge.lake.mirror import MIRRORS, sync_mirror
from pitch_edge.odds.steam import detect_steam
from pitch_edge.odds.utils import no_vig_probabilities
from pitch_edge.paper.pricing import power_devig_odds
from pitch_edge.paper.settle import LakeCloseSource, PmBar, _kickoff_moved, _own_snapshot_close, _pm_bar_close
from pitch_edge.paper.store import PaperStore, Quote
from pitch_edge.rag.documents import match_documents

pytestmark = pytest.mark.unit

KO = datetime(2026, 10, 24, 14, 0, tzinfo=UTC)
CAPTURED = KO - timedelta(hours=5)
SEL = ("home", "draw", "away")


def _both_orders(rows: list[Any]) -> list[list[Any]]:
    """`rows` as given and reversed."""
    return [list(rows), list(reversed(rows))]


class DuckLake:
    """A lake stand-in that runs each query on an in-memory DuckDB holding hand-built tables (UTC session)."""

    def __init__(self) -> None:
        self.con = duckdb.connect()
        self.con.execute("SET TimeZone = 'UTC'")

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        return self.con.execute(query, params or []).df()


# --------------------------------------------------------------------------- 1. lake mirror, one row per key
class BatchLake:
    """Serves one fixed record batch to `sync_mirror`."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def sql_batches(self, query, params=None):
        yield pa.RecordBatch.from_pylist(self.rows)


@pytest.mark.parametrize("with_scraped_at", [True, False])
def test_mirror_keeps_one_row_per_key_by_a_total_order(tmp_path, with_scraped_at):
    stamp = datetime(2026, 2, 1)
    rows = [{"coach_id": 1, "name": n} for n in ("Delta", "Bravo", "Alpha", "Charlie")]
    if with_scraped_at:
        rows = [{**r, "scraped_at": stamp} for r in rows]
        rows += [
            {"coach_id": 1, "name": "Aardvark", "scraped_at": stamp - timedelta(days=1)},  # older: loses
            {"coach_id": 1, "name": "Aaron", "scraped_at": None},  # no scrape time: loses
        ]
    picks = []
    for i, order in enumerate(_both_orders(rows)):
        with Warehouse(tmp_path / f"w{i}.duckdb") as wh:
            sync_mirror(wh, BatchLake(order), MIRRORS["tm_managers"])
            picks.append(wh.read("lake_tm_managers")["name"].tolist())
    assert picks == [["Alpha"], ["Alpha"]]


# --------------------------------------------------------------------------- paper store helpers
def _quote(eid: str, selection: str, *, source="odds_api:pinnacle", ko=KO, captured=CAPTURED, **kw) -> Quote:
    return Quote(
        source=source,
        source_event_id=eid,
        home="Ipswich Town",
        away="Nottingham Forest",
        league="EPL",
        kickoff_at=ko,
        selection=selection,
        captured_at=captured,
        **kw,
    )


def _stores(tmp_path, quote_sets: list[list[Quote]]):
    """One fresh paper store per quote list, each filled in the list's order."""
    for i, quotes in enumerate(quote_sets):
        store = PaperStore(tmp_path / f"paper{i}.duckdb")
        try:
            store.add_quotes(quotes)
            yield store
        finally:
            store.close()


# --------------------------------------------------------------------------- 2. latest_quotes
def test_latest_quotes_breaks_a_capture_time_tie_by_event_id(tmp_path):
    quotes = [_quote(eid, "home", decimal_odds=odds) for eid, odds in (("e3", 2.3), ("e1", 2.1), ("e2", 2.2))]
    picks = []
    for store in _stores(tmp_path, _both_orders(quotes)):
        got = store.latest_quotes(CAPTURED + timedelta(minutes=1))
        assert got["fixture_key"].nunique() == 1  # the three events are one fixture
        picks.append(got[["source_event_id", "decimal_odds"]].values.tolist())
    assert picks == [[["e1", 2.1]], [["e1", 2.1]]]


# --------------------------------------------------------------------------- 3. known_kickoffs
def test_known_kickoffs_breaks_a_capture_time_tie_by_the_later_kickoff(tmp_path):
    kickoffs = (KO, KO + timedelta(hours=2), KO + timedelta(hours=1))
    quotes = [_quote("e1", s, ko=ko, decimal_odds=2.0) for s, ko in zip(SEL, kickoffs, strict=True)]
    picks = [store.known_kickoffs("odds_api:pinnacle") for store in _stores(tmp_path, _both_orders(quotes))]
    assert picks == [{"e1": KO + timedelta(hours=2)}] * 2


# --------------------------------------------------------------------------- 4. OddsPapi Pinnacle close
def _oddspapi_lake(odds_rows: list[tuple]) -> DuckLake:
    lake = DuckLake()
    lake.con.execute(
        "CREATE TABLE oddspapi_fixtures (fixture_id VARCHAR, division VARCHAR, home_team VARCHAR, "
        "away_team VARCHAR, kickoff_utc TIMESTAMPTZ)"
    )
    lake.con.execute(
        "INSERT INTO oddspapi_fixtures VALUES ('f1', 'UCL', 'Paris Saint-Germain', 'FC Barcelona', ?)", [KO]
    )
    lake.con.execute(
        "CREATE TABLE oddspapi_odds (fixture_id VARCHAR, bookmaker VARCHAR, market VARCHAR, outcome VARCHAR, "
        "price DOUBLE, active BOOLEAN, observed_at TIMESTAMPTZ)"
    )
    for row in odds_rows:
        lake.con.execute("INSERT INTO oddspapi_odds VALUES ('f1', 'pinnacle', '1x2', ?, ?, ?, ?)", list(row))
    return lake


@pytest.mark.parametrize(
    ("tied_home", "valid"),
    [
        ((1.95, True), False),  # another price at the latest observed_at
        ((1.90, False), False),  # the same price, but inactive
        ((1.90, True), True),  # an exact duplicate: interchangeable
    ],
)
def test_pinnacle_close_refuses_two_different_rows_at_the_latest_observed_at(tied_home, valid):
    last = KO - timedelta(minutes=3)
    rows = [
        ("home", 1.80, True, KO - timedelta(hours=2)),
        ("home", 1.90, True, last),
        ("home", *tied_home, last),
        ("draw", 3.80, True, last),
        ("away", 4.20, True, last),
    ]
    closes = [
        LakeCloseSource(_oddspapi_lake(order)).pinnacle_close("UCL", KO, "Paris Saint Germain", "Barcelona")
        for order in _both_orders(rows)
    ]
    assert closes[0] == closes[1]
    if valid:
        assert closes[0].odds == (1.90, 3.80, 4.20)
    else:
        assert closes[0].odds is None and not closes[0].retry
        assert "two Pinnacle rows at the latest observed_at" in closes[0].reason
        assert "home" in closes[0].reason


# --------------------------------------------------------------------------- 5. Polymarket last bar
def _pm_quotes(tokens: dict[str, str], eid: str = "pm1") -> list[Quote]:
    return [
        _quote(eid, s, source="polymarket", bid=0.3, ask=0.32, instrument_id=tokens[s], captured=CAPTURED) for s in SEL
    ]


def _bars_lake(bars: list[tuple]) -> DuckLake:
    lake = DuckLake()
    lake.con.execute("CREATE TABLE pm_soccer_bars_1h (asset_id VARCHAR, hour_start TIMESTAMPTZ, close DOUBLE)")
    for row in bars:
        lake.con.execute("INSERT INTO pm_soccer_bars_1h VALUES (?, ?, ?)", list(row))
    return lake


def test_pm_bar_close_refuses_two_different_closes_at_the_newest_hour(tmp_path):
    tokens = {s: f"tok-{s}" for s in SEL}
    hour = KO - timedelta(hours=2)
    bars = [
        ("tok-home", hour - timedelta(hours=1), 0.38),
        ("tok-home", hour, 0.40),
        ("tok-home", hour, 0.45),
        ("tok-draw", hour, 0.30),
        ("tok-away", hour, 0.30),
    ]
    store = PaperStore(tmp_path / "paper.duckdb")
    try:
        store.add_quotes(_pm_quotes(tokens))
        fixture = str(store.fixtures()["fixture_key"].iloc[0])
        answers = []
        for order in _both_orders(bars):
            source = LakeCloseSource(_bars_lake(order))
            home = _pm_bar_close(store, source, fixture, KO, "home")
            draw = _pm_bar_close(store, source, fixture, KO, "draw")
            answers.append((home, draw))
    finally:
        store.close()
    assert answers[0] == answers[1]
    (home_close, home_reason), (draw_close, _) = answers[0]
    assert home_close is None and "newest hour" in home_reason and "home" in home_reason
    # the conflicting home token drops out of the three-way normalisation; the draw settles on its own bar
    assert draw_close is not None and draw_close.closing_price == pytest.approx(1 / 0.30)
    assert draw_close.source == "lake:pm_soccer_bars_1h(single-token)"


# --------------------------------------------------------------------------- 6. Polymarket token per selection
class RecordingBars:
    """A `CloseSource` that records which tokens are asked for and answers each with a fresh bar."""

    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    def pm_last_bars(self, asset_ids, before):
        self.asked.append(list(asset_ids))
        return {a: PmBar(0.33, before - timedelta(hours=1)) for a in asset_ids}


def test_pm_token_tie_on_capture_time_picks_the_largest_instrument_id(tmp_path):
    first = _pm_quotes({"home": "tok-home-b", "draw": "tok-draw", "away": "tok-away"}, eid="pm1")
    second = [_quote("pm2", "home", source="polymarket", bid=0.3, ask=0.32, instrument_id="tok-home-a")]
    asked = []
    for store in _stores(tmp_path, [first + second, second + first]):
        fixture = str(store.fixtures()["fixture_key"].iloc[0])
        assert store.quotes_for_fixture(fixture, before=KO)["source_event_id"].nunique() == 2
        source = RecordingBars()
        _pm_bar_close(store, source, fixture, KO, "home")
        asked.append(source.asked)
    assert asked == [[["tok-away", "tok-draw", "tok-home-b"]]] * 2


# --------------------------------------------------------------------------- 7. usual XI
USUAL = list(range(1, 11))  # players 1-10 lead the club on minutes
TIED = [50505, 99357, 55136, 5317, 33947, 67024]  # tie for the eleventh place; 5317 is the smallest id
ALL_PLAYERS = USUAL + TIED


def _seed_usual_xi(wh: Warehouse, reverse: bool) -> None:
    """Club 1: `USUAL` lead on minutes and `TIED` tie for the eleventh place. Game 100 on 2024-08-01: club 1
    fields `USUAL` and every tied player but 5317. The tied ids are scattered so that the unbroken tie does not
    happen to follow player id."""
    apps = []
    for pid in ALL_PLAYERS:
        n_games = 10 if pid in USUAL else 5
        for g in range(n_games):
            apps.append(
                {
                    "appearance_id": f"{pid}_{g}",
                    "player_id": pid,
                    "player_club_id": 1,
                    "date": pd.Timestamp("2024-06-01") + pd.Timedelta(days=g),
                    "minutes_played": 90,
                }
            )
    if reverse:
        apps.reverse()
    wh.upsert("tm_appearances", pd.DataFrame(apps), keys=["appearance_id"])
    values = dict.fromkeys(USUAL, 1_000_000) | dict.fromkeys(TIED, 2_000_000) | {5317: 4_000_000}
    vals = pd.DataFrame(
        {"player_id": list(values), "date": pd.Timestamp("2024-01-01"), "market_value_in_eur": list(values.values())}
    )
    wh.upsert("tm_player_valuations", vals, keys=["player_id", "date"])
    wh.upsert(
        "tm_players",
        pd.DataFrame({"player_id": ALL_PLAYERS, "name": [f"P{pid}" for pid in ALL_PLAYERS]}),
        keys=["player_id"],
    )
    wh.upsert(
        "tm_games",
        pd.DataFrame(
            {"game_id": [100], "date": [pd.Timestamp("2024-08-01")], "home_club_id": [1], "away_club_id": [2]}
        ),
        keys=["game_id"],
    )
    lineup = [(pid, "starting_lineup") for pid in USUAL] + [(pid, "substitutes") for pid in TIED if pid != 5317]
    wh.upsert(
        "tm_game_lineups",
        pd.DataFrame(
            {
                "game_lineups_id": [f"l{pid}" for pid, _ in lineup],
                "game_id": 100,
                "club_id": 1,
                "player_id": [pid for pid, _ in lineup],
                "type": [t for _, t in lineup],
            }
        ),
        keys=["game_lineups_id"],
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_squad_value_usual_xi_breaks_a_minutes_tie_by_player_id(tmp_path, reverse):
    with Warehouse(tmp_path / "wh.duckdb") as wh:
        _seed_usual_xi(wh, reverse)
        out = squad_value_context(wh, pd.DataFrame({"match_id": ["m1"], "tm_game_id": [100]}))
    # usual XI = `USUAL` and 5317 (the smallest tied id), who is absent: 4M missing of 14M usual
    assert out["sv_home_missing_pct"].iloc[0] == pytest.approx(4 / 14)


@pytest.mark.parametrize("reverse", [False, True])
def test_live_lineups_usual_xi_breaks_a_minutes_tie_by_player_id(tmp_path, reverse):
    with Warehouse(tmp_path / "wh.duckdb") as wh:
        _seed_usual_xi(wh, reverse)
        xi = _usual_xi_values(wh, 1, date(2024, 8, 1))
    assert sorted(xi["player_id"].tolist()) == [*USUAL, 5317]


# --------------------------------------------------------------------------- 8. referee / game id per match
@pytest.mark.parametrize("reverse", [False, True])
def test_transfermarkt_context_picks_the_smallest_game_id_for_a_match(tmp_path, reverse):
    games = [("g3", "C Ref"), ("g1", "A Ref"), ("g2", "B Ref")]  # three games on one date between one pair
    if reverse:
        games.reverse()
    with Warehouse(tmp_path / "wh.duckdb") as wh:
        wh.upsert(
            "tm_clubs",
            pd.DataFrame({"club_id": ["1", "2"], "name": ["Team00 FC", "Team01 FC"], "last_season": [2025, 2025]}),
            keys=["club_id"],
        )
        wh.upsert(
            "tm_games",
            pd.DataFrame(
                {
                    "game_id": [g for g, _ in games],
                    "date": pd.Timestamp("2020-03-07"),
                    "home_club_id": 1,
                    "away_club_id": 2,
                    "competition_type": "domestic_league",
                    "referee": [r for _, r in games],
                }
            ),
            keys=["game_id"],
        )
        wh.upsert(
            "tm_appearances",
            pd.DataFrame(
                {
                    "appearance_id": ["a1"],
                    "game_id": ["g0"],
                    "player_club_id": [1],
                    "date": [pd.Timestamp("2020-03-01")],
                    "minutes_played": [90],
                }
            ),
            keys=["appearance_id"],
        )
        matches = pd.DataFrame(
            {"match_id": ["x1"], "date": [pd.Timestamp("2020-03-07")], "home_team": ["Team00"], "away_team": ["Team01"]}
        )
        ctx = transfermarkt_context(wh, matches).iloc[0]
    assert (ctx["tm_game_id"], ctx["tm_referee"]) == ("g1", "A Ref")


# --------------------------------------------------------------------------- 9. steam first / last snapshot
def _snap(book: str, prices: tuple[float, float, float], ts: str) -> list[dict]:
    return [
        {"match_id": "m1", "bookmaker": book, "side": s, "price": p, "snapshot_ts": pd.Timestamp(ts)}
        for s, p in zip(SEL, prices, strict=True)
    ]


def test_steam_breaks_a_snapshot_time_tie_by_price():
    rows = (
        _snap("PS", (2.20, 3.4, 3.3), "2024-01-01T08:00")
        + _snap("PS", (1.85, 3.6, 4.2), "2024-01-01T10:00")
        + [
            {
                "match_id": "m1",
                "bookmaker": "PS",
                "side": "home",
                "price": 1.75,
                "snapshot_ts": pd.Timestamp("2024-01-01T10:00"),
            }
        ]
        + _snap("SOFT", (2.20, 3.4, 3.3), "2024-01-01T08:00")
        + _snap("SOFT", (2.20, 3.4, 3.3), "2024-01-01T10:00")
    )
    flags = [detect_steam(pd.DataFrame(order), sharp_book="PS") for order in _both_orders(rows)]
    assert flags[0] == flags[1]
    home = [f for f in flags[0] if f.side == "home"]
    # the last PS snapshot is the highest price at the tied time: home 1.85
    assert len(home) == 1 and home[0].sharp_probability == pytest.approx(no_vig_probabilities(1.85, 3.6, 4.2)[0])


# --------------------------------------------------------------------------- 10. Transfermarkt club name
@pytest.mark.parametrize("reverse", [False, True])
def test_substitution_profile_names_a_club_by_its_smallest_name(tmp_path, reverse):
    names = ["Arsenal FC", "Arsenal London", "Arsenal Football Club", "The Arsenal", "Arsenal", "Gunners", "AFC"]
    events = [(f"e{i}", i, 11, name, 60 + i) for i, name in enumerate(names)]  # one club, a name per game
    if reverse:
        events.reverse()
    con = duckdb.connect(str(tmp_path / "transfermarkt-datasets.duckdb"))
    con.execute(
        "CREATE TABLE game_events (game_event_id VARCHAR, game_id INT, club_id INT, club_name VARCHAR, "
        "minute INT, type VARCHAR)"
    )
    for e in events:
        con.execute("INSERT INTO game_events VALUES (?, ?, ?, ?, ?, 'Substitutions')", list(e))
    con.close()
    prof = TransfermarktOpenSource(cache_dir=tmp_path).substitution_profile(min_games=1)
    assert prof["club_name"].tolist() == ["AFC"]


# =========================================================================== follow-up picks
# --------------------------------------------------------------------------- live_lineups valuation lookups
def _valuation_rows(wh: Warehouse, rows: list[tuple[int, str, int]]) -> None:
    """Create `tm_player_valuations` and insert `rows` (player_id, date, value) one by one in the given order,
    bypassing the warehouse key so two values of one (player, date) can be stored."""
    wh._conn.execute("CREATE TABLE tm_player_valuations (player_id BIGINT, date TIMESTAMP, market_value_in_eur BIGINT)")
    for row in rows:
        wh._conn.execute("INSERT INTO tm_player_valuations VALUES (?, ?, ?)", list(row))


def test_usual_xi_value_breaks_a_same_day_valuation_tie_by_the_larger_value(tmp_path):
    same_day = [(1, "2024-01-01", 5_000_000), (1, "2024-01-01", 7_000_000), (1, "2024-01-01", 6_000_000)]
    picks = []
    for i, order in enumerate(_both_orders(same_day)):
        with Warehouse(tmp_path / f"wh{i}.duckdb") as wh:
            _valuation_rows(wh, order)
            apps = pd.DataFrame(
                {
                    "appearance_id": ["a1"],
                    "player_id": [1],
                    "player_club_id": [1],
                    "date": [pd.Timestamp("2024-06-01")],
                    "minutes_played": [90],
                }
            )
            wh.upsert("tm_appearances", apps, keys=["appearance_id"])
            wh.upsert("tm_players", pd.DataFrame({"player_id": [1], "name": ["P1"]}), keys=["player_id"])
            picks.append(_usual_xi_values(wh, 1, date(2024, 8, 1))["value"].tolist())
    assert picks == [[7_000_000], [7_000_000]]


@pytest.mark.parametrize("reverse", [False, True])
def test_latest_value_refuses_a_name_shared_by_two_players(tmp_path, reverse):
    players = [(7, "Danilo"), (3, "Danilo"), (5, "Rodri")]
    if reverse:
        players.reverse()
    with Warehouse(tmp_path / "wh.duckdb") as wh:
        _valuation_rows(wh, [(7, "2024-01-01", 10_000_000), (3, "2024-01-01", 20_000_000), (5, "2024-01-01", 9)])
        wh.upsert(
            "tm_players",
            pd.DataFrame({"player_id": [p for p, _ in players], "name": [n for _, n in players]}),
            keys=["player_id"],
        )
        assert _latest_value(wh, "danilo", date(2024, 8, 1)) == 0.0  # ambiguous: no value rather than a guess
        assert _latest_value(wh, "Rodri", date(2024, 8, 1)) == 9.0


# --------------------------------------------------------------------------- squad_value as-of value and game pick
def _seed_one_game_lineups(wh: Warehouse, games: dict[int, list[int]], vals: list[tuple[int, str, int]]) -> None:
    """tm_games / tm_game_lineups for club 1 at home to club 2 on 2024-08-01, `games` = {game_id: starters}, and
    the valuations `vals` (player_id, date, value) upserted in the given order. `tm_appearances` gets one row of
    an unrelated club when it does not exist yet, so the usual-XI step has its table."""
    if not wh.table_exists("tm_appearances"):
        wh.upsert(
            "tm_appearances",
            pd.DataFrame(
                {
                    "appearance_id": ["other"],
                    "player_id": [999],
                    "player_club_id": [9],
                    "date": [pd.Timestamp("2024-06-01")],
                    "minutes_played": [90],
                }
            ),
            keys=["appearance_id"],
        )
    wh.upsert(
        "tm_games",
        pd.DataFrame(
            {"game_id": list(games), "date": pd.Timestamp("2024-08-01"), "home_club_id": 1, "away_club_id": 2}
        ),
        keys=["game_id"],
    )
    lineups = [(g, p) for g, starters in games.items() for p in starters]
    wh.upsert(
        "tm_game_lineups",
        pd.DataFrame(
            {
                "game_lineups_id": [f"{g}_{p}" for g, p in lineups],
                "game_id": [g for g, _ in lineups],
                "club_id": 1,
                "player_id": [p for _, p in lineups],
                "type": "starting_lineup",
            }
        ),
        keys=["game_lineups_id"],
    )
    for pid, when, value in vals:
        wh.upsert(
            "tm_player_valuations",
            pd.DataFrame({"player_id": [pid], "date": [pd.Timestamp(when)], "market_value_in_eur": [value]}),
            keys=["player_id", "date"],
        )


def test_squad_value_takes_the_latest_valuation_of_a_day(tmp_path):
    vals = [(5, "2024-01-01 00:00", 9_000_000), (5, "2024-01-01 12:00", 3_000_000)]  # one day, two stamps
    picks = []
    for i, order in enumerate(_both_orders(vals)):
        with Warehouse(tmp_path / f"wh{i}.duckdb") as wh:
            _seed_one_game_lineups(wh, {100: [5]}, order)
            out = squad_value_context(wh, pd.DataFrame({"match_id": ["m1"], "tm_game_id": [100]}))
            picks.append(float(out["sv_home_xi_value"].iloc[0]))
    assert picks == [pytest.approx(np.log1p(3_000_000))] * 2


def test_squad_value_uses_the_smallest_game_id_when_a_match_links_two_games(tmp_path):
    vals = [(1, "2024-01-01", 1_000_000), (2, "2024-01-01", 2_000_000)]
    links = [("m1", 101), ("m1", 100)]
    picks = []
    for i, order in enumerate(_both_orders(links)):
        with Warehouse(tmp_path / f"wh{i}.duckdb") as wh:
            _seed_one_game_lineups(wh, {100: [1], 101: [2]}, vals)
            ctx = pd.DataFrame({"match_id": [m for m, _ in order], "tm_game_id": [g for _, g in order]})
            out = squad_value_context(wh, ctx)
            picks.append(out[["match_id", "sv_home_xi_value"]].values.tolist())
    assert picks == [[["m1", pytest.approx(np.log1p(1_000_000))]]] * 2


# --------------------------------------------------------------------------- PE-45: missing share
@pytest.mark.parametrize(("absent", "share"), [((), 0.0), ((11,), 1 / 11), (tuple(range(1, 12)), 1.0)])
def test_squad_value_missing_share_is_zero_one_part_or_all(tmp_path, absent, share):
    """Usual XI: players 1-11 (club 1, equal value). Today's lineup fields the usual XI minus `absent`, plus
    player 99 so the lineup is never empty."""
    with Warehouse(tmp_path / "wh.duckdb") as wh:
        apps = pd.DataFrame(
            {
                "appearance_id": [f"a{p}" for p in range(1, 12)],
                "player_id": list(range(1, 12)),
                "player_club_id": 1,
                "date": pd.Timestamp("2024-06-01"),
                "minutes_played": 90,
            }
        )
        wh.upsert("tm_appearances", apps, keys=["appearance_id"])
        starters = [p for p in range(1, 12) if p not in absent] + [99]
        _seed_one_game_lineups(wh, {100: starters}, [(p, "2024-01-01", 1_000_000) for p in [*range(1, 12), 99]])
        out = squad_value_context(wh, pd.DataFrame({"match_id": ["m1"], "tm_game_id": [100]}))
    assert out["sv_home_missing_pct"].iloc[0] == pytest.approx(share)


# --------------------------------------------------------------------------- settle: kickoff move, own snapshot
def test_kickoff_moved_breaks_a_capture_time_tie_by_the_later_kickoff(tmp_path):
    late = CAPTURED + timedelta(hours=3)
    quotes = [
        _quote("e1", "home", source="odds_api:pinnacle", decimal_odds=2.0),
        _quote("e1", "home", source="odds_api:pinnacle", decimal_odds=2.0, captured=late),
        _quote(
            "e1", "home", source="odds_api:betfair_ex_uk", decimal_odds=2.0, captured=late, ko=KO + timedelta(hours=5)
        ),
    ]
    picks = []
    for store in _stores(tmp_path, [quotes, [quotes[0], quotes[2], quotes[1]]]):
        fixture = str(store.fixtures()["fixture_key"].iloc[0])
        assert store.quotes_for_fixture(fixture, before=KO + timedelta(days=1))["fixture_key"].nunique() == 1
        picks.append(_kickoff_moved(store, fixture, KO, KO + timedelta(days=1)))
    assert picks == [KO + timedelta(hours=5)] * 2


def test_own_snapshot_exchange_close_takes_the_smallest_event_id_at_a_tied_capture(tmp_path):
    decided = CAPTURED - timedelta(hours=1)
    books = {"pm1": (0.40, 0.30, 0.30), "pm2": (0.50, 0.25, 0.25)}
    quotes = [
        _quote(eid, s, source="polymarket", bid=p - 0.02, ask=p + 0.02, instrument_id=f"{eid}-{s}")
        for eid, probs in books.items()
        for s, p in zip(SEL, probs, strict=True)
    ]
    picks = []
    for store in _stores(tmp_path, _both_orders(quotes)):
        fixture = str(store.fixtures()["fixture_key"].iloc[0])
        assert store.quotes_for_fixture(fixture, before=KO)["source_event_id"].nunique() == 2
        close = _own_snapshot_close(store, fixture, KO, "home", "polymarket", decided)
        picks.append(None if close is None else close.closing_price)
    assert picks == [pytest.approx(1 / 0.40)] * 2


def test_own_snapshot_bookmaker_close_takes_the_smallest_event_id_at_a_tied_capture(tmp_path):
    decided = CAPTURED - timedelta(hours=1)
    books = {"e1": (2.0, 3.5, 4.0), "e2": (2.2, 3.4, 3.6)}
    quotes = [_quote(eid, s, decimal_odds=o) for eid, odds in books.items() for s, o in zip(SEL, odds, strict=True)]
    picks = []
    for store in _stores(tmp_path, _both_orders(quotes)):
        fixture = str(store.fixtures()["fixture_key"].iloc[0])
        close = _own_snapshot_close(store, fixture, KO, "home", "odds_api:williamhill", decided)
        picks.append(None if close is None else (close.closing_price, close.fair_close_odds))
    fair_home = power_devig_odds(list(books["e1"]))[0]
    assert picks == [(2.0, pytest.approx(1 / fair_home))] * 2


# =========================================================================== sweep picks
# --------------------------------------------------------------------------- context.club_id_map
@pytest.mark.parametrize("reverse", [False, True])
def test_club_id_map_breaks_a_last_season_tie_by_the_smallest_club_id(tmp_path, reverse):
    clubs = [("7", "Team00 FC"), ("12", "Team00 Club"), ("3", "Team00")]
    if reverse:
        clubs.reverse()
    with Warehouse(tmp_path / "wh.duckdb") as wh:
        wh.upsert(
            "tm_clubs",
            pd.DataFrame({"club_id": [c for c, _ in clubs], "name": [n for _, n in clubs], "last_season": 2025}),
            keys=["club_id"],
        )
        assert club_id_map(wh, ["Team00", "Team01"]) == {"Team00": 3}


# --------------------------------------------------------------------------- settle.fd_match
class FrameLake:
    """Answers every query with one fixed frame."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def sql(self, query, params=None):
        return self.frame


FD_COLS = [
    "match_date",
    "home_team",
    "away_team",
    "ft_result",
    "betfair_ex_home_close",
    "betfair_ex_draw_close",
    "betfair_ex_away_close",
]


def test_fd_match_refuses_two_equally_similar_fixtures():
    rows = [
        (pd.Timestamp("2026-10-24"), "Alphaville United", "Betaford Town", "H", 1.8, 3.9, 5.0),
        (pd.Timestamp("2026-10-24"), "Alphaville City", "Betaford Town", "A", 2.5, 3.3, 3.0),
    ]
    picks = [
        LakeCloseSource(FrameLake(pd.DataFrame(order, columns=FD_COLS))).fd_match(
            "EPL", KO, "Alphaville", "Betaford Town"
        )
        for order in _both_orders(rows)
    ]
    assert picks == [None, None]


# --------------------------------------------------------------------------- qa_data.resolve_fixture
def test_resolve_fixture_takes_the_earliest_meeting_in_the_window():
    rows = [
        ("m-late", pd.Timestamp("2024-03-20"), "Arsenal", "Chelsea"),
        ("m-cup", pd.Timestamp("2024-03-05"), "Chelsea", "Arsenal"),
        ("m-early", pd.Timestamp("2024-03-05"), "Arsenal", "Chelsea"),
    ]
    picks = []
    for order in _both_orders(rows):
        feats = pd.DataFrame(order, columns=["match_id", "date", "home_team", "away_team"]).assign(
            PSH=2.0, PSD=3.4, PSA=3.8
        )
        got = resolve_fixture(feats, pd.Timestamp("2024-03-01"), "Arsenal", "Chelsea")
        picks.append(got["match_id"].tolist())
    assert picks == [["m-early"]] * 2


# --------------------------------------------------------------------------- backtest report: best staking rule
def test_best_strategy_breaks_a_mean_clv_tie_by_strategy_name():
    rows = [
        {"model": "gbdt", "strategy": "kelly", "n_bets": 10, "mean_clv_pct": 1.5},
        {"model": "gbdt", "strategy": "flat", "n_bets": 10, "mean_clv_pct": 1.5},
        {"model": "gbdt", "strategy": "value", "n_bets": 10, "mean_clv_pct": 0.5},
        {"model": "dc", "strategy": "flat", "n_bets": 10, "mean_clv_pct": -0.2},
    ]
    picks = [_best_strategy(pd.DataFrame(order))[["model", "strategy"]].values.tolist() for order in _both_orders(rows)]
    assert picks == [[["dc", "flat"], ["gbdt", "flat"]]] * 2


# --------------------------------------------------------------------------- RAG cut points
class FakeIndex:
    """Collects the documents `build_rag_index` adds."""

    def __init__(self) -> None:
        self.ids: list[str] = []

    def add(self, docs) -> None:
        self.ids += [d.doc_id for d in docs]


def test_match_documents_cut_breaks_a_date_tie_by_match_id():
    rows = [
        {"match_id": f"m{i:02d}", "date": pd.Timestamp("2024-03-02"), "home_team": "A", "away_team": "B",
         "home_goals": 1, "away_goals": 0}
        for i in range(30)
    ] + [{"match_id": "old", "date": pd.Timestamp("2024-03-01"), "home_team": "A", "away_team": "B",
          "home_goals": 0, "away_goals": 0}]  # fmt: skip
    picks = [[d.doc_id for d in match_documents(pd.DataFrame(order), limit=5)] for order in _both_orders(rows)]
    assert picks == [[f"match:m{i:02d}" for i in range(5)]] * 2


def _rag_ids(tmp_path, name: str, tables: dict[str, tuple[pd.DataFrame, list[str]]], **kw) -> list[str]:
    """Doc ids `build_rag_index` adds from a warehouse holding `tables` ({table: (frame, keys)}), frames stored
    in their row order."""
    with Warehouse(tmp_path / f"{name}.duckdb") as wh:
        for table, (frame, keys) in tables.items():
            wh.upsert(table, frame, keys=keys)
        index = FakeIndex()
        pipeline.build_rag_index(wh, index, **kw)  # type: ignore[arg-type]
    return sorted(index.ids)


def test_rag_prediction_cut_breaks_a_date_tie_by_match_id(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "get_settings", lambda: type("S", (), {"backtest_dir": tmp_path / "none"})())
    preds = pd.DataFrame(
        {
            "match_id": [f"m{i:04d}" for i in range(1502)],
            "model_name": "gbdt",
            "run_id": "r1",
            "date": pd.Timestamp("2024-03-02"),
            "home_team": "A",
            "away_team": "B",
            "p_home": 0.5,
            "p_draw": 0.3,
            "p_away": 0.2,
        }
    )
    picks = [
        _rag_ids(tmp_path, f"p{i}", {"model_predictions": (f, ["match_id", "model_name", "run_id"])})
        for i, f in enumerate(_both_orders_frame(preds))
    ]
    assert picks[0] == picks[1] == sorted(f"pred:gbdt:m{i:04d}" for i in range(2, 1502))


def test_rag_news_cut_breaks_a_time_tie_by_item_id(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "get_settings", lambda: type("S", (), {"backtest_dir": tmp_path / "none"})())
    news = pd.DataFrame(
        {
            "item_id": [f"n{i:04d}" for i in range(2003)],
            "published_at": pd.Timestamp("2024-03-02 10:00"),
            "title": "t",
            "summary": "s",
        }
    )
    picks = [
        _rag_ids(tmp_path, f"n{i}", {"news_items": (f, ["item_id"])}) for i, f in enumerate(_both_orders_frame(news))
    ]
    assert picks[0] == picks[1] == sorted(f"news:n{i:04d}" for i in range(3, 2003))


def test_rag_espn_cut_breaks_a_date_tie_by_event_id(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "get_settings", lambda: type("S", (), {"backtest_dir": tmp_path / "none"})())
    fx = pd.DataFrame(
        {
            "eventId": list(range(40)),
            "matched_match_id": [f"m{i:02d}" for i in range(40)],
            "date": pd.Timestamp("2024-03-02"),
            "home_team": "A",
            "away_team": "B",
            "league_code": "E0",
            "homeTeamScore": 1,
            "awayTeamScore": 0,
            "homeTeamId": 1,
            "awayTeamId": 2,
            "statusId": 28,
        }
    )
    stats = pd.DataFrame({"eventId": [0], "teamId": [1]})
    events = pd.DataFrame({"eventId": [0], "keyEventOrder": [1], "keyEventShortText": ["goal"]})
    picks = [
        _rag_ids(
            tmp_path,
            f"e{i}",
            {
                "espn_fixtures_mapped": (f, ["eventId"]),
                "espn_team_stats": (stats, ["eventId", "teamId"]),
                "espn_key_events": (events, ["eventId", "keyEventOrder"]),
            },
            max_matches=5,
        )
        for i, f in enumerate(_both_orders_frame(fx))
    ]
    assert picks[0] == picks[1] == sorted(f"espn:m{i:02d}" for i in range(5))


def _both_orders_frame(frame: pd.DataFrame) -> list[pd.DataFrame]:
    return [frame.reset_index(drop=True), frame.iloc[::-1].reset_index(drop=True)]


def test_upcoming_fixtures_cut_breaks_a_date_tie_by_teams(monkeypatch):
    day = (pd.Timestamp.now().normalize() + pd.Timedelta(days=3)).date()
    teams = [f"Club{i:02d}" for i in range(70)]
    fixtures = pd.DataFrame({"date": [day] * 70, "home_team": teams, "away_team": teams[1:] + teams[:1]})

    class FakeOpenFootball:
        order = fixtures

        def fetch_upcoming_fixtures(self, code, year):
            return FakeOpenFootball.order.copy()

    monkeypatch.setattr(pipeline, "OpenFootballSource", FakeOpenFootball)
    monkeypatch.setattr(pipeline, "_apply_confirmed_lineups", lambda wh, fx: fx)
    features = pd.DataFrame(
        {"home_team": teams, "away_team": teams[1:] + teams[:1], "date": pd.Timestamp("2024-01-01")}
    )
    picks = []
    for order in _both_orders_frame(fixtures):
        FakeOpenFootball.order = order
        got = pipeline.upcoming_fixture_frame(None, features, leagues=["E0"])  # type: ignore[arg-type]
        picks.append(sorted(got["home_team"]))
    assert picks[0] == picks[1] == teams[:60]


# --------------------------------------------------------------------------- paper store: collect log
def test_collect_log_reads_break_a_call_time_tie_conservatively(tmp_path):
    t = CAPTURED
    calls = [
        {"remaining": 480, "used": 20, "cost": 2},
        {"remaining": 470, "used": 30, "cost": 3},
        {"remaining": 490, "used": 10, "cost": 1},
    ]
    picks = []
    for i, order in enumerate(_both_orders(calls)):
        store = PaperStore(tmp_path / f"paper{i}.duckdb")
        try:
            for c in order:
                store.record_call("odds_api", "soccer_epl", status="ok", called_at=t, **c)
            picks.append((store.budget("odds_api"), store.last_used("odds_api"), store.cost_per_call("odds_api")))
        finally:
            store.close()
    assert picks == [(470, 30, 3)] * 2


# --------------------------------------------------------------------------- venues: one venue per club
@pytest.mark.parametrize("reverse", [False, True])
def test_country_venues_keep_the_smallest_venue_name_per_club(tmp_path, reverse):
    bindings = [
        {"clubLabel": {"value": "Arsenal"}, "venueLabel": {"value": "Highbury"}, "coord": {"value": "Point(-0.10 51.55)"}},
        {"clubLabel": {"value": "Arsenal"}, "venueLabel": {"value": "Emirates Stadium"}, "coord": {"value": "Point(-0.11 51.56)"}},
        {"clubLabel": {"value": "Chelsea"}, "venueLabel": {"value": "Stamford Bridge"}, "coord": {"value": "Point(-0.19 51.48)"}},
    ]  # fmt: skip
    if reverse:
        bindings.reverse()

    class FakeHttp:
        def get_json(self, url, params=None, headers=None):
            return {"results": {"bindings": bindings}}

    got = VenueGeocoder(cache_dir=tmp_path, http=FakeHttp()).fetch_country_venues("England")  # type: ignore[arg-type]
    assert got.set_index("club")["venue"].to_dict() == {"Arsenal": "Emirates Stadium", "Chelsea": "Stamford Bridge"}
