"""Every one-row-per-group pick resolves ties by a total order (PE-36).

Each test builds the same tied rows twice, inserted in opposite orders, and requires the same pick in both,
naming the row the documented tie-break chooses. Nothing touches the network or the real lake: lake queries
run against an in-memory DuckDB that holds the tables the query reads.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pytest

from pitch_edge.data.sources.transfermarkt_open import TransfermarktOpenSource
from pitch_edge.data.storage import Warehouse
from pitch_edge.features.context import transfermarkt_context
from pitch_edge.features.live_lineups import _usual_xi_values
from pitch_edge.features.squad_value import squad_value_context
from pitch_edge.lake.mirror import MIRRORS, sync_mirror
from pitch_edge.odds.steam import detect_steam
from pitch_edge.odds.utils import no_vig_probabilities
from pitch_edge.paper.settle import LakeCloseSource, PmBar, _pm_bar_close
from pitch_edge.paper.store import PaperStore, Quote

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
