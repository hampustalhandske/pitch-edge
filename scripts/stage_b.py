import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from pitch_edge.artifacts import build_data_universe
from pitch_edge.config import get_settings
from pitch_edge.data.ingest import (
    ingest_kalshi,
    ingest_news,
    ingest_polymarket,
    ingest_sweden,
    ingest_thesportsdb_sweden,
    ingest_transfermarkt_open,
)
from pitch_edge.data.storage import Warehouse

s = get_settings()
s.ensure_dirs()
t0 = time.time()
with Warehouse(s.db_path) as wh:
    print("sweden", ingest_sweden(wh), flush=True)
    print("kalshi", ingest_kalshi(wh), flush=True)
    print("news", ingest_news(wh), flush=True)
    print("polymarket", ingest_polymarket(wh), flush=True)
    print("thesportsdb", ingest_thesportsdb_sweden(wh), flush=True)
    print("transfermarkt", ingest_transfermarkt_open(wh), round(time.time() - t0), "s", flush=True)
    for t in (
        "tm_games",
        "tm_game_events",
        "tm_appearances",
        "tm_players",
        "tm_player_valuations",
        "tm_game_lineups",
        "tsdb_events",
        "tsdb_players",
        "market_snapshots",
        "news_items",
    ):
        print(t, wh.count(t), flush=True)
    build_data_universe(wh)
print("STAGE B COMPLETE", round(time.time() - t0), "s", flush=True)
