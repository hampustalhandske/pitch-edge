import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from pitch_edge.artifacts import build_data_universe
from pitch_edge.config import get_settings
from pitch_edge.data.ingest import ingest_transfermarkt_open
from pitch_edge.data.storage import Warehouse

s = get_settings()
t0 = time.time()
with Warehouse(s.db_path) as wh:
    print("transfermarkt", ingest_transfermarkt_open(wh), round(time.time() - t0), "s", flush=True)
    for t in (
        "tm_games",
        "tm_game_events",
        "tm_appearances",
        "tm_players",
        "tm_player_valuations",
        "tm_game_lineups",
        "tm_clubs",
        "tm_competitions",
    ):
        print(t, wh.count(t), flush=True)
    build_data_universe(wh)
print("STAGE B2 COMPLETE", flush=True)
