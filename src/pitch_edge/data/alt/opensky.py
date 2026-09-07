"""OpenSky Network connector — connector is real, the *feature* is stubbed.

Anonymous access to `/states/all` is free (rate-limited), so we can list
aircraft in a bounding box around an airport right now. Attributing a given
aircraft to a specific team's charter, however, needs licensed tail-number /
operator data we do not have, so nothing here feeds a model. The
fatigue/travel index uses fixture congestion + venue distance instead (see
`travel.py`). Kept so the plug-in point exists and is exercised by tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient

logger = logging.getLogger(__name__)

STATES_URL = "https://opensky-network.org/api/states/all"
STATE_COLUMNS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact", "longitude", "latitude",
    "baro_altitude", "on_ground", "velocity", "true_track", "vertical_rate", "sensors", "geo_altitude",
    "squawk", "spi", "position_source",
]


class OpenSkySource:
    name = "opensky"
    implemented_as_feature = False  # documented honestly in README/dashboard

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "opensky"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=10.0)

    def fetch_states_in_bbox(self, lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> pd.DataFrame:
        params = {"lamin": lat_min, "lomin": lon_min, "lamax": lat_max, "lomax": lon_max}
        try:
            payload = self.http.get_json(STATES_URL, params=params, cache=False)
        except Exception as exc:
            logger.info("OpenSky unavailable: %s", exc)
            return pd.DataFrame(columns=STATE_COLUMNS)
        states = payload.get("states") or []
        df = pd.DataFrame([s[: len(STATE_COLUMNS)] for s in states], columns=STATE_COLUMNS)
        if not df.empty:
            df["callsign"] = df["callsign"].astype(str).str.strip()
            df["snapshot_ts"] = pd.to_datetime(payload.get("time"), unit="s", errors="coerce")
        return df
