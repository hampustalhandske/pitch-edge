"""Open-Meteo historical weather at kickoff (free, keyless, no ToS friction).

One archive request per (venue, date-range) returns hourly temperature,
precipitation, wind and humidity; we pick the kickoff hour per match. Used as
a total-goals / tempo feature — rain and wind measurably suppress goals.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"]


class OpenMeteoWeather:
    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "open_meteo"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=0.7)

    def fetch_hourly(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        params = {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "start_date": start_date, "end_date": end_date,
            "hourly": ",".join(HOURLY_VARS), "timezone": "UTC",
        }
        payload = self.http.get_json(ARCHIVE_URL, params=params)
        hourly = payload.get("hourly", {})
        df = pd.DataFrame(hourly)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"])
        return df

    def forecast_at(self, lat: float, lon: float, when: pd.Timestamp) -> dict | None:
        """Forecast for one kickoff hour (≤ 16 days ahead); None when unavailable. Never cached."""
        when = pd.Timestamp(when)
        params = {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "start_date": when.strftime("%Y-%m-%d"), "end_date": when.strftime("%Y-%m-%d"),
            "hourly": ",".join(HOURLY_VARS), "timezone": "UTC",
        }
        try:
            payload = self.http.get_json(FORECAST_URL, params=params, cache=False)
        except Exception as exc:  # noqa: BLE001
            logger.info("Open-Meteo forecast unavailable: %s", exc)
            return None
        df = pd.DataFrame(payload.get("hourly", {}))
        if df.empty:
            return None
        df["time"] = pd.to_datetime(df["time"])
        i = df["time"].sub(when.floor("h")).abs().idxmin()
        rec = df.loc[i]
        return {f"wx_{v}": (float(rec[v]) if pd.notna(rec.get(v)) else None) for v in HOURLY_VARS}

    def weather_for_matches(self, matches: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
        """matches: match_id, date, home_team, optional kickoff_time ('HH:MM').
        coords: team, lat, lon. Returns one row per match with weather at kickoff hour."""
        if matches.empty or coords.empty:
            return pd.DataFrame(columns=["match_id", *[f"wx_{v}" for v in HOURLY_VARS]])
        coord_map = coords.set_index("team")[["lat", "lon"]].to_dict("index")
        out = []
        m = matches.copy()
        m["kick_hour"] = _kick_hour(m)
        for team, grp in m.groupby("home_team"):
            c = coord_map.get(team)
            if c is None:
                continue
            for _, season_grp in grp.groupby(grp["date"].dt.year):
                start = season_grp["date"].min().strftime("%Y-%m-%d")
                end = season_grp["date"].max().strftime("%Y-%m-%d")
                try:
                    hourly = self.fetch_hourly(c["lat"], c["lon"], start, end)
                except Exception as exc:
                    logger.warning("Open-Meteo failed for %s (%s..%s): %s", team, start, end, exc)
                    continue
                if hourly.empty:
                    continue
                hourly = hourly.set_index("time")
                for _, row in season_grp.iterrows():
                    ts = pd.Timestamp(row["date"]).normalize() + pd.Timedelta(hours=int(row["kick_hour"]))
                    if ts not in hourly.index:
                        nearest = hourly.index.get_indexer([ts], method="nearest")[0]
                        if nearest < 0:
                            continue
                        rec = hourly.iloc[nearest]
                    else:
                        rec = hourly.loc[ts]
                    out.append({"match_id": row["match_id"], **{f"wx_{v}": rec.get(v) for v in HOURLY_VARS}})
        return pd.DataFrame(out)


def _kick_hour(m: pd.DataFrame) -> pd.Series:
    if "kickoff_time" in m.columns:
        hours = pd.to_datetime(m["kickoff_time"].astype(str), format="%H:%M", errors="coerce").dt.hour
        return hours.fillna(15).astype(int)
    return pd.Series(15, index=m.index)
