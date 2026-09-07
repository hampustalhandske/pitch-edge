"""Fatigue / travel index from public data only.

Charter-flight tracking (OpenSky) cannot be tied to a specific team without
licensed tail-number data, so the *implemented* index uses two things we
can compute exactly: the away side's great-circle distance between home
grounds (from geocoded venues) and each team's fixture congestion (rest
days, matches in the trailing 7/14 days). Both are documented in the
literature as affecting cover rates. OpenSky stays a connector-only stub.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def rest_and_congestion(matches: pd.DataFrame) -> pd.DataFrame:
    """Per match: rest days and trailing-window match counts for both teams.

    Strictly uses matches dated *before* each fixture — no leakage.
    """
    long = pd.concat(
        [
            matches[["match_id", "date", "home_team"]].rename(columns={"home_team": "team"}).assign(is_home=1),
            matches[["match_id", "date", "away_team"]].rename(columns={"away_team": "team"}).assign(is_home=0),
        ]
    ).sort_values(["team", "date"]).reset_index(drop=True)
    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days

    def trailing(grp: pd.DataFrame) -> pd.DataFrame:
        dates = grp["date"].to_numpy()
        n7 = np.array([((dates < d) & (dates >= d - np.timedelta64(7, "D"))).sum() for d in dates])
        n14 = np.array([((dates < d) & (dates >= d - np.timedelta64(14, "D"))).sum() for d in dates])
        return pd.DataFrame({"matches_last_7d": n7, "matches_last_14d": n14}, index=grp.index)

    tr = long.groupby("team", group_keys=False)[["date"]].apply(trailing)
    long = long.join(tr)

    home = long[long["is_home"] == 1].set_index("match_id")[["rest_days", "matches_last_7d", "matches_last_14d"]]
    away = long[long["is_home"] == 0].set_index("match_id")[["rest_days", "matches_last_7d", "matches_last_14d"]]
    out = home.add_prefix("home_").join(away.add_prefix("away_"), how="outer")
    return out.reset_index()


def travel_distance(matches: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Away team's distance (km) from its own ground to the venue."""
    if coords.empty:
        return pd.DataFrame({"match_id": matches["match_id"], "away_travel_km": np.nan})
    cm = coords.set_index("team")[["lat", "lon"]].to_dict("index")
    dist = []
    for _, r in matches.iterrows():
        h, a = cm.get(r["home_team"]), cm.get(r["away_team"])
        dist.append(haversine_km(h["lat"], h["lon"], a["lat"], a["lon"]) if h and a else np.nan)
    return pd.DataFrame({"match_id": matches["match_id"].to_numpy(), "away_travel_km": dist})


def fatigue_index(features: pd.DataFrame) -> pd.Series:
    """Composite away-side fatigue score in [0, ~3]: short rest + congestion + long trip."""
    rest = features.get("away_rest_days", pd.Series(np.nan, index=features.index)).fillna(7).clip(1, 14)
    cong = features.get("away_matches_last_14d", pd.Series(0, index=features.index)).fillna(0)
    km = features.get("away_travel_km", pd.Series(np.nan, index=features.index)).fillna(0)
    return (1 - (rest - 1) / 13) + cong / 4 + (km / 1000).clip(0, 1)
