"""Alternative / unconventional data sources.

Status per source (kept in sync with README):
- venues + weather (Wikidata, Open-Meteo): REAL, feature used
- travel/fatigue (geodesic distance + fixture congestion): REAL, feature used
- referee tendencies (football-data.co.uk stats): REAL, feature used
- news sentiment velocity (RSS + VADER): REAL data, feature computed live, not in the backtested models
- pmxt archive (Polymarket soccer markets, historical): REAL, discovery via Gamma `/events/keyset`
  + hourly orderbook backfill via `r2v2.pmxt.dev` — replaces the old live-snapshot
  `polymarket.py`/`kalshi.py`/`odds_api.py` connectors (removed: no historical depth, and
  `polymarket.py`'s own `/markets?tag_slug=` filter was confirmed broken)
- opensky: connector only, no feature (attribution to teams impossible with free data)
- crowd audio: feature extraction real, no licensed data feed → stubbed
"""

from pitch_edge.data.alt.news import NewsScanner, sentiment_velocity
from pitch_edge.data.alt.opensky import OpenSkySource
from pitch_edge.data.alt.pmxt_archive import PMXTArchiveSource
from pitch_edge.data.alt.referee import referee_features
from pitch_edge.data.alt.travel import fatigue_index, haversine_km, rest_and_congestion, travel_distance
from pitch_edge.data.alt.venues import VenueGeocoder
from pitch_edge.data.alt.weather import OpenMeteoWeather

__all__ = [
    "NewsScanner",
    "OpenMeteoWeather",
    "OpenSkySource",
    "PMXTArchiveSource",
    "VenueGeocoder",
    "fatigue_index",
    "haversine_km",
    "referee_features",
    "rest_and_congestion",
    "sentiment_velocity",
    "travel_distance",
]
