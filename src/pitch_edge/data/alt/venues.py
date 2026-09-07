"""Stadium geolocation: Wikidata (bulk, cached) with a bundled fallback.

Coordinates are the join key for two real alternative-data features: weather
at kickoff (Open-Meteo) and the away side's travel distance (fatigue/travel
index). Wikidata is CC0 and its SPARQL endpoint is free; we issue one bulk
query per country and cache the result, so the endpoint sees a handful of
requests in total rather than one per club.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient
from pitch_edge.data.teams import TeamNameResolver

logger = logging.getLogger(__name__)

SPARQL_URL = "https://query.wikidata.org/sparql"

COUNTRY_QIDS = {
    "England": "Q21", "Scotland": "Q22", "Wales": "Q25", "Germany": "Q183", "Spain": "Q29", "Italy": "Q38",
    "France": "Q142", "Netherlands": "Q55", "Belgium": "Q31", "Portugal": "Q45", "Turkey": "Q43", "Greece": "Q41",
    "Argentina": "Q414", "Austria": "Q40", "Brazil": "Q155", "China": "Q148", "Denmark": "Q35", "Finland": "Q33",
    "Ireland": "Q27", "Japan": "Q17", "Mexico": "Q96", "Norway": "Q20", "Poland": "Q36", "Romania": "Q218",
    "Russia": "Q159", "Sweden": "Q34", "Switzerland": "Q39", "USA": "Q30",
}

# Fallback so the feature pipeline works fully offline for the flagship league.
BUNDLED_VENUES: dict[str, tuple[float, float]] = {
    "Arsenal": (51.5549, -0.1084), "Aston Villa": (52.5092, -1.8848), "Bournemouth": (50.7352, -1.8383),
    "Brentford": (51.4907, -0.2886), "Brighton": (50.8616, -0.0837), "Burnley": (53.7890, -2.2302),
    "Chelsea": (51.4817, -0.1910), "Crystal Palace": (51.3983, -0.0856), "Everton": (53.4388, -2.9663),
    "Fulham": (51.4750, -0.2217), "Ipswich": (52.0550, 1.1447), "Leeds": (53.7778, -1.5722),
    "Leicester": (52.6204, -1.1422), "Liverpool": (53.4308, -2.9608), "Luton": (51.8840, -0.4316),
    "Man City": (53.4831, -2.2004), "Man United": (53.4631, -2.2913), "Newcastle": (54.9756, -1.6217),
    "Nott'm Forest": (52.9399, -1.1329), "Sheffield United": (53.3703, -1.4710), "Southampton": (50.9058, -1.3911),
    "Sunderland": (54.9146, -1.3882), "Tottenham": (51.6043, -0.0664), "West Ham": (51.5387, -0.0166),
    "Wolves": (52.5903, -2.1305), "Norwich": (52.6222, 1.3091), "Watford": (51.6499, -0.4015),
    "West Brom": (52.5090, -1.9639), "Sheffield Weds": (53.4115, -1.5006), "Middlesbrough": (54.5781, -1.2168),
    "Stoke": (52.9884, -2.1755), "Swansea": (51.6428, -3.9347), "Cardiff": (51.4728, -3.2030),
    "Hull": (53.7466, -0.3678), "QPR": (51.5093, -0.2321), "Millwall": (51.4859, -0.0509),
    "Coventry": (52.4481, -1.4956), "Blackburn": (53.7286, -2.4893), "Preston": (53.7722, -2.6880),
    "Bristol City": (51.4400, -2.6205), "Plymouth": (50.3882, -4.1509), "Portsmouth": (50.7964, -1.0639),
    "Derby": (52.9150, -1.4472), "Oxford": (51.7163, -1.2081), "Birmingham": (52.4756, -1.8681),
    "Wrexham": (53.0517, -3.0038), "Charlton": (51.4865, 0.0364), "Bolton": (53.5805, -2.5357),
}


class VenueGeocoder:
    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "wikidata"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=3.0, timeout_s=90)

    def fetch_country_venues(self, country: str) -> pd.DataFrame:
        """All football clubs in `country` with a geocoded home venue (Wikidata)."""
        qid = COUNTRY_QIDS.get(country)
        if qid is None:
            return pd.DataFrame(columns=["club", "venue", "lat", "lon", "country"])
        query = f"""
        SELECT ?clubLabel ?venueLabel ?coord WHERE {{
          ?club wdt:P31 wd:Q476028; wdt:P17 wd:{qid}; wdt:P115 ?venue.
          ?venue wdt:P625 ?coord.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        try:
            payload = self.http.get_json(
                SPARQL_URL, params={"format": "json", "query": query}, headers={"Accept": "application/sparql-results+json"}
            )
        except Exception as exc:
            logger.warning("Wikidata venue query failed for %s: %s", country, exc)
            return pd.DataFrame(columns=["club", "venue", "lat", "lon", "country"])
        rows = []
        for b in payload.get("results", {}).get("bindings", []):
            coord = b.get("coord", {}).get("value", "")
            m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", coord)
            if not m:
                continue
            rows.append(
                {
                    "club": b.get("clubLabel", {}).get("value"),
                    "venue": b.get("venueLabel", {}).get("value"),
                    "lon": float(m.group(1)),
                    "lat": float(m.group(2)),
                    "country": country,
                }
            )
        return pd.DataFrame(rows).drop_duplicates(subset="club")

    def build_team_coordinates(self, teams: list[str], countries: list[str]) -> pd.DataFrame:
        """Map canonical team names -> (lat, lon), Wikidata first, bundled fallback second."""
        resolver = TeamNameResolver(teams)
        found: dict[str, tuple[float, float, str]] = {}
        for country in dict.fromkeys(countries):
            venues = self.fetch_country_venues(country)
            for _, row in venues.iterrows():
                canonical = resolver.resolve(str(row["club"]), cutoff=0.9)
                if canonical and canonical not in found:
                    found[canonical] = (row["lat"], row["lon"], "wikidata")
        for team in teams:
            if team not in found and team in BUNDLED_VENUES:
                lat, lon = BUNDLED_VENUES[team]
                found[team] = (lat, lon, "bundled")
        out = pd.DataFrame(
            [{"team": t, "lat": v[0], "lon": v[1], "geo_source": v[2]} for t, v in found.items()]
        )
        return out
