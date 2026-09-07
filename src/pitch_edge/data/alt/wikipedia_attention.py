"""Attention-velocity signal from Wikipedia pageviews (free, keyless, generous ToS).

The Wikimedia REST API (`wikimedia.org/api/rest_v1/metrics/pageviews`) publishes daily (and
hourly) per-article view counts back to 2015-07. A multi-sigma spike in a club's article the day
before kickoff, with no corresponding item in the RSS feeds yet, is a machine-detectable
"something is happening that has not hit the English-language press" signal — a cleaner,
ToS-clean proxy for the sentiment-velocity idea than social media.

Leakage rule: the feature for a match on day D uses views up to and including D-1 only (the
daily count for D-1 is final at 00:00 UTC on D, before any kickoff), against a trailing baseline
ending on D-2. Match-day views are never used — they contain the result.

Article titles are resolved once per club via the MediaWiki search API (a published API, not a
crawl) with a small override table for the awkward short names in the spine.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient
from pitch_edge.data.teams import ALIASES, normalise

logger = logging.getLogger(__name__)

PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/user/"
    "{article}/{granularity}/{start}/{end}"
)
SEARCH_URL = "https://{lang}.wikipedia.org/w/api.php"

# spine (football-data.co.uk) short name -> English Wikipedia article title
TITLE_OVERRIDES: dict[str, str] = {
    "Man United": "Manchester United F.C.",
    "Man City": "Manchester City F.C.",
    "Nott'm Forest": "Nottingham Forest F.C.",
    "Wolves": "Wolverhampton Wanderers F.C.",
    "Brighton": "Brighton & Hove Albion F.C.",
    "West Brom": "West Bromwich Albion F.C.",
    "Sheffield United": "Sheffield United F.C.",
    "Sheffield Weds": "Sheffield Wednesday F.C.",
    "QPR": "Queens Park Rangers F.C.",
    "Ath Madrid": "Atlético Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Sociedad": "Real Sociedad",
    "Espanol": "RCD Espanyol",
    "Vallecano": "Rayo Vallecano",
    "Celta": "RC Celta de Vigo",
    "Betis": "Real Betis",
    "La Coruna": "Deportivo de La Coruña",
    "Santander": "Racing de Santander",
    "Alaves": "Deportivo Alavés",
    "Sp Gijon": "Sporting de Gijón",
    "Inter": "Inter Milan",
    "Milan": "AC Milan",
    "Roma": "AS Roma",
    "Lazio": "SS Lazio",
    "Verona": "Hellas Verona FC",
    "Napoli": "SSC Napoli",
    "Juventus": "Juventus FC",
    "Fiorentina": "ACF Fiorentina",
    "Atalanta": "Atalanta BC",
    "Torino": "Torino FC",
    "Bologna": "Bologna FC 1909",
    "Udinese": "Udinese Calcio",
    "Sassuolo": "U.S. Sassuolo Calcio",
    "Genoa": "Genoa CFC",
    "Cagliari": "Cagliari Calcio",
    "Parma": "Parma Calcio 1913",
    "Spal": "S.P.A.L.",
    "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer 04 Leverkusen",
    "M'gladbach": "Borussia Mönchengladbach",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "1. FC Köln",
    "Mainz": "1. FSV Mainz 05",
    "Hertha": "Hertha BSC",
    "Schalke 04": "FC Schalke 04",
    "Hamburg": "Hamburger SV",
    "Stuttgart": "VfB Stuttgart",
    "Wolfsburg": "VfL Wolfsburg",
    "Freiburg": "SC Freiburg",
    "Augsburg": "FC Augsburg",
    "Bochum": "VfL Bochum",
    "Darmstadt": "SV Darmstadt 98",
    "Bielefeld": "Arminia Bielefeld",
    "Greuther Furth": "SpVgg Greuther Fürth",
    "Paderborn": "SC Paderborn 07",
    "Fortuna Dusseldorf": "Fortuna Düsseldorf",
    "Hannover": "Hannover 96",
    "Nurnberg": "1. FC Nürnberg",
    "Ingolstadt": "FC Ingolstadt 04",
    "Paris SG": "Paris Saint-Germain F.C.",
    "Marseille": "Olympique de Marseille",
    "Lyon": "Olympique Lyonnais",
    "St Etienne": "AS Saint-Étienne",
    "Monaco": "AS Monaco FC",
    "Lille": "Lille OSC",
    "Nice": "OGC Nice",
    "Rennes": "Stade Rennais F.C.",
    "Lens": "RC Lens",
    "Reims": "Stade de Reims",
    "Strasbourg": "RC Strasbourg Alsace",
    "Nantes": "FC Nantes",
    "Montpellier": "Montpellier HSC",
    "Toulouse": "Toulouse FC",
    "Brest": "Stade Brestois 29",
    "Lorient": "FC Lorient",
    "Le Havre": "Le Havre AC",
    "Metz": "FC Metz",
    "Auxerre": "AJ Auxerre",
    "Angers": "Angers SCO",
    "Clermont": "Clermont Foot",
    "Bordeaux": "FC Girondins de Bordeaux",
    "Nimes": "Nîmes Olympique",
    "Amiens": "Amiens SC",
    "Dijon": "Dijon FCO",
    "Caen": "Stade Malherbe Caen",
    "Guingamp": "En Avant Guingamp",
    "Troyes": "ES Troyes AC",
    "Ajaccio": "AC Ajaccio",
    "Bastia": "SC Bastia",
    "Sp Braga": "S.C. Braga",
    "Guimaraes": "Vitória S.C.",
    "Sporting CP": "Sporting CP",
    "PSV Eindhoven": "PSV Eindhoven",
    "St. Gilloise": "Royale Union Saint-Gilloise",
    "Standard": "Standard Liège",
    "Sp Lisbon": "Sporting CP",
    "Setubal": "Vitória F.C.",
    "Boavista": "Boavista F.C.",
    "Preston": "Preston North End F.C.",
    "Peterboro": "Peterborough United F.C.",
    "MGladbach": "Borussia Mönchengladbach",
    "Brescia": "Brescia Calcio",
    "Ajaccio GFCO": "Gazélec Ajaccio",
    "Lokeren": "K.S.C. Lokeren Oost-Vlaanderen",
    "Charleroi": "R. Charleroi S.C.",
    "Seraing": "R.F.C. Seraing",
    "Westerlo": "K.V.C. Westerlo",
    "Dender": "F.C.V. Dender E.H.",
    "RWD Molenbeek": "RWD Molenbeek",
    "Waasland-Beveren": "S.K. Beveren",
    "Beveren": "S.K. Beveren",
    "Mouscron": "Royal Excel Mouscron",
    # Türkiye — short spine names that search resolves to players or cities
    "Altay": "Altay S.K.",
    "Goztep": "Göztepe S.K.",
    "Kasimpasa": "Kasımpaşa S.K.",
    "Buyuksehyr": "İstanbul Başakşehir F.K.",
    "Osmanlispor": "Osmanlıspor",
    "Bodrumspor": "Bodrum F.K.",
    "Amedspor": "Amed S.K.",
    "Ad. Demirspor": "Adana Demirspor",
    "Mersin Idman Yurdu": "Mersin İdman Yurdu",
    "Erzurum BB": "Erzurumspor FK",
    "Erzurumspor": "Erzurumspor FK",
    "Akhisar Belediyespor": "Akhisarspor",
    "Karabukspor": "Kardemir Karabükspor",
    "Rizespor": "Çaykur Rizespor",
    "Ankaragucu": "MKE Ankaragücü",
}
_BAD_TITLE = re.compile(r"^\d{4}|season|list of|records|history of|stadium|supporters|rivalry|managers|\(footballer", re.I)
_CLUB_TITLE = re.compile(r"\b(F\.?C\.?|C\.?F\.?|S\.?C\.?|S\.?K\.?|A\.?F\.?C\.?|B\.?K\.?|I\.?F\.?|spor|FK|SV|BSC|SSC|AC|AS|US|CD|UD|RCD)\b")


class WikipediaAttentionSource:
    name = "wikipedia_pageviews"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None, lang: str = "en"):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "wikipedia"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=0.25)
        self.lang = lang

    # ---------------------------------------------------------------- titles
    def search_candidates(self, team: str) -> list[str]:
        """Search phrases from most to least specific: alias long names, then the spine name itself."""
        key = normalise(team)
        longs = sorted({k for k, v in ALIASES.items() if v == team and normalise(k) != key}, key=len, reverse=True)
        return [f"{q} football club" for q in [*longs, team]]

    def resolve_title(self, team: str) -> str | None:
        if team in TITLE_OVERRIDES:
            return TITLE_OVERRIDES[team]
        tokens = set(normalise(team).split())
        for query in self.search_candidates(team):
            try:
                payload = self.http.get_json(
                    SEARCH_URL.format(lang=self.lang),
                    params={"action": "query", "list": "search", "srsearch": query, "srlimit": 5, "format": "json"},
                )
            except Exception as exc:  # noqa: BLE001
                logger.info("wikipedia search %r failed: %s", query, exc)
                continue
            for hit in payload.get("query", {}).get("search", []):
                title = str(hit.get("title", ""))
                snippet = re.sub(r"<[^>]+>", "", str(hit.get("snippet", ""))).lower()
                if _BAD_TITLE.search(title) or "footballer" in snippet or "manager" in snippet[:80]:
                    continue  # a player/manager page, or a season/list page
                looks_like_club = "club" in snippet or bool(_CLUB_TITLE.search(title))
                if looks_like_club and (tokens & set(normalise(title).split()) or _CLUB_TITLE.search(title)):
                    return title
        return None

    def resolve_titles(self, teams: list[str]) -> pd.DataFrame:
        rows = [{"team": t, "article": self.resolve_title(t), "lang": self.lang} for t in teams]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------- pageviews
    def fetch_daily(self, article: str, start: str, end: str, project: str | None = None) -> pd.DataFrame:
        """Daily views for one article, fetched in calendar-year chunks so immutable history is cached."""
        project = project or f"{self.lang}.wikipedia"
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        frames = []
        for year in range(s.year, e.year + 1):
            a = max(s, pd.Timestamp(year=year, month=1, day=1))
            b = min(e, pd.Timestamp(year=year, month=12, day=31))
            url = PAGEVIEWS_URL.format(
                project=project,
                article=quote(article.replace(" ", "_"), safe=""),
                granularity="daily",
                start=a.strftime("%Y%m%d00"),
                end=b.strftime("%Y%m%d00"),
            )
            # the chunk containing "today" changes every day: never serve it from the cache
            live = b >= pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
            try:
                payload = self.http.get_json(url, cache=not live)
            except Exception as exc:  # noqa: BLE001 - 404 = no data for that range
                logger.info("pageviews %s %s: %s", article, year, exc)
                continue
            frames.append(self.parse_items(payload.get("items", []), article))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["article", "date", "views"])
        return pd.concat(frames, ignore_index=True).drop_duplicates(["article", "date"])

    @staticmethod
    def parse_items(items: list[dict], article: str) -> pd.DataFrame:
        rows = []
        for it in items or []:
            ts = str(it.get("timestamp", ""))[:8]
            try:
                rows.append({"article": article, "date": pd.Timestamp(ts), "views": int(it.get("views", 0))})
            except (ValueError, TypeError):
                continue
        return pd.DataFrame(rows)

    def pageviews_for_teams(self, titles: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        frames = []
        for r in titles.dropna(subset=["article"]).itertuples(index=False):
            df = self.fetch_daily(str(r.article), start, end)
            if not df.empty:
                frames.append(df.assign(team=r.team))
        if not frames:
            return pd.DataFrame(columns=["team", "article", "date", "views"])
        return pd.concat(frames, ignore_index=True)[["team", "article", "date", "views"]]


# -------------------------------------------------------------------- feature
def attention_anomaly(
    pageviews: pd.DataFrame, baseline_days: int = 28, lead_days: int = 1, min_baseline: int = 10
) -> pd.DataFrame:
    """Per (team, date): log-ratio and robust z of views on date-lead_days vs the trailing baseline
    ending one day earlier. Rows are indexed by the *match* date, so a join on (team, date) is
    leakage-free by construction."""
    if pageviews.empty:
        return pd.DataFrame(columns=["team", "date", "pv_views", "pv_anom", "pv_z", "pv_baseline"])
    out = []
    for team, g in pageviews.groupby("team"):
        s = g.set_index(pd.to_datetime(g["date"]))["views"].astype(float).sort_index()
        s = s[~s.index.duplicated()].asfreq("D")
        lv = np.log1p(s)
        base = lv.shift(lead_days + 1).rolling(baseline_days, min_periods=min_baseline)
        med = base.median()
        mad = base.apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
        cur = lv.shift(lead_days)
        anom = cur - med
        z = anom / (1.4826 * mad + 0.05)
        out.append(
            pd.DataFrame(
                {
                    "team": team,
                    "date": s.index,
                    "pv_views": s.shift(lead_days).to_numpy(),
                    "pv_anom": anom.to_numpy(),
                    "pv_z": z.to_numpy(),
                    "pv_baseline": np.expm1(med).to_numpy(),
                }
            )
        )
    return pd.concat(out, ignore_index=True).dropna(subset=["pv_anom"])


def attention_features(matches: pd.DataFrame, pageviews: pd.DataFrame, **kw) -> pd.DataFrame:
    """match_id -> pv_home_anom, pv_away_anom, pv_home_z, pv_away_z, pv_diff (home minus away)."""
    cols = ["match_id", "pv_home_anom", "pv_away_anom", "pv_home_z", "pv_away_z", "pv_diff"]
    if matches.empty or pageviews.empty:
        return pd.DataFrame(columns=cols)
    an = attention_anomaly(pageviews, **kw)
    if an.empty:
        return pd.DataFrame(columns=cols)
    m = matches[["match_id", "date", "home_team", "away_team"]].copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    an["date"] = pd.to_datetime(an["date"]).dt.normalize()
    h = an.rename(columns={"team": "home_team", "pv_anom": "pv_home_anom", "pv_z": "pv_home_z"})
    a = an.rename(columns={"team": "away_team", "pv_anom": "pv_away_anom", "pv_z": "pv_away_z"})
    m = m.merge(h[["home_team", "date", "pv_home_anom", "pv_home_z"]], on=["home_team", "date"], how="left")
    m = m.merge(a[["away_team", "date", "pv_away_anom", "pv_away_z"]], on=["away_team", "date"], how="left")
    m["pv_diff"] = m["pv_home_anom"] - m["pv_away_anom"]
    return m[cols]
