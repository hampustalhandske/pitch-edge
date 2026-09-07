"""Team-name harmonisation across sources.

football-data.co.uk names are the canonical form (it is the backtesting
spine, so everything else joins onto it). Other sources spell clubs
differently — Club Elo drops spaces ("ManUnited"), openfootball appends
"FC", StatsBomb uses full names — so we normalise, apply an explicit alias
table for the awkward cases, and fall back to fuzzy matching against the set
of canonical names actually present in the database.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

_STRIP_TOKENS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "us", "sv", "vfb", "vfl", "fsv", "tsg", "bsc", "rcd", "rc",
    "cd", "ud", "sd", "ca", "club", "de", "futbol", "calcio", "1", "1899", "1846", "1904", "1909", "1913",
    "the",
}

# alias (normalised) -> canonical football-data.co.uk name
ALIASES: dict[str, str] = {
    # England
    "manunited": "Man United", "manchester united": "Man United", "manchester utd": "Man United",
    "mancity": "Man City", "manchester city": "Man City",
    "tottenham hotspur": "Tottenham", "spurs": "Tottenham",
    "wolverhampton wanderers": "Wolves", "wolverhampton": "Wolves",
    "newcastle united": "Newcastle", "newcastle utd": "Newcastle",
    "nottingham forest": "Nott'm Forest", "forest": "Nott'm Forest", "nottm forest": "Nott'm Forest",
    "west ham united": "West Ham", "westham": "West Ham",
    "brighton hove albion": "Brighton", "brighton and hove albion": "Brighton", "brighton & hove albion": "Brighton",
    "sheffield united": "Sheffield United", "sheffield utd": "Sheffield United",
    "leicester city": "Leicester", "leeds united": "Leeds", "norwich city": "Norwich",
    "west bromwich albion": "West Brom", "westbrom": "West Brom", "west bromwich": "West Brom",
    "queens park rangers": "QPR", "hull city": "Hull", "cardiff city": "Cardiff", "swansea city": "Swansea",
    "stoke city": "Stoke", "birmingham city": "Birmingham", "blackburn rovers": "Blackburn",
    "bolton wanderers": "Bolton", "wigan athletic": "Wigan", "huddersfield town": "Huddersfield",
    "luton town": "Luton", "ipswich town": "Ipswich", "afc bournemouth": "Bournemouth",
    "crystal palace": "Crystal Palace", "aston villa": "Aston Villa",
    # Germany
    "bayern munich": "Bayern Munich", "bayern munchen": "Bayern Munich", "bayern": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich",
    "borussia dortmund": "Dortmund", "bayer leverkusen": "Leverkusen", "bayer 04 leverkusen": "Leverkusen",
    "borussia monchengladbach": "M'gladbach", "monchengladbach": "M'gladbach", "gladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt", "frankfurt": "Ein Frankfurt",
    "tsg hoffenheim": "Hoffenheim", "1899 hoffenheim": "Hoffenheim",
    "rb leipzig": "RB Leipzig", "leipzig": "RB Leipzig",
    "vfb stuttgart": "Stuttgart", "vfl wolfsburg": "Wolfsburg", "sc freiburg": "Freiburg",
    "1 fsv mainz 05": "Mainz", "mainz 05": "Mainz", "fsv mainz": "Mainz",
    "1 fc koln": "FC Koln", "koln": "FC Koln", "cologne": "FC Koln", "1 fc union berlin": "Union Berlin",
    "fc augsburg": "Augsburg", "werder bremen": "Werder Bremen", "sv werder bremen": "Werder Bremen",
    "hertha bsc": "Hertha", "hertha berlin": "Hertha", "fc schalke 04": "Schalke 04", "schalke": "Schalke 04",
    "vfl bochum": "Bochum", "1 fc heidenheim": "Heidenheim", "fc st pauli": "St Pauli", "holstein kiel": "Holstein Kiel",
    "hamburger sv": "Hamburg", "hamburg": "Hamburg",
    # Spain
    "real madrid": "Real Madrid", "fc barcelona": "Barcelona", "atletico madrid": "Ath Madrid", "atletico": "Ath Madrid",
    "atletico de madrid": "Ath Madrid", "athletic bilbao": "Ath Bilbao", "athletic club": "Ath Bilbao", "bilbao": "Ath Bilbao",
    "real sociedad": "Sociedad", "sociedad": "Sociedad", "real betis": "Betis", "sevilla fc": "Sevilla",
    "villarreal cf": "Villarreal", "valencia cf": "Valencia", "celta vigo": "Celta", "celta de vigo": "Celta",
    "rcd espanyol": "Espanol", "espanyol": "Espanol", "rcd mallorca": "Mallorca", "getafe cf": "Getafe",
    "ca osasuna": "Osasuna", "deportivo alaves": "Alaves", "rayo vallecano": "Vallecano", "real valladolid": "Valladolid",
    "girona fc": "Girona", "cadiz cf": "Cadiz", "ud las palmas": "Las Palmas", "cd leganes": "Leganes",
    "granada cf": "Granada", "elche cf": "Elche", "levante ud": "Levante", "sd eibar": "Eibar", "sd huesca": "Huesca",
    # Italy
    "inter milan": "Inter", "internazionale": "Inter", "fc internazionale milano": "Inter", "ac milan": "Milan",
    "juventus fc": "Juventus", "ssc napoli": "Napoli", "as roma": "Roma", "ss lazio": "Lazio",
    "atalanta bc": "Atalanta", "acf fiorentina": "Fiorentina", "torino fc": "Torino", "bologna fc": "Bologna",
    "udinese calcio": "Udinese", "us sassuolo": "Sassuolo", "hellas verona": "Verona", "genoa cfc": "Genoa",
    "us lecce": "Lecce", "cagliari calcio": "Cagliari", "empoli fc": "Empoli", "ac monza": "Monza",
    "frosinone calcio": "Frosinone", "us salernitana": "Salernitana", "parma calcio": "Parma", "como 1907": "Como",
    "venezia fc": "Venezia", "spezia calcio": "Spezia", "us cremonese": "Cremonese", "uc sampdoria": "Sampdoria",
    # France
    "paris saint germain": "Paris SG", "paris sg": "Paris SG", "psg": "Paris SG", "paris saint-germain": "Paris SG",
    "olympique marseille": "Marseille", "olympique de marseille": "Marseille", "olympique lyonnais": "Lyon",
    "as monaco": "Monaco", "losc lille": "Lille", "lille osc": "Lille", "stade rennais": "Rennes", "rennes": "Rennes",
    "ogc nice": "Nice", "rc lens": "Lens", "stade de reims": "Reims", "rc strasbourg": "Strasbourg",
    "fc nantes": "Nantes", "montpellier hsc": "Montpellier", "toulouse fc": "Toulouse", "stade brestois": "Brest",
    "brest": "Brest", "fc lorient": "Lorient", "le havre ac": "Le Havre", "fc metz": "Metz", "aj auxerre": "Auxerre",
    "as saint etienne": "St Etienne", "saint etienne": "St Etienne", "angers sco": "Angers", "clermont foot": "Clermont",
    # Netherlands / Portugal / Belgium / Scotland / Turkey
    "afc ajax": "Ajax", "psv eindhoven": "PSV Eindhoven", "psv": "PSV Eindhoven", "feyenoord rotterdam": "Feyenoord",
    "az alkmaar": "AZ Alkmaar", "fc twente": "Twente", "fc utrecht": "Utrecht", "sc heerenveen": "Heerenveen",
    "fc porto": "Porto", "sl benfica": "Benfica", "sporting cp": "Sporting CP", "sporting lisbon": "Sporting CP",
    "sporting braga": "Sp Braga", "sc braga": "Sp Braga", "braga": "Sp Braga", "vitoria guimaraes": "Guimaraes",
    "club brugge": "Club Brugge", "club brugge kv": "Club Brugge", "rsc anderlecht": "Anderlecht",
    "krc genk": "Genk", "royal antwerp": "Antwerp", "union saint gilloise": "St. Gilloise", "union sg": "St. Gilloise",
    "royale union saint-gilloise": "St. Gilloise", "standard liege": "Standard", "kaa gent": "Gent",
    "celtic fc": "Celtic", "rangers fc": "Rangers", "heart of midlothian": "Hearts", "hibernian": "Hibernian",
    "aberdeen fc": "Aberdeen",
    "galatasaray sk": "Galatasaray", "fenerbahce sk": "Fenerbahce", "besiktas jk": "Besiktas", "trabzonspor": "Trabzonspor",
}


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalise(name: str) -> str:
    """Lower-case, accent-free, punctuation-free, club-suffix-free key."""
    text = _strip_accents(str(name)).lower()
    text = text.replace("&", " and ").replace("-", " ").replace(".", "").replace("'", "")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [t for t in text.split() if t not in _STRIP_TOKENS]
    return " ".join(tokens).strip()


class TeamNameResolver:
    def __init__(self, canonical_names: list[str] | set[str] | None = None):
        self.canonical: set[str] = set(canonical_names or [])
        self._norm_to_canonical: dict[str, str] = {normalise(n): n for n in self.canonical}
        self._alias_norm: dict[str, str] = {normalise(k): v for k, v in ALIASES.items()}


    def resolve(self, name: str, cutoff: float = 0.86) -> str | None:
        """Return the canonical name for `name`, or None if no confident match."""
        if name in self.canonical:
            return name
        key = normalise(name)
        if key in self._norm_to_canonical:
            return self._norm_to_canonical[key]
        if key in self._alias_norm:
            alias_target = self._alias_norm[key]
            if not self.canonical or alias_target in self.canonical:
                return alias_target
            alias_key = normalise(alias_target)
            if alias_key in self._norm_to_canonical:
                return self._norm_to_canonical[alias_key]
        if not self.canonical:
            return None
        candidates = difflib.get_close_matches(key, list(self._norm_to_canonical), n=1, cutoff=cutoff)
        if candidates:
            return self._norm_to_canonical[candidates[0]]
        return None
