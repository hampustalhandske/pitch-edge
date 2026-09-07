"""Club Elo connector (clubelo.com / api.clubelo.com).

Free, keyless CSV API. Two endpoints: every club's rating on a given date,
and one club's full rating history. Club Elo asks for reasonable request
rates and non-commercial use; we cache every response and never request the
same date/club twice. Names are Club Elo's compact form ("ManUnited") and go
through `TeamNameResolver`.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient

logger = logging.getLogger(__name__)

BASE_URL = "http://api.clubelo.com"


class ClubEloSource:
    name = "club_elo"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "club_elo"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=2.0)

    def fetch_ratings_by_date(self, date: str, force_refresh: bool = False) -> pd.DataFrame:
        """All club ratings as of `date` (YYYY-MM-DD)."""
        text = self.http.get_text(f"{BASE_URL}/{date}", force=force_refresh)
        df = pd.read_csv(StringIO(text))
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"club": "team"})
        df["as_of"] = pd.Timestamp(date)
        for col in ("from", "to"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

