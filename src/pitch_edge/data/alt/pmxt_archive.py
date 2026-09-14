"""PMXT hourly orderbook archive — historical Polymarket tick data, covering all of soccer.

Two-step pipeline, matching how the archive is actually shaped:

1. **Market discovery** (`discover_soccer_markets_for_day` / `discover_soccer_markets_incremental`):
   Gamma's `/events/keyset` endpoint, filtered by the single umbrella tag `soccer` (confirmed
   live: even an obscure Moroccan Botola League match carries `tags: ['sports', 'games',
   'soccer']` — no per-league tag needed at all; `tag_slug=football` is a red herring, confirmed
   live to mean American football/NFL on this platform, not soccer) and by `end_date_min`/
   `end_date_max` scoped to one calendar day at a time (`endDate` tracks a match's real
   kickoff/resolution; `startDate` is when Polymarket *listed* the market, often ~2 weeks
   earlier, so it's the wrong field to slice by day).

   Querying "every soccer market, all time" in one sweep is what the old (deleted) version of
   this function did, via `tag_slug` + naive `while True` cursor pagination — **confirmed live
   to be broken**: `/events/keyset`'s `next_cursor` starts repeating the identical value (and
   identical page of ~100 events) forever once real pages run out, instead of returning an
   empty page or no cursor. A day-scoped query never hits this: a single day's soccer matches
   are far under the ~100-per-page ceiling, so `next_cursor` comes back empty on page 1 and the
   loop finishes correctly, with real matches from any day in history (not just whatever a
   broken multi-page sweep could reach — confirmed live for both 2026-06-15 and 2026-01-10,
   each returning genuinely different, correctly-dated matches).

   `/markets?tag_slug=...` (what the old, now-deleted `polymarket.py`/`kalshi.py` live
   snapshot connectors used) is broken in a different way: confirmed live that `tag_slug` is
   silently ignored by that endpoint entirely — a bogus tag (`tag_slug=totally-bogus-tag-xyz`)
   returns the exact same "trending markets" listing as `tag_slug=premier-league`. `/events`
   (singular, non-keyset) *does* filter correctly but is deprecated with a sunset date already
   in the past (2026-05-01) per its own response headers — `/events/keyset` is the supported
   replacement and is what's used here.

   `discover_soccer_markets_incremental` walks backward day by day from today (plus a buffer —
   see below) and stops as soon as it reaches a day already recorded in `pmxt_days_scanned`, so
   a first run walks the full history and every run after that only touches the new day(s)
   since the last run — never re-scanning what's already known.

2. **Orderbook ingestion** (`fetch_orderbook_hour`): only real executed trades
   (`event_type = 'last_trade_price'`) are kept — see that method's own docstring for the
   measured 99.8%-was-noise finding behind that choice. PMXT publishes one Parquet file per
   hour at `https://r2v2.pmxt.dev/polymarket_orderbook_<YYYY-MM-DD>T<HH>.parquet`
   (confirmed live: no API key needed for the file itself, ~100-550MB per hour across *all*
   of Polymarket, every market — size varies a lot by hour). DuckDB's `httpfs` extension
   reads it directly over HTTP and filters server-request-side to only the `market` values
   found in step 1 before a single row is materialized into pandas — the `market` column is
   a BLOB whose bytes are the UTF-8 `conditionId` string (confirmed live: `CAST(market AS
   VARCHAR)` reproduces the exact `0x...` string Gamma returns), not something that needs
   hex-decoding.

   **Coverage is NOT one continuous range — it's genuinely gappy, confirmed by direct HTTP
   checks against `r2v2.pmxt.dev` (not the `archive.pmxt.dev` *website*, which serves a
   fake 200-OK HTML shell for literally any path and cannot be used to check existence).**
   Dense hourly files run 2026-04-13T19 through 2026-08-10T00 (confirmed exact boundary:
   `...08-09T23` exists, `...08-10T00` exists, `...08-10T01` does not) — PMXT's own docs
   attribute this class of gap to "ingestion failures," despite also claiming v2 made gaps
   "rare." After that, nothing until three isolated hours on 2026-09-09 (15, 16, 17), then
   nothing again through the date this was last checked (2026-09-13). `pmxt_hour_exists`
   does a cheap HEAD request (no DuckDB/httpfs) before ever attempting the real fetch, so a
   range backfill spanning a gap skips it in milliseconds instead of opening a remote-parquet
   connection per missing hour.

   A predecessor "v1" archive (`r2.pmxt.dev`, distinct host) DOES exist as real extra history —
   confirmed live via direct HTTP probing: substantial (400-600MB/hour) files run from at least
   2026-02-22 through 2026-04-16, overlapping v2's own start. It is NOT a drop-in extension,
   though: its schema is entirely different (5 columns — `timestamp_received`,
   `timestamp_created_at`, `market_id`, `update_type`, and a nested JSON `data` field — not v2's
   flat `condition_id`/`asset_id`/`price`/`size`), and every sampled hour across Feb/March/April
   shows only two `update_type` values, `price_change` (a bid/ask quote update) and
   `book_snapshot` — never a real trade print. Using v1 to extend history would mean pricing off
   the bid/ask midpoint as a proxy, not a real executed trade, and needs its own parser; not yet
   built.

Nothing here is traded; this is historical/near-historical read-only market data.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
ARCHIVE_URL_TEMPLATE = "https://r2v2.pmxt.dev/polymarket_orderbook_{date}T{hour:02d}.parquet"
SOCCER_TAG = "soccer"
# How far past "today" to look for matches whose end_date is beyond the archive's own newest
# hour but which could still have real pre-match ticks inside it (see module docstring).
FUTURE_BUFFER_DAYS = 14


class PMXTArchiveSource:
    name = "pmxt_archive"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "pmxt_archive"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=0.5)

    # ------------------------------------------------------------- discovery
    def discover_soccer_markets_for_day(self, day: date, page_limit: int = 100) -> pd.DataFrame:
        """Every soccer market whose `end_date` (real kickoff/resolution, not listing date)
        falls on `day`, keyed by `condition_id`. A single day is always far under the ~100
        events/page ceiling, so this never hits the stuck-cursor bug real full-history sweeps
        do (see module docstring) — `next_cursor` should come back empty after page 1, but the
        same repeat-guard is kept here anyway as a cheap safety net."""
        day_str = day.isoformat()
        next_day_str = (day + timedelta(days=1)).isoformat()
        rows: dict[str, dict] = {}
        for closed in (False, True):
            cursor: str | None = None
            seen_cursors: set[str] = set()
            page = 0
            while True:
                page += 1
                params = {
                    "tag_slug": SOCCER_TAG,
                    "closed": str(closed).lower(),
                    "end_date_min": day_str,
                    "end_date_max": next_day_str,
                    "limit": page_limit,
                }
                if cursor:
                    params["cursor"] = cursor
                payload = self.http.get_json(f"{GAMMA_URL}/events/keyset", params=params, cache=False)
                events = payload.get("events") or []
                if not events:
                    break
                for event in events:
                    for m in event.get("markets") or []:
                        condition_id = m.get("conditionId")
                        if not condition_id:
                            continue
                        rows[condition_id] = {
                            "condition_id": condition_id,
                            "question": m.get("question"),
                            "event_title": event.get("title"),
                            "group_item_title": m.get("groupItemTitle"),
                            "slug": m.get("slug"),
                            "closed": bool(m.get("closed")),
                            "end_date": pd.to_datetime(m.get("endDate"), errors="coerce", utc=True),
                            "clob_token_ids": json.dumps(_as_list(m.get("clobTokenIds"))),
                            "discovered_day": day_str,
                        }
                next_cursor = payload.get("next_cursor")
                if not next_cursor or next_cursor in seen_cursors:
                    break
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            logger.info(
                "discover_soccer_markets_for_day: %s closed=%s pages=%d markets_so_far=%d",
                day_str, closed, page, len(rows),
            )
        return pd.DataFrame(rows.values())

    # ------------------------------------------------------------ ingestion
    def fetch_orderbook_hour(self, date: str, hour: int, condition_ids: list[str]) -> pd.DataFrame:
        """One hour of PMXT **real executed trades** (`event_type = 'last_trade_price'`) for
        `condition_ids` (soccer only).

        Only `last_trade_price` rows are kept — confirmed live that this is where the actual
        signal is: for one real, busy market this event type was 780 rows/day of genuine trades
        (irregular timing, non-round trade sizes — a real buyer and seller agreeing on a price),
        versus 356,332 rows/day of `price_change` (repeated re-quoting at only ~149 distinct
        prices — almost entirely market-maker bot noise, confirmed by checking distinct price
        counts) and `book`/`tick_size_change` rows whose price/size/best_bid/best_ask columns
        are entirely NULL for every row (their real content — full order-book depth — was
        already excluded on purpose; see the module docstring). Filtering `event_type` here,
        inside the same DuckDB query that filters `condition_ids`, cut one busy market's daily
        row count by ~450x with zero loss of real trading signal.

        Runs the filter inside DuckDB against the remote Parquet file over `httpfs` — only the
        matching rows (and only the columns selected below) are ever materialized into pandas,
        never the full ~16M-row hour.
        """
        if not condition_ids:
            return pd.DataFrame()
        url = ARCHIVE_URL_TEMPLATE.format(date=date, hour=hour)
        if not pmxt_hour_exists(url, self.http):
            return pd.DataFrame()
        con = duckdb.connect()
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
            con.register("condition_ids", pd.DataFrame({"condition_id": condition_ids}))
            df = con.execute(
                f"""
                SELECT
                    CAST(market AS VARCHAR) AS condition_id,
                    asset_id,
                    timestamp,
                    timestamp_received,
                    side,
                    price,
                    size
                FROM read_parquet('{url}') AS ob
                WHERE event_type = 'last_trade_price'
                  AND CAST(ob.market AS VARCHAR) IN (SELECT condition_id FROM condition_ids)
                """
            ).df()
        except duckdb.IOException as exc:
            logger.info("PMXT archive hour %sT%02d unavailable: %s", date, hour, exc)
            return pd.DataFrame()
        finally:
            con.close()
        df["source_hour"] = f"{date}T{hour:02d}"
        return df


def pmxt_hour_exists(url: str, http: CachedHttpClient) -> bool:
    """Cheap existence probe (a plain HEAD, no DuckDB/httpfs) — the archive is genuinely gappy
    (see module docstring), so checking first means a range backfill skips a missing hour in
    milliseconds instead of opening a remote-parquet connection that's doomed to fail."""
    try:
        resp = http.session.head(url, timeout=10)
        return resp.status_code == 200
    except Exception as exc:  # noqa: BLE001 - a network blip here should not abort a backfill
        logger.info("pmxt_hour_exists: HEAD failed for %s (%s); treating as absent", url, exc)
        return False


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []
