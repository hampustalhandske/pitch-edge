"""PMXT Polymarket archive connector: discovery (mocked Gamma) and hour ingestion wiring."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import responses

from pitch_edge.data.alt.pmxt_archive import GAMMA_URL, PMXTArchiveSource

pytestmark = pytest.mark.unit


@responses.activate
def test_discover_soccer_markets_for_day_dedupes_across_closed_states(tmp_path):
    page_open = {
        "events": [
            {
                "title": "Arsenal FC vs. Chelsea FC",
                "markets": [
                    {
                        "conditionId": "0xaaa",
                        "question": "Will Arsenal FC win?",
                        "groupItemTitle": "Arsenal",
                        "slug": "arsenal-v-chelsea",
                        "closed": False,
                        "endDate": "2026-10-01T00:00:00Z",
                        "clobTokenIds": '["111", "222"]',
                    }
                ],
            }
        ]
    }
    page_closed = {"events": []}
    responses.add(responses.GET, f"{GAMMA_URL}/events/keyset", json=page_open)
    responses.add(responses.GET, f"{GAMMA_URL}/events/keyset", json=page_closed)

    src = PMXTArchiveSource(cache_dir=tmp_path)
    df = src.discover_soccer_markets_for_day(date(2026, 10, 1))

    assert len(df) == 1
    assert df.iloc[0]["condition_id"] == "0xaaa"
    assert df.iloc[0]["clob_token_ids"] == '["111", "222"]'
    assert df.iloc[0]["discovered_day"] == "2026-10-01"
    # both closed=false and closed=true must have been queried
    assert len(responses.calls) == 2
    assert responses.calls[0].request.params["closed"] == "false"
    assert responses.calls[1].request.params["closed"] == "true"
    # scoped to the single day, not open-ended
    assert responses.calls[0].request.params["end_date_min"] == "2026-10-01"
    assert responses.calls[0].request.params["end_date_max"] == "2026-10-02"


@responses.activate
def test_discover_soccer_markets_for_day_stops_on_repeated_cursor(tmp_path):
    """Confirmed live: Gamma's `/events/keyset` can return the same `next_cursor` (and the same
    page) forever instead of signalling end-of-results — the loop must not hang."""
    looping_page = {
        "events": [{"title": "X", "markets": [{"conditionId": "0xzzz", "clobTokenIds": "[]"}]}],
        "next_cursor": "stuck-cursor",
    }
    responses.add(responses.GET, f"{GAMMA_URL}/events/keyset", json=looping_page)
    responses.add(responses.GET, f"{GAMMA_URL}/events/keyset", json=looping_page)
    responses.add(responses.GET, f"{GAMMA_URL}/events/keyset", json={"events": []})

    src = PMXTArchiveSource(cache_dir=tmp_path)
    df = src.discover_soccer_markets_for_day(date(2026, 10, 1))

    assert len(df) == 1
    # closed=false: page 1, then page 2 confirms the cursor repeated -> stop (2 calls);
    # closed=true: one more call (1 call) -> 3 total, never an infinite loop.
    assert len(responses.calls) == 3


def test_fetch_orderbook_hour_empty_condition_ids_short_circuits(tmp_path):
    src = PMXTArchiveSource(cache_dir=tmp_path)
    df = src.fetch_orderbook_hour("2026-09-09", 17, [])
    assert df.empty


def test_fetch_orderbook_hour_filters_by_condition_id_and_trades_only(tmp_path, monkeypatch):
    """`fetch_orderbook_hour` builds the archive URL from date/hour, filters to the given
    condition_ids, AND keeps only real executed trades (`event_type = 'last_trade_price'`) —
    `price_change`/`book`/`tick_size_change` rows (confirmed live to be ~99.8% market-maker
    quote noise, not real trades) must never appear in the result, even for a wanted market.
    Verified against a local Parquet file standing in for the remote archive (real remote
    reachability is exercised manually, not in the mocked unit suite)."""
    local_parquet = tmp_path / "hour.parquet"
    raw = pd.DataFrame(
        {
            "market": [b"0xaaa", b"0xaaa", b"0xaaa", b"0xbbb"],
            "asset_id": ["111", "111", "111", "999"],
            "timestamp": pd.to_datetime(["2026-09-09T17:00:00Z"] * 4),
            "timestamp_received": pd.to_datetime(["2026-09-09T17:00:00Z"] * 4),
            "event_type": ["last_trade_price", "price_change", "book", "last_trade_price"],
            "side": ["BUY", "SELL", None, "SELL"],
            "price": [0.55, 0.60, None, 0.10],
            "size": [10.0, 20.0, None, 5.0],
        }
    )
    raw.to_parquet(local_parquet)

    import pitch_edge.data.alt.pmxt_archive as pa

    # Template has no `{date}`/`{hour}` placeholders, so `.format()` just returns it unchanged —
    # points fetch_orderbook_hour at the local stand-in file instead of the real remote archive.
    monkeypatch.setattr(pa, "ARCHIVE_URL_TEMPLATE", str(local_parquet))
    # The archive is genuinely gappy (see module docstring), so a real HEAD-existence check
    # gates every fetch — stub it here since there's nothing at an http(s) URL to HEAD in this test.
    monkeypatch.setattr(pa, "pmxt_hour_exists", lambda url, http: True)

    src = PMXTArchiveSource(cache_dir=tmp_path)
    # Ask for both markets — 0xbbb's only row is a real trade but should still be excluded
    # because it's not in condition_ids, proving the two filters are independent (AND, not OR).
    df = src.fetch_orderbook_hour("2026-09-09", 17, ["0xaaa"])
    assert len(df) == 1
    assert df.iloc[0]["condition_id"] == "0xaaa"
    assert df.iloc[0]["price"] == 0.55
    assert df["source_hour"].iloc[0] == "2026-09-09T17"
    assert "event_type" not in df.columns
    assert "best_bid" not in df.columns


def test_fetch_orderbook_hour_skips_missing_hour_without_duckdb(tmp_path, monkeypatch):
    """A gap hour (HEAD returns non-200) must short-circuit before ever touching DuckDB/httpfs —
    the whole point of the existence check is never opening a remote-parquet connection that's
    doomed to 404 (the archive has a real multi-week gap, confirmed live)."""
    import pitch_edge.data.alt.pmxt_archive as pa

    monkeypatch.setattr(pa, "pmxt_hour_exists", lambda url, http: False)

    def _boom(*a, **k):
        raise AssertionError("duckdb.connect should never be called for a missing hour")

    monkeypatch.setattr(pa.duckdb, "connect", _boom)

    src = PMXTArchiveSource(cache_dir=tmp_path)
    df = src.fetch_orderbook_hour("2026-08-10", 1, ["0xaaa"])
    assert df.empty


@responses.activate
def test_pmxt_hour_exists_checks_head_status(tmp_path):
    from pitch_edge.data.alt.pmxt_archive import ARCHIVE_URL_TEMPLATE, pmxt_hour_exists

    ok_url = ARCHIVE_URL_TEMPLATE.format(date="2026-08-09", hour=23)
    gap_url = ARCHIVE_URL_TEMPLATE.format(date="2026-08-10", hour=1)
    responses.add(responses.HEAD, ok_url, status=200)
    responses.add(responses.HEAD, gap_url, status=404)

    src = PMXTArchiveSource(cache_dir=tmp_path)
    assert pmxt_hour_exists(ok_url, src.http) is True
    assert pmxt_hour_exists(gap_url, src.http) is False
