"""PMXT ingestion orchestration in `data/ingest.py`: order and skip behaviour."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from pitch_edge.data import ingest as ing
from pitch_edge.data.storage import Warehouse

pytestmark = pytest.mark.unit


def _fake_hour_frame(date: str, hour: int, condition_id: str = "0xaaa") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "condition_id": condition_id,
                "asset_id": "111",
                "timestamp": pd.Timestamp.utcnow(),
                "timestamp_received": pd.Timestamp.utcnow(),
                "side": "BUY",
                "price": 0.5,
                "size": 1.0,
                "source_hour": f"{date}T{hour:02d}",
            }
        ]
    )


def test_ingest_pmxt_orderbook_range_walks_newest_first(tmp_path, monkeypatch):
    wh = Warehouse(tmp_path / "wh.duckdb")
    wh.upsert("dim_soccer_markets", pd.DataFrame([{"condition_id": "0xaaa"}]))

    calls: list[str] = []

    class FakeSrc:
        def fetch_orderbook_hour(self, date, hour, ids):
            calls.append(f"{date}T{hour:02d}")
            return _fake_hour_frame(date, hour)

    monkeypatch.setattr("pitch_edge.data.alt.pmxt_archive.PMXTArchiveSource", FakeSrc)

    ing.ingest_pmxt_orderbook_range(wh, datetime(2026, 4, 13, 19), datetime(2026, 4, 13, 22))

    # newest hour first, oldest last — matches discovery's walk-backward order
    assert calls == ["2026-04-13T22", "2026-04-13T21", "2026-04-13T20", "2026-04-13T19"]
    wh.close()


def test_ingest_pmxt_orderbook_range_skips_already_ingested_hours(tmp_path, monkeypatch):
    wh = Warehouse(tmp_path / "wh.duckdb")
    wh.upsert("dim_soccer_markets", pd.DataFrame([{"condition_id": "0xaaa"}]))

    calls: list[str] = []

    class FakeSrc:
        def fetch_orderbook_hour(self, date, hour, ids):
            calls.append(f"{date}T{hour:02d}")
            return _fake_hour_frame(date, hour)

    monkeypatch.setattr("pitch_edge.data.alt.pmxt_archive.PMXTArchiveSource", FakeSrc)

    # First pass ingests hour 20 only.
    ing.ingest_pmxt_orderbook_hour(wh, "2026-04-13", 20)
    assert calls == ["2026-04-13T20"]

    # A range spanning it again must not re-fetch that hour.
    calls.clear()
    ing.ingest_pmxt_orderbook_range(wh, datetime(2026, 4, 13, 19), datetime(2026, 4, 13, 21))
    assert calls == ["2026-04-13T21", "2026-04-13T19"]
    wh.close()
