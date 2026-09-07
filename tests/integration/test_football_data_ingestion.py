"""Connector -> odds melt -> warehouse -> wide join, with HTTP mocked (offline, deterministic)."""

from __future__ import annotations

import pytest
import responses

from pitch_edge.data.ingest import store_matches
from pitch_edge.data.sources.football_data_co_uk import FootballDataCoUkSource

pytestmark = pytest.mark.integration


@responses.activate
def test_store_matches_persists_core_and_odds(tmp_path, warehouse, football_data_csv_bytes):
    responses.add(responses.GET, "https://www.football-data.co.uk/mmz4281/2324/E0.csv", body=football_data_csv_bytes)
    df = FootballDataCoUkSource(cache_dir=tmp_path).fetch_matches("E0", [2023])
    assert store_matches(warehouse, df) == 4
    assert store_matches(warehouse, df) == 0  # idempotent
    assert warehouse.count("odds") == 4 * (3 + 3 + 3 + 2)  # B365, PS, PSC (1x2) + B365 over/under per match
    wide = warehouse.matches_with_closing_odds()
    row = wide.set_index("match_id").loc["fd_E0_2324_20230812_Arsenal_ManUnited"]
    assert row["PSH"] == 1.85 and row["PSCH"] == 1.82 and row["referee"] == "M Oliver"


@responses.activate
def test_cache_prevents_refetch(tmp_path, football_data_csv_bytes):
    responses.add(responses.GET, "https://www.football-data.co.uk/mmz4281/2324/E0.csv", body=football_data_csv_bytes)
    src = FootballDataCoUkSource(cache_dir=tmp_path)
    src.fetch_matches("E0", [2023])
    src.fetch_matches("E0", [2023])
    assert len(responses.calls) == 1
