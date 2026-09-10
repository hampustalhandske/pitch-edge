from __future__ import annotations

import pytest
import responses

from pitch_edge.data.sources.football_data_co_uk import (
    FootballDataCoUkSource,
    odds_wide_to_long,
    season_code,
    season_label,
)

pytestmark = pytest.mark.unit


def test_season_codes():
    assert season_code(2023) == "2324" and season_code(2009) == "0910"
    assert season_label(2023) == "2023/24"


@responses.activate
def test_main_file_parsing_keeps_stats_and_both_odds_families(tmp_path, football_data_csv_bytes):
    responses.add(responses.GET, "https://www.football-data.co.uk/mmz4281/2324/E0.csv", body=football_data_csv_bytes)
    df = FootballDataCoUkSource(cache_dir=tmp_path).fetch_matches("E0", [2023])
    assert len(df) == 4
    assert df["league"].iloc[0] == "England - Premier League" and df["league_code"].iloc[0] == "E0"
    for col in ("referee", "home_shots", "away_yellows", "kickoff_time", "PSH", "PSCH", "B365>2.5"):
        assert col in df.columns
    assert df["match_id"].str.startswith("fd_E0_2324_").all()
    assert df["match_id"].is_unique


@responses.activate
def test_missing_season_is_skipped_not_fatal(tmp_path, football_data_csv_bytes):
    responses.add(responses.GET, "https://www.football-data.co.uk/mmz4281/2324/E0.csv", body=football_data_csv_bytes)
    responses.add(responses.GET, "https://www.football-data.co.uk/mmz4281/2425/E0.csv", status=404)
    df = FootballDataCoUkSource(cache_dir=tmp_path).fetch_matches("E0", [2023, 2024])
    assert len(df) == 4


def test_unknown_league_code_rejected(tmp_path):
    with pytest.raises(ValueError):
        FootballDataCoUkSource(cache_dir=tmp_path).fetch_matches("XX", [2023])


@responses.activate
def test_extra_league_file_parsing(tmp_path, extra_league_csv_bytes):
    responses.add(responses.GET, "https://www.football-data.co.uk/new/SWE.csv", body=extra_league_csv_bytes)
    df = FootballDataCoUkSource(cache_dir=tmp_path).fetch_extra_league("SWE")
    assert len(df) == 2 and df["country"].iloc[0] == "Sweden" and df["league"].iloc[0] == "Sweden - Allsvenskan"
    assert {"PSCH", "BFECH", "AvgCA"} <= set(df.columns)


def test_archive_main_file_used_instead_of_network(tmp_path, football_data_csv_bytes, monkeypatch):
    archive_dir = tmp_path / "archive"
    season_dir = archive_dir / "main" / "main" / "23_24"
    season_dir.mkdir(parents=True)
    (season_dir / "E0.csv").write_bytes(football_data_csv_bytes)

    class ExplodingHttp:
        def get_bytes(self, *args, **kwargs):
            raise AssertionError("network path should not be called when archive file exists")

    src = FootballDataCoUkSource(cache_dir=tmp_path / "cache", http=ExplodingHttp(), archive_dir=archive_dir)
    df = src.fetch_matches("E0", [2023])
    assert len(df) == 4
    assert df["match_id"].str.startswith("fd_E0_2324_").all()


def test_archive_extra_file_used_instead_of_network(tmp_path, extra_league_csv_bytes):
    archive_dir = tmp_path / "archive"
    extra_dir = archive_dir / "extra"
    extra_dir.mkdir(parents=True)
    (extra_dir / "SWE.csv").write_bytes(extra_league_csv_bytes)

    class ExplodingHttp:
        def get_bytes(self, *args, **kwargs):
            raise AssertionError("network path should not be called when archive file exists")

    src = FootballDataCoUkSource(cache_dir=tmp_path / "cache", http=ExplodingHttp(), archive_dir=archive_dir)
    df = src.fetch_extra_league("SWE")
    assert len(df) == 2 and df["country"].iloc[0] == "Sweden"


@responses.activate
def test_odds_wide_to_long_marks_closing_and_totals(tmp_path, football_data_csv_bytes):
    responses.add(responses.GET, "https://www.football-data.co.uk/mmz4281/2324/E0.csv", body=football_data_csv_bytes)
    df = FootballDataCoUkSource(cache_dir=tmp_path).fetch_matches("E0", [2023])
    long = odds_wide_to_long(df)
    ps = long[(long["bookmaker"] == "PS") & (long["match_id"] == df["match_id"].iloc[0])]
    assert ps["is_closing"].sum() == 3 and (~ps["is_closing"]).sum() == 3
    totals = long[long["market"] == "totals_2.5"]
    assert set(totals["side"]) == {"over", "under"}
    assert (long["price"] > 1).all()
