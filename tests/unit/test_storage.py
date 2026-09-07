from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.data.storage import Warehouse

pytestmark = pytest.mark.unit


def _matches(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "date": pd.date_range("2023-08-01", periods=n),
            "league": "L",
            "league_code": "L0",
            "season": "2023/24",
            "home_team": [f"H{i}" for i in range(n)],
            "away_team": [f"A{i}" for i in range(n)],
            "home_goals": 1.0,
            "away_goals": 0.0,
            "source": "test",
        }
    )


def test_upsert_creates_table_and_is_idempotent(warehouse):
    assert warehouse.upsert("matches", _matches()) == 3
    assert warehouse.upsert("matches", _matches()) == 0
    assert warehouse.count("matches") == 3


def test_upsert_widens_schema_with_new_columns(warehouse):
    warehouse.upsert("matches", _matches())
    wider = _matches(5).assign(referee="R")
    assert warehouse.upsert("matches", wider) == 2
    assert "referee" in warehouse.columns("matches")
    df = warehouse.read("matches")
    assert df["referee"].notna().sum() == 2


def test_upsert_requires_key_columns(warehouse):
    with pytest.raises(ValueError, match="missing key columns"):
        warehouse.upsert("matches", pd.DataFrame({"foo": [1]}))


def test_upsert_unknown_table_needs_keys(warehouse):
    with pytest.raises(ValueError, match="No natural key"):
        warehouse.upsert("mystery", pd.DataFrame({"a": [1]}))
    assert warehouse.upsert("mystery", pd.DataFrame({"a": [1]}), keys=["a"]) == 1


def test_replace_and_read_where(warehouse):
    warehouse.replace("features", pd.DataFrame({"match_id": ["a", "b"], "x": [1.0, 2.0]}))
    assert warehouse.count("features") == 2
    out = warehouse.read("features", "x > ?", [1.5])
    assert out["match_id"].tolist() == ["b"]


def test_odds_long_join_produces_wide_columns(warehouse):
    warehouse.upsert("matches", _matches(1))
    odds = pd.DataFrame(
        {
            "match_id": ["m0"] * 6,
            "bookmaker": ["PS"] * 6,
            "market": ["1x2"] * 6,
            "side": ["home", "draw", "away"] * 2,
            "price": [1.9, 3.5, 4.2, 1.85, 3.6, 4.4],
            "is_closing": [False] * 3 + [True] * 3,
            "snapshot_ts": pd.Timestamp("2023-08-01"),
        }
    )
    warehouse.upsert("odds", odds)
    wide = warehouse.matches_with_closing_odds()
    assert wide.loc[0, "PSH"] == 1.9 and wide.loc[0, "PSCH"] == 1.85


def test_health_and_log_run(warehouse):
    warehouse.log_run("src_a", "ok", rows=10)
    warehouse.log_run("src_a", "error", error="boom")
    warehouse.log_run("src_b", "ok", rows=3)
    h = warehouse.health().set_index("source")
    assert h.loc["src_a", "errors"] == 1
    assert h.loc["src_a", "last_status"] == "error"
    assert pd.notna(h.loc["src_a", "last_success"])


def test_export_parquet(warehouse, tmp_path):
    warehouse.upsert("matches", _matches())
    paths = warehouse.export_parquet(tmp_path / "lake")
    assert any(p.name == "matches.parquet" for p in paths)
    assert len(pd.read_parquet(tmp_path / "lake" / "matches.parquet")) == 3


def test_reopen_persists(tmp_path):
    p = tmp_path / "x.duckdb"
    with Warehouse(p) as wh:
        wh.upsert("matches", _matches())
    with Warehouse(p) as wh:
        assert wh.count("matches") == 3
