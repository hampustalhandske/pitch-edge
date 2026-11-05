from __future__ import annotations

import json
from datetime import UTC, datetime
from datetime import date as Day

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy.stats import poisson

from pitch_edge.data.storage import Warehouse

# NiceGUI's browser-less `user` fixture (tests/integration/test_dashboard.py) — a pure ASGI
# simulation (httpx + starlette transport), no real browser/websocket, so it stays fast and
# network-free like the rest of the suite. Must be declared in the top-level conftest.
pytest_plugins = ["nicegui.testing.user_plugin"]


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Every test gets its own PITCH_EDGE_DATA_DIR so nothing touches the real warehouse."""
    monkeypatch.setenv("PITCH_EDGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PITCH_EDGE_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("PITCH_EDGE_HTTP_MIN_INTERVAL", "0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    yield


@pytest.fixture
def small_match_history() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    rows = []
    for date in pd.date_range("2023-08-01", periods=24, freq="7D"):
        home, away = rng.choice(teams, size=2, replace=False)
        rows.append(
            {
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_goals": int(rng.poisson(1.5)),
                "away_goals": int(rng.poisson(1.1)),
            }
        )
    return pd.DataFrame(rows)


def _outcome_probs(lam_h: float, lam_a: float, max_goals: int = 10) -> tuple[float, float, float]:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
    m /= m.sum()
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def make_synthetic_league(
    n_teams: int = 12, n_seasons: int = 4, seed: int = 42, league_code: str = "SYN"
) -> pd.DataFrame:
    """Round-robin seasons with true strengths, shots/fouls/cards stats, referee, and
    Pinnacle early (PS*) + closing (PSC*) + B365 odds derived from the true probabilities
    with noise, so CLV is meaningful."""
    rng = np.random.default_rng(seed)
    teams = [f"Team{i:02d}" for i in range(n_teams)]
    strength = {t: rng.normal(0, 0.35) for t in teams}
    refs = [f"Ref {c}" for c in "ABCDEF"]
    rows = []
    date = pd.Timestamp("2018-08-04")
    mid = 0
    for s in range(n_seasons):
        season = f"{2018 + s}/{(2019 + s) % 100:02d}"
        pairs = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(pairs)
        for k, (h, a) in enumerate(pairs):
            if k % (n_teams // 2) == 0:
                date += pd.Timedelta(days=7)
            lam_h = np.exp(0.25 + strength[h] - 0.6 * strength[a])
            lam_a = np.exp(strength[a] - 0.6 * strength[h])
            hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
            ph, pd_, pa = _outcome_probs(lam_h, lam_a)
            noise = rng.normal(0, 0.02, 3)
            early = np.clip(np.array([ph, pd_, pa]) + noise, 0.04, 0.9)
            early /= early.sum()
            closing = np.clip(np.array([ph, pd_, pa]) + rng.normal(0, 0.008, 3), 0.04, 0.9)
            closing /= closing.sum()
            hs, as_ = int(rng.poisson(9 + 4 * lam_h)), int(rng.poisson(8 + 4 * lam_a))
            rows.append(
                {
                    "match_id": f"syn_{mid}",
                    "date": date,
                    "country": "Synthland",
                    "league": "Synthland - Premier",
                    "league_code": league_code,
                    "season": season,
                    "home_team": h,
                    "away_team": a,
                    "home_goals": float(hg),
                    "away_goals": float(ag),
                    "source": "synthetic",
                    "referee": refs[mid % len(refs)],
                    "home_shots": hs,
                    "away_shots": as_,
                    "home_shots_on_target": int(hs * 0.35),
                    "away_shots_on_target": int(as_ * 0.35),
                    "home_fouls": int(rng.poisson(10)),
                    "away_fouls": int(rng.poisson(12)),
                    "home_corners": int(rng.poisson(5)),
                    "away_corners": int(rng.poisson(4)),
                    "home_yellows": int(rng.poisson(1.5)),
                    "away_yellows": int(rng.poisson(2.0)),
                    "home_reds": 0,
                    "away_reds": int(rng.random() < 0.05),
                    "PSH": round(1 / (early[0] * 1.03), 2),
                    "PSD": round(1 / (early[1] * 1.03), 2),
                    "PSA": round(1 / (early[2] * 1.03), 2),
                    "PSCH": round(1 / (closing[0] * 1.025), 2),
                    "PSCD": round(1 / (closing[1] * 1.025), 2),
                    "PSCA": round(1 / (closing[2] * 1.025), 2),
                    "B365H": round(1 / (early[0] * 1.06), 2),
                    "B365D": round(1 / (early[1] * 1.06), 2),
                    "B365A": round(1 / (early[2] * 1.06), 2),
                }
            )
            mid += 1
        date += pd.Timedelta(days=60)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_league_matches() -> pd.DataFrame:
    return make_synthetic_league()


@pytest.fixture
def warehouse(tmp_path) -> Warehouse:
    wh = Warehouse(tmp_path / "wh.duckdb")
    yield wh
    wh.close()


_E0_STATS = ("ht_home_goals", "ht_away_goals", "home_shots", "away_shots", "home_shots_on_target",
             "away_shots_on_target", "home_fouls", "away_fouls", "home_corners", "away_corners",
             "home_yellow", "away_yellow", "home_red", "away_red")  # fmt: skip
# date, kickoff, home, away, FT goals, referee, stats (in _E0_STATS order), odds_all columns
_E0_FIXTURES = [
    (Day(2023, 8, 12), "20:00", "Arsenal", "Man United", (2, 1), "M Oliver", (1, 0, 15, 9, 6, 3, 10, 12, 7, 3, 1, 2, 0, 0),
     {"B365H": 1.80, "B365D": 3.60, "B365A": 4.50, "PSH": 1.85, "PSD": 3.70, "PSA": 4.40, "B365>2.5": 1.70,
      "B365<2.5": 2.20, "PSCH": 1.82, "PSCD": 3.75, "PSCA": 4.55}),
    (Day(2023, 8, 13), "14:00", "Chelsea", "Liverpool", (1, 1), "A Taylor", (0, 1, 12, 14, 4, 5, 9, 11, 5, 6, 2, 1, 0, 0),
     {"B365H": 2.90, "B365D": 3.30, "B365A": 2.50, "PSH": 2.95, "PSD": 3.35, "PSA": 2.45, "B365>2.5": 1.75,
      "B365<2.5": 2.10, "PSCH": 2.90, "PSCD": 3.40, "PSCA": 2.50}),
    (Day(2023, 8, 19), "15:00", "Man United", "Chelsea", (0, 2), "P Tierney", (0, 1, 10, 13, 3, 6, 13, 8, 4, 5, 3, 1, 0, 0),
     {"B365H": 2.20, "B365D": 3.40, "B365A": 3.30, "PSH": 2.25, "PSD": 3.45, "PSA": 3.20, "B365>2.5": 1.80,
      "B365<2.5": 2.05, "PSCH": 2.30, "PSCD": 3.40, "PSCA": 3.15}),
    (Day(2023, 8, 20), "16:30", "Liverpool", "Arsenal", (3, 1), "M Oliver", (2, 0, 18, 8, 8, 3, 8, 14, 9, 2, 0, 2, 0, 1),
     {"B365H": 1.95, "B365D": 3.60, "B365A": 3.90, "PSH": 1.90, "PSD": 3.65, "PSA": 4.00, "B365>2.5": 1.60,
      "B365<2.5": 2.40, "PSCH": 1.88, "PSCD": 3.70, "PSCA": 4.10}),
]  # fmt: skip
_SWE_FIXTURES = [
    (Day(2023, 4, 1), "Malmo FF", "AIK", (2, 0),
     {"PSCH": 1.8, "PSCD": 3.6, "PSCA": 4.6, "MaxCH": 1.9, "MaxCD": 3.8, "MaxCA": 4.9, "AvgCH": 1.78, "AvgCD": 3.5,
      "AvgCA": 4.4, "BFECH": 1.88, "BFECD": 3.7, "BFECA": 4.7}),
    (Day(2023, 4, 2), "Hammarby", "Djurgarden", (1, 1),
     {"PSCH": 2.4, "PSCD": 3.3, "PSCA": 3.0, "MaxCH": 2.5, "MaxCD": 3.4, "MaxCA": 3.1, "AvgCH": 2.35, "AvgCD": 3.25,
      "AvgCA": 2.95, "BFECH": 2.45, "BFECD": 3.35, "BFECA": 3.05}),
]  # fmt: skip


@pytest.fixture
def lake_fd_rows() -> list[dict]:
    """Lake `fd_matches` rows (the lake's column names) for four 2023/24 Premier League matches and two
    Allsvenskan matches: the football-data CSV fixtures of the old ingestion, as the lake serves them."""
    blank = {"ht_result": None, "ft_result": None, "source_file": "fixture", "scraped_at": None, "attendance": None}
    rows: list[dict] = []
    for day, ko, home, away, (hg, ag), ref, stats, odds in _E0_FIXTURES:
        rows.append(
            {"division": "E0", "country": "England", "league": "Premier League", "season": "2023-2024",
             "match_date": day, "kickoff_time": ko, "home_team": home, "away_team": away,
             "ft_home_goals": hg, "ft_away_goals": ag, "referee": ref, **dict(zip(_E0_STATS, stats, strict=True)),
             "odds_all": json.dumps(odds), **blank}
        )  # fmt: skip
    for day, home, away, (hg, ag), odds in _SWE_FIXTURES:
        rows.append(
            {"division": "SWE", "country": "Sweden", "league": "Allsvenskan", "season": "2023",
             "match_date": day, "kickoff_time": "15:00", "home_team": home, "away_team": away,
             "ft_home_goals": hg, "ft_away_goals": ag, "referee": None, **dict.fromkeys(_E0_STATS),
             "odds_all": json.dumps(odds), **blank}
        )  # fmt: skip
    for r in rows:  # the mirror query's derived known-at column: match date 00:00 UTC
        d = r["match_date"]
        r["open_known_by"] = datetime(d.year, d.month, d.day, tzinfo=UTC)
    return rows


class FakeFdLake:
    """Stands in for `Lake`: answers the `fd_matches` mirror query with canned rows inside the requested
    `[start, end)` window, and the freshness aggregate with the per-division newest completed date."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.calls: list[tuple[str, list]] = []
        self.version = "fake"

    def __enter__(self) -> FakeFdLake:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def sql_batches(self, query, params=None):
        self.calls.append((query, params or []))
        lo, hi = Day.fromisoformat(params[0]), Day.fromisoformat(params[1])
        keep = [r for r in self.rows if lo <= r["match_date"] < hi]
        if keep:
            yield pa.RecordBatch.from_pylist(keep)

    def sql(self, query, params=None):
        self.calls.append((query, params or []))
        newest: dict[str, Day] = {}
        for r in self.rows:
            if r["division"] in (params or []) and r["ft_home_goals"] is not None:
                newest[r["division"]] = max(newest.get(r["division"], r["match_date"]), r["match_date"])
        return pd.DataFrame({"division": list(newest), "d": [pd.Timestamp(v) for v in newest.values()]})


@pytest.fixture
def fd_lake(lake_fd_rows) -> FakeFdLake:
    return FakeFdLake(lake_fd_rows)


@pytest.fixture
def statsbomb_events_df() -> pd.DataFrame:
    """Two tiny teams with a handful of passes and shots — enough for graphs and in-play states."""
    rng = np.random.default_rng(1)
    rows = []
    players = {
        "Home FC": [(1, "H One"), (2, "H Two"), (3, "H Three")],
        "Away FC": [(11, "A One"), (12, "A Two"), (13, "A Three")],
    }
    idx = 0
    for team, pl in players.items():
        for step, minute in enumerate(range(0, 90, 3)):
            src = pl[step % 3]
            dst = pl[(step + 1) % 3]
            rows.append(
                {
                    "statsbomb_match_id": 999,
                    "event_id": f"e{idx}",
                    "index": idx,
                    "period": 1 if minute < 45 else 2,
                    "minute": minute,
                    "second": 0,
                    "type": "Pass",
                    "possession": idx // 5,
                    "team": team,
                    "player": src[1],
                    "player_id": src[0],
                    "position": "Midfield",
                    "x": float(rng.uniform(20, 100)),
                    "y": float(rng.uniform(0, 80)),
                    "pass_end_x": float(rng.uniform(40, 120)),
                    "pass_end_y": float(rng.uniform(0, 80)),
                    "pass_recipient": dst[1],
                    "pass_recipient_id": dst[0],
                    "pass_outcome": None if rng.random() < 0.8 else "Incomplete",
                    "pass_length": 15.0,
                    "shot_xg": None,
                    "shot_outcome": None,
                    "under_pressure": bool(rng.random() < 0.3),
                }
            )
            idx += 1
        for minute in (12, 40, 67, 88):
            rows.append(
                {
                    "statsbomb_match_id": 999,
                    "event_id": f"e{idx}",
                    "index": idx,
                    "period": 1 if minute < 45 else 2,
                    "minute": minute,
                    "second": 30,
                    "type": "Shot",
                    "possession": idx // 5,
                    "team": team,
                    "player": pl[0][1],
                    "player_id": pl[0][0],
                    "position": "Forward",
                    "x": 105.0,
                    "y": 40.0,
                    "pass_end_x": None,
                    "pass_end_y": None,
                    "pass_recipient": None,
                    "pass_recipient_id": None,
                    "pass_outcome": None,
                    "pass_length": None,
                    "shot_xg": float(rng.uniform(0.05, 0.5)),
                    "shot_outcome": "Goal" if (team == "Home FC" and minute == 40) else "Saved",
                    "under_pressure": False,
                }
            )
            idx += 1
    return pd.DataFrame(rows)
