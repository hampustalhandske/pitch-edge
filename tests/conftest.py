from __future__ import annotations

import numpy as np
import pandas as pd
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


@pytest.fixture
def football_data_csv_bytes() -> bytes:
    header = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,"
        "B365H,B365D,B365A,PSH,PSD,PSA,B365>2.5,B365<2.5,PSCH,PSCD,PSCA\n"
    )
    rows = [
        "E0,12/08/2023,20:00,Arsenal,Man United,2,1,H,1,0,H,M Oliver,15,9,6,3,10,12,7,3,1,2,0,0,1.80,3.60,4.50,1.85,3.70,4.40,1.70,2.20,1.82,3.75,4.55",
        "E0,13/08/2023,14:00,Chelsea,Liverpool,1,1,D,0,1,A,A Taylor,12,14,4,5,9,11,5,6,2,1,0,0,2.90,3.30,2.50,2.95,3.35,2.45,1.75,2.10,2.90,3.40,2.50",
        "E0,19/08/2023,15:00,Man United,Chelsea,0,2,A,0,1,A,P Tierney,10,13,3,6,13,8,4,5,3,1,0,0,2.20,3.40,3.30,2.25,3.45,3.20,1.80,2.05,2.30,3.40,3.15",
        "E0,20/08/2023,16:30,Liverpool,Arsenal,3,1,H,2,0,H,M Oliver,18,8,8,3,8,14,9,2,0,2,0,1,1.95,3.60,3.90,1.90,3.65,4.00,1.60,2.40,1.88,3.70,4.10",
    ]
    return (header + "\n".join(rows) + "\n").encode("utf-8")


@pytest.fixture
def extra_league_csv_bytes() -> bytes:
    header = "Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,BFECH,BFECD,BFECA\n"
    rows = [
        "Sweden,Allsvenskan,2023,01/04/2023,15:00,Malmo FF,AIK,2,0,H,1.8,3.6,4.6,1.9,3.8,4.9,1.78,3.5,4.4,1.88,3.7,4.7",
        "Sweden,Allsvenskan,2023,02/04/2023,15:00,Hammarby,Djurgarden,1,1,D,2.4,3.3,3.0,2.5,3.4,3.1,2.35,3.25,2.95,2.45,3.35,3.05",
    ]
    return ("﻿" + header + "\n".join(rows) + "\n").encode("utf-8")


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
