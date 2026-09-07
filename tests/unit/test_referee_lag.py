"""Referee-announcement lag: page parsing, snapshot pairing, tendency table, pre-registered event study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import responses

from pitch_edge.backtest.event_study import referee_announcement_study, referee_tendency_table
from pitch_edge.data.alt.referee_announcements import (
    RefereeAnnouncementSource,
    pair_with_snapshots,
    parse_assignments,
)

pytestmark = pytest.mark.unit

HTML = """
<h2>Matchweek 4 appointments — 13 September 2026</h2>
<div class="fixture">Brighton and Hove Albion v Nottingham Forest <span>Referee: Michael Oliver</span> Assistants: X, Y</div>
<div class="fixture">Arsenal v Chelsea <span>Referee: Anthony Taylor</span></div>
<p>Wolves v Everton — VAR: Someone Else</p>
"""


def test_parse_assignments_resolves_teams_and_referees():
    df = parse_assignments(HTML, teams=["Brighton", "Nott'm Forest", "Arsenal", "Chelsea", "Wolves", "Everton"])
    assert len(df) == 2
    first = df.iloc[0]
    assert (
        first["home_team"] == "Brighton"
        and first["away_team"] == "Nott'm Forest"
        and first["referee"] == "Michael Oliver"
    )
    assert first["fixture_key"] == "brighton|nottm forest|2026-09-13"


@responses.activate
def test_poll_uses_robots_and_tags_source(tmp_path):
    responses.add(responses.GET, "https://example.org/robots.txt", body="User-agent: *\nAllow: /\n")
    responses.add(responses.GET, "https://example.org/appointments", body=HTML)
    src = RefereeAnnouncementSource(
        cache_dir=tmp_path, sources={"test": {"url": "https://example.org/appointments", "league_code": "E0"}}
    )
    df = src.poll(teams=["Brighton", "Nott'm Forest", "Arsenal", "Chelsea"], now=pd.Timestamp("2026-09-10 09:00"))
    assert (
        len(df) == 2
        and set(df["source"]) == {"test"}
        and df["announced_at"].iloc[0] == pd.Timestamp("2026-09-10 09:00")
    )


def test_pair_with_snapshots_before_and_after():
    ann = pd.DataFrame(
        {
            "fixture_key": ["brighton|nottm forest|2026-09-13"],
            "referee": ["Michael Oliver"],
            "announced_at": [pd.Timestamp("2026-09-10 09:00")],
        }
    )
    rows = []
    for ts, ph in ((pd.Timestamp("2026-09-10 08:00"), 0.40), (pd.Timestamp("2026-09-10 10:00"), 0.44)):
        for o, p in (("Brighton", ph), ("Tie", 0.28), ("Nottingham Forest", 1 - ph - 0.28)):
            rows.append(
                {
                    "venue": "kalshi",
                    "question": "Brighton vs Nottingham Forest game on Sep 13, 2026",
                    "outcome": o,
                    "probability": p,
                    "snapshot_ts": ts,
                }
            )
    paired = pair_with_snapshots(ann, pd.DataFrame(rows))
    home = paired[paired["outcome"] == "home"].iloc[0]
    assert home["p_before"] == pytest.approx(0.40) and home["move"] == pytest.approx(0.04) and home["gap_hours"] == 2


def test_tendency_table_and_study_verdicts(synthetic_league_matches):
    tend = referee_tendency_table(synthetic_league_matches, min_matches=20)
    assert len(tend) == 6 and np.isclose(tend["z_cards"].mean(), 0, atol=1e-9)
    # not enough paired announcements -> insufficient, never a made-up number
    empty = referee_announcement_study(pd.DataFrame(), tend)
    assert empty.iloc[-1]["verdict"].startswith("insufficient")
    rng = np.random.default_rng(0)
    strong_ref = tend.assign(z=tend[["z_cards", "z_home_bias"]].abs().max(axis=1)).sort_values("z").iloc[-1]["referee"]
    neutral_ref = tend.assign(z=tend[["z_cards", "z_home_bias"]].abs().max(axis=1)).sort_values("z").iloc[0]["referee"]
    paired = pd.DataFrame(
        [
            {
                "fixture_key": f"a|b|{i}",
                "referee": strong_ref,
                "venue": "kalshi",
                "outcome": "home",
                "move": rng.normal(0, 0.03),
            }
            for i in range(20)
        ]
        + [
            {
                "fixture_key": f"c|d|{i}",
                "referee": neutral_ref,
                "venue": "kalshi",
                "outcome": "home",
                "move": rng.normal(0, 0.03),
            }
            for i in range(20)
        ]
    )
    tend.loc[tend["referee"] == strong_ref, "z_cards"] = 2.0
    tend.loc[tend["referee"] == neutral_ref, ["z_cards", "z_home_bias"]] = 0.0
    out = referee_announcement_study(paired, tend)
    assert set(out["group"]) == {"strong_tendency", "neutral", "ALL"} and out.iloc[-1]["n"] == 40
    assert out.iloc[-1]["verdict"].startswith(("H0", "H1"))
