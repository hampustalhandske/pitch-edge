"""Ingest defaults reach the season under way, so a new season is fetched without a code change."""

from __future__ import annotations

from datetime import date

import pytest

from pitch_edge.data.ingest import latest_season_start

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("today", "start"),
    [(date(2026, 10, 28), 2026), (date(2026, 7, 1), 2026), (date(2026, 6, 30), 2025), (date(2027, 1, 15), 2026)],
)
def test_latest_season_start_turns_over_on_1_july(today, start):
    assert latest_season_start(today) == start
