from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.odds.steam import detect_divergence, detect_steam, flags_to_frame

pytestmark = pytest.mark.unit


def _snap(book: str, h: float, d: float, a: float, ts: str = "2024-01-01T10:00") -> list[dict]:
    return [
        {"match_id": "m1", "bookmaker": book, "side": s, "price": p, "snapshot_ts": pd.Timestamp(ts)}
        for s, p in zip(("home", "draw", "away"), (h, d, a), strict=True)
    ]


def test_divergence_flags_soft_book_too_generous():
    snap = pd.DataFrame(_snap("PS", 1.70, 3.8, 5.5) + _snap("SOFT", 2.10, 3.5, 3.6))
    flags = detect_divergence(snap, sharp_book="PS", min_gap=0.03)
    assert flags and flags[0].side == "home" and flags[0].soft_book == "SOFT" and flags[0].gap > 0.03


def test_no_divergence_when_books_agree():
    snap = pd.DataFrame(_snap("PS", 2.0, 3.4, 3.8) + _snap("SOFT", 2.02, 3.4, 3.75))
    assert detect_divergence(snap) == []


def test_steam_detected_when_sharp_moves_and_soft_lags():
    hist = pd.DataFrame(
        _snap("PS", 2.20, 3.4, 3.3, "2024-01-01T08:00")
        + _snap("PS", 1.85, 3.6, 4.2, "2024-01-01T10:00")  # sharp shortens home strongly
        + _snap("SOFT", 2.20, 3.4, 3.3, "2024-01-01T08:00")
        + _snap("SOFT", 2.20, 3.4, 3.3, "2024-01-01T10:00")  # soft hasn't moved
    )
    flags = detect_steam(hist, sharp_book="PS")
    assert any(f.kind == "steam" and f.side == "home" for f in flags)
    df = flags_to_frame(flags)
    assert {"match_id", "side", "gap", "kind"} <= set(df.columns)


def test_steam_ignored_when_soft_follows():
    hist = pd.DataFrame(
        _snap("PS", 2.20, 3.4, 3.3, "2024-01-01T08:00")
        + _snap("PS", 1.85, 3.6, 4.2, "2024-01-01T10:00")
        + _snap("SOFT", 2.20, 3.4, 3.3, "2024-01-01T08:00")
        + _snap("SOFT", 1.88, 3.6, 4.1, "2024-01-01T10:00")
    )
    assert detect_steam(hist) == []
