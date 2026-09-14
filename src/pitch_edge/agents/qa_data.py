"""Deterministic data assembly for the `ask` agent: candidate fixtures, model predictions, and
market-vs-model edges — all as-of aware, no LLM anywhere in this module.

A fixture only ever becomes a candidate if it carries a real (non-synthetic) Pinnacle early quote
(`PSH`/`PSD`/`PSA`) — there is no live-odds feed today, so "upcoming" here always means an already-
archived historical match being treated as if it were upcoming relative to a past `as_of`.
"""

from __future__ import annotations

import pandas as pd

from pitch_edge.backtest.as_of import filter_before, filter_window
from pitch_edge.data.teams import TeamNameResolver
from pitch_edge.models.base import OUTCOMES, MatchModel
from pitch_edge.odds.edge import compute_edge

REAL_ODDS_COLUMNS = ("PSH", "PSD", "PSA")


def real_odds_mask(df: pd.DataFrame) -> pd.Series:
    if not set(REAL_ODDS_COLUMNS) <= set(df.columns):
        return pd.Series(False, index=df.index)
    return df[list(REAL_ODDS_COLUMNS)].notna().all(axis=1)


def top_bets_candidates(
    features: pd.DataFrame, as_of: pd.Timestamp, window_days: int = 14, pmxt_match_ids: set[str] | None = None
) -> pd.DataFrame:
    """Fixtures in `[as_of, as_of + window_days)`. A real bet recommendation needs a real market
    to time an entry against — Polymarket now, not the old football-data.co.uk Pinnacle quote
    (that archive's coverage stops in Jan 2026, before Polymarket data starts, so requiring both
    would always return nothing). When `pmxt_match_ids` is given, only fixtures mapped in
    `pmxt_match_map` qualify; without it (no warehouse wired in), falls back to the old
    Pinnacle-quote gate so this function still degrades sanely for callers that predate PMXT."""
    window = filter_window(features, as_of, window_days)
    if pmxt_match_ids is not None:
        return window[window["match_id"].isin(pmxt_match_ids)].copy()
    return window[real_odds_mask(window)].copy()


def resolve_fixture(
    features: pd.DataFrame, as_of: pd.Timestamp, home_team: str, away_team: str, window_days: int = 60
) -> pd.DataFrame | None:
    """The candidate window's row for this fixture (fuzzy team-name matched, either side), or
    `None` if no real-odds fixture between these two teams exists in the window."""
    window = filter_window(features, as_of, window_days)
    candidates = window[real_odds_mask(window)]
    if candidates.empty:
        return None
    names = set(candidates["home_team"]) | set(candidates["away_team"])
    resolver = TeamNameResolver(names)
    home, away = resolver.resolve(home_team), resolver.resolve(away_team)
    if not home or not away:
        return None
    match = candidates[(candidates["home_team"] == home) & (candidates["away_team"] == away)]
    if match.empty:
        match = candidates[(candidates["home_team"] == away) & (candidates["away_team"] == home)]
    return match.iloc[[0]] if not match.empty else None


def run_models(
    features: pd.DataFrame, fixtures: pd.DataFrame, as_of: pd.Timestamp, models: list[MatchModel]
) -> pd.DataFrame:
    """Fit every model fresh on data strictly before `as_of` (no weights are persisted anywhere in
    this project — every backtest fold already refits from scratch), then predict every candidate
    fixture. One row per (model, fixture)."""
    if fixtures.empty:
        return pd.DataFrame(columns=["match_id", "model", "p_home", "p_draw", "p_away"])
    train = filter_before(features, as_of)
    X = fixtures.copy()
    X["date"] = pd.to_datetime(X["date"])
    rows = []
    for m in models:
        m.fit(train)
        probs = m.predict_proba(X)
        for mid, h, d, a in zip(X["match_id"], probs["home"], probs["draw"], probs["away"], strict=True):
            rows.append(
                {"match_id": mid, "model": m.name, "p_home": float(h), "p_draw": float(d), "p_away": float(a)}
            )
    return pd.DataFrame(rows)


def market_edges(fixtures: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, fixture, outcome): `model_probability`, `market_probability`, `edge`,
    `decimal_odds` — computed from each fixture's own real early Pinnacle quote."""
    if predictions.empty:
        return pd.DataFrame(
            columns=["match_id", "model", "outcome", "model_probability", "market_probability", "edge", "decimal_odds"]
        )
    odds_by_match = fixtures.set_index("match_id")[list(REAL_ODDS_COLUMNS)]
    rows = []
    for _, r in predictions.iterrows():
        if r["match_id"] not in odds_by_match.index:
            continue
        h_odds, d_odds, a_odds = odds_by_match.loc[r["match_id"]]
        edge = compute_edge({o: r[f"p_{o}"] for o in OUTCOMES}, h_odds, d_odds, a_odds)
        for outcome, stats in edge.items():
            rows.append({"match_id": r["match_id"], "model": r["model"], "outcome": outcome, **stats})
    return pd.DataFrame(rows)


def pmxt_market_edges(predictions: pd.DataFrame, live_market: dict[str, dict]) -> pd.DataFrame:
    """Same shape as `market_edges`, but priced against real Polymarket no-vig probabilities
    (`live_market`, from `PolymarketOddsProvider` ticks strictly before `as_of`) instead of the
    old Pinnacle quote — the real market this project actually times entries against now."""
    rows = []
    for _, r in predictions.iterrows():
        m = live_market.get(r["match_id"])
        if not m:
            continue
        for outcome in OUTCOMES:
            model_p = float(r[f"p_{outcome}"])
            market_p = float(m[f"p_{outcome}"])
            rows.append(
                {
                    "match_id": r["match_id"],
                    "model": r["model"],
                    "outcome": outcome,
                    "model_probability": model_p,
                    "market_probability": market_p,
                    "edge": model_p - market_p,
                    "decimal_odds": 1.0 / market_p if market_p > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)
