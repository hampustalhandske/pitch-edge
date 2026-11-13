"""Deterministic data assembly for the `ask` agent: candidate fixtures, model predictions, and
market-vs-model edges — all as-of aware, no LLM anywhere in this module.

A fixture only ever becomes a candidate if it carries a real (non-synthetic) Pinnacle early quote
(`PSH`/`PSD`/`PSA`) — there is no live-odds feed today, so "upcoming" here always means an already-
archived historical match being treated as if it were upcoming relative to a past `as_of`.

`annotate_candidates` attaches to each candidate the evidence status of the model that produced
its number, read from the research ladder summary (`research/summary.py`): a candidate is only a
*validated edge* when that model maps to a registered hypothesis at or above the display floor.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pitch_edge.backtest.as_of import filter_before, filter_window
from pitch_edge.data.teams import TeamNameResolver
from pitch_edge.models.base import OUTCOMES, MatchModel
from pitch_edge.odds.edge import compute_edge
from pitch_edge.research.summary import BASE_DISPLAY_FLOOR, meets_floor

REAL_ODDS_COLUMNS = ("PSH", "PSD", "PSA")

# The `ask` models (models.default_models) and the registered hypothesis (research/registry/<id>.yaml)
# whose rung a candidate from that model carries: each model's R7 bet-rule hypothesis (bet the
# best-edge outcome against the Pinnacle open, scored by CLV against the de-vigged close), the
# return-kind test of the model-vs-market rule `ask` ranks with. Its probability-kind companion,
# `outcome-models-<model>-edge-bits-v1`, cannot make a candidate an edge (research/STANDARDS.md L4).
ASK_MODEL_HYPOTHESES: dict[str, str | None] = {
    "dixon_coles": "outcome-models-dixon-coles-bet-rule-v1",
    "gbdt": "outcome-models-gbdt-bet-rule-v1",
    "stochastic_strength": "outcome-models-stochastic-strength-bet-rule-v1",
}
STALE_DATA_DAYS = 7  # a league's data is flagged when it ends more than this many days before the as-of date
NOT_VALIDATED_LABEL = "model-vs-market disagreement, not a validated edge"
VALIDATED_LABEL = "validated edge"


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
    `None` if no real-odds fixture between these two teams exists in the window. When the two meet more than
    once with the asked home side, or else more than once reversed, the earliest (by date, then match_id) is
    returned."""
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
    return match.sort_values(["date", "match_id"], kind="stable").iloc[[0]] if not match.empty else None


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
            rows.append({"match_id": mid, "model": m.name, "p_home": float(h), "p_draw": float(d), "p_away": float(a)})
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


def _rung_in(research: dict[str, Any], hypothesis_id: str) -> str | None:
    for fam in research.get("families", []):
        for h in fam.get("hypotheses", []):
            if h.get("id") == hypothesis_id:
                return str(h.get("rung"))
    return None


def candidate_evidence(
    model: str | None, research: dict[str, Any], model_hypotheses: dict[str, str | None] | None = None
) -> dict[str, Any]:
    """Evidence status of `model` given the ladder summary `research` (`EvidenceSummary.to_dict()`).

    Returns the model, its registered hypothesis id (None when it has none), that hypothesis' rung
    (`unregistered` / `not in ledger` otherwise), the display floor, whether the candidate counts as a
    validated edge, a short `status` and the `label` the answer must use for it."""
    mapping = ASK_MODEL_HYPOTHESES if model_hypotheses is None else model_hypotheses
    floor = str(research.get("display_floor", BASE_DISPLAY_FLOOR))
    hypothesis_id = mapping.get(model) if model is not None else None
    if hypothesis_id is None:
        rung, status = "unregistered", "no registered evidence / below floor"
    else:
        found = _rung_in(research, hypothesis_id)
        rung = found if found is not None else "not in ledger"
        status = f"{rung}, {'at or above' if meets_floor(rung, floor) else 'below'} display floor {floor}"
    validated = hypothesis_id is not None and meets_floor(rung, floor)
    return {
        "model": model,
        "hypothesis_id": hypothesis_id,
        "rung": rung,
        "display_floor": floor,
        "validated_edge": validated,
        "status": status,
        "label": VALIDATED_LABEL if validated else NOT_VALIDATED_LABEL,
    }


def annotate_candidates(
    candidates: list[dict[str, Any]],
    model_by_match: dict[str, str | None],
    research: dict[str, Any],
    model_hypotheses: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """Copies of `candidates` (same order) each with an `evidence` entry from `candidate_evidence` for
    the model that produced its number (`model_by_match[match_id]`, None when no model did)."""
    return [
        {**c, "evidence": candidate_evidence(model_by_match.get(c["match_id"]), research, model_hypotheses)}
        for c in candidates
    ]


def last_match_dates(wh) -> dict[str, pd.Timestamp]:
    """Read-only: the date of the latest match per `league_code` in the warehouse `matches` table
    (a TIMESTAMP column, truncated to the day). Empty when the table is missing or has no rows."""
    if not wh.table_exists("matches"):
        return {}
    df = wh.query(
        "SELECT league_code, CAST(MAX(date) AS DATE) AS last_date FROM matches "
        "WHERE league_code IS NOT NULL AND date IS NOT NULL GROUP BY league_code"
    )
    return {str(r.league_code): pd.Timestamp(r.last_date) for r in df.itertuples()}


def data_freshness_notes(
    last_dates: dict[str, pd.Timestamp], leagues: list[str], as_of: pd.Timestamp, max_gap_days: int = STALE_DATA_DAYS
) -> list[str]:
    """One line per league (sorted by code) whose data ends more than `max_gap_days` before `as_of`,
    e.g. "Data for E0 ends 2026-09-03 (55 days before 2026-10-28)". Leagues with no known last date,
    or whose data reaches within `max_gap_days` of `as_of` (or beyond it), produce no line."""
    as_day = (
        pd.Timestamp(as_of).tz_localize(None).normalize()
        if pd.Timestamp(as_of).tzinfo
        else pd.Timestamp(as_of).normalize()
    )
    notes: list[str] = []
    for league in sorted(set(leagues)):
        last = last_dates.get(league)
        if last is None:
            continue
        gap = (as_day - pd.Timestamp(last).normalize()).days
        if gap > max_gap_days:
            notes.append(f"Data for {league} ends {pd.Timestamp(last).date()} ({gap} days before {as_day.date()})")
    return notes


def freshness_for(wh, leagues: list[str], as_of: pd.Timestamp) -> list[str]:
    """`data_freshness_notes` for the warehouse `wh`; empty when `wh` is None or cannot be read."""
    if wh is None or not leagues:
        return []
    try:
        return data_freshness_notes(last_match_dates(wh), leagues, as_of)
    except Exception:  # noqa: BLE001 - an unreadable warehouse must never block an answer
        return []
