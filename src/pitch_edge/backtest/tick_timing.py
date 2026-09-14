"""Tick-timing backtest: does entering when the model and Polymarket's live price disagree pay off?

Deliberately separate from `backtest/engine.py`: that engine is built around exactly two price
points per match (an early quote, a closing quote) and date-level fold boundaries. Here a match
carries many timestamped Polymarket ticks before kickoff (`odds/providers.py::PolymarketOddsProvider`),
and it's the entry *policy* itself being compared, not just a staking strategy.

Model scoring follows quantitative-finance convention: risk-adjusted return (Sharpe ratio,
`backtest/metrics.py::sharpe_ratio`) is the primary ranking criterion across entry policies, per
this project's own `metrics.py` docstring — "CLV, not raw ROI, is the standard proof of edge...
ROI over a realistic sample is dominated by variance." Sharpe extends that same logic to comparing
policies against each other, not just against zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitch_edge.backtest.kelly import fractional_kelly_stake
from pitch_edge.backtest.metrics import clv as clv_metric
from pitch_edge.backtest.metrics import roi as roi_metric
from pitch_edge.backtest.metrics import sharpe_ratio
from pitch_edge.data.storage import Warehouse
from pitch_edge.models.base import OUTCOMES, MatchModel
from pitch_edge.models.gbdt import GBDTMatchModel
from pitch_edge.odds.providers import PolymarketOddsProvider
from pitch_edge.odds.utils import no_vig_probabilities
from pitch_edge.pipeline import load_feature_frame

logger = logging.getLogger(__name__)

ENTRY_POLICIES = ("first_tick", "threshold_cross", "last_tick")


@dataclass
class TickTimingConfig:
    edge_threshold: float = 0.03  # same value the deleted agents/risk.py::RiskLimits.min_edge used
    retrain_every_days: int = 30
    min_train_matches: int = 400
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.05
    bankroll0: float = 1000.0
    min_odds: float = 1.2
    max_odds: float = 12.0


def _model() -> MatchModel:
    """Gradient-boosted trees on the tabular feature store — the standard, robust choice for this
    kind of structured prediction task, and the model this project's own `models/gbdt.py` docstring
    frames as the default. `include_market=False`: we're computing edge against the Polymarket
    market itself, so feeding the model football-data.co.uk's own market price would conflate the
    two markets rather than compare the model honestly against a market it never saw."""
    return GBDTMatchModel(include_market=False)


def _pick_entries(edges: list[float], cfg: TickTimingConfig) -> dict[str, int | None]:
    """Index into a match's chronological quote list each policy would enter at. `threshold_cross`
    is `None` (never enters) if the edge condition is never met for this match."""
    entries: dict[str, int | None] = {"first_tick": 0, "last_tick": len(edges) - 1, "threshold_cross": None}
    for i, e in enumerate(edges):
        if e >= cfg.edge_threshold:
            entries["threshold_cross"] = i
            break
    return entries


def run_tick_timing_backtest(wh: Warehouse, cfg: TickTimingConfig | None = None) -> dict[str, pd.DataFrame]:
    """One row per (match, entry policy) in `bets`; one row per entry policy in `summary`, ranked
    by Sharpe ratio (primary quant-finance selection criterion) with ROI/CLV/hit-rate reported
    alongside. Models are refit every `retrain_every_days` on strictly-earlier data, same
    walk-forward discipline as `backtest/engine.py`."""
    cfg = cfg or TickTimingConfig()
    mapped = wh.query("SELECT DISTINCT match_id FROM pmxt_match_map")
    if mapped.empty:
        logger.warning("run_tick_timing_backtest: pmxt_match_map is empty — run `pitch-edge map-pmxt` first")
        return {"bets": pd.DataFrame(), "summary": pd.DataFrame()}

    # PMXT covers any league Polymarket lists (obscure leagues included), not just the
    # pre-registered major-league set load_feature_frame defaults to — so pass every league
    # code actually present rather than let it silently narrow the training/test pool.
    all_leagues = wh.query("SELECT DISTINCT league_code FROM matches")["league_code"].dropna().tolist()
    features = load_feature_frame(wh, leagues=all_leagues)
    if features.empty:
        logger.warning("run_tick_timing_backtest: feature store is empty — run `pitch-edge features` first")
        return {"bets": pd.DataFrame(), "summary": pd.DataFrame()}
    features = features.sort_values("date").reset_index(drop=True)
    features["date"] = pd.to_datetime(features["date"])
    mapped_ids = set(mapped["match_id"])
    test_pool = features[features["match_id"].isin(mapped_ids)]
    if test_pool.empty:
        logger.warning("run_tick_timing_backtest: no feature-store rows for any pmxt-mapped match")
        return {"bets": pd.DataFrame(), "summary": pd.DataFrame()}

    provider = PolymarketOddsProvider(wh)
    bet_rows: list[dict] = []
    bankroll = dict.fromkeys(ENTRY_POLICIES, cfg.bankroll0)

    fold_starts = pd.date_range(test_pool["date"].min(), test_pool["date"].max(), freq=f"{cfg.retrain_every_days}D")
    for fold_start in fold_starts:
        fold_end = fold_start + pd.Timedelta(days=cfg.retrain_every_days)
        train = features[features["date"] < fold_start]
        test = test_pool[(test_pool["date"] >= fold_start) & (test_pool["date"] < fold_end)]
        if test.empty or len(train) < cfg.min_train_matches:
            continue
        model = _model()
        model.fit(train)
        preds = model.predict_proba(test)

        for idx, row in test.iterrows():
            match_id = row["match_id"]
            p = preds.loc[idx]
            model_probs = {o: float(p[o]) for o in OUTCOMES}
            favored = max(model_probs, key=lambda o: model_probs[o])
            quotes = provider.get_quotes(match_id)
            if not quotes:
                continue
            edges = []
            for q in quotes:
                mkt_probs = dict(zip(OUTCOMES, no_vig_probabilities(q.home_odds, q.draw_odds, q.away_odds), strict=True))
                edges.append(model_probs[favored] - mkt_probs[favored])
            entries = _pick_entries(edges, cfg)
            closing_odds = getattr(quotes[-1], f"{favored}_odds")
            won = int(row["result"] == OUTCOMES.index(favored))

            for policy, i in entries.items():
                if i is None:
                    continue
                entry_odds = getattr(quotes[i], f"{favored}_odds")
                if not (cfg.min_odds <= entry_odds <= cfg.max_odds):
                    continue
                stake = fractional_kelly_stake(
                    model_probs[favored], entry_odds, bankroll[policy],
                    fraction=cfg.kelly_fraction, max_stake_pct=cfg.max_stake_pct,
                )
                if stake <= 0:
                    continue
                profit = stake * (entry_odds - 1) if won else -stake
                bankroll[policy] += profit
                bet_rows.append(
                    {
                        "match_id": match_id,
                        "policy": policy,
                        "outcome": favored,
                        "model_p": model_probs[favored],
                        "entry_odds": entry_odds,
                        "closing_odds": closing_odds,
                        "entry_ts": quotes[i].timestamp,
                        "edge_at_entry": edges[i],
                        "stake": stake,
                        "won": won,
                        "profit": profit,
                    }
                )

    bets = pd.DataFrame(bet_rows)
    if bets.empty:
        return {"bets": bets, "summary": pd.DataFrame()}

    summary_rows = []
    for policy, g in bets.groupby("policy"):
        returns = (g["profit"] / g["stake"]).to_numpy()
        summary_rows.append(
            {
                "policy": policy,
                "n_bets": len(g),
                "roi": roi_metric(g["stake"].to_numpy(), g["profit"].to_numpy()),
                "mean_clv_pct": float(np.mean(clv_metric(g["entry_odds"].to_numpy(), g["closing_odds"].to_numpy()))),
                "sharpe": sharpe_ratio(returns) if len(returns) > 1 else 0.0,
                "hit_rate": float(g["won"].mean()),
                "final_bankroll": bankroll[policy],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return {"bets": bets, "summary": summary}
