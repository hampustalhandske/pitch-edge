"""Writes the six R7 registration files (research/registry/outcome-models-*-v1.yaml): for each `ask` model
(dixon_coles, gbdt, stochastic_strength) an edge-in-bits hypothesis and a bet-rule CLV hypothesis."""

from __future__ import annotations

import sys
from pathlib import Path

REG = Path(__file__).resolve().parents[2] / "registry"
REGISTERED_AT = sys.argv[1]

MODELS = {
    "dixon_coles": ("dixon-coles", "Dixon-Coles (DixonColesMatchModel: one bivariate-Poisson fit per league, xi=0.0018, 730-day lookback)"),
    "gbdt": ("gbdt", "GBDT (GBDTMatchModel(include_market=False): sklearn HistGradientBoosting, 300 iters, depth 4, no market features)"),
    "stochastic_strength": ("stochastic-strength", "stochastic_strength (StochasticStrengthModel: Elo with Ornstein-Uhlenbeck Monte-Carlo noise, 2000 sims)"),
}

COMMON_DECISION = """\
  The football-data.co.uk Pinnacle opening snapshot (PSH/PSD/PSA: Friday afternoon for Fri-Mon games, Tuesday afternoon
  for Tue-Thu games; R1 uses the latest Tue/Fri 18:00 Europe/London before kickoff as open_known_at). The model's
  information cutoff is the start date of its walk-forward fold: it is fitted on matches with date < fold_start and
  predicts the 14 days [fold_start, fold_start+14d) without refitting, so the model never sees anything dated on or
  after the fold start (0-13 days before the match). Every feature is a rolling/expanding/sequential quantity shifted
  by one match, or an external pre-match rating (home_elo/away_elo, probed in exploratory data to move with the
  previous match's surprise and not the next one). Asserted in code (research/studies/r7-outcome-models/pit.py): (i)
  feature-builder truncation invariance (features rebuilt from a spine cut at a date equal the full-spine features for
  every earlier match), (ii) walk-forward truncation invariance (predictions of a run on the frame cut at 2020-06-30
  equal the full run's for every match up to the cut), (iii) train.max(date) < fold_start <= match date for every
  prediction. Results, closing prices and the Pinnacle close are targets/benchmarks only."""

COMMON_UNIVERSE = """\
  Matches of the 11 DEFAULT_BACKTEST_LEAGUES (E0, E1, D1, SP1, I1, F1, N1, P1, B1, SC0, T1; spine from
  `load_feature_frame(min_date=2015-07-01)` of the warehouse, read-only, snapshotted to data/research/r7/features.parquet)
  that have a complete Pinnacle open (PSH/PSD/PSA) AND a complete Pinnacle close (PSCH/PSCD/PSCA) and a result,
  predicted by a model fitted walk-forward (retrain every 14 days = the `ask` candidate window; first fold after 400
  training matches). Matches without both Pinnacle quotes are dropped and counted per league/season in the report."""

WINDOW = """\
  exploratory: [2015-07-01, 2019-06-30]   # seasons before 2019/20; may be looked at to design
  evaluation:  [2019-07-01, 2025-06-30]   # seasons 2019/20-2024/25, scored once after registration"""

NOTES_COMMON = """\
  Why this window: from counts only, in the warehouse frame Pinnacle open+close are present for 3,140 / 4,309 / 3,900 /
  3,835 / 2,647 / 3,795 matches in 2019/20 ... 2024/25 (2023/24 is 68% covered: a coverage hole, reported per league),
  1,736 of 3,761 in 2025/26 (to 2025-09) and none from 2026-02, so 2024/25 is the last complete season. Code:
  research/studies/r7-outcome-models/{snapshot,pit,generate,run}.py; per-match outputs in data/research/r7/. The
  production `ask` agent (agents/qa_graph.py::gather) fits `default_models()` fresh at as_of on the rows dated before
  as_of of its own feature frame (CLI: min_date 2018-07-01; dashboard: all rows) and serves the RAW model probabilities
  (no isotonic calibration); this study therefore scores the raw `raw_*` probabilities of the walk-forward and reports
  the engine's isotonic-calibrated `p_*` as a secondary. The walk-forward refit cadence (14 days) matches the ask
  window; the train window starts 2015-07 (not the warehouse's full history). Weather features are observed, not
  forecast, values (a mild approximation of a forecast; `wx_*` are inputs to gbdt only)."""


def edge_bits(slug: str, model: str, desc: str) -> str:
    return f"""\
id: outcome-models-{slug}-edge-bits-v1
family: outcome-models
registered_at: {REGISTERED_AT}
author: quant-researcher
question: >
  Does the `ask` agent's {desc} predict match outcomes better than the de-vigged Pinnacle close (log-loss edge in bits,
  walk-forward, 2019/20-2024/25)?
hypothesis:
  h1: >
    The model's mean per-match log2 p_model(result) - log2 p_close(result) (p_close = power-de-vigged Pinnacle close) is
    > 0 over the evaluation window (positive = model better than the sharp close).
  h0: mean edge_bits <= 0
  direction: positive
  disconfirming: >
    95% match-day block-bootstrap CI of mean edge_bits includes 0 or lies below it; or the sign is negative in 2 of 3
    chronological thirds; or, with any one league removed, the 95% CI includes 0 (STANDARDS L3e). Expected, from the
    existing report (all models below market, best -0.024 bits): a clean negative.
decision_time: >
{COMMON_DECISION}
universe: >
{COMMON_UNIVERSE}
window:
{WINDOW}
benchmark: >
  Pinnacle closing 1X2 (PSCH/PSCD/PSCA), de-vigged by the power method (STANDARDS). Sensitivities (secondary metrics):
  Shin de-vig of the close, multiplicative de-vig of the Pinnacle OPEN (the market price `ask` compares against), and the
  engine's isotonic-calibrated probabilities.
primary_metric: edge_bits
metric_kind: probability
secondary_metrics:
  - edge_bits_shin
  - edge_bits_vs_open_mult
  - edge_bits_calibrated
costs: none (probability scoring; no bet is placed)
grid:
  probabilities: [raw]
min_n: 300
literature_ceiling: >
  Published public-data models land between -0.03 and +0.005 bits vs the de-vigged Pinnacle close; above +0.02 bits is a
  leakage suspect (STANDARDS).
notes: >
  Model under test: {desc}. value per match = log2 p_model(result) - log2 p_close(result); `raw` = the model's own
  probabilities exactly as `ask` serves them. Block = ISO match date; league = league_code (leave-one-league-out). The
  engine run uses WalkForwardConfig(min_train_matches=400, retrain_every_days=14, calibrate=True) with no staking
  strategies; seeds fixed (HistGradientBoosting random_state=42, StochasticStrengthModel seed=0).
{NOTES_COMMON}
  Before registration only coverage counts, an Elo-column timing probe on pre-2019/20 rows (exploratory) and the existing
  warehouse report (all models below market) were looked at; no metric of this study was computed on any window.
"""


def bet_rule(slug: str, model: str, desc: str) -> str:
    return f"""\
id: outcome-models-{slug}-bet-rule-v1
family: outcome-models
registered_at: {REGISTERED_AT}
author: quant-researcher
question: >
  Does the engine's model-vs-market bet rule applied to the `ask` agent's {desc} (bet the outcome with the largest
  edge = p_model - no-vig Pinnacle open probability when the edge exceeds 3%, at the quoted Pinnacle open price) beat the
  de-vigged Pinnacle close (CLV), walk-forward, 2019/20-2024/25?
hypothesis:
  h1: >
    Mean per-bet CLV = ln(PS_odds_taken * p_close) (p_close = power-de-vigged Pinnacle close probability of the bet
    outcome) is > 0 over the evaluation window, taking the Pinnacle open odds including their margin.
  h0: mean CLV <= 0
  direction: positive
  disconfirming: >
    95% match-day block-bootstrap CI of mean CLV includes 0 or lies below it; or the sign is negative in 2 of 3
    chronological thirds; or, with any one league removed, the CI includes 0. Expected: negative (taking Pinnacle's open
    with its margin against a sharp close costs roughly the margin on a rule with no information).
decision_time: >
{COMMON_DECISION}
universe: >
{COMMON_UNIVERSE}
window:
{WINDOW}
benchmark: >
  Pinnacle closing 1X2, de-vigged by the power method (STANDARDS: CLV = log of taken odds over de-vigged closing odds).
  Shin close de-vig as a sensitivity.
primary_metric: clv_log
metric_kind: return
secondary_metrics:
  - roi_flat
  - clv_log_shin
  - clv_log_top4_per_window
costs: >
  The Pinnacle opening margin is paid (the bet is taken at the quoted PS odds, never at fair odds); no commission or
  stake limit is modelled beyond that (Pinnacle charges no commission).
grid:
  edge_threshold: [0.03]
min_n: 300
literature_ceiling: >
  CLV vs Pinnacle close from public-information models: a few tenths of a percent at most; above +2% per bet over
  hundreds of bets is a leakage suspect (STANDARDS).
notes: >
  Rule (src/pitch_edge/backtest/engine.py::_place_bets, the production backtest): per match, edge_o = p_raw_o - m_o
  with m = multiplicative no-vig of PS (exactly agents/qa_data.py::market_edges / odds.utils.no_vig_probabilities),
  candidates have edge_o > edge_threshold and 1.2 <= PS_o <= 12, bet only the best-edge outcome (max_bets_per_match=1).
  The edge uses the model's RAW probabilities (what `ask` serves; the engine bets its calibrated p). The production
  threshold is the engine's WalkForwardConfig default 0.03 = the single grid point. `ask` itself has NO threshold:
  agents/qa_graph.py::gather ranks the candidate window's fixtures by best-outcome edge and keeps the top n (default 4,
  14-day window); that literal rule is the secondary `clv_log_top4_per_window` (consecutive 14-day windows from
  2019-07-01, every match with a complete Pinnacle pair is a candidate, top 4 by best-outcome edge, any sign) and is
  not the primary. Secondary `roi_flat` = flat 1-unit return per bet (won: PS-1, lost: -1). `ask` ranks with the model
  its slice evidence trusts per fixture and, with a warehouse wired in, prices against Polymarket instead of Pinnacle;
  neither is testable historically (no multi-season Polymarket history), so this study scores each model's own rule
  against the Pinnacle open. Block = ISO match date; league = league_code.
{NOTES_COMMON}
  Before registration only coverage counts, an Elo-column timing probe on pre-2019/20 rows and the existing warehouse
  report were looked at; no metric of this study was computed on any window.
"""


for model, (slug, desc) in MODELS.items():
    (REG / f"outcome-models-{slug}-edge-bits-v1.yaml").write_text(edge_bits(slug, model, desc))
    (REG / f"outcome-models-{slug}-bet-rule-v1.yaml").write_text(bet_rule(slug, model, desc))
print("written")
