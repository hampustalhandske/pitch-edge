# R7 — pitch-edge's own outcome models against the de-vigged Pinnacle close

Family `outcome-models`. Six registrations (ledger 2026-10-21 06:58:16–25Z, written by `make_registrations.py`; all
trials 18:54–18:55Z): `research/registry/outcome-models-{dixon-coles,gbdt,stochastic-strength}-edge-bits-v1.yaml`
(probability kind) and `…-{dixon-coles,gbdt,stochastic-strength}-bet-rule-v1.yaml` (return kind). Code:
`snapshot.py`, `generate.py`, `pit.py`, `run.py` (exploratory probe: `probe_elo.py`). Outputs: `data/research/r7/`.
Ledger rows: 5 per hypothesis, 30 in the family.

## Question
Do the three models the `ask` agent ranks candidates with (`models.default_models()`: Dixon-Coles, GBDT without market
features, stochastic_strength) forecast 1X2 outcomes better than the de-vigged Pinnacle close (edge in bits), and does
the engine's model-vs-market bet rule on each of them beat the close (CLV)?

## Data, decision time, windows
Warehouse `load_feature_frame(min_date=2015-07-01)`, read-only, snapshotted to `features.parquet` (sha256
b6d80c39…), 11 `DEFAULT_BACKTEST_LEAGUES`. Walk-forward with the production `WalkForwardBacktester`
(`min_train_matches=400`, refit every 14 days = the `ask` candidate window, isotonic calibration fitted on realised
past folds; HGB `random_state=42`, stochastic `seed=0`). The model's information cutoff is its fold start: fitted on
matches dated before it, predicting the next 14 days. Decision price = football-data Pinnacle open (PSH/PSD/PSA, the
Tue/Fri snapshot); benchmark = Pinnacle close (PSCH/PSCD/PSCA), power de-vig (Shin as a secondary). Scored matches
need a result and complete open + close. Exploratory 2015-07 → 2019-06 (n = 14,325); evaluation 2019/20 → 2024/25:
**n = 21,626 matches, 1,299 match-days** (1,647 of 23,273 predicted matches dropped for lack of a Pinnacle pair — the
2023/24 hole, 2,647 of 3,885, plus `cfmd_` spine rows that never carry Pinnacle odds; see threats).

Point in time, asserted in `pit.py` (`data/research/r7/pit.json`): feature-builder truncation invariance (spine cut at
2022-01-01; 25,030 rows × 86 columns, no unexplained difference — 2 rows of `ref_*` differ only inside same-referee
same-day pairs, 8 such rows in the evaluation window; `sv_*` differ but no `ask` model reads them); walk-forward
truncation invariance (run on the frame cut at 2020-06-30 equals the full run on 17,936 rows, max |Δp| = 0.0 for all
three models); fold discipline (`train_max_date < fold_start <= date`, `date − fold_start < 14 d`) on all 37,994 rows.
`run.py` re-checks the cut-vs-full predictions itself and records `clean` only if all of these hold.

## Method
- edge-bits: per match `log2 p_model(result) − log2 p_close(result)`, the model's **raw** probabilities (what `ask`
  serves); secondaries: Shin close, multiplicative de-vigged open, the engine's isotonic-calibrated probabilities.
- bet-rule (`engine._place_bets` on raw probabilities): edge = p_raw − multiplicative no-vig Pinnacle open; bet the
  best-edge outcome when edge > 0.03 and 1.2 ≤ odds ≤ 12, at the quoted open (margin paid). CLV = ln(odds × p_close).
  Secondaries: flat ROI, Shin CLV, and `ask`'s literal rule (top 4 matches by best-outcome edge per consecutive 14-day
  window, any sign).
- Match-day block bootstrap, 2,000 draws, seed 20261021, one-sided p in the registered (positive) direction;
  leave-one-league-out CIs from `run_trial`. Each hypothesis has one grid point; the primary is also recorded on the
  exploratory window (counted in the family).

## Headline (evaluation; first and only run per config is the result of record)
| model | edge_bits vs power close [95% CI] | bet rule CLV [95% CI], n bets | ROI flat (not primary) |
|---|---|---|---|
| Dixon-Coles | −0.0637 [−0.0733, −0.0546] | −4.97% [−5.15, −4.79], 15,454 | −7.4% [−10.2, −4.8] |
| GBDT | **−0.0336 [−0.0380, −0.0295]** | **−3.86% [−4.04, −3.69]**, 16,247 | −3.3% [−5.7, −0.9] |
| stochastic_strength | −0.1069 [−0.1159, −0.0982] | −4.97% [−5.13, −4.81], 20,135 | −6.2% [−8.3, −4.3] |

All six one-sided p = 1.00 (ROI GBDT p = 0.995). Every model forecasts worse than the close and every bet rule loses
to it, by far more than the Pinnacle open itself does: the open trails the close by only −0.0058 bits [−0.0074,
−0.0041] on the same matches.

| secondary (evaluation) | Dixon-Coles | GBDT | stochastic_strength |
|---|---|---|---|
| edge_bits, Shin close | −0.0636 | −0.0335 | −0.1068 |
| edge_bits vs multiplicative open | −0.0580 [−0.0675, −0.0486] | −0.0278 [−0.0319, −0.0238] | −0.1011 [−0.1101, −0.0925] |
| edge_bits, isotonic-calibrated | −0.0472 [−0.0553, −0.0401] | −0.0319 [−0.0358, −0.0281] | −0.0365 [−0.0450, −0.0296] |
| CLV, Shin close | −4.67% | −3.68% | −4.93% |
| CLV, `ask` top-4 per 14 days (n = 525) | −3.73% [−4.65, −2.81] | −2.23% [−3.04, −1.36] | −5.01% [−5.99, −4.00] |
| primary on exploratory window | −0.0730 bits / −3.51% | −0.0751 bits / −2.81% | −0.1024 bits / −3.46% |

Calibration helps the stochastic model most (−0.107 → −0.037 bits: its raw probabilities are badly scaled — it almost
never prices a draw edge, 1 draw bet of 20,135); it still trails the close. Scale: GBDT, the best, is ≈ 6× further from
the close than the open is, and just below the published range for public-data models (−0.03 to +0.005 bits).

## Stability (evaluation, primary metric; every cell's 95% CI excludes 0 on the negative side)
| | 19/20 | 20/21 | 21/22 | 22/23 | 23/24 | 24/25 | thirds | leagues < 0 | leave-one-league-out |
|---|---|---|---|---|---|---|---|---|---|
| DC bits | −.060 | −.053 | −.086 | −.054 | −.079 | −.056 | −.057 / −.071 / −.063 | 11/11 (best E1 −.032, worst T1 −.139) | all CI hi < 0 |
| GBDT bits | −.037 | −.037 | −.038 | −.022 | −.033 | −.035 | −.037 / −.030 / −.034 | 11/11 (best D1 −.021, worst N1 −.046) | est −.033 to −.035, CI hi ≤ −.028 |
| SS bits | −.093 | −.108 | −.112 | −.084 | −.118 | −.128 | −.103 / −.098 / −.120 | 11/11 (best E1 −.056, worst N1 −.154) | all CI hi < 0 |
| DC CLV % | −5.3 | −4.4 | −4.7 | −4.1 | −5.2 | −6.3 | −4.8 / −4.5 / −5.7 | 11/11 | all CI hi < 0 |
| GBDT CLV % | −4.2 | −3.4 | −4.0 | −3.1 | −3.7 | −4.9 | −3.7 / −3.6 / −4.3 | 11/11 (best I1 −2.8) | all CI hi < 0 |
| SS CLV % | −5.1 | −4.5 | −4.7 | −4.1 | −5.3 | −6.3 | −4.7 / −4.4 / −5.7 | 11/11 | all CI hi < 0 |

Every season, third and league is negative with a CI below zero; no league with n < 100 (smallest SC0, 1,241 matches /
792 DC bets). Per-slice tables: `data/research/r7/<model>_{edge,bet}_stability.csv`; coverage by league × season:
`<model>_coverage.csv`.

## Ladder (`pitch-edge research grade <id>`)
| rung / criterion | DC bits | GBDT bits | SS bits | DC bet | GBDT bet | SS bet |
|---|---|---|---|---|---|---|
| L0 registered before first trial, file unchanged, chain intact | pass | pass | pass | pass | pass | pass |
| L1 clean flag (pit.py + cut-vs-full rerun) | pass | pass | pass | pass | pass | pass |
| L1 provenance (dirty tree; code 48e62eaa… archived) | pass | pass | pass | pass | pass | pass |
| L1 reproduction (`run_trial` re-run) | pass | pass | pass | pass | pass | pass |
| L2 CI excludes 0 in registered direction | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** |
| L2 n ≥ 300, 2,000 draws, ≥ 30 blocks, α ≤ .05 | pass | pass | pass | pass | pass | pass |
| L3–L5 | blocked | blocked | blocked | blocked | blocked | blocked |

**All six hypotheses stop at L1: a clean, reproduced, registered negative.** BH across the family is moot (every
p = 1). Nothing here is an edge; no auditor pass was requested for a null.

## Deviations and notes (disclosed, none changes a registered quantity)
1. **The point-in-time check first failed for stochastic_strength.** `StochasticStrengthModel.predict_proba` drew its
   Monte Carlo noise from one generator advanced on every call, with shape `(n_sims, len(X))`, and filled a missing Elo
   with the mean of `X` itself. A match's probabilities therefore depended on how many matches shared its batch, which
   rows they were and how many predictions came before — so the run on the frame cut at 2020-06-30 did not reproduce the
   full run (pit.py assertion 2 failed), and the missing-Elo fill used other same-fold matches' ratings. Fixed in
   `src/pitch_edge/models/stochastic.py` before any trial (common random numbers: one fixed set of standard-normal draws
   per model scaled by the fitted σ; missing Elo filled with the training mean), with tests in
   `tests/unit/test_stochastic_strength.py` (batch independence, repeatability, a later missing Elo does not move
   earlier rows, walk-forward truncation invariance). Predictions were regenerated (`pred_stochastic_strength*.parquet`,
   20:22 local) and pit.py then passed with max |Δp| = 0.0. Predictions from before the fix were never scored. The
   registration's model description (Elo + OU Monte-Carlo noise, 2,000 sims, seed 0) still holds; the model under test
   is the fixed one, which is also what `ask` serves now.
2. `run.py` resolved the repository root one level too shallow (`HERE.parents[1]`), so it could not find the
   registrations; corrected to `parents[2]` before any trial. Before the counted run it was smoke-tested on a **scratch
   copy** of the ledger with only exploratory-window data (2015-07 → 2019-06, split into two pseudo-windows); those
   numbers were seen and nothing was changed afterwards.
3. Choices made in `run.py` before any counted run that the registration does not spell out: model probabilities are
   clipped to [1e-12, 1] before the log (the `backtest/slices.py` / engine convention). It only binds for the
   isotonic-calibrated secondary, which assigns exact zeros: 3 DC and 4 stochastic evaluation matches whose result had
   calibrated p = 0 (≈ −38 bits each). Without them the calibrated secondary is −0.0419 (DC) and −0.0299 (SS) bits.
   The `clean` flag additionally requires that no feature row differs outside a same-day referee pair. The engine and
   model sources are included in the hashed/archived code files.

## Threats to validity
- **Decision time vs model cutoff.** The registered cutoff is the fold start (≤ match date), not the open snapshot: on
  8,612 of 21,626 evaluation matches (40%) the fold started after the snapshot day, so the model knew 1–3 days of other
  results the open did not. This can only flatter the models; it cannot explain the negative result.
- **Duplicate matches in the spine.** Of 1,886 `cfmd_` rows (no Pinnacle odds, never scored), ≥ 698 are the same match
  as an `fd_` row under a differently spelled team (e.g. `MGladbach` / `Mgladbach`). They are in the training data
  (≈ 1.8% of rows), splitting a club's history across two identities in Dixon-Coles/Elo-type features. This degrades
  the models a little (and does the same in production `ask`); it cannot create a negative of this size.
- **Evaluation worse than exploratory for CLV** (all three ≈ −3.5% → −5.0%) while GBDT's bits improved (−0.075 →
  −0.034; exploratory GBDT per season −0.143 / −0.064 / −0.063 / −0.042 — a learning curve from 400 training matches).
  No sign of leakage (the smell is "better in evaluation"); the CLV drift is unexplained and is not needed for the null.
- 2023/24 has 68% Pinnacle coverage (holes in every league), so that season is under-represented; the result holds in
  every other season.
- The `ask` agent prices against Polymarket when the warehouse provides it and ranks with whichever model its slice
  evidence trusts per fixture; neither is testable on 2019–2025 history. This study grades each model's own rule
  against the Pinnacle open only.

## What this means for `ask`
`agents/qa_data.py::ASK_MODEL_HYPOTHESES` maps each `ask` model to its R7 bet-rule hypothesis, so every candidate
carries the rung of record (L1, below the L4 floor) and is labelled "model-vs-market disagreement, not a validated
edge". The models should be read as descriptions of the matches, not as prices to bet against.

## Next test
A descriptive, unregistered decomposition (evaluation): CLV = open margin paid (≈ −3.2% for all three) + fair-open →
fair-close move of the chosen outcome (DC −1.7%, GBDT −0.6%, SS −1.7%); betting every in-range outcome at the open gives
−4.6%. GBDT's choices may carry a little information about the close's direction relative to an odds-matched random
choice. Register that as a paired test (GBDT bet vs odds-band-matched placebo on the same matches, CLV difference),
alongside a re-run on a deduplicated spine; neither would make GBDT beat the close.
