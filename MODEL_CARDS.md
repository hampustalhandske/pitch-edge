# Model cards

Machine-readable cards are written on every run to `data/backtest/main/model_card_<model>.json` (local,
gitignored — `reports/main/CASE_STUDY.md` is the public write-up); this page is the human summary of the
2026-09-06 run and predates the removal of the GNN/in-play models from the registry.
All match-outcome models share the same walk-forward folds, the same isotonic post-calibration (fitted on
realised past folds only) and the same staking rules, so their numbers are directly comparable.

Common training window: 11 European divisions (E0, E1, D1, SP1, I1, F1, N1, P1, B1, SC0, T1), 2015-07-01 →
2026-09-03, 41,939 matches, 90 walk-forward folds (45-day retrain), 40,894 out-of-sample predictions each.

| | dixon_coles | gbdt | gbdt_mkt | gru_sequence | transformer_sequence |
|---|---|---|---|---|---|
| family | bivariate Poisson + τ low-score correction, exponential decay (ξ = 0.0018/day), one fit per league, 730-day window | histogram GBDT (scikit-learn HGB backend on this machine; LightGBM when importable), 300 iters, depth 4, lr 0.03 | same + early no-vig market probabilities (`mkt_home_p/draw/away`, overround) | PyTorch Lightning GRU (hidden 32) over each team's last 10 matches; step features gf/ga/shots/sot/home flag/points/days gap; + Elo gap | same recipe, `nn.TransformerEncoder` (2 layers, 4 heads, learned positions, mean pooling) |
| training | MLE (L-BFGS-B, bounded) | gradient boosting | gradient boosting | time-ordered 85/15 split inside each fold; AdamW lr 5e-4, ReduceLROnPlateau, grad-clip 1.0, 25 epochs; early stopping (patience 6) only when the validation slice has ≥ 200 rows; **temperature scaling** (softening only, T ∈ [1, 2.5], ≥ 300 val rows) — a sharpening T fitted on 64-row slices was the cause of a worse-than-uniform first run | same |
| features | teams, goals, dates | 58 cols: Elo (ours + CFMD), rolling 5/10 form (goals, shots, SoT, corners, points), rest/congestion, travel km, fatigue index, referee bias, kickoff weather, league id | 62 cols (+ 4 market) | sequences + elo_diff | sequences + elo_diff |
| leakage checks | trained strictly before fold start; per-league fit | rolling stats shifted one match; Elo sequential; closing odds never used; all-NaN/constant columns dropped per fit | early price only (`PS*` when present, else market-average) — never `PSC*` | sequences strictly before fixture date; validation slice is the most recent 15 %, never shuffled; `observe()` only adds realised results | same |
| log-loss (market 0.9716; 2018→ slice market 0.9743) | 1.0084 | 0.9975 | 0.9885 | 1.0418 (2018→ slice, 29,240 preds) | 1.0364 (2018→ slice, 29,240 preds) |
| multiclass Brier (market 0.5778) | 0.596 | 0.5936 | 0.5868 | — | — |
| known weaknesses | no player information; per-league fit ignores continental form | weather feature slightly harmful out of sample (ablation −0.0011); referee feature inactive on the fallback spine | inherits the market's information — good log-loss, no *independent* edge by construction | ~10 matches of history is thin; under-performs trees on aggregate features | same, plus attention needs more data than the folds provide |
| intended use | baseline every other model must beat | primary pre-match model | "does anything beat the price?" test | deep-learning comparison point | attention-vs-recurrence comparison |
| not for | staking decisions without calibration | — | claiming edge (it *is* the market) | — | — |

## Phase 5 feature additions (all models that read the feature store)

Twelve pre-match columns were added to `BASE_FEATURES` on 2026-09-06; the walk-forward protocol is unchanged.

| group | columns | leakage rule | source |
|---|---|---|---|
| `wiki_attention` | `pv_home_anom`, `pv_away_anom`, `pv_home_z`, `pv_away_z`, `pv_diff` | for a match on day D only views ≤ D-1 are used (final at 00:00 UTC on D); baseline = days D-29 … D-2; match-day views never enter | `data/alt/wikipedia_attention.py` |
| `rotation_load` | `rot_home_minutes_7d`, `rot_away_minutes_7d`, `rot_home_days_since_any`, `rot_away_days_since_any`, `rot_home_midweek_cup`, `rot_away_midweek_cup`, `rot_load_diff` | appearance minutes dated ≤ D-1 only; games ≤ D-1 only | `features/context.py` (Transfermarkt open extract) |
| `referee` (activated) | `ref_cards_per_game`, `ref_home_bias` (unchanged definitions) | the referee name is joined from Transfermarkt games on (date, home club, away club); tendency is the expanding mean over strictly prior matches, ≥ 5 prior | `data/alt/referee.py` + `features/context.py` |

Ablation verdicts for these groups are in `CASE_STUDY.md` Result 7; the dossier's section 1 quotes the model
card's feature count and leakage checks verbatim.

### Fixture predictions used by Match Intel

For an unplayed fixture the dossier quotes a `fixture_predictions` row written by `pitch_edge.intel.predict`:
the named model is fitted once on the persisted feature store, each side's latest pre-match snapshot is
carried forward (the `upcoming_fixture_frame` convention), and the raw probability is stored **without** fold
calibration and labelled as such. It is never used for staking, never enters a backtest, and always appears
next to the backtest evidence for that model.

## player_embedding_gnn

* GraphSAGE (PyTorch Geometric), 2 layers, 16-d embeddings; nodes = players who acted or received a pass
  for a team in a match; 13 node features (passes, completion, progressive passes, shots, xG, pressures,
  carries, dribbles, recoveries, interceptions, mean x/y, under-pressure rate); edges = completed passes.
* Objective: link prediction (positive edges vs random negatives) + team xG-share head; 40 epochs. A GAT
  encoder (`conv="gat"`) is available on the same objective.
* Evaluation: 15 % of each graph's edges are held out before training; **held-out-edge AUROC** is reported
  in `data/artifacts/gnn_card.json` (torchmetrics) and on the dashboard — the embedding is measured, not assumed.
* Data: StatsBomb Open Data, 149 matches (La Liga 2019-21, PL 2015-16, Bundesliga 2023-24, WC 2022, Euro 2020) → 1,829 players.
* Use: "find players in a similar passing role"; team-strength descriptors. Not a rating of quality.
* Limits: open-data coverage is a handful of competitions; embeddings drift with formation/role.

## inplay_gru

* GRU (hidden 32) over 5-minute state vectors: minute fraction, score diff, xG diff, red-card diff (parsed as 0 for now),
  shots in last 10', Elo prior; logistic baseline on the same states for comparison. 40 epochs.
* Data: StatsBomb event streams for the matches above → 3,494 state rows.
* Limits: no Betfair in-play line is available to test "is the market already right"; red cards not parsed.

## Calibration of all cards

Reliability diagrams and monthly Brier trends (model vs market) are on the dashboard's **Calibration** tab
for every model and refresh with each run.
