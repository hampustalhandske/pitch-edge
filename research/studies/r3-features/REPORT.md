# R3 — What predicts the sharp close beyond the soft-consensus gap?

Family `closing-line-prediction`. Registrations: `research/registry/closing-line-prediction-{pi-rating,clubelo,
xg-finishing,manager-change,schedule}-v1.yaml`. Trials in `data/research/ledger.duckdb` (grade with
`pitch-edge research grade <id>`). Code: `../r3-common/`, `run.py`, `explore.py`. Outputs: `data/research/r3/`.

## Data and decision time
Lake snapshot 2026-10-20T20:53Z plus warehouse `tm_games` (read-only). 97,645 Pinnacle-close matches 2012-07 →
2025-06 in 22 divisions, plus 2,929 post-2025-07 Betfair-close matches. Decision time = the football-data
Friday/Tuesday snapshot; features use only information dated strictly before that day (asserted on every row).
Join rates (`data/research/r3/join_rates.csv`): Understat 83–84% in the top 5 leagues; Transfermarkt ids (mapped
by fixture co-occurrence, 353 fd teams) 100% for E0, SP1, I1, D1, F1, N1, SC0 and 24–41% in second tiers; Club Elo
(2022-09+) 98–99% E0/D1, 70–88% F1/I1/SP1.

**Data gap (PE-26).** football-data published no Betfair Exchange close for 2026-04-20 to 2026-05-03 and for about
half the matches of 2026-04-17, 04-19 and 05-04. The post-2025-07 Betfair-close matches also need a Pinnacle open, and
the snapshot has none after 2026-01-15, so the 2,929 post matches end on 2026-01-15 and none of the 597 main-league
matches dated 2026-04-17 to 05-04 enters the panel; 432 of those 597 also have no Betfair close (counted on
`data/research/r3/fd_odds.parquet`). The post-2025 numbers are the recorded run; the gap removes no match from them.

## Method
Target: logit move from power- (Shin as sensitivity) de-vigged Pinnacle open to close. Baseline: per-outcome OLS
on the consensus gaps g_avg, g_max (R2-Q2). Feature model adds the group's columns, fitted on the same rows.
Walk-forward: fit on all prior seasons, predict the next. Primary metric: Δ log-loss of the de-vigged close in bits
(feature model − baseline, positive = better), match-day block bootstrap. Evaluation 2019-07 → 2025-06 unless stated.

## Results
| Hypothesis | Rung (harness) | Δ bits [95% CI] | n |
|---|---|---|---|
| xg-finishing-v1 — rolling goals − xG, shrunk | L3 (criteria; not audited) | Shin +1.59e-5 [+5.8e-6, +2.7e-5]; power +1.50e-5 [+4.0e-6, +2.7e-5]; move R² +0.28 pp [+0.04, +0.54] | 10,647 |
| clubelo-v1 — Club Elo vs Pinnacle open | L1 | +3.1e-6 [−1.0e-6, +7.4e-6] (eval 2023-07 → 2025-06 only) | 5,294 |
| pi-rating-v1 (Constantinou & Fenton 2013) | L1 | −5.4e-7 [−2.5e-6, +1.6e-6]; move R² −0.07 pp | 45,418 |
| manager-change-v1 × fixture difficulty | L1 | −2.2e-7 [−1.6e-6, +1.1e-6] | 24,755 |
| schedule-v1 — rest and congestion | L1 | −9.6e-7 [−3.9e-6, +2.1e-6] | 24,749 |
| injury load | not registered | not testable point-in-time (`tm_injuries_log` first-seen ≥ 2026-09-15; `tm_injuries` survivorship-biased: 356 spells in 2012 vs 7,561 in 2025) — forward only | — |

xG detail: BH q = 0.0067; 2/3 thirds positive; leave-one-league-out 6/6 positive; within ceiling. By season
+3.0e-5, +2.6e-5, −1.8e-5, −0.1e-5, +2.6e-5, +3.5e-5. Bundesliga carries most of it (+4.7e-5), Serie A ≈ 0.
Exploratory 2015/16–2018/19 +1.8e-5 (evaluation not better than exploration). Post-2025-07 Betfair segment
−2.4e-6 [−3.2e-5, +2.4e-5], n = 894 (null).

Scale: the baseline's OOS R² of the move is 0.8–1.0%; the mean open→close KL is ≈ 0.004 bits, so xG removes
≈ 0.4% of the open→close divergence — statistically detectable, economically negligible on its own.

Pi-rating: exploratory +9.3e-6, vanished after 2019/20 — the same season football-data's Avg/Max columns change
source, which changes the baseline. Club Elo: the warehouse's xgabora Elo values don't match Club Elo's series, so
only the lake's 2022-09+ data was used.

## Audit (2026-10-20): NEEDS-WORK
`research/audits/closing-line-prediction-xg-finishing-v1.md`. Re-derived exactly with independent code; no
look-ahead found; placebos (team shuffle, match-day shuffle, season lag) all kill the effect. But the effect is
**Bundesliga-driven**: D1 supplies 57% of it from 17% of rows (+5.0e-5 [2.2e-5, 7.7e-5]); without D1 it is
+7.4e-6 [−5.3e-6, +2.1e-5], p = 0.13. The "6/6 leave-one-league-out positive" above checks signs only and does not
meet the (now explicit) STANDARDS L3(e) CI rule. Middle third 2021-08 → 2023-05 is −1.2e-5. Richer baseline (open
level + square) partly absorbs it (+1.42e-5, ISO-week CI touches 0). Current standing: **L2**, not L3.
Needed: a v2 registration defining the single-division disconfirmer in advance (or stating a Bundesliga-specific
hypothesis) and confirmation on unscored data (2025/26+ or forward).

## Threats to validity
Tiny effect sizes; team-mapping tables built after the fact; snapshot time inferred, not recorded; ladder grades
were computed before the harness fixes from its 2026-10-20 code review and are provisional until re-graded.

## Disclosures
A pre-registration dry run on a scratch ledger (exploratory windows) also printed post-2025 Betfair secondaries for
pi-rating and manager-change; nothing was changed after seeing them. The first counted run crashed before
recording (ledger path resolved relative to cwd); the stray empty ledger was deleted. Git tree dirty; a code hash
is stored in each trial's `extra_json`.

## Next
Independent audit of the xG luck re-derivation; a v2 xG specification with separate attack and defence terms;
injury features forward-only.

## Re-grade through the hardened harness (2026-10-21)

All 5 feature groups x 2 de-vig methods x 4 metrics (40 trials) were rerun through `run_trial` with `league` = division
(under 100 rows pooled), `code_files` and `n_blocks`, after a check-only pass (`R3_CHECK_ONLY=1`) showed every headline
(n, estimate, both CI bounds) equal to the earlier ledger row to 1e-9; the walk-forward parquets were byte-for-value
identical. The `data/research/r3/panel.parquet` snapshot was not rebuilt (same sha f9a147fb29d8), so the warehouse was not
needed. The reruns now make `reruns reproduce` pass.

**Rung of record: unchanged, L0.** The result of record is the first (pre-chain, legacy) row, which has no `code_hash`
and `git_dirty` true, so `provenance recorded` fails and a rerun cannot supply it.

What-if (native rerun as the result of record; not a rung of record):

| group | delta_kl_bits [95% CI], power | what-if rung | why it stops |
|---|---|---|---|
| xg-finishing | +1.50e-5 [+4.0e-6, +2.7e-5] n=10,647 | L2 | L3(e): carried by D1. Without D1 +7.8e-6 [-4.5e-6, +2.0e-5] (shin: +9.6e-6 [-1.9e-6, +2.1e-5]); 5 of 6 leave-one-out CIs exclude 0; thirds +2.7e-5 / -1.1e-5 / +3.0e-5 (2 of 3). DSR and PBO n/a (probability metric); BH q=0.007 |
| pi-rating | -5.4e-7 [-2.5e-6, +1.6e-6] n=45,418 | L1 | CI includes 0 |
| clubelo | +3.1e-6 [-1.0e-6, +7.4e-6] n=5,294 | L1 | CI includes 0 |
| manager-change | -2.2e-7 [-1.6e-6, +1.1e-6] n=24,755 | L1 | CI includes 0 |
| schedule | -9.6e-7 [-3.9e-6, +2.1e-6] n=24,749 | L1 | CI includes 0 |

The new L3(e) agrees with the xG audit (`research/audits/closing-line-prediction-xg-finishing-v1.md`: NEEDS-WORK,
Bundesliga-driven, without D1 +7.4e-6 [-5.3e-6, +2.1e-5]).
