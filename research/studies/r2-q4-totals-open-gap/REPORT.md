# R2-Q4 — Does the consensus-gap signal work for totals (O/U 2.5) and the Asian handicap?

Registration: `research/registry/bookmaker-line-structure-totals-open-gap-v1.yaml`. Ledger: `trials.csv` (24 trials).

## Method
Pinnacle O/U 2.5 exists from 2019/20 only. Exploratory 2019-07 → 2021-06 (14,407 matches); evaluation
2021-07 → 2025-06 (30,970); post segment 2025-07+ (2,920, Pinnacle close where present).
gap = p_avg_over_open − p_pin_over_open (two-way power de-vig). Back over if gap > θ, under if gap < −θ, at
Pinnacle open; CLV vs power-de-vigged Pinnacle close. Predictability: OOS R² of the move on the logit gap.
Asian handicap: the same rule at the market line, scored only where `ah_line_close == ah_line_open` — a hindsight
selection (38% of matches have a moved line), so AH is capped at L1 and reported as feasibility only.

## Headline (evaluation, O/U)
| θ | n | CLV |
|---|---|---|
| 0 | 30,913 | −4.47% [−4.56, −4.38] |
| 0.01 | 2,885 | −4.22% [−4.50, −3.93] |
| 0.02 | 197 | −2.63% |
| 0.03 | 48 | −0.18% |
| 0.05 | 20 | +1.75% (noise) |

Every configuration with n ≥ 300 is negative; ROI at θ = 0 is −4.3%. Predictability: OOS R² +0.15%
[0.03, 0.26] (family q = 0.069), but the slope flips sign — exploratory −0.07 [−0.15, 0.00], evaluation +0.15
[0.07, 0.23]: fragile. Post segment θ = 0: −4.4% (n = 2,917). AH: CLV −2.45% (θ = 0, n = 19,250); −0.8% at
θ = 0.02 (n = 61); move-on-gap slope ≈ 0.8 in the hindsight-selected subset.

## Stability
O/U θ = 0 by third: −4.25, −4.47, −4.70%. 0/22 divisions positive.

## Ladder
L0, L1 pass; L2 fail → rung **L1 (null)**. AH capped at L1 by design.

## Threats to validity
Same snapshot-asynchrony caveat as Q2; no O/U max columns before 2019/20; no Betfair O/U close in the lake for
the post segment.

## Next test
None warranted for the totals gap. An AH test needs line-adjusted probability conversion (pricing a moved line
onto the open line), not line-equality selection.

## Re-grade through the hardened harness (2026-10-21)

Every registered grid point of the evaluation window was recomputed from `data/research/r2/fd.parquet` as per-decision
CLV (`decision_id` = division|date|home|away, `block` = match-day, `league` = division, divisions under 100 bets pooled
as OTHER) and recorded with `pitch_edge.research.run_trial` (`run.py --harness --record`, code hashed and archived under
`data/research/bookmaker-line-structure-totals-open-gap-v1/code/`). Before recording, each point was compared with the trial imported from `trials.csv`: bet
count equal and mean equal to the 6 significant figures `trials.csv` keeps, in all 5 points; CI bounds differ by at
most 6% of a half-width (the study's multinomial match-day bootstrap and the harness's integer block draws are
different engines, so the CIs cannot match to 1e-9).

**Rung of record: unchanged, L0.** The result of record for each grid point is still the imported row, and L1 reads
`clean` and provenance from that row (no `clean` flag, no `code_hash`). The native rows below are reproduction checks
that the gates cannot credit: the imported rows carry the metric name `clv_log_mean`, a native run must use the registered primary string, so the two are never compared.

What-if (native reproduction as record): **L1**. Evaluation CLV (O/U 2.5, Pinnacle open vs power-de-vigged close):
theta 0 -4.47% [-4.56, -4.38] n=30,913; 0.01 -4.22% n=2,885; 0.02 -2.63% [-3.71, -1.58] n=197; 0.03 -0.18% n=48;
0.05 +1.75% [-2.34, +7.05] n=20. No point has a CI above 0. A null/negative result.

The "what-if" rung treats the first native run as the result of record. It is not a rung of record: those runs came
after the numbers were seen.
