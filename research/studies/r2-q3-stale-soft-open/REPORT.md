# R2-Q3 — Stale soft lines: does the best open price beat Pinnacle's fair odds, and then the close?

Registration: `research/registry/bookmaker-line-structure-stale-soft-open-v1.yaml`. Ledger: `trials.csv` (20 trials).

## Method
Decision at the football-data pre-closing snapshot (Friday or Tuesday afternoon). Fair odds at decision =
1 / power-de-vigged Pinnacle **open** (never the close). edge = log(price × p_pin_open); one bet per match on the
largest-edge outcome if edge > θ. Price source: `max_open` (an **upper bound** — books limit winners, and max
prices can be stale or erroneous) or `bet365_open`. Quotes above 1.5× the Pinnacle open are treated as data
errors. Score: CLV vs power-de-vigged Pinnacle close. Post segment 2025-07+: Betfair open as fair price, Betfair
close as benchmark.

**Data gap (PE-26).** football-data published no Betfair Exchange close for 2026-04-20 to 2026-05-03 and for about
half the matches of 2026-04-17, 04-19 and 05-04. In the post segment, 170 matches dated 2026-04-17 to 05-04 pass the
other eligibility filters (main-league file, result, Betfair open, max and bet365 open); 5 of them have no Betfair
close and are not scored. The other 427 of the 597 main-league matches in the span lack a Betfair open, so the filter
removes them regardless; 432 of the 597 have no close (counted on `data/research/r2/fd.parquet`). The post-segment
numbers are the recorded run; they exclude those matches.

## How often (evaluation)
max_open beats Pinnacle-open fair on ≥ 1 outcome in 65% of matches (26% of outcomes); 59% of those also beat the
fair close. bet365: 12% of matches (4.1% of outcomes); 54% also beat the close. Exploratory: 77% / 19% of matches.

## Headline (evaluation)
| source | θ | n | CLV | ROI |
|---|---|---|---|---|
| max | 0 | 29,512 | +0.97% [0.87, 1.07] | +0.8% [−0.9, 2.5] |
| max | 0.02 | 11,172 | **+2.08% [1.91, 2.26]** | +1.7% [−1.5, 4.9] |
| max | 0.05 | 2,104 | +3.18% [2.61, 3.77] | +6.7% [−2.2, 15.7] |
| bet365 | 0 | 5,597 | −0.15% [−0.41, 0.13] | +3.2% |
| bet365 | 0.02 | 1,994 | +0.46% [−0.06, 0.95] | +1.4% |
| bet365 | 0.05 | 445 | +1.21% [−0.27, 2.75] | +8.1% |

Exploratory: max +1.33 / +2.65 / +5.23%; bet365 +0.03 / +1.30 / +3.38% — the bet365 signal decays. Unconditional
reference (every outcome at max_open): −2.4% in evaluation, so the Pinnacle-open filter supplies the positive CLV.
The edge at open on max θ = 0.02 bets averaged 3.9%; realised CLV is 2.1% — Pinnacle moved about half-way toward
the max price. Cap sensitivity: no cap +2.12%, cap 1.25 +2.09%. Post segment (Betfair reference + benchmark): max
θ = 0.02 +0.88% [0.13, 1.60] (n = 856); bet365 −3.3% to −7.4%, suggesting the Betfair "open" is a noisy fair
reference.

## Stability (max, θ = 0.02)
Thirds +2.15 / +2.28 / +1.63%; 22/22 divisions positive (1.0% to 4.0%); leave-one-division-out [2.04, 2.13]%.
bet365 θ = 0.02 by third: +1.16\* / +0.35 / −0.72% — decaying.

## Ladder
| Rung | max_open | bet365_open |
|---|---|---|
| L0 | pass | pass |
| L1 | pass (open-only features asserted; rerun identical) | pass |
| L2 | pass: +2.08% [1.91, 2.26], n = 11,172 | fail: θ = 0.02 CI spans 0 |
| L3a family BH | pass (q ≈ 0) | — |
| L3b DSR | contested: ≈1.0 with null variance, ≈0 with cross-trial variance (inflated by vig-negative trials) | — |
| L3c PBO | 0.0 | — |
| L3d literature ceiling | flagged: at the +2% leakage-suspect line; Kaunitz et al. (2017) report positive max-odds returns that were then stake-limited | — |
| L3e sub-periods | pass | — |

Rung: max_open **L2**, as an upper-bound price, not an achievable edge; nothing is called an edge before the
auditor signs. bet365_open stays at L1.

## Threats to validity
1. Max is an upper bound across many books, including palpable errors, stake limits, and pre-2019/20 Betbrain
   books a bettor may not have been able to use.
2. The snapshot may not be simultaneous: if max was captured after the Pinnacle open, this is look-ahead.
3. The benchmark de-vig shifts CLV by about 1 pp on longshot-heavy bets (mean odds 3.7); no Shin or
   multiplicative sensitivity was registered for Q3.
4. ROI is not significant.

## Next test
Skeptic-auditor re-derivation from `odds_all`; a v2 registration with achievable single books (bet365 plus the
other named books in `odds_all`) and Shin/multiplicative benchmarks; forward paper test (L5) against live soft
prices via the-odds-api.

## Re-grade through the hardened harness (2026-10-21)

Every registered grid point of the evaluation window was recomputed from `data/research/r2/fd.parquet` as per-decision
CLV (`decision_id` = division|date|home|away, `block` = match-day, `league` = division, divisions under 100 bets pooled
as OTHER) and recorded with `pitch_edge.research.run_trial` (`run.py --harness --record`, code hashed and archived under
`data/research/bookmaker-line-structure-stale-soft-open-v1/code/`). Before recording, each point was compared with the trial imported from `trials.csv`: bet
count equal and mean equal to the 6 significant figures `trials.csv` keeps, in all 6 points; CI bounds differ by at
most 14% of a half-width (the study's multinomial match-day bootstrap and the harness's integer block draws are
different engines, so the CIs cannot match to 1e-9).

**Rung of record: unchanged, L0.** The result of record for each grid point is still the imported row, and L1 reads
`clean` and provenance from that row (no `clean` flag, no `code_hash`). The native rows below are reproduction checks
that the gates cannot credit: the imported rows are off-grid (they use the config key `source`, the registration says `price_source`) and name the metric `clv_log_mean`. The auto-selected row is the imported off-grid max_open theta=0.05 row (p=0 beats the native p=0.0005).

What-if (native reproduction as record, each point graded with `--config-hash`-style selection):

| point | CLV [95% CI] | n | blocks | what-if rung | what stops the next rung |
|---|---|---|---|---|---|
| max_open, theta 0 | +0.97% [+0.87, +1.07] | 29,512 | 1,535 | L2 | DSR 0.0 (SR variance pooled over 16 returns, flagged in the audit) |
| max_open, theta 0.02 | +2.08% [+1.90, +2.27] | 11,172 | 1,354 | L2 | DSR 0.0; within-ceiling false (at the +2% line) |
| max_open, theta 0.05 | +3.18% [+2.61, +3.77] | 2,104 | 829 | L2 | DSR 0.0; ceiling false |
| bet365_open, theta 0.02 | +0.46% [-0.06, +0.97] | 1,994 | 730 | L1 | CI includes 0 |

BH passes (q=0.014, m=347), PBO 0.0, 3/3 sub-periods positive and 22/22 leagues keep a CI above 0 with one league removed
for max theta 0/0.02. The auditor's SIGNED covers only the descriptive claim about quoted max prices.

The "what-if" rung treats the first native run as the result of record. It is not a rung of record: those runs came
after the numbers were seen.
