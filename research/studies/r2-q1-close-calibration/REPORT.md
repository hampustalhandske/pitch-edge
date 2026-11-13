# R2-Q1 — Is the de-vigged sharp close calibrated?

Registration: `research/registry/bookmaker-line-structure-close-calibration-v1.yaml` (sha256 0a70f757…), family
`bookmaker-line-structure`. Code: `run.py` + `../r2-common/`. Ledger: `trials.csv` (254 trials). Outputs: `data/research/r2/q1/`.

## Question
Is the de-vigged Pinnacle 1X2 close (and Betfair Exchange close, 2024-08+) free of favourite–longshot (FLB)
and draw bias, per division with BH across divisions; which de-vig method is best calibrated out of sample?

## Data / window
Lake `fd_matches` snapshot 2026-10-20 17:56:52Z, all 38 divisions. Exploratory 2012-07 → 2019-06 (82,670
matches / 2,385 match-days); evaluation 2019-07 → 2025-06 (72,538 / 2,034 days); Betfair close 2024-08 → 2026-10
(25,878, evaluation only).

**Data gap (PE-26).** football-data published no Betfair Exchange close for 2026-04-20 to 2026-05-03 and for about
half the matches of 2026-04-17, 04-19 and 05-04. Of the 1,029 matches in the Betfair-close window dated 2026-04-17 to
05-04, 863 have no close and are not scored (counted on `data/research/r2/fd.parquet`; the study's other filter, a
result, removes none of them). The numbers above are the recorded run; they exclude those matches.

## Method
FLB = OLS slope b in y = a + b·p (home+away rows; b > 1 ⇒ FLB). Draw bias = mean(y_draw − p_draw). De-vig
comparison = paired log-loss difference. Match-day block bootstrap, 2,000 draws; BH q = 0.10 within test across
divisions and family-wide. No decision features (targets only).

## Headline (evaluation, Pinnacle close)
| | multiplicative | power | Shin |
|---|---|---|---|
| FLB slope−1 | +0.050 [0.034, 0.066] | +0.011 [−0.004, 0.026] | +0.021 [0.006, 0.037] |
| draw bias (pp) | +0.05 [−0.25, 0.38] | +0.37 [0.06, 0.69] | +0.24 [−0.06, 0.57] |
| log-loss | 1.00120 | 1.00093 | 1.00096 |
| ECE | 0.0045 | 0.0015 | 0.0018 |

The power-de-vigged close is calibrated; the multiplicative FLB is a de-vig artefact. Power (the registered
exploratory winner) beats multiplicative out of sample (−0.000266 nats [−0.000406, −0.000123]) but not Shin
(Shin − power +0.000031 [−0.000021, 0.000082]) → power ≈ Shin > multiplicative; "power best" is not confirmed.
Betfair close (power): FLB +0.015 [−0.010, 0.040], draw +0.19 pp [−0.35, 0.73], both null; on 16,538 overlapping
matches its log-loss vs Pinnacle is +0.00069 [−0.0006, 0.0028] — equally sharp.

## Stability
FLB by evaluation third (power): −0.020, +0.034\*, +0.015 (unstable). Draw: +0.24, +0.21, +0.64\* pp (weak).
Exploratory: FLB +0.012 [−0.003, 0.027]; draw +0.15 [−0.14, 0.45]. Per division (evaluation, power): CHN
+0.129 [0.063, 0.195] (q = 0.004) and T1 +0.152 [0.061, 0.239] (q = 0.017) survive family BH; T1 has the same sign
in exploratory (+0.076) and on the Betfair close (+0.175), CHN does not. No division has a significant draw bias
in evaluation; I2's exploratory draw bias did not replicate.

## Ladder (calibration study, not a strategy)
| Rung | Result |
|---|---|
| L0 | pass — registered before first run |
| L1 | pass — no features; rerun identical |
| L2 analogue | pooled power: fail (calibrated); T1/CHN FLB: pass |
| L3 analogue | T1: family BH pass and 3/3 segments same sign; draw bias unstable |

Pooled measures: L1 null. T1 FLB is a market-structure finding, not a strategy (exploiting it needs the close).

## Threats to validity
The linear slope can hide tail bias (see `reliability_*.csv`); extra-league "close" columns are assumed to be
closing prices; T1/CHN may reflect data or integrity quality rather than pricing.

## Next test
T1 favourite backing at the Betfair close net of commission (own registration); report Shin and power as
benchmark de-vig methods.

## Re-grade through the hardened harness (2026-10-21)

Not run through `run_trial`; rung of record stays **L0** (imported rows). The registration cannot be scored as written:
`primary_metric` names three different statistics in one string (`run_trial` takes one metric name, which must equal the
registered string), the FLB slope is an OLS coefficient and not a mean of per-decision values, and the grid is the
Cartesian product of unrelated axes (benchmark x devig x per_division_tests x devig_pairwise = 36 points, none of which
identifies a statistic). All 254 imported rows are off-grid. Inventing a mapping would change what a trial means, so none
was recorded. A v2 registration with one hypothesis per statistic (draw bias, FLB slope, paired log-loss difference) is
needed before this study can be graded; it makes no edge claim.
