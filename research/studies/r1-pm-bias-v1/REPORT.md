# r1-pm-bias-v1 — Favourite-longshot and draw bias of Polymarket vs the sharp close

**Question.** Against the de-vigged Betfair Exchange close, does Polymarket at kickoff−1h over-price longshots
(close probability < 0.20), under-price favourites (> 0.60), or mis-price the draw? Registration:
`research/registry/r1-pm-bias-v1.yaml`. Ledger: `trials.csv`.

## Data, window, n
Universe, kickoff rule, PM snapshot rule and windows as `r1-pm-accuracy-v1`. Evaluation 2025-11-01 → 2026-10-19:
1,174 matches, 213 match-days, three outcome rows each; PM power-normalised.

## Method
Per outcome gap = PM probability − Betfair close probability; mean per segment with match-day block bootstrap
(5,000 draws), one-sided in the registered direction.

## Headline (evaluation)
| Segment | n outcomes (days) | mean gap, pp [95% CI] | BH-adj p | exploratory |
|---|---|---|---|---|
| Longshots (q < 0.20) | 538 (160) | +0.37 [+0.23, +0.51] | 0.002 | +0.36 [+0.21, +0.50] (n = 271) |
| Favourites (q > 0.60) | 270 (130) | −0.61 [−0.92, −0.32] | 0.002 | −0.43 [−0.69, −0.15] (n = 134) |
| Draw | 1,174 (213) | +0.01 [−0.05, +0.06] | 1.0 | −0.13 [−0.22, −0.03] |

Gap by close-probability bucket (home/away, evaluation, pp) — monotone:
| bucket | 0–.1 | .1–.2 | .2–.3 | .3–.4 | .4–.5 | .5–.6 | .6–.7 | .7–1 |
|---|---|---|---|---|---|---|---|---|
| gap | +0.30 | +0.33 | +0.26 | +0.06 | −0.12 | −0.29 | −0.51 | −0.77 |

Draws: no mean bias but compression toward ≈ 0.25 (post-hoc, not a trial): draws priced 0.1–0.2 by the close
+0.49 pp [0.28, 0.72]; priced 0.3–0.4 −0.32 pp [−0.43, −0.20].

## Stability (evaluation)
- Longshots: all three thirds positive with CIs excluding 0 (+0.33, +0.25, +0.67); E0 +0.29, pooled-small +0.39.
- Favourites: thirds −0.30, −0.46, −1.46; only the last excludes 0.
- De-vig sensitivity (post-hoc): with the Betfair close de-vigged multiplicatively, longshot +0.23 [0.09, 0.37],
  favourite −0.33 [−0.64, −0.04] — the power method takes more margin from longshots and explains ≈ 40% of the
  measured bias. Raw unnormalised PM: longshot +0.32, favourite −0.64.

## Ladder
| Rung | Criterion | Longshot | Favourite | Draw |
|---|---|---|---|---|
| L0 | registered | pass | pass | pass |
| L1 | PIT asserted, reproducible | pass\* | pass\* | pass\* |
| L2 | CI excludes 0, n ≥ 300 | pass | fail (n = 270) | fail (null) |
| L3a | BH q < 0.10 (27 family trials) | pass | (pass) | — |
| L3b/c | DSR / PBO | n/a (descriptive) | | |
| L3d | ceiling (≈ 5 pp) | pass | | |
| L3e | 2/3 thirds, not one league | pass | | |
| L4 | auditor | not requested | | |

\*Study code uncommitted at ledger sha 6d29de4.

**Reading.** A small favourite-longshot bias relative to the Betfair close: longshots ≈ 0.2–0.4 pp too high,
strong favourites ≈ 0.3–0.8 pp too low, depending on de-vig. Not tradable: smaller than the cheapest crossing
cost (half a 1¢ tick plus the 0.05·p·(1−p) taker fee ≈ 1.1¢ at p = 0.15).

## Threats to validity
Last-trade prices, not mids: retail takers buying longshot Yes print at the ask (Roll noise SD ≈ 0.7¢), which
alone could produce +0.3 pp — attack this first. Size depends on de-vig. Bucketing by close probability conditions
on a hindsight benchmark (fine for description, not a decision rule).

## Next
Repeat with order-book mids from `pm_orderbook_snapshots` (2026-04 → 2026-09); if the gap vanishes at the mid it
was microstructure.
