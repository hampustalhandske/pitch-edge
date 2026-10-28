# H3 — A soft-books-only model of the sharp close, bet at the best soft open

Registration: `research/registry/soft-open-vs-betfair-gbm-close-v1.yaml` (family `soft-open-vs-betfair`, family
N 4). Code: `gbm_close.py`, `run.py`, shared `../soft-open-vs-betfair-common/sobf.py`. Counted trial: one, on the
evaluation window. Data: lake build `e3800576ec61`, football-data main-league files, pulled 2026-10-28.

**Most of this CLV is line shopping, not the model.** The same bet rule with the soft-average de-vig in place of
the model's forecast (the placebo) earns +1.43% [+0.28, +2.61] on 443 bets. The model's own rule earns +1.77%. The
roughly 0.3 pp between them was not tested. Both come from betting at football-data's Max price, the best price
over many books, which no single venue offers (E2 P5). **This is not an edge a bettor at one book could take.**

The primary test fails narrowly. Mean CLV +1.77% on 330 bets, 95% CI [+0.44%, +3.06%], one-sided p 0.0065,
against the family level at N 4 of 1 − 0.975^(1/4) = 0.0063. Rung: L1 (clean, counted null at the family level).

## Question
Can a model built only from soft-book opens predict the sharp close well enough to find best soft opens priced
above it? The model's inputs are the average, max and bet365 levels and spreads. It has no Pinnacle or Betfair
inputs, so it could run live on books football-data still publishes.

## Method
- Model: HistGradientBoosting, R6's frozen "tiny" setting, one regressor per outcome.
  - Target: the logit of the power-de-vigged Pinnacle close.
  - Features: 28 soft-open features (9 raw odds, 9 de-vigged logits, 6 spreads, 3 overrounds) and an era flag. No
    Pinnacle, Betfair, close or result column is an input (asserted).
- Folds: walk-forward by season (July to June), each season predicted from all earlier ones.
- The final model is fitted once on 2012-07 to 2024-06 and is not refitted on the evaluation window.
- Rule: q_hat is the normalised model forecast. Bet the outcome with the largest log(max_open × q_hat) when it
  exceeds θ; one bet per match, flat stakes.
- Score: CLV = log(price × p_close), with p_close the power-de-vigged Betfair Exchange close.
  - The CI is a match-day block bootstrap with 2,000 draws.
  - The evaluation window is 2025-07-01 to 2026-10-19, minus 2026-04-17 to 05-04 (PE-26; 597 rows).
  - Betfair close books are healthy: 0.07% have a book sum above 1.10.

## Selection (exploratory, out-of-fold, against the Pinnacle close)
The registered rule takes the highest total CLV with at least 1,000 bets, ties going to the lower θ. A total-CLV
criterion favours the lowest θ, and it did.

| θ | bets | mean CLV | 95% CI | total CLV |
|---|---|---|---|---|
| **0.01** | **31,088** | **+0.88%** | [+0.75, +1.01] | **275.1** |
| 0.02 | 18,264 | +1.00% | [+0.81, +1.19] | 182.1 |
| 0.03 | 12,138 | +0.80% | [+0.55, +1.06] | 97.1 |

## Headline (evaluation, θ 0.01)
| rule | n | match-days | mean CLV | 95% CI | one-sided p | flat ROI |
|---|---|---|---|---|---|---|
| model q_hat (counted) | 330 | 142 | +1.77% | [+0.44, +3.06] | 0.0065 | −1.6% |
| placebo: soft-average de-vig (descriptive) | 443 | 165 | +1.43% | [+0.28, +2.61] | — | — |

- Log-loss edge of q_hat over the soft-average de-vig, measured against the Betfair close: 1.0e-4 bits per match.
  The forecast is barely closer to the close than the plain soft average.
- Stability by calendar third: +0.05%, +2.62%, +2.20%. No division has 100 bets, so all are pooled as OTHER.
- The result is under the +3% literature ceiling.
- **Bet rate.** The rule bet on 37% of out-of-fold matches in training and 3.8% in evaluation. θ did not carry
  over across eras: football-data's Max before 2019 came from a different set of books, and the era flag does not
  absorb that.

## Ladder
| rung | status |
|---|---|
| L0 registered | yes, before any outcome was read |
| L1 clean | yes: features asserted point-in-time and soft-only; the rerun reproduces; git sha, snapshot and counts recorded |
| L2 signal | **no**: CI excludes 0 on ≥ 300 bets, but p 0.0065 is above the family level 0.0063 (STANDARDS: the family's multiple-testing level applies) |

## Threats to validity
- **Line shopping, not prediction (main).** Max-over-books CLV is positive almost by construction, as the R2-Q3
  audit's placebo showed, and the placebo here earns most of the result.
- **Train and test use different benchmarks.** The model is trained and θ selected against Pinnacle, but it is
  scored against Betfair, so θ carrying over is an assumption.
- **The evaluation window was reused.** Other families already scored rules on it.
- **The model's increment over the placebo was not registered as the test.** A model-vs-placebo test would be a
  new trial.

## Next test
None registered. A test of the model's own increment would pair each model bet with the placebo on the same
matches, at a single named venue (bet365 open), scored on forward paper bets.
