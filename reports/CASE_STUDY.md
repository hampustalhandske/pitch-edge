# Case studies

This file is an index, not a report. Each study has its own write-up — the question, data and window, method,
headline with its 95% CI and n, stability checks, the ladder gate by gate, and threats to validity — at
`research/studies/<id>/REPORT.md`; independent audits are in `research/audits/`. Walk-forward backtests of the
project's own match models write a one-page `reports/<label>/CASE_STUDY.md`. Every number below is taken from
the linked report.

**Rung of record** is the rung the research ledger assigns today (`pitch-edge research status`), which can be
lower than the rung a report claims: the ledger requires provable pre-registration and a reproduced result of
record (`research/STANDARDS.md`). `none` means L0 itself fails (the registration time cannot be shown to precede
the trials). Predictive and descriptive findings top out at L4 and are never an edge; only return-kind
strategies can be.

**Headline:** display floor L4, **no validated edge**. The only L4 findings are R6's predictive closing-line
results.

## Research studies

| Study | Question | Rung of record | Headline |
|---|---|---|---|
| [r1-pm-accuracy-v1](../research/studies/r1-pm-accuracy-v1/REPORT.md) | Is Polymarket's 1X2 price a worse forecast than the de-vigged Betfair close? | none | Null at T−1h: +0.0029 bits [−0.0020, +0.0077], n=1,173; T−24h +0.0108 [+0.0016, +0.0195], n=777, gone when the last trade is ≤ 6h old |
| [r1-pm-bias-v1](../research/studies/r1-pm-bias-v1/REPORT.md) | Does Polymarket show favourite-longshot or draw bias vs the Betfair close? | none | Longshots over-priced +0.37 pp [+0.23, +0.51], n=538; favourites −0.61 pp, n=270; draw null — below the cost of crossing, not tradable |
| [r1-pm-convergence-v1](../research/studies/r1-pm-convergence-v1/REPORT.md) | Does Polymarket move toward the sharp open? | none | Negative slopes (−0.41 / −0.15 / −0.09 at T−24/6/2h) equal what bid-ask bounce alone implies; the sharp line moves toward Polymarket instead |
| [r1-pm-gap-trade-v1](../research/studies/r1-pm-gap-trade-v1/REPORT.md) | Buy Polymarket outcomes priced away from the sharp open (net of fees, spread)? | none | Every informative trial negative: best-n CLV −3.69% [−5.01, −2.41], n=75; PBO 0.63 |
| [r2-q1-close-calibration](../research/studies/r2-q1-close-calibration/REPORT.md) | Is the de-vigged sharp close calibrated, and which de-vig method is best? | L0 | Power-de-vigged Pinnacle close calibrated: FLB slope−1 +0.011 [−0.004, 0.026], n=72,538; power ≈ Shin > multiplicative |
| [r2-q2-open-consensus-gap](../research/studies/r2-q2-open-consensus-gap/REPORT.md) | Does the soft-book consensus at the open predict Pinnacle's move? | L1 | Move is predictable (OOS R² +0.65% [0.47, 0.83]) but backing at Pinnacle open loses: CLV −5.03% [−5.17, −4.89], n=45,466 |
| [r2-q3-stale-soft-open](../research/studies/r2-q3-stale-soft-open/REPORT.md) | Does the best open price beat Pinnacle's fair open, and then the close? | L2 | Max odds θ=0.02: CLV +2.08% [1.91, 2.26], n=11,172 — an upper bound on prices that cannot reliably be taken; audit signs it as a quoted-price fact only; bet365 version +0.46% [−0.06, 0.95] rejected |
| [r2-q4-totals-open-gap](../research/studies/r2-q4-totals-open-gap/REPORT.md) | Does the consensus-gap signal work for O/U 2.5 and the Asian handicap? | L1 | O/U CLV −4.47% [−4.56, −4.38], n=30,913; every configuration with n ≥ 300 negative |
| [r3-features](../research/studies/r3-features/REPORT.md) | What predicts the sharp close beyond the soft-consensus gap? | xg-finishing L2; pi-rating, clubelo, manager-change, schedule L1 | xG finishing luck +1.59e-5 bits [+5.8e-6, +2.7e-5], n=10,647, Bundesliga-driven (audit: NEEDS-WORK); the other groups null; injury load not testable point-in-time |
| [r3-strategy](../research/studies/r3-strategy/REPORT.md) | Back a soft book when it beats the model-predicted fair close? | soft-book-vs-model-close L1; bw-fair-ablation L2 (post-hoc, rejected) | bwin θ=0 CLV +1.63% [+0.87, +2.41], n=408, ROI −0.7%; the xG model adds almost nothing — it is stale bwin quotes. Not an edge |
| [r4-crowd-attention](../research/studies/r4-crowd-attention/REPORT.md) | Does pre-kickoff Bluesky/Reddit activity predict Polymarket's move? | attention-size, mention-skew, sentiment-skew L1 | All six Bluesky trials null (e.g. attention Δ OOS R² +0.00032 [−0.00089, +0.00134], n=1,555); Reddit untestable (threads open at kickoff) |
| [r5-pm-micro-flb-v1](../research/studies/r5-pm-micro-flb-v1/REPORT.md) | Does R1's favourite-longshot gap survive at the order-book mid? | L1 | Longshots +1.45 pp [+0.74, +2.06], n=39; favourites +2.78 pp, n=22; mid ≈ last trade — inconclusive (stale, sparse books) |
| [r5-pm-micro-accuracy-v1](../research/studies/r5-pm-micro-accuracy-v1/REPORT.md) | Is Polymarket's mid a worse forecast than the Betfair close? | L1 | Null and underpowered: T−1h +0.0150 bits [−0.0204, +0.0436], n=93 |
| [r5-pm-micro-convergence-v1](../research/studies/r5-pm-micro-convergence-v1/REPORT.md) | Does Polymarket converge to the sharp open on bounce-free mids? | L1 | Uninformative: n=19 / 24 / 30 over 5–10 match-days |
| [r5-xv-agreement-v1](../research/studies/r5-xv-agreement-v1/REPORT.md) | How far apart are Kalshi and Polymarket for the same outcome and hour? | L1 | Within costs: 0.22–0.57% of hours exceed the round-trip cost (5,494 events); median gap 0.5¢ |
| [r5-xv-arb-v1](../research/studies/r5-xv-arb-v1/REPORT.md) | Do riskless Kalshi/Polymarket combinations exist after fees? | L2 (rejected) | +1.32¢ [+0.58, +2.21], n=307 is a stale-book artefact: fills on a newly observed book −3.27¢ [−4.47, −2.16], n=54 |
| [r5-xv-leadlag-v1](../research/studies/r5-xv-leadlag-v1/REPORT.md) | Does Kalshi or Polymarket lead? | L1 | Both lead each other (0.117 vs 0.054 next-hour slopes, n=126,796); Kalshi's lead comes from thin Polymarket hours — inconclusive |
| [r6-cross-book-close](../research/studies/r6-cross-book-close/REPORT.md) | How much better does every book's open predict the sharp close than the consensus gap, and which books are informative? | cross-book-linear, cross-book-gbm, book-informativeness L4 (audit signed) | GBM +6.61e-5 bits [+5.75e-5, +7.41e-5], n=45,467 (≈ 2% of open→close divergence); VC and Betfair informative, BW, IW, 1xBet stale. Predictive, not a strategy |
| [r7-outcome-models](../research/studies/r7-outcome-models/REPORT.md) | Do the `ask` models (Dixon-Coles, GBDT, stochastic strength) beat the de-vigged Pinnacle close, as forecasts and through the engine's bet rule? | all six `outcome-models-*` L1 (clean null) | Every model loses: GBDT −0.0336 bits [−0.0380, −0.0295], DC −0.0637, stochastic −0.1069, n=21,626 (2019/20–2024/25); bet-rule CLV −3.86% (GBDT, n=16,247) to −4.97%; negative in every season and league |

Registered with no report yet: the forward tests `bookmaker-line-structure-live-soft-vs-sharp-v1` / `-v2`
(L0, collecting in the paper loop).

## Backtests of the match models

| Label | What it covers | Headline |
|---|---|---|
| [main](main/CASE_STUDY.md) | `pitch-edge backtest --label main`, early Pinnacle prices, CLV vs Pinnacle close. The roster in this run (gru_sequence, transformer_sequence, gbdt_mkt, dixon_coles, gbdt) differs from today's `available_models()` | Every model loses to the market: edge_bits −0.037 (gbdt_mkt) to −0.278 (gbdt) |
