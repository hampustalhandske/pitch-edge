# r1-pm-accuracy-v1 — Is Polymarket's 1X2 price less accurate than the sharp book?

**Question.** Is Polymarket's (PM) 1X2 price (three binary markets, power-normalised) a worse forecast of the
result than the de-vigged Betfair Exchange close, at kickoff−1h and kickoff−24h; and is PM at kickoff−24h worse
than the de-vigged sharp *open*? Registration: `research/registry/r1-pm-accuracy-v1.yaml` (family
`polymarket-efficiency`). Ledger: `trials.csv`.

## Data, window, n
- Lake snapshot 2026-10-20T18:03Z: 5,881 PM 1X2 map rows, 1,977 football-data matches, 331,140 hourly bars,
  pulled once by `../r1-common/pull.py` into `data/research/r1/`.
- Kickoff: football-data `kickoff_time` (UK local) converted Europe/London → UTC equals PM `close_at` to the
  minute for 1,912 / 1,977 matches (96.7%). A match is kept when every `close_at` agrees with that kickoff within
  30 min (30 dropped); 32 more are dropped for not having exactly one market per outcome. 1,915 kept. Drops lean
  slightly toward 2026 and E0 (34% of drops vs 28% of kept).
- PM price at T: last trade from a bar with `hour_start + 1h ≤ T`; Yes token merged with the complement of No;
  each of the three last trades ≤ 24h old; raw sum in [0.9, 1.1].
- Benchmarks: Betfair Exchange close (power de-vig, used only when booksum ≤ 1.05); "sharp open" = Pinnacle open,
  else Betfair open.
- Windows (chosen from counts only): exploratory 2023-09-01 → 2025-10-31 (510 matches); evaluation
  2025-11-01 → 2026-10-19 (1,405 matches, 226 match-days).

## Method
Per match LL(PM) − LL(benchmark) in bits on the football-data result (PM resolution agreed with it for 100% of
exploratory matches); mean with match-day block bootstrap (5,000 draws). Positive = PM worse.

## Headline (evaluation)
| Comparison | n (days) | Δ log-loss bits [95% CI] | Brier Δ [95% CI] | BH-adj p |
|---|---|---|---|---|
| PM T−1h vs Betfair close | 1,173 (213) | +0.0029 [−0.0020, +0.0077] | +0.0013 [−0.0010, +0.0035] | 0.44 |
| PM T−24h vs Betfair close | 777 (185) | +0.0108 [+0.0016, +0.0195] | +0.0046 [+0.0002, +0.0088] | 0.037 |
| PM T−24h vs sharp open | 558 (137) | +0.0024 [−0.0019, +0.0066] | +0.0011 [−0.0007, +0.0028] | 0.44 |

Exploratory: same sign and similar size (+0.0031, +0.0093, +0.0011).

## Stability and sensitivity (evaluation)
- T−1h by thirds −0.0020, +0.0039, +0.0124 (all CIs span 0); by league E0 −0.0037, E1 +0.0033, USA +0.0149
  [−0.0003, 0.0293], pooled-small +0.0015.
- T−24h by thirds +0.0019, +0.0138, +0.0223; E0 +0.0074, USA +0.0294 [0.0048, 0.0504], pooled-small +0.0067.
  Fragile: requiring last trades ≤ 6h old gives +0.0026 [−0.0090, +0.0136] (n = 510); vs Pinnacle close −0.0020
  (n = 214).
- Pinnacle close vs Betfair close on the same 273 matches: Pinnacle worse by +0.0037 [0.0003, 0.0076] bits —
  Betfair is the right benchmark here.
- Multiplicative instead of power normalisation for PM: no change (+0.0031 at T−1h).

## Ladder
| Rung | Criterion | T−1h vs close | T−24h vs close | T−24h vs open |
|---|---|---|---|---|
| L0 | registered before scoring | pass | pass | pass |
| L1 | PIT asserted, rerun reproduces | pass\* | pass\* | pass\* |
| L2 | CI excludes 0, n ≥ 300 | fail (null) | pass | fail (null) |
| L3a | BH q < 0.10 in family (27 trials) | — | pass (0.037) | — |
| L3b/c | DSR / PBO | — | n/a (not a strategy) | — |
| L3d | within ceiling (0.10 bits) | — | pass | — |
| L3e | 2/3 thirds same sign, not one league | — | pass | — |
| L4 | auditor | — | not requested | — |

\*Ledger records git sha 6d29de4 but the study code is not yet committed.

**Reading.** An hour before kickoff PM is statistically indistinguishable from the Betfair close. A day out it
trails the close by ≈ 0.01 bits — expected, since the close carries 24 more hours of information — and is no
worse than the sharp open from about the same time. The T−24h deficit disappears once stale prices are excluded.
No PM inefficiency is shown.

## Threats to validity
Last trade is not the mid (bid-ask bounce adds noise, against PM); staleness and raw-sum filters change the
sample (864 / 1,405 at T−24h); football-data's Betfair close may not be the true last matched price; unmapped
markets are not missing at random (1,977 fd matches out of ≈ 25k PM 1X2 markets).

## Next
Score the PM mid from `pm_orderbook_snapshots` (2026-04 → 2026-09); split by pre-T volume tercile.
