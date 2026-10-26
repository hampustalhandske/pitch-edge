# r1-pm-convergence-v1 — Does Polymarket move toward the sharp open?

**Question.** At kickoff−T, does gap = p_PM(T) − q_open (de-vigged sharp open) predict PM's move to its last
pre-kickoff trade, move = p_last − p_PM(T)? Convergence shows as slope < 0 (full convergence −1). Registration:
`research/registry/r1-pm-convergence-v1.yaml`. Ledger: `trials.csv`.

## Data
Universe and open point-in-time rule as `r1-pm-gap-trade-v1`. p_PM(T) must be a trade ≤ 6h old; p_last is the last
trade from a bar ending by kickoff, ≤ 3h old. Evaluation 2025-11-01 → 2026-10-19; rows are outcomes (three per
match), pooled.

## Headline (evaluation)
| T | n (days) | slope [95% CI] | BH-adj p | bounce-only slope (post-hoc) | sharp open→close slope on gap |
|---|---|---|---|---|---|
| 24h | 848 (110) | −0.41 [−0.60, −0.22] | 0.002 | −0.41 | +0.56 [+0.37, +0.76] |
| 6h | 1,901 (121) | −0.15 [−0.22, −0.08] | 0.002 | −0.17 | +0.84 [+0.76, +0.92] |
| 2h | 2,096 (144) | −0.09 [−0.14, −0.04] | 0.002 | −0.15 | +0.91 [+0.85, +0.97] |

Exploratory slopes similar (−0.38, −0.16, −0.10). T−6h stability: thirds −0.16, −0.14, −0.16 (last CI spans 0);
home −0.07, draw −0.37, away −0.17; E0 alone −0.04 [−0.15, 0.07] — carried by E1, draws and smaller leagues.

## The diagnostic that changes the reading (post-hoc, not a registered trial)
Last-trade prices contain bid-ask bounce noise e, which sits in both gap and move and mechanically gives a slope of
−σe²/var(gap) with no information flow. Roll's estimator on 67,309 consecutive hourly price-change pairs (same
markets, pre-kickoff) gives σe ≈ 0.70¢, implying bounce-only slopes −0.41 / −0.17 / −0.15 at T = 24 / 6 / 2h —
equal to or larger in magnitude than every observed slope. The two-regressor fit (move ~ p_PM(T) + q_open) has
equal and opposite coefficients (−0.149 / +0.149 at T−6h) and adds nothing. Meanwhile the sharp line moves toward
PM: open→close slope on the same gap +0.56 to +0.91.

## Ladder
| Rung | Criterion | Result |
|---|---|---|
| L0 | registered | pass |
| L1 | PIT asserted; reproducible | pass (code uncommitted at sha 6d29de4) |
| L2 | CI excludes 0 in registered direction, n ≥ 300 | pass on the registered metric (all three T) |
| L3a | BH q < 0.10 across 27 family trials | pass |
| L3b/c | DSR / PBO | n/a (not a return series) |
| L3d | ceiling: the registration says slopes 0 to −0.3 are indistinguishable from bounce | fail as evidence of convergence — observed ≈ bounce-implied |
| L3e | 2/3 thirds same sign | pass (T−6h, T−2h), but not carried by E0 |

Rung: **L2 on the registered statistic, substantively null.** No evidence PM converges to the sharp open beyond
last-trade noise; information flows the other way — the football-data open is stale and the sharp close ends up
near where PM already was.

## Threats
The bounce estimate pools all pre-kickoff bars including thin early hours — a mid-price version is needed before
claiming the slope is exactly zero. Sharp-open sources are mixed (Pinnacle and Betfair).

## Next
Rerun with order-book mids (`pm_orderbook_snapshots`, 2026-04 → 2026-09); lead-lag of PM vs a live sharp quote
(R5), not football-data's Friday snapshot.
