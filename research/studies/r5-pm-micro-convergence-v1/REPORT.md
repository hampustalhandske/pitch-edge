# r5-pm-micro-convergence-v1 — Bounce-free convergence of Polymarket toward the sharp open

**Question.** R1's convergence slopes (−0.41 / −0.15 / −0.09) equal what last-trade bounce implies. On mids (no
bounce), is the slope still negative? Registration: `research/registry/r5-pm-micro-convergence-v1.yaml`.

## Data and n
Each row needs a book at T ≤ 6h old, a last book before kickoff ≤ 3h old, spread ≤ 5¢, and a sharp open known by T.
The sharp open exists for only 27% of 2026 matches → n = **19 / 24 / 30** over 5 / 6 / 10 days.

## Method
OLS slope of move = mid(last) − mid(T) on gap = mid(T) − q_open, scored as mean influence values in `run_trial`;
exact block CIs stored alongside.

## Headline
| T | Slope [influence CI] | Exact block CI | Last-trade slope, same rows | Sharp open→close slope |
|---|---|---|---|---|
| 24h | −0.16 [−0.81, +0.33] | [−0.48, +1.10] | −0.11 | +0.32 |
| 6h | −0.14 [−0.24, +0.01] | [−0.31, +0.05] | −0.55 | +1.07 |
| 2h | −0.08 [−0.75, +0.86] | [−1.01, +0.69] | −0.47 | +1.36 |

## Ladder
L0, L1 pass; L2 fail everywhere. **Rung L1 — uninformative** (the bootstrap is barely defined with 5–10 blocks).

## Next
A bounce-free convergence test on Kalshi mids against a live sharp quote (paper loop data).
