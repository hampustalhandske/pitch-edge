# R3 strategy — Back a single soft book when it beats the model-predicted fair close

Registrations: `research/registry/closing-line-prediction-soft-book-vs-model-close-v1.yaml` and the post-hoc
`closing-line-prediction-bw-fair-ablation-v1.yaml` (counted in the family). Code: `run.py`, `ablation.py`.
Outputs: `data/research/r3/strategy/`.

## Rule
Registered after the xG feature group passed L2 on the move, using the xG closing-line model. Books pre-specified
from open-price coverage only: bet365 (B365), bwin (BW), William Hill (WH). Grid: book × θ ∈ {0, 0.02}. At the
football-data open snapshot, back an outcome when log(price × model fair-close probability) > θ. Score: CLV vs the
power-de-vigged Pinnacle close (Shin as sensitivity). Evaluation 2019-07 → 2025-06.

**Data gap (PE-26).** football-data published no Betfair Exchange close for 2026-04-20 to 2026-05-03 and for about
half the matches of 2026-04-17, 04-19 and 05-04. The post-2025 Betfair rows come from the R3 panel, which ends on
2026-01-15 because it requires a Pinnacle open; none of the 597 main-league matches dated 2026-04-17 to 05-04 is in it
(432 of them also have no Betfair close, counted on `data/research/r3/fd_odds.parquet`). The post-2025 numbers are the
recorded run; the gap removes no match from them.

## Results
| book, θ | CLV [95% CI] | n | note |
|---|---|---|---|
| BW, 0 | +1.63% [+0.87, +2.41] | 408 | Shin close +1.41%; ROI −0.7% [−13, +13]; 3/3 thirds and 5/5 leagues positive; post-2025 Betfair +0.33% [−0.99, +1.80], n = 24 |
| B365, 0 | +0.37% [−0.47, +1.28] | 398 | |
| WH, 0 | +0.09% [−1.95, +2.06] | 200 | |
| any, θ = 0.02 | — | 44–84 | all below min_n |

Rungs: the harness auto-selects the lowest-p grid point, BW θ = 0.02 (n = 44), so the hypothesis grades **L1**.
The BW θ = 0 point graded with `--config-hash 12f7ce51…` is **L2**; L3 fails on DSR = 0.03 — computed with SR
variance pooled across the family's mixed metrics (Δ-bits and CLV), the artefact flagged in the R2-Q3 audit and the
harness review; with null variance 1/n it would be 0.99. PBO not evaluable (one block common to all grid points).
Leave-one-league-out degenerate (every league < 100 bets, pooled).

## Post-hoc ablation (`bw-fair-ablation-v1`, registered after seeing BW)
- Baseline-model (no xG) fair close: +1.51% [+0.73, +2.22], n = 375 — 83% the same bets. L2 (DSR fail).
- De-vigged Pinnacle open as fair price: +0.87% [+0.28, +1.42], n = 710.
On the bets, bwin's price was 3.5% above Pinnacle's open; bwin shortened its own price 4.8% by its close;
Pinnacle moved only 0.3% toward it. This is R2-Q3's stale-soft-line effect at one named book — the xG model adds
almost nothing. The auditor's quote-sanity filter (drop quotes above the aggregator Max, 7% of bets) gives
+1.41% [+0.62, +2.17] (diagnostic, not a counted trial).

## What a sceptic should attack first
1. Availability: per the R2-Q3 audit placebo, positive CLV is almost mechanical once a quote above fair exists;
   only a forward test can show the bwin quote was takeable, and bwin limits winning accounts.
2. Simultaneity: no per-row timestamp; if BW's pre-closing quote predates the snapshot, day-before features could
   lead it slightly.
3. Selection: the harness picks the lowest-p grid point; a v2 must pre-commit the selection rule.

## Verdict
Not an edge. A candidate for the forward test `bookmaker-line-structure-live-soft-vs-sharp-v1` (live soft vs sharp
at the same timestamp, quote-sanity rule), already registered and collecting.

## Disclosures
Between the B365 and BW/WH runs `run.py` was changed to make recording idempotent and skip single-match-day
segments (no scoring change; those trials carry different code hashes). The YAML `registered_at` (21:20Z) is later
than the ledger registration (21:07:40Z); the ledger is authoritative.

## Re-grade through the hardened harness (2026-10-21)

The 20 recorded (book, theta, metric) trials were rerun through `run_trial` (`R3_REPRODUCE=1`) with `league`
(division, under 100 rows pooled) and `code_files`, after a check-only pass showed each headline equal to the earlier
row to 1e-9. One post-Betfair segment (B365 theta 0.02, n=1 match-day) cannot be bootstrapped and was not recorded,
as before.

**Rung of record: unchanged, L0** (the legacy result of record has no `code_hash`; `provenance recorded` fails and a
rerun cannot supply it).

What-if (native rerun as the result of record; not a rung of record): auto-selected BW theta 0.02 (+5.63% [+2.95, +8.03],
n=44) is L1 (n < 300); BW theta 0 (+1.63% [+0.87, +2.41], n=408) is L2, L3 fails on DSR 0.016 (SR variance pooled over
8 returns, the artefact already noted above) and leave-one-league-out is not evaluable (division pooling leaves a single
league, so the registration would need `single_league` or a per-book league rule); BH q=0.0013, PBO 0.108, 3/3 thirds
positive. B365 and WH theta 0 and 0.02 have CIs that include 0 (WH theta 0.02 at the 0.05 edge) and sit at L1. The
conclusion stays the one above: not an edge.
