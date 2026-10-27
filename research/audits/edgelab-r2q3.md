# edgelab second opinion: R2-Q3 bet365_open (L4(b) record)

Study: `bookmaker-line-structure-stale-soft-open-v1`, the registered achievable-book variant (`bet365_open`, θ = 0.02).
Audit of record: `research/audits/bookmaker-line-structure-stale-soft-open-v1.md`, claim (b) **REJECTED**.
Code: `research/audits/edgelab-r2q3/{build_dataset.py,pe_side.py,make_strategy.py}`; spec and every modelling choice:
`research/audits/edgelab-r2q3/strategies/pitch-edge-r2q3-bet365/notes.md`.

## What was submitted

- Rule: at the football-data open snapshot, back the outcome with the largest `log(p_pin_open × bet365_open)` if it
  exceeds 0.02 (power de-vig, 1.5× data-error cap). Flat 0.5% of NAV per bet, held to resolution. One trial, no sweep.
- Data: one synthetic binary market per match (its largest-edge outcome), ask = 1/odds, bid = 0.98 × ask, two
  identical snapshots (open hour and +1 h, the persistence assumption), no closing snapshot, no `market_type`
  feature (no Polymarket fee), grading policy commission 1 bp. Matches with < 1 h 30 min lead dropped (1,307 of 97,577).
- Window `data.range` 2012-07-27 → 2025-06-01 (includes the 2012-2019 exploratory era), `fast` profile, grading
  policy `edgelab-2026.10.3`, holdout fraction 0.2, never unlocked (`holdout_access_number` 0).
- Run `2c65a816-7731-47ef-ad23-51aee6e0da1c`, folder `runs/2026-10-24_1630_2c65a816` (local copy under
  `data/research/r2/edgelab/strategies/pitch-edge-r2q3-bet365/`).

## edgelab grade: **inconclusive** ("Insufficient evidence")

Blocking: `hac_t`, `psr`, `dsr`, `spa`. Sharpe 0.194 (upper 95% 0.654); "needs ~105 years against the 12.9 used".

| gate | status | value (threshold) | informative? |
|---|---|---|---|
| data_audit / insolvency | green / green | no critical failures; NAV never ≤ 0 | yes (data shape only) |
| hac_t | red | 0.70 (≥ 2), p = 0.24 | yes |
| psr | red | 0.759 (≥ 0.95) | yes |
| dsr | red | 0.759 (≥ 0.95), N = 1 | partly: N = 1, not 355 (EL-4) |
| spa | red | p = 0.24 (≤ 0.05), 50 reps | yes |
| pbo | amber | none: one trial, CSCV impossible | no |
| track_record | red | 112,632 bars vs MinTRL 618,680 | no: hourly-bar basis (EL-11) |
| cpcv | green | 0.429 (> 0) | no: 21 identical OOS Sharpes (EL-9) |
| holdout | amber | locked | no |
| random_entry | amber | every replay undefined | no (EL-10) |
| breakeven_cost | green | ≥ 3.0, right-censored | no: margin is in the price, costs are 1 bp |
| delay_retention | amber | 0 of 4,942 bets fillable at +1 bar | no (EL-10, two-snapshot book) |
| plateau | not evaluated | no grid | no |
| factor_alpha | not evaluated | equity only | no |
| calibration | amber | market − model Brier 4.41e-5, p = 0.36 | yes, but against the fill price (see below) |

Robustness context: cost sweep 0.5×-3× moves Sharpe 0.194 → 0.192; remove best 1/5/10 bets 0.161 / 0.060 / −0.040;
thirds 0.091 / 0.098 / 0.444; 20 noise draws all identical (0.194). Data-fit: min detectable Sharpe 1.63 at 10.3 years.

## Reconciliation (same decisions)

All pitch-edge numbers recomputed for this record from `r2q3-bet365-pe_bets.parquet` and the exported dataset;
CIs are match-day block bootstraps (B = 2000, from `r2q3-bet365-pe_side.json`).

| quantity | edgelab | pitch-edge | match |
|---|---|---|---|
| bets | 4,942 fills (all synthetic book) | 4,942; the dataset's markets with model edge > 0.02 at the ask are set-equal to `pe_bets` | exact |
| per-bet fields | 500 listed bets (EL-8 cap; 2023-04-21 → 2025-05-30) | price = 1/odds (max diff 1e-16), forecast = `p_open` (exact), outcome = `y` (all) | exact on the 500 |
| hit rate / mean entry price | 27.60% / 0.27029 | 27.60% / 0.27029 (= mean 1/odds; fills at the ask, not the mid) | exact |
| return per bet | ROI 1.035% (stake-weighted, NAV-proportional stakes, 1 bp commission) | unit-stake ROI +1.75% [−3.33, +6.79]; NAV-path replay of edgelab's sizing: 1.035%, final NAV 127.5k (edgelab CAGR 1.909% over 12.86 y implies 127.5k); stakes within 0.06% on the 500 | reconciled |
| Sharpe / t | 0.194 annual, HAC t 0.70 (hourly bars) | per-bet SR 0.0094, × √(384 bets/yr) = 0.185; iid t 0.66 | consistent |
| Brier, model vs market | 0.186810 vs 0.186854 (Δ 4.41e-5, p = 0.36) | Pinnacle open 0.186810, bet365 open ask 0.186854 (exact); synthetic mid 0.186896; **Pinnacle close 0.185751** | exact |
| CLV vs Pinnacle close | none (no close in the book) | whole window +0.94% [+0.65, +1.26] (n 4,942); 2012-07 → 2019-06 +1.30% [+0.92, +1.68] (n 2,996); **study evaluation 2019-07 → 2025-06 +0.39% [−0.12, +0.90] (n 1,946)**; first 80% +1.12% [+0.80, +1.45]; last 20% −0.27% [−1.18, +0.64] | n/a |
| CLV vs Betfair close | none | only 209 bets carry a Betfair close (2024-08-09 → 2025-06-01): −2.51% [−4.47, −0.71]; Pinnacle close on the same 209: −1.31% [−3.24, +0.41] | n/a |

The study's registered evaluation number is +0.46% [−0.06, +0.97] on n = 1,994; this dataset has 48 fewer
evaluation-window bets, attributed to the short-lead drop (not itemised; unverified).

## Why the numbers differ

- **Window.** edgelab scores 2012-2025; the study scores 2019-07 → 2025-06. bet365's CLV is significant only in the
  exploratory era (+1.30%) and decays to +0.39% (CI spans 0) in evaluation, the audit's "decaying" finding.
  edgelab's P&L-based statistics cannot see this: ROI is noise-dominated (unit CI ±5 pp) in every window.
- **Fills and the one-bar delay.** Fills equal pitch-edge's price by construction: the build copies the open quote
  to the next hour. Whether the bet365 open survived an hour is assumed, not tested; edgelab's delay test cannot
  test it because the book has no third snapshot (EL-10).
- **Synthetic 2% spread (EL-12).** Sets the mark (mid = 0.99 × ask) and the stake (cash 1.0101 × fraction); no
  effect on per-bet return. edgelab's "market" Brier uses the ask.
- **Flat 1 bp commission.** 266 currency units over the run; the bookmaker margin sits in the price, so the cost
  gates are trivially green.
- **Stake weighting.** NAV-proportional stakes give 1.04% vs 1.75% unit-stake ROI; replicated exactly by replaying
  the sizing over the same 4,942 bets.
- **No closing price.** edgelab has no CLV. Its calibration gate asks whether Pinnacle open beats bet365 open as a
  forecast (yes, by 4.4e-5, not significant). The Pinnacle close beats both by about 1.1e-3, the information this
  strategy never sees.
- **N = 1 vs 355 (EL-4).** edgelab deflates over its one-point grid, so DSR = PSR = 0.759; pitch-edge's family holds
  355 trials (`research/LOG.md` line 51). DSR only falls with N; the gate fails either way.
- **Hourly-bar Sharpe (EL-11).** 2,977 active of 112,632 bars; MinTRL and "years needed" are framed in bars and years.
  The per-bet annualised Sharpe (0.185) agrees, so the basis does not change this verdict.

## Not explained (for edgelab, unverified)

1. **Holdout coverage.** edgelab reports the holdout as locked, but its scored series covers the whole range:
   112,632 bars = 12.86 years (`live_end` 2025-06-01, `holdout_start` null), all 4,942 fills (635 fall after the
   80% split, 2022-11-05), the listed bets run to 2025-05-30, and the NAV replay reproduces its CAGR only with all
   4,942 bets. The data-fit stage uses 10.3 years (80%) while the assessment says 12.9 years used. Either the 20%
   holdout lies outside `data.range`, or the search-range restriction is not applied to the scored series for this
   provider. Filed in edgelab as EL-23 (must); edgelab's own trace finds the holdout carved out only for
   validation while inference uses the full series, so the "locked" holdout bars feed the grade. Consequence: `notes.md`'s "headline scored mostly on 2012-2022" and the
   `holdout_20pct_not_scored_by_edgelab` key in `r2q3-bet365-pe_side.json` do not describe this run. The same check
   applies to the `r2q3-max` run.
2. **CPCV value.** The 21 identical OOS Sharpes are 0.429, not the full-sample 0.194; why is unknown (part of EL-9).

## Verdict

**Consistent with the study audit's rejection of the bet365 strategy.** The decisions match bet for bet. edgelab
finds no significant return (HAC t 0.70, PSR/DSR 0.759, SPA p 0.24), and pitch-edge finds no significant
evaluation-window CLV (+0.39% [−0.12, +0.90]), a decay from the exploratory era, and a negative Betfair-close CLV
on the 209 bets that carry one. Neither engine shows an edge. L4(b) is recorded as **inconclusive** (not
`promising`, not a data-fit refusal). No rung change: bet365_open stays below L2, and the audit's REJECTED stands.
