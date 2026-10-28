# r5-xv-leadlag-v1 — Does Kalshi or Polymarket lead?

**Question.** Does one venue's hourly move predict the other's next-hour move, and which venue's price carries
information about the Betfair close that the other lacks? Registration: `research/registry/r5-xv-leadlag-v1.yaml`.

## Data and n
Hourly: Kalshi mids at H−1, H, H+1 and a Polymarket trade in each of those hours — 126,796 rows, 290 days. Close:
the last hour ≤ kickoff−1h with a Polymarket book mid, an `fd_match_key` and a Betfair close — 276 outcomes, 22 days.

**Data gap (PE-26).** football-data published no Betfair Exchange close for 2026-04-20 to 2026-05-03 and for about
half the matches of 2026-04-17, 04-19 and 05-04. Of the 897 evaluation-window events with an `fd_match_key`, 132 are
dated 2026-04-17 to 05-04 and 113 of them have no close and are not scored (counted on `data/research/r5/fd.parquet`,
before the study's price filters). The numbers above are the recorded run; they exclude those matches.

## Headline
- Kalshi → Polymarket next-hour slope **0.117 [0.060, 0.178]** (thirds 0.036 / 0.231 / 0.181).
- Polymarket → Kalshi **0.054 [0.013, 0.101]** (thirds 0.012 / 0.116 / 0.085).
- Contemporaneous slope 0.36; own-lag Polymarket −0.23 (bounce), Kalshi −0.10. Per-series leave-one-out all positive.
- Close test: Kalshi information about the Betfair close b = **0.91 [0.66, 1.19]**; "Polymarket adds information"
  0.09 [−0.19, +0.34].

## Ladder
Hourly: L2 pass; L3 BH pass (q 0.0009 / 0.008), PBO 0.04, 3/3 thirds, but DSR "fail" — Sharpe statistics applied to a
slope, which STANDARDS L3(f) now marks `n/a` for predictive metrics. Rung **L2** (provisional). Close: n 276 < 300 and
22 blocks < 30 → **L1**.

## Post-hoc
By Polymarket trade count in hour H: 1 trade — Kalshi→PM 0.13, PM→Kalshi −0.01; 5–19 trades — 0.076 vs 0.073; 20+
trades — 0.22 vs 0.16 (both CIs touch 0). Information flows both ways in active hours; Kalshi's lead comes from thin
Polymarket hours (non-synchronous sampling: the Kalshi quote is at hour end, the Polymarket price is the last trade
somewhere inside the hour — registered as a threat in advance). Close test with fresh books ≤ 1h: b = 1.10 [0.37,
1.64] (n = 44); with a fresh Polymarket last trade: 0.46 [−0.00, 0.86] (bounce-biased). Inconclusive.

## Next
Lead-lag against a live Betfair/Pinnacle quote from the paper loop, with synchronised snapshots.
