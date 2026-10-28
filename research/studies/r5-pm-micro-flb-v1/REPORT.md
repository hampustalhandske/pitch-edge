# r5-pm-micro-flb-v1 — Does R1's Polymarket favourite-longshot gap survive at the order-book mid?

**Question.** R1 (`r1-pm-bias-v1`) found Polymarket longshots over-priced by +0.37 pp and favourites under-priced
by 0.61 pp at kickoff−1h on last trades, vs the de-vigged Betfair close. Does it survive at the order-book mid (if
it vanishes, it was microstructure)? Registration: `research/registry/r5-pm-micro-flb-v1.yaml` (family
`polymarket-efficiency`).

## Data and n
R1 snapshot 2026-10-20T18:03Z (universe, kickoff, de-vig, last trades). `pm_orderbook_snapshots` pulled
2026-10-20T21:10Z: 125,702 pmxt rows (2026-04-09 → 06-10) + 294 clob rows (09-29) — about one snapshot per market
every 4h; median book 6.6h old at T−1h. 454 R1 matches kick off in the window; 303 inside pmxt coverage; 185 with
three usable books (two-sided, spread ≤ 5¢, age ≤ 24h, raw sum 0.9–1.1); **93** also have a valid Betfair close
(missing for 44%). 39 longshot outcomes (18 days), 22 favourite outcomes (17 days).

**Data gap (PE-26).** football-data published no Betfair Exchange close for 2026-04-20 to 2026-05-03 and for about
half the matches of 2026-04-17, 04-19 and 05-04. Of the 185 matches with three usable books, 92 lack a valid close,
and all 92 are dated 2026-04-17 to 05-04 (167 of the 454 matches in the book window are, and 132 of those have no
close; counted on `data/research/r1/fd.parquet` through the study's loader). The numbers above are the recorded run;
they exclude those matches.

## Method
Per outcome s·(p_mid − q_close), s = +1 for q < 0.2, −1 for q > 0.6; p_mid power-normalised. Match-day block
bootstrap (5,000) via `run_trial`. Secondaries on the same rows: last trade, paired last − mid, depth-1¢ microprice.

## Headline (pp)
| Segment | Mid | Last trade (same rows) | Last − mid | Microprice |
|---|---|---|---|---|
| Longshots | +1.45 [+0.74, +2.06] | +1.29 [+0.54, +1.99] (n = 35) | −0.02 [−0.35, +0.29] | +1.48 |
| Favourites | +2.78 [+1.48, +4.03] | +2.74 [+1.36, +4.15] (n = 20) | +0.23 [−0.55, +1.15] | +2.85 |

Draw gap at mid −0.13 [−0.42, +0.15] (n = 93). Thirds: longshots 1.62 / 2.03 / 0.51; favourites 3.56 / 3.13 / 1.27.
No division has ≥ 20 rows. Post-hoc by oldest-book age: ≤ 3h +1.26 (n = 27); 3–6h +3.10; 6–12h +3.33; 12–24h +1.18.

## Ladder
L0 pass · L1 pass (code 2cf556c3f197 archived) · L2: CI excludes 0 but n 39 < 300 and 18 blocks < 30 → fail ·
L3 blocked (BH q would be 0.005; favourites breach the registered 2 pp ceiling). **Rung L1.**

## Reading
Mid ≈ last trade, so the microstructure explanation is not supported; the gap is ≈ 4× R1's at both prices and this
sample cannot explain R1's number.

## Threats
Stale books combined with bucketing on the hindsight close (regression to the mean); tiny end-of-season sample;
Betfair close coverage not missing at random. `registered_at` was typed as 21:40Z; the ledger registration is
21:11:31Z, before the trials.

## Next
v2 once the pmxt archive reaches 2026-08-10 and the live poll resumes: books ≤ 1h old, last-trade age matched.
