# r5-xv-arb-v1 — Riskless Kalshi/Polymarket combinations after fees (paper)

**Question.** Do riskless combinations exist after taker fees? *cross*: Yes on one venue + No on the other for the
same outcome; *dutch*: all three Yes (or all three No), each leg on its cheaper venue, both venues quoting every
outcome. The best combination at H fires if edge > θ and is executed with the same legs at H+1h quotes.
Registration: `research/registry/r5-xv-arb-v1.yaml`. (One design change came from exploration — both venues must
quote every outcome — and was written into the registration before any evaluation run.)

## Data
Primary pricing: true Polymarket book (coverage 04-09 → 06-10 and 09-29). Secondary: Polymarket at last trade ±
max(0.5¢, half the high−low) over the whole evaluation window.

## Headline (P&L per unit payout, evaluation, book pricing)
| Type | θ | Mean P&L | n (events) |
|---|---|---|---|
| cross | 0 | +0.18¢ [−0.12, +0.47] | 1,068 |
| cross | 0.01 | +1.32¢ [+0.58, +2.21] | 307 (112) |
| cross | 0.02 | +2.48¢ [+0.91, +4.44] | 131 |
| dutch | 0 | +0.18¢ [−0.35, +0.69] | 332 |
| dutch | 0.01 | +0.90¢ [−0.12, +1.78] | 147 |
| dutch | 0.02 | +1.96¢ [+0.14, +4.02] | 60 |

Proxy pricing is negative at every θ (cross −1.24¢ [−1.53, −0.96] n = 549 to −2.72¢; dutch −1.38¢ to −2.62¢).
Thirds for cross θ = 0.01: 2.21 / 0.93 / −0.17.

## Ladder (cross θ = 0.01)
L2 pass; L3: BH q = 0.0009 pass, **DSR 4e-14 fail**, PBO 0.05 pass, ceiling pass, thirds 2/3, league criterion not
evaluated. Harness rung **L2** (provisional — graded before the harness fixes).

## Post-hoc decomposition (not trials) — why this is not an edge
Median Polymarket book age at signal: 9.8h (the archive holds ≈ one snapshot per market per 4h). Fills that reuse the
same Polymarket snapshot: +2.30¢ [+1.56, +3.27] (n = 253). Fills on a newly observed Polymarket book: **−3.27¢
[−4.47, −2.16]** (n = 54). Both books ≤ 1h old: −1.31¢ (n = 14). A stale book both creates the signal and "fills" it
an hour later at a price nobody may have been offering.

**Reading.** An artefact of stale Polymarket quotes; no tradable cross-venue arbitrage is shown. Not an edge.

## Threats
No depth at the touch; venue resolution-rule risk (33 Kalshi outcomes settled `scalar`); Kalshi per-order cent
rounding ignored; capital lock-up ignored.

## Next
Forward test with live books from both venues (P1 paper loop), quotes ≤ 5 minutes old.
