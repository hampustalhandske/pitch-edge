# r5-pm-micro-accuracy-v1 — Is Polymarket's mid a worse forecast than the Betfair close?

**Question.** At the order-book mid, is Polymarket's 1X2 forecast worse than the Betfair close at T−1h and T−24h?
(R1 on last trades: null at T−1h, +0.0108 bits at T−24h.) Registration:
`research/registry/r5-pm-micro-accuracy-v1.yaml`.

## Data and n
Universe and filters as `r5-pm-micro-flb-v1`. Target: football-data result (hindsight, target only). T−1h: 93
matches / 23 days; T−24h: 98 / 22.

**Data gap (PE-26).** football-data published no Betfair Exchange close for 2026-04-20 to 2026-05-03 and for about
half the matches of 2026-04-17, 04-19 and 05-04. Of the 454 matches in the book window, 167 are dated 2026-04-17 to
05-04 and 132 of them have no close. Of the matches with three usable books, 92 of 185 (T−1h) and 111 of 209 (T−24h)
have no close and are not scored; 92 and 110 of them are dated in that span (counted on `data/research/r1/fd.parquet`
through the study's loader). The numbers above exclude them.

## Method
Per match LL(PM mid) − LL(Betfair close) in bits; positive = Polymarket worse.

## Headline (bits)
| T | Mid − close | Last trade − close (same matches) | Paired mid − last |
|---|---|---|---|
| 1h | +0.0150 [−0.0204, +0.0436] | +0.0076 [−0.0241, +0.0360] (n = 87) | +0.0011 [−0.0088, +0.0122] |
| 24h | +0.0158 [−0.0278, +0.0573] | +0.0315 [−0.0080, +0.0694] (n = 74) | +0.0003 [−0.0066, +0.0069] |

Thirds at T−1h +0.026 / +0.036 / −0.035; MLS (39) +0.015, pooled rest +0.015.

## Ladder
L0, L1 pass; L2 fail (CI spans 0; n 93; 23 blocks). **Rung L1 — null and underpowered** (a ±0.03-bit CI cannot
detect R1-sized effects).

## Threats
Stale books; 44% of Betfair closes missing; end-of-season sample; same `registered_at` note as flb-v1.

## Next
Rerun when the book archive is extended; ≈ 1,000 matches would give ±0.005 bits.
