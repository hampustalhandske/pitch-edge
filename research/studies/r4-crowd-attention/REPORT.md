# R4 — Does pre-kickoff crowd activity predict Polymarket's move? (family `crowd-attention`)

Registrations (ledger 2026-10-21 07:27:40Z; first trial 07:28:04Z): `research/registry/r4-bsky-attention-size-v1.yaml`,
`r4-bsky-mention-skew-v1.yaml`, `r4-bsky-sentiment-skew-v1.yaml`. Code: `research/studies/r4-common/` (pull, linking,
lexicon, panel, model, runner; `explore.py` exploratory, `posthoc.py` post-hoc) and `r4-bsky-*-v1/run.py`. Outputs:
`data/research/r4/`.

## Answer
No, not out of sample. Reddit could not be tested as specified; none of the six registered Bluesky trials beats the
Polymarket-only baseline. **All six are L1** (L2 fails: every CI includes 0). Family BH: smallest p 0.28.

## Headline (evaluation 2026-03-01 → 2026-10-19; each trial run once)
| Hypothesis | Lookback | n (match-days) | Δ OOS R² [95% CI] | p |
|---|---|---|---|---|
| attention → size of PM move | 6h | 1,555 sides (157) | −0.00000 [−0.00089, +0.00083] | 0.50 |
| attention → size | 24h | 1,555 (157) | +0.00032 [−0.00089, +0.00134] | 0.28 |
| mention skew → direction | 6h | 443 matches (117) | −0.00120 [−0.00481, +0.00235] | 0.75 |
| mention skew → direction | 24h | 443 (117) | −0.00238 [−0.00948, +0.00412] | 0.75 |
| lexicon sentiment skew → direction | 6h | 443 (117) | +0.00037 [−0.00282, +0.00372] | 0.41 |
| lexicon sentiment skew → direction | 24h | 443 (117) | +0.00142 [−0.00470, +0.00805] | 0.33 |

The CIs rule out attention adding more than ≈ 0.13% of the variance of PM's remaining move size, and skew more than
≈ 0.2–0.8% of its direction. PM's own baseline explains 1.5% of move size and nothing of direction (OOS R² −0.3%),
consistent with R1/R5. Thirds all span 0; per league (post-hoc) attention is positive nowhere (EPL −0.0049, Serie A
−0.0036, Championship +0.0006).

Post-hoc, not counted: the 24h sentiment-skew coefficient fitted on evaluation rows is +0.013 [−0.002, +0.030], right
sign and positive in all 8 walk-forward folds — ≈ 0.4¢ per SD of skew against a 6.6¢ SD move and ≈ 1.75¢ per side to
cross Polymarket (fee + one-tick half-spread): not tradable even if real.

## Data, linking, missingness
- **Reddit:** 2,673 of 12,046 PM 1X2 groups link to a "Match Thread: A vs B" (22.2%; EPL 94%, Serie A 84%, Bundesliga
  73%, UCL 73%, LaLiga 41%, Championship 13%, MLS 4%). Threads open at kickoff (median −0.01h; 94.7% within ±1h of PM
  `close_at`, which also confirms `close_at` = kickoff); only 1.16% have any comment before ko−1h. No pre-decision
  feature exists, so no trial was spent on Reddit.
- **Bluesky:** `bluesky_post_matches` links posts to club/competition *search queries*, not matches; search backfill
  covers 2026-01-05 onward only. Of 8,844 PM groups (kickoff 2026-01-20 → 10-19), 8.3% have both clubs covered, 20.7%
  at least one (EPL 89% both; MLS/Argentina/Brazil/World Cup 0). Point-in-time column: `indexed_at` (`created_at` is
  client-set, ≈ 3% back- or future-dated).
- Linked/covered matches are much more liquid than the rest (median trades 489 vs 64 Reddit; 300 vs 65 Bluesky).

## Method
Decision hours H = ko−24h … ko−2h. Bounce-robust target: the move from the first bar with `hour_start ≥ H` (within 3h)
to the last pre-kickoff price, so the bounce at H is not shared with the own-recent-move baseline (exploratory check:
slope on the own 6h move −0.048 from the price at H vs +0.045 from the bar after H). Baseline: hours-to-kickoff fixed
effects + PM's recent move, price level and trade count. OLS, monthly expanding walk-forward. Metric: Δ OOS R² per
decision, match-day block bootstrap (5,000). Exploratory 2026-01-05 → 02-28 (cut chosen on counts), evaluation from
2026-03-01.

## Threats
Generic queries ("Athletic", "Derby", "Spurs" catches the NBA, "Inter", "Milan", city names) and the retrospective
backfill (deleted posts missing; ≈ 92% of hits read Jan–Jul vs ≈ 100% Aug–Oct) bias toward zero. The target skips the
first 1–3 hours after H by design. Evaluation is ≈ 80% March–May 2026, in the most-watched leagues. H2's
leave-one-league-out was degenerate (pooled into one group) — moot at L1.

## Next
Bluesky posts in the lineup window (ko−60 to −45 min) against Polymarket order-book snapshots, with stricter linking
(football share ≥ 0.7, or both club names in one post).
