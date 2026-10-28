# R2-Q2 — Does the soft-book consensus at "open" predict Pinnacle's move?

Registration: `research/registry/bookmaker-line-structure-open-consensus-gap-v1.yaml`. Ledger: `trials.csv`
(24 rows, including 2 post-hoc diagnostics).

## What "open" means
football-data `notes.txt`: "These are for pre-closing odds…" and "Betting odds for weekend games are collected
Friday afternoons, and on Tuesday afternoons for midweek games." The decision time is that snapshot, 1–3 days
before kickoff. There is no per-row timestamp; simultaneity across books is assumed.

## Method
Universe: 22 main leagues with Pinnacle open+close, avg/max open and a result. Exploratory 2012-07 → 2019-06
(52,177); evaluation 2019-07 → 2025-06 (45,467); post segment 2025-07+ (2,929, Betfair close).
Features (all from `*_open`, asserted in code): g_avg = logit p_avg_open − logit p_pin_open (power de-vig);
g_max = log(max_open / pin_open), capped at 1.5. Predictability: move (logit p_pin_close − logit p_pin_open) on
g_avg, g_max — fitted on exploratory, scored by out-of-sample R² on evaluation. Strategy: back the outcome with the
largest gap > θ at Pinnacle open; CLV vs power-de-vigged Pinnacle close.

**Data gap (PE-26).** football-data published no Betfair Exchange close for 2026-04-20 to 2026-05-03 and for about
half the matches of 2026-04-17, 04-19 and 05-04. None of the 1,029 post-segment matches dated 2026-04-17 to 05-04 has
a Pinnacle open (the snapshot has none after 2026-01-15), which the universe requires, so none is scored whether or
not it has a close; 863 of the 1,029 also have no Betfair close (counted on `data/research/r2/fd.parquet`). The
post-segment numbers are the recorded run; the gap removes no match from them.

## Headline
Predictability: OOS R² +0.65% [0.47, 0.83] (136k outcome rows). In-window slope on g_avg 0.38 (exploratory),
0.24 (evaluation); with probability-level and outcome controls (post-hoc) 0.39 [0.36, 0.43] and 0.44 [0.38, 0.49].
Pinnacle moves about 40% of the way toward the de-vigged consensus.

Strategy (evaluation; honest null):
| θ | n | CLV | ROI |
|---|---|---|---|
| 0 | 45,466 | −5.03% [−5.17, −4.89] | −5.7% |
| 0.01 | 5,219 | −4.61% [−4.97, −4.26] | −5.9% |
| 0.02 | 155 (< min_n) | −4.80% [−7.45, −2.19] | +1.9% (noise) |
| 0.03 | 13 | −12.1% | — |
| 0.05 | 2 | — | — |

Reference: backing every outcome at Pinnacle open gives −5.2% (exploratory −3.6%); the selection adds only
0.2–0.8 pp against a 3.3–5% Pinnacle open margin. Benchmark sensitivity at θ = 0.02: Shin close −4.6%,
multiplicative close −3.8%. PBO 0.0002 is uninformative (configurations that bet least lose least). Post segment,
θ = 0: −5.5% [−6.8, −4.6]; θ ≥ 0.02 has n ≤ 110.

## Stability
θ = 0 by third: −4.0, −5.0, −6.0%. 0/22 divisions positive. Leave-one-division-out [−5.10, −4.92]%.

## Ladder
L0 pass · L1 pass (open-only features asserted; rerun identical) · L2 fail (every θ with n ≥ 300 below −4.5%).
Strategy rung **L1**. The predictability passes an OOS CI and family BH as a descriptive finding but cannot be
monetised at Pinnacle.

## Threats to validity
If avg/max were captured after the Pinnacle open, the predictability is partly look-ahead. Column definitions
change at 2019/20. The registered θ grid was too coarse for the gap distribution (median 0.5 pp), so θ ≥ 0.02 is
under-powered.

## Next test
Use the lagging soft book's move instead (Q3); test g_avg as a model feature scored in bits against the close.

## Re-grade through the hardened harness (2026-10-21)

Every registered grid point of the evaluation window was recomputed from `data/research/r2/fd.parquet` as per-decision
CLV (`decision_id` = division|date|home|away, `block` = match-day, `league` = division, divisions under 100 bets pooled
as OTHER) and recorded with `pitch_edge.research.run_trial` (`run.py --harness --record`, code hashed and archived under
`data/research/bookmaker-line-structure-open-consensus-gap-v1/code/`). Before recording, each point was compared with the trial imported from `trials.csv`: bet
count equal and mean equal to the 6 significant figures `trials.csv` keeps, in all 5 points; CI bounds differ by at
most 10% of a half-width (the study's multinomial match-day bootstrap and the harness's integer block draws are
different engines, so the CIs cannot match to 1e-9).

**Rung of record: unchanged, L0.** The result of record for each grid point is still the imported row, and L1 reads
`clean` and provenance from that row (no `clean` flag, no `code_hash`). The native rows below are reproduction checks
that the gates cannot credit: the imported rows carry the metric name `clv_log_mean`, a native run must use the registered primary string, so the two are never compared.

What-if (native reproduction as record): **L1**. Evaluation CLV at Pinnacle open vs the power-de-vigged Pinnacle close:
theta 0 -5.03% [-5.17, -4.89] n=45,466; 0.01 -4.61% [-5.00, -4.25] n=5,219; 0.02 -4.80% [-7.36, -2.20] n=155; 0.03 -12.1% n=13;
0.05 n=2. No grid point has a CI above 0, so L2 fails. This is a null/negative result and stays one.

The "what-if" rung treats the first native run as the result of record. It is not a rung of record: those runs came
after the numbers were seen.
