# Agentic replay-eval — does the reviewer's trust/distrust rule track anything real?

## Question

`agents/reviewer.py` decides, deterministically, whether a proposed signal should be trusted:
`trust` if its routed model has positive real walk-forward edge (`edge_bits`) in that league *and*
the odds behind it are real (not synthetic); `distrust` if the edge is negative; `needs_info` if
there's no backtest evidence at all. Does that rule actually correlate with what happens in real
football matches, or is it just a plausible-looking heuristic? This replay answers that the only
honest way available: train strictly before a cutoff date, run the exact agentic pipeline forward
as if the next month's fixtures were still upcoming (no peeking), then reveal what really happened.

## Setup

- **T0 = 2025-12-15**, replay window **2025-12-15 → 2026-01-14** (30 days), computed as
  `max(real Pinnacle closing date) - 30d` so both sides of the cutoff have genuine closing prices.
- **Training fold**: every match strictly before T0 across 11 leagues (E0, E1, D1, SP1, I1, F1, N1,
  P1, B1, SC0, T1), 2,141–5,454 rows per league. All 6 registered models were retrained on this
  fold alone: `dixon_coles`, `gbdt`, `gbdt_mkt`, `gbdt_squadval`, `gbdt_mkt_squadval`,
  `transformer_sequence`. Evidence: `data/backtest/replay/replay_train/by_league.csv` (66 rows).
- **Replay window**: the exact `agentic-signals` graph (`select_fixtures → gather_context →
  inference → odds → edge_detector → risk_manager → review_proposals`) run over the window's 318
  real fixtures with their true dates/teams but *no* access to results or closing prices — odds
  came from the same synthetic-Elo fallback a live run would use for a fixture with no live quote
  yet. Risk sizing: ¼-Kelly, 3% max stake, 3% minimum edge to propose (same defaults as a live run).
- **Scoring**: only after the graph reached the human-approval gate were real Pinnacle closing
  prices (`PSCH`/`PSCD`/`PSCA`) and real results revealed, to compare the model's probability
  against the market's own real closing (de-vigged) probability for the picked outcome.

## Headline result

Every model's real edge_bits was negative in every one of the 11 leagues on this training fold —
consistent with the project's main backtest (`reports/main` note in `CLAUDE.md`): no cherry-picking
happened to produce this replay window, the market beats every registered model on the pre-T0 data
too. Best/worst per model (min |edge_bits| across its 11 leagues):

| model | best league_code | best edge_bits | worst league_code | worst edge_bits |
|---|---|---|---|---|
| gbdt_mkt_squadval | B1/T1 | -0.018 | N1 | -0.045 |
| gbdt_mkt | SC0 | -0.018 | E0 | -0.051 |
| gbdt_squadval | E1 | -0.028 | P1 | -0.059 |
| gbdt | D1 | -0.024 | SC0 | -0.069 |
| dixon_coles | SC0 | -0.035 | SP1 | -0.070 |
| transformer_sequence | E1 | -0.058 | N1 | -0.183 |

Because the router always picks the *least-bad* available model per league (never `None` unless a
league has zero evidence — see `agents/router.py`), and every model was negative everywhere, the
reviewer correctly flagged **every single proposal in this window as `distrust`** — there was no
league where the router could honestly hand out a `trust` verdict. Out of 318 fixtures scouted,
only **4 cleared the risk manager's edge/odds thresholds** to become proposals at all, and all 4
had real closing prices to score against:

| verdict | n | hit_rate | mean_edge_vs_real_close | log_loss |
|---|---|---|---|---|
| distrust | 4 | 0.75 | +0.060 | 0.281 |

(`data/backtest/replay/replay_eval.csv`, `replay_eval_summary.csv`)

## Caveats

- **n=4.** This is nowhere near enough to judge the rule's real-world calibration — a 75% hit rate
  on 4 bets is well within noise for a coin-flip-adjacent process. Nothing here should be read as
  "the model works."
- **No `trust` bucket to compare against.** Because every model lost to the market on this fold,
  the replay could only ever produce `distrust` verdicts — it exercised the deterministic rule's
  *mechanics* end-to-end (real evidence in, real verdict out, nothing invented) but could not test
  whether `trust` actually discriminates better outcomes than `distrust`, since there was no
  `trust` case in this window at all. A fair discriminating-power test needs a period (real or
  historical) where at least one league genuinely had positive real edge_bits.
- **Whole-window scope, single T0.** One 30-day window is one draw from a noisy process; a proper
  validation would run this across several non-overlapping T0 folds and pool the scored proposals.

## Verdict

The agentic pipeline is mechanically sound end to end — real training-fold evidence, honest
per-league routing (still to the least-bad model even when everything is negative), a
deterministic reviewer that correctly withheld trust everywhere the evidence didn't support it, and
correct after-the-fact scoring against real closing prices. That is the thing this replay was
actually built to prove, and it held up with zero LLM calls, zero invented numbers, and full
reproducibility. What it did **not** produce — because the underlying models still lose to the
market on real data, exactly as the main backtest already found — is any evidence that this
system currently identifies a profitable edge. The honest reading of the n=4 result is "too small
to mean anything," not "75% hit rate." Re-running this across multiple T0 folds once the feature
work in progress moves at least one league's real edge_bits positive is the natural next test.
