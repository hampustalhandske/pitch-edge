---
name: research-protocol
description: The pitch-edge research protocol — how to pre-register a betting-market hypothesis, run it as counted trials, score it against the sharp market with honest uncertainty, grade it on the L0–L5 evidence ladder in research/STANDARDS.md, and write it up. Load before designing, running, auditing or reporting any strategy, model or feature study. Triggers on: hypothesis, pre-register, study, edge, strategy test, evidence ladder, rung, champion, research harness.
---

# Research protocol

`research/STANDARDS.md` is the law; this skill is the procedure.

## 1. Register (rung L0)

Create `research/registry/<id>.yaml` **and run `pitch-edge research register research/registry/<id>.yaml`**
**before** scoring anything that touches outcomes or closing prices in the evaluation window. The ledger's
registration time is the only one that counts: never type a guessed `registered_at` into the YAML (four R1
hypotheses lost L0 because their typed time was later than their trials). `<id>` is
`<family>-<short-slug>-v<n>`, and the file name must equal the id. Never edit a registration after a run; bump
`v<n>` instead (the old one stays, and its trials still count). `metric_kind: return | probability | score` is
required from 2026-10-21 (DSR/PBO apply to `return` only).

```yaml
id: pm-vs-sharp-fav-bias-v1
family: polymarket-efficiency        # multiple-testing family (BH across every id in it)
registered_at: 2026-10-20T21:00:00Z
author: quant-researcher
question: >
  Do Polymarket 1X2 prices 24h before kickoff over-price favourites relative to the de-vigged sharp close?
hypothesis:
  h1: Polymarket implied probability for the favourite exceeds the de-vigged sharp close by > 0
  h0: no difference
  direction: positive                 # positive | negative | two-sided
  disconfirming: 95% CI of the mean gap includes 0, or sign flips in 2 of 3 sub-periods
decision_time: kickoff - 24h          # when a decision would be taken; every input must be known by then
universe: Polymarket win_home/draw/win_away markets mapped to fd_matches (pm_soccer_match_map)
window:
  exploratory: [2023-09-01, 2025-06-30]   # may look at outcomes here to design
  evaluation:  [2025-07-01, 2026-10-19]   # scored once, after registration
benchmark: betfair_ex close de-vigged (power); pinnacle close where present (sensitivity)
primary_metric: mean_prob_gap          # or clv_log | edge_bits | brier_skill
secondary_metrics: [roi_after_fees, brier_skill]
costs: polymarket taker fee 0.05*p*(1-p) per share + half bar high-low spread
grid:                                  # every combination is one trial
  favourite_threshold: [0.5, 0.6]
min_n: 300
literature_ceiling: gap of a few percentage points at most for liquid matches
```

## 2. Run (L1–L2)

- Through the research harness: a script in `research/studies/<id>/run.py` calls
  `pitch_edge.research.run_trial(registration, config, per_decision, metric_name, ledger, data_snapshot,
  code_files=[...])` once per grid point. `per_decision` has columns `decision_id, block` (match-day, ISO date),
  `value`, and `league` (needed for the leave-one-league-out gate). It refuses unregistered/edited ids, off-grid
  configs and unregistered metrics, recomputes the bootstrap to record `reproduced`, hashes the code (archived
  when the tree is dirty), and writes one hash-chained ledger row. The **first** run per config is the result of
  record; a rerun must reproduce it exactly. Every registered grid point counts in BH/DSR (unrun = p 1).
- `data_snapshot` names the pull: its date and the lake build that served it (`Lake.version`, the API's
  `X-Lakehouse-Version`, e.g. `lake 856ab2f64e97 pulled 2026-10-28`). The pull script also writes that build into
  its snapshot file next to the row counts.
- Ledger writes retry on DuckDB lock contention; still read back your trial with
  `pitch-edge research ledger --id <id>`. Other commands: `status`, `grade <id> [--config-hash]`,
  `import-trials`, `review` (records an audit / edgelab / paper verdict, optionally scoped to a config subset).
- Outputs go to `data/research/<id>/` (gitignored). The study folder holds code and the report only.
- Uncertainty: block bootstrap with match-day blocks (≥ 30 blocks for L2), ≥ 2,000 resamples, α ≤ 0.05,
  percentile CI. For metrics on
  probabilities, also report the paired difference vs the benchmark.
- Stability: report the metric per chronological third and per league (leagues with n < 100 pooled).
- Costs: always net of the costs in the registration. A gross number may be shown beside it, labelled.

## 3. Grade (L3)

For a family, collect every trial in the ledger, compute p-values (one-sided per registered direction),
apply BH at q = 0.10. Compute PSR/DSR with the ledger's trial count for return-like metrics, and PBO via
CSCV (16 blocks) when the grid has ≥ 4 points. Compare the effect to the literature ceiling. Write the
rung reached with each criterion's pass/fail and the number behind it.

## 4. Review (L4) and forward test (L5)

- Ask the orchestrator for a `skeptic-auditor` pass. Nothing is called "an edge" anywhere (report, LOG,
  ask agent, dashboard) before the auditor signs.
- Then `edgelab-liaison` for edgelab's grade (see `edgelab-review`).
- L5 needs the paper-trading ledger: decisions are written before kickoff, are immutable, and settle
  against the sharp close later.

## 5. Write up

`research/studies/<id>/REPORT.md`, one page: question → registration link → data/window/n →
method → headline with CI → stability table → ladder table (each rung, criterion, evidence) → threats to
validity → next test. Then one line in `research/LOG.md`:
`2026-10-20 | <id> | L2 | CLV +0.4% [0.1, 0.7] n=812 | auditor: pending`.

## Known traps in this project's data (each one cost a study on 2026-10-20/21)
- **Last-trade bounce (Polymarket, Kalshi):** a last-trade price carries bid-ask bounce (σ ≈ 0.7¢). Any regression
  of a later move on a gap that contains the same price gets a mechanical negative slope (R1-Q4 convergence was
  entirely bounce). Start targets at the bar *after* the decision hour, or use mids.
- **Stale order books:** the PMXT archive has ≈ one Polymarket book snapshot per market per 4h. A stale book both
  creates a "signal" and "fills" it later (R5 cross-venue arb: +1.3¢ on reused snapshots, −3.3¢ on fresh ones).
  Always split results by quote age.
- **football-data "open" is a Fri/Tue afternoon snapshot**, 1–3 days before kickoff, with no per-row timestamp.
  By match day it is stale; sharp closes move toward Polymarket, not the other way (R1). Per-book capture timing
  was tested and found synchronous (R6 audit bracket test), except Betfair Exchange open (unverifiable).
- **football-data Avg/Max changed source in 2019/20** — any baseline built on them breaks there; fit era-aware
  baselines or the post-2019 gain is overstated (R6 audit: ≈ 20%).
- **Pinnacle close ends 2025-09** in the lake; Betfair Exchange close is the 2025-10+ benchmark.
- **Max odds are quoted prices, not bettable prices.** Positive CLV is almost mechanical once a quote above fair
  exists (R2-Q3 audit placebo); availability can only be shown forward.
- **Signs are not stability:** "6/6 leagues positive" hid a Bundesliga-only effect (R3 xG). The gate checks
  leave-one-league-out CIs.
- **Typed timestamps lie:** never put a guessed `registered_at` in a YAML; YAML 1.1 reads `yes`/`no` as booleans
  (quote them).
- **Reddit match threads open at kickoff** — no pre-match signal; Reddit `score` is as-retrieved. Bluesky links are
  to search queries, not matches; use `indexed_at`.
- **Injury first-seen times exist only from 2026-09-15** (no point-in-time injury history).
- **DuckDB timezone:** the session runs Europe/Stockholm; bind TIMESTAMPTZ bounds as UTC explicitly.

## Smells that mean stop and check for leakage
- Any feature computed with a window that includes the match itself (rolling means without `shift(1)`).
- Joins on match date without kickoff time when the decision time is the same day.
- `scraped_at` used as if it were publication time, or vice versa.
- Results that are better in the evaluation window than in the exploratory one.
- An effect above the literature ceiling.
