# Case studies

This file is an index, not a report. Each test PITCH-EDGE runs — a walk-forward backtest, a
feature-group ablation, the agentic replay-eval — gets its own one-page, honest write-up at
`reports/<label>/CASE_STUDY.md`: the question being tested, the setup, the headline result read
plainly (a model that loses to the closing price is reported as losing to it), the caveats that
matter, and a verdict. `reports/` holds nothing else — no raw CSVs, parquet or per-match data.
The supporting machine-readable numbers behind every write-up live locally under
`data/backtest/<label>/` and `data/artifacts/` (gitignored, not published).

Case studies are produced by the `case-study` Claude Code skill
(`.claude/skills/case-study/SKILL.md`): it runs the relevant `pitch-edge` command, reads only the
resulting local data, and writes (or refreshes) that test's `CASE_STUDY.md` — never inventing a
number that isn't traceable to a file on disk.

## Available case studies

The previous version of this file described a run from 2026-09-06 with hardcoded numbers that no
longer match the current model registry (the GRU sequence model, GNN player embeddings and the
in-play model were since removed; `odds/leadlag.py` and `backtest/event_study.py`, the two modules
behind the old lead-lag and referee-lag results, no longer exist in the codebase) — rather than
leave stale figures in place, this index was reset. The model registry has since changed again:
`gbdt_mkt`/`gbdt_squadval`/`gbdt_mkt_squadval` are gone (no live-odds feed means a market-odds
feature is just bias on the live `ask` path), replaced by a five-model roster — `dixon_coles`,
`gbdt` (no market odds), `stochastic_strength` (Monte Carlo on Elo), `transformer_sequence`,
`sentiment_only` — and `replay-eval`/`agentic-signals` were removed in the Phase 8 scope
narrowing (see `CLAUDE.md`). Run the `case-study` skill against a label below to regenerate a
real one:

| Label | Command | Status |
|---|---|---|
| `main` | `pitch-edge backtest --label main --models dixon_coles,gbdt,stochastic_strength,transformer_sequence,sentiment_only` | not yet regenerated |
| `developing` | `pitch-edge backtest --label developing --leagues <16 under-covered divisions>` | not yet regenerated |
| `ablation` | `pitch-edge ablation` | not yet regenerated |
| `pmxt_timing` | `pitch-edge map-pmxt && pitch-edge backtest-timing --label pmxt_timing` | smoke-tested against real Polymarket data (124 matches) — real mechanism, sample far too small for a real verdict; see `data/backtest/pmxt_timing/` |

## What's still true regardless of specific numbers

- Every model is compared against the market's own closing (or best-available) price on the same
  out-of-sample walk-forward folds — never a shuffled split, never priced at a fantasy line.
- Losing is the expected, honest baseline for a public-data model marked against the close, and
  it gets published exactly like a win would.
- `reports/rag_eval.csv`-style retrieval-quality checks, referee/lead-lag event studies, and any
  other standalone evaluation get the same treatment: a real run, a real `CASE_STUDY.md`, no
  numbers carried over from a previous codebase state.
