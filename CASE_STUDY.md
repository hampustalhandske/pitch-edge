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

_None yet under this system._ The previous version of this file described a run from
2026-09-06 with hardcoded numbers that no longer match the current model registry (the GRU
sequence model, GNN player embeddings and the in-play model were since removed;
`odds/leadlag.py` and `backtest/event_study.py`, the two modules behind the old lead-lag and
referee-lag results, no longer exist in the codebase) — rather than leave stale figures in place,
this index was reset. Run the `case-study` skill against a label below to regenerate a real one:

| Label | Command | Status |
|---|---|---|
| `main` | `pitch-edge backtest --label main` | not yet regenerated |
| `developing` | `pitch-edge backtest --label developing --leagues <16 under-covered divisions>` | not yet regenerated |
| `squad_value` | `pitch-edge backtest --label squad_value --models gbdt,gbdt_mkt,gbdt_squadval,gbdt_mkt_squadval` | not yet regenerated |
| `ablation` | `pitch-edge ablation` | not yet regenerated |
| `replay` | `pitch-edge replay-eval` | not yet regenerated |

## What's still true regardless of specific numbers

- Every model is compared against the market's own closing (or best-available) price on the same
  out-of-sample walk-forward folds — never a shuffled split, never priced at a fantasy line.
- Losing is the expected, honest baseline for a public-data model marked against the close, and
  it gets published exactly like a win would.
- `reports/rag_eval.csv`-style retrieval-quality checks, referee/lead-lag event studies, and any
  other standalone evaluation get the same treatment: a real run, a real `CASE_STUDY.md`, no
  numbers carried over from a previous codebase state.
