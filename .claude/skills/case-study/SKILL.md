---
name: case-study
description: >
  Run a named pitch-edge test (a walk-forward backtest, a feature-group ablation, or the
  agentic replay-eval) and turn its results into an honest, industry-standard case-study
  write-up at reports/<label>/CASE_STUDY.md, then refresh the root CASE_STUDY.md index. Reads
  only the local machine-readable output under data/backtest/ and data/artifacts/ (never
  writes raw data into reports/ — that directory holds nothing but case studies). Triggers on:
  case study, write a case study, run a backtest and report on it, evaluate this test, honest
  report for <test>, /case-study.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
framework_version: 1.0.0
---

# Case study

## What this is

pitch-edge separates its **public artifact** (`reports/<label>/CASE_STUDY.md` — a one-page,
honest write-up, git-tracked) from its **internal data** (`data/backtest/<label>/` and
`data/artifacts/` — CSVs, model cards, per-fixture scores, gitignored). `pitch-edge backtest`
already writes both automatically: a machine-generated one-pager at
`reports/<label>/CASE_STUDY.md` (`backtest/report.py::render_report` — headline edge vs market,
best staking result per model) and the supporting CSVs under `data/backtest/<label>/`.

This skill is for going beyond that auto one-pager — running (or reading) a specific test and
writing the kind of case-study narrative a reviewer would expect: the question being tested, the
setup, the headline result read plainly (including when the model loses, which on this project's
public-data slices it usually does), the caveats that actually matter, and a verdict. It replaces
`reports/<label>/CASE_STUDY.md` with that fuller write-up and keeps the root `CASE_STUDY.md` as a
short index linking to every test that has one.

## Non-negotiables

- **Never invent a number.** Every figure in the write-up must come from a file under
  `data/backtest/<label>/` or `data/artifacts/`, or from a warehouse query — cite the source
  file so a reviewer can check it.
- **Losing results are published exactly like winning ones.** This project's whole point is
  honesty about whether public data beats the closing price — see the root `CLAUDE.md`
  "Non-negotiables" and the root `CASE_STUDY.md`'s own history. Don't soften a negative
  `edge_bits` number or bury it.
- **`reports/` holds only `<label>/CASE_STUDY.md` files and the root index — nothing else.** Any
  CSV/JSON/parquet a test produces belongs under `data/backtest/` or `data/artifacts/` (already
  gitignored). If a command you're using would write something else into `reports/`, redirect
  it or move the output out before finishing.
- **Long backtests must survive a laptop sleep.** `main`/`developing`-sized runs can take a long
  time; launch them with `nohup caffeinate -i uv run pitch-edge <cmd> ... &` per `scripts/README.md`
  and poll rather than blocking, exactly as CLAUDE.md requires for any long pipeline stage.
- **DuckDB is single-writer.** Never launch a test run while another `pitch-edge` stage might
  still be writing to the warehouse.

## Where each test's numbers live

| Test type | Command | Local data (never in `reports/`) |
|---|---|---|
| Walk-forward backtest | `pitch-edge backtest --label <name> [--leagues ...] [--models ...] [--min-date ...]` | `data/backtest/<name>/{backtest_summary.csv, calibration.csv, by_league.csv, preamble.txt, model_card_*.json}` — also already produces the auto one-pager at `reports/<name>/CASE_STUDY.md` |
| Feature-group ablation | `pitch-edge ablation --groups <a,b,c,...>` | `data/artifacts/ablation.csv` — no automatic report; this skill is what turns it into one |
| Agentic replay-eval | `pitch-edge replay-eval [--as-of YYYY-MM-DD]` | `data/backtest/replay/{replay_eval.csv, replay_eval_summary.csv}` (train-only fold evidence under `data/backtest/replay_train/`) — no automatic report |

## Steps

1. **Clarify the target.** Which test/label, and whether to re-run it or just write up numbers
   that already exist under `data/backtest/` or `data/artifacts/`. Don't re-run a multi-hour
   backtest just to refresh prose if the underlying data hasn't changed — ask first.
2. **Run it, if needed**, using the command from the table above. For a full backtest, launch
   with `nohup caffeinate -i uv run pitch-edge backtest --label <name> ... > logs/<name>.log 2>&1 &`
   and wait for it to finish before reading results — don't guess at numbers mid-run.
3. **Read the local files** for that test (the table above). Cross-check the auto one-pager
   (`reports/<label>/CASE_STUDY.md`, if the command wrote one) against the CSVs — it's a good
   starting point for the headline numbers but not the finished write-up.
4. **Write `reports/<label>/CASE_STUDY.md`** (Write tool, overwriting the auto one-pager) with:
   - **Question** — one line, stated as it would have been before looking at results if this is
     a pre-registered test (compare the "Result 7" pre-registration pattern in the root
     `CASE_STUDY.md` for the house style).
   - **Setup** — data slice, leagues/divisions, date range, row counts, walk-forward config
     (retrain-every-days, edge threshold), and any staking rule used.
   - **Headline result** — the actual table(s): model vs market log-loss / edge_bits, best
     staking result per model. Read it plainly; state directly whether the model beat the
     closing price.
   - **Caveats** — anything that would make a careful reader doubt the number at face value
     (missing early prices, low coverage, small n, a known data-quality gap).
   - **Verdict** — one paragraph, no hedging beyond what the caveats already justify.
   Keep it to roughly one page — one headline table, one staking table, nothing per-division or
   per-strategy (that detail stays in the local CSVs for anyone who wants to dig in).
5. **Update the root `CASE_STUDY.md`** — add or refresh a single-line index entry linking to
   `reports/<label>/CASE_STUDY.md` with its one-line verdict. Only add an entry once the report
   file actually exists; never describe a test that hasn't been run.
6. **Tell the user the headline number plainly** in your reply — including if it's a loss.
