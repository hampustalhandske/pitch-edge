# From hypothesis to result: the pitch-edge research pipeline

How a betting idea becomes a graded result in pitch-edge. Every step is code in this repo or a file it writes;
nothing is promoted by editing a report. The evidence ladder itself is `research/STANDARDS.md`.

## 1. Hypothesis and registration (rung L0)

A study starts as a YAML file in `research/registry/<id>.yaml`, written before any outcome in the test window is
scored. It states:

- `id` and `family`: the id is versioned (`-v1`, `-v2`); the family groups hypotheses that share a
  multiple-testing correction (e.g. `bookmaker-line-structure`).
- `question` and `hypothesis`: `h1`, `h0`, `direction`, and the `disconfirming` result.
- `decision_time`: the moment a decision is taken, and therefore the point-in-time cut for every input.
- `window`: `exploratory` (the training and design period) and `evaluation` (the test period). Only the
  evaluation window can be graded.
- `benchmark`: the sharp reference the result is scored against. For pre-match bookmaker markets that is the
  power-de-vigged Pinnacle close where present, otherwise the de-vigged Betfair Exchange close.
- `primary_metric` and `metric_kind`: `return` (e.g. `clv_log`, the log of taken odds over the de-vigged close
  odds, per bet at flat stakes), `probability` or `score`. Only return-kind strategies can ever be shown as an
  edge.
- `grid`: every parameter combination that will be tried. Each grid point is one trial; there is no
  untracked tuning.
- `min_n`, `costs` and `literature_ceiling`: an effect above the ceiling is treated as a leakage suspect, not a
  win.

`pitch-edge research register` hashes the file (sha256) and records it in the ledger. Editing a registered
file is refused; a change means a new id.

## 2. Data, point in time

Inputs come from the owner's lakehouse through its HTTP API (`pitch_edge.lake.client.Lake`, one token per
project, one query at a time), either pulled directly by the study code or mirrored into the local DuckDB
warehouse by `pitch-edge lake-sync` (`lake_*` tables, checkpointed per window). Every input row carries a
`known_at` timestamp, and the study asserts `known_at <= decision_time` in code. Known traps (snapshot
timing of football-data's "open", the 2019/20 source change, the Pinnacle-close end date, stale prediction-market
books, last-trade bounce) are listed in the research-protocol skill and must be handled in the study code.

## 3. Study code and trials (rung L1)

Study code lives in `research/studies/<id>/`. It builds one row per decision (`decision_id`, `block` = the
resampling unit, usually the match-day, `value` = the metric) and calls `pitch_edge.research.trial.run_trial`
once per registered grid point and window. `run_trial`:

- computes the mean with a match-day block-bootstrap 95% CI (at least 2,000 draws) and one-sided p-values;
- checks the row against the registration (registered grid point, metric and direction, unchanged file);
- records git sha, config hash, data snapshot, row counts and the study's code hash;
- appends it to the research ledger, `data/research/ledger.duckdb`.

The ledger is append-only and hash-chained: each entry's hash is sha256 of the previous hash, the table name
and the canonical JSON of the row, so an edited or deleted row is detectable (`verify_chain`). Dropped ideas
keep counting. The first run of a grid point is the result of record; later runs are reproduction checks, and
L1 requires a rerun that reproduces the record exactly (native runs) or to a stated tolerance (imported runs).

## 4. The ladder (L2 to L5)

`pitch-edge research grade <id>` reads the ledger and evaluates `research/STANDARDS.md` gate by gate:

- **L2 signal**: walk-forward, out of sample, evaluation window only. The primary metric is positive against
  the sharp benchmark, with a block-bootstrap 95% CI excluding zero, on at least 300 decisions.
- **L3 survives search**:
  - Benjamini-Hochberg across every test the family has ever run;
  - the deflated Sharpe ratio at the family's trial count;
  - PBO below 0.25 for parameter grids;
  - the effect within the literature ceiling;
  - the same sign in 2 of 3 sub-periods, and a CI that still excludes zero with any one league removed.
- **L4 independent**: an auditor agent re-derives the headline from raw tables with its own code and signs
  `research/audits/<id>.md`. For strategies, an independent engine grades it as well. Predictive and
  descriptive findings stop at L4.
- **L5 forward**: paper-traded on fixtures that had not kicked off when the decision was logged, at least 150
  settled decisions, CI reported.

The display floor for the `ask` agent and the dashboard is the highest rung any return-kind champion holds.
Below it, numbers are labelled with their rung and never called an edge.

## 5. The paper loop (rung L5)

`scripts/paper_loop.sh` runs hourly under launchd. Each cycle:

1. `collect`: live quotes from the-odds-api (Pinnacle, Betfair and soft books), Polymarket and Kalshi.
2. `decide`: every registered paper strategy decides on the latest quotes. Decisions are immutable and
   hash-chained (`data/paper/paper.duckdb` plus a JSONL log), with the code hash recorded.
3. `settle`: settles decisions against lake closes (the Betfair Exchange close and the result from
   football-data, or the last pre-kickoff Polymarket bar). Fallbacks are provisional and never enter the
   metric.

`pitch-edge paper report` gives per-strategy CLV with a match-day block-bootstrap CI and ROI for the
evaluation segment after registration. Next to each result it gives the window's coverage: the loop runs on a
laptop, and cycles missed while it sleeps are counted and stated, never filled. Nothing ever places a bet.

## 6. Independent grading

A strategy that reaches L3 is exported as a data bundle (`pitch_edge.edgelab.export`): quotes, timestamps and
the reference close, built by the study's own point-in-time code. It is submitted to the owner's grading engine
with its ledger id, family and family N. The engine's grade is reconciled bet for bet with pitch-edge's numbers
in `research/audits/`, and disagreements are written up (example: `research/audits/edgelab-r2q3.md`).

## Where things are

| Step | Code or file |
|---|---|
| Registration | `research/registry/*.yaml`, `src/pitch_edge/research/registry.py` |
| Ledger | `src/pitch_edge/research/ledger.py`, `data/research/ledger.duckdb` |
| Trials | `src/pitch_edge/research/trial.py`, `research/studies/<id>/` |
| Statistics and de-vig | `src/pitch_edge/research/stats.py`, `devig.py` |
| Ladder | `research/STANDARDS.md`, `src/pitch_edge/research/gates.py` |
| Reports and audits | `research/studies/<id>/REPORT.md`, `research/audits/`, index `reports/CASE_STUDY.md` |
| Paper loop | `scripts/paper_loop.sh`, `src/pitch_edge/paper/` |
| Lake access | `src/pitch_edge/lake/` |
| Export for independent grading | `src/pitch_edge/edgelab/` |
