# Issues

Work on pitch-edge is tracked here, one entry per issue. Ids are `PE-<n>`; issues in the other projects pitch-edge
works with are referred to by their own ids (`EL-` edgelab, `LK-` the lakehouse, `AT-` agentic-trading). Each entry
gives its status (open, in-progress, done, wontfix), priority (must, should or nice, with the shared requirement it
serves), who raised it, the evidence, the acceptance criterion, what blocks it and what it blocks, and the commits
that worked on it. Closed issues stay in the file with the commit that closed them. Every commit names its issue
(`Refs PE-<n>` or `Closes PE-<n>`).

Shared requirements referred to below: M3 typed submission and result contract; M5 trial accounting across
projects; M8 traceability; S2 lake capacity; S4 decimal-odds instrument with CLV; N4 always-on host.

## PE-1 Track pitch-edge work as issues
- Status: done
- Priority: must (M8)
- Raised by: joint plan, phase 0
- Evidence: work items lived only in the local, untracked `research/HANDOFF.md`.
- Acceptance: this file exists, is tracked, and every phase-0 item and known defect has an entry.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-1`.

## PE-2 Settle paper decisions on every loop cycle and state coverage
- Status: done
- Priority: must (forward evidence, L5)
- Raised by: pitch-edge
- Evidence: `scripts/paper_loop.sh` settled once every 24 h; on 2026-10-27 the settle ran at 08:39 UTC and the
  lake's `fd_matches` closes arrived at 10:08 UTC, so every weekend decision waited a further day (1 final settlement
  out of about 300 decisions). The Mac sleeps: 47 loop cycles ran from 2026-10-20 to 2026-10-27 where about 170 were
  expected, none on Saturday 2026-10-25, and quotes missed while asleep cannot be backfilled.
- Acceptance: `paper settle` runs on every hourly cycle and the daily mechanism is gone; missed cycles are derived
  and shown by `paper status`; `paper report` states the share of each evaluation window covered by a cycle next
  to its CLV; gaps are never filled.
- Result: `paper settle` runs every cycle; `paper status` shows cycles observed against expected and the longest
  gaps (derived from `decide_log`, one entry per cycle); `paper report` adds each primary segment's window
  coverage; the loop logs `missed N cycles since <ts>` when it wakes after a gap within one process run (a gap
  across a restart shows only in `paper status`). The hourly settle's cost is PE-18.
- Blocked by: none. Blocks: any L5 claim.
- Commits: see git log for `PE-2`.

## PE-3 lake-sync uses at most one lake query slot and waits when the lake is busy
- Status: done
- Priority: should (S2)
- Raised by: joint plan
- Evidence: the lake API serves a few concurrent queries per service and answers an extra one with HTTP 503 and
  `Retry-After`; `pitch_edge.lake.client.Lake` treated a 503 as an error.
- Acceptance: lake-sync issues its queries strictly one after another (a test fails on overlapping queries);
  `Lake` waits `Retry-After` (capped) on a 503 and retries a bounded number of times.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-3`.

## PE-4 Delete dead code
- Status: done
- Priority: should (owner rule: delete what is unused)
- Raised by: pitch-edge
- Evidence: `pitch-edge artifacts` wrote dashboard JSONs the dashboard does not read; the `dossiers` key in
  `TABLE_KEYS` has no table; the case-study skill still describes a `replay-eval` command that no longer exists;
  `rag/fixture_context.py` and its test describe a replay harness that does not exist.
- Acceptance: the three are gone with their tests and doc mentions; lint, types and tests pass.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-4`.

## PE-5 Reconcile edgelab's R2-Q3 bet365 grade with pitch-edge's numbers
- Status: done
- Priority: must (L4(b) evidence for R2-Q3 is recorded only once reconciled)
- Raised by: pitch-edge
- Evidence: edgelab graded `pitch-edge-r2q3-bet365` inconclusive on 2026-10-24 (Sharpe 0.19, HAC t 0.70, DSR 0.76
  at N=1); no write-up compares it with pitch-edge's CLV on the same 4,942 bets.
- Acceptance: `research/audits/edgelab-r2q3.md` gives edgelab's gates, pitch-edge's numbers for the same decisions,
  each difference explained or marked unexplained, and the rung consequence.
- Result: edgelab and pitch-edge agree bet for bet (4,942 bets, same prices, outcomes, hit rate and stake-weighted
  return); the grade stays inconclusive, consistent with the audit's rejection of the bet365 strategy; no rung
  change. Found on the way: edgelab's scored series includes the locked 20% holdout (all bets up to 2025-05-30
  are in the headline), filed as EL-23; the CPCV value (0.429 against a full-sample 0.194) is unexplained
  (EL-9). `notes.md` of the strategy and the `pe_side` summary describe the holdout as unscored; the audit
  states that this is not what the run shows.
- Blocked by: none. Blocks: none. Related: EL-4, EL-8, EL-9, EL-10, EL-11, EL-12, EL-23.
- Commits: see git log for `PE-5`.

## PE-6 Move the R2-Q3 strategy folders from edgelab into pitch-edge
- Status: done
- Priority: must (strategies live in the client, joint plan B1)
- Raised by: joint plan
- Evidence: `pitch-edge-r2q3-bet365` and `pitch-edge-r2q3-max` sat untracked inside edgelab's `strategies/`.
- Acceptance: the strategy code and specs are tracked under `research/audits/edgelab-r2q3/strategies/` (with
  repo-relative data paths and without edgelab's private grading-policy block, the exact submitted hashes recorded
  in its README); the complete folders, run output included, are archived byte-identical under the git-ignored
  `data/research/r2/edgelab/strategies/`; edgelab removes its copies.
- Blocked by: none. Blocks: EL-21.
- Commits: see git log for `PE-6`.

## PE-7 Unload the finished edgelab bet365 launchd job
- Status: done
- Priority: must (phase 0)
- Raised by: joint plan
- Evidence: launchd job `com.pitchedge.edgelab.bet365` stayed loaded after its run exited 0 on 2026-10-24.
- Acceptance: the job is no longer loaded. Done 2026-10-27 with `launchctl bootout`; no repo change.
- Blocked by: none. Blocks: none.
- Commits: none.

## PE-8 Declare the trial family and N on every edgelab submission
- Status: open
- Priority: must (M5)
- Raised by: pitch-edge
- Evidence: edgelab graded the bet365 strategy at N=1 while its family in pitch-edge's ledger held 355 trials, so
  its deflated Sharpe understated the search.
- Acceptance: each submission carries the pitch-edge ledger id, the family and the family N; edgelab's result shows
  the N it used.
- Blocked by: EL-1 (M3 contract), EL-4 (M5 trial accounting). Blocks: any further counted edgelab run.
- Commits: none yet.

## PE-9 Submit through edgelab's client and delete the hand-built path
- Status: open
- Priority: should (S4, phase 3)
- Raised by: joint plan
- Evidence: `src/pitch_edge/edgelab/scaffold.py` and `research/audits/edgelab-r2q3/make_strategy.py` write
  strategy folders for edgelab's tree; `research/audits/edgelab-r2q3/pe_side.py` reconciles CLV by hand.
- Acceptance: `export.py` builds a bundle carrying quotes, timestamps and the reference close (Pinnacle de-vigged
  close where present, else the Betfair Exchange de-vigged close) and submits it through edgelab's client;
  `scaffold.py`, `make_strategy.py` and the hand CLV code are deleted in the same change.
- Blocked by: EL-1, EL-2 (job worker), EL-6 (decimal-odds instrument with CLV). Blocks: none.
- Commits: none yet.

## PE-10 Provide acceptance data for edgelab's bet-level mode
- Status: open
- Priority: should (S1)
- Raised by: pitch-edge
- Evidence: the R2-Q3 bet365 bundle (4,942 bets, matched one-to-one with edgelab's fills) has per-bet CLV and returns
  computed by pitch-edge.
- Acceptance: edgelab's bet-level statistic reproduces pitch-edge's per-bet mean and match-day block-bootstrap CI
  on that bundle within a stated tolerance.
- Blocked by: EL-5, EL-11. Blocks: none.
- Commits: none yet.

## PE-11 Make the repo's lint and format gate pass
- Status: done
- Priority: must (a commit is safe only when the gates pass)
- Raised by: pitch-edge
- Evidence: `ruff check .` failed on `research/studies/r7-outcome-models/generate.py` (C408) and `ruff format
  --check .` on four R7 study scripts.
- Acceptance: `ruff check .` and `ruff format --check .` pass; the R7 scripts change formatting only.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-11`.

## PE-12 Duplicate matches in the warehouse spine
- Status: done
- Priority: should (correct training data for the `ask` models)
- Raised by: R7 study
- Evidence: at least 698 `cfmd_` matches duplicate `fd_` matches under variant team spellings
  (`research/studies/r7-outcome-models/REPORT.md`).
- Acceptance: one row per real match; a test fails on a duplicate fixture under two name spellings.
- Blocked by: none. Blocks: none.
- Result: the xgabora connector (`club_football_match_data`) is removed. `ingest_everything` first runs
  `purge_duplicate_fixtures`, which deletes each non-football-data row that `find_duplicate_fixtures` pairs with a
  football-data row, with its `odds`, `weather` and `features` rows; unpaired rows are kept. `pitch-edge health` and
  the refresh summary report duplicate fixtures (`data/quality.py`). Measured read-only on the warehouse: 53,959
  `cfmd_` rows; 1,780 duplicate pairs, all `cfmd_` against `fd_`; 0 non-`cfmd_` duplicates (none between football-data
  and openfootball); 1,779 matches (with 17,142 odds, 302 weather and 713 features rows) to be purged at the next
  ingest.
- Owner decision: no code ingests the roughly 52,000 remaining `cfmd_`-only matches any more, so a freshly built
  warehouse would not have them while this one does. The owner decides whether to re-add the source or drop the rows.
- Commits: see git log for `PE-12`.

## PE-13 UCL paper decisions have no benchmark
- Status: open
- Priority: should (registered universe of the forward test)
- Raised by: pitch-edge
- Evidence: football-data has no UCL division and the lake has no bookmaker close for UCL, so UCL decisions are
  collected but reported as no benchmark; the registration of `bookmaker-line-structure-live-soft-vs-sharp-v1`
  lists UCL.
- Acceptance: either UCL leaves the paper strategies' universe (stated as a deviation in the report), or a sharp
  UCL close source is found and used.
- Lead: the lake's planned OddsPapi recorder (LK-45) would give Pinnacle and Singbet UCL price paths from
  2026-01 with kickoff times, and FotMob the scores (LK-27).
- Blocked by: LK-45. Blocks: none.
- Commits: none yet.

## PE-14 Paper strategies that cannot produce evidence
- Status: done
- Priority: should (owner rule: every piece of work serves a goal)
- Raised by: pitch-edge
- Evidence: `pm-vs-sharp-gap` runs with placeholder parameters (hypothesis `pending-r1`, unregistered, 4 decisions
  in a week); the registered sanity rule of `soft-vs-sharp-live` ("price <= the max quote in the response") can
  never bind on live data.
- Acceptance: `pm-vs-sharp-gap` is registered or removed from the loop; the v1 rule's tautology is recorded as a
  deviation in its report.
- Blocked by: none. Blocks: none.
- Result: `pm-vs-sharp-gap` is removed from the paper loop (its past decisions stay in the ledger and still
  report as unregistered); the v1 sanity-rule tautology is recorded as a deviation in the paper report notes.
- Commits: see git log for `PE-14`.

## PE-15 Stale docs
- Status: open
- Priority: nice
- Raised by: pitch-edge
- Evidence: `reports/main/CASE_STUDY.md` covers an older model roster; `pitch-edge refresh --fast` help mentions
  GRU; the `backtest/as_of.py` docstring says `ODDS_API_KEY` is unset; `ODDS_API_KEY` and `LAKEHOUSE_*` are outside
  the `keys.py` registry.
- Acceptance: each statement matches the code.
- Progress: the `refresh --fast` help, the `as_of.py` and `keys.py` docstrings and the `pull_prices.py` docstring
  now match the code (`API_KEYS.md` already lists the lake and odds-api credentials outside the registry). Left:
  `reports/main/CASE_STUDY.md` needs a rerun of the main backtest on the current model roster.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-15`.

## PE-16 The paper loop runs on a laptop that sleeps
- Status: open (deferred by the owner)
- Priority: nice (N4)
- Raised by: pitch-edge
- Evidence: see PE-2.
- Acceptance: the loop runs on an always-on host; until then PE-2's coverage figures go with every forward result.
- Blocked by: owner. Blocks: none.
- Commits: none.

## PE-17 Export the research ledger for edgelab's trial-count check
- Status: done
- Priority: must (M5)
- Raised by: edgelab (EL-4)
- Evidence: EL-4 verifies a declared trial count by recomputing pitch-edge's hash chain. The chain is global
  across families (it starts at genesis, 672 entries on 2026-10-27, plus 47 legacy rows written before the chain
  existed), and the gates' family N counts tests, not rows (`research/gates.py`, `_family_tests`): the R2-Q3
  family had 338 trial rows, 300 distinct grid points and 355 tests.
- Acceptance: `pitch-edge research` exports the chain from genesis to a given entry with the row bodies, and the
  family's test keys behind N, in the form EL-4 specifies; a test recomputes the chain from the export alone.
- Result: `pitch-edge research export-ledger --family <f> --upto <row_hash> --out <file>.jsonl` in the shape
  agreed with edgelab: every chain entry from genesis to `--upto` (all tables and families) as
  `{table, row_canonical, prev_hash, row_hash}` with the exact hashed string, then the family's unchained rows
  (`chained: false`), then a summary (n_tests, distinct (hypothesis_id, config_hash), return-kind count,
  test_keys `"<hypothesis_id>|<config_hash>"`). Tests recompute the chain from the file and show that one changed
  byte breaks every later hash. On a copy of the live ledger all 672 entries recompute; bookmaker-line-structure
  has 338 trials, 300 distinct keys, 46 return-kind and 1 unchained row (the gates' N of 355 also counts windows
  and imported keys, so it is at least the distinct count edgelab recomputes).
- Blocked by: none. Blocks: PE-8.
- Commits: see git log for `PE-17`.

## PE-18 The hourly settle queries the lake once per open decision
- Status: done
- Priority: should (S2)
- Raised by: pitch-edge
- Evidence: `paper/settle.py` looks up `fd_matches` (and Polymarket bars) separately for every decision past
  kickoff without a final settlement, re-polling provisional ones for up to 30 days; with settlement on every
  cycle (PE-2) that is one lake query per open decision per hour.
- Acceptance: one lake query per league and date window per settle call, with the same settlements as before.
- Result: `LakeCloseSource` keeps every lake answer for the settle call it serves: one fd_matches query per
  (division, kickoff date) and one bar query per token set, and a failed query is not repeated within the call.
  The decision logic is unchanged; mocked tests count the queries.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-18`.

## PE-19 Document the research pipeline for the other projects
- Status: done
- Priority: should (round 2: agentic-trading adopts pitch-edge's pipeline)
- Raised by: coordinator, round 2
- Evidence: the pipeline was spread over `research/STANDARDS.md`, the research modules' docstrings and the
  local handoff notes.
- Acceptance: `docs/PIPELINE.md` explains registration, point-in-time data, study code and trials, the ledger,
  replication, the L0-L5 ladder, the paper loop and independent grading, with the file for each step; sent to
  agentic-trading and edgelab.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-19`.

## PE-20 The family N for the deflated Sharpe counts tests of every metric kind
- Status: open
- Priority: should (M5)
- Raised by: pitch-edge (E2 review)
- Evidence: `research/gates.py` `_dsr` uses N = every test in the family, while the variance of Sharpe ratios
  comes only from same-kind (return) tests. In `bookmaker-line-structure`, 254 of 338 trial rows are R2-Q1
  calibration tests with a probability metric that could never be selected as a strategy, so the R2-Q3 DSR was
  deflated at N=355.
- Acceptance: N counts the return-kind tests of the family, the total is reported beside it, and the rule
  matches the one agreed for edgelab (E2); a test covers a mixed-kind family.
- Blocked by: E2 agreement. Blocks: PE-17.
- Commits: none yet.

## PE-21 Keep the research-ledger format stable for agentic-trading's ledger
- Status: done
- Priority: should (M5)
- Raised by: agentic-trading (AT-40), agreed 2026-10-28
- Evidence: agentic-trading writes its own small ledger that matches pitch-edge's hash chain byte for byte
  (columns, canonical JSON and chain hash in `src/pitch_edge/research/ledger.py`), so one checker and edgelab's
  declared-N check (EL-4) work on both. The format exists only as code.
- Acceptance: the format (columns, canonicalisation, chain hash, genesis, row keys) and two test vectors are
  written down beside the ledger, and a pitch-edge test asserts those vectors, so a format change cannot pass
  unnoticed.
- Result: `tests/unit/test_research_ledger_format.py` states the format and asserts two vectors (registration
  hash `13c98f3c…`, trial hash `18879784…`); agentic-trading reproduced both with an independent implementation.
  Any future change to the canonicalisation or separators gets a format version number recorded with the
  chain, so a shared checker can tell the versions apart.
- Blocked by: none. Blocks: AT-40.
- Commits: none yet.

## PE-22 R3 features drop ClubElo and xG for clubs with two football-data spellings
- Status: done
- Priority: should (wrong numbers on the next data pull)
- Raised by: lakehouse audit (PE-A1)
- Evidence: `research/studies/r3-common/features.py` `fd_team_keys` inverts football-data key -> club_id and
  keeps one spelling per club, and the ClubElo and Understat merges joined on the raw panel key. After the
  lake's club-registry merge four clubs have two spellings (Sparta Rotterdam, Oster, Estrela, Torpedo
  Moscow), so matches written the other way would get NaN Elo and xG. The local R3 snapshot predates the
  merge and has no club with two spellings, so recorded R3 results are unaffected.
- Acceptance: both merges go through the club_id, so every spelling of a club gets the same features; on the
  current snapshot the change leaves every panel key as it is.
- Result: `canonical_key` maps panel and feature keys to one spelling per club_id before both merges; all 622
  snapshot keys are unchanged, and two spellings of one club merge in a synthetic check.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-22`.

## PE-23 Unused lake helpers in study code
- Status: done
- Priority: nice (simplicity)
- Raised by: lakehouse audit (PE-A2, PE-A3)
- Evidence: `research/studies/r3-common/lake.py` `describe` called `Lake.describe`, which the client does not
  have, and had no callers. `research/studies/r2-common/lake_sql.py` had no callers and defaulted to the JSON
  format, which the API caps at 10,000 rows.
- Acceptance: both are deleted; studies read the lake through `pitch_edge.lake.client.Lake`.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-23`.

## PE-24 Exported strategies filter pandas frames on every stake call
- Status: open
- Priority: nice (S2)
- Raised by: pitch-edge (E1 profile, with edgelab EL-32)
- Evidence: in a profile of edgelab's backtest of the R2-Q3 bet365 strategy (2018-08 to 2020-03, 14,064
  bars), the strategy's `stake` and `_forecast_now` (`research/audits/edgelab-r2q3/strategies/*/strategy.py`)
  took about 12% of runtime, 6.9 ms per call, filtering arrow-backed pandas frames each time. The rest is
  edgelab's per-bar view build, filed as EL-32.
- Acceptance: the next strategy pitch-edge exports indexes its features by (ts, symbol) once at load, and
  its per-call cost is measured in the same profile. The submitted R2-Q3 folders stay as submitted.
- Blocked by: PE-9. Blocks: none.
- Commits: none yet.

## PE-25 Record which lake build answered a pull
- Status: done
- Priority: should (M8)
- Raised by: lakehouse (LK-1: the API reports a committed build in `X-Lakehouse-Version`)
- Evidence: study pulls and lake-sync record the pull time and row counts but not the lake build, so a
  result cannot be traced to the lake code that produced its data.
- Acceptance: `Lake.version` holds the build that answered the last query; lake-sync records it per window
  in `lake_sync_state.lake_version` (existing warehouses gain the column); new study pulls write it into their
  snapshot file. The first two are done with mocked tests. The study rule is in the research-protocol skill
  (`data_snapshot` and the snapshot file name the lake build); finished studies' pulls predate the header and
  are left as they are.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-25`.

## PE-26 No sharp close in fd_matches from 2026-04-17 to 05-04
- Status: done
- Priority: nice (honest reporting)
- Raised by: lakehouse (LK-42)
- Evidence: `fd_matches` has no Betfair Exchange close on any match from 2026-04-20 to 2026-05-03 and on about
  half the matches of 04-17, 04-19 and 05-04 (about 830 matches, every division); RUS has none in 2026.
  football-data never published these closes (the lake re-read the source files on 2026-10-27), so the lake
  matches its source and the recorded runs stand. Studies scoring against that close drop those matches; none
  falls back to a soft close. Affected: R2-Q1 and R2-Q3 post segments, the R1 Polymarket studies (evaluation
  2025-11 to 2026-10), R5 micro (evaluation 2026-04-09 to 09-29) and R5 lead-lag. R2-Q2, R3 and R6 lose nothing:
  their universes need a Pinnacle open, which the snapshots lack after 2026-01-15, so their post segments end
  there. R5 agreement and arbitrage never read the close. Paper decisions start on 2026-10-20 and are unaffected.
- Acceptance: each affected report states the gap and its match count beside the numbers it affects. If the
  lake's OddsPapi backfill (LK-45) gives a sharp close for the span, a run on it is reported as a sensitivity;
  the recorded first run stays the result of record.
- Result: a "Data gap (PE-26)" paragraph in 14 study reports, with counts from each study's local snapshot (for
  example R2-Q1 863 of 1,029 matches in the span, R5 micro FLB 92 of 107, R2-Q3 5 of 170 eligible).
- Blocked by: none. Blocks: none.
- Commits: none yet.

## PE-27 Propose new hypotheses tradable on current data
- Status: in-progress (waiting for the owner's approval)
- Priority: should (owner goal: more strategies tested on data we have)
- Raised by: coordinator, round 2
- Evidence: Betfair historical data is not obtainable from Sweden and no paid datasets are bought, so new tests
  must use football-data opens and closes, the Betfair close, Polymarket and Kalshi bars, and OddsPapi paths
  from 2026-01 (LK-45).
- Acceptance: a one-page draft of 3-6 hypotheses (at least one basic rule and one ML idea) with data, windows,
  expected test bets and power at +2% CLV; the owner approves or rejects each before any registration.
- Result so far: `research/proposals/2026-10-28-new-hypotheses.md` (commit 5ff6d18) proposes H1 favourite rule,
  H2 end-of-season motivation rule, H3 soft-books-only closing-line model (ML), H4 Pinnacle-to-bet365 lag on
  OddsPapi paths (after LK-45).
- Blocked by: owner approval; H4 also LK-45. Blocks: none.
- Commits: see git log for `PE-27`.
