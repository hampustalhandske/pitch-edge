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
- Paused (owner, 2026-10-28): the launchd agent `local.pitch-edge.paper-loop` is unloaded and the loop exited at
  2026-10-28T07:48:17Z after its 07:15-07:27Z cycle; the plist stays in `~/Library/LaunchAgents/`. Cycles from then
  until the agent is reloaded are paused, not missed: the pause start is recorded in the paper ledger's `pauses`
  table (`paper pause --since … --reason …`), and `paper status` shows "paused since <start>" while no cycle has run
  and, after reload, counts the gap holding the pause as paused cycles apart from missed ones. Every forward result
  states the paused period beside its coverage. The 14 decisions waiting for a lake close are left as they
  are and settle on the first cycle after reload
  (`launchctl load ~/Library/LaunchAgents/local.pitch-edge.paper-loop.plist`).
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
- Progress: `pitch_edge.edgelab.write_submission` writes edgelab's `submission.v1` manifest with the trial
  declaration from the ledger (family, N = the family's trial rows up to a chain hash, the test keys and the
  ledger export inside the folder); `edgelab submission check` accepts it. Left: edgelab using the declared N in
  its deflated Sharpe (EL-4), then a counted run that shows it (owner approval).
- Blocked by: EL-4 (M5 trial accounting); EL-1 is done. Blocks: any further counted edgelab run.
- Commits: see git log for `PE-8`.

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
- Readers of the `cfmd_`-only rows (measured read-only): the six `outcome-models-*-v1` registrations train on 1,489 of
  them through `load_feature_frame` (1,188 in 2023/24, the season with no football-data Pinnacle close, and 301 in
  2026/27; never scored, since they have no Pinnacle price); `ask` and the dashboard take the only 2026/27 fixtures
  of the 11 backtest leagues from them; 39 of those matches are in `pmxt_match_map`; `model_predictions` and
  `backtest_bets` hold rows for them. Owner decision (2026-10-28): keep every `cfmd_`-only row until another source
  covers 2023/24 and 2026/27 for those leagues.
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
- Lead: LK-45 is built (lake b3abc46): OddsPapi Pinnacle, Singbet and Betfair Exchange price paths for UCL,
  UEL and UECL from 2026-01-13, with kickoff UTC; the close is the last row with observed_at <= kickoff_utc
  (median 2.9 minutes before kickoff for Pinnacle on UCL). Outcome codes 101/102/103 = home/draw/away are verified
  (the lake matched OddsPapi's Betfair close to football-data's on 54 of 54 matches, 2026-01-13 to 01-17). Next:
  once the backfill finishes (~2026-10-30), settle UCL paper decisions against the de-vigged Pinnacle close from it.
- Progress (uncommitted, not closed): `paper/settle.py` settles UCL/UEL bookmaker decisions against the de-vigged
  OddsPapi Pinnacle close (`lake:oddspapi.pinnacle_close`; fixture matched by competition, kickoff within 3 h and
  team names via `fixtures.oddspapi_team`; valid only with all three outcomes present, active, price > 1, the
  latest row at most 15 min before kickoff, and a plausible book sum; one lake query per competition and day per
  settle call). A fixture OddsPapi does not list yet, or lists without Pinnacle rows, is retried and never flagged
  before the 30-day horizon; an invalid close gets a `no_benchmark` flag with the reason. `settlements` gain
  `sharp_source`. `paper report` shows a `benchmark` column (`betfair_close` / `pinnacle_close`) with separate rows;
  v2's arm table and `research/summary.py` exclude the Pinnacle-benchmarked rows. The registrations name the
  Betfair Exchange close, so the Pinnacle rows are a deviation to state. Read-only match test on the 2026-10-28
  copy of the paper ledger: 12 UCL decisions of 2026-10-21 (all provisional) match 12/12 fixtures, 11 get a valid
  close (Real Madrid v RB Leipzig is stale, 22 min); the 29 of 3-4 November are listed by OddsPapi without
  Pinnacle rows (recorder stalled, LK-61) and wait. Not yet run: `paper settle` on the live ledger.
- Blocked by: LK-61 (recorder stalled since 2026-10-28) for decisions after 2026-11-01. Blocks: none.
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
- State: the loop is paused by the owner from 2026-10-28T07:48:17Z (see PE-2); the host question returns when it is
  reloaded.
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
  deflated at N=355. N also counts trial rows, so a re-run of the same (hypothesis, config) counts as a second
  test; edgelab counts distinct (hypothesis_id, config_hash) keys and keeps re-evaluations apart (EL-4). The
  submission's `declared_n` (PE-8) uses the same rows count as `_dsr`, so the two change together.
- Acceptance: N counts the distinct return-kind (hypothesis, config) tests of the family, the total is reported
  beside it, `_dsr` and the submission's `declared_n` use the same rule, and the rule matches the one agreed for
  edgelab (E2); a test covers a mixed-kind family and a re-run config.
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
- Addendum (2026-11-05, for AT-40): a third vector covers a `reviews` row (`REVIEW_HASH` `297a6a2f…`, chained after the
  trial vector). Its `detail_json` is `json.dumps(detail, sort_keys=True)` with the default separators, embedded in
  the row JSON as a string. The `amendment` review kind (fe4f694) is a new value of `kind`, not a format change, so
  the format has no version number yet.
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
- Lead for the sensitivity (lake, 2026-11-05): OddsPapi has Betfair Exchange 1X2 paths for 2026-04-19 to 05-04;
  in E0, E1, D1, SP1, I1, F1 and SC0, 173 of 174 fixtures have a betfair-ex path and all 174 a Pinnacle path. A
  close from it is the last row with observed_at <= kickoff_utc. Not run; the recorded first runs stay the result
  of record.
- Blocked by: none. Blocks: none.
- Commits: none yet.

## PE-27 Propose new hypotheses tradable on current data
- Status: in-progress (registered; scoring next)
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
- Registered (owner approved all four, 2026-10-28): family `soft-open-vs-betfair`, one test config each, so the
  family N is 4: `soft-open-vs-betfair-fav-open-v1` (H1), `-late-season-stakes-v1` (H2), `-gbm-close-v1` (H3),
  `-exchange-lag-v1` (H4; Pinnacle steam against Betfair Exchange lag). Each registration's `selection:` block fixes
  how its parameters are chosen on the exploratory window; those candidate runs are kept in the study's
  `trials.csv`, not on the ledger, and only the selected config is scored once on the evaluation window. Next: score
  H1 and H3; count H2's train-window bets before any outcome is read, then score; score H4 after the LK-45 backfill.
  H1's selected config can fall below `min_n` 300 (about 200 bets at θ = 1%).
- H4 (2026-11-05): LK-45 is complete (6,993 fixtures, 6,900 with a path; Pinnacle on 99.6% and Betfair Exchange
  on 98.3% of them). Two more amendments are chained before any pull of record, count or selection: a price <= 1.0
  is no valid quote (the lake holds 518 betfair-ex rows at 0.0 on 2026-02-05/06), and the readings fixed while
  writing the code. Study code in `research/studies/soft-open-vs-betfair-exchange-lag/` (pull, outcome-blind
  count, selection, one score). The evaluation window has 2,388 fixtures with both books, so `min_n` 300 needs
  about one bet per eight fixtures.
- Blocked by: H4's score waits for a stable lake build (LK-61). Blocks: none.
- Commits: see git log for `PE-27`.

## PE-28 The warehouse spine has no 2026/27 football-data matches
- Status: done
- Priority: should (`ask` and the dashboard answer from a stale current season)
- Raised by: coordinator, after the PE-12 decision
- Evidence (2026-10-28, read-only): in the 11 backtest leagues the warehouse's football-data rows end 2026-05-02 to
  05-24, and the only 2026/27 rows are 301 `cfmd_` matches ending 2026-08-30 to 09-03, 55 days behind today. The
  lake mirror `lake_fd_matches` already holds 683 2026/27 matches in those leagues through 2026-10-11 to 10-19. The
  cause is in pitch-edge: `ingest_football_data`, `ingest_openfootball` and `ingest_club_elo` defaulted to seasons
  or dates ending 2025, so the 2026/27 files were never requested. Re-sourcing through the lake is not needed for
  these leagues; the other 28 `cfmd_`-only league codes have no football-data source (PE-12).
- Acceptance: the defaults reach the season under way (`latest_season_start`); after the next `pitch-edge ingest`,
  each of the 11 leagues' last match is within a week of the lake's `fd_matches`, and `pitch-edge health` shows it.
  Until that ingest runs, `ask` and the dashboard state the last match date of each league they answer from.
- Result (2026-10-28): `ingest_football_data` for season 2026 (22 main divisions, no extra leagues, no duplicate
  purge, warehouse backed up first) added 1,981 matches and 90,712 odds rows; the 11 leagues now end 2026-10-24 to
  10-26, later than the lake mirror (10-11 to 10-19). The features stage was not re-run. The 301 `cfmd_` 2026/27
  matches now duplicate football-data fixtures; the next full `ingest` purges them as duplicates (PE-12's purge),
  while the other `cfmd_`-only rows stay.
- Blocked by: none. Blocks: none.
- Commits: see git log for `PE-28`.

## PE-29 Read football-data from the lake only
- Status: open
- Priority: should (joint plan: the lake serves all data; one source, no second copy that can go stale)
- Raised by: coordinator, after PE-28
- Evidence (2026-10-28): the lake's `fd_matches` is fetched daily (latest run 11:04 UTC); in the 11 backtest leagues it
  ends 2026-10-24 to 10-26. pitch-edge holds two other copies:
  - its own ingest (`data/ingest.py` `ingest_football_data`), which feeds the warehouse spine that `ask`, the
    dashboard and the backtests read, and reached the same dates;
  - the warehouse's `lake_fd_matches` mirror from `lake-sync`, which ends 10-11 to 10-19. The warehouse holds no
    `lake_sync_runs`/`lake_sync_state` rows, so the mirror has not been refreshed since it was built, and nothing
    schedules `lake-sync`.
  The studies and the paper settle read the lake API directly.
- Acceptance: the warehouse spine's football-data matches and odds come from the lake, either read through the API
  at decision time or through a `lake-sync` that runs on a schedule and reports its lag in `pitch-edge health`;
  `ingest_football_data` and its connector are removed in the same change; a test covers the spine built from the
  lake rows, with the same columns and point-in-time stamps.
- Blocked by: none. Blocks: none.
- Commits: none yet.
- Progress (2026-11-05, design (b)): the spine's football-data `matches` and `odds` are built from the
  `lake_fd_matches` mirror by `src/pitch_edge/lake/spine.py` (same columns, `match_id` scheme, `source` label and
  `snapshot_ts` stamps; incremental by calendar-year window, checkpoint + fingerprint in `lake_sync_state`, one
  transaction per window). `ingest_football_data`, the connector, `PITCH_EDGE_FD_ARCHIVE_DIR` and their tests are
  deleted; the vocabulary `store_matches` needs lives in `data/fd_vocab.py`. `ingest`/`refresh` call
  `build_football_data_spine`; `lake-sync` rebuilds the spine after `fd_matches`. `scripts/lake_sync_fd.sh` and
  `scripts/launchd/local.pitch-edge.lake-sync-fd.plist.example` (14:00 and 20:00 local, `--skip-if-busy`) are written,
  not installed. `pitch-edge health` shows mirror last sync, spine last build and per-league lag (warns above 3 days
  behind the lake, 48 h since the sync, or any spine-behind-mirror gap). Dry-run against the stale mirror in the
  warehouse (read-only, nothing written): 227,435 spine rows, 6,951,795 odds rows, identical to the stored
  football-data rows on every shared id (main-division odds exact) except: (1) the 16 extra-league files' Pinnacle,
  Max and Avg prices are now closing (`is_closing` True, 450,890 rows; the CSV parse had stored them as opens); (2)
  `ROU` league label `Romania - Superliga` (was `Liga 1`), 513 `ARG` rows capitalised differently; (3) 242 `MEX` rows use
  the lake's `UNAM Pumas` spelling, and 688 `MEX`/`BRA` ids change with the lake's club renames (`Tigres UANL`,
  `Athletico-PR`; no features, weather or backtest rows keyed on them); (4) two `P1` fixtures the lake re-dated by one
  day change id (18 `backtest_bets`, 6 `model_predictions` rows keyed on the old ids); (5) 8 lake-corrected stat cells and
  trailing whitespace trimmed from 86 referee/kickoff strings; (6) +15,851 rows the stored spine lacks (extra leagues
  since 2024-02, and lower-division matches from 2024-02 to 2024-06, e.g. SP2 172, F2 153, E3 152, E2 148). Still open: the first real `lake-sync` plus
  spine build (needs a warehouse slot), the owner's launchd bootstrap, then close.

## PE-30 The football-data CSV ingest stored extra-league closes as opens
- Status: done
- Priority: should (point-in-time: no hindsight price may pass as a pre-match one)
- Raised by: pitch-edge (PE-29 dry run)
- Evidence: for football-data's 16 extra-league files (`fdx_` match ids: MLS, Brazil, Argentina, Mexico, Japan,
  Romania, Poland, Sweden, Norway, Russia, Denmark, China, Ireland, Finland, Austria, Switzerland), the columns
  `PSCH`/`MaxCH`/`AvgCH` (Pinnacle, max and average closing 1X2) were parsed as `PS`/`Max`/`Avg` with
  `is_closing` false. Warehouse table `odds`, source `football_data_co_uk`: 305,069 rows on `fdx_` matches
  (2012-03-02 to 2024-02-22) before the rebuild, 0 after. Readers that could have taken these closes for opens:
  `Warehouse.matches_with_closing_odds` (columns `PSH/PSD/PSA`, `AvgH…`, `MaxH…`), and through it the feature build
  (`features/build.py` early-odds features, so `features` rows of `fdx_` matches), `ask` (`agents/qa_data.py`
  `REAL_ODDS_COLUMNS`), the RAG match documents (`rag/documents.py`, "Pinnacle pre-match odds"), the backtest
  engine's bet price and `pipeline.py`. Unaffected: the 11 backtest leagues (main divisions, `fd_` ids) and every
  registered study (they read the lake directly).
- Acceptance: the spine stores these prices as closes, and the derived tables built from the old rows are rebuilt.
- Result: the spine built from the lake (654f2ee, first build 2026-11-05) stores them as closes. Left as a
  follow-up, not a blocker: `features` rows and the RAG index for `fdx_` matches still hold values derived from
  the old rows until the next `refresh` (a warehouse slot).
- Blocked by: none. Blocks: none.
- Commits: 654f2ee (Refs PE-29).

## PE-31 Match-keyed rows outlive the matches the spine replaces
- Status: open
- Priority: nice (traceability of stored results)
- Raised by: pitch-edge (PE-29 first sync)
- Evidence: `store_matches` deletes a non-football-data `matches` row (and its `odds`) when football-data brings
  the same fixture, but leaves its rows in `features`, `weather`, `model_predictions` and `backtest_bets`. On
  2026-11-05, after the first spine build from the lake, about 84,000 `cfmd_` ids (dated 2005 to 2026) are in
  those tables with no `matches` row: `backtest_bets` 584,420 rows (70,577 ids), `model_predictions` 302,534 rows,
  `features` 41,203 ids, `weather` 12,046 ids. Most predate the build (for example the 2026/27 `cfmd_` ids replaced by
  the football-data 2026/27 ingest on 2026-10-28); the build replaced 4,889 more `cfmd_` matches (2024 fixtures
  football-data now has) and re-keyed two `P1` fixtures the lake re-dated by one day:
  `fd_P1_1920_20200201_SantaClara_PacosFerreira` and `fd_P1_2021_20210103_SantaClara_Benfica`
  (18 `backtest_bets`, 6 `model_predictions` rows). `pitch-edge health` also lists 1,591 `cfmd_`/football-data
  duplicate pairs, which `purge_duplicate_fixtures` removes at the next `ingest`/`refresh` (PE-12).
- Readers (checked 2026-11-05): no report, case study, `ask` answer or dashboard figure reads `backtest_bets`.
  Case studies are written from the CSVs a backtest run produces, `ask` fits its models fresh per question, and the
  dashboard reads neither table; `pipeline.latest_backtest_tables` reads only the latest run (0 orphaned of its
  4,531 bets) and is called only by a test. The orphaned bets sit in older runs (for example `main:gbdt`,
  `main:dixon_coles` and `ping:dixon_coles` of run 20260906T083733). The one live reader is the RAG index: it takes
  the latest `model_predictions` run, in which 262 of 2,384 rows are orphaned `cfmd_` ids (2024-02 to 2026-08), as
  prediction documents. Those documents name teams and dates, not match ids, so they still describe the right
  fixture, possibly beside a football-data duplicate.
- Decision so far: every orphaned row is kept (2026-11-05); this entry is their flag.
- Acceptance: either the replacement step carries the match-keyed rows to the new id or records the old-to-new
  id, or `pitch-edge health` reports orphaned match-keyed rows by table, so a stored result can always be traced.
- Blocked by: none. Blocks: none.
- Commits: none yet.

## PE-32 CI never runs the R6 live-model paper strategy tests
- Status: open
- Priority: nice (test coverage)
- Raised by: pitch-edge (local CI run on `c482c75`, 2026-11-05)
- Evidence: all 10 tests in `tests/unit/test_paper_v2.py` carry `needs_coef`, which skips them unless
  `data/research/r6/v2_live_coefficients.json` exists. `data/` is git-ignored, so on a clean checkout (CI) the
  `soft-vs-sharp-live-v2` golden test, the coefficient-hash guard and the decision-time tests are all skipped.
- Fix: split the tests. Keep the hash check against the real file behind `needs_coef`. Run the model,
  hash-guard and decision-time logic against a small synthetic coefficients file written in `tmp_path`, so they
  run everywhere.
- Blocked by: none. Blocks: none.

## PE-33 Paper settle can take a days-old Polymarket bar as the close
- Status: open
- Priority: high (paper CLV and P&L correctness)
- Raised by: pitch-edge (lake LK-88, 2026-11-06)
- Evidence: `LakeCloseSource.pm_last_bars` takes the last hourly bar per token that is complete by kickoff,
  looking back 14 days, and returns the price without the bar's time. `_pm_bar_close` then uses it as the close
  with no age check. The lake's `pm_soccer_bars_1h` has no bars after 2026-11-02 02:00 UTC (LK-87, LK-88).
  So a Polymarket decision with a later kickoff settles on a bar that ended hours or days before kickoff,
  and it is labelled `lake:pm_soccer_bars_1h` like a true close. The same happens for any token that stops
  trading. On 2026-11-06 the paper ledger holds one unsettled Polymarket decision with kickoff 2026-11-02 17:30,
  so its close would come from a bar more than 15 hours old. The paper loop is unloaded and
  `paper settle` has not been run since, so nothing stale has been written yet.
- Fix: return each bar's `hour_start` with its close. Refuse a close whose newest bar ended more than a set
  age before kickoff (choose the limit from the bar-age distribution of settled decisions, the way
  `ODDSPAPI_MAX_CLOSE_AGE` bounds the Pinnacle close). Let such a decision fall to the provisional fallback or
  a `no_benchmark` flag. `tests/unit/test_paper_settle_lake_equivalence.py` repeats the query and needs the
  same change.
- Until fixed: do not run `paper settle` on the live paper ledger.
- Decision (2026-11-06, master go with four conditions): a refused close becomes `no_benchmark` with its reason and
  is reported apart from the rest (as in PE-13). No other source's price replaces it, so no provisional row. Past
  settlements are not rewritten. The paper loop stays unloaded.
- Limit: **3 hours** between the end of a token's newest complete bar and kickoff (`PM_MAX_BAR_AGE`). It was
  chosen from the bar-age distribution alone, with no P&L or CLV looked at. Sample: every Yes token of a mapped
  Polymarket 1X2 market (`win_home`/`draw`/`win_away` with an `fd_match_key`) whose `close_at` (kickoff) is in
  2025-08-01..2026-10-31. That is 8,992 markets; 8,438 have a bar in the 14 days before kickoff. Age = kickoff
  minus (newest `hour_start` + 1 h). Median 0.5 h, p75 0.75 h, p90 2.5 h, p95 8 h, p99 113 h, max 317 h.
  Cumulative share of markets with a bar: 82.4% under 1 h, 87.8% under 2 h, 90.6% under 3 h, 94.0% under 6 h,
  97.5% under 24 h. Reason: hourly bars and kickoffs at :15/:30/:45 put the freshest possible bar up to 0.75 h
  before kickoff. New markets come in at 460 per extra hour between 1 and 2 h and 235 between 2 and 3 h. After
  that the rate drops to about 95 per hour (3-6 h) and 30 per hour (6-12 h), a long tail of tokens that stopped
  trading. So 3 h allows the hourly grid plus one quiet hour, and anything older is a stale price, not a close.
- Re-check of past settlements: all 88 are bookmaker decisions settled on `lake:fd_matches.betfair_ex_close`. None
  used a Polymarket bar, so 0 would be refused. The provisional rows are 12: 9 Pinnacle-venue rows and 3
  Polymarket-venue rows (kickoff 2026-10-21). One of the Polymarket rows uses our own Polymarket snapshot; the
  other two have no close. Those three tokens have no bar at all in the 14 days before kickoff, so the age rule
  does not apply to them, and their rows stay as written.
- Progress (2026-11-06): `PM_MAX_BAR_AGE` implemented in `paper/settle.py`. `pm_last_bars` now returns each bar's
  end time (`PmBar`). A refused close waits inside `grace`. After that it gets one `no_benchmark` flag with its
  reason and no provisional row, and it can still settle on a fresh bar until `final_horizon`. A stale bar on a
  token other than the decision's own drops only the three-way normalisation. 6 new tests.
- Read-only check against the live lake: of the 3 unsettled Polymarket decisions, 2026-11-02 17:30 is refused (own bar
  17.5 h old). 2026-11-01 14:00 has fresh bars (0 h) and would settle. 2026-10-31 15:30 has no bars at all.
- Open question: a Polymarket decision whose tokens have **no** bar at all still gets the existing fallback after
  `grace`, our own last pre-kickoff snapshot written as a provisional row. That happened on 2026-10-21 and would
  happen to the 2026-10-31 decision. The age rule does not cover this case. Treating it as `no_benchmark` too is
  the master's call.
- Blocked by: none. Blocks: the next live `paper settle`.
