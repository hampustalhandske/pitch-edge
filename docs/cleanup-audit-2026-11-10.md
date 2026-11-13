# pitch-edge cleanup audit (2026-11-10)

This audit is read-only and covers code, `docs/ISSUES.md`, tests and CI config at `c57e7be`. It ran no test
suites and sent no lake requests. The standard is the master's engineering rules (1-10) and section 1 of the
cleanup plan. New findings are filed as PE-36 to PE-44 in `docs/ISSUES.md`, which holds the full evidence. This
file is the ordered list.

Batches:
- **B1** is offline work in pitch-edge only.
- **B2** needs another project's release (named per item).
- **Later** means it does not stand between pitch-edge and trusted results.

## 1. The ten rules

| Rule | Verdict | Evidence | Finding |
|---|---|---|---|
| 1 Determinism | **gap** | Ten row picks can tie. Examples: `lake/mirror.py:360-363`; `paper/store.py:466,474`; `paper/settle.py:183-184,254,336`; `features/squad_value.py:133`; `features/live_lineups.py:90`; `features/context.py:134`; `odds/steam.py:90-92`. No two-process or kill-resume test exists anywhere. Float aggregates pass: SQL sums are over integer-valued minutes, numpy reductions run in a fixed order, and every RNG is seeded (`research/stats.py:115`, `models/stochastic.py:38`, `models/gbdt.py:49`, `models/sequence.py:211`). | PE-36, PE-37 |
| 2 Complete checkpoints | **gap** | Checkpoints are transactional by construction: the mirror commits a window and its checkpoint in one transaction (`lake/mirror.py:340-375`), and the spine and the ledgers are append-only. But resume is tested only by in-process raises, with no generic state walk. Module-level state has no census: `research/summary.py:543-544`, `dashboard/web.py:240`, `paper/strategies.py:425`. | PE-37, PE-38 |
| 3 Vendored contracts | **gap** | The edgelab manifest and bundle checks skip in CI (`test_edgelab_submission.py:116`, `test_edgelab_export.py:203`). `strategy.yaml` is hand-templated (`edgelab/scaffold.py:257-275`). There are no vendored lake schemas. The ledger format has test vectors but no exported, versioned schema (PE-21). | PE-39 |
| 4 Fail closed | **gap** | The mirror dedupes by its key columns when there is no `scraped_at` (`lake/mirror.py:362`). A null lake build is accepted (`lake/client.py:119`). Model fallbacks go unrecorded (`models/poisson.py:67-73`, `models/dixon_coles.py:145-150`). CI runs only on `main` and on PRs (`.github/workflows/ci.yml:3-6`). These pass: the paper pause fails closed (`scripts/paper_loop.sh:53-60`), the gate scripts refuse a non-commit, ladder criteria read `not-evaluated` and block (`research/gates.py:845-866`), and the paper settle writes fallbacks only to `provisional_settlements`. The `ask`/LLM fallbacks are narration by design and are excluded. | PE-40 |
| 5 Negative controls | **gap** | No census of the ladder criteria (`research/gates.py:105-130`). The six study test files are outside `testpaths` (`pyproject.toml:75`). The 13 CI skips are unbounded (10 of them are PE-32). The gate scripts were never shown to fail on a planted failure. | PE-41, PE-32 |
| 6 Resilient boundaries | **gap** | The lake client retries only HTTP 503 (`lake/client.py:112-122`). The web collectors pass: `data/http.py:139,199` retries 5xx/429 with backoff and a circuit breaker. Slot release is on the lake's side (LK-111). | PE-42 |
| 7 Data keeps history | **gap** (consumer) | Reads use the current views and record only the build. The R2/R3/R6/R7 pulls have no snapshot file (`data/research/{r2,r3,r6,r7}`). The warehouse spine is rebuilt whole (PE-31). These pass: the research ledger is append-only and hash-chained; the paper ledger is append-only. | PE-43, PE-31, PE-34 |
| 8 Controls in code | **gap** | The paper pause now lives in the ledger (PE-35, `paper/cli.py` gate, `scripts/paper_loop.sh:53`), and the odds-api budget is in code (`paper/collectors.py:172`). But the lake freeze is held only by discipline: `lake-sync-fd` will run at 14:00. Scheduled jobs also run the live working tree (`scripts/lake_sync_fd.sh:17`, `scripts/paper_loop.sh:42`). | PE-44 |
| 9 Evidence in repo | **pass** | Run outputs go to the git-ignored `data/` (study pulls under `data/research/<id>/`, gate logs under `data/logs/`). No code writes to `/private/tmp`. One past loss is recorded: the non-counted `r2q3-max-nopersist` dry-run strategy lived in a wiped scratchpad (`research/HANDOFF.md:376`). | none |
| 10 Run when ready | **pass** | No booked windows in pitch-edge. The fixed `lake-sync-fd` times (14:00/20:00) carry their reason in the plist: the lake fetches at about 11:00 UTC, and the second start is a retry. | none |

## 2. Findings in order (new and re-triaged)

Each finding lists the rule it breaks, its size, the test that proves the fix with its negative control, what it
blocks or is blocked by, and its batch.

**Must**

1. **PE-36 Total order on row picks** (rule 1)
   - Size: M.
   - Test: per pick, insert the tied rows in both orders and require the same pick.
   - Negative control: the test fails with the tie-break removed.
   - Blocks: the paper re-settle check, the mirror re-sync and the feature store.
   - Batch: B1.
2. **PE-37 Two-process and kill-resume tests** (rules 1 and 2) for the trial run with grade and ledger export,
   paper decide/settle/report, mirror and spine, edgelab export and submission, and backtests.
   - Size: L.
   - Test: two subprocesses must produce byte-identical output. A SIGKILL mid-run, then a resume, must equal an
     uninterrupted run.
   - Negative control: a planted unseeded draw and a planted skipped checkpoint each make it fail.
   - Blocked by: PE-36.
   - Blocks: every rerun.
   - Batch: B1.
3. **PE-41 Coverage census for gates and suites** (rule 5)
   - Size: M.
   - Test: every criterion is reached as pass and as fail/not-evaluated; the study tests run in CI; the skip set
     is pinned.
   - Negative control: delete one gate's fail-case test, or add a skip, and the census fails.
   - Batch: B1.
4. **PE-32 R6 v2 paper tests skip in CI** (rule 5; re-triaged from backlog to must)
   - Size: S.
   - Test: a synthetic coefficients file in `tmp_path`, with the hash guard kept behind `needs_coef`.
   - Negative control: a wrong coefficient hash fails.
   - Batch: B1.
5. **PE-39 Vendored, versioned contracts** (rule 3)
   - Size: L.
   - Test: every written manifest, bundle and `strategy.yaml`, and every read lake frame, validates against a
     vendored schema in CI without the other repo.
   - Negative control: a planted bad artifact fails.
   - Blocked by: edgelab's schemas (with EL-38/EL-74) and the lake's table schemas.
   - Blocks: the next counted edgelab run, and agentic's vendoring of the ledger format.
   - Batch: the producer-side ledger format in B1; the rest in B2.
6. **PE-8 Declared N on edgelab submissions** (M5, existing)
   - Size: S on our side.
   - Test: a submission's `declared_n` equals the ledger rule.
   - Negative control: a re-run config row changes N under the old rule.
   - Blocked by: EL-4 and PE-20.
   - Blocks: regrading R2-Q3 bet365.
   - Batch: B2.
7. **PE-20 Family N counts distinct return-kind tests** (M5, existing)
   - Size: S.
   - Test: a mixed-kind family with a re-run config.
   - Negative control: the old N (355) is reproduced by the old rule only.
   - Blocked by: the E2 agreement with edgelab.
   - Batch: B2.
8. **PE-35 Paper loop obeys the ledger pause.** Built and CI-passed. It is done once the owner pushes `c57e7be`.
   No further work.

**Should**

9. **PE-40 Fail-closed gaps** (rule 4)
   - Size: M.
   - Test: plant a batch with no order, a null build, and a forced model fallback; each must refuse or be recorded.
     CI must run on every branch.
   - Batch: B1.
10. **PE-44 Scheduled jobs pinned to a sha and freeze-aware** (rule 8)
    - Size: M.
    - Test: plant a dirty tree and an active freeze; each must refuse.
    - Interim, while B1 is being built: disable `lake-sync-fd` or point it at a pinned worktree (owner/master
      call).
    - Batch: B1.
11. **PE-42 Lake client retries transport errors** (rule 6)
    - Size: S.
    - Test: mocked ConnectError, timeout, 502 and a cut stream are retried and then succeed.
    - Negative control: a 400 is not retried.
    - Batch: B1.
12. **PE-38 Module-state census** (rule 2)
    - Size: S.
    - Test: the census allowlist.
    - Negative control: a planted dict global fails.
    - Batch: B1.
13. **PE-43 Pin lake reads to (build, T)** (rule 7; must before a counted re-pull)
    - Size: M.
    - Test: a mocked lake with two builds returns each build's rows.
    - Negative control: an unpinned read is refused.
    - Blocked by: LK-109 and LK-13.
    - Batch: B2.
14. **PE-34 Re-sync `lake_tm_injuries`** (existing)
    - Size: S.
    - Test: the mirror holds one row per key, matching the lake's tie-broken view.
    - Blocked by: LK-13.
    - Batch: B2.
15. **PE-13 UCL benchmark** (existing)
    - Its settle code is committed in `af38e21`; the issue text said uncommitted and is now corrected. Its
      OddsPapi close carries a PE-36 tie.
    - Remaining: a `paper settle` run on the live ledger, which waits for the loop's reload (owner).
    - Batch: after B1.
16. **PE-31 Orphaned match-keyed rows** (traceability, rule 7, existing)
    - Size: M.
    - Test: `health` reports orphans by table.
    - Negative control: a replaced match leaves a counted orphan.
    - Batch: B1 or later.
17. **PE-9 Submit through edgelab's client** (existing)
    - Size: M.
    - Blocked by: EL-2 and EL-6. Overlaps PE-39 (it removes the hand-built scaffold).
    - Batch: B2.

**Later**: PE-10 (edgelab bet-level acceptance data; blocked by EL-5, EL-11), PE-16 (always-on host, deferred by
the owner), PE-24 (export performance).

## 3. Cross-project contracts

| Contract | Direction | Schema and version | Vendored | Tested in CI without the other repo |
|---|---|---|---|---|
| Lake HTTP API (`/v1/sql` arrow, `X-Lakehouse-Version`, 503 + `Retry-After`) | consumes | none; described in the `lake/client.py` docstring | client code is a vendored copy; no schema | yes, mocked (`test_lake_client.py`), but the mocks are hand-built |
| Lake tables read (`fd_matches`, `oddspapi_*`, `pm_soccer_*`, `kalshi_*`, `us_team_matches`, `clubelo_ratings`, `tm_*`, `club_alias`) | consumes | none (the lake's `describe` only) | no | hand-built frames only |
| edgelab `submission.v1` manifest | produces | `SCHEMA_VERSION = 1` (`edgelab/submission.py:31`) | no | **no**: the check skips without `EDGELAB_DIR` |
| edgelab data bundle (book snapshots, resolutions, features, bars) | produces | edgelab's `data.schemas`, unversioned on our side | no | **no**: skips |
| edgelab `strategy.yaml` | produces | hand template, `version: "1.0.0"` | no | no (the AT-151 pattern) |
| edgelab result/verdict (L4 reviews) | consumes | recorded by hand through `research review` | no | n/a (no code reads it) |
| Research-ledger chain format and ledger export | produces, for agentic-trading (AT-40) and edgelab (EL-4) | no version number; three fixed vectors | consumers reimplement it | producer-side vectors only |
| `docs/PIPELINE.md` (PE-19) | produces (prose) | none | n/a | n/a |

Every row marked "no" is PE-39.

## 4. Results and runs to redo after the fixes (owner ruling 11-10 07:40 (4))

1. **Research ledger.** This covers every counted trial: 681 trial rows, 37 hypotheses and 7 families, from
   2026-10-20 to 11-06.
   - Rerun after PE-36 and PE-37, in fresh processes, against the stored pulls. Every rerun must reproduce its
     result of record, and L1's "reruns reproduce" criterion records the outcome.
   - Order: the signed L4 findings first (R6 H1-H3), then R2-Q3 (the best strategy, at L2), then the rest.
   - The R2, R3, R6 and R7 pulls have no snapshot file, so their data cannot be named by build. A fresh,
     build-pinned pull is a new snapshot (PE-43, after LK-109), not a reproduction. **Owner question:** are
     reruns from the stored pulls enough, or must they be re-pulled pinned?
2. **edgelab R2-Q3 bet365 grade.** It was graded at N=1 and must be regraded with the declared family N (PE-8,
   PE-20, EL-4). This is a counted run and needs the owner's approval.
3. **Lake mirrors.**
   - Re-sync all 10 after PE-36 and LK-13. The bars and candles take about 25 minutes, so this is a heavy job and
     goes in the queue.
   - Rebuild the spine from them, which also covers PE-34's `lake_tm_injuries` re-sync.
4. **Paper ledger.**
   - After PE-36, a read-only check (after "record done") of whether any final settlement hit an OddsPapi or bar
     tie.
   - The ledger is append-only, so a wrong settlement is flagged, not rewritten.
   - Regenerate `paper report`. Coverage is already 9-11% under the PE-35 rule.
5. **Evidence-layer backtests and case studies** (below the display floor): rerun after the PE-36 feature tie-breaks
   (squad value, usual XI, referee). Later.

## 5. Restarts

- **Paper loop (launchd `local.pitch-edge.paper-loop`):** disabled. The PE-35 fix takes effect when the owner
  reloads it with `launchctl enable gui/$(id -u)/local.pitch-edge.paper-loop` and then a load. Nothing to restart
  now.
- **lake-sync-fd (launchd, loaded):** it picks up whatever code is checked out at its next start, so it needs no
  restart. That is also the PE-44 risk. If PE-44 pins it to a worktree, the plist changes and the agent must be
  unloaded and loaded.
- **Lake API restart for LK-111:** pitch-edge needs no restart. A sync running at that moment fails and resumes at
  its next start (PE-42 makes it retry). Each session's lakehouse MCP connection may need a reconnect.
- **Dashboard (`pitch-edge serve`):** not running. If it is started before the fixes, restart it afterwards,
  because `_ASK_CACHE` holds the graph for the process's life.

## Counts

- **By rule (gaps):**
  - R1: 2 (PE-36, PE-37)
  - R2: 2 (PE-37, PE-38)
  - R3: 1 (PE-39)
  - R4: 1 (PE-40)
  - R5: 2 (PE-41, PE-32)
  - R6: 1 (PE-42)
  - R7: 3 (PE-43, PE-31, PE-34)
  - R8: 1 (PE-44)
  - R9: 0
  - R10: 0
- **By triage:**
  - must 8: PE-36, 37, 41, 32, 39, 8, 20, 35 (35 is done pending the push).
  - should 9: PE-40, 44, 42, 38, 43, 34, 13, 31, 9.
  - later 3: PE-10, 16, 24.
