# Run scripts

On an Apple-silicon Mac whose shell runs under Rosetta, prefix `uv` commands with `arch -arm64`.

## Scheduled jobs run a pinned commit

The launchd agents below never run the working checkout. They run `scripts/paper_loop.sh` and
`scripts/lake_sync_fd.sh` from the detached worktree `.worktrees/scheduled`, at a sha the owner names:

```bash
# 1. stop the agents first (a running loop keeps executing the worktree's files as they change)
launchctl bootout gui/$(id -u)/local.pitch-edge.paper-loop
launchctl bootout gui/$(id -u)/local.pitch-edge.lake-sync-fd
# 2. pin: create or move .worktrees/scheduled to <sha>, `uv sync --extra dev --locked` there, record the pin
scripts/scheduled_pin.sh <sha>
# 3. install the plists (see each job below) and bootstrap them again
```

`scheduled_pin.sh` refuses (exit 2) anything that is not a commit and (exit 3) a `.worktrees/scheduled` that is
dirty or not a worktree of this repository. When the checkout or `uv sync` fails, or leaves the worktree changed,
it records nothing (exit 1). On success it appends `<utc time> <full sha>` to `data/control/scheduled_pin.log`.

The plists set `PITCH_EDGE_DATA_DIR=<repo>/data` and `PITCH_EDGE_ENV_FILE=<repo>/.env`, so the pinned worktree
writes the main checkout's warehouse, ledgers and logs and reads its `.env` (both variables default to `data` and
`.env` in the tree the script runs from).

**Guard** (`scripts/scheduled_guard.sh`, run once at the start of both scripts): the script logs a
`refusing to start: …` line and exits 0 (so launchd's KeepAlive does not respawn the paper loop) when

- `git status --porcelain` in its tree fails or prints anything (modified, staged or untracked files);
- its HEAD is not the last sha in `<data dir>/control/scheduled_pin.log`, or that log has no sha.

## Lake freeze

A lake freeze (the lake's owner is rebuilding or migrating it) is recorded in code, in the append-only log
`data/control/lake_freeze.jsonl`. While it is active, `Lake()` refuses with `LakeFrozenError` (so no sync, settle
or study pull reads the lake), `pitch-edge lake-sync` prints `skipped: lake freeze since … (reason)` and exits 0
without contacting the lake, and `pitch-edge control gate` exits 10. A freeze lasts from its start until the first
thaw after it. A freeze log that cannot be read or parsed counts as frozen (gate exit 10, `Lake()` refuses) until it
is repaired by hand.

```bash
arch -arm64 uv run pitch-edge control freeze --reason "lake rebuild LK-111" [--at 2026-11-10T09:11:00Z]
arch -arm64 uv run pitch-edge control status        # frozen since when and why, or "lake not frozen"
arch -arm64 uv run pitch-edge control gate          # exit 0 not frozen, 10 frozen (for scripts)
arch -arm64 uv run pitch-edge control thaw --reason "rebuild done" [--at …]   # refused when nothing is frozen
```

## Forward paper-trading loop (always on)

`scripts/paper_loop.sh` runs the paper ledger (`pitch-edge paper …`, own DB `data/paper/paper.duckdb`, separate
from the warehouse), from the pinned worktree after the guard above. Paper only: every HTTP call is a public, read-only market-data GET and no code path places
an order.

- **Every hour:** `paper collect --sources polymarket,kalshi` (public books), `paper collect --sources odds_api`
  (the-odds-api, regions uk,eu, 2 credits per league; each league at most once per 23.5 h, never while fewer
  than 60 credits remain, league list trimmed so the projected spend stays ≤ 450 credits/month), then
  `paper decide` (every registered paper strategy).
- **Every hour, after `decide`:** `paper settle` against the lake's closing prices, so a decision settles in the
  first cycle after the lake has its close.
- **Missed cycles:** a cycle that starts more than 1.5 x the interval after the previous one is logged as
  `missed N cycles since <ts>`. `paper status` lists cycles observed vs expected and the longest gaps (derived
  from `decide_log`); `paper report` prints each primary segment's `window_coverage` (share of evaluation-window
  hours with a cycle outside a pause) and `window_paused` next to its CLV. Gaps are reported, never filled.
- **Pause:** the pause lives in the paper ledger, not in launchd. `paper pause --since <utc> --reason …` records
  it; at the start of every cycle `paper gate` reads it, and while a pause is active the loop logs
  `paused in the ledger` and exits 0 (launchd does not restart a clean exit), so a reboot or reload cannot trade.
  `paper collect`, `decide` and `settle` also do nothing while paused. Only `paper resume --at <utc> --reason …`
  ends a pause; hours between the two count as paused, not as coverage.
- **Log:** `data/logs/paper_loop.log`; launchd's own stdout/stderr go to `data/logs/paper_loop.launchd.log`.
- **Single instance:** the script writes its pid to `data/paper/paper_loop.pid` and refuses to start while that
  pid is alive; the pidfile is removed on exit.
- **Sleep-safe:** it waits in 60-second steps until the next cycle is due by wall clock, so after the lid is
  opened an overdue cycle starts within a minute. Quotes that could only have been collected while the machine
  slept are lost (market snapshots cannot be backfilled).

It runs under the launchd agent `local.pitch-edge.paper-loop`
(template `scripts/launchd/local.pitch-edge.paper-loop.plist.example`): started at login (`RunAtLoad`), restarted after a crash
(`KeepAlive` on unsuccessful exit, throttled to once a minute), not restarted after a clean exit. launchd needs
absolute paths, so the template has `__REPO_DIR__` / `__HOME__` placeholders that the install step fills in for
this checkout; re-run it if the repo moves.

```bash
# install / start (after scripts/scheduled_pin.sh <sha>; the agent runs .worktrees/scheduled/scripts/paper_loop.sh)
sed -e "s|__REPO_DIR__|$PWD|g" -e "s|__HOME__|$HOME|g" \
  scripts/launchd/local.pitch-edge.paper-loop.plist.example > ~/Library/LaunchAgents/local.pitch-edge.paper-loop.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.pitch-edge.paper-loop.plist

# check
launchctl print gui/$(id -u)/local.pitch-edge.paper-loop | grep -E 'state|pid'
tail data/logs/paper_loop.log
arch -arm64 uv run pitch-edge paper status     # row counts, quotes per source, odds-api credits left
arch -arm64 uv run pitch-edge paper report     # per strategy: decisions, settlements, CLV with CI, ROI

# pause (the loop exits at its next cycle and stays stopped across reboots until a resume)
arch -arm64 uv run pitch-edge paper pause --since 2026-10-28T07:48:17Z --reason "owner: not needed yet"
# resume (then start the agent again)
arch -arm64 uv run pitch-edge paper resume --at <now, ISO 8601 UTC> --reason "owner reloaded the loop"

# stop the agent itself (stays stopped until the next login or bootstrap; `launchctl disable` makes it stick)
launchctl bootout gui/$(id -u)/local.pitch-edge.paper-loop
```

Without launchd (e.g. for a one-off session): `PITCH_EDGE_DATA_DIR=$PWD/data PITCH_EDGE_ENV_FILE=$PWD/.env nohup
caffeinate -i .worktrees/scheduled/scripts/paper_loop.sh >/dev/null 2>&1 &`, stopped with
`kill $(cat data/paper/paper_loop.pid)`. The interval can be overridden with
`PAPER_LOOP_INTERVAL_S` (default 3600).

## Football-data mirror and spine (twice daily)

`scripts/lake_sync_fd.sh` runs, after the guard above, `pitch-edge control gate` (while the lake is frozen it logs
`skipped: lake frozen` and exits 0; if the gate itself fails it exits 1 without syncing) and then
`pitch-edge lake-sync --only fd_matches --skip-if-busy`: it mirrors the lake's
`fd_matches` into `lake_fd_matches` (pulled in yearly windows, each committed with its checkpoint) and then rebuilds
the warehouse spine's football-data `matches` and `odds` rows from the mirror. The spine rebuild is incremental: a
calendar-year window is redone only when its mirror rows changed (fingerprint kept in `lake_sync_state`).
`pitch-edge ingest` / `refresh` rebuild the spine from the mirror too (no network), so they are only as fresh as the
last sync.

- **Schedule:** launchd agent `local.pitch-edge.lake-sync-fd` (template
  `scripts/launchd/local.pitch-edge.lake-sync-fd.plist.example`) starts the script at 14:00 and 20:00 local time,
  under `caffeinate -i`. The lake fetches football-data at about 11:00 UTC; 14:00 is 12:00 UTC in Stockholm summer time,
  so adjust both hours for another zone (keep the first at or after 12:00 UTC). A start missed while the Mac slept runs
  once on wake.
- **Warehouse busy:** DuckDB is single-writer. If `data/pitch_edge.duckdb` is open in another process, the run prints
  `skipped: warehouse busy`, pulls nothing and exits 0; the later start retries. A run killed half-way resumes at its
  first uncommitted window next time.
- **Log:** `data/logs/lake_sync_fd.log` (one `start` and one `exit <code>` line per run, plus the sync output);
  launchd's own stdout/stderr go to `data/logs/lake_sync_fd.launchd.log`.
- **Lag:** `pitch-edge health` prints the mirror's last sync, the spine's last build and, per backtest league, the
  newest completed match in the mirror, in the spine and in the lake. It warns when the lake is more than 3 days ahead
  of the mirror, when the mirror was last synced more than 48 h ago, or when the spine is behind the mirror.

```bash
# install / start (the owner does this; nothing here loads it). Pin first: scripts/scheduled_pin.sh <sha>
sed -e "s|__REPO_DIR__|$PWD|g" -e "s|__HOME__|$HOME|g" \
  scripts/launchd/local.pitch-edge.lake-sync-fd.plist.example > ~/Library/LaunchAgents/local.pitch-edge.lake-sync-fd.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.pitch-edge.lake-sync-fd.plist

# check
launchctl print gui/$(id -u)/local.pitch-edge.lake-sync-fd | grep -E 'state|runs|last exit'
tail data/logs/lake_sync_fd.log
arch -arm64 uv run pitch-edge health              # football-data lag table and warnings

# run now (same as a scheduled start)
launchctl kickstart gui/$(id -u)/local.pitch-edge.lake-sync-fd

# stop (stays stopped until the next bootstrap)
launchctl bootout gui/$(id -u)/local.pitch-edge.lake-sync-fd
```

Without launchd: `.worktrees/scheduled/scripts/lake_sync_fd.sh` with `PITCH_EDGE_DATA_DIR=$PWD/data` and
`PITCH_EDGE_ENV_FILE=$PWD/.env` (run from a checkout that is not the pinned sha, it refuses). The first run on a warehouse with no mirror pulls the whole table
(about 230k rows, 23 yearly windows) and builds every spine window.

## Warehouse summary

`uv run python scripts/data_summary.py [--json out.json]` lists every table in the warehouse (row count, column
count, date range where one exists), read from `information_schema.tables` through a read-only connection. Use
it to check whether `src/pitch_edge/features/README.md`'s numbers are current or to find tables no doc mentions.

## Pipeline runs

These are `pitch-edge` commands rather than scripts; run long ones with `nohup caffeinate -i … &` so a laptop
sleep doesn't stall them, and never run two warehouse-writing stages at once (DuckDB is single-writer).

- `uv run pitch-edge refresh [--fast]` — ingest → features → backtest → RAG index in one shot.
- `uv run pitch-edge schedule` — the APScheduler loop: hourly news/markets job, nightly full refresh at 03:30.
- `uv run pitch-edge lake-sync [--only …] [--full-refresh] [--no-spine] [--skip-if-busy]` — mirror lakehouse tables
  into `lake_*` tables; checkpointed per window, so re-running the same command resumes an unfinished sync. Syncing
  `fd_matches` also rebuilds the football-data spine (`--no-spine` skips that).
