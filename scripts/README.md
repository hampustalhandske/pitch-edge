# Run scripts

On an Apple-silicon Mac whose shell runs under Rosetta, prefix `uv` commands with `arch -arm64`.

## Forward paper-trading loop (always on)

`scripts/paper_loop.sh` runs the paper ledger (`pitch-edge paper …`, own DB `data/paper/paper.duckdb`, separate
from the warehouse). Paper only: every HTTP call is a public, read-only market-data GET and no code path places
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
  hours with a cycle) next to its CLV. Gaps are reported, never filled.
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
# install / start
sed -e "s|__REPO_DIR__|$PWD|g" -e "s|__HOME__|$HOME|g" \
  scripts/launchd/local.pitch-edge.paper-loop.plist.example > ~/Library/LaunchAgents/local.pitch-edge.paper-loop.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.pitch-edge.paper-loop.plist

# check
launchctl print gui/$(id -u)/local.pitch-edge.paper-loop | grep -E 'state|pid'
tail data/logs/paper_loop.log
arch -arm64 uv run pitch-edge paper status     # row counts, quotes per source, odds-api credits left
arch -arm64 uv run pitch-edge paper report     # per strategy: decisions, settlements, CLV with CI, ROI

# stop (stays stopped until the next login or bootstrap)
launchctl bootout gui/$(id -u)/local.pitch-edge.paper-loop
```

Without launchd (e.g. for a one-off session): `nohup caffeinate -i scripts/paper_loop.sh >/dev/null 2>&1 &`,
stopped with `kill $(cat data/paper/paper_loop.pid)`. The interval can be overridden with
`PAPER_LOOP_INTERVAL_S` (default 3600).

## Football-data mirror and spine (twice daily)

`scripts/lake_sync_fd.sh` runs `pitch-edge lake-sync --only fd_matches --skip-if-busy`: it mirrors the lake's
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
# install / start (the owner does this; nothing here loads it)
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

Without launchd: `scripts/lake_sync_fd.sh`. The first run on a warehouse with no mirror pulls the whole table
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
