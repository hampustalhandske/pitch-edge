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

## Warehouse summary

`uv run python scripts/data_summary.py [--json out.json]` lists every table in the warehouse (row count, column
count, date range where one exists), read from `information_schema.tables` through a read-only connection. Use
it to check whether `src/pitch_edge/features/README.md`'s numbers are current or to find tables no doc mentions.

## Pipeline runs

These are `pitch-edge` commands rather than scripts; run long ones with `nohup caffeinate -i … &` so a laptop
sleep doesn't stall them, and never run two warehouse-writing stages at once (DuckDB is single-writer).

- `uv run pitch-edge refresh [--fast]` — ingest → features → backtest → RAG index in one shot.
- `uv run pitch-edge schedule` — the APScheduler loop: hourly news/markets job, nightly full refresh at 03:30.
- `uv run pitch-edge lake-sync [--only …] [--full-refresh]` — mirror lakehouse tables into `lake_*` tables;
  checkpointed per window, so re-running the same command resumes an unfinished sync.
