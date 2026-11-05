#!/usr/bin/env bash
# Daily football-data refresh: mirror the lake's fd_matches into the warehouse (`lake_fd_matches`), then rebuild the
# football-data `matches`/`odds` spine rows from it.
#
# Runs once per launchd start (scripts/launchd/local.pitch-edge.lake-sync-fd.plist.example). If another process holds
# data/pitch_edge.duckdb (DuckDB is single-writer: a refresh, a backtest, the dashboard), the sync logs
# "skipped: warehouse busy" and exits 0; the next scheduled start retries. The sync is checkpointed per window, so a
# run killed half-way (lid closed, reboot) resumes where it stopped on the next start.
# Log: data/logs/lake_sync_fd.log. Manual run: scripts/lake_sync_fd.sh
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="data/logs/lake_sync_fd.log"
mkdir -p data/logs
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "$(stamp) lake-sync fd_matches start" >>"$LOG"
arch -arm64 uv run pitch-edge lake-sync --only fd_matches --skip-if-busy >>"$LOG" 2>&1
rc=$?
echo "$(stamp) lake-sync fd_matches exit $rc" >>"$LOG"
exit $rc
