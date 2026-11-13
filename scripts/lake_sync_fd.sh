#!/usr/bin/env bash
# Daily football-data refresh: mirror the lake's fd_matches into the warehouse (`lake_fd_matches`), then rebuild the
# football-data `matches`/`odds` spine rows from it.
#
# Runs once per launchd start (scripts/launchd/local.pitch-edge.lake-sync-fd.plist.example), from the pinned
# worktree `.worktrees/scheduled` (scripts/scheduled_pin.sh). In order:
#   1. Guard (scripts/scheduled_guard.sh): refuses, logs a line and exits 0 when this tree has uncommitted changes or
#      its HEAD is not the last sha in the pin log.
#   2. `pitch-edge control gate`: while a lake freeze is recorded it logs "skipped: lake frozen" and exits 0; if the
#      gate itself fails it logs that and exits 1 without syncing.
#   3. `pitch-edge lake-sync --only fd_matches --skip-if-busy`. If another process holds the warehouse (DuckDB is
#      single-writer) the sync logs "skipped: warehouse busy" and exits 0; the next scheduled start retries. The sync
#      is checkpointed per window, so a run killed half-way (lid closed, reboot) resumes where it stopped.
# Data directory (logs, pin log, warehouse): $PITCH_EDGE_DATA_DIR, default `data` in this tree. The .env is read from
# $PITCH_EDGE_ENV_FILE (default `.env` in this tree). Log: <data dir>/logs/lake_sync_fd.log.
# Manual run: scripts/lake_sync_fd.sh
set -u
cd "$(dirname "$0")/.." || exit 1
DATA_DIR="${PITCH_EDGE_DATA_DIR:-data}"
LOG="$DATA_DIR/logs/lake_sync_fd.log"
PIN_LOG="$DATA_DIR/control/scheduled_pin.log"
mkdir -p "$DATA_DIR/logs"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(stamp) $*" >>"$LOG"; }
pe() { arch -arm64 uv run pitch-edge "$@" >>"$LOG" 2>&1; }

if ! . scripts/scheduled_guard.sh; then
  log "refusing to start: scripts/scheduled_guard.sh is missing"
  exit 0
fi
scheduled_guard || exit 0

pe control gate; gate_rc=$?
if [ "$gate_rc" -eq 10 ]; then
  log "skipped: lake frozen (control gate); no sync"
  exit 0
fi
if [ "$gate_rc" -ne 0 ]; then
  log "freeze check failed (exit $gate_rc); no sync"
  exit 1
fi

log "lake-sync fd_matches start"
pe lake-sync --only fd_matches --skip-if-busy
rc=$?
log "lake-sync fd_matches exit $rc"
exit $rc
