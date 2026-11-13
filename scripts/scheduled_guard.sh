# Start-of-run guard for the scheduled scripts (paper_loop.sh, lake_sync_fd.sh); sourced, not run.
#
# Scheduled jobs run only committed code at the sha the owner pinned with scripts/scheduled_pin.sh. The caller has
# already `cd`-ed to its own tree and defines `log` (one line to its log) and `PIN_LOG` (the pin log, whose last
# line is "<utc time> <full sha>"). `scheduled_guard` returns 0 when the tree may run, and otherwise logs why not
# and returns 1:
#   - `git status --porcelain` fails or prints anything (uncommitted or untracked changes);
#   - HEAD cannot be read;
#   - the pin log has no sha, or HEAD is not its last sha.

scheduled_guard() {
  local changes head pin
  if ! changes=$(git status --porcelain 2>/dev/null); then
    log "refusing to start: git status failed in $PWD"
    return 1
  fi
  if [ -n "$changes" ]; then
    log "refusing to start: $PWD has uncommitted changes ($(printf '%s\n' "$changes" | wc -l | tr -d ' ') paths)"
    return 1
  fi
  if ! head=$(git rev-parse --verify --quiet HEAD); then
    log "refusing to start: cannot read HEAD in $PWD"
    return 1
  fi
  pin=$(tail -n 1 "$PIN_LOG" 2>/dev/null | awk '{print $2}')
  if [ -z "$pin" ]; then
    log "refusing to start: no pinned sha in $PIN_LOG (run scripts/scheduled_pin.sh <sha>)"
    return 1
  fi
  if [ "$head" != "$pin" ]; then
    log "refusing to start: HEAD $head is not the pinned sha $pin ($PIN_LOG)"
    return 1
  fi
  return 0
}
