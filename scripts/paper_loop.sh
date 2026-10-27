#!/usr/bin/env bash
# Forward paper-trading loop (paper only: read-only market data, never places an order).
#
# Runs under launchd (scripts/launchd/local.pitch-edge.paper-loop.plist: starts at login, restarts after a
# crash or reboot). Closing the lid pauses it; on wake an overdue cycle starts within a minute. Manual run:
#   nohup scripts/paper_loop.sh >/dev/null 2>&1 &
#
# Every hour: collect Polymarket + Kalshi books, collect the-odds-api (every book in regions uk,eu, 2 credits
# per league; the collector calls each league at most once per 23.5 h, never while credits remaining < 60,
# and trims the league list so the projected monthly spend stays <= 450), then run every strategy.
# Every hour, after that: settle finished decisions against the lake's closing prices, so a decision settles
# in the first cycle after the lake has its close. A cycle that starts more than 1.5 x the interval after the
# previous one (the Mac slept) is logged as "missed N cycles since <ts>".
# Log: data/logs/paper_loop.log. Stop: kill the PID in data/paper/paper_loop.pid (the trap removes the
# pidfile; a second copy refuses to start while that pid is alive).
set -u
cd "$(dirname "$0")/.." || exit 1

INTERVAL_S="${PAPER_LOOP_INTERVAL_S:-3600}"
LOG="data/logs/paper_loop.log"
PIDFILE="data/paper/paper_loop.pid"
mkdir -p data/logs data/paper

# Single instance: refuse to start while another loop's pid is alive.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) refusing to start: loop already running (pid $(cat "$PIDFILE"))" >>"$LOG"
  exit 0  # a clean exit, so launchd's KeepAlive (restart on crash only) does not respawn a duplicate
fi
echo $$ > "$PIDFILE"
SLEEP_PID=""
cleanup() {
  [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null
  [ "$(cat "$PIDFILE" 2>/dev/null)" = "$$" ] && rm -f "$PIDFILE"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) paper loop exiting (pid $$)" >>"$LOG"
}
trap cleanup EXIT
trap 'exit 0' TERM INT

pe() { arch -arm64 uv run pitch-edge "$@" >>"$LOG" 2>&1; }
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >>"$LOG"; }

log "paper loop started (pid $$, interval ${INTERVAL_S}s)"
last_start=0
while true; do
  cycle_start=$(date +%s)
  if [ "$last_start" -gt 0 ] && [ $((cycle_start - last_start)) -gt $((INTERVAL_S * 3 / 2)) ]; then
    log "missed $(( (cycle_start - last_start + INTERVAL_S / 2) / INTERVAL_S - 1 )) cycles since $(date -u -r "$last_start" +%Y-%m-%dT%H:%M:%SZ)"
  fi
  last_start=$cycle_start
  log "cycle start"
  pe paper collect --sources polymarket,kalshi || log "collect polymarket,kalshi exited non-zero"
  pe -v paper collect --sources odds_api || log "collect odds_api exited non-zero"
  pe paper decide || log "decide exited non-zero"
  pe paper settle || log "settle exited non-zero"
  now=$(date +%s)
  log "cycle end"
  # Wall-clock schedule: wait in short steps until the next cycle is due, so after the machine wakes from sleep
  # (lid closed) an overdue cycle starts within a minute instead of after the remainder of a frozen hour-long sleep.
  next_due=$((now + INTERVAL_S))
  while [ "$(date +%s)" -lt "$next_due" ]; do
    sleep 60 &
    SLEEP_PID=$!
    wait "$SLEEP_PID"  # interruptible: a TERM is handled at once
  done
done
