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
# Once a day: settle finished decisions against the lake's closing prices.
# Log: data/logs/paper_loop.log. Stop: kill the PID in data/paper/paper_loop.pid (the trap removes the
# pidfile; a second copy refuses to start while that pid is alive).
set -u
cd "$(dirname "$0")/.." || exit 1

INTERVAL_S="${PAPER_LOOP_INTERVAL_S:-3600}"
SETTLE_EVERY_S="${PAPER_SETTLE_EVERY_S:-86400}"
LOG="data/logs/paper_loop.log"
STAMP="data/paper/.last_settle"
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

log "paper loop started (pid $$, interval ${INTERVAL_S}s, settle every ${SETTLE_EVERY_S}s)"
while true; do
  log "cycle start"
  pe paper collect --sources polymarket,kalshi || log "collect polymarket,kalshi exited non-zero"
  pe -v paper collect --sources odds_api || log "collect odds_api exited non-zero"
  pe paper decide || log "decide exited non-zero"
  now=$(date +%s)
  last=$(cat "$STAMP" 2>/dev/null || echo 0)
  if [ $((now - last)) -ge "$SETTLE_EVERY_S" ]; then
    if pe paper settle; then echo "$now" > "$STAMP"; else log "settle exited non-zero"; fi
  fi
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
