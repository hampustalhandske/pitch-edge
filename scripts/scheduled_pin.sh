#!/usr/bin/env bash
# Pin the scheduled jobs to one commit: scripts/scheduled_pin.sh <sha>
#
# Creates (or moves) the detached worktree <repo>/.worktrees/scheduled at <sha>, runs
# `arch -arm64 uv sync --extra dev --locked` there, and appends "<utc time> <full sha>" to the pin log
# <data dir>/control/scheduled_pin.log (<data dir> is $PITCH_EDGE_DATA_DIR, default <repo>/data). <repo> is the main
# checkout, also when this script is run from another worktree. The launchd agents run paper_loop.sh and
# lake_sync_fd.sh from that worktree, and both refuse to start unless its HEAD is the last sha in the pin log and
# it has no uncommitted changes (scripts/scheduled_guard.sh).
#
# Refuses (exit 2) when <sha> is not a commit, and (exit 3) when .worktrees/scheduled exists but is not a clean
# worktree of this repository. When the checkout or `uv sync` fails, or leaves the worktree dirty, the pin is not
# recorded (exit 1), so the scheduled jobs keep refusing. Stop the running agents before moving the pin: a loop
# that is running keeps executing the files of the worktree as they change.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMMON=$(git -C "$SCRIPT_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || {
  echo "refused: $SCRIPT_DIR is not in a git repository"
  exit 2
}
REPO=$(dirname "$COMMON")
cd "$REPO" || exit 1
SHA=$(git rev-parse --verify --quiet "${1:-}^{commit}") || { echo "refused: '${1:-}' is not a commit"; exit 2; }
WT="$REPO/.worktrees/scheduled"
DATA_DIR="${PITCH_EDGE_DATA_DIR:-$REPO/data}"
PIN_LOG="$DATA_DIR/control/scheduled_pin.log"

if [ -e "$WT" ]; then
  wt_common=$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
  wt_top=$(git -C "$WT" rev-parse --show-toplevel 2>/dev/null)
  if [ "$wt_common" != "$COMMON" ] || [ -z "$wt_top" ] || [ "$(cd "$wt_top" && pwd -P)" != "$(cd "$WT" && pwd -P)" ]; then
    echo "refused: $WT exists and is not a worktree of $REPO"
    exit 3
  fi
  if [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "refused: $WT has uncommitted changes"
    exit 3
  fi
  git -C "$WT" checkout --quiet --detach "$SHA" || { echo "failed: checkout of $SHA in $WT; pin not recorded"; exit 1; }
else
  mkdir -p "$REPO/.worktrees"
  git worktree add --quiet --detach "$WT" "$SHA" || { echo "failed: git worktree add $WT $SHA; pin not recorded"; exit 1; }
fi

(cd "$WT" && arch -arm64 uv sync --extra dev --locked) || { echo "failed: uv sync in $WT; pin not recorded"; exit 1; }
if [ -n "$(git -C "$WT" status --porcelain)" ]; then
  echo "failed: $WT has changes after uv sync; pin not recorded"
  exit 1
fi

mkdir -p "$(dirname "$PIN_LOG")"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $SHA" >>"$PIN_LOG"
echo "pinned $WT at $SHA (recorded in $PIN_LOG)"
