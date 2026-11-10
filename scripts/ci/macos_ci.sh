#!/bin/bash
# Run the steps of .github/workflows/ci.yml job `test` on macOS, on a clean in-repo worktree of one sha, with no
# project env (env -i). Reports pass/fail, wall time and peak RSS per step, then a verdict: `=== GATE PASS` when
# every step exits 0, else `=== GATE FAIL (<failed steps>)` and exit 1. Log: data/logs/ci-macos-<sha7>.log.
# Usage (repo root): scripts/ci/macos_ci.sh <sha>
set -uo pipefail
SHA=$(git rev-parse --verify --quiet "${1:-}^{commit}") || { echo "refused: '${1:-}' is not a commit; give one sha"; exit 2; }
WT=".worktrees/ci-${SHA:0:7}"; mkdir -p data/logs; LOG="$PWD/data/logs/ci-macos-${SHA:0:7}.log"
UVDIR=$(dirname "$(which uv)")
git worktree add --detach "$WT" "$SHA" >/dev/null 2>&1 || { echo "worktree add failed"; exit 1; }
cd "$WT"
FAILED=""
step() {
  local name=$1; shift
  echo "=== STEP $name"
  local start=$(date +%s)
  /usr/bin/time -l env -i HOME="$HOME" PATH="$UVDIR:/usr/bin:/bin:/usr/sbin:/sbin" TERM=dumb CI=true LANG=en_US.UTF-8 \
    arch -arm64 "$@" > step.out 2> step.time; local rc=$?
  tail -5 step.out
  local peak=$(awk '/maximum resident set size/{printf "%.0f", $1/1048576}' step.time)
  echo "=== RESULT rc=$rc wall=$(( $(date +%s) - start ))s peak_rss=${peak}MB"
  rm -f step.out step.time
  [ "$rc" -eq 0 ] || FAILED="${FAILED:+$FAILED, }$name"
}
{
  echo "sha $SHA"
  step "uv python install 3.11" uv python install 3.11
  step "uv sync --extra dev" uv sync --extra dev
  step "uv run ruff check ." uv run ruff check .
  step "uv run mypy src" uv run mypy src
  step "uv run pytest" env PITCH_EDGE_EMBEDDING_MODEL=tfidf-only uv run pytest -q -rs -p no:cacheprovider --cov=pitch_edge --cov-report=term-missing
  echo "=== git status after run"; git status --short
  if [ -z "$FAILED" ]; then echo "=== GATE PASS"; else echo "=== GATE FAIL ($FAILED)"; fi
} > "$LOG" 2>&1
cd - >/dev/null; git worktree remove --force "$WT"
echo "log: $LOG"; grep -E "^=== (STEP|RESULT|GATE)|passed|failed" "$LOG"
[ -z "$FAILED" ] || exit 1
