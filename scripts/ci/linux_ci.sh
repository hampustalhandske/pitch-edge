#!/bin/bash
# Run the steps of .github/workflows/ci.yml job `test` in a Linux container on a clean checkout of one sha.
# Ends with a verdict: `=== GATE PASS` when the checkout and every step exit 0, else `=== GATE FAIL (<failed
# steps>)` and exit 1. Log: data/logs/ci-linux-<sha7>.log.
# Usage: scripts/ci/linux_ci.sh <sha>   (from the repo root; starts Docker Desktop and quits it on exit)
# KEEP_DOCKER=1 leaves Docker running on exit, for when another project's Linux CI runs right after in the same
# session; the host then quits it after the last run with: scripts/ci/linux_ci.sh --quit-docker
# The repo's .git is mounted read-only and cloned inside the container, so the container never writes to the repo.
set -uo pipefail
quit_docker() {
  osascript -e 'quit app "Docker Desktop"' || echo "DOCKER QUIT FAILED: tell the master at once"
  for _ in $(seq 1 60); do pgrep -fl com.docker >/dev/null || break; sleep 2; done
  if pgrep -fl com.docker; then echo "DOCKER STILL RUNNING: tell the master at once"; else echo "docker quit: ok"; fi
}
if [ "${1:-}" = "--quit-docker" ]; then quit_docker; exit 0; fi
SHA=$(git rev-parse --verify --quiet "${1:-}^{commit}") || { echo "refused: '${1:-}' is not a commit; give one sha"; exit 2; }
mkdir -p data/logs; LOG="data/logs/ci-linux-${SHA:0:7}.log"
IMAGE="ghcr.io/astral-sh/uv:python3.11-bookworm"
# No persistent uv cache volume: a full cache is about 6.4 GB, over the owner's 5 GB limit.

if [ "${KEEP_DOCKER:-0}" = "1" ]; then
  trap 'echo "KEEP_DOCKER=1: Docker left running for the next run; quit it with scripts/ci/linux_ci.sh --quit-docker"' EXIT
else
  trap quit_docker EXIT
fi
open -a "Docker Desktop"
for _ in $(seq 1 90); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null 2>&1 || { echo "docker daemon did not start"; exit 1; }
docker pull -q "$IMAGE" >/dev/null
{
  echo "sha $SHA"
  echo "image $IMAGE $(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"
  ( while sleep 3; do docker stats --no-stream --format '{{.MemUsage}}' pe-ci 2>/dev/null; done ) > "$LOG.mem" &
  STATS=$!
  START=$(date +%s)
  docker run --rm --name pe-ci --platform linux/arm64 -v "$PWD/.git:/repo.git:ro" -e CI=true -e SHA="$SHA" "$IMAGE" bash -c '
    FAILED=""
    step() { echo "=== STEP $1"; n=$1; shift; s=$(date +%s); "$@"; rc=$?; echo "=== RESULT rc=$rc wall=$(( $(date +%s) - s ))s"; [ $rc -eq 0 ] || FAILED="${FAILED:+$FAILED, }$n"; return 0; }
    uname -m; cat /etc/os-release | head -1
    git config --global --add safe.directory "*"
    git clone -q --no-checkout /repo.git /work && cd /work && git checkout -q --detach "$SHA" && git log --oneline -1 \
      || { echo "=== GATE FAIL (checkout)"; exit 1; }
    step "uv python install 3.11" uv python install 3.11
    step "uv sync --extra dev" uv sync --extra dev
    step "uv run ruff check ." uv run ruff check .
    step "uv run mypy src" uv run mypy src
    step "uv run pytest" env PITCH_EDGE_EMBEDDING_MODEL=tfidf-only uv run pytest -q -rs --cov=pitch_edge --cov-report=term-missing
    echo "=== git status after run"; git status --short
    if [ -z "$FAILED" ]; then echo "=== GATE PASS"; else echo "=== GATE FAIL ($FAILED)"; exit 1; fi
  '
  RUN_RC=$?
  if [ "$RUN_RC" -ne 0 ] && ! grep -q "^=== GATE FAIL" "$LOG"; then echo "=== GATE FAIL (docker run rc=$RUN_RC)"; fi
  echo "=== wall total $(( $(date +%s) - START ))s"
  kill $STATS 2>/dev/null
  echo "=== container peak memory $(awk '{print $1}' "$LOG.mem" | sort -h | tail -1)"
} > "$LOG" 2>&1
echo "log: $LOG"
grep -E "^=== |passed|failed" "$LOG"
[ "$RUN_RC" -eq 0 ] || exit 1
