#!/usr/bin/env bash
# Smoke test: build the image and run the organisers' command
# on a temporary copy of the samples with --network none and without LLM_* variables.
# The source samples are never modified.
#
# Usage: scripts/smoke.sh [--no-build] [--pytest] [--live]
#   --no-build  reuse an existing hack:latest image
#   --pytest    also run the test suite on Linux/Python 3.13 in a throwaway dev container
#   --live      also solve the samples through the model with T0 disabled. SPENDS KEY QUOTA.
#               Docker reads LLM_BASE_URL and LLM_API_KEY from .env; nothing prints them.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="hack:latest"
TEAM="${TEAM:-smoke}"
SAMPLES="${SAMPLES:-$REPO/task-dir}"
PYTHON="${PYTHON:-python}"
MAX_BUILD_S=1000
MAX_IMAGE_BYTES=$((4 * 1024 * 1024 * 1024))

BUILD=1
PYTEST=0
LIVE=0
for arg in "$@"; do
  case "$arg" in
    --no-build) BUILD=0 ;;
    --pytest) PYTEST=1 ;;
    --live) LIVE=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Git Bash rewrites /app/... and /data/... arguments into Windows paths unless told not to.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
host_path() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }

fail() { echo "SMOKE FAILED: $*" >&2; exit 1; }
step() { echo; echo "== $* =="; }

snapshot() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum); }

[ -d "$SAMPLES" ] || fail "samples directory not found: $SAMPLES"
SAMPLES_BEFORE="$(snapshot "$SAMPLES")"

if [ "$BUILD" = 1 ]; then
  step "1. docker build"
  started=$(date +%s)
  docker build -t "$IMAGE" "$(host_path "$REPO")"
  elapsed=$(( $(date +%s) - started ))
  echo "build time: ${elapsed}s (limit ${MAX_BUILD_S}s; a warm cache makes this number optimistic)"
  [ "$elapsed" -lt "$MAX_BUILD_S" ] || fail "build took ${elapsed}s"
fi
size=$(docker image inspect "$IMAGE" --format '{{.Size}}')
echo "image size: $((size / 1024 / 1024)) MiB (limit 4096 MiB)"
[ "$size" -lt "$MAX_IMAGE_BYTES" ] || fail "image is larger than 4 GB"

step "2. temporary copy of the samples without 'expected'"
TASKS="$(mktemp -d)"
trap 'rm -rf "$TASKS"' EXIT
cp -R "$SAMPLES"/. "$TASKS"/
# Dev-only: the key is removed without ever being used; the harness itself never reads it.
"$PYTHON" - "$(host_path "$TASKS")" <<'PY'
import json
import pathlib
import sys

for path in pathlib.Path(sys.argv[1]).glob("*/task.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("expected", None)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
! grep -rqs '"expected"' "$TASKS" || fail "'expected' is still present in the copy"
chmod -R a+rwX "$TASKS" 2>/dev/null || true
COPY_BEFORE="$(snapshot "$TASKS")"

step "3. organisers' command, --network none, no LLM_* variables"
set +e
env -u LLM_BASE_URL -u LLM_API_KEY -u BUDGET \
  timeout -k 1000s 1h docker run --rm --name "run-${TEAM}" \
  --network none \
  -v "$(host_path "$TASKS"):/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 8 --memory 16g --pids-limit 512 \
  --workdir /app --entrypoint python \
  "$IMAGE" /app/run_agent.py --tasks-dir /data/tasks 2>&1 | tee "$TASKS.log"
code=${PIPESTATUS[0]}
set -e
trap 'rm -rf "$TASKS" "$TASKS.log"' EXIT
[ "$code" = 0 ] || fail "container exited with code $code"
! grep -q "Traceback (most recent call last)" "$TASKS.log" || fail "traceback in the container output"
grep -q "loaded 8 tasks" "$TASKS.log" || echo "note: the run did not report 8 loaded tasks"

step "4. checks outside the container"
[ "$(snapshot "$SAMPLES")" = "$SAMPLES_BEFORE" ] || fail "samples directory was modified"
answers=$(find "$TASKS" -name answer.txt | wc -l)
echo "answer.txt files in the copy: $answers"
if [ "$answers" = 0 ]; then
  [ "$(snapshot "$TASKS")" = "$COPY_BEFORE" ] || fail "the copy changed although nothing was solved"
fi
COPY_AFTER_NO_ANSWERS="$(cd "$TASKS" && find . -type f ! -name answer.txt ! -name mylib.py -print0 | sort -z | xargs -0 sha256sum)"
COPY_BEFORE_NO_ANSWERS="$(printf '%s\n' "$COPY_BEFORE" | grep -v -e '/answer\.txt$' -e '/mylib\.py$')"
[ "$COPY_AFTER_NO_ANSWERS" = "$COPY_BEFORE_NO_ANSWERS" ] || fail "files other than answer.txt and mylib.py changed"

! find "$TASKS" -name __pycache__ | grep -q . || fail "__pycache__ appeared in the tasks directory"
# The judge is the only code that reads `expected`; it takes it from the pristine samples.
"$PYTHON" "$(host_path "$REPO/scripts/check.py")" "$(host_path "$TASKS")" \
  --reference "$(host_path "$SAMPLES")" || fail "phase 0 did not solve every sample"

if [ "$LIVE" = 1 ]; then
  step "live. model path: T0 disabled, proxy from .env (spends quota)"
  [ -f "$REPO/.env" ] || fail ".env not found"
  LIVE_TASKS="$(mktemp -d)"
  trap 'rm -rf "$TASKS" "$TASKS.log" "$LIVE_TASKS" "$LIVE_TASKS.log"' EXIT
  cp -R "$SAMPLES"/. "$LIVE_TASKS"/
  chmod -R a+rwX "$LIVE_TASKS" 2>/dev/null || true
  set +e
  timeout -k 1000s 1h docker run --rm --name "run-${TEAM}-live"     --env-file "$(host_path "$REPO/.env")" -e DISABLE_T0=1 -e T1_MODEL     -v "$(host_path "$LIVE_TASKS"):/data/tasks:rw"     --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp     --cap-drop ALL --security-opt no-new-privileges     --cpus 8 --memory 16g --pids-limit 512     --workdir /app --entrypoint python     "$IMAGE" /app/run_agent.py --tasks-dir /data/tasks 2>&1 | tee "$LIVE_TASKS.log"
  code=${PIPESTATUS[0]}
  set -e
  [ "$code" = 0 ] || fail "live container exited with code $code"
  grep -E "phase 1 (done|cost)|model tier" "$LIVE_TASKS.log" || echo "note: phase 1 did not run"
  "$PYTHON" "$(host_path "$REPO/scripts/check.py")" "$(host_path "$LIVE_TASKS")"     --reference "$(host_path "$SAMPLES")" || echo "LIVE: not every sample was solved by the model"
  [ "$(snapshot "$SAMPLES")" = "$SAMPLES_BEFORE" ] || fail "samples directory was modified"
fi

if [ "$PYTEST" = 1 ]; then
  step "5. pytest and ruff on Linux / Python 3.13 (dev container, needs network for uv sync)"
  docker run --rm -v "$(host_path "$REPO"):/repo:ro" --tmpfs /tmp:rw,exec,size=2g \
    -e HOME=/tmp -e UV_PROJECT_ENVIRONMENT=/tmp/venv -e UV_CACHE_DIR=/tmp/uv-cache \
    -e UV_LINK_MODE=copy -e RUFF_CACHE_DIR=/tmp/ruff-cache --workdir /repo \
    python:3.13.5-slim-bookworm sh -c '
      pip install --no-cache-dir --quiet --root-user-action=ignore uv==0.9.5 &&
      uv sync --locked --quiet &&
      uv run --no-sync ruff check . &&
      uv run --no-sync ruff format --check . &&
      uv run --no-sync pytest -q -p no:cacheprovider'
fi

echo
echo "SMOKE OK"
