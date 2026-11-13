# openJiuwen Challenge 2026

A token-budget-aware agent harness built with the openJiuwen SDK. It solves tasks with deterministic solvers first, then uses model calls when needed while tracking a shared token budget.

## Build and run

```bash
docker build -t hack:latest .
docker run --rm \
  -e LLM_BASE_URL -e LLM_API_KEY \
  -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 4 --memory 16g --pids-limit 512 \
  --workdir /app --entrypoint python \
  hack:latest /app/run_agent.py --tasks-dir /data/tasks
```

`TASKS` is the local directory containing the task directories. Each solved task receives an `answer.txt`. Without `LLM_BASE_URL` and `LLM_API_KEY`, only deterministic solvers run.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BUDGET` | `100000` | Shared budget: prompt tokens / 10 + completion tokens |
| `LLM_CONCURRENCY` | `3` | Initial number of parallel model sessions |
| `T1_MODEL`, `T2_MODEL` | `qwen3.8-27b-noreason`, `qwen3.8-27b` | Models used by T1 and T2 |
| `ANSWER_WHEN_UNSURE` | `1` | Allow a free, well-defined answer for a task suspected to be infeasible |
| `TRAJECTORY_DIR` | `/tmp/run/logs` | Directory for `trajectory.jsonl` |
| `OUT` / `--out` | unset | Optional aggregate `answers.json` |

## Development

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The smoke test requires an external directory of sample tasks: `SAMPLES=/path/to/samples bash scripts/smoke.sh --pytest`. Its live mode uses model quota.

## Structure

- `run_agent.py` — entry point.
- `src/` — task loading, scheduling, solvers, model gateway and result handling.
- `tests/` — automated tests.
- `scripts/` — development checks, task generator and proxy probe.
