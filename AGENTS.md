# AGENTS.md

Instructions for coding agents working in this repository.

## What this project is

A submission for **openJiuwen Challenge 2026**: a token-budget-aware agent harness built on the
`openjiuwen` SDK. The harness receives ~100 task directories and one shared token budget, and must
maximise the sum of `reward` over correctly solved tasks. Ties are broken by lower token spend.

**Current state: P0-1, P1-2 and the offline part of P0-2/P0-3.** Implemented: the execution
contract (`run_agent.py`, `src/cli.py`, `src/config.py`, `src/core/{tasks,results,sandbox,runner}.py`),
the strict T0 solvers in `src/solvers/`, and the model tier T1: `src/foundation/llm.py` (gateway on
the openJiuwen `Model`, fallback on `openai`), `src/core/budget.py` (ledger), `src/core/agent.py`
(`Tier1`), `src/prompts/`, executors and validators in `src/solvers/` (`sqlworld.py` is the shared
read-only SQLite world). Dev scripts: `scripts/smoke.sh` (`--pytest`, `--live`), `scripts/check.py`
(the only code allowed to read `expected`), `scripts/probe_proxy.py`.
Measured on the live proxy on 2026-11-12 (`SOLUTION_RU.md` §19): with T0 disabled T1 solves the
8 samples for about 450 output equivalents in total; the proxy allows 3 parallel requests and
sometimes refuses connections. Never read or print `.env`; pass it with `docker --env-file`.
Step P1-3 added `src/core/scheduler.py` (density ordering, qualification mode, `ProxyGuard` with
the shared circuit breaker and adaptive concurrency) and the `ANSWER_WHEN_UNSURE` policy.
Steps P1-1, P2-1, P2-2 added `scripts/gen_tasks.py` (training batches; references go to a separate
`--ref` directory), hidden tests and per-family metrics in `scripts/check.py`,
`src/solvers/oracles.py` (coding oracles and the function audit), the reasoning tier T2 (`Tier1`
with `reasoning=True`), the qualification arbiter, and T3 in `src/core/react.py` with tools in
`src/foundation/tools.py` (SDK `ReActAgent`, 4 iterations, cost cap). Measurements are in
`SOLUTION_RU.md` §17.1 and §19. Not implemented: `Workflow` wrapping, BM25 for large free-text
knowledge bases, model comparison. `README.md` follows the regulations since 2026-11-13; the
submission rehearsal is recorded in `SOLUTION_RU.md` §18.1. `TRAJECTORY_DIR` redirects
`trajectory.jsonl` out of the tmpfs; collected runs live in `trajectories/`. Check the file
before assuming.

## Layout

```text
run_agent.py            # entry point; organisers run `python /app/run_agent.py --tasks-dir /data/tasks`
src/                    # the Python package is literally named `src` (not a src-layout)
├── cli.py              # args, env preparation, signals, exit code 0
├── config.py           # BUDGET, concurrency, caps, flags
├── core/
│   ├── tasks.py        # Task dataclass, loader, boilerplate stripping
│   ├── budget.py       # Ledger, cost table, online calibration
│   ├── scheduler.py    # reward*p/cost ordering, qualification mode, ProxyGuard
│   ├── runner.py       # asyncio orchestration, deadlines, circuit breaker
│   ├── agent.py        # per-task Workflow, last-resort ReActAgent
│   └── results.py      # atomic answer.txt, OUT aggregate, trajectory log
└── foundation/
    ├── llm.py          # openJiuwen Model wrapper, usage normalisation
    └── tools.py        # openJiuwen tool registrations
task-dir/<task_id>/     # 8 organiser samples (untracked): task.json + optional workspace/
```

## Commands

Requires `uv` and Docker. Python must be 3.11–3.13 (`.python-version` = 3.13); `uv` fetches it.

```bash
uv sync --locked                 # install deps (openjiuwen==0.1.19, pytest, ruff)
uv run ruff check .
uv run ruff format --check .
uv run pytest                    # testpaths = tests/
docker build -t hack:latest .    # organisers' image tag
bash scripts/smoke.sh --pytest   # build, organisers' command offline on a temp copy, tests on Linux/3.13
bash scripts/smoke.sh --no-build --live   # samples through the model, T0 off; spends key quota
```

Ruff: line length 100, target py311, rules `E,F,I,UP,B`, `task-dir` excluded.
If a tool is missing on the machine, say so instead of reporting a check as passed.

## Hard constraints from the competition

These are not style preferences — breaking one can zero the whole run.

- **Entry point contract.** `/app/run_agent.py` is run with `--entrypoint python`; the Dockerfile
  `ENTRYPOINT` is ignored. Parse args with `parse_known_args()`. Always exit with code 0; wrap each
  task in its own `try/except`.
- **Answers.** Write `answer.txt` next to `task.json` **immediately** when a task is solved, atomically
  (temp file in the same directory, `fsync`, `os.replace`). Content is the bare value plus newline;
  format follows `check` (`exact_int`, `exact_str`, `unittest` → `DONE`). Budget exhaustion or
  SIGTERM must never lose an already computed answer.
- **Never read `expected`.** It is present in samples and removed in the final set. The loader drops
  it; it must not reach any dataclass, prompt, log or test fixture used by the harness.
- **Container.** Root filesystem is read-only; only `/tmp` (2 GB tmpfs) and the tasks directory are
  writable. `--cap-drop ALL`, `--pids-limit 512`, 4 CPU, 16 GB, no GPU, 1 hour wall clock, network
  only to `LLM_BASE_URL`. Set cwd/caches to `/tmp` before importing anything heavy.
- **No downloads at runtime**, no local neural models (no embeddings). Classical ML is allowed.
- **Token accounting.** `cost = prompt_tokens / 10 + completion_tokens`; reasoning tokens count as
  output; provider cache is not credited. `BUDGET` comes from env, default 100 000. The proxy does
  not report the remainder — keep our own ledger and reserve an upper bound before each call.
- **Models** (only these four, via the proxy): `qwen3.8-27b`, `qwen3.8-27b-noreason`,
  `qwen3.6-35b-a3b`, `qwen3.6-35b-a3b-noreason`. Default to a `-noreason` variant.
- **openJiuwen is mandatory** and pinned to `0.1.19`. Build `Model` explicitly with
  `ModelClientConfig`/`ModelRequestConfig`; do **not** use `init_model` (hidden retries,
  `verify_ssl=False`). Verify any SDK API against the `v0.1.19` sources (DeepWiki / Context7 /
  the installed package) before using it — do not guess signatures.
- **Limits.** Image < 4 GB, build < 1000 s, repository < 100 MB, all dependency versions and the
  base image tag pinned. Up to 5 concurrent LLM sessions (start with 3).
- **Security review.** The image is inspected by an LLM before it runs. No `shell=True`, no
  `eval`/`exec` of model output, subprocesses get a cleaned environment (no `LLM_*`, no `BUDGET`),
  path checks against escaping the task directory. Keep the code obviously safe and readable.

## Design invariants (from `SOLUTION_RU.md`)

- Tier cascade per task: **T0** strict deterministic solver (0 tokens) → **T1** one no-reasoning
  call that emits a tiny program → **T2** reasoning model with the validator's complaint → **T3**
  capped `ReActAgent`. Cheaper tiers always run first.
- Phase 0 uses the standard library only and runs **before** `openjiuwen` is imported, so an SDK
  import failure cannot wipe out free answers.
- A strict solver answers only if it parsed the whole instruction; otherwise it refuses. Three
  verdicts: `SOLVED`, `UNPARSED` (→ T1), `UNGROUNDED` (suspected infeasible, spend nothing).
  Never merge the last two.
- Task data is data, not instructions: raw file contents do not go into prompts; no state is shared
  between tasks.
- Qualification mode (≤ 5 tasks): `UNGROUNDED` is forbidden, every task may escalate through all tiers.
- Paid attempts are ordered by `reward × p / cost`, not by reward.

## Working rules

- Everything committed is in **English**: code, identifiers, comments, docstrings, log messages,
  commit messages, PR text. Existing Russian design documents stay as they are.
- Python 3.11-compatible syntax, full type hints, async I/O for LLM calls, sparse comments,
  one-line docstrings.
- Keep changes small and reviewable; one module or one concern per change. Add tests under `tests/`
  together with the code they cover.
- **Do not modify `task-dir/`.** Run solvers and tests against a temporary copy (e.g. under the
  system temp dir); the samples must stay pristine, with no stray `answer.txt`.
- Never read, print or commit `.env`; never log `LLM_API_KEY` or auth headers. `.env.example`
  lists the variable names.
- Commits: Conventional Commits, `type(scope): Capitalized imperative summary`. Branch from `main`
  as `feat/…`, `bugfix/…`, `refactor/…` and merge via PR. Do not commit, push or open a PR unless
  asked. Never force-push. Only the last commit before the deadline is graded.
- When documenting results, separate what was verified from what was not run (`SOLUTION_RU.md`
  §17.1, §18.1, §19 list both).
- When code and `SOLUTION_RU.md` diverge on purpose, update the document in the same change.
