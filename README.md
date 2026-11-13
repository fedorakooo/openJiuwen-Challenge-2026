# openJiuwen Challenge 2026 — token-budget-aware agent harness

A harness on the `openjiuwen` SDK (pinned to `0.1.19`) that receives a directory of tasks and one
shared token budget and maximises the sum of `reward` over correctly solved tasks. It spends tokens
only where a free deterministic solver cannot answer, and never spends them on tasks it can show to
be infeasible.

The full design, measurements and the list of unverified assumptions are in
[SOLUTION_RU.md](SOLUTION_RU.md) (Russian). Where this README and that document disagree, the
document and the code win.

## Build and run

```bash
docker build -t hack:latest .
```

The organisers' command works as is; `ENTRYPOINT` is not relied on:

```bash
timeout -k 1000s 1h docker run --rm --name "run-${TEAM}" \
  --network hack-net -e LLM_BASE_URL -e LLM_API_KEY \
  -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 8 --memory 16g --pids-limit 512 \
  --workdir /app --entrypoint python \
  "hack:latest" /app/run_agent.py --tasks-dir /data/tasks
```

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY` | — | Proxy address and key. Without them only the zero-token solvers run |
| `BUDGET` | `100000` | Budget in output equivalents (`prompt_tokens / 10 + completion_tokens`) |
| `ANSWER_WHEN_UNSURE` | `1` | Write a free, well-defined answer for a task suspected to be infeasible (see below) |
| `LLM_CONCURRENCY` | `3` | Starting number of parallel model sessions, 1–5 |
| `T1_MODEL`, `T2_MODEL` | `qwen3.8-27b-noreason`, `qwen3.8-27b` | Only the four allowed model names are accepted |
| `DISABLE_T0`, `DISABLE_T2`, `DISABLE_T3` | `0` | Switch a tier off (development and measurements) |
| `RUN_DEADLINE_S`, `PHASE0_CAP_S` | `3000`, `480` | Own deadline (50 min) and the time cap of the free phase |
| `OUT` / `--out` | — | Optional aggregate `answers.json`; `answer.txt` files are written regardless |
| `TRAJECTORY_DIR` | — | Absolute path of a writable directory for `trajectory.jsonl`; otherwise it stays in `/tmp/run/logs` |

The tasks directory is taken from `--tasks-dir`, then `--tasks`, then a positional argument, then
`TASKS`, then `/data/tasks`. Unknown arguments are ignored. The process always exits with code 0.

Dependencies are pinned in `uv.lock`; the base image is `python:3.13.5-slim-bookworm`, `uv` is
`0.9.5`. Nothing is downloaded at run time. The key is passed only at run time: `.env` is ignored by
git and excluded from the build context.

### Development

```bash
uv sync --locked
uv run ruff check . && uv run ruff format --check .
uv run pytest
bash scripts/smoke.sh --pytest          # build, the organisers' command offline, tests on Linux/3.13
bash scripts/smoke.sh --no-build --live # samples through the model with T0 off; spends key quota
python scripts/gen_tasks.py --out /tmp/b/tasks --ref /tmp/b/ref --count 100 --seed 1
python scripts/check.py /tmp/b/run --reference /tmp/b/ref --by-family
```

`scripts/check.py` is the only code that reads `expected`. The harness loader drops that field
before a task object is built.

## How it works

Each task goes through a cascade; a cheaper tier always runs first.

| Tier | What it does | Cost |
|---|---|---|
| **T0** | Strict deterministic solver per task type. Answers only if it parsed the whole instruction | 0 tokens |
| **T1** | One call to a no-reasoning model that emits a tiny program: integer statements for `math`, one `SELECT` for `data_analysis` and `search`, changed lines for `coding`. The program is executed locally and validated | about 45–70 per task on the samples |
| **T2** | Reasoning model with the validator's complaint, one attempt, non-coding tasks only | 100–950 per task measured |
| **T3** | SDK `ReActAgent` with four tools, at most 4 iterations and a per-task cost cap `min(3000, 3% of BUDGET)`; only for unsolved tasks with a reward at or above the median | 280–730 per task measured |

The model never computes the answer itself: arithmetic is evaluated by an AST walker over a small
grammar, SQL runs in an in-memory SQLite database built from the task files, code patches are
checked by running the tests. Raw file contents do not go into prompts — the model sees the
question, the schema and value dictionaries.

openJiuwen is used for every model call (`Model` built explicitly from `ModelClientConfig` and
`ModelRequestConfig`; `init_model` is not used because it adds retries and disables TLS
verification) and for the agent tier (`ReActAgent`, tools registered with the SDK `tool`
decorator). The pipeline is not wrapped in a `Workflow`.

## How the token cost is estimated

The proxy does not report the remaining budget, so the harness keeps its own ledger
(`src/core/budget.py`).

- **Before a call** an upper bound is reserved: `ceil(prompt_chars / chars_per_token / 10) +
  max_tokens`. A call that does not fit into `BUDGET` minus a 5% margin is not made.
- **After a call** the reservation is replaced by the real cost from `usage`
  (`prompt_tokens / 10 + completion_tokens`; reasoning tokens are already inside
  `completion_tokens` on this proxy). The characters-per-token ratio starts at 3.0 and is corrected
  online from every response (3.15–3.31 on our prompts).
- **If `usage` is missing or the call fails after being sent**, the whole reservation stays spent.
  A refused TCP connection was never sent, so its reservation is released.
- **Ordering.** Paid attempts are ordered by `reward × p(success) / expected cost`, not by reward.
  Priors per task type and tier are in the cost table; expected costs are updated from observed
  calls.
- **Output is kept short by construction**: `max_tokens` is 96–200 for T1, a `stop` on the newline,
  few-shot prompts with a one-line output format.
- **Proxy failures**: 429 is retried twice, connection refusals three times (unbilled), server
  errors with pauses 5/15/45 s and then a shared circuit breaker; 401/403 stops the model tiers.
  If the proxy keeps failing while the ledger is above 90% of the budget, the budget is treated as
  exhausted.

## How infeasible tasks are found

Detection is deterministic and free; it happens in T0 before any token is spent
(`src/solvers/`). A solver returns one of three verdicts that are never merged:

| Verdict | Meaning | What happens |
|---|---|---|
| `SOLVED` | The whole instruction was parsed and answered | `answer.txt` is written at once |
| `UNPARSED` | The wording was not recognised | The task goes to T1 |
| `UNGROUNDED` | The instruction was understood, but it refers to something the data does not contain | No tokens are spent |

Signals of `UNGROUNDED`: a product, column or month that is absent from the table; an entity that
is not in the knowledge base, is ambiguous, or has a dangling or missing link; a quantity the word
problem never gives; tests whose assertions contradict each other. In the model tier the same
check is repeated on the generated program: SQL literals must exist in the data, numbers in a math
program must come from the question, and the model may answer `X` ("cannot be answered exactly").
`X` is not accepted at once when the question names things that do exist in the data — the model
is asked a second time.

Not treated as infeasibility: an unrecognised wording, a model mistake, a red test, a failed proxy
call, a row count in the instruction that differs from the file.

**`ANSWER_WHEN_UNSURE` (on by default).** A wrong answer and a missing answer both score zero, so
detection exists to stop spending, not to hide a free answer. When a well-defined value exists (a
sum over an empty filter is `0`), it is written; when there is nothing to compute, no file is
created. If answering an infeasible task turns out to be penalised, set `ANSWER_WHEN_UNSURE=0`
and the same signals become skips.

**Qualification mode (5 tasks or fewer).** Every task is assumed feasible: `UNGROUNDED` escalates
instead of skipping, T0 answers are verified by T1, and a T0 answer is replaced only if T1 and T2
independently agree on a different one.

## Trajectories

Every solver verdict and every model call is appended to `trajectory.jsonl`: task, tier, verdict,
whether the answer was published, the evidence (the program, SQL or patch that produced it), and
for model calls the model, reserved bound, `prompt_tokens`, `completion_tokens`, cost, latency and
`finish_reason`. Prompts, task file contents and keys are not logged.

| File | Run | Result |
|---|---|---|
| [trajectories/samples_offline.jsonl](trajectories/samples_offline.jsonl) | 8 organiser samples, `--network none`, organisers' flags | 8/8 solved by T0 for 0 tokens in 7 s |
| [trajectories/generated_200_seed1_offline.jsonl](trajectories/generated_200_seed1_offline.jsonl) | 200 generated tasks (`gen_tasks.py --seed 1`), `--network none`, 4 CPUs | T0: 89 of 166 feasible solved, 89 deferred to the model, 132 s |
| [trajectories/generated_100_seed4_live.jsonl](trajectories/generated_100_seed4_live.jsonl) | 100 fresh generated tasks (`--seed 4`, never used for tuning), live proxy, `BUDGET=3500`, organisers' flags with 4 CPUs | 83/83 feasible solved, reward 259/259, no wrong answers; T0 solved 44, T1 solved 39 for 2620 output equivalents (48 calls, about 64 per task); 111 s wall clock |

In the live run 4 of 17 infeasible tasks received an answer (the free `0` for a product that
has no rows, `ANSWER_WHEN_UNSURE`); the other 13 were skipped. T2 and T3 were not needed.

Trajectories of the earlier live runs were not kept: they were written to the container's
tmpfs before `TRAJECTORY_DIR` existed. Their aggregate results:

| Live run (model `qwen3.8-27b-noreason`, concurrency 3) | Result |
|---|---|
| 8 samples with T0 disabled | 8/8 in three of five runs, about 450 output equivalents in total |
| 100 generated tasks (seed 2), after tuning on this batch | 84/84 feasible, reward 252/252, 1938 output equivalents — optimistic, the batch was used for tuning |
| 80 generated tasks (seed 3), `BUDGET=1650` | 47/65: the budget ran out, 18 tasks never got a model call; no model mistakes among the attempted ones |

## Non-obvious details

- **The free phase runs before the SDK is imported.** Phase 0 uses the standard library only, so
  an `openjiuwen` import failure cannot cost the free answers. The SDK is imported in a background
  thread meanwhile.
- **The working directory moves to `/tmp/run` before any heavy import.** The SDK creates `./logs`
  on import and fails in a read-only directory; caches and temp files also go to `/tmp`.
- **Degraded mode.** If the SDK cannot be imported, calls go through the `openai` package (already
  an SDK dependency) with the same accounting. Without `LLM_*` variables only T0 runs.
- **Answers are written immediately and atomically** (temp file in the same directory, `fsync`,
  `os.replace`). SIGTERM or an exhausted budget cannot lose a computed answer. The harness stops
  on its own after 50 minutes.
- **`answer.txt` is duplicated** into `workspace/` when the task has one, in case the judge looks
  there. For `coding` the fixed `mylib.py` is written in place and the answer is `DONE`; test
  files are never changed.
- **Answer format follows `check`**: `exact_int` gives `158`, not `158.0`; `exact_str` keeps the
  value as it is in the source.
- **Coding repair without the model.** T0 searches single-token mutations of `mylib.py` until the
  tests pass. Functions with a known reference (`src/solvers/oracles.py`, 13 of them) are also
  compared with it on random inputs, which catches bugs the visible tests do not cover.
- **One event loop for all model tiers.** The SDK HTTP client is bound to the first loop.
- **SDK logging is silenced**: by default it prints every prompt to stdout.

## Security model

- No `shell=True`, no `eval`/`exec` of model output anywhere in the image.
- Model output is data: a math program is parsed into an AST and walked by our own evaluator; SQL
  runs in SQLite under an authorizer that allows a single read-only `SELECT`; a code patch only
  replaces lines of `mylib.py` in a copy and is executed solely by running the tests.
- Subprocesses (test runs) get a cleaned environment without `LLM_*` and `BUDGET`, a copy of the
  workspace under `/tmp`, their own process group, a hard timeout and `setrlimit` on CPU, memory,
  file size and process count.
- Paths are resolved and checked against escaping the task directory.
- Task files are data, not instructions: their raw contents do not reach prompts, and no state is
  shared between tasks.
- The only network destination is `LLM_BASE_URL`. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; no
  local neural models.
- Keys and auth headers are never logged; the repository contains no keys.

## Verified and not verified

Verified by running (13 November, Docker Desktop on Windows):

| Check | Result |
|---|---|
| Build without cache from an export of the files to be committed | 180 s (limit 1000 s); base image layers were already local |
| `docker images` size | 1.36 GB (limit 4 GB) |
| Repository size without `.git` | 1.5 MB (limit 100 MB) |
| Organisers' command, `--network none`, read-only root, `--cap-drop ALL` | exit 0, 8/8 samples |
| 200 generated tasks offline with 4 CPUs | exit 0, 132 s, no traceback |
| `docker kill --signal TERM` in the middle of a run | exit 0 within 4 s, the answers already written are intact |
| Tests | 300 on Linux/Python 3.13 in a container |
| Live run of 100 fresh tasks under the organisers' flags, `BUDGET=3500` | exit 0, 111 s, 83/83 feasible, 2620 output equivalents by our ledger |

Not verified:

- A clone of the final commit: the rehearsal used an export of the working tree.
- T2 and T3 inside a full run: the live batch was solved by T0 and T1 alone, and with a
  budget below about 22 000 the T3 cap is smaller than the bound of a single agent call.
- Whether our ledger matches the proxy's own accounting: the proxy does not report it.
- Accuracy on wordings other than the organisers' samples and our own generator, whose
  paraphrases were written by hand by the same author as the solvers.
- The comparison of the two no-reasoning models.

## Layout

```text
run_agent.py     entry point
src/cli.py       arguments, environment, signals, phases
src/config.py    settings from the environment
src/core/        tasks, budget ledger, scheduler, runner, model tier, agent tier, results, sandbox
src/foundation/  openJiuwen Model gateway, tools of the agent
src/solvers/     strict solvers, SQL world, coding repair and oracles
src/prompts/     few-shot prompts
scripts/         smoke test, task generator, judge, proxy probe (development only)
tests/           pytest suite
trajectories/    collected runs
```
