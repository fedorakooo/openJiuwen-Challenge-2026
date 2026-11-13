"""Command line entry: arguments, environment preparation, signals; standard library only."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path

from src.config import DEFAULT_TASKS_DIR, Settings, load_settings
from src.core.results import (
    TRAJECTORY_FILE,
    ResultWriter,
    TrajectoryLog,
    resolve_aggregate_path,
)
from src.core.runner import Clock, RunStats, Solver, arbitrate, run_phase0, run_phase1
from src.core.scheduler import (
    ProxyGuard,
    complaint_for_reasoning,
    is_qualification,
    order_by_density,
    worth_an_agent,
)
from src.core.tasks import Outcome, load_tasks

logger = logging.getLogger("src")

_RUN_DIR_NAME = "run"
_CANARY_WAIT_S = 130.0
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse known arguments only; unknown ones are ignored instead of aborting."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--tasks-dir", dest="tasks_dir")
    parser.add_argument("--tasks", dest="tasks")
    parser.add_argument("--out", dest="out")
    parser.add_argument("positional", nargs="?")
    try:
        args, _unknown = parser.parse_known_args(list(argv))
    except SystemExit:  # e.g. an option given without its value
        args = argparse.Namespace(tasks_dir=None, tasks=None, out=None, positional=None)
    return args


def resolve_tasks_dir(args: argparse.Namespace, env: Mapping[str, str]) -> Path:
    """Precedence: --tasks-dir, --tasks, positional argument, TASKS, /data/tasks."""
    for candidate in (args.tasks_dir, args.tasks, args.positional, env.get("TASKS")):
        if candidate and candidate.strip():
            return Path(candidate.strip())
    return Path(DEFAULT_TASKS_DIR)


def _writable_tmp_root() -> Path:
    candidates = [Path("/tmp")] if os.name == "posix" else []
    candidates.append(Path(tempfile.gettempdir()))
    for candidate in candidates:
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate
    return candidates[-1]


def prepare_environment() -> Path:
    """Move cwd, temp and cache locations to the writable tmpfs; return the run directory."""
    root = _writable_tmp_root()
    run_dir = root / _RUN_DIR_NAME
    cache_dir = root / ".cache"
    mpl_dir = root / ".mpl"
    for directory in (run_dir, run_dir / "logs", cache_dir, mpl_dir):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    os.environ["TMPDIR"] = str(root)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ.setdefault("HOME", str(root))
    os.environ.update(_OFFLINE_ENV)
    tempfile.tempdir = None  # make tempfile pick up the new TMPDIR
    try:
        os.chdir(run_dir)
    except OSError:
        os.chdir(root)
        run_dir = root
    return run_dir


def _configure_logging(run_dir: Path) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        handlers.append(logging.FileHandler(run_dir / "logs" / "run.log", encoding="utf-8"))
    except OSError:
        pass
    for stale in list(logger.handlers):
        logger.removeHandler(stale)
        stale.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def install_signal_handlers(stop: threading.Event) -> None:
    """SIGTERM and SIGINT only raise the stop flag; answers are already on disk."""

    def handle(signum: int, _frame: object) -> None:
        stop.set()
        logger.warning("signal %d received, stopping", signum)

    for name in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            signal.signal(signum, handle)
        except (OSError, ValueError):  # not the main thread, or unsupported on this platform
            pass


def _default_solvers() -> Mapping[str, Solver]:
    """Strict T0 solvers; a broken solver package must not cost the whole run."""
    try:
        from src.solvers import SOLVERS
    except Exception:
        logger.exception("T0 solvers are unavailable")
        return {}
    return SOLVERS


def _start_canary() -> Future[bool] | None:
    """Start the SDK import check in parallel with phase 0, if the proxy is configured."""
    if not os.environ.get("LLM_BASE_URL", "").strip() or not os.environ.get("LLM_API_KEY"):
        logger.info("LLM_BASE_URL or LLM_API_KEY is not set: only zero-token solvers will run")
        return None
    try:
        from src.foundation.llm import start_canary

        return start_canary()
    except Exception:
        logger.exception("cannot start the SDK canary")
        return None


def _run_model_phase(
    stats: RunStats,
    rewards: Sequence[float],
    qualification: bool,
    canary: Future[bool],
    settings: Settings,
    writer: ResultWriter,
    trajectory: TrajectoryLog,
    clock: Clock,
    stop: threading.Event,
) -> None:
    """Paid tiers for the tasks the strict solvers refused: T1, then T2, then the agent."""
    try:
        sdk_importable = canary.result(timeout=_CANARY_WAIT_S)
    except Exception:
        sdk_importable = False
    from src.core.agent import Tier1
    from src.core.budget import Ledger
    from src.foundation.llm import NOREASON_TIMEOUT_S, REASON_TIMEOUT_S, build_gateway

    gateway = build_gateway(os.environ, sdk_importable)
    if gateway is None:
        return
    ledger = Ledger(settings.budget)
    guard = ProxyGuard(
        ledger,
        settings.llm_concurrency,
        can_wait=lambda seconds: clock.can_start(seconds + NOREASON_TIMEOUT_S),
    )
    shared = {
        "writer": writer,
        "trajectory": trajectory,
        "clock": clock,
        "stop": stop,
        "concurrency": settings.llm_concurrency,
        "answer_when_unsure": settings.answer_when_unsure,
    }
    outcomes: dict[str, Outcome] = {}
    disagreements: dict[str, str] = {}

    def report(tier: str, model: str, solved: int, attempted: int) -> None:
        logger.info(
            "%s done via %s (%s): %d solved of %d; ledger: %d calls, spent %.1f of %d "
            "(unsettled %.1f), %.2f chars per token",
            tier,
            gateway.name,
            model,
            solved,
            attempted,
            ledger.calls,
            ledger.spent,
            settings.budget,
            ledger.unknown,
            ledger.chars_per_token,
        )

    async def phases() -> None:
        tier1 = Tier1(
            gateway, ledger, trajectory, settings.t1_model, guard=guard, qualification=qualification
        )
        deferred = order_by_density(stats.deferred, ledger)
        solved = await run_phase1(
            deferred,
            tier1.solve,
            attempt_timeout_s=NOREASON_TIMEOUT_S,
            verify=stats.verify,
            outcomes=outcomes,
            disagreements=disagreements,
            **shared,
        )
        report("T1", settings.t1_model, solved, len(deferred))
        _log_costs("T1", tier1.task_cost)

        if settings.t2_enabled and not stop.is_set() and guard.dead_reason is None:
            tier2 = Tier1(
                gateway,
                ledger,
                trajectory,
                settings.t2_model,
                guard=guard,
                qualification=qualification,
                tier="T2",
                reasoning=True,
            )
            # Code goes straight to the agent: a reasoning model spends its whole output
            # budget thinking about a patch, while the agent can run the tests.
            complaints = {
                task.task_id: complaint
                for task in deferred
                if task.task_id in outcomes
                and task.task_type != "coding"
                and (complaint := complaint_for_reasoning(outcomes[task.task_id], qualification))
            }
            retry = order_by_density(
                [task for task in deferred if task.task_id in complaints], ledger, "T2"
            )
            solved = await run_phase1(
                retry,
                lambda task: tier2.solve(task, complaints[task.task_id]),
                attempt_timeout_s=REASON_TIMEOUT_S,
                tier="T2",
                outcomes=outcomes,
                **shared,
            )
            replaced = await arbitrate(stats.verify, disagreements, tier2.solve, writer, stop)
            report("T2", settings.t2_model, solved, len(retry))
            if disagreements:
                logger.info(
                    "arbiter: %d of %d disputed answers replaced", replaced, len(disagreements)
                )
            _log_costs("T2", tier2.task_cost)

        candidates = [
            task for task in deferred if worth_an_agent(task, outcomes.get(task.task_id), rewards)
        ]
        agent_ready = settings.t3_enabled and gateway.name == "openjiuwen" and candidates
        if agent_ready and not stop.is_set() and guard.dead_reason is None:
            from src.core.react import AgentTier

            agent = AgentTier(
                os.environ.get("LLM_BASE_URL", ""),
                os.environ.get("LLM_API_KEY", "").strip(),
                settings.t2_model,
                ledger,
                trajectory,
                guard,
            )
            ordered = order_by_density(candidates, ledger, "T3")
            solved = await run_phase1(
                ordered, agent.solve, attempt_timeout_s=REASON_TIMEOUT_S, tier="T3", **shared
            )
            report("T3", settings.t2_model, solved, len(ordered))
            _log_costs("T3", agent.task_cost)

    asyncio.run(phases())
    if guard.dead_reason is not None:
        logger.error("model tiers were stopped: %s", guard.dead_reason)


def _open_trajectory(run_dir: Path) -> TrajectoryLog:
    """Trajectory goes to the tmpfs unless TRAJECTORY_DIR names a writable directory."""
    wanted = os.environ.get("TRAJECTORY_DIR", "").strip()
    if wanted and Path(wanted).is_absolute() and Path(wanted).is_dir():
        trajectory = TrajectoryLog(Path(wanted) / TRAJECTORY_FILE)
        if trajectory.enabled:
            return trajectory
    return TrajectoryLog(run_dir / "logs" / TRAJECTORY_FILE)


def _log_costs(tier: str, costs: Mapping[str, float]) -> None:
    text = ", ".join(f"{task_id}={cost:.1f}" for task_id, cost in sorted(costs.items()))
    logger.info("%s cost per task: %s", tier, text or "no settled calls")


def main(argv: Sequence[str] | None = None, solvers: Mapping[str, Solver] | None = None) -> int:
    """Run the harness; always returns 0."""
    stop = threading.Event()
    trajectory = TrajectoryLog(None)
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        # Resolve user paths before cwd moves to the tmpfs.
        tasks_dir = resolve_tasks_dir(args, os.environ).absolute()
        out = args.out or os.environ.get("OUT")
        aggregate_path = resolve_aggregate_path(str(Path(out).absolute()) if out else None)

        settings = load_settings()
        clock = Clock(settings)
        run_dir = prepare_environment()
        _configure_logging(run_dir)
        install_signal_handlers(stop)

        tasks = load_tasks(tasks_dir)
        qualification = is_qualification(len(tasks))
        logger.info(
            "loaded %d tasks, budget %d, qualification mode: %s",
            len(tasks),
            settings.budget,
            qualification,
        )

        trajectory = _open_trajectory(run_dir)
        writer = ResultWriter(aggregate_path)
        canary = _start_canary()
        if solvers is None:
            solvers = _default_solvers() if settings.t0_enabled else {}
        stats = run_phase0(
            tasks,
            solvers,
            writer,
            trajectory,
            clock,
            stop,
            answer_when_unsure=settings.answer_when_unsure,
            qualification=qualification,
        )
        logger.info(
            "phase 0 done in %.2fs: %d solved, %d ungrounded (%d unsure answers written), "
            "%d deferred of %d",
            clock.elapsed(),
            stats.solved,
            stats.ungrounded,
            stats.unsure,
            len(stats.deferred),
            stats.total,
        )
        if (stats.deferred or stats.verify) and canary is not None and not stop.is_set():
            _run_model_phase(
                stats,
                [task.reward for task in tasks],
                qualification,
                canary,
                settings,
                writer,
                trajectory,
                clock,
                stop,
            )
    except Exception:  # exit code 0 is part of the contract
        logger.exception("run aborted")
    finally:
        trajectory.close()
    return 0
