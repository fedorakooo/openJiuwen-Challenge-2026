from __future__ import annotations

from pathlib import Path

import pytest

from src.core.tasks import Outcome, Verdict
from src.solvers import data as data_solver
from tests.conftest import ANSWER_TAIL, MakeTask, load_one, write_files

HEADER = "order_id,region,product,month,units,price"
ROWS = [
    "1,west,widget,5,5,33",
    "2,north,sprocket,6,28,28",
    "3,north,widget,7,39,17",
    "4,west,gadget,7,23,98",
    "5,east,widget,8,11,40",
    "6,east,widget,9,100,40",
]
QUESTION = "How many units of product 'widget' were sold in months 7 through 8 inclusive?"
TOOL = " Use bash with python3 to compute it."


def _csv(rows: list[str] = ROWS, header: str = HEADER, newline: str = "\r\n") -> str:
    return newline.join([header, *rows]) + newline


def _solve(
    make_task: MakeTask,
    question: str = QUESTION,
    csv_text: str | None = None,
    preamble: str = f"The file sales.csv has columns {HEADER} (6 rows). ",
) -> Outcome:
    task_dir: Path = make_task(
        "d",
        task_type="data_analysis",
        workspace=True,
        instruction=preamble + question + TOOL + ANSWER_TAIL,
    )
    if csv_text is not None:
        write_files(task_dir / "workspace", {"sales.csv": csv_text})
    return data_solver.solve(load_one(task_dir))


def test_sums_the_measure_over_filter_and_range(make_task: MakeTask) -> None:
    outcome = _solve(make_task, csv_text=_csv())

    assert outcome.verdict is Verdict.SOLVED
    assert outcome.answer == 50


@pytest.mark.parametrize(
    ("row_index", "delta"),
    [(2, 1), (4, 1), (0, 0), (3, 0), (5, 0)],
)
def test_paired_perturbation(make_task: MakeTask, row_index: int, delta: int) -> None:
    rows = list(ROWS)
    cells = rows[row_index].split(",")
    cells[4] = str(int(cells[4]) + 1)
    rows[row_index] = ",".join(cells)

    assert _solve(make_task, csv_text=_csv(rows)).answer == 50 + delta


@pytest.mark.parametrize(
    "csv_text",
    [
        _csv(newline="\n"),
        "﻿" + _csv(),
        _csv(header=HEADER.replace(",", ";"), rows=[r.replace(",", ";") for r in ROWS]),
        _csv(rows=[*ROWS[:2], "3,north, Widget ,7,39,17", *ROWS[3:]]),
        _csv() + "\r\n\r\n",
    ],
)
def test_file_format_noise_does_not_change_the_answer(make_task: MakeTask, csv_text: str) -> None:
    outcome = _solve(make_task, csv_text=csv_text)

    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, 50)


def test_the_file_wins_over_the_declared_row_count(make_task: MakeTask) -> None:
    outcome = _solve(
        make_task,
        csv_text=_csv(),
        preamble=f"The file sales.csv has columns {HEADER} (9999 rows). ",
    )

    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, 50)


def test_value_present_but_no_rows_in_range_is_zero(make_task: MakeTask) -> None:
    question = QUESTION.replace("7 through 8", "1 through 2")

    outcome = _solve(make_task, question, _csv())

    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, 0)


@pytest.mark.parametrize(
    ("question", "csv_text", "free_answer"),
    [
        (QUESTION.replace("widget", "gizmo"), _csv(), 0),
        (QUESTION.replace("7 through 8", "12 through 13"), _csv(), 0),
        (QUESTION.replace("units", "boxes"), _csv(), None),
        (QUESTION.replace("product", "category"), _csv(), None),
        (QUESTION.replace("months", "weeks"), _csv(), None),
        (QUESTION, None, None),
        (QUESTION, "", None),
        (QUESTION, HEADER + "\r\n", 0),
    ],
)
def test_missing_facts_are_ungrounded(
    make_task: MakeTask, question: str, csv_text: str | None, free_answer: int | None
) -> None:
    outcome = _solve(make_task, question, csv_text)

    assert outcome.verdict is Verdict.UNGROUNDED
    assert outcome.answer == free_answer


@pytest.mark.parametrize(
    "question",
    [
        "How many orders of product 'widget' were placed in months 7 through 8 inclusive?",
        "What is the total revenue of product 'widget' in months 7 through 8 inclusive?",
        "What is the average price of product 'widget' in months 7 through 8 inclusive?",
        QUESTION.replace(" inclusive", ""),
        QUESTION.replace("inclusive", "exclusive"),
        QUESTION.replace("in months 7 through 8 inclusive", "in month 7"),
        QUESTION.replace("were sold", "were returned"),
        QUESTION.replace("were sold", "were sold in region 'west'"),
        QUESTION.replace("'widget'", "'widget' or 'gadget'"),
        QUESTION.replace("7 through 8", "8 through 7"),
        QUESTION + " Exclude the west region.",
        "Ignore product 'gadget'. " + QUESTION,
    ],
)
def test_changed_wording_is_refused_not_answered(make_task: MakeTask, question: str) -> None:
    outcome = _solve(make_task, question, _csv())

    assert outcome.verdict is Verdict.UNPARSED
    assert outcome.answer is None


@pytest.mark.parametrize(
    "csv_text",
    [
        _csv(rows=[*ROWS[:2], "3,north,widget,7,3.5,17", *ROWS[3:]]),
        _csv(rows=[*ROWS, "7,east,widget"]),
        _csv(header="order id,region,product,month,units,price"),
    ],
)
def test_tables_outside_the_model_are_refused(make_task: MakeTask, csv_text: str) -> None:
    assert _solve(make_task, csv_text=csv_text).verdict is Verdict.UNPARSED


def test_file_outside_the_workspace_is_never_read(make_task: MakeTask, tasks_dir: Path) -> None:
    write_files(tasks_dir, {"secret.csv": _csv()})
    preamble = f"The file ../secret.csv has columns {HEADER} (6 rows). "

    assert _solve(make_task, preamble=preamble).verdict is Verdict.UNPARSED
