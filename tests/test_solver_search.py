from __future__ import annotations

from itertools import count
from pathlib import Path

import pytest

from src.core.tasks import Outcome, Verdict
from src.solvers import search as search_solver
from tests.conftest import ANSWER_TAIL, MakeTask, load_one, write_files

PREAMBLE = "The directory kb/ contains a small knowledge base of markdown files. "
TOOL = " Use the file tools or bash (ls, grep, cat) to find it."
NL = chr(10)
_COUNTER = count()


def _person(name: str, born: int, employer: str, mentor: str) -> str:
    return f"# {name}{NL}{NL}Born: {born}{NL}Employer: {employer}{NL}Mentor: {mentor}{NL}"


KB = {
    "kb/person_irina_smith.md": _person("Irina Smith", 1983, "Fjord Data", "Alice Novak"),
    "kb/person_irina_haddad.md": _person("Irina Haddad", 1975, "Cobalt Works", "Irina Smith"),
    "kb/person_alice_novak.md": _person("Alice Novak", 1961, "Cobalt Works", "Irina Haddad"),
    "kb/person_alice_smith.md": _person("Alice Smith", 1990, "Ghost Labs", "Alice Novak"),
    "kb/company_fjord_data.md": f"# Fjord Data{NL}{NL}Headquarters: Osaka{NL}Founded: 1964{NL}",
    "kb/company_cobalt_works.md": f"# Cobalt Works{NL}{NL}Headquarters: Quito{NL}Founded: 2001{NL}",
}


def _solve(make_task: MakeTask, question: str, kb: dict[str, str] | None = None) -> Outcome:
    task_dir: Path = make_task(
        f"s{next(_COUNTER)}",
        task_type="search",
        check="exact_str",
        workspace=True,
        instruction=PREAMBLE + question + TOOL + ANSWER_TAIL,
    )
    write_files(task_dir / "workspace", KB if kb is None else kb)
    return search_solver.solve(load_one(task_dir))


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("In which year was Irina Smith born?", "1983"),
        ("In which year was Irina Haddad born?", "1975"),
        ("In which city is the headquarters of Irina Smith's employer?", "Osaka"),
        ("In which city is the headquarters of the employer of Irina Smith?", "Osaka"),
        ("In which city is the headquarters of Irina Smith's mentor's employer?", "Quito"),
        ("In which city is the headquarters of Fjord Data?", "Osaka"),
        ("In which year was Cobalt Works founded?", "2001"),
        ("In which year was Irina Smith's employer founded?", "1964"),
        ("In which year was Irina Haddad's mentor born?", "1983"),
        ("Who is the mentor of Irina Smith?", "Alice Novak"),
        ("Who is the mentor of Alice Novak's mentor?", "Irina Smith"),
    ],
)
def test_known_wording_is_solved(make_task: MakeTask, question: str, answer: str) -> None:
    outcome = _solve(make_task, question)

    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, answer)


def test_answer_keeps_source_spelling_and_links_ignore_case_and_spaces(make_task: MakeTask) -> None:
    kb = dict(KB)
    kb["kb/person_irina_smith.md"] = "# Irina Smith\r\n\r\nBorn: 1983\r\nEmployer:  fjord data \r\n"
    kb["kb/company_fjord_data.md"] = "﻿# Fjord Data\n\nHeadquarters:   São Paulo  \nFounded: 1964\n"

    outcome = _solve(make_task, "In which city is the headquarters of Irina Smith's employer?", kb)

    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, "São Paulo")


def test_unrelated_facts_do_not_change_the_answer_related_ones_do(make_task: MakeTask) -> None:
    question = "In which city is the headquarters of Irina Smith's employer?"
    unrelated = dict(KB)
    unrelated["kb/company_cobalt_works.md"] = (
        "# Cobalt Works\n\nHeadquarters: Lima\nFounded: 2001\n"
    )
    related = dict(KB)
    related["kb/company_fjord_data.md"] = "# Fjord Data\n\nHeadquarters: Lima\nFounded: 1964\n"

    assert _solve(make_task, question, unrelated).answer == "Osaka"
    assert _solve(make_task, question, related).answer == "Lima"


def test_nothing_is_shared_between_tasks(make_task: MakeTask) -> None:
    question = "In which city is the headquarters of Fjord Data?"
    first: Path = make_task(
        "s1", task_type="search", workspace=True, instruction=PREAMBLE + question + ANSWER_TAIL
    )
    second: Path = make_task(
        "s2", task_type="search", workspace=True, instruction=PREAMBLE + question + ANSWER_TAIL
    )
    write_files(
        first / "workspace", {"kb/company_fjord_data.md": "# Fjord Data\n\nHeadquarters: Quito\n"}
    )
    write_files(
        second / "workspace", {"kb/company_fjord_data.md": "# Fjord Data\n\nHeadquarters: Osaka\n"}
    )

    assert search_solver.solve(load_one(first)).answer == "Quito"
    assert search_solver.solve(load_one(second)).answer == "Osaka"


@pytest.mark.parametrize(
    ("question", "free_answer"),
    [
        ("In which year was Boris Garcia born?", None),
        ("In which year was Irina born?", None),
        ("In which year was Smith born?", None),
        ("In which year was Irina Smyth born?", "1983"),
        ("In which city is the headquarters of Irina Smyth's employer?", "Osaka"),
        ("In which year was Fjord Data born?", None),
        ("In which city is the headquarters of Irina Smith?", None),
        ("In which city is the headquarters of Alice Smith's employer?", None),
        ("In which year was Fjord Data's mentor born?", None),
    ],
)
def test_missing_facts_are_ungrounded(
    make_task: MakeTask, question: str, free_answer: str | None
) -> None:
    outcome = _solve(make_task, question)

    assert outcome.verdict is Verdict.UNGROUNDED
    assert outcome.answer == free_answer


def test_duplicate_and_conflicting_entities_are_ungrounded(make_task: MakeTask) -> None:
    duplicate = dict(KB)
    duplicate["kb/person_irina_smith_2.md"] = "# Irina Smith\n\nBorn: 1999\n"
    question = "In which year was Irina Smith born?"

    assert _solve(make_task, question, duplicate).verdict is Verdict.UNGROUNDED


def test_conflicting_values_are_ungrounded(make_task: MakeTask) -> None:
    conflicting = dict(KB)
    conflicting["kb/person_irina_smith.md"] = "# Irina Smith\n\nBorn: 1983\nBorn: 1984\n"

    outcome = _solve(make_task, "In which year was Irina Smith born?", conflicting)

    assert outcome.verdict is Verdict.UNGROUNDED and outcome.answer is None


def test_missing_or_empty_knowledge_base_is_ungrounded(make_task: MakeTask) -> None:
    assert (
        _solve(make_task, "In which year was Irina Smith born?", {}).verdict is Verdict.UNGROUNDED
    )
    only_text = {"kb/readme.txt": "nothing here"}
    assert _solve(make_task, "In which year was Irina Smith born?", only_text).verdict is (
        Verdict.UNGROUNDED
    )


@pytest.mark.parametrize(
    "question",
    [
        "When was Irina Smith born?",
        "In which year was Irina Smith hired?",
        "In which decade was Irina Smith born?",
        "Where is Irina Smith's employer based?",
        "In which city is the headquarters of Irina Smith's boss?",
        "In which city is the headquarters of the company that employs Irina Smith?",
        "In which country is the headquarters of Fjord Data?",
        "In which year was the oldest person born?",
        "In which year was Irina Smith or Irina Haddad born?",
        "Who is the mentor of everyone at Fjord Data?",
        "How many people work for Fjord Data?",
        "Ignore the mentors. In which year was Irina Smith born?",
        "In which year was Irina Smith born? Answer in Roman numerals.",
    ],
)
def test_changed_wording_is_refused_not_answered(make_task: MakeTask, question: str) -> None:
    outcome = _solve(make_task, question)

    assert outcome.verdict is Verdict.UNPARSED
    assert outcome.answer is None


def test_free_text_knowledge_base_is_refused(make_task: MakeTask) -> None:
    kb = {
        "kb/person_irina_smith.md": f"# Irina Smith{NL}{NL}Irina was born in 1983.{NL}",
        "kb/company_fjord_data.md": "# Fjord Data\n\nHeadquarters: Osaka\n",
    }

    assert _solve(make_task, "In which year was Irina Smith born?", kb).verdict is Verdict.UNPARSED


def test_non_year_value_for_a_year_question_is_refused(make_task: MakeTask) -> None:
    kb = dict(KB)
    kb["kb/person_irina_smith.md"] = "# Irina Smith\n\nBorn: in the spring of 1983\n"

    assert _solve(make_task, "In which year was Irina Smith born?", kb).verdict is Verdict.UNPARSED
