"""Tests that run against the real 2026 timetable, calendar and 2024 handbook."""
import pytest

from chatbot.engine import KnowledgeBase, answer_query


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def ask(kb, q):
    return answer_query(kb, q)


# ---- timetable -----------------------------------------------------------
def test_module_on_monday(kb):
    a = ask(kb, "At what time is 5EEE421L on Monday?")
    assert a.source == "timetable"
    assert "10:30-11:20" in a.text and "11:30-12:20" in a.text


def test_module_not_on_tuesday(kb):
    a = ask(kb, "Is 5EEE421L scheduled on Tuesday?")
    assert a.source == "timetable"
    assert "does not appear to be scheduled on Tuesday" in a.text
    assert "Monday" in a.text and "Friday" in a.text  # tells the student when it IS scheduled


def test_module_whole_week(kb):
    a = ask(kb, "When is 5EEE421L?")
    assert "Monday" in a.text and "Friday" in a.text


def test_unknown_module(kb):
    assert "not found in the timetable" in ask(kb, "What time is 9ZZZ999L on Monday?").text


def test_group_letters_are_reported(kb):
    assert "(Group A)" in ask(kb, "When is 4MTH271L on Monday?").text


# ---- calendar ------------------------------------------------------------
def test_first_semester_start(kb):
    a = ask(kb, "When does the first semester start?")
    assert a.source == "calendar"
    assert a.text.splitlines()[0].startswith("- **Monday 9 February 2026**")


def test_second_semester_exams(kb):
    a = ask(kb, "When do the second semester exams start?")
    assert "Monday 19 October 2026" in a.text


def test_public_holidays_in_month(kb):
    a = ask(kb, "Which public holidays are in April?")
    assert "Freedom Day" in a.text and "Good Friday" in a.text


# ---- handbook ------------------------------------------------------------
def test_dp_requirement(kb):
    a = ask(kb, "What is the DP requirement?")
    assert a.source == "handbook" and "40%" in a.text


def test_entry_requirement_lists_the_real_requirements(kb):
    a = ask(kb, "What is the entry requirement for engineering?")
    assert a.source == "handbook"
    assert "30 matriculation points" in a.text
    assert "Mathematics" in a.text


def test_module_prerequisites_use_the_handbook(kb):
    a = ask(kb, "What are the prerequisites for 5EEE421L?")
    assert a.source == "handbook"
    assert "Engineering Systems Design" in a.text and "5EEE342" in a.text


# ---- out of scope --------------------------------------------------------
@pytest.mark.parametrize("q", ["What is the capital of France?", "I like pizza", ""])
def test_out_of_scope(kb, q):
    assert ask(kb, q).source == "none"
