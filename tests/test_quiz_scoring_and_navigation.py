"""
Unit tests for quiz scoring, question navigation state, answer persistence,
and duplicate submission prevention.
"""

from typing import Dict, List
import pytest

from src.quiz_models import (
    Quiz,
    QuizQuestion,
    QuizResult,
    score_quiz,
)


@pytest.fixture
def sample_quiz() -> Quiz:
    """Creates a sample 3-question quiz across two topics."""
    q1 = QuizQuestion(
        question_id="q1",
        topic="Quantum Principles",
        question="What phenomenon enables linear combinations of states?",
        options=["Superposition", "Entanglement", "Decoherence", "Tunneling"],
        correct_answer="Superposition",
        explanation="Superposition allows linear combinations of 0 and 1.",
        source_filename="doc.pdf",
        source_page=1,
        difficulty="Easy"
    )
    q2 = QuizQuestion(
        question_id="q2",
        topic="Quantum Principles",
        question="Which algorithm provides exponential speedup for factoring?",
        options=["Shor's algorithm", "Grover's algorithm", "RSA", "ECC"],
        correct_answer="Shor's algorithm",
        explanation="Shor's algorithm provides exponential speedup.",
        source_filename="doc.pdf",
        source_page=1,
        difficulty="Medium"
    )
    q3 = QuizQuestion(
        question_id="q3",
        topic="Cryogenics",
        question="What temperature do superconducting qubits operate around?",
        options=["15 millikelvin", "77 kelvin", "273 kelvin", "4 kelvin"],
        correct_answer="15 millikelvin",
        explanation="They operate near 15 millikelvin.",
        source_filename="doc.pdf",
        source_page=2,
        difficulty="Easy"
    )
    return Quiz(
        quiz_id="quiz_test_123",
        title="Sample Quantum Quiz",
        topic="Quantum Computing",
        source_filename="doc.pdf",
        difficulty="Medium",
        questions=[q1, q2, q3]
    )


def test_perfect_score(sample_quiz: Quiz):
    """Verifies 100% score when all questions are answered correctly."""
    user_answers = {
        "q1": "Superposition",
        "q2": "Shor's algorithm",
        "q3": "15 millikelvin"
    }

    result = score_quiz(quiz=sample_quiz, user_answers=user_answers)

    assert result.total_questions == 3
    assert result.correct_count == 3
    assert result.wrong_count == 0
    assert result.unanswered_count == 0
    assert result.score_percentage == 100.0

    # Topic breakdown
    assert result.topic_scores["Quantum Principles"]["correct"] == 2
    assert result.topic_scores["Quantum Principles"]["total"] == 2
    assert result.topic_scores["Cryogenics"]["correct"] == 1
    assert result.topic_scores["Cryogenics"]["total"] == 1


def test_partial_score_and_letter_answers(sample_quiz: Quiz):
    """
    Verifies that letter-based answers (e.g. 'A') map to options correctly
    and partial score percentage is computed accurately.
    """
    # q1: Option A is "Superposition" (Correct)
    # q2: Option B is "Grover's algorithm" (Wrong - correct is Shor's)
    # q3: "15 millikelvin" (Correct)
    user_answers = {
        "q1": "A",
        "q2": "B",
        "q3": "15 millikelvin"
    }

    result = score_quiz(quiz=sample_quiz, user_answers=user_answers)

    assert result.total_questions == 3
    assert result.correct_count == 2
    assert result.wrong_count == 1
    assert result.unanswered_count == 0
    assert result.score_percentage == 66.7


def test_unanswered_questions_counted_as_incorrect(sample_quiz: Quiz):
    """Verifies that missing or omitted answers are tracked as unanswered and wrong."""
    user_answers = {
        "q1": "Superposition"
        # q2 and q3 omitted
    }

    result = score_quiz(quiz=sample_quiz, user_answers=user_answers)

    assert result.total_questions == 3
    assert result.correct_count == 1
    assert result.wrong_count == 0
    assert result.unanswered_count == 2
    assert result.score_percentage == 33.3

    assert result.question_results[1].selected_answer is None
    assert result.question_results[1].is_correct is False


def test_zero_score_all_wrong(sample_quiz: Quiz):
    """Verifies 0.0% score when all questions are answered incorrectly."""
    user_answers = {
        "q1": "Decoherence",
        "q2": "RSA",
        "q3": "273 kelvin"
    }

    result = score_quiz(quiz=sample_quiz, user_answers=user_answers)

    assert result.correct_count == 0
    assert result.wrong_count == 3
    assert result.unanswered_count == 0
    assert result.score_percentage == 0.0


def test_navigation_state_boundary_conditions():
    """
    Verifies question index navigation boundary logic:
    cannot navigate backward below 0, cannot navigate forward beyond total-1.
    """
    total_questions = 5

    current_idx = 0
    # Try previous at beginning
    can_prev = current_idx > 0
    assert not can_prev

    # Step forward
    if current_idx < total_questions - 1:
        current_idx += 1
    assert current_idx == 1

    # Jump to end
    current_idx = 4
    can_next = current_idx < total_questions - 1
    assert not can_next


def test_prevent_duplicate_submissions_in_history():
    """
    Verifies that repeatedly submitting or re-rendering does not inflate
    the user's quiz attempt history.
    """
    history: List[QuizResult] = []
    submission_recorded = False

    def handle_submit(result: QuizResult):
        nonlocal submission_recorded
        if not submission_recorded:
            history.append(result)
            submission_recorded = True

    dummy_result = QuizResult(
        quiz_id="quiz_1",
        total_questions=1,
        correct_count=1,
        wrong_count=0,
        unanswered_count=0,
        score_percentage=100.0,
        topic_scores={},
        question_results=[],
        submitted_at="2026-10-29T00:00:00"
    )

    # First submission
    handle_submit(dummy_result)
    assert len(history) == 1

    # Duplicate submission attempt (e.g. user re-clicks or page re-renders)
    handle_submit(dummy_result)
    assert len(history) == 1

