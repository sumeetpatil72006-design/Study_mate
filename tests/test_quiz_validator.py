"""
Unit tests for QuizValidator.
Validates question schema, exactly 4 distinct options, answer presence,
duplicate question rejection, and factual grounding against context chunks.
"""

import pytest

from src.exceptions import QuizValidationError
from src.models import RetrievalResult
from src.quiz_models import QuizQuestion
from src.quiz_validator import QuizValidator


@pytest.fixture
def sample_context_chunks():
    """Provides sample retrieval chunks for grounding verification."""
    return [
        RetrievalResult(
            chunk_id="ch_p1",
            content="Quantum computing harnesses qubits which utilize superposition and entanglement. Shor's algorithm provides exponential speedup.",
            source_filename="quantum.pdf",
            page_number=1,
            score=0.95,
            metadata={}
        ),
        RetrievalResult(
            chunk_id="ch_p2",
            content="Superconducting qubits operate at cryogenic temperatures near 15 millikelvin inside dilution refrigerators.",
            source_filename="quantum.pdf",
            page_number=2,
            score=0.90,
            metadata={}
        )
    ]


@pytest.fixture
def valid_question_data():
    return {
        "question_id": "q1",
        "topic": "Cryogenics",
        "question": "What temperature do superconducting qubits operate near?",
        "options": [
            "15 millikelvin",
            "300 kelvin",
            "77 kelvin",
            "4 millikelvin"
        ],
        "correct_answer": "15 millikelvin",
        "explanation": "Superconducting qubits operate at cryogenic temperatures near 15 millikelvin inside dilution refrigerators.",
        "source_filename": "quantum.pdf",
        "source_page": 2,
        "difficulty": "Easy"
    }


def test_valid_question_passes_validation(valid_question_data, sample_context_chunks):
    """Verifies that a well-formed grounded question passes validation without error."""
    q = QuizValidator.validate_single_question(
        raw_q=valid_question_data,
        context_chunks=sample_context_chunks,
        expected_filename="quantum.pdf",
        allowed_pages={1, 2}
    )
    assert isinstance(q, QuizQuestion)
    assert q.question_id == "q1"
    assert q.correct_answer == "15 millikelvin"
    assert len(q.options) == 4
    assert q.source_page == 2


def test_missing_required_fields_rejected(valid_question_data, sample_context_chunks):
    """Verifies that questions missing required fields are rejected."""
    incomplete = dict(valid_question_data)
    del incomplete["explanation"]

    with pytest.raises(QuizValidationError) as exc:
        QuizValidator.validate_single_question(
            raw_q=incomplete,
            context_chunks=sample_context_chunks
        )
    assert "missing required fields" in str(exc.value).lower()


def test_options_count_must_be_exactly_four(valid_question_data, sample_context_chunks):
    """Verifies that questions with fewer or more than 4 options are rejected."""
    # 3 options
    three_opts = dict(valid_question_data)
    three_opts["options"] = ["Option 1", "Option 2", "Option 3"]
    with pytest.raises(QuizValidationError) as exc:
        QuizValidator.validate_single_question(three_opts, sample_context_chunks)
    assert "exactly 4 options" in str(exc.value)

    # 5 options
    five_opts = dict(valid_question_data)
    five_opts["options"] = ["O1", "O2", "O3", "O4", "O5"]
    with pytest.raises(QuizValidationError) as exc:
        QuizValidator.validate_single_question(five_opts, sample_context_chunks)
    assert "exactly 4 options" in str(exc.value)


def test_duplicate_options_rejected(valid_question_data, sample_context_chunks):
    """Verifies that questions with duplicate options are rejected."""
    dup_opts = dict(valid_question_data)
    dup_opts["options"] = [
        "15 millikelvin",
        "300 kelvin",
        "15 millikelvin",  # Duplicate
        "4 millikelvin"
    ]
    with pytest.raises(QuizValidationError) as exc:
        QuizValidator.validate_single_question(dup_opts, sample_context_chunks)
    assert "distinct" in str(exc.value).lower()


def test_empty_options_rejected(valid_question_data, sample_context_chunks):
    """Verifies that questions with empty option strings are rejected."""
    empty_opts = dict(valid_question_data)
    empty_opts["options"] = ["15 millikelvin", "", "77 kelvin", "4 millikelvin"]
    with pytest.raises(QuizValidationError) as exc:
        QuizValidator.validate_single_question(empty_opts, sample_context_chunks)
    assert "empty strings" in str(exc.value).lower()


def test_correct_answer_not_in_options_rejected(valid_question_data, sample_context_chunks):
    """Verifies that questions whose correct_answer does not match any option are rejected."""
    bad_ans = dict(valid_question_data)
    bad_ans["correct_answer"] = "999 kelvin"  # Not in options
    with pytest.raises(QuizValidationError) as exc:
        QuizValidator.validate_single_question(bad_ans, sample_context_chunks)
    assert "does not match any of the 4 options" in str(exc.value)


def test_letter_based_correct_answer_resolved(valid_question_data, sample_context_chunks):
    """Verifies that letter correct_answers (e.g. 'A') are resolved to the matching option."""
    letter_ans = dict(valid_question_data)
    letter_ans["correct_answer"] = "A"
    q = QuizValidator.validate_single_question(letter_ans, sample_context_chunks)
    assert q.correct_answer == "15 millikelvin"


def test_hallucinated_page_number_rejected(valid_question_data, sample_context_chunks):
    """Verifies that citing a non-existent page number raises QuizValidationError."""
    bad_page = dict(valid_question_data)
    bad_page["source_page"] = 99  # Allowed pages are only {1, 2}
    with pytest.raises(QuizValidationError) as exc:
        QuizValidator.validate_single_question(
            bad_page,
            sample_context_chunks,
            allowed_pages={1, 2}
        )
    assert "source_page 99 is not among the document's valid pages" in str(exc.value)


def test_unsupported_facts_rejected(sample_context_chunks):
    """
    Verifies that a question containing facts completely absent from the context chunks
    is rejected as an unsupported fact.
    """
    unsupported_q = {
        "question_id": "q_alien",
        "topic": "Space Exploration",
        "question": "What is the atmospheric composition of Jupiter's moon Europa?",
        "options": [
            "Molecular oxygen",
            "Liquid methane",
            "Carbon dioxide",
            "Argon gas"
        ],
        "correct_answer": "Molecular oxygen",
        "explanation": "Europa has a tenuous atmosphere composed mostly of oxygen.",
        "source_filename": "quantum.pdf",
        "source_page": 1,
        "difficulty": "Medium"
    }

    with pytest.raises(QuizValidationError) as exc:
        QuizValidator.validate_single_question(
            unsupported_q,
            sample_context_chunks,
            allowed_pages={1, 2}
        )
    assert "unsupported fact" in str(exc.value).lower()


def test_duplicate_questions_filtered_in_batch(valid_question_data, sample_context_chunks):
    """Verifies that duplicate questions within a batch are detected and filtered."""
    q1 = dict(valid_question_data)
    q1["question_id"] = "q1"

    q2_dup = dict(valid_question_data)
    q2_dup["question_id"] = "q2"
    q2_dup["question"] = "What temperature do superconducting qubits operate near?   "  # Duplicate

    q3_distinct = dict(valid_question_data)
    q3_distinct["question_id"] = "q3"
    q3_distinct["question"] = "Which algorithm provides exponential speedup according to page 1?"
    q3_distinct["options"] = ["Shor's algorithm", "Grover's algorithm", "Bubble sort", "Dijkstra"]
    q3_distinct["correct_answer"] = "Shor's algorithm"
    q3_distinct["explanation"] = "Shor's algorithm unlocks exponential speedup."
    q3_distinct["source_page"] = 1

    valid_questions = QuizValidator.validate_quiz_questions(
        raw_questions=[q1, q2_dup, q3_distinct],
        context_chunks=sample_context_chunks,
        allowed_pages={1, 2},
        min_required=1
    )

    # Only 2 distinct questions should survive (q1 and q3)
    assert len(valid_questions) == 2
    assert valid_questions[0].question_id == "q1"
    assert valid_questions[1].question_id == "q3"

