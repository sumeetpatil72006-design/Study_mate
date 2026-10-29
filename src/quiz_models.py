"""
Data models and scoring logic for topic extraction and quiz generation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class Topic(BaseModel):
    """Represents a discovered study topic grounded in document content."""
    topic_id: str
    name: str
    description: str
    source_filename: str
    source_pages: List[int] = Field(default_factory=list)
    chunk_count: int = 0


class QuizQuestion(BaseModel):
    """
    Represents a validated multiple-choice quiz question.
    Requires exactly four distinct options and a matching correct answer.
    """
    question_id: str
    topic: str
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    source_filename: str
    source_page: int
    difficulty: str = "Medium"

    @field_validator("options")
    @classmethod
    def validate_options_count(cls, v: List[str]) -> List[str]:
        if len(v) != 4:
            raise ValueError(f"Question must have exactly 4 options, got {len(v)}")
        cleaned = [opt.strip() for opt in v if opt and opt.strip()]
        if len(cleaned) != 4:
            raise ValueError("All 4 options must be non-empty strings")
        # Check distinct options (case-insensitive)
        if len({opt.lower() for opt in cleaned}) != 4:
            raise ValueError("All 4 options must be distinct")
        return cleaned

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("correct_answer cannot be empty")
        return v.strip()

    @field_validator("source_page")
    @classmethod
    def validate_source_page_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"source_page must be a positive integer, got {v}")
        return v


class QuizConfig(BaseModel):
    """Configuration for quiz generation."""
    topic: str
    num_questions: int = 5
    difficulty: str = "Medium"
    question_type: str = "MCQ"
    source_filename: Optional[str] = None


class Quiz(BaseModel):
    """A generated and validated collection of quiz questions."""
    quiz_id: str
    title: str
    topic: str
    source_filename: str
    difficulty: str
    questions: List[QuizQuestion]
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class QuestionResult(BaseModel):
    """Outcome for a single quiz question after learner submission."""
    question_id: str
    question: str
    options: List[str]
    selected_answer: Optional[str] = None
    correct_answer: str
    is_correct: bool
    explanation: str
    source_filename: str
    source_page: int
    difficulty: str
    topic: str


class QuizResult(BaseModel):
    """Aggregated scoring and review results for a submitted quiz."""
    quiz_id: str
    total_questions: int
    correct_count: int
    wrong_count: int
    unanswered_count: int
    score_percentage: float
    topic_scores: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    question_results: List[QuestionResult] = Field(default_factory=list)
    submitted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def score_quiz(quiz: Quiz, user_answers: Dict[str, str]) -> QuizResult:
    """
    Calculates overall score, percentage, correct/wrong/unanswered counts,
    and topic-wise performance for a quiz submission.

    Args:
        quiz: The Quiz instance being submitted.
        user_answers: Mapping of question_id -> selected option text or key.

    Returns:
        QuizResult with detailed question-level outcomes and topic breakdown.
    """
    total = len(quiz.questions)
    correct_count = 0
    wrong_count = 0
    unanswered_count = 0

    question_results: List[QuestionResult] = []
    topic_scores: Dict[str, Dict[str, int]] = {}

    for q in quiz.questions:
        topic_name = q.topic or quiz.topic or "General"
        if topic_name not in topic_scores:
            topic_scores[topic_name] = {"correct": 0, "total": 0}
        topic_scores[topic_name]["total"] += 1

        selected = user_answers.get(q.question_id)
        if selected is not None:
            selected = str(selected).strip()

        # Check correctness
        is_correct = False
        if not selected:
            unanswered_count += 1
            is_correct = False
        else:
            # Direct match
            if selected.lower() == q.correct_answer.strip().lower():
                is_correct = True
            # Check letter-based answer (e.g. "A" / "B" / "C" / "D" mapping)
            elif len(selected) == 1 and selected.upper() in ["A", "B", "C", "D"]:
                idx = ord(selected.upper()) - ord("A")
                if 0 <= idx < len(q.options):
                    if q.options[idx].strip().lower() == q.correct_answer.strip().lower():
                        is_correct = True

            if is_correct:
                correct_count += 1
                topic_scores[topic_name]["correct"] += 1
            else:
                wrong_count += 1

        question_results.append(
            QuestionResult(
                question_id=q.question_id,
                question=q.question,
                options=q.options,
                selected_answer=selected,
                correct_answer=q.correct_answer,
                is_correct=is_correct,
                explanation=q.explanation,
                source_filename=q.source_filename,
                source_page=q.source_page,
                difficulty=q.difficulty,
                topic=topic_name,
            )
        )

    percentage = round((correct_count / total * 100.0), 1) if total > 0 else 0.0

    return QuizResult(
        quiz_id=quiz.quiz_id,
        total_questions=total,
        correct_count=correct_count,
        wrong_count=wrong_count,
        unanswered_count=unanswered_count,
        score_percentage=percentage,
        topic_scores=topic_scores,
        question_results=question_results,
        submitted_at=datetime.now(timezone.utc).isoformat(),
    )

