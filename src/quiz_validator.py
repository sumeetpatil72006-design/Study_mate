"""
Validation and verification service for generated quizzes.
Enforces schema requirements, option uniqueness, answer presence, citation integrity,
duplicate prevention, and grounding against retrieved document context.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.exceptions import QuizValidationError
from src.logger import logger
from src.models import RetrievalResult
from src.quiz_models import QuizQuestion

# Common stopwords to exclude during content grounding checks
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did",
    "do", "does", "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "isn't", "it", "its", "itself", "let's", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "would", "you", "your", "yours", "yourself", "yourselves"
}


class QuizValidator:
    """
    Validates quiz questions and sets of questions against strict structural
    and factual consistency constraints.
    """

    @staticmethod
    def extract_content_tokens(text: str, min_len: int = 3) -> Set[str]:
        """Extracts normalized content words excluding punctuation and stopwords."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {t for t in tokens if len(t) >= min_len and t not in STOPWORDS}

    @classmethod
    def validate_grounding(
        cls,
        question: str,
        correct_answer: str,
        context_chunks: List[RetrievalResult],
        source_page: int
    ) -> Tuple[bool, str]:
        """
        Validates that the question and correct answer are factually supported by
        the retrieved context chunks, and that the cited page has textual backing.
        """
        if not context_chunks:
            return False, "No context chunks provided to ground question."

        # Verify page existence in retrieved chunks
        chunk_pages = {c.page_number for c in context_chunks}
        if source_page not in chunk_pages:
            return (
                False,
                f"Cited source_page {source_page} not found in retrieved chunks {sorted(list(chunk_pages))}."
            )

        # Build full context text and specific page context text
        full_context = " ".join(c.content.lower() for c in context_chunks)
        page_chunks = [c for c in context_chunks if c.page_number == source_page]
        page_context = " ".join(c.content.lower() for c in page_chunks)

        # Extract content tokens from question + answer
        q_tokens = cls.extract_content_tokens(question)
        ans_tokens = cls.extract_content_tokens(correct_answer)
        combined_tokens = q_tokens | ans_tokens

        if not combined_tokens:
            return True, "No content tokens to verify."

        # Check overlap with full context
        full_overlap = {t for t in combined_tokens if t in full_context}
        if not full_overlap:
            return (
                False,
                f"Question and answer contain no substantive terms present in the document context. "
                f"Tested tokens: {list(combined_tokens)[:5]}"
            )

        # Check overlap with cited page
        page_overlap = {t for t in combined_tokens if t in page_context}
        if not page_overlap:
            return (
                False,
                f"Cited page {source_page} does not contain any key terms from the question/answer. "
                f"Key terms: {list(combined_tokens)[:5]}"
            )

        return True, "Grounded"

    @classmethod
    def validate_single_question(
        cls,
        raw_q: Any,
        context_chunks: List[RetrievalResult],
        expected_filename: Optional[str] = None,
        allowed_pages: Optional[Set[int]] = None
    ) -> QuizQuestion:
        """
        Validates a single question dictionary or model against all rules.

        Raises:
            QuizValidationError: If any rule is violated.
        """
        if isinstance(raw_q, QuizQuestion):
            data = raw_q.model_dump()
        elif isinstance(raw_q, dict):
            data = dict(raw_q)
        else:
            raise QuizValidationError(f"Invalid question type: {type(raw_q)}")

        # Check required fields
        required_fields = [
            "question_id",
            "topic",
            "question",
            "options",
            "correct_answer",
            "explanation",
            "source_filename",
            "source_page",
        ]
        missing = [f for f in required_fields if f not in data or data[f] is None]
        if missing:
            raise QuizValidationError(f"Question missing required fields: {missing}")

        q_id = str(data["question_id"]).strip()
        topic = str(data["topic"]).strip()
        question = str(data["question"]).strip()
        explanation = str(data["explanation"]).strip()
        source_filename = str(data["source_filename"]).strip()
        raw_correct = str(data["correct_answer"]).strip()
        difficulty = str(data.get("difficulty", "Medium")).strip().capitalize()

        if difficulty not in ["Easy", "Medium", "Hard"]:
            difficulty = "Medium"

        if len(question) < 10:
            raise QuizValidationError(
                f"Question text too short ({len(question)} chars): '{question}'"
            )
        if not explanation:
            raise QuizValidationError("Question explanation cannot be empty.")
        if not topic:
            raise QuizValidationError("Question topic cannot be empty.")

        # Validate options
        raw_options = data["options"]
        if not isinstance(raw_options, (list, tuple)):
            raise QuizValidationError("Options must be a list of 4 choices.")
        if len(raw_options) != 4:
            raise QuizValidationError(
                f"Question must have exactly 4 options, found {len(raw_options)}."
            )

        options = [str(opt).strip() for opt in raw_options]
        if any(not opt for opt in options):
            raise QuizValidationError("One or more options are empty strings.")

        # Verify options are distinct
        distinct_opts = {opt.lower() for opt in options}
        if len(distinct_opts) != 4:
            raise QuizValidationError(
                f"All 4 options must be distinct. Options provided: {options}"
            )

        # Resolve correct_answer
        resolved_answer: Optional[str] = None
        # Check direct text match (case-insensitive)
        for opt in options:
            if raw_correct.lower() == opt.lower():
                resolved_answer = opt
                break

        # Check letter index match (A, B, C, D)
        if not resolved_answer and len(raw_correct) == 1 and raw_correct.upper() in ["A", "B", "C", "D"]:
            idx = ord(raw_correct.upper()) - ord("A")
            if 0 <= idx < len(options):
                resolved_answer = options[idx]

        # Check prefixed match (e.g. "Option A" or "1.")
        if not resolved_answer:
            letter_match = re.match(r"^(?:option\s+)?([a-d])[\.:\)]?", raw_correct, re.IGNORECASE)
            if letter_match:
                idx = ord(letter_match.group(1).upper()) - ord("A")
                if 0 <= idx < len(options):
                    resolved_answer = options[idx]

        if not resolved_answer:
            raise QuizValidationError(
                f"correct_answer '{raw_correct}' does not match any of the 4 options: {options}"
            )

        # Validate source_page
        try:
            source_page = int(data["source_page"])
            if source_page < 1:
                raise ValueError()
        except (ValueError, TypeError):
            raise QuizValidationError(
                f"source_page must be a positive integer, got: {data['source_page']}"
            )

        if allowed_pages is not None and source_page not in allowed_pages:
            raise QuizValidationError(
                f"source_page {source_page} is not among the document's valid pages {sorted(list(allowed_pages))}."
            )

        # Validate source_filename
        if expected_filename:
            if source_filename.lower() != expected_filename.lower():
                logger.warning(
                    f"Question source_filename '{source_filename}' does not match expected '{expected_filename}'. Overriding."
                )
                source_filename = expected_filename

        # Validate Grounding against context chunks
        if context_chunks:
            is_grounded, reason = cls.validate_grounding(
                question=question,
                correct_answer=resolved_answer,
                context_chunks=context_chunks,
                source_page=source_page
            )
            if not is_grounded:
                raise QuizValidationError(f"Unsupported fact / citation: {reason}")

        return QuizQuestion(
            question_id=q_id,
            topic=topic,
            question=question,
            options=options,
            correct_answer=resolved_answer,
            explanation=explanation,
            source_filename=source_filename,
            source_page=source_page,
            difficulty=difficulty,
        )

    @classmethod
    def validate_quiz_questions(
        cls,
        raw_questions: List[Any],
        context_chunks: List[RetrievalResult],
        expected_filename: Optional[str] = None,
        allowed_pages: Optional[Set[int]] = None,
        min_required: int = 1
    ) -> List[QuizQuestion]:
        """
        Validates a list of questions, drops malformed, invalid, or duplicate questions,
        and enforces that at least `min_required` valid questions remain.

        Raises:
            QuizValidationError: If zero questions pass validation or if fewer than `min_required`.
        """
        if not raw_questions:
            raise QuizValidationError("Cannot validate empty questions list.")

        valid_questions: List[QuizQuestion] = []
        seen_question_signatures: Set[str] = set()
        errors: List[str] = []

        for i, raw_q in enumerate(raw_questions, start=1):
            try:
                question_obj = cls.validate_single_question(
                    raw_q=raw_q,
                    context_chunks=context_chunks,
                    expected_filename=expected_filename,
                    allowed_pages=allowed_pages
                )

                # Duplicate detection by normalized question string
                signature = re.sub(r"[^a-z0-9]", "", question_obj.question.lower())
                if signature in seen_question_signatures:
                    logger.warning(
                        f"Rejecting duplicate question: '{question_obj.question}'"
                    )
                    errors.append(f"Q{i}: Duplicate question detected.")
                    continue

                seen_question_signatures.add(signature)
                valid_questions.append(question_obj)

            except QuizValidationError as qe:
                logger.warning(f"Rejecting invalid question #{i}: {qe}")
                errors.append(f"Q{i}: {qe}")
            except Exception as e:
                logger.warning(f"Unexpected error validating question #{i}: {e}")
                errors.append(f"Q{i}: Unexpected validation error: {e}")

        if len(valid_questions) < min_required:
            error_details = "\n - ".join(errors)
            raise QuizValidationError(
                f"Generated quiz failed validation: only {len(valid_questions)} valid questions produced "
                f"(minimum {min_required} required).\nValidation rejection details:\n - {error_details}"
            )

        logger.info(
            f"Successfully validated {len(valid_questions)}/{len(raw_questions)} questions."
        )
        return valid_questions

