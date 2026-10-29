"""
Quiz generation engine coordinating scoped retrieval, structured Gemini prompting,
JSON extraction, and strict validation against source document context.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Set
from src.exceptions import (
    InsufficientContentForQuizError,
    QuizGenerationError,
    QuizValidationError,
)
from src.generator import GeminiGenerator
from src.logger import logger
from src.models import RetrievalResult
from src.quiz_models import Quiz, QuizQuestion, Topic
from src.quiz_validator import QuizValidator
from src.retriever import Retriever

QUIZ_GENERATOR_SYSTEM_PROMPT = """You are an expert assessment creator and teacher.
Generate multiple-choice study questions (MCQs) STRICTLY AND EXCLUSIVELY grounded in the provided document excerpts.

CRITICAL ASSESSMENT RULES:
1. Grounding: Only test facts and concepts that are explicitly stated in the context excerpts. Do NOT guess, extrapolate, or bring in external knowledge.
2. Structure: Every question must have EXACTLY 4 distinct, plausible options.
3. Answer: The correct_answer MUST match one of the 4 options verbatim.
4. Explanation: Provide a helpful explanation that directly cites or paraphrases the source text.
5. Citation: Provide the exact source_filename and source_page where the fact is stated.
6. Difficulty ({difficulty}):
   - Easy: Direct recall of explicitly defined terms or numbers.
   - Medium: Application, relationships, or differentiating concepts stated in the text.
   - Hard: Analysis of complex details or distinguishing subtle nuances stated in the text.
7. Format: Return ONLY a valid JSON array of objects with NO conversational text or codeblock wrappers.

JSON Object Schema:
{{
  "question_id": "q1",
  "topic": "{topic_name}",
  "question": "Clear question text ending with a question mark?",
  "options": [
    "Option A",
    "Option B",
    "Option C",
    "Option D"
  ],
  "correct_answer": "Option A",
  "explanation": "Detailed explanation grounded in the text...",
  "source_filename": "{source_filename}",
  "source_page": 1,
  "difficulty": "{difficulty}"
}}
"""

QUIZ_GENERATOR_USER_TEMPLATE = """Document Excerpts for Topic '{topic_name}':
{context}

Target Topic: {topic_name}
Target Difficulty: {difficulty}
Number of Questions Requested: {num_questions}

Generate {num_questions} high-quality, distinct multiple-choice questions grounded in the context above.
Return ONLY a valid JSON array of question objects."""


class QuizGenerator:
    """
    Coordinates scoped document retrieval, structured question generation with Gemini,
    and rigorous schema/fact validation.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: GeminiGenerator,
        validator: Optional[QuizValidator] = None,
        min_context_words: int = 25
    ):
        """
        Initialize QuizGenerator.

        Args:
            retriever: Initialized Retriever instance.
            generator: Initialized GeminiGenerator instance.
            validator: Optional QuizValidator instance.
            min_context_words: Minimum word count required in retrieved context.
        """
        self.retriever = retriever
        self.generator = generator
        self.validator = validator or QuizValidator()
        self.min_context_words = min_context_words

    def retrieve_topic_context(
        self,
        topic: Topic,
        top_k: int = 8
    ) -> List[RetrievalResult]:
        """
        Retrieves context chunks scoped strictly to the selected document and topic.

        Args:
            topic: Target Topic instance.
            top_k: Maximum number of chunks to retrieve.

        Returns:
            List of RetrievalResult objects isolated to the topic's source document.
        """
        # Formulate query from topic name and description
        search_query = f"{topic.name}: {topic.description}".strip()

        # Scoped metadata filter ensures document isolation
        where_filter = {"source": topic.source_filename}

        retrieved = self.retriever.retrieve(
            query=search_query,
            top_k=top_k,
            where=where_filter
        )

        # Fallback: if semantic search returns 0 chunks, fetch document's raw chunks
        if not retrieved:
            raw_chunks = self.retriever.vector_store.get_document_chunks(
                source_filename=topic.source_filename
            )
            # Filter by topic source_pages if specified
            if topic.source_pages:
                raw_chunks = [
                    c for c in raw_chunks
                    if int(c.get("metadata", {}).get("page", 1)) in topic.source_pages
                ]

            retrieved = [
                RetrievalResult(
                    chunk_id=c["chunk_id"],
                    content=c["content"],
                    source_filename=c.get("metadata", {}).get("source", topic.source_filename),
                    page_number=int(c.get("metadata", {}).get("page", 1)),
                    score=1.0,
                    metadata=c.get("metadata", {})
                )
                for c in raw_chunks[:top_k]
            ]

        return retrieved

    def generate_quiz(
        self,
        topic: Topic,
        num_questions: int = 5,
        difficulty: str = "Medium",
        question_type: str = "MCQ"
    ) -> Quiz:
        """
        Generates a validated quiz for the specified topic and document.

        Args:
            topic: The selected Topic object.
            num_questions: Desired number of questions (default 5).
            difficulty: "Easy", "Medium", or "Hard".
            question_type: Question type (default "MCQ").

        Returns:
            Validated Quiz object.

        Raises:
            InsufficientContentForQuizError: If the document lacks sufficient content.
            QuizGenerationError: If Gemini API fails or returns unparseable output.
            QuizValidationError: If generated questions fail schema or grounding validation.
        """
        difficulty = difficulty.capitalize()
        if difficulty not in ["Easy", "Medium", "Hard"]:
            difficulty = "Medium"

        num_questions = max(1, min(num_questions, 15))

        logger.info(
            f"Generating {num_questions} {difficulty} questions for topic '{topic.name}' "
            f"from '{topic.source_filename}'"
        )

        # Step 1: Retrieve context scoped to document
        retrieved_chunks = self.retrieve_topic_context(topic=topic, top_k=max(6, num_questions * 2))

        if not retrieved_chunks:
            raise InsufficientContentForQuizError(
                f"No relevant content could be retrieved from '{topic.source_filename}' "
                f"for topic '{topic.name}'."
            )

        # Step 2: Verify sufficient content length
        total_text = " ".join(c.content for c in retrieved_chunks)
        word_count = len(re.findall(r"\b\w+\b", total_text))
        if word_count < self.min_context_words:
            raise InsufficientContentForQuizError(
                f"The context for topic '{topic.name}' in '{topic.source_filename}' is too brief "
                f"({word_count} words; minimum {self.min_context_words} required) to generate a reliable quiz."
            )

        # Step 3: Format context for prompt
        formatted_context, has_context = self.retriever.format_context_for_prompt(retrieved_chunks)
        if not has_context:
            raise InsufficientContentForQuizError(
                f"Failed to format document context for topic '{topic.name}'."
            )

        # Step 4: Construct system and user prompts
        sys_prompt = QUIZ_GENERATOR_SYSTEM_PROMPT.format(
            difficulty=difficulty,
            topic_name=topic.name,
            source_filename=topic.source_filename
        )
        user_prompt = QUIZ_GENERATOR_USER_TEMPLATE.format(
            topic_name=topic.name,
            difficulty=difficulty,
            num_questions=num_questions,
            context=formatted_context
        )

        # Step 5: Call Gemini API
        try:
            response = self.generator._invoke_api_with_retry(
                user_content=user_prompt,
                system_instruction=sys_prompt
            )
        except Exception as e:
            logger.error(f"Gemini API invocation failed during quiz generation: {e}")
            raise QuizGenerationError(f"Failed to generate quiz with Gemini: {e}") from e

        # Step 6: Extract and parse JSON
        raw_text = ""
        if hasattr(response, "output_text") and isinstance(response.output_text, str):
            raw_text = response.output_text.strip()
        elif hasattr(response, "text") and isinstance(response.text, str):
            raw_text = response.text.strip()
        elif hasattr(response, "output_text") and response.output_text is not None:
            raw_text = str(response.output_text).strip()
        elif hasattr(response, "text") and response.text is not None:
            raw_text = str(response.text).strip()

        if not raw_text:
            raise QuizGenerationError("Gemini returned an empty response.")

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        start_idx = cleaned.find("[")
        end_idx = cleaned.rfind("]")
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx : end_idx + 1]

        try:
            parsed_json = json.loads(cleaned)
        except Exception as je:
            logger.error(f"Failed to parse JSON quiz output: {je}\nRaw output:\n{raw_text[:500]}")
            raise QuizGenerationError(
                f"Gemini returned invalid JSON for quiz questions: {je}"
            ) from je

        if not isinstance(parsed_json, list):
            if isinstance(parsed_json, dict) and "questions" in parsed_json:
                parsed_json = parsed_json["questions"]
            else:
                raise QuizGenerationError(
                    f"Expected a JSON array of questions, but received {type(parsed_json).__name__}."
                )

        # Step 7: Validate generated questions with strict rules
        allowed_pages: Set[int] = {c.page_number for c in retrieved_chunks}
        # Include any known pages from topic
        if topic.source_pages:
            allowed_pages.update(topic.source_pages)

        validated_questions = self.validator.validate_quiz_questions(
            raw_questions=parsed_json,
            context_chunks=retrieved_chunks,
            expected_filename=topic.source_filename,
            allowed_pages=allowed_pages,
            min_required=1
        )

        quiz_id = f"quiz_{uuid.uuid4().hex[:10]}"
        quiz = Quiz(
            quiz_id=quiz_id,
            title=f"{topic.name} Quiz",
            topic=topic.name,
            source_filename=topic.source_filename,
            difficulty=difficulty,
            questions=validated_questions[:num_questions]
        )

        logger.info(
            f"Successfully generated quiz '{quiz.quiz_id}' with {len(quiz.questions)} questions."
        )
        return quiz
