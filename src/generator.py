"""
Gemini API integration for RAG answer generation with citations and fallback handling.
"""

import os
import time
from typing import Any, List, Optional, Tuple
from google import genai
from google.genai import types

from src.exceptions import APIKeyMissingError, GeneratorError
from src.logger import logger
from src.models import RetrievalResult, SourceCitation

INSUFFICIENT_CONTEXT_MESSAGE = (
    "The provided documents do not contain enough information to answer this question."
)

SYSTEM_PROMPT = """You are a helpful, accurate, and objective AI Knowledge Assistant.
Your task is to answer the user's question STRICTLY based on the provided document excerpts below.

CRITICAL INSTRUCTIONS:
1. ONLY use information directly stated or clearly implied in the provided context excerpts.
2. If the excerpts DO NOT contain sufficient information to answer the question accurately, or if no context is provided, you MUST reply with exactly:
   "The provided documents do not contain enough information to answer this question."
   Do NOT extrapolate, guess, speculate, or rely on external general knowledge.
3. When answering, cite the sources in-line using the format: [Source: filename, Page: page_number].
4. Keep answers clear, professional, and well-structured.
"""

USER_PROMPT_TEMPLATE = """Document Excerpts:
{context}

User Question:
{query}

Answer:"""

class GeminiGenerator:
    """
    RAG generator powered by the Google GenAI SDK.
    Supports current Gemini models (e.g. gemini-3.8-flash, gemini-2.5-flash, gemini-2.5-pro).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
        temperature: float = 0.2,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        backoff_factor: float = 2.0
    ):
        """
        Initialize the Gemini generator.

        Args:
            api_key: Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY env vars.
            model_name: Gemini model name (default: gemini-3.8-flash).
            temperature: Low temperature for factual RAG responses.
            max_retries: Maximum attempts for transient 503 errors.
            initial_retry_delay: Initial sleep duration in seconds before retry.
            backoff_factor: Multiplier for exponential backoff between retries.
        """
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.backoff_factor = backoff_factor
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        """Returns or initializes the Google GenAI client."""
        if self._client is None:
            if not self.api_key:
                raise APIKeyMissingError(
                    "Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _is_transient_error(self, exc: Exception) -> bool:
        """
        Determines if an exception is a transient 503 Service Unavailable or temporary error.
        """
        # 1. Check direct error code attribute
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in (503, 504, 429):
            return True

        # 2. Check nested response status code
        resp = getattr(exc, "response", None)
        if resp is not None:
            resp_code = getattr(resp, "status_code", None)
            if resp_code in (503, 504, 429):
                return True

        # 3. Check error message string for transient indicators
        err_msg = str(exc).lower()
        transient_phrases = [
            "503",
            "service unavailable",
            "temporarily unavailable",
            "unavailable",
            "overloaded",
            "resource_exhausted",
        ]
        return any(phrase in err_msg for phrase in transient_phrases)

    def _invoke_api_with_retry(
        self,
        user_content: str,
        system_instruction: Optional[str] = None
    ) -> Any:
        """
        Invokes Gemini API with exponential backoff for temporary 503 errors.
        Prefers the Interactions API and falls back to models.generate_content if needed.
        """
        sys_instruction = system_instruction or SYSTEM_PROMPT
        last_error: Optional[Exception] = None
        delay = self.initial_retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                # 1. Prefer Interactions API (client.interactions.create)
                if hasattr(self.client, "interactions") and hasattr(self.client.interactions, "create"):
                    try:
                        logger.debug(
                            f"Invoking client.interactions.create (attempt {attempt}/{self.max_retries}) "
                            f"for model '{self.model_name}'"
                        )
                        return self.client.interactions.create(
                            model=self.model_name,
                            input=user_content,
                            system_instruction=sys_instruction,
                            generation_config={
                                "temperature": self.temperature,
                            },
                        )
                    except Exception as ie:
                        # If it's a transient 503/server error, let retry handler handle it
                        if self._is_transient_error(ie):
                            raise
                        # If Interactions API failed due to routing or unsupported method, fall back to generateContent
                        logger.warning(
                            f"Interactions API call failed for model '{self.model_name}': {ie}. "
                            "Falling back to models.generate_content."
                        )

                # 2. Call models.generate_content
                config = types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    temperature=self.temperature,
                )
                logger.debug(
                    f"Invoking client.models.generate_content (attempt {attempt}/{self.max_retries}) "
                    f"for model '{self.model_name}'"
                )
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_content,
                    config=config,
                )

            except (APIKeyMissingError, GeneratorError):
                raise
            except Exception as e:
                last_error = e
                if self._is_transient_error(e) and attempt < self.max_retries:
                    logger.warning(
                        f"Transient Gemini API error (attempt {attempt}/{self.max_retries}): {e}. "
                        f"Retrying with exponential backoff in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    logger.error(f"Gemini API generation failed (attempt {attempt}/{self.max_retries}): {e}")
                    raise GeneratorError(f"Error calling Gemini API: {e}") from e

        raise GeneratorError(f"Gemini API failed after {self.max_retries} attempts: {last_error}")

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        context_str: Optional[str] = None
    ) -> Tuple[str, List[SourceCitation], bool]:
        """
        Generates an answer based on retrieved context.

        Args:
            query: The user's question.
            retrieved_chunks: List of retrieved chunks.
            context_str: Optional pre-formatted context string.

        Returns:
            Tuple of (answer_text, citations_list, is_insufficient_context_bool).
        """
        # If no chunks were retrieved at all, return the insufficient context message immediately
        if not retrieved_chunks:
            logger.info("No retrieved chunks. Returning insufficient context response.")
            return INSUFFICIENT_CONTEXT_MESSAGE, [], True

        # Build citations list from the retrieved chunks
        citations: List[SourceCitation] = []
        for chunk in retrieved_chunks:
            # First 150 chars as brief excerpt
            excerpt = chunk.content[:150].strip().replace("\n", " ") + ("..." if len(chunk.content) > 150 else "")
            citations.append(
                SourceCitation(
                    filename=chunk.source_filename,
                    page_number=chunk.page_number,
                    excerpt=excerpt
                )
            )

        # Build context if not pre-provided
        if not context_str:
            blocks = []
            for i, chunk in enumerate(retrieved_chunks, start=1):
                blocks.append(
                    f"[Document: {chunk.source_filename} | Page: {chunk.page_number}]\n{chunk.content}"
                )
            context_str = "\n\n---\n\n".join(blocks)

        user_content = USER_PROMPT_TEMPLATE.format(context=context_str, query=query)

        logger.info(f"Generating answer using {self.model_name} for query: '{query}'")

        try:
            response = self._invoke_api_with_retry(user_content)

            # Extract generated response text
            raw_answer = ""
            if hasattr(response, "output_text") and isinstance(response.output_text, str):
                raw_answer = response.output_text.strip()
            elif hasattr(response, "text") and isinstance(response.text, str):
                raw_answer = response.text.strip()
            elif hasattr(response, "steps") and response.steps:
                for step in response.steps:
                    if getattr(step, "type", None) == "model_output":
                        content = getattr(step, "content", [])
                        if isinstance(content, list):
                            for part in content:
                                if getattr(part, "type", None) == "text":
                                    raw_answer += getattr(part, "text", "") or ""
                raw_answer = raw_answer.strip()
            elif hasattr(response, "output_text") and response.output_text is not None:
                raw_answer = str(response.output_text).strip()
            elif hasattr(response, "text") and response.text is not None:
                raw_answer = str(response.text).strip()

            if not raw_answer:
                return INSUFFICIENT_CONTEXT_MESSAGE, citations, True

            # Check if answer indicates insufficient context
            insufficient = (
                INSUFFICIENT_CONTEXT_MESSAGE.lower() in raw_answer.lower()
                or "documents do not contain enough information" in raw_answer.lower()
                or "does not contain enough information" in raw_answer.lower()
            )

            return raw_answer, citations, insufficient

        except (APIKeyMissingError, GeneratorError):
            raise
        except Exception as e:
            logger.error(f"Gemini API generation failed: {e}")
            raise GeneratorError(f"Error calling Gemini API: {e}") from e

