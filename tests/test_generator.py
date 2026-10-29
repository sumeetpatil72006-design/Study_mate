"""
Unit tests for Gemini Generator, prompt construction, citations, and insufficient context handling.
Verifies integration with gemini-3.8-flash, the official Google Gen AI SDK Interactions API,
models.generate_content fallback, and retry handling for 503 errors.
"""

from unittest.mock import MagicMock, patch
import os
import pytest

from src.exceptions import APIKeyMissingError, GeneratorError
from src.generator import GeminiGenerator, INSUFFICIENT_CONTEXT_MESSAGE
from src.models import RetrievalResult

def test_default_model_is_gemini_3_8_flash():
    """Verifies that generator defaults to gemini-3.8-flash."""
    generator = GeminiGenerator(api_key="mock_key")
    assert generator.model_name == "gemini-3.8-flash"

def test_insufficient_context_when_no_chunks_retrieved():
    """Verifies that generator returns the exact insufficient context message when no chunks are retrieved."""
    generator = GeminiGenerator(api_key="mock_key")
    answer, citations, is_insufficient = generator.generate_answer(
        query="What is the speed of light?",
        retrieved_chunks=[]
    )

    assert answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert len(citations) == 0
    assert is_insufficient is True

def test_citations_built_from_retrieved_chunks_via_interactions_api():
    """
    Verifies that citations correctly capture filename, page number, and excerpt,
    and that client.interactions.create is invoked with context and user query.
    """
    generator = GeminiGenerator(api_key="mock_key")

    chunks = [
        RetrievalResult(
            chunk_id="ch1",
            content="Qubits operate at 15 millikelvin inside dilution refrigerators.",
            source_filename="quantum_paper.pdf",
            page_number=2,
            score=0.92,
            metadata={}
        ),
        RetrievalResult(
            chunk_id="ch2",
            content="Superconducting quantum circuits require microwave pulses.",
            source_filename="hardware.pdf",
            page_number=4,
            score=0.85,
            metadata={}
        )
    ]

    # Mock the Gemini client Interactions API call
    mock_response = MagicMock()
    mock_response.output_text = (
        "Superconducting qubits are kept at 15 millikelvin [Source: quantum_paper.pdf, Page: 2] "
        "and controlled using microwave pulses [Source: hardware.pdf, Page: 4]."
    )
    mock_response.text = mock_response.output_text

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    generator._client = mock_client

    answer, citations, is_insufficient = generator.generate_answer(
        query="What temperature do qubits operate at?",
        retrieved_chunks=chunks
    )

    assert not is_insufficient
    assert len(citations) == 2
    assert citations[0].filename == "quantum_paper.pdf"
    assert citations[0].page_number == 2
    assert "15 millikelvin" in citations[0].excerpt
    assert citations[1].filename == "hardware.pdf"
    assert citations[1].page_number == 4

    # Verify that interactions.create was called with model, input containing query & context
    mock_client.interactions.create.assert_called_once()
    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    assert "What temperature do qubits operate at?" in call_kwargs["input"]
    assert "quantum_paper.pdf" in call_kwargs["input"]
    assert "15 millikelvin" in call_kwargs["input"]
    assert call_kwargs["system_instruction"] is not None
    assert call_kwargs["generation_config"]["temperature"] == 0.2

def test_selected_model_is_used_by_generator():
    """Verifies that changing model_name (e.g. to gemini-2.5-pro) is properly passed to the API."""
    generator = GeminiGenerator(api_key="mock_key", model_name="gemini-2.5-pro")

    chunks = [
        RetrievalResult(
            chunk_id="ch1",
            content="Context text.",
            source_filename="doc.pdf",
            page_number=1,
            score=0.9,
            metadata={}
        )
    ]

    mock_response = MagicMock()
    mock_response.output_text = "Answer from gemini-2.5-pro [Source: doc.pdf, Page: 1]."
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    generator._client = mock_client

    answer, citations, is_insufficient = generator.generate_answer(
        query="Test query?",
        retrieved_chunks=chunks
    )

    assert not is_insufficient
    assert "gemini-2.5-pro" in answer
    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-pro"

def test_llm_flags_insufficient_context_in_response():
    """Verifies that if LLM responds saying documents don't have enough info, is_insufficient is True."""
    generator = GeminiGenerator(api_key="mock_key")

    chunks = [
        RetrievalResult(
            chunk_id="ch1",
            content="This document talks about garden vegetables and tomatoes.",
            source_filename="gardening.pdf",
            page_number=1,
            score=0.45,
            metadata={}
        )
    ]

    mock_response = MagicMock()
    mock_response.output_text = "The provided documents do not contain enough information to answer this question."
    mock_response.text = mock_response.output_text

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    generator._client = mock_client

    answer, citations, is_insufficient = generator.generate_answer(
        query="What is the GDP of France?",
        retrieved_chunks=chunks
    )

    assert is_insufficient is True
    assert INSUFFICIENT_CONTEXT_MESSAGE in answer

def test_missing_api_key_raises_error(monkeypatch):
    """Verifies that APIKeyMissingError is raised when no API key is provided."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    generator = GeminiGenerator(api_key="")

    with pytest.raises(APIKeyMissingError) as exc_info:
        _ = generator.client
    assert "Gemini API key is missing" in str(exc_info.value)

def test_generator_error_handling_when_api_fails():
    """Verifies that API errors are wrapped in GeneratorError (which also satisfies RuntimeError)."""
    generator = GeminiGenerator(api_key="mock_key", max_retries=1)

    chunks = [
        RetrievalResult(
            chunk_id="ch1",
            content="Sample text.",
            source_filename="doc.pdf",
            page_number=1,
            score=0.9,
            metadata={}
        )
    ]

    mock_client = MagicMock()
    mock_client.interactions.create.side_effect = Exception("Google GenAI connection timeout")
    mock_client.models.generate_content.side_effect = Exception("Google GenAI connection timeout")
    generator._client = mock_client

    with pytest.raises(GeneratorError) as exc_info:
        generator.generate_answer(query="Question?", retrieved_chunks=chunks)

    assert "Error calling Gemini API" in str(exc_info.value)
    assert isinstance(exc_info.value, RuntimeError)

def test_fallback_when_interactions_not_available():
    """Verifies fallback to models.generate_content when client lacks interactions attribute."""
    generator = GeminiGenerator(api_key="mock_key")

    chunks = [
        RetrievalResult(
            chunk_id="ch1",
            content="Some fallback content.",
            source_filename="fallback.pdf",
            page_number=1,
            score=0.88,
            metadata={}
        )
    ]

    mock_response = MagicMock()
    mock_response.text = "Answer from models.generate_content [Source: fallback.pdf, Page: 1]."

    # Mock client with only models attribute (spec without interactions)
    mock_client = MagicMock(spec=["models"])
    mock_client.models.generate_content.return_value = mock_response
    generator._client = mock_client

    answer, citations, is_insufficient = generator.generate_answer(
        query="Fallback query?",
        retrieved_chunks=chunks
    )

    assert not is_insufficient
    assert "fallback.pdf" in answer
    mock_client.models.generate_content.assert_called_once()

def test_interactions_api_failure_falls_back_to_generate_content():
    """Verifies that when interactions.create fails, generator falls back to models.generate_content."""
    generator = GeminiGenerator(api_key="mock_key", max_retries=1)

    chunks = [
        RetrievalResult(
            chunk_id="ch1",
            content="Interactions fallback content.",
            source_filename="doc.pdf",
            page_number=1,
            score=0.88,
            metadata={}
        )
    ]

    mock_fallback_response = MagicMock()
    mock_fallback_response.text = "Success from fallback generateContent [Source: doc.pdf, Page: 1]."

    mock_client = MagicMock()
    mock_client.interactions.create.side_effect = Exception("Interactions endpoint not enabled")
    mock_client.models.generate_content.return_value = mock_fallback_response
    generator._client = mock_client

    answer, citations, is_insufficient = generator.generate_answer(
        query="Test query?",
        retrieved_chunks=chunks
    )

    assert not is_insufficient
    assert "Success from fallback generateContent" in answer
    mock_client.interactions.create.assert_called_once()
    mock_client.models.generate_content.assert_called_once()

def test_retry_on_503_service_unavailable_recovers():
    """
    Verifies that a temporary 503 error triggers retry with backoff,
    and succeeds once the server recovers.
    """
    generator = GeminiGenerator(
        api_key="mock_key",
        max_retries=3,
        initial_retry_delay=0.001,
        backoff_factor=1.5
    )

    chunks = [
        RetrievalResult(
            chunk_id="ch1",
            content="503 test content.",
            source_filename="doc.pdf",
            page_number=1,
            score=0.9,
            metadata={}
        )
    ]

    mock_success = MagicMock()
    mock_success.output_text = "Recovered answer [Source: doc.pdf, Page: 1]."

    mock_503_error = Exception("503 Service Unavailable: The model is temporarily overloaded.")

    mock_client = MagicMock()
    # First attempt raises 503, second attempt succeeds
    mock_client.interactions.create.side_effect = [mock_503_error, mock_success]
    generator._client = mock_client

    with patch("time.sleep") as mock_sleep:
        answer, citations, is_insufficient = generator.generate_answer(
            query="Test query?",
            retrieved_chunks=chunks
        )

        assert not is_insufficient
        assert "Recovered answer" in answer
        assert mock_client.interactions.create.call_count == 2
        mock_sleep.assert_called_once_with(0.001)

def test_retry_exhausted_on_repeated_503_errors():
    """
    Verifies that when 503 errors persist past max_retries, GeneratorError is raised.
    """
    generator = GeminiGenerator(
        api_key="mock_key",
        max_retries=3,
        initial_retry_delay=0.001,
        backoff_factor=2.0
    )

    chunks = [
        RetrievalResult(
            chunk_id="ch1",
            content="Persistent 503 content.",
            source_filename="doc.pdf",
            page_number=1,
            score=0.9,
            metadata={}
        )
    ]

    mock_client = MagicMock()
    mock_client.interactions.create.side_effect = Exception("503 Service Unavailable")
    generator._client = mock_client

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(GeneratorError) as exc_info:
            generator.generate_answer(query="Query?", retrieved_chunks=chunks)

        assert "503" in str(exc_info.value)
        assert mock_client.interactions.create.call_count == 3
        assert mock_sleep.call_count == 2  # Sleeps before attempt 2 and attempt 3

def test_non_transient_error_does_not_retry():
    """
    Verifies that permanent errors (e.g. 404 or invalid model) fail immediately without retrying.
    """
    generator = GeminiGenerator(
        api_key="mock_key",
        max_retries=3,
        initial_retry_delay=0.001
    )

    chunks = [
        RetrievalResult(
            chunk_id="ch1",
            content="Invalid model content.",
            source_filename="doc.pdf",
            page_number=1,
            score=0.9,
            metadata={}
        )
    ]

    mock_client = MagicMock()
    mock_client.interactions.create.side_effect = Exception("404 Not Found: models/unknown is not found")
    # And models.generate_content also raises 404
    mock_client.models.generate_content.side_effect = Exception("404 Not Found: models/unknown is not found")
    generator._client = mock_client

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(GeneratorError) as exc_info:
            generator.generate_answer(query="Query?", retrieved_chunks=chunks)

        assert "404 Not Found" in str(exc_info.value)
        # Should not sleep or retry transiently
        mock_sleep.assert_not_called()
