"""
Unit tests for topic discovery and extraction from document chunks.
Verifies grounding in document content, page-level citation links,
insufficient content handling, heuristic extraction, and mocked LLM discovery.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.exceptions import InsufficientContentForQuizError
from src.quiz_models import Topic
from src.rag_pipeline import RAGPipeline
from src.topic_extractor import TopicExtractor


def test_heuristic_topic_extraction_grounded_in_pdf(sample_pdf_path: Path, temp_chroma_dir: str, mock_embedding_service):
    """
    Verifies that topics extracted from an indexed PDF are grounded in document content
    and linked to valid page numbers and filename.
    """
    from src.config import AppConfig

    config = AppConfig(chroma_persist_directory=temp_chroma_dir)
    pipeline = RAGPipeline(config=config)
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service

    # Ingest 3-page quantum sample PDF
    pipeline.ingest_pdf(sample_pdf_path, filename="sample_knowledge.pdf")

    topics = pipeline.discover_topics(source_filename="sample_knowledge.pdf")

    assert len(topics) >= 1
    for topic in topics:
        assert isinstance(topic, Topic)
        assert topic.source_filename == "sample_knowledge.pdf"
        assert len(topic.source_pages) > 0
        # All source pages must be within the 3-page sample document
        for p in topic.source_pages:
            assert p in [1, 2, 3]
        assert len(topic.name) > 0
        assert len(topic.description) > 0


def test_topic_extraction_detects_headings(sample_pdf_path: Path, temp_chroma_dir: str, mock_embedding_service):
    """
    Verifies that heuristic discovery detects explicit headings from the sample document:
    e.g. 'Quantum Computing Overview', 'Hardware and Qubit Technologies', 'Quantum Error Correction'.
    """
    from src.config import AppConfig

    config = AppConfig(chroma_persist_directory=temp_chroma_dir)
    pipeline = RAGPipeline(config=config)
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service
    pipeline.ingest_pdf(sample_pdf_path, filename="sample_knowledge.pdf")

    topics = pipeline.discover_topics(source_filename="sample_knowledge.pdf")
    topic_names = [t.name.lower() for t in topics]

    # At least one of the prominent headings should be captured
    found_any = any(
        "quantum" in name or "hardware" in name or "qubit" in name or "error" in name
        for name in topic_names
    )
    assert found_any, f"Expected headings not found in: {topic_names}"


def test_insufficient_content_raises_error(pdf_with_short_text: Path, temp_chroma_dir: str, mock_embedding_service):
    """
    Verifies that a document with minimal textual content raises InsufficientContentForQuizError.
    """
    from src.config import AppConfig

    config = AppConfig(chroma_persist_directory=temp_chroma_dir)
    pipeline = RAGPipeline(config=config)
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service
    pipeline.ingest_pdf(pdf_with_short_text, filename="short_text.pdf")

    # Document has only ~8 words total, well below min threshold
    with pytest.raises(InsufficientContentForQuizError) as exc_info:
        pipeline.discover_topics(source_filename="short_text.pdf")

    assert "below the minimum threshold" in str(exc_info.value) or "insufficient" in str(exc_info.value).lower()


def test_nonexistent_document_raises_insufficient_content(temp_chroma_dir: str):
    """Verifies that requesting topics for an unindexed document raises InsufficientContentForQuizError."""
    from src.config import AppConfig

    config = AppConfig(chroma_persist_directory=temp_chroma_dir)
    pipeline = RAGPipeline(config=config)

    with pytest.raises(InsufficientContentForQuizError):
        pipeline.discover_topics(source_filename="ghost_document.pdf")


def test_llm_topic_discovery_with_mocked_gemini(sample_pdf_path: Path, temp_chroma_dir: str, mock_embedding_service):
    """
    Verifies that LLM-based topic discovery parses structured JSON responses from Gemini,
    validates page numbers against the document, and constructs Topic objects.
    """
    from src.config import AppConfig

    config = AppConfig(chroma_persist_directory=temp_chroma_dir, gemini_api_key="mock_key")
    pipeline = RAGPipeline(config=config)
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service
    pipeline.ingest_pdf(sample_pdf_path, filename="sample_knowledge.pdf")

    mock_llm_json = """[
        {
            "name": "Superposition and Qubits",
            "description": "Exploration of quantum linear combinations and Shor's algorithm.",
            "source_pages": [1]
        },
        {
            "name": "Cryogenic Hardware",
            "description": "Superconducting qubits operating at 15 millikelvin inside dilution refrigerators.",
            "source_pages": [2]
        }
    ]"""

    mock_response = MagicMock()
    mock_response.output_text = mock_llm_json
    mock_response.text = mock_llm_json

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    pipeline.generator._client = mock_client

    topics = pipeline.discover_topics(source_filename="sample_knowledge.pdf")

    assert len(topics) == 2
    assert topics[0].name == "Superposition and Qubits"
    assert topics[0].source_pages == [1]
    assert topics[0].source_filename == "sample_knowledge.pdf"
    assert topics[1].name == "Cryogenic Hardware"
    assert topics[1].source_pages == [2]


def test_llm_topic_discovery_filters_hallucinated_pages(sample_pdf_path: Path, temp_chroma_dir: str, mock_embedding_service):
    """
    Verifies that if Gemini returns pages outside the actual document (e.g. page 99 for a 3-page doc),
    they are filtered out and assigned to valid pages.
    """
    from src.config import AppConfig

    config = AppConfig(chroma_persist_directory=temp_chroma_dir, gemini_api_key="mock_key")
    pipeline = RAGPipeline(config=config)
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service
    pipeline.ingest_pdf(sample_pdf_path, filename="sample_knowledge.pdf")

    # LLM hallucinates page 99
    mock_llm_json = """[
        {
            "name": "Hallucinated Topic",
            "description": "Discussion referencing non-existent page 99.",
            "source_pages": [99]
        }
    ]"""

    mock_response = MagicMock()
    mock_response.output_text = mock_llm_json
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    pipeline.generator._client = mock_client

    topics = pipeline.discover_topics(source_filename="sample_knowledge.pdf")

    assert len(topics) == 1
    # Page 99 must be stripped and replaced with a valid document page
    assert 99 not in topics[0].source_pages
    assert all(p in [1, 2, 3] for p in topics[0].source_pages)
