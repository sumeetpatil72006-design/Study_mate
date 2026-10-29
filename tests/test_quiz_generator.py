"""
Unit tests for QuizGenerator.
Verifies scoped document retrieval, Gemini prompting, structured JSON handling,
document isolation, and handling of insufficient content or malformed output.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.config import AppConfig
from src.exceptions import (
    InsufficientContentForQuizError,
    QuizGenerationError,
    QuizValidationError,
)
from src.quiz_generator import QuizGenerator
from src.quiz_models import Quiz, Topic
from src.rag_pipeline import RAGPipeline


@pytest.fixture
def test_pipeline(sample_pdf_path: Path, temp_chroma_dir: str, mock_embedding_service):
    """Initializes a pipeline with an indexed sample document and mock API key."""
    config = AppConfig(
        chroma_persist_directory=temp_chroma_dir,
        gemini_api_key="mock_key",
        gemini_model="gemini-3.8-flash"
    )
    pipeline = RAGPipeline(config=config)
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service
    pipeline.ingest_pdf(sample_pdf_path, filename="sample_knowledge.pdf")
    return pipeline


def test_quiz_generation_with_mocked_gemini(test_pipeline: RAGPipeline):
    """
    Verifies that QuizGenerator produces a validated Quiz object
    using mocked Gemini JSON responses.
    """
    topic = Topic(
        topic_id="topic_1",
        name="Quantum Overview",
        description="Principles of superposition, qubits, and Shor's algorithm.",
        source_filename="sample_knowledge.pdf",
        source_pages=[1]
    )

    mock_json_response = """[
        {
            "question_id": "q1",
            "topic": "Quantum Overview",
            "question": "What principle allows qubits to exist in linear combinations?",
            "options": [
                "Superposition",
                "Classical binary",
                "Cryogenic cooling",
                "Thermal dissipation"
            ],
            "correct_answer": "Superposition",
            "explanation": "Qubits utilize superposition to exist in linear combinations of both states.",
            "source_filename": "sample_knowledge.pdf",
            "source_page": 1,
            "difficulty": "Easy"
        },
        {
            "question_id": "q2",
            "topic": "Quantum Overview",
            "question": "Which algorithm demonstrates exponential speedup for qubits in the document?",
            "options": [
                "Shor's algorithm",
                "Binary search",
                "Quick sort",
                "Bellman-Ford"
            ],
            "correct_answer": "Shor's algorithm",
            "explanation": "Shor's algorithm unlocks exponential computational speedups.",
            "source_filename": "sample_knowledge.pdf",
            "source_page": 1,
            "difficulty": "Easy"
        }
    ]"""

    mock_response = MagicMock()
    mock_response.output_text = mock_json_response
    mock_response.text = mock_json_response

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    test_pipeline.generator._client = mock_client

    quiz = test_pipeline.generate_quiz(
        topic=topic,
        num_questions=2,
        difficulty="Easy"
    )

    assert isinstance(quiz, Quiz)
    assert quiz.topic == "Quantum Overview"
    assert quiz.source_filename == "sample_knowledge.pdf"
    assert quiz.difficulty == "Easy"
    assert len(quiz.questions) == 2
    assert quiz.questions[0].correct_answer == "Superposition"
    assert quiz.questions[1].correct_answer == "Shor's algorithm"


def test_document_isolation_during_retrieval(temp_chroma_dir: str, mock_embedding_service):
    """
    Verifies that context retrieval is strictly scoped to the selected document
    and never includes chunks from other indexed documents.
    """
    import fitz

    # Create two distinct documents
    doc_a_path = Path(temp_chroma_dir) / "doc_a.pdf"
    doc_b_path = Path(temp_chroma_dir) / "doc_b.pdf"

    pdf_a = fitz.open()
    p_a = pdf_a.new_page()
    p_a.insert_text((50, 72), "Astrophysics and black holes event horizon spacetime singularity.")
    pdf_a.save(str(doc_a_path))
    pdf_a.close()

    pdf_b = fitz.open()
    p_b = pdf_b.new_page()
    p_b.insert_text((50, 72), "Molecular biology DNA replication enzymes genetics chromosomes.")
    pdf_b.save(str(doc_b_path))
    pdf_b.close()

    config = AppConfig(chroma_persist_directory=temp_chroma_dir)
    pipeline = RAGPipeline(config=config)
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service

    pipeline.ingest_pdf(doc_a_path, filename="doc_a.pdf")
    pipeline.ingest_pdf(doc_b_path, filename="doc_b.pdf")

    topic_a = Topic(
        topic_id="top_a",
        name="Black Holes",
        description="Event horizon and singularity in astrophysics.",
        source_filename="doc_a.pdf",
        source_pages=[1]
    )

    retrieved_chunks = pipeline.quiz_generator.retrieve_topic_context(topic=topic_a)

    assert len(retrieved_chunks) > 0
    # Every chunk must belong exclusively to doc_a.pdf
    for chunk in retrieved_chunks:
        assert chunk.source_filename == "doc_a.pdf"
        assert "biology" not in chunk.content.lower()
        assert "dna" not in chunk.content.lower()


def test_insufficient_content_raises_error_for_quiz(pdf_with_short_text: Path, temp_chroma_dir: str, mock_embedding_service):
    """
    Verifies that attempting to generate a quiz on notes lacking sufficient content
    raises InsufficientContentForQuizError.
    """
    config = AppConfig(chroma_persist_directory=temp_chroma_dir, gemini_api_key="mock_key")
    pipeline = RAGPipeline(config=config)
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service
    pipeline.ingest_pdf(pdf_with_short_text, filename="short_text.pdf")

    topic = Topic(
        topic_id="short_topic",
        name="Candidate Elimination",
        description="Version space",
        source_filename="short_text.pdf",
        source_pages=[1]
    )

    with pytest.raises(InsufficientContentForQuizError):
        pipeline.generate_quiz(topic=topic, num_questions=3)


def test_malformed_gemini_json_raises_generation_error(test_pipeline: RAGPipeline):
    """Verifies that invalid or non-JSON output from Gemini raises QuizGenerationError."""
    topic = Topic(
        topic_id="top_1",
        name="Quantum Overview",
        description="General overview.",
        source_filename="sample_knowledge.pdf",
        source_pages=[1]
    )

    # Gemini returns prose instead of JSON
    mock_response = MagicMock()
    mock_response.output_text = "Here are some questions: 1. What is quantum? Answer: yes."
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = mock_response
    test_pipeline.generator._client = mock_client

    with pytest.raises(QuizGenerationError):
        test_pipeline.generate_quiz(topic=topic, num_questions=2)
