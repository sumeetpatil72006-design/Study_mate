"""
Integration tests for the end-to-end RAG pipeline.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.config import AppConfig
from src.document_loader import DuplicateDocumentError, NoTextExtractedError
from src.models import RAGResponse
from src.rag_pipeline import RAGPipeline

def test_rag_pipeline_end_to_end(
    sample_pdf_path: Path,
    temp_chroma_dir: str,
    mock_embedding_service
):
    """
    Verifies full RAG flow: ingest PDF -> index in ChromaDB -> query -> citations -> database summary.
    """
    config = AppConfig(
        gemini_api_key="mock_test_key",
        gemini_model="gemini-3.8-flash",
        chroma_persist_directory=temp_chroma_dir,
        chunk_size=300,
        chunk_overlap=50,
        top_k=2
    )

    pipeline = RAGPipeline(config=config)
    # Replace embedding service with fast mock to keep test quick and offline
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service
    pipeline.chunker.chunk_size = 300
    pipeline.chunker.chunk_overlap = 50

    # 1. Querying empty database
    empty_res = pipeline.ask("What is a qubit?")
    assert empty_res.is_insufficient_context is True
    assert "upload a PDF" in empty_res.answer
    assert empty_res.model_name == "gemini-3.8-flash"

    # 2. Ingest valid PDF
    stats = pipeline.ingest_pdf(sample_pdf_path)
    assert stats["total_pages"] == 3
    assert stats["total_chunks"] >= 3
    assert stats["source_filename"] == sample_pdf_path.name

    # 3. Ingesting duplicate should fail
    with pytest.raises(DuplicateDocumentError):
        pipeline.ingest_pdf(sample_pdf_path, check_duplicate=True)

    # 4. Check database summary
    summary = pipeline.get_database_summary()
    assert summary["total_documents"] == 1
    assert summary["total_chunks"] >= 3

    # 5. Mock generator answer
    mock_llm_response = MagicMock()
    mock_llm_response.output_text = (
        "Qubits leverage quantum superposition and entanglement to represent multiple states "
        "[Source: sample_knowledge.pdf, Page: 1]."
    )
    mock_llm_response.text = mock_llm_response.output_text
    pipeline.generator._client = MagicMock()
    pipeline.generator._client.interactions.create.return_value = mock_llm_response

    # 6. Execute query
    response: RAGResponse = pipeline.ask("What makes qubits different from classical bits?")

    assert isinstance(response, RAGResponse)
    assert not response.is_insufficient_context
    assert len(response.citations) > 0
    assert response.citations[0].filename == sample_pdf_path.name
    assert "sample_knowledge.pdf" in response.answer
    assert response.model_name == "gemini-3.8-flash"

    # 7. Delete document
    file_hash = stats["file_hash"]
    deleted = pipeline.delete_document(file_hash)
    assert deleted > 0
    assert pipeline.get_database_summary()["total_documents"] == 0

def test_rag_pipeline_non_text_pdf_rejected(
    scanned_pdf_path: Path,
    temp_chroma_dir: str,
    mock_embedding_service
):
    """
    Verifies that non-text/scanned PDFs are rejected by the pipeline with NoTextExtractedError.
    """
    config = AppConfig(
        gemini_api_key="mock_key",
        chroma_persist_directory=temp_chroma_dir
    )
    pipeline = RAGPipeline(config=config)
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service

    with pytest.raises(NoTextExtractedError) as exc_info:
        pipeline.ingest_pdf(scanned_pdf_path)
    assert "no selectable text found" in str(exc_info.value).lower()
    assert pipeline.get_database_summary()["total_documents"] == 0

def test_rag_pipeline_whitespace_pdf_rejected(
    whitespace_only_pdf: Path,
    temp_chroma_dir: str,
    mock_embedding_service
):
    """
    Verifies that a 4-page whitespace-only PDF is rejected by the pipeline,
    preventing 0-chunk vector insertions.
    """
    config = AppConfig(
        gemini_api_key="mock_key",
        chroma_persist_directory=temp_chroma_dir
    )
    pipeline = RAGPipeline(config=config)
    pipeline.embedding_service = mock_embedding_service
    pipeline.retriever.embedding_service = mock_embedding_service

    with pytest.raises(NoTextExtractedError) as exc_info:
        pipeline.ingest_pdf(whitespace_only_pdf)
    assert "no selectable text found" in str(exc_info.value).lower()
    assert pipeline.get_database_summary()["total_documents"] == 0
