"""
Unit tests for text chunking, overlap, and metadata preservation.
"""

import pytest
from src.exceptions import NoTextExtractedError
from src.models import DocumentPage
from src.text_splitter import DocumentChunker

def test_chunking_preserves_metadata():
    """Verifies that chunking retains page number, filename, and hash metadata."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    pages = [
        DocumentPage(
            source_filename="quantum_guide.pdf",
            page_number=1,
            total_pages=2,
            content="Qubits leverage superposition. Quantum entanglement enables high speed parallel operations.",
            file_hash="abc123456789",
            metadata={"source": "quantum_guide.pdf", "page": 1}
        ),
        DocumentPage(
            source_filename="quantum_guide.pdf",
            page_number=2,
            total_pages=2,
            content="Cryogenic dilution refrigerators cool superconducting chips to near absolute zero.",
            file_hash="abc123456789",
            metadata={"source": "quantum_guide.pdf", "page": 2}
        )
    ]

    chunks = chunker.split_pages(pages)
    assert len(chunks) >= 2

    for c in chunks:
        assert c.source_filename == "quantum_guide.pdf"
        assert c.file_hash == "abc123456789"
        assert c.page_number in [1, 2]
        assert "source" in c.metadata
        assert "page" in c.metadata
        assert c.metadata["page"] == c.page_number
        assert len(c.content) <= 150  # within chunk bounds

def test_chunk_size_and_overlap_parameters():
    """Verifies that invalid chunking parameters raise appropriate ValueErrors."""
    # Chunk size <= 0
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=0, chunk_overlap=0)

    # Chunk overlap >= chunk size
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=100)

    # Negative chunk overlap
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=-10)

def test_empty_page_handling():
    """Verifies that empty list returns empty chunks and all-blank pages raise NoTextExtractedError."""
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)

    # Empty list returns empty list
    assert chunker.split_pages([]) == []

    # Whitespace-only page raises NoTextExtractedError
    pages = [
        DocumentPage(
            source_filename="blank.pdf",
            page_number=1,
            total_pages=1,
            content="   \n\n   ",
            file_hash="blankhash"
        )
    ]

    with pytest.raises(NoTextExtractedError) as exc_info:
        chunker.split_pages(pages)
    assert "empty or contain only whitespace" in str(exc_info.value)

def test_interspersed_empty_pages_preserve_numbering():
    """Verifies that an empty page between text pages does not shift page numbering."""
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)

    pages = [
        DocumentPage(
            source_filename="manual.pdf",
            page_number=1,
            total_pages=3,
            content="Chapter 1 introduces the core algorithms and basic definitions.",
            file_hash="hash123"
        ),
        DocumentPage(
            source_filename="manual.pdf",
            page_number=2,
            total_pages=3,
            content="",
            file_hash="hash123",
            metadata={"is_empty": True}
        ),
        DocumentPage(
            source_filename="manual.pdf",
            page_number=3,
            total_pages=3,
            content="Chapter 2 covers experimental results and validation metrics.",
            file_hash="hash123"
        )
    ]

    chunks = chunker.split_pages(pages)
    assert len(chunks) == 2

    # Chunk 0 corresponds to Page 1
    assert chunks[0].page_number == 1
    assert "Chapter 1" in chunks[0].content
    assert chunks[0].metadata["page"] == 1

    # Chunk 1 corresponds to Page 3 (Page 2 was empty and skipped without shifting Page 3)
    assert chunks[1].page_number == 3
    assert "Chapter 2" in chunks[1].content
    assert chunks[1].metadata["page"] == 3

def test_short_text_chunks_created():
    """Verifies that pages with short text are properly chunked and not dropped."""
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=150)
    pages = [
        DocumentPage(
            source_filename="brief.pdf",
            page_number=1,
            total_pages=1,
            content="Candidate Elimination Algorithm",
            file_hash="hashshort"
        )
    ]
    chunks = chunker.split_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].content == "Candidate Elimination Algorithm"
    assert chunks[0].page_number == 1
    assert chunks[0].source_filename == "brief.pdf"
    assert chunks[0].metadata["source"] == "brief.pdf"
    assert chunks[0].metadata["page"] == 1

