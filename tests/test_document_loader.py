"""
Unit tests for text-based PDF document extraction, metadata preservation,
whitespace normalization, empty page handling, and error validation.
"""

import io
from pathlib import Path
import pytest

from src.document_loader import (
    DocumentLoaderError,
    DuplicateDocumentError,
    EncryptedPDFError,
    InvalidPDFError,
    NoTextExtractedError,
    PDFDocumentLoader,
    normalize_text,
    is_meaningful_text,
)
from src.models import DocumentPage

def test_load_valid_pdf_file_path(sample_pdf_path: Path):
    """Verifies that a valid text PDF file path loads all pages with accurate metadata."""
    loader = PDFDocumentLoader()
    pages, file_hash = loader.load_pdf(sample_pdf_path)

    assert len(pages) == 3
    assert len(file_hash) == 64  # SHA-256 length

    # Page 1 checks
    p1: DocumentPage = pages[0]
    assert p1.page_number == 1
    assert p1.total_pages == 3
    assert p1.source_filename == sample_pdf_path.name
    assert "Quantum Computing Overview" in p1.content
    assert p1.file_hash == file_hash
    assert p1.metadata["is_empty"] is False

    # Page 2 checks
    p2: DocumentPage = pages[1]
    assert p2.page_number == 2
    assert p2.total_pages == 3
    assert "Superconducting qubits" in p2.content

    # Page 3 checks
    p3: DocumentPage = pages[2]
    assert p3.page_number == 3
    assert p3.total_pages == 3
    assert "Quantum Error Correction" in p3.content

def test_load_pdf_from_bytes_and_bytesio(sample_pdf_bytes: bytes):
    """Verifies that raw bytes and BytesIO buffers can be extracted properly."""
    loader = PDFDocumentLoader()

    # Test raw bytes
    pages_bytes, hash_bytes = loader.load_pdf(sample_pdf_bytes, filename="memory_doc.pdf")
    assert len(pages_bytes) == 3
    assert pages_bytes[0].source_filename == "memory_doc.pdf"

    # Test BytesIO
    stream = io.BytesIO(sample_pdf_bytes)
    pages_stream, hash_stream = loader.load_pdf(stream, filename="stream_doc.pdf", check_duplicate=False)
    assert len(pages_stream) == 3
    assert hash_bytes == hash_stream

def test_whitespace_normalization(pdf_with_code_and_table: Path):
    """
    Verifies that whitespace normalization collapses excessive empty lines
    while preserving code block indentation and table formatting.
    """
    loader = PDFDocumentLoader()
    pages, _ = loader.load_pdf(pdf_with_code_and_table)

    assert len(pages) == 1
    content = pages[0].content

    # Indentation in code is preserved
    assert "    dw = weights * 0.01" in content
    assert "    return dw, db" in content

    # Table columns alignment is preserved
    assert "Accuracy" in content and "98.5%" in content

    # Excessive blank lines (4+) are collapsed to at most 2 newlines
    assert "\n\n\n" not in content

def test_handle_empty_pages_gracefully(pdf_with_empty_page: Path):
    """
    Verifies that a document containing a blank page does not fail,
    preserves correct page numbers (1, 2, 3), and flags the blank page accurately.
    """
    loader = PDFDocumentLoader()
    pages, file_hash = loader.load_pdf(pdf_with_empty_page)

    assert len(pages) == 3

    # Page 1 has text
    assert pages[0].page_number == 1
    assert pages[0].content == "Content on page one."
    assert pages[0].metadata["is_empty"] is False

    # Page 2 is empty
    assert pages[1].page_number == 2
    assert pages[1].content == ""
    assert pages[1].metadata["is_empty"] is True

    # Page 3 has text
    assert pages[2].page_number == 3
    assert pages[2].content == "Content on page three."
    assert pages[2].metadata["is_empty"] is False

def test_completely_empty_pdf_raises_error(completely_empty_pdf: Path):
    """Verifies that a PDF where all pages are completely blank raises NoTextExtractedError."""
    loader = PDFDocumentLoader()
    with pytest.raises(NoTextExtractedError) as exc_info:
        loader.load_pdf(completely_empty_pdf)
    assert "no selectable text found" in str(exc_info.value).lower()

def test_whitespace_only_4pages_pdf_rejected(whitespace_only_pdf: Path):
    """
    Verifies that a multi-page PDF containing only whitespace/tabs is rejected
    with a clear user-friendly NoTextExtractedError, preventing 0-chunk bugs.
    """
    loader = PDFDocumentLoader()
    with pytest.raises(NoTextExtractedError) as exc_info:
        loader.load_pdf(whitespace_only_pdf)
    assert "no selectable text found" in str(exc_info.value).lower()

def test_scanned_image_pdf_rejected_with_friendly_error(scanned_pdf_path: Path):
    """
    Verifies that an image-only/scanned PDF raises NoTextExtractedError
    explaining that only digitally generated PDFs with selectable text are supported.
    """
    loader = PDFDocumentLoader()
    with pytest.raises(NoTextExtractedError) as exc_info:
        loader.load_pdf(scanned_pdf_path)
    
    error_msg = str(exc_info.value).lower()
    assert "selectable text" in error_msg
    assert "scanned or image-only" in error_msg

def test_encrypted_password_protected_pdf(encrypted_pdf_path: Path):
    """Verifies that password-protected PDFs raise EncryptedPDFError."""
    loader = PDFDocumentLoader()
    with pytest.raises(EncryptedPDFError) as exc_info:
        loader.load_pdf(encrypted_pdf_path)
    assert "password-protected" in str(exc_info.value).lower() or "encrypted" in str(exc_info.value).lower()

def test_duplicate_document_handling(sample_pdf_path: Path):
    """Verifies that loading the same document twice triggers DuplicateDocumentError."""
    loader = PDFDocumentLoader()
    pages, file_hash = loader.load_pdf(sample_pdf_path)
    loader.mark_as_processed(file_hash)

    assert loader.is_duplicate(file_hash) is True

    # Loading again with check_duplicate=True should raise DuplicateDocumentError
    with pytest.raises(DuplicateDocumentError) as exc_info:
        loader.load_pdf(sample_pdf_path, check_duplicate=True)
    assert "already been loaded" in str(exc_info.value)

    # Loading with check_duplicate=False should succeed
    pages_bypass, _ = loader.load_pdf(sample_pdf_path, check_duplicate=False)
    assert len(pages_bypass) == 3

def test_invalid_and_corrupt_pdf(invalid_pdf_path: Path):
    """Verifies that corrupt non-PDF files raise InvalidPDFError."""
    loader = PDFDocumentLoader()
    with pytest.raises(InvalidPDFError) as exc_info:
        loader.load_pdf(invalid_pdf_path)
    assert "not a valid PDF" in str(exc_info.value) or "Failed to parse" in str(exc_info.value)

def test_empty_file_handling(tmp_path: Path):
    """Verifies that a 0-byte file raises an InvalidPDFError."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    loader = PDFDocumentLoader()
    with pytest.raises(InvalidPDFError) as exc_info:
        loader.load_pdf(empty_file)
    assert "empty" in str(exc_info.value).lower()

def test_nonexistent_file():
    """Verifies that non-existent paths raise FileNotFoundError."""
    loader = PDFDocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_pdf("this_file_does_not_exist_at_all.pdf")
