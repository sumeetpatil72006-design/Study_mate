"""
Pytest fixtures for testing RAG pipeline components.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator
import pymupdf as fitz
import pytest

from src.config import AppConfig
from src.embeddings import EmbeddingService
from src.models import DocumentChunk, DocumentPage
from src.vector_store import ChromaVectorStore

@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """
    Creates a valid 3-page PDF with distinct text on each page for testing.
    """
    pdf_path = tmp_path / "sample_knowledge.pdf"
    doc = fitz.open()

    # Page 1: Overview of Quantum Computing
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Quantum Computing Overview:\n"
        "Quantum computing harnesses qubits which utilize superposition and entanglement.\n"
        "Unlike classical bits which are 0 or 1, qubits can exist in linear combinations of both states.\n"
        "This unlocks exponential computational speedups for specific algorithms like Shor's algorithm."
    )

    # Page 2: Qubit Technologies and Hardware
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "Hardware and Qubit Technologies:\n"
        "Leading qubit implementations include superconducting circuits, trapped ions, and photonic qubits.\n"
        "Superconducting qubits operate at cryogenic temperatures near absolute zero, around 15 millikelvin.\n"
        "Dilution refrigerators are critical infrastructure for maintaining quantum coherence."
    )

    # Page 3: Quantum Error Correction
    page3 = doc.new_page()
    page3.insert_text(
        (50, 72),
        "Quantum Error Correction (QEC):\n"
        "QEC is necessary because quantum states are fragile to environmental decoherence.\n"
        "Surface codes require hundreds of physical qubits to represent a single fault-tolerant logical qubit.\n"
        "Fault tolerance is widely considered the threshold for practical quantum supremacy."
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

@pytest.fixture
def sample_pdf_bytes(sample_pdf_path: Path) -> bytes:
    """Returns raw bytes of sample PDF."""
    return sample_pdf_path.read_bytes()

@pytest.fixture
def pdf_with_empty_page(tmp_path: Path) -> Path:
    """Creates a 3-page PDF where page 1 and 3 have text, but page 2 is empty."""
    pdf_path = tmp_path / "pdf_with_empty_page.pdf"
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 72), "Content on page one.")

    p2 = doc.new_page()  # Intentionally blank

    p3 = doc.new_page()
    p3.insert_text((50, 72), "Content on page three.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

@pytest.fixture
def completely_empty_pdf(tmp_path: Path) -> Path:
    """Creates a PDF where all pages are completely blank."""
    pdf_path = tmp_path / "completely_empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

@pytest.fixture
def whitespace_only_pdf(tmp_path: Path) -> Path:
    """
    Creates a 4-page PDF containing only whitespace, blank lines, and tabs.
    Simulates documents that produce 0 chunks if not caught properly.
    """
    pdf_path = tmp_path / "whitespace_only_4pages.pdf"
    doc = fitz.open()
    for _ in range(4):
        page = doc.new_page()
        page.insert_text((50, 72), "   \n\t   \n\n   ")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

@pytest.fixture
def pdf_with_code_and_table(tmp_path: Path) -> Path:
    """Creates a PDF containing Python code with indentation and a columnar table."""
    pdf_path = tmp_path / "code_and_table.pdf"
    doc = fitz.open()
    page = doc.new_page()
    content = (
        "def compute_gradients(weights, bias):\n"
        "    dw = weights * 0.01\n"
        "    db = bias * 0.01\n"
        "    return dw, db\n\n\n\n"
        "Metric      Train       Test\n"
        "Accuracy    98.5%       96.2%\n"
        "Loss        0.042       0.081\n"
    )
    page.insert_text((50, 72), content)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

@pytest.fixture
def pdf_with_short_text(tmp_path: Path) -> Path:
    """Creates a PDF where pages contain short single-line phrases."""
    pdf_path = tmp_path / "short_text.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "Candidate Elimination Algorithm")
    p2 = doc.new_page()
    p2.insert_text((50, 72), "Version Space Representation")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

@pytest.fixture
def scanned_pdf_path(tmp_path: Path) -> Path:
    """
    Creates a scanned image PDF (no embedded text layer) containing
    text rendered as an image bitmap, simulating 'ML lab prog 3.pdf'.
    """
    pdf_path = tmp_path / "scanned_lab_prog.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Render text onto an offscreen canvas and convert to raster image
    temp_doc = fitz.open()
    temp_page = temp_doc.new_page(width=595, height=842)
    temp_page.insert_text(
        (72, 100),
        "Machine Learning Lab Program 3: Candidate Elimination Algorithm",
        fontsize=16
    )
    temp_page.insert_text(
        (72, 140),
        "Initialize G to the set of maximally general hypotheses in H.",
        fontsize=12
    )
    temp_page.insert_text(
        (72, 160),
        "Initialize S to the set of maximally specific hypotheses in H.",
        fontsize=12
    )
    pix = temp_page.get_pixmap(dpi=150)
    temp_doc.close()

    # Insert image bitmap covering the page (zero text objects)
    page.insert_image(page.rect, pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

@pytest.fixture
def encrypted_pdf_path(tmp_path: Path) -> Path:
    """Creates a password-protected encrypted PDF."""
    pdf_path = tmp_path / "encrypted_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Top secret content.")
    # Encrypt with owner and user passwords
    perm = int(
        fitz.PDF_PERM_ACCESSIBILITY
        | fitz.PDF_PERM_PRINT
        | fitz.PDF_PERM_COPY
        | fitz.PDF_PERM_ANNOTATE
    )
    doc.save(
        str(pdf_path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner_pass",
        user_pw="secret123",
        permissions=perm
    )
    doc.close()
    return pdf_path

@pytest.fixture
def invalid_pdf_path(tmp_path: Path) -> Path:
    """Creates a corrupted non-PDF file for error testing."""
    invalid_path = tmp_path / "corrupt.pdf"
    invalid_path.write_bytes(b"This is completely invalid binary data not starting with PDF header.")
    return invalid_path

@pytest.fixture
def temp_chroma_dir() -> Generator[str, None, None]:
    """Provides a fresh temporary directory for ChromaDB tests and cleans it up."""
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

class MockEmbeddingService:
    """Fast deterministic mock for embeddings to accelerate vector tests."""
    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    @staticmethod
    def name() -> str:
        return "mock_embedding"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self):
        return ["cosine", "l2", "ip"]

    def is_legacy(self) -> bool:
        return False

    def get_config(self) -> dict:
        return {"dimension": self.dimension}

    @staticmethod
    def build_from_config(config: dict) -> "MockEmbeddingService":
        return MockEmbeddingService(dimension=config.get("dimension", 16))

    def embed_query(self, text: str):
        # Deterministic pseudo-embedding based on string hash
        val = sum(ord(c) for c in text) % 100 / 100.0
        vec = [val] * self.dimension
        return vec

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.embed_documents(input)

@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    return MockEmbeddingService()

@pytest.fixture
def test_vector_store(temp_chroma_dir: str, mock_embedding_service: MockEmbeddingService) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_directory=temp_chroma_dir,
        collection_name="test_collection",
        embedding_function=mock_embedding_service
    )
