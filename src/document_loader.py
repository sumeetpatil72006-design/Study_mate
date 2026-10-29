"""
PDF Document Loader and Text Extractor using PyMuPDF.
Focuses exclusively on text-based, digitally generated PDFs containing selectable text.
Includes page metadata extraction, duplicate detection, input validation,
careful whitespace normalization, and graceful empty page handling.
"""

import hashlib
import io
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union
import pymupdf as fitz

from src.exceptions import (
    DocumentLoaderError,
    DuplicateDocumentError,
    EncryptedPDFError,
    InvalidPDFError,
    NoTextExtractedError,
)
from src.logger import logger
from src.models import DocumentPage

__all__ = [
    "PDFDocumentLoader",
    "DocumentLoaderError",
    "DuplicateDocumentError",
    "EncryptedPDFError",
    "InvalidPDFError",
    "NoTextExtractedError",
    "normalize_text",
    "is_meaningful_text",
]

def normalize_text(text: str) -> str:
    """
    Normalizes whitespace carefully without damaging code snippets,
    tables, or paragraph structures.

    - Replaces form feeds and carriage returns with standard newlines.
    - Strips trailing whitespace on each line while preserving leading indentation.
    - Collapses 3 or more consecutive newlines into 2 (preserving paragraphs).
    - Preserves column spacing and code indentation.
    """
    if not text:
        return ""
    # Standardize line endings and form feeds
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    # Strip trailing whitespace on each line while preserving leading indentation
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3 or more consecutive blank lines into standard paragraph separation (2 newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_meaningful_text(text: str) -> bool:
    """
    Returns True if text contains at least one alphanumeric or word character,
    ensuring whitespace-only or symbol-only content is not treated as valid text.
    """
    return bool(re.search(r"\w", text))


class PDFDocumentLoader:
    """
    Extracts text and metadata page-by-page from digitally generated PDF files using PyMuPDF.
    Preserves source filename, 1-based page numbers, total page count, and document hash.
    Gracefully handles empty pages in multi-page documents and rejects documents with no extractable text.
    """

    def __init__(
        self,
        processed_hashes: Optional[Set[str]] = None,
        **kwargs  # Accept kwargs for backward compatibility
    ):
        """
        Initialize the loader.

        Args:
            processed_hashes: Optional set of already processed document SHA-256 hashes.
        """
        self.processed_hashes: Set[str] = set(processed_hashes or [])

    @staticmethod
    def compute_hash(data: Union[bytes, str, Path]) -> str:
        """Computes SHA-256 hash of file content for integrity and duplicate detection."""
        hasher = hashlib.sha256()
        if isinstance(data, (str, Path)):
            with open(data, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        elif isinstance(data, (bytes, bytearray)):
            hasher.update(data)
        else:
            raise ValueError("Unsupported data type for hash computation.")
        return hasher.hexdigest()

    def is_duplicate(self, file_hash: str) -> bool:
        """Checks if the document hash is already registered."""
        return file_hash in self.processed_hashes

    def mark_as_processed(self, file_hash: str) -> None:
        """Registers a document hash into processed set."""
        self.processed_hashes.add(file_hash)

    def load_pdf(
        self,
        file_input: Union[str, Path, bytes, io.BytesIO],
        filename: Optional[str] = None,
        check_duplicate: bool = True,
        **kwargs  # Accept kwargs for backward compatibility
    ) -> Tuple[List[DocumentPage], str]:
        """
        Extracts pages and text from a text-based PDF file.

        Args:
            file_input: File path (str/Path) or bytes / BytesIO buffer.
            filename: Human-friendly name of the file (inferred if path is passed).
            check_duplicate: If True, raises DuplicateDocumentError if already processed.

        Returns:
            Tuple of (List[DocumentPage], file_hash).

        Raises:
            InvalidPDFError: If file is empty, corrupted, or not a PDF.
            EncryptedPDFError: If file is password-protected and cannot be opened.
            DuplicateDocumentError: If document was already loaded.
            NoTextExtractedError: If no selectable text could be extracted.
        """
        file_bytes: bytes
        source_name: str

        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if not path.is_file():
                raise InvalidPDFError(f"Target is not a regular file: {path}")
            source_name = filename or path.name
            try:
                file_bytes = path.read_bytes()
            except Exception as e:
                raise DocumentLoaderError(f"Failed to read file '{source_name}': {type(e).__name__}") from e
        elif isinstance(file_input, bytes):
            file_bytes = file_input
            source_name = filename or "uploaded_document.pdf"
        elif isinstance(file_input, io.BytesIO):
            file_bytes = file_input.getvalue()
            source_name = filename or "uploaded_document.pdf"
        else:
            raise InvalidPDFError(f"Unsupported file input type: {type(file_input)}")

        # Validation: check non-empty
        if len(file_bytes) == 0:
            raise InvalidPDFError(f"Uploaded file '{source_name}' is empty (0 bytes).")

        # Validation: check PDF header
        if not file_bytes.startswith(b"%PDF"):
            raise InvalidPDFError(f"File '{source_name}' is not a valid PDF (missing %PDF signature).")

        file_hash = self.compute_hash(file_bytes)
        if check_duplicate and self.is_duplicate(file_hash):
            logger.warning(f"Duplicate document detected: '{source_name}' (hash: {file_hash[:8]}...)")
            raise DuplicateDocumentError(
                f"Document '{source_name}' (hash {file_hash[:8]}) has already been loaded."
            )

        # Open with PyMuPDF
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            logger.error(f"PyMuPDF failed to parse '{source_name}': {type(e).__name__}")
            raise InvalidPDFError(f"Failed to parse PDF '{source_name}': {type(e).__name__}") from e

        try:
            if doc.is_encrypted:
                # Attempt empty string decryption, otherwise reject
                if not doc.authenticate(""):
                    raise EncryptedPDFError(
                        f"PDF '{source_name}' is password-protected or encrypted. Please provide an unlocked document."
                    )

            total_pages = len(doc)
            if total_pages == 0:
                raise InvalidPDFError(f"PDF '{source_name}' contains 0 pages.")

            logger.info(
                f"Extracting text from '{source_name}' ({total_pages} pages, hash: {file_hash[:8]}...)"
            )

            pages: List[DocumentPage] = []

            for page_idx in range(total_pages):
                page_num = page_idx + 1  # 1-based indexing for citations
                try:
                    page = doc.load_page(page_idx)
                    raw_text = page.get_text("text") or ""
                    cleaned_text = normalize_text(raw_text)
                    is_empty = not is_meaningful_text(cleaned_text)

                    if is_empty:
                        cleaned_text = ""

                    doc_page = DocumentPage(
                        source_filename=source_name,
                        page_number=page_num,
                        total_pages=total_pages,
                        content=cleaned_text,
                        file_hash=file_hash,
                        metadata={
                            "source": source_name,
                            "page": page_num,
                            "total_pages": total_pages,
                            "file_hash": file_hash,
                            "char_count": len(cleaned_text),
                            "is_empty": is_empty,
                        }
                    )
                    pages.append(doc_page)

                except Exception as e:
                    logger.warning(f"Error processing page {page_num} of '{source_name}': {type(e).__name__}")
                    # Append placeholder for failed page to preserve correct page numbering
                    pages.append(
                        DocumentPage(
                            source_filename=source_name,
                            page_number=page_num,
                            total_pages=total_pages,
                            content="",
                            file_hash=file_hash,
                            metadata={
                                "source": source_name,
                                "page": page_num,
                                "total_pages": total_pages,
                                "file_hash": file_hash,
                                "char_count": 0,
                                "is_empty": True,
                                "has_error": True,
                                "error_type": type(e).__name__,
                            }
                        )
                    )
        finally:
            doc.close()

        # Check overall extracted text across the entire PDF
        meaningful_pages = [p for p in pages if not p.metadata.get("is_empty", False)]
        total_chars = sum(len(p.content) for p in meaningful_pages)

        logger.info(
            f"Extracted {len(pages)} pages from '{source_name}' "
            f"(pages with text: {len(meaningful_pages)}, total chars: {total_chars})."
        )

        if not meaningful_pages or total_chars == 0:
            raise NoTextExtractedError(
                f"No selectable text found in '{source_name}'. "
                "This assistant only supports digitally generated PDFs with selectable text. "
                "Scanned or image-only PDFs without a text layer are not supported."
            )

        return pages, file_hash
