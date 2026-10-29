"""
Configurable text splitter preserving page and document metadata.
"""

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.exceptions import NoTextExtractedError
from src.logger import logger
from src.models import DocumentChunk, DocumentPage

class DocumentChunker:
    """
    Splits DocumentPages into DocumentChunks with configurable size and overlap.
    Preserves source filename, page number, document hash, and chunk order.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum character count per chunk.
            chunk_overlap: Overlap between consecutive chunks.
            separators: Priority list of string separators for splitting.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )

    def split_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """
        Splits a list of DocumentPages into DocumentChunks.
        Splits page by page to ensure accurate page-level citations.

        Args:
            pages: List of DocumentPage objects.

        Returns:
            List of DocumentChunk objects with metadata attached.

        Raises:
            NoTextExtractedError: If pages contain no extractable or segmentable text.
        """
        if not pages:
            return []

        doc_name = pages[0].source_filename if pages else "document"
        non_empty_pages = [p for p in pages if p.content and p.content.strip()]
        if not non_empty_pages:
            logger.warning(
                f"Attempted to chunk {len(pages)} pages from '{doc_name}', but all pages are empty."
            )
            raise NoTextExtractedError(
                f"Cannot chunk '{doc_name}': all {len(pages)} pages are empty or contain only whitespace."
            )

        all_chunks: List[DocumentChunk] = []

        for page in pages:
            raw_text = page.content.strip()
            if not raw_text:
                continue

            text_segments = self._splitter.split_text(raw_text)
            # Ensure short text is never lost if splitter returns an empty list
            if not text_segments:
                text_segments = [raw_text]

            for chunk_idx, segment in enumerate(text_segments):
                clean_segment = segment.strip()
                if not clean_segment:
                    continue

                # Unique chunk ID: {file_hash_prefix}_p{page}_c{chunk_index}
                hash_prefix = page.file_hash[:8] if page.file_hash else "doc"
                chunk_id = f"{hash_prefix}_p{page.page_number}_c{chunk_idx}"

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=clean_segment,
                    source_filename=page.source_filename,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    file_hash=page.file_hash,
                    metadata={
                        "source": page.source_filename,
                        "page": page.page_number,
                        "chunk_index": chunk_idx,
                        "file_hash": page.file_hash,
                        "total_pages": page.total_pages,
                        "char_length": len(clean_segment),
                    }
                )
                all_chunks.append(chunk)

        if not all_chunks:
            logger.warning(
                f"Chunking yielded 0 chunks from {len(pages)} pages of '{doc_name}' "
                f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})."
            )
            raise NoTextExtractedError(
                f"Failed to produce any text chunks from {len(pages)} pages of '{doc_name}'. "
                "The text content may be too short or only whitespace."
            )

        logger.info(
            f"Chunked {len(pages)} pages into {len(all_chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})."
        )
        return all_chunks

