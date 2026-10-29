"""
Data models for the RAG pipeline.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class DocumentPage(BaseModel):
    """Represents raw text extracted from a single page of a document."""
    source_filename: str
    page_number: int
    total_pages: int
    content: str
    file_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentChunk(BaseModel):
    """Represents a text chunk produced by the chunking stage."""
    chunk_id: str
    content: str
    source_filename: str
    page_number: int
    chunk_index: int
    file_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    """Represents a retrieved chunk with its similarity score."""
    chunk_id: str
    content: str
    source_filename: str
    page_number: int
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SourceCitation(BaseModel):
    """Citation information linking answer statements back to source documents."""
    filename: str
    page_number: int
    excerpt: str

class RAGResponse(BaseModel):
    """Final response returned by the RAG pipeline."""
    query: str
    answer: str
    citations: List[SourceCitation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievalResult] = Field(default_factory=list)
    is_insufficient_context: bool = False
    model_name: Optional[str] = None

