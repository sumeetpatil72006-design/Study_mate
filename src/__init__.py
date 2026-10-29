"""
AI Knowledge Assistant - RAG Pipeline Package
"""

from src.exceptions import (
    RAGAssistantError,
    DocumentLoaderError,
    InvalidPDFError,
    EncryptedPDFError,
    DuplicateDocumentError,
    NoTextExtractedError,
    VectorStoreError,
    GeneratorError,
    APIKeyMissingError,
)

__version__ = "0.1.0"

__all__ = [
    "RAGAssistantError",
    "DocumentLoaderError",
    "InvalidPDFError",
    "EncryptedPDFError",
    "DuplicateDocumentError",
    "NoTextExtractedError",
    "VectorStoreError",
    "GeneratorError",
    "APIKeyMissingError",
]
