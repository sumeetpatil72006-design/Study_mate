"""
Centralized exception hierarchy for the AI Knowledge Assistant (RAG).
Ensures consistent definitions, imports, and error handling across modules.
"""

class RAGAssistantError(Exception):
    """Base exception for all RAG Assistant application errors."""
    pass

# Document Loader Exceptions
class DocumentLoaderError(RAGAssistantError):
    """Base exception for document loading and extraction errors."""
    pass

class InvalidPDFError(DocumentLoaderError):
    """Raised when an uploaded file is not a valid PDF or is corrupted."""
    pass

class EncryptedPDFError(DocumentLoaderError):
    """Raised when an uploaded PDF is password-protected or encrypted."""
    pass

class DuplicateDocumentError(DocumentLoaderError):
    """Raised when an uploaded file is already processed."""
    pass

class NoTextExtractedError(DocumentLoaderError):
    """Raised when no textual content could be extracted from a PDF (e.g. scanned image)."""
    pass

# Vector Store Exceptions
class VectorStoreError(RAGAssistantError):
    """Base exception for vector store operations."""
    pass

# Generator Exceptions
class GeneratorError(RAGAssistantError, RuntimeError):
    """Base exception for answer generation errors."""
    pass

class APIKeyMissingError(GeneratorError, ValueError):
    """Raised when the Gemini API key is missing or not configured."""
    pass

# Quiz Exceptions
class QuizError(RAGAssistantError):
    """Base exception for all quiz-related errors."""
    pass

class QuizGenerationError(QuizError, GeneratorError):
    """Raised when quiz generation fails."""
    pass

class QuizValidationError(QuizError, ValueError):
    """Raised when quiz schema or question validation fails."""
    pass

class InsufficientContentForQuizError(QuizError):
    """Raised when document content is insufficient to construct a reliable quiz."""
    pass

