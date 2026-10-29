"""
Configuration settings for the RAG Assistant application.
Focuses on text-based PDFs, persistent ChromaDB, and Google Gemini API.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class AppConfig:
    """Central configuration class with validation."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        chroma_persist_directory: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        top_k: Optional[int] = None,
        **kwargs  # Accept additional kwargs for backward compatibility
    ):
        # API Keys: prioritize explicit argument, then GEMINI_API_KEY, then GOOGLE_API_KEY
        self.gemini_api_key = (
            gemini_api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        
        # Storage
        self.chroma_persist_directory = Path(
            chroma_persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
        ).resolve()
        
        # Embeddings & Retrieval
        self.embedding_model_name = (
            embedding_model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        )
        
        # Chunking parameters
        self.chunk_size = int(chunk_size or os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = int(chunk_overlap or os.getenv("CHUNK_OVERLAP", "150"))
        self.top_k = int(top_k or os.getenv("TOP_K", "4"))

        self.validate()

    def validate(self) -> None:
        """Validates configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

    @property
    def has_api_key(self) -> bool:
        """Returns True if a Gemini API key is configured."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    def __repr__(self) -> str:
        return (
            f"AppConfig(model='{self.gemini_model}', "
            f"persist_dir='{self.chroma_persist_directory}', "
            f"embedding='{self.embedding_model_name}', "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
            f"top_k={self.top_k}, api_key_configured={self.has_api_key})"
        )

# Global default configuration instance
default_config = AppConfig()
