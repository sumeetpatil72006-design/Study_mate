"""
Embedding generation service using Sentence Transformers.
"""

from typing import List, Optional, Union
import numpy as np

from src.logger import logger

class EmbeddingService:
    """
    Wrapper around SentenceTransformer models for generating vector embeddings.
    Implements ChromaDB-compatible embedding function interface.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize the embedding service.

        Args:
            model_name: HuggingFace model identifier (e.g. 'all-MiniLM-L6-v2').
            device: 'cpu', 'cuda', or None for automatic selection.
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        logger.info(f"Initialized EmbeddingService with model: {model_name}")

    @property
    def model(self):
        """Lazy load SentenceTransformer model when needed."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            from sentence_transformers import SentenceTransformer
            try:
                # Prioritize local cached files to avoid network latency/timeouts
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    local_files_only=True
                )
            except Exception:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
            logger.info(f"Model '{self.model_name}' loaded successfully.")
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Computes dense embeddings for a list of document strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vector embeddings.
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Computes a single dense embedding vector for a query string.

        Args:
            text: Query string.

        Returns:
            Float vector embedding.
        """
        if not text or not text.strip():
            raise ValueError("Query string cannot be empty.")
        embedding = self.model.encode(
            text.strip(),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()

    @staticmethod
    def name() -> str:
        """Name of the embedding function for ChromaDB configuration."""
        return "sentence_transformer_all_minilm_l6_v2"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def is_legacy(self) -> bool:
        return False

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> "EmbeddingService":
        return EmbeddingService(model_name=config.get("model_name", "all-MiniLM-L6-v2"))

    def __call__(self, input: Union[List[str], str]) -> List[List[float]]:
        """
        ChromaDB EmbeddingFunction interface compatibility.
        """
        if isinstance(input, str):
            input = [input]
        return self.embed_documents(input)
