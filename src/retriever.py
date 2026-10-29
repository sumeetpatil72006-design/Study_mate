"""
Semantic retriever coordinating embedding generation and vector search.
"""

from typing import List, Optional, Tuple
from src.embeddings import EmbeddingService
from src.logger import logger
from src.models import RetrievalResult
from src.vector_store import ChromaVectorStore

class Retriever:
    """
    Coordinates semantic vector retrieval from ChromaDB using SentenceTransformers.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_service: EmbeddingService,
        default_top_k: int = 4,
        similarity_threshold: Optional[float] = None
    ):
        """
        Initialize Retriever.

        Args:
            vector_store: Initialized ChromaVectorStore instance.
            embedding_service: Initialized EmbeddingService.
            default_top_k: Default number of chunks to return.
            similarity_threshold: Minimum similarity score [0.0, 1.0] to consider relevant.
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        where: Optional[dict] = None
    ) -> List[RetrievalResult]:
        """
        Executes semantic search for the given user query.

        Args:
            query: User's question or search query.
            top_k: Optional override for number of retrieved chunks.
            threshold: Optional override for similarity score cutoff.
            where: Optional metadata filter for scoped retrieval (e.g. by document).

        Returns:
            List of RetrievalResult objects.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever.")
            return []

        clean_query = query.strip()
        k = top_k if (top_k is not None and top_k > 0) else self.default_top_k
        thresh = threshold if threshold is not None else self.similarity_threshold

        logger.info(f"Retrieving top {k} relevant chunks for query: '{clean_query}' (where={where})")

        # Generate query vector
        query_vector = self.embedding_service.embed_query(clean_query)

        # Query vector store
        results = self.vector_store.query_by_embedding(query_vector=query_vector, top_k=k, where=where)

        # Apply threshold filtering if specified
        if thresh is not None:
            filtered = [r for r in results if r.score >= thresh]
            logger.info(
                f"Retrieved {len(results)} chunks; {len(filtered)} met similarity threshold >= {thresh:.2f}"
            )
            return filtered

        return results

    @staticmethod
    def format_context_for_prompt(results: List[RetrievalResult]) -> Tuple[str, bool]:
        """
        Formats retrieved chunks into structured context blocks for the LLM prompt.

        Returns:
            Tuple of (formatted_context_string, has_context_bool).
        """
        if not results:
            return "No relevant documents found.", False

        context_blocks = []
        for i, res in enumerate(results, start=1):
            block = (
                f"[Source {i}: {res.source_filename} | Page: {res.page_number} | Relevance: {res.score:.2f}]\n"
                f"{res.content.strip()}"
            )
            context_blocks.append(block)

        formatted_context = "\n\n---\n\n".join(context_blocks)
        return formatted_context, True

