"""
ChromaDB persistent vector store management.
Handles indexing, duplicate detection, chunk storage, and vector retrieval.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import chromadb
from chromadb.config import Settings

from src.exceptions import VectorStoreError
from src.logger import logger
from src.models import DocumentChunk, RetrievalResult

__all__ = [
    "ChromaVectorStore",
    "VectorStoreError",
]

class ChromaVectorStore:
    """
    Persistent ChromaDB vector store wrapper for document chunks.
    Maintains embeddings, metadata, and enables duplicate detection.
    """

    def __init__(
        self,
        persist_directory: Union[str, Path] = "./data/chroma",
        collection_name: str = "knowledge_assistant",
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize the ChromaDB persistent client.

        Args:
            persist_directory: Folder path where ChromaDB persists SQLite & index data.
            collection_name: Name of the vector collection.
            embedding_function: Optional embedding function object.
        """
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        logger.info(f"Connecting to ChromaDB at: {self.persist_directory}")
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                f"Collection '{self.collection_name}' ready. Total chunks indexed: {self.collection.count()}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise VectorStoreError(f"ChromaDB initialization failed: {e}") from e

    def count(self) -> int:
        """Returns total number of chunks currently stored in the collection."""
        return self.collection.count()

    def get_indexed_hashes(self) -> Set[str]:
        """Returns a set of all unique document hashes stored in the collection."""
        try:
            results = self.collection.get(include=["metadatas"])
            hashes = set()
            for meta in results.get("metadatas", []):
                if meta and "file_hash" in meta:
                    hashes.add(meta["file_hash"])
            return hashes
        except Exception as e:
            logger.warning(f"Could not retrieve indexed hashes: {e}")
            return set()

    def is_document_indexed(self, file_hash: str) -> bool:
        """Checks whether chunks for a document hash are already present."""
        if not file_hash:
            return False
        return file_hash in self.get_indexed_hashes()

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Returns summary info of all documents stored in the database.
        Each entry has {file_hash, source, total_chunks, pages: set of page numbers}.
        """
        try:
            results = self.collection.get(include=["metadatas"])
            docs: Dict[str, Dict[str, Any]] = {}
            for meta in results.get("metadatas", []):
                if not meta:
                    continue
                file_hash = meta.get("file_hash", "unknown")
                source = meta.get("source", "unknown")
                page = meta.get("page", 1)

                if file_hash not in docs:
                    docs[file_hash] = {
                        "file_hash": file_hash,
                        "source": source,
                        "total_chunks": 0,
                        "pages": set(),
                        "total_pages": meta.get("total_pages", 1)
                    }
                docs[file_hash]["total_chunks"] += 1
                docs[file_hash]["pages"].add(page)

            # Format set to list / count
            summary = []
            for doc in docs.values():
                summary.append({
                    "file_hash": doc["file_hash"],
                    "source": doc["source"],
                    "total_chunks": doc["total_chunks"],
                    "page_count": len(doc["pages"]),
                    "total_pages": doc["total_pages"]
                })
            return summary
        except Exception as e:
            logger.error(f"Failed to get indexed documents summary: {e}")
            return []

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 100
    ) -> int:
        """
        Adds text chunks to the Chroma collection with metadata and vectors.

        Args:
            chunks: List of DocumentChunk models.
            embeddings: Optional pre-computed float embeddings list matching chunks length.
            batch_size: Batch size for inserting into ChromaDB.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        # Validate embeddings alignment if passed
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) must match chunks count ({len(chunks)})"
            )

        total_added = 0
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch_chunks]
            documents = [c.content for c in batch_chunks]
            metadatas = [c.metadata for c in batch_chunks]

            batch_kwargs: Dict[str, Any] = {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
            }

            if embeddings is not None:
                batch_kwargs["embeddings"] = embeddings[i : i + batch_size]

            try:
                self.collection.upsert(**batch_kwargs)
                total_added += len(batch_chunks)
            except Exception as e:
                logger.error(f"Failed to add batch of chunks to ChromaDB: {e}")
                raise VectorStoreError(f"Failed to insert chunks into ChromaDB: {e}") from e

        logger.info(f"Successfully upserted {total_added} chunks into '{self.collection_name}'.")
        return total_added

    def query_by_embedding(
        self,
        query_vector: List[float],
        top_k: int = 4,
        where: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Performs semantic vector search in ChromaDB.

        Args:
            query_vector: Dense embedding vector of the query.
            top_k: Number of nearest neighbors to retrieve.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of RetrievalResult objects sorted by similarity.
        """
        if self.count() == 0:
            logger.info("ChromaDB collection is empty. Returning 0 results.")
            return []

        # Ensure top_k does not exceed stored chunks count
        k = max(1, min(top_k, self.count()))

        query_params: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            query_params["where"] = where

        try:
            results = self.collection.query(**query_params)
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise VectorStoreError(f"Vector search failed: {e}") from e

        retrieval_results: List[RetrievalResult] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, content, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns cosine distance (in [0, 2]). Cosine similarity = 1 - distance
            # Normalize to similarity score where higher is more similar
            similarity_score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0

            source_name = meta.get("source", "Unknown")
            page_num = int(meta.get("page", 1))

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=content,
                    source_filename=source_name,
                    page_number=page_num,
                    score=similarity_score,
                    metadata=meta
                )
            )

        # Sort descending by similarity score
        retrieval_results.sort(key=lambda r: r.score, reverse=True)
        return retrieval_results

    def get_document_chunks(
        self,
        file_hash: Optional[str] = None,
        source_filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all chunks belonging to a specific document without semantic search.

        Args:
            file_hash: Optional file hash to filter by.
            source_filename: Optional source filename to filter by.

        Returns:
            List of dictionaries containing chunk_id, content, and metadata.
        """
        where = {}
        if file_hash:
            where["file_hash"] = file_hash
        elif source_filename:
            where["source"] = source_filename

        try:
            if where:
                results = self.collection.get(where=where, include=["documents", "metadatas"])
            else:
                results = self.collection.get(include=["documents", "metadatas"])

            chunks: List[Dict[str, Any]] = []
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])

            for cid, doc, meta in zip(ids, docs, metas):
                chunks.append({
                    "chunk_id": cid,
                    "content": doc,
                    "metadata": meta or {}
                })
            return chunks
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return []

    def delete_document(self, file_hash: str) -> int:
        """
        Deletes all chunks associated with a specific file_hash.

        Returns:
            Estimated count of deleted items.
        """
        if not file_hash:
            return 0
        try:
            before_count = self.count()
            self.collection.delete(where={"file_hash": file_hash})
            after_count = self.count()
            deleted = before_count - after_count
            logger.info(f"Deleted {deleted} chunks for file_hash '{file_hash[:8]}'.")
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete document with hash {file_hash}: {e}")
            raise VectorStoreError(f"Failed to delete document: {e}") from e

    def clear(self) -> None:
        """Empties the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared ChromaDB collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB: {e}")
            raise VectorStoreError(f"Failed to clear ChromaDB: {e}") from e
