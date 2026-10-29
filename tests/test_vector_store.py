"""
Unit tests for ChromaVectorStore persistence, indexing, duplicate tracking, and queries.
"""

from typing import List
import pytest
from src.models import DocumentChunk, RetrievalResult
from src.vector_store import ChromaVectorStore

def test_vector_store_add_and_count(test_vector_store: ChromaVectorStore):
    """Verifies adding document chunks to ChromaDB and checking count."""
    assert test_vector_store.count() == 0

    chunks = [
        DocumentChunk(
            chunk_id="doc1_p1_c0",
            content="Superconducting qubits require dilution refrigerators.",
            source_filename="quantum.pdf",
            page_number=1,
            chunk_index=0,
            file_hash="hash_alpha",
            metadata={"source": "quantum.pdf", "page": 1, "file_hash": "hash_alpha"}
        ),
        DocumentChunk(
            chunk_id="doc1_p2_c0",
            content="Trapped ion qubits have long coherence times.",
            source_filename="quantum.pdf",
            page_number=2,
            chunk_index=0,
            file_hash="hash_alpha",
            metadata={"source": "quantum.pdf", "page": 2, "file_hash": "hash_alpha"}
        )
    ]

    added = test_vector_store.add_chunks(chunks)
    assert added == 2
    assert test_vector_store.count() == 2

    # Check duplicate tracking
    assert test_vector_store.is_document_indexed("hash_alpha") is True
    assert test_vector_store.is_document_indexed("non_existent_hash") is False

def test_vector_store_query(test_vector_store: ChromaVectorStore, mock_embedding_service):
    """Verifies semantic query retrieval and ranking."""
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            content="Qubits and superposition principles in physics.",
            source_filename="physics.pdf",
            page_number=1,
            chunk_index=0,
            file_hash="h1",
            metadata={"source": "physics.pdf", "page": 1, "file_hash": "h1"}
        ),
        DocumentChunk(
            chunk_id="c2",
            content="Baking chocolate chip cookies in an oven.",
            source_filename="recipes.pdf",
            page_number=5,
            chunk_index=0,
            file_hash="h2",
            metadata={"source": "recipes.pdf", "page": 5, "file_hash": "h2"}
        )
    ]
    test_vector_store.add_chunks(chunks)

    query_vec = mock_embedding_service.embed_query("quantum qubits")
    results: List[RetrievalResult] = test_vector_store.query_by_embedding(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].source_filename in ["physics.pdf", "recipes.pdf"]
    assert results[0].score >= 0.0

def test_delete_document_and_clear(test_vector_store: ChromaVectorStore):
    """Verifies deleting chunks by document hash and clearing the collection."""
    chunks = [
        DocumentChunk(
            chunk_id="c_docA_0",
            content="Content from Doc A",
            source_filename="docA.pdf",
            page_number=1,
            chunk_index=0,
            file_hash="hash_A",
            metadata={"source": "docA.pdf", "page": 1, "file_hash": "hash_A"}
        ),
        DocumentChunk(
            chunk_id="c_docB_0",
            content="Content from Doc B",
            source_filename="docB.pdf",
            page_number=1,
            chunk_index=0,
            file_hash="hash_B",
            metadata={"source": "docB.pdf", "page": 1, "file_hash": "hash_B"}
        )
    ]
    test_vector_store.add_chunks(chunks)
    assert test_vector_store.count() == 2

    # Delete Doc A
    test_vector_store.delete_document("hash_A")
    assert test_vector_store.count() == 1
    assert test_vector_store.is_document_indexed("hash_A") is False
    assert test_vector_store.is_document_indexed("hash_B") is True

    # Clear all
    test_vector_store.clear()
    assert test_vector_store.count() == 0

