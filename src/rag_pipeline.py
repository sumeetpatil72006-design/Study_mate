"""
End-to-end RAG Pipeline orchestrator.
Coordinates document extraction, chunking, embedding, vector storage, retrieval, and generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import io

from src.config import AppConfig, default_config
from src.document_loader import PDFDocumentLoader, NoTextExtractedError
from src.embeddings import EmbeddingService
from src.generator import GeminiGenerator, INSUFFICIENT_CONTEXT_MESSAGE
from src.logger import logger
from src.models import DocumentChunk, DocumentPage, RAGResponse, RetrievalResult
from src.quiz_generator import QuizGenerator
from src.quiz_models import Quiz, Topic
from src.retriever import Retriever
from src.text_splitter import DocumentChunker
from src.topic_extractor import TopicExtractor
from src.vector_store import ChromaVectorStore

class RAGPipeline:
    """
    Main orchestrator for the Retrieval-Augmented Generation (RAG) assistant.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the RAG pipeline with components based on configuration.
        """
        self.config = config or default_config

        logger.info(f"Initializing RAG Pipeline with config: {self.config}")

        # 1. Document Loader & Chunker
        self.document_loader = PDFDocumentLoader()
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # 2. Embedding Service
        self.embedding_service = EmbeddingService(
            model_name=self.config.embedding_model_name
        )

        # 3. Persistent ChromaDB Store
        self.vector_store = ChromaVectorStore(
            persist_directory=self.config.chroma_persist_directory,
            collection_name="knowledge_assistant",
            embedding_function=self.embedding_service
        )

        # Sync document_loader processed hashes with already stored ChromaDB documents
        for h in self.vector_store.get_indexed_hashes():
            self.document_loader.mark_as_processed(h)

        # 4. Retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            default_top_k=self.config.top_k
        )

        # 5. Gemini Generator
        self.generator = GeminiGenerator(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_model
        )

        # 6. Topic Extractor & Quiz Generator
        self.topic_extractor = TopicExtractor(
            vector_store=self.vector_store,
            generator=self.generator
        )
        self.quiz_generator = QuizGenerator(
            retriever=self.retriever,
            generator=self.generator
        )

        logger.info("RAG Pipeline initialization complete.")

    def ingest_pdf(
        self,
        file_input: Union[str, Path, bytes, io.BytesIO],
        filename: Optional[str] = None,
        check_duplicate: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Processes and indexes a text-based PDF document into the knowledge base.

        Args:
            file_input: Path to file or bytes buffer.
            filename: Display filename.
            check_duplicate: Whether to reject already indexed files.

        Returns:
            Dictionary with ingestion statistics (filename, total_pages, total_chunks, file_hash).
        """
        # Step 1: Extract pages and metadata
        pages, file_hash = self.document_loader.load_pdf(
            file_input=file_input,
            filename=filename,
            check_duplicate=check_duplicate,
            **kwargs
        )

        # Step 2: Chunk pages into segments
        chunks: List[DocumentChunk] = self.chunker.split_pages(pages)
        if not chunks:
            raise NoTextExtractedError(
                f"No textual content could be chunked from '{filename or 'file'}'. "
                "The document may contain only empty or unextractable pages."
            )

        # Step 3: Compute embeddings
        texts = [c.content for c in chunks]
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_service.embed_documents(texts)

        # Step 4: Add chunks to vector store
        self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

        # Step 5: Mark file hash as processed
        self.document_loader.mark_as_processed(file_hash)

        source_name = filename or (pages[0].source_filename if pages else "uploaded_document.pdf")
        logger.info(
            f"Successfully indexed document '{source_name}' ({len(pages)} pages, {len(chunks)} chunks)."
        )

        return {
            "source_filename": source_name,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "file_hash": file_hash
        }

    def ask(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> RAGResponse:
        """
        Answers a user query using the RAG pipeline.

        Args:
            query: The question asked by the user.
            top_k: Optional number of context chunks to retrieve.

        Returns:
            RAGResponse object containing answer, citations, and metadata.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        clean_query = query.strip()
        logger.info(f"Received query: '{clean_query}'")

        # Check if vector store has any documents
        total_chunks = self.vector_store.count()
        if total_chunks == 0:
            logger.info("Knowledge base is empty. Returning insufficient context response.")
            return RAGResponse(
                query=clean_query,
                answer=(
                    "No documents have been uploaded to the knowledge base yet. "
                    "Please upload a PDF document first."
                ),
                citations=[],
                retrieved_chunks=[],
                is_insufficient_context=True,
                model_name=self.generator.model_name
            )

        # Step 1: Semantic retrieval
        retrieved_chunks: List[RetrievalResult] = self.retriever.retrieve(
            query=clean_query,
            top_k=top_k or self.config.top_k
        )

        # Step 2: Format context
        context_str, has_context = self.retriever.format_context_for_prompt(retrieved_chunks)

        if not has_context:
            return RAGResponse(
                query=clean_query,
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                citations=[],
                retrieved_chunks=[],
                is_insufficient_context=True,
                model_name=self.generator.model_name
            )

        # Step 3: LLM generation
        answer, citations, is_insufficient = self.generator.generate_answer(
            query=clean_query,
            retrieved_chunks=retrieved_chunks,
            context_str=context_str
        )

        return RAGResponse(
            query=clean_query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            is_insufficient_context=is_insufficient,
            model_name=self.generator.model_name
        )

    def get_database_summary(self) -> Dict[str, Any]:
        """
        Returns stats about the vector store and indexed files.
        """
        docs = self.vector_store.get_indexed_documents()
        return {
            "total_chunks": self.vector_store.count(),
            "total_documents": len(docs),
            "documents": docs
        }

    def delete_document(self, file_hash: str) -> int:
        """
        Removes a document by hash from vector store and loader tracking.
        """
        deleted = self.vector_store.delete_document(file_hash)
        if file_hash in self.document_loader.processed_hashes:
            self.document_loader.processed_hashes.remove(file_hash)
        return deleted

    def clear_database(self) -> None:
        """
        Completely resets vector store and document registry.
        """
        self.vector_store.clear()
        self.document_loader.processed_hashes.clear()
        logger.info("Cleared all documents from RAG Pipeline.")

    def discover_topics(
        self,
        source_filename: Optional[str] = None,
        file_hash: Optional[str] = None
    ) -> List[Topic]:
        """
        Extracts study topics grounded in document content and page numbers.

        Args:
            source_filename: Filename of the target document.
            file_hash: Hash of the target document.

        Returns:
            List of discovered Topic objects.
        """
        return self.topic_extractor.extract_topics(
            source_filename=source_filename,
            file_hash=file_hash
        )

    def generate_quiz(
        self,
        topic: Topic,
        num_questions: int = 5,
        difficulty: str = "Medium",
        question_type: str = "MCQ"
    ) -> Quiz:
        """
        Generates and validates a multiple-choice quiz scoped to a specific topic and document.

        Args:
            topic: Topic object to generate the quiz for.
            num_questions: Desired question count (default 5).
            difficulty: "Easy", "Medium", or "Hard".
            question_type: "MCQ".

        Returns:
            Validated Quiz object.
        """
        return self.quiz_generator.generate_quiz(
            topic=topic,
            num_questions=num_questions,
            difficulty=difficulty,
            question_type=question_type
        )
