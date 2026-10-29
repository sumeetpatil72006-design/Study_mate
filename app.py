"""
Streamlit Chatbot UI for the AI Knowledge Assistant (RAG).
Includes document upload, real-time vector indexing, chat history, and source citations.
"""

import os
import time
from typing import List
import streamlit as st

from src.config import AppConfig, default_config
from src.document_loader import (
    DocumentLoaderError,
    DuplicateDocumentError,
    EncryptedPDFError,
    InvalidPDFError,
    NoTextExtractedError,
)
from src.logger import logger
from src.models import RAGResponse
from src.quiz_ui import render_quiz_tab
from src.rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="AI Knowledge Assistant (RAG)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished modern look
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E3A8A;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-radius: 4px;
        font-size: 0.92rem;
    }
    .citation-meta {
        font-weight: 600;
        color: #1E40AF;
        margin-bottom: 0.3rem;
    }
    .citation-excerpt {
        color: #374151;
        font-style: italic;
    }
    .badge-success {
        color: #065F46;
        background-color: #D1FAE5;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warning {
        color: #92400E;
        background-color: #FEF3C7;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner=False)
def get_rag_pipeline(
    gemini_key: str,
    gemini_model: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int
) -> RAGPipeline:
    """
    Cached initializer for RAGPipeline.
    Ensures model weights and ChromaDB client are loaded once.
    """
    config = AppConfig(
        gemini_api_key=gemini_key,
        gemini_model=gemini_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k
    )
    return RAGPipeline(config=config)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_history" not in st.session_state:
    st.session_state.uploaded_files_history = []

# Sidebar: Configuration and Document Management
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/knowledge-sharing.png", width=70)
    st.title("Settings & Docs")

    # API Key Configuration
    st.subheader("1. API Configuration")
    env_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    
    if env_api_key:
        st.markdown('<span class="badge-success">✓ API Key loaded from .env</span>', unsafe_allow_html=True)
        user_api_key = env_api_key
    else:
        st.markdown('<span class="badge-warning">⚠ API Key not found in .env</span>', unsafe_allow_html=True)
        user_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste your Gemini API key...",
            help="Get your key at https://aistudio.google.com/"
        )

    # Model Selection
    selected_model = st.selectbox(
        "Gemini Model",
        options=["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
    )

    # Retrieval & Chunking Parameters
    with st.expander("⚙ Retrieval & Chunking Settings", expanded=False):
        chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=800, step=100)
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=150, step=25)
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=4, step=1)

    # Initialize Pipeline
    pipeline: RAGPipeline = get_rag_pipeline(
        gemini_key=user_api_key,
        gemini_model=selected_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k
    )

    # Ensure pipeline generator has current key and model
    if user_api_key:
        pipeline.generator.api_key = user_api_key
        pipeline.generator._client = None
    pipeline.generator.model_name = selected_model
    pipeline.retriever.default_top_k = top_k

    # Document Upload Section
    st.divider()
    st.subheader("2. Upload Knowledge Documents")
    st.caption("Supports digitally generated PDFs containing selectable text.")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload text-based PDF documents to index into the knowledge base."
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_identifier not in st.session_state.uploaded_files_history:
                with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                    try:
                        start_time = time.time()
                        stats = pipeline.ingest_pdf(
                            file_input=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            check_duplicate=True
                        )
                        elapsed = time.time() - start_time
                        st.session_state.uploaded_files_history.append(file_identifier)
                        st.success(
                            f" Indexed **{uploaded_file.name}**\n\n"
                            f"- Pages: {stats['total_pages']}\n"
                            f"- Chunks: {stats['total_chunks']}\n"
                            f"- Time: {elapsed:.2f}s"
                        )
                    except DuplicateDocumentError:
                        st.info(f"'{uploaded_file.name}' has already been indexed. Skipping re-processing.")
                        st.session_state.uploaded_files_history.append(file_identifier)
                    except NoTextExtractedError as e:
                        st.error(f"📄 Text Extraction: {e}")
                    except EncryptedPDFError as e:
                        st.error(f"🔒 Password Protected: {e}")
                    except InvalidPDFError as e:
                        st.error(f"⚠️ Invalid PDF '{uploaded_file.name}': {e}")
                    except Exception as e:
                        st.error(f"Error processing '{uploaded_file.name}': {e}")

    # Indexed Documents Summary
    st.divider()
    st.subheader("3. Indexed Knowledge Base")
    summary = pipeline.get_database_summary()
    st.write(f"**Total Documents:** {summary['total_documents']}")
    st.write(f"**Total Vector Chunks:** {summary['total_chunks']}")

    if summary["documents"]:
        with st.expander("📄 View Indexed Documents", expanded=False):
            for doc in summary["documents"]:
                st.markdown(
                    f"**{doc['source']}**  \n"
                    f"Chunks: `{doc['total_chunks']}` | Pages: `{doc['page_count']}`  \n"
                    f"<small>Hash: `{doc['file_hash'][:8]}...`</small>",
                    unsafe_allow_html=True
                )
                if st.button(f"🗑 Remove {doc['source'][:15]}", key=f"del_{doc['file_hash']}"):
                    pipeline.delete_document(doc['file_hash'])
                    st.rerun()

    if summary["total_chunks"] > 0:
        if st.button("🚨 Clear Entire Knowledge Base", type="secondary"):
            pipeline.clear_database()
            st.session_state.uploaded_files_history.clear()
            st.session_state.messages.clear()
            st.success("Knowledge Base cleared successfully.")
            st.rerun()

    # RAG Architecture Info
    with st.expander("ℹ️ How This RAG System Works", expanded=False):
        st.markdown(
            """
            **Architecture Overview:**
            1. **PDF Extraction:** PyMuPDF extracts text page-by-page preserving document metadata.
            2. **Chunking:** Text is split using Recursive Character Text Splitter with configurable chunk size & overlap.
            3. **Vector Embeddings:** Generated locally using `Sentence Transformers (all-MiniLM-L6-v2)`.
            4. **Persistent Vector Store:** ChromaDB stores chunk embeddings, text, and source page metadata.
            5. **Semantic Retrieval:** Cosine similarity retrieves the most relevant chunks.
            6. **Grounded Generation:** Gemini synthesizes answers strictly using retrieved context with exact page citations.
            """
        )

# Main Chat Interface
st.markdown('<div class="main-title">🧠 AI Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Retrieval-Augmented Generation (RAG) with local embeddings, ChromaDB, and Gemini API.</div>',
    unsafe_allow_html=True
)

# Tabs for Chat Q&A and Topic Quiz
tab_chat, tab_quiz = st.tabs(["💬 Ask Questions (RAG Chat)", "📝 Topic Quiz"])

with tab_chat:
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Display citations if present
            if msg.get("citations"):
                with st.expander(f"📚 Sources & Citations ({len(msg['citations'])} excerpts)", expanded=False):
                    for cit in msg["citations"]:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-meta">📄 {cit['filename']} — Page {cit['page_number']}</div>
                                <div class="citation-excerpt">"{cit['excerpt']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Quick Prompts if Chat History is Empty
    if not st.session_state.messages:
        st.info("👋 **Welcome!** Upload one or more PDF documents in the sidebar, then ask questions about them below.")
        st.markdown("**Sample questions you can ask once documents are uploaded:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- *'What are the primary findings in this document?'*")
            st.markdown("- *'Can you summarize section 2 in bullet points?'*")
        with col2:
            st.markdown("- *'What methodologies were utilized?'*")
            st.markdown("- *'What are the conclusions and future directions?'*")

    # Chat Input & Response Generation
    if user_prompt := st.chat_input("Ask a question about your uploaded documents..."):
        # Input validation
        if not user_prompt.strip():
            st.warning("Please enter a valid question.")
            st.stop()

        # Append user question to history
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base & generating answer..."):
                try:
                    response: RAGResponse = pipeline.ask(query=user_prompt)

                    # Render answer
                    st.markdown(response.answer)

                    # Render citations
                    citations_data = []
                    if response.citations:
                        citations_data = [
                            {
                                "filename": c.filename,
                                "page_number": c.page_number,
                                "excerpt": c.excerpt
                            }
                            for c in response.citations
                        ]
                        with st.expander(f"📚 Sources & Citations ({len(response.citations)} excerpts)", expanded=True):
                            for c in response.citations:
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <div class="citation-meta">📄 {c.filename} — Page {c.page_number}</div>
                                        <div class="citation-excerpt">"{c.excerpt}"</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    # Highlight insufficient context if flagged
                    if response.is_insufficient_context:
                        st.warning("Notice: The assistant determined that the uploaded documents do not contain enough information to fully answer this query.")

                    # Save response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "citations": citations_data,
                        "is_insufficient_context": response.is_insufficient_context
                    })

                except ValueError as ve:
                    st.error(f"Configuration or Input Error: {ve}")
                except Exception as e:
                    logger.error(f"Unexpected chat error: {e}", exc_info=True)
                    st.error(f"An error occurred while answering your question: {e}")

with tab_quiz:
    render_quiz_tab(pipeline=pipeline)


