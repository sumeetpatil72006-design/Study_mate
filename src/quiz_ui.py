"""
Streamlit UI components for Topic Extraction and Quiz Generator.
Handles topic selection, quiz configuration, single-question navigation,
answer recording, submission, scoring, review cards, and history tracking.
"""

from typing import Dict, List, Optional
import streamlit as st

from src.exceptions import (
    InsufficientContentForQuizError,
    QuizError,
    QuizGenerationError,
    QuizValidationError,
)
from src.logger import logger
from src.quiz_models import Quiz, QuizQuestion, QuizResult, Topic, score_quiz
from src.rag_pipeline import RAGPipeline


def init_quiz_session_state() -> None:
    """Initializes Streamlit session state keys for the quiz generator."""
    if "quiz_current_quiz" not in st.session_state:
        st.session_state.quiz_current_quiz = None
    if "quiz_current_index" not in st.session_state:
        st.session_state.quiz_current_index = 0
    if "quiz_user_answers" not in st.session_state:
        st.session_state.quiz_user_answers = {}
    if "quiz_is_submitted" not in st.session_state:
        st.session_state.quiz_is_submitted = False
    if "quiz_result" not in st.session_state:
        st.session_state.quiz_result = None
    if "quiz_history" not in st.session_state:
        st.session_state.quiz_history = []
    if "quiz_discovered_topics" not in st.session_state:
        st.session_state.quiz_discovered_topics = {}
    if "quiz_submission_recorded" not in st.session_state:
        st.session_state.quiz_submission_recorded = False


def render_quiz_tab(pipeline: RAGPipeline) -> None:
    """
    Renders the complete Topic Extraction & Quiz Generator UI.

    Args:
        pipeline: Active RAGPipeline instance.
    """
    init_quiz_session_state()

    db_summary = pipeline.get_database_summary()
    indexed_docs = db_summary.get("documents", [])

    if not indexed_docs:
        st.info(
            "📚 **No documents indexed yet.**\n\n"
            "Upload one or more text-based PDF documents using the sidebar to discover study topics "
            "and generate interactive quizzes."
        )
        return

    # View 1: Results and Review View
    if st.session_state.quiz_is_submitted and st.session_state.quiz_result:
        _render_quiz_results_view()
        return

    # View 2: Active Quiz In-Progress View
    if st.session_state.quiz_current_quiz is not None:
        _render_active_quiz_view()
        return

    # View 3: Setup & Configuration View
    _render_quiz_setup_view(pipeline=pipeline, indexed_docs=indexed_docs)


def _render_quiz_setup_view(pipeline: RAGPipeline, indexed_docs: List[Dict]) -> None:
    """Renders document selection, topic discovery, and quiz configuration."""
    st.markdown("### 📝 Study Quiz Generator")
    st.caption("Select an indexed document, discover grounded topics, and configure your practice quiz.")

    # Step 1: Select Document
    doc_filenames = [doc["source"] for doc in indexed_docs]
    selected_doc_name = st.selectbox(
        "1. Select Document",
        options=doc_filenames,
        index=0,
        help="Choose an uploaded document to study."
    )

    # Step 2: Topic Discovery
    st.markdown("#### 2. Discovered Study Topics")
    cached_topics = st.session_state.quiz_discovered_topics.get(selected_doc_name, [])

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        discover_clicked = st.button("🔍 Discover Topics", use_container_width=True)

    if discover_clicked or not cached_topics:
        with st.spinner(f"Extracting grounded study topics from '{selected_doc_name}'..."):
            try:
                topics = pipeline.discover_topics(source_filename=selected_doc_name)
                st.session_state.quiz_discovered_topics[selected_doc_name] = topics
                cached_topics = topics
            except InsufficientContentForQuizError as ice:
                st.warning(f"⚠️ {ice}")
                cached_topics = []
            except Exception as e:
                logger.error(f"Error discovering topics: {e}")
                st.error(f"Failed to discover topics: {e}")
                cached_topics = []

    if not cached_topics:
        st.warning(
            "No topics could be discovered from this document. "
            "The document might contain insufficient textual content."
        )
        return

    # Topic Selection
    topic_map = {t.name: t for t in cached_topics}
    topic_options = list(topic_map.keys())

    selected_topic_name = st.selectbox(
        "Choose a topic to test:",
        options=topic_options,
        index=0
    )
    chosen_topic = topic_map[selected_topic_name]

    # Show topic details
    pages_str = ", ".join(str(p) for p in chosen_topic.source_pages) if chosen_topic.source_pages else "All"
    st.info(
        f"**Topic Details:**  \n"
        f"📖 **Description:** {chosen_topic.description}  \n"
        f"📄 **Cited Pages:** Page(s) {pages_str} | **Relevant Chunks:** {chosen_topic.chunk_count}"
    )

    st.divider()

    # Step 3: Quiz Configuration
    st.markdown("#### 3. Quiz Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

    with cfg_col1:
        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=10,
            value=5,
            help="Default is 5 questions."
        )

    with cfg_col2:
        difficulty = st.selectbox(
            "Difficulty Level",
            options=["Medium", "Easy", "Hard"],
            index=0,
            help="Difficulty of questions grounded in text."
        )

    with cfg_col3:
        question_type = st.selectbox(
            "Question Type",
            options=["Multiple Choice (MCQ)"],
            index=0,
            disabled=True,
            help="Phase 1 supports Multiple Choice Questions."
        )

    # Step 4: Generate Quiz Button
    st.write("")
    if st.button("🚀 Generate & Start Quiz", type="primary", use_container_width=True):
        with st.spinner("Retrieving scoped context and generating grounded questions with Gemini..."):
            try:
                quiz = pipeline.generate_quiz(
                    topic=chosen_topic,
                    num_questions=num_questions,
                    difficulty=difficulty,
                    question_type=question_type
                )
                # Initialize session state for active quiz
                st.session_state.quiz_current_quiz = quiz
                st.session_state.quiz_current_index = 0
                st.session_state.quiz_user_answers = {}
                st.session_state.quiz_is_submitted = False
                st.session_state.quiz_result = None
                st.session_state.quiz_submission_recorded = False
                st.rerun()

            except InsufficientContentForQuizError as ice:
                st.warning(f"⚠️ Insufficient Content: {ice}")
            except QuizValidationError as qve:
                st.error(f"⚠️ Quiz Validation Error: {qve}")
            except QuizGenerationError as qge:
                st.error(f"⚠️ Generation Failed: {qge}")
            except Exception as e:
                logger.error(f"Unexpected error starting quiz: {e}", exc_info=True)
                st.error(f"An unexpected error occurred: {e}")

    # Step 5: Past Quiz History
    if st.session_state.quiz_history:
        st.divider()
        with st.expander(f"📊 Past Quiz Attempts ({len(st.session_state.quiz_history)})", expanded=False):
            for i, hist in enumerate(reversed(st.session_state.quiz_history), start=1):
                st.markdown(
                    f"**Attempt #{len(st.session_state.quiz_history) - i + 1}** — "
                    f"Score: `{hist.correct_count}/{hist.total_questions}` ({hist.score_percentage:.1f}%) — "
                    f"<small>{hist.submitted_at[:19].replace('T', ' ')} UTC</small>",
                    unsafe_allow_html=True
                )


def _render_active_quiz_view() -> None:
    """
    Renders the active in-progress quiz showing one question at a time.
    Strictly conceals correct answers and explanations until submission.
    """
    quiz: Quiz = st.session_state.quiz_current_quiz
    idx: int = st.session_state.quiz_current_index
    total = len(quiz.questions)
    current_q: QuizQuestion = quiz.questions[idx]

    # Progress and Header
    hdr_col1, hdr_col2 = st.columns([3, 1])
    with hdr_col1:
        st.markdown(f"### 📝 {quiz.title}")
        st.caption(f"Document: **{quiz.source_filename}** | Difficulty: **{quiz.difficulty}**")
    with hdr_col2:
        st.markdown(f"**Question {idx + 1} of {total}**")
        st.progress((idx + 1) / total)

    st.write("")

    # Question Card Container
    with st.container():
        st.markdown(f"#### Q{idx + 1}. {current_q.question}")

        # Restore previously selected answer if present
        saved_answer = st.session_state.quiz_user_answers.get(current_q.question_id)
        default_index = None
        if saved_answer and saved_answer in current_q.options:
            default_index = current_q.options.index(saved_answer)

        # Radio button selection
        selected_option = st.radio(
            "Select one answer:",
            options=current_q.options,
            index=default_index,
            key=f"q_radio_{current_q.question_id}",
            help="Choose the best answer based on your notes."
        )

        # Save selected answer immediately
        if selected_option:
            st.session_state.quiz_user_answers[current_q.question_id] = selected_option

    st.write("")
    st.divider()

    # Navigation Controls
    answered_count = len(st.session_state.quiz_user_answers)
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1.5, 1])

    with nav_col1:
        if st.button("⬅ Previous", disabled=(idx == 0), use_container_width=True):
            st.session_state.quiz_current_index -= 1
            st.rerun()

    with nav_col2:
        if idx < total - 1:
            if st.button("Next ➡", use_container_width=True):
                st.session_state.quiz_current_index += 1
                st.rerun()

    with nav_col3:
        # Show submit button
        submit_label = f"✅ Submit Quiz ({answered_count}/{total} Answered)"
        if st.button(submit_label, type="primary", use_container_width=True):
            if answered_count < total:
                st.warning(
                    f"You have answered {answered_count} of {total} questions. "
                    "Unanswered questions will be counted as incorrect."
                )

            # Score quiz and record submission
            result = score_quiz(quiz, st.session_state.quiz_user_answers)
            st.session_state.quiz_result = result
            st.session_state.quiz_is_submitted = True

            # Prevent duplicate submissions from inflating history
            if not st.session_state.quiz_submission_recorded:
                st.session_state.quiz_history.append(result)
                st.session_state.quiz_submission_recorded = True

            st.rerun()

    with nav_col4:
        if st.button("❌ Quit Quiz", use_container_width=True):
            st.session_state.quiz_current_quiz = None
            st.session_state.quiz_user_answers = {}
            st.session_state.quiz_current_index = 0
            st.session_state.quiz_is_submitted = False
            st.session_state.quiz_result = None
            st.rerun()


def _render_quiz_results_view() -> None:
    """
    Renders post-submission results, score metrics, topic breakdown,
    and question-by-question review with citations and explanations.
    """
    result: QuizResult = st.session_state.quiz_result
    quiz: Quiz = st.session_state.quiz_current_quiz

    st.markdown("### 📊 Quiz Results & Review")
    st.caption(f"Topic: **{quiz.topic}** | Document: **{quiz.source_filename}**")

    # Score Metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Score", f"{result.correct_count} / {result.total_questions}")
    with m_col2:
        st.metric("Percentage", f"{result.score_percentage:.1f}%")
    with m_col3:
        st.metric("Correct Answers", result.correct_count)
    with m_col4:
        st.metric("Wrong / Unanswered", result.wrong_count + result.unanswered_count)

    # Topic-wise Results Breakdown
    if result.topic_scores:
        st.markdown("#### 📈 Topic Performance")
        for topic_name, score_data in result.topic_scores.items():
            t_correct = score_data["correct"]
            t_total = score_data["total"]
            t_pct = (t_correct / t_total * 100.0) if t_total > 0 else 0.0
            st.write(f"- **{topic_name}:** {t_correct}/{t_total} correct ({t_pct:.1f}%)")

    st.divider()

    # Detailed Question Review
    st.markdown("#### 🔍 Detailed Question Review")
    for i, qr in enumerate(result.question_results, start=1):
        status_color = "#10B981" if qr.is_correct else "#EF4444"
        status_text = "✓ Correct" if qr.is_correct else "✗ Incorrect"
        if not qr.selected_answer:
            status_text = "⚠ Unanswered"
            status_color = "#F59E0B"

        with st.expander(f"Question {i}: {status_text} — {qr.question[:70]}...", expanded=True):
            st.markdown(f"**Question:** {qr.question}")

            col_ans1, col_ans2 = st.columns(2)
            with col_ans1:
                st.markdown(
                    f"**Your Answer:**  \n"
                    f"<span style='color: {status_color}; font-weight: 600;'>"
                    f"{qr.selected_answer or 'None (Unanswered)'}"
                    f"</span>",
                    unsafe_allow_html=True
                )
            with col_ans2:
                st.markdown(
                    f"**Correct Answer:**  \n"
                    f"<span style='color: #10B981; font-weight: 600;'>{qr.correct_answer}</span>",
                    unsafe_allow_html=True
                )

            st.write("")
            st.markdown(f"**Explanation:**  \n{qr.explanation}")

            # Grounded Source Citation
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border-left: 4px solid #3B82F6; padding: 0.6rem 0.8rem; border-radius: 4px; font-size: 0.88rem; margin-top: 0.5rem;">
                    <strong>📄 Source Citation:</strong> {qr.source_filename} — <strong>Page:</strong> {qr.source_page}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")
    st.divider()

    # Actions: Retake or Start New
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        if st.button("🔄 Retake This Quiz", type="primary", use_container_width=True):
            st.session_state.quiz_user_answers = {}
            st.session_state.quiz_current_index = 0
            st.session_state.quiz_is_submitted = False
            st.session_state.quiz_result = None
            st.session_state.quiz_submission_recorded = False
            st.rerun()

    with act_col2:
        if st.button("📚 Create New Quiz", use_container_width=True):
            st.session_state.quiz_current_quiz = None
            st.session_state.quiz_user_answers = {}
            st.session_state.quiz_current_index = 0
            st.session_state.quiz_is_submitted = False
            st.session_state.quiz_result = None
            st.session_state.quiz_submission_recorded = False
            st.rerun()

