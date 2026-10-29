"""
Topic extraction and discovery from indexed PDF chunks.
Extracts study topics grounded in actual document content and linked to source filenames/pages.
Supports Gemini-assisted discovery with rigorous grounding checks and deterministic heuristic fallback.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set
from src.exceptions import InsufficientContentForQuizError, QuizError
from src.generator import GeminiGenerator
from src.logger import logger
from src.quiz_models import Topic
from src.vector_store import ChromaVectorStore

TOPIC_DISCOVERY_SYSTEM_PROMPT = """You are an expert educational content analyzer.
Analyze the following document excerpts and identify the major study topics covered in the text.

RULES:
1. Ground every topic strictly in the provided document content.
2. Link every topic to the exact page number(s) where it is discussed.
3. For each topic, provide:
   - name: Concise, clear topic title (2 to 6 words)
   - description: 1 to 2 sentences summarizing the key concepts
   - source_pages: JSON array of integer page numbers where this topic appears
4. Return ONLY a valid JSON array of objects matching the schema below. No markdown or conversational text.

Schema:
[
  {
    "name": "Topic Title",
    "description": "Brief explanation of concepts...",
    "source_pages": [1, 2]
  }
]
"""

TOPIC_DISCOVERY_USER_TEMPLATE = """Document: {filename}
Total Pages: {total_pages}

Document Excerpts by Page:
{excerpts}

Extract 2 to 6 distinct, grounded study topics."""


class TopicExtractor:
    """
    Extracts study topics from indexed document chunks in ChromaDB.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        generator: Optional[GeminiGenerator] = None,
        min_words_threshold: int = 25
    ):
        """
        Initialize TopicExtractor.

        Args:
            vector_store: Initialized ChromaVectorStore instance.
            generator: Optional GeminiGenerator instance for LLM discovery.
            min_words_threshold: Minimum total word count required in document.
        """
        self.vector_store = vector_store
        self.generator = generator
        self.min_words_threshold = min_words_threshold

    def extract_topics(
        self,
        source_filename: Optional[str] = None,
        file_hash: Optional[str] = None
    ) -> List[Topic]:
        """
        Extracts study topics for the specified document.

        Args:
            source_filename: Filename of the indexed document.
            file_hash: Hash of the indexed document.

        Returns:
            List of validated, grounded Topic objects.

        Raises:
            InsufficientContentForQuizError: If the document lacks sufficient content.
        """
        # Retrieve chunks for the specified document
        chunks = self.vector_store.get_document_chunks(
            file_hash=file_hash,
            source_filename=source_filename
        )

        doc_label = source_filename or (file_hash[:8] if file_hash else "Unknown document")

        if not chunks:
            raise InsufficientContentForQuizError(
                f"No indexed content found for document '{doc_label}'. "
                "Please verify the document has been successfully processed and indexed."
            )

        # Check total word count across chunks
        total_text = " ".join(c.get("content", "") for c in chunks)
        word_count = len(re.findall(r"\b\w+\b", total_text))

        if word_count < self.min_words_threshold:
            raise InsufficientContentForQuizError(
                f"Document '{doc_label}' contains only {word_count} words, which is below the "
                f"minimum threshold ({self.min_words_threshold} words) needed to discover study topics."
            )

        # Determine effective filename
        effective_filename = source_filename
        if not effective_filename and chunks:
            effective_filename = chunks[0].get("metadata", {}).get("source", "document.pdf")

        # Organize chunks by page
        pages_dict: Dict[int, List[Dict[str, Any]]] = {}
        for c in chunks:
            meta = c.get("metadata", {})
            page_num = int(meta.get("page", 1))
            pages_dict.setdefault(page_num, []).append(c)

        all_doc_pages = set(pages_dict.keys())

        # Attempt LLM-based discovery if generator has an API key
        if self.generator and self.generator.api_key:
            try:
                llm_topics = self._discover_topics_via_llm(
                    chunks=chunks,
                    pages_dict=pages_dict,
                    filename=effective_filename,
                    all_doc_pages=all_doc_pages
                )
                if llm_topics:
                    logger.info(
                        f"LLM successfully discovered {len(llm_topics)} topics for '{effective_filename}'."
                    )
                    return llm_topics
            except Exception as e:
                logger.warning(
                    f"LLM topic extraction failed for '{effective_filename}': {e}. "
                    "Falling back to deterministic heuristic extraction."
                )

        # Fallback: Deterministic heuristic extraction
        heuristic_topics = self._discover_topics_heuristically(
            pages_dict=pages_dict,
            filename=effective_filename
        )

        logger.info(
            f"Extracted {len(heuristic_topics)} topics via heuristic extractor for '{effective_filename}'."
        )
        return heuristic_topics

    def _discover_topics_via_llm(
        self,
        chunks: List[Dict[str, Any]],
        pages_dict: Dict[int, List[Dict[str, Any]]],
        filename: str,
        all_doc_pages: Set[int]
    ) -> List[Topic]:
        """Discovers grounded topics using Gemini API with validation."""
        # Build concise excerpts by page
        page_excerpts = []
        for p in sorted(pages_dict.keys()):
            combined = " ".join(c.get("content", "") for c in pages_dict[p])
            excerpt = combined[:600].replace("\n", " ").strip()
            page_excerpts.append(f"[Page {p}]: {excerpt}")

        excerpts_str = "\n\n".join(page_excerpts)
        user_prompt = TOPIC_DISCOVERY_USER_TEMPLATE.format(
            filename=filename,
            total_pages=len(pages_dict),
            excerpts=excerpts_str
        )

        response = self.generator._invoke_api_with_retry(
            user_content=user_prompt,
            system_instruction=TOPIC_DISCOVERY_SYSTEM_PROMPT
        )

        # Extract text response
        raw_text = ""
        if hasattr(response, "output_text") and isinstance(response.output_text, str):
            raw_text = response.output_text.strip()
        elif hasattr(response, "text") and isinstance(response.text, str):
            raw_text = response.text.strip()
        elif hasattr(response, "output_text") and response.output_text is not None:
            raw_text = str(response.output_text).strip()
        elif hasattr(response, "text") and response.text is not None:
            raw_text = str(response.text).strip()

        # Parse JSON
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        # Find JSON array
        start_idx = cleaned.find("[")
        end_idx = cleaned.rfind("]")
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx : end_idx + 1]

        data = json.loads(cleaned)
        if not isinstance(data, list):
            return []

        topics: List[Topic] = []
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            desc = str(item.get("description", "")).strip()
            raw_pages = item.get("source_pages", [])

            if not name or not desc:
                continue

            # Validate and filter pages
            valid_pages = []
            if isinstance(raw_pages, list):
                for p in raw_pages:
                    try:
                        p_int = int(p)
                        if p_int in all_doc_pages:
                            valid_pages.append(p_int)
                    except (ValueError, TypeError):
                        continue

            if not valid_pages:
                # If LLM didn't return valid pages, assign first page
                valid_pages = [min(all_doc_pages)]

            valid_pages = sorted(list(set(valid_pages)))

            # Count chunks covering these pages
            chunk_count = sum(
                len(pages_dict.get(p, [])) for p in valid_pages
            )

            topic_id = f"topic_{re.sub(r'[^a-z0-9]', '_', name.lower())[:20]}_{idx}"
            topics.append(
                Topic(
                    topic_id=topic_id,
                    name=name,
                    description=desc,
                    source_filename=filename,
                    source_pages=valid_pages,
                    chunk_count=chunk_count
                )
            )

        return topics

    def _discover_topics_heuristically(
        self,
        pages_dict: Dict[int, List[Dict[str, Any]]],
        filename: str
    ) -> List[Topic]:
        """
        Deterministic, robust topic discovery based on headings, colons,
        and prominent sentence structures in chunks.
        """
        topics: List[Topic] = []
        seen_names: Set[str] = set()

        for page_num in sorted(pages_dict.keys()):
            page_chunks = pages_dict[page_num]
            combined_text = "\n".join(c.get("content", "") for c in page_chunks).strip()
            lines = [line.strip() for line in combined_text.splitlines() if line.strip()]

            if not lines:
                continue

            # Check 1: Explicit heading ending with colon, e.g. "Quantum Computing Overview:"
            heading_found = False
            for line in lines[:5]:
                colon_match = re.match(r"^([A-Z][A-Za-z0-9\s,\-–\(\)]{3,60})\s*:\s*(.*)$", line)
                if colon_match:
                    candidate_name = colon_match.group(1).strip()
                    remainder = colon_match.group(2).strip()

                    norm_key = re.sub(r"[^a-z0-9]", "", candidate_name.lower())
                    if norm_key not in seen_names:
                        seen_names.add(norm_key)
                        # Build description
                        desc = remainder or (lines[1] if len(lines) > 1 else f"Concepts on {candidate_name}")
                        desc = desc[:180].strip()
                        topic_id = f"topic_p{page_num}_{len(topics)+1}"
                        topics.append(
                            Topic(
                                topic_id=topic_id,
                                name=candidate_name,
                                description=desc,
                                source_filename=filename,
                                source_pages=[page_num],
                                chunk_count=len(page_chunks)
                            )
                        )
                        heading_found = True
                        break

            if heading_found:
                continue

            # Check 2: First line as title if short (<= 60 chars) and capitalized
            first_line = lines[0]
            if len(first_line) <= 60 and first_line[0].isupper() and not first_line.endswith("."):
                candidate_name = first_line.rstrip(":")
                norm_key = re.sub(r"[^a-z0-9]", "", candidate_name.lower())
                if norm_key not in seen_names:
                    seen_names.add(norm_key)
                    desc = lines[1][:180] if len(lines) > 1 else f"Discussion of {candidate_name}"
                    topics.append(
                        Topic(
                            topic_id=f"topic_p{page_num}_{len(topics)+1}",
                            name=candidate_name,
                            description=desc,
                            source_filename=filename,
                            source_pages=[page_num],
                            chunk_count=len(page_chunks)
                        )
                    )
                    continue

            # Check 3: Fallback topic for the page based on first sentence
            first_sentence = re.split(r"(?<=[.!?])\s+", lines[0])[0]
            words = first_sentence.split()
            candidate_name = " ".join(words[:4]).rstrip(",.:;")
            if not candidate_name:
                candidate_name = f"Page {page_num} Overview"

            norm_key = re.sub(r"[^a-z0-9]", "", candidate_name.lower())
            if norm_key not in seen_names:
                seen_names.add(norm_key)
                topics.append(
                    Topic(
                        topic_id=f"topic_p{page_num}_{len(topics)+1}",
                        name=candidate_name,
                        description=first_sentence[:180],
                        source_filename=filename,
                        source_pages=[page_num],
                        chunk_count=len(page_chunks)
                    )
                )

        return topics

