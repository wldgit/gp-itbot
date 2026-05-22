import time
from dataclasses import dataclass
from typing import Any

import chromadb
from openai import OpenAI

from app.app_logger import logger
from app.config import settings
from app.logging_service import log_runtime_event
from app.prompt_loader import load_system_prompt
from app.support_messages import format_no_context_fallback
from app.text_safety import mask_sensitive_data


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]
    found_context: bool
    response_time_ms: int


def dedupe_sources(sources: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in sources:
        source = (raw or "").strip()
        if not source or source in seen:
            continue
        seen.add(source)
        result.append(source)
    return result


def append_sources_to_answer(answer: str, sources: list[str]) -> str:
    unique_sources = dedupe_sources(sources)
    if not unique_sources:
        return answer
    lines = ["Использованные документы:"]
    lines.extend(f"- {source}" for source in unique_sources)
    block = "\n".join(lines)
    body = (answer or "").rstrip()
    if not body:
        return block
    return f"{body}\n\n{block}"


CHUNK_PREVIEW_MAX_CHARS = 120


def chunk_text_preview(text: str, max_chars: int = CHUNK_PREVIEW_MAX_CHARS) -> str:
    safe = mask_sensitive_data(text or "")
    one_line = " ".join(safe.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[:max_chars] + "..."


def build_retrieval_results_for_log(
    docs: list[str],
    metadatas: list[dict[str, Any]],
    distances: list[float | None],
    threshold: float,
) -> list[dict[str, Any]]:
    """One entry per raw Chroma hit (query order), for INFO logs and runtime_events."""
    retrieval_results: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        distance = distances[idx] if idx < len(distances) else None
        relevance_score = distance_to_relevance_score(distance)
        accepted = relevance_score >= threshold
        source = meta.get("source", "unknown")
        section = meta.get("section", "") or ""
        preview = chunk_text_preview(doc)

        logger.info(
            "RAG chunk result. rank=%s accepted=%s source=%s section=%r distance=%s "
            "relevance_score=%.3f threshold=%.3f preview=%r",
            idx + 1,
            accepted,
            source,
            section,
            distance,
            relevance_score,
            threshold,
            preview,
        )
        retrieval_results.append(
            {
                "rank": idx + 1,
                "accepted": accepted,
                "source": source,
                "section": section,
                "distance": distance,
                "relevance_score": relevance_score,
                "preview": preview,
            }
        )
    return retrieval_results


def distance_to_relevance_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if distance < 0:
        distance = 0.0
    return 1.0 / (1.0 + distance)


def filter_chunks_by_relevance(
    docs: list[str],
    metadatas: list[dict[str, Any]],
    distances: list[float | None],
    threshold: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    min_score = settings.min_relevance_score if threshold is None else threshold
    accepted_results: list[dict[str, Any]] = []
    rejected_results: list[dict[str, Any]] = []

    for idx, doc in enumerate(docs):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        distance = distances[idx] if idx < len(distances) else None
        relevance_score = distance_to_relevance_score(distance)
        source = meta.get("source", "unknown")
        section = meta.get("section", "") or ""

        item = {
            "document": doc,
            "metadata": meta,
            "distance": distance,
            "relevance_score": relevance_score,
            "source": source,
            "section": section,
        }

        if relevance_score >= min_score:
            accepted_results.append(item)
            logger.debug(
                "RAG chunk accepted. source=%s section=%s distance=%s relevance_score=%.3f",
                source,
                section,
                distance,
                relevance_score,
            )
        else:
            rejected_results.append(item)
            logger.debug(
                "RAG chunk rejected. source=%s section=%s distance=%s relevance_score=%.3f threshold=%.3f",
                source,
                section,
                distance,
                relevance_score,
                min_score,
            )

    accepted_results.sort(key=lambda item: item["relevance_score"], reverse=True)
    return accepted_results, rejected_results


def format_context_for_model(item: dict[str, Any]) -> str:
    source = item["source"]
    section = item.get("section") or ""
    doc = item["document"]
    lines = [f"Источник: {source}"]
    if section:
        lines.append(f"Раздел: {section}")
    lines.append(f"Фрагмент:\n{doc}")
    return "\n".join(lines)


class RagService:
    def __init__(self):
        self.system_prompt = load_system_prompt(settings.system_prompt_file)
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            settings.chroma_collection_name
        )

        logger.info(
            "RagService initialized. chroma_path=%s collection=%s model=%s embedding_model=%s",
            settings.chroma_path,
            settings.chroma_collection_name,
            settings.openai_model,
            settings.openai_embedding_model,
        )
        log_runtime_event(
            "INFO",
            "rag_initialized",
            "RAG service initialized",
            {
                "chroma_path": settings.chroma_path,
                "collection_name": settings.chroma_collection_name,
                "model": settings.openai_model,
                "embedding_model": settings.openai_embedding_model,
            },
        )

    def _embed(self, text: str) -> list[float]:
        logger.debug("Creating embedding. chars=%s", len(text))
        response = self.openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _query_chroma(self, question: str) -> tuple[list[str], list[dict[str, Any]], list[float | None]]:
        safe_question = mask_sensitive_data(question)
        embedding = self._embed(safe_question)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=settings.top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []
        return docs, metadatas, distances

    def search_context(self, question: str) -> tuple[list[str], list[str], bool]:
        started = time.monotonic()

        try:
            docs, metadatas, distances = self._query_chroma(question)
            threshold = settings.min_relevance_score
            accepted_results, rejected_results = filter_chunks_by_relevance(
                docs, metadatas, distances, threshold=threshold
            )
            retrieval_results = build_retrieval_results_for_log(
                docs, metadatas, distances, threshold
            )

            logger.info(
                "RAG retrieval completed. raw_results=%s accepted=%s rejected=%s min_relevance_score=%.3f",
                len(docs),
                len(accepted_results),
                len(rejected_results),
                threshold,
            )

            if len(docs) > 0 and len(accepted_results) == 0:
                logger.warning(
                    "RAG search returned chunks, but all were below relevance threshold. "
                    "raw_results=%s threshold=%.3f",
                    len(docs),
                    threshold,
                )

            contexts = [format_context_for_model(item) for item in accepted_results]
            sources = dedupe_sources([item["source"] for item in accepted_results])
            found = len(accepted_results) > 0
            elapsed = int((time.monotonic() - started) * 1000)

            logger.info(
                "RAG search completed. found=%s, chunks=%s, elapsed_ms=%s, sources=%s",
                found,
                len(contexts),
                elapsed,
                sources,
            )
            log_runtime_event(
                "INFO",
                "rag_search_completed",
                "RAG search completed",
                {
                    "found": found,
                    "raw_chunks": len(docs),
                    "accepted_chunks": len(accepted_results),
                    "rejected_chunks": len(rejected_results),
                    "min_relevance_score": threshold,
                    "accepted_sources": sources,
                    "retrieval_results": retrieval_results,
                    "elapsed_ms": elapsed,
                },
            )
            return contexts, sources, found

        except Exception as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            logger.exception("RAG search failed. elapsed_ms=%s", elapsed)
            log_runtime_event(
                "ERROR",
                "rag_search_failed",
                str(exc),
                {"elapsed_ms": elapsed},
            )
            raise

    def debug_search_context(self, question: str) -> list[dict[str, Any]]:
        docs, metadatas, distances = self._query_chroma(question)
        accepted_results, rejected_results = filter_chunks_by_relevance(
            docs, metadatas, distances
        )

        debug_rows: list[dict[str, Any]] = []
        for item in accepted_results:
            debug_rows.append(
                {
                    "source": item["source"],
                    "section": item["section"],
                    "distance": item["distance"],
                    "relevance_score": item["relevance_score"],
                    "accepted": True,
                    "text_preview": chunk_text_preview(item["document"] or "", max_chars=300),
                }
            )
        for item in rejected_results:
            debug_rows.append(
                {
                    "source": item["source"],
                    "section": item["section"],
                    "distance": item["distance"],
                    "relevance_score": item["relevance_score"],
                    "accepted": False,
                    "text_preview": chunk_text_preview(item["document"] or "", max_chars=300),
                }
            )
        return debug_rows

    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> RagAnswer:
        started = time.monotonic()
        contexts, sources, found_context = self.search_context(question)

        if not found_context:
            elapsed = int((time.monotonic() - started) * 1000)
            logger.info("No context found for question. elapsed_ms=%s", elapsed)
            return RagAnswer(
                answer=format_no_context_fallback(),
                sources=[],
                found_context=False,
                response_time_ms=elapsed,
            )

        context_block = "\n\n---\n\n".join(contexts)
        user_prompt = f"""Вопрос пользователя:
{mask_sensitive_data(question)}

Найденные фрагменты базы знаний:
{context_block}

Сформируй ответ для пользователя.
"""

        prior: list[dict[str, str]] = list(history or [])
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            *prior,
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0.2,
            )

            answer = response.choices[0].message.content or ""
            answer = append_sources_to_answer(answer, sources)
            elapsed = int((time.monotonic() - started) * 1000)

            logger.info("Answer generated. elapsed_ms=%s, sources=%s", elapsed, sources)
            log_runtime_event(
                "INFO",
                "answer_generated",
                "Answer generated successfully",
                {
                    "elapsed_ms": elapsed,
                    "sources": sources,
                    "answer_chars": len(answer),
                },
            )
            return RagAnswer(
                answer=answer,
                sources=sources,
                found_context=True,
                response_time_ms=elapsed,
            )

        except Exception as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            logger.exception("Answer generation failed. elapsed_ms=%s", elapsed)
            log_runtime_event(
                "ERROR",
                "answer_generation_failed",
                str(exc),
                {
                    "elapsed_ms": elapsed,
                    "sources": sources,
                },
            )
            raise
