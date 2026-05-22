import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

import chromadb
from openai import OpenAI

from app.app_logger import logger
from app.config import settings
from app.logging_service import log_runtime_event
from app.prompt_loader import load_system_prompt
from app.support_messages import format_no_context_fallback, format_no_direct_answer_fallback
from app.text_safety import mask_sensitive_data

RetrievalQuality = Literal["no_context", "borderline", "confident"]

GROUNDEDNESS_SYSTEM_PROMPT = """Определи, содержит ли контекст прямой ответ на вопрос пользователя.
Прямой ответ означает, что в контексте явно описана нужная проблема, технология или процедура.
Если контекст только похож по общей теме, но не содержит прямой инструкции, верни has_direct_answer=false.
Не используй общие знания.

Верни только JSON без пояснений и без Markdown:
{
  "has_direct_answer": true,
  "reason": "коротко"
}"""


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]
    found_context: bool
    response_time_ms: int


@dataclass
class RetrievalOutcome:
    contexts: list[str]
    sources: list[str]
    accepted_results: list[dict[str, Any]]
    accepted_count: int
    best_score: float
    quality: RetrievalQuality
    retrieval_results: list[dict[str, Any]]


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
    min_score = settings.rag_min_score if threshold is None else threshold
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


def evaluate_retrieval_quality(
    accepted_results: list[dict[str, Any]],
) -> tuple[RetrievalQuality, float]:
    """Accepted chunks are already >= RAG_MIN_SCORE."""
    if not accepted_results:
        return "no_context", 0.0

    best_score = max(item["relevance_score"] for item in accepted_results)
    if best_score >= settings.rag_confident_score:
        return "confident", best_score
    return "borderline", best_score


def parse_groundedness_response(raw: str) -> tuple[bool, str]:
    if not raw or not raw.strip():
        return False, "empty response"

    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return False, "invalid JSON"
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return False, "invalid JSON"

    has_direct = bool(payload.get("has_direct_answer", False))
    reason = str(payload.get("reason", "")).strip() or "no reason"
    return has_direct, reason


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

    def _retrieve(self, question: str) -> RetrievalOutcome:
        started = time.monotonic()

        docs, metadatas, distances = self._query_chroma(question)
        min_score = settings.rag_min_score
        accepted_results, rejected_results = filter_chunks_by_relevance(
            docs, metadatas, distances, threshold=min_score
        )
        retrieval_results = build_retrieval_results_for_log(
            docs, metadatas, distances, min_score
        )

        accepted_count = len(accepted_results)
        sources = dedupe_sources([item["source"] for item in accepted_results])
        quality, best_score = evaluate_retrieval_quality(accepted_results)
        contexts = [format_context_for_model(item) for item in accepted_results]
        elapsed = int((time.monotonic() - started) * 1000)

        logger.info(
            "RAG retrieval completed. mode=%s raw_results=%s accepted=%s rejected=%s "
            "RAG_MIN_SCORE=%.3f RAG_CONFIDENT_SCORE=%.3f best_score=%.3f sources=%s",
            quality,
            len(docs),
            accepted_count,
            len(rejected_results),
            min_score,
            settings.rag_confident_score,
            best_score,
            sources,
        )

        if len(docs) > 0 and accepted_count == 0:
            logger.warning(
                "RAG no_context: all chunks below RAG_MIN_SCORE. raw_results=%s threshold=%.3f",
                len(docs),
                min_score,
            )

        log_runtime_event(
            "INFO",
            "rag_search_completed",
            "RAG search completed",
            {
                "mode": quality,
                "best_score": best_score,
                "accepted_count": accepted_count,
                "raw_chunks": len(docs),
                "accepted_chunks": accepted_count,
                "rejected_chunks": len(rejected_results),
                "rag_min_score": min_score,
                "rag_confident_score": settings.rag_confident_score,
                "accepted_sources": sources,
                "retrieval_results": retrieval_results,
                "elapsed_ms": elapsed,
            },
        )

        return RetrievalOutcome(
            contexts=contexts,
            sources=sources,
            accepted_results=accepted_results,
            accepted_count=accepted_count,
            best_score=best_score,
            quality=quality,
            retrieval_results=retrieval_results,
        )

    def search_context(self, question: str) -> tuple[list[str], list[str], bool]:
        try:
            outcome = self._retrieve(question)
            found = outcome.quality == "confident"
            return outcome.contexts, outcome.sources, found
        except Exception as exc:
            logger.exception("RAG search failed.")
            log_runtime_event("ERROR", "rag_search_failed", str(exc), {})
            raise

    def _check_context_groundedness(
        self,
        question: str,
        contexts: list[str],
    ) -> tuple[bool, str]:
        context_block = "\n\n---\n\n".join(contexts)
        user_prompt = f"""Вопрос пользователя:
{mask_sensitive_data(question)}

Контекст из базы знаний:
{context_block}
"""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": GROUNDEDNESS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            return parse_groundedness_response(raw)
        except Exception as exc:
            err = str(exc).lower()
            if "response_format" in err or "unsupported" in err:
                response = self.openai_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": GROUNDEDNESS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
                raw = response.choices[0].message.content or ""
                return parse_groundedness_response(raw)
            logger.exception("Groundedness check failed")
            return False, str(exc)

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

        try:
            outcome = self._retrieve(question)
        except Exception:
            raise

        if outcome.quality == "no_context":
            elapsed = int((time.monotonic() - started) * 1000)
            logger.info(
                "RAG no_context. best_score=%.3f RAG_MIN_SCORE=%.3f elapsed_ms=%s",
                outcome.best_score,
                settings.rag_min_score,
                elapsed,
            )
            return RagAnswer(
                answer=format_no_context_fallback(),
                sources=[],
                found_context=False,
                response_time_ms=elapsed,
            )

        if outcome.quality == "borderline":
            has_direct, reason = self._check_context_groundedness(question, outcome.contexts)
            logger.info(
                "RAG borderline groundedness. has_direct_answer=%s best_score=%.3f "
                "RAG_MIN_SCORE=%.3f RAG_CONFIDENT_SCORE=%.3f sources=%s reason=%s",
                has_direct,
                outcome.best_score,
                settings.rag_min_score,
                settings.rag_confident_score,
                outcome.sources,
                reason,
            )
            if not has_direct:
                elapsed = int((time.monotonic() - started) * 1000)
                log_runtime_event(
                    "INFO",
                    "rag_borderline_rejected",
                    "Borderline RAG context rejected by groundedness check",
                    {
                        "mode": "no_context",
                        "has_direct_answer": False,
                        "best_score": outcome.best_score,
                        "sources": outcome.sources,
                        "reason": reason,
                        "elapsed_ms": elapsed,
                    },
                )
                return RagAnswer(
                    answer=format_no_direct_answer_fallback(),
                    sources=[],
                    found_context=False,
                    response_time_ms=elapsed,
                )
            log_runtime_event(
                "INFO",
                "rag_borderline_accepted",
                "Borderline RAG context passed groundedness check",
                {
                    "mode": "borderline",
                    "has_direct_answer": True,
                    "best_score": outcome.best_score,
                    "sources": outcome.sources,
                    "reason": reason,
                },
            )

        contexts = outcome.contexts
        sources = outcome.sources
        context_block = "\n\n---\n\n".join(contexts)
        user_prompt = f"""Вопрос пользователя:
{mask_sensitive_data(question)}

Найденные фрагменты базы знаний:
{context_block}

Сформируй ответ для пользователя только на основе этих фрагментов.
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
