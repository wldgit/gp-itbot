from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from app.app_logger import logger
from app.config import settings
from app.prompt_loader import load_prompt

VALID_DOC_TYPES = frozenset(
    {
        "faq",
        "instruction",
        "troubleshooting",
        "policy",
        "support_contacts",
        "reference",
        "mixed",
        "unknown",
    }
)

VALID_CHUNKING_STRATEGIES = frozenset(
    {
        "qa_pairs",
        "sections",
        "troubleshooting_blocks",
        "whole_document",
        "mixed_by_sections",
        "fallback_token_chunks",
    }
)

VALID_SECTION_TYPES = VALID_DOC_TYPES - {"mixed"}

DEFAULT_DOCUMENT_PROFILER_PROMPT = (
    "Ты анализируешь документ для базы знаний ИТ-ассистента. "
    "Верни только JSON с полями doc_type, chunking_strategy, max_chunk_tokens, "
    "overlap_tokens, confidence, signals, section_profiles."
)


@dataclass
class SectionProfile:
    heading: str
    section_type: str
    chunking_strategy: str


@dataclass
class DocumentProfile:
    doc_type: str
    chunking_strategy: str
    max_chunk_tokens: int
    overlap_tokens: int
    preserve_headings: bool = True
    repeat_document_title: bool = True
    repeat_section_title: bool = True
    repeat_question_in_split_answer: bool = True
    confidence: float = 0.0
    signals: list[str] = field(default_factory=list)
    section_profiles: list[SectionProfile] = field(default_factory=list)


@dataclass
class ProfilingResult:
    profile: DocumentProfile
    truncated_for_profiling: bool
    document_length: int


def fallback_profile() -> DocumentProfile:
    return DocumentProfile(
        doc_type="unknown",
        chunking_strategy="fallback_token_chunks",
        max_chunk_tokens=settings.default_chunk_max_tokens,
        overlap_tokens=settings.default_chunk_overlap_tokens,
        preserve_headings=True,
        repeat_document_title=True,
        repeat_section_title=True,
        repeat_question_in_split_answer=True,
        confidence=0.0,
        signals=["fallback profile"],
        section_profiles=[],
    )


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_tokens(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _normalize_profile(profile: DocumentProfile) -> DocumentProfile:
    if profile.doc_type == "faq" and profile.chunking_strategy != "qa_pairs":
        profile.chunking_strategy = "qa_pairs"
    if profile.doc_type == "support_contacts" and profile.chunking_strategy != "sections":
        profile.chunking_strategy = "sections"
    if profile.doc_type == "mixed" and profile.chunking_strategy != "mixed_by_sections":
        profile.chunking_strategy = "mixed_by_sections"
    if profile.doc_type == "unknown" and profile.chunking_strategy != "fallback_token_chunks":
        profile.chunking_strategy = "fallback_token_chunks"
    return profile


def validate_document_profile(raw: dict) -> DocumentProfile:
    doc_type = str(raw.get("doc_type", "unknown")).strip().lower()
    if doc_type not in VALID_DOC_TYPES:
        doc_type = "unknown"

    chunking_strategy = str(raw.get("chunking_strategy", "fallback_token_chunks")).strip().lower()
    if chunking_strategy not in VALID_CHUNKING_STRATEGIES:
        chunking_strategy = "fallback_token_chunks"

    try:
        confidence = _clamp_confidence(float(raw.get("confidence", 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0

    try:
        max_chunk_tokens = int(raw.get("max_chunk_tokens", settings.default_chunk_max_tokens))
    except (TypeError, ValueError):
        max_chunk_tokens = settings.default_chunk_max_tokens

    try:
        overlap_tokens = int(raw.get("overlap_tokens", settings.default_chunk_overlap_tokens))
    except (TypeError, ValueError):
        overlap_tokens = settings.default_chunk_overlap_tokens

    max_chunk_tokens = _clamp_tokens(max_chunk_tokens, 250, 1200)
    overlap_tokens = _clamp_tokens(overlap_tokens, 0, 200)
    max_overlap = max(0, int(max_chunk_tokens * 0.3))
    overlap_tokens = min(overlap_tokens, max_overlap)

    signals = raw.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    signals = [str(item) for item in signals[:10]]

    section_profiles: list[SectionProfile] = []
    raw_sections = raw.get("section_profiles") or []
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, dict):
                continue
            section_type = str(item.get("section_type", "unknown")).strip().lower()
            if section_type not in VALID_SECTION_TYPES:
                section_type = "unknown"
            sec_strategy = str(item.get("chunking_strategy", "sections")).strip().lower()
            if sec_strategy not in VALID_CHUNKING_STRATEGIES:
                sec_strategy = "sections"
            section_profiles.append(
                SectionProfile(
                    heading=str(item.get("heading", "")).strip(),
                    section_type=section_type,
                    chunking_strategy=sec_strategy,
                )
            )

    profile = DocumentProfile(
        doc_type=doc_type,
        chunking_strategy=chunking_strategy,
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
        preserve_headings=bool(raw.get("preserve_headings", True)),
        repeat_document_title=bool(raw.get("repeat_document_title", True)),
        repeat_section_title=bool(raw.get("repeat_section_title", True)),
        repeat_question_in_split_answer=bool(raw.get("repeat_question_in_split_answer", True)),
        confidence=confidence,
        signals=signals,
        section_profiles=section_profiles,
    )
    return _normalize_profile(profile)


def parse_profiler_response(raw: str) -> DocumentProfile | None:
    if not raw or not raw.strip():
        return None

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
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict):
        return None

    profile = validate_document_profile(payload)
    if profile.confidence < settings.document_profiler_confidence_threshold:
        logger.warning(
            "Document profiler confidence below threshold: %.2f < %.2f",
            profile.confidence,
            settings.document_profiler_confidence_threshold,
        )
        return None
    return profile


def _build_user_message(filename: str, text: str) -> tuple[str, bool]:
    truncated = False
    body = text
    if len(body) > settings.document_profiler_max_input_chars:
        body = body[: settings.document_profiler_max_input_chars]
        truncated = True
        logger.warning(
            "Document text truncated for profiling. filename=%s limit=%s",
            filename,
            settings.document_profiler_max_input_chars,
        )
    user_content = f"Имя файла:\n{filename}\n\nТекст документа:\n{body}"
    return user_content, truncated


def _call_profiler_model(client: OpenAI, system_prompt: str, user_content: str) -> str:
    kwargs: dict = {
        "model": settings.document_profiler_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if "gpt-5" not in settings.document_profiler_model:
        kwargs["temperature"] = 0

    try:
        response = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        err = str(exc).lower()
        if "response_format" in err or "unsupported" in err:
            response = client.chat.completions.create(**kwargs)
        else:
            raise
    return response.choices[0].message.content or ""


class DocumentProfilerService:
    def __init__(self) -> None:
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.system_prompt = load_prompt(
            settings.document_profiler_prompt_file,
            DEFAULT_DOCUMENT_PROFILER_PROMPT,
            "Document profiler prompt",
        )

    def profile_document(self, filename: str, text: str) -> ProfilingResult:
        document_length = len(text or "")
        if not settings.document_profiler_enabled:
            return ProfilingResult(
                profile=fallback_profile(),
                truncated_for_profiling=False,
                document_length=document_length,
            )

        user_content, truncated = _build_user_message(filename, text or "")
        try:
            raw = _call_profiler_model(self.openai_client, self.system_prompt, user_content)
            profile = parse_profiler_response(raw)
            if profile is None:
                logger.warning(
                    "Document profiler returned invalid or low-confidence JSON for %s",
                    filename,
                )
                return ProfilingResult(
                    profile=fallback_profile(),
                    truncated_for_profiling=truncated,
                    document_length=document_length,
                )
            return ProfilingResult(
                profile=profile,
                truncated_for_profiling=truncated,
                document_length=document_length,
            )
        except Exception:
            logger.warning("Document profiler API call failed for %s", filename, exc_info=True)
            return ProfilingResult(
                profile=fallback_profile(),
                truncated_for_profiling=truncated,
                document_length=document_length,
            )
