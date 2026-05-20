from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.app_logger import logger
from app.config import settings
from ingest.document_profiler import DocumentProfile, SectionProfile

MARKDOWN_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
QA_BLOCK_RE = re.compile(
    r"(?:^|\n)(?:Вопрос|вопрос)\s*:\s*(.+?)\n(?:Ответ|ответ)\s*:\s*(.+?)(?=\n(?:Вопрос|вопрос)\s*:|$)",
    re.DOTALL,
)
QA_QA_BLOCK_RE = re.compile(
    r"(?:^|\n)(?:Q|q)\s*:\s*(.+?)\n(?:A|a)\s*:\s*(.+?)(?=\n(?:Q|q)\s*:|$)",
    re.DOTALL,
)
NUMBERED_QUESTION_RE = re.compile(
    r"(?:^|\n)(\d+[\.\)])\s+(.+?\?)\s*\n([\s\S]+?)(?=\n\d+[\.\)]\s+.+?\?|\Z)",
)
@dataclass
class DocumentChunk:
    text: str
    section: str = ""
    question: str = ""
    extra_metadata: dict = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _basename(filename: str) -> str:
    return filename.replace("\\", "/").rsplit("/", 1)[-1]


def _document_title(text: str, filename: str) -> str:
    match = MARKDOWN_HEADER_RE.search(text)
    if match:
        return match.group(2).strip()
    return _basename(filename)


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _prefix_lines(
    profile: DocumentProfile,
    title: str,
    section: str = "",
    question: str = "",
) -> list[str]:
    lines: list[str] = []
    if profile.repeat_document_title and title:
        lines.append(f"Документ: {title}")
    if profile.repeat_section_title and section:
        if profile.doc_type == "troubleshooting" or profile.chunking_strategy == "troubleshooting_blocks":
            lines.append(f"Проблема/раздел: {section}")
        else:
            lines.append(f"Раздел: {section}")
    if question and (profile.repeat_question_in_split_answer or profile.chunking_strategy == "qa_pairs"):
        lines.append(f"Вопрос: {question}")
    return lines


def _pack_paragraphs(
    paragraphs: list[str],
    max_tokens: int,
    overlap_tokens: int,
    prefix: str = "",
) -> list[str]:
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = estimate_tokens(prefix)

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        body = "\n\n".join(current)
        chunks.append(f"{prefix}\n\n{body}".strip() if prefix else body)
        current = []
        current_tokens = estimate_tokens(prefix)

    for paragraph in paragraphs:
        para_tokens = estimate_tokens(paragraph)
        if current and current_tokens + para_tokens > max_tokens:
            previous_paras = list(current)
            flush()
            if overlap_tokens > 0 and previous_paras:
                overlap_paras: list[str] = []
                overlap_count = 0
                for prev_para in reversed(previous_paras):
                    overlap_count += estimate_tokens(prev_para)
                    overlap_paras.insert(0, prev_para)
                    if overlap_count >= overlap_tokens:
                        break
                current = overlap_paras
                current_tokens = estimate_tokens(prefix) + sum(estimate_tokens(p) for p in current)
            if para_tokens > max_tokens:
                chunks.append(f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph)
                current = []
                current_tokens = estimate_tokens(prefix)
                continue
        current.append(paragraph)
        current_tokens += para_tokens

    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def _format_section_body(heading: str, body: str) -> str:
    heading = (heading or "").strip()
    body = (body or "").strip()

    if heading and body:
        return f"## {heading}\n\n{body}"
    if body:
        return body
    return ""


def _strip_chunk_prefix(
    chunk_text: str,
    profile: DocumentProfile,
    title: str,
    section: str,
) -> str:
    prefix = "\n".join(_prefix_lines(profile, title, section))
    text = chunk_text.strip()
    if prefix and text.startswith(prefix):
        return text[len(prefix) :].lstrip("\n").strip()
    return text


def _merge_section_names(existing: str, title: str, extra_headings: list[str]) -> str:
    headings: list[str] = []
    if existing and existing != title:
        headings.extend(part.strip() for part in existing.split(";") if part.strip())
    for heading in extra_headings:
        if heading and heading not in headings:
            headings.append(heading)
    return "; ".join(headings) if headings else title


def _append_merged_section_chunk(
    result: list[DocumentChunk],
    profile: DocumentProfile,
    title: str,
    buffer_parts: list[str],
    buffer_headings: list[str],
    *,
    section_type: str = "",
) -> None:
    if not buffer_parts:
        return

    body = "\n\n".join(buffer_parts).strip()
    if not body:
        return

    section_name = "; ".join(buffer_headings) if buffer_headings else title
    prefix = "\n".join(_prefix_lines(profile, title, section_name))
    chunk_text = f"{prefix}\n\n{body}".strip() if prefix else body
    extra_metadata: dict = {}
    if section_type:
        extra_metadata["section_type"] = section_type
    result.append(
        DocumentChunk(text=chunk_text, section=section_name, extra_metadata=extra_metadata)
    )


def _has_markdown_headers(text: str) -> bool:
    return bool(re.search(r"^#{1,3}\s+", text, re.MULTILINE))


def _sections_to_text(sections: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for heading, body in sections:
        formatted = _format_section_body(heading, body)
        if formatted:
            parts.append(formatted)
    return "\n\n".join(parts).strip()


def _pack_plain_paragraphs(text: str, filename: str, profile: DocumentProfile) -> list[DocumentChunk]:
    title = _document_title(text, filename)
    prefix = "\n".join(_prefix_lines(profile, title))
    result: list[DocumentChunk] = []
    for chunk_text in _pack_paragraphs(
        _split_paragraphs(text),
        profile.max_chunk_tokens,
        profile.overlap_tokens,
        prefix=prefix,
    ):
        result.append(DocumentChunk(text=chunk_text))
    return result


def _merge_adjacent_small_chunks(
    chunks: list[DocumentChunk],
    profile: DocumentProfile,
    title: str,
) -> list[DocumentChunk]:
    if len(chunks) < 2:
        return chunks

    min_tokens = settings.section_min_chunk_tokens
    max_tokens = profile.max_chunk_tokens
    merged: list[DocumentChunk] = []

    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue

        prev = merged[-1]
        prev_type = prev.extra_metadata.get("section_type", "")
        curr_type = chunk.extra_metadata.get("section_type", "")
        if prev_type and curr_type and prev_type != curr_type:
            merged.append(chunk)
            continue

        prev_body = _strip_chunk_prefix(prev.text, profile, title, prev.section)
        curr_body = _strip_chunk_prefix(chunk.text, profile, title, chunk.section)
        curr_tokens = estimate_tokens(curr_body)
        combined_body = f"{prev_body}\n\n{curr_body}".strip()

        if curr_tokens < min_tokens and estimate_tokens(combined_body) <= max_tokens:
            extra_headings = [h.strip() for h in chunk.section.split(";") if h.strip()]
            if chunk.section == title:
                extra_headings = []
            section_name = _merge_section_names(prev.section, title, extra_headings)
            prefix = "\n".join(_prefix_lines(profile, title, section_name))
            chunk_text = f"{prefix}\n\n{combined_body}".strip() if prefix else combined_body
            extra_metadata = dict(prev.extra_metadata)
            merged[-1] = DocumentChunk(
                text=chunk_text,
                section=section_name,
                extra_metadata=extra_metadata,
            )
        else:
            merged.append(chunk)

    return merged


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    matches = list(MARKDOWN_HEADER_RE.finditer(text))
    if not matches:
        return [("", text.strip())] if text.strip() else []

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for idx, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))
    return sections


def _pack_markdown_sections(
    text: str,
    filename: str,
    profile: DocumentProfile,
    sections: list[tuple[str, str]] | None = None,
    *,
    section_type: str = "",
) -> list[DocumentChunk]:
    title = _document_title(text, filename)
    section_list = sections if sections is not None else _split_markdown_sections(text)
    result: list[DocumentChunk] = []
    buffer_parts: list[str] = []
    buffer_headings: list[str] = []
    min_tokens = settings.section_min_chunk_tokens
    max_tokens = profile.max_chunk_tokens
    resolved_section_type = section_type or profile.doc_type

    def flush_buffer(*, final: bool = False) -> None:
        nonlocal buffer_parts, buffer_headings
        if not buffer_parts:
            return

        body = "\n\n".join(buffer_parts).strip()
        if not body:
            buffer_parts = []
            buffer_headings = []
            return

        body_tokens = estimate_tokens(body)
        if final and body_tokens < min_tokens and result:
            prev = result[-1]
            prev_body = _strip_chunk_prefix(prev.text, profile, title, prev.section)
            merged_body = f"{prev_body}\n\n{body}".strip()
            if estimate_tokens(merged_body) <= max_tokens:
                section_name = _merge_section_names(prev.section, title, buffer_headings)
                prefix = "\n".join(_prefix_lines(profile, title, section_name))
                chunk_text = f"{prefix}\n\n{merged_body}".strip() if prefix else merged_body
                extra_metadata = dict(prev.extra_metadata)
                if resolved_section_type:
                    extra_metadata["section_type"] = resolved_section_type
                result[-1] = DocumentChunk(
                    text=chunk_text,
                    section=section_name,
                    extra_metadata=extra_metadata,
                )
                buffer_parts = []
                buffer_headings = []
                return

        _append_merged_section_chunk(
            result,
            profile,
            title,
            buffer_parts,
            buffer_headings,
            section_type=resolved_section_type,
        )
        buffer_parts = []
        buffer_headings = []

    for heading, body in section_list:
        section_text = _format_section_body(heading, body)
        if not section_text:
            continue

        section_tokens = estimate_tokens(section_text)

        if section_tokens > max_tokens:
            flush_buffer()
            prefix = "\n".join(_prefix_lines(profile, title, heading))
            for chunk_text in _pack_paragraphs(
                _split_paragraphs(body),
                max_tokens,
                profile.overlap_tokens,
                prefix=prefix,
            ):
                extra_metadata: dict = {}
                if resolved_section_type:
                    extra_metadata["section_type"] = resolved_section_type
                result.append(
                    DocumentChunk(
                        text=chunk_text,
                        section=heading,
                        extra_metadata=extra_metadata,
                    )
                )
            continue

        candidate_body = "\n\n".join(buffer_parts + [section_text]).strip()
        if buffer_parts and estimate_tokens(candidate_body) > max_tokens:
            flush_buffer()

        buffer_parts.append(section_text)
        if heading:
            buffer_headings.append(heading)

        if estimate_tokens("\n\n".join(buffer_parts)) >= min_tokens:
            flush_buffer()

    flush_buffer(final=True)
    return result


def _chunk_sections(
    text: str,
    filename: str,
    profile: DocumentProfile,
    section_label: str = "Раздел",
) -> list[DocumentChunk]:
    chunks = _pack_markdown_sections(text, filename, profile)
    if chunks:
        return chunks
    if text.strip():
        return _pack_plain_paragraphs(text, filename, profile)
    return []


def _chunk_fallback_token_chunks(
    text: str,
    filename: str,
    profile: DocumentProfile,
) -> list[DocumentChunk]:
    if _has_markdown_headers(text):
        chunks = _pack_markdown_sections(text, filename, profile)
        if chunks:
            return chunks
    if text.strip():
        return _pack_plain_paragraphs(text, filename, profile)
    return []


def _extract_qa_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for match in QA_BLOCK_RE.finditer(text):
        pairs.append((match.group(1).strip(), match.group(2).strip()))

    for match in QA_QA_BLOCK_RE.finditer(text):
        pairs.append((match.group(1).strip(), match.group(2).strip()))

    if pairs:
        return pairs

    for match in NUMBERED_QUESTION_RE.finditer(text):
        pairs.append((match.group(2).strip(), match.group(3).strip()))

    if pairs:
        return pairs

    for heading, body in _split_markdown_sections(text):
        if "?" in heading:
            pairs.append((heading, body))

    return pairs


def _chunk_qa_pairs(text: str, filename: str, profile: DocumentProfile) -> list[DocumentChunk]:
    title = _document_title(text, filename)
    pairs = _extract_qa_pairs(text)
    if not pairs:
        return _chunk_sections(text, filename, profile)

    result: list[DocumentChunk] = []
    for question, answer in pairs:
        prefix = "\n".join(_prefix_lines(profile, title, question=question))
        answer_paragraphs = _split_paragraphs(answer)
        answer_tokens = estimate_tokens(answer)
        if answer_tokens <= profile.max_chunk_tokens - estimate_tokens(prefix):
            chunk_text = f"{prefix}\n\n{answer}".strip()
            result.append(DocumentChunk(text=chunk_text, section="", question=question))
            continue

        for chunk_text in _pack_paragraphs(
            answer_paragraphs,
            profile.max_chunk_tokens,
            profile.overlap_tokens,
            prefix=prefix,
        ):
            result.append(DocumentChunk(text=chunk_text, section="", question=question))

    return result or _chunk_fallback_token_chunks(text, filename, profile)


def _chunk_troubleshooting_blocks(
    text: str,
    filename: str,
    profile: DocumentProfile,
) -> list[DocumentChunk]:
    ts_profile = DocumentProfile(
        doc_type="troubleshooting",
        chunking_strategy=profile.chunking_strategy,
        max_chunk_tokens=profile.max_chunk_tokens,
        overlap_tokens=profile.overlap_tokens,
        preserve_headings=profile.preserve_headings,
        repeat_document_title=profile.repeat_document_title,
        repeat_section_title=profile.repeat_section_title,
        repeat_question_in_split_answer=profile.repeat_question_in_split_answer,
        confidence=profile.confidence,
        signals=profile.signals,
        section_profiles=profile.section_profiles,
    )
    if _has_markdown_headers(text):
        chunks = _pack_markdown_sections(text, filename, ts_profile, section_type="troubleshooting")
        if chunks:
            return chunks
    return _chunk_fallback_token_chunks(text, filename, ts_profile)


def _chunk_whole_document(text: str, filename: str, profile: DocumentProfile) -> list[DocumentChunk]:
    title = _document_title(text, filename)
    if estimate_tokens(text) <= profile.max_chunk_tokens:
        prefix = "\n".join(_prefix_lines(profile, title))
        chunk_text = f"{prefix}\n\n{text}".strip() if prefix else text.strip()
        return [DocumentChunk(text=chunk_text)]
    return _chunk_fallback_token_chunks(text, filename, profile)


def _section_profile_map(profile: DocumentProfile) -> dict[str, SectionProfile]:
    mapping: dict[str, SectionProfile] = {}
    for section in profile.section_profiles:
        if section.heading:
            mapping[section.heading.lower()] = section
    return mapping


def _profile_for_section_type(
    profile: DocumentProfile,
    section_type: str,
    chunking_strategy: str = "sections",
) -> DocumentProfile:
    return DocumentProfile(
        doc_type=section_type,
        chunking_strategy=chunking_strategy,
        max_chunk_tokens=profile.max_chunk_tokens,
        overlap_tokens=profile.overlap_tokens,
        preserve_headings=profile.preserve_headings,
        repeat_document_title=profile.repeat_document_title,
        repeat_section_title=profile.repeat_section_title,
        repeat_question_in_split_answer=profile.repeat_question_in_split_answer,
        confidence=profile.confidence,
        signals=profile.signals,
        section_profiles=[],
    )


def _chunk_mixed_by_sections(text: str, filename: str, profile: DocumentProfile) -> list[DocumentChunk]:
    title = _document_title(text, filename)

    if not profile.section_profiles:
        return _pack_markdown_sections(text, filename, profile)

    section_map = _section_profile_map(profile)
    sections = _split_markdown_sections(text)
    result: list[DocumentChunk] = []

    pack_buffer: list[tuple[str, str]] = []
    pack_section_type = profile.doc_type

    def flush_pack_buffer() -> None:
        nonlocal pack_buffer, pack_section_type
        if not pack_buffer:
            return
        block_text = _sections_to_text(pack_buffer)
        if not block_text:
            pack_buffer = []
            return
        section_profile = _profile_for_section_type(profile, pack_section_type)
        result.extend(
            _pack_markdown_sections(
                block_text,
                filename,
                section_profile,
                section_type=pack_section_type,
            )
        )
        pack_buffer = []

    for heading, body in sections:
        sec_profile = section_map.get(heading.lower())
        if sec_profile and sec_profile.chunking_strategy == "qa_pairs":
            flush_pack_buffer()
            section_doc = _profile_for_section_type(profile, sec_profile.section_type, "qa_pairs")
            full_section_text = _format_section_body(heading, body) or body
            qa_chunks = _chunk_qa_pairs(full_section_text, filename, section_doc)
            for chunk in qa_chunks:
                extra_metadata = dict(chunk.extra_metadata)
                extra_metadata["section_type"] = sec_profile.section_type
                result.append(
                    DocumentChunk(
                        text=chunk.text,
                        section=chunk.section,
                        question=chunk.question,
                        extra_metadata=extra_metadata,
                    )
                )
            continue

        section_type = sec_profile.section_type if sec_profile else profile.doc_type
        if pack_buffer and section_type != pack_section_type:
            flush_pack_buffer()
        pack_section_type = section_type
        pack_buffer.append((heading, body))

    flush_pack_buffer()

    if not result:
        return _pack_markdown_sections(text, filename, profile)

    return _merge_adjacent_small_chunks(result, profile, title)


def chunk_document(text: str, filename: str, profile: DocumentProfile) -> list[DocumentChunk]:
    if not text or not text.strip():
        return []

    strategy = profile.chunking_strategy
    if strategy == "qa_pairs":
        chunks = _chunk_qa_pairs(text, filename, profile)
    elif strategy == "sections":
        chunks = _chunk_sections(text, filename, profile)
    elif strategy == "troubleshooting_blocks":
        chunks = _chunk_troubleshooting_blocks(text, filename, profile)
    elif strategy == "whole_document":
        chunks = _chunk_whole_document(text, filename, profile)
    elif strategy == "mixed_by_sections":
        chunks = _chunk_mixed_by_sections(text, filename, profile)
    elif strategy == "fallback_token_chunks":
        chunks = _chunk_fallback_token_chunks(text, filename, profile)
    else:
        logger.warning(
            "Unknown chunking strategy: %s for %s, using fallback_token_chunks",
            strategy,
            filename,
        )
        chunks = _chunk_fallback_token_chunks(text, filename, profile)

    if not chunks:
        chunks = _chunk_fallback_token_chunks(text, filename, profile)
    return chunks
