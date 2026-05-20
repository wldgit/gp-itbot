import hashlib
import json
import sys
from pathlib import Path

import chromadb
from docx import Document
from openai import OpenAI
from pypdf import PdfReader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.app_logger import logger
from app.config import PROJECT_ROOT, settings
from ingest.document_chunker import DocumentChunk, chunk_document
from ingest.document_profiler import DocumentProfilerService, fallback_profile
from app.logging_service import log_runtime_event
from ingest.text_cleaner import clean_text


def read_txt_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_document(path: Path) -> str:
    if path.suffix.lower() in [".txt", ".md"]:
        return read_txt_md(path)

    if path.suffix.lower() == ".docx":
        return read_docx(path)

    if path.suffix.lower() == ".pdf":
        return read_pdf(path)

    return ""


def make_id(source: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}:{chunk_index}:{text[:100]}".encode("utf-8")).hexdigest()[:24]
    return f"{source}-{chunk_index}-{digest}".replace("/", "_")


def _document_title_from_text(text: str, filename: str) -> str:
    import re

    match = re.search(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return filename


def _build_chunk_metadata(
    source: str,
    chunk_index: int,
    chunk: DocumentChunk,
    profile,
    title: str,
) -> dict:
    metadata = {
        "source": source,
        "chunk_index": chunk_index,
        "doc_type": profile.doc_type,
        "chunking_strategy": profile.chunking_strategy,
        "profile_confidence": profile.confidence,
        "section": chunk.section or "",
        "question": chunk.question or "",
        "title": title,
    }
    metadata.update(chunk.extra_metadata)
    return metadata


def get_ingest_collection(chroma_client: chromadb.PersistentClient):
    collection_name = settings.chroma_collection_name

    if settings.recreate_chroma_collection:
        logger.warning(
            "Recreating Chroma collection before ingest. collection=%s path=%s",
            collection_name,
            settings.chroma_path,
        )
        try:
            chroma_client.delete_collection(collection_name)
            logger.info("Deleted existing Chroma collection: %s", collection_name)
        except Exception as exc:
            logger.info(
                "Chroma collection was not deleted, probably does not exist yet. "
                "collection=%s error=%s",
                collection_name,
                str(exc),
            )

    collection = chroma_client.get_or_create_collection(collection_name)
    return collection


def _write_preprocessing_report(report_path: Path, report: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    logger.info(
        "Starting document ingestion. docs_path=%s chroma_path=%s collection=%s "
        "recreate_collection=%s profiler_enabled=%s",
        settings.docs_path,
        settings.chroma_path,
        settings.chroma_collection_name,
        settings.recreate_chroma_collection,
        settings.document_profiler_enabled,
    )

    openai_client = OpenAI(api_key=settings.openai_api_key)
    chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = get_ingest_collection(chroma_client)
    profiler = DocumentProfilerService()

    docs_path = Path(settings.docs_path)
    files = [
        path
        for path in docs_path.rglob("*")
        if path.suffix.lower() in [".txt", ".md", ".docx", ".pdf"]
    ]

    logger.info("Documents found: %s", len(files))
    total_chunks = 0
    report_entries: list[dict] = []

    for path in files:
        source = str(path.relative_to(docs_path))
        try:
            raw_text = read_document(path)
            cleaned_text = clean_text(raw_text)

            if settings.document_profiler_enabled:
                profiling = profiler.profile_document(path.name, cleaned_text)
                profile = profiling.profile
                truncated_for_profiling = profiling.truncated_for_profiling
                document_length = profiling.document_length
            else:
                profile = fallback_profile()
                truncated_for_profiling = False
                document_length = len(cleaned_text)

            title = _document_title_from_text(cleaned_text, path.name)
            doc_chunks = chunk_document(cleaned_text, path.name, profile)

            logger.info(
                "Processed document. filename=%s document_length=%s truncated_for_profiling=%s "
                "doc_type=%s chunking_strategy=%s max_chunk_tokens=%s overlap_tokens=%s "
                "confidence=%.2f signals=%s chunks_count=%s",
                path.name,
                document_length,
                truncated_for_profiling,
                profile.doc_type,
                profile.chunking_strategy,
                profile.max_chunk_tokens,
                profile.overlap_tokens,
                profile.confidence,
                profile.signals,
                len(doc_chunks),
            )

            report_entries.append(
                {
                    "filename": path.name,
                    "collection_name": settings.chroma_collection_name,
                    "recreate_chroma_collection": settings.recreate_chroma_collection,
                    "document_length": document_length,
                    "truncated_for_profiling": truncated_for_profiling,
                    "doc_type": profile.doc_type,
                    "chunking_strategy": profile.chunking_strategy,
                    "max_chunk_tokens": profile.max_chunk_tokens,
                    "overlap_tokens": profile.overlap_tokens,
                    "confidence": profile.confidence,
                    "signals": profile.signals,
                    "chunks_count": len(doc_chunks),
                }
            )

            for index, chunk in enumerate(doc_chunks):
                embedding = openai_client.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=chunk.text,
                ).data[0].embedding

                metadata = _build_chunk_metadata(source, index, chunk, profile, title)
                collection.upsert(
                    ids=[make_id(source, index, chunk.text)],
                    documents=[chunk.text],
                    embeddings=[embedding],
                    metadatas=[metadata],
                )
                total_chunks += 1

        except Exception as exc:
            logger.exception("Failed to ingest document: %s", path)
            log_runtime_event(
                "ERROR",
                "ingest_document_failed",
                str(exc),
                {"path": str(path)},
            )

    try:
        collection_count = collection.count()
    except Exception:
        collection_count = None

    report_payload = {
        "summary": {
            "collection_name": settings.chroma_collection_name,
            "recreate_chroma_collection": settings.recreate_chroma_collection,
            "files_count": len(files),
            "chunks_indexed_this_run": total_chunks,
            "collection_count": collection_count,
        },
        "documents": report_entries,
    }

    report_path = Path(settings.preprocessing_report_path)
    try:
        _write_preprocessing_report(report_path, report_payload)
        logger.info("Preprocessing report saved: %s", report_path)
    except Exception:
        fallback_report = PROJECT_ROOT / "logs" / "preprocessing_report.json"
        logger.warning("Failed to save report to %s, trying %s", report_path, fallback_report)
        _write_preprocessing_report(fallback_report, report_payload)

    ingest_summary_path = PROJECT_ROOT / "logs" / "ingest_summary.json"
    try:
        _write_preprocessing_report(ingest_summary_path, report_payload["summary"])
        logger.info("Ingest summary saved: %s", ingest_summary_path)
    except Exception:
        logger.warning("Failed to save ingest summary to %s", ingest_summary_path)

    logger.info(
        "Document ingestion completed. files=%s chunks_indexed_this_run=%s collection_count=%s",
        len(files),
        total_chunks,
        collection_count,
    )
    log_runtime_event(
        "INFO",
        "ingest_completed",
        "Document ingestion completed",
        {
            "files": len(files),
            "collection_name": settings.chroma_collection_name,
            "recreate_chroma_collection": settings.recreate_chroma_collection,
            "chunks_indexed_this_run": total_chunks,
            "collection_count": collection_count,
        },
    )


if __name__ == "__main__":
    main()
