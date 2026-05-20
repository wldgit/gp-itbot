"""Manual check: profile + chunk one document. Usage: python scripts/check_document_profiler.py path/to/doc.md"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.ingest_docs import read_document  # noqa: E402
from ingest.text_cleaner import clean_text  # noqa: E402
from ingest.document_profiler import DocumentProfilerService  # noqa: E402
from ingest.document_chunker import chunk_document  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_document_profiler.py <path-to-document>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    text = clean_text(read_document(path))
    profiler = DocumentProfilerService()
    result = profiler.profile_document(path.name, text)
    profile = result.profile
    chunks = chunk_document(text, path.name, profile)

    print("=== Document profile ===")
    print(f"filename: {path.name}")
    print(f"document_length: {result.document_length}")
    print(f"truncated_for_profiling: {result.truncated_for_profiling}")
    print(f"doc_type: {profile.doc_type}")
    print(f"chunking_strategy: {profile.chunking_strategy}")
    print(f"max_chunk_tokens: {profile.max_chunk_tokens}")
    print(f"overlap_tokens: {profile.overlap_tokens}")
    print(f"confidence: {profile.confidence:.2f}")
    print(f"signals: {profile.signals}")
    print(f"chunks_count: {len(chunks)}\n")

    for idx, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {idx + 1} ---")
        print(chunk.text[:800])
        if len(chunk.text) > 800:
            print("...")
        print()


if __name__ == "__main__":
    main()
