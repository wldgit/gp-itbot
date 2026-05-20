"""Debug RAG retrieval without answer generation. Usage: python scripts/check_retrieval.py \"your question\""""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.rag_service import RagService  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/check_retrieval.py "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Question is empty.")
        sys.exit(1)

    service = RagService()
    rows = service.debug_search_context(question)

    print(f"Question: {question}")
    print(f"MIN_RELEVANCE_SCORE: {settings.min_relevance_score}")
    print(f"TOP_K: {settings.top_k}")
    print(f"Results: {len(rows)}\n")

    for idx, row in enumerate(rows, start=1):
        status = "ACCEPTED" if row["accepted"] else "REJECTED"
        print(f"--- [{idx}] {status} ---")
        print(f"source: {row['source']}")
        print(f"section: {row['section']}")
        print(f"distance: {row['distance']}")
        print(f"relevance_score: {row['relevance_score']:.3f}")
        print(f"preview: {row['text_preview']}")
        print()


if __name__ == "__main__":
    main()
