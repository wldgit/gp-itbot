"""Show Chroma collection stats. Usage: python scripts/check_chroma_collection.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402

from app.config import settings  # noqa: E402


def main() -> None:
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(settings.chroma_collection_name)

    try:
        count = collection.count()
    except Exception as exc:
        print(f"Failed to get collection count: {exc}")
        count = None

    print(f"chroma_path: {settings.chroma_path}")
    print(f"collection_name: {settings.chroma_collection_name}")
    print(f"collection_count: {count}")

    try:
        sample = collection.peek(limit=5)
        documents = sample.get("documents") or []
        metadatas = sample.get("metadatas") or []
        print(f"\nSample records (up to 5): {len(documents)}")
        for idx, doc in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            source = meta.get("source", "unknown")
            section = meta.get("section", "")
            doc_type = meta.get("doc_type", "")
            preview = (doc or "")[:120].replace("\n", " ")
            print(f"- source={source} section={section} doc_type={doc_type}")
            print(f"  preview={preview}")
    except Exception as exc:
        print(f"\nCould not peek collection sample: {exc}")


if __name__ == "__main__":
    main()
