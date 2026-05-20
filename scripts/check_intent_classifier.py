"""Manual check for intent classifier. Usage: python scripts/check_intent_classifier.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.intent_service import IntentService, resolve_intent_decision  # noqa: E402

SAMPLES = [
    ("привет", "greeting -> no RAG"),
    ("спасибо", "greeting -> no RAG"),
    ("почему трава зеленая", "out_of_scope -> no RAG"),
    ("расскажи анекдот", "out_of_scope -> no RAG"),
    ("не работает монитор", "it_support_request -> RAG"),
    ("как обратиться в поддержку", "it_support_request -> RAG"),
    ("не подключается VPN", "it_support_request -> RAG"),
]


def main() -> None:
    service = IntentService()
    print("Intent classifier manual check\n")
    for text, note in SAMPLES:
        result = service.classify(text)
        decision = resolve_intent_decision(result)
        print(f"Q: {text}")
        print(f"   intent={result.intent} confidence={result.confidence:.2f} decision={decision}")
        print(f"   ({note})\n")


if __name__ == "__main__":
    main()
