"""Run questions from tests/test_questions.md through bot logic. Usage: python scripts/run_test_questions.py"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.intent_service import IntentService, resolve_intent_decision  # noqa: E402
from app.rag_service import RagService  # noqa: E402
from app.support_messages import format_greeting_response, format_out_of_scope_response  # noqa: E402

QUESTIONS_PATH = ROOT / "tests" / "test_questions.md"
OUT_OF_SCOPE_NUMS = set(range(21, 26))


def load_questions(path: Path) -> list[tuple[int, str, str]]:
    text = path.read_text(encoding="utf-8")
    items: list[tuple[int, str, str]] = []
    section = ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        match = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
        if match:
            items.append((int(match.group(1)), section, match.group(2).strip()))
    return items


def route_question(question: str, intent_service: IntentService, rag: RagService) -> dict:
    if settings.intent_classifier_enabled:
        classification = intent_service.classify(question)
        decision = resolve_intent_decision(classification)
        intent = classification.intent
        confidence = classification.confidence
    else:
        intent = "it_support_request"
        confidence = 0.0
        decision = "rag"

    if decision == "greeting":
        return {
            "intent": intent,
            "confidence": confidence,
            "decision": decision,
            "answer": format_greeting_response(),
            "found_context": False,
            "sources": [],
        }
    if decision == "out_of_scope":
        return {
            "intent": intent,
            "confidence": confidence,
            "decision": decision,
            "answer": format_out_of_scope_response(),
            "found_context": False,
            "sources": [],
        }

    result = rag.answer(question, history=[])
    return {
        "intent": intent,
        "confidence": confidence,
        "decision": decision,
        "answer": result.answer,
        "found_context": result.found_context,
        "sources": result.sources,
    }


def evaluate(num: int, section: str, question: str, run: dict) -> tuple[bool, str]:
    decision = run["decision"]
    answer = (run["answer"] or "").lower()
    found = run["found_context"]

    if num in OUT_OF_SCOPE_NUMS:
        if decision != "out_of_scope":
            return False, f"ожидался out_of_scope, получен {decision}"
        off_topic_hints = ("погод", "коммерческ", "простуд", "акци", "придумай пароль")
        if any(h in answer for h in off_topic_hints) and "ит" not in answer[:80]:
            return False, "ответ пытается ответить на off-topic тему"
        return True, "корректный отказ вне ИТ"

    if decision == "greeting":
        return False, "IT-вопрос ошибочно обработан как greeting"
    if decision == "out_of_scope":
        return False, "IT-вопрос ошибочно обработан как out_of_scope"

    if decision != "rag":
        return False, f"ожидался RAG, получен {decision}"

    topic_keywords = {
        "VPN Fortinet": ("vpn", "forti", "fortinet"),
        "Почта": ("почт", "outlook", "thunderbird", "яндекс", "спам"),
        "Windows и учетная запись": (
            "парол",
            "аккаунт",
            "wi-fi",
            "wifi",
            "windows",
            "учетн",
        ),
        "Доступы и эскалация": (
            "доступ",
            "поддерж",
            "обращ",
            "критич",
            "telegram",
            "/support",
        ),
    }
    keys = topic_keywords.get(section, ())
    if keys and not any(k in answer for k in keys):
        return False, "ответ не по теме раздела"

    if not found and "не нашел" in answer and "баз" in answer:
        return False, "RAG не нашел контекст (no_context fallback)"

    return True, "релевантный ответ из базы или по теме"


def main() -> None:
    questions = load_questions(QUESTIONS_PATH)
    intent_service = IntentService()
    rag = RagService()

    print(f"Intent classifier: {settings.intent_classifier_enabled}")
    print(f"Model: {settings.intent_model}\n")
    print("=" * 80)

    passed = 0
    for num, section, question in questions:
        run = route_question(question, intent_service, rag)
        ok, reason = evaluate(num, section, question, run)
        if ok:
            passed += 1

        print(f"\n### {num}. [{section}] {question}")
        print(
            f"intent={run['intent']} conf={run['confidence']:.2f} "
            f"decision={run['decision']} found_context={run['found_context']}"
        )
        if run["sources"]:
            print(f"sources: {', '.join(run['sources'][:3])}")
        print(f"Оценка: {'OK' if ok else 'FAIL'} — {reason}")
        print("-" * 40)
        answer = run["answer"] or ""
        print(answer[:1200] + ("..." if len(answer) > 1200 else ""))

    print("\n" + "=" * 80)
    print(f"Итого: {passed}/{len(questions)} правильных")


if __name__ == "__main__":
    main()
