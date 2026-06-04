"""Run questions from tests/test_questions.md through bot logic.

Usage:
  python scripts/run_test_questions.py
  python scripts/run_test_questions.py --output logs/test_questions_run.txt
"""
import argparse
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "logs" / "test_questions_run.txt"


def configure_utf8_io() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


configure_utf8_io()

from app.config import settings  # noqa: E402
from app.intent_service import IntentService, resolve_intent_decision  # noqa: E402
from app.rag_service import RagService  # noqa: E402
from app.support_messages import format_greeting_response, format_out_of_scope_response  # noqa: E402

QUESTIONS_PATH = ROOT / "tests" / "test_questions.md"
OUT_OF_SCOPE_NUMS = set(range(21, 26))


class ReportWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file = path.open("w", encoding="utf-8", newline="\n")

    def line(self, text: str = "") -> None:
        self._file.write(text + "\n")
        self._file.flush()
        print(text)

    def close(self) -> None:
        self._file.close()

    @property
    def path(self) -> Path:
        return self._path


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
    parser = argparse.ArgumentParser(description="Run test questions from tests/test_questions.md")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"UTF-8 report path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    logging.getLogger("gp_itbot").setLevel(logging.WARNING)

    report = ReportWriter(args.output)
    try:
        questions = load_questions(QUESTIONS_PATH)
        intent_service = IntentService()
        rag = RagService()

        report.line(f"Intent classifier: {settings.intent_classifier_enabled}")
        report.line(f"Model: {settings.intent_model}")
        report.line(f"Report file: {report.path}")
        report.line()
        report.line("=" * 80)

        passed = 0
        for num, section, question in questions:
            started = time.monotonic()
            run = route_question(question, intent_service, rag)
            elapsed_s = time.monotonic() - started
            ok, reason = evaluate(num, section, question, run)
            if ok:
                passed += 1

            report.line()
            report.line(f"### {num}. [{section}] {question}")
            report.line(f"time={elapsed_s:.1f}s")
            report.line(
                f"intent={run['intent']} conf={run['confidence']:.2f} "
                f"decision={run['decision']} found_context={run['found_context']}"
            )
            if run["sources"]:
                report.line(f"sources: {', '.join(run['sources'][:3])}")
            report.line(f"Оценка: {'OK' if ok else 'FAIL'} — {reason}")
            report.line("-" * 40)
            answer = run["answer"] or ""
            report.line(answer[:1200] + ("..." if len(answer) > 1200 else ""))

        report.line()
        report.line("=" * 80)
        report.line(f"Итого: {passed}/{len(questions)} правильных")
    finally:
        report.close()


if __name__ == "__main__":
    main()
