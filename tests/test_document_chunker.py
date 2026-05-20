import unittest
from pathlib import Path

from app.config import settings
from ingest.document_chunker import (
    _document_title,
    _strip_chunk_prefix,
    chunk_document,
    estimate_tokens,
)
from ingest.document_profiler import DocumentProfile, validate_document_profile
from ingest.text_cleaner import clean_text


def _profile(
    strategy: str,
    max_tokens: int = 700,
    overlap: int = 100,
    doc_type: str = "unknown",
) -> DocumentProfile:
    return DocumentProfile(
        doc_type=doc_type,
        chunking_strategy=strategy,
        max_chunk_tokens=max_tokens,
        overlap_tokens=overlap,
    )


def _chunk_body_tokens(chunk, profile, filename: str) -> int:
    title = _document_title(chunk.text, filename)
    body = _strip_chunk_prefix(chunk.text, profile, title, chunk.section)
    return estimate_tokens(body)


class DocumentChunkerTests(unittest.TestCase):
    def test_qa_pairs_vopros_otvet(self):
        text = (
            "Вопрос: Как сбросить пароль?\n"
            "Ответ: Обратитесь в ИТ-поддержку.\n\n"
            "Вопрос: Где взять VPN?\n"
            "Ответ: Установите FortiClient."
        )
        chunks = chunk_document(text, "faq.txt", _profile("qa_pairs", 500, 50))
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("Вопрос:" in chunk.text for chunk in chunks))

    def test_qa_pairs_markdown_headers(self):
        text = "## Как подключить VPN?\n\nСкачайте FortiClient.\n\n## Не работает почта?\n\nПерезапустите Outlook."
        chunks = chunk_document(text, "faq.md", _profile("qa_pairs", 500, 50))
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(any("?" in chunk.text for chunk in chunks))

    def test_sections(self):
        text = "# Заголовок\n\nВводный текст.\n\n## Раздел 1\n\nТекст раздела 1.\n\n## Раздел 2\n\nТекст раздела 2."
        chunks = chunk_document(text, "guide.md", _profile("sections", 700, 100))
        self.assertGreaterEqual(len(chunks), 1)
        joined = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("Раздел 1", joined)
        self.assertIn("Раздел 2", joined)

    def test_fallback_not_empty(self):
        text = "Простой текст без структуры.\n\nВторой абзац с описанием."
        chunks = chunk_document(text, "plain.txt", _profile("fallback_token_chunks", 200, 50))
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(chunk.text.strip() for chunk in chunks))

    def test_sections_merges_short_sections(self):
        text = (
            "# Инструкция\n\n"
            "Короткое описание.\n\n"
            "## Описание\n\nТекст описания.\n\n"
            "## Когда использовать\n\nИспользуйте при настройке почты.\n\n"
            "## Перед началом\n\nПроверьте логин и пароль.\n\n"
            "## Шаги\n\n1. Откройте Outlook.\n2. Добавьте учетную запись.\n\n"
            "## Если не получилось\n\nОбратитесь в поддержку."
        )

        chunks = chunk_document(text, "outlook_setup.md", _profile("sections", 700, 100))

        self.assertLess(len(chunks), 5)

        joined = "\n".join(chunk.text for chunk in chunks)

        self.assertIn("Описание", joined)
        self.assertIn("Когда использовать", joined)
        self.assertIn("Перед началом", joined)
        self.assertIn("Шаги", joined)
        self.assertIn("Если не получилось", joined)

    def test_trailing_small_buffer_merges_with_previous(self):
        paragraph = "Текст шага настройки с подробным описанием действий. " * 12
        text = (
            "# Инструкция\n\n"
            f"## Раздел 1\n\n{paragraph}\n\n"
            f"## Раздел 2\n\n{paragraph}\n\n"
            f"## Раздел 3\n\n{paragraph}\n\n"
            "## Хвост\n\nКороткий финальный раздел."
        )

        chunks = chunk_document(text, "guide.md", _profile("sections", 700, 100))

        self.assertEqual(len(chunks), 1)
        self.assertIn("Хвост", chunks[0].text)
        self.assertIn("Раздел 3", chunks[0].text)

    def test_troubleshooting_blocks_merges_short_sections(self):
        text = (
            "# Ошибки Fortinet VPN\n\n"
            "Краткое введение о типичных сбоях подключения.\n\n"
            "## Коды ошибок\n\n"
            "| Код | Описание |\n"
            "| --- | --- |\n"
            "| -20199 | Проблема с сертификатом |\n"
            "| -7200 | Таймаут сессии |\n\n"
            "## Что делать\n\n"
            "1. Перезапустите FortiClient.\n"
            "2. Проверьте VPN-профиль.\n\n"
            "## Когда обращаться в поддержку\n\n"
            "Если ошибка повторяется более 15 минут."
        )
        profile = _profile("troubleshooting_blocks", 700, 80, doc_type="troubleshooting")
        chunks = chunk_document(text, "vpn_fortinet_errors.md", profile)

        self.assertLess(len(chunks), 4)
        joined = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("Коды ошибок", joined)
        self.assertIn("-20199", joined)
        self.assertIn("Когда обращаться", joined)

        min_tokens = settings.section_min_chunk_tokens
        for chunk in chunks:
            tokens = _chunk_body_tokens(chunk, profile, "vpn_fortinet_errors.md")
            if len(chunks) > 1:
                self.assertGreaterEqual(
                    tokens,
                    min_tokens,
                    msg=f"unexpected micro-chunk: {chunk.section!r} ({tokens} tokens)",
                )

    def test_mixed_by_sections_merges_short_sections(self):
        text = (
            "# Смешанный документ\n\n"
            "Короткое введение.\n\n"
            "## Раздел A\n\nТекст раздела A.\n\n"
            "## Раздел B\n\nТекст раздела B.\n\n"
            "## Раздел C\n\nТекст раздела C."
        )
        profile = DocumentProfile(
            doc_type="mixed",
            chunking_strategy="mixed_by_sections",
            max_chunk_tokens=700,
            overlap_tokens=80,
            section_profiles=[],
        )
        chunks = chunk_document(text, "mixed.md", profile)

        self.assertLess(len(chunks), 4)
        joined = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("Раздел A", joined)
        self.assertIn("Раздел B", joined)
        self.assertIn("Раздел C", joined)

    def test_fallback_uses_markdown_sections_when_headers_present(self):
        text = (
            "# Документ\n\n"
            "## Раздел 1\n\nПервый абзац с описанием.\n\n"
            "## Раздел 2\n\nВторой абзац с описанием."
        )
        chunks = chunk_document(text, "doc.md", _profile("fallback_token_chunks", 700, 100))

        self.assertGreaterEqual(len(chunks), 1)
        self.assertLess(len(chunks), 2)
        joined = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("## Раздел 1", joined)
        self.assertIn("## Раздел 2", joined)

    def test_qa_pairs_one_chunk_per_pair(self):
        text = (
            "Вопрос: Как сбросить пароль?\n"
            "Ответ: Обратитесь в ИТ-поддержку и подтвердите личность.\n\n"
            "Вопрос: Где взять VPN?\n"
            "Ответ: Установите FortiClient и импортируйте профиль."
        )
        chunks = chunk_document(text, "faq.txt", _profile("qa_pairs", 500, 50))
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all("Вопрос:" in chunk.text for chunk in chunks))

    def test_support_contacts_escalation_uses_sections(self):
        path = Path("data/docs/support_contacts_escalation.md")
        if not path.exists():
            self.skipTest("support_contacts_escalation.md not found")

        text = clean_text(path.read_text(encoding="utf-8"))
        profile = validate_document_profile(
            {
                "doc_type": "support_contacts",
                "chunking_strategy": "sections",
                "max_chunk_tokens": 500,
                "overlap_tokens": 0,
                "confidence": 0.9,
                "signals": ["support contacts doc"],
                "section_profiles": [],
            }
        )
        self.assertEqual(profile.doc_type, "support_contacts")
        self.assertEqual(profile.chunking_strategy, "sections")

        chunks = chunk_document(text, path.name, profile)

        self.assertGreaterEqual(len(chunks), 3)
        self.assertLess(len(chunks), 20)

        joined = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("Каналы обращения", joined)
        self.assertIn("эскалации", joined)

        for chunk in chunks:
            lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
            self.assertFalse(
                len(lines) == 1 and lines[0].startswith("Email:"),
                "unexpected single-line contact key chunk",
            )


if __name__ == "__main__":
    unittest.main()
