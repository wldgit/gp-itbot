import unittest
from unittest.mock import MagicMock, patch

from repositories.support_escalation_repository import (
    SQLiteSupportEscalationRepository,
    SupportEscalationRecord,
)
from services.escalation_service import (
    REASON_BUTTON,
    REASON_COMMAND,
    EscalationService,
    TelegramUserInfo,
    build_email_body,
    build_email_subject,
    build_problem_summary,
    format_conversation_excerpt,
)


class EscalationFormattingTests(unittest.TestCase):
    def test_problem_summary_uses_last_user_message(self):
        messages = [
            {"role": "user", "content": "VPN не работает"},
            {"role": "assistant", "content": "Перезагрузите клиент"},
            {"role": "user", "content": "Не помогло"},
        ]
        self.assertEqual(build_problem_summary(messages), "Не помогло")

    def test_conversation_excerpt_labels_roles(self):
        excerpt = format_conversation_excerpt(
            [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Здравствуйте"}]
        )
        self.assertIn("Пользователь: Привет", excerpt)
        self.assertIn("Ассистент: Здравствуйте", excerpt)

    def test_email_subject_contains_username(self):
        subject = build_email_subject(
            TelegramUserInfo(123, "test_user", "Ivan", "Petrov")
        )
        self.assertIn("@test_user", subject)

    def test_email_body_contains_reason_and_history(self):
        user = TelegramUserInfo(42, None, "Anna", None)
        from datetime import datetime, timezone

        body = build_email_body(
            user,
            reason=REASON_COMMAND,
            problem_summary="Не открывается почта",
            conversation_excerpt="Пользователь: Не открывается почта",
            submitted_at=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
        )
        self.assertIn(REASON_COMMAND, body)
        self.assertIn("42", body)
        self.assertIn("Не открывается почта", body)


class EscalationServiceTests(unittest.TestCase):
    def test_send_escalation_success(self):
        repo = SQLiteSupportEscalationRepository(":memory:")
        email_service = MagicMock()
        service = EscalationService(email_service=email_service)
        service._repository = repo

        user = TelegramUserInfo(99, "worker", "Olga", "Sidorova")
        messages = [{"role": "user", "content": "Принтер не печатает"}]

        with patch("services.escalation_service.settings") as mock_settings:
            mock_settings.support_email = "it@example.com"
            result = service.send_escalation(user, messages, via_command=False)

        self.assertTrue(result.success)
        email_service.send_plain_text.assert_called_once()
        args = email_service.send_plain_text.call_args[0]
        self.assertEqual(args[0], "it@example.com")
        self.assertIn("Принтер не печатает", args[2])
        self.assertIn(REASON_BUTTON, args[2])

    def test_send_escalation_failure_records_error(self):
        repo = SQLiteSupportEscalationRepository(":memory:")
        email_service = MagicMock()
        email_service.send_plain_text.side_effect = RuntimeError("SMTP down")
        service = EscalationService(email_service=email_service)
        service._repository = repo

        user = TelegramUserInfo(1, None, "A", None)
        with patch("services.escalation_service.settings") as mock_settings:
            mock_settings.support_email = "it@example.com"
            result = service.send_escalation(user, [], via_command=True)

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "SMTP down")

        with repo._connect() as conn:
            row = conn.execute(
                "SELECT email_sent, error_message, reason FROM support_escalations WHERE id = ?",
                (result.record_id,),
            ).fetchone()
        self.assertEqual(row[0], 0)
        self.assertEqual(row[1], "SMTP down")
        self.assertEqual(row[2], REASON_COMMAND)


if __name__ == "__main__":
    unittest.main()
