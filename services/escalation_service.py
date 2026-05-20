from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from app.app_logger import logger
from app.config import settings
from app.text_safety import mask_sensitive_data
from repositories.support_escalation_repository import (
    SupportEscalationRecord,
    get_support_escalation_repository,
)
from services.email_notification_service import EmailNotificationService


REASON_COMMAND = "Пользователь запросил передачу обращения через команду /support"
REASON_BUTTON = (
    'Пользователь запросил передачу обращения через кнопку «Передать в поддержку»'
)


@dataclass
class TelegramUserInfo:
    user_id: int
    username: str | None
    first_name: str | None
    last_name: str | None


@dataclass
class EscalationResult:
    success: bool
    record_id: int | None = None
    error_message: str | None = None


def build_telegram_full_name(first_name: str | None, last_name: str | None) -> str:
    parts = [p.strip() for p in (first_name or "", last_name or "") if p and p.strip()]
    return " ".join(parts) if parts else "—"


def format_conversation_excerpt(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "История диалога пуста."

    lines: list[str] = []
    for item in messages:
        role = item.get("role", "")
        content = mask_sensitive_data((item.get("content") or "").strip())
        if not content:
            continue
        label = "Пользователь" if role == "user" else "Ассистент"
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines) if lines else "История диалога пуста."


def build_problem_summary(messages: list[dict[str, str]]) -> str:
    user_messages = [
        mask_sensitive_data((m.get("content") or "").strip())
        for m in messages
        if m.get("role") == "user" and (m.get("content") or "").strip()
    ]
    if not user_messages:
        return "Пользователь не описал проблему в текущем диалоге."
    if len(user_messages) == 1:
        return user_messages[-1]
    return user_messages[-1]


def build_email_subject(user: TelegramUserInfo) -> str:
    username_part = f"@{user.username}" if user.username else f"id{user.user_id}"
    return f"[IT Bot] Обращение из Telegram ({username_part})"


def build_email_body(
    user: TelegramUserInfo,
    *,
    reason: str,
    problem_summary: str,
    conversation_excerpt: str,
    submitted_at: datetime,
) -> str:
    full_name = build_telegram_full_name(user.first_name, user.last_name)
    username_line = f"@{user.username}" if user.username else "—"
    submitted_str = submitted_at.strftime("%Y-%m-%d %H:%M:%S %Z").strip()

    return (
        "Обращение из Telegram-бота ИТ-поддержки\n"
        "========================================\n\n"
        f"Дата и время обращения: {submitted_str}\n"
        f"Telegram user_id: {user.user_id}\n"
        f"Telegram username: {username_line}\n"
        f"Имя в Telegram: {full_name}\n\n"
        f"Причина передачи: {reason}\n\n"
        "Краткое описание проблемы:\n"
        f"{problem_summary}\n\n"
        "История диалога:\n"
        "----------------\n"
        f"{conversation_excerpt}\n"
    )


class EscalationService:
    def __init__(
        self,
        email_service: EmailNotificationService | None = None,
    ) -> None:
        self._email_service = email_service or EmailNotificationService()
        self._repository = get_support_escalation_repository()

    def send_escalation(
        self,
        user: TelegramUserInfo,
        conversation_messages: list[dict[str, str]],
        *,
        via_command: bool,
    ) -> EscalationResult:
        reason = REASON_COMMAND if via_command else REASON_BUTTON
        submitted_at = datetime.now(timezone.utc)
        problem_summary = build_problem_summary(conversation_messages)
        conversation_excerpt = format_conversation_excerpt(conversation_messages)
        email_to = settings.support_email.strip()
        email_subject = build_email_subject(user)
        full_name = build_telegram_full_name(user.first_name, user.last_name)

        record = SupportEscalationRecord(
            telegram_user_id=user.user_id,
            telegram_username=user.username,
            telegram_full_name=full_name,
            problem_summary=problem_summary,
            conversation_excerpt=conversation_excerpt,
            reason=reason,
            email_to=email_to,
            email_subject=email_subject,
            email_sent=False,
        )
        record_id = self._repository.create(record)

        try:
            body = build_email_body(
                user,
                reason=reason,
                problem_summary=problem_summary,
                conversation_excerpt=conversation_excerpt,
                submitted_at=submitted_at,
            )
            self._email_service.send_plain_text(email_to, email_subject, body)
            sent_at = int(time.time())
            self._repository.mark_sent(record_id, sent_at)
            logger.info(
                "Support escalation email sent. record_id=%s user_id=%s",
                record_id,
                user.user_id,
            )
            return EscalationResult(success=True, record_id=record_id)
        except Exception as exc:
            error_message = str(exc)
            logger.exception(
                "Support escalation email failed. record_id=%s user_id=%s",
                record_id,
                user.user_id,
            )
            self._repository.mark_failed(record_id, error_message)
            return EscalationResult(
                success=False,
                record_id=record_id,
                error_message=error_message,
            )
