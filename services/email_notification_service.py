from __future__ import annotations

import smtplib
from email.message import EmailMessage

from app.config import settings


class EmailNotificationService:
    def send_plain_text(self, to_address: str, subject: str, body: str) -> None:
        if not settings.support_email_enabled:
            raise RuntimeError("SUPPORT_EMAIL_ENABLED is false")

        if not settings.smtp_host.strip():
            raise RuntimeError("SMTP_HOST is not configured")

        recipient = to_address.strip()
        if not recipient:
            raise RuntimeError("Recipient email is empty")

        sender = (settings.smtp_from or settings.smtp_user or settings.support_email).strip()
        if not sender:
            raise RuntimeError("SMTP_FROM or SMTP_USER is not configured")

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = sender
        message["To"] = recipient
        message.set_content(body, charset="utf-8")

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
            smtp.ehlo()
            if settings.smtp_port == 587:
                smtp.starttls()
                smtp.ehlo()
            if settings.smtp_user:
                smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(message)
