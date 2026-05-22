from app.config import settings


def _support_contact_lines() -> list[str]:
    lines: list[str] = []
    if settings.support_email.strip():
        lines.append(f"Email: {settings.support_email.strip()}")
    if settings.support_telegram.strip():
        lines.append(f"Telegram: {settings.support_telegram.strip()}")
    if settings.support_phone.strip():
        lines.append(f"Телефон: {settings.support_phone.strip()}")
    if settings.support_teams.strip():
        lines.append(f"Teams: {settings.support_teams.strip()}")
    if settings.support_portal.strip():
        lines.append(f"Портал: {settings.support_portal.strip()}")
    return lines


def format_help_text() -> str:
    return (
        "Доступные команды:\n\n"
        "/start — приветствие и краткое описание бота; сбрасывает историю диалога\n"
        "/help — эта справка по командам и кнопкам\n"
        "/clear — очистить историю текущего диалога\n"
        "/support — передать обращение в ИТ-поддержку по email\n\n"
        "Кнопки под полем ввода:\n"
        "👍 Помогло — отметить, что ответ помог\n"
        "🛟 Передать в поддержку — то же, что /support\n"
        "✏️ Новый вопрос — сбросить историю и задать новый вопрос\n\n"
        "Любой другой текст — вопрос по ИТ (почта, VPN, Windows, пароли, доступы и т.д.)."
    )


def format_greeting_response() -> str:
    return (
        "Здравствуйте! Я ИИ-ассистент первой линии ИТ-поддержки компании MedRetail.\n\n"
        "Я помогу с ИТ-вопросами: доступами, почтой, VPN, сетью, "
        "компьютером, принтером или корпоративными сервисами. Опишите, что случилось."
    )


def format_out_of_scope_response() -> str:
    return (
        "Я помогаю только с ИТ-вопросами: доступы, почта, VPN, сеть, рабочий компьютер, "
        "принтеры и корпоративные сервисы. Опишите, пожалуйста, вашу ИТ-проблему."
    )


def format_support_contacts_text() -> str:
    contact_lines = _support_contact_lines()
    body = "\n".join(contact_lines) if contact_lines else "Контакты поддержки не настроены."
    return (
        "Обратиться в ИТ-поддержку можно так:\n\n"
        f"{body}\n\n"
        "Если хотите передать обращение через бота, используйте команду /support."
    )


def format_escalation_confirm_prompt() -> str:
    return (
        "Я отправлю обращение в ИТ-поддержку.\n\n"
        "В письмо будут включены:\n"
        "ваше имя в Telegram, username или Telegram ID, дата обращения "
        "и краткая история этого диалога.\n\n"
        "Отправить обращение?"
    )


def format_escalation_cancelled() -> str:
    return (
        "Хорошо, обращение не отправлено.\n"
        "Если понадобится помощь специалиста, используйте команду /support."
    )


def format_escalation_sent() -> str:
    return (
        "Обращение отправлено в ИТ-поддержку.\n"
        "Специалист сможет связаться с вами в Telegram."
    )


def format_escalation_send_failed() -> str:
    return (
        "Не удалось отправить обращение в поддержку из-за технической ошибки.\n"
        "Попробуйте позже или обратитесь в поддержку напрямую."
    )


def format_no_direct_answer_fallback() -> str:
    return (
        "Я не нашел подходящую инструкцию в базе знаний по этому вопросу. "
        "Рекомендую обратиться в ИТ-поддержку через /support."
    )


def format_no_context_fallback() -> str:
    contact_lines = _support_contact_lines()
    contacts_block = "\n".join(contact_lines) if contact_lines else "Контакты поддержки не настроены."
    return (
        "Я не нашел подходящую инструкцию в базе знаний.\n\n"
        "Рекомендую обратиться в ИТ-поддержку:\n"
        f"{contacts_block}\n\n"
        "Также можно использовать команду /support, чтобы передать обращение через бота."
    )
