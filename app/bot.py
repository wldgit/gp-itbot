import asyncio
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, KeyboardButton, Message, ReplyKeyboardMarkup

from app.app_logger import logger
from app.chat_history import ChatHistoryStore
from app.config import settings
from app.logging_service import InteractionLog, log_repository, log_runtime_event
from app.intent_service import IntentService, resolve_intent_decision
from app.rag_service import RagService
from app.support_messages import (
    format_escalation_cancelled,
    format_escalation_confirm_prompt,
    format_escalation_send_failed,
    format_escalation_sent,
    format_greeting_response,
    format_help_text,
    format_out_of_scope_response,
    format_support_contacts_text,
)
from app.text_safety import contains_secret_like_data, mask_sensitive_data
from services.escalation_service import EscalationService, TelegramUserInfo


class SupportEscalation(StatesGroup):
    confirm = State()


BTN_HELPED = "👍 Помогло"
BTN_SUPPORT = "🛟 Передать в поддержку"
BTN_NEW_QUESTION = "✏️ Новый вопрос"
BTN_CONFIRM_YES = "Да, отправить"
BTN_CONFIRM_NO = "Нет"

MAIN_MENU_TEXTS = frozenset({BTN_HELPED, BTN_SUPPORT, BTN_NEW_QUESTION})
CONFIRM_TEXTS = frozenset({BTN_CONFIRM_YES, BTN_CONFIRM_NO})


def main_reply_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_HELPED), KeyboardButton(text=BTN_SUPPORT)],
            [KeyboardButton(text=BTN_NEW_QUESTION)],
        ],
        resize_keyboard=True,
        input_field_placeholder="Введите вопрос или выберите действие",
    )


def confirm_reply_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_CONFIRM_YES), KeyboardButton(text=BTN_CONFIRM_NO)],
        ],
        resize_keyboard=True,
        input_field_placeholder="Подтвердите отправку обращения",
    )


rag_service = RagService()
intent_service = IntentService()
chat_history = ChatHistoryStore(settings.chat_history_max_messages)
escalation_service = EscalationService()


def _telegram_user_info(message: Message) -> TelegramUserInfo:
    user = message.from_user
    return TelegramUserInfo(
        user_id=user.id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
    )


def _log_intent_decision(
    user_id: int,
    intent: str,
    confidence: float,
    decision: str,
) -> None:
    logger.info(
        "Intent routing. user_id=%s intent_classifier_enabled=%s intent=%s confidence=%.2f decision=%s",
        user_id,
        settings.intent_classifier_enabled,
        intent,
        confidence,
        decision,
    )
    log_runtime_event(
        "INFO",
        "intent_routing",
        "Intent classification completed",
        {
            "user_id": user_id,
            "intent_classifier_enabled": settings.intent_classifier_enabled,
            "intent": intent,
            "confidence": confidence,
            "decision": decision,
        },
    )


async def start_support_escalation(
    message: Message,
    state: FSMContext,
    *,
    via_command: bool,
) -> None:
    logger.info(
        "Support escalation confirmation started. user_id=%s via_command=%s",
        message.from_user.id,
        via_command,
    )
    await state.set_state(SupportEscalation.confirm)
    await state.update_data(via_command=via_command)
    await message.answer(
        format_escalation_confirm_prompt(),
        reply_markup=confirm_reply_keyboard(),
    )


async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    logger.info(
        "Command /start from user_id=%s username=%s",
        message.from_user.id,
        message.from_user.username,
    )
    chat_history.clear(message.from_user.id)
    text = (
        "Здравствуйте! Я ИИ-ассистент первой линии ИТ-поддержки компании MedRetail.\n\n"
        "Я помогаю с ИТ-вопросами: доступами, почтой, VPN, сетью, компьютером, принтером или корпоративными сервисами.\n\n"
        "Список доступных команд: /help\n\n"
    )
    await message.answer(text, reply_markup=main_reply_keyboard())


async def cmd_help(message: Message, state: FSMContext):
    await state.clear()
    logger.info("Command /help from user_id=%s", message.from_user.id)
    await message.answer(format_help_text(), reply_markup=main_reply_keyboard())


async def cmd_clear(message: Message, state: FSMContext):
    await state.clear()
    chat_history.clear(message.from_user.id)
    await message.answer("История диалога очищена.", reply_markup=main_reply_keyboard())


async def cmd_support(message: Message, state: FSMContext):
    await state.clear()
    await start_support_escalation(message, state, via_command=True)


async def support_escalation_confirm(message: Message, state: FSMContext):
    text = (message.text or "").strip()
    user_id = message.from_user.id
    data = await state.get_data()
    via_command = bool(data.get("via_command", False))

    if text == BTN_CONFIRM_NO:
        await state.clear()
        logger.info("Support escalation cancelled. user_id=%s", user_id)
        log_runtime_event(
            "INFO",
            "support_escalation_cancelled",
            "User cancelled support escalation",
            {"user_id": user_id, "username": message.from_user.username},
        )
        await message.answer(
            format_escalation_cancelled(),
            reply_markup=main_reply_keyboard(),
        )
        return

    if text != BTN_CONFIRM_YES:
        await message.answer(
            "Пожалуйста, нажмите «Да, отправить» или «Нет».",
            reply_markup=confirm_reply_keyboard(),
        )
        return

    await state.clear()
    started = time.monotonic()
    conversation = chat_history.get_messages(user_id)
    user_info = _telegram_user_info(message)

    result = await asyncio.to_thread(
        escalation_service.send_escalation,
        user_info,
        conversation,
        via_command=via_command,
    )
    elapsed = int((time.monotonic() - started) * 1000)

    if result.success:
        answer = format_escalation_sent()
        status = "support_escalation_sent"
        log_runtime_event(
            "INFO",
            "support_escalation_sent",
            "Support escalation email sent",
            {
                "user_id": user_id,
                "username": message.from_user.username,
                "record_id": result.record_id,
            },
        )
    else:
        answer = format_escalation_send_failed()
        status = "support_escalation_failed"
        log_runtime_event(
            "ERROR",
            "support_escalation_failed",
            result.error_message or "unknown error",
            {
                "user_id": user_id,
                "username": message.from_user.username,
                "record_id": result.record_id,
            },
        )

    log_repository.log_interaction(
        InteractionLog(
            user_id=user_id,
            username=message.from_user.username,
            question="/support escalation",
            answer=answer,
            sources=[],
            transferred_to_support=result.success,
            response_time_ms=elapsed,
            status=status,
            error_message=result.error_message,
        )
    )
    await message.answer(answer, reply_markup=main_reply_keyboard())


async def main_menu_buttons(message: Message, state: FSMContext):
    text = (message.text or "").strip()
    user_id = message.from_user.id

    if text == BTN_HELPED:
        await state.clear()
        logger.info("Feedback: helped. user_id=%s", user_id)
        log_runtime_event(
            "INFO",
            "feedback_helped",
            "User marked response as helpful",
            {"user_id": user_id, "username": message.from_user.username},
        )
        await message.answer(
            "Рады были помочь. При необходимости задайте следующий вопрос текстом.",
            reply_markup=main_reply_keyboard(),
        )
        return

    if text == BTN_SUPPORT:
        await state.clear()
        await start_support_escalation(message, state, via_command=False)
        return

    if text == BTN_NEW_QUESTION:
        await state.clear()
        chat_history.clear(user_id)
        logger.info("New question via button. user_id=%s", user_id)
        log_runtime_event(
            "INFO",
            "dialog_new_question",
            "User started a new question (history cleared)",
            {"user_id": user_id, "username": message.from_user.username},
        )
        await message.answer(
            "История диалога сброшена. Опишите новый вопрос одним сообщением.",
            reply_markup=main_reply_keyboard(),
        )
        return


async def handle_question(message: Message, state: FSMContext):
    question = message.text or ""
    started = time.monotonic()

    logger.info(
        "Incoming question. user_id=%s username=%s chars=%s",
        message.from_user.id,
        message.from_user.username,
        len(question),
    )

    if contains_secret_like_data(question):
        logger.warning("Secret-like data detected in message. user_id=%s", message.from_user.id)
        await message.answer(
            "Пожалуйста, не отправляйте в чат пароли, одноразовые коды и другие секретные данные.",
            reply_markup=main_reply_keyboard(),
        )
        return

    user_id = message.from_user.id

    if settings.intent_classifier_enabled:
        classification = intent_service.classify(question)
        decision = resolve_intent_decision(classification)
        _log_intent_decision(user_id, classification.intent, classification.confidence, decision)

        if decision == "greeting":
            answer = format_greeting_response()
            await message.answer(answer, reply_markup=main_reply_keyboard())
            log_repository.log_interaction(
                InteractionLog(
                    user_id=user_id,
                    username=message.from_user.username,
                    question=question,
                    answer=answer,
                    sources=[],
                    transferred_to_support=False,
                    response_time_ms=int((time.monotonic() - started) * 1000),
                    status="intent_greeting",
                )
            )
            return

        if decision == "out_of_scope":
            answer = format_out_of_scope_response()
            await message.answer(answer, reply_markup=main_reply_keyboard())
            log_repository.log_interaction(
                InteractionLog(
                    user_id=user_id,
                    username=message.from_user.username,
                    question=question,
                    answer=answer,
                    sources=[],
                    transferred_to_support=False,
                    response_time_ms=int((time.monotonic() - started) * 1000),
                    status="intent_out_of_scope",
                )
            )
            return

        if decision == "support_contacts":
            answer = format_support_contacts_text()
            await message.answer(answer, reply_markup=main_reply_keyboard())
            log_repository.log_interaction(
                InteractionLog(
                    user_id=user_id,
                    username=message.from_user.username,
                    question=question,
                    answer=answer,
                    sources=[],
                    transferred_to_support=False,
                    response_time_ms=int((time.monotonic() - started) * 1000),
                    status="intent_support_contacts",
                )
            )
            return

    prior = chat_history.get_messages(user_id)

    try:
        result = rag_service.answer(question, history=prior)
        await message.answer(result.answer, reply_markup=main_reply_keyboard())

        safe_q = mask_sensitive_data(question)
        chat_history.append(user_id, "user", safe_q)
        chat_history.append(user_id, "assistant", result.answer)

        log_repository.log_interaction(
            InteractionLog(
                user_id=message.from_user.id,
                username=message.from_user.username,
                question=question,
                answer=result.answer,
                sources=result.sources,
                transferred_to_support=False,
                response_time_ms=result.response_time_ms,
                status="ok" if result.found_context else "no_context",
            )
        )

    except Exception as exc:
        elapsed = int((time.monotonic() - started) * 1000)

        logger.exception(
            "Failed to handle question. user_id=%s elapsed_ms=%s",
            message.from_user.id,
            elapsed,
        )
        log_runtime_event(
            "ERROR",
            "message_handling_failed",
            str(exc),
            {
                "user_id": message.from_user.id,
                "username": message.from_user.username,
                "elapsed_ms": elapsed,
            },
        )
        log_repository.log_interaction(
            InteractionLog(
                user_id=message.from_user.id,
                username=message.from_user.username,
                question=question,
                answer="Произошла техническая ошибка. Попробуйте позже или используйте /support.",
                sources=[],
                transferred_to_support=False,
                response_time_ms=elapsed,
                status="error",
                error_message=str(exc),
            )
        )
        await message.answer(
            "Произошла техническая ошибка. Попробуйте позже или используйте /support.",
            reply_markup=main_reply_keyboard(),
        )


async def main():
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    logger.info(
        "Starting bot. environment=%s log_backend=%s",
        settings.environment,
        settings.log_backend,
    )
    log_runtime_event(
        "INFO",
        "bot_starting",
        "Bot is starting",
        {
            "environment": settings.environment,
            "log_backend": settings.log_backend,
        },
    )

    bot = Bot(token=settings.telegram_bot_token)
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Приветствие и сброс истории"),
            BotCommand(command="help", description="Справка по командам"),
            BotCommand(command="clear", description="Очистить историю диалога"),
            BotCommand(command="support", description="Передать обращение в поддержку"),
        ]
    )
    dp = Dispatcher(storage=MemoryStorage())

    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(cmd_clear, Command("clear"))
    dp.message.register(cmd_support, Command("support"))
    dp.message.register(support_escalation_confirm, SupportEscalation.confirm)
    dp.message.register(main_menu_buttons, F.text.in_(MAIN_MENU_TEXTS))
    dp.message.register(handle_question, F.text)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
