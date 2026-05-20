from pathlib import Path

from app.app_logger import logger


DEFAULT_SYSTEM_PROMPT = """Ты — ИИ-ассистент первой линии ИТ-поддержки компании MelRetail.
Отвечай только на основе базы знаний. Если ответа нет, предложи обратиться в поддержку.
Не запрашивай пароли, одноразовые коды и персональные данные.
"""


def load_prompt(path: str, default: str, label: str = "prompt") -> str:
    prompt_path = Path(path)

    try:
        if not prompt_path.exists():
            logger.warning("%s file not found: %s. Using default.", label, path)
            return default

        prompt = prompt_path.read_text(encoding="utf-8").strip()

        if not prompt:
            logger.warning("%s file is empty: %s. Using default.", label, path)
            return default

        logger.info("%s loaded from %s, chars=%s", label, path, len(prompt))
        return prompt

    except Exception:
        logger.exception("Failed to load %s from %s. Using default.", label, path)
        return default


def load_system_prompt(path: str) -> str:
    return load_prompt(path, DEFAULT_SYSTEM_PROMPT, "System prompt")
