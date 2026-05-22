from pathlib import Path

from app.app_logger import logger


def load_prompt(path: str, default: str, label: str = "prompt") -> str:
    """Load optional prompt files (intent, document profiler) with soft fallback."""
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
    """Load main RAG system prompt; fail fast if file is missing, empty, or unreadable."""
    prompt_path = Path(path)
    label = "System prompt"

    if not prompt_path.exists():
        logger.error("%s file not found: %s", label, prompt_path)
        raise RuntimeError(f"{label} file not found: {prompt_path}")

    try:
        raw = prompt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.exception("%s file is not valid UTF-8: %s", label, prompt_path)
        raise RuntimeError(f"{label} file is not valid UTF-8: {prompt_path}") from None
    except OSError:
        logger.exception("Failed to read %s from %s", label, prompt_path)
        raise RuntimeError(f"Failed to read {label} file: {prompt_path}") from None

    prompt = raw.strip()
    if not prompt:
        logger.error("%s file is empty: %s", label, prompt_path)
        raise RuntimeError(f"{label} file is empty: {prompt_path}")

    logger.info("%s loaded from %s, chars=%s", label, prompt_path, len(prompt))
    return prompt
