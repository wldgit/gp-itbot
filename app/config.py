import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _resolve_path(env_value: str, default: str) -> str:
    """Resolve paths for local run and Docker (/app/... exists only in container)."""
    raw = (env_value or default).strip()
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/app/"):
        return str((PROJECT_ROOT / normalized[5:]).resolve())
    path = Path(raw)
    if not path.is_absolute():
        return str((PROJECT_ROOT / path).resolve())
    if path.exists():
        return str(path.resolve())
    return str(path)


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    chroma_path: str = _resolve_path(os.getenv("CHROMA_PATH", ""), "./data/chroma")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")
    recreate_chroma_collection: bool = _env_bool("RECREATE_CHROMA_COLLECTION", True)
    docs_path: str = _resolve_path(os.getenv("DOCS_PATH", ""), "./data/docs")
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    document_profiler_enabled: bool = _env_bool("DOCUMENT_PROFILER_ENABLED", True)
    document_profiler_model: str = os.getenv("DOCUMENT_PROFILER_MODEL", "gpt-5-nano")
    document_profiler_confidence_threshold: float = float(
        os.getenv("DOCUMENT_PROFILER_CONFIDENCE_THRESHOLD", "0.65")
    )
    document_profiler_max_input_chars: int = int(
        os.getenv("DOCUMENT_PROFILER_MAX_INPUT_CHARS", "30000")
    )
    preprocessing_report_path: str = _resolve_path(
        os.getenv("PREPROCESSING_REPORT_PATH", ""), "./data/preprocessing_report.json"
    )
    document_profiler_prompt_file: str = _resolve_path(
        os.getenv("DOCUMENT_PROFILER_PROMPT_FILE", ""),
        "./prompts/document_profiler_prompt.md",
    )

    default_chunk_max_tokens: int = int(os.getenv("DEFAULT_CHUNK_MAX_TOKENS", "700"))
    default_chunk_overlap_tokens: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP_TOKENS", "100"))
    section_min_chunk_tokens: int = int(os.getenv("SECTION_MIN_CHUNK_TOKENS", "150"))

    faq_chunk_max_tokens: int = int(os.getenv("FAQ_CHUNK_MAX_TOKENS", "450"))
    faq_chunk_overlap_tokens: int = int(os.getenv("FAQ_CHUNK_OVERLAP_TOKENS", "50"))

    instruction_chunk_max_tokens: int = int(os.getenv("INSTRUCTION_CHUNK_MAX_TOKENS", "700"))
    instruction_chunk_overlap_tokens: int = int(os.getenv("INSTRUCTION_CHUNK_OVERLAP_TOKENS", "100"))

    troubleshooting_chunk_max_tokens: int = int(
        os.getenv("TROUBLESHOOTING_CHUNK_MAX_TOKENS", "650")
    )
    troubleshooting_chunk_overlap_tokens: int = int(
        os.getenv("TROUBLESHOOTING_CHUNK_OVERLAP_TOKENS", "80")
    )

    policy_chunk_max_tokens: int = int(os.getenv("POLICY_CHUNK_MAX_TOKENS", "900"))
    policy_chunk_overlap_tokens: int = int(os.getenv("POLICY_CHUNK_OVERLAP_TOKENS", "120"))

    support_contacts_chunk_max_tokens: int = int(
        os.getenv("SUPPORT_CONTACTS_CHUNK_MAX_TOKENS", "500")
    )
    support_contacts_chunk_overlap_tokens: int = int(
        os.getenv("SUPPORT_CONTACTS_CHUNK_OVERLAP_TOKENS", "0")
    )

    reference_chunk_max_tokens: int = int(os.getenv("REFERENCE_CHUNK_MAX_TOKENS", "800"))
    reference_chunk_overlap_tokens: int = int(os.getenv("REFERENCE_CHUNK_OVERLAP_TOKENS", "100"))

    chat_history_max_messages: int = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "10"))

    intent_classifier_enabled: bool = _env_bool("INTENT_CLASSIFIER_ENABLED", True)
    intent_model: str = os.getenv("INTENT_MODEL", "gpt-5-nano")
    intent_prompt_file: str = _resolve_path(
        os.getenv("INTENT_PROMPT_FILE", ""), "./prompts/intent_classifier_prompt.md"
    )
    intent_confidence_threshold: float = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.65"))
    intent_offtopic_confidence_threshold: float = float(
        os.getenv("INTENT_OFFTOPIC_CONFIDENCE_THRESHOLD", "0.80")
    )
    intent_reasoning_effort: str = os.getenv("INTENT_REASONING_EFFORT", "low")

    support_email: str = os.getenv("SUPPORT_EMAIL", "it@medretail.com")
    support_email_enabled: bool = _env_bool("SUPPORT_EMAIL_ENABLED", True)
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", "")
    support_telegram: str = os.getenv("SUPPORT_TELEGRAM", "@it_support_medretail")
    support_phone: str = os.getenv("SUPPORT_PHONE", "+7 (495) 100-10-01")
    support_teams: str = os.getenv("SUPPORT_TEAMS", "")
    support_portal: str = os.getenv("SUPPORT_PORTAL", "")

    system_prompt_file: str = _resolve_path(
        os.getenv("SYSTEM_PROMPT_FILE", ""), "./prompts/system_prompt.md"
    )

    log_backend: str = os.getenv("LOG_BACKEND", "sqlite").lower()
    sqlite_log_path: str = _resolve_path(
        os.getenv("SQLITE_LOG_PATH", ""), "./logs/app_logs.sqlite3"
    )

    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "gp_itbot_logs")
    postgres_user: str = os.getenv("POSTGRES_USER", "gp_itbot")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "gp_itbot_password")

    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "dev")


settings = Settings()
