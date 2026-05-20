import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json

from app.app_logger import logger
from app.config import settings
from app.text_safety import mask_sensitive_data


@dataclass
class InteractionLog:
    user_id: int
    username: str | None
    question: str
    answer: str
    sources: list[str]
    transferred_to_support: bool
    response_time_ms: int
    status: str
    error_message: str | None = None


class BaseLogRepository:
    def init_storage(self) -> None:
        raise NotImplementedError

    def log_interaction(self, event: InteractionLog) -> None:
        raise NotImplementedError

    def log_runtime_event(
        self,
        level: str,
        event_type: str,
        message: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError


class SQLiteLogRepository(BaseLogRepository):
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_storage()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def init_storage(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at INTEGER NOT NULL,
                    user_id INTEGER,
                    username TEXT,
                    question TEXT,
                    answer TEXT,
                    sources TEXT,
                    transferred_to_support INTEGER,
                    response_time_ms INTEGER,
                    status TEXT,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at INTEGER NOT NULL,
                    level TEXT,
                    event_type TEXT,
                    message TEXT,
                    payload TEXT
                )
                """
            )

    def log_interaction(self, event: InteractionLog) -> None:
        safe_event = asdict(event)
        safe_event["question"] = mask_sensitive_data(event.question)
        safe_event["answer"] = mask_sensitive_data(event.answer)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interactions (
                    created_at, user_id, username, question, answer, sources,
                    transferred_to_support, response_time_ms, status, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()),
                    event.user_id,
                    event.username,
                    safe_event["question"],
                    safe_event["answer"],
                    json.dumps(event.sources, ensure_ascii=False),
                    1 if event.transferred_to_support else 0,
                    event.response_time_ms,
                    event.status,
                    event.error_message,
                ),
            )

    def log_runtime_event(
        self,
        level: str,
        event_type: str,
        message: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_events (created_at, level, event_type, message, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()),
                    level,
                    event_type,
                    message,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )


class PostgresLogRepository(BaseLogRepository):
    def __init__(self):
        self.init_storage()

    def _connect(self):
        return psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )

    def init_storage(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS interactions (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        user_id BIGINT,
                        username TEXT,
                        question TEXT,
                        answer TEXT,
                        sources JSONB,
                        transferred_to_support BOOLEAN,
                        response_time_ms INTEGER,
                        status TEXT,
                        error_message TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_events (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        level TEXT,
                        event_type TEXT,
                        message TEXT,
                        payload JSONB
                    )
                    """
                )

    def log_interaction(self, event: InteractionLog) -> None:
        safe_question = mask_sensitive_data(event.question)
        safe_answer = mask_sensitive_data(event.answer)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO interactions (
                        user_id, username, question, answer, sources,
                        transferred_to_support, response_time_ms, status, error_message
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.user_id,
                        event.username,
                        safe_question,
                        safe_answer,
                        Json(event.sources),
                        event.transferred_to_support,
                        event.response_time_ms,
                        event.status,
                        event.error_message,
                    ),
                )

    def log_runtime_event(
        self,
        level: str,
        event_type: str,
        message: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runtime_events (level, event_type, message, payload)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (level, event_type, message, Json(payload or {})),
                )


def create_log_repository() -> BaseLogRepository:
    if settings.log_backend == "postgres":
        logger.info("Initializing PostgreSQL logging backend")
        return PostgresLogRepository()

    logger.info("Initializing SQLite logging backend: %s", settings.sqlite_log_path)
    return SQLiteLogRepository(settings.sqlite_log_path)


log_repository = create_log_repository()


def log_runtime_event(
    level: str,
    event_type: str,
    message: str,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    try:
        log_repository.log_runtime_event(level, event_type, message, payload)
    except Exception:
        logger.exception("Failed to persist runtime event: %s | %s", event_type, message)
