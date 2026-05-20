from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2

from app.app_logger import logger
from app.config import settings


@dataclass
class SupportEscalationRecord:
    telegram_user_id: int
    telegram_username: str | None
    telegram_full_name: str
    problem_summary: str
    conversation_excerpt: str
    reason: str
    email_to: str
    email_subject: str
    email_sent: bool
    email_sent_at: int | None = None
    error_message: str | None = None
    id: int | None = None
    created_at: int | None = None


class BaseSupportEscalationRepository:
    def init_storage(self) -> None:
        raise NotImplementedError

    def create(self, record: SupportEscalationRecord) -> int:
        raise NotImplementedError

    def mark_sent(self, record_id: int, sent_at: int) -> None:
        raise NotImplementedError

    def mark_failed(self, record_id: int, error_message: str) -> None:
        raise NotImplementedError


class SQLiteSupportEscalationRepository(BaseSupportEscalationRepository):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._memory_conn: sqlite3.Connection | None = None
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_storage()

    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:")
            return self._memory_conn
        return sqlite3.connect(self.db_path)

    def init_storage(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS support_escalations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_user_id INTEGER NOT NULL,
                    telegram_username TEXT,
                    telegram_full_name TEXT,
                    problem_summary TEXT,
                    conversation_excerpt TEXT,
                    reason TEXT,
                    email_to TEXT,
                    email_subject TEXT,
                    email_sent INTEGER NOT NULL DEFAULT 0,
                    email_sent_at INTEGER,
                    error_message TEXT,
                    created_at INTEGER NOT NULL
                )
                """
            )

    def create(self, record: SupportEscalationRecord) -> int:
        created_at = int(time.time())
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO support_escalations (
                    telegram_user_id, telegram_username, telegram_full_name,
                    problem_summary, conversation_excerpt, reason,
                    email_to, email_subject, email_sent, email_sent_at,
                    error_message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.telegram_user_id,
                    record.telegram_username,
                    record.telegram_full_name,
                    record.problem_summary,
                    record.conversation_excerpt,
                    record.reason,
                    record.email_to,
                    record.email_subject,
                    1 if record.email_sent else 0,
                    record.email_sent_at,
                    record.error_message,
                    created_at,
                ),
            )
            return int(cursor.lastrowid)

    def mark_sent(self, record_id: int, sent_at: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE support_escalations
                SET email_sent = 1, email_sent_at = ?, error_message = NULL
                WHERE id = ?
                """,
                (sent_at, record_id),
            )

    def mark_failed(self, record_id: int, error_message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE support_escalations
                SET email_sent = 0, error_message = ?
                WHERE id = ?
                """,
                (error_message, record_id),
            )


class PostgresSupportEscalationRepository(BaseSupportEscalationRepository):
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
                    CREATE TABLE IF NOT EXISTS support_escalations (
                        id BIGSERIAL PRIMARY KEY,
                        telegram_user_id BIGINT NOT NULL,
                        telegram_username TEXT,
                        telegram_full_name TEXT,
                        problem_summary TEXT,
                        conversation_excerpt TEXT,
                        reason TEXT,
                        email_to TEXT,
                        email_subject TEXT,
                        email_sent BOOLEAN NOT NULL DEFAULT FALSE,
                        email_sent_at TIMESTAMPTZ,
                        error_message TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )

    def create(self, record: SupportEscalationRecord) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO support_escalations (
                        telegram_user_id, telegram_username, telegram_full_name,
                        problem_summary, conversation_excerpt, reason,
                        email_to, email_subject, email_sent, email_sent_at,
                        error_message
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        record.telegram_user_id,
                        record.telegram_username,
                        record.telegram_full_name,
                        record.problem_summary,
                        record.conversation_excerpt,
                        record.reason,
                        record.email_to,
                        record.email_subject,
                        record.email_sent,
                        datetime.fromtimestamp(record.email_sent_at, tz=timezone.utc)
                        if record.email_sent_at
                        else None,
                        record.error_message,
                    ),
                )
                row = cur.fetchone()
                return int(row[0])

    def mark_sent(self, record_id: int, sent_at: int) -> None:
        sent_dt = datetime.fromtimestamp(sent_at, tz=timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE support_escalations
                    SET email_sent = TRUE, email_sent_at = %s, error_message = NULL
                    WHERE id = %s
                    """,
                    (sent_dt, record_id),
                )

    def mark_failed(self, record_id: int, error_message: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE support_escalations
                    SET email_sent = FALSE, error_message = %s
                    WHERE id = %s
                    """,
                    (error_message, record_id),
                )


def create_support_escalation_repository() -> BaseSupportEscalationRepository:
    if settings.log_backend == "postgres":
        logger.info("Initializing PostgreSQL support_escalations repository")
        repo = PostgresSupportEscalationRepository()
        repo.init_storage()
        return repo

    logger.info(
        "Initializing SQLite support_escalations repository: %s",
        settings.sqlite_log_path,
    )
    return SQLiteSupportEscalationRepository(settings.sqlite_log_path)


support_escalation_repository: Optional[BaseSupportEscalationRepository] = None


def get_support_escalation_repository() -> BaseSupportEscalationRepository:
    global support_escalation_repository
    if support_escalation_repository is None:
        support_escalation_repository = create_support_escalation_repository()
    return support_escalation_repository
