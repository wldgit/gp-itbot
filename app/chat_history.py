from __future__ import annotations

from collections import deque
from threading import Lock


class ChatHistoryStore:
    """In-memory chat turns per Telegram user (role/content), bounded for token safety."""

    def __init__(self, max_messages: int) -> None:
        self._max_messages = max(2, max_messages)
        self._lock = Lock()
        self._by_user: dict[int, deque[dict[str, str]]] = {}

    def get_messages(self, user_id: int) -> list[dict[str, str]]:
        with self._lock:
            dq = self._by_user.get(user_id)
            if not dq:
                return []
            return [{"role": m["role"], "content": m["content"]} for m in dq]

    def append(self, user_id: int, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        with self._lock:
            dq = self._by_user.setdefault(user_id, deque(maxlen=self._max_messages))
            dq.append({"role": role, "content": content})

    def clear(self, user_id: int) -> None:
        with self._lock:
            self._by_user.pop(user_id, None)
