"""会话级运行状态。

只保存在内存中，可随插件重启重建。不放长期事实；长期数据统一落 SQLite。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class SessionState:
    session_id: str
    last_extract_at: float = 0.0
    last_recall_at: float = 0.0
    last_trace_id: str = ""
    pending_extract: bool = False
    injected_at: dict[str, float] = field(default_factory=dict)
    last_query: str = ""
    last_injected_text: str = ""

    def mark_injected(self, memory_ids: list[str], query: str, trace_id: str) -> None:
        stamp = time.monotonic()
        for memory_id in memory_ids:
            self.injected_at[memory_id] = stamp
        self.last_recall_at = stamp
        self.last_query = query
        self.last_trace_id = trace_id

    def was_recently_injected(self, memory_id: str, *, ttl_seconds: int = 600) -> bool:
        stamp = self.injected_at.get(memory_id)
        if stamp is None:
            return False
        return time.monotonic() - stamp <= max(1, ttl_seconds)

    def prune_injected(self, *, ttl_seconds: int = 3600) -> None:
        cutoff = time.monotonic() - max(1, ttl_seconds)
        self.injected_at = {
            key: stamp for key, stamp in self.injected_at.items() if stamp >= cutoff
        }


class SessionStateStore:
    def __init__(self) -> None:
        self._states: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        key = session_id or "default"
        state = self._states.get(key)
        if state is None:
            state = SessionState(session_id=key)
            self._states[key] = state
        return state

    def clear(self, session_id: str = "") -> None:
        if session_id:
            self._states.pop(session_id, None)
        else:
            self._states.clear()
