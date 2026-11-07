"""短期会话缓冲。

长期记忆在外部库，短期上下文留在内存。这里保存最近若干轮消息，作为小模型
抽取与查询改写的上下文。插件重启后缓冲会重建；已提交的记忆不受影响。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

from utils.ids import new_trace_id
from utils.time_utils import now_iso
from utils.text import compact_text, normalize_text


@dataclass(slots=True)
class Turn:
    id: str
    role: str
    content: str
    created_at: str = field(default_factory=now_iso)
    extracted: bool = False
    suppressed: bool = False


class SessionBuffer:
    def __init__(self, *, max_turns: int = 20, ttl_seconds: int = 3600):
        self.max_turns = max(4, max_turns)
        self.ttl_seconds = max(60, ttl_seconds)
        self._turns: dict[str, deque[Turn]] = defaultdict(
            lambda: deque(maxlen=self.max_turns)
        )
        self._last_active: dict[str, float] = {}

    def _prune(self, session_id: str) -> None:
        now = time.monotonic()
        last = self._last_active.get(session_id, now)
        if now - last > self.ttl_seconds:
            self._turns.pop(session_id, None)
            self._last_active.pop(session_id, None)
            return
        self._last_active[session_id] = now
        # 顺带清理全局过期会话，避免长时间运行内存增长。
        expired = [
            key for key, ts in self._last_active.items()
            if now - ts > self.ttl_seconds
        ]
        for key in expired:
            self._turns.pop(key, None)
            self._last_active.pop(key, None)

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        suppressed: bool = False,
    ) -> Turn | None:
        text = normalize_text(content)
        if not session_id or not text:
            return None
        self._prune(session_id)
        turn = Turn(
            id=new_trace_id().replace("trc_", "turn_"),
            role=role,
            content=text,
            suppressed=suppressed,
        )
        self._turns[session_id].append(turn)
        return turn

    def turns(self, session_id: str, *, limit: int = 0) -> list[Turn]:
        self._prune(session_id)
        values = list(self._turns.get(session_id, ()))
        return values[-limit:] if limit > 0 else values

    def pending_turns(self, session_id: str) -> list[Turn]:
        self._prune(session_id)
        return [
            turn for turn in self._turns.get(session_id, ())
            if not turn.extracted and not turn.suppressed
        ]

    def mark_extracted(self, session_id: str, turn_ids: Iterable[str]) -> None:
        wanted = set(turn_ids)
        for turn in self._turns.get(session_id, ()):
            if turn.id in wanted:
                turn.extracted = True

    def suppress_text(self, session_id: str, text: str) -> None:
        """用户说“别记”时，把当前缓冲中相同内容标记为不抽取。"""
        target = compact_text(text)
        if not target:
            return
        for turn in self._turns.get(session_id, ()):
            if compact_text(turn.content) == target:
                turn.suppressed = True
                turn.extracted = True

    def clear(self, session_id: str = "") -> None:
        if session_id:
            self._turns.pop(session_id, None)
            self._last_active.pop(session_id, None)
        else:
            self._turns.clear()
            self._last_active.clear()

    def session_count(self) -> int:
        return len(self._turns)
