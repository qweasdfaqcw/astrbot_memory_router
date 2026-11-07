"""按会话的检索限流。

两个维度：
- 每分钟检索次数；
- 每小时注入 token 预算。

只限制“自动预取/工具检索”，不限制后台维护与用户显式命令的读取。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(slots=True)
class RateDecision:
    allowed: bool
    reason: str = ""
    retry_after_seconds: int = 0


class RateLimiter:
    def __init__(self, *, per_minute: int = 20, tokens_per_hour: int = 12000):
        self.per_minute = max(1, int(per_minute))
        self.tokens_per_hour = max(100, int(tokens_per_hour))
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._tokens: dict[str, deque[tuple[float, int]]] = defaultdict(deque)

    def _prune(self, session_id: str, now: float) -> None:
        calls = self._calls[session_id]
        while calls and now - calls[0] > 60:
            calls.popleft()
        tokens = self._tokens[session_id]
        while tokens and now - tokens[0][0] > 3600:
            tokens.popleft()

    def check(self, session_id: str, *, estimated_tokens: int = 0) -> RateDecision:
        now = time.monotonic()
        key = session_id or "default"
        self._prune(key, now)
        calls = self._calls[key]
        if len(calls) >= self.per_minute:
            retry = max(1, int(60 - (now - calls[0])))
            return RateDecision(False, "每分钟检索次数超限", retry)
        used = sum(item[1] for item in self._tokens[key])
        if used + max(0, estimated_tokens) > self.tokens_per_hour:
            return RateDecision(False, "每小时注入 token 预算超限", 3600)
        return RateDecision(True)

    def consume(self, session_id: str, *, tokens: int = 0) -> None:
        now = time.monotonic()
        key = session_id or "default"
        self._prune(key, now)
        self._calls[key].append(now)
        if tokens > 0:
            self._tokens[key].append((now, int(tokens)))

    def clear(self, session_id: str = "") -> None:
        if session_id:
            self._calls.pop(session_id, None)
            self._tokens.pop(session_id, None)
        else:
            self._calls.clear()
            self._tokens.clear()
