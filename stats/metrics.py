"""内存评估指标。进程内累计，重启清零；结构化日志与 /memory stats 可读。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MetricsCollector:
    recalls: int = 0
    injections: int = 0
    injected_items: int = 0
    injected_tokens: int = 0
    candidate_total: int = 0
    blocked_sensitive: int = 0
    blocked_injection: int = 0
    degraded: int = 0
    writes: int = 0
    merges: int = 0
    conflicts: int = 0
    latencies_ms: list[int] = field(default_factory=list)
    errors: int = 0

    def observe_recall(
        self,
        *,
        latency_ms: int,
        candidates: int,
        injected_items: int,
        injected_tokens: int,
        blocked_sensitive: int = 0,
        blocked_injection: int = 0,
    ) -> None:
        self.recalls += 1
        self.candidate_total += max(0, candidates)
        self.latencies_ms.append(max(0, int(latency_ms)))
        if len(self.latencies_ms) > 500:
            self.latencies_ms = self.latencies_ms[-500:]
        if injected_items > 0:
            self.injections += 1
            self.injected_items += injected_items
            self.injected_tokens += max(0, injected_tokens)
        self.blocked_sensitive += max(0, blocked_sensitive)
        self.blocked_injection += max(0, blocked_injection)

    def observe_write(self, *, created: int, merged: int, conflicts: int, degraded: bool) -> None:
        self.writes += max(0, created)
        self.merges += max(0, merged)
        self.conflicts += max(0, conflicts)
        if degraded:
            self.degraded += 1

    def observe_error(self) -> None:
        self.errors += 1

    def snapshot(self) -> dict[str, Any]:
        latencies = self.latencies_ms or [0]
        return {
            "recalls": self.recalls,
            "injections": self.injections,
            "injected_items": self.injected_items,
            "injected_tokens": self.injected_tokens,
            "candidate_total": self.candidate_total,
            "blocked_sensitive": self.blocked_sensitive,
            "blocked_injection": self.blocked_injection,
            "degraded": self.degraded,
            "writes": self.writes,
            "merges": self.merges,
            "conflicts": self.conflicts,
            "errors": self.errors,
            "latency_ms_p50": int(statistics.median(latencies)),
            "latency_ms_p95": int(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]),
            "injection_precision_proxy": (
                round(self.injected_items / max(1, self.recalls), 4)
            ),
        }

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]
