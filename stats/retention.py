"""留存分计算。

留存分只用于“排序、压缩、归档”的参考，绝不能单独决定删除。公式：

    初始重要性 × 时间衰减
    + min(访问增益上限, 分层访问增益)
    + 用户确认增益
    + 情绪显著性增益
    - 纠正/过时惩罚

访问增益分层加权并取 log，同时用“不同会话数”削弱同一会话反复召回造成的
自我强化；候选召回、注入、实际使用、用户确认分别计数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from store.models import HALF_LIFE_DAYS, AccessStats
from utils.time_utils import age_days, half_life_decay, parse_iso

_ACCESS_WEIGHTS = {
    "candidate_recalled": 0.015,
    "injected": 0.035,
    "used_by_main_model": 0.09,
    "user_confirmed": 0.16,
    "revived": 0.02,
}
_PENALTY_WEIGHTS = {
    "user_corrected": 0.18,
}
_MAX_GAIN = 0.28


@dataclass(slots=True)
class RetentionBreakdown:
    score: float
    initial: float
    decay: float
    access_gain: float
    confirmation_gain: float
    emotion_gain: float
    correction_penalty: float
    pinned: bool = False
    formula: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "initial": round(self.initial, 4),
            "decay": round(self.decay, 4),
            "access_gain": round(self.access_gain, 4),
            "confirmation_gain": round(self.confirmation_gain, 4),
            "emotion_gain": round(self.emotion_gain, 4),
            "correction_penalty": round(self.correction_penalty, 4),
            "pinned": self.pinned,
            "formula": self.formula,
            "details": self.details,
        }

    def explain(self) -> str:
        return (
            f"留存分 {self.score:.3f} = 初始 {self.initial:.3f} × 衰减 {self.decay:.3f}"
            f" + 访问增益 {self.access_gain:.3f}"
            f" + 确认增益 {self.confirmation_gain:.3f}"
            f" + 情绪增益 {self.emotion_gain:.3f}"
            f" - 纠正惩罚 {self.correction_penalty:.3f}"
            + ("（用户标记长期保留）" if self.pinned else "")
        )


def compute_retention(
    memory: Mapping[str, Any],
    stats: AccessStats | None = None,
    *,
    now_iso_value: str | None = None,
) -> RetentionBreakdown:
    stats = stats or AccessStats()
    memory_type = str(memory.get("memory_type") or "fact")
    half_life = HALF_LIFE_DAYS.get(memory_type, 180.0)
    reference = (
        str(memory.get("last_verified_at") or "")
        or str(memory.get("updated_at") or "")
        or str(memory.get("created_at") or "")
    )
    age = age_days(reference, now=parse_iso(now_iso_value))
    decay = half_life_decay(age, half_life)
    initial = max(0.0, min(1.0, float(memory.get("importance") or 0.5))) * decay

    access_gain = 0.0
    for event_type, weight in _ACCESS_WEIGHTS.items():
        total = stats.count(event_type)
        recent = stats.recent(event_type)
        sessions = max(1, stats.sessions(event_type))
        # 近期访问额外加权；同会话重复访问按 sqrt 降权，避免自我强化。
        weighted_total = total + recent * 0.8
        effective = min(weighted_total, sessions * 3) + math.sqrt(
            max(0.0, weighted_total - sessions * 3)
        )
        access_gain += weight * math.log1p(max(0.0, effective))
    access_gain = min(_MAX_GAIN, max(0.0, access_gain))

    confirmation_gain = min(0.12, 0.08 * math.log1p(stats.sessions("user_confirmed")))
    emotion_gain = 0.05 * max(0.0, min(1.0, float(memory.get("emotion_arousal") or 0.0)))

    correction_penalty = 0.0
    for event_type, weight in _PENALTY_WEIGHTS.items():
        correction_penalty += weight * min(3.0, stats.count(event_type))
    correction_penalty = min(0.45, correction_penalty)

    pinned = bool(memory.get("pinned"))
    score = initial + access_gain + confirmation_gain + emotion_gain - correction_penalty
    if pinned:
        score = max(score, 0.65)
    score = max(0.0, min(1.0, score))
    return RetentionBreakdown(
        score=score,
        initial=initial,
        decay=decay,
        access_gain=access_gain,
        confirmation_gain=confirmation_gain,
        emotion_gain=emotion_gain,
        correction_penalty=correction_penalty,
        pinned=pinned,
        formula="importance*decay + capped_access_gain + confirmation_gain + emotion_gain - correction_penalty",
        details={
            "memory_type": memory_type,
            "half_life_days": half_life,
            "age_days": round(age, 2),
            "counts": stats.to_dict(),
        },
    )


def should_archive(
    memory: Mapping[str, Any],
    breakdown: RetentionBreakdown,
    *,
    auto_archive: bool,
    archive_after_days: int,
    threshold: float,
) -> bool:
    """自动归档条件：未被用户保留、非约束、久未访问、留存分低。

    自动流程绝不删除；归档后仍可按标签检索/复活。
    """
    if not auto_archive or bool(memory.get("pinned")):
        return False
    if str(memory.get("memory_type")) in {"constraint"}:
        return False
    if str(memory.get("lifecycle")) != "active":
        return False
    reference = str(memory.get("last_access_at") or memory.get("updated_at") or memory.get("created_at") or "")
    if age_days(reference) < archive_after_days:
        return False
    return breakdown.score < threshold


def should_compress(
    memory: Mapping[str, Any],
    breakdown: RetentionBreakdown,
    *,
    auto_compress: bool,
    compress_after_days: int,
) -> bool:
    if not auto_compress or bool(memory.get("pinned")):
        return False
    if str(memory.get("memory_type")) != "event":
        return False
    reference = str(memory.get("updated_at") or memory.get("created_at") or "")
    return age_days(reference) >= compress_after_days and breakdown.score < 0.45


def retention_reason(memory: Mapping[str, Any], breakdown: RetentionBreakdown) -> str:
    """给 /memory why 用的可解释文本。"""
    reasons: list[str] = []
    if breakdown.pinned:
        reasons.append("用户 /memory keep 标记为长期保留，自动流程不会删除")
    if breakdown.decay < 0.5:
        reasons.append("时间衰减明显")
    if breakdown.access_gain >= 0.1:
        reasons.append("有持续实际访问")
    if breakdown.confirmation_gain > 0:
        reasons.append("用户确认过")
    if breakdown.correction_penalty > 0:
        reasons.append("有纠正/否定记录")
    if not reasons:
        reasons.append("按类型初始重要性保留")
    return "；".join(reasons)
