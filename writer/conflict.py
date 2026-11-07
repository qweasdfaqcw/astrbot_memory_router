"""去重合并与冲突检测。

这里只做“发现冲突并标注”，不调用小模型裁决真伪。冲突记录保留双方 ID，
由用户确认/纠正命令解决，或由后续用户本人陈述覆盖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from utils.text import jaccard, tokenize

_NEGATION_RE = re.compile(
    r"(?i)\b(not|never|no|don't|doesn't|didn't|isn't|aren't|cannot|can't)\b|"
    r"不喜欢|不想要|不愿意|不是|没有|没|不|别|禁止|勿|非|从不|绝不"
)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_CONFLICT_TYPES = {"fact", "preference", "constraint", "relationship", "task_state"}


@dataclass(slots=True)
class ConflictFinding:
    existing: dict[str, Any]
    conflict_type: str
    similarity: float
    reason: str


def normalized_hash_payload(content: str) -> str:
    return re.sub(r"\s+", "", content.strip().lower())


def has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(text or ""))


def polarity_conflict(a: str, b: str) -> bool:
    return has_negation(a) != has_negation(b)


def numbers_conflict(a: str, b: str) -> bool:
    numbers_a = set(_NUMBER_RE.findall(a or ""))
    numbers_b = set(_NUMBER_RE.findall(b or ""))
    return bool(numbers_a and numbers_b and numbers_a.isdisjoint(numbers_b))


def is_near_duplicate(a: str, b: str, threshold: float = 0.9) -> bool:
    return jaccard(a, b) >= threshold


def detect_conflicts(
    new_content: str,
    new_type: str,
    new_source: str,
    existing: list[dict[str, Any]],
) -> list[ConflictFinding]:
    """对候选与新记忆做冲突检测，返回所有值得记录的冲突。"""
    if new_type not in _CONFLICT_TYPES:
        return []
    findings: list[ConflictFinding] = []
    new_tokens = set(tokenize(new_content))
    for old in existing:
        old_id = str(old.get("id") or "")
        if not old_id:
            continue
        old_content = str(old.get("content") or "")
        old_type = str(old.get("memory_type") or "")
        old_source = str(old.get("source") or "")
        if old_type != new_type:
            continue
        similarity = jaccard(new_content, old_content)
        shared = new_tokens & set(tokenize(old_content))
        # 转述 vs 本人陈述不作为“真假冲突”，只作为来源差异；避免低质量冲突刷屏。
        if {new_source, old_source} == {"user", "third_party"}:
            continue
        if similarity >= 0.9:
            continue
        if similarity < 0.28 and len(shared) < 2:
            continue
        reason = ""
        conflict_type = ""
        if polarity_conflict(new_content, old_content) and similarity >= 0.2:
            reason = "肯定/否定方向相反"
            conflict_type = "negation"
        elif numbers_conflict(new_content, old_content) and similarity >= 0.35:
            reason = "关键数值不一致"
            conflict_type = "numeric"
        elif new_type == "preference" and similarity >= 0.45:
            reason = "同一对象上的偏好表述可能已变化"
            conflict_type = "preference_direction"
        elif new_type in {"fact", "relationship"} and similarity >= 0.55:
            reason = "同一槽位出现不同说法"
            conflict_type = "value_mismatch"
        if conflict_type:
            findings.append(
                ConflictFinding(
                    existing=old,
                    conflict_type=conflict_type,
                    similarity=similarity,
                    reason=reason,
                )
            )
    return findings
