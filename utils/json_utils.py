"""严格 JSON 解析与类型校验工具。

小模型输出经常包含 Markdown 代码块、前后解释文字或尾逗号。这里只做
「尽量安全地取出 JSON」，业务字段验证由调用方完成，绝不把解析失败当成成功。
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class JSONParseError(ValueError):
    """JSON 无法解析。"""


def _candidate_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    for match in _FENCE_RE.finditer(text or ""):
        fragments.append(match.group(1).strip())
    stripped = (text or "").strip()
    if stripped:
        fragments.append(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        fragments.append(stripped[start : end + 1])
    start, end = stripped.find("["), stripped.rfind("]")
    if start != -1 and end > start:
        fragments.append(stripped[start : end + 1])
    # 去重并保持顺序
    seen: set[str] = set()
    result: list[str] = []
    for item in fragments:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def loads_lenient(text: str) -> Any:
    """从文本中提取并解析 JSON；失败抛出 JSONParseError。"""
    errors: list[str] = []
    for fragment in _candidate_fragments(text):
        for candidate in (fragment, _remove_trailing_commas(fragment)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
    raise JSONParseError(errors[-1] if errors else "empty JSON output")


def _remove_trailing_commas(text: str) -> str:
    """移除对象/数组末尾多余逗号。"""
    return re.sub(r",\s*([}\]])", r"\1", text)


def as_dict(value: Any) -> dict[str, Any]:
    """确保是 dict，否则返回空字典。"""
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """确保是 list。"""
    return value if isinstance(value, list) else []


def coerce_float(value: Any, default: float = 0.0, minimum: float | None = None,
                 maximum: float | None = None) -> float:
    """把模型输出转为 float 并夹紧范围。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_int(value: Any, default: int = 0, minimum: int | None = None,
               maximum: int | None = None) -> int:
    """把模型输出转为 int 并夹紧范围。"""
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_bool(value: Any, default: bool = False) -> bool:
    """把模型输出的常见布尔写法转为 bool。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "是", "对"}:
            return True
        if normalized in {"false", "no", "0", "否", "不"}:
            return False
    return default


def dump(data: Any) -> str:
    """稳定 JSON 输出，便于日志与工具返回。"""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
