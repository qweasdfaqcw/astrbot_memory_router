"""时间处理工具。

所有持久化时间统一使用 UTC ISO-8601 字符串，避免跨时区比较问题。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def utcnow() -> datetime:
    """返回当前 UTC 时间（带时区）。"""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """当前 UTC 时间的 ISO 字符串。"""
    return utcnow().isoformat(timespec="seconds")


def to_iso(value: datetime | None = None) -> str:
    """把 datetime 转为 UTC ISO 字符串。"""
    if value is None:
        value = utcnow()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str | None) -> datetime | None:
    """解析 ISO 时间，兼容 ``Z`` 后缀；失败返回 None。"""
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def age_days(value: str | None, *, now: datetime | None = None) -> float:
    """返回某个 ISO 时间距现在多少天；无法解析时返回一个很大的值。"""
    parsed = parse_iso(value)
    if parsed is None:
        return 36500.0
    current = now or utcnow()
    return max(0.0, (current - parsed).total_seconds() / 86400.0)


def days_ago(days: float, *, now: datetime | None = None) -> str:
    """返回 N 天前的 ISO 时间。"""
    current = now or utcnow()
    return to_iso(current - timedelta(days=days))


def half_life_decay(age: float, half_life_days: float) -> float:
    """指数半衰期衰减；半衰期非正时按 1 天处理。"""
    half_life = max(0.5, float(half_life_days))
    return 0.5 ** (max(0.0, age) / half_life)
