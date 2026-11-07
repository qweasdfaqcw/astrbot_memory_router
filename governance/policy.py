"""权限与敏感策略。

这里是“能不能读/能不能返回”的最后一道业务闸门：
- 主模型与路由小模型永远不能读取 sensitive/secret 记忆；
- 只有用户显式命令（caller=user 且 require_user_auth）可以读取自己的敏感记忆；
- owner 隔离由 SQLiteStore 再校验一次，双保险。
"""

from __future__ import annotations

from typing import Any, Mapping

from store.models import (
    SENSITIVITY_RANK,
    AccessContext,
    SearchFilters,
    unique_strings,
)
from writer.sanitizer import mask_sensitive


class PermissionDenied(PermissionError):
    """越权或敏感策略拒绝。"""


def can_read_sensitive(context: AccessContext, caller: str, require_user_auth: bool) -> bool:
    return (
        caller == "user"
        and require_user_auth
        and bool(context.user_id)
        and not context.is_group
    )


def sensitivity_clearance(caller: str, context: AccessContext, require_user_auth: bool) -> str:
    """返回当前调用可返回的最高敏感级别。"""
    if can_read_sensitive(context, caller, require_user_auth):
        return "secret"
    return "normal"


def filter_sensitivity_for_caller(
    filters: SearchFilters,
    caller: str,
    context: AccessContext,
    require_user_auth: bool,
) -> SearchFilters:
    """根据调用者收紧 sensitivity_max，并复制对象避免修改调用方状态。"""
    clearance = sensitivity_clearance(caller, context, require_user_auth)
    requested_rank = SENSITIVITY_RANK.get(filters.sensitivity_max, 0)
    clearance_rank = SENSITIVITY_RANK[clearance]
    effective = "normal"
    for name, rank in SENSITIVITY_RANK.items():
        if rank == min(requested_rank, clearance_rank):
            effective = name
            break
    copied = SearchFilters(**{
        field: getattr(filters, field)
        for field in SearchFilters.__dataclass_fields__
    })
    copied.sensitivity_max = effective
    return copied


def validate_filter_scope(context: AccessContext, filters: SearchFilters) -> None:
    """请求的 owner 范围不能超出当前会话可见范围。"""
    if filters.owner_scope:
        allowed = {scope for scope, _ in context.accessible_scopes()}
        if filters.owner_scope not in allowed:
            raise PermissionDenied(f"无权访问 owner_scope={filters.owner_scope}")
        if filters.owner_id and not context.can_access(filters.owner_scope, filters.owner_id):
            raise PermissionDenied("无权访问指定 owner_id 的记忆")
    if filters.owner_id and not filters.owner_scope:
        if filters.owner_id in {"global", "*"}:
            return
        if not any(owner_id == filters.owner_id for _, owner_id in context.accessible_scopes()):
            # 不能用 owner_id 把查询扩展到其他用户/群。
            raise PermissionDenied("无权访问指定 owner_id 的记忆")


def ensure_memory_access(context: AccessContext, memory: Mapping[str, Any]) -> None:
    scope = str(memory.get("owner_scope") or "")
    owner_id = str(memory.get("owner_id") or "")
    if not context.can_access(scope, owner_id):
        raise PermissionDenied("记忆不属于当前会话可见范围")


def ensure_memory_mutation(context: AccessContext, memory: Mapping[str, Any]) -> None:
    """修改/删除权限：个人记忆归本人；群/全局记忆需要管理员。"""
    ensure_memory_access(context, memory)
    scope = str(memory.get("owner_scope") or "personal")
    if scope in {"group", "global"} and not context.is_admin:
        raise PermissionDenied("修改群记忆或全局记忆需要管理员权限")


def can_hard_delete(context: AccessContext, *, enabled: bool) -> bool:
    return bool(enabled and context.is_admin)


def sanitize_audit_detail(detail: Mapping[str, Any]) -> dict[str, Any]:
    """审计详情脱敏，避免审计日志变成隐私泄露源。"""
    result: dict[str, Any] = {}
    for key, value in detail.items():
        if isinstance(value, str):
            result[key] = mask_sensitive(value)[:1000]
        elif isinstance(value, (int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = [
                mask_sensitive(str(item))[:300] if not isinstance(item, (int, float, bool)) else item
                for item in list(value)[:30]
            ]
        elif isinstance(value, Mapping):
            result[key] = sanitize_audit_detail(value)
        else:
            result[key] = mask_sensitive(str(value))[:300]
    return result


def normalize_filter_input(value: Any) -> dict[str, Any]:
    """memory_access 的 filters 既可能是对象也可能是 JSON 字符串。"""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        import json

        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def normalize_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return unique_strings(value.replace("，", ",").split(","), limit=50)
    if isinstance(value, (list, tuple, set)):
        return unique_strings(value, limit=50)
    return []
