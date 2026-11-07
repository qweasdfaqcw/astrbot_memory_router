"""ID 工具。"""

from __future__ import annotations

import uuid


def new_memory_id() -> str:
    """生成可读、可排序的短记忆 ID。"""
    return "mem_" + uuid.uuid4().hex[:20]


def new_trace_id() -> str:
    """生成召回链路 ID。"""
    return "trc_" + uuid.uuid4().hex[:16]


def new_conflict_id() -> str:
    """生成冲突记录 ID。"""
    return "cft_" + uuid.uuid4().hex[:16]


def new_audit_id() -> str:
    """生成审计记录 ID。"""
    return "aud_" + uuid.uuid4().hex[:16]
