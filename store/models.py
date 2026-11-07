"""存储层数据模型与常量。

这里集中定义记忆类型、生命周期、可见范围、检索过滤条件，避免各模块散落
魔法字符串。SQLite 行统一转为普通 dict，便于跨模块与 JSON 序列化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from utils.json_utils import as_list, coerce_float, coerce_int
from utils.time_utils import now_iso

MEMORY_TYPES: tuple[str, ...] = (
    "fact",
    "preference",
    "event",
    "task_state",
    "relationship",
    "constraint",
)

OWNER_SCOPES: tuple[str, ...] = ("personal", "group", "global")
LIFECYCLES: tuple[str, ...] = ("active", "compressed", "archived", "soft_deleted")
SENSITIVITY_LEVELS: tuple[str, ...] = ("normal", "sensitive", "secret")
SENSITIVITY_RANK = {name: index for index, name in enumerate(SENSITIVITY_LEVELS)}
STATUSES: tuple[str, ...] = ("active", "disputed", "superseded", "deleted")

CALLERS = ("main_model", "router_small", "system", "user")
MODES = ("search", "fetch", "prefetch", "probe")
ACCESS_EVENT_TYPES = (
    "candidate_recalled",
    "injected",
    "used_by_main_model",
    "user_confirmed",
    "user_corrected",
    "revived",
)

# 不同 memory_type 的时间半衰期（天）。事实/偏好/关系更耐久，事件和任务状态更易过时。
HALF_LIFE_DAYS: dict[str, float] = {
    "fact": 365.0,
    "preference": 365.0,
    "event": 30.0,
    "task_state": 14.0,
    "relationship": 540.0,
    "constraint": 730.0,
}

# 初始重要性默认值；模型输出会被夹紧，不让模型把一切都标成 1.0。
DEFAULT_IMPORTANCE: dict[str, float] = {
    "fact": 0.55,
    "preference": 0.60,
    "event": 0.45,
    "task_state": 0.65,
    "relationship": 0.60,
    "constraint": 0.80,
}


@dataclass(slots=True)
class AccessContext:
    """一次调用的访问身份与隔离边界。

    该对象只由 AstrBot 适配层或测试代码构造，工具层不得接受模型自定义的
    owner 字段来扩大可见范围。
    """

    user_id: str
    session_id: str = ""
    group_id: str = ""
    umo: str = ""
    sender_name: str = ""
    is_group: bool = False
    is_admin: bool = False
    channel: str = ""
    allow_personal_in_group: bool = False

    @property
    def effective_user_id(self) -> str:
        return self.user_id or "anonymous"

    def accessible_scopes(self) -> list[tuple[str, str]]:
        """返回当前会话可读的所有 (owner_scope, owner_id)。"""
        scopes: list[tuple[str, str]] = [("global", "*")]
        if self.is_group:
            if self.group_id:
                scopes.append(("group", self.group_id))
            if self.allow_personal_in_group and self.user_id:
                scopes.append(("personal", self.user_id))
        elif self.user_id:
            scopes.append(("personal", self.user_id))
        return scopes

    def can_access(self, owner_scope: str, owner_id: str) -> bool:
        """严格判断是否可读某条记忆。"""
        if owner_scope == "global":
            return True
        if owner_scope == "group":
            return bool(self.is_group and self.group_id and owner_id == self.group_id)
        if owner_scope == "personal":
            if self.user_id and owner_id == self.user_id:
                return not self.is_group or self.allow_personal_in_group
            return False
        return False

    def write_scope(self) -> tuple[str, str]:
        """当前会话新记忆默认归属。"""
        if self.is_group and self.group_id:
            return "group", self.group_id
        return "personal", self.effective_user_id


@dataclass(slots=True)
class SearchFilters:
    """统一过滤条件。字段名与 memory_access 的 filters 对齐。"""

    time_range: dict[str, str] = field(default_factory=dict)
    scene: str = ""
    task: str = ""
    topic: str = ""
    entities: list[str] = field(default_factory=list)
    event_type: str = ""
    emotion: str = ""
    source: str = ""
    confidence_min: float = 0.0
    sensitivity_max: str = "normal"
    lifecycle: list[str] = field(default_factory=lambda: ["active"])
    owner_scope: str = ""
    owner_id: str = ""
    tags_any: list[str] = field(default_factory=list)
    tags_all: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    status: list[str] = field(default_factory=lambda: ["active", "disputed"])
    offset: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "SearchFilters":
        data = raw or {}
        time_range = data.get("time_range") or {}
        if not isinstance(time_range, Mapping):
            time_range = {}
        lifecycle = data.get("lifecycle")
        if isinstance(lifecycle, str):
            lifecycle_list = [lifecycle] if lifecycle else ["active"]
        elif isinstance(lifecycle, list):
            lifecycle_list = [str(item) for item in lifecycle if str(item)]
        else:
            lifecycle_list = ["active"]
        status = data.get("status")
        if isinstance(status, str):
            status_list = [status]
        elif isinstance(status, list) and status:
            status_list = [str(item) for item in status]
        else:
            status_list = ["active", "disputed"]
        sensitivity = str(data.get("sensitivity_max") or "normal")
        if sensitivity not in SENSITIVITY_RANK:
            sensitivity = "normal"
        return cls(
            time_range={
                "from": str(time_range.get("from") or ""),
                "to": str(time_range.get("to") or ""),
            },
            scene=str(data.get("scene") or "").strip(),
            task=str(data.get("task") or "").strip(),
            topic=str(data.get("topic") or "").strip(),
            entities=[str(item).strip() for item in as_list(data.get("entities")) if str(item).strip()],
            event_type=str(data.get("event_type") or "").strip(),
            emotion=str(data.get("emotion") or "").strip(),
            source=str(data.get("source") or "").strip(),
            confidence_min=coerce_float(data.get("confidence_min"), 0.0, 0.0, 1.0),
            sensitivity_max=sensitivity,
            lifecycle=lifecycle_list or ["active"],
            owner_scope=str(data.get("owner_scope") or "").strip(),
            owner_id=str(data.get("owner_id") or "").strip(),
            tags_any=[str(item).strip() for item in as_list(data.get("tags_any")) if str(item).strip()],
            tags_all=[str(item).strip() for item in as_list(data.get("tags_all")) if str(item).strip()],
            memory_ids=[
                str(item).strip() for item in as_list(data.get("memory_ids")) if str(item).strip()
            ],
            status=status_list,
            offset=coerce_int(data.get("offset"), 0, 0, 100000),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "time_range": self.time_range,
            "scene": self.scene,
            "task": self.task,
            "topic": self.topic,
            "entities": self.entities,
            "event_type": self.event_type,
            "emotion": self.emotion,
            "source": self.source,
            "confidence_min": self.confidence_min,
            "sensitivity_max": self.sensitivity_max,
            "lifecycle": self.lifecycle,
            "owner_scope": self.owner_scope,
            "owner_id": self.owner_id,
            "tags_any": self.tags_any,
            "tags_all": self.tags_all,
            "memory_ids": self.memory_ids,
            "offset": self.offset,
        }


@dataclass(slots=True)
class MemoryRecord:
    """写入数据库的完整记忆记录。"""

    content: str
    summary: str
    memory_type: str
    owner_scope: str
    owner_id: str
    source: str = "user"
    source_ref: str = ""
    confidence: float = 0.7
    sensitivity: str = "normal"
    importance: float = 0.5
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    scene: str = ""
    task: str = ""
    event_type: str = ""
    entities: list[str] = field(default_factory=list)
    injectable: bool = True
    content_hash: str = ""
    id: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    last_verified_at: str = field(default_factory=now_iso)
    last_access_at: str | None = None
    lifecycle: str = "active"
    compressed_from: list[str] = field(default_factory=list)
    compressed_into: str = ""
    status: str = "active"
    write_epoch: int = 0
    schema_version: int = 1
    retention_score: float = 0.0
    pinned: bool = False
    deleted_at: str | None = None
    archived_at: str | None = None

    def __post_init__(self) -> None:
        if self.memory_type not in MEMORY_TYPES:
            self.memory_type = "fact"
        if self.owner_scope not in OWNER_SCOPES:
            self.owner_scope = "personal"
        if self.lifecycle not in LIFECYCLES:
            self.lifecycle = "active"
        if self.status not in STATUSES:
            self.status = "active"
        if self.sensitivity not in SENSITIVITY_RANK:
            self.sensitivity = "normal"
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.importance = max(0.0, min(1.0, float(self.importance)))
        self.emotion_valence = max(-1.0, min(1.0, float(self.emotion_valence)))
        self.emotion_arousal = max(0.0, min(1.0, float(self.emotion_arousal)))


@dataclass(slots=True)
class TagInput:
    """写入标签的输入对象。"""

    namespace: str
    tag: str
    weight: float = 1.0


@dataclass(slots=True)
class AccessStats:
    """单条记忆的访问统计。"""

    counts: dict[str, int] = field(default_factory=dict)
    session_counts: dict[str, int] = field(default_factory=dict)
    recent_counts: dict[str, int] = field(default_factory=dict)
    last_access_at: str | None = None

    def count(self, event_type: str) -> int:
        return int(self.counts.get(event_type, 0))

    def recent(self, event_type: str) -> int:
        return int(self.recent_counts.get(event_type, 0))

    def sessions(self, event_type: str) -> int:
        return int(self.session_counts.get(event_type, 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "recent_30d": dict(self.recent_counts),
            "unique_sessions": dict(self.session_counts),
            "last_access_at": self.last_access_at,
        }


def row_to_dict(row: Any) -> dict[str, Any]:
    """sqlite3.Row -> dict。"""
    if row is None:
        return {}
    return {key: row[key] for key in row.keys()}


def decode_memory_row(row: Any) -> dict[str, Any]:
    """把数据库行转为带解析字段的 dict。"""
    data = row_to_dict(row)
    if not data:
        return {}
    for key in ("compressed_from", "entities"):
        raw = data.get(key)
        if isinstance(raw, str):
            try:
                import json

                parsed = json.loads(raw)
                data[key] = parsed if isinstance(parsed, list) else []
            except (ValueError, TypeError):
                data[key] = []
        elif raw is None:
            data[key] = []
    data["injectable"] = bool(data.get("injectable", 1))
    data["pinned"] = bool(data.get("pinned", 0))
    return data


def unique_strings(values: Iterable[Any], limit: int = 64) -> list[str]:
    """保序去重并限制数量。"""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= limit:
            break
    return result
