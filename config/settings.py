"""插件配置的强类型视图。

AstrBot 通过 ``_conf_schema.json`` 生成字典样式配置，本模块负责把不可信的
输入（WebUI、旧配置、手工 JSON）归一为稳定的 ``MemorySettings``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "是"}:
            return True
        if lowered in {"0", "false", "no", "off", "否"}:
            return False
    return default


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, result))


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, result))


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value).strip()


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.replace("，", ",").split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


@dataclass(slots=True)
class MemorySettings:
    """插件运行配置。所有字段都有安全默认值。"""

    enabled: bool = True
    data_dir: str = ""
    db_filename: str = "memory_router.sqlite3"

    extract_enabled: bool = True
    explicit_remember_enabled: bool = True
    extraction_every_n_messages: int = 6
    extraction_min_chars: int = 60
    extraction_debounce_seconds: float = 1.5
    extraction_timeout_seconds: float = 25.0
    extraction_max_items: int = 5
    buffer_max_turns: int = 20
    buffer_ttl_seconds: int = 3600
    small_model_provider_id: str = ""

    retrieval_enabled: bool = True
    injection_enabled: bool = True
    prefetch_max_latency_ms: int = 800
    candidate_limit: int = 30
    gate_threshold: float = 0.42
    top_k: int = 5
    max_injected_tokens: int = 700
    query_rewrite_with_llm: bool = False
    llm_rerank_enabled: bool = False
    rerank_candidate_limit: int = 12
    vector_backend: str = "none"
    embedding_dim: int = 384
    vector_weight: float = 0.18

    allow_personal_in_group: bool = False
    inject_third_party: bool = False

    auto_maintenance: bool = True
    maintenance_interval_seconds: int = 3600
    auto_archive: bool = True
    archive_after_days: int = 180
    archive_retention_threshold: float = 0.12
    auto_compress: bool = False
    compress_after_days: int = 90
    compress_min_group: int = 5

    rate_limit_per_minute: int = 20
    rate_limit_tokens_per_hour: int = 12000
    admin_user_ids: list[str] = field(default_factory=list)
    hard_delete_enabled: bool = False

    def __post_init__(self) -> None:
        # 数据库文件名只允许文件名，避免配置中的路径穿越。
        self.db_filename = Path(self.db_filename).name or "memory_router.sqlite3"
        if self.vector_backend not in {"none", "hash"}:
            self.vector_backend = "none"
        if self.extraction_every_n_messages < 1:
            self.extraction_every_n_messages = 1

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "MemorySettings":
        raw: Mapping[str, Any] = config or {}
        # AstrBot 配置有时会包一层 ``config``；兼容但只读第一层。
        if not raw and isinstance(config, Mapping):
            raw = config
        defaults = cls()
        return cls(
            enabled=_as_bool(raw.get("enabled"), defaults.enabled),
            data_dir=_as_str(raw.get("data_dir"), defaults.data_dir),
            db_filename=_as_str(raw.get("db_filename"), defaults.db_filename) or defaults.db_filename,
            extract_enabled=_as_bool(raw.get("extract_enabled"), defaults.extract_enabled),
            explicit_remember_enabled=_as_bool(
                raw.get("explicit_remember_enabled"), defaults.explicit_remember_enabled
            ),
            extraction_every_n_messages=_as_int(
                raw.get("extraction_every_n_messages"),
                defaults.extraction_every_n_messages,
                1,
                50,
            ),
            extraction_min_chars=_as_int(
                raw.get("extraction_min_chars"), defaults.extraction_min_chars, 10, 5000
            ),
            extraction_debounce_seconds=_as_float(
                raw.get("extraction_debounce_seconds"),
                defaults.extraction_debounce_seconds,
                0.0,
                30.0,
            ),
            extraction_timeout_seconds=_as_float(
                raw.get("extraction_timeout_seconds"),
                defaults.extraction_timeout_seconds,
                3.0,
                120.0,
            ),
            extraction_max_items=_as_int(
                raw.get("extraction_max_items"), defaults.extraction_max_items, 1, 20
            ),
            buffer_max_turns=_as_int(
                raw.get("buffer_max_turns"), defaults.buffer_max_turns, 4, 100
            ),
            buffer_ttl_seconds=_as_int(
                raw.get("buffer_ttl_seconds"), defaults.buffer_ttl_seconds, 60, 86400
            ),
            small_model_provider_id=_as_str(
                raw.get("small_model_provider_id"), defaults.small_model_provider_id
            ),
            retrieval_enabled=_as_bool(raw.get("retrieval_enabled"), defaults.retrieval_enabled),
            injection_enabled=_as_bool(raw.get("injection_enabled"), defaults.injection_enabled),
            prefetch_max_latency_ms=_as_int(
                raw.get("prefetch_max_latency_ms"),
                defaults.prefetch_max_latency_ms,
                50,
                5000,
            ),
            candidate_limit=_as_int(
                raw.get("candidate_limit"), defaults.candidate_limit, 5, 200
            ),
            gate_threshold=_as_float(
                raw.get("gate_threshold"), defaults.gate_threshold, 0.0, 1.0
            ),
            top_k=_as_int(raw.get("top_k"), defaults.top_k, 1, 20),
            max_injected_tokens=_as_int(
                raw.get("max_injected_tokens"), defaults.max_injected_tokens, 100, 8000
            ),
            query_rewrite_with_llm=_as_bool(
                raw.get("query_rewrite_with_llm"), defaults.query_rewrite_with_llm
            ),
            llm_rerank_enabled=_as_bool(
                raw.get("llm_rerank_enabled"), defaults.llm_rerank_enabled
            ),
            rerank_candidate_limit=_as_int(
                raw.get("rerank_candidate_limit"), defaults.rerank_candidate_limit, 3, 30
            ),
            vector_backend=_as_str(raw.get("vector_backend"), defaults.vector_backend),
            embedding_dim=_as_int(
                raw.get("embedding_dim"), defaults.embedding_dim, 64, 4096
            ),
            vector_weight=_as_float(
                raw.get("vector_weight"), defaults.vector_weight, 0.0, 0.6
            ),
            allow_personal_in_group=_as_bool(
                raw.get("allow_personal_in_group"), defaults.allow_personal_in_group
            ),
            inject_third_party=_as_bool(
                raw.get("inject_third_party"), defaults.inject_third_party
            ),
            auto_maintenance=_as_bool(
                raw.get("auto_maintenance"), defaults.auto_maintenance
            ),
            maintenance_interval_seconds=_as_int(
                raw.get("maintenance_interval_seconds"),
                defaults.maintenance_interval_seconds,
                60,
                86400,
            ),
            auto_archive=_as_bool(raw.get("auto_archive"), defaults.auto_archive),
            archive_after_days=_as_int(
                raw.get("archive_after_days"), defaults.archive_after_days, 30, 3650
            ),
            archive_retention_threshold=_as_float(
                raw.get("archive_retention_threshold"),
                defaults.archive_retention_threshold,
                0.0,
                0.8,
            ),
            auto_compress=_as_bool(raw.get("auto_compress"), defaults.auto_compress),
            compress_after_days=_as_int(
                raw.get("compress_after_days"), defaults.compress_after_days, 7, 3650
            ),
            compress_min_group=_as_int(
                raw.get("compress_min_group"), defaults.compress_min_group, 2, 50
            ),
            rate_limit_per_minute=_as_int(
                raw.get("rate_limit_per_minute"), defaults.rate_limit_per_minute, 1, 500
            ),
            rate_limit_tokens_per_hour=_as_int(
                raw.get("rate_limit_tokens_per_hour"),
                defaults.rate_limit_tokens_per_hour,
                100,
                200000,
            ),
            admin_user_ids=_as_str_list(raw.get("admin_user_ids")),
            hard_delete_enabled=_as_bool(
                raw.get("hard_delete_enabled"), defaults.hard_delete_enabled
            ),
        )

    def to_public_dict(self) -> dict[str, Any]:
        """导出不含敏感信息的公开配置，便于审计与调试。"""
        data = {
            "enabled": self.enabled,
            "db_filename": self.db_filename,
            "extract_enabled": self.extract_enabled,
            "retrieval_enabled": self.retrieval_enabled,
            "injection_enabled": self.injection_enabled,
            "gate_threshold": self.gate_threshold,
            "top_k": self.top_k,
            "max_injected_tokens": self.max_injected_tokens,
            "vector_backend": self.vector_backend,
            "auto_maintenance": self.auto_maintenance,
            "auto_archive": self.auto_archive,
            "auto_compress": self.auto_compress,
        }
        return data
