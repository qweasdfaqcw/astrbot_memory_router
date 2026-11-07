"""AstrBot 适配层（事件、配置路径、请求注入、历史读取）。

AstrBot 的 API 在不同版本间会有细微差异。本文件把可能有差异的调用集中
在这里，全部使用 getattr + try/except 做防御；上层业务不直接猜 API。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from store.models import AccessContext
from utils.json_utils import dump
from utils.time_utils import now_iso


def get_event_text(event: Any) -> str:
    """尽量稳定地取出纯文本消息。"""
    text = getattr(event, "message_str", None)
    if text is None:
        message_obj = getattr(event, "message_obj", None)
        text = getattr(message_obj, "message_str", "") if message_obj else ""
    return str(text or "").strip()


def get_event_message_obj(event: Any) -> Any:
    return getattr(event, "message_obj", None)


def get_sender_id(event: Any) -> str:
    getter = getattr(event, "get_sender_id", None)
    if callable(getter):
        try:
            value = getter()
            if value:
                return str(value)
        except Exception:  # noqa: BLE001
            pass
    message_obj = get_event_message_obj(event)
    sender = getattr(message_obj, "sender", None) if message_obj else None
    for attr in ("user_id", "id", "qq"):
        value = getattr(sender, attr, None) if sender else None
        if value:
            return str(value)
    return ""


def get_sender_name(event: Any) -> str:
    getter = getattr(event, "get_sender_name", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except Exception:  # noqa: BLE001
            pass
    message_obj = get_event_message_obj(event)
    sender = getattr(message_obj, "sender", None) if message_obj else None
    for attr in ("nickname", "name", "card"):
        value = getattr(sender, attr, None) if sender else None
        if value:
            return str(value)
    return ""


def get_group_id(event: Any) -> str:
    getter = getattr(event, "get_group_id", None)
    if callable(getter):
        try:
            value = getter()
            if value:
                return str(value)
        except Exception:  # noqa: BLE001
            pass
    message_obj = get_event_message_obj(event)
    value = getattr(message_obj, "group_id", "") if message_obj else ""
    return str(value or "")


def get_session_id(event: Any) -> str:
    message_obj = get_event_message_obj(event)
    for source in (event, message_obj):
        for attr in ("session_id", "unified_msg_origin"):
            value = getattr(source, attr, None) if source else None
            if value:
                return str(value)
    return ""


def get_umo(event: Any) -> str:
    return str(getattr(event, "unified_msg_origin", "") or get_session_id(event))


def is_admin(event: Any, configured_admin_ids: list[str] | None = None) -> bool:
    """兼容 AstrBot 可能存在的管理员标记；配置名单最可靠。"""
    configured = {str(item) for item in (configured_admin_ids or [])}
    sender_id = get_sender_id(event)
    if sender_id and sender_id in configured:
        return True
    for attr in ("is_admin", "is_admin_user"):
        value = getattr(event, attr, None)
        if callable(value):
            try:
                if value():
                    return True
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(value, bool) and value:
            return True
    role = getattr(event, "role", "")
    return str(role).lower() in {"admin", "administrator", "owner"}


def build_access_context(
    event: Any,
    *,
    admin_ids: list[str] | None = None,
    allow_personal_in_group: bool = False,
) -> AccessContext:
    """从 AstrBot 事件构造严格隔离的访问上下文。"""
    user_id = get_sender_id(event) or "anonymous"
    group_id = get_group_id(event)
    session_id = get_session_id(event)
    platform = ""
    for attr in ("get_platform_name", "platform_name"):
        value = getattr(event, attr, None)
        if callable(value):
            try:
                platform = str(value() or "")
                break
            except Exception:  # noqa: BLE001
                pass
        elif value:
            platform = str(value)
            break
    return AccessContext(
        user_id=user_id,
        session_id=session_id,
        group_id=group_id,
        umo=get_umo(event),
        sender_name=get_sender_name(event),
        is_group=bool(group_id),
        is_admin=is_admin(event, admin_ids),
        channel=platform,
        allow_personal_in_group=allow_personal_in_group,
    )


def is_command_text(text: str) -> bool:
    return str(text or "").strip().startswith(("/", "／"))


def resolve_data_directory(configured: str = "") -> Path:
    """解析 AstrBot data 目录，禁止回退到插件自身目录。"""
    if configured:
        path = Path(os.path.expanduser(configured)).resolve()
        plugin_root = Path(__file__).resolve().parents[1]
        if path == plugin_root or plugin_root in path.parents:
            raise ValueError("data_dir 不得指向插件自身目录，更新/重装会丢数据")
        path.mkdir(parents=True, exist_ok=True)
        return path
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path

        base = Path(get_astrbot_data_path())
    except Exception:  # noqa: BLE001 - 适配不同 AstrBot 版本
        env_path = os.environ.get("ASTRBOT_DATA_PATH", "").strip()
        if not env_path:
            raise RuntimeError(
                "无法定位 AstrBot data 目录。请升级 AstrBot，或在插件配置中显式设置 data_dir。"
            )
        base = Path(env_path)
    path = (base / "plugin_data" / "astrbot_memory_router").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


async def get_current_provider_id(context: Any, event: Any) -> str:
    """获取当前会话聊天模型 ID；失败返回空串。"""
    umo = get_umo(event)
    getter = getattr(context, "get_current_chat_provider_id", None)
    if not callable(getter):
        return ""
    try:
        if umo:
            value = await getter(umo=umo)
        else:
            value = await getter()
        return str(value or "")
    except Exception:  # noqa: BLE001
        return ""


async def get_recent_history(
    context: Any, event: Any, *, limit: int = 8
) -> list[dict[str, str]]:
    """可选读取 AstrBot 会话历史；不存在时返回空列表，由本地缓冲兜底。

    不同 AstrBot 版本 conversation_manager 的方法名不同，所以只做防御式尝试，
    不把某个具体 API 写死到业务层。
    """
    manager = getattr(context, "conversation_manager", None)
    if manager is None:
        return []
    umo = get_umo(event)
    conversation = None
    for method_name in (
        "get_conversation",
        "get_conversation_by_umo",
        "get_or_create_conversation",
    ):
        method = getattr(manager, method_name, None)
        if not callable(method):
            continue
        try:
            conversation = await method(umo=umo) if umo else await method()
        except TypeError:
            try:
                conversation = await method(umo)
            except Exception:  # noqa: BLE001
                conversation = None
        except Exception:  # noqa: BLE001
            conversation = None
        if conversation is not None:
            break
    if conversation is None:
        return []
    history = getattr(conversation, "history", None) or getattr(conversation, "messages", None)
    if not isinstance(history, list):
        return []
    result: list[dict[str, str]] = []
    for item in history[-max(1, limit):]:
        if isinstance(item, dict):
            role = str(item.get("role") or "")
            content = item.get("content") or item.get("text") or ""
        else:
            role = str(getattr(item, "role", "") or "")
            content = getattr(item, "content", "") or getattr(item, "text", "")
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        if content:
            result.append({"role": role, "content": str(content)[:2000]})
    return result


def append_dynamic_context(request: Any, text: str) -> str:
    """把本轮动态记忆追加到用户消息内容块，不污染系统提示缓存。

    返回注入位置：``extra_user_content_parts`` / ``system_prompt`` / ``none``。
    """
    if not text:
        return "none"
    parts = getattr(request, "extra_user_content_parts", None)
    if parts is not None:
        try:
            from astrbot.core.agent.message import TextPart

            parts.append(TextPart(text=text))
            return "extra_user_content_parts"
        except Exception:  # noqa: BLE001 - 尝试不同版本的简单对象
            try:
                from types import SimpleNamespace

                parts.append(SimpleNamespace(text=text))
                return "extra_user_content_parts"
            except Exception:  # noqa: BLE001
                pass
    system_prompt = getattr(request, "system_prompt", None)
    if isinstance(system_prompt, str):
        # 仅作为最后兜底；README 说明这会削弱提示词缓存。
        request.system_prompt = system_prompt + "\n\n" + text
        return "system_prompt"
    return "none"


def format_trace_meta(**kwargs: Any) -> str:
    """trace 元数据统一脱敏为 JSON 字符串。"""
    return dump(kwargs)


def utc_timestamp() -> str:
    return now_iso()
