"""插件入口加载测试（用最小 AstrBot stub，不依赖真实安装）。"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import sys
import types
from pathlib import Path


def _install_astrbot_stubs(tmp_path: Path) -> None:
    if "astrbot" in sys.modules:
        return
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("test.astrbot")
    event = types.ModuleType("astrbot.api.event")
    star = types.ModuleType("astrbot.api.star")

    class AstrMessageEvent:  # noqa: D401
        pass

    class EventMessageType:
        ALL = "all"
        PRIVATE_MESSAGE = "private"
        GROUP_MESSAGE = "group"

    class _Filter:
        @staticmethod
        def command(*args, **kwargs):
            def deco(func):
                func.__command__ = args[0] if args else kwargs.get("name")
                return func
            return deco

        @staticmethod
        def event_message_type(*args, **kwargs):
            def deco(func):
                func.__event_type__ = args[0] if args else None
                return func
            return deco

        @staticmethod
        def on_llm_request(*args, **kwargs):
            def deco(func):
                func.__llm_request_hook__ = True
                return func
            return deco

        @staticmethod
        def llm_tool(name=None, **kwargs):
            def deco(func):
                func.__llm_tool_name__ = name or func.__name__
                return func
            return deco

    _Filter.EventMessageType = EventMessageType
    event.filter = _Filter()
    event.AstrMessageEvent = AstrMessageEvent
    event.MessageEventResult = type("MessageEventResult", (), {})

    class Context:
        pass

    class Star:
        def __init__(self, context):
            self.context = context
            self.name = ""

    star.Context = Context
    star.Star = Star
    astrbot.api = api
    astrbot.api.event = event
    astrbot.api.star = star
    astrbot.api.logger = api.logger

    core = types.ModuleType("astrbot.core")
    utils = types.ModuleType("astrbot.core.utils")
    astrbot_path = types.ModuleType("astrbot.core.utils.astrbot_path")

    def get_astrbot_data_path():
        return str(tmp_path / "data")

    astrbot_path.get_astrbot_data_path = get_astrbot_data_path
    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.star": star,
            "astrbot.core": core,
            "astrbot.core.utils": utils,
            "astrbot.core.utils.astrbot_path": astrbot_path,
        }
    )


def test_plugin_class_and_tool_docs_load(tmp_path: Path):
    _install_astrbot_stubs(tmp_path)
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    plugin_cls = main.MemoryRouterPlugin
    assert hasattr(plugin_cls, "on_message_capture")
    assert hasattr(plugin_cls, "on_llm_request")
    assert hasattr(plugin_cls, "memory_command")
    assert hasattr(plugin_cls, "memory_access_tool")
    assert hasattr(plugin_cls, "memory_probe_tool")
    doc = plugin_cls.memory_access_tool.__doc__ or ""
    assert "Args:" in doc
    for parameter in ("caller", "mode", "query", "filters", "budget", "options"):
        assert f"{parameter}(" in doc
    assert plugin_cls.memory_access_tool.__llm_tool_name__ == "memory_access"
    assert plugin_cls.memory_probe_tool.__llm_tool_name__ == "memory_probe"
    # AstrBot 当前工具执行器优先处理协程/返回 str 或 CallToolResult；
    # 工具不能是 async generator，否则 yield 的 MessageEventResult 会被当成直接发消息。
    assert inspect.iscoroutinefunction(plugin_cls.memory_access_tool)
    assert inspect.iscoroutinefunction(plugin_cls.memory_probe_tool)
    assert not inspect.isasyncgenfunction(plugin_cls.memory_access_tool)
    assert not inspect.isasyncgenfunction(plugin_cls.memory_probe_tool)
    assert inspect.iscoroutinefunction(plugin_cls.on_llm_request)
    assert inspect.iscoroutinefunction(plugin_cls.on_message_capture)

    plugin = plugin_cls(context=object(), config={})
    assert not plugin._fatal
    assert plugin.store is not None

    class FakeEvent:
        unified_msg_origin = "test:private:u1"
        message_str = "我上次说的咖啡偏好"
        session_id = "s1"

        class MessageObj:
            sender = types.SimpleNamespace(user_id="u1", nickname="tester")
            session_id = "s1"
            group_id = ""
            message_str = "我上次说的咖啡偏好"

        message_obj = MessageObj()

    result = asyncio.run(
        plugin.memory_access_tool(
            FakeEvent(), mode="search", query="咖啡", filters={}, budget={}, options={}
        )
    )
    # mcp 未安装时回退为 str；安装时返回 CallToolResult。
    assert isinstance(result, str) or hasattr(result, "content")
    asyncio.run(plugin.terminate())
