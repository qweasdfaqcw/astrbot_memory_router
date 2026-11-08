"""测试环境的最小 AstrBot 日志桩。

插件源码必须使用 ``from astrbot.api import logger``，因此直接运行单元测试时
需要先注入一个最小的 ``astrbot.api`` 模块。本文件只用于测试，不会随插件运行。
"""

from __future__ import annotations

import sys
import types
from typing import Any


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


class NullLogger:
    """无副作用日志器，避免测试依赖 AstrBot 运行时。"""

    debug = _noop
    info = _noop
    warning = _noop
    error = _noop
    exception = _noop
    critical = _noop


def install_logger_stub() -> None:
    """确保 ``astrbot.api.logger`` 存在；已有真实模块时不覆盖。"""
    astrbot = sys.modules.get("astrbot")
    if astrbot is None:
        astrbot = types.ModuleType("astrbot")
        astrbot.__path__ = []  # 标记为包
        sys.modules["astrbot"] = astrbot

    api = sys.modules.get("astrbot.api")
    if api is None:
        api = types.ModuleType("astrbot.api")
        sys.modules["astrbot.api"] = api
        setattr(astrbot, "api", api)

    if not hasattr(api, "logger"):
        setattr(api, "logger", NullLogger())
