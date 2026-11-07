"""提示词加载。

提示词放在独立文件中，改提示词不需要改 Python。使用简单的 ``{{name}}``
占位符，避免 JSON 花括号与 ``str.format`` 冲突。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """读取提示词；文件缺失时返回空串并由调用方降级。"""
    path = _PROMPT_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def render_prompt(name: str, **values: object) -> str:
    """替换 ``{{key}}`` 占位符。"""
    template = load_prompt(name)
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template
