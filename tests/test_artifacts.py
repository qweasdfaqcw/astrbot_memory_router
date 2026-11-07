"""交付物静态校验。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_schema_and_example_config_are_valid_json():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    example = json.loads((ROOT / "config" / "example_config.json").read_text(encoding="utf-8"))
    supported = {
        "string", "text", "int", "float", "bool", "object", "list", "dict",
        "template_list", "file",
    }
    for key, value in schema.items():
        assert value.get("type") in supported, key
    for key in example:
        assert key in schema, f"示例配置字段缺少 schema: {key}"


def test_metadata_and_entrypoint_exist():
    metadata = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    assert "name: astrbot_memory_router" in metadata
    assert (ROOT / "main.py").is_file()
    assert "from astrbot.api.star import Context, Star" in (
        ROOT / "main.py"
    ).read_text(encoding="utf-8")
    assert "@filter.llm_tool(name=\"memory_access\")" in (
        ROOT / "main.py"
    ).read_text(encoding="utf-8")


def test_prompts_exist_and_use_placeholders():
    for name in ("extraction.txt", "routing.txt", "query_rewrite.txt", "main_model_usage.txt"):
        assert (ROOT / "prompts" / name).is_file()
    extraction = (ROOT / "prompts" / "extraction.txt").read_text(encoding="utf-8")
    assert "{{context}}" in extraction
    assert "\"memories\"" in extraction
