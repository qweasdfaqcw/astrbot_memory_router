"""小模型调用适配。

只使用 AstrBot 的 ``context.llm_generate``；不直接发 HTTP，不依赖 requests。
所有失败统一抛 ``LLMUnavailable``，由业务层降级到规则抽取 / 关键词召回。
"""

from __future__ import annotations

import asyncio
from typing import Any

from adapter.astrbot_adapter import get_current_provider_id
from prompts import load_prompt
from utils.json_utils import JSONParseError, as_dict, loads_lenient


class LLMUnavailable(RuntimeError):
    """小模型不可用或返回无法修复。"""


class LLMAdapter:
    def __init__(self, context: Any, model_provider_id: str = ""):
        self.context = context
        self.model_provider_id = model_provider_id.strip()

    async def _provider_for(self, event: Any) -> str:
        if self.model_provider_id:
            return self.model_provider_id
        if event is not None:
            return await get_current_provider_id(self.context, event)
        return ""

    @staticmethod
    def _response_text(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        for attr in ("completion_text", "text", "content", "result"):
            value = getattr(response, attr, None)
            if value is not None:
                return str(value)
        if isinstance(response, dict):
            for key in ("completion_text", "text", "content", "result"):
                if response.get(key) is not None:
                    return str(response[key])
        return str(response)

    async def generate_text(
        self,
        prompt: str,
        *,
        event: Any = None,
        timeout: float = 25.0,
        provider_id: str = "",
    ) -> str:
        """调用小模型并返回文本；失败抛 LLMUnavailable。"""
        provider = (provider_id or await self._provider_for(event)).strip()
        if not provider:
            raise LLMUnavailable("未配置小模型提供商，且无法获取当前会话提供商")
        if not prompt.strip():
            raise LLMUnavailable("prompt 为空")
        prompt = prompt[:30000]
        generator = getattr(self.context, "llm_generate", None)
        if not callable(generator):
            raise LLMUnavailable("当前 AstrBot 不支持 context.llm_generate")
        try:
            response = await asyncio.wait_for(
                generator(chat_provider_id=provider, prompt=prompt),
                timeout=max(1.0, timeout),
            )
        except asyncio.TimeoutError as exc:
            raise LLMUnavailable("小模型调用超时") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"小模型调用失败: {exc}") from exc
        text = self._response_text(response).strip()
        if not text:
            raise LLMUnavailable("小模型返回空文本")
        return text

    async def generate_json(
        self,
        prompt: str,
        *,
        event: Any = None,
        timeout: float = 25.0,
        retry_repair: bool = True,
    ) -> dict[str, Any]:
        """调用小模型并严格解析 JSON；失败时做一次修复重试。"""
        text = await self.generate_text(prompt, event=event, timeout=timeout)
        try:
            parsed = loads_lenient(text)
            return as_dict(parsed)
        except JSONParseError as first_error:
            if not retry_repair:
                raise LLMUnavailable(f"JSON 解析失败: {first_error}") from first_error
            repair_prompt = (
                "你上一次的输出不是合法 JSON。请只输出一个 JSON 对象，"
                "不要 Markdown、不要解释、不要尾逗号。\n\n"
                f"上一次输出：\n{text[:8000]}\n\n"
                "如果无法修复，输出 {\"memories\":[]}（抽取任务）或 "
                "{\"scores\":[]}（重排任务）或 {\"queries\":[]}（改写任务）。"
            )
            try:
                repaired = await self.generate_text(
                    repair_prompt, event=event, timeout=timeout
                )
                parsed = loads_lenient(repaired)
                return as_dict(parsed)
            except (LLMUnavailable, JSONParseError) as second_error:
                raise LLMUnavailable(
                    f"JSON 解析失败且修复失败: {second_error}"
                ) from second_error

    async def extract_json(
        self,
        prompt: str,
        *,
        event: Any = None,
        timeout: float = 25.0,
    ) -> dict[str, Any]:
        """语义化封装，便于测试替换。"""
        return await self.generate_json(prompt, event=event, timeout=timeout)

    def usage_rules(self) -> str:
        """主模型工具使用说明（写入稳定 system prompt）。"""
        return load_prompt("main_model_usage.txt").strip()
