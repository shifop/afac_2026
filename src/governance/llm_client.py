"""LLM 客户端 - 统一的大模型调用接口"""
import json
import time
import asyncio
from typing import Dict, Any, Optional, List
import logging; logger = logging.getLogger(__name__)

import openai


class LLMClient:
    """LLM 调用客户端，支持重试、并发控制"""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "qwen3.7-plus",
        temperature: float = 0.0,
        timeout: int = 60,
        max_retries: int = 3,
        max_concurrency: int = 10,
    ) -> None:
        self.client = openai.OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._total_tokens = 0
        self._call_count = 0

    def chat(
        self,
        messages: list[dict],
        response_format: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """同步调用 LLM，返回 (content, usage)"""
        extra_body = {"enable_thinking": False, "thinking": {"type": "disabled"}}
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format=response_format,
                    timeout=self.timeout,
                    extra_body=extra_body,
                )
                content = response.choices[0].message.content.strip()
                usage = response.usage
                self._total_tokens += usage.total_tokens if usage else 0
                self._call_count += 1
                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                    },
                }
            except openai.RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"LLM 限流 (尝试 {attempt + 1}/{self.max_retries})，等待 {wait}s")
                time.sleep(wait)
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                wait = 2 ** attempt
                logger.warning(f"LLM 网络错误 (尝试 {attempt + 1}/{self.max_retries}): {e}，等待 {wait}s")
                time.sleep(wait)
            except openai.APIError as e:
                logger.error(f"LLM API 错误 (尝试 {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 调用失败，已重试 {self.max_retries} 次")

    async def chat_async(
        self,
        messages: list[dict],
        response_format: Optional[dict] = None,
    ) -> dict:
        """异步调用 LLM"""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.chat(messages, response_format)
            )

    def parse_json_response(self, raw: str, label: str = "LLM") -> Any:
        """从 LLM 原始响应中解析 JSON"""
        raw = raw.strip()
        # 移除 markdown 代码块
        if raw.startswith("```"):
            lines = raw.split("\n")
            end_idx = None
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            if end_idx is not None:
                raw = "\n".join(lines[1:end_idx])
            else:
                raw = "\n".join(lines[1:])
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取 JSON 对象/数组
            import re
            match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", raw)
            if match:
                return json.loads(match.group())
            raise ValueError(f"{label} 响应无法解析为 JSON: {raw[:200]}")

    @property
    def stats(self) -> dict:
        return {
            "total_tokens": self._total_tokens,
            "call_count": self._call_count,
        }
