"""阅读指南生成模块 - 章节摘要、content_info、intent_guide"""
import json
from typing import Dict, Any, Optional
import logging; logger = logging.getLogger(__name__)

from .llm_client import LLMClient


_GUIDE_SYSTEM_PROMPT = """你将收到一个从金融文档中切分出来的章节文本。你需要生成一份阅读指南，以JSON格式输出。

输出字段：
- summary: 不超过50字的章节核心内容概括。
- content_info: 字符串数组，每个以"您可以了解"开头，列出3-5个具体信息点。
- intent_guide: 字符串数组，每个以"如果您想了解"开头，列出3-5条指引。

注意：summary必须准确精炼；两个数组内容需基于原文，避免空泛。"""


class ReadingGuideGenerator:
    """阅读指南生成器"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, section_path: str, text: str) -> Optional[Dict[str, Any]]:
        """为章节生成阅读指南"""
        if not text.strip():
            return None

        # 如果 text 太长，截断（约 12K 字符）
        max_input = 12000
        if len(text) > max_input:
            text = text[:max_input] + "\n...(文本过长已截断)"

        user_input = json.dumps(
            {"section_path": section_path, "text": text},
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": _GUIDE_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        try:
            resp = self.llm.chat(messages, response_format={"type": "json_object"})
            raw = resp["content"]
            guide = self.llm.parse_json_response(raw, "阅读指南")
            # 确保字段完整
            guide.setdefault("summary", "")
            guide.setdefault("content_info", [])
            guide.setdefault("intent_guide", [])
            # 截断 summary
            if len(guide.get("summary", "")) > 80:
                guide["summary"] = guide["summary"][:80]
            return guide
        except Exception as e:
            logger.error(f"阅读指南生成失败: {e}")
            return {
                "summary": text[:50].replace("\n", " "),
                "content_info": [f"您可以了解{section_path}的相关内容"],
                "intent_guide": [f"如果您想了解{section_path}的详细信息"],
            }
