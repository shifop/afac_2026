"""阅读指南生成模块 - 为查询阶段提供紧凑的文档导航地图

设计目标：
  查询时 LLM 先扫阅读指南决定加载哪些章节，因此每条指南必须极短（≤80 chars），
  用关键词而非完整句，让 LLM 用最少 token 完成"问题→章节"的路由决策。

输出格式：
  has:  本节包含什么（关键词逗号分隔，≤50字）— 用于关键词命中判断
  for:  适合回答哪类问题（≤5个短标签）— 用于语义匹配
  skip: 是否可跳过（封面/目录/团队/免责等非正文章节）"""

import json
from typing import Dict, Any, Optional
import logging; logger = logging.getLogger(__name__)

from .llm_client import LLMClient, _truncate


_GUIDE_SYSTEM_PROMPT = """你是一个金融文档索引助手。你会收到一个文档章节，需要生成极简的导航标签，供下游大模型快速判断"是否要加载此章节"。

输出 JSON（严格遵守字段含义）：
- has: 字符串。本节具体包含哪些信息？用关键词/短语列举，逗号分隔，≤80字。优先列出可验证的事实、数据指标、实体名称、条款类型。要具体不要概括。
  例如："营业收入/净利润/资产负债率/现金流/分红方案/每股收益"
  而不是："财务数据"
  对于数据密集章节，宁可多列几个关键词也不要遗漏重要主题。
- for: 字符串数组，≤5个。本节适合回答什么类型的问题？用简短标签（2-4字），从提问者角度命名。
  例如：["财务指标","盈利能力","负债水平","分红政策"]
  而不是：["您可以了解公司的财务表现"]
  标签应覆盖本节的主要信息维度，确保与可能的问题术语对齐（如同时包含"ROE"和"净资产收益率"）。
- skip: 布尔值。本节是否为非正文内容？以下类型直接标 true：封面/标题页、目录/TOC、团队介绍/作者简介/个人履历、免责声明/分析师声明/版权声明、附录/参考文献/数据来源。其余标 false。

核心原则：标签要具体可检索，帮助下游 LLM 用最少 token 完成"问题→章节"的路由决策。"""


class ReadingGuideGenerator:
    """阅读指南生成器 — 输出紧凑导航标签供查询阶段使用"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, section_path: str, text: str) -> Optional[Dict[str, Any]]:
        """为章节生成阅读指南"""
        if not text.strip():
            return None

        # 只需要前 6000 字符即可判断章节内容（比之前的 12000 更省输入 token）
        max_input = 6000
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
            resp = self.llm.chat(messages, response_format={"type": "json_object"}, label="指南")
            raw = resp["content"]
            guide = self.llm.parse_json_response(raw, "阅读指南")

            # 规范化字段
            has = guide.get("has", "")[:80]
            for_tags = guide.get("for", [])
            if isinstance(for_tags, str):
                for_tags = [t.strip() for t in for_tags.split(",") if t.strip()]
            for_tags = [t[:20] for t in for_tags[:5]]
            skip = bool(guide.get("skip", False))

            result = {"has": has, "for": for_tags, "skip": skip}

            logger.info(
                f"[指南] \"{section_path[:50]}\" | "
                f"has: {has[:60]} | "
                f"for: {for_tags} | "
                f"skip: {skip}"
            )
            return result
        except Exception as e:
            logger.error(f"阅读指南生成失败: {e}")
            return {
                "has": text[:80].replace("\n", " "),
                "for": [],
                "skip": False,
            }
