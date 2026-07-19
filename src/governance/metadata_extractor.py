"""文档元数据抽取模块 - LLM 驱动的结构化元数据提取"""
import json
from typing import Dict, Any
import logging; logger = logging.getLogger(__name__)

from .models import DocType
from .llm_client import LLMClient

# 各文档类型的元数据提取提示词
_METADATA_PROMPTS: Dict[DocType, dict] = {
    DocType.INSURANCE_CONTRACT: {
        "fields": {
            "contract_name": "保险产品名称",
            "insurer": "保险公司名称",
        },
        "hint": "合同名称通常为「XXX保险条款」或「XXX保险」格式，保险公司名称通常包含「保险股份有限公司」等字样。",
    },
    DocType.ANNUAL_REPORT: {
        "fields": {
            "stock_code": "股票代码（如 601668.SH）",
            "company_name": "公司全称",
            "fiscal_year_end": "财年截止日期（如 2024-12-31）",
            "report_year": "报告年份（整数，如 2024）",
        },
        "hint": "股票代码和公司名称通常在报告封面，财年截止日通常表述为「截至XXXX年XX月XX日」。",
    },
    DocType.PROSPECTUS: {
        "fields": {
            "doc_type_name": "文档类型（如「公司债募集说明书」、「购买资产报告书」、「招股说明书」等）",
            "issuer_name": "发行公司/发行人全称",
        },
        "hint": "文档类型通常在封面或标题中明确标注，发行人名称通常在「发行人」或「发行人名称」后。",
    },
    DocType.FINANCIAL_REGULATION: {
        "fields": {
            "title": "法规名称",
        },
        "hint": "法规名称通常是文档的主标题。",
    },
    DocType.RESEARCH_REPORT: {
        "fields": {
            "title": "报告标题",
            "institution": "撰写机构名称（证券公司/研究所名称）",
        },
        "hint": "报告标题通常在首页，机构名称通常包含「证券」、「研究所」等字样。",
    },
}

_SYSTEM_PROMPT = """你是一个金融文档元数据抽取助手。根据给定的文档类型，从文档内容中提取结构化元数据字段。

规则：
1. 严格输出 JSON，不包含任何解释。
2. 如果某个字段无法从文档中提取，将其值设为 null。
3. 输出格式：{"field_name": "value", ...}"""


class MetadataExtractor:
    """文档元数据抽取器"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(self, content: str, doc_type: DocType, title: str = "") -> Dict[str, Any]:
        """
        从文档内容中提取结构化元数据。

        参数:
            content: 文档全文
            doc_type: 文档类型
            title: 已知的文档标题（从 markdown 标题提取）

        返回:
            包含各字段的字典，字段名与 data/documents.all structured_data 对齐
        """
        config = _METADATA_PROMPTS.get(doc_type)
        if not config:
            logger.info(f"[元数据] 文档类型 {doc_type.value} 无元数据配置，跳过")
            return {}

        fields_desc = "\n".join(
            f"  - {name}: {desc}" for name, desc in config["fields"].items()
        )
        user_prompt = (
            f"文档类型：{doc_type.value}\n"
            f"已知标题：{title}\n\n"
            f"需要提取的字段：\n{fields_desc}\n\n"
            f"提示：{config['hint']}\n\n"
            f"文档内容（前8000字）：\n{content[:8000]}"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            resp = self.llm.chat(messages, response_format={"type": "json_object"}, label="元数据")
            raw = resp["content"]
            data = self.llm.parse_json_response(raw, "metadata")
            # 只保留配置中定义的字段
            result = {}
            for field_name in config["fields"]:
                result[field_name] = data.get(field_name)
            logger.info(f"[元数据] 提取完成: {json.dumps(result, ensure_ascii=False)[:200]}")
            return result
        except Exception as e:
            logger.error(f"[元数据] 提取失败: {e}")
            return {f: None for f in config["fields"]}
