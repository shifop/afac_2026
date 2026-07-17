# src/extraction/llm_extractor.py
import json
import logging
from typing import List, Dict, Tuple, Any
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 提示词模板
# ──────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """\
你是一个专业的保险条款信息抽取助手。你的任务是从给定的保险条款文本中抽取出结构化的实体和关系。
请严格按照以下 JSON 格式输出，不要添加任何额外的解释或文字，只输出一个 JSON 对象：
{
  "entities": [
    {
      "name": "实体名称",
      "desc": "对实体的简短解释或描述"
    }
  ],
  "relations": [
    {
      "subject": "主体实体名称（必须与 entities 中的某个 name 完全一致）",
      "predicate": "关系类型，例如：提供、定义、包含、属于、触发、不保、限制等",
      "object": "客体实体名称（必须与 entities 中的某个 name 完全一致）",
      "desc": "对该关系的一句话说明"
    }
  ]
}

抽取重点（请从以下角度识别实体和关系，但不要局限于此）：
- 保险产品名称及其提供的保障（如重大疾病保险金、身故保险金）
- 责任免除情形（即不保什么）
- 重要期限（等待期、犹豫期、宽限期等）
- 合同中的重要角色（投保人、被保险人、受益人）
- 疾病或伤残名称及其定义或分组
- 现金价值、保单贷款等权益

注意：
1. entities 列表中每个元素必须包含 "name" 和 "desc" 字段。
2. relations 列表中每个元素必须包含 "subject"、"predicate"、"object"、"desc" 字段，且 subject 和 object 必须能在 entities 中找到完全匹配的 name。
3. 只输出 JSON，不要包含 markdown 代码块标记（如 ```json），只输出纯 JSON 文本。
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """\
请从以下保险条款文本中抽取出实体和关系：

{content}
"""


def _call_llm(system_prompt: str, user_prompt: str, model_name: str = "qwen3.7-plus") -> str:
    """
    调用大语言模型，返回生成的原始文本。
    此处以 OpenAI API 为例，请根据实际使用的后端替换此函数。
    """
    # 如果你使用其他模型（如本地 Llama、ChatGLM 等），请修改本函数
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # 低温度保证输出稳定
        extra_body = {
            "enable_thinking": False,
            "thinking":{"type": "disabled"}
        }
    )
    return response.choices[0].message.content


def _parse_llm_output(text: str) -> Tuple[List[Dict], List[Dict]]:
    """
    尝试将 LLM 返回的文本解析为包含 entities 和 relations 的字典。
    支持纯 JSON 以及包裹在 markdown 代码块中的情况。
    解析失败时返回两个空列表。
    """
    import re

    # 尝试直接解析
    try:
        data = json.loads(text)
        return data.get("entities", []), data.get("relations", [])
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 中的内容
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("entities", []), data.get("relations", [])
        except json.JSONDecodeError:
            pass

    logger.warning("无法从 LLM 输出中解析出有效的 JSON，返回空结果。原始输出: %s", text[:200])
    return [], []


def extract_entities_relations(
    content: str,
    model_name: str = "qwen3.7-plus",
    max_content_length: int = 3000
) -> Tuple[List[Dict], List[Dict]]:
    """
    从保险条款文本中抽取实体和关系。

    Args:
        content: 待抽取的文本（通常是单个 chunk 的内容）。
        model_name: 使用的 LLM 模型名称，默认为 "gpt-4"。
        max_content_length: 传入模型的最大字符数，超出部分将截断（避免 token 超限）。

    Returns:
        entities: 实体列表，每个元素格式为 {"name": str, "desc": str}
        relations: 关系列表，每个元素格式为 {"subject": str, "predicate": str, "object": str, "desc": str}
    """
    if not content or not content.strip():
        return [], []

    # 截取合适长度，避免模型输入超限
    trimmed_content = content[:max_content_length]

    # 构建提示词
    user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(content=trimmed_content)

    # 调用大模型
    try:
        raw_output = _call_llm(EXTRACTION_SYSTEM_PROMPT, user_prompt, model_name)
    except Exception as e:
        logger.error("调用 LLM 失败: %s", e)
        return [], []

    # 解析输出
    entities, relations = _parse_llm_output(raw_output)

    # 可选：对抽取结果进行基本校验和清理
    # 例如确保 relation 中的 subject/object 真实存在于 entities 中，不存在则删除该 relation
    entity_names = {e["name"] for e in entities if "name" in e}
    valid_relations = []
    for rel in relations:
        if rel.get("subject") in entity_names and rel.get("object") in entity_names:
            valid_relations.append(rel)
        else:
            logger.debug("移除无效关系（实体不存在）：%s", rel)

    return entities, valid_relations

def enrich_chunks(chunks):
    for chunk in chunks:
        if not chunk.get("entities") or not chunk.get("relations"):
            content = chunk.get("content", "")
            if content:
                entities, relations = extract_entities_relations(content)
                chunk["entities"] = entities
                chunk["relations"] = relations
    return chunks