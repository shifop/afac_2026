import json
from typing import List, Dict, Any, Optional
import openai  # 示例，实际使用时可按需替换

def extract_entities_relations_with_llm(content: str) -> tuple[List[Dict], List[Dict]]:
    """
    调用大模型抽取 content 中的实体和关系。
    返回 (entities, relations) 两个列表。
    此处为占位实现，实际应替换为真正的 LLM 调用。
    """
    # TODO: 替换为真实的大模型调用逻辑，例如：
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": f"从以下文本中抽取实体和关系：\n{content}"}]
    # )
    # 解析 response 得到 entities 和 relations 列表
    
    # 占位示例
    entities = []
    relations = []
    if "重大疾病" in content:
        entities.append({"name": "重大疾病", "desc": "指合同约定的重大疾病"})
    if "身故" in content:
        entities.append({"name": "身故", "desc": "指被保险人死亡"})
    if len(entities) >= 2:
        relations.append({
            "subject": entities[0]["name"],
            "predicate": "定义",
            "object": entities[1]["name"],
            "desc": "保障内容"
        })
    return entities, relations


def process_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    遍历 chunk 列表，对 entities 或 relations 为空的 chunk，
    调用大模型抽取实体和关系，并将结果写回原字典。
    
    Args:
        chunks: 每个元素包含 content、entities、relations 等字段的列表。
        
    Returns:
        更新后的 chunks 列表（原地修改并返回）。
    """
    for chunk in chunks:
        entities = chunk.get("entities")
        relations = chunk.get("relations")
        # 判断是否为空：空列表、None 或缺失均视为空
        if not entities or not relations:
            content = chunk.get("content", "")
            if not content:
                # 无内容时无法抽取，跳过
                continue
            new_entities, new_relations = extract_entities_relations_with_llm(content)
            chunk["entities"] = new_entities
            chunk["relations"] = new_relations
    return chunks