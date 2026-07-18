"""实体关系抽取模块 - LLM 驱动的 NER/RE"""
import json
from typing import List, Dict, Any, Optional
import logging; logger = logging.getLogger(__name__)

from .models import (
    ExtractedEntity, ExtractedRelation, ExtractionResult, DocType,
    ENTITY_TYPES, RELATION_TYPES, new_id,
)
from .llm_client import LLMClient


# 系统提示词（通用版）
_SYSTEM_PROMPT = """你是一个金融文档结构化抽取助手。你的任务是从给定的文本片段中提取实体和关系。
文本片段已附带定位信息（location）。你必须严格输出JSON，不包含任何解释。

实体类型枚举：
Organization, Person, MonetaryAmount, Percentage, Date, FinancialMetric, LegalDocument, Clause, InsuranceProduct, InsuranceCoverage, Exclusion, Industry, Rating, LegalTerm, Report

关系类型枚举：
has_director (Organization->Person)
has_shareholder (Organization->Organization|Person)
reports_financial (Organization->FinancialMetric, 属性含value_entity_id指向MonetaryAmount)
has_credit_rating (Organization->Rating)
references_law (Clause->LegalDocument)
defines_term (Clause->LegalTerm)
covers_risk (InsuranceProduct->InsuranceCoverage)
excludes_risk (InsuranceProduct->Exclusion)
analyzes_industry (Report->Industry)
gives_opinion (Person->Organization|Industry)

抽取规则：
1. 实体必须有 entity_id(临时), entity_type, mention, canonical_name, attributes(字典)。
2. 关系必须有 relation_id, relation_type, head_entity_id, tail_entity_id, properties(字典), evidence(直接复制输入中的location对象)。
3. 如果多个关系涉及同一实体，复用entity_id。
4. 无实体或关系时输出空的entities和relations数组。
5. 对于代词，mention应使用所指代的实体名称，并在attributes.original_mention中记录原代词。

输出JSON结构：
{"entities": [...], "relations": [...]}"""

# 文档类型增强指令
_TYPE_ENHANCEMENTS: Dict[DocType, str] = {
    DocType.ANNUAL_REPORT: """
强化：FinancialMetric必须是标准化指标名（如"营业收入""净利润"），attributes包含period。
MonetaryAmount的attributes必须包含currency和标准数值。reports_financial关系必须连接Organization和FinancialMetric，并通过value_entity_id指向对应的MonetaryAmount实体。""",
    DocType.PROSPECTUS: """
强化：FinancialMetric必须是标准化指标名（如"营业收入""净利润"），attributes包含period。
MonetaryAmount的attributes必须包含currency和标准数值。reports_financial关系必须连接Organization和FinancialMetric，并通过value_entity_id指向对应的MonetaryAmount实体。""",
    DocType.INSURANCE_CONTRACT: """
强化：实体InsuranceProduct（产品全称），InsuranceCoverage（责任名），Exclusion（除外事项）。
关系covers_risk属性需包括触发条件、给付标准；excludes_risk属性需包括原因。""",
}


class EntityExtractor:
    """实体关系抽取器"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract_from_sentences(
        self,
        sentences_data: List[Dict[str, Any]],
        doc_type: DocType = DocType.UNKNOWN,
    ) -> List[ExtractionResult]:
        """
        批量抽取：将多个句子的 location+text 作为 LLM 输入。

        参数:
            sentences_data: [{"sentence_id": "...", "text": "...", "location": {...}}, ...]
        返回:
            ExtractionResult 列表
        """
        if not sentences_data:
            return []

        # 构建系统提示词
        system_msg = _SYSTEM_PROMPT
        enhancement = _TYPE_ENHANCEMENTS.get(doc_type, "")
        if enhancement:
            system_msg += "\n" + enhancement

        # 用户输入
        user_input = json.dumps(sentences_data, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"现在请处理以下文本。\n\n{user_input}"},
        ]

        try:
            resp = self.llm.chat(messages, response_format={"type": "json_object"})
            raw = resp["content"]
            data = self.llm.parse_json_response(raw, "NER/RE")
            return self._parse_response(sentences_data, data)
        except Exception as e:
            logger.error(f"实体关系抽取失败: {e}")
            # 返回空结果
            return [
                ExtractionResult(
                    sentence_id=s["sentence_id"],
                    text=s["text"],
                    location=s["location"],
                )
                for s in sentences_data
            ]

    def _parse_response(
        self,
        sentences_data: List[Dict[str, Any]],
        data: dict,
    ) -> List[ExtractionResult]:
        """将 LLM 返回的 entities/relations 匹配回句子"""
        entities_list: List[dict] = data.get("entities", [])
        relations_list: List[dict] = data.get("relations", [])

        # 按 sentence_id 分组证据
        # 首先尝试从 evidence 中获取 sentence_id
        results: dict[str, ExtractionResult] = {}

        for s in sentences_data:
            sid = s["sentence_id"]
            results[sid] = ExtractionResult(
                sentence_id=sid,
                text=s["text"],
                location=s["location"],
            )

        # 解析实体
        entity_map: dict[str, ExtractedEntity] = {}
        for ent_data in entities_list:
            eid = ent_data.get("entity_id", new_id("ent"))
            entity = ExtractedEntity(
                entity_id=eid,
                entity_type=ent_data.get("entity_type", ""),
                mention=ent_data.get("mention", ""),
                canonical_name=ent_data.get("canonical_name", ent_data.get("mention", "")),
                attributes=ent_data.get("attributes", {}),
            )
            entity_map[eid] = entity

        # 解析关系，从 evidence 确定归属句子
        rel_to_sid: dict[str, str] = {}
        for rel_data in relations_list:
            evidence = rel_data.get("evidence", {})
            sid = evidence.get("sentence_id", "")
            # 如果 evidence 中没有 sentence_id，尝试其他方式
            if not sid:
                # 从 head/tail entity 推断
                head_id = rel_data.get("head_entity_id", "")
                tail_id = rel_data.get("tail_entity_id", "")
                # 默认放到第一个句子上
                sid = sentences_data[0]["sentence_id"] if sentences_data else ""

            relation = ExtractedRelation(
                relation_id=rel_data.get("relation_id", new_id("rel")),
                relation_type=rel_data.get("relation_type", ""),
                head_entity_id=rel_data.get("head_entity_id", ""),
                tail_entity_id=rel_data.get("tail_entity_id", ""),
                properties=rel_data.get("properties", {}),
                evidence=evidence,
            )
            if sid in results:
                results[sid].relations.append(relation)
            rel_to_sid[relation.relation_id] = sid

        # 将实体也分配到对应句子（基于关系中的引用），并回写 sentence_id
        assigned_entities: set[str] = set()
        for rel in [r for res in results.values() for r in res.relations]:
            if rel.head_entity_id in entity_map:
                eid = rel.head_entity_id
                if eid not in assigned_entities:
                    sid = rel_to_sid.get(rel.relation_id, "")
                    if sid in results:
                        entity_map[eid].sentence_id = sid
                        results[sid].entities.append(entity_map[eid])
                        assigned_entities.add(eid)
            if rel.tail_entity_id in entity_map:
                eid = rel.tail_entity_id
                if eid not in assigned_entities:
                    sid = rel_to_sid.get(rel.relation_id, "")
                    if sid in results:
                        entity_map[eid].sentence_id = sid
                        results[sid].entities.append(entity_map[eid])
                        assigned_entities.add(eid)

        # 未分配到任何关系的实体，放到第一个句子
        for eid, entity in entity_map.items():
            if eid not in assigned_entities:
                first_sid = sentences_data[0]["sentence_id"] if sentences_data else ""
                if first_sid in results:
                    entity.sentence_id = first_sid
                    results[first_sid].entities.append(entity)

        return list(results.values())
