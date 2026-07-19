"""实体关系抽取模块 - LLM 驱动的 NER/RE"""
import json
from typing import List, Dict, Any, Optional
import logging; logger = logging.getLogger(__name__)

from .models import (
    ExtractedEntity, ExtractedRelation, ExtractionResult, DocType,
    ENTITY_TYPES, RELATION_TYPES, new_id,
)
from .llm_client import LLMClient, _truncate


# 系统提示词（简化版：LLM 只负责信息抽取，不生成 ID、不复制 location）
_SYSTEM_PROMPT = """你是一个金融文档结构化抽取助手。你的任务是从给定的文本片段列表中提取实体和关系。

输入格式：每项包含 sentence_index（整数序号）和 text（句子文本）。
你需要为每个抽取出的实体和关系标注它来源于哪个句子（sentence_index）。

实体类型枚举：
Organization, Person, MonetaryAmount, Percentage, Date, FinancialMetric, LegalDocument, Clause, InsuranceProduct, InsuranceCoverage, Exclusion, Industry, Rating, LegalTerm, Report

关系类型枚举：
has_director (Organization->Person)
has_shareholder (Organization->Organization|Person)
reports_financial (Organization->FinancialMetric, 属性含value_entity_name指向MonetaryAmount的canonical_name)
has_credit_rating (Organization->Rating)
references_law (Clause->LegalDocument)
defines_term (Clause->LegalTerm)
covers_risk (InsuranceProduct->InsuranceCoverage)
excludes_risk (InsuranceProduct->Exclusion)
analyzes_industry (Report->Industry)
gives_opinion (Person->Organization|Industry)

抽取规则：
1. 实体：entity_type 必须从枚举中选择，canonical_name 是实体的标准化名称，mention 是原文中的提及文本（可选，默认同 canonical_name），attributes 为字典。

2. 关系：relation_type 必须从枚举中选择，head_entity_name 和 tail_entity_name 必须对应本批次中某个实体的 canonical_name，properties 为字典。

3. 每个实体/关系都需标注 sentence_index（整数），指明来源于输入中的哪个句子。

4. 无实体或关系时输出空的 entities 和 relations 数组。

输出JSON结构：
{"entities": [{"entity_type": "...", "canonical_name": "...", "mention": "...", "attributes": {...}, "sentence_index": 0}], "relations": [{"relation_type": "...", "head_entity_name": "...", "tail_entity_name": "...", "properties": {...}, "sentence_index": 0}]}"""

# 文档类型增强指令
_TYPE_ENHANCEMENTS: Dict[DocType, str] = {
    DocType.ANNUAL_REPORT: """
强化：FinancialMetric必须是标准化指标名（如"营业收入""净利润"），attributes包含period。
MonetaryAmount的attributes必须包含currency和标准数值。reports_financial关系必须连接Organization和FinancialMetric，并在properties中通过value_entity_name指向对应的MonetaryAmount实体（使用其canonical_name）。""",
    DocType.PROSPECTUS: """
强化：FinancialMetric必须是标准化指标名（如"营业收入""净利润"），attributes包含period。
MonetaryAmount的attributes必须包含currency和标准数值。reports_financial关系必须连接Organization和FinancialMetric，并在properties中通过value_entity_name指向对应的MonetaryAmount实体（使用其canonical_name）。""",
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
        批量抽取：将多个句子的 sentence_index + text 作为 LLM 输入。

        参数:
            sentences_data: [{"sentence_id": "...", "text": "...", "location": {...}}, ...]
        返回:
            ExtractionResult 列表
        """
        if not sentences_data:
            return []

        # 为每个批次生成唯一前缀，避免跨批次实体/关系 ID 冲突
        batch_prefix = new_id("b")

        # 构建系统提示词
        system_msg = _SYSTEM_PROMPT
        enhancement = _TYPE_ENHANCEMENTS.get(doc_type, "")
        if enhancement:
            system_msg += "\n" + enhancement

        # 构建简化输入：只用 sentence_index + text，去掉 location 等冗余信息
        simplified_input = [
            {"sentence_index": i, "text": s["text"]}
            for i, s in enumerate(sentences_data)
        ]
        user_input = json.dumps(simplified_input, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"现在请处理以下文本。\n\n{user_input}"},
        ]

        try:
            # 批次日志
            sent_count = len(sentences_data)
            total_chars = sum(len(s.get("text", "")) for s in sentences_data)
            first_text = sentences_data[0].get("text", "") if sentences_data else ""
            logger.info(
                f"[抽取] 批次 {sent_count} 句, 共 {total_chars:,} chars | "
                f"首句: {_truncate(first_text, 120)}"
            )

            resp = self.llm.chat(messages, response_format={"type": "json_object"}, label="抽取")
            raw = resp["content"]
            data = self.llm.parse_json_response(raw, "NER/RE")

            # 输出摘要
            ent_count = len(data.get("entities", []))
            rel_count = len(data.get("relations", []))
            logger.info(
                f"[抽取] 结果: {ent_count} 实体, {rel_count} 关系 | "
                f"输出截选: {_truncate(raw, 200)}"
            )

            return self._parse_response(sentences_data, data, batch_prefix)
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
        batch_prefix: str = "",
    ) -> List[ExtractionResult]:
        """
        将 LLM 返回的简化 entities/relations 转换为 ExtractionResult 列表。

        LLM 输出：
          entities: [{entity_type, canonical_name, mention?, attributes, sentence_index}]
          relations: [{relation_type, head_entity_name, tail_entity_name, properties, sentence_index}]

        代码负责：
          - 用 UUID 生成 entity_id / relation_id
          - 用 sentence_index 查找 sentence_id 和 location
          - 用 entity_name 匹配关系的 head/tail 到实体
        """
        entities_list: List[dict] = data.get("entities", [])
        relations_list: List[dict] = data.get("relations", [])
        if len(entities_list)==0 and len(relations_list)==0:
            print('')

        batch_size = len(sentences_data)

        # ---- Step 1: 生成实体 ----
        entity_map: Dict[str, ExtractedEntity] = {}          # entity_id → ExtractedEntity
        name_to_entities: Dict[str, List[str]] = {}           # canonical_name → [entity_id, ...]

        for i, ent_data in enumerate(entities_list):
            canonical_name = (ent_data.get("canonical_name") or "").strip()
            if not canonical_name:
                logger.warning(f"[抽取] 实体 {i} canonical_name 为空，跳过")
                continue

            entity_type = ent_data.get("entity_type", "")
            if entity_type not in ENTITY_TYPES:
                logger.warning(f"[抽取] 实体类型不在枚举中: {entity_type} (name={canonical_name})")
                # 仍然保留，后处理会再次校验

            mention = ent_data.get("mention") or canonical_name
            attributes = ent_data.get("attributes") or {}

            # 用 sentence_index 查找对应的 sentence_id
            sent_idx = self._safe_index(ent_data.get("sentence_index"), batch_size)
            sid = sentences_data[sent_idx]["sentence_id"]

            entity_id = f"{batch_prefix}_{new_id('ent')}"
            entity = ExtractedEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                mention=mention,
                canonical_name=canonical_name,
                attributes=attributes,
                sentence_id=sid,
                sentence_ids=[sid],
            )
            entity_map[entity_id] = entity

            # 维护 name → [entity_id] 索引
            if canonical_name not in name_to_entities:
                name_to_entities[canonical_name] = []
            name_to_entities[canonical_name].append(entity_id)

        # ---- Step 2: 初始化结果容器（每个输入句子一个） ----
        results: Dict[str, ExtractionResult] = {}
        for s in sentences_data:
            sid = s["sentence_id"]
            results[sid] = ExtractionResult(
                sentence_id=sid,
                text=s["text"],
                location=s["location"],
            )

        # ---- Step 3: 处理关系 ----
        for i, rel_data in enumerate(relations_list):
            relation_type = rel_data.get("relation_type", "")
            if relation_type not in RELATION_TYPES:
                logger.warning(f"[抽取] 关系类型不在枚举中: {relation_type}")
                # 仍然保留，后处理会再次校验

            head_name = (rel_data.get("head_entity_name") or "").strip()
            tail_name = (rel_data.get("tail_entity_name") or "").strip()
            if not head_name or not tail_name:
                logger.warning(f"[抽取] 关系 {i} head_entity_name 或 tail_entity_name 为空，跳过")
                continue

            # 用 sentence_index 确定关系和 evidence
            rel_sent_idx = self._safe_index(rel_data.get("sentence_index"), batch_size)
            rel_sid = sentences_data[rel_sent_idx]["sentence_id"]
            rel_location = sentences_data[rel_sent_idx]["location"]

            # 匹配 head entity：优先同 sentence_index，回退到任意
            head_id = self._match_entity(name_to_entities, entity_map, head_name, rel_sid)
            tail_id = self._match_entity(name_to_entities, entity_map, tail_name, rel_sid)

            if not head_id:
                logger.warning(f"[抽取] 关系 head_entity_name=\"{head_name}\" 在批次内未找到，跳过")
                continue
            if not tail_id:
                logger.warning(f"[抽取] 关系 tail_entity_name=\"{tail_name}\" 在批次内未找到，跳过")
                continue

            relation_id = f"{batch_prefix}_{new_id('rel')}"
            relation = ExtractedRelation(
                relation_id=relation_id,
                relation_type=relation_type,
                head_entity_id=head_id,
                tail_entity_id=tail_id,
                properties=rel_data.get("properties") or {},
                evidence=rel_location if isinstance(rel_location, dict) else (
                    rel_location.to_dict() if hasattr(rel_location, 'to_dict') else {}
                ),
            )

            if rel_sid in results:
                results[rel_sid].relations.append(relation)

        # ---- Step 4: 将实体添加到对应句子 ----
        for entity in entity_map.values():
            sid = entity.sentence_id
            if sid in results:
                results[sid].entities.append(entity)
            else:
                # 兜底：sentence_id 无效时放到第一个句子
                first_sid = sentences_data[0]["sentence_id"] if sentences_data else ""
                if first_sid in results:
                    results[first_sid].entities.append(entity)
                    logger.debug(f"[抽取] 实体 {entity.entity_id} sentence_id 无效，放入首句")

        return list(results.values())

    # ---- 内部辅助 ----

    @staticmethod
    def _safe_index(raw_index: Any, batch_size: int) -> int:
        """将 LLM 输出的 sentence_index 安全地 clamp 到有效范围"""
        try:
            idx = int(raw_index)
        except (ValueError, TypeError):
            return 0
        return max(0, min(idx, batch_size - 1))

    @staticmethod
    def _match_entity(
        name_to_entities: Dict[str, List[str]],
        entity_map: Dict[str, ExtractedEntity],
        entity_name: str,
        preferred_sid: str,
    ) -> Optional[str]:
        """
        按 canonical_name 查找 entity_id。
        优先匹配 sentence_id == preferred_sid 的实体，回退到任意匹配。
        """
        candidates = name_to_entities.get(entity_name, [])
        if not candidates:
            return None

        # 优先：同 sentence_id
        for eid in candidates:
            if entity_map[eid].sentence_id == preferred_sid:
                return eid

        # 回退：第一个
        return candidates[0]
