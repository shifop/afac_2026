"""后处理与质量校验模块 - JSON Schema验证、值域检查、置信度过滤、共指消解"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging; logger = logging.getLogger(__name__)

from .models import (
    ExtractedEntity, ExtractedRelation, ExtractionResult,
    ENTITY_TYPES, RELATION_TYPES, RELATION_CONSTRAINTS,
)


class PostProcessor:
    """后处理与质量校验器"""

    def __init__(self) -> None:
        self.rejected: list[dict] = []

    def process(
        self,
        results: List[ExtractionResult],
        confidence_threshold: float = 0.6,
    ) -> List[ExtractionResult]:
        """对抽取结果进行全量后处理"""
        for result in results:
            # 1. JSON Schema 验证
            self._validate_schema(result)
            # 2. 引用完整性检查
            self._check_reference_integrity(result)
            # 3. 值域与逻辑校验
            self._validate_value_domain(result)
            # 4. 关系类型约束检查
            self._validate_relation_constraints(result)
            # 5. 置信度过滤
            self._filter_confidence(result, confidence_threshold)
            # 6. 共指消解
            self._resolve_coreferences(result)
        return results

    def _validate_schema(self, result: ExtractionResult) -> None:
        """JSON Schema 验证：检查实体和关系必填字段"""
        valid_entities = []
        for ent in result.entities:
            if not ent.entity_id or not ent.entity_type:
                self._reject(result.sentence_id, "schema_validation",
                             f"实体缺少必填字段: {ent}")
                continue
            if ent.entity_type not in ENTITY_TYPES:
                self._reject(result.sentence_id, "schema_validation",
                             f"实体类型不在枚举中: {ent.entity_type}")
                continue
            valid_entities.append(ent)
        result.entities = valid_entities

        valid_relations = []
        for rel in result.relations:
            if not rel.relation_id or not rel.relation_type:
                self._reject(result.sentence_id, "schema_validation",
                             f"关系缺少必填字段: {rel}")
                continue
            if rel.relation_type not in RELATION_TYPES:
                self._reject(result.sentence_id, "schema_validation",
                             f"关系类型不在枚举中: {rel.relation_type}")
                continue
            valid_relations.append(rel)
        result.relations = valid_relations

    def _check_reference_integrity(self, result: ExtractionResult) -> None:
        """引用完整性：关系的 head/tail entity_id 必须在 entities 数组中存在"""
        entity_ids = {e.entity_id for e in result.entities}
        valid_relations = []
        for rel in result.relations:
            if rel.head_entity_id not in entity_ids:
                self._reject(result.sentence_id, "reference_integrity",
                             f"关系引用不存在的 head entity: {rel.head_entity_id}")
                continue
            if rel.tail_entity_id not in entity_ids:
                self._reject(result.sentence_id, "reference_integrity",
                             f"关系引用不存在的 tail entity: {rel.tail_entity_id}")
                continue
            valid_relations.append(rel)
        result.relations = valid_relations

    def _validate_value_domain(self, result: ExtractionResult) -> None:
        """值域与逻辑校验"""
        for ent in result.entities:
            attrs = ent.attributes
            if ent.entity_type == "MonetaryAmount":
                # amount 必须是数字
                if "amount" in attrs:
                    try:
                        attrs["amount"] = float(attrs["amount"])
                    except (ValueError, TypeError):
                        # 尝试从 mention 中提取
                        mention_num = self._extract_number(ent.mention)
                        if mention_num is not None:
                            attrs["amount"] = mention_num
                            ent.source = "auto_fixed"
                        else:
                            self._reject(result.sentence_id, "value_domain",
                                         f"MonetaryAmount amount 无效: {attrs.get('amount')}")
                            continue
                # currency 标准化
                if "currency" in attrs:
                    currency = str(attrs["currency"])
                    if "人民" in currency or currency == "元":
                        attrs["currency"] = "CNY"
            elif ent.entity_type == "Percentage":
                if "value" in attrs:
                    try:
                        val = float(attrs["value"])
                        if val < 0 or val > 100:
                            val = val / 100.0 if val > 100 else val
                        attrs["value"] = val
                    except (ValueError, TypeError):
                        self._reject(result.sentence_id, "value_domain",
                                     f"Percentage value 无效: {attrs.get('value')}")
            elif ent.entity_type == "Date":
                if "value" in attrs:
                    attrs["value"] = self._normalize_date(str(attrs["value"]))

    def _validate_relation_constraints(self, result: ExtractionResult) -> None:
        """关系类型约束检查"""
        entity_types = {e.entity_id: e.entity_type for e in result.entities}
        valid_relations = []
        for rel in result.relations:
            constraint = RELATION_CONSTRAINTS.get(rel.relation_type)
            if constraint:
                head_type = entity_types.get(rel.head_entity_id, "")
                tail_type = entity_types.get(rel.tail_entity_id, "")
                expected_head = constraint[0]
                expected_tail = constraint[1]

                if isinstance(expected_tail, tuple):
                    tail_ok = tail_type in expected_tail
                else:
                    tail_ok = (tail_type == expected_tail)

                if head_type != expected_head or not tail_ok:
                    self._reject(result.sentence_id, "relation_constraint",
                                 f"关系 {rel.relation_type} 类型约束不满足: "
                                 f"head={head_type}(期望{expected_head}), "
                                 f"tail={tail_type}(期望{expected_tail})")
                    continue
            valid_relations.append(rel)
        result.relations = valid_relations

    def _filter_confidence(self, result: ExtractionResult, threshold: float) -> None:
        """置信度过滤"""
        result.entities = [e for e in result.entities if e.confidence >= threshold]
        result.relations = [r for r in result.relations if r.confidence >= threshold]

    def _resolve_coreferences(self, result: ExtractionResult) -> None:
        """共指消解：代词替换为段落内最近同类型实体"""
        for ent in result.entities:
            original = ent.attributes.get("original_mention", "")
            if original and len(original) <= 3:  # 代词或泛指
                # 在同结果中找最近的同类型实体
                for other in reversed(result.entities):
                    if other.entity_id != ent.entity_id and other.entity_type == ent.entity_type:
                        if len(other.canonical_name) > 3:
                            ent.mention = other.canonical_name
                            ent.canonical_name = other.canonical_name
                            ent.source = "auto_fixed"
                            break
                else:
                    ent.attributes["co_ref_unresolved"] = True

    def _reject(self, sentence_id: str, stage: str, reason: str) -> None:
        self.rejected.append({
            "sentence_id": sentence_id,
            "stage": stage,
            "reason": reason,
        })

    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数值"""
        # 匹配中文数字表达
        match = re.search(r"(\d+[\d,.]*)\s*(?:亿|万|千|百)?\s*(?:元|美元|港元|%|％)?", text)
        if match:
            num_str = match.group(1).replace(",", "")
            try:
                num = float(num_str)
                if "亿" in text[match.start():match.end()]:
                    num *= 100000000
                elif "万" in text[match.start():match.end()]:
                    num *= 10000
                return num
            except ValueError:
                pass
        return None

    def _normalize_date(self, date_str: str) -> str:
        """标准化日期格式为 YYYY-MM-DD"""
        # 已经是标准格式
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str
        # 年份
        year_match = re.search(r"(\d{4})\s*年", date_str)
        if not year_match:
            return date_str
        year = year_match.group(1)
        month_match = re.search(r"(\d{1,2})\s*月", date_str)
        month = month_match.group(1).zfill(2) if month_match else "01"
        day_match = re.search(r"(\d{1,2})\s*日", date_str)
        day = day_match.group(1).zfill(2) if day_match else "01"
        return f"{year}-{month}-{day}"
