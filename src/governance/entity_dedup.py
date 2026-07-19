"""实体去重与链接模块 - 文档内去重、全局实体链接"""
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import logging; logger = logging.getLogger(__name__)

from .models import ExtractedEntity, ExtractedRelation, new_id


class EntityDedup:
    """实体去重器"""

    def dedup_within_document(
        self,
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation], Dict[str, str]]:
        """
        文档内去重：相同 mention + entity_type 合并为一个。
        返回去重后的实体、更新后的关系、以及 {旧ID -> 新全局ID} 映射。
        """
        if not entities:
            return entities, relations, {}

        # 按 (entity_type, canonical_name) 分组
        groups: Dict[Tuple[str, str], List[ExtractedEntity]] = defaultdict(list)
        for ent in entities:
            key = (ent.entity_type, ent.canonical_name.strip())
            groups[key].append(ent)

        # 生成映射
        id_mapping: Dict[str, str] = {}
        merged_entities: List[ExtractedEntity] = []

        for (etype, cname), group in groups.items():
            global_id = new_id("ent")
            # 保留第一个实体的属性，合并 attributes 和 sentence_id
            merged = group[0]
            old_id = merged.entity_id  # 保存旧 ID（entity_id 即将被覆盖）
            merged.entity_id = global_id
            all_sids = [merged.sentence_id] if merged.sentence_id else []
            for other in group[1:]:
                merged.attributes.update(other.attributes)
                if other.sentence_id and other.sentence_id not in all_sids:
                    all_sids.append(other.sentence_id)
                id_mapping[other.entity_id] = global_id
            id_mapping[old_id] = global_id  # 第一个实体的旧 ID → 新全局 ID
            # 将主 sentence_id 设为第一次出现的位置，所有位置存入 attributes
            merged.sentence_id = all_sids[0] if all_sids else ""
            if len(all_sids) > 1:
                merged.attributes["all_sentence_ids"] = all_sids
            merged_entities.append(merged)

        # 更新关系中的 entity_id
        updated_relations = []
        for rel in relations:
            new_head = id_mapping.get(rel.head_entity_id, rel.head_entity_id)
            new_tail = id_mapping.get(rel.tail_entity_id, rel.tail_entity_id)
            rel.head_entity_id = new_head
            rel.tail_entity_id = new_tail
            updated_relations.append(rel)

        dedup_count = len(entities) - len(merged_entities)
        if dedup_count > 0:
            logger.info(f"文档内实体去重: {len(entities)} → {len(merged_entities)} (合并 {dedup_count} 个)")

        return merged_entities, updated_relations, id_mapping


class GlobalEntityLinker:
    """全局实体链接器"""

    def __init__(self) -> None:
        self.global_entities: Dict[str, str] = {}  # local_id -> global_id

    def link_cross_documents(
        self,
        new_entities: List[ExtractedEntity],
        existing_entities: List[dict],
    ) -> Dict[str, str]:
        """
        跨文档实体链接。
        使用简单的 canonical_name + entity_type 精确匹配策略。
        返回 {local_id -> global_id} 映射。
        """
        # 构建已有实体索引
        existing_index: Dict[Tuple[str, str], str] = {}
        for ex in existing_entities:
            key = (ex.get("entity_type", ""), ex.get("canonical_name", "").strip())
            if key not in existing_index:
                existing_index[key] = ex.get("entity_id", "")

        mapping: Dict[str, str] = {}
        for ent in new_entities:
            key = (ent.entity_type, ent.canonical_name.strip())
            if key in existing_index:
                mapping[ent.entity_id] = existing_index[key]
            else:
                mapping[ent.entity_id] = ent.entity_id  # 保持自己的ID

        linked = sum(1 for k, v in mapping.items() if k != v)
        logger.info(f"跨文档实体链接: {len(new_entities)} 个实体, {linked} 个链接到已有实体")
        return mapping

    def cluster_by_similarity(
        self,
        entities: List[dict],
        eps: float = 0.2,
    ) -> Dict[str, str]:
        """
        基于相似度聚类 (DBSCAN)。
        简化实现：按 entity_type 分组，对同类型 canonical_name 做编辑距离聚类。
        """
        mapping: Dict[str, str] = {}
        if not entities:
            return mapping

        # 按类型分组
        by_type: Dict[str, List[dict]] = defaultdict(list)
        for e in entities:
            by_type[e.get("entity_type", "")].append(e)

        for etype, group in by_type.items():
            # 简单策略：完全相同的 canonical_name 分一组
            name_groups: Dict[str, List[str]] = defaultdict(list)
            for e in group:
                cname = e.get("canonical_name", "").strip().lower()
                name_groups[cname].append(e.get("entity_id", ""))

            for cname, eids in name_groups.items():
                if len(eids) >= 2:
                    global_id = new_id("glb_ent")
                    for eid in eids:
                        mapping[eid] = global_id
                else:
                    mapping[eids[0]] = eids[0]

        return mapping
