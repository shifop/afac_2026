from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import logging
import jieba
from whoosh import query as wquery
from whoosh.query import Query
from src.models.document import Chunk
from src.models.query import StructuredQuery, QueryField, RetrievedChunk
from src.models.schema import ChannelWeights
from src.indexer.index_manager import IndexManager

from copy import deepcopy

logger = logging.getLogger(__name__)

MIN_SCORE_THRESHOLD = 0.01

# 通道 → chunk 索引字段映射（仅用于标题和内容通道）
CHUNK_CHANNEL_FIELD_MAP = {
    "title": ["section_title", "section_path"],
    "content": ["content"],
}

# 实体/关系独立索引字段映射
META_FIELDS = ["insurer", "contract_name"]
ENTITY_FIELDS = ["name", "desc"]
RELATION_FIELDS = ["subject", "predicate", "object"]

# 通道 → 索引类型映射
CHANNEL_INDEX_MAP = {
    "entity": "entity",
    "relation": "relation",
    "title": "chunk",
    "content": "chunk",
}


class WhooshSearcher:
    def __init__(
        self,
        index: IndexManager,
        chunk_dict: Dict[str, Chunk],
        context_window: Optional[Dict] = None,
    ):
        self.index = index
        self.chunk_dict = chunk_dict
        self.context_window = context_window or {"prev": 1, "next": 1}

    def search(
        self,
        structured_q: StructuredQuery,
        top_k: int = 10,
        doc_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        # 查询meta数据，限制后续查询范围
        structured_q_copy = deepcopy(structured_q)
        doc_ids = []
        for filter_ in structured_q_copy.filters:
            structured_q_copy.fields = filter_
            meta_query = self._build_channel_query(structured_q_copy, META_FIELDS)
            
            if meta_query is not None:
                hits = self.index.search_metas_for_scoring(
                    meta_query
                )
                doc_ids.extend([_[0] for _ in hits[:2]])
        if len(doc_ids)==0:
            doc_ids = None
        # 构建文档过滤器
        doc_filter = self._build_doc_filter(doc_ids)

        # 收集各通道的 (chunk_id, score) 字典
        channel_scores: Dict[str, Dict[str, float]] = {}

        # 1. 处理实体通道
        entity_query = self._build_channel_query(structured_q, ENTITY_FIELDS)
        if entity_query is not None:
            hits = self.index.search_entities_for_scoring(
                entity_query,
                limit=structured_q.entity_limit,
                doc_filter=doc_filter,
            )
            channel_scores["entity"] = self._aggregate_max_score(hits)
        else:
            channel_scores["entity"] = {}

        # 2. 处理关系通道
        relation_query = self._build_channel_query(structured_q, RELATION_FIELDS)
        if relation_query is not None:
            hits = self.index.search_relations_for_scoring(
                relation_query,
                limit=structured_q.relation_limit,
                doc_filter=doc_filter,
            )
            channel_scores["relation"] = self._aggregate_max_score(hits)
        else:
            channel_scores["relation"] = {}

        # 3. 处理标题通道和内容通道（基于 chunk 索引）
        for ch, index_fields in CHUNK_CHANNEL_FIELD_MAP.items():
            q = self._build_channel_query(structured_q, index_fields)
            if q is not None:
                # 在 chunk 索引上搜索
                if ch == "content" and structured_q.free_text:
                    # 若有 free_text，合并为 Or 查询
                    fallback = self._build_content_fallback(structured_q.free_text)
                    q = wquery.Or([q, fallback])
                self._search_chunk_channel(ch, q, doc_filter, channel_scores)
            else:
                # 即使无结构化字段，free_text 也可能给 content 通道添加查询
                if ch == "content" and structured_q.free_text:
                    fallback = self._build_content_fallback(structured_q.free_text)
                    self._search_chunk_channel(ch, fallback, doc_filter, channel_scores)
                else:
                    channel_scores[ch] = {}

        # 移除空通道
        active_channels = {ch for ch, scores in channel_scores.items() if scores}
        if not active_channels:
            logger.warning("无有效查询，返回空列表")
            return []

        # 4. 归一化与加权求和
        norm_scores = self._normalize_channel_scores(channel_scores)
        total_scores = defaultdict(float)
        ch_weights = structured_q.channel_weights

        # 计算有效通道权重和
        total_weight = sum(
            getattr(ch_weights, f"{ch}_channel", 0.0) for ch in active_channels
        )
        if total_weight == 0:
            total_weight = 1.0

        for ch in active_channels:
            w = getattr(ch_weights, f"{ch}_channel", 0.0)
            normalized_w = w / total_weight
            for cid, s in norm_scores[ch].items():
                total_scores[cid] += normalized_w * s

        # 5. 排序、截断、后处理
        sorted_ids = sorted(
            total_scores.keys(), key=lambda x: total_scores[x], reverse=True
        )[:top_k]

        results = []
        for cid in sorted_ids:
            if total_scores[cid] < MIN_SCORE_THRESHOLD:
                continue
            chunk = self.chunk_dict.get(cid)
            if chunk is None:
                continue
            prev_chunks = self._get_adjacent_chunks(chunk, "prev")
            next_chunks = self._get_adjacent_chunks(chunk, "next")
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=total_scores[cid],
                    prev_chunks=prev_chunks,
                    next_chunks=next_chunks,
                )
            )
        return results

    def _search_chunk_channel(
        self,
        channel: str,
        query: Query,
        doc_filter: Optional[Query],
        channel_scores: dict,
        limit:int=20
    ):
        """在 chunk 索引上执行通道查询，结果存入 channel_scores"""
        with self.index.index.searcher() as searcher:
            q = query
            if doc_filter is not None:
                q = wquery.And([doc_filter, q])
            try:
                results = searcher.search(q, limit=limit)
                channel_scores[channel] = {
                    hit["chunk_id"]: hit.score for hit in results
                }
                logger.debug(f"通道 {channel} 命中 {len(channel_scores[channel])} 条")
            except Exception as e:
                logger.error(f"通道 {channel} 搜索失败: {e}")
                channel_scores[channel] = {}

    def _build_channel_query(
        self, structured_q: StructuredQuery, index_fields: List[str]
    ) -> Optional[Query]:
        """构造通道查询，适用于 chunk/实体/关系索引"""
        clauses = []
        for qf in structured_q.fields:
            if qf.value is None or qf.target_index_field not in index_fields:
                continue
            boost = qf.weight
            sub_clauses = self._build_field_query(
                qf.target_index_field, qf.value, qf.match_method, boost
            )
            clauses.extend(sub_clauses)
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return wquery.Or(clauses)

    def _build_field_query(
        self, field: str, value: str, match_method: str, boost: float
    ) -> List[Query]:
        clauses = []
        tokens = jieba.lcut(value)
        tokens = [t.strip() for t in tokens if t.strip()]

        if match_method == "exact_then_fuzzy":
            for token in tokens:
                clauses.append(wquery.Term(field, token.lower(), boost=boost * 1.0))
                if len(token) <= 4:
                    clauses.append(
                        wquery.FuzzyTerm(
                            field, token.lower(), maxdist=1, boost=boost * 0.6
                        )
                    )
        elif match_method == "fuzzy":
            for token in tokens:
                clauses.append(wquery.Term(field, token.lower(), boost=boost * 1.0))
                if len(token) <= 4:
                    clauses.append(
                        wquery.FuzzyTerm(
                            field, token.lower(), maxdist=1, boost=boost * 0.5
                        )
                    )
        elif match_method == "substring":
            for token in tokens:
                clauses.append(wquery.Term(field, token.lower(), boost=boost))
        else:
            for token in tokens:
                clauses.append(wquery.Term(field, token.lower(), boost=boost))
        return clauses

    def _build_doc_filter(
        self, doc_ids: Optional[List[str]]
    ) -> Optional[Query]:
        if not doc_ids:
            return None
        clauses = [wquery.Term("doc_id", did) for did in doc_ids]
        if len(clauses) == 1:
            return clauses[0]
        return wquery.Or(clauses)

    def _build_content_fallback(self, free_text: str) -> wquery.Query:
        tokens = jieba.lcut(free_text)
        tokens = [t.strip().lower() for t in tokens if t.strip()]
        if not tokens:
            return wquery.Every()
        clauses = [wquery.Term("content", t, boost=0.3) for t in tokens]
        return wquery.Or(clauses)

    def _aggregate_max_score(
        self, hits: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """按 chunk_id 聚合，取最大得分"""
        agg = {}
        for chunk_id, score in hits:
            if chunk_id not in agg or score > agg[chunk_id]:
                agg[chunk_id] = score
        return agg

    def _normalize_channel_scores(
        self, channel_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        norm = {}
        for ch, scores in channel_scores.items():
            if not scores:
                norm[ch] = {}
                continue
            vals = list(scores.values())
            vmin, vmax = min(vals), max(vals)
            if vmax == vmin:
                norm[ch] = {k: 1.0 for k in scores}
            else:
                norm[ch] = {
                    k: (v - vmin) / (vmax - vmin) for k, v in scores.items()
                }
        return norm

    def _get_adjacent_chunks(self, chunk: Chunk, direction: str) -> List[Chunk]:
        result = []
        count = self.context_window.get(direction, 1)
        current = chunk
        for _ in range(count):
            adj_id = (
                current.prev_chunk_id if direction == "prev" else current.next_chunk_id
            )
            if adj_id and adj_id in self.chunk_dict:
                adj_chunk = self.chunk_dict[adj_id]
                result.append(adj_chunk)
                current = adj_chunk
            else:
                break
        return result
