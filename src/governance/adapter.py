"""
治理管道输出 → 索引器输入 适配层

将 governance pipeline 的 sections/paragraphs/sentences + 顶层 entities/relations
转换为 IndexManager 期望的 chunks (embedded entities/relations) + structured_data 格式。
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class GovernanceAdapter:
    """将治理格式 (output.v2) 适配为索引器格式 (documents.all)"""

    def __init__(self, target_chunk_chars: int = 3000):
        self.target_chunk_chars = target_chunk_chars

    # ── 主入口 ─────────────────────────────────────────────

    def adapt(self, governance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换单个治理文档为索引器兼容格式。

        Args:
            governance_data: 治理管道 _build_result 产出的字典

        Returns:
            包含 chunks / structured_data / doc_tree / version_ts 的字典，
            可直接被 DocumentLoader 加载。
        """
        doc_id = governance_data["doc_id"]
        sections = governance_data.get("sections", [])
        title = governance_data.get("title", "")
        doc_type = governance_data.get("doc_type", "")

        # 1. 从 sections 扁平化为 chunks
        chunks, sent_to_chunk = self._sections_to_chunks(doc_id, sections, title)

        # 2. 构建 sentence → chunk 的扩展索引 (包含 sub-sentence 的父段落)
        sent_to_chunk = self._expand_sent_mapping(sections, sent_to_chunk)

        # 3. 分配治理实体到 chunks
        self._distribute_entities(
            governance_data.get("entities", []), sent_to_chunk, chunks
        )

        # 4. 分配治理关系到 chunks
        self._distribute_relations(
            governance_data.get("relations", []),
            governance_data.get("entities", []),
            sections,
            chunks,
        )

        # 5. 链接 chunks
        for i, chunk in enumerate(chunks):
            chunk["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
            chunk["next_chunk_id"] = (
                chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else None
            )

        # 6. 构建 structured_data 和 doc_tree
        structured_data = self._build_structured_data(governance_data)
        doc_tree = self._build_doc_tree(sections, title)

        # 7. 将 doc_tree 嵌入第一个 chunk (兼容 IndexManager 的加载逻辑)
        if chunks:
            chunks[0]["doc_tree"] = doc_tree

        # 生成 version_ts
        version_ts = int(time.time() * 1000)

        return {
            "doc_id": doc_id,
            "doc_type": self._normalise_doc_type(doc_type),
            "version_ts": version_ts,
            "structured_data": structured_data,
            "chunks": chunks,
        }

    # ── Chunk 构建 ──────────────────────────────────────────

    def _sections_to_chunks(
        self,
        doc_id: str,
        sections: List[Dict],
        title: str,
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        将 sections → chunks。
        返回 (chunks_list, sentence_id→chunk_id 映射)
        """
        chunks: List[Dict] = []
        sent_to_chunk: Dict[str, str] = {}
        chunk_index = 0

        # 无内容的根 section
        root_sections = [s for s in sections if s.get("paragraphs")]
        if not root_sections and sections:
            # 所有 section 都没段落：用第一个 section 作为根
            root_sections = sections[:1]

        for section in root_sections:
            section_path = section.get("section_path", "")
            section_title = section.get("title", "")

            # 收集该 section 下所有句子
            all_sents: List[Dict] = []
            for para in section.get("paragraphs", []):
                all_sents.extend(para.get("sentences", []))

            if not all_sents:
                continue

            # 按字符数分块
            sub_chunks = self._split_sents_into_chunks(
                all_sents, section_path, section_title
            )

            for sc in sub_chunks:
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                chunk = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "section_title": sc["section_title"],
                    "section_path": sc["section_path"],
                    "content": sc["content"],
                    "chunk_index": chunk_index,
                    "prev_chunk_id": None,
                    "next_chunk_id": None,
                    "entities": [],
                    "relations": [],
                }
                chunks.append(chunk)
                for s in sc["sentences"]:
                    sent_to_chunk[s["sentence_id"]] = chunk_id
                chunk_index += 1

        # 附加 reading_guide 信息到对应 section 的 chunk 中
        # - content: 首 chunk 注入完整指南前缀
        # - section_title: 所有 chunk 注入功能标签（对齐 LLM 提取的 section_title 语义类别）
        for section in sections:
            guide = section.get("reading_guide")
            if not guide or guide.get("skip"):
                continue
            sp = section.get("section_path", "")
            guide_text = self._format_reading_guide(guide)
            title_tags = self._format_guide_title_tags(guide)
            if not guide_text and not title_tags:
                continue

            content_injected = False
            for chunk in chunks:
                if chunk["section_path"] != sp:
                    continue
                # 首 chunk: content 注入完整指南前缀
                if guide_text and not content_injected:
                    chunk["content"] = guide_text + "\n\n" + chunk["content"]
                    content_injected = True
                # 所有 chunk: section_title 注入功能标签
                if title_tags and title_tags not in chunk["section_title"]:
                    chunk["section_title"] = f"{chunk['section_title']} [{title_tags}]"

        return chunks, sent_to_chunk

    def _split_sents_into_chunks(
        self,
        sentences: List[Dict],
        section_path: str,
        section_title: str,
    ) -> List[Dict]:
        """将句子列表按目标字符数切分为多个 chunk 组"""
        result: List[Dict] = []
        buf_sents: List[Dict] = []
        buf_chars = 0

        for s in sentences:
            text = s.get("text", "")
            if buf_sents and buf_chars + len(text) > self.target_chunk_chars:
                result.append({
                    "sentences": buf_sents,
                    "section_path": section_path,
                    "section_title": section_title,
                    "content": "\n".join(x["text"] for x in buf_sents),
                })
                buf_sents = []
                buf_chars = 0
            buf_sents.append(s)
            buf_chars += len(text)

        if buf_sents:
            result.append({
                "sentences": buf_sents,
                "section_path": section_path,
                "section_title": section_title,
                "content": "\n".join(x["text"] for x in buf_sents),
            })
        return result

    @staticmethod
    def _format_reading_guide(guide: Dict) -> str:
        """将 reading_guide 格式化为可索引文本（注入 content）"""
        parts = []
        has = guide.get("has", "")
        if has:
            parts.append(f"[关键内容] {has}")
        for_tag = guide.get("for", [])
        if for_tag:
            parts.append(f"[适用问题] {', '.join(for_tag)}")
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _format_guide_title_tags(guide: Dict) -> str:
        """提取 reading_guide 的功能标签（注入 section_title 用于对齐 LLM 语义类别检索）"""
        tags = []
        for_tag = guide.get("for", [])
        if for_tag:
            tags.extend(for_tag)
        # 从 has 中提取前几个关键词作为补充
        has = guide.get("has", "")
        if has:
            keywords = [kw.strip() for kw in has.split(",")[:5]]
            tags.extend(keywords)
        return ", ".join(tags) if tags else ""

    # ── sentence → chunk 扩展索引 ───────────────────────────

    def _expand_sent_mapping(
        self,
        sections: List[Dict],
        sent_to_chunk: Dict[str, str],
    ) -> Dict[str, str]:
        """
        扩展映射：除了精确匹配 sentence_id，还处理 sub-sentence (长段落拆分产生的 _subN_s0)。
        策略: 将 _subN_s0 映射到与 _sX 所在的同一个 chunk。
        """
        import re

        expanded: Dict[str, str] = dict(sent_to_chunk)

        for section in sections:
            for para in section.get("paragraphs", []):
                for s in para.get("sentences", []):
                    sid = s["sentence_id"]
                    if sid not in sent_to_chunk:
                        # 尝试匹配同一段落中的其他句子
                        para_id = re.sub(r"_s\d+$", "", sid)
                        for known_sid, chunk_id in sent_to_chunk.items():
                            if known_sid.startswith(para_id):
                                expanded[sid] = chunk_id
                                break
        return expanded

    # ── 实体分配 ────────────────────────────────────────────

    def _distribute_entities(
        self,
        entities: List[Dict],
        sent_to_chunk: Dict[str, str],
        chunks: List[Dict],
    ) -> None:
        """将顶层实体按 sentence_id 分配到对应 chunk"""
        chunk_index: Dict[str, Dict] = {c["chunk_id"]: c for c in chunks}
        unassigned: List[Dict] = []

        for e in entities:
            sid = e.get("sentence_id", "")
            chunk_id = sent_to_chunk.get(sid)

            if chunk_id and chunk_id in chunk_index:
                chunk_index[chunk_id]["entities"].append(self._adapt_entity(e))
                continue

            # 尝试 section_path fallback
            section_path = self._extract_section_path_from_sid(sid)
            if section_path:
                for c in chunks:
                    if c["section_path"] == section_path:
                        c["entities"].append(self._adapt_entity(e))
                        break
                else:
                    unassigned.append(e)
            else:
                unassigned.append(e)

        # 兜底: tbl_* sentence_ids（表格实体）或其他无法匹配的实体，
        # 分配到第一个同 section_path 的 chunk，或第一个 chunk
        if unassigned and chunks:
            for e in unassigned:
                chunks[0]["entities"].append(self._adapt_entity(e))
            logger.debug(
                f"适配器: {len(unassigned)} 个实体无法精确定位到 chunk，"
                f"已分配到首 chunk (多为表格实体)"
            )

    @staticmethod
    def _adapt_entity(e: Dict) -> Dict:
        """治理实体 → 索引器实体"""
        name = e.get("canonical_name") or e.get("mention", "")
        desc_parts = []
        # 如果 mention 不同于 canonical_name，保留 mention
        if (
            e.get("mention")
            and e.get("canonical_name")
            and e["mention"] != e["canonical_name"]
        ):
            desc_parts.append(f"原文: {e['mention']}")
        # 属性
        for k, v in (e.get("attributes") or {}).items():
            if v:
                desc_parts.append(f"{k}: {v}")
        desc = "; ".join(desc_parts) if desc_parts else ""
        return {
            "name": name,
            "desc": desc,
            "entity_type": e.get("entity_type", ""),
        }

    @staticmethod
    def _extract_section_path_from_sid(sentence_id: str) -> str:
        """
        从 sentence_id 提取 section_path。
        格式: doc_id#section_path > ... > title_p0_s0
        """
        if "#" not in sentence_id:
            return ""
        after_doc = sentence_id.split("#", 1)[1]  # "section_path > ... > title_p0_s0"
        # 去掉末尾的 _pN_sM
        import re

        path = re.sub(r"_p\d+.*$", "", after_doc)
        return path if path != after_doc else ""

    # ── 关系分配 ────────────────────────────────────────────

    def _distribute_relations(
        self,
        relations: List[Dict],
        entities: List[Dict],
        sections: List[Dict],
        chunks: List[Dict],
    ) -> None:
        """将顶层关系分配到对应 chunk，同时解析 entity_id → 实体名称"""
        entity_map = self._build_entity_name_map(entities)

        chunk_index: Dict[str, Dict] = {c["chunk_id"]: c for c in chunks}

        for r in relations:
            # 解析实体名称
            head_id = r.get("head_entity_id", "")
            tail_id = r.get("tail_entity_id", "")
            head_name = entity_map.get(head_id, head_id)
            tail_name = entity_map.get(tail_id, tail_id)

            adapted_rel = {
                "subject": head_name,
                "predicate": self._translate_relation_type(r.get("relation_type", "")),
                "object": tail_name,
                "desc": self._build_relation_desc(r),
            }

            # 确定 chunk: 优先用 evidence.section_path
            evidence = r.get("evidence", {})
            section_path = evidence.get("section_path", "")
            target = None
            if section_path:
                for c in chunks:
                    if c["section_path"] == section_path:
                        target = c
                        break
            # Fallback: 第一个 chunk
            if target is None and chunks:
                target = chunks[0]

            if target:
                target["relations"].append(adapted_rel)

    @staticmethod
    def _build_entity_name_map(entities: List[Dict]) -> Dict[str, str]:
        """构建 entity_id → 名称 的映射"""
        mapping: Dict[str, str] = {}
        for e in entities:
            eid = e.get("entity_id", "")
            name = e.get("canonical_name") or e.get("mention", "")
            if eid:
                mapping[eid] = name
        return mapping

    @staticmethod
    def _translate_relation_type(rel_type: str) -> str:
        """将治理管道的关系类型翻译为中英文可检索文本"""
        TYPE_MAP = {
            "has_director": "董事",
            "has_shareholder": "股东",
            "reports_financial": "财务指标",
            "has_credit_rating": "信用评级",
            "references_law": "法律引用",
            "defines_term": "术语定义",
            "covers_risk": "承保风险",
            "excludes_risk": "除外风险",
            "analyzes_industry": "行业分析",
            "gives_opinion": "意见观点",
        }
        cn = TYPE_MAP.get(rel_type, rel_type)
        return f"{cn} ({rel_type})"

    @staticmethod
    def _build_relation_desc(r: Dict) -> str:
        """从 relation 的 evidence 和 properties 构建描述"""
        parts = []
        evidence = r.get("evidence", {})
        if evidence.get("section_path"):
            parts.append(f"所在章节: {evidence['section_path']}")
        props = r.get("properties", {})
        if props:
            parts.append(
                ", ".join(f"{k}={v}" for k, v in props.items() if v)
            )
        return "; ".join(parts)

    # ── structured_data ─────────────────────────────────────

    @staticmethod
    def _build_structured_data(data: Dict) -> Dict:
        """构建与 IndexManager 期望兼容的 structured_data"""
        title = data.get("title", "")
        doc_type = data.get("doc_type", "")

        result = {
            "title": title,
            "doc_type": doc_type,
            "file_path": data.get("file_path", ""),
            "processed_at": data.get("processed_at", ""),
        }

        # 从 sections 中提取更准确的合同名 / 产品名
        sections = data.get("sections", [])
        substantive_title = GovernanceAdapter._find_substantive_title(sections, title)

        # 预填所有可能的字段
        result.setdefault("contract_name", "")
        result.setdefault("insurer", "")
        result.setdefault("company_name", "")
        result.setdefault("fiscal_year_end", "")
        result.setdefault("stock_code", "")
        result.setdefault("institution", "")
        result.setdefault("issuer_name", "")
        result.setdefault("doc_type_name", "")
        result.setdefault("title", title)

        if doc_type == "保险合同":
            insurer, contract_name = GovernanceAdapter._parse_insurance_meta(
                substantive_title, sections
            )
            result["contract_name"] = contract_name or substantive_title or title
            result["insurer"] = insurer or ""

        elif doc_type == "年报":
            result["company_name"] = substantive_title or title

        elif doc_type == "行业研报":
            result["title"] = substantive_title or title

        elif doc_type == "金融法规":
            result["title"] = substantive_title or title

        elif doc_type == "募集说明书":
            result["issuer_name"] = substantive_title or title
            result["doc_type_name"] = doc_type

        return result

    @staticmethod
    def _find_substantive_title(sections: List[Dict], fallback: str) -> str:
        """找到第一个有实质内容的 section title（跳过阅读指引、目录等）"""
        skip_keywords = ["阅读指引", "阅 读 指 引", "目录", "条款目录"]
        for s in sections:
            t = s.get("title", "").strip()
            if t and not any(kw in t for kw in skip_keywords):
                guide = s.get("reading_guide")
                if guide and guide.get("skip"):
                    continue
                return t
        return fallback

    @staticmethod
    def _parse_insurance_meta(title: str, sections: List[Dict]) -> tuple:
        """
        尝试从标题和章节中提取 (保险公司名称, 产品名称)。
        """
        insurer = ""
        contract_name = title

        # 策略1: 在标题中查找保险公司名
        for sep in ["财产保险股份有限公司", "人寿保险股份有限公司", "健康保险股份有限公司",
                     "养老保险股份有限公司", "财产保险公司", "人寿保险公司"]:
            if sep in title:
                idx = title.index(sep) + len(sep)
                insurer = title[:idx]
                contract_name = title[idx:].strip()
                break

        # 策略2: 在 section content 中查找保险公司名
        if not insurer:
            for s in sections[:3]:
                for p in s.get("paragraphs", []):
                    content = p.get("content", "")
                    for keyword in ["财产保险股份有限公司", "人寿保险股份有限公司",
                                     "健康保险股份有限公司", "养老保险股份有限公司"]:
                        if keyword in content:
                            pos = content.index(keyword) + len(keyword)
                            # 从公司名末尾往前找到起点 (取最近的"公司"或标点之后)
                            prefix = content[:pos]
                            # 找到真正的公司名起点
                            name_start = 0
                            for marker in ["指", "是", "为", "：", ":", "。", "，", "、", "\n"]:
                                idx = prefix.rfind(marker)
                                if idx > name_start:
                                    name_start = idx + 1
                            candidate = prefix[name_start:pos].strip()
                            # 清理：去掉引导词和括号
                            for bad_start in ["的", "由", "向", "对", "以"]:
                                if candidate.startswith(bad_start) and len(candidate) > 3:
                                    candidate = candidate[1:]
                            if len(candidate) > 4 and "保险" in candidate:
                                insurer = candidate
                                break
                    if insurer:
                        break
                if insurer:
                    break

        # 策略3: 从 section_path 中提取产品名
        if contract_name == title:
            for s in sections:
                path = s.get("section_path", "")
                if "利益条款" in path or "条款" == path[-2:]:
                    # 取 section_path 的第一段（通常是产品名）
                    parts = [p.strip() for p in path.split(">")]
                    if parts and len(parts[0]) > 4:
                        contract_name = parts[0]
                        break

        return insurer, contract_name

    # ── doc_tree ────────────────────────────────────────────

    @staticmethod
    def _build_doc_tree(sections: List[Dict], root_title: str) -> Dict:
        """
        从 sections 的 section_path 构建嵌套目录树。
        使用 chunk 索引范围标注每个节点的起止 chunk。
        """
        if not sections:
            return {"title": root_title, "children": []}

        root: Dict = {"title": root_title, "children": []}
        node_registry: Dict[str, Dict] = {}

        for i, s in enumerate(sections):
            sp = s.get("section_path", "")
            if not sp:
                continue
            parts = [p.strip() for p in sp.split(">") if p.strip()]
            if not parts:
                continue

            current = root
            for depth, part in enumerate(parts):
                key = " > ".join(parts[: depth + 1])
                if key not in node_registry:
                    node = {
                        "title": part,
                        "children": [],
                        "chunk_id_start": f"_chunk_{i}",
                        "chunk_id_end": f"_chunk_{i}",
                    }
                    current.setdefault("children", []).append(node)
                    node_registry[key] = node
                else:
                    node = node_registry[key]
                current = node

        return root

    # ── doc_type 规范化 ─────────────────────────────────────

    @staticmethod
    def _normalise_doc_type(governance_type: str) -> str:
        """将治理管道的 doc_type 映射为索引器的英文常量"""
        MAP = {
            "年报": "ANNUAL",
            "募集说明书": "PROSPECTUS",
            "保险合同": "CONTRACT",
            "金融法规": "REG",
            "行业研报": "RESEARCH",
        }
        return MAP.get(governance_type, governance_type.upper())
