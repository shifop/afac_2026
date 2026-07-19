"""治理 Pipeline - 文档结构化抽取全流程"""
import asyncio
import hashlib
import json
import time
import os
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple
import logging; logger = logging.getLogger(__name__)

_CACHE_FILE = "_cache.json"

from .models import (
    DocumentMeta, Section, Paragraph, Sentence,
    ExtractedEntity, ExtractedRelation, ExtractionResult,
    DocType, Location, new_id,
)
from .preprocessor import DocumentPreprocessor
from .splitter import StructuralSplitter
from .table_processor import TableProcessor
from .long_paragraph import LongParagraphHandler
from .llm_client import LLMClient
from .entity_extractor import EntityExtractor
from .reading_guide import ReadingGuideGenerator
from .post_processor import PostProcessor
from .entity_dedup import EntityDedup


class GovernancePipeline:
    """数据治理主流水线 — 处理 MD 文件，输出结构化 JSON"""

    def __init__(
        self,
        llm_client: LLMClient,
        batch_chars: int = 4000,
        max_concurrent: int = 8,
        use_cache: bool = True,
    ) -> None:
        self.llm = llm_client
        self.batch_chars = batch_chars          # 合并后每批目标字符数
        self.max_concurrent = max_concurrent    # 并发 LLM 调用上限
        self.use_cache = use_cache              # 是否启用输出缓存（跳过已处理文件）
        self.preprocessor = DocumentPreprocessor()
        self.splitter = StructuralSplitter()
        self.table_processor = TableProcessor()
        self.long_para_handler = LongParagraphHandler(max_chars=1000)
        self.extractor = EntityExtractor(self.llm)
        self.guide_gen = ReadingGuideGenerator(self.llm)
        self.post_processor = PostProcessor()
        self.entity_dedup = EntityDedup()

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """处理单个文件，返回完整结构化结果"""
        start_time = time.time()
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        logger.info(f"╔══════════════════════════════════════════════════╗")
        logger.info(f"║  开始处理: {os.path.basename(file_path)[:50]}")
        logger.info(f"║  原始大小: {len(raw_content):,} chars")
        logger.info(f"╚══════════════════════════════════════════════════╝")

        # Phase 1: 预处理
        logger.info(f"[阶段 1/5] 预处理...")
        content, meta = self.preprocessor.process(file_path, raw_content)
        doc_id = meta.doc_id
        logger.info(f"[阶段 1/5] 完成 → doc_id={doc_id}, type={meta.doc_type.value}, title={meta.title[:60]}")

        # Phase 2: 拆分 + 表格
        logger.info(f"[阶段 2/5] 结构拆分 + 表格解析...")
        sections = self.splitter.split(doc_id, content)
        all_tables = self._process_tables(sections)
        para_count = sum(len(s.paragraphs) for s in sections)
        sent_count = sum(sum(len(p.sentences) for p in s.paragraphs) for s in sections)
        logger.info(f"[阶段 2/5] 完成 → {len(sections)} 章节, {para_count} 段落, {sent_count} 句子, {len(all_tables)} 表格")

        # Phase 3: 实体关系抽取 (LLM 密集)
        logger.info(f"[阶段 3/5] 实体关系抽取 (LLM)...")
        all_sentences, all_extraction_results = self._collect_sentences_and_extract(
            sections, meta.doc_type, all_tables
        )
        raw_ent = sum(len(r.entities) for r in all_extraction_results)
        raw_rel = sum(len(r.relations) for r in all_extraction_results)
        logger.info(f"[阶段 3/5] 完成 → 原始 {raw_ent} 实体, {raw_rel} 关系 | "
                    f"累计 LLM: {self.llm.stats['call_count']} 次, {self.llm.stats['total_tokens']:,} tokens")

        # Phase 4: 后处理 + 去重
        logger.info(f"[阶段 4/5] 后处理 + 去重...")
        all_extraction_results = self.post_processor.process(all_extraction_results)
        rejected_count = len(self.post_processor.rejected)

        all_entities = [e for r in all_extraction_results for e in r.entities]
        all_relations = [r for res in all_extraction_results for r in res.relations]
        merged_entities, merged_relations, _ = self.entity_dedup.dedup_within_document(
            all_entities, all_relations
        )
        logger.info(f"[阶段 4/5] 完成 → {len(merged_entities)} 实体 (去重后), "
                    f"{len(merged_relations)} 关系, {rejected_count} 被拒")

        # Phase 5: 阅读指南 (LLM) — 并发执行
        logger.info(f"[阶段 5/5] 阅读指南生成 (LLM) — {len(sections)} 章节 (并发)...")
        guide_ok, guide_fail = asyncio.run(
            self._generate_guides_async(sections)
        )
        logger.info(f"[阶段 5/5] 完成 → {guide_ok} 成功, {guide_fail} 失败 | "
                    f"累计 LLM: {self.llm.stats['call_count']} 次, {self.llm.stats['total_tokens']:,} tokens")

        elapsed = time.time() - start_time

        # ---- 输出详细统计 ----
        self._log_statistics(
            sections, all_extraction_results, merged_entities, merged_relations,
            rejected_count, elapsed,
        )

        logger.info(
            f"╔══════════════════════════════════════════════════╗\n"
            f"║  处理完成: {os.path.basename(file_path)[:50]}\n"
            f"║  耗时: {elapsed:.1f}s | LLM: {self.llm.stats['call_count']} 次, "
            f"{self.llm.stats['total_tokens']:,} tokens\n"
            f"║  结果: {len(sections)} 章节, {para_count} 段落, {len(merged_entities)} 实体, "
            f"{len(merged_relations)} 关系\n"
            f"╚══════════════════════════════════════════════════╝"
        )
        return self._build_result(
            meta, sections, merged_entities, merged_relations,
            rejected_count, elapsed, file_path,
        )

    # ---- 缓存 ----

    def _load_cache(self, output_dir: str) -> Dict[str, Any]:
        """加载缓存文件 {file_key: {doc_id, mtime, size, processed_at}}"""
        cache_path = os.path.join(output_dir, _CACHE_FILE)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"缓存文件损坏，将重建: {cache_path}")
        return {}

    def _save_cache(self, output_dir: str, cache: Dict[str, Any]) -> None:
        """保存缓存文件"""
        os.makedirs(output_dir, exist_ok=True)
        cache_path = os.path.join(output_dir, _CACHE_FILE)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def _get_file_key(self, file_path: str) -> str:
        """生成文件的唯一标识键 (基于绝对路径的 hash)"""
        abs_path = os.path.abspath(file_path)
        return hashlib.md5(abs_path.encode()).hexdigest()[:12]

    def _is_cached(self, file_path: str, cache: Dict[str, Any]) -> Optional[str]:
        """
        检查文件是否已处理且未变更。
        返回已缓存的 doc_id，或 None (需要重新处理)。
        """
        if not self.use_cache:
            return None
        key = self._get_file_key(file_path)
        if key not in cache:
            return None
        entry = cache[key]
        try:
            stat = os.stat(file_path)
            if stat.st_mtime != entry.get("mtime", 0):
                return None  # 文件已修改
            if stat.st_size != entry.get("size", 0):
                return None  # 文件大小已变
        except OSError:
            return None
        # 验证输出的 JSON 确实存在
        doc_id = entry.get("doc_id", "")
        return doc_id if doc_id else None

    def process_file_or_dir(
        self, path: str, output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """处理文件或目录，结果保存为 JSON。启用缓存时跳过已处理的文件。"""
        input_path = Path(path)
        results: List[Dict[str, Any]] = []
        out = output_dir or str(input_path if input_path.is_dir() else input_path.parent)
        cache = self._load_cache(out)
        skipped = 0

        if input_path.is_file():
            cached_id = self._is_cached(str(input_path), cache)
            if cached_id:
                logger.info(f"⏭️  跳过 (已缓存): {input_path.name} → {cached_id}")
                skipped += 1
            else:
                result = self.process_file(str(input_path))
                self._save_result(result, out)
                cache[self._get_file_key(str(input_path))] = self._cache_entry(str(input_path), result)
                results.append(result)

        elif input_path.is_dir():
            md_files = sorted(input_path.rglob("*.md"))
            logger.info(f"扫描到 {len(md_files)} 个 MD 文件")
            for f in md_files:
                try:
                    cached_id = self._is_cached(str(f), cache)
                    if cached_id:
                        logger.info(f"  ⏭️  跳过: {f.name} → {cached_id}")
                        skipped += 1
                        continue
                    result = self.process_file(str(f))
                    self._save_result(result, out)
                    cache[self._get_file_key(str(f))] = self._cache_entry(str(f), result)
                    results.append(result)
                    logger.info(f"  ✅ {f.name} → {result['doc_id']}")
                except Exception as e:
                    logger.error(f"  ❌ {f.name}: {e}")
        else:
            raise FileNotFoundError(f"路径不存在: {path}")

        # 保存缓存
        if self.use_cache:
            self._save_cache(out, cache)

        if skipped:
            logger.info(f"共跳过 {skipped} 个已缓存文件, 处理 {len(results)} 个新文件")
        return results

    def _cache_entry(self, file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """构建缓存条目"""
        try:
            stat = os.stat(file_path)
            return {
                "doc_id": result["doc_id"],
                "title": result.get("title", ""),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "processed_at": result.get("processed_at", ""),
            }
        except OSError:
            return {"doc_id": result["doc_id"], "title": result.get("title", ""), "mtime": 0, "size": 0}

    # ---- 输出 ----

    def _save_result(self, result: Dict[str, Any], output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        doc_id = result["doc_id"]
        out_path = os.path.join(output_dir, f"{doc_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存: {out_path}")
        return out_path

    def _build_result(
        self,
        meta: DocumentMeta,
        sections: List[Section],
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
        rejected_count: int,
        elapsed: float,
        file_path: str,
    ) -> Dict[str, Any]:
        return {
            "doc_id": meta.doc_id,
            "title": meta.title,
            "doc_type": meta.doc_type.value,
            "file_path": file_path,
            "processed_at": meta.upload_time,
            "stats": {
                "sections": len(sections),
                "paragraphs": sum(len(s.paragraphs) for s in sections),
                "sentences": sum(sum(len(p.sentences) for p in s.paragraphs) for s in sections),
                "entities": len(entities),
                "relations": len(relations),
                "rejected": rejected_count,
                "llm_calls": self.llm.stats["call_count"],
                "llm_tokens": self.llm.stats["total_tokens"],
                "elapsed_seconds": round(elapsed, 2),
            },
            "sections": [
                {
                    "section_id": s.section_id,
                    "section_path": s.section_path,
                    "title": s.title,
                    "level": s.level,
                    "reading_guide": s.reading_guide,
                    "paragraphs": [
                        {
                            "paragraph_id": p.paragraph_id,
                            "para_index": p.para_index,
                            "content": p.content,
                            "sentences": [
                                {
                                    "sentence_id": sent.sentence_id,
                                    "text": sent.text,
                                    "location": sent.location.to_dict(),
                                }
                                for sent in p.sentences
                            ],
                        }
                        for p in s.paragraphs
                    ],
                }
                for s in sections
            ],
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "entity_type": e.entity_type,
                    "mention": e.mention,
                    "canonical_name": e.canonical_name,
                    "attributes": e.attributes,
                    "sentence_id": e.sentence_id,
                    "sentence_ids": e.sentence_ids,
                    "source": e.source,
                    "confidence": e.confidence,
                }
                for e in entities
            ],
            "relations": [
                {
                    "relation_id": r.relation_id,
                    "relation_type": r.relation_type,
                    "head_entity_id": r.head_entity_id,
                    "tail_entity_id": r.tail_entity_id,
                    "properties": r.properties,
                    "evidence": r.evidence,
                    "source": r.source,
                    "confidence": r.confidence,
                }
                for r in relations
            ],
            "rejected": [
                {"stage": r["stage"], "reason": r["reason"], "sentence_id": r.get("sentence_id", "")}
                for r in self.post_processor.rejected
            ],
        }

    # ---- 内部处理 ----

    def _process_tables(self, sections: List[Section]) -> List[Dict[str, Any]]:
        all_tables: List[Dict[str, Any]] = []
        for section in sections:
            for para in section.paragraphs:
                table = self.table_processor.detect_and_parse(
                    section.doc_id, section.section_path, para.para_index, para.content
                )
                if table:
                    prev_idx = para.para_index - 1
                    prev_text = ""
                    if 0 <= prev_idx < len(section.paragraphs):
                        prev_text = section.paragraphs[prev_idx].content
                    table_data = {
                        "table": table, "prev_text": prev_text,
                        "doc_id": section.doc_id,
                        "para_sentence_id": (
                            para.sentences[0].sentence_id if para.sentences
                            else para.paragraph_id
                        ),
                    }
                    all_tables.append(table_data)
                    section.tables.append(table_data)
                    ctx = self.table_processor.build_context_for_llm(table, prev_text)
                    section.raw_text += f"\n\n[表格摘要] {ctx[:500]}"
        return all_tables

    def _collect_sentences_and_extract(
        self, sections: List[Section], doc_type: DocType, all_tables: List[Dict[str, Any]]
    ) -> Tuple[List[Sentence], List[ExtractionResult]]:
        """收集所有句子批次，合并小批次，并发执行 LLM 抽取"""
        all_sentences: List[Sentence] = []
        raw_batches: List[List[Dict[str, Any]]] = []

        # ---- 收集段落句子 ----
        for section in sections:
            for para in section.paragraphs:
                if self.long_para_handler.needs_split(para.content):
                    sub_texts = self.long_para_handler.split_with_context(para.content)
                    for sub_idx, sub_text in enumerate(sub_texts):
                        sub_sents = self._make_sub_sentences(para, sub_idx, sub_text)
                        all_sentences.extend(sub_sents)
                        if sub_sents:
                            raw_batches.append([
                                {"sentence_id": s.sentence_id, "text": s.text, "location": s.location.to_dict()}
                                for s in sub_sents
                            ])
                else:
                    all_sentences.extend(para.sentences)
                    if para.sentences:
                        raw_batches.append([
                            {"sentence_id": s.sentence_id, "text": s.text, "location": s.location.to_dict()}
                            for s in para.sentences
                        ])

        # ---- 收集表格 — 整个表格作为一个批次 (而非逐行) ----
        for tbl_data in all_tables:
            table = tbl_data["table"]
            doc_id = tbl_data.get("doc_id", "")
            para_sid = tbl_data.get("para_sentence_id", table.table_id)

            table_batch: List[Dict[str, Any]] = []
            for row_idx, row in enumerate(table.rows):
                row_dict = dict(zip(table.headers, row))
                table_batch.append({
                    "sentence_id": para_sid,
                    "text": f"[表 行{row_idx}] {row_dict}",
                    "location": {
                        "doc_id": doc_id,
                        "section_path": table.section_path,
                        "paragraph_index": table.paragraph_index,
                        "table_id": table.table_id,
                        "row_index": row_idx,
                        "is_table_row": True,
                    },
                })
            if table_batch:
                raw_batches.append(table_batch)

        # ---- 合并小批次 ----
        merged_batches = self._merge_batches(raw_batches)
        logger.info(
            f"[抽取] 原始 {len(raw_batches)} 批次 → 合并为 {len(merged_batches)} 批次 "
            f"({sum(len(b) for b in merged_batches)} 个输入项)"
        )

        # ---- 并发执行 ----
        all_results = asyncio.run(self._extract_batches_async(merged_batches, doc_type))

        return all_sentences, all_results

    def _merge_batches(
        self, batches: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        """将小批次合并到目标大小，减少 LLM 调用次数"""
        merged: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_chars = 0

        for batch in batches:
            batch_chars = sum(len(s.get("text", "")) for s in batch)
            if current and current_chars + batch_chars > self.batch_chars:
                merged.append(current)
                current = []
                current_chars = 0
            current.extend(batch)
            current_chars += batch_chars

        if current:
            merged.append(current)

        return merged

    async def _extract_batches_async(
        self, batches: List[List[Dict[str, Any]]], doc_type: DocType
    ) -> List[ExtractionResult]:
        """并发执行多个批次的 LLM 抽取"""
        sem = asyncio.Semaphore(self.max_concurrent)

        async def extract_one(batch: List[Dict[str, Any]], idx: int) -> List[ExtractionResult]:
            async with sem:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: self.extractor.extract_from_sentences(batch, doc_type)
                )

        tasks = [extract_one(b, i) for i, b in enumerate(batches)]
        logger.info(f"[抽取] 启动 {len(tasks)} 个并发抽取任务 (最大并发: {self.max_concurrent})")

        # 并发执行所有任务
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: List[ExtractionResult] = []
        failed = 0
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"[抽取] 批次 {i} 失败: {result}")
                failed += 1
            else:
                all_results.extend(result)

        if failed:
            logger.warning(f"[抽取] {failed}/{len(batches)} 批次失败")
        logger.info(
            f"[抽取] 并发完成 → {len(all_results)} 个结果 | "
            f"LLM: {self.llm.stats['call_count']} 次, {self.llm.stats['total_tokens']:,} tokens"
        )
        return all_results

    async def _generate_guides_async(
        self, sections: List[Section]
    ) -> Tuple[int, int]:
        """并发生成所有章节的阅读指南"""
        sem = asyncio.Semaphore(self.max_concurrent)
        ok = 0
        fail = 0

        async def generate_one(section: Section) -> bool:
            async with sem:
                try:
                    loop = asyncio.get_event_loop()
                    guide = await loop.run_in_executor(
                        None,
                        lambda: self.guide_gen.generate(section.section_path, section.raw_text),
                    )
                    section.reading_guide = guide
                    return True
                except Exception as e:
                    logger.error(f"阅读指南失败 section={section.section_id}: {e}")
                    return False

        tasks = [generate_one(s) for s in sections]
        results = await asyncio.gather(*tasks)
        ok = sum(1 for r in results if r)
        fail = len(results) - ok
        return ok, fail

    # ---- 统计 ----

    def _log_statistics(
        self,
        sections: List[Section],
        extraction_results: List[ExtractionResult],
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
        rejected_count: int,
        elapsed: float,
    ) -> None:
        """输出详细的文档统计分布"""
        # 收集基础数据
        all_paras = [p for s in sections for p in s.paragraphs]
        all_sents = [sent for s in sections for p in s.paragraphs for sent in p.sentences]

        para_lens = [len(p.content) for p in all_paras]
        sent_lens = [len(sent.text) for sent in all_sents]
        sents_per_para = [len(p.sentences) for p in all_paras]

        # 从 ExtractionResult 按 paragraph_id 聚合实体/关系数 (去重前)
        para_ent_count: Dict[str, int] = {}
        para_rel_count: Dict[str, int] = {}
        for r in extraction_results:
            # sentence_id → paragraph_id (去掉 _sN 或 _subN_s0 后缀)
            import re
            para_id = re.sub(r'_(sub\d+_)?s\d+$', '', r.sentence_id)
            para_ent_count[para_id] = para_ent_count.get(para_id, 0) + len(r.entities)
            para_rel_count[para_id] = para_rel_count.get(para_id, 0) + len(r.relations)

        ent_per_para = list(para_ent_count.values())
        rel_per_para = list(para_rel_count.values())

        # 实体类型分布
        ent_type_dist: Dict[str, int] = {}
        for e in entities:
            ent_type_dist[e.entity_type] = ent_type_dist.get(e.entity_type, 0) + 1

        # 关系类型分布
        rel_type_dist: Dict[str, int] = {}
        for r in relations:
            rel_type_dist[r.relation_type] = rel_type_dist.get(r.relation_type, 0) + 1

        def _bucket(values: List[int], boundaries: List[int]) -> Dict[str, int]:
            """按区间分桶统计"""
            buckets: Dict[str, int] = {}
            counts = [0] * (len(boundaries) + 1)
            for v in values:
                placed = False
                for i, b in enumerate(boundaries):
                    if v < b:
                        counts[i] += 1
                        placed = True
                        break
                if not placed:
                    counts[-1] += 1
            for i, c in enumerate(counts):
                if i == 0:
                    label = f"<{boundaries[0]}"
                elif i == len(boundaries):
                    label = f">={boundaries[-1]}"
                else:
                    label = f"{boundaries[i-1]}-{boundaries[i]-1}"
                buckets[label] = c
            return buckets

        def _percent(n: int, total: int) -> str:
            if total == 0:
                return "0%"
            return f"{n/total*100:.1f}%"

        def _avg_median(values: List[int]) -> Tuple[float, float]:
            if not values:
                return 0.0, 0.0
            avg = sum(values) / len(values)
            s = sorted(values)
            mid = len(s) // 2
            median = s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2
            return round(avg, 1), round(median, 1)

        # 分桶边界
        para_len_bounds = [200, 500, 1000, 2000, 5000]
        sent_len_bounds = [30, 60, 120, 250, 500]
        sents_para_bounds = [1, 3, 5, 10]
        ent_para_bounds = [0, 3, 6, 10, 20]
        rel_para_bounds = [0, 2, 5, 10]

        # 格式化输出
        lines = [
            f"\n{'='*60}",
            f"  文档统计报告",
            f"{'='*60}",
            f"  耗时: {elapsed:.1f}s  被拒: {rejected_count}",
            f"",
            f"  ── 基本数量 ──",
            f"  章节: {len(sections)}  段落: {len(all_paras)}  句子: {len(all_sents)}",
            f"  实体: {len(entities)}  关系: {len(relations)}",
            f"",
            f"  ── 段落长度分布 (chars) ──",
        ]
        pa_avg, pa_med = _avg_median(para_lens)
        lines.append(f"  均值: {pa_avg}  中位数: {pa_med}")
        for label, count in _bucket(para_lens, para_len_bounds).items():
            bar = "█" * max(1, count * 40 // max(1, len(all_paras)))
            lines.append(f"    {label:>8}: {count:>4} ({_percent(count, len(all_paras)):>5}) {bar}")

        lines.append(f"\n  ── 句子长度分布 (chars) ──")
        sa_avg, sa_med = _avg_median(sent_lens)
        lines.append(f"  均值: {sa_avg}  中位数: {sa_med}")
        for label, count in _bucket(sent_lens, sent_len_bounds).items():
            bar = "█" * max(1, count * 40 // max(1, len(all_sents)))
            lines.append(f"    {label:>8}: {count:>4} ({_percent(count, len(all_sents)):>5}) {bar}")

        lines.append(f"\n  ── 每段落句子数分布 ──")
        sp_avg, sp_med = _avg_median(sents_per_para)
        lines.append(f"  均值: {sp_avg}  中位数: {sp_med}")
        for label, count in _bucket(sents_per_para, sents_para_bounds).items():
            bar = "█" * max(1, count * 40 // max(1, len(all_paras)))
            lines.append(f"    {label:>8}: {count:>4} ({_percent(count, len(all_paras)):>5}) {bar}")

        lines.append(f"\n  ── 每段落实体数分布 (去重前) ──")
        if ent_per_para:
            ep_avg, ep_med = _avg_median(ent_per_para)
            lines.append(f"  有实体的段落: {len(ent_per_para)}/{len(all_paras)}  均值: {ep_avg}  中位数: {ep_med}")
            for label, count in _bucket(ent_per_para, ent_para_bounds).items():
                bar = "█" * max(1, count * 40 // max(1, len(ent_per_para)))
                lines.append(f"    {label:>8}: {count:>4} ({_percent(count, len(ent_per_para)):>5}) {bar}")
        else:
            lines.append("  (无实体数据)")

        lines.append(f"\n  ── 每段落关系数分布 (去重前) ──")
        if rel_per_para:
            rp_avg, rp_med = _avg_median(rel_per_para)
            lines.append(f"  有关系的段落: {len(rel_per_para)}/{len(all_paras)}  均值: {rp_avg}  中位数: {rp_med}")
            for label, count in _bucket(rel_per_para, rel_para_bounds).items():
                bar = "█" * max(1, count * 40 // max(1, len(rel_per_para)))
                lines.append(f"    {label:>8}: {count:>4} ({_percent(count, len(rel_per_para)):>5}) {bar}")
        else:
            lines.append("  (无关系数据)")

        lines.append(f"\n  ── 实体类型分布 ──")
        for etype, count in sorted(ent_type_dist.items(), key=lambda x: -x[1]):
            bar = "█" * max(1, count * 30 // max(1, len(entities)))
            lines.append(f"    {etype:<22}: {count:>4} ({_percent(count, len(entities)):>5}) {bar}")

        lines.append(f"\n  ── 关系类型分布 ──")
        for rtype, count in sorted(rel_type_dist.items(), key=lambda x: -x[1]):
            bar = "█" * max(1, count * 30 // max(1, len(relations)))
            lines.append(f"    {rtype:<22}: {count:>4} ({_percent(count, len(relations)):>5}) {bar}")

        lines.append(f"\n{'='*60}\n")
        logger.info("\n".join(lines))

    # ---- 内部辅助 ----

    def _make_sub_sentences(self, para: Paragraph, sub_idx: int, sub_text: str) -> List[Sentence]:
        sent_id = f"{para.paragraph_id}_sub{sub_idx}_s0"
        doc_id = para.paragraph_id.split("#")[0]
        sp = "#".join(para.paragraph_id.split("#")[1:]).rsplit("_p", 1)[0] if "#" in para.paragraph_id else ""
        loc = Location(doc_id=doc_id, section_path=sp, paragraph_index=para.para_index, sentence_index=sub_idx)
        return [Sentence(sentence_id=sent_id, text=sub_text, location=loc)]
