"""治理 Pipeline - 文档结构化抽取全流程"""
import asyncio
import json
import time
import os
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple
import logging; logger = logging.getLogger(__name__)

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
    ) -> None:
        self.llm = llm_client
        self.batch_chars = batch_chars          # 合并后每批目标字符数
        self.max_concurrent = max_concurrent    # 并发 LLM 调用上限
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

    def process_file_or_dir(
        self, path: str, output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """处理文件或目录，结果保存为 JSON"""
        input_path = Path(path)
        results: List[Dict[str, Any]] = []

        if input_path.is_file():
            result = self.process_file(str(input_path))
            self._save_result(result, output_dir or str(input_path.parent))
            results.append(result)

        elif input_path.is_dir():
            md_files = sorted(input_path.rglob("*.md"))
            logger.info(f"扫描到 {len(md_files)} 个 MD 文件")
            out = output_dir or str(input_path)
            for f in md_files:
                try:
                    result = self.process_file(str(f))
                    self._save_result(result, out)
                    results.append(result)
                    logger.info(f"  ✅ {f.name} → {result['doc_id']}")
                except Exception as e:
                    logger.error(f"  ❌ {f.name}: {e}")
        else:
            raise FileNotFoundError(f"路径不存在: {path}")

        return results

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
                    virtual_sents = self.table_processor.to_virtual_sentences(
                        section.doc_id, section.section_path, para.para_index, table
                    )
                    table_data = {"table": table, "virtual_sentences": virtual_sents, "prev_text": prev_text}
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
            virtual_sents = tbl_data["virtual_sentences"]
            table = tbl_data["table"]
            prev_text = tbl_data.get("prev_text", "")
            all_sentences.extend(virtual_sents)

            ctx = self.table_processor.build_context_for_llm(table, prev_text)
            table_batch: List[Dict[str, Any]] = []
            for row_idx, row in enumerate(table.rows):
                row_dict = dict(zip(table.headers, row))
                table_batch.append({
                    "sentence_id": f"{table.table_id}_row{row_idx}",
                    "text": f"表格上下文：{ctx[:500]}\n当前行数据：{row_dict}",
                    "location": {
                        "doc_id": virtual_sents[0].location.doc_id if virtual_sents else "",
                        "section_path": table.section_path,
                        "paragraph_index": table.paragraph_index,
                        "table_id": table.table_id,
                        "row_index": row_idx,
                        "is_cell": True,
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

    def _make_sub_sentences(self, para: Paragraph, sub_idx: int, sub_text: str) -> List[Sentence]:
        sent_id = f"{para.paragraph_id}_sub{sub_idx}_s0"
        doc_id = para.paragraph_id.split("#")[0]
        sp = "#".join(para.paragraph_id.split("#")[1:]).rsplit("_p", 1)[0] if "#" in para.paragraph_id else ""
        loc = Location(doc_id=doc_id, section_path=sp, paragraph_index=para.para_index, sentence_index=sub_idx)
        return [Sentence(sentence_id=sent_id, text=sub_text, location=loc)]
