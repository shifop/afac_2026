"""治理 Pipeline - 文档结构化抽取全流程"""
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

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
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

        logger.info(f"处理文档: {file_path}")
        content, meta = self.preprocessor.process(file_path, raw_content)
        doc_id = meta.doc_id

        sections = self.splitter.split(doc_id, content)
        all_tables = self._process_tables(sections)
        all_sentences, all_extraction_results = self._collect_sentences_and_extract(
            sections, meta.doc_type, all_tables
        )

        all_extraction_results = self.post_processor.process(all_extraction_results)
        rejected_count = len(self.post_processor.rejected)

        all_entities = [e for r in all_extraction_results for e in r.entities]
        all_relations = [r for res in all_extraction_results for r in res.relations]
        merged_entities, merged_relations, _ = self.entity_dedup.dedup_within_document(
            all_entities, all_relations
        )

        for section in sections:
            try:
                guide = self.guide_gen.generate(section.section_path, section.raw_text)
                section.reading_guide = guide
            except Exception as e:
                logger.error(f"阅读指南失败 section={section.section_id}: {e}")

        elapsed = time.time() - start_time
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
        all_sentences: List[Sentence] = []
        all_results: List[ExtractionResult] = []

        for section in sections:
            for para in section.paragraphs:
                if self.long_para_handler.needs_split(para.content):
                    sub_texts = self.long_para_handler.split_with_context(para.content)
                    for sub_idx, sub_text in enumerate(sub_texts):
                        sub_sents = self._make_sub_sentences(para, sub_idx, sub_text)
                        all_sentences.extend(sub_sents)
                        if sub_sents:
                            batch = [{"sentence_id": s.sentence_id, "text": s.text, "location": s.location.to_dict()} for s in sub_sents]
                            all_results.extend(self.extractor.extract_from_sentences(batch, doc_type))
                else:
                    batch = [{"sentence_id": s.sentence_id, "text": s.text, "location": s.location.to_dict()} for s in para.sentences]
                    all_sentences.extend(para.sentences)
                    if batch:
                        all_results.extend(self.extractor.extract_from_sentences(batch, doc_type))

        for tbl_data in all_tables:
            virtual_sents = tbl_data["virtual_sentences"]
            table = tbl_data["table"]
            prev_text = tbl_data.get("prev_text", "")
            all_sentences.extend(virtual_sents)
            for row_idx, row in enumerate(table.rows):
                row_dict = dict(zip(table.headers, row))
                ctx = self.table_processor.build_context_for_llm(table, prev_text)
                row_input = [{
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
                }]
                all_results.extend(self.extractor.extract_from_sentences(row_input, doc_type))

        return all_sentences, all_results

    def _make_sub_sentences(self, para: Paragraph, sub_idx: int, sub_text: str) -> List[Sentence]:
        sent_id = f"{para.paragraph_id}_sub{sub_idx}_s0"
        doc_id = para.paragraph_id.split("#")[0]
        sp = "#".join(para.paragraph_id.split("#")[1:]).rsplit("_p", 1)[0] if "#" in para.paragraph_id else ""
        loc = Location(doc_id=doc_id, section_path=sp, paragraph_index=para.para_index, sentence_index=sub_idx)
        return [Sentence(sentence_id=sent_id, text=sub_text, location=loc)]
