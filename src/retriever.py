import os
import json
import logging
from typing import Dict, List, Optional, Any
from copy import deepcopy

import yaml

from src.models.document import Document, Chunk
from src.models.schema import CategorySchema, CategoryConfig
from src.models.query import StructuredQuery, RetrievedChunk
from src.classifier.config_manager import ConfigManager
from src.classifier.classifier import Classifier
from src.extractor.llm_extractor import LLMExtractor, ExtractionError
from src.extractor.orchestrator import ExtractorOrchestrator, QueryBuilder
from src.search.whoosh_searcher import WhooshSearcher
from src.indexer.index_manager import IndexManager
from src.indexer.insurance_index_manager import InsuranceIndexManager

logger = logging.getLogger(__name__)

INDEX_MANAGER_MAP = {
    "default":IndexManager,
    "insurance":InsuranceIndexManager
}

class Retriever:
    def __init__(
        self,
        config_dir: str = "config",
        documents_dir: str = "./data/documents",
        index_dir: str = "./data/index",
        force_rebuild: bool = False,
        llm_api_base: str = "http://localhost:1234/v1",
        llm_api_key: str = "not-needed",
        llm_model: str = "qwen2.5-7b-instruct",
        llm_timeout: int = 30,
        llm_max_retries: int = 2,
        llm_temperature: float = 0.0,
        context_window: Optional[Dict[str, int]] = None,
        index_manager_type:str="default"
    ):
        self.config_manager = ConfigManager(config_dir)
        self.config_manager.load_all()

        self.index_manager = INDEX_MANAGER_MAP.get(index_manager_type, IndexManager)(
            index_dir=index_dir,
            documents_dir=documents_dir,
            force_rebuild=force_rebuild,
        )
        self.index_manager.initialize()

        self.classifier = Classifier(self.config_manager)

        self.llm_extractor = LLMExtractor(
            api_base=llm_api_base,
            api_key=llm_api_key,
            model=llm_model,
            temperature=llm_temperature,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
        )

        self.orchestrator = ExtractorOrchestrator(self.llm_extractor)

        self.searcher = WhooshSearcher(
            index=self.index_manager,
            chunk_dict=self.index_manager.chunk_dict,
            context_window=context_window or {"prev": 1, "next": 1},
        )

        logger.info("Retriever 初始化完成")

    def retrieve(self, question: str, top_k: int = 10,
                 doc_ids: Optional[List[str]] = None, category_name: Optional[str] = None, filter_by_llm=True) -> Dict[str, Any]:
        
        if category_name:
            category_config, schema = self.classifier.get_category_by_name(category_name)
        else:
            category_config, schema = self.classifier.classify(question)

        warning = None

        # 构建过滤参数
        try:
            schema_copy = deepcopy(schema)
            schema_copy.fields = schema_copy.filters
            filter_list = self.llm_extractor.extract_filter(question, schema_copy)
        except ExtractionError:
            schema = self.config_manager.get_default_schema()
            filter_list = [{"contract_name": question}, doc_ids]
            warning = "LLM提取失败，已降级为全文检索"

        # 先确认检索范围
        candidate_docs = {}

        for filter_,_ in filter_list:
            structured_q = QueryBuilder.build(schema, filter_, [filter_])
            structured_q.free_text = question
            result = self.searcher.filter(structured_q, 3, 2)
            for doc in result:
                candidate_docs[doc['doc_id']] = doc

        # 使用大模型确认检索的id
        if filter_by_llm:
            try:
                doc_ids_ = self.llm_extractor.filter(question, candidate_docs)
            except ExtractionError:
                doc_ids_ = [k for k in candidate_docs]
                warning = "LLM提取失败，已降级为全文检索"
        else:
            doc_ids_ = [k for k in candidate_docs]

        # 构建查询参数
        try:
            extracted_list = self.llm_extractor.extract(question, schema)
        except ExtractionError:
            schema = self.config_manager.get_default_schema()
            extracted_list = [[{"query_text": question}, doc_ids]]
            warning = "LLM提取失败，已降级为全文检索"

        result = []
        for [extracted_dict, _] in extracted_list:
            structured_q = QueryBuilder.build(schema, extracted_dict, [_[0] for _ in filter_list])
            structured_q.free_text = question
            for ids in doc_ids_:
                results = self.searcher.search(structured_q, top_k=top_k, doc_ids=[ids])

                extracted_fields = {qf.field_name: qf.value for qf in structured_q.fields}
                result.append({
                    "question": question,
                    "category_id": category_config.id,
                    "category_name": category_config.name,
                    "extracted_fields": extracted_fields,
                    "results": results,
                    "total": len(results),
                    "warning": warning,
                })
        return result, extracted_list

    def classify(self, question: str) -> Dict[str, Any]:
        category_config, schema = self.classifier.classify(question)
        return {
            "category_id": category_config.id,
            "category_name": category_config.name,
            "schema": schema,
        }

    def extract(self, question: str, category_id: Optional[str] = None) -> Dict[str, Any]:
        if category_id:
            schema = self.config_manager.get_schema(category_id)
            if schema is None:
                raise ValueError(f"未找到类别: {category_id}")
        else:
            _, schema = self.classifier.classify(question)

        extracted_dict = self.llm_extractor.extract(question, schema)
        return {
            "category_id": schema.category_id,
            "extracted_fields": extracted_dict,
        }

    def search_only(self, question: str, top_k: int = 10,
                    doc_ids: Optional[List[str]] = None) -> List[RetrievedChunk]:
        category_config, schema = self.classifier.classify(question)

        try:
            extracted_dict = self.llm_extractor.extract(question, schema)
        except ExtractionError:
            schema = self.config_manager.get_default_schema()
            extracted_dict = {"query_text": question}

        structured_q = QueryBuilder.build(schema, extracted_dict, question=question)
        structured_q.free_text = question

        return self.searcher.search(structured_q, top_k=top_k, doc_ids=doc_ids)

    def rebuild_index(self):
        self.index_manager.rebuild_index()
        self.searcher.chunk_dict = self.index_manager.chunk_dict
        self.searcher.index = self.index_manager.index
        logger.info("索引重建完成")
