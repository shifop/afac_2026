import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.models.api import (
    RetrieveRequest,
    RetrieveResponse,
    HealthResponse,
    ErrorDetail,
)
from src.models.schema import CategoryConfig, CategorySchema
from src.models.query import StructuredQuery, RetrievedChunk
from src.classifier.classifier import Classifier
from src.extractor.llm_extractor import LLMExtractor, ExtractionError
from src.extractor.orchestrator import ExtractorOrchestrator, QueryBuilder
from src.search.whoosh_searcher import WhooshSearcher
from src.classifier.config_manager import ConfigManager

logger = logging.getLogger(__name__)

router = APIRouter()

classifier: Optional[Classifier] = None
orchestrator: Optional[ExtractorOrchestrator] = None
searcher: Optional[WhooshSearcher] = None
config_manager: Optional[ConfigManager] = None


def init_pipeline(
    _classifier: Classifier,
    _orchestrator: ExtractorOrchestrator,
    _searcher: WhooshSearcher,
    _config_manager: ConfigManager,
):
    global classifier, orchestrator, searcher, config_manager
    classifier = _classifier
    orchestrator = _orchestrator
    searcher = _searcher
    config_manager = _config_manager


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    if classifier is None or searcher is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    question = request.question
    options = request.options or {}

    category_config, schema = classifier.classify(question)

    warning = None
    try:
        extracted_dict = orchestrator.llm_extractor.extract(question, schema)
    except ExtractionError:
        schema = config_manager.get_default_schema()
        extracted_dict = {"query_text": question}
        warning = "LLM提取失败，已降级为全文检索"

    structured_q = QueryBuilder.build(schema, extracted_dict)
    structured_q.free_text = question

    top_k = options.top_k if options else 10
    doc_ids = options.doc_ids if options else None
    results = searcher.search(structured_q, top_k=top_k, doc_ids=doc_ids)

    extracted_fields = {qf.field_name: qf.value for qf in structured_q.fields}

    response = RetrieveResponse(
        request_id=str(uuid.uuid4()),
        category_used=category_config.id,
        category_name=category_config.name,
        extracted_fields=extracted_fields,
        results=results,
        total=len(results),
        returned=len(results),
        warning=warning,
    )
    return response


@router.get("/health", response_model=HealthResponse)
def health():
    index_ready = searcher is not None and searcher.index is not None
    return HealthResponse(status="ok", index_ready=index_ready)