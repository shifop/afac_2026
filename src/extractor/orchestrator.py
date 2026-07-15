import logging
from typing import Dict, Optional

from src.models.schema import CategorySchema, ChannelWeights
from src.models.query import QueryField, StructuredQuery
from src.extractor.llm_extractor import LLMExtractor, ExtractionError

logger = logging.getLogger(__name__)


class QueryBuilder:
    @staticmethod
    def build(schema: CategorySchema, extracted: Dict[str, Optional[str]], filters: Dict[str, Optional[str]]) -> StructuredQuery:
        fields = []
        for sf in schema.fields:
            val = extracted.get(sf.field_name)
            fields.append(QueryField(
                field_name=sf.field_name,
                value=val,
                target_index_field=sf.target_index_field,
                match_method=sf.match_method,
                weight=sf.weight,
            ))

        filter_list = []
        for filter_ in filters:
            filter_list.append([])
            for sf in schema.filters:
                val = filter_.get(sf.field_name)
                filter_list[-1].append(QueryField(
                    field_name=sf.field_name,
                    value=val,
                    target_index_field=sf.target_index_field,
                    match_method=sf.match_method,
                    weight=sf.weight,
                ))

        has_value = any(qf.value is not None for qf in fields)
        if not has_value:
            fields = [QueryField(
                field_name="query_text",
                value="PLACEHOLDER",
                target_index_field="content",
                match_method="fuzzy",
                weight=1.0,
            )]
            channel_weights = ChannelWeights(
                entity_channel=0.0,
                relation_channel=0.0,
                title_channel=0.0,
                content_channel=1.0,
            )
        else:
            channel_weights = schema.default_channel_weights
        return StructuredQuery(
            category_id=schema.category_id,
            fields=fields,
            filters=filter_list,
            free_text="PLACEHOLDER",
            channel_weights=channel_weights,
        )


class ExtractorOrchestrator:
    def __init__(self, llm_extractor: LLMExtractor):
        self.llm_extractor = llm_extractor

    def extract_with_fallback(self, question: str, schema: CategorySchema) -> Dict[str, Optional[str]]:
        try:
            return self.llm_extractor.extract(question, schema)
        except ExtractionError as e:
            logger.warning(f"LLM提取失败，降级为全文检索: {e}")
            return {"query_text": question}
