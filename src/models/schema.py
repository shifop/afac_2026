from pydantic import BaseModel
from typing import List, Optional, Literal


class CategoryConfigKeywordGroup(BaseModel):
    words: List[str]
    weight: float


class CategoryConfigClassification(BaseModel):
    method: Literal["keyword", "llm", "default"]
    keywords: Optional[List[CategoryConfigKeywordGroup]] = None
    threshold: Optional[float] = None


class CategoryConfig(BaseModel):
    id: str
    name: str
    description: str = ""
    schema_file: str
    classification: CategoryConfigClassification


class SchemaField(BaseModel):
    field_name: str
    field_type: str
    description: str
    target_index_field: str
    match_method: Literal["exact_then_fuzzy", "fuzzy", "substring"]
    weight: float
    required: bool = False


class ChannelWeights(BaseModel):
    entity_channel: float = 0.4
    relation_channel: float = 0.3
    title_channel: float = 0.2
    content_channel: float = 0.1


class CategorySchema(BaseModel):
    category_id: str
    category_name: str
    description: str = ""
    fields: List[SchemaField]
    filters: Optional[List[SchemaField]] = None
    default_channel_weights: ChannelWeights
