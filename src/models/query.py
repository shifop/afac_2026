from pydantic import BaseModel
from typing import List, Optional
from src.models.document import Chunk
from src.models.schema import ChannelWeights


class QueryField(BaseModel):
    field_name: str
    value: Optional[str] = None
    target_index_field: str
    match_method: str
    weight: float


class StructuredQuery(BaseModel):
    category_id: str
    fields: List[QueryField]
    filters: Optional[List[List[QueryField]]] = None
    free_text: Optional[str] = None
    channel_weights: ChannelWeights
    entity_limit: int = 10000
    relation_limit: int = 10000


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    prev_chunks: List[Chunk] = []
    next_chunks: List[Chunk] = []
