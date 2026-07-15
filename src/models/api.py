from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from src.models.query import RetrievedChunk


class RetrieveOptions(BaseModel):
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.01, ge=0.0, le=1.0)
    include_adjacent: bool = True
    adjacent_count: int = Field(default=1, ge=0, le=5)
    doc_ids: Optional[List[str]] = None


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    options: Optional[RetrieveOptions] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = {}


class RetrieveResponse(BaseModel):
    request_id: str
    category_used: str
    category_name: str
    extracted_fields: Dict[str, Optional[str]] = {}
    results: List[RetrievedChunk] = []
    total: int = 0
    returned: int = 0
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    index_ready: bool