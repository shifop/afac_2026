"""核心数据模型定义"""
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class DocType(str, Enum):
    ANNUAL_REPORT = "年报"
    PROSPECTUS = "募集说明书"
    INSURANCE_CONTRACT = "保险合同"
    FINANCIAL_REGULATION = "金融法规"
    RESEARCH_REPORT = "行业研报"
    UNKNOWN = "unknown"


ENTITY_TYPES = [
    "Organization", "Person", "MonetaryAmount", "Percentage",
    "Date", "FinancialMetric", "LegalDocument", "Clause",
    "InsuranceProduct", "InsuranceCoverage", "Exclusion",
    "Industry", "Rating", "LegalTerm", "Report",
]

RELATION_TYPES = [
    "has_director", "has_shareholder", "reports_financial",
    "has_credit_rating", "references_law", "defines_term",
    "covers_risk", "excludes_risk", "analyzes_industry", "gives_opinion",
]

# 关系类型约束: relation_type -> (head_type, tail_type)
RELATION_CONSTRAINTS: Dict[str, tuple] = {
    "has_director": ("Organization", "Person"),
    "has_shareholder": ("Organization", ("Organization", "Person")),
    "reports_financial": ("Organization", "FinancialMetric"),
    "has_credit_rating": ("Organization", "Rating"),
    "references_law": ("Clause", "LegalDocument"),
    "defines_term": ("Clause", "LegalTerm"),
    "covers_risk": ("InsuranceProduct", "InsuranceCoverage"),
    "excludes_risk": ("InsuranceProduct", "Exclusion"),
    "analyzes_industry": ("Report", "Industry"),
    "gives_opinion": ("Person", ("Organization", "Industry")),
}


@dataclass
class Location:
    """句子/单元格定位信息"""
    doc_id: str
    section_path: str = ""
    paragraph_index: int = -1
    sentence_index: int = -1
    table_id: Optional[str] = None
    row_index: Optional[int] = None
    column_index: Optional[int] = None
    is_cell: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "doc_id": self.doc_id,
            "section_path": self.section_path,
            "paragraph_index": self.paragraph_index,
            "sentence_index": self.sentence_index,
        }
        if self.is_cell:
            d["table_id"] = self.table_id
            d["row_index"] = self.row_index
            d["column_index"] = self.column_index
            d["is_cell"] = True
        return d


@dataclass
class Sentence:
    """句子对象"""
    sentence_id: str
    text: str
    location: Location


@dataclass
class Paragraph:
    """段落对象"""
    paragraph_id: str
    section_id: str
    para_index: int
    content: str
    sentences: List[Sentence] = field(default_factory=list)


@dataclass
class Section:
    """章节对象"""
    section_id: str
    doc_id: str
    section_path: str
    title: str
    level: int
    para_start: int
    para_end: int
    raw_text: str = ""
    reading_guide: Optional[Dict[str, Any]] = None
    paragraphs: List[Paragraph] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TableInfo:
    """表格信息"""
    table_id: str
    headers: List[str]
    rows: List[List[str]]
    section_path: str
    paragraph_index: int
    caption: str = ""
    summary: str = ""


@dataclass
class ExtractedEntity:
    """抽取的实体"""
    entity_id: str  # 临时ID
    entity_type: str
    mention: str
    canonical_name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    sentence_id: str = ""
    sentence_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "auto"  # auto / auto_fixed / dual_model


@dataclass
class ExtractedRelation:
    """抽取的关系"""
    relation_id: str
    relation_type: str
    head_entity_id: str
    tail_entity_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "auto"


@dataclass
class ExtractionResult:
    """抽取结果"""
    sentence_id: str
    text: str
    location: Location
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    raw_llm_response: Optional[str] = None


@dataclass
class DocumentMeta:
    """文档元数据"""
    doc_id: str
    title: str
    doc_type: DocType
    file_path: str = ""
    upload_time: str = ""
    status: str = "processing"  # processing / completed / error
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured_data: Dict[str, Any] = field(default_factory=dict)
