from pydantic import BaseModel
from typing import List, Optional, Dict


class Entity(BaseModel):
    name: str
    desc: str = ""
    entity_type: str = ""


class Relation(BaseModel):
    subject: str = ""
    predicate: str
    object: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    section_title: str = ""
    section_path: str = ""
    content: str
    chunk_index: int = 0
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    entities: List[Entity] = []
    relations: List[Relation] = []


class TreeNode(BaseModel):
    title: str
    level: int = 0
    children: List["TreeNode"] = []


class Document(BaseModel):
    doc_id: str
    doc_type: str = ""
    version_ts: int
    chunks: List[Chunk] = []
    doc_tree: List[TreeNode] = []
    meta:Optional[Dict] = None
