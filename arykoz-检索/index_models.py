#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class IndexManifest:
    schema_version: str = "2.1"
    chunker_version: str = "2.1"
    normalizer_version: str = "1.0"
    created_at: str = ""
    source_hashes: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocMetaV2:
    doc_id: str
    domain: str
    title: str
    filepath: str
    doc_subtype: str = ""
    year: Optional[int] = None
    issuer: Optional[str] = None
    key_entities: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    total_chars: int = 0
    source_hash: str = ""
    document_family_id: str = ""
    version_date: str = ""
    effective_date: str = ""
    extraction_variant: str = ""


@dataclass
class ChunkV2:
    chunk_id: str
    doc_id: str
    chunk_type: str
    title: str
    level: int
    char_start: int
    char_end: int
    text: str
    section_path: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metric_hint: str = ""
    source_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableAtom:
    table_id: str
    doc_id: str
    section_path: List[str]
    table_title: str
    unit_context: str
    row_header: str
    column_headers: List[str]
    cells: Dict[str, str]
    char_start: int
    char_end: int
    raw_text: str


@dataclass
class ValueRecordV2:
    value_id: str
    doc_id: str
    chunk_id: str
    metric: str
    vtype: str
    value_norm: Optional[float]
    unit_norm: str
    value_raw: str
    raw_text: str
    char_pos: int
    time_hint: str = ""
    table_id: str = ""
    row_header: str = ""
    col_header: str = ""


@dataclass
class EntityEntryV2:
    entity: str
    entity_type: str
    domain: str
    doc_ids: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)


@dataclass
class IndexV2:
    manifest: IndexManifest = field(default_factory=IndexManifest)
    meta_table: Dict[str, DocMetaV2] = field(default_factory=dict)
    chunk_index: Dict[str, List[ChunkV2]] = field(default_factory=dict)
    chunk_by_id: Dict[str, ChunkV2] = field(default_factory=dict)
    entity_lexicon: Dict[str, EntityEntryV2] = field(default_factory=dict)
    value_index: List[ValueRecordV2] = field(default_factory=list)
    clause_kw_index: Dict[str, List[Tuple[str, str, float]]] = field(default_factory=dict)
    bm25: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    doc_identities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tables: Dict[str, TableAtom] = field(default_factory=dict)
    structured_fields: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(default_factory=dict)
