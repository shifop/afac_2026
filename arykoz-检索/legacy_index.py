#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe adapter for the complete legacy BIndex pickle and IndexV2."""
from __future__ import annotations

import os
import pickle
import sys
import types
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from text_utils import canonical_title, stable_hash, tokenize


@dataclass
class DocMeta:
    doc_id: str
    domain: str
    title: str
    filepath: str
    doc_subtype: str = ""
    key_entities: List[str] = field(default_factory=list)
    year: Optional[int] = None
    issuer: Optional[str] = None
    summary: str = ""
    total_chars: int = 0
    aliases: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_type: str
    title: str
    level: int
    char_start: int
    char_end: int
    text: str
    entities: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metric_hint: str = ""


@dataclass
class ValueRecord:
    value_id: str
    doc_id: str
    chunk_id: str
    metric: str
    vtype: str
    value_norm: Optional[float]
    unit_norm: str
    value_raw: str
    raw_text: str
    line_start: int
    char_pos: int


@dataclass
class EntityEntry:
    entity: str
    entity_type: str
    domain: str
    doc_ids: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)


@dataclass
class BIndex:
    meta_table: Dict[str, DocMeta] = field(default_factory=dict)
    chunk_index: Dict[str, List[Chunk]] = field(default_factory=dict)
    entity_lexicon: Dict[str, EntityEntry] = field(default_factory=dict)
    value_index: List[ValueRecord] = field(default_factory=list)
    clause_kw_index: Dict[str, List[Tuple[str, str, float]]] = field(default_factory=dict)
    bm25: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    chunk_by_id: Dict[str, Chunk] = field(default_factory=dict)
    doc_identities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    llm_doc_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    llm_section_annotations: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _install_pickle_aliases() -> None:
    module = sys.modules.get("build_b_index")
    if module is None:
        module = types.ModuleType("build_b_index")
        sys.modules["build_b_index"] = module
    for name in ("BIndex", "DocMeta", "Chunk", "EntityEntry", "ValueRecord"):
        if not hasattr(module, name):
            setattr(module, name, globals()[name])
    import __main__
    for name in ("BIndex", "DocMeta", "Chunk", "EntityEntry", "ValueRecord"):
        if not hasattr(__main__, name):
            setattr(__main__, name, globals()[name])


def load_index_object(path: str) -> Any:
    _install_pickle_aliases()
    with open(path, "rb") as handle:
        return pickle.load(handle)


class RuntimeIndex:
    """Normalizes legacy BIndex and the new IndexV2 to one runtime API."""
    def __init__(self, raw: Any, *, index_path: str = ""):
        self.raw = raw
        self.index_path = index_path
        self.meta_table = dict(getattr(raw, "meta_table", {}) or getattr(raw, "docs", {}) or {})
        self.chunk_index = dict(getattr(raw, "chunk_index", {}) or {})
        if not self.chunk_index and hasattr(raw, "chunks"):
            grouped: Dict[str, List[Any]] = defaultdict(list)
            for chunk in getattr(raw, "chunks", {}).values():
                grouped[str(chunk.doc_id)].append(chunk)
            self.chunk_index = dict(grouped)
        self.chunk_by_id = dict(getattr(raw, "chunk_by_id", {}) or {})
        if not self.chunk_by_id:
            self.chunk_by_id = {
                str(chunk.chunk_id): chunk
                for chunks in self.chunk_index.values() for chunk in chunks
            }
        self.entity_lexicon = dict(getattr(raw, "entity_lexicon", {}) or {})
        self.value_index = list(getattr(raw, "value_index", []) or [])
        self.clause_kw_index = dict(getattr(raw, "clause_kw_index", {}) or {})
        self.bm25 = dict(getattr(raw, "bm25", {}) or {})
        self.doc_identities = dict(getattr(raw, "doc_identities", {}) or {})
        self.tables = dict(getattr(raw, "tables", {}) or {})
        self.tables_by_doc: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
        for key, table in self.tables.items():
            self.tables_by_doc[str(getattr(table, "doc_id", "") or "")].append((str(key), table))
        self.structured_fields = dict(getattr(raw, "structured_fields", {}) or {})
        self.manifest = getattr(raw, "manifest", None)
        self.schema_version = str(getattr(self.manifest, "schema_version", "legacy-v1"))

        self.values_by_doc: Dict[str, List[Any]] = defaultdict(list)
        self.values_by_chunk: Dict[str, List[Any]] = defaultdict(list)
        for value in self.value_index:
            self.values_by_doc[str(value.doc_id)].append(value)
            self.values_by_chunk[str(value.chunk_id)].append(value)

        self.metrics_by_doc: Dict[str, set] = defaultdict(set)
        for did, chunks in self.chunk_index.items():
            for chunk in chunks:
                hint = str(getattr(chunk, "metric_hint", "") or "")
                self.metrics_by_doc[str(did)].update(x for x in hint.split(",") if x)
                self.metrics_by_doc[str(did)].update(getattr(chunk, "keywords", []) or [])
        for value in self.value_index:
            if getattr(value, "metric", ""):
                self.metrics_by_doc[str(value.doc_id)].add(str(value.metric))

        self.family_by_doc: Dict[str, str] = {}
        for did, meta in self.meta_table.items():
            family = str(getattr(meta, "document_family_id", "") or "")
            if not family:
                family = canonical_title(str(getattr(meta, "title", "") or did)) or str(did)
            self.family_by_doc[str(did)] = family

        self._doc_text_cache: Dict[str, str] = {}
        self._doc_token_cache: Dict[str, List[str]] = {}
        self._vocabulary = self._build_vocabulary()

    @classmethod
    def load(cls, path: str) -> "RuntimeIndex":
        return cls(load_index_object(path), index_path=path)

    def _build_vocabulary(self) -> List[str]:
        values = set()
        for name, entry in self.entity_lexicon.items():
            values.add(str(name))
            values.update(str(x) for x in getattr(entry, "aliases", []) or [])
        for identity in self.doc_identities.values():
            for value in identity.values():
                if isinstance(value, list):
                    values.update(str(x) for x in value)
                elif value:
                    values.add(str(value))
        for metrics in self.metrics_by_doc.values():
            values.update(str(x) for x in metrics)
        return sorted((x for x in values if len(x) >= 2), key=len, reverse=True)

    def all_doc_ids(self, domain: Optional[str] = None) -> List[str]:
        if not domain:
            return list(self.meta_table.keys())
        return [str(did) for did, meta in self.meta_table.items() if str(getattr(meta, "domain", "")) == domain]

    def get_chunks(self, doc_id: str) -> List[Any]:
        return list(self.chunk_index.get(str(doc_id), []) or [])

    def get_doc_text(self, doc_id: str) -> str:
        did = str(doc_id)
        if did in self._doc_text_cache:
            return self._doc_text_cache[did]
        meta = self.meta_table.get(did)
        path = str(getattr(meta, "filepath", "") or "") if meta else ""
        text = ""
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
            except OSError:
                text = ""
        if not text:
            # Legacy chunks can overlap. Reconstruct by char offsets when valid.
            chunks = sorted(self.get_chunks(did), key=lambda item: (int(getattr(item, "char_start", 0)), int(getattr(item, "char_end", 0))))
            max_end = max((int(getattr(item, "char_end", 0) or 0) for item in chunks), default=0)
            if 0 < max_end <= 5_000_000:
                chars = [" "] * max_end
                for chunk in chunks:
                    start = int(getattr(chunk, "char_start", 0) or 0)
                    body = str(getattr(chunk, "text", "") or "")
                    end = min(max_end, start + len(body))
                    if start < 0 or start >= end:
                        continue
                    chars[start:end] = body[:end - start]
                text = "".join(chars).rstrip()
            else:
                text = "\n\n".join(str(getattr(item, "text", "") or "") for item in chunks)
        self._doc_text_cache[did] = text
        return text

    def get_doc_tokens(self, doc_id: str) -> List[str]:
        did = str(doc_id)
        if did not in self._doc_token_cache:
            self._doc_token_cache[did] = tokenize(self.get_doc_text(did), self._vocabulary)
        return self._doc_token_cache[did]

    def source_hash(self, doc_id: str) -> str:
        meta = self.meta_table.get(str(doc_id))
        value = str(getattr(meta, "source_hash", "") or "") if meta else ""
        return value or stable_hash(self.get_doc_text(str(doc_id)))
