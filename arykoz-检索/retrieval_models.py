#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified data contracts for AFAC retrieval V2.

These contracts are intentionally independent from the legacy retriever.  The
reasoning layer may keep using its own ClaimSpec/EvidenceSpan classes; the
adapter converts by field name and does not create a second claim parser.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class EvidenceObligation:
    obligation_id: str
    claim_id: str
    obligation_type: str
    required_docs: List[str] = field(default_factory=list)
    required_subjects: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    required_years: List[str] = field(default_factory=list)
    required_values: List[str] = field(default_factory=list)
    min_distinct_docs: int = 1
    require_table_header: bool = False
    require_unit_context: bool = False
    require_full_document_scan: bool = False
    require_exception_context: bool = False
    require_formula_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalPlan:
    question_id: str
    question: str
    options: Dict[str, str]
    domain: str
    answer_format: str
    claims_by_option: Dict[str, List[Any]] = field(default_factory=dict)
    hard_doc_ids: List[str] = field(default_factory=list)
    subject_groups: List[List[str]] = field(default_factory=list)
    year_groups: List[List[str]] = field(default_factory=list)
    obligations: Dict[str, List[EvidenceObligation]] = field(default_factory=dict)
    ordered_doc_ids: List[str] = field(default_factory=list)
    max_doc_rounds: int = 2
    max_evidence_rounds: int = 2


@dataclass
class EvidenceSpanV2:
    evidence_id: str
    doc_id: str
    text: str
    score: float = 0.0
    source: str = "claim_retrieval"
    chunk_id: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    matched_terms: List[str] = field(default_factory=list)
    section_path: List[str] = field(default_factory=list)
    page_hint: Optional[str] = None
    table_id: Optional[str] = None
    row_header: Optional[str] = None
    column_headers: List[str] = field(default_factory=list)
    unit_context: Optional[str] = None
    source_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkHit:
    """Legacy-compatible chunk result consumed by V6.1."""
    doc_id: str
    chunk_id: str
    score: float
    snippet: str
    char_start: int
    char_end: int
    reason: str = ""


@dataclass
class RetrievalResult:
    """Legacy-compatible public return type."""
    doc_ids: List[str]
    doc_scores: Dict[str, float]
    chunks: List[ChunkHit]
    confidence: float
    fallback_level: int


@dataclass
class RetrievalResultV2:
    doc_ids: List[str]
    doc_scores: Dict[str, float]
    doc_reasons: Dict[str, List[str]]
    option_evidence: Dict[str, List[EvidenceSpanV2]]
    claim_evidence: Dict[str, List[EvidenceSpanV2]]
    obligation_status: Dict[str, Dict[str, Any]]
    missing_obligations: List[Dict[str, Any]]
    confidence: float
    fallback_level: int
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_ids": self.doc_ids,
            "doc_scores": self.doc_scores,
            "doc_reasons": self.doc_reasons,
            "option_evidence": {
                key: [span.to_dict() for span in spans]
                for key, spans in self.option_evidence.items()
            },
            "claim_evidence": {
                key: [span.to_dict() for span in spans]
                for key, spans in self.claim_evidence.items()
            },
            "obligation_status": self.obligation_status,
            "missing_obligations": self.missing_obligations,
            "confidence": self.confidence,
            "fallback_level": self.fallback_level,
            "audit": self.audit,
        }

    def to_legacy(self, *, max_chunks: int = 24) -> RetrievalResult:
        """Flatten claim evidence without losing per-document coverage."""
        seen = set()
        chunks: List[ChunkHit] = []
        all_spans: List[EvidenceSpanV2] = []
        for spans in self.claim_evidence.values():
            all_spans.extend(spans)
        all_spans.sort(key=lambda item: (-item.score, item.doc_id, item.char_start or -1))

        # First keep one span from each routed document, then fill by score.
        ordered: List[EvidenceSpanV2] = []
        for did in self.doc_ids:
            hit = next((span for span in all_spans if span.doc_id == did), None)
            if hit is not None:
                ordered.append(hit)
        ordered.extend(all_spans)

        for span in ordered:
            key = (span.doc_id, span.char_start, span.char_end, span.text[:120])
            if key in seen:
                continue
            seen.add(key)
            chunks.append(ChunkHit(
                doc_id=span.doc_id,
                chunk_id=span.chunk_id or span.evidence_id,
                score=float(span.score),
                snippet=span.text[:500],
                char_start=int(span.char_start or 0),
                char_end=int(span.char_end or max(1, len(span.text))),
                reason=",".join(span.matched_terms[:8]) or span.source,
            ))
            if len(chunks) >= max_chunks:
                break
        return RetrievalResult(
            doc_ids=list(self.doc_ids),
            doc_scores=dict(self.doc_scores),
            chunks=chunks,
            confidence=float(self.confidence),
            fallback_level=int(self.fallback_level),
        )


def claim_attr(claim: Any, name: str, default: Any = None) -> Any:
    if isinstance(claim, dict):
        return claim.get(name, default)
    return getattr(claim, name, default)


def unique_strings(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out
