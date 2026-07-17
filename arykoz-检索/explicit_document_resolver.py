#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve explicit document identities before ordinary retrieval.

Stage A.2 separates *identity routing* from first/second truth inference.
Only visible identifiers, titles, product names, issuers and report years are
used.  The resolver never evaluates whether an option is true.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


_EXPLICIT_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<prefix>fc|fin|ins|reg|res)_text_0*(?P<number>\d+)(?![A-Za-z0-9])",
    re.I,
)
_QUOTED_TITLE_RE = re.compile(r"《([^》]{3,160})》")
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})\s*年?")
_RELATIVE_ROLE_RE = re.compile(r"第一份|第二份|第一篇|第二篇|前者|后者|文档一|文档二|文件一|文件二")
_GENERIC_ALIASES = {
    "公司", "发行人", "文档", "文件", "报告", "年度报告", "募集说明书", "保险条款",
    "合同", "办法", "规定", "证券", "银行", "保险", "中国", "集团",
}


def _norm(value: Any) -> str:
    return re.sub(r"[\s\u3000，。；：、,.!?！？:;()（）\[\]【】\"'“”‘’·—\-]", "", str(value or "")).lower()


def _unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


@dataclass
class ExplicitDocumentResolution:
    doc_ids: List[str] = field(default_factory=list)
    ordered_doc_ids: List[str] = field(default_factory=list)
    reference_map: Dict[str, str] = field(default_factory=dict)
    matches: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_references: List[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "none"
    order_semantic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExplicitDocumentResolver:
    """Index-backed resolver for visible document names and identifiers."""

    def __init__(self, runtime_index: Any):
        self.index = runtime_index
        self._alias_rows: Dict[str, List[Tuple[str, str, str]]] = {}
        self._alias_docs: Dict[str, Dict[str, List[str]]] = {}
        self._build_aliases()

    @staticmethod
    def _preference(index: Any, doc_id: str) -> Tuple[int, int, str]:
        meta = index.meta_table[str(doc_id)]
        role = str(getattr(meta, "extraction_variant", "") or "")
        role_rank = {
            "normative_text": 0, "enforcement_decision": 0,
            "legacy_reconstructed": 1, "mineru": 3,
            "revision_explanation": 5,
        }.get(role, 2)
        did = str(doc_id)
        if did.startswith("strict_v3_") or (did.startswith("csrc_") and "_att1" in did):
            source_rank = 0
        elif did.startswith("regulatory_mineru_"):
            source_rank = 3
        elif "_att2" in did or "_att3" in did or "_att4" in did:
            source_rank = 4
        else:
            source_rank = 1
        return role_rank, source_rank, did

    def _values_for_doc(self, doc_id: str) -> List[Tuple[str, str]]:
        meta = self.index.meta_table[doc_id]
        values: List[Tuple[str, str]] = [
            (str(doc_id), "doc_id"),
            (str(getattr(meta, "title", "") or ""), "title"),
            (str(getattr(meta, "issuer", "") or ""), "issuer"),
        ]
        values.extend((str(x), "meta_alias") for x in getattr(meta, "aliases", []) or [])
        identity = dict(getattr(self.index, "doc_identities", {}).get(doc_id, {}) or {})
        for key, value in identity.items():
            if isinstance(value, (list, tuple, set)):
                values.extend((str(x), str(key)) for x in value)
            elif value:
                values.append((str(value), str(key)))
        # Product questions often use the commercial short name while the index
        # stores the full legal product title.  Derive only deterministic suffix
        # removals; do not invent semantic aliases.
        product_values = [v for v, kind in values if kind == "product"]
        for product in product_values:
            short = re.sub(r"（[^）]{1,20}）", "", product)
            short = re.sub(r"(?:专属商业养老保险|商业养老保险|养老年金保险|终身寿险|医疗保险|保险条款)$", "", short).strip()
            if len(_norm(short)) >= 4 and short != product:
                values.append((short, "product_short"))
            compact_short = re.sub(r"(?:住院)?\d+(?:\.\d+)?$", "", short).strip()
            if len(_norm(compact_short)) >= 4 and compact_short not in {short, product}:
                values.append((compact_short, "product_short"))
            if short.startswith("国寿") and len(_norm(short[2:])) >= 4:
                values.append((short[2:], "product_short"))
        return [(v.strip(), kind) for v, kind in values if str(v).strip()]

    def _build_aliases(self) -> None:
        by_domain: Dict[str, Dict[str, List[Tuple[str, str, str]]]] = {}
        for doc_id, meta in self.index.meta_table.items():
            domain = str(getattr(meta, "domain", "") or "")
            domain_map = by_domain.setdefault(domain, {})
            for alias, kind in self._values_for_doc(str(doc_id)):
                norm = _norm(alias)
                if len(norm) < 3 or norm in {_norm(x) for x in _GENERIC_ALIASES}:
                    continue
                domain_map.setdefault(norm, []).append((alias, str(doc_id), kind))
        for domain, mapping in by_domain.items():
            rows: List[Tuple[str, str, str]] = []
            alias_docs: Dict[str, List[str]] = {}
            for norm, triples in mapping.items():
                docs = _unique(row[1] for row in triples)
                alias_docs[norm] = docs
                # Preserve one representative literal/kind per document.
                seen = set()
                for literal, doc_id, kind in triples:
                    key = (doc_id, kind)
                    if key not in seen:
                        seen.add(key)
                        rows.append((literal, doc_id, kind))
            rows.sort(key=lambda row: (-len(_norm(row[0])), self._preference(self.index, row[1])))
            self._alias_rows[domain] = rows
            self._alias_docs[domain] = alias_docs

    def _exact_code_matches(self, text: str, domain: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        alias_docs = self._alias_docs.get(domain, {})
        for match in _EXPLICIT_CODE_RE.finditer(text):
            literal = match.group(0)
            docs = alias_docs.get(_norm(literal), [])
            if not docs:
                # Contract aliases normalize to fc_text_010 in the index.  Also
                # try the canonical textNN form without guessing from rank.
                number = int(match.group("number"))
                candidates = [f"text{number:02d}", f"text{number}"]
                docs = [did for did in candidates if did in self.index.meta_table and str(getattr(self.index.meta_table[did], "domain", "")) == domain]
            if len(docs) == 1:
                rows.append({"literal": literal, "doc_id": docs[0], "kind": "explicit_code", "start": match.start(), "confidence": 1.0})
            else:
                rows.append({"literal": literal, "doc_id": "", "kind": "explicit_code", "start": match.start(), "confidence": 0.0, "candidates": docs})
        return rows

    def _title_matches(self, text: str, domain: str) -> List[Dict[str, Any]]:
        normalized_text = _norm(text)
        rows: List[Dict[str, Any]] = []
        # Quoted law/report titles are stronger than generic issuer aliases.
        quoted = [(m.group(1), m.start()) for m in _QUOTED_TITLE_RE.finditer(text)]
        for literal, start in quoted:
            norm = _norm(literal)
            candidates: List[str] = []
            for alias_norm, docs in self._alias_docs.get(domain, {}).items():
                if norm == alias_norm or (len(norm) >= 6 and (norm in alias_norm or alias_norm in norm)):
                    candidates.extend(docs)
            candidates = _unique(candidates)
            if candidates:
                # One representative per canonical family; prefer normative text.
                by_family: Dict[str, List[str]] = {}
                for did in candidates:
                    family = str(getattr(self.index, "family_by_doc", {}).get(did, did))
                    by_family.setdefault(family, []).append(did)
                chosen = [min(group, key=lambda did: self._preference(self.index, did)) for group in by_family.values()]
                chosen.sort(key=lambda did: self._preference(self.index, did))
                # An exact title should identify one family.  If several
                # unrelated families remain, retain them as unresolved.
                if len(chosen) == 1:
                    rows.append({"literal": literal, "doc_id": chosen[0], "kind": "quoted_title", "start": start, "confidence": 0.99})
                else:
                    rows.append({"literal": literal, "doc_id": "", "kind": "quoted_title", "start": start, "confidence": 0.0, "candidates": chosen})

        # Long, visible product/title aliases.  Option text may name a product;
        # that is safe for routing but never used as first/second order.
        occupied: List[Tuple[int, int]] = []
        for literal, did, kind in self._alias_rows.get(domain, []):
            norm = _norm(literal)
            if len(norm) < 4 or norm not in normalized_text:
                continue
            # Issuer-only aliases shared by multiple annual reports are handled
            # by the year-aware pass below.
            docs_for_alias = self._alias_docs.get(domain, {}).get(norm, [])
            if len(docs_for_alias) != 1 and kind in {"issuer", "aliases"}:
                continue
            start = text.find(literal)
            if start < 0:
                # Normalization may remove spaces/punctuation; use the normalized
                # position only for ordering, not for substring extraction.
                start = normalized_text.find(norm)
            end = start + max(1, len(literal))
            if any(start < b and end > a for a, b in occupied):
                continue
            occupied.append((start, end))
            rows.append({"literal": literal, "doc_id": did, "kind": f"identity:{kind}", "start": max(0, start), "confidence": 0.96 if kind in {"product", "title", "short", "code"} else 0.9})
        return rows

    def _issuer_year_matches(self, text: str, domain: str) -> List[Dict[str, Any]]:
        years_in_order = [(m.group(1), m.start()) for m in _YEAR_RE.finditer(text)]
        if not years_in_order:
            return []
        normalized_text = _norm(text)
        rows: List[Dict[str, Any]] = []
        issuers: Dict[str, List[str]] = {}
        literals: Dict[str, str] = {}
        for did, meta in self.index.meta_table.items():
            if str(getattr(meta, "domain", "")) != domain:
                continue
            issuer = str(getattr(meta, "issuer", "") or "")
            if not issuer:
                identity = dict(getattr(self.index, "doc_identities", {}).get(str(did), {}) or {})
                issuer = str(identity.get("issuer", "") or "")
            norm = _norm(issuer)
            if len(norm) >= 3:
                issuers.setdefault(norm, []).append(str(did))
                literals[norm] = issuer
        for issuer_norm, docs in issuers.items():
            if issuer_norm not in normalized_text:
                continue
            issuer_pos = normalized_text.find(issuer_norm)
            for year, year_pos in years_in_order:
                candidates = []
                for did in docs:
                    meta = self.index.meta_table[did]
                    meta_year = str(getattr(meta, "year", "") or "")
                    title = str(getattr(meta, "title", "") or "")
                    if year == meta_year or year in title or year in did:
                        candidates.append(did)
                candidates = _unique(candidates)
                if len(candidates) == 1:
                    rows.append({
                        "literal": f"{literals[issuer_norm]}{year}年",
                        "doc_id": candidates[0], "kind": "issuer_year",
                        "start": min(issuer_pos, year_pos), "confidence": 0.98,
                    })
        return rows

    def resolve_text(self, text: str, domain: str, *, allow_issuer_year: bool = True) -> ExplicitDocumentResolution:
        raw = str(text or "")
        matches = self._exact_code_matches(raw, domain)
        matches.extend(self._title_matches(raw, domain))
        if allow_issuer_year:
            matches.extend(self._issuer_year_matches(raw, domain))
        # Deduplicate by document, preserving the earliest and strongest visible
        # reference.  Unresolved explicit references remain auditable.
        matches.sort(key=lambda row: (int(row.get("start", 10**9)), -float(row.get("confidence", 0.0)), str(row.get("doc_id", ""))))
        resolved_rows: List[Dict[str, Any]] = []
        unresolved: List[str] = []
        seen_docs = set()
        reference_map: Dict[str, str] = {}
        for row in matches:
            did = str(row.get("doc_id", "") or "")
            literal = str(row.get("literal", "") or "")
            if not did:
                if literal:
                    unresolved.append(literal)
                continue
            if literal:
                reference_map[literal] = did
                reference_map[_norm(literal)] = did
            if did in seen_docs:
                continue
            seen_docs.add(did)
            resolved_rows.append(row)
        docs = [str(row["doc_id"]) for row in resolved_rows]
        unresolved = [literal for literal in unresolved if _norm(literal) not in reference_map]
        confidence = min((float(row.get("confidence", 0.0)) for row in resolved_rows), default=0.0)
        return ExplicitDocumentResolution(
            doc_ids=docs,
            ordered_doc_ids=list(docs),
            reference_map=reference_map,
            matches=resolved_rows,
            unresolved_references=_unique(unresolved),
            confidence=confidence,
            method="explicit_identity" if docs else "none",
            order_semantic=bool(len(docs) >= 2 and _RELATIVE_ROLE_RE.search(raw)),
        )

    def resolve(self, *, question: str, options: Mapping[str, str], domain: str) -> ExplicitDocumentResolution:
        # Question references define semantic mention order.  Option references
        # are used only to widen the hard routing set.
        question_result = self.resolve_text(question, domain)
        option_rows: List[Dict[str, Any]] = []
        option_map: Dict[str, str] = {}
        unresolved: List[str] = list(question_result.unresolved_references)
        seen = set(question_result.doc_ids)
        docs = list(question_result.doc_ids)
        for key in sorted(options):
            result = self.resolve_text(str(options[key]), domain)
            option_map.update(result.reference_map)
            unresolved.extend(result.unresolved_references)
            for row in result.matches:
                did = str(row.get("doc_id", ""))
                if did and did not in seen:
                    seen.add(did)
                    copied = dict(row)
                    copied["option_key"] = str(key)
                    copied["routing_only"] = True
                    option_rows.append(copied)
                    docs.append(did)
        reference_map = dict(question_result.reference_map)
        reference_map.update(option_map)
        matches = list(question_result.matches) + option_rows
        return ExplicitDocumentResolution(
            doc_ids=docs,
            ordered_doc_ids=list(question_result.ordered_doc_ids),
            reference_map=reference_map,
            matches=matches,
            unresolved_references=_unique(unresolved),
            confidence=min([float(x.get("confidence", 0.0)) for x in matches] or [0.0]),
            method="explicit_identity" if docs else "none",
            order_semantic=question_result.order_semantic,
        )


__all__ = ["ExplicitDocumentResolution", "ExplicitDocumentResolver"]
