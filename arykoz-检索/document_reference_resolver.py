#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve relative document references without using retrieval rank as order.

AFAC blind questions may physically omit ``doc_ids`` while options still refer to
"第一份文档/第二份文档/前者/后者".  A flat retriever can find the right
corpus documents but cannot safely interpret those ordinal roles.  This module
adds a bounded, auditable resolution stage:

1. authoritative input order, when the question actually provides it;
2. deterministic constraint scoring over retrieved evidence and document
   identity cards;
3. optional Qwen-only resolution when deterministic evidence is ambiguous;
4. explicit abstention when neither path can uniquely bind both roles.

The resolver never uses qid, answer labels, ground truth, or the retrieval list
position as the meaning of first/second.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_FIRST_RE = re.compile(r"第一份|第一篇|前一份|前者|文档一|文件一")
_SECOND_RE = re.compile(r"第二份|第二篇|后一份|后者|文档二|文件二")
_PAIR_RE = re.compile(r"两份文档|两篇文档|两份文件|两篇文件|二者|两者|双方")
_SEPARATE_DOC_RE = re.compile(r"(?:文档|文件|报告|条款).{0,10}分别|分别.{0,10}(?:文档|文件|报告|条款)")
_COMPANY_RE = re.compile(
    r"[A-Za-z0-9\u4e00-\u9fff（）()·—\-]{3,60}?(?:股份有限公司|有限责任公司|集团有限公司|有限公司|证券股份有限公司|保险股份有限公司)"
)
_RATING_RE = re.compile(r"(?<![A-Z])(?:AAA|AA\+|AA-|AA|A\+|A-|BBB\+|BBB-|BBB)(?![A-Z])", re.I)
_CODE_RE = re.compile(r"(?<!\d)(?:[036]\d{5})(?!\d)")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(-?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|元/股|元|%|％|天|日|年|个月|月|倍)?")

_TRANSACTION_TERMS = (
    "发行股份购买资产", "发行股份及支付现金购买资产", "募集配套资金",
    "重大资产重组", "关联交易报告书", "可转换公司债券", "公司债券募集说明书",
)
_ROLE_PREDICATE_NOISE = {
    "均为", "都是", "高于", "低于", "大于", "小于", "超过", "不超过", "至少", "至多",
    "正确", "符合", "详细列出", "明确给出",
}


def _extract_identity_anchors(raw: str, claim: Any) -> List[Tuple[str, float]]:
    """Extract role-binding anchors, excluding generic truth predicates.

    Role-specific absolute facts (named entity, code, transaction type, exact
    field/value) may identify a document. Universal and comparative truth
    predicates are not identity evidence by themselves.
    """
    text = str(raw or "")
    values: List[Tuple[str, float]] = []
    for item in _COMPANY_RE.findall(text):
        cleaned = re.sub(r"^.*?(?:发行人(?:名称)?|主体|公司)(?:为|是|：|:)?", "", item).strip()
        values.append((cleaned or item, 10.0))
    for item in _CODE_RE.findall(text):
        values.append((item, 9.0))
    for term in _TRANSACTION_TERMS:
        if term in text:
            values.append((term, 7.0))
    # Exact field/value is allowed only as a role-specific identity constraint.
    fields = [str(x) for x in (_get(claim, "field_candidates", []) or []) if str(x)]
    if fields:
        for number, unit in _NUMBER_RE.findall(text):
            token = f"{number}{unit}" if unit else number
            if unit or "." in number:
                values.append((token, 6.0))
        for field in fields:
            if len(field) >= 2 and field not in _GENERIC:
                values.append((field, 3.5))
    for item in _get(claim, "subject_candidates", []) or []:
        item = str(item or "").strip()
        if len(item) >= 4 and item not in _GENERIC and not any(noise in item for noise in _ROLE_PREDICATE_NOISE):
            cleaned = _FIRST_RE.sub("", _SECOND_RE.sub("", item)).strip("中的是为：: ")
            if cleaned:
                values.append((cleaned, 2.5))
    by_norm: Dict[str, Tuple[str, float]] = {}
    for value, weight in values:
        key = _norm(value)
        if len(key) < 2:
            continue
        if key not in by_norm or weight > by_norm[key][1]:
            by_norm[key] = (value, weight)
    return sorted(by_norm.values(), key=lambda x: (-x[1], -len(x[0])))[:14]


def _usage_from_response(response: Any) -> Dict[str, int]:
    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }
    if not usage["total_tokens"]:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage


def _merge_usage(left: Mapping[str, int], right: Mapping[str, int]) -> Dict[str, int]:
    out = {key: int(left.get(key, 0) or 0) + int(right.get(key, 0) or 0)
           for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    if not out["total_tokens"]:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def _parse_json_object(content: str) -> Dict[str, Any]:
    raw = str(content or "").strip()
    raw = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", raw, flags=re.I | re.S).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start:end + 1])
    if not isinstance(data, Mapping):
        raise ValueError("binding response must be a JSON object")
    return dict(data)

_GENERIC = {
    "第一份文档", "第二份文档", "第一篇文档", "第二篇文档", "前者", "后者",
    "两份文档", "两篇文档", "两份文件", "两篇文件", "文档", "文件", "两者", "二者",
    "明确", "显示", "表明", "涉及", "包含", "提及", "规定", "属于", "均为", "都是",
    "第一份", "第二份", "第一篇", "第二篇", "前一份", "后一份",
}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _norm(text: Any) -> str:
    return re.sub(r"[\s\u3000，。；：、,.!?！？:;()（）\[\]【】\"'“”‘’]", "", str(text or "")).lower()


def _unique(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _role_mentions(text: str) -> Tuple[bool, bool, bool]:
    raw = str(text or "")
    first = bool(_FIRST_RE.search(raw))
    second = bool(_SECOND_RE.search(raw))
    pair = bool(_PAIR_RE.search(raw) or _SEPARATE_DOC_RE.search(raw) or (first and second))
    return first, second, pair


def _extract_anchors(raw: str, claim: Any) -> List[Tuple[str, float]]:
    """Return lexical anchors with conservative specificity weights."""
    values: List[Tuple[str, float]] = []
    text = str(raw or "")
    for item in _COMPANY_RE.findall(text):
        values.append((item, 6.0))
    for item in _RATING_RE.findall(text):
        values.append((item.upper(), 3.5))
    for item in _CODE_RE.findall(text):
        values.append((item, 5.0))
    for number, unit in _NUMBER_RE.findall(text):
        token = f"{number}{unit}" if unit else number
        # Bare small integers are often ordinals/page numbers and are weak.
        values.append((token, 3.0 if unit else 1.0))

    for item in _get(claim, "subject_candidates", []) or []:
        item = str(item or "").strip()
        if len(item) >= 3 and item not in _GENERIC and not _FIRST_RE.fullmatch(item) and not _SECOND_RE.fullmatch(item):
            values.append((item, 4.0 if len(item) >= 6 else 2.0))
    for item in _get(claim, "field_candidates", []) or []:
        item = str(item or "").strip()
        if len(item) >= 2 and item not in _GENERIC:
            values.append((item, 2.0))
    for item in _get(claim, "keywords", []) or []:
        item = str(item or "").strip()
        if len(item) >= 3 and item not in _GENERIC and not re.fullmatch(r"(?:第一|第二|前|后).{0,3}文档", item):
            values.append((item, 0.7))

    # Preserve the strongest weight for duplicated normalized anchors.
    by_norm: Dict[str, Tuple[str, float]] = {}
    for text_value, weight in values:
        key = _norm(text_value)
        if len(key) < 2:
            continue
        if key not in by_norm or weight > by_norm[key][1]:
            by_norm[key] = (text_value, weight)
    return sorted(by_norm.values(), key=lambda item: (-item[1], -len(item[0])))[:18]


def _span_dict(span: Any) -> Dict[str, Any]:
    if isinstance(span, Mapping):
        return dict(span)
    if hasattr(span, "to_dict"):
        return dict(span.to_dict())
    return {
        "evidence_id": _get(span, "evidence_id", ""),
        "doc_id": _get(span, "doc_id", ""),
        "text": _get(span, "text", ""),
        "score": _get(span, "score", 0.0),
        "source": _get(span, "source", ""),
        "metadata": dict(_get(span, "metadata", {}) or {}),
    }


def _normalized_number(raw_value: str, unit: str) -> Optional[Tuple[Decimal, str]]:
    try:
        value = Decimal(str(raw_value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    u = str(unit or "").replace("％", "%")
    if u == "亿元":
        return value * Decimal("100000000"), "元"
    if u == "万元":
        return value * Decimal("10000"), "元"
    if u in {"日", "天"}:
        return value, "天"
    if u == "月":
        return value, "月"
    if u == "个月":
        return value, "月"
    return value, u


def _numbers_from_span(span: Mapping[str, Any]) -> List[Tuple[Decimal, str]]:
    metadata = dict(span.get("metadata", {}) or {})
    # Structured values/table cells are the adjudicable scope.  When they
    # contain numbers, do not contaminate them with page numbers or unrelated
    # figures from the surrounding raw text.
    structured_texts = [
        str(metadata.get("value", "")),
        str(metadata.get("structured_value", "")),
        str(metadata.get("cell_value", "")),
    ]
    structured: List[Tuple[Decimal, str]] = []
    for text in structured_texts:
        for raw, unit in _NUMBER_RE.findall(text):
            parsed = _normalized_number(raw, unit)
            if parsed and parsed not in structured:
                structured.append(parsed)
    if structured:
        return structured[:8]
    out: List[Tuple[Decimal, str]] = []
    for raw, unit in _NUMBER_RE.findall(str(span.get("text", ""))):
        parsed = _normalized_number(raw, unit)
        if parsed and parsed not in out:
            out.append(parsed)
    return out[:8]


@dataclass
class DocumentBinding:
    needed: bool
    status: str = "NOT_REQUIRED"  # NOT_REQUIRED/AUTHORITATIVE/RESOLVED/PARTIAL/UNRESOLVED
    role_to_doc_id: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    method: str = "none"
    order_semantic: bool = True
    reason: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    candidate_pairs: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_roles: List[str] = field(default_factory=list)
    llm_calls: int = 0
    retry_count: int = 0
    warnings: List[str] = field(default_factory=list)
    identity_support: Dict[str, List[str]] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    })

    @property
    def resolved(self) -> bool:
        return bool(
            self.status in {"AUTHORITATIVE", "RESOLVED"}
            and self.role_to_doc_id.get("first")
            and self.role_to_doc_id.get("second")
            and self.role_to_doc_id.get("first") != self.role_to_doc_id.get("second")
        )

    @property
    def ordered_doc_ids(self) -> List[str]:
        if not self.resolved:
            return []
        return [self.role_to_doc_id["first"], self.role_to_doc_id["second"]]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["resolved"] = self.resolved
        data["ordered_doc_ids"] = self.ordered_doc_ids
        return data


class DocumentReferenceResolver:
    """Resolve first/second document roles from evidence, never from rank."""

    def __init__(self, runtime_index: Any, *, llm_client: Any = None, model: str = "",
                 enable_llm: bool = True, deterministic_threshold: float = 11.0,
                 deterministic_margin: float = 3.0):
        self.index = runtime_index
        self.llm_client = llm_client
        self.model = model or "qwen-plus"
        self.enable_llm = bool(enable_llm and llm_client is not None)
        self.deterministic_threshold = float(deterministic_threshold)
        self.deterministic_margin = float(deterministic_margin)

    @staticmethod
    def needs_resolution(question: str, options: Mapping[str, str]) -> bool:
        corpus = " ".join([str(question or "")] + [str(v) for v in options.values()])
        first, second, pair = _role_mentions(corpus)
        return bool(first or second or pair)

    def resolve(self, *, question: str, options: Mapping[str, str],
                claims_by_option: Mapping[str, Sequence[Any]], retrieval_result: Any,
                authoritative_ordered_doc_ids: Sequence[str] = (),
                authoritative_order_semantic: bool = True,
                authoritative_method: str = "input_doc_ids") -> DocumentBinding:
        needed = self.needs_resolution(question, options)
        authoritative = [str(x) for x in authoritative_ordered_doc_ids if str(x)]
        if len(authoritative) >= 2:
            method = str(authoritative_method or "input_doc_ids")
            return DocumentBinding(
                needed=needed, status="AUTHORITATIVE",
                role_to_doc_id={"first": authoritative[0], "second": authoritative[1]},
                confidence=1.0, method=method,
                order_semantic=bool(authoritative_order_semantic),
                reason=(
                    "Question input supplied an authoritative document order."
                    if method == "input_doc_ids"
                    else "Two explicit document identities define the authoritative target pair; retrieval rank was not used."
                ),
            )
        if not needed:
            return DocumentBinding(needed=False)

        candidates = [str(x) for x in _get(retrieval_result, "doc_ids", []) or [] if str(x)]
        candidates = list(dict.fromkeys(candidates))[:12]
        if len(candidates) < 2:
            return DocumentBinding(
                needed=True, status="UNRESOLVED", confidence=0.0,
                method="insufficient_candidates", reason="Fewer than two candidate documents were retrieved.",
                unresolved_roles=["first", "second"],
            )

        corpus = " ".join([str(question or "")] + [str(v) for v in options.values()])
        mentions_first, mentions_second, mentions_pair = _role_mentions(corpus)
        pair_only = bool(mentions_pair and not mentions_first and not mentions_second)
        claim_evidence = _get(retrieval_result, "claim_evidence", {}) or {}
        role_scores, pair_scores, evidence_catalog = self._score_candidates(
            claims_by_option, claim_evidence, candidates
        )
        pair_rows = self._rank_pairs(candidates, role_scores, pair_scores)
        deterministic = (
            self._deterministic_unordered_pair(candidates, pair_scores, evidence_catalog)
            if pair_only else self._deterministic_binding(pair_rows, evidence_catalog)
        )
        if deterministic.resolved:
            return deterministic

        if self.enable_llm:
            llm_binding = self._resolve_with_llm(
                question=question, options=options, candidates=candidates,
                role_scores=role_scores, pair_rows=pair_rows,
                evidence_catalog=evidence_catalog, claims_by_option=claims_by_option,
            )
            if llm_binding.resolved:
                return llm_binding
            # Preserve deterministic diagnostics when the LLM abstains.
            deterministic.llm_calls = llm_binding.llm_calls
            deterministic.retry_count = llm_binding.retry_count
            deterministic.warnings = list(llm_binding.warnings)
            deterministic.identity_support = dict(llm_binding.identity_support)
            deterministic.token_usage = llm_binding.token_usage
            deterministic.reason = (deterministic.reason + " " + llm_binding.reason).strip()

        return deterministic

    def _identity_text(self, doc_id: str) -> str:
        meta = self.index.meta_table.get(str(doc_id)) if getattr(self.index, "meta_table", None) else None
        identity = dict(getattr(self.index, "doc_identities", {}).get(str(doc_id), {}) or {})
        pieces: List[str] = [str(doc_id)]
        if meta is not None:
            for name in ("title", "issuer", "year", "doc_subtype"):
                value = getattr(meta, name, "")
                if value:
                    pieces.append(str(value))
            pieces.extend(str(x) for x in getattr(meta, "key_entities", []) or [])
            pieces.extend(str(x) for x in getattr(meta, "aliases", []) or [])
        for value in identity.values():
            if isinstance(value, (list, tuple, set)):
                pieces.extend(str(x) for x in value)
            elif value:
                pieces.append(str(value))
        return " | ".join(_unique(pieces))

    def _document_probe_texts(self, doc_id: str) -> Iterable[str]:
        """Yield identity and chunk text without using retrieval rank."""
        yield self._identity_text(doc_id)
        chunks = list(getattr(self.index, "chunk_index", {}).get(str(doc_id), []) or [])
        for chunk in chunks:
            text = str(_get(chunk, "text", "") or "")
            if text:
                yield text
        structured = dict(getattr(self.index, "structured_fields", {}).get(str(doc_id), {}) or {})
        for rows in structured.values():
            for row in rows or []:
                if isinstance(row, Mapping):
                    yield str(row.get("raw_evidence", "") or row.get("value", ""))

    def _probe_identity_anchors(self, doc_id: str, anchors: Sequence[Tuple[str, float]]) -> Tuple[float, List[str]]:
        pending = { _norm(text): (text, float(weight)) for text, weight in anchors if _norm(text) }
        if not pending:
            return 0.0, []
        matched: List[str] = []
        score = 0.0
        for probe in self._document_probe_texts(doc_id):
            norm_probe = _norm(probe)
            if not norm_probe:
                continue
            for key, (raw, weight) in list(pending.items()):
                if key in norm_probe:
                    matched.append(raw)
                    score += weight
                    pending.pop(key, None)
            if not pending:
                break
        # Exact decimal/unit anchors are exceptionally discriminative in
        # financial documents and must outweigh generic field co-occurrence.
        if any(re.search(r"\d+\.\d+\s*(?:%|％|亿元|万元|元|倍|天|日|年)?$", str(x)) for x in matched):
            score += 14.0
        if any(_CODE_RE.fullmatch(str(x)) for x in matched):
            score += 10.0
        return min(score, 36.0), _unique(matched)

    @staticmethod
    def _binding_constraints(claims_by_option: Mapping[str, Sequence[Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        roles: List[Dict[str, Any]] = []
        pairs: List[Dict[str, Any]] = []
        for option_key, claims in claims_by_option.items():
            for claim in claims:
                raw = str(_get(claim, "raw", "") or "")
                first, second, pair = _role_mentions(raw)
                row = {
                    "option": str(option_key), "claim_id": str(_get(claim, "claim_id", "")),
                    "text": raw[:300],
                }
                if first ^ second:
                    row["role"] = "first" if first else "second"
                    row["identity_anchors"] = [x[0] for x in _extract_identity_anchors(raw, claim)]
                    roles.append(row)
                elif pair or (first and second):
                    row["usage"] = "tie_breaker_only"
                    pairs.append(row)
        return roles, pairs

    def _score_candidates(self, claims_by_option: Mapping[str, Sequence[Any]],
                          claim_evidence: Mapping[str, Sequence[Any]], candidates: Sequence[str]
                          ) -> Tuple[Dict[str, Dict[str, float]], Dict[Tuple[str, str], float], Dict[str, Dict[str, Any]]]:
        role_scores: Dict[str, Dict[str, float]] = {
            "first": {did: 0.0 for did in candidates},
            "second": {did: 0.0 for did in candidates},
        }
        pair_scores: Dict[Tuple[str, str], float] = {(a, b): 0.0 for a in candidates for b in candidates if a != b}
        evidence_catalog: Dict[str, Dict[str, Any]] = {}
        deferred_relations: List[Tuple[str, Dict[str, List[Tuple[Decimal, str]]], str]] = []

        for option_claims in claims_by_option.values():
            for claim in option_claims:
                raw = str(_get(claim, "raw", "") or "")
                first, second, pair = _role_mentions(raw)
                if not (first or second or pair):
                    continue
                claim_id = str(_get(claim, "claim_id", ""))
                role_specific = bool(first ^ second)
                anchors = _extract_identity_anchors(raw, claim) if role_specific else _extract_anchors(raw, claim)
                per_doc: Dict[str, float] = {did: 0.0 for did in candidates}
                values_by_doc: Dict[str, List[Tuple[Decimal, str]]] = {did: [] for did in candidates}
                structured_value_docs: set[str] = set()
                for raw_span in claim_evidence.get(claim_id, []) or []:
                    span = _span_dict(raw_span)
                    did = str(span.get("doc_id", ""))
                    if did not in per_doc:
                        continue
                    eid = str(span.get("evidence_id", ""))
                    text = str(span.get("text", ""))
                    metadata = dict(span.get("metadata", {}) or {})
                    norm_text = _norm(text + " " + json.dumps(metadata, ensure_ascii=False))
                    score = min(1.0, math.log1p(max(0.0, float(span.get("score", 0.0) or 0.0))) / 4.0)
                    matched: List[str] = []
                    for anchor, weight in anchors:
                        if _norm(anchor) and _norm(anchor) in norm_text:
                            score += weight if weight >= 5.0 else weight * 0.35
                            matched.append(anchor)
                    field_value = str(metadata.get("field", ""))
                    for field in _get(claim, "field_candidates", []) or []:
                        if role_specific and field and (_norm(field) in _norm(field_value) or _norm(field_value) in _norm(field)):
                            score += 1.5
                            matched.append(str(field))
                            break
                    source = str(span.get("source", ""))
                    if source in {"structured_field", "table", "table_atom"}:
                        score += 0.5
                    per_doc[did] = max(per_doc[did], score)
                    span_numbers = _numbers_from_span(span)
                    if source in {"structured_field", "table", "table_atom"} and span_numbers:
                        if did not in structured_value_docs:
                            values_by_doc[did] = []
                            structured_value_docs.add(did)
                        for item in span_numbers:
                            if item not in values_by_doc[did]:
                                values_by_doc[did].append(item)
                    elif did not in structured_value_docs:
                        for item in span_numbers:
                            if item not in values_by_doc[did]:
                                values_by_doc[did].append(item)
                    if eid:
                        evidence_catalog[eid] = {
                            "evidence_id": eid, "doc_id": did, "text": text[:1000],
                            "claim_id": claim_id, "score": round(score, 4),
                            "matched_anchors": _unique(matched),
                        }

                # Search the actual candidate document, not just top-k evidence.
                # This recovers identity phrases and exact values that fell below
                # the first retrieval cut (e.g. transaction type / 43.24%).
                if role_specific:
                    for did in candidates:
                        probe_score, probe_hits = self._probe_identity_anchors(did, anchors)
                        per_doc[did] += probe_score
                        if probe_hits:
                            synthetic_id = f"identity_probe:{did}:{claim_id}"
                            evidence_catalog.setdefault(synthetic_id, {
                                "evidence_id": synthetic_id, "doc_id": did,
                                "text": " | ".join(probe_hits)[:1000], "claim_id": claim_id,
                                "score": round(probe_score, 4), "matched_anchors": probe_hits,
                            })

                if first and not second:
                    for did, score in per_doc.items():
                        role_scores["first"][did] += score
                elif second and not first:
                    for did, score in per_doc.items():
                        role_scores["second"][did] += score
                else:
                    # Pair/universal statements may identify an unordered partner
                    # set, but never establish first/second order by themselves.
                    for left in candidates:
                        for right in candidates:
                            if left != right:
                                pair_scores[(left, right)] += min(per_doc[left], per_doc[right]) * 0.45

                comparator = str(_get(claim, "comparator", "") or "")
                if first and second and comparator in {"GT", "GTE", "LT", "LTE"}:
                    deferred_relations.append((raw, values_by_doc, comparator))

        # Comparative truth predicates are a bounded tie-breaker only after at
        # least one role has independent role-specific identity support.
        identity_seeded = (
            max(role_scores["first"].values(), default=0.0) >= 6.0
            or max(role_scores["second"].values(), default=0.0) >= 6.0
        )
        if identity_seeded:
            for raw, values_by_doc, comparator in deferred_relations:
                first_pos = min([p for p in [raw.find("第一份"), raw.find("第一篇"), raw.find("前者")] if p >= 0] or [10**9])
                second_pos = min([p for p in [raw.find("第二份"), raw.find("第二篇"), raw.find("后者")] if p >= 0] or [10**9])
                left_role = "first" if first_pos < second_pos else "second"
                for first_doc in candidates:
                    for second_doc in candidates:
                        if first_doc == second_doc:
                            continue
                        left_doc = first_doc if left_role == "first" else second_doc
                        right_doc = second_doc if left_role == "first" else first_doc
                        relation = self._numeric_relation(values_by_doc[left_doc], values_by_doc[right_doc], comparator)
                        if relation is True:
                            pair_scores[(first_doc, second_doc)] += 2.0
                        elif relation is False:
                            pair_scores[(first_doc, second_doc)] -= 1.0
        return role_scores, pair_scores, evidence_catalog

    @staticmethod
    def _numeric_relation(left_values: Sequence[Tuple[Decimal, str]], right_values: Sequence[Tuple[Decimal, str]],
                          comparator: str) -> Optional[bool]:
        left_by_unit: Dict[str, set] = {}
        right_by_unit: Dict[str, set] = {}
        for value, unit in left_values:
            left_by_unit.setdefault(unit, set()).add(value)
        for value, unit in right_values:
            right_by_unit.setdefault(unit, set()).add(value)
        common = [unit for unit in left_by_unit if unit and unit in right_by_unit]
        determinate: List[bool] = []
        for unit in common:
            if len(left_by_unit[unit]) != 1 or len(right_by_unit[unit]) != 1:
                continue
            left = next(iter(left_by_unit[unit]))
            right = next(iter(right_by_unit[unit]))
            if comparator == "GT":
                determinate.append(left > right)
            elif comparator == "GTE":
                determinate.append(left >= right)
            elif comparator == "LT":
                determinate.append(left < right)
            elif comparator == "LTE":
                determinate.append(left <= right)
        if len(set(determinate)) == 1 and determinate:
            return determinate[0]
        return None

    def _rank_pairs(self, candidates: Sequence[str], role_scores: Mapping[str, Mapping[str, float]],
                    pair_scores: Mapping[Tuple[str, str], float]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for first in candidates:
            for second in candidates:
                if first == second:
                    continue
                score = (
                    float(role_scores.get("first", {}).get(first, 0.0))
                    + float(role_scores.get("second", {}).get(second, 0.0))
                    + float(pair_scores.get((first, second), 0.0))
                )
                rows.append({
                    "first": first, "second": second, "score": round(score, 4),
                    "first_score": round(float(role_scores.get("first", {}).get(first, 0.0)), 4),
                    "second_score": round(float(role_scores.get("second", {}).get(second, 0.0)), 4),
                    "pair_score": round(float(pair_scores.get((first, second), 0.0)), 4),
                })
        rows.sort(key=lambda row: (-row["score"], row["first"], row["second"]))
        return rows[:12]

    def _deterministic_unordered_pair(self, candidates: Sequence[str],
                                      pair_scores: Mapping[Tuple[str, str], float],
                                      evidence_catalog: Mapping[str, Mapping[str, Any]]) -> DocumentBinding:
        unordered: List[Dict[str, Any]] = []
        seen = set()
        for left in candidates:
            for right in candidates:
                if left == right:
                    continue
                pair = tuple(sorted((left, right)))
                if pair in seen:
                    continue
                seen.add(pair)
                score = max(float(pair_scores.get((left, right), 0.0)), float(pair_scores.get((right, left), 0.0)))
                unordered.append({"docs": list(pair), "score": round(score, 4)})
        unordered.sort(key=lambda row: (-row["score"], row["docs"]))
        if not unordered:
            return DocumentBinding(needed=True, status="UNRESOLVED", confidence=0.0,
                                   method="deterministic_unordered_pair", order_semantic=False,
                                   reason="No unordered pair could be scored.", unresolved_roles=["document_pair"])
        top = unordered[0]
        runner = float(unordered[1]["score"]) if len(unordered) > 1 else 0.0
        margin = float(top["score"]) - runner
        resolved = float(top["score"]) >= 0.9 and margin >= 0.8
        evidence_ids = [eid for eid, row in evidence_catalog.items() if row.get("doc_id") in set(top["docs"])][:12]
        if resolved:
            # The lexical order is only an internal carrier for ClaimCompiler;
            # no semantic first/second claim exists in pair-only questions.
            first, second = top["docs"]
            return DocumentBinding(
                needed=True, status="RESOLVED", role_to_doc_id={"first": first, "second": second},
                confidence=min(0.95, 0.65 + 0.04 * margin), method="deterministic_unordered_pair",
                order_semantic=False, reason=f"Unique unordered document pair recovered with margin={margin:.3f}.",
                evidence_ids=evidence_ids, candidate_pairs=unordered[:5],
            )
        return DocumentBinding(
            needed=True, status="UNRESOLVED", confidence=min(0.69, 0.2 + 0.03 * max(0.0, float(top["score"]))),
            method="deterministic_unordered_pair", order_semantic=False,
            reason=f"Unordered pair evidence was ambiguous (top={float(top['score']):.3f}, margin={margin:.3f}).",
            evidence_ids=evidence_ids, candidate_pairs=unordered[:5], unresolved_roles=["document_pair"],
        )

    def _deterministic_binding(self, pair_rows: Sequence[Mapping[str, Any]],
                               evidence_catalog: Mapping[str, Mapping[str, Any]]) -> DocumentBinding:
        if not pair_rows:
            return DocumentBinding(
                needed=True, status="UNRESOLVED", confidence=0.0,
                method="deterministic_constraints", reason="No distinct document pair could be scored.",
                unresolved_roles=["first", "second"],
            )
        top = dict(pair_rows[0])
        second_score = float(pair_rows[1]["score"]) if len(pair_rows) > 1 else 0.0
        margin = float(top["score"]) - second_score
        first_score = float(top.get("first_score", 0.0))
        second_role_score = float(top.get("second_score", 0.0))
        strong_roles = first_score >= 4.0 and second_role_score >= 4.0
        resolved = bool(float(top["score"]) >= self.deterministic_threshold and margin >= self.deterministic_margin and strong_roles)
        confidence = min(0.98, max(0.0, 0.45 + 0.025 * float(top["score"]) + 0.04 * margin)) if resolved else min(0.69, max(0.05, 0.2 + 0.02 * max(0.0, float(top["score"])) + 0.02 * max(0.0, margin)))
        evidence_ids = [
            eid for eid, row in evidence_catalog.items()
            if row.get("doc_id") in {top.get("first"), top.get("second")} and float(row.get("score", 0.0)) >= 4.0
        ][:12]
        if resolved:
            return DocumentBinding(
                needed=True, status="RESOLVED",
                role_to_doc_id={"first": str(top["first"]), "second": str(top["second"])},
                confidence=round(confidence, 4), method="deterministic_constraints",
                reason=f"Top pair exceeded threshold with margin={margin:.3f}; retrieval order was not used.",
                evidence_ids=evidence_ids, candidate_pairs=[dict(x) for x in pair_rows[:5]],
            )
        return DocumentBinding(
            needed=True, status="UNRESOLVED", confidence=round(confidence, 4),
            method="deterministic_constraints",
            reason=f"Pair evidence was ambiguous (top={float(top['score']):.3f}, margin={margin:.3f}).",
            evidence_ids=evidence_ids, candidate_pairs=[dict(x) for x in pair_rows[:5]],
            unresolved_roles=["first", "second"],
        )

    def _compact_candidate_context(self, candidates: Sequence[str], evidence_catalog: Mapping[str, Mapping[str, Any]],
                                   pair_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        priority_docs: List[str] = []
        for row in pair_rows[:5]:
            priority_docs.extend([str(row.get("first", "")), str(row.get("second", ""))])
        priority_docs.extend(candidates)
        selected_docs = list(dict.fromkeys(x for x in priority_docs if x))[:8]
        evidence_map: Dict[str, str] = {}
        cards: List[Dict[str, Any]] = []
        for did in selected_docs:
            rows = [dict(row) for row in evidence_catalog.values() if str(row.get("doc_id", "")) == did]
            rows.sort(key=lambda row: (-float(row.get("score", 0.0)), str(row.get("evidence_id", ""))))
            compact_evidence = []
            for row in rows[:4]:
                eid = str(row.get("evidence_id", ""))
                text = str(row.get("text", ""))[:360]
                if eid and text:
                    evidence_map[eid] = did
                    compact_evidence.append({
                        "evidence_id": eid, "text": text,
                        "matched_anchors": row.get("matched_anchors", []),
                    })
            cards.append({
                "doc_id": did,
                "identity": self._identity_text(did)[:500],
                "evidence": compact_evidence,
            })
        return cards, evidence_map

    def _resolve_with_llm(self, *, question: str, options: Mapping[str, str], candidates: Sequence[str],
                          role_scores: Mapping[str, Mapping[str, float]], pair_rows: Sequence[Mapping[str, Any]],
                          evidence_catalog: Mapping[str, Mapping[str, Any]],
                          claims_by_option: Mapping[str, Sequence[Any]]) -> DocumentBinding:
        cards, evidence_map = self._compact_candidate_context(candidates, evidence_catalog, pair_rows)
        role_constraints, pair_constraints = self._binding_constraints(claims_by_option)
        if len(cards) < 2 or not evidence_map or not role_constraints:
            return DocumentBinding(
                needed=True, status="UNRESOLVED", confidence=0.0, method="qwen_binding",
                reason="Qwen binding skipped because role-specific identity evidence was insufficient.",
                unresolved_roles=["first", "second"],
            )
        system = """你是金融多文档题目的文档角色绑定器。只做文档身份绑定，不判断选项对错。
只能使用role_constraints中的角色专属绝对锚点（公司名、代码、报告/交易类型、字段+精确值）和逐字证据。
禁止使用候选顺序、doc_id数字、检索排名、常识；禁止用“均为/高于/低于/正确”等选项真假作为主要身份依据。
pair_constraints只能在至少一个角色已由绝对锚点确定后用于并列候选的次级排除。
证据不能唯一确定时必须UNRESOLVED。只输出极简JSON。"""
        base_payload = {
            "candidate_documents": cards,
            "role_constraints": role_constraints,
            "pair_constraints": pair_constraints[:4],
            "deterministic_pairs": list(pair_rows[:5]),
        }
        user = json.dumps(base_payload, ensure_ascii=False) + "\n输出格式：" + (
            '{"status":"RESOLVED|UNRESOLVED","first_doc_id":"","second_doc_id":"",'
            '"confidence":0.0,"first_evidence_id":"","second_evidence_id":""}'
        )

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        warnings: List[str] = []
        calls = 0
        data: Dict[str, Any] = {}
        for attempt in range(2):
            attempt_system = system if attempt == 0 else (
                "修复上一轮JSON格式错误。不要解释，只输出单行JSON；所有字符串必须闭合。"
            )
            attempt_user = user if attempt == 0 else (
                json.dumps({"candidate_doc_ids": [x["doc_id"] for x in cards],
                            "role_constraints": role_constraints}, ensure_ascii=False)
                + '\n只输出：{"status":"RESOLVED|UNRESOLVED","first_doc_id":"",'
                  '"second_doc_id":"","confidence":0.0,"first_evidence_id":"",'
                  '"second_evidence_id":""}'
            )
            response = None
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": attempt_system},
                              {"role": "user", "content": attempt_user}],
                    temperature=0, max_tokens=260 if attempt == 0 else 160,
                    response_format={"type": "json_object"},
                )
                calls += 1
                # Usage is recorded before parsing so malformed JSON cannot hide cost.
                total_usage = _merge_usage(total_usage, _usage_from_response(response))
                content = str(response.choices[0].message.content or "")
                data = _parse_json_object(content)
                break
            except Exception as exc:
                if response is None:
                    calls += 1
                warnings.append(f"attempt_{attempt + 1}:{type(exc).__name__}:{exc}")
                data = {}
                if attempt == 0:
                    continue
        if not data:
            return DocumentBinding(
                needed=True, status="UNRESOLVED", confidence=0.0, method="qwen_binding",
                reason="Qwen binding JSON failed after one format retry.",
                unresolved_roles=["first", "second"], llm_calls=calls,
                retry_count=max(0, calls - 1), warnings=warnings, token_usage=total_usage,
                candidate_pairs=[dict(x) for x in pair_rows[:5]],
            )

        status = str(data.get("status", "")).upper()
        first = str(data.get("first_doc_id", ""))
        second = str(data.get("second_doc_id", ""))
        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        first_citations = [str(data.get("first_evidence_id", "") or "")]
        second_citations = [str(data.get("second_evidence_id", "") or "")]
        # Backward compatibility with the older nested citation schema.
        citations_obj = data.get("citations", {}) if isinstance(data.get("citations", {}), Mapping) else {}
        first_citations += [str(x) for x in citations_obj.get("first", []) if str(x)]
        second_citations += [str(x) for x in citations_obj.get("second", []) if str(x)]
        first_citations = _unique(first_citations)
        second_citations = _unique(second_citations)
        valid = bool(
            status == "RESOLVED" and first in candidates and second in candidates and first != second
            and confidence >= 0.75
            and any(evidence_map.get(eid) == first for eid in first_citations)
            and any(evidence_map.get(eid) == second for eid in second_citations)
        )
        identity_support = {"first": first_citations, "second": second_citations}
        if not valid:
            return DocumentBinding(
                needed=True, status="UNRESOLVED", confidence=min(confidence, 0.74),
                method="qwen_binding", reason="Qwen did not return a citation-backed unique role pair.",
                evidence_ids=_unique(first_citations + second_citations), identity_support=identity_support,
                candidate_pairs=[dict(x) for x in pair_rows[:5]], unresolved_roles=["first", "second"],
                llm_calls=calls, retry_count=max(0, calls - 1), warnings=warnings,
                token_usage=total_usage,
            )
        return DocumentBinding(
            needed=True, status="RESOLVED", role_to_doc_id={"first": first, "second": second},
            confidence=min(0.99, confidence), method="qwen_binding",
            reason="Citation-backed role-specific identity anchors uniquely selected the document pair.",
            evidence_ids=_unique(first_citations + second_citations), identity_support=identity_support,
            candidate_pairs=[dict(x) for x in pair_rows[:5]], llm_calls=calls,
            retry_count=max(0, calls - 1), warnings=warnings, token_usage=total_usage,
        )


__all__ = ["DocumentBinding", "DocumentReferenceResolver"]
