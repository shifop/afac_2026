#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Share auditable evidence across options of the same question.

The ledger never creates facts.  It only reuses a span already retrieved for
another claim when subject/field/year signals overlap.  This fixes option
islands such as one dividend option locating the 2025 amount while another
comparison option cannot see it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from retrieval_models import EvidenceSpanV2, claim_attr

_GENERIC = {
    "文档", "文件", "报告", "公司", "第一份", "第二份", "两份文档", "两篇文档",
    "正确", "错误", "描述", "情况", "数据", "数值", "指标",
}
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def _norm(text: Any) -> str:
    return re.sub(r"[\s，。；：、,.!?！？:;()（）\[\]【】\"'“”‘’·—\-]", "", str(text or "")).lower()


def _unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _claim_signature(claim: Any) -> Dict[str, List[str]]:
    raw = str(claim_attr(claim, "raw", "") or "")
    subjects = [str(x) for x in (claim_attr(claim, "subject_candidates", []) or [])]
    fields = [str(x) for x in (claim_attr(claim, "field_candidates", []) or [])]
    years = [str(x) for x in (claim_attr(claim, "years", []) or [])]
    years.extend(_YEAR_RE.findall(raw))
    subjects = [x for x in subjects if len(_norm(x)) >= 3 and x not in _GENERIC]
    fields = [x for x in fields if len(_norm(x)) >= 2 and x not in _GENERIC]
    return {"subjects": _unique(subjects), "fields": _unique(fields), "years": _unique(years)}


def _span_text(span: EvidenceSpanV2) -> str:
    meta = dict(getattr(span, "metadata", {}) or {})
    return " ".join([
        str(getattr(span, "text", "") or ""),
        str(meta.get("field", "") or ""),
        str(meta.get("value", "") or ""),
        " ".join(str(x) for x in (meta.get("doc_identity_terms", []) or [])),
        " ".join(str(x) for x in (getattr(span, "matched_terms", []) or [])),
    ])


def _match_score(signature: Mapping[str, Sequence[str]], span: EvidenceSpanV2) -> float:
    text = _norm(_span_text(span))
    if not text:
        return 0.0
    score = 0.0
    subject_hits = [s for s in signature.get("subjects", []) if _norm(s) and _norm(s) in text]
    field_hits = [f for f in signature.get("fields", []) if _norm(f) and _norm(f) in text]
    year_hits = [y for y in signature.get("years", []) if y and y in text]
    if subject_hits:
        score += 6.0
    if field_hits:
        score += 7.0
    if year_hits:
        score += 3.0
    # A field match is mandatory unless the claim has no field candidate.
    if signature.get("fields") and not field_hits:
        return 0.0
    # When a concrete subject is available, do not share evidence from another
    # entity merely because the field/year matches.
    if signature.get("subjects") and not subject_hits:
        identity = " ".join(str(x) for x in (getattr(span, "metadata", {}) or {}).get("doc_identity_terms", []) or [])
        if not any(_norm(s) in _norm(identity) or _norm(identity) in _norm(s) for s in signature.get("subjects", []) if _norm(s)):
            return 0.0
    score += min(2.0, float(getattr(span, "score", 0.0) or 0.0) / 50.0)
    return score


def apply_question_evidence_ledger(retrieval: Any, *, claims_by_option: Mapping[str, Sequence[Any]],
                                   max_shared_per_claim: int = 3) -> Dict[str, Any]:
    claim_evidence = getattr(retrieval, "claim_evidence", None)
    option_evidence = getattr(retrieval, "option_evidence", None)
    if not isinstance(claim_evidence, dict) or not isinstance(option_evidence, dict):
        return {"shared_spans": 0, "rows": []}

    all_spans: Dict[str, EvidenceSpanV2] = {}
    for spans in claim_evidence.values():
        for span in spans or []:
            eid = str(getattr(span, "evidence_id", "") or "")
            if eid and eid not in all_spans:
                all_spans[eid] = span

    rows: List[Dict[str, Any]] = []
    shared_total = 0
    for option_key, claims in claims_by_option.items():
        for claim in claims:
            claim_id = str(claim_attr(claim, "claim_id", "") or "")
            existing = list(claim_evidence.get(claim_id, []) or [])
            existing_ids = {str(getattr(span, "evidence_id", "") or "") for span in existing}
            signature = _claim_signature(claim)
            candidates: List[Tuple[float, EvidenceSpanV2]] = []
            for eid, span in all_spans.items():
                if eid in existing_ids:
                    continue
                score = _match_score(signature, span)
                if score >= 7.0:
                    candidates.append((score, span))
            candidates.sort(key=lambda item: (-item[0], -float(getattr(item[1], "score", 0.0) or 0.0), str(getattr(item[1], "evidence_id", ""))))
            additions = [span for _, span in candidates[:max_shared_per_claim]]
            if additions:
                claim_evidence[claim_id] = existing + additions
                shared_total += len(additions)
                rows.append({
                    "option": str(option_key), "claim_id": claim_id,
                    "evidence_ids": [str(getattr(span, "evidence_id", "")) for span in additions],
                })

    # Rebuild option evidence after sharing.
    for option_key, claims in claims_by_option.items():
        merged: List[EvidenceSpanV2] = []
        seen = set()
        for claim in claims:
            for span in claim_evidence.get(str(claim_attr(claim, "claim_id", "")), []) or []:
                eid = str(getattr(span, "evidence_id", "") or "")
                if eid and eid not in seen:
                    seen.add(eid)
                    merged.append(span)
        option_evidence[str(option_key)] = merged

    audit = {"shared_spans": shared_total, "rows": rows}
    getattr(retrieval, "audit", {}).setdefault("question_evidence_ledger", audit)
    return audit


__all__ = ["apply_question_evidence_ledger"]
