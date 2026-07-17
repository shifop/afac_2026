#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a claim-balanced, context-preserving evidence bundle per question."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from score_optimized_config import ScoreOptimizedConfig, load_score_optimized_config


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _short(value: Any, limit: int = 120) -> str:
    return _clean_text(value)[:limit]


@dataclass
class BundleResult:
    payload: Dict[str, Any]
    audit: Dict[str, Any]


class EvidenceBundleBuilder:
    """Allocate evidence by unresolved claim and document, not global Top-K.

    C1 ranked every span globally.  A high-scoring document could therefore
    consume the whole bundle and erase the other side of a comparison.  C1.1
    first rotates across claims, then across documents within each claim.
    """

    def __init__(self, config: ScoreOptimizedConfig | None = None):
        self.config = config or load_score_optimized_config()

    def build(
        self,
        *,
        qid: str,
        question: str,
        options: Mapping[str, str],
        answer_format: str,
        decisions: Mapping[str, Any],
        unresolved: Mapping[str, Sequence[Any]],
        evidence_by_option: Mapping[str, Sequence[Any]],
        evidence_by_claim: Mapping[str, Sequence[Any]] | None = None,
        document_binding: Mapping[str, Any] | None = None,
        document_catalog: Mapping[str, Any] | None = None,
    ) -> BundleResult:
        unresolved_candidates = {
            key: list(unresolved.get(key, []) or [])[: self.config.max_claims_per_option]
            for key in sorted(options)
        }
        selected_review = {key: [] for key in sorted(options)}
        for claim_number in range(self.config.max_claims_per_option):
            for option_key in sorted(options):
                if sum(len(rows) for rows in selected_review.values()) >= self.config.max_review_claims:
                    break
                rows = unresolved_candidates.get(option_key, [])
                if claim_number < len(rows):
                    selected_review[option_key].append(rows[claim_number])

        option_rows: List[Dict[str, Any]] = []
        review_claim_ids: List[str] = []
        claim_to_option: Dict[str, str] = {}
        claim_requirements: Dict[str, Dict[str, Any]] = {}
        for option_key in sorted(options):
            decision = decisions.get(option_key)
            local_claims = []
            for claim in list(_get(decision, "claim_decisions", []) or []):
                local_claims.append({
                    "claim_id": str(_get(claim, "claim_id", "")),
                    "local_verdict": str(_get(claim, "verdict", "UNKNOWN")),
                    "trust": self._trust_hint(str(_get(claim, "source", ""))),
                    "source": str(_get(claim, "source", "")),
                    "reason": _short(_get(claim, "reason", ""), 180),
                    "evidence_ids": [str(x) for x in (_get(claim, "evidence_ids", []) or [])][:4],
                    "missing_fields": [str(x) for x in (_get(claim, "missing_fields", []) or [])][:5],
                })
            review_rows = []
            for claim in selected_review.get(option_key, []):
                claim_id = str(_get(claim, "claim_id", ""))
                if not claim_id:
                    continue
                review_claim_ids.append(claim_id)
                claim_to_option[claim_id] = option_key
                doc_ids = list(dict.fromkeys(
                    str(x) for x in (
                        list(_get(claim, "doc_hints", []) or [])
                        + list(_get(claim, "comparison_doc_ids", []) or [])
                    ) if str(x)
                ))
                requires_multi_doc = bool(_get(claim, "requires_multi_doc", False))
                claim_requirements[claim_id] = {
                    "requires_multi_doc": requires_multi_doc,
                    "required_doc_ids": doc_ids[:6],
                }
                review_rows.append({
                    "claim_id": claim_id,
                    "statement": _short(_get(claim, "raw", ""), 520),
                    "subjects": [str(x) for x in (_get(claim, "subject_candidates", []) or [])][:5],
                    "fields": [str(x) for x in (_get(claim, "field_candidates", []) or [])][:5],
                    "years": [str(x) for x in (_get(claim, "years", []) or [])][:4],
                    "required_doc_ids": doc_ids[:6],
                    "requires_multi_doc": requires_multi_doc,
                })
            option_rows.append({
                "key": option_key,
                "text": _short(options.get(option_key, ""), 760),
                "local_option_verdict": str(_get(decision, "verdict", "UNKNOWN")),
                "local_claims": local_claims,
                "review_claims": review_rows,
            })

        catalog = dict(document_catalog or {})
        evidence_by_claim = dict(evidence_by_claim or {})
        candidate_rows: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        claim_candidates: Dict[str, List[str]] = {claim_id: [] for claim_id in review_claim_ids}

        def register(span: Any, *, option_key: str = "", claim_id: str = "") -> None:
            evidence_id = str(_get(span, "evidence_id", ""))
            text = _clean_text(_get(span, "text", ""))
            if not evidence_id or not text:
                return
            score = float(_get(span, "score", 0.0) or 0.0)
            if evidence_id not in candidate_rows:
                metadata = dict(_get(span, "metadata", {}) or {})
                doc_id = str(_get(span, "doc_id", ""))
                row = {
                    "evidence_id": evidence_id,
                    "doc_id": doc_id,
                    "document_context": self._document_context(doc_id, metadata, catalog),
                    "text": text[: self.config.max_chars_per_span],
                    "source": str(_get(span, "source", "")),
                    "for_options": [],
                    "for_claims": [],
                }
                candidate_rows[evidence_id] = (score, row)
            else:
                old_score, row = candidate_rows[evidence_id]
                candidate_rows[evidence_id] = (max(old_score, score), row)
            row = candidate_rows[evidence_id][1]
            if option_key and option_key not in row["for_options"]:
                row["for_options"].append(option_key)
            if claim_id and claim_id not in row["for_claims"]:
                row["for_claims"].append(claim_id)
            if claim_id and evidence_id not in claim_candidates.setdefault(claim_id, []):
                claim_candidates[claim_id].append(evidence_id)

        # Prefer true claim-local retrieval rows.  Fall back to option-local
        # rows for compatibility with older retrieval objects.
        for claim_id in review_claim_ids:
            option_key = claim_to_option.get(claim_id, "")
            rows = list(evidence_by_claim.get(claim_id, []) or [])
            if not rows:
                rows = list(evidence_by_option.get(option_key, []) or [])
            for span in rows:
                register(span, option_key=option_key, claim_id=claim_id)

        # Keep any remaining option evidence available for the final fill pass.
        for option_key, spans in evidence_by_option.items():
            for span in spans or []:
                register(span, option_key=str(option_key))

        for claim_id, ids in claim_candidates.items():
            ids.sort(key=lambda eid: (candidate_rows[eid][0], eid), reverse=True)

        selected_ids: List[str] = []
        selected_set: set[str] = set()
        seen_docs_by_claim: Dict[str, set[str]] = {claim_id: set() for claim_id in review_claim_ids}
        cursors = {claim_id: 0 for claim_id in review_claim_ids}

        # Round-robin over claims.  For multi-document claims, prefer an unseen
        # document before selecting a second span from an already represented one.
        while len(selected_ids) < self.config.max_evidence_spans:
            progressed = False
            for claim_id in review_claim_ids:
                ids = claim_candidates.get(claim_id, [])
                remaining = [eid for eid in ids if eid not in selected_set]
                if not remaining:
                    continue
                requirement = claim_requirements.get(claim_id, {})
                chosen = remaining[0]
                if requirement.get("requires_multi_doc"):
                    required_docs = set(str(x) for x in requirement.get("required_doc_ids", []) if str(x))
                    unseen_required = [
                        eid for eid in remaining
                        if candidate_rows[eid][1].get("doc_id") in required_docs
                        and candidate_rows[eid][1].get("doc_id") not in seen_docs_by_claim[claim_id]
                    ]
                    unseen_any = [
                        eid for eid in remaining
                        if candidate_rows[eid][1].get("doc_id") not in seen_docs_by_claim[claim_id]
                    ]
                    chosen = (unseen_required or unseen_any or remaining)[0]
                selected_ids.append(chosen)
                selected_set.add(chosen)
                seen_docs_by_claim[claim_id].add(str(candidate_rows[chosen][1].get("doc_id", "")))
                cursors[claim_id] += 1
                progressed = True
                if len(selected_ids) >= self.config.max_evidence_spans:
                    break
            if not progressed:
                break

        # Fill unused capacity by score without disturbing the balanced prefix.
        global_ids = sorted(candidate_rows, key=lambda eid: (candidate_rows[eid][0], eid), reverse=True)
        for evidence_id in global_ids:
            if len(selected_ids) >= self.config.max_evidence_spans:
                break
            if evidence_id not in selected_set:
                selected_ids.append(evidence_id)
                selected_set.add(evidence_id)

        evidence_rows: List[Dict[str, Any]] = []
        evidence_chars = 0
        for evidence_id in selected_ids:
            row = dict(candidate_rows[evidence_id][1])
            remaining = self.config.max_evidence_chars - evidence_chars
            if remaining <= 120:
                break
            row["text"] = str(row.get("text", ""))[:remaining]
            evidence_chars += len(row["text"])
            evidence_rows.append(row)

        binding = dict(document_binding or {})
        payload = {
            "qid": str(qid),
            "question": _short(question, 1_100),
            "answer_format": str(answer_format or ""),
            "options": option_rows,
            "evidence": evidence_rows,
            "document_binding": {
                "status": str(binding.get("status", "")),
                "needed": bool(binding.get("needed", False)),
                "resolved": bool(binding.get("resolved", False)),
                "ordered_doc_ids": [str(x) for x in (binding.get("ordered_doc_ids", []) or [])][:6],
                "order_semantic": bool(binding.get("order_semantic", True)),
            },
            "review_claim_ids": review_claim_ids,
            "claim_requirements": claim_requirements,
        }
        payload, shrink_rounds = self._fit_payload(payload)
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        represented = {
            claim_id: sorted({
                str(row.get("doc_id", "")) for row in payload.get("evidence", [])
                if claim_id in (row.get("for_claims", []) or []) and row.get("doc_id")
            })
            for claim_id in review_claim_ids
        }
        return BundleResult(
            payload=payload,
            audit={
                "prompt_payload_chars": len(serialized),
                "evidence_spans": len(payload.get("evidence", [])),
                "evidence_chars": sum(len(str(x.get("text", ""))) for x in payload.get("evidence", [])),
                "candidate_evidence_spans": len(candidate_rows),
                "review_claims": len(review_claim_ids),
                "candidate_review_claims": sum(len(rows) for rows in unresolved_candidates.values()),
                "review_claims_truncated": max(0, sum(len(rows) for rows in unresolved_candidates.values()) - len(review_claim_ids)),
                "claim_document_coverage": represented,
                "selection_policy": "claim_document_round_robin_then_score",
                "shrink_rounds": shrink_rounds,
                "within_payload_cap": len(serialized) <= self.config.max_prompt_chars,
            },
        )

    @staticmethod
    def _document_context(doc_id: str, metadata: Mapping[str, Any], catalog: Mapping[str, Any]) -> Dict[str, Any]:
        catalog_value = catalog.get(doc_id, {})
        if isinstance(catalog_value, Mapping):
            base = dict(catalog_value)
        else:
            base = {"document_name": os.path.basename(str(catalog_value or ""))}
        aliases = {
            "document_name": ("document_name", "title", "file_name", "filename"),
            "company": ("company", "issuer", "entity", "product"),
            "year": ("year", "report_year", "period"),
            "section": ("section", "section_title", "heading"),
            "page": ("page", "page_no", "page_number"),
            "unit": ("unit_context", "unit", "currency"),
            "table": ("table_id", "table_title", "row_header", "column_headers"),
        }
        context: Dict[str, Any] = {"doc_id": doc_id}
        merged = dict(base)
        merged.update(dict(metadata or {}))
        for output_name, keys in aliases.items():
            values: List[str] = []
            for key in keys:
                value = merged.get(key)
                if isinstance(value, (list, tuple)):
                    values.extend(_short(x, 80) for x in value if _short(x, 80))
                elif value not in (None, ""):
                    values.append(_short(value, 120))
            values = list(dict.fromkeys(x for x in values if x))
            if values:
                context[output_name] = values[0] if len(values) == 1 else values[:4]
        if "document_name" not in context and doc_id:
            context["document_name"] = doc_id
        return context

    def _fit_payload(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        shrink_rounds = 0
        while len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) > self.config.max_prompt_chars:
            evidence = list(payload.get("evidence", []) or [])
            if evidence and len(evidence) > 4:
                evidence.pop()
                payload["evidence"] = evidence
            elif evidence and max(len(str(x.get("text", ""))) for x in evidence) > 260:
                for row in evidence:
                    row["text"] = str(row.get("text", ""))[: max(260, len(str(row.get("text", ""))) * 3 // 4)]
            else:
                payload["question"] = str(payload.get("question", ""))[:700]
                for option in payload.get("options", []) or []:
                    option["text"] = str(option.get("text", ""))[:420]
                    for claim in option.get("local_claims", []) or []:
                        claim["reason"] = str(claim.get("reason", ""))[:80]
            shrink_rounds += 1
            if shrink_rounds >= 16:
                break
        return payload, shrink_rounds

    @staticmethod
    def _trust_hint(source: str) -> str:
        hard_prefixes = (
            "EXACT_TEXT", "EXPLICIT_DOC_TERM_RULE", "FULL_DOCUMENT_ABSENCE_RULE",
            "CROSS_DOC_NUMERIC_RULE", "FACT_INDEX_TREND", "INS_RULE",
            "INSURANCE_CALC_RULE", "FINANCIAL_CALC_RELIABLE",
        )
        return "HARD" if str(source or "").startswith(hard_prefixes) else "REVIEW"
