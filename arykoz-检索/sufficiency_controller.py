#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence obligation evaluation and bounded gap-driven re-retrieval."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
import re

from field_ontology import aliases_for_field
from retrieval_models import EvidenceObligation, EvidenceSpanV2, claim_attr, unique_strings


class SufficiencyController:
    def _components(self, spans: Sequence[EvidenceSpanV2]) -> str:
        return "\n".join(span.text for span in spans)

    def evaluate_obligation(self, obligation: EvidenceObligation,
                            spans: Sequence[EvidenceSpanV2]) -> Tuple[bool, Dict[str, Any]]:
        docs = {span.doc_id for span in spans}
        text = self._components(spans)
        required_docs = set(obligation.required_docs)
        doc_ok = required_docs.issubset(docs) if required_docs else len(docs) >= obligation.min_distinct_docs
        subject_hits = {
            subject: any(
                subject in span.text or any(subject in identity or identity in subject for identity in span.metadata.get("doc_identity_terms", []) if identity)
                for span in spans
            )
            for subject in obligation.required_subjects
        }
        field_hits = {
            field: any(alias in text for alias in aliases_for_field(field, adjudication=False))
            for field in obligation.required_fields
        }
        year_hits = {year: year in text for year in obligation.required_years}
        compact_text = re.sub(r"[\s,，]", "", text)
        value_hits = {value: re.sub(r"[\s,，]", "", value) in compact_text for value in obligation.required_values}
        table_heuristic = any(
            len(re.findall(r"20\d{2}年?|本期|上期|期末|期初", span.text)) >= 2
            and len(re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", span.text)) >= 3
            for span in spans
        )
        table_ok = not obligation.require_table_header or any(
            span.table_id or span.column_headers or span.metadata.get("table_header") for span in spans
        ) or table_heuristic
        unit_ok = not obligation.require_unit_context or any(
            span.unit_context or any(unit in span.text for unit in ["单位", "亿元", "万元", "%", "元/股"])
            for span in spans
        )
        full_scan_ok = not obligation.require_full_document_scan or (
            bool(spans) and all(span.metadata.get("full_scan_completed") for span in spans if span.doc_id in (required_docs or docs))
        )
        exception_ok = not obligation.require_exception_context or any(
            term in text for term in ["除外", "例外", "除非", "但", "如果", "若", "否则"]
        )
        formula_hits = {component: component in text for component in obligation.require_formula_components}

        kind = obligation.obligation_type
        if kind == "DOC_IDENTITY":
            satisfied = doc_ok
        elif kind in {"CROSS_DOC_PAIR", "UNIVERSAL_MULTI_DOC"}:
            satisfied = doc_ok and len(docs) >= obligation.min_distinct_docs and all(subject_hits.values()) and all(field_hits.values())
        elif kind == "FULL_DOCUMENT_ABSENCE":
            satisfied = doc_ok and full_scan_ok
        elif kind == "TABLE_HEADER_ROW_UNIT":
            satisfied = doc_ok and all(field_hits.values()) and all(year_hits.values()) and table_ok and unit_ok
        elif kind == "YEAR_VALUE_PAIR":
            satisfied = doc_ok and all(field_hits.values()) and len([value for value in year_hits.values() if value]) >= min(2, len(year_hits)) and table_ok and unit_ok
        elif kind == "RULE_CONDITION_EXCEPTION":
            satisfied = doc_ok and all(field_hits.values()) and exception_ok
        elif kind == "FORMULA_COMPONENTS":
            satisfied = doc_ok and all(subject_hits.values()) and all(field_hits.values()) and all(formula_hits.values())
        elif kind == "FIELD_VALUE":
            satisfied = doc_ok and all(subject_hits.values()) and all(field_hits.values()) and all(year_hits.values()) and all(value_hits.values())
        else:
            satisfied = doc_ok and all(subject_hits.values()) and all(field_hits.values()) and all(year_hits.values())

        return satisfied, {
            "satisfied": satisfied, "docs": sorted(docs), "doc_ok": doc_ok,
            "subject_hits": subject_hits, "field_hits": field_hits, "year_hits": year_hits, "value_hits": value_hits,
            "table_ok": table_ok, "unit_ok": unit_ok, "full_scan_ok": full_scan_ok,
            "exception_ok": exception_ok, "formula_hits": formula_hits,
        }

    def evaluate(self, obligations_by_claim: Dict[str, List[EvidenceObligation]],
                 claim_evidence: Dict[str, List[EvidenceSpanV2]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], float]:
        status = {}
        missing = []
        total = 0
        met = 0
        for claim_id, obligations in obligations_by_claim.items():
            spans = claim_evidence.get(claim_id, [])
            for obligation in obligations:
                total += 1
                satisfied, details = self.evaluate_obligation(obligation, spans)
                status[obligation.obligation_id] = details
                if satisfied:
                    met += 1
                else:
                    missing.append({
                        "claim_id": claim_id,
                        "obligation": obligation.to_dict(),
                        "details": details,
                    })
        return status, missing, met / max(1, total)

    @staticmethod
    def gap_request(missing: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
        requests: Dict[str, Dict[str, List[str]]] = {}
        for item in missing:
            claim_id = item["claim_id"]
            obligation = item["obligation"]
            details = item["details"]
            row = requests.setdefault(claim_id, {"required_docs": [], "extra_terms": []})
            row["required_docs"].extend(obligation.get("required_docs", []))
            row["extra_terms"].extend(
                key for key, hit in details.get("subject_hits", {}).items() if not hit
            )
            row["extra_terms"].extend(
                key for key, hit in details.get("field_hits", {}).items() if not hit
            )
            row["extra_terms"].extend(
                key for key, hit in details.get("year_hits", {}).items() if not hit
            )
            row["extra_terms"].extend(
                key for key, hit in details.get("formula_hits", {}).items() if not hit
            )
            row["extra_terms"].extend(
                key for key, hit in details.get("value_hits", {}).items() if not hit
            )
        for row in requests.values():
            row["required_docs"] = unique_strings(row["required_docs"])
            row["extra_terms"] = unique_strings(row["extra_terms"])
        return requests
