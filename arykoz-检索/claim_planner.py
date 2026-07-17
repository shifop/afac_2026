#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared retrieval planning built directly from the V6.1 ClaimCompiler."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from retrieval_models import EvidenceObligation, RetrievalPlan, claim_attr, unique_strings


class SharedClaimCompiler:
    def __init__(self, compiler: Any = None, fact_index: Any = None):
        if compiler is not None:
            self.compiler = compiler
            return
        try:
            from reasoning_engine import ClaimCompiler
        except ImportError as exc:
            raise ImportError(
                "V6.1 reasoning_engine.py must be on PYTHONPATH; retrieval V2 does not maintain a second Claim parser"
            ) from exc
        self.compiler = ClaimCompiler(fact_index=fact_index)

    def compile(self, question: str, options: Mapping[str, str], domain: str,
                answer_format: str, ordered_doc_ids: Sequence[str] = ()) -> Dict[str, List[Any]]:
        return self.compiler.compile_options(
            question, options, domain, answer_format,
            ordered_doc_ids=list(ordered_doc_ids or []),
        )


def _clean_subjects(values: Sequence[str], raw: str = "") -> List[str]:
    raw = str(raw or "")
    candidates: List[str] = []
    for original in values:
        value = str(original or "").strip()
        # Subjects inherited only from the question are not evidence obligations
        # for every option. Keep a compiler candidate only when it appears in the
        # atomic claim itself.
        if value and value not in raw:
            continue
        for marker in ["明确指定", "指定", "认为", "显示", "表明"]:
            if marker in value:
                value = value.split(marker, 1)[-1].strip()
        value = re.sub(r"^(?:第一份|第二份|前一份|后一份|两份|该份)?文档(?:的)?", "", value)
        generic_markers = ["下列关于", "关于公司", "年归属于上市公司", "该公司", "公司经营", "文档"]
        if value and len(value) <= 50 and not any(marker in value for marker in generic_markers):
            candidates.append(value)

    # Product-ranking options often express entities as 产品名(计算值).
    for match in re.finditer(r"(?:^|[><=])\s*([^()><=]{2,30}?)\s*[（(]", raw):
        value = match.group(1).strip(" ，、")
        if value:
            candidates.append(value)
    # Company/year form such as 宇信科技 2025年...
    match = re.match(r"\s*([一-鿿A-Za-z0-9·]{2,24})\s*20\d{2}年?", raw)
    if match:
        candidates.append(match.group(1))

    candidates = unique_strings(candidates)
    # If one candidate is a clean prefix of a noisy longer phrase, keep the
    # shorter entity (韩国寿险 rather than 韩国寿险银保渠道保费贡献率在).
    clean: List[str] = []
    for value in sorted(candidates, key=len):
        if any(existing in value for existing in clean):
            continue
        clean.append(value)
    return clean


class ObligationBuilder:
    FORMULA_TERMS = ["公式", "计算", "赔付金额", "身故保险金", "现金价值", "免赔额", "赔付比例"]
    EXCEPTION_TERMS = ["如果", "若", "当", "除", "例外", "但", "否则", "前提"]

    def build(self, claim: Any, domain: str) -> List[EvidenceObligation]:
        claim_id = str(claim_attr(claim, "claim_id", ""))
        raw = str(claim_attr(claim, "raw", "") or "")
        subjects = _clean_subjects(claim_attr(claim, "subject_candidates", []) or [], raw)
        fields = unique_strings(claim_attr(claim, "field_candidates", []) or [])
        years = unique_strings(claim_attr(claim, "years", []) or [])
        docs = unique_strings(claim_attr(claim, "doc_hints", []) or [])
        comparison_docs = unique_strings(claim_attr(claim, "comparison_doc_ids", []) or [])
        comparator = str(claim_attr(claim, "comparator", "EQ") or "EQ")
        universal_signal = bool(re.search(r"均|全部|所有(?!人)|两份|双方|二者", raw))
        universal = bool(claim_attr(claim, "universal", False)) and universal_signal
        compiler_multi_doc = bool(claim_attr(claim, "requires_multi_doc", False))
        raw_multi_signal = any(mark in raw for mark in ["两份", "双方", "二者", "相比", "高于", "低于", "早于", "晚于", ">", "<", "="])
        multi_doc = bool(comparison_docs or len(docs) >= 2 or (len(subjects) >= 2 and raw_multi_signal))
        literal_absence = bool(claim_attr(claim, "literal_absence", False))
        numbers = list(claim_attr(claim, "numbers", []) or [])
        non_year_numbers = []
        for number in numbers:
            unit = number.get("unit", "") if isinstance(number, dict) else getattr(number, "unit", "")
            if str(unit) != "年":
                non_year_numbers.append(number)
        required_values = []
        for number in non_year_numbers:
            required_values.append(str(number.get("raw", "") if isinstance(number, dict) else getattr(number, "raw", "")))

        obligations: List[EvidenceObligation] = []
        serial = 0

        def add(kind: str, **kwargs) -> None:
            nonlocal serial
            serial += 1
            obligations.append(EvidenceObligation(
                obligation_id=f"{claim_id}:O{serial}", claim_id=claim_id,
                obligation_type=kind, **kwargs,
            ))

        if docs:
            add("DOC_IDENTITY", required_docs=docs, min_distinct_docs=len(docs))
        if literal_absence:
            add("FULL_DOCUMENT_ABSENCE", required_docs=docs,
                required_fields=fields, require_full_document_scan=True,
                min_distinct_docs=max(1, len(docs)))
        else:
            add("SUBJECT_FIELD", required_docs=docs, required_subjects=subjects,
                required_fields=fields, required_years=years)

        # Insurance calculation results are derived values; source clauses may
        # contain formula components rather than the option's computed number.
        if non_year_numbers and not (domain == "insurance" and len(subjects) >= 2):
            add("FIELD_VALUE", required_docs=docs, required_subjects=subjects,
                required_fields=fields, required_years=years, required_values=required_values)

        if comparator in {"UP", "DOWN"} or len(years) >= 2:
            add("YEAR_VALUE_PAIR", required_docs=docs, required_subjects=subjects,
                required_fields=fields, required_years=years,
                require_table_header=domain == "financial_reports",
                require_unit_context=domain == "financial_reports")

        required_multi_docs = comparison_docs or docs
        if multi_doc or universal:
            add("UNIVERSAL_MULTI_DOC" if universal else "CROSS_DOC_PAIR",
                required_docs=required_multi_docs, required_subjects=subjects,
                required_fields=fields, required_years=years,
                min_distinct_docs=max(2, len(required_multi_docs)))

        if domain == "financial_reports" and fields and (years or numbers or comparator != "EQ"):
            add("TABLE_HEADER_ROW_UNIT", required_docs=docs, required_subjects=subjects,
                required_fields=fields, required_years=years,
                require_table_header=True, require_unit_context=True)

        if domain in {"insurance", "regulatory", "financial_contracts"} and any(term in raw for term in self.EXCEPTION_TERMS):
            add("RULE_CONDITION_EXCEPTION", required_docs=docs, required_subjects=subjects,
                required_fields=fields, require_exception_context=True)

        if domain == "insurance" and (any(term in raw for term in self.FORMULA_TERMS) or len(subjects) >= 2):
            components = [term for term in ["保险金额", "现金价值", "免赔额", "赔付比例", "已交保费"] if term in raw]
            formula_fields = fields or (["身故保险金"] if len(subjects) >= 2 else [])
            add("FORMULA_COMPONENTS", required_docs=docs, required_subjects=subjects,
                required_fields=formula_fields, require_formula_components=components,
                min_distinct_docs=max(1, min(len(subjects), 4)))
        return obligations


class RetrievalPlanBuilder:
    def __init__(self, compiler: SharedClaimCompiler, obligation_builder: Optional[ObligationBuilder] = None):
        self.compiler = compiler
        self.obligation_builder = obligation_builder or ObligationBuilder()

    def build(self, *, qid: str, question: str, options: Mapping[str, str], domain: str,
              answer_format: str, ordered_doc_ids: Sequence[str] = (),
              hard_doc_ids: Sequence[str] = (),
              claims_by_option: Optional[Dict[str, List[Any]]] = None) -> RetrievalPlan:
        claims = claims_by_option or self.compiler.compile(question, options, domain, answer_format, ordered_doc_ids)
        obligations: Dict[str, List[EvidenceObligation]] = {}
        hard_docs: List[str] = list(ordered_doc_ids or []) + list(hard_doc_ids or [])
        subject_groups: List[List[str]] = []
        year_groups: List[List[str]] = []
        for option_claims in claims.values():
            for claim in option_claims:
                cid = str(claim_attr(claim, "claim_id", ""))
                obligations[cid] = self.obligation_builder.build(claim, domain)
                hard_docs.extend(claim_attr(claim, "doc_hints", []) or [])
                subjects = _clean_subjects(claim_attr(claim, "subject_candidates", []) or [], str(claim_attr(claim, "raw", "") or ""))
                years = unique_strings(claim_attr(claim, "years", []) or [])
                if subjects:
                    subject_groups.append(subjects)
                if years:
                    year_groups.append(years)
        return RetrievalPlan(
            question_id=qid or hashlib.sha1(question.encode("utf-8")).hexdigest()[:12],
            question=question, options={str(k): str(v) for k, v in options.items()},
            domain=domain, answer_format=answer_format,
            claims_by_option=claims, hard_doc_ids=unique_strings(hard_docs),
            subject_groups=subject_groups, year_groups=year_groups,
            obligations=obligations, ordered_doc_ids=list(ordered_doc_ids or []),
        )
