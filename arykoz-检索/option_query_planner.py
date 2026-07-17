#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Option/claim-level query decomposition for AFAC Stage B1-Lite.

The planner is deliberately deterministic and answer-blind.  It creates a
small set of lexical BM25/query channels from the already compiled ClaimSpec.
No embedding, vector database, answer key, qid heuristic, or external model is
used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence

from field_ontology import CLAUSE_TERMS, aliases_for_field
from retrieval_models import claim_attr, unique_strings


@dataclass(frozen=True)
class QueryVariant:
    query_id: str
    text: str
    weight: float
    purpose: str
    terms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_GENERIC = {
    "文档", "文件", "报告", "公司", "关于", "以下", "正确", "错误", "分别",
    "两份", "一份", "第一份", "第二份", "前者", "后者", "其中", "以及",
    "应当", "可以", "不得", "有权", "是否", "数值", "情况", "相关",
}

_CONDITION_TERMS = [
    "如果", "若", "当", "在", "自", "截至", "至少", "不超过", "超过", "未",
    "但", "除外", "前提", "条件", "导致", "发现", "完成", "提前", "之日起",
]

_DOMAIN_TERMS = {
    "financial_reports": ["合并口径", "母公司口径", "归属于上市公司股东", "同比", "报告期", "年度报告", "单位"],
    "financial_contracts": ["发行人", "本期债券", "本次债券", "违约", "补偿", "兑付", "回售", "赎回", "本金", "利息"],
    "insurance": ["保险责任", "身故保险金", "现金价值", "个人账户价值", "保单账户价值", "基本保险金额", "赔付比例", "免赔额", "退保"],
    "regulatory": ["应当", "不得", "报告", "提交", "备案", "审批", "期限", "施行", "保存", "保密", "例外"],
    "research": ["预计", "预测", "市场规模", "复合增长率", "CAGR", "同比增长", "营收", "保费", "出货量", "渗透率"],
}


def _number_raw(number: Any) -> str:
    if isinstance(number, dict):
        return str(number.get("raw", "") or "")
    return str(getattr(number, "raw", "") or "")


def _clean_term(value: Any) -> str:
    text = str(value or "").strip(" ，。；：:()（）[]【】\n\t")
    text = re.sub(r"^(?:第一份|第二份|前者|后者|两份)(?:文档|文件)?(?:中|的)?", "", text)
    return text


def _specific_terms(values: Iterable[Any], *, min_len: int = 2, max_len: int = 48) -> List[str]:
    out: List[str] = []
    for value in values:
        text = _clean_term(value)
        if not text or text in _GENERIC or len(text) < min_len or len(text) > max_len:
            continue
        out.append(text)
    return unique_strings(out)


def _raw_phrases(raw: str) -> List[str]:
    phrases: List[str] = []
    for token in re.findall(r"\d+(?:\.\d+)?\s*(?:亿元|万元|元|%|％|个工作日|工作日|日|天|年|个月|月|倍)?", raw):
        token = token.replace(" ", "").replace("％", "%")
        if token:
            phrases.append(token)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.+\-]{1,24}|[\u4e00-\u9fff]{2,16}", raw):
        if token not in _GENERIC:
            phrases.append(token)
    return unique_strings(phrases)


class OptionQueryPlanner:
    """Build at most four deterministic query channels per atomic claim."""

    def __init__(self, *, max_variants: int = 4, max_terms: int = 18):
        self.max_variants = max(2, int(max_variants))
        self.max_terms = max(8, int(max_terms))

    def _base_terms(self, claim: Any, domain: str) -> Dict[str, List[str]]:
        raw = str(claim_attr(claim, "raw", "") or "")
        subjects = _specific_terms(claim_attr(claim, "subject_candidates", []) or [], min_len=2)
        fields: List[str] = []
        for field in claim_attr(claim, "field_candidates", []) or []:
            fields.extend(aliases_for_field(str(field), adjudication=False))
        fields = _specific_terms(fields, min_len=2)
        years = _specific_terms(claim_attr(claim, "years", []) or [], min_len=4, max_len=8)
        numbers = _specific_terms((_number_raw(n).replace(" ", "").replace("％", "%") for n in claim_attr(claim, "numbers", []) or []), min_len=1, max_len=24)
        keywords = _specific_terms(claim_attr(claim, "keywords", []) or [], min_len=2, max_len=24)
        clauses: List[str] = []
        for category in CLAUSE_TERMS.values():
            clauses.extend(term for term in category if term in raw)
        clauses.extend(term for term in _DOMAIN_TERMS.get(domain, []) if term in raw)
        conditions = [term for term in _CONDITION_TERMS if term in raw]
        return {
            "subjects": unique_strings(subjects),
            "fields": unique_strings(fields),
            "years": unique_strings(years),
            "numbers": unique_strings(numbers),
            "keywords": unique_strings(keywords),
            "clauses": unique_strings(clauses),
            "conditions": unique_strings(conditions),
            "raw_phrases": _raw_phrases(raw),
        }

    @staticmethod
    def _join(terms: Sequence[str]) -> str:
        return " ".join(unique_strings(terms))

    def build(self, claim: Any, domain: str, *, question: str = "", option_text: str = "") -> List[QueryVariant]:
        claim_id = str(claim_attr(claim, "claim_id", "") or "claim")
        raw = str(claim_attr(claim, "raw", "") or option_text or "")
        parts = self._base_terms(claim, domain)
        variants: List[QueryVariant] = []

        def add(suffix: str, terms: Sequence[str], weight: float, purpose: str) -> None:
            clean = unique_strings(_specific_terms(terms, min_len=1, max_len=64))[:self.max_terms]
            text = self._join(clean)
            if not text:
                return
            signature = tuple(clean)
            if any(tuple(row.terms) == signature for row in variants):
                return
            variants.append(QueryVariant(f"{claim_id}:{suffix}", text, float(weight), purpose, clean))

        # 1) Full atomic claim retains condition and word order.
        add("raw", [raw], 1.0, "atomic_claim")

        # 2) Identity + field + time is the highest precision general channel.
        add(
            "identity_field",
            parts["subjects"] + parts["fields"] + parts["years"] + parts["conditions"] + parts["clauses"],
            1.45,
            "identity_field_time",
        )

        # 3) Exact numeric/metric channel prevents generic theme words from
        # suppressing the one report or table containing the requested value.
        if parts["numbers"] or parts["years"]:
            add(
                "value",
                parts["subjects"] + parts["fields"] + parts["years"] + parts["numbers"],
                1.65,
                "exact_value_metric",
            )

        # 4) Domain-specialized clause/formula channel.
        special: List[str] = []
        if domain == "insurance":
            special.extend(parts["subjects"])
            if "身故保险金" in (question + raw):
                special.extend([
                    "身故保险金额", "身故给付比例", "基本保险金额", "个人账户价值",
                    "保单账户价值", "累计已交保险费", "累计已领取养老年金", "较大值",
                    "18周岁", "41周岁", "61周岁", "160%", "140%", "120%",
                ])
            elif "退保" in (question + raw):
                special.extend(["退保费用比例", "现金价值", "个人账户价值", "累计收益", "累计所交保险费", "75%"])
            elif "医疗费用" in (question + raw) or "免赔额" in (question + raw):
                special.extend(["共享免赔额", "年度免赔额", "基本医疗保险", "赔付比例", "100%", "医疗费用"])
        elif domain == "regulatory":
            special.extend(parts["subjects"] + parts["clauses"] + parts["conditions"] + parts["numbers"])
            special.extend(term for term in ["应当", "不得", "报告", "提交", "备案", "施行之日起", "保存期限", "例外"] if term in (question + raw))
        elif domain == "research":
            special.extend(parts["subjects"] + parts["years"] + parts["fields"] + parts["numbers"])
            special.extend(term for term in ["预计", "预测", "市场规模", "复合增长率", "CAGR", "同比增长", "营收", "保费贡献率", "出货量"] if term in (question + raw))
        elif domain == "financial_contracts":
            special.extend(parts["subjects"] + parts["fields"] + parts["numbers"] + parts["clauses"])
        elif domain == "financial_reports":
            special.extend(parts["subjects"] + parts["years"] + parts["fields"] + parts["numbers"])
            special.extend(term for term in ["合并口径", "母公司口径", "归属于上市公司股东", "同比"] if term in (question + raw))
        add("domain", special, 1.8, f"{domain}_specialized")

        # Keep a compact deterministic set.  The raw channel is always first;
        # remaining variants are ordered by weight descending.
        if len(variants) > self.max_variants:
            raw_rows = [v for v in variants if v.purpose == "atomic_claim"][:1]
            rest = sorted((v for v in variants if v.purpose != "atomic_claim"), key=lambda v: (-v.weight, v.query_id))
            variants = raw_rows + rest[: self.max_variants - len(raw_rows)]
        return variants


__all__ = ["OptionQueryPlanner", "QueryVariant"]
