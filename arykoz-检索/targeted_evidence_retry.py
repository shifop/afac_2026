#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded full-document evidence retry for Stage A.1.

This is not a second retriever and does not use vectors.  It is activated only
for already-bound documents and claims whose evidence obligations explicitly
need historical/range/scope/controlling-shareholder context.  The scan returns
small auditable windows and never changes DocumentBinding.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from retrieval_models import EvidenceSpanV2, claim_attr

_TRIGGER_RE = re.compile(
    r"历史|最近三年|近三年|报告期各期|区间|至\s*\d|控股股东|实际控制人|"
    r"标的公司|合并口径|母公司口径|资产负债率|发行股份(?:及支付现金)?购买资产|募集配套资金|"
    r"身故保险金|退保|现金价值|个人账户价值|保单账户价值|免赔额|赔付比例|家庭共享|"
    r"共享免赔额|累计收益|已交保险费|养老年金|医疗费用|"
    r"违约|补偿|兑付日|回售|赎回|本金|利息|通知期限|报告出具|"
    r"保存期限|保密|不得向|差异报告|备案信息|识别核实|施行之日起|"
    r"市场规模|保费贡献率|营收同比|同比增长|复合增速|CAGR|"
    r"研发投入强度|研发投入占营业收入|研发投入总额占营业收入|研发费用占营业收入|"
    r"现金分红占|归母净利润|经营活动产生的现金流量净额|每10股|每股派发"
)
_LITERAL_TERMS = (
    "资产负债率", "合并口径", "母公司口径", "最近三年", "报告期各期",
    "标的公司", "控股股东", "实际控制人", "发行股份购买资产",
    "发行股份及支付现金购买资产", "募集配套资金", "净利润",
    "身故保险金", "退保费用比例", "现金价值", "个人账户价值", "保单账户价值",
    "免赔额", "赔付比例", "家庭共享", "共享免赔额", "累计收益", "累计已交保险费",
    "养老保险金", "医疗费用补偿",
    "违约", "违约责任", "补偿", "资产减值补偿", "兑付日", "回售", "赎回",
    "本金", "利息", "通知期限", "报告出具之日起", "保存期限", "客户身份资料",
    "保密", "不得向任何单位和个人提供", "差异报告", "备案信息", "识别核实",
    "施行之日起", "市场规模", "保费贡献率", "复合增长率", "CAGR", "营收同比增长",
    "研发投入金额", "研发投入占营业收入比例", "研发投入总额占营业收入比例",
    "研发费用占营业收入比例", "营业收入", "营业总收入",
    "经营活动产生的现金流量净额", "现金分红占归母净利润比例",
    "现金分红占合并报表归属于上市公司股东净利润的比例", "每10股派发现金红利",
)
_TEXT_TERM_STOP = {
    "受托管理人", "发行概况", "发行金额", "主体评级", "主承销商",
    "工作日", "个月内", "应当", "不得", "可以", "有权", "义务",
    "文档", "文件", "报告", "公司", "相关条款",
}
_RELATIVE_NOISE = re.compile(r"第一份|第二份|第一篇|第二篇|前者|后者|两份文档|两篇文档|文档|文件")
_NUMBER_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|元|%|％|天|日|年|个月|月|倍)?")


def _unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _claim_terms(claim: Any) -> Tuple[List[str], List[str]]:
    raw = str(claim_attr(claim, "raw", "") or "")
    terms: List[str] = []
    for value in claim_attr(claim, "field_candidates", []) or []:
        value = _RELATIVE_NOISE.sub("", str(value or "")).strip()
        if len(value) >= 2:
            terms.append(value)
    for value in claim_attr(claim, "subject_candidates", []) or []:
        value = _RELATIVE_NOISE.sub("", str(value or "")).strip()
        if len(value) >= 3:
            terms.append(value)
    for term in _LITERAL_TERMS:
        if term in raw:
            terms.append(term)
    # ClaimCompiler already produces option-local lexical terms.  Reuse only
    # specific terms here; generic domain boilerplate would create query drift.
    for value in claim_attr(claim, "keywords", []) or []:
        value = _RELATIVE_NOISE.sub("", str(value or "")).strip()
        if 2 <= len(value) <= 24 and value not in _TEXT_TERM_STOP:
            terms.append(value)
    numbers: List[str] = []
    for number in claim_attr(claim, "numbers", []) or []:
        token = str(claim_attr(number, "raw", "") or "").replace(" ", "")
        if token:
            numbers.append(token.replace("％", "%"))
    if not numbers:
        numbers = [x.replace(" ", "").replace("％", "%") for x in _NUMBER_TOKEN_RE.findall(raw)]
    return _unique(terms)[:14], _unique(numbers)[:8]


def _path_for(doc_map: Mapping[str, str], domain: str, doc_id: str) -> str:
    for key in (f"{domain}:{doc_id}", doc_id):
        path = str(doc_map.get(key, "") or "")
        if path and os.path.exists(path):
            return path
    return ""


def _candidate_positions(text: str, terms: Sequence[str], numbers: Sequence[str]) -> List[int]:
    positions: List[int] = []
    formula_priority = {
        "160%", "75%", "100%", "基本保险金额", "个人账户价值", "保单账户价值",
        "累计已交保险费", "累计已领取养老年金", "共享免赔额", "年度免赔额",
        "资产负债率", "控股股东", "发行股份购买资产",
        "研发投入占营业收入比例", "研发投入总额占营业收入比例",
        "经营活动产生的现金流量净额", "现金分红占归母净利润比例",
    }
    primary = sorted(list(terms) or list(numbers), key=lambda term: (0 if term in formula_priority or "%" in term else 1, -len(term)))
    for term in primary:
        start = 0
        per_term = 0
        while term and per_term < 12 and len(positions) < 160:
            idx = text.find(term, start)
            if idx < 0:
                break
            positions.append(idx)
            per_term += 1
            start = idx + max(1, len(term))
    # Exact decimals are high-value even when the field label is separated.
    for token in numbers:
        bare = token.replace("%", "").replace("％", "")
        if "." not in bare:
            continue
        idx = text.find(bare)
        if idx >= 0:
            positions.append(idx)
    return sorted(set(positions))


def _score_window(window: str, terms: Sequence[str], numbers: Sequence[str], raw: str) -> float:
    """Rank evidence-bearing prose/table windows above TOC and nearby noise.

    Exact values and obligation co-occurrence are intentionally dominant.  The
    scan is a bounded evidence retry, not a generic relevance retriever, so a
    window containing the requested 43.24% or a three-year ratio series must
    outrank an early table-of-contents hit containing only the field name.
    """
    score = 0.0
    normalized_window = window.replace("％", "%")
    exact_hits = 0
    for term in terms:
        if term in window:
            score += 6.0 if term in {
                "资产负债率", "标的公司", "控股股东", "实际控制人",
                "发行股份购买资产", "发行股份及支付现金购买资产", "募集配套资金",
            } else 2.5
    for token in numbers:
        normalized = token.replace("％", "%").replace(" ", "")
        compact_window = re.sub(r"\s+", "", normalized_window)
        if normalized and normalized in compact_window:
            score += 42.0
            exact_hits += 1
        else:
            bare = re.sub(r"(?:亿元|万元|元|%|天|日|年|个月|月|倍)$", "", normalized)
            if bare and bare in normalized_window:
                score += 14.0
    percent_values = re.findall(r"(?<!\d)\d+(?:\.\d+)?%", normalized_window)
    historical_signal = bool(re.search(r"最近三年|近三年|报告期各期|报告期内|历史数据|分别为", window))
    field_present = any(term in window for term in terms if len(term) >= 2)
    if historical_signal and field_present:
        score += 10.0
    ratio_obligation = bool("资产负债率" in raw or any("%" in x.replace("％", "%") for x in numbers))
    if ratio_obligation and ("历史" in raw or "最近三年" in raw or "区间" in raw) \
            and "资产负债率" in window and len(set(percent_values)) >= 2:
        score += 24.0 + min(10.0, 2.0 * len(set(percent_values)))
    if "资产负债率" in raw and "资产负债率" in window and len(set(percent_values)) >= 2:
        score += 18.0
    if "控股股东" in raw:
        if "标的公司" in window and "控股股东" in window:
            score += 24.0
        elif "控股股东" in window:
            score += 8.0
    legal_terms = [
        "违约", "补偿", "兑付日", "回售", "赎回", "本金", "利息",
        "保存期限", "客户身份资料", "保密", "差异报告", "识别核实",
        "报告出具之日起", "通知",
    ]
    if any(term in raw for term in legal_terms):
        hits = sum(1 for term in legal_terms if term in raw and term in window)
        score += 9.0 * hits
        if hits >= 2:
            score += 14.0
        if re.search(r"应当|不得|期限|日内|年内|责任|计算|基数", window):
            score += 8.0
    research_terms = ["市场规模", "保费贡献率", "复合增长率", "CAGR", "同比增长", "营收"]
    if any(term in raw for term in research_terms):
        hits = sum(1 for term in research_terms if term in raw and term in window)
        score += 8.0 * hits
        if any(token in window.replace("％", "%") for token in numbers):
            score += 12.0
    formula_terms = [
        "身故保险金", "现金价值", "个人账户价值", "保单账户价值",
        "退保费用比例", "免赔额", "赔付比例", "累计收益", "累计已交保险费",
    ]
    if any(term in raw for term in formula_terms):
        formula_hits = sum(1 for term in formula_terms if term in window)
        score += 8.0 * formula_hits
        if re.search(r"较大值|等于|计算|公式|乘以|扣除", window):
            score += 12.0
        if re.search(r"第[六七八九十0-9]+个?保单年度|第六年及以后", window):
            score += 8.0

    transaction_terms = [
        "发行股份购买资产", "发行股份及支付现金购买资产", "募集配套资金",
    ]
    if any(term in raw for term in transaction_terms):
        tx_hits = sum(1 for term in transaction_terms if term in window)
        score += 14.0 * tx_hits
        if tx_hits >= 2:
            score += 12.0
    # Typical Chinese PDF TOC lines contain long dot leaders and page numbers.
    # They are useful for routing but not admissible evidence.
    if re.search(r"[.。·…]{12,}", window) or window.count("...") >= 3:
        score -= 35.0
    if re.search(r"目录|目\s*录", window[:240]):
        score -= 12.0
    # Exact numeric anchors should never be displaced by a generic early hit.
    if exact_hits:
        score += 8.0 * exact_hits
    return score


def _scan_document(path: str, *, doc_id: str, claim_id: str, raw: str,
                   terms: Sequence[str], numbers: Sequence[str], max_spans: int) -> List[EvidenceSpanV2]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    positions = _candidate_positions(text, terms, numbers)
    rows: List[Tuple[float, int, int, str]] = []
    radius = 1400 if re.search(r"历史|最近三年|报告期各期|区间|资产负债率|身故保险金|退保|免赔额|赔付比例|现金价值|违约|补偿|兑付日|保存期限|差异报告|市场规模|保费贡献率|研发投入强度|研发投入占营业收入|现金分红占|经营活动产生的现金流量净额", raw) else 850
    for pos in positions:
        start = max(0, pos - radius)
        end = min(len(text), pos + radius)
        window = text[start:end].strip()
        score = _score_window(window, terms, numbers, raw)
        if score > 0:
            rows.append((score, start, end, window))
    rows.sort(key=lambda row: (-row[0], row[1]))
    selected: List[EvidenceSpanV2] = []
    occupied: List[Tuple[int, int]] = []
    content_seen = set()
    for score, start, end, window in rows:
        if any(max(start, a) < min(end, b) and (min(end, b) - max(start, a)) > 0.65 * min(end - start, b - a)
               for a, b in occupied):
            continue
        fingerprint = re.sub(r"\s+", "", window)[:1800]
        if fingerprint in content_seen:
            continue
        content_seen.add(fingerprint)
        occupied.append((start, end))
        selected.append(EvidenceSpanV2(
            evidence_id=f"targeted:{doc_id}:{claim_id}:{start}",
            doc_id=doc_id, text=window[:2600], score=100.0 + score,
            source="targeted_full_doc_scan", chunk_id=None,
            char_start=start, char_end=end,
            matched_terms=[term for term in terms if term in window][:10]
                          + [token for token in numbers if token.replace("％", "%") in window.replace("％", "%")][:6],
            section_path=[], page_hint=None, table_id=None, row_header=None,
            column_headers=[], unit_context=None,
            metadata={
                "retry_reason": "bound_document_evidence_obligation",
                "claim_id": claim_id,
                "source_path": path,
            },
        ))
        if len(selected) >= max_spans:
            break
    return selected


def augment_targeted_evidence(retrieval: Any, *, claims_by_option: Mapping[str, Sequence[Any]],
                              doc_map: Mapping[str, str], domain: str,
                              explicit_doc_ids: Sequence[str] = (),
                              claim_doc_targets: Mapping[str, Sequence[str]] | None = None,
                              question: str = "",
                              max_spans_per_claim: int = 2) -> Dict[str, Any]:
    """Prepend small full-document windows for explicit hard evidence gaps."""
    claim_evidence = getattr(retrieval, "claim_evidence", None)
    option_evidence = getattr(retrieval, "option_evidence", None)
    if not isinstance(claim_evidence, dict) or not isinstance(option_evidence, dict):
        return {"triggered_claims": 0, "added_spans": 0, "numeric_triggered_claims": 0, "non_numeric_triggered_claims": 0, "rows": []}
    audit_rows: List[Dict[str, Any]] = []
    triggered = 0
    added = 0
    numeric_triggered = 0
    non_numeric_triggered = 0
    explicit_doc_ids = _unique(str(x) for x in explicit_doc_ids if str(x))
    claim_doc_targets = dict(claim_doc_targets or {})
    for option_key, claims in claims_by_option.items():
        for claim in claims:
            raw = str(claim_attr(claim, "raw", "") or "")
            obligation_text = (str(question or "") + " " + raw).strip()
            claim_id = str(claim_attr(claim, "claim_id", ""))
            target_docs = _unique(str(x) for x in (claim_attr(claim, "doc_hints", []) or []) if str(x))
            target_docs = _unique(target_docs + [str(x) for x in claim_doc_targets.get(claim_id, []) if str(x)])
            identity_scoped = bool(target_docs)
            if not target_docs and explicit_doc_ids:
                # Option-level deterministic decomposition: each atomic claim
                # scans only explicitly named documents, not the whole corpus.
                target_docs = list(explicit_doc_ids)
                identity_scoped = True
            if not target_docs and bool(claim_attr(claim, "requires_multi_doc", False)):
                target_docs = list(getattr(retrieval, "doc_ids", []) or [])[:2]
            if not target_docs:
                continue
            trigger_match = bool(_TRIGGER_RE.search(obligation_text))
            # Explicit identity is itself a safe retry scope.  This extends the
            # retry from numeric obligations to legal clauses and report facts
            # without introducing a generic agent loop.
            if not trigger_match and not identity_scoped:
                continue
            terms, numbers = _claim_terms(claim)
            # Question-level inputs and formula cues are shared by every option
            # in calculation questions; add only literal, auditable terms.
            for term in _LITERAL_TERMS:
                if term in obligation_text and term not in terms:
                    terms.append(term)
            if "身故保险金" in obligation_text:
                for term in [
                    "保单账户价值", "基本保险金额", "个人账户价值", "160%",
                    "累计已交保险费", "累计已领取养老年金", "较大值", "领取日前",
                ]:
                    if term not in terms:
                        terms.append(term)
            if "退保" in obligation_text:
                for term in ["累计所交保险费", "保单账户累计收益", "75%", "退保费用比例", "现金价值"]:
                    if term not in terms:
                        terms.append(term)
            if "医疗费用" in obligation_text:
                for term in ["共享免赔额", "年度免赔额", "基本医疗保险", "赔付比例", "100%"]:
                    if term not in terms:
                        terms.append(term)
            formula_obligation = any(term in obligation_text for term in ["身故保险金", "退保", "医疗费用", "免赔额", "赔付比例"])
            if not formula_obligation:
                question_numeric_text = re.sub(
                    r"(?:fc[_\s]?)?text[_\s]?\d+|insurance[_\s]?\d+", "",
                    str(question or ""), flags=re.I,
                )
                for token in _NUMBER_TOKEN_RE.findall(question_numeric_text):
                    token = token.replace(" ", "").replace("％", "%")
                    if token and token not in numbers:
                        numbers.append(token)
            else:
                # Inputs and proposed answers do not appear in policy clauses.
                # Using them as exact scan anchors would pull unrelated 20-day
                # waiting-period text instead of the formula.
                numbers = []
            terms = terms[:18]
            numbers = numbers[:12]
            if not terms and not numbers:
                continue
            triggered += 1
            if numbers or re.search(r"\d|%|％", obligation_text):
                numeric_triggered += 1
            else:
                non_numeric_triggered += 1
            new_spans: List[EvidenceSpanV2] = []
            max_docs = 4 if domain == "insurance" else 2
            per_claim_budget = max(max_spans_per_claim, min(max_docs, len(target_docs)))
            for doc_id in target_docs[:max_docs]:
                path = _path_for(doc_map, domain, doc_id)
                if not path:
                    continue
                new_spans.extend(_scan_document(
                    path, doc_id=doc_id, claim_id=claim_id, raw=obligation_text,
                    terms=terms, numbers=numbers, max_spans=per_claim_budget,
                ))
            # Enforce a per-claim, not per-document, budget.  Preserve one
            # strongest span per target document, then fill remaining slots by
            # global score.  This prevents a two-document claim from injecting
            # eight retry windows ahead of the normal evidence context.
            new_spans.sort(key=lambda span: (-float(getattr(span, "score", 0.0) or 0.0), str(getattr(span, "doc_id", "")), int(getattr(span, "char_start", 0) or 0)))
            if len(new_spans) > per_claim_budget:
                selected_spans: List[EvidenceSpanV2] = []
                selected_ids = set()
                for doc_id in target_docs[:max_docs]:
                    best = next((span for span in new_spans if str(getattr(span, "doc_id", "")) == doc_id), None)
                    if best is not None:
                        selected_spans.append(best)
                        selected_ids.add(id(best))
                for span in new_spans:
                    if len(selected_spans) >= per_claim_budget:
                        break
                    if id(span) not in selected_ids:
                        selected_spans.append(span)
                        selected_ids.add(id(span))
                new_spans = selected_spans[:per_claim_budget]
            existing = list(claim_evidence.get(claim_id, []) or [])
            seen = set()
            merged: List[EvidenceSpanV2] = []
            for span in new_spans + existing:
                key = (str(getattr(span, "doc_id", "")), int(getattr(span, "char_start", 0) or 0),
                       str(getattr(span, "text", ""))[:120])
                if key in seen:
                    continue
                seen.add(key)
                merged.append(span)
            claim_evidence[claim_id] = merged
            added += len(new_spans)
            audit_rows.append({
                "option": str(option_key), "claim_id": claim_id,
                "trigger_reason": "lexical_obligation" if trigger_match else "explicit_identity_scope",
                "target_docs": target_docs[:max_docs], "terms": terms,
                "numbers": numbers, "added": len(new_spans),
                "evidence_ids": [span.evidence_id for span in new_spans],
            })
    # Rebuild per-option lists so targeted spans are visible before the 8-span
    # reasoning context cut.
    for option_key, claims in claims_by_option.items():
        rows: List[EvidenceSpanV2] = []
        seen = set()
        for claim in claims:
            for span in claim_evidence.get(str(claim_attr(claim, "claim_id", "")), []) or []:
                key = str(getattr(span, "evidence_id", ""))
                if key and key not in seen:
                    seen.add(key)
                    rows.append(span)
        option_evidence[str(option_key)] = rows
    audit = {
        "triggered_claims": triggered,
        "added_spans": added,
        "numeric_triggered_claims": numeric_triggered,
        "non_numeric_triggered_claims": non_numeric_triggered,
        "rows": audit_rows,
    }
    getattr(retrieval, "audit", {}).setdefault("targeted_evidence_retry", audit)
    return audit


__all__ = ["augment_targeted_evidence"]
