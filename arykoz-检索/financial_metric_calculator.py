#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence-gated deterministic financial calculations for AFAC B1.1.

This module intentionally covers a small set of arithmetic operations that are
safe to execute from annual-report evidence. It never guesses a company,
period, metric or unit. Unsupported or ambiguous cases return UNKNOWN and fall
through to the normal verifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SUPPORT = "SUPPORT"
CONTRADICT = "CONTRADICT"
UNKNOWN = "UNKNOWN"


@dataclass
class FinancialCalculationResult:
    verdict: str
    reason: str
    evidence_ids: List[str]
    quote: str
    calculated: Dict[str, str]
    operation: str = ""
    reliability_passed: bool = False
    reliability_reason: str = ""
    bindings: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.bindings is None:
            self.bindings = {}


_NUM = r"[-+]?\(?\d[\d,]*(?:\.\d+)?\)?"
_UNIT_SCALE = {
    "": Decimal("1"), "元": Decimal("1"), "人民币元": Decimal("1"),
    "千元": Decimal("1000"), "万元": Decimal("10000"),
    "百万元": Decimal("1000000"), "亿元": Decimal("100000000"),
}

METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "revenue": ("营业收入", "营业总收入"),
    "profit": ("归属于上市公司股东的净利润", "归属于母公司股东的净利润", "归母净利润"),
    "operating_cashflow": ("经营活动产生的现金流量净额", "经营活动现金流量净额", "经营活动产生的现金流量净"),
    "rd_amount": ("研发投入金额", "研发投入合计", "研发投入总额", "研发费用"),
    "cash_dividend": ("合计拟派发现金红利", "拟派发现金红利总额", "现金分红总额", "现金分红金额"),
}

RATIO_ALIASES: Dict[str, Tuple[str, ...]] = {
    "rd_intensity": (
        "研发投入占营业收入比例", "研发投入总额占营业收入比例",
        "研发费用占营业收入比例", "研发投入占营收比例", "研发费用率",
    ),
    "cash_dividend_ratio": (
        "现金分红占合并报表归属于上市公司股东净利润的比例",
        "现金分红占归属于上市公司股东净利润的比例",
        "现金分红占归母净利润比例", "现金分红占净利润比例",
    ),
}


def _get(row: Any, name: str, default: Any = "") -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _norm(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).replace("％", "%")


def _dec(raw: Any) -> Optional[Decimal]:
    text = str(raw or "").replace(",", "").strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return -value if negative else value


def _doc_year(doc_id: str) -> int:
    match = re.search(r"20\d{2}", str(doc_id or ""))
    return int(match.group(0)) if match else 0


def _issuer_key(doc_id: str) -> str:
    return re.sub(r"^annual_|_20\d{2}_report$", "", str(doc_id or ""))


# row = (doc_id, year, value, unit, evidence_id, quote)
MetricRow = Tuple[str, int, Decimal, str, str, str]


class FinancialMetricCalculator:
    def __init__(self) -> None:
        self._audit: Dict[str, Any] = {
            "attempts": 0, "support": 0, "contradict": 0, "unknown": 0,
            "operations": {}, "ambiguous_rejected": 0,
            "reliability_passed": 0, "reliability_rejected": 0,
            "scope_incomplete_rejected": 0, "subject_year_rejected": 0,
            "semantic_operation_rejected": 0, "outlier_rejected": 0,
            "shared_evidence_filtered": 0,
        }
        self._context: Dict[str, Any] = {}

    def audit(self) -> Dict[str, Any]:
        return {**self._audit, "operations": dict(self._audit["operations"])}

    @staticmethod
    def _labels(evidence: Sequence[Any]) -> Dict[str, List[str]]:
        labels: Dict[str, List[str]] = {}
        for span in evidence:
            did = str(_get(span, "doc_id", "") or "")
            if not did:
                continue
            vals = labels.setdefault(did, [did, _issuer_key(did)])
            meta = dict(_get(span, "metadata", {}) or {})
            for key in ("issuer", "company", "title"):
                if meta.get(key):
                    vals.append(str(meta[key]))
            vals.extend(str(x) for x in meta.get("doc_identity_terms", []) or [] if str(x))
        return {did: list(dict.fromkeys(x for x in vals if len(_norm(x)) >= 2)) for did, vals in labels.items()}

    @staticmethod
    def _unit_before(text: str, pos: int, alias: str) -> str:
        # Prefer a unit attached to the row label, then a nearby table header.
        attached = re.search(re.escape(alias) + r"\s*[（(]\s*(人民币元|百万元|千元|万元|亿元|元)\s*[）)]", text[pos:pos + len(alias) + 30])
        if attached:
            return attached.group(1)
        before = text[max(0, pos - 180):pos]
        headers = re.findall(r"单位\s*[:：]\s*(人民币元|百万元|千元|万元|亿元|元)", before)
        return headers[-1] if headers else ""

    @classmethod
    def _metric_rows(cls, evidence: Sequence[Any], metric: str) -> List[MetricRow]:
        aliases = METRIC_ALIASES[metric]
        rows: List[MetricRow] = []
        for span in evidence:
            did = str(_get(span, "doc_id", "") or "")
            text = str(_get(span, "text", "") or "").replace("％", "%")
            eid = str(_get(span, "evidence_id", "") or "")
            if not did or not text:
                continue
            year = _doc_year(did)
            for alias in aliases:
                for match in re.finditer(re.escape(alias), text):
                    tail = text[match.end():match.end() + 180]
                    # Only accept the first table-like number close to the metric.
                    value_match = re.search(r"^[^\d()]{0,45}(" + _NUM + r")\s*(人民币元|百万元|千元|万元|亿元|元)?", tail)
                    if not value_match:
                        continue
                    value = _dec(value_match.group(1))
                    if value is None:
                        continue
                    explicit = str(value_match.group(2) or "")
                    unit = explicit or cls._unit_before(text, match.start(), alias)
                    # Unitless small values are almost always page/section numbers.
                    if not unit and abs(value) < Decimal("10000"):
                        continue
                    normalized = value * _UNIT_SCALE.get(unit, Decimal("1"))
                    quote = text[max(0, match.start() - 90):match.end() + 150]
                    rows.append((did, year, normalized, unit, eid, quote))
                    break
        # Keep one strongest current-period value per doc. Prefer explicit units,
        # larger magnitudes and shorter table-like quotes.
        best: Dict[str, MetricRow] = {}
        for row in rows:
            old = best.get(row[0])
            score = (bool(row[3]), abs(row[2]), -len(row[5]))
            old_score = (bool(old[3]), abs(old[2]), -len(old[5])) if old else None
            if old is None or score > old_score:
                best[row[0]] = row
        return list(best.values())

    @classmethod
    def _ratio_rows(cls, evidence: Sequence[Any], ratio_name: str) -> List[MetricRow]:
        aliases = RATIO_ALIASES[ratio_name]
        rows: List[MetricRow] = []
        for span in evidence:
            did = str(_get(span, "doc_id", "") or "")
            text = str(_get(span, "text", "") or "").replace("％", "%")
            eid = str(_get(span, "evidence_id", "") or "")
            if not did or not text:
                continue
            current_year = _doc_year(did)
            for alias in aliases:
                for match in re.finditer(re.escape(alias), text):
                    window = text[match.end():match.end() + 180]
                    pcts = re.findall(r"(-?\d+(?:\.\d+)?)\s*%", window)
                    if not pcts:
                        continue
                    quote = text[max(0, match.start() - 100):match.end() + 180]
                    rows.append((did, current_year, Decimal(pcts[0]) / Decimal("100"), "%", eid, quote))
                    # Annual tables often contain current and prior year values.
                    header = text[max(0, match.start() - 160):match.start()]
                    years = [int(x) for x in re.findall(r"20\d{2}", header)]
                    if len(pcts) >= 2 and len(set(years)) >= 2:
                        prior = sorted(set(years), reverse=True)[1]
                        rows.append((did, prior, Decimal(pcts[1]) / Decimal("100"), "%", eid, quote))
                    break
        # Deduplicate exact doc/year/value rows.
        output: List[MetricRow] = []
        seen = set()
        for row in rows:
            key = (row[0], row[1], row[2])
            if key not in seen:
                seen.add(key)
                output.append(row)
        return output

    @classmethod
    def _derived_ratios(cls, evidence: Sequence[Any], ratio_name: str) -> List[MetricRow]:
        direct = cls._ratio_rows(evidence, ratio_name)
        by_doc_year: Dict[Tuple[str, int], MetricRow] = {(r[0], r[1]): r for r in direct}
        if ratio_name == "rd_intensity":
            numerators = cls._metric_rows(evidence, "rd_amount")
            denominators = cls._metric_rows(evidence, "revenue")
        else:
            numerators = cls._metric_rows(evidence, "cash_dividend")
            denominators = cls._metric_rows(evidence, "profit")
        for num in numerators:
            for den in denominators:
                if num[0] != den[0] or den[2] == 0:
                    continue
                numerator_value, denominator_value = num[2], den[2]
                # Evidence cards may omit a repeated table-level unit on one
                # row.  When exactly one row carries a known unit, treat the
                # unitless row as using the same table scale.  This is safe
                # within one annual-report document and is audited in the
                # returned evidence pair.
                if not num[3] and den[3] in _UNIT_SCALE:
                    numerator_value *= _UNIT_SCALE[den[3]]
                elif num[3] in _UNIT_SCALE and not den[3]:
                    denominator_value *= _UNIT_SCALE[num[3]]
                ratio = numerator_value / denominator_value
                # When unit propagation produces an implausible ratio,
                # try again without unit scaling.  R&D intensity and
                # cash-dividend ratios are bounded proportions; a value
                # over 150 % almost always signals a unit-mismatch
                # rather than a genuine financial extreme.
                if ratio <= 0 or ratio > Decimal("1.5"):
                    fallback = num[2] / den[2]
                    if Decimal("0") < fallback <= Decimal("1.5"):
                        ratio = fallback
                        numerator_value, denominator_value = num[2], den[2]
                    else:
                        continue
                key = (num[0], num[1])
                # Prefer an explicit direct ratio when available.
                if key not in by_doc_year:
                    by_doc_year[key] = (
                        num[0], num[1], ratio, "%",
                        num[4] or den[4], (num[5] + " | " + den[5])[:600],
                    )
        return list(by_doc_year.values())

    @staticmethod
    def _subject_order(raw: str, rows: Sequence[MetricRow], labels: Mapping[str, Sequence[str]]) -> List[MetricRow]:
        compact = _norm(raw)
        def position(row: MetricRow) -> Tuple[int, int, str]:
            found = [compact.find(_norm(term)) for term in labels.get(row[0], []) if _norm(term) and compact.find(_norm(term)) >= 0]
            return (min(found) if found else 10**9, -row[1], row[0])
        return sorted(rows, key=position)

    @staticmethod
    def _named_doc_ids(raw: str, labels: Mapping[str, Sequence[str]]) -> List[str]:
        compact = _norm(raw)
        return [did for did, terms in labels.items() if any(_norm(term) in compact for term in terms if len(_norm(term)) >= 2)]

    @staticmethod
    def _comparison(raw: str, left: Decimal, right: Decimal) -> Optional[bool]:
        if re.search(r"不低于|至少", raw): return left >= right
        if re.search(r"不超过|至多", raw): return left <= right
        if re.search(r"高于|大于|超过|更高|优于", raw): return left > right
        if re.search(r"低于|小于|不足|更低", raw): return left < right
        return None

    @staticmethod
    def _claimed_percent(raw: str) -> Optional[Decimal]:
        matches = re.findall(r"(?<!20)(\d+(?:\.\d+)?)\s*%", raw.replace("％", "%"))
        return Decimal(matches[-1]) / Decimal("100") if matches else None

    @staticmethod
    def _growth_intent(raw: str) -> bool:
        return bool(re.search(r"增速|增长率|同比(?:增长|下降|增幅|降幅)?|双位数增长|较.{0,16}(?:增长|下降)", raw))

    @staticmethod
    def _ratio_intent(raw: str) -> bool:
        return bool(re.search(r"占.{0,18}(?:比例|比重)|强度|率(?:高于|低于|上升|下降|提高|降低)", raw))

    @staticmethod
    def _candidate_allowed_docs(claim: Any, raw: str, labels: Mapping[str, Sequence[str]],
                                explicit_allowed: Sequence[str]) -> List[str]:
        allowed = [str(x) for x in explicit_allowed if str(x)]
        claim_docs = [str(x) for x in list(getattr(claim, "doc_hints", []) or [])
                      + list(getattr(claim, "comparison_doc_ids", []) or []) if str(x)]
        if claim_docs:
            allowed = [x for x in allowed if x in claim_docs] or claim_docs
        named = FinancialMetricCalculator._named_doc_ids(raw, labels)
        if named:
            allowed = [x for x in allowed if x in named] or named
        return list(dict.fromkeys(allowed))

    @staticmethod
    def _named_subject_year_pairs(raw: str, labels: Mapping[str, Sequence[str]],
                                  allowed_docs: Sequence[str]) -> Dict[str, List[int]]:
        compact = _norm(raw)
        years = [int(x) for x in re.findall(r"20\d{2}", compact)]
        pairs: Dict[str, List[int]] = {}
        named = FinancialMetricCalculator._named_doc_ids(raw, labels)
        for did in named or list(allowed_docs):
            terms = [t for t in labels.get(did, []) if len(_norm(t)) >= 2]
            positions = [compact.find(_norm(t)) for t in terms if compact.find(_norm(t)) >= 0]
            local: List[int] = []
            doc_year = _doc_year(did)
            if years and doc_year in years:
                local = [doc_year]
            elif positions and years:
                pos = min(positions)
                year_positions = [(abs(m.start() - pos), int(m.group(0))) for m in re.finditer(r"20\d{2}", compact)]
                if year_positions:
                    local = [min(year_positions)[1]]
            if not local and len(years) == 1:
                local = years[:]
            pairs[did] = local
        return pairs

    def _validate_reliability(self, operation: str, verdict: str,
                              rows: Sequence[MetricRow], calculated: Dict[str, str]) -> Tuple[bool, str, Dict[str, Any]]:
        ctx = dict(self._context or {})
        raw = str(ctx.get("raw", "") or "")
        allowed = set(ctx.get("allowed_doc_ids", []) or [])
        scope_complete = bool(ctx.get("scope_complete", True))
        labels = dict(ctx.get("labels", {}) or {})
        bindings: Dict[str, Any] = {
            "allowed_doc_ids": sorted(allowed),
            "used_doc_ids": sorted({r[0] for r in rows}),
            "used_doc_years": sorted({f"{r[0]}:{r[1]}" for r in rows}),
            "operation": operation,
        }
        if verdict not in {SUPPORT, CONTRADICT}:
            return False, "calculation_not_decisive", bindings
        if not scope_complete:
            self._audit["scope_incomplete_rejected"] += 1
            return False, "financial_scope_incomplete", bindings
        if not rows:
            return False, "no_bound_input_rows", bindings
        if allowed and any(r[0] not in allowed for r in rows):
            self._audit["subject_year_rejected"] += 1
            return False, "used_document_outside_question_scope", bindings
        if self._growth_intent(raw) and operation in {"revenue_comparison", "cashflow_revenue_ratio"}:
            self._audit["semantic_operation_rejected"] += 1
            return False, "growth_claim_cannot_use_absolute_amount_operation", bindings
        if self._ratio_intent(raw) and operation == "revenue_comparison":
            self._audit["semantic_operation_rejected"] += 1
            return False, "ratio_claim_cannot_use_absolute_revenue_operation", bindings
        # Every numeric row must be tied to the annual-report year encoded in
        # its document id. This blocks a 2025 evidence card from serving a 2024
        # clause merely because both years are printed in the same table.
        for row in rows:
            doc_year = _doc_year(row[0])
            comparative_ratio_row = bool(
                row[3] == "%" and row[1] and str(row[1]) in str(row[5] or "")
            )
            if doc_year and row[1] and doc_year != row[1] and not comparative_ratio_row:
                self._audit["subject_year_rejected"] += 1
                return False, f"row_year_{row[1]}_does_not_match_document_year_{doc_year}", bindings
        pairs = self._named_subject_year_pairs(raw, labels, sorted(allowed))
        bindings["required_subject_year_pairs"] = pairs
        used_pairs = {(r[0], r[1]) for r in rows}
        for did, years in pairs.items():
            if did not in {r[0] for r in rows}:
                self._audit["subject_year_rejected"] += 1
                return False, f"named_subject_missing:{did}", bindings
            if years and not any((did, year) in used_pairs for year in years):
                self._audit["subject_year_rejected"] += 1
                return False, f"named_subject_year_missing:{did}:{years}", bindings
        if operation in {"rd_intensity", "cash_dividend_ratio"}:
            vals = [r[2] for r in rows]
            if any(v <= 0 or v > Decimal("1") for v in vals):
                self._audit["outlier_rejected"] += 1
                return False, "ratio_outside_0_to_100_percent", bindings
        if operation == "cashflow_revenue_ratio":
            vals = []
            for value in calculated.values():
                try: vals.append(abs(Decimal(str(value))))
                except Exception: pass
            if not vals or any(v > Decimal("0.8") for v in vals):
                self._audit["outlier_rejected"] += 1
                return False, "cashflow_revenue_ratio_outlier_or_missing", bindings
        # A comparison/trend must use distinct bound facts rather than one row
        # duplicated into both sides.
        if len(rows) >= 2:
            identities = {(r[0], r[1], r[2], r[4]) for r in rows}
            if len(identities) < len(rows):
                self._audit["subject_year_rejected"] += 1
                return False, "duplicate_value_record_used_for_multiple_sides", bindings
        return True, "all_subject_year_metric_unit_operation_gates_passed", bindings

    def _finish(self, operation: str, verdict: str, reason: str,
                rows: Sequence[MetricRow], calculated: Dict[str, str]) -> FinancialCalculationResult:
        self._audit["operations"][operation or "none"] = int(self._audit["operations"].get(operation or "none", 0)) + 1
        reliable, reliability_reason, bindings = self._validate_reliability(operation, verdict, rows, calculated)
        if verdict in {SUPPORT, CONTRADICT} and not reliable:
            verdict = UNKNOWN
            reason = f"可靠性门禁拒绝确定性裁决：{reliability_reason}；原计算={reason}"
            self._audit["reliability_rejected"] += 1
        elif reliable:
            self._audit["reliability_passed"] += 1
        self._audit["support" if verdict == SUPPORT else "contradict" if verdict == CONTRADICT else "unknown"] += 1
        eids: List[str] = []
        quotes: List[str] = []
        for row in rows:
            if row[4] and row[4] not in eids: eids.append(row[4])
            if row[5] and row[5] not in quotes: quotes.append(row[5])
        return FinancialCalculationResult(
            verdict, reason, eids[:8], " | ".join(quotes)[:1800], calculated,
            operation, reliable, reliability_reason, bindings,
        )

    def execute(self, *, claim: Any, option_text: str, question: str,
                evidence: Sequence[Any], allowed_doc_ids: Optional[Sequence[str]] = None,
                scope_complete: bool = True) -> FinancialCalculationResult:
        self._audit["attempts"] += 1
        raw = str(getattr(claim, "raw", "") or option_text or "")
        initial_labels = self._labels(evidence)
        allowed = self._candidate_allowed_docs(claim, raw, initial_labels, list(allowed_doc_ids or []))
        if allowed:
            before = len(evidence)
            evidence = [span for span in evidence if str(_get(span, "doc_id", "") or "") in set(allowed)]
            self._audit["shared_evidence_filtered"] += max(0, before - len(evidence))
        labels = self._labels(evidence)
        self._context = {
            "raw": raw,
            "allowed_doc_ids": allowed,
            "scope_complete": bool(scope_complete),
            "labels": labels,
        }
        full = raw + " " + str(question or "")
        if not scope_complete:
            return self._finish("scope_gate", UNKNOWN, "题干主体-年份作用域未完整解析，禁止确定性财务计算", [], {})

        ratio_name = ""
        if re.search(r"研发(?:投入|费用).{0,10}(?:强度|占.{0,8}(?:营业收入|营收)(?:的)?.{0,5}(?:比例|比重))", raw):
            ratio_name = "rd_intensity"
        elif "研发投入强度" in raw:
            ratio_name = "rd_intensity"
        elif "现金分红" in raw and re.search(r"占.{0,16}(?:归母|归属于上市公司股东|净利润).{0,8}(?:比例|比重)", raw):
            ratio_name = "cash_dividend_ratio"

        if ratio_name:
            operation = ratio_name
            rows = self._derived_ratios(evidence, ratio_name)
            named = self._named_doc_ids(raw, labels)
            if named:
                rows = [r for r in rows if r[0] in named]
            years = [int(x) for x in re.findall(r"20\d{2}", raw)]
            claimed = self._claimed_percent(raw)
            calculated = {f"{r[0]}:{r[1]}": str(r[2]) for r in rows}
            used: List[MetricRow] = []
            verdict = UNKNOWN
            reason = "派生比例缺少唯一主体、期间或指标证据"

            # Direct numeric assertion for one named document.
            current_rows = [r for r in rows if not years or r[1] in years]
            if claimed is not None and len(current_rows) == 1:
                used = current_rows
                tolerance = Decimal("0.0015")  # 0.15 percentage point
                verdict = SUPPORT if abs(current_rows[0][2] - claimed) <= tolerance else CONTRADICT
                reason = f"{operation}计算值={current_rows[0][2]*100:.4f}%，题述={claimed*100:.4f}%"
            # Cross-company comparison.
            elif len(named) >= 2:
                by_doc = []
                for did in named:
                    candidates = [r for r in rows if r[0] == did]
                    if years:
                        candidates = [r for r in candidates if r[1] in years]
                    if candidates:
                        by_doc.append(sorted(candidates, key=lambda r: r[1], reverse=True)[0])
                if len(by_doc) == len(named):
                    ordered = self._subject_order(raw, by_doc, labels)
                    result = self._comparison(raw, ordered[0][2], ordered[1][2])
                    if result is not None:
                        used = ordered
                        verdict = SUPPORT if result else CONTRADICT
                        reason = f"{operation}比较：{ordered[0][0]}={ordered[0][2]*100:.4f}%，{ordered[1][0]}={ordered[1][2]*100:.4f}%"
            # Same-company two-period trend.
            elif re.search(r"上升|提升|提高|增长|下降|下滑|减少", raw):
                by_issuer: Dict[str, List[MetricRow]] = {}
                for row in rows:
                    by_issuer.setdefault(_issuer_key(row[0]), []).append(row)
                groups = [g for g in by_issuer.values() if len({r[1] for r in g}) >= 2]
                if len(groups) == 1:
                    ordered = sorted(groups[0], key=lambda r: r[1], reverse=True)
                    newest = ordered[0]
                    older = next((r for r in ordered[1:] if r[1] < newest[1]), None)
                    if older is not None:
                        expected_up = not bool(re.search(r"下降|下滑|减少|降低", raw))
                        actual_up = newest[2] > older[2]
                        used = [newest, older]
                        verdict = SUPPORT if actual_up == expected_up else CONTRADICT
                        reason = f"{operation}期间比较：{newest[1]}={newest[2]*100:.4f}%，{older[1]}={older[2]*100:.4f}%"
            if verdict == UNKNOWN:
                self._audit["ambiguous_rejected"] += 1
            return self._finish(operation, verdict, reason, used, calculated)

        # Operating cash flow divided by revenue.
        if re.search(r"经营活动.*现金流.{0,16}(?:营业收入|营收)", full) and re.search(r"一半|十分之一|百分之|比例|比重", full):
            operation = "cashflow_revenue_ratio"
            cash = self._metric_rows(evidence, "operating_cashflow")
            revenue = self._metric_rows(evidence, "revenue")
            ratios: Dict[str, Tuple[Decimal, List[MetricRow]]] = {}
            for c in cash:
                for r in revenue:
                    if c[0] == r[0] and r[2] != 0:
                        value = c[2] / r[2]
                        if Decimal("-2") < value < Decimal("2"):
                            ratios[c[0]] = (value, [c, r])
            named = self._named_doc_ids(raw, labels)
            checks: List[bool] = []
            used: List[MetricRow] = []
            calculated: Dict[str, str] = {}
            compact = _norm(raw)
            for did in named or ratios.keys():
                if did not in ratios: continue
                pos = min([compact.find(_norm(t)) for t in labels.get(did, []) if compact.find(_norm(t)) >= 0] or [0])
                clause = compact[pos:pos + 100]
                threshold = Decimal("0.5") if "一半" in clause else Decimal("0.1") if "十分之一" in clause else None
                if threshold is None: continue
                value, ev = ratios[did]
                expected = value < threshold if "低于" in clause else value > threshold if re.search(r"超过|高于", clause) else None
                if expected is not None:
                    checks.append(expected); used.extend(ev); calculated[did] = str(value)
            verdict = SUPPORT if checks and all(checks) else CONTRADICT if checks else UNKNOWN
            reason = "经营现金流/营业收入：" + "，".join(f"{k}={Decimal(v)*100:.4f}%" for k,v in calculated.items()) if checks else "缺少可唯一配对的经营现金流和营业收入"
            if verdict == UNKNOWN: self._audit["ambiguous_rejected"] += 1
            return self._finish(operation, verdict, reason, used, calculated)

        # Growth-rate claims require period-delta facts. They must never be
        # answered by comparing absolute revenue amounts.
        if self._growth_intent(raw) and re.search(r"营业收入|营收|净利润|现金流", raw):
            self._audit["semantic_operation_rejected"] += 1
            return self._finish(
                "unsupported_growth_rate", UNKNOWN,
                "增长率/增速命题缺少安全的两期增长率计算模板，交回Qwen核验",
                [], {},
            )

        # Cross-company revenue comparison, including a multiple threshold.
        if re.search(r"营业收入|营收规模|同期营收", raw) and re.search(r"两倍|大于|高于|超过|低于", raw):
            operation = "revenue_comparison"
            rows = self._metric_rows(evidence, "revenue")
            named = self._named_doc_ids(raw, labels)
            by_doc = [next((r for r in rows if r[0] == did), None) for did in named]
            by_doc = [r for r in by_doc if r is not None]
            verdict = UNKNOWN; reason = "缺少两家公司的唯一营业收入证据"; used: List[MetricRow] = []
            calculated = {r[0]: str(r[2]) for r in by_doc}
            if len(by_doc) == 2:
                ordered = self._subject_order(raw, by_doc, labels)
                if "两倍" in raw:
                    ok = ordered[0][2] > ordered[1][2] * Decimal("2")
                else:
                    cmp = self._comparison(raw, ordered[0][2], ordered[1][2]); ok = cmp if cmp is not None else None
                if ok is not None:
                    verdict = SUPPORT if ok else CONTRADICT; used = ordered
                    reason = f"营业收入比较：{ordered[0][0]}={ordered[0][2]}元，{ordered[1][0]}={ordered[1][2]}元"
            if verdict == UNKNOWN: self._audit["ambiguous_rejected"] += 1
            return self._finish(operation, verdict, reason, used, calculated)

        return self._finish("none", UNKNOWN, "不属于安全的财务派生计算模板", [], {})


__all__ = ["FinancialMetricCalculator", "FinancialCalculationResult"]
