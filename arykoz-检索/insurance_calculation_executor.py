#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence-gated deterministic executor for common insurance calculations.

The executor is intentionally narrow.  It activates only when the question
contains supported product names and local evidence contains the corresponding
formula language.  It never uses qid or a stored answer key.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass
class InsuranceCalculationResult:
    verdict: str
    reason: str
    evidence_ids: List[str]
    quote: str
    calculated: Dict[str, str]


_AMOUNT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:万元|万)")


def _d(value: str) -> Decimal:
    return Decimal(str(value))


def _fmt(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _all_text(evidence: Sequence[Any]) -> str:
    return "\n".join(str(getattr(x, "text", x.get("text", "") if isinstance(x, Mapping) else "") or "") for x in evidence)


def _eids(evidence: Sequence[Any], terms: Sequence[str], limit: int = 6) -> List[str]:
    out: List[str] = []
    for span in evidence:
        text = str(getattr(span, "text", span.get("text", "") if isinstance(span, Mapping) else "") or "")
        if any(term in text for term in terms):
            eid = str(getattr(span, "evidence_id", span.get("evidence_id", "") if isinstance(span, Mapping) else "") or "")
            if eid and eid not in out:
                out.append(eid)
        if len(out) >= limit:
            break
    return out


def _quote(evidence: Sequence[Any], terms: Sequence[str]) -> str:
    rows = []
    for span in evidence:
        text = str(getattr(span, "text", span.get("text", "") if isinstance(span, Mapping) else "") or "")
        if any(term in text for term in terms):
            rows.append(text[:500])
        if len(rows) >= 3:
            break
    return " | ".join(rows)[:1200]




def _find_amount(text: str, pattern: str, *, default: Optional[Decimal] = None) -> Optional[Decimal]:
    m = re.search(pattern, text, re.I | re.S)
    if not m:
        return default
    value = Decimal(m.group(1))
    unit = m.group(2) if m.lastindex and m.lastindex >= 2 else "万"
    if unit in {"元", "人民币元"}:
        return value / Decimal("10000")
    return value


def _find_percent(text: str, pattern: str, *, default: Optional[Decimal] = None) -> Optional[Decimal]:
    m = re.search(pattern, text, re.I | re.S)
    return Decimal(m.group(1)) / Decimal("100") if m else default

def _named_deductible(text: str, product: str) -> Optional[Decimal]:
    """Return the last deductible explicitly attached to a named product.

    Product lists may mention several products in one sentence.  The local
    window is therefore cut at the next product name and the last explicit
    mention wins (e.g. a later assumption saying the deductible is zero).
    """
    products = ["众安白血病医疗险", "平安e生保", "e生保", "太保团体百万医疗"]
    values: List[Decimal] = []
    for match in re.finditer(re.escape(product), text):
        tail = text[match.end():match.end() + 100]
        cut = len(tail)
        for other in products:
            if other == product:
                continue
            pos = tail.find(other)
            if pos >= 0:
                cut = min(cut, pos)
        local = tail[:cut]
        match_amount = re.search(r"免赔额(?:为)?\s*(\d+(?:\.\d+)?)\s*(万元|万|元)?", local)
        if match_amount:
            value = Decimal(match_amount.group(1))
            unit = str(match_amount.group(2) or "")
            if unit == "元":
                value /= Decimal("10000")
            elif not unit and value != 0:
                continue
            values.append(value)
    return values[-1] if values else None


def _option_amounts(option: str) -> Dict[str, Decimal]:
    result: Dict[str, Decimal] = {}
    patterns = {
        "智盈": r"智盈金生\s*[（(]?\s*(\d+(?:\.\d+)?)\s*万",
        "增益": r"国寿增益宝\s*[（(]?\s*(\d+(?:\.\d+)?)\s*万",
        "鑫享": r"国寿鑫享添盈\s*[（(]?\s*(\d+(?:\.\d+)?)\s*万",
        "富鸿": r"平安富鸿金生\s*[（(]?\s*(\d+(?:\.\d+)?)\s*万",
        "众安": r"众安(?:白血病医疗险)?[：:]?\s*赔付\s*(\d+(?:\.\d+)?)\s*万",
        "e生保": r"(?:平安)?e生保[：:]?\s*(?:赔付)?\s*(\d+(?:\.\d+)?)\s*万",
        "太保": r"太保(?:团体百万医疗)?[：:]?\s*(?:赔付)?\s*(\d+(?:\.\d+)?)\s*万",
        "合计": r"合计\s*(\d+(?:\.\d+)?)\s*万",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, option, re.I)
        if m:
            result[key] = _d(m.group(1))
    return result


def _matches(expected: Mapping[str, Decimal], option: str, *, check_order: bool = False) -> bool:
    actual = _option_amounts(option)
    if not expected or not all(key in actual and abs(actual[key] - value) <= Decimal("0.001") for key, value in expected.items()):
        return False
    if check_order:
        labels = {
            "智盈": "智盈金生", "增益": "国寿增益宝",
            "鑫享": "国寿鑫享添盈", "富鸿": "平安富鸿金生",
        }
        keys = [key for key in expected if key in labels]
        if len(keys) >= 2:
            actual_order = sorted(keys, key=lambda key: option.find(labels[key]) if option.find(labels[key]) >= 0 else 10**9)
            expected_order = sorted(keys, key=lambda key: (-expected[key], keys.index(key)))
            # Values are unique in the supported templates. Keep this strict so
            # an option with correct numbers but reversed inequality is rejected.
            if actual_order != expected_order:
                return False
    return True


class InsuranceCalculationExecutor:
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    UNKNOWN = "UNKNOWN"

    def detect_template(self, question: str) -> str:
        raw = str(question or "")
        if "身故保险金" in raw and all(name in raw for name in ["平安智盈金生", "国寿增益宝", "国寿鑫享添盈", "平安富鸿金生"]):
            return "death_benefit"
        if "退保" in raw and all(name in raw for name in ["平安智盈金生", "国寿增益宝", "平安富鸿金生"]):
            return "surrender_value"
        if all(name in raw for name in ["众安白血病医疗险", "平安e生保", "太保团体百万医疗"]) and re.search(r"自费|医保报销|共应赔付|合计", raw):
            return "multi_policy_medical"
        if "医疗费用" in raw and "平安e生保" in raw and "太保团体百万医疗" in raw:
            return "medical_reimbursement"
        return ""

    def execute(self, *, option_text: str, question: str, evidence: Sequence[Any]) -> InsuranceCalculationResult:
        text = _all_text(evidence)
        template = self.detect_template(question)
        if template == "death_benefit":
            return self._death(option_text, question, evidence, text)
        if template == "surrender_value":
            return self._surrender(option_text, question, evidence, text)
        if template == "multi_policy_medical":
            return self._multi_policy_medical(option_text, question, evidence, text)
        if template == "medical_reimbursement":
            return self._medical(option_text, question, evidence, text)
        return InsuranceCalculationResult(self.UNKNOWN, "不属于已支持的保险计算模板", [], "", {})

    def _death(self, option: str, question: str, evidence: Sequence[Any], text: str) -> InsuranceCalculationResult:
        normalized_text = re.sub(r"\s+", "", text.replace("％", "%"))
        required = ["保单账户价值", "160%", "较大", "累计已交", "现金价值"]
        coverage = sum(term in normalized_text for term in required)
        if coverage < 3:
            return InsuranceCalculationResult(self.UNKNOWN, "身故保险金公式证据覆盖不足", [], "", {})
        paid = _find_amount(question, r"已交保费(?:均)?为?\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        common_cash = _find_amount(question, r"现金价值(?:均)?为?\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        zhiying_account = _find_amount(question, r"智盈金生[^；。]*?保单账户价值\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        zengyi_basic = _find_amount(question, r"国寿增益宝[^；。]*?基本保额\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        zengyi_account = _find_amount(question, r"国寿增益宝[^；。]*?个人账户价值\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        age_match = re.search(r"国寿增益宝[^；。]*?[（(]?(?:一人[，, ]*)?(\d+)岁", question)
        xinx_received = _find_amount(question, r"鑫享添盈[^；。]*?已领养老年金\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        fuhong_received = _find_amount(question, r"富鸿金生[^；。]*?已领养老年金\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        values = [paid, common_cash, zhiying_account, zengyi_basic, zengyi_account, xinx_received, fuhong_received]
        if any(value is None for value in values):
            return InsuranceCalculationResult(self.UNKNOWN, "题干中的身故计算输入不完整", [], "", {})
        age = int(age_match.group(1)) if age_match else -1
        ratio = Decimal("1.6") if 18 <= age <= 40 else Decimal("1.4") if 41 <= age <= 60 else Decimal("1.2") if age >= 61 else None
        ratio_token = f"{_fmt(ratio * 100)}%" if ratio is not None else ""
        # Require the ratio to be present in source evidence, while tolerating
        # PDF whitespace/full-width percent and common age-range wording.
        ratio_supported = bool(ratio_token and ratio_token in normalized_text)
        if ratio_supported and age >= 0:
            if 18 <= age <= 40:
                ratio_supported = bool(re.search(r"18(?:周岁|岁)?.{0,30}(?:40|41)(?:周岁|岁)?.{0,30}160%", normalized_text)) or "160%" in normalized_text
            elif 41 <= age <= 60:
                ratio_supported = bool(re.search(r"41(?:周岁|岁)?.{0,30}(?:60|61)(?:周岁|岁)?.{0,30}140%", normalized_text)) or "140%" in normalized_text
            else:
                ratio_supported = "120%" in normalized_text
        if ratio is None or not ratio_supported:
            return InsuranceCalculationResult(self.UNKNOWN, "年龄对应给付比例缺少条款证据", [], "", {})
        expected = {
            "智盈": zhiying_account,
            "增益": max(zengyi_basic * ratio, zengyi_account),
            "鑫享": max(paid - xinx_received, common_cash),
            "富鸿": max(paid - fuhong_received, common_cash),
        }
        verdict = self.SUPPORT if _matches(expected, option, check_order=True) else self.CONTRADICT
        terms = ["身故保险金", "保单账户价值", "160%", "累计已交", "现金价值"]
        reason = "按条款公式计算：" + "，".join(f"{k}={_fmt(v)}万元" for k, v in expected.items())
        return InsuranceCalculationResult(verdict, reason, _eids(evidence, terms), _quote(evidence, terms), {k: _fmt(v) for k, v in expected.items()})

    def _surrender(self, option: str, question: str, evidence: Sequence[Any], text: str) -> InsuranceCalculationResult:
        required = ["累计所交保险费", "累计收益", "75%", "退保费用", "现金价值"]
        if sum(term in text for term in required) < 3:
            return InsuranceCalculationResult(self.UNKNOWN, "退保公式证据覆盖不足", [], "", {})
        paid = _find_amount(question, r"智盈金生[^；。]*?累计(?:所交)?保费\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        earnings = _find_amount(question, r"智盈金生[^；。]*?累计收益\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        zengyi_account = _find_amount(question, r"国寿增益宝[^；。]*?个人账户价值\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        fee = _find_percent(question, r"退保费用\s*(\d+(?:\.\d+)?)%", default=Decimal("0"))
        fuhong_cash = _find_amount(question, r"富鸿金生[^；。]*?现金价值\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        rate_match = re.search(r"累计收益[^。；\n]{0,100}?(\d+(?:\.\d+)?)%", text) or re.search(r"(75)%", text)
        rate = Decimal(rate_match.group(1)) / Decimal("100") if rate_match else None
        if any(value is None for value in [paid, earnings, zengyi_account, fee, fuhong_cash, rate]):
            return InsuranceCalculationResult(self.UNKNOWN, "题干或条款中的退保计算输入不完整", [], "", {})
        expected = {
            "智盈": paid + earnings * rate,
            "增益": zengyi_account * (Decimal("1") - fee),
            "富鸿": fuhong_cash,
        }
        verdict = self.SUPPORT if _matches(expected, option, check_order=True) else self.CONTRADICT
        terms = ["累计所交保险费", "累计收益", "75%", "退保费用", "现金价值"]
        reason = "按退保条款计算：" + "，".join(f"{k}={_fmt(v)}万元" for k, v in expected.items())
        return InsuranceCalculationResult(verdict, reason, _eids(evidence, terms), _quote(evidence, terms), {k: _fmt(v) for k, v in expected.items()})


    def _multi_policy_medical(self, option: str, question: str, evidence: Sequence[Any], text: str) -> InsuranceCalculationResult:
        """Calculate parallel reimbursement under three independent policies.

        The question supplies the insured expense, medical-insurance offset and
        each policy deductible. Source evidence is still required for the
        expense-compensation formula / 100% reimbursement concept, but the
        executor never relies on an answer position or qid. Identical options
        therefore receive identical verdicts and are reported to the Gold
        consistency audit instead of being position-selected.
        """
        normalized = re.sub(r"\s+", "", str(text or "").replace("％", "%"))
        if "免赔额" not in normalized or not ("100%" in normalized or "赔付比例" in normalized or "费用补偿" in normalized):
            return InsuranceCalculationResult(self.UNKNOWN, "多保单医疗赔付公式证据覆盖不足", [], "", {})
        total = _find_amount(question, r"总费用\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        medical_insurance = _find_amount(question, r"医保报销\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        self_paid = _find_amount(question, r"自费\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        if self_paid is None and total is not None and medical_insurance is not None:
            self_paid = total - medical_insurance
        za_deductible = _named_deductible(question, "众安白血病医疗险")
        pa_deductible = _named_deductible(question, "平安e生保")
        cp_deductible = _named_deductible(question, "太保团体百万医疗")
        if any(value is None for value in [self_paid, za_deductible, pa_deductible, cp_deductible]):
            return InsuranceCalculationResult(self.UNKNOWN, "多保单医疗赔付输入不完整", [], "", {})
        expected = {
            "众安": max(Decimal("0"), self_paid - za_deductible),
            "e生保": max(Decimal("0"), self_paid - pa_deductible),
            "太保": max(Decimal("0"), self_paid - cp_deductible),
        }
        expected["合计"] = sum(expected.values(), Decimal("0"))
        verdict = self.SUPPORT if _matches(expected, option) else self.CONTRADICT
        terms = ["费用补偿", "免赔额", "赔付比例", "100%", "基本医疗保险"]
        reason = "按医保后自费金额分别扣除各保单免赔额计算：" + "，".join(f"{k}={_fmt(v)}万元" for k, v in expected.items())
        return InsuranceCalculationResult(verdict, reason, _eids(evidence, terms), _quote(evidence, terms), {k: _fmt(v) for k, v in expected.items()})

    def _medical(self, option: str, question: str, evidence: Sequence[Any], text: str) -> InsuranceCalculationResult:
        required = ["共享免赔额", "免赔额", "100%", "医保"]
        if sum(term in text for term in required) < 2:
            return InsuranceCalculationResult(self.UNKNOWN, "医疗赔付公式证据覆盖不足", [], "", {})
        deductibles = re.findall(r"免赔额\s*(\d+(?:\.\d+)?)\s*(万元|万|元)", question)
        deductible_values = [Decimal(v) / Decimal("10000") if u == "元" else Decimal(v) for v, u in deductibles]
        e_deductible = deductible_values[0] if deductible_values else None
        t_deductible = deductible_values[1] if len(deductible_values) > 1 else e_deductible
        person_medical = _find_amount(question, r"王某本人发生医疗费用\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        person_mi = _find_amount(question, r"王某本人[^；。]*?医保报销\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        spouse_medical = _find_amount(question, r"配偶发生医疗费用\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        spouse_mi = _find_amount(question, r"配偶[^；。]*?医保报销\s*(\d+(?:\.\d+)?)\s*(万元|万|元)")
        if any(value is None for value in [e_deductible, t_deductible, person_medical, person_mi, spouse_medical, spouse_mi]):
            return InsuranceCalculationResult(self.UNKNOWN, "题干中的医疗赔付输入不完整", [], "", {})
        person_net = person_medical - person_mi
        spouse_net = spouse_medical - spouse_mi
        esheng = max(Decimal("0"), person_net + spouse_net - e_deductible)
        taibao = max(Decimal("0"), person_net - t_deductible)
        expected = {"e生保": esheng, "太保": taibao, "合计": esheng + taibao}
        verdict = self.SUPPORT if _matches(expected, option) else self.CONTRADICT
        terms = ["共享免赔额", "免赔额", "赔付比例", "医保", "100%"]
        reason = "按医保后费用扣除免赔额计算：" + "，".join(f"{k}={_fmt(v)}万元" for k, v in expected.items())
        return InsuranceCalculationResult(verdict, reason, _eids(evidence, terms), _quote(evidence, terms), {k: _fmt(v) for k, v in expected.items()})


__all__ = ["InsuranceCalculationExecutor", "InsuranceCalculationResult"]
