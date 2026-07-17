#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
insurance_rules.py — V5 Stage 2 保险域规则引擎（0-token 预判定）

在调LLM前，对ins域题目先走通用保险条款规则做0-token预判定。
严格遵守：
- C6 规则优先：简单险责题、明显错误选项走规则直接判CONTRADICTED
- 保守原则：宁可UNKNOWN交LLM，绝不误判SUPPORTED；CONTRADICTED必须有明确原文字符串命中
- 不做产品特判（不硬编码某具体产品的等待期天数），只做通用条款逻辑
- 输出 {option_key: ('SUPPORTED'|'CONTRADICTED'|'UNKNOWN', rule_name, evidence_snippet)}
- UNKNOWN选项才交给LLM判
- 规则命中的选项在option_sources里标记为'INS_RULE'，并在notes里写明命中规则和原文位置

规则覆盖：
a. 等待期规则：选项涉及"等待期内赔付/给付保险金"且条款明确等待期天数+等待期内免责→CONTRADICTED
b. 犹豫期规则：犹豫期内退保=退还保费（扣工本费），选项说"扣除手续费"→CONTRADICTED
c. 免责条款关键词命中：选项提到的赔付情形在"责任免除"章节明确列出→CONTRADICTED
d. 免赔额/赔付比例规则：赔付金额 ≤（损失-免赔额）×赔付比例；选项说"全额赔付"在有免赔额/比例<100%时→CONTRADICTED
e. 数值计算检查：对计算题选项，用正则提取原文数字做简单算术验证（赔付=（损失-免赔）×比例，不超过保额）
f. 法定免责情形：酒驾/无证/故意/战争/核辐射/自杀2年内等常规免责→若选项说赔→CONTRADICTED
"""
from __future__ import annotations
import re
from typing import Dict, Tuple, List, Any, Optional


# ──────────────────────────────────────
# 关键词/模式库（通用条款逻辑，不特判具体产品）
# ──────────────────────────────────────

# f. 法定/常规免责情形（保险行业通用免责）—— 若选项说这些情形"赔付/给付保险金"→CONTRADICTED
#    注意：必须同时满足"条款责任免除章节明确列出"+"选项说赔付"才触发
STATUTORY_EXCLUSION_KEYWORDS = [
    '酒后驾驶', '醉酒驾驶', '酒驾', '醉驾', '无合法有效驾驶证', '无驾驶证', '无证驾驶',
    '故意犯罪', '故意行为', '抗拒依法采取的刑事强制措施', '自杀', '故意自伤',
    '战争', '军事冲突', '暴乱', '武装叛乱', '核爆炸', '核辐射', '核污染',
    '潜水', '攀岩', '探险', '武术比赛', '摔跤比赛', '特技表演', '赛马', '赛车',
    '吸食', '注射毒品', '未遵医嘱', '私自服用', '斗殴',
    '艾滋病', 'HIV感染', '先天性疾病', '遗传性疾病',
    '妊娠', '流产', '分娩',  # 部分医疗险免责
]

# 赔付关键词
PAYOUT_KEYWORDS = ['赔付', '给付保险金', '赔偿保险金', '给付', '赔偿', '报销', '承担保险责任', '支付保险金']

# b. 犹豫期相关
HESITATION_KEYWORDS = ['犹豫期', '冷静期']
HESITATION_REFUND_PATTERNS = [
    # 犹豫期退保 → 退还 / 全额退还 / 扣除工本费后退还（放宽匹配：不强制"解除/退保"紧跟，只要犹豫期附近出现退费描述即可）
    (re.compile(r'犹豫期[^\n。；]{0,80}?退还\s*(所交|全部|已交|您(?:所|已)?交纳)?(?:的)?\s*保险费'), 'REFUND_PREMIUM'),
    (re.compile(r'犹豫期[^\n。；]{0,80}?扣除[^。；\n]{0,30}工本费'), 'REFUND_PREMIUM_DEDUCT_COST'),
    (re.compile(r'犹豫期[^。；\n]{0,50}?(解除|退保)[^。；\n]{0,200}?退还[^。；\n]{0,50}保险费'), 'REFUND_PREMIUM'),
]
HESITATION_WRONG_KEYWORDS = ['扣除手续费', '退还现金价值', '扣除保费']  # 这些是犹豫期后退保的做法

# a. 等待期相关
WAITING_PERIOD_EXCLUSION_PATTERNS = [
    re.compile(r'等待期[^\n。；]{0,80}?(不承担|不给付|不予|不赔付|免责|除外)'),
    re.compile(r'(?:自合同成立|生效)[^\n。；]{0,20}?(\d+)\s*(日|天)[^\n。；]{0,30}?(等待期|观察期)[^\n。；]{0,40}?(不承担|不给付|不予)'),
]
WAITING_PERIOD_PAYOUT_CLAIM_HINTS = ['等待期内', '等待期 内', '观察期内']

# c. 责任免除章节定位
EXCLUSION_SECTION_PATTERNS = [
    # 模式1：标题格式"第X章/第X条 责任免除"，贪婪到下一个同级/高级标题
    re.compile(r'(?:^|\n)\s*(?:第[一二三四五六七八九十百千\d]+\s*[章节条])?\s*[^\n]{0,10}?责\s*任\s*免\s*除[\s\S]{0,8000}?(?=(?:^|\n)\s*(?:第[一二三四五六七八九十百千\d]+\s*[章节条])[^\n]*(?:保险责任|保险金额|保险费|犹豫期|等待期|合同|申请|释义|退保|现金价值)|\Z)', re.M),
    re.compile(r'(?:^|\n)\s*[（(]?\s*[二2三四五六七八1-9]+\s*[）)]?\s*责\s*任\s*免\s*除[\s\S]{0,8000}?(?=\n\s*[（(]?\s*[三四五六七八1-9]+\s*[）)]\s*\S|\Z)', re.M),
    # 模式2：直接匹配"责任免除"关键词后续8000字
    re.compile(r'责\s*任\s*免\s*除[\s\S]{0,8000}'),
]

# d. 免赔额/赔付比例
DEDUCTIBLE_PATTERNS = [
    re.compile(r'免赔额\s*(?:为|是)?\s*([\d,]+\.?\d*)\s*(万)?元'),
    re.compile(r'(?:年度|每次)?\s*免赔额\s*[:：]?\s*([\d,]+\.?\d*)\s*(万)?元'),
]
PAYOUT_RATIO_PATTERNS = [
    re.compile(r'赔付比例\s*(?:为|是)?\s*(\d{1,3})\s*%'),
    re.compile(r'(?:按|按照)\s*(\d{1,3})\s*%\s*(?:的比例)?\s*(?:给付|赔付|报销)'),
]
SUM_INSURED_PATTERNS = [
    re.compile(r'基本保险金额\s*(?:为|是)?\s*([\d,]+\.?\d*)\s*(万)?元'),
    re.compile(r'保险金额\s*(?:为|是)?\s*([\d,]+\.?\d*)\s*(万)?元'),
]

# 全额赔付关键词
FULL_PAYOUT_HINTS = ['全额赔付', '全额给付', '全额赔偿', '100%赔付', '100%报销', '全部报销', '全部赔付']


def _parse_amount(num_str: str, unit_wan: Optional[str]) -> float:
    """解析金额，统一为元为单位的数值。"""
    try:
        val = float(str(num_str).replace(',', ''))
    except Exception:
        return 0.0
    if unit_wan == '万':
        val *= 10000
    return val


def _extract_number(text: str) -> Optional[float]:
    """从字符串里抽第一个数字"""
    m = re.search(r'([\d,]+\.?\d*)', text)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except Exception:
            return None
    return None


def _find_section(context: str, section_names: List[str], max_chars: int = 3000) -> str:
    """从文本中定位指定章节的内容（贪婪匹配到下一章标题或结束）。"""
    for name in section_names:
        # 找标题位置
        pat = re.compile(r'(?:^|\n)\s*(?:第[一二三四五六七八九十百千\d]+\s*[章节条])?\s*[^\n。；]{0,10}?' + re.escape(name) + r'[^\n]{0,30}', re.M)
        m = pat.search(context)
        if m:
            start = m.start()
            # 向后取到下一个大标题
            tail = context[start + len(m.group()):start + len(m.group()) + max_chars]
            end_m = re.search(r'\n\s*(?:第[一二三四五六七八九十百千\d]+\s*[章节条]|[（(]?\s*[一二三四五六七八九十\d]+\s*[）)]\s*\S)', tail)
            if end_m:
                tail = tail[:end_m.start()]
            return m.group() + tail
    return ''


def _locate_exclusion_section(context: str) -> str:
    """定位"责任免除"章节内容。"""
    for pat in EXCLUSION_SECTION_PATTERNS:
        m = pat.search(context)
        if m:
            return m.group(0)
    # 简单兜底：搜"责任免除"关键词附近2000字
    idx = context.find('责任免除')
    if idx >= 0:
        return context[max(0, idx - 200):idx + 3000]
    return ''


def _rule_waiting_period(option_text: str, context: str) -> Optional[Tuple[str, str, str]]:
    """规则a：等待期内赔付→CONTRADICTED（仅在条款明确等待期+等待期不赔时触发）。"""
    # 选项是否提到等待期内赔付？
    has_waiting_payout = False
    if any(h in option_text for h in WAITING_PERIOD_PAYOUT_CLAIM_HINTS):
        if any(p in option_text for p in PAYOUT_KEYWORDS + ['承担保险责任', '保险责任', '赔付', '给付']):
            has_waiting_payout = True
    if not has_waiting_payout:
        # 更宽松：选项提到"等待期"+"赔付/给付/承担"
        if '等待期' in option_text and any(p in option_text for p in PAYOUT_KEYWORDS + ['承担', '给付', '赔付', '赔偿']):
            has_waiting_payout = True
    if not has_waiting_payout:
        return None

    # 条款里是否明确写了等待期不赔？
    for pat in WAITING_PERIOD_EXCLUSION_PATTERNS:
        m = pat.search(context)
        if m:
            snippet = m.group(0)[:120]
            return ('CONTRADICTED', 'WAITING_PERIOD_NO_PAYOUT',
                    f'条款明确等待期内不赔：{snippet}')
    return None


def _rule_hesitation(option_text: str, context: str) -> Optional[Tuple[str, str, str]]:
    """规则b：犹豫期退保错说成"扣除手续费/退还现金价值"→CONTRADICTED。"""
    if not any(h in option_text for h in HESITATION_KEYWORDS):
        return None
    if not any(kw in option_text for kw in ['退保', '解除合同', '解除']):
        return None

    # 选项主张错误的退费方式？
    option_wrong = any(kw in option_text for kw in HESITATION_WRONG_KEYWORDS)
    if not option_wrong:
        return None

    # 条款是否明确犹豫期退"保险费/扣除工本费"？
    for pat, _tag in HESITATION_REFUND_PATTERNS:
        m = pat.search(context)
        if m:
            snippet = m.group(0)[:150]
            return ('CONTRADICTED', 'HESITATION_REFUND_WRONG',
                    f'犹豫期退保条款规定：{snippet}')
    return None


def _rule_statutory_exclusion(option_text: str, context: str) -> Optional[Tuple[str, str, str]]:
    """规则f+c：选项提到的情形属于常规免责且在"责任免除"章节明确列出，却主张赔付→CONTRADICTED。"""
    # 选项是否主张赔付？
    if not any(p in option_text for p in PAYOUT_KEYWORDS + ['承担保险责任', '赔付', '赔偿', '给付']):
        return None
    # 否定的不算（如"不赔/不承担"是对的）
    if any(neg in option_text for neg in ['不赔', '不承担', '不予赔付', '不赔付', '不给付']):
        return None

    # 定位责任免除章节
    exclusion_sec = _locate_exclusion_section(context)
    if not exclusion_sec:
        return None

    # 选项提到了哪个免责情形？
    for kw in STATUTORY_EXCLUSION_KEYWORDS:
        if kw in option_text and kw in exclusion_sec:
            # 在责任免除章节找包含该关键词的原句；
            # 该句本身不必包含否定词（因为责任免除章节的标题已表明全章为"不赔"），
            # 只需要关键词所在句子的上下文总段落含有否定语义即可。
            ctx_window = 400  # 关键词前后窗口
            idx = exclusion_sec.find(kw)
            window = exclusion_sec[max(0, idx-50):idx+ctx_window]
            has_neg_signal = any(neg in window for neg in [
                '不承担', '不给付', '不予', '除外', '免责', '不赔偿', '不负',
                '责任免除', '不承担给付', '我们不', '本公司不',
            ])
            if has_neg_signal:
                snippet = kw + ' ... ' + window.split('\n')[0][:80]
                return ('CONTRADICTED', f'STATUTORY_EXCLUSION:{kw}',
                        f'责任免除章节明确列明（含"{kw}"）：{snippet[:150]}')
    return None


def _rule_full_payout_with_deductible(option_text: str, context: str) -> Optional[Tuple[str, str, str]]:
    """规则d：有免赔额/赔付比例<100%时，选项说"全额赔付"→CONTRADICTED。"""
    if not any(h in option_text for h in FULL_PAYOUT_HINTS):
        return None
    if not any(p in option_text for p in PAYOUT_KEYWORDS + ['报销', '赔付', '给付']):
        return None

    # 是否有免赔额？
    has_deductible = False
    ded_snippet = ''
    for pat in DEDUCTIBLE_PATTERNS:
        m = pat.search(context)
        if m:
            val = _parse_amount(m.group(1), m.group(2) if m.lastindex >= 2 else None)
            if val > 0:
                has_deductible = True
                ded_snippet = m.group(0)[:80]
                break

    # 是否赔付比例 < 100%？
    ratio_less_100 = False
    ratio_snippet = ''
    for pat in PAYOUT_RATIO_PATTERNS:
        m = pat.search(context)
        if m:
            try:
                r = int(m.group(1))
                if 0 < r < 100:
                    ratio_less_100 = True
                    ratio_snippet = m.group(0)[:80]
                    break
            except Exception:
                pass

    if has_deductible or ratio_less_100:
        reason_parts = []
        if ded_snippet:
            reason_parts.append(ded_snippet)
        if ratio_snippet:
            reason_parts.append(ratio_snippet)
        return ('CONTRADICTED', 'FULL_PAYOUT_WITH_DEDUCTIBLE',
                f'存在免赔/比例<100%，与全额赔付矛盾：{"; ".join(reason_parts)}')
    return None


def _rule_numeric_calc(option_text: str, context: str) -> Optional[Tuple[str, str, str]]:
    """规则e：简单数值计算验证——若选项是计算题答案且数字与（损失-免赔）×比例明显不符→CONTRADICTED。
    保守策略：只在所有数字都能从原文精确提取、且计算结果与选项差异>10%时判CONTRADICTED，否则UNKNOWN。
    """
    # 选项是否以一个数字作为赔付金额答案？
    opt_num_m = re.search(r'(?:赔付|赔偿|给付|报销|应支付|共)(?:金额|保险金)?\s*(?:为|是|等于|约为|≈)?\s*([\d,]+\.?\d*)\s*(万)?元', option_text)
    if not opt_num_m:
        # 选项结尾的数字+单位
        opt_num_m = re.search(r'([\d,]+\.?\d*)\s*(万)?元\s*[。.？?]?$', option_text.strip())
    if not opt_num_m:
        return None

    try:
        opt_val = _parse_amount(opt_num_m.group(1), opt_num_m.group(2) if opt_num_m.lastindex >= 2 else None)
    except Exception:
        return None
    if opt_val <= 0:
        return None

    # 从上下文提取：损失金额、免赔额、赔付比例、保额
    loss = None
    # 损失/费用提取：覆盖"医疗费用/损失/合理费用/支出/总费用/实际支出/发生费用/花费/住院费/住院共花/共花费/医疗费"
    loss_m = re.search(
        r'(?:医疗费用|医疗|住院费用|住院(?:共|共计|合计|花费|花了)?|损失|合理(?:且|的)?必要(?:的)?费用|合理费用|支出|总费用|实际支出|发生(?:的)?(?:医疗)?费用|共(?:花费|花了|计|合计|支出|发生|产生)|花费|花了)[^。；\n：:]*?(\d[\d,.]*)\s*(万)?元',
        context
    )
    if loss_m:
        loss = _parse_amount(loss_m.group(1), loss_m.group(2) if loss_m.lastindex >= 2 else None)

    ded = 0.0
    for pat in DEDUCTIBLE_PATTERNS:
        dm = pat.search(context)
        if dm:
            ded = _parse_amount(dm.group(1), dm.group(2) if dm.lastindex >= 2 else None)
            break

    ratio = 1.0
    for pat in PAYOUT_RATIO_PATTERNS:
        rm = pat.search(context)
        if rm:
            try:
                ratio = int(rm.group(1)) / 100.0
                break
            except Exception:
                pass

    insured = float('inf')  # 默认无上限
    for pat in SUM_INSURED_PATTERNS:
        sm = pat.search(context)
        if sm:
            insured = _parse_amount(sm.group(1), sm.group(2) if sm.lastindex >= 2 else None)
            break

    # 保守：必须能提取到损失金额才算
    if loss is None or loss <= 0:
        return None

    # 计算理论赔付
    expected = max(0.0, (loss - ded)) * ratio
    expected = min(expected, insured)

    if expected <= 0:
        return None

    # 与选项值比较，差异>10%且选项不是约数描述时判CONTRADICTED
    if '约' in option_text or '≈' in option_text or '大概' in option_text:
        tol = max(0.20, abs(expected) * 0.20)
    else:
        tol = max(0.10 * abs(expected), 100)  # 至少差100元或10%

    if abs(opt_val - expected) > tol and opt_val > expected * 1.1:
        # 选项明显算多了才CONTRADICTED（算少了可能是其他因素，UNKNOWN）
        return ('CONTRADICTED', 'CALC_MISMATCH',
                f'按(损失{loss:.0f}-免赔{ded:.0f})×比例{ratio:.0%}={expected:.0f}元(保额上限{insured:.0f})，选项={opt_val:.0f}元明显偏高')
    return None


# 规则按顺序执行，命中即返回
_RULES = [
    _rule_waiting_period,
    _rule_hesitation,
    _rule_statutory_exclusion,
    _rule_full_payout_with_deductible,
    _rule_numeric_calc,
]


def apply_insurance_rules(question_text: str,
                          options: Dict[str, str],
                          context: str) -> Dict[str, Dict[str, Any]]:
    """对ins域题目应用0-token规则引擎预判定。

    Args:
        question_text: 题干
        options: {A: text, B: text, ...}
        context: 压缩后的文档原文（已做V5 Stage 1上下文压缩）

    Returns:
        {option_key: {
            'verdict': 'SUPPORTED'|'CONTRADICTED'|'UNKNOWN',
            'rule': str 规则名（UNKNOWN时为空）,
            'evidence': str 原文证据片段（UNKNOWN时为空）,
        }}

    保守原则：
        - SUPPORTED：规则引擎**不**做SUPPORTED判定（避免误选），所有"疑似正确"都返回UNKNOWN交LLM
        - CONTRADICTED：只有明确原文命中才判
        - 未命中任何规则 → UNKNOWN
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not context or len(context) < 50:
        for k in (options or {}):
            results[k] = {'verdict': 'UNKNOWN', 'rule': '', 'evidence': ''}
        return results

    for opt_key, opt_text in (options or {}).items():
        verdict = 'UNKNOWN'
        hit_rule = ''
        evidence = ''
        for rule_fn in _RULES:
            try:
                hit = rule_fn(opt_text, context)
            except Exception as e:
                # 规则执行异常不影响整体
                hit = None
            if hit:
                v, r, ev = hit
                # 规则引擎**只判CONTRADICTED**，不直接判SUPPORTED
                if v == 'CONTRADICTED':
                    verdict = 'CONTRADICTED'
                    hit_rule = r
                    evidence = ev
                    break
        results[opt_key] = {
            'verdict': verdict,
            'rule': hit_rule,
            'evidence': evidence,
        }
    return results


def format_rules_hint(rule_results: Dict[str, Dict[str, Any]]) -> str:
    """把规则引擎结果格式化为LLM prompt里的提示行。"""
    parts = []
    for k in sorted(rule_results.keys()):
        r = rule_results[k]
        if r['verdict'] == 'CONTRADICTED':
            parts.append(f"{k}=CONTRADICTED({r['rule']})")
    if not parts:
        return '规则引擎预判定：无命中，所有选项需结合原文独立判断。'
    return '规则引擎已预判定：' + '，'.join(parts) + '。请结合原文复核确认，未列出的选项需独立判断。'


if __name__ == '__main__':
    # 简易自测
    ctx = """
第二章 保险责任
在本合同保险期间内，我们承担下列保险责任...

第三章 责任免除
因下列情形之一导致被保险人发生保险事故的，我们不承担给付保险金的责任：
（1）投保人对被保险人的故意杀害、故意伤害；
（2）被保险人故意犯罪或者抗拒依法采取的刑事强制措施；
（3）被保险人自本合同成立或者合同效力恢复之日起2年内自杀，但被保险人自杀时为无民事行为能力人的除外；
（4）被保险人酒后驾驶、无合法有效驾驶证驾驶，或驾驶无有效行驶证的机动车；
（5）战争、军事冲突、暴乱或武装叛乱；
（6）核爆炸、核辐射或核污染。

第四条 等待期
自本合同生效之日起90日为等待期。等待期内被保险人因疾病发生保险事故的，我们不承担保险责任。

第七条 犹豫期
自您签收本合同次日起，有15日的犹豫期。若您在犹豫期内解除合同，我们将在扣除不超过10元的工本费后退还您所交纳的保险费。

保险金额：本合同的基本保险金额为100000元。
免赔额：年度免赔额为10000元。
赔付比例：我们对超过免赔额的部分按80%的比例给付医疗保险金。
"""
    opts = {
        'A': '被保险人等待期内因疾病发生保险事故，保险公司应当承担保险责任',
        'B': '投保人犹豫期内退保，保险公司扣除手续费后退还保险费',
        'C': '被保险人酒后驾车发生事故，保险公司应当给付保险金',
        'D': '被保险人因疾病住院花费50000元，保险公司应全额赔付50000元',
    }
    res = apply_insurance_rules('关于某医疗保险条款，下列说法正确的是', opts, ctx)
    for k, v in res.items():
        print(f'{k}: {v["verdict"]} rule={v["rule"]}')
        if v['evidence']:
            print(f'   evidence: {v["evidence"][:100]}')
