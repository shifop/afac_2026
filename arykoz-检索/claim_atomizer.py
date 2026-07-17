#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
claim_atomizer.py — V5 Stage 2 选项原子化拆分模块

功能：将每个选项文本拆分为1-N个原子命题(atomic claim)，为后续三态判定提供结构化输入。
严格遵守C6约束：纯规则/正则拆分，不调LLM，0 token 消耗。

原子命题类型：
- NUMERIC        数值断言（X=Y / X>Y / X<Y）
- ATTRIBUTE      主体属性断言（A是B / A为B / A的B为C）
- RELATION       关系断言（A大于/小于B）
- TEMPORAL       时间断言（在T之前/之后/内/超过T）
- EXISTENTIAL    存在性断言（存在X / 包括X / 含有X）
- CONDITIONAL    条件断言（如果A则B / 当A时B / 在X情况下B）
- NEGATION       否定断言（不包括X / 不属于X / 不得X）
- COMPLEX        复杂句，无法用规则拆，整句作为1个原子claim

拆分策略：
1. 先按并列连词（、，、，和、以及、并且、；）切分并列原子命题
2. 对每个子句识别谓词类型：
   - 数值关系：高于/低于/大于/小于/等于/超过/不足/达到/约为/≈
   - 属性：是/为/即/指的是/称为
   - 时间：在...之前/之后/以内/超过...天/年
   - 否定：不/无/非/未/不得/不能
   - 条件：如果/若/当...时/在...情况下/前提是
3. 保留 modality（肯定/否定）
4. 若子句仍含多个谓词或嵌套结构，标注为COMPLEX不继续拆

输出：每个选项对应 atomic_claims 列表，每个 claim 是 dict:
    {
        'type': str,          # 类型枚举
        'subject': str,       # 主体（谓词左边）
        'predicate': str,     # 谓词
        'object': str,        # 客体/数值/时间值
        'modality': str,      # 'positive' / 'negative'
        'raw': str,           # 原子claim原文
    }

本模块不做对错判定，只做结构化拆分。
"""
from __future__ import annotations
import re
from typing import List, Dict, Any


# ──────────────────────────────────────
# 拆分用关键词/正则
# ──────────────────────────────────────

# 并列连词：用于第一层切分
_COORD_SPLIT_RE = re.compile(
    r'[、，；;]|以及|并且|而且|同时|另外|此外'
)

# 条件句引导词
_CONDITION_MARKERS = [
    '如果', '若', '假如', '假设', '在.*?情况下', '在.*?情形下',
    '当.*?时', '前提是', '只要', '除非',
]
_CONDITION_RE = re.compile(
    r'(如果|若|假如|假设|前提是|只要)([^，。；]*?)[，。；则]?|'
    r'在(.{1,30}?)的?情况下|在(.{1,30}?)的?情形下|当(.{1,30}?)时'
)

# 数值关系谓词（优先级高，先匹配）
# 注意：pattern 不使用 ^ 锚点，允许句中匹配；主体采用非贪婪+回溯，客体数字必须出现
_NUMERIC_PRED_PATTERNS = [
    # 比较级 —— 有明确谓词
    (re.compile(r'(.+?)\s*(高于|大于|超过|超出|多于)\s*([\d,.]+\s*(?:万亿|千亿|百亿|十亿|亿|万|千|百|元|美元|%|个百分点|倍|日|天|个月|年|岁|小时|股|份|张|万元|亿元)?(?:左右|以上|以下)?)'), 'GT'),
    (re.compile(r'(.+?)\s*(低于|小于|不足|少于|不到)\s*([\d,.]+\s*(?:万亿|千亿|百亿|十亿|亿|万|千|百|元|美元|%|个百分点|倍|日|天|个月|年|岁|小时|股|份|张|万元|亿元)?(?:左右|以上|以下)?)'), 'LT'),
    (re.compile(r'(.+?)\s*(不低于|不少于|不小于|至少|≥|>=)\s*([\d,.]+\s*(?:万亿|千亿|百亿|十亿|亿|万|千|百|元|美元|%|个百分点|倍|日|天|个月|年|岁)?)'), 'GTE'),
    (re.compile(r'(.+?)\s*(不高于|不超过|不大于|至多|≤|<=)\s*([\d,.]+\s*(?:万亿|千亿|百亿|十亿|亿|万|千|百|元|美元|%|个百分点|倍|日|天|个月|年|岁)?)'), 'LTE'),
    (re.compile(r'(.+?)\s*(等于|为|是|约为|≈|约|达到|增至|下降至|增长至)\s*([\d,.]+\s*(?:万亿|千亿|百亿|十亿|亿|万|千|百|元|美元|%|个百分点|倍|日|天|个月|年|岁|小时|股|份|张|万元|亿元)?)'), 'EQ'),
    (re.compile(r'(.+?)\s*(同比增长|同比下降|同比增加|同比减少|环比增长|环比下降|增长|下降|增加|减少|上升|下滑)\s*([\d,.]+\s*%?)'), 'CHANGE'),
    # 隐式数值断言：<metric名词短语> <number>（如"营业收入7771亿元"、"基本保险金额10万元"）
    # 要求：主体必须以汉字结尾（指标名），客体必须以数字开头
    (re.compile(r'([\u4e00-\u9fa5][\u4e00-\u9fa5A-Za-z0-9]{1,28}?)\s{0,2}(达|约|近|超|超过|逾|为|是|约为|高达)?\s{0,2}(\d[\d,.]*\s*(?:万亿|千亿|百亿|十亿|亿|万|千|百|元|美元|%|个百分点|倍|日|天|个月|年|岁|万元|亿元)(?:左右|以上|以下|多)?)'), 'EQ_IMPLICIT'),
]

# 数值/单位识别
_NUMBER_RE = re.compile(r'([\d,]+\.?\d*)\s*(万亿|千亿|百亿|十亿|亿|万|千|百|元|美元|%|个百分点|倍|日|天|个月|年|岁|小时|股|份|张|万元|亿元)?')

# 时间断言
_TEMPORAL_PATTERNS = [
    (re.compile(r'(.+?)\s*(在|于)\s*([\d]{4}年(?:[\d]{1,2}月)?(?:[\d]{1,2}日)?)\s*(?:之前|以前|前|之后|以后|后|以内|内|期间)?'), 'TIME_POINT'),
    (re.compile(r'(.+?)\s*(?:自|从)\s*([^，。；]{1,20}?)\s*(?:之日起|起)\s*(\d+)\s*(日|天|个月|年|工作日)(?:内|以内|之内|前|后)?'), 'TIME_DURATION'),
    (re.compile(r'(.+?)\s*(超过|不足|不满)\s*(\d+)\s*(日|天|个月|年|工作日)'), 'TIME_COMPARE'),
    (re.compile(r'等待期\s*(?:为|是)?\s*(\d+)\s*(日|天)'), 'WAITING_PERIOD'),
]

# 属性断言（A是B / A为B / A的B为C）
_ATTRIBUTE_PATTERNS = [
    re.compile(r'^(.{1,30}?)\s*(?:是指|是|为|即|称为|属于)\s*(.{1,40}?)$'),
    re.compile(r'^(.{1,30}?)\s*的\s*(.{1,20}?)\s*(?:为|是|等于|有)\s*(.{1,40}?)$'),
]

# 否定词
_NEGATION_WORDS = ['不', '无', '非', '未', '没', '否', '禁止', '不得', '不能', '不可', '不予']

# 存在性
_EXISTENTIAL_PATTERNS = [
    re.compile(r'(?:存在|包括|包含|含有|涵盖|涉及|有)\s*(.{1,40})'),
]

# 主体抽取用：选项开头到第一个谓词之间视为主体
_SUBJECT_STOP_WORDS = {'是', '为', '在', '有', '对', '与', '和', '或', '将', '被', '把', '从', '向', '到', '比'}


def _detect_modality(text: str) -> str:
    """检测句子的肯否定模态"""
    # 简单启发：如果出现明显否定词且不是"不超过/不少于"这类词组内嵌，标记为 negative
    for neg in ['不', '无', '非', '未', '没', '不得', '不能', '不可', '不予', '禁止']:
        if neg in text:
            # 排除"不低于/不高于/不超过/不少于/不大于"这类肯定比较
            # 这些在数值谓词里已经单独处理
            return 'negative'
    return 'positive'


def _is_mostly_numeric_clause(text: str) -> bool:
    """判断子句是否主要是数值比较"""
    return bool(re.search(r'[\d]+\s*[%万亿千元日天年个倍]+', text)) and bool(
        re.search(r'高于|低于|大于|小于|等于|超过|不足|达到|约为|≈|是|为', text)
    )


def _classify_clause(clause: str) -> Dict[str, Any]:
    """对单个子句做类型判定，返回一个 atomic claim dict。"""
    clause = clause.strip().strip('，。；,.; ')
    if not clause:
        return None

    modality = _detect_modality(clause)
    claim_type = 'COMPLEX'
    subject = ''
    predicate = ''
    obj = ''

    # 1. 优先匹配数值关系
    for pat, rel in _NUMERIC_PRED_PATTERNS:
        m = pat.search(clause)
        if m:
            n_groups = len(m.groups())
            if rel == 'EQ_IMPLICIT':
                # 3 groups: subject, 达/约/近(可空), object
                subj_raw = (m.group(1) or '').strip()
                pred_word = (m.group(2) or '').strip()
                obj_raw = (m.group(3) or '').strip()
                # 防误匹配：主体必须含汉字，且客体确实包含数字
                if not re.search(r'[\u4e00-\u9fa5]', subj_raw):
                    continue
                if not re.search(r'\d', obj_raw):
                    continue
                # 主体末尾不能是单独的数字/标点（切到真正名词）
                subject = subj_raw
                predicate = pred_word if pred_word else '为'
                obj = obj_raw
            else:
                subject = (m.group(1) or '').strip()
                predicate = m.group(2).strip()
                obj = (m.group(3) or '').strip()
                if not re.search(r'\d', obj):
                    continue
            # 处理特殊：不低于/不高于等本身已带否定
            if predicate in ('不低于', '不少于', '不小于', '不高于', '不超过', '不大于'):
                modality = 'positive'
            claim_type = 'NUMERIC'
            return {
                'type': claim_type, 'relation': rel,
                'subject': subject, 'predicate': predicate, 'object': obj,
                'modality': modality, 'raw': clause,
            }

    # 2. 时间断言
    for pat, kind in _TEMPORAL_PATTERNS:
        m = pat.search(clause)
        if m:
            groups = [g for g in m.groups() if g]
            subject = (groups[0] if groups else '').strip()
            obj = groups[-1].strip() if groups else ''
            predicate = kind
            claim_type = 'TEMPORAL'
            return {
                'type': claim_type, 'relation': kind,
                'subject': subject, 'predicate': predicate, 'object': obj,
                'modality': modality, 'raw': clause,
            }

    # 3. 条件句
    cm = _CONDITION_RE.search(clause)
    if cm:
        claim_type = 'CONDITIONAL'
        # 提取条件部分和结果部分
        cond_part = ''
        for i, g in enumerate(cm.groups()):
            if g and i > 0:
                cond_part = g.strip()
                break
        result_part = clause[cm.end():].strip('，。；,.; ')
        predicate = 'IF-THEN'
        subject = cond_part
        obj = result_part
        return {
            'type': claim_type,
            'subject': subject, 'predicate': predicate, 'object': obj,
            'modality': modality, 'raw': clause,
        }

    # 4. 存在性
    for pat in _EXISTENTIAL_PATTERNS:
        m = pat.search(clause)
        if m and not clause.startswith(('是', '为')):
            claim_type = 'EXISTENTIAL'
            predicate = m.group(0)[:2]
            obj = m.group(1).strip()
            subject = clause[:m.start()].strip() or '本合同/本产品'
            return {
                'type': claim_type,
                'subject': subject, 'predicate': predicate, 'object': obj,
                'modality': modality, 'raw': clause,
            }

    # 5. 属性断言
    for pat in _ATTRIBUTE_PATTERNS:
        m = pat.match(clause)
        if m:
            groups = [g.strip() for g in m.groups() if g and g.strip()]
            if len(groups) >= 2:
                if len(groups) == 2:
                    subject, obj = groups[0], groups[1]
                    predicate = '是/为'
                else:
                    subject, attr, obj = groups[0], groups[1], groups[2]
                    predicate = f'{attr}为'
                claim_type = 'ATTRIBUTE'
                # 若object是数值，升级为NUMERIC
                if re.search(r'^[\d,.]+\s*[%万亿千元日天年个倍]', obj):
                    claim_type = 'NUMERIC'
                return {
                    'type': claim_type,
                    'subject': subject, 'predicate': predicate, 'object': obj,
                    'modality': modality, 'raw': clause,
                }

    # 6. 简单关系/动作句：按第一个动词/谓词位置切分主体和谓宾
    # 兜底：整句作为 COMPLEX
    # 尝试提取一个粗略主语（句首到第一个停用词/标点）
    tokens = list(re.finditer(r'[\u4e00-\u9fa5A-Za-z0-9%]+', clause))
    subj_end = 0
    for tk in tokens:
        w = tk.group()
        if w in _SUBJECT_STOP_WORDS and tk.start() > 0:
            subj_end = tk.start()
            predicate = w
            obj = clause[tk.end():].strip()
            subject = clause[:tk.start()].strip()
            break
    else:
        subject = clause
        predicate = ''
        obj = ''

    # 如果句子过短（<8字）或者没有识别到任何谓词结构，整句做 ATTRIBUTE/COMPLEX
    if len(clause) <= 6:
        claim_type = 'ATTRIBUTE'
        predicate = predicate or '是/为'
        obj = obj or clause
    else:
        claim_type = 'COMPLEX'
        predicate = predicate or ''

    return {
        'type': claim_type,
        'subject': subject, 'predicate': predicate, 'object': obj,
        'modality': modality, 'raw': clause,
    }


def _split_coordinates(text: str) -> List[str]:
    """按并列连词切分，但保留整体名词短语。
    简单策略：先按并列符切，若切出的碎片过短（<2字）且前一个碎片看起来也是名词，视为同一名词短语的并列修饰，合并。
    """
    # 先按正则切分
    parts = _COORD_SPLIT_RE.split(text)
    parts = [p.strip('，。；,.; ') for p in parts if p and p.strip('，。；,.; ')]

    # 合并规则：如果某段不含任何谓词（是/为/高于/低于/有/在等），且前后段是同一主语下的并列宾语/谓语，
    # 我们保持分离（每段独立做atomic claim），由上层根据subject复用判断；
    # 过短碎片(<2字)说明是连词位置切错，合并回前段
    merged = []
    for p in parts:
        if len(p) < 2 and merged:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)
    return merged if merged else [text]


def atomize_option(option_text: str) -> List[Dict[str, Any]]:
    """将单个选项文本拆分为原子命题列表。

    Args:
        option_text: 单个选项的文本（不含选项字母前缀）

    Returns:
        List[atomic_claim dict]，至少包含1个claim
    """
    option_text = str(option_text or '').strip()
    if not option_text:
        return [{'type': 'COMPLEX', 'subject': '', 'predicate': '', 'object': '',
                 'modality': 'positive', 'raw': ''}]

    # 去掉选项前缀 "A." / "A、" / "A．" 等
    option_text = re.sub(r'^[A-Da-d][\.、．.、]\s*', '', option_text).strip()

    # Step 1: 并列切分
    clauses = _split_coordinates(option_text)

    # Step 2: 若只有1段且含明确条件句，仍先把条件整体作为CONDITIONAL，不强行拆
    # Step 3: 逐段分类
    claims = []
    for c in clauses:
        c = c.strip()
        if not c:
            continue
        claim = _classify_clause(c)
        if claim:
            claims.append(claim)

    # Step 4: 去重（raw完全相同的去重）
    seen_raw = set()
    deduped = []
    for cl in claims:
        if cl['raw'] not in seen_raw:
            seen_raw.add(cl['raw'])
            deduped.append(cl)

    # 保底：至少1个
    if not deduped:
        deduped = [{'type': 'COMPLEX', 'subject': '', 'predicate': '', 'object': '',
                    'modality': 'positive', 'raw': option_text}]

    return deduped


def atomize_options(options: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """批量原子化多个选项。

    Args:
        options: {A: text, B: text, ...}

    Returns:
        {A: [claim1, claim2, ...], B: [...], ...}
    """
    return {k: atomize_option(v) for k, v in (options or {}).items()}


def format_atomic_claims_for_prompt(atomic_map: Dict[str, List[Dict[str, Any]]]) -> str:
    """把原子化结果格式化为LLM prompt可读的字符串。"""
    lines = []
    for opt_key in sorted(atomic_map.keys()):
        claims = atomic_map[opt_key]
        lines.append(f'【选项{opt_key}】共{len(claims)}个原子命题：')
        for i, c in enumerate(claims, 1):
            modality_tag = '否' if c['modality'] == 'negative' else '肯'
            subj = c['subject'] or '(无主)'
            pred = c['predicate'] or '(无谓)'
            obj = c['object'] or '(无宾)'
            lines.append(f'  {i}. [{c["type"]}|{modality_tag}] 主体={subj} | 谓词={pred} | 客体={obj}')
            if c['type'] == 'COMPLEX':
                lines.append(f'     原文：{c["raw"]}')
    return '\n'.join(lines)


if __name__ == '__main__':
    # 自测：5种典型选项
    samples = {
        'A': '等待期内出险，保险公司不承担保险责任，且不退还保费',
        'B': '2024年比亚迪营业收入7771亿元，同比增长28.5%',
        'C': '投保人自合同成立之日起超过30日未支付保费，合同效力中止',
        'D': '如果被保险人在保险期间内发生意外伤害，保险公司按基本保险金额给付保险金',
        'E': '下列关于信息披露义务人的说法正确的是',  # COMPLEX整句
    }
    for k, v in samples.items():
        print(f'\n=== {k}. {v} ===')
        for cl in atomize_option(v):
            print(f'  [{cl["type"]}|{cl["modality"]}] subj={cl["subject"][:20]!r} '
                  f'pred={cl["predicate"]!r} obj={cl["object"][:30]!r}')
