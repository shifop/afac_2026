"""P1-3: 轻量FactIndex — 统一事实索引

只抽与100题相关的字段，先服务影子验证器。后续扩展为全量KG。

统一格式（KG和evidence_ledger共享）:
  fact_index lookup by (entity, field, time) → FactRecord

Usage:
  idx = build_lightweight_index(extracted_texts)
  facts = idx.lookup('比亚迪', '研发投入占营收比例', '2024年')
  → [FactRecord(value='6.97%', source_doc=..., page=...)]
"""

import re, json, os, sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime

# ═══════════════════════════════════════
# FactIndex data model
# ═══════════════════════════════════════

@dataclass
class FactRecord:
    """统一事实记录 — KG和evidence_ledger共享

    E-2.6: 质量门禁——无raw_evidence+source_doc+line_start的fact不能直接裁决
    """
    entity: str           # 主体（公司/产品/法规）
    field: str            # 字段名（营收/等待期/信用评级）
    value_original: str   # 原始值（含单位，"7771亿元"）
    value_normalized: float = None  # 归一化数值（纯数字，统一单位到元/%/etc）
    unit: str = None      # 原始单位
    time: str = None      # 时间（2024年/报告期末/前三季度）
    source_doc: str = None  # 来源文档
    page: int = None      # 页码
    line: int = None      # 行号（保留向后兼容）
    line_start: int = None  # 起始行号
    line_end: int = None    # 结束行号
    char_start: int = None  # 原文字符起点（V2正式定位）
    char_end: int = None    # 原文字符终点
    tool: str = 'regex'   # 提取工具
    confidence: str = 'MEDIUM'  # HIGH/MEDIUM/LOW
    # E-2.6+ fields
    raw_evidence: str = None       # 原文摘录（必须存在才能裁决）
    table_row_header: str = None   # 表格行标题
    table_col_header: str = None   # 表格列标题（含年份信息）
    condition: str = None          # 适用条件/前提
    modality: str = None           # fact/prediction/rule
    fact_type: str = 'atomic_fact' # atomic_fact/formula_fact/rule_fact/relation_fact
    # A++++: resolve status
    resolve_status: str = 'UNKNOWN'  # ADJUDICABLE/FAILED_LOCATE/FACT_REJECTED/RETRIEVAL_HINT_ONLY/TABLE_AMBIGUOUS

    def lookup_key(self):
        """唯一查找键"""
        return (self.entity, self.field, self.time or '')

    def to_dict(self):
        return asdict(self)

    @property
    def is_adjudicable(self):
        """A++++.5: 白名单门禁——只有resolve_status=ADJUDICABLE才可裁决"""
        if self.resolve_status != 'ADJUDICABLE':
            return False
        return (self.raw_evidence is not None and
                self.source_doc is not None and
                (self.char_start is not None or self.line_start is not None or self.line is not None))


# E-6.2: 字段别名映射（保守：只覆盖财报高频无歧义词）
# E-6.3: 精准扩展——仅添加语义确实等价的定量字段（共21条映射）
FIELD_ALIASES = {
    # 现金流
    '现金流': ['经营现金流', '经营现金流同比'],
    '现金流表现': ['经营现金流', '经营现金流同比'],
    '筹资活动现金流': ['筹资活动现金流减少'],
    # 营收
    '营收': ['营业收入', '营业总收入'],
    '营业总收入': ['营业收入'],
    '营收规模': ['营业收入'],
    '营业收入同比': ['营收同比增长'],
    '营业总收入增长率': ['营收同比增长'],
    '营收增速': ['营收同比增长'],
    # 利润
    '利润': ['净利润', '归母净利润'],
    '净利润表现': ['净利润', '归母净利润', '净利润同比'],
    '归母净利润增速': ['归母净利润同比'],
    '净利润降幅': ['归母净利润同比'],
    '归母净利润增长': ['归母净利润同比'],
    # 研发
    '研发': ['研发投入', '研发费用', '研发投入占营收比例'],
    '研发投入强度': ['研发投入占营收比例', '研发费用占营收比例'],
    '研发投入占比': ['研发投入占营收比例'],
    '研发费用占比': ['研发费用占营收比例'],
    '研发投入占营收比例': ['研发投入占营收比例', '研发费用占营收比重'],  # cross-entity lookup
    # 资产负债率
    '资产负债率': ['资产负债率'],
    # 分红
    '分红': ['现金分红', '每股分红', '现金分红占归母净利润比例'],
    '每股现金分红': ['每股分红'],
    '每股派息': ['每股分红'],
    '现金分红': ['每股分红', '每10股分红', '现金分红占归母净利润比例', '现金分红比例'],
    '现金分红政策': ['现金分红比例', '现金分红占归母净利润比例'],
    '每10股分红': ['每股分红', '每10股分红'],
    '现金分红占净利润比例': ['现金分红占归母净利润比例'],
    # 评级
    '评级': ['主体信用评级', '主体评级', '债项评级'],
    # 法规：scope后缀编码在字段名中
    '存量客户识别核实期限': ['存量客户识别核实期限_较高风险'],
    # 综合
    '业绩': ['营业收入', '净利润', '归母净利润'],
}


class FactIndex:
    """统一事实索引"""

    def __init__(self):
        self.facts = []  # List[FactRecord]
        self._index = defaultdict(list)  # (entity, field, time) → [FactRecord]

    def _normalize_time(self, time_str):
        """归一化时间格式: '2025年度'→'2025年', '2025年前三季度'→'2025年'"""
        if not time_str:
            return ''
        import re
        # Extract year
        m = re.search(r'(202[0-9])', str(time_str))
        if m:
            return f'{m.group(1)}年'
        return str(time_str)

    def add(self, fact: FactRecord):
        # Persist normalized time in the key; otherwise '2025年度' can never
        # match a normalized '2025年' lookup deterministically.
        fact.time = self._normalize_time(fact.time) if fact.time else None
        self.facts.append(fact)
        key = fact.lookup_key()
        self._index[key].append(fact)

    def lookup(self, entity, field, time=None, fuzzy=False):
        """精确或模糊查找（含时间归一化）

        Args:
            entity: 主体名称
            field: 字段名
            time: 时间（可选，None=不限制时间）
            fuzzy: True → 模糊匹配entity和field

        Returns: (found, List[FactRecord], match_type)
        """
        time_norm = self._normalize_time(time) if time else ''
        # Exact match with normalized time
        key = (entity, field, time_norm)
        if key in self._index:
            return True, self._index[key], 'exact_match'

        # Original time exact match (backward compat)
        key_orig = (entity, field, time or '')
        if key_orig != key and key_orig in self._index:
            return True, self._index[key_orig], 'exact_match'

        # Partial time match
        if time:
            for (e, f, t), facts in self._index.items():
                if e == entity and f == field and t and (time_norm in t or (time or '') in t):
                    return True, facts, 'time_relaxed'

        if not fuzzy:
            return False, [], 'not_found'

        # Fuzzy entity match (substring)
        results = []
        for (e, f, t), facts in self._index.items():
            if (entity in e or e in entity) and field == f:
                if not time or not t or time in t:
                    results.extend(facts)
        return (True, results, 'fuzzy_match') if results else (False, [], 'not_found')

    def lookup_with_aliases(self, entity, field, time=None):
        """Conservative alias lookup.

        Only field aliases classified as exact equivalence may participate in
        adjudication.  Broad terms such as "利润" or "评级" remain retrieval
        hints and are not expanded to multiple non-equivalent accounting or
        contract fields here.
        """
        found, facts, mt = self.lookup(entity, field, time, fuzzy=False)
        if found and facts:
            return found, facts, mt
        try:
            from field_ontology import aliases_for_field
            aliases = aliases_for_field(field, adjudication=True)
        except Exception:
            aliases = [field]
        all_facts = []
        for alias_field in aliases:
            if alias_field == field:
                continue
            found, facts, _ = self.lookup(entity, alias_field, time, fuzzy=False)
            if found:
                all_facts.extend(facts)
        if all_facts:
            return True, all_facts, 'exact_alias_match'
        # Entity substring matching is retained only with the exact requested
        # field; no relationship expansion is performed.
        return self.lookup(entity, field, time, fuzzy=True)

    def stats(self):
        """索引统计"""
        return {
            'total_facts': len(self.facts),
            'unique_entities': len(set(f.entity for f in self.facts)),
            'unique_fields': len(set(f.field for f in self.facts)),
            'index_keys': len(self._index),
        }

    def export_ledger_format(self):
        """导出为evidence_ledger可直接使用的格式"""
        ledger = {}
        for fact in self.facts:
            entry = {
                'value': fact.value_original,
                'source': {
                    'doc': fact.source_doc,
                    'page': fact.page,
                    'line': fact.line,
                    'tool': fact.tool
                },
                'confidence': fact.confidence,
            }
            key = f'{fact.entity}.{fact.field}'
            if key not in ledger:
                ledger[key] = []
            ledger[key].append(entry)
        return ledger


# ═══════════════════════════════════════
# Lightweight builder: regex extraction from text
# ═══════════════════════════════════════

# Patterns for common financial facts
FACT_PATTERNS = {
    # 营收
    '营业收入': [
        (r'(?:营业收入|营业总收入|营收)[^\d]*?([\d,]+\.?\d*\s*(?:万亿元|千亿元|亿元|万元))', 'amount'),
    ],
    '归母净利润': [
        (r'(?:归母净利润|归属于母公司[^\d]*?净利润)[^\d]*?([\d,]+\.?\d*\s*(?:万亿元|千亿元|亿元|万元))', 'amount'),
    ],
    '研发投入': [
        (r'(?:研发投入|研发费用)[^\d]*?([\d,]+\.?\d*\s*(?:万亿元|千亿元|亿元|万元))', 'amount'),
    ],
    '研发投入占营收比例': [
        (r'(?:研发投入|研发费用)[^占]*?占[^\d]*?(?:营收|营业收入)[^\d]*?([\d.]+%)', 'ratio'),
        (r'(?:研发占比|研发费用率)[^\d]*?([\d.]+%)', 'ratio'),
    ],
    '经营现金流': [
        (r'(?:经营[^\d]*?现金流|经营活动[^\d]*?现金流量净额)[^\d]*?([\d,]+\.?\d*\s*(?:万亿元|千亿元|亿元|万元))', 'amount'),
    ],
    '现金分红': [
        (r'(?:现金分红|派发现金红利)[^\d]*?([\d,]+\.?\d*\s*(?:万亿元|千亿元|亿元|万元))', 'amount'),
    ],
    '每股分红': [
        (r'(?:每股分红|每股派息|每股股利)[^\d]*?([\d.]+元)', 'per_share'),
        (r'每\s*10\s*股[^\d]*?([\d.]+元)', 'per_10_share'),
    ],
    '信用评级': [
        (r'(?:主体信用评级|主体评级|信用等级)[为是：:]\s*(A{1,3}[+-]?)', 'rating'),
    ],
    '等待期': [
        (r'等待期[为是：:]\s*([\d]+)\s*(?:天|日)', 'term_days'),
    ],
    '免赔额': [
        (r'免赔额[为是：:]\s*([\d,]+\.?\d*\s*(?:万元|元))', 'amount'),
    ],
    '身故保险金': [
        (r'身故保险金[为是：:按均以]?\s*([^\n，。；]+(?:公式|计算)[^\n。；]*)', 'formula'),
    ],
}


def extract_facts_from_text(text, source_doc, entity_hint=None):
    """从文本中提取常见金融字段的FactRecord

    Args:
        text: 文档文本
        source_doc: 来源文档名
        entity_hint: 主体名称提示（从文档名推断）

    Returns: List[FactRecord]
    """
    facts = []

    # Infer entity from doc name or text
    entity = entity_hint or 'unknown'
    if not entity_hint:
        # Try to extract company name from first few lines
        first_lines = '\n'.join(text.split('\n')[:20])
        company_match = re.search(r'(比亚迪|宁德时代|美的集团|中国建筑|中国移动|平安[一-鿿]{1,6}|'
                                   r'国寿[一-鿿]{1,6}|太保[一-鿿]{1,6}|众安[一-鿿]{1,6}|'
                                   r'广晟控股|厦门金圆|力诺投资|科源制药|海峡股份|'
                                   r'西部证券|安克创新|芯原[一-鿿]{0,4}|宇信科技|天阳科技|长亮科技)',
                                   first_lines)
        if company_match:
            entity = company_match.group(1)

    # Infer time from text
    time = None
    year_match = re.search(r'(202[0-9])\s*(?:年|年度|财年)', text[:2000])
    if year_match:
        time = f'{year_match.group(1)}年'

    lines = text.split('\n')
    for field_name, patterns in FACT_PATTERNS.items():
        for pattern, value_type in patterns:
            for i, line in enumerate(lines):
                matches = re.findall(pattern, line)
                for m in matches:
                    fact = FactRecord(
                        entity=entity,
                        field=field_name,
                        value_original=m.strip(),
                        unit=_infer_unit(m, value_type),
                        time=time,
                        source_doc=source_doc,
                        line=i,
                        tool='regex_lightweight',
                        confidence='LOW',  # Regex extraction, low confidence
                    )
                    facts.append(fact)

    return facts


def _infer_unit(value_str, value_type):
    """推断单位"""
    if value_type == 'amount':
        for unit in ['万亿元', '千亿元', '亿元', '万元', '千元', '元']:
            if unit in value_str:
                return unit
    elif value_type == 'ratio':
        return '%'
    elif value_type == 'per_share':
        return '元每股'
    elif value_type == 'term_days':
        return '天'
    elif value_type == 'rating':
        return 'rating'
    return 'unknown'


# ═══════════════════════════════════════
# Build lightweight index from all docs
# ═══════════════════════════════════════

def build_lightweight_index(extracted_dir, doc_map=None):
    """为影子验证器构建轻量FactIndex

    只抽与100题claim相关的字段。不追求全量KG。

    Args:
        extracted_dir: 提取文本目录
        doc_map: {domain:doc_id → filepath} 映射（可选）

    Returns: FactIndex
    """
    idx = FactIndex()

    if not os.path.exists(extracted_dir):
        print(f'  ⚠️  Extracted dir not found: {extracted_dir}')
        return idx

    txt_files = [f for f in os.listdir(extracted_dir) if f.endswith('.txt')]
    print(f'Building lightweight FactIndex from {len(txt_files)} files...')

    for fname in txt_files:
        filepath = os.path.join(extracted_dir, fname)
        try:
            text = open(filepath, encoding='utf-8').read()
            if len(text) < 100:
                continue

            # Only process first 30000 chars for speed (most facts in front matter)
            facts = extract_facts_from_text(text[:50000], source_doc=fname)
            for fact in facts:
                idx.add(fact)

        except Exception as e:
            continue

    stats = idx.stats()
    print(f'  Indexed {stats["total_facts"]} facts from {stats["unique_entities"]} entities '
          f'({stats["unique_fields"]} fields, {stats["index_keys"]} keys)')

    return idx


# ═══════════════════════════════════════
# Integration: use FactIndex to enhance verifier
# ═══════════════════════════════════════

def lookup_claim_in_index(claim, fact_index):
    """从FactIndex查找claim的对应事实

    Returns: (found, facts, match_type)
    """
    entity = claim.get('subject', '')
    field = claim.get('metric', '')
    time = claim.get('time')

    if not entity or not field:
        return False, [], 'missing entity or field'

    # Try exact lookup (now includes time normalization)
    found, facts, match_type = fact_index.lookup(entity, field, time)
    if found and facts:
        return True, facts, match_type

    # Try fuzzy lookup
    found, facts, match_type = fact_index.lookup(entity, field, time, fuzzy=True)
    if found and facts:
        return True, facts, 'fuzzy_match'

    # Try without time constraint
    found, facts, match_type = fact_index.lookup(entity, field, fuzzy=True)
    if found and facts:
        return True, facts, 'time_relaxed'

    return False, [], 'not_found'
