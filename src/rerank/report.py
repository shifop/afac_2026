import re
import math
from typing import List, Dict, Optional, Any, Set, Tuple
from difflib import SequenceMatcher
from collections import defaultdict

try:
    from bs4 import BeautifulSoup, Tag
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    raise ImportError("请安装 beautifulsoup4: pip install beautifulsoup4")


# ========================= 同义词与比较词映射表 =========================
FINANCIAL_SYNONYMS = {
    "营业收入": ["营收", "总收入", "主营业务收入", "销售额"],
    "净利润": ["净利", "纯利润", "归属于母公司股东的净利润", "归母净利润"],
    "毛利率": ["毛利"],
    "研发费用": ["研发投入", "研发开支"],
    "资产负债率": ["负债率"],
    "每股收益": ["EPS", "每股盈利"],
    "净资产收益率": ["ROE", "权益回报率"],
    "总资产": ["资产总计", "资产总额"],
    "总负债": ["负债合计", "负债总额"],
    "海外业务营收": ["海外收入", "境外营收", "境外收入"],
    "营收占比": ["收入占比", "比重"],
    "同比增速": ["同比增长率", "增幅", "增速"],
}
SYNONYM_TO_STANDARD = {}
for std, aliases in FINANCIAL_SYNONYMS.items():
    for alias in aliases:
        SYNONYM_TO_STANDARD[alias] = std
    SYNONYM_TO_STANDARD[std] = std

COMPARISON_SYNONYMS = {
    "同比": ["较上年", "同比增长", "同比变动", "比上年", "较去年同期"],
    "占比": ["比例", "占", "百分比", "份额", "贡献率"],
    "高于": ["大于", "超过", "优于", "超越"],
    "低于": ["小于", "不足", "低于", "落后"],
    "增长": ["增加", "上升", "提高", "增幅", "上涨"],
    "下降": ["减少", "降低", "下滑", "降幅", "跌落"],
}


# ========================= 工具函数 =========================
def normalize_text(s: str) -> str:
    """去除多余空格/换行，全角转半角，统一小写"""
    if not s:
        return ""
    s = s.replace('\n', ' ').replace('\r', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    result = []
    for ch in s:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(ch)
    return ''.join(result).lower()


def contains_number(text: str) -> bool:
    """检测是否包含数值（含百分数、千分位、负数）"""
    pattern = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%?|[-+]?\.\d+\s*%?'
    return bool(re.search(pattern, text))


def is_aggregate_row(cells_text: List[str]) -> bool:
    """判断一行是否为小计/合计行"""
    joined = ' '.join(cells_text).lower()
    keywords = ['合计', '小计', '总计', 'sum', 'total', '累计', '全部','总收']
    return any(kw in joined for kw in keywords)


def fuzzy_contains(query: str, text: str, synonyms: List[str] = None,
                   threshold: float = 0.8, short_threshold: float = 0.9) -> bool:
    """
    判断文本中是否包含查询词或其同义词（模糊匹配）。
    """
    terms = [query]
    if synonyms:
        terms.extend(synonyms)
    norm_text = normalize_text(text)
    for term in terms:
        norm_term = normalize_text(term)
        if norm_term in norm_text:
            return True
        if len(norm_term) <= 3:
            win_size = len(norm_term)
            for i in range(len(norm_text) - win_size + 1):
                window = norm_text[i:i + win_size]
                if SequenceMatcher(None, norm_term, window).ratio() >= short_threshold:
                    return True
        else:
            if len(norm_text) < 2000:
                if SequenceMatcher(None, norm_term, norm_text).ratio() >= threshold:
                    return True
    return False


def fuzzy_match_score(query: str, text: str,
                      synonyms: Optional[List[str]] = None,
                      exact_threshold: float = 0.85,
                      partial_threshold: float = 0.6) -> float:
    """
    判断查询词（含同义词）是否在文本中出现，返回 1.0（完全命中）、0.7（部分命中）或 0.0（未命中）。
    """
    terms = [query]
    if synonyms:
        terms.extend(synonyms)
    norm_text = normalize_text(text)
    for term in terms:
        norm_term = normalize_text(term)
        if norm_term in norm_text:
            return 1.0
        if len(norm_term) <= 4:
            win_size = len(norm_term)
            best = 0.0
            for i in range(len(norm_text) - win_size + 1):
                window = norm_text[i:i + win_size]
                sim = SequenceMatcher(None, norm_term, window).ratio()
                if sim > best:
                    best = sim
            if best >= exact_threshold:
                return 1.0
            elif best >= partial_threshold:
                return 0.7
        else:
            if len(norm_text) < 2000:
                sim = SequenceMatcher(None, norm_term, norm_text).ratio()
                if sim >= exact_threshold:
                    return 1.0
                elif sim >= partial_threshold:
                    return 0.7
    return 0.0


def get_chunk_text(chunk) -> str:
    """从chunk对象中获取综合文本"""
    content = getattr(chunk, 'content', '')
    entities = getattr(chunk, 'entities', [])
    relations = getattr(chunk, 'relations', [])
    parts = [content]
    for ent in entities:
        name = getattr(ent, 'name', '')
        desc = getattr(ent, 'desc', '')
        parts.append(f"{name} {desc}")
    for rel in relations:
        subj = getattr(rel, 'subject', '')
        pred = getattr(rel, 'predicate', '')
        obj = getattr(rel, 'object', '')
        parts.append(f"{subj} {pred} {obj}")
    # return ' '.join(parts)
    return content


# ========================= 表格解析与裁剪 =========================
def parse_html_table(html_content: str) -> Optional[List[List[str]]]:
    """将HTML表格解析为二维文本数组"""
    if not HAS_BS4:
        return None
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if not table:
        return None
    grid = []
    for row in table.find_all('tr'):
        cells = row.find_all(['th', 'td'])
        row_data = [cell.get_text(separator=' ', strip=True) for cell in cells]
        if row_data:
            grid.append(row_data)
    return grid


def locate_matches(grid: List[List[str]], indicators: List[str]) -> Dict:
    """
    使用模糊匹配在表格中定位指标。
    """
    row_matches = {}
    col_matches = {}
    for indicator in indicators:
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if fuzzy_match_score(indicator, cell, synonyms=FINANCIAL_SYNONYMS.get(indicator)) > 0:
                    if len(grid[0]) <= 3 and c == 0:
                        row_matches.setdefault(indicator, []).append((r, c))
                    elif r == 0:
                        col_matches.setdefault(indicator, []).append((r, c))
                    else:
                        col_matches.setdefault(indicator, []).append((r, c))
    return {'row': row_matches, 'col': col_matches}


# ========================= 表格展开工具 =========================
def expand_table(soup: BeautifulSoup) -> Tuple[List[List[Tag]], List[List[str]], int, int]:
    """
    将HTML表格展开为规整的二维网格，处理colspan/rowspan。
    返回:
        tag_grid: 二维单元格Tag列表 (行, 列)
        text_grid: 二维单元格文本列表 (行, 列)
        rows: 行数
        cols: 列数
    """
    table = soup.find('table')
    if not table:
        return [], [], 0, 0

    rows = table.find_all('tr')
    # 先确定最大列数
    max_cols = 0
    row_cells = []
    for row in rows:
        cells = row.find_all(['th', 'td'])
        row_cells.append(cells)
        col_count = 0
        for cell in cells:
            colspan = int(cell.get('colspan', 1))
            col_count += colspan
        max_cols = max(max_cols, col_count)

    n_rows = len(rows)
    n_cols = max_cols

    # 初始化网格，None表示被合并占位
    tag_grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    text_grid = [['' for _ in range(n_cols)] for _ in range(n_rows)]

    for r, cells in enumerate(row_cells):
        c = 0
        for cell in cells:
            # 跳过被上方rowspan占据的列
            while c < n_cols and tag_grid[r][c] is not None:
                c += 1
            if c >= n_cols:
                break

            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            cell_text = cell.get_text(separator=' ', strip=True)

            # 填充单元格到所有被合并的位置
            for dr in range(rowspan):
                rr = r + dr
                if rr >= n_rows:
                    break
                for dc in range(colspan):
                    cc = c + dc
                    if cc >= n_cols:
                        break
                    # 复制原始标签（浅拷贝即可，用于重建时保留类型）
                    new_tag = Tag(name=cell.name)
                    for attr, val in cell.attrs.items():
                        new_tag[attr] = val
                    # 移除colspan/rowspan，因为我们已展开
                    if 'colspan' in new_tag.attrs: del new_tag.attrs['colspan']
                    if 'rowspan' in new_tag.attrs: del new_tag.attrs['rowspan']
                    # 设置显示文本
                    new_tag.string = cell_text if (dr == 0 and dc == 0) else ''  # 仅主单元格显示文本，其余留空
                    tag_grid[rr][cc] = new_tag
                    text_grid[rr][cc] = cell_text if (dr == 0 and dc == 0) else ''

            c += colspan

    # 清理未填充的单元格（例如rowspan最后几行未完全覆盖）
    for r in range(n_rows):
        for c in range(n_cols):
            if tag_grid[r][c] is None:
                # 创建空td
                tag_grid[r][c] = Tag(name='td')
                text_grid[r][c] = ''

    return tag_grid, text_grid, n_rows, n_cols


# ========================= 表头识别与匹配 =========================
def identify_header_role(text_grid: List[List[str]], r: int, c: int, n_rows: int, n_cols: int) -> str:
    """
    通过遍历下方和右侧的数字单元格数量，判断单元格是行标题还是列标题。
    返回: 'row_header' 或 'column_header' 或 'unknown'
    """
    # 计算下方数字单元格数量（不包含自己）
    below_nums = 0
    for rr in range(r + 1, n_rows):
        cell_text = text_grid[rr][c]
        if contains_number(cell_text):
            below_nums += 1
        elif cell_text.strip() == '':
            continue
        else:
            break  # 遇到非数字非空即停止

    # 计算右侧数字单元格数量
    right_nums = 0
    for cc in range(c + 1, n_cols):
        cell_text = text_grid[r][cc]
        if contains_number(cell_text):
            right_nums += 1
        elif cell_text.strip() == '':
            continue
        else:
            break

    if below_nums > right_nums:
        return 'column_header'
    elif right_nums > below_nums:
        return 'row_header'
    elif below_nums == 0 and right_nums == 0:
        return 'unknown'  # 无数字跟随，可能为孤立标签，保留整行/列均无妨，暂且当列标题
    else:
        # 相等时，优先判定为列表头（更常见）
        return 'column_header'


def match_indicator_in_cell(indicator: str, cell_text: str) -> bool:
    """判断指标或比较词是否在单元格文本中命中（模糊匹配）"""
    synonyms = FINANCIAL_SYNONYMS.get(indicator, [])
    # 如果是比较词，使用COMPARISON_SYNONYMS
    # 统一处理：尝试指标同义词匹配，若失败再尝试比较词同义词（但调用者应区分）
    # 这里简化：先用fuzzy_match_score，大于0视为命中
    return fuzzy_match_score(indicator, cell_text, synonyms=synonyms) > 0


def match_comparison_in_cell(comparison: str, cell_text: str) -> bool:
    synonyms = COMPARISON_SYNONYMS.get(comparison, [])
    return fuzzy_match_score(comparison, cell_text, synonyms=synonyms) > 0


# ========================= 新增：分组覆盖评分器 =========================
class CoverageRanker:
    """基于分组需求的打分器，强调组内覆盖率和跨组覆盖数"""
    def __init__(self, requirement_groups: List[List[str]]):
        self.groups = requirement_groups
        self.max_scores = {
            'coverage': 50,
            'number': 30,
            'structure': 10,
            'novelty': 10
        }
        self.all_keywords = list({w for g in requirement_groups for w in g})

    def _group_coverage_score(self, content: str) -> float:
        if not self.groups:
            return float(self.max_scores['coverage'])
        total = 0.0
        for group in self.groups:
            hit = sum(1 for word in group
                      if fuzzy_match_score(word, content,
                                           synonyms=FINANCIAL_SYNONYMS.get(word, [])) > 0)
            total += hit / len(group)
        return (total / len(self.groups)) * self.max_scores['coverage']

    def _number_existence_score(self, content: str) -> float:
        if not self.all_keywords:
            return float(self.max_scores['number'])
        if not contains_number(content):
            return 0.0
        effective = sum(1 for kw in self.all_keywords
                        if fuzzy_match_score(kw, content,
                                             synonyms=FINANCIAL_SYNONYMS.get(kw, [])) > 0)
        return (effective / len(self.all_keywords)) * self.max_scores['number']

    def _structure_score(self, content: str, has_table: bool,
                         has_aggregate: bool, is_continuous_text: bool,
                         length: int) -> float:
        score = 0.0
        if has_table:
            score += 10
            if has_aggregate:
                score += 3
        elif is_continuous_text:
            score += 2
        if length > 800 and self._group_coverage_score(content) <= 5:
            score -= 2
        return max(0, min(self.max_scores['structure'], score))

    def score_chunk(self, chunk, novelty_score: float = 0.0) -> float:
        content = get_chunk_text(chunk)
        has_table = '<table' in content.lower()
        grid = None
        has_aggregate = False
        if has_table:
            grid = parse_html_table(content)
            if grid:
                has_aggregate = any(is_aggregate_row(row) for row in grid)
        is_continuous = not has_table and len(content.split('\n')) < 5

        s1 = self._group_coverage_score(content)
        s2 = self._number_existence_score(content)
        s3 = self._structure_score(content, has_table, has_aggregate, is_continuous, len(content))
        s4 = min(self.max_scores['novelty'], novelty_score)
        return min(100.0, s1 + s2 + s3 + s4)


# ========================= 修改后的表格裁剪（统一关键词匹配） =========================
# ========================= 修复后的表格裁剪 =========================
def clip_table(html_content: str, keywords: List[str]) -> str:
    """
    裁剪表格：保留与关键词相关的行/列，并确保数字列与合计行不丢失。
    如果 html_content 包含表格外的文本，本函数只处理内部表格，
    调用者应负责拼接前后文。
    """
    def match_keyword_in_cell(keyword: str, cell_text: str) -> bool:
        if fuzzy_match_score(keyword, cell_text,
                             synonyms=FINANCIAL_SYNONYMS.get(keyword, [])) > 0:
            return True
        if fuzzy_match_score(keyword, cell_text,
                             synonyms=COMPARISON_SYNONYMS.get(keyword, [])) > 0:
            return True
        return False

    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if not table:
        return html_content

    tag_grid, text_grid, n_rows, n_cols = expand_table(soup)
    if n_rows == 0 or n_cols == 0:
        return html_content

    keep_rows: Set[int] = set()
    keep_cols: Set[int] = set()

    # 1. 匹配关键词所在单元格，根据角色保留行/列
    for r in range(n_rows):
        for c in range(n_cols):
            cell_text = text_grid[r][c]
            if contains_number(cell_text):
                continue
            role = identify_header_role(text_grid, r, c, n_rows, n_cols)

            # 单层匹配
            hit = any(match_keyword_in_cell(kw, cell_text) for kw in keywords)
            # 合并多层表头匹配（针对列表头）
            if not hit and role == 'column_header':
                combined = []
                for rr in range(r, -1, -1):
                    txt = text_grid[rr][c]
                    if not contains_number(txt):
                        combined.insert(0, txt)
                    else:
                        break
                if combined:
                    full_label = ' '.join(combined)
                    hit = any(match_keyword_in_cell(kw, full_label) for kw in keywords)
            if hit:
                if role == 'column_header':
                    keep_cols.add(c)
                    # 列表头所在的行通常也要保留，以免丢失表头行
                    keep_rows.add(r)
                elif role == 'row_header':
                    keep_rows.add(r)
                    # 行标题命中：主动保留该行中有数字的列
                    for cc in range(n_cols):
                        if contains_number(text_grid[r][cc]):
                            keep_cols.add(cc)
                else:
                    keep_rows.add(r)
                    keep_cols.add(c)

    # 2. 合计行无条件保留
    for r in range(n_rows):
        row_text = ' '.join(text_grid[r][c] for c in range(n_cols))
        if is_aggregate_row(row_text.split()):
            keep_rows.add(r)

    # 3. 表头安全保留：前3行中的非数字行
    for r in range(min(3, n_rows)):
        if any(not contains_number(text_grid[r][c]) for c in range(n_cols)):
            keep_rows.add(r)

    # 4. 保底：若没有保留任何列，保留所有列；若没有保留任何行，保留所有行
    if not keep_cols:
        keep_cols = set(range(n_cols))
    if not keep_rows:
        keep_rows = set(range(n_rows))

    # 5. 空行清理（合计行豁免）
    final_rows = set(keep_rows)
    aggregate_rows = {r for r in range(n_rows)
                      if is_aggregate_row(' '.join(text_grid[r][c] for c in range(n_cols)).split())}
    for r in list(final_rows):
        if r in aggregate_rows:
            continue
        # 只要在任一保留列上非空，就保留该行
        if all(text_grid[r][c].strip() in ('', '-', '—', 'N/A', '无', '/')
               for c in keep_cols):
            final_rows.remove(r)

    # 6. 空列清理：但数字列永远保留，只移除纯文本且全空的列
    final_cols = set(keep_cols)
    for c in list(final_cols):
        # 如果该列在任一保留行上有数字，无条件保留
        if any(contains_number(text_grid[r][c]) for r in final_rows):
            continue
        # 否则，若在所有保留行上都是占位符，则移除
        if all(text_grid[r][c].strip() in ('', '-', '—', 'N/A', '无', '/')
               for r in final_rows):
            final_cols.remove(c)

    if not final_rows or not final_cols:
        return str(table)   # 裁剪过激，返回原始表格

    # 7. 重建表格
    new_table = soup.new_tag('table')
    for attr, val in table.attrs.items():
        new_table[attr] = val

    for r in sorted(final_rows):
        new_row = soup.new_tag('tr')
        for c in sorted(final_cols):
            cell_tag = tag_grid[r][c]
            new_cell = soup.new_tag(cell_tag.name)
            for attr, val in cell_tag.attrs.items():
                if attr not in ('colspan', 'rowspan'):
                    new_cell[attr] = val
            new_cell.string = cell_tag.get_text(separator=' ', strip=True)
            new_row.append(new_cell)
        new_table.append(new_row)

    return str(new_table)


# ========================= 修改后的新颖性计算（统一关键词） =========================
def compute_novelty_scores(chunks: List[Any], keywords: List[str]) -> Dict[Any, float]:
    """基于所有关键词计算独有性得分，key 为 chunk_id"""
    id_map = {}
    for i, chunk in enumerate(chunks):
        cid = getattr(chunk, 'chunk_id', f'chunk_{i}')
        id_map[cid] = chunk
    cids = list(id_map.keys())
    N = len(chunks)
    if N <= 1:
        return {cid: 0.0 for cid in cids}

    df = {kw: 0 for kw in keywords}
    for chunk in chunks:
        content = get_chunk_text(chunk)
        for kw in keywords:
            if (fuzzy_match_score(kw, content, synonyms=FINANCIAL_SYNONYMS.get(kw, [])) > 0 or
                fuzzy_match_score(kw, content, synonyms=COMPARISON_SYNONYMS.get(kw, [])) > 0):
                df[kw] += 1

    def idf_weight(d):
        if d == 0:
            return 0.0
        return max(0.0, math.log((N + 0.5) / (d + 0.5)))

    idf_vals = {kw: idf_weight(df[kw]) for kw in keywords}

    raw_scores = {}
    for cid in cids:
        chunk = id_map[cid]
        content = get_chunk_text(chunk)
        raw = 0.0
        for kw in keywords:
            if (fuzzy_match_score(kw, content, synonyms=FINANCIAL_SYNONYMS.get(kw, [])) > 0 or
                fuzzy_match_score(kw, content, synonyms=COMPARISON_SYNONYMS.get(kw, [])) > 0):
                raw += idf_vals[kw]
        raw_scores[cid] = raw

    max_raw = max(raw_scores.values()) if raw_scores else 1.0
    alpha = math.atanh(0.95) / max_raw if max_raw != 0 else 1.0
    return {cid: round(math.tanh(raw * alpha) * 10.0, 2) for cid, raw in raw_scores.items()}


# ========================= 贪心覆盖选择器 =========================
def greedy_coverage_selection(
    chunk_data: List[Tuple[Any, float, Dict[str, float], int]],
    requirement_groups: List[List[str]],
    budget: int,
    min_score: float = 0.0,
    alpha:float = 1.0, # 得分权重
    beta:float = 1.0 # 增益权重
) -> List[Any]:
    """
    基于质量增益的贪心选择：
    - chunk_data: [(chunk, score, keyword_scores, length), ...]
      keyword_scores: 关键词 -> 匹配得分 (1.0 或 0.7)
    - 维护 best_coverage: {关键词: 当前已选最高匹配分}
    - 文档保底 + 全局质量增益贪心
    """
    all_keywords = set(w for g in requirement_groups for w in g)
    best_coverage = {kw: 0.0 for kw in all_keywords}   # 当前全局最佳覆盖质量
    selected = []
    total_length = 0

    # 过滤低分 chunk
    candidates = [(chunk, score, kw_scores, length)
                  for chunk, score, kw_scores, length in chunk_data
                  if score >= min_score]
    if not candidates:
        return []

    # ---------- 辅助函数 ----------
    def compute_gain(kw_scores: Dict[str, float], current_best: Dict[str, float]) -> float:
        """计算该 chunk 对当前覆盖质量的提升总量"""
        gain = 0.0
        for kw, score in kw_scores.items():
            current = current_best.get(kw, 0.0)
            if score > current:
                gain += score - current
        return gain

    def evaluate(chunk_info, current_best):
        chunk, score, kw_scores, length = chunk_info
        gain = compute_gain(kw_scores, current_best)
        if gain == 0:
            return 0.0, length, 0.0
        score_factor = score / 100.0
        effective_cost = max(math.log(1+min(length, 1000)), 1)
        value = (alpha * score_factor + 0.2) * (beta * gain) / effective_cost
        return value, length, gain

    # ---------- 第一阶段：按文档保底 ----------
    # 计算文档重要度（该文档内最高 score）
    doc_best = {}
    doc_candidates = {}
    for chunk, score, kw_scores, length in candidates:
        doc_id = getattr(chunk, 'doc_id', 'unknown')
        doc_best[doc_id] = max(doc_best.get(doc_id, 0), score)
        doc_candidates.setdefault(doc_id, []).append((chunk, score, kw_scores, length))
    sorted_docs = sorted(doc_best.keys(), key=lambda d: doc_best[d], reverse=True)

    selected_docs = set()
    for doc_id in sorted_docs:
        if total_length >= budget:
            break
        if doc_id in selected_docs:
            continue
        # 从该文档候选中选出对当前 best_coverage 增益最大的 chunk（已排除低分）
        best_value = -1.0
        best_item = None
        best_idx_global = -1
        for i, item in enumerate(doc_candidates[doc_id]):
            val, cost, gain = evaluate(item, best_coverage)
            if val > best_value or (val == best_value and (best_item is None or item[3] < best_item[3])):
                best_value = val
                best_item = item
                # 全局索引需要从 candidates 中找到对应位置
                best_idx_global = candidates.index(item)
        if best_item is None or best_value <= 0:
            continue
        if total_length + best_item[3] > budget:
            continue
        # 选中
        chosen = candidates.pop(best_idx_global)
        selected.append(chosen[0])
        total_length += chosen[3]
        # 更新 best_coverage
        for kw, score in chosen[2].items():
            best_coverage[kw] = max(best_coverage.get(kw, 0), score)
        selected_docs.add(doc_id)

    # ---------- 第二阶段：全局质量贪心 ----------
    while candidates and total_length < budget:
        best_value = -1.0
        best_idx = -1
        best_cost = 0
        for i, (chunk, score, kw_scores, length) in enumerate(candidates):
            val, cost, gain = evaluate((chunk, score, kw_scores, length), best_coverage)
            if gain == 0:
                continue
            if val > best_value:
                best_value = val
                best_idx = i
                best_cost = cost
        if best_idx == -1:
            break
        if total_length + best_cost > budget:
            break
        chosen = candidates.pop(best_idx)
        selected.append(chosen[0])
        total_length += best_cost
        for kw, score in chosen[2].items():
            best_coverage[kw] = max(best_coverage.get(kw, 0), score)

    return selected


# ========================= 新的主函数 =========================
def batch_rerank_and_clip(total_chunks, ids2name,
                          requirement_groups: List[List[str]],
                          B: int,
                          delta: int = 0,
                          min_score: float = 0.0) -> str:
    # 1. 去重收集 chunk 对象
    seen_ids = set()
    chunk_list = []
    for item in total_chunks:
        c = item.chunk
        cid = getattr(c, 'chunk_id', None)
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            chunk_list.append(c)

    if not chunk_list:
        return ""

    all_keywords = list({w for g in requirement_groups for w in g})

    # 2. 表格裁剪（保留表格前后文本）
    for c in chunk_list:
        original = getattr(c, 'content', '')
        if '<table' in original.lower():
            try:
                # 分割出表格前后的文本
                parts = original.split('<table', 1)
                before = parts[0]
                rest = parts[1]
                table_part, _, after = rest.partition('</table>')
                # 重新组装完整表格HTML
                full_table = f"<table{table_part}</table>"
                clipped_table = clip_table(full_table, all_keywords)
                # 如果裁剪后仍包含 <table>，则替换；否则保留原样
                if '<table' in clipped_table.lower():
                    c.content = before + clipped_table + after
                else:
                    # 裁剪可能返回了非表格内容，保持原内容不变
                    pass
            except Exception:
                pass  # 裁剪失败则保留原始内容

    # 3. 新颖性得分
    novelty_dict = compute_novelty_scores(chunk_list, all_keywords)

    # 4. 分组覆盖评分与覆盖信息
    ranker = CoverageRanker(requirement_groups)
    chunk_infos = []
    for c in chunk_list:
        content = get_chunk_text(c)
        cid = getattr(c, 'chunk_id', '')
        nov = novelty_dict.get(cid, 0.0)
        score = ranker.score_chunk(c, novelty_score=nov)

        # 构建 keyword_scores 字典
        keyword_scores = {}
        for kw in all_keywords:
            s = fuzzy_match_score(kw, content, synonyms=FINANCIAL_SYNONYMS.get(kw, []))
            if s == 0:
                s = fuzzy_match_score(kw, content, synonyms=COMPARISON_SYNONYMS.get(kw, []))
            if s > 0:
                keyword_scores[kw] = s
        chunk_infos.append((c, score, keyword_scores, len(content)))

    # 5. 贪心覆盖选择
    budget = B + delta
    docs = defaultdict(list)
    for chunk in chunk_infos:
        docs[chunk[0].doc_id].append(chunk)
    
    selected = []
    for doc_chunks in docs.values():
        selected +=greedy_coverage_selection(doc_chunks, requirement_groups, budget, min_score, alpha=10.0)
    selected.sort(key=lambda x:x.chunk_index)

    # 6. 生成 Markdown
    lines = []
    for c in selected:
        doc_id = getattr(c, 'doc_id', 'unknown')
        doc_name = ids2name.get(doc_id, doc_id) if ids2name else doc_id
        title = getattr(c, 'section_title', '') or getattr(c, 'section_path', '')
        header = f"## {doc_name}"
        if title:
            header += f" - {title}"
        lines.append(header)
        lines.append(getattr(c, 'content', ''))
        lines.append("")
        lines.append("---")
        lines.append("")
    return '\n'.join(lines)