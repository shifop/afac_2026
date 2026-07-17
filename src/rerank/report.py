import re
import math
from typing import List, Dict, Optional, Any, Set, Tuple
from difflib import SequenceMatcher
from .insurance import select_chunks_with_budget

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
    return ' '.join(parts)


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


# ========================= 主裁剪函数 =========================
def clip_table(html_content: str,
                  indicators: List[str],
                  comparison_words: List[str]) -> str:
    """
    裁剪表格：
    1. 解析并展开合并单元格。
    2. 识别所有非数字单元格为标签，判断其行列角色。
    3. 多层表头：分别对每层单单元格匹配，也对合并后的多层文本匹配；任一命中则保留对应列（列表头）或行（行表头）。
    4. 合计/小计行无条件保留。
    5. 清理空行/列。
    6. 重建HTML表格。
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    if not soup.find('table'):
        return html_content

    tag_grid, text_grid, n_rows, n_cols = expand_table(soup)
    if n_rows == 0 or n_cols == 0:
        return html_content

    # 需要保留的行和列索引
    keep_rows: Set[int] = set()
    keep_cols: Set[int] = set()

    # 收集所有指标+比较词的列表（统一匹配）
    match_terms = list(indicators) + list(comparison_words)

    # 遍历所有单元格，进行标签识别和匹配
    for r in range(n_rows):
        for c in range(n_cols):
            cell_text = text_grid[r][c]
            if contains_number(cell_text):
                # 数字单元格暂不处理（会通过行/列保留关联）
                continue

            # 非数字单元格 → 标签
            # 判定角色
            role = identify_header_role(text_grid, r, c, n_rows, n_cols)

            # 多层表头处理：若角色为列表头，则需要同时考虑该列上下表头行合并匹配
            # 但我们采用简化策略：在匹配时，除了匹配当前单元格，还匹配从当前单元格开始向上合并的所有表头文本（如果角色是列表头）
            # 这里实现单层匹配和合并多层匹配。

            # 单层匹配
            hit = False
            for term in match_terms:
                if term in indicators:
                    if match_indicator_in_cell(term, cell_text):
                        hit = True
                        break
                else:
                    if match_comparison_in_cell(term, cell_text):
                        hit = True
                        break

            # 合并多层表头匹配（仅对列表头有意义，因为列标题通常位于上方）
            if not hit and role == 'column_header':
                # 向上收集同一列的所有非数字单元格文本，合并后匹配
                combined_text = []
                for rr in range(r, -1, -1):
                    txt = text_grid[rr][c]
                    if not contains_number(txt):
                        combined_text.insert(0, txt)
                    else:
                        break
                if combined_text:
                    full_label = ' '.join(combined_text)
                    for term in match_terms:
                        if term in indicators:
                            if match_indicator_in_cell(term, full_label):
                                hit = True
                                break
                        else:
                            if match_comparison_in_cell(term, full_label):
                                hit = True
                                break

            # 如果命中，根据角色保留行或列
            if hit:
                if role == 'column_header':
                    keep_cols.add(c)
                elif role == 'row_header':
                    keep_rows.add(r)
                else:  # unknown 保守保留整行整列（但可能重复）
                    keep_rows.add(r)
                    keep_cols.add(c)

    # 保留与行表头相关联的数字列？根据之前规则：行表头命中保留整行所有列，所以无需额外操作，整行保留即可。
    # 但是注意：行保留后，所有列都会被保留，可能引入无关列。但规则是保留整行所有列，所以可以接受。
    # 如果需要保留列，就加进keep_cols。

    # 合计/小计行无条件保留
    for r in range(n_rows):
        row_text = ' '.join(text_grid[r][c] for c in range(n_cols))
        if is_aggregate_row(row_text.split()):  # 使用之前定义的is_aggregate_row
            keep_rows.add(r)

    # 保证表头行（前面几行通常是表头）至少有一行保留？不必强制，但如果保留了列，第一行（通常表头）也应该保留。
    # 这里可以加一个安全措施：如果保留了列，自动保留第一行（假设第一行是表头），除非第一行全是数字？但基本表头都在前几行，简单处理：保留前3行中非数字行。
    for r in range(min(3, n_rows)):
        if any(not contains_number(text_grid[r][c]) for c in range(n_cols)):
            keep_rows.add(r)

    # 如果只保留了列但没有保留任何行（不太可能），至少保留所有行
    if keep_cols and not keep_rows:
        keep_rows = set(range(n_rows))

    # 如果只保留了行但没有列，保留所有列
    if keep_rows and not keep_cols:
        keep_cols = set(range(n_cols))

    # 清理空行/列（合计行豁免）
    # 先确定哪些行是合计行
    aggregate_rows = set()
    for r in range(n_rows):
        row_text = ' '.join(text_grid[r][c] for c in range(n_cols))
        if is_aggregate_row(row_text.split()):
            aggregate_rows.add(r)

    # 空行清理
    final_rows = set(keep_rows)
    for r in list(final_rows):
        if r in aggregate_rows:
            continue
        row_all_empty = True
        for c in range(n_cols):
            if c in keep_cols and text_grid[r][c].strip() not in ('', '-', '—', 'N/A', '无', '/'):
                row_all_empty = False
                break
        if row_all_empty:
            final_rows.remove(r)

    # 空列清理
    final_cols = set(keep_cols)
    for c in list(final_cols):
        col_all_empty = True
        for r in range(n_rows):
            if r in final_rows and text_grid[r][c].strip() not in ('', '-', '—', 'N/A', '无', '/'):
                col_all_empty = False
                break
        if col_all_empty:
            final_cols.remove(c)

    # 确保至少有一行一列
    if not final_rows or not final_cols:
        return html_content  # 裁剪失败，返回原表

    # 重建HTML表格
    new_table = soup.new_tag('table')
    # 复制原table属性
    original_table = soup.find('table')
    for attr, val in original_table.attrs.items():
        new_table[attr] = val

    for r in sorted(final_rows):
        new_row = soup.new_tag('tr')
        for c in sorted(final_cols):
            cell_tag = tag_grid[r][c]
            # 创建新单元格，保留类型（th/td）和属性（除colspan/rowspan）
            new_cell = soup.new_tag(cell_tag.name)
            for attr, val in cell_tag.attrs.items():
                if attr not in ('colspan', 'rowspan'):
                    new_cell[attr] = val
            # 设置文本
            if cell_tag.string:
                new_cell.string = cell_tag.string
            else:
                # 如果原单元格有子元素，复制innerHTML（简单处理：用文本即可）
                new_cell.string = cell_tag.get_text(separator=' ', strip=True)
            new_row.append(new_cell)
        new_table.append(new_row)

    return str(new_table)


# ========================= 精排打分器 =========================
class FinancialChunkRanker:
    def __init__(self, indicators: List[str], comparison_words: List[str]):
        self.indicators = indicators
        self.comparison_words = comparison_words
        self.max_scores = {
            'indicator': 35,
            'number': 30,
            'comparison': 15,
            'structure': 10,
            'novelty': 10
        }
        if not self.comparison_words:
            self.max_scores['indicator'] += self.max_scores.pop('comparison')
            self.max_scores['comparison'] = 0

    def _indicator_score(self, content: str) -> float:
        if not self.indicators:
            return float(self.max_scores['indicator'])
        total = 0.0
        for ind in self.indicators:
            total += fuzzy_match_score(ind, content, synonyms=FINANCIAL_SYNONYMS.get(ind, []))
        score = (total / len(self.indicators)) * self.max_scores['indicator']
        if total == len(self.indicators):  # 全部完全命中
            score = min(self.max_scores['indicator'], score + 2)
        return score

    def _number_existence_score(self, content: str) -> float:
        if not self.indicators:
            return float(self.max_scores['number'])
        if not contains_number(content):
            return 0.0
        effective = 0
        for ind in self.indicators:
            if fuzzy_match_score(ind, content, synonyms=FINANCIAL_SYNONYMS.get(ind, [])) > 0:
                effective += 1
        return (effective / len(self.indicators)) * self.max_scores['number']

    def _comparison_score(self, content: str) -> float:
        if not self.comparison_words:
            return 0.0
        hit = 0
        for comp in self.comparison_words:
            if fuzzy_contains(comp, content, synonyms=COMPARISON_SYNONYMS.get(comp, []),
                              threshold=0.8, short_threshold=0.9):
                hit += 1
        total_comp = len(self.comparison_words)
        return (hit / total_comp) * self.max_scores['comparison'] if total_comp else 0.0

    def _structure_score(self, content: str, has_table: bool,
                         has_aggregate: bool, is_continuous_text: bool,
                         length: int) -> float:
        score = 0.0
        if has_table:
            score += 5
            if has_aggregate:
                score += 3
        elif is_continuous_text:
            score += 2
        if length > 800 and self._indicator_score(content) <= 5:
            score -= 2
        return max(0, min(self.max_scores['structure'], score))

    def score_chunk(self, chunk: Dict, novelty_score: float = 0.0) -> float:
        if chunk.chunk_id=='8133a1f41b7b04cedfbc3933025ea8c4_chunk_108_v1782061791':
            print('')
        content = get_chunk_text(chunk)
        has_table = '<table' in content.lower()
        grid = None
        has_aggregate = False
        if has_table:
            grid = parse_html_table(content)
            if grid:
                has_aggregate = any(is_aggregate_row(row) for row in grid)
        is_continuous = not has_table and len(content.split('\n')) < 5

        s1 = self._indicator_score(content)
        s2 = self._number_existence_score(content)
        s3 = self._comparison_score(content)
        s4 = self._structure_score(content, has_table, has_aggregate, is_continuous, len(content))
        s5 = min(self.max_scores['novelty'], novelty_score)

        total = s1 + s2 + s3 + s4 + s5
        return min(100.0, total)


# ========================= 独有性计算（改用属性访问） =========================
def compute_novelty_scores(chunks: List[Any],
                           indicators: List[str],
                           comparisons: List[str]) -> Dict[Any, float]:
    id_map = {}
    for i, chunk in enumerate(chunks):
        cid = getattr(chunk, 'chunk_id', f'chunk_{i}')
        id_map[cid] = chunk
    cids = list(id_map.keys())
    N = len(chunks)
    if N <= 1:
        return {cid: 0.0 for cid in cids}

    indicator_df = {ind: 0 for ind in indicators}
    comparison_df = {comp: 0 for comp in comparisons}

    for chunk in chunks:
        content = get_chunk_text(chunk)
        for ind in indicators:
            if fuzzy_match_score(ind, content, synonyms=FINANCIAL_SYNONYMS.get(ind, [])) > 0:
                indicator_df[ind] += 1
        for comp in comparisons:
            if fuzzy_contains(comp, content, synonyms=COMPARISON_SYNONYMS.get(comp, []),
                              threshold=0.8, short_threshold=0.9):
                comparison_df[comp] += 1

    def idf_weight(df):
        if df == 0:
            return 0.0
        return max(0.0, math.log((N + 0.5) / (df + 0.5)))

    indicator_idf = {ind: idf_weight(df) for ind, df in indicator_df.items()}
    comparison_idf = {comp: idf_weight(df) for comp, df in comparison_df.items()}

    raw_scores = {}
    for cid in cids:
        chunk = id_map[cid]
        content = get_chunk_text(chunk)
        raw = 0.0
        for ind in indicators:
            if fuzzy_contains(ind, content, synonyms=FINANCIAL_SYNONYMS.get(ind, [])):
                raw += indicator_idf[ind]
        for comp in comparisons:
            if fuzzy_contains(comp, content, synonyms=COMPARISON_SYNONYMS.get(comp, []),
                              threshold=0.8, short_threshold=0.9):
                raw += 0.8 * comparison_idf[comp]
        raw_scores[cid] = raw

    if not raw_scores:
        return {cid: 0.0 for cid in cids}

    max_raw = max(raw_scores.values()) if raw_scores else 1.0
    alpha = math.atanh(0.95) / max_raw if max_raw != 0 else 1.0

    return {cid: round(math.tanh(raw * alpha) * 10.0, 2) for cid, raw in raw_scores.items()}


# ========================= 主流程（对象 → 字典） =========================
def rerank_and_clip(chunks: List[Any],
                    question_indicators: List[str],
                    question_comparisons: List[str],
                    top_k: int = 5) -> List[Dict]:
    novelty_dict = compute_novelty_scores(chunks, question_indicators, question_comparisons)
    ranker = FinancialChunkRanker(question_indicators, question_comparisons)

    scored = []
    for chunk in chunks:
        cid = getattr(chunk, 'chunk_id', '')
        nov = novelty_dict.get(cid, 0.0)
        score = ranker.score_chunk(chunk, novelty_score=nov)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    result = []
    for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
        # 从对象属性构建字典
        new_chunk = {
            'chunk_id': getattr(chunk, 'chunk_id', ''),
            'doc_id': getattr(chunk, 'doc_id', ''),
            'section_title': getattr(chunk, 'section_title', ''),
            'section_path': getattr(chunk, 'section_path', ''),
            'chunk_index': getattr(chunk, 'chunk_index', 0),
            'content': getattr(chunk, 'content', ''),
            'entities': getattr(chunk, 'entities', []),
            'relations': getattr(chunk, 'relations', []),
        }
        new_chunk['rank_score'] = round(score, 2)
        new_chunk['rank'] = rank

        content = new_chunk['content']
        if '<table' in content.lower():
            try:
                text = content.split('<table')[0]
                text2 = content.split('</table>')[-1]
                clipped = text+"\n"+clip_table(content[len(text):-len(text2)], question_indicators, question_comparisons)+'\n'+text2
                new_chunk['content'] = clipped
                new_chunk['clipped'] = True
            except Exception as e:
                new_chunk['clipped'] = False
                new_chunk['clip_error'] = str(e)
        else:
            new_chunk['clipped'] = False
        result.append(new_chunk)

    return result


# ========================= 新增：全局打分函数 =========================
def score_all_chunks(chunks: List[Any],
                     indicators: List[str],
                     comparisons: List[str]) -> Dict[Any, float]:
    """对所有片段进行精排打分，返回 {chunk_id: score}"""
    novelty_dict = compute_novelty_scores(chunks, indicators, comparisons)
    ranker = FinancialChunkRanker(indicators, comparisons)
    scores = {}
    for chunk in chunks:
        cid = getattr(chunk, 'chunk_id', '')
        nov = novelty_dict.get(cid, 0.0)
        score = ranker.score_chunk(chunk, novelty_score=nov)
        scores[cid] = score
    return scores


# ========================= 修改后的 batch_rerank_and_clip =========================
def batch_rerank_and_clip(total_chunks, ids2name,
                          question_indicators: List[str],
                          question_comparisons: List[str],
                          B: int,
                          delta: int = 0,
                          min_score: float = 0.0) -> str:
    """
    优化版：精排 + 表格裁剪 + 预算选择，不硬截断 top_k。
    
    :param total_chunks: 元素具有 .chunk 属性的列表（chunk 对象需包含 chunk_id, doc_id, content, chunk_index, section_path 等）
    :param ids2name: doc_id -> 文档名称映射
    :param question_indicators: 问题中的财务指标
    :param question_comparisons: 问题中的比较词
    :param B: 目标字符数预算
    :param delta: 允许的弹性字符数（总上限 = B + delta）
    :param min_score: 片段最低得分阈值
    :return: Markdown 字符串
    """
    # 1. 去重并收集原始 chunk 对象
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

    # 2. 精排打分（基于原始完整内容）
    scores = score_all_chunks(chunk_list, question_indicators, question_comparisons)

    # 3. 对每个片段执行表格裁剪，并更新内容为裁剪后的版本
    #    同时计算裁剪后的实际成本（字符数）
    for c in chunk_list:
        original_content = getattr(c, 'content', '')
        if '<table' in original_content.lower():
            try:
                clipped = clip_table(original_content, question_indicators, question_comparisons)
                # 直接修改内容（假设对象属性可写；若不可写，可改用其他方式）
                c.content = clipped
            except Exception:
                pass  # 裁剪失败则保留原内容

    # 4. 构造 select_chunks_with_budget 所需的输入格式
    #    每个元素需有 .chunk 和 .score 属性
    class ScoredChunk:
        def __init__(self, chunk, score):
            self.chunk = chunk
            self.score = score

    scored_items = [ScoredChunk(c, scores[c.chunk_id]) for c in chunk_list]

    # 5. 调用预算选择，生成最终 Markdown
    md = select_chunks_with_budget(scored_items, ids2name, B, delta, min_score)
    return md


# ========================= 测试示例 =========================
if __name__ == "__main__":
    sample = [
        {
            "chunk_id": "chunk_001",
            "content": """
            <table>
                <tr><th>项目</th><th>2024年</th><th>2023年</th></tr>
                <tr><td>营业收入</td><td>3620.58亿元</td><td>2986.34亿元</td></tr>
                <tr><td>净利润</td><td>441.21亿元</td><td>378.53亿元</td></tr>
                <tr><td>海外收入</td><td>1020.33亿元</td><td>845.12亿元</td></tr>
                <tr><td>合计</td><td>—</td><td>—</td></tr>
            </table>
            """
        },
        {
            "chunk_id": "chunk_002",
            "content": "2024年公司海外业务收入达到1020.33亿元，占营收比重提升至28.2%，较上年增长20.7%。"
        },
        {
            "chunk_id": "chunk_003",
            "content": "公司研发投入持续增加，2024年研发费用为154.2亿元，同比增长12.3%。"
        }
    ]

    indicators = ["海外业务营收", "营收占比", "同比增速"]
    comparisons = ["同比", "占比"]

    top_chunks = rerank_and_clip(sample, indicators, comparisons, top_k=5)

    for c in top_chunks:
        print(f"Chunk: {c['chunk_id']}, Score: {c['rank_score']}, Clipped: {c['clipped']}")
        print(c['content'][:300])
        print("------")