import re
import math
from typing import List, Dict, Tuple, Optional, Union
from difflib import SequenceMatcher
from itertools import chain

try:
    from bs4 import BeautifulSoup, Tag
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    raise ImportError("请安装 beautifulsoup4: pip install beautifulsoup4")

# ========================= 同义词表（可按业务持续扩充） =========================
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
}
# 反向映射，方便查找
SYNONYM_TO_STANDARD = {}
for standard, aliases in FINANCIAL_SYNONYMS.items():
    for alias in aliases:
        SYNONYM_TO_STANDARD[alias] = standard
    SYNONYM_TO_STANDARD[standard] = standard  # 标准名自身也映射

# 比较词同义映射
COMPARISON_SYNONYMS = {
    "同比": ["较上年", "同比增长", "同比变动", "比上年", "较去年同期"],
    "占比": ["比例", "占", "百分比", "份额"],
    "高于": ["大于", "超过", "优于"],
    "低于": ["小于", "不足", "低于"],
    "增长": ["增加", "上升", "提高", "增幅"],
    "下降": ["减少", "降低", "下滑", "降幅"],
}


# ========================= 基础工具函数 =========================
def normalize_text(s: str) -> str:
    """去除多余空格、换行，全角转半角"""
    s = s.replace('\n', ' ').replace('\r', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    # 简单全角转半角
    result = []
    for ch in s:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(ch)
    return ''.join(result)


def fuzzy_match(query: str, target: str, threshold_exact: float = 0.9,
                threshold_partial: float = 0.6) -> float:
    """
    返回匹配得分：1.0 完全命中，0.7 部分命中，0.0 未命中
    基于字符级序列相似度
    """
    q = normalize_text(query)
    t = normalize_text(target)
    # 预处理：去掉括号内容等（可调）
    ratio = SequenceMatcher(None, q, t).ratio()
    if ratio >= threshold_exact:
        return 1.0
    elif ratio >= threshold_partial:
        return 0.7
    else:
        return 0.0


def contains_number(text: str) -> bool:
    """判断字符串中是否包含数值（含百分数、千分位）"""
    pattern = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%?|[-+]?\.\d+\s*%?'
    return bool(re.search(pattern, text))


def has_comparison_word(text: str, comparison_words: List[str]) -> int:
    """统计片段中命中的比较词数量（含同义扩展）"""
    count = 0
    lower_text = normalize_text(text).lower()
    for word in comparison_words:
        # 原词
        if word.lower() in lower_text:
            count += 1
            continue
        # 同义词
        synonyms = COMPARISON_SYNONYMS.get(word, [])
        for syn in synonyms:
            if syn.lower() in lower_text:
                count += 1
                break
    return count


def is_aggregate_row(cells_text: List[str]) -> bool:
    """判断一行是否为小计/合计行"""
    joined = ' '.join(cells_text)
    keywords = ['合计', '小计', '总计', 'sum', 'total', '累计']
    return any(kw in joined.lower() for kw in keywords)


# ========================= 表格解析与裁剪 =========================
def parse_html_table(html_content: str) -> Optional[List[List[str]]]:
    """将HTML表格解析为二维列表，保留合并单元格的基本信息（简化版）"""
    if not HAS_BS4:
        return None
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if not table:
        return None
    grid = []
    rows = table.find_all('tr')
    for row in rows:
        cells = row.find_all(['th', 'td'])
        row_data = []
        for cell in cells:
            # 处理colspan/rowspan（简单扩展，这里仅做标记，实际裁剪时会重新构建表格）
            text = cell.get_text(separator=' ', strip=True)
            row_data.append(text)
        if row_data:
            grid.append(row_data)
    return grid


def locate_matches(grid: List[List[str]], indicators: List[str]) -> Dict:
    """
    在表格二维数组中定位指标匹配的单元格。
    返回：{
        'row_matched': {指标: [(row_idx, col_idx), ...]},
        'col_matched': {指标: [(row_idx, col_idx), ...]},
    }
    """
    row_matches = {}
    col_matches = {}
    for i_idx, indicator in enumerate(indicators):
        # 扩展同义词
        candidates = [indicator] + FINANCIAL_SYNONYMS.get(indicator, [])
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                for cand in candidates:
                    score = fuzzy_match(cand, cell, threshold_exact=0.9, threshold_partial=0.6)
                    if score > 0:
                        # 粗略判断：如果表格列数较少（<=3），可能为键值对，优先行匹配
                        if len(grid[0]) <= 3 and c == 0:  # 指标在第一列
                            row_matches.setdefault(indicator, []).append((r, c))
                        elif r == 0:  # 第一行表头
                            col_matches.setdefault(indicator, []).append((r, c))
                        else:
                            # 其他位置也记录下来，后续决策
                            col_matches.setdefault(indicator, []).append((r, c))
    return {'row': row_matches, 'col': col_matches}


def clip_table(html_content: str, indicators: List[str]) -> str:
    """
    裁剪表格：保留与指标相关的列/行，并保留合计行。
    返回裁剪后的HTML表格字符串。
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if not table:
        return html_content  # 非表格直接返回

    rows = table.find_all('tr')
    if not rows:
        return html_content

    # 解析为二维文本数组，同时保留对BeautifulSoup元素的引用
    grid = []
    row_elements = []  # 保存每一行的Tag元素
    for row in rows:
        cells = row.find_all(['th', 'td'])
        if not cells:
            continue
        row_texts = [cell.get_text(separator=' ', strip=True) for cell in cells]
        grid.append(row_texts)
        row_elements.append((row, cells))

    if not grid:
        return html_content

    matches = locate_matches(grid, indicators)
    row_matched_indicators = matches['row']
    col_matched_indicators = matches['col']

    # 确定要保留的行索引和列索引
    keep_rows = set()
    keep_cols = set()

    # 1. 根据行匹配（键值对形式）保留整行
    for indicator, positions in row_matched_indicators.items():
        for r, c in positions:
            keep_rows.add(r)
            # 对于键值对表，数值一般在同行的下一列，也保留
            if len(grid[0]) > c + 1:
                keep_cols.add(c + 1)  # 保留数值列

    # 2. 根据列匹配（表头形式）保留整列
    for indicator, positions in col_matched_indicators.items():
        for r, c in positions:
            if r == 0:  # 表头行
                keep_cols.add(c)
                # 如果该列下方有数值行，那些行也应保留（通过后续逻辑）
                for row_idx in range(1, len(grid)):
                    if c < len(grid[row_idx]) and contains_number(grid[row_idx][c]):
                        keep_rows.add(row_idx)
            else:
                # 其他位置的匹配，保留所在行
                keep_rows.add(r)

    # 3. 强制保留表头行（第一行）和所有合计行
    if len(grid) > 0:
        keep_rows.add(0)  # 表头
    for r, row_texts in enumerate(grid):
        if is_aggregate_row(row_texts):
            keep_rows.add(r)

    # 4. 如果没有明确保留列，但保留了行，则保留所有列（以免丢失数据）
    if keep_rows and not keep_cols:
        keep_cols = set(range(len(grid[0])))

    # 5. 重建裁剪后的表格
    new_table = soup.new_tag('table')
    # 复制原table的属性（如class）
    for attr, value in table.attrs.items():
        new_table[attr] = value

    for r_idx, (row_tag, cells) in enumerate(row_elements):
        if r_idx not in keep_rows:
            continue
        new_row = soup.new_tag('tr')
        # 复制行的属性
        for attr, value in row_tag.attrs.items():
            new_row[attr] = value
        for c_idx, cell in enumerate(cells):
            if not keep_cols or c_idx in keep_cols:
                new_cell = soup.new_tag(cell.name)
                # 复制属性
                for attr, value in cell.attrs.items():
                    new_cell[attr] = value
                # 保留内部HTML
                new_cell.append(BeautifulSoup(str(cell), 'html.parser'))
                new_row.append(new_cell)
        new_table.append(new_row)

    return str(new_table)


# ========================= 精排打分 =========================
class FinancialChunkRanker:
    def __init__(self, indicators: List[str], comparison_words: List[str],
                 weights: Optional[Dict[str, float]] = None):
        self.indicators = indicators
        self.comparison_words = comparison_words
        # 默认权重
        self.weights = weights or {
            'indicator': 0.4,
            'number': 0.3,
            'comparison': 0.2,
            'structure': 0.1
        }
        # 如果没有比较词，调整权重
        if not self.comparison_words:
            self.weights['indicator'] += self.weights.pop('comparison', 0.2)

    def _indicator_score(self, content: str) -> float:
        """指标命中分（满分40）"""
        if not self.indicators:
            return 40.0
        total_hit = 0.0
        for ind in self.indicators:
            # 尝试直接匹配指标或同义词
            max_score = 0.0
            # 检查原词
            max_score = max(max_score, fuzzy_match(ind, content))
            # 检查同义词
            for syn in FINANCIAL_SYNONYMS.get(ind, []):
                max_score = max(max_score, fuzzy_match(syn, content))
            total_hit += max_score
        score = (total_hit / len(self.indicators)) * 40
        # 完全命中加成
        if total_hit == len(self.indicators):
            score = min(40, score + 4)
        return score

    def _number_existence_score(self, content: str, is_table: bool = False) -> float:
        """数值存在分（满分30）"""
        if not self.indicators:
            return 30.0
        # 对于表格内容，我们已经解析过；这里使用简单窗口法
        effective_pairs = 0
        # 简单的全文本窗口检测
        for ind in self.indicators:
            # 寻找指标出现的位置
            candidates = [ind] + FINANCIAL_SYNONYMS.get(ind, [])
            found_num = False
            for cand in candidates:
                # 在content中查找cand的位置
                start = 0
                while True:
                    pos = content.find(cand, start)
                    if pos == -1:
                        break
                    # 检查前后30个字符内是否有数字
                    window_start = max(0, pos - 30)
                    window_end = min(len(content), pos + len(cand) + 30)
                    window_text = content[window_start:window_end]
                    if contains_number(window_text):
                        found_num = True
                        break
                    start = pos + len(cand)
                if found_num:
                    break
            if found_num:
                effective_pairs += 1
        return (effective_pairs / len(self.indicators)) * 30

    def _comparison_score(self, content: str) -> float:
        """比较词覆盖分（满分20）"""
        if not self.comparison_words:
            return 0.0  # 权重已转移到指标分，这里返回0
        total = len(self.comparison_words)
        hit = has_comparison_word(content, self.comparison_words)
        return (hit / total) * 20 if total > 0 else 0.0

    def _structure_score(self, content: str, has_table: bool, has_aggregate: bool,
                         is_continuous_text: bool, length: int) -> float:
        """结构优势分（满分10）"""
        score = 0.0
        if has_table:
            score += 5
            if has_aggregate:
                score += 3
        elif is_continuous_text:
            score += 2
        if length > 800 and self._indicator_score(content) <= 10:
            score -= 2
        return max(0, min(10, score))

    def score_chunk(self, chunk: dict) -> float:
        """对一个检索片段打分"""
        content = chunk.content
        # 判断是否包含表格
        has_table = '<table' in content.lower()
        # 解析表格以获得更深层信息
        grid = None
        has_aggregate = False
        if has_table:
            grid = parse_html_table(content)
            if grid:
                has_aggregate = any(is_aggregate_row(row) for row in grid)
        # 其他结构判断
        is_continuous_text = not has_table and len(content.split('\n')) < 5  # 简单判断为连续段落

        s1 = self._indicator_score(content)
        s2 = self._number_existence_score(content, is_table=has_table)
        s3 = self._comparison_score(content)
        s4 = self._structure_score(content, has_table, has_aggregate, is_continuous_text, len(content))

        # 如果比较词为空，指标得分已加成，因此直接用权重计算
        total = (s1 * (self.weights.get('indicator', 0.4) / 0.4 if not self.comparison_words else 1)
                 + s2 + s3 + s4)  # 注意权重系数归一化
        # 重新用标准权重计算
        if not self.comparison_words:
            # 此时指标权重变为0.6，数值0.3，结构0.1
            total = s1 * (0.6/0.4) + s2 + s4  # s3=0
        else:
            total = s1 + s2 + s3 + s4
        # 截断到100
        return min(100, total)


# ========================= 主流程 =========================
def rerank_and_clip(chunks: List[dict], question_indicators: List[str],
                    question_comparisons: List[str], top_k: int = 5) -> List[dict]:
    """
    对检索片段进行精排和裁剪
    :param chunks: 检索结果列表，每个元素至少包含 content 字段
    :param question_indicators: 从问题中抽取的财务指标列表
    :param question_comparisons: 比较词列表（如 ['同比', '增长']）
    :param top_k: 保留前K个片段进行裁剪
    :return: 排序并裁剪后的片段列表，每个片段增加 rank_score 字段
    """
    ranker = FinancialChunkRanker(question_indicators, question_comparisons)

    # 1. 计算所有片段得分
    scored_chunks = []
    for chunk in chunks:
        score = ranker.score_chunk(chunk)
        scored_chunks.append((score, chunk))

    # 2. 按得分降序排序
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # 3. 取前K个进行裁剪
    result = []
    for rank, (score, chunk) in enumerate(scored_chunks[:top_k], start=1):
        new_chunk = dict(chunk)
        new_chunk['rank_score'] = score
        new_chunk['rank'] = rank
        content = chunk.content
        if '<table' in content.lower():
            try:
                clipped_content = clip_table(content, question_indicators)
                new_chunk['content'] = clipped_content
                new_chunk['clipped'] = True
            except Exception as e:
                # 裁剪失败则保留原文
                new_chunk['clipped'] = False
                new_chunk['clip_error'] = str(e)
        else:
            # 非表格片段完整保留
            new_chunk['clipped'] = False
        result.append(new_chunk)

    return result

def batch_rerank_and_clip(total_chunks, ids2name, question_indicators: List[str],
                    question_comparisons: List[str], top_k: int = 5):
    # 1. 去重并构造候选信息
    seen_ids = set()
    chunk_list = []
    for item in total_chunks:
        if item.chunk.chunk_id not in seen_ids:
            seen_ids.add(item.chunk.chunk_id)
            cost = len(item.chunk.content)  # 字符数近似 token 数
            chunk_list.append(item.chunk)

    # 2. 按文档分组，应用最低分数阈值（并保证每篇至少一个候选）
    docs_candidates = {}   # doc_id -> list of candidates
    for cand in chunk_list:
        doc_id = cand.doc_id
        if doc_id not in docs_candidates:
            docs_candidates[doc_id] = []
        docs_candidates[doc_id].append(cand)

    docs = {}
    for k,v in docs_candidates.items():
        chunks = rerank_and_clip(v, question_indicators, question_comparisons, top_k)
        docs[k] = chunks
    
    used_titles = []
    md_lines = []
    for doc_id, chunks in docs.items():
        md_lines.append(f"# {ids2name[doc_id]}")
        # 按 chunk_index 排序
        chunks.sort(key=lambda x: x['chunk_index'])
        for chunk in chunks:
            section_path = chunk['section_path']
            if section_path not in used_titles:
                used_titles.append(section_path)
                md_lines.append(f"## {section_path}")
            md_lines.append(chunk['content'])

    return "\n".join(md_lines)



# ========================= 示例用法 =========================
if __name__ == "__main__":
    # 模拟检索返回的片段（使用你提供的样例）
    sample_chunks = [
        {
            "chunk_id": "300750_2024_chunk_7_v1782048616",
            "content": """
            <table>
                <tr><td>股票简称</td><td>宁德时代</td></tr>
                <tr><td>营业收入</td><td>3620.58亿元</td></tr>
                <tr><td>净利润</td><td>441.21亿元</td></tr>
                <tr><td>同比增长</td><td>22.3%</td></tr>
                <tr><td>合计</td><td>—</td></tr>
            </table>
            """
        },
        {
            "chunk_id": "some_text_chunk",
            "content": "2024年，公司实现营业收入3620.58亿元，较上年增长22.3%；归属于母公司股东的净利润为441.21亿元，同比增长18.5%。"
        }
    ]

    # 模拟问题抽取
    indicators = ["营业收入", "净利润"]
    comparisons = ["同比", "增长"]

    top_chunks = rerank_and_clip(sample_chunks, indicators, comparisons, top_k=5)

    for c in top_chunks:
        print(f"Chunk ID: {c['chunk_id']}, Score: {c['rank_score']:.2f}, Clipped: {c['clipped']}")
        print(c['content'][:200])
        print("------")