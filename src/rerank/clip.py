import re
from typing import List, Dict, Optional, Set, Tuple
from bs4 import BeautifulSoup, Tag

# 假设以下函数已定义：
# normalize_text, contains_number, fuzzy_match_score, fuzzy_contains
# FINANCIAL_SYNONYMS, COMPARISON_SYNONYMS


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
def clip_table_v2(html_content: str,
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