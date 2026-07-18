"""表格处理模块 - HTML/Markdown 表格检测解析、虚拟句子生成、表格摘要"""
import re
from typing import List, Dict, Any, Optional
import logging; logger = logging.getLogger(__name__)

from .models import Sentence, Location, TableInfo, new_id


_HTML_TABLE_RE = re.compile(r"<table>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_TR_RE = re.compile(r"<tr>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD_RE = re.compile(r"<t[hd]([^>]*)>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r'(rowspan|colspan)\s*=\s*["\']?(\d+)["\']?', re.IGNORECASE)
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")


class TableProcessor:
    """表格处理器"""

    # ---- 对外接口 ----

    def detect_and_parse(
        self, doc_id: str, section_path: str, para_index: int, text: str
    ) -> Optional[TableInfo]:
        """检测并解析表格。优先 HTML 表格，其次 Markdown 表格。"""
        table = self._parse_html_table(text)
        if table is not None:
            table.section_path = section_path
            table.paragraph_index = para_index
            return table
        return self._parse_md_table(text, section_path, para_index)

    def to_virtual_sentences(
        self, doc_id: str, section_path: str, para_index: int, table: TableInfo
    ) -> List[Sentence]:
        """将表格单元格转换为虚拟句子"""
        sentences: List[Sentence] = []
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell_text in enumerate(row):
                if not cell_text.strip():
                    continue
                sent_id = f"{doc_id}#{section_path}_p{para_index}_t{row_idx}_{col_idx}"
                loc = Location(
                    doc_id=doc_id,
                    section_path=section_path,
                    paragraph_index=para_index,
                    sentence_index=-1,
                    table_id=table.table_id,
                    row_index=row_idx,
                    column_index=col_idx,
                    is_cell=True,
                )
                sentences.append(Sentence(sentence_id=sent_id, text=cell_text, location=loc))
        return sentences

    def build_context_for_llm(
        self, table: TableInfo, prev_paragraph_text: str = ""
    ) -> str:
        """为 LLM 调用构建表格上下文"""
        parts: list[str] = []
        if table.caption:
            parts.append(f"表格标题：{table.caption}")
        if prev_paragraph_text and len(prev_paragraph_text) < 200:
            parts.append(f"前文说明：{prev_paragraph_text}")
        parts.append(f"表头：{' | '.join(table.headers)}")
        parts.append("数据行：")
        for i, row in enumerate(table.rows):
            row_dict = dict(zip(table.headers, row))
            parts.append(f"  行{i}: {row_dict}")
        return "\n".join(parts)

    def generate_summary_prompt(self, table: TableInfo, prev_paragraph_text: str = "") -> str:
        context = self.build_context_for_llm(table, prev_paragraph_text)
        return (
            "请用一段简短的话（不超过80字）概括以下表格的内容，"
            "并推测数值列的单位（如'单位：百万元'）。\n\n" + context
        )

    # ---- HTML 表格解析 ----

    def _parse_html_table(self, text: str) -> Optional[TableInfo]:
        m = _HTML_TABLE_RE.search(text)
        if not m:
            return None

        body = m.group(1)
        rows_raw: list[list[str]] = []
        spans: list[list[tuple[int, int]]] = []  # (rowspan, colspan)

        for tr_m in _TR_RE.finditer(body):
            cells: list[str] = []
            cell_spans: list[tuple[int, int]] = []
            for td_m in _TD_RE.finditer(tr_m.group(1)):
                attrs = td_m.group(1)
                raw_content = td_m.group(2)
                # 去除内部 HTML 标签
                cell_text = _STRIP_TAGS_RE.sub("", raw_content).strip()
                rs, cs = 1, 1
                for attr_m in _ATTR_RE.finditer(attrs):
                    if attr_m.group(1).lower() == "rowspan":
                        rs = int(attr_m.group(2))
                    else:
                        cs = int(attr_m.group(2))
                cells.append(cell_text)
                cell_spans.append((rs, cs))
            if cells:
                rows_raw.append(cells)
                spans.append(cell_spans)

        if not rows_raw:
            return None

        # 展开 rowspan / colspan → 规整矩阵
        grid, headers = self._expand_spans(rows_raw, spans)
        if not headers or len(grid) == 0:
            return None

        table_id = new_id("tbl")
        return TableInfo(
            table_id=table_id,
            headers=headers,
            rows=grid,
            section_path="",
            paragraph_index=0,
        )

    def _expand_spans(
        self, rows_raw: list[list[str]], spans: list[list[tuple[int, int]]]
    ) -> tuple[list[list[str]], list[str]]:
        """展开 rowspan/colspan，返回规整的 (数据行列表, 表头列表)"""
        # 先计算总列数
        total_cols = 0
        for row_cells, row_spans in zip(rows_raw, spans):
            col_count = sum(cs for _, cs in row_spans)
            total_cols = max(total_cols, col_count)

        # 构建占位矩阵，先填 None
        max_rows = sum(max(rs for rs, _ in row) for row in spans)
        max_rows = max(max_rows, len(rows_raw))
        matrix: list[list[Optional[str]]] = [[None] * total_cols for _ in range(max_rows)]

        # 填充
        for r, (cells, row_spans) in enumerate(zip(rows_raw, spans)):
            c = 0
            # 跳过已被上面 rowspan 占用的列
            while c < total_cols and matrix[r][c] is not None:
                c += 1
            for cell_text, (rs, cs) in zip(cells, row_spans):
                while c < total_cols and matrix[r][c] is not None:
                    c += 1
                for dr in range(rs):
                    for dc in range(cs):
                        nr, nc = r + dr, c + dc
                        if nr < max_rows and nc < total_cols:
                            matrix[nr][nc] = cell_text
                c += cs

        # 提取表头（第一行非None行）
        header_row: list[str] = []
        header_idx = 0
        for r in range(len(matrix)):
            if any(v is not None for v in matrix[r]):
                header_row = [v if v is not None else "" for v in matrix[r]]
                header_idx = r
                break

        # 数据行：表头之后的行
        data_rows: list[list[str]] = []
        for r in range(header_idx + 1, len(matrix)):
            row = [v if v is not None else "" for v in matrix[r]]
            if any(v.strip() for v in row):
                data_rows.append(row)

        return data_rows, header_row

    # ---- Markdown 表格解析 ----

    def _parse_md_table(self, text: str, section_path: str, para_index: int) -> Optional[TableInfo]:
        """解析 Markdown 表格"""
        lines = text.strip().split("\n")
        if len(lines) < 2:
            return None

        pipe_lines = [i for i, ln in enumerate(lines) if "|" in ln]
        if len(pipe_lines) < 2:
            return None

        start = pipe_lines[0]
        end = start
        for i in range(1, len(pipe_lines)):
            if pipe_lines[i] == pipe_lines[i - 1] + 1:
                end = pipe_lines[i]
            else:
                break

        table_lines = lines[start : end + 1]
        if len(table_lines) < 2:
            return None

        # 第二行必须是分隔行
        if not re.match(r"^\|[\s\-:\|]+$", table_lines[1].strip()):
            return None

        headers = self._parse_md_row(table_lines[0])
        if not headers:
            return None

        rows = []
        for ln in table_lines[2:]:
            row = self._parse_md_row(ln)
            if row:
                while len(row) < len(headers):
                    row.append("")
                rows.append(row[: len(headers)])

        table_id = new_id("tbl")
        return TableInfo(
            table_id=table_id,
            headers=headers,
            rows=rows,
            section_path=section_path,
            paragraph_index=para_index,
        )

    def _parse_md_row(self, line: str) -> List[str]:
        line = line.strip().strip("|")
        return [c.strip() for c in line.split("|")]
