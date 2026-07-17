#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boundary-preserving domain chunker.

Unlike the legacy implementation, normal-sized sections are emitted directly;
only genuinely short sibling segments are merged, and every oversize segment is
split before being emitted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from text_utils import stable_hash


@dataclass
class ChunkDraft:
    title: str
    level: int
    chunk_type: str
    char_start: int
    char_end: int
    text: str
    section_path: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class TableDraft:
    title: str
    char_start: int
    char_end: int
    raw_text: str
    unit_context: str = ""
    column_headers: List[str] = field(default_factory=list)
    rows: List[Tuple[str, List[str]]] = field(default_factory=list)
    section_path: List[str] = field(default_factory=list)


DOMAIN_PATTERNS = {
    "insurance": [
        (1, "chapter", re.compile(r"^\s*第\s*[一二三四五六七八九十百零\d]+\s*章\s*(.+)$")),
        (2, "clause", re.compile(r"^\s*(\d+\.\d+(?:\.\d+)*)\s+(.+)$")),
        (2, "section", re.compile(r"^\s*[（(]?[一二三四五六七八九十\d]+[）)]?[、.]\s*(.+)$")),
    ],
    "regulatory": [
        (1, "chapter", re.compile(r"^\s*第\s*[一二三四五六七八九十百零\d]+\s*章\s*(.*)$")),
        (2, "section", re.compile(r"^\s*第\s*[一二三四五六七八九十百零\d]+\s*节\s*(.*)$")),
        (3, "article", re.compile(r"^\s*第\s*[一二三四五六七八九十百零\d]+\s*条\s*(.*)$")),
        (4, "paragraph", re.compile(r"^\s*[（(][一二三四五六七八九十\d]+[）)]\s*(.*)$")),
    ],
    "financial_contracts": [
        (1, "chapter", re.compile(r"^\s*第\s*[一二三四五六七八九十百零\d]+\s*章\s*(.+)$")),
        (2, "section", re.compile(r"^\s*第\s*[一二三四五六七八九十百零\d]+\s*节\s*(.+)$")),
        (3, "article", re.compile(r"^\s*第\s*[一二三四五六七八九十百零\d]+\s*条\s*(.+)$")),
        (3, "subsection", re.compile(r"^\s*[一二三四五六七八九十百]+、\s*(.+)$")),
    ],
    "financial_reports": [
        (1, "section", re.compile(r"^\s*第\s*[一二三四五六七八九十百零\d]+\s*节\s*(.+)$")),
        (1, "markdown", re.compile(r"^\s*#\s+(.+)$")),
        (2, "markdown", re.compile(r"^\s*##\s+(.+)$")),
        (3, "markdown", re.compile(r"^\s*###\s+(.+)$")),
        (2, "subsection", re.compile(r"^\s*[一二三四五六七八九十百]+、\s*(.+)$")),
        (3, "subsection", re.compile(r"^\s*（[一二三四五六七八九十百]+）\s*(.+)$")),
    ],
    "research": [
        (1, "markdown", re.compile(r"^\s*#\s+(.+)$")),
        (2, "markdown", re.compile(r"^\s*##\s+(.+)$")),
        (3, "markdown", re.compile(r"^\s*###\s+(.+)$")),
        (2, "section", re.compile(r"^\s*[一二三四五六七八九十百]+、\s*(.+)$")),
        (3, "section", re.compile(r"^\s*\d+(?:\.\d+)+\s+(.+)$")),
    ],
}


def _line_spans(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    for match in re.finditer(r".*(?:\n|$)", text):
        if match.start() == match.end():
            continue
        spans.append((match.start(), match.end(), match.group(0).rstrip("\n")))
    return spans


def _heading(line: str, domain: str):
    for level, kind, pattern in DOMAIN_PATTERNS.get(domain, []):
        match = pattern.match(line)
        if match:
            groups = [g for g in match.groups() if g]
            title = (groups[-1] if groups else line).strip() or line.strip()
            return level, kind, title[:180]
    return None


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for match in re.finditer(r"\S(?:.|\n)*?\S(?=\n\s*\n|\Z)", text):
        spans.append((match.start(), match.end()))
    if not spans and text.strip():
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        spans.append((start, end))
    return spans


def _safe_cut(body: str, limit: int, minimum: int) -> int:
    low = max(minimum, int(limit * 0.55))
    for token in ("\n", "。", "；", ";"):
        pos = body.rfind(token, low, limit)
        if pos >= low:
            return pos + 1
    return limit


def split_oversize(segment: ChunkDraft, *, min_chars: int, max_chars: int) -> List[ChunkDraft]:
    if len(segment.text) <= max_chars:
        return [segment]
    out: List[ChunkDraft] = []
    offset = 0
    while len(segment.text) - offset > max_chars:
        remaining = segment.text[offset:]
        cut = _safe_cut(remaining, max_chars, min_chars)
        piece = remaining[:cut]
        start = segment.char_start + offset
        out.append(ChunkDraft(
            title=segment.title,
            level=segment.level,
            chunk_type=segment.chunk_type,
            char_start=start,
            char_end=start + len(piece),
            text=piece,
            section_path=list(segment.section_path),
            metadata={**segment.metadata, "split_part": len(out) + 1},
        ))
        offset += cut
    tail = segment.text[offset:]
    if tail.strip():
        start = segment.char_start + offset
        out.append(ChunkDraft(
            title=segment.title,
            level=segment.level,
            chunk_type=segment.chunk_type,
            char_start=start,
            char_end=start + len(tail),
            text=tail,
            section_path=list(segment.section_path),
            metadata={**segment.metadata, "split_part": len(out) + 1},
        ))
    return out


def merge_short_siblings(segments: Sequence[ChunkDraft], *, min_chars: int, max_chars: int) -> List[ChunkDraft]:
    out: List[ChunkDraft] = []
    index = 0
    while index < len(segments):
        current = segments[index]
        # Normal-size chunks are emitted immediately. This is the key legacy bug fix.
        if len(current.text) >= min_chars:
            out.extend(split_oversize(current, min_chars=min_chars, max_chars=max_chars))
            index += 1
            continue
        merged = current
        next_index = index + 1
        while next_index < len(segments) and len(merged.text) < min_chars:
            nxt = segments[next_index]
            same_parent = merged.section_path[:-1] == nxt.section_path[:-1]
            compatible = same_parent and merged.chunk_type == nxt.chunk_type
            combined_len = nxt.char_end - merged.char_start
            if not compatible or combined_len > max_chars:
                break
            merged = ChunkDraft(
                title=merged.title,
                level=min(merged.level, nxt.level),
                chunk_type=merged.chunk_type,
                char_start=merged.char_start,
                char_end=nxt.char_end,
                text=merged.text + ("\n" if not merged.text.endswith("\n") else "") + nxt.text,
                section_path=list(merged.section_path),
                metadata={**merged.metadata, "merged_short": True},
            )
            next_index += 1
        out.extend(split_oversize(merged, min_chars=min_chars, max_chars=max_chars))
        index = next_index
    return out


class DomainChunker:
    def __init__(self, *, min_chars: int = 180, max_chars: int = 2400):
        if min_chars <= 0 or max_chars <= min_chars:
            raise ValueError("invalid chunk size configuration")
        self.min_chars = min_chars
        self.max_chars = max_chars

    def split(self, text: str, domain: str, doc_id: str) -> List[ChunkDraft]:
        text = str(text or "")
        lines = _line_spans(text)
        headings = []
        for start, end, line in lines:
            parsed = _heading(line, domain)
            if parsed:
                headings.append((start, end, *parsed))

        raw: List[ChunkDraft] = []
        if headings:
            stack: Dict[int, str] = {}
            for idx, (start, line_end, level, kind, title) in enumerate(headings):
                end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
                for depth in list(stack):
                    if depth >= level:
                        stack.pop(depth, None)
                stack[level] = title
                section_path = [stack[d] for d in sorted(stack)]
                body = text[start:end]
                if body.strip():
                    raw.append(ChunkDraft(title, level, kind, start, end, body, section_path))
            # Preserve preface before the first heading.
            if headings[0][0] > 0 and text[:headings[0][0]].strip():
                raw.insert(0, ChunkDraft("文档首页", 0, "preface", 0, headings[0][0], text[:headings[0][0]], ["文档首页"]))
        else:
            for idx, (start, end) in enumerate(_paragraph_spans(text), 1):
                body = text[start:end]
                raw.append(ChunkDraft(f"片段{idx}", 1, "paragraph", start, end, body, [f"片段{idx}"]))

        chunks = merge_short_siblings(raw, min_chars=self.min_chars, max_chars=self.max_chars)
        # Defensive invariant: no oversize chunks and offsets map back to source.
        validated = []
        for chunk in chunks:
            if len(chunk.text) > self.max_chars:
                raise AssertionError(f"oversize chunk after split: {doc_id} {len(chunk.text)}")
            source = text[chunk.char_start:chunk.char_end]
            if source != chunk.text:
                # Merged chunks may include a synthetic newline. Restore exact source span.
                chunk.text = source
            chunk.metadata["source_hash"] = stable_hash(chunk.text)
            validated.append(chunk)
        return validated

    def extract_tables(self, text: str, domain: str, chunks: Sequence[ChunkDraft]) -> List[TableDraft]:
        if domain not in {"financial_reports", "research", "financial_contracts"}:
            return []
        lines = _line_spans(text)
        table_like = []
        for idx, (start, end, line) in enumerate(lines):
            numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", line)
            separated = bool(re.search(r"\t|\s{2,}|\|", line))
            if len(numbers) >= 2 and separated and 4 <= len(line.strip()) <= 500:
                table_like.append(idx)
        if not table_like:
            return []
        groups: List[List[int]] = []
        for idx in table_like:
            if not groups or idx - groups[-1][-1] > 1:
                groups.append([idx])
            else:
                groups[-1].append(idx)
        out: List[TableDraft] = []
        for group in groups:
            if len(group) < 2:
                continue
            first = max(0, group[0] - 2)
            last = min(len(lines) - 1, group[-1] + 1)
            start = lines[first][0]
            end = lines[last][1]
            raw = text[start:end]
            before = "\n".join(item[2] for item in lines[max(0, first - 4):first + 1])
            unit_match = re.search(r"单位[：:]?\s*([^\s，。;；]{1,12})", before + "\n" + raw)
            years = list(dict.fromkeys(re.findall(r"20\d{2}年?|本期|上期|期末|期初", raw[:1000])))
            title = next((item[2].strip() for item in reversed(lines[max(0, first - 4):first]) if item[2].strip()), "表格")
            section = next((chunk.section_path for chunk in chunks if chunk.char_start <= start < chunk.char_end), [])
            rows = []
            for _, _, line in lines[group[0]:group[-1] + 1]:
                cells = [x.strip() for x in re.split(r"\t|\s{2,}|\|", line) if x.strip()]
                if len(cells) >= 2:
                    rows.append((cells[0], cells[1:]))
            out.append(TableDraft(
                title=title[:180], char_start=start, char_end=end, raw_text=raw,
                unit_context=unit_match.group(1) if unit_match else "",
                column_headers=years, rows=rows, section_path=list(section),
            ))
        return out
