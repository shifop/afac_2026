#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact, quote-safe evidence cards for AFAC Stage B1-Lite.

Cards are assembled only from exact substrings of the source evidence span.
They retain section/table context while removing duplicated long PDF windows.
The compressor never paraphrases evidence and therefore remains compatible
with strict quote validation.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from retrieval_models import EvidenceSpanV2, claim_attr, unique_strings
from text_utils import normalize_text


_BOUNDARY_RE = re.compile(r"(?<=[。！？；\n])")


def _number_raw(number: Any) -> str:
    if isinstance(number, dict):
        return str(number.get("raw", "") or "")
    return str(getattr(number, "raw", "") or "")


def _claim_terms(claim: Any) -> List[str]:
    terms: List[str] = []
    terms.extend(str(x) for x in claim_attr(claim, "subject_candidates", []) or [])
    terms.extend(str(x) for x in claim_attr(claim, "field_candidates", []) or [])
    terms.extend(str(x) for x in claim_attr(claim, "years", []) or [])
    terms.extend(_number_raw(x).replace(" ", "").replace("％", "%") for x in claim_attr(claim, "numbers", []) or [])
    terms.extend(str(x) for x in claim_attr(claim, "keywords", []) or [])
    raw = str(claim_attr(claim, "raw", "") or "")
    for term in [
        "应当", "不得", "至少", "不超过", "超过", "如果", "若", "但", "除外",
        "身故保险金", "退保", "现金价值", "免赔额", "赔付比例", "市场规模",
        "复合增长率", "同比", "资产负债率", "违约", "补偿", "兑付", "保存期限",
    ]:
        if term in raw:
            terms.append(term)
    return unique_strings(term.strip() for term in terms if len(str(term).strip()) >= 2)[:24]


def _split_units(text: str) -> List[Tuple[int, int, str]]:
    rows: List[Tuple[int, int, str]] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        end = match.end()
        value = text[start:end].strip()
        if value:
            rows.append((start, end, value))
        start = end
    if start < len(text):
        value = text[start:].strip()
        if value:
            rows.append((start, len(text), value))
    if not rows and text:
        rows.append((0, len(text), text))
    return rows


def _prefix(span: EvidenceSpanV2) -> str:
    parts: List[str] = []
    section_path = [str(x).strip() for x in span.section_path or [] if str(x).strip()]
    if section_path:
        parts.append("章节：" + " > ".join(section_path[-3:]))
    if span.row_header:
        parts.append("表格行：" + str(span.row_header))
    if span.column_headers:
        parts.append("表头：" + " | ".join(str(x) for x in span.column_headers[:8]))
    if span.unit_context:
        parts.append("单位：" + str(span.unit_context))
    return "\n".join(parts)


class EvidenceCardCompressor:
    def __init__(self, *, max_chars: int = 1200, context_units: int = 1):
        self.max_chars = max(500, int(max_chars))
        self.context_units = max(0, int(context_units))

    def compress(self, span: EvidenceSpanV2, claim: Any = None) -> EvidenceSpanV2:
        text = str(span.text or "")
        before = len(text)
        if before <= self.max_chars:
            metadata = dict(span.metadata)
            metadata.setdefault("evidence_card", False)
            return replace(span, metadata=metadata)

        terms = unique_strings(list(span.matched_terms or []) + _claim_terms(claim))
        units = _split_units(text)
        chosen_indices: List[int] = []
        for index, (_, _, unit) in enumerate(units):
            compact = unit.replace("％", "%").replace(" ", "")
            if any(term.replace("％", "%").replace(" ", "") in compact for term in terms if term):
                for offset in range(-self.context_units, self.context_units + 1):
                    target = index + offset
                    if 0 <= target < len(units) and target not in chosen_indices:
                        chosen_indices.append(target)

        # Fallback to a centered slice around the first matched term.  If no
        # term is found, keep the leading context rather than fabricating a
        # summary.
        if not chosen_indices:
            positions = [text.find(term) for term in terms if term and text.find(term) >= 0]
            if positions:
                center = min(positions)
                start = max(0, center - self.max_chars // 3)
                end = min(len(text), start + self.max_chars)
                body = text[start:end].strip()
            else:
                body = text[: self.max_chars].strip()
        else:
            chosen_indices.sort()
            excerpts: List[str] = []
            used = 0
            previous = None
            for index in chosen_indices:
                value = units[index][2]
                separator = "\n…\n" if previous is not None and index > previous + 1 else ""
                addition = separator + value
                if excerpts and used + len(addition) > self.max_chars:
                    continue
                excerpts.append(addition)
                used += len(addition)
                previous = index
            body = "".join(excerpts).strip()
            if not body:
                body = text[: self.max_chars].strip()

        header = _prefix(span)
        available = self.max_chars - (len(header) + 1 if header else 0)
        if len(body) > available:
            body = body[:available]
            cut = max(body.rfind("。"), body.rfind("\n"), body.rfind("；"))
            if cut >= int(available * 0.65):
                body = body[: cut + 1]
        card_text = f"{header}\n{body}".strip() if header else body

        metadata = dict(span.metadata)
        metadata.update({
            "evidence_card": True,
            "evidence_card_chars_before": before,
            "evidence_card_chars_after": len(card_text),
            "evidence_card_exact_substrings": True,
        })
        return replace(span, text=card_text, metadata=metadata)

    def compress_many(self, spans: Sequence[EvidenceSpanV2], claim: Any = None) -> Tuple[List[EvidenceSpanV2], Dict[str, int]]:
        before_chars = sum(len(str(span.text or "")) for span in spans)
        compressed = [self.compress(span, claim=claim) for span in spans]
        # Collapse exact/near-identical cards originating from overlapping
        # query channels.  Identity is document + normalized card content.
        out: List[EvidenceSpanV2] = []
        seen = set()
        duplicates = 0
        for span in sorted(compressed, key=lambda item: (-float(item.score), item.doc_id, item.char_start or -1)):
            key = (span.doc_id, normalize_text(span.text)[:700])
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            out.append(span)
        after_chars = sum(len(str(span.text or "")) for span in out)
        return out, {
            "chars_before": before_chars,
            "chars_after": after_chars,
            "chars_saved": max(0, before_chars - after_chars),
            "duplicates_removed": duplicates,
            "spans_before": len(spans),
            "spans_after": len(out),
        }


__all__ = ["EvidenceCardCompressor"]
