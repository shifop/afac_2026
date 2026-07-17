#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

_PUNCT = set('，。！？；：、“”"\'（）《》【】()[]{},.?!;:·—…-\\/|')
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9.%＋+\-]+")


def normalize_text(text: str) -> str:
    return re.sub(r"[\s\u3000，。；：、,.;:()（）\[\]【】《》‘’“”\"']+", "", str(text or "")).lower()


def stable_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()


def tokenize(text: str, vocabulary: Sequence[str] = ()) -> List[str]:
    """Dictionary-aware tokens plus Chinese bigrams; no embedding dependency."""
    text = str(text or "")
    vocab = sorted({str(v) for v in vocabulary if v and len(str(v)) >= 2}, key=len, reverse=True)
    tokens: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or ch in _PUNCT:
            i += 1
            continue
        # Match against the original string at position i.  Using text[i:]
        # here copies the remaining suffix on every character and becomes
        # effectively quadratic for long financial documents.
        m = _LATIN_TOKEN_RE.match(text, i)
        if m:
            tok = m.group(0).lower()
            tokens.append(tok)
            i = m.end()
            continue
        if "\u4e00" <= ch <= "\u9fff":
            matched = next((word for word in vocab if text.startswith(word, i)), None)
            if matched:
                tokens.append(matched)
                i += len(matched)
                continue
            tokens.append(ch)
            if i + 1 < len(text) and "\u4e00" <= text[i + 1] <= "\u9fff":
                tokens.append(text[i:i + 2])
            i += 1
            continue
        i += 1
    return tokens


def extract_years(text: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"(?<!\d)(20\d{2})(?:\s*年|年度)?", str(text or ""))))


def extract_number_strings(text: str) -> List[str]:
    pattern = r"-?\d[\d,]*(?:\.\d+)?\s*(?:万亿元|千亿元|亿元|万元|千元|元/股|元每股|元|%|％|倍|个?工作日|个月|日|天|年)?"
    return list(dict.fromkeys(m.group(0).strip() for m in re.finditer(pattern, str(text or ""))))


def canonical_title(title: str) -> str:
    value = normalize_text(title)
    value = re.sub(r"^(?:附件\d*|附件|附录)", "", value)
    value = re.sub(r"(?:全文|正式稿|修订稿|征求意见稿|附件)$", "", value)
    return value


def rrf_score(ranks: Iterable[int], k: int = 60) -> float:
    return sum(1.0 / (k + rank) for rank in ranks if rank > 0)


def bm25_score(query_tokens: Sequence[str], doc_tokens: Sequence[str], doc_freqs: Dict[str, int],
               avgdl: float, n_docs: int, *, k1: float = 1.5, b: float = 0.75) -> float:
    if not query_tokens or not doc_tokens or n_docs <= 0:
        return 0.0
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for token in query_tokens:
        freq = tf.get(token, 0)
        if not freq:
            continue
        df = doc_freqs.get(token, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        denom = freq + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
        score += idf * freq * (k1 + 1.0) / denom
    return score


def match_positions(text: str, terms: Sequence[str], *, per_term_limit: int = 12) -> List[Tuple[int, str]]:
    positions: List[Tuple[int, str]] = []
    for term in sorted({str(t) for t in terms if t and len(str(t)) >= 2}, key=len, reverse=True):
        start = 0
        count = 0
        while count < per_term_limit:
            idx = text.find(term, start)
            if idx < 0:
                break
            positions.append((idx, term))
            start = idx + max(1, len(term))
            count += 1
    return sorted(positions)


def centered_window(text: str, position: int, *, before: int = 500, after: int = 1000) -> Tuple[int, int, str]:
    start = max(0, position - before)
    end = min(len(text), position + after)
    # Prefer sentence or newline boundaries.
    left = max(text.rfind("\n", start, position), text.rfind("。", start, position))
    if left >= start:
        start = left + 1
    candidates = [p for p in (text.find("\n", position + 240, end), text.find("。", position + 240, end)) if p >= 0]
    if candidates:
        end = min(candidates) + 1
    if end - start < 320:
        end = min(len(text), start + max(320, after))
    return start, end, text[start:end]

_REGULATORY_TITLE_SUFFIXES = (
    "法", "办法", "规定", "规则", "准则", "指引", "条例", "细则",
    "决定", "决定书", "通知", "公告", "内容与格式", "实施意见",
)


def clean_ocr_inline(text: str) -> str:
    """Collapse OCR/PDF line breaks without destroying Chinese titles."""
    value = str(text or "").replace("\ufeff", "")
    value = re.sub(r"---\s*Page\s*\d+\s*---", " ", value, flags=re.I)
    value = re.sub(r"[\s|]+", "", value)
    return value.strip("：:，,。；;|-")


def _regulatory_aliases(title: str) -> List[str]:
    title = clean_ocr_inline(title)
    if not title:
        return []
    aliases = [title]
    # Underlying rule name in amendment/explanatory titles.
    aliases.extend(clean_ocr_inline(x) for x in re.findall(r"《([^》]{3,100})》", title))
    replacements = [
        ("上市公司信息披露管理办法", ["信息披露管理办法", "信披办法"]),
        ("上市公司治理准则", ["治理准则", "上市公司治理规范"]),
        ("上市公司章程指引", ["章程指引"]),
        ("证券公司分类监管规定", ["证券公司分类评价规定", "分类监管规定", "分类评价规定"]),
        ("证券公司分类评价规定", ["证券公司分类监管规定", "分类监管规定", "分类评价规定"]),
        ("半年度报告的内容与格式", ["半年度报告内容与格式准则", "半年度报告编制准则", "半年度报告"]),
        ("年度报告的内容与格式", ["年度报告内容与格式准则", "年度报告编制准则", "年度报告"]),
        ("中华人民共和国反洗钱法", ["反洗钱法"]),
        ("中华人民共和国证券法", ["证券法"]),
        ("中华人民共和国公司法", ["公司法"]),
    ]
    for needle, values in replacements:
        if needle == "年度报告的内容与格式" and "半年度报告的内容与格式" in title:
            continue
        if needle in title:
            aliases.extend(values)
    if title.endswith("管理办法"):
        aliases.append(title.replace("管理办法", "办法"))
    if title.startswith("上市公司") and len(title) > 6:
        aliases.append(title.replace("上市公司", "", 1))
    return list(dict.fromkeys(x for x in aliases if len(x) >= 2))


def extract_regulatory_identity(text: str, *, doc_id: str = "", fallback_title: str = "") -> Dict[str, object]:
    """Derive the actual regulatory document identity from its leading page.

    Legacy metadata frequently selected the first cited law (for example
    ``中华人民共和国公司法``) instead of the document's own title.  This
    routine prioritizes the first-page heading and gives enforcement decisions
    a unique family, which is essential in a dense regulatory corpus.
    """
    raw = str(text or "")
    head = raw[:12_000]
    compact = clean_ocr_inline(head[:4_000])
    role = "normative_text"
    title = ""
    aliases: List[str] = []
    family_title = ""

    # Strict-v3 filenames carry a reliable title or an order with the rule name
    # in parentheses.  Preserve both as aliases and use the inner rule as family.
    strict_match = re.match(r"strict_v3_\d+_(.+)", str(doc_id or ""))
    if strict_match:
        stem = clean_ocr_inline(strict_match.group(1))
        aliases.append(stem)
        inner = [clean_ocr_inline(x) for x in re.findall(r"[（(]([^）)]{4,120})[）)]", stem)]
        inner = [x for x in inner if any(suffix in x for suffix in _REGULATORY_TITLE_SUFFIXES)]
        title = inner[-1] if inner else stem
        family_title = title

    # Enforcement decisions generally have no visible title in the extracted
    # text.  Use decision type, document number and first party to avoid grouping
    # hundreds of decisions under the cited Securities Law.
    has_party = bool(re.search(r"当事人\s*[：:]", head[:3_000]))
    has_investigation = "立案调查" in head[:8_000] or "调查、审理" in head[:8_000]
    if has_party and has_investigation:
        role = "enforcement_decision"
        decision_type = "市场禁入决定书" if "市场禁入" in head[:5_000] else "行政处罚决定书"
        number_match = re.search(r"〔\s*(20\d{2})\s*〕\s*(\d+)\s*号", head[:500])
        number = f"〔{number_match.group(1)}〕{number_match.group(2)}号" if number_match else ""
        party_match = re.search(r"当事人\s*[：:]\s*([^，,。；;\n|]{2,80})", head[:2_000])
        party = clean_ocr_inline(party_match.group(1)) if party_match else ""
        # Remove explanatory parenthesis from the compact display name.
        party_short = re.split(r"[（(]", party, maxsplit=1)[0][:32]
        title = f"{decision_type}{number}" + (f"（{party_short}）" if party_short else "")
        family_title = f"{decision_type}{number or doc_id}"
        aliases.extend([decision_type, number, party, party_short])
        if "会计师事务所" in head[:4_000] or "签字注册会计师" in head[:4_000]:
            aliases.extend(["审计机构行政处罚", "会计师事务所行政处罚", "审计未勤勉尽责"])
        if "信息披露违法" in head[:5_000]:
            aliases.extend(["信息披露违法行政处罚", "信息披露违法违规"])
        if "处罚时效" in raw or "连续、继续状态" in raw or "连续继续状态" in raw:
            aliases.extend(["行政处罚时效", "处罚时效", "连续继续状态"])
        if decision_type == "市场禁入决定书":
            aliases.extend(["市场禁入", "市场禁入案例"])

    # Explicit format-standard heading, often split over two PDF lines.
    if role != "enforcement_decision":
        format_match = re.search(
            r"公开发行证券的公司信息披露内容与格式准则\s*第\s*(\d+)\s*号[—－\-–]*\s*([^第]{2,45}?内容与格式)",
            compact,
        )
        if format_match:
            subject = clean_ocr_inline(format_match.group(2))
            title = f"公开发行证券的公司信息披露内容与格式准则第{format_match.group(1)}号——{subject}"
            family_title = title

    # First-page visible heading.  Only inspect text before the first article so
    # cited laws inside Article 1 cannot replace the document's own title.
    if role != "enforcement_decision" and not title:
        boundary_positions = [p for p in (head.find("第一章"), head.find("第一条")) if p >= 0]
        prefix = head[:min(boundary_positions)] if boundary_positions else head[:2_000]
        lines = []
        for raw_line in prefix.splitlines():
            line = clean_ocr_inline(raw_line)
            if not line or re.fullmatch(r"\d+", line) or line.startswith("---Page"):
                continue
            lines.append(line)
        # Revision descriptions are supporting material, not the normative text.
        revision = next((line for line in lines[:8] if "修订说明" in line), "")
        if revision:
            role = "revision_explanation"
            title = revision
            quoted = re.findall(r"《([^》]{4,120})》", revision)
            family_title = clean_ocr_inline(quoted[0]) if quoted else revision.replace("修订说明", "")
        else:
            candidates = []
            for i, line in enumerate(lines[:8]):
                joined = line
                if i + 1 < len(lines):
                    joined2 = line + lines[i + 1]
                    if "内容与格式" in joined2:
                        joined = joined2
                if any(suffix in joined for suffix in _REGULATORY_TITLE_SUFFIXES):
                    candidates.append(joined)
            if candidates:
                title = candidates[0]
                family_title = title

    fallback = clean_ocr_inline(fallback_title)
    if not title:
        title = fallback or str(doc_id or "")
    if not family_title:
        family_title = title
    aliases.extend(_regulatory_aliases(title))
    aliases.extend(_regulatory_aliases(family_title))
    aliases = list(dict.fromkeys(x for x in aliases if x and len(x) >= 2))
    return {
        "title": title,
        "aliases": aliases,
        "family_title": family_title,
        "role": role,
    }
