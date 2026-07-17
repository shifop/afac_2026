#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build AFAC IndexV2 from the complete text corpus.

No V1/V2 annotation scorer is used.  Valuable legacy assets retained here are:
domain-aware file identity, dictionary/BM25 retrieval, entity aliases, numeric
anchors, clause keywords, contract front-page fields and insurance identity.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from chunker_v2 import DomainChunker
from field_ontology import CLAUSE_TERMS, EXACT_FIELD_ALIASES, RETRIEVAL_FIELD_ALIASES
from index_models import (ChunkV2, DocMetaV2, EntityEntryV2, IndexManifest,
                          IndexV2, TableAtom, ValueRecordV2)
from text_utils import (canonical_title, extract_number_strings, extract_regulatory_identity,
                        extract_years, stable_hash, tokenize)


FILE_RULES = [
    (re.compile(r"^insurance_(\d+)\.txt$"), "insurance", lambda m: m.group(1), "insurance_clause"),
    (re.compile(r"^financial_contracts_(text\d+)\.txt$"), "financial_contracts", lambda m: m.group(1), "bond"),
    (re.compile(r"^financial_reports_(annual_.+_\d{4}_report)\.txt$"), "financial_reports", lambda m: m.group(1), "annual"),
    (re.compile(r"^research_(pack2_text\d+)\.txt$"), "research", lambda m: m.group(1), "research_report"),
    (re.compile(r"^(csrc_[\w_]+)\.txt$"), "regulatory", lambda m: m.group(1), "csrc"),
    (re.compile(r"^(strict_v3_\d+_.+)\.txt$"), "regulatory", lambda m: m.group(1), "strict_v3"),
    (re.compile(r"^(regulatory_mineru_[a-f0-9]+)\.txt$"), "regulatory", lambda m: m.group(1), "mineru_law"),
]

CONTRACT_FIELDS = [
    "发行人", "注册金额", "本期发行金额", "发行金额", "担保情况", "主体评级", "债项评级",
    "评级展望", "主承销商", "牵头主承销商", "联席主承销商", "簿记管理人",
    "债券受托管理人", "受托管理人", "评级机构", "股票简称", "股票代码", "证券代码",
]

VALUE_RE = re.compile(
    r"-?\d[\d,]*(?:\.\d+)?\s*(万亿元|千亿元|亿元|万元|千元|元/股|元每股|元|%|％|倍|个?工作日|个月|日|天|年)"
)
RATING_RE = re.compile(r"(?<![A-Za-z])A{1,3}[+-]?(?![A-Za-z+])", re.I)


def infer_meta(path: str) -> Optional[DocMetaV2]:
    filename = os.path.basename(path)
    matched = None
    for pattern, domain, doc_id_fn, subtype in FILE_RULES:
        match = pattern.match(filename)
        if match:
            matched = (domain, doc_id_fn(match), subtype)
            break
    if not matched:
        return None
    domain, doc_id, subtype = matched
    with open(path, encoding="utf-8", errors="ignore") as handle:
        head = handle.read(5000)
    title = ""
    if domain in {"insurance", "regulatory"}:
        match = re.search(r"《([^》]{4,80})》", head)
        if match:
            title = match.group(1)
    if domain == "financial_contracts":
        match = re.search(r"([^\n]{4,100}(?:募集说明书|发行文件|债券))", head)
        if match:
            title = match.group(1).strip()
    if not title:
        title = next((line.strip(" #\t") for line in head.splitlines() if 4 <= len(line.strip()) <= 120), doc_id)
    aliases = []
    family_title = title
    extraction_variant = "mineru" if "mineru" in filename else "text"
    if domain == "regulatory":
        with open(path, encoding="utf-8", errors="ignore") as source_handle:
            regulatory_text = source_handle.read()
        identity = extract_regulatory_identity(regulatory_text, doc_id=doc_id, fallback_title=title)
        title = str(identity["title"])
        aliases = list(identity["aliases"])
        family_title = str(identity["family_title"])
        extraction_variant = str(identity["role"])
    elif domain == "research":
        family_title = doc_id
    years = extract_years(filename + " " + head[:2000])
    issuer = ""
    match = re.search(r"发行人[：:]\s*([^\n]{2,80})", head)
    if match:
        issuer = match.group(1).strip()[:80]
    content_hash = stable_hash(open(path, encoding="utf-8", errors="ignore").read())
    return DocMetaV2(
        doc_id=doc_id, domain=domain, title=title, filepath=path, doc_subtype=subtype,
        year=int(years[0]) if years else None, issuer=issuer or None,
        key_entities=[issuer] if issuer else [], aliases=aliases, total_chars=os.path.getsize(path),
        source_hash=content_hash, document_family_id=canonical_title(family_title) or doc_id,
        extraction_variant=extraction_variant,
    )


def scan_documents(directories: Iterable[str]) -> List[DocMetaV2]:
    metas: List[DocMetaV2] = []
    seen = set()
    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(directory, filename)
            meta = infer_meta(path)
            if meta and meta.doc_id not in seen:
                metas.append(meta)
                seen.add(meta.doc_id)
    return metas


def _normalize_numeric(raw: str, unit: str) -> Tuple[Optional[float], str, str]:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", raw)
    if not match:
        return None, unit, "raw"
    value = float(match.group(0).replace(",", ""))
    if unit in {"%", "％"}:
        return value / 100.0, "ratio", "percent"
    if unit == "万元":
        return value * 10_000, "元", "amount"
    if unit == "千元":
        return value * 1_000, "元", "amount"
    if unit == "亿元":
        return value * 100_000_000, "元", "amount"
    if unit == "千亿元":
        return value * 100_000_000_000, "元", "amount"
    if unit == "万亿元":
        return value * 1_000_000_000_000, "元", "amount"
    if unit in {"元/股", "元每股"}:
        return value, "元/股", "per_share"
    if unit in {"日", "天", "工作日", "个工作日"}:
        return value, "日", "term"
    if unit == "个月":
        return value * 30, "日", "term"
    if unit == "年":
        return value * 365, "日", "term"
    if unit == "倍":
        return value, "ratio", "ratio"
    return value, unit, "amount" if unit == "元" else "raw"


def _field_near(text: str, position: int) -> str:
    window = text[max(0, position - 80):position + 30]
    candidates = []
    for standard, aliases in {**RETRIEVAL_FIELD_ALIASES, **EXACT_FIELD_ALIASES}.items():
        for alias in [standard] + aliases:
            idx = window.rfind(alias)
            if idx >= 0:
                candidates.append((idx, len(alias), standard))
    return max(candidates, default=(-1, -1, ""))[2]


def extract_values(chunk: ChunkV2) -> List[ValueRecordV2]:
    records: List[ValueRecordV2] = []
    for idx, match in enumerate(VALUE_RE.finditer(chunk.text), 1):
        unit = match.group(1)
        norm, norm_unit, vtype = _normalize_numeric(match.group(0), unit)
        field = _field_near(chunk.text, match.start())
        context = chunk.text[max(0, match.start() - 100):min(len(chunk.text), match.end() + 100)]
        years = extract_years(context)
        records.append(ValueRecordV2(
            value_id=f"{chunk.chunk_id}#V{idx}", doc_id=chunk.doc_id, chunk_id=chunk.chunk_id,
            metric=field, vtype=vtype, value_norm=norm, unit_norm=norm_unit,
            value_raw=match.group(0), raw_text=context, char_pos=match.start(),
            time_hint=f"{years[0]}年" if years else "",
        ))
    for idx, match in enumerate(RATING_RE.finditer(chunk.text), len(records) + 1):
        context = chunk.text[max(0, match.start() - 80):min(len(chunk.text), match.end() + 80)]
        field = "主体评级" if "主体" in context else "债项评级" if "债项" in context or "债券" in context else "评级"
        records.append(ValueRecordV2(
            value_id=f"{chunk.chunk_id}#V{idx}", doc_id=chunk.doc_id, chunk_id=chunk.chunk_id,
            metric=field, vtype="rating", value_norm=None, unit_norm="rating",
            value_raw=match.group(0).upper(), raw_text=context, char_pos=match.start(),
        ))
    return records


def extract_structured_fields(meta: DocMetaV2, text: str) -> Dict[str, List[Dict]]:
    result: Dict[str, List[Dict]] = defaultdict(list)
    if meta.domain == "financial_contracts":
        head = text[:20_000]
        lines = list(re.finditer(r"[^\n]+", head))
        for line_match in lines:
            line = line_match.group(0).strip()
            for field in CONTRACT_FIELDS:
                pos = line.find(field)
                if pos < 0:
                    continue
                value = re.sub(rf"^.*?{re.escape(field)}[：:\s]*", "", line).strip(" ：:")
                if value and len(value) <= 180:
                    result[field].append({
                        "value": value, "char_start": line_match.start(), "char_end": line_match.end(),
                        "raw_evidence": line,
                    })
    return dict(result)


def build_index(directories: Iterable[str], output_path: str, *, export_dir: str = "",
                min_chars: int = 180, max_chars: int = 2400) -> IndexV2:
    metas = scan_documents(directories)
    if not metas:
        raise RuntimeError("no source documents matched the AFAC filename rules")
    index = IndexV2()
    index.manifest = IndexManifest(created_at=datetime.now().isoformat())
    chunker = DomainChunker(min_chars=min_chars, max_chars=max_chars)
    # Persist query-time BM25 postings in the index.  Online retrieval must not
    # rebuild a full-corpus inverted index on the first B-list question.
    domain_doc_tokens: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    vocabulary = set()
    value_count = 0

    # Seed aliases from config when available.
    config_entities: Dict[str, Tuple[str, List[str]]] = {}
    try:
        import b_config as cfg
        for attr, entity_type in [
            ("A_LIST_COMPANIES", "company"), ("A_LIST_INSURANCE_PRODUCTS", "insurance_product"),
            ("A_LIST_BONDS", "bond"), ("A_LIST_LAWS", "law")]:
            for standard, aliases in getattr(cfg, attr, {}).items():
                config_entities[standard] = (entity_type, list(dict.fromkeys([standard] + list(aliases))))
    except Exception:
        pass

    for meta in metas:
        with open(meta.filepath, encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
        index.meta_table[meta.doc_id] = meta
        index.manifest.source_hashes[meta.doc_id] = meta.source_hash
        drafts = chunker.split(text, meta.domain, meta.doc_id)
        chunks: List[ChunkV2] = []
        for seq, draft in enumerate(drafts, 1):
            chunk_id = f"{meta.doc_id}#C{seq}"
            values = extract_number_strings(draft.text)
            keywords = []
            for standard, aliases in {**RETRIEVAL_FIELD_ALIASES, **EXACT_FIELD_ALIASES}.items():
                if any(alias in draft.text for alias in [standard] + aliases):
                    keywords.append(standard)
            chunk = ChunkV2(
                chunk_id=chunk_id, doc_id=meta.doc_id, chunk_type=draft.chunk_type,
                title=draft.title, level=draft.level, char_start=draft.char_start,
                char_end=draft.char_end, text=draft.text, section_path=draft.section_path,
                values=values[:50], keywords=keywords, metric_hint=",".join(keywords[:12]),
                source_hash=stable_hash(draft.text), metadata=draft.metadata,
            )
            chunks.append(chunk)
            index.chunk_by_id[chunk_id] = chunk
            records = extract_values(chunk)
            value_count += len(records)
            index.value_index.extend(records)
            for category, terms in CLAUSE_TERMS.items():
                for term in terms:
                    if term in chunk.text:
                        index.clause_kw_index.setdefault(term, []).append((meta.doc_id, chunk_id, 1.0))
        index.chunk_index[meta.doc_id] = chunks

        tables = chunker.extract_tables(text, meta.domain, drafts)
        for table_seq, table in enumerate(tables, 1):
            table_id = f"{meta.doc_id}#T{table_seq}"
            for row_header, cells in table.rows:
                index.tables[f"{table_id}:{len(index.tables)}"] = TableAtom(
                    table_id=table_id, doc_id=meta.doc_id, section_path=table.section_path,
                    table_title=table.title, unit_context=table.unit_context,
                    row_header=row_header, column_headers=table.column_headers,
                    cells={str(i): value for i, value in enumerate(cells)},
                    char_start=table.char_start, char_end=table.char_end, raw_text=table.raw_text,
                )
        index.structured_fields[meta.doc_id] = extract_structured_fields(meta, text)

        identity = {"title": meta.title, "aliases": list(meta.aliases)}
        if meta.issuer:
            identity["issuer"] = meta.issuer
        index.doc_identities[meta.doc_id] = identity

        doc_tokens = tokenize(text, vocabulary)
        domain_doc_tokens[meta.domain].append((meta.doc_id, doc_tokens))

    # Entity lexicon: exact config entities plus document titles/issuers.
    for standard, (etype, aliases) in config_entities.items():
        doc_ids = []
        chunk_ids = []
        for did, chunks in index.chunk_index.items():
            if any(alias and (alias in index.meta_table[did].title or any(alias in c.text for c in chunks[:6])) for alias in aliases):
                doc_ids.append(did)
                chunk_ids.extend(c.chunk_id for c in chunks if any(alias in c.text for alias in aliases))
        if doc_ids:
            domain = index.meta_table[doc_ids[0]].domain
            index.entity_lexicon[standard] = EntityEntryV2(standard, etype, domain, doc_ids, aliases, chunk_ids[:500])
    for did, meta in index.meta_table.items():
        for entity, etype in [(meta.issuer, "issuer"), (meta.title, "document")]:
            if entity and entity not in index.entity_lexicon:
                index.entity_lexicon[entity] = EntityEntryV2(entity, etype, meta.domain, [did], [entity], [])

    vocabulary.update(index.entity_lexicon.keys())
    for domain, rows in domain_doc_tokens.items():
        df = Counter()
        lengths = []
        doc_ids = []
        postings = defaultdict(list)
        for did, tokens in rows:
            # Query-time routing uses the same stable character-bigram / Latin
            # token stream.  Single Chinese characters are intentionally
            # omitted to reduce noise and index size.
            counts = Counter(token for token in tokens if len(token) >= 2)
            doc_ids.append(did)
            lengths.append(max(1, sum(counts.values())))
            df.update(counts.keys())
            for token, frequency in counts.items():
                postings[token].append((did, int(frequency)))
        index.bm25[domain] = {
            "doc_ids": doc_ids,
            "doc_freqs": dict(df),
            "doc_lengths": lengths,
            "avgdl": sum(lengths) / max(1, len(lengths)),
            "postings": dict(postings),
            "tokenizer": "char_bigram_v1",
        }

    max_chunk = max((len(chunk.text) for chunk in index.chunk_by_id.values()), default=0)
    if max_chunk > max_chars:
        raise AssertionError(f"chunk invariant failed: max={max_chunk} limit={max_chars}")
    index.manifest.statistics = {
        "documents": len(index.meta_table), "chunks": len(index.chunk_by_id),
        "values": value_count, "tables": len(index.tables), "max_chunk_chars": max_chunk,
        "domains": dict(Counter(meta.domain for meta in index.meta_table.values())),
        "bm25_posting_terms": {domain: len(stats.get("postings", {})) for domain, stats in index.bm25.items()},
        "legacy_v1v2_scorer_retained": False,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Save only after every derived field and validation has completed.
    with output.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if export_dir:
        export = Path(export_dir)
        export.mkdir(parents=True, exist_ok=True)
        (export / "manifest.json").write_text(json.dumps(asdict(index.manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        with (export / "docs.jsonl").open("w", encoding="utf-8") as handle:
            for meta in index.meta_table.values():
                handle.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
        with (export / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in index.chunk_by_id.values():
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        with (export / "tables.jsonl").open("w", encoding="utf-8") as handle:
            for table in index.tables.values():
                handle.write(json.dumps(asdict(table), ensure_ascii=False) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-dir", action="append", default=[])
    parser.add_argument("--output", default="b_index_v2.pkl")
    parser.add_argument("--export-dir", default="index_v2")
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--max-chars", type=int, default=2400)
    args = parser.parse_args()
    directories = list(args.extracted_dir)
    if not directories:
        try:
            import b_config as cfg
            directories = [getattr(cfg, "EXTRACTED_DIR", ""), getattr(cfg, "REGULATORY_V2_DIR", "")]
        except Exception:
            directories = []
    index = build_index(directories, args.output, export_dir=args.export_dir,
                        min_chars=args.min_chars, max_chars=args.max_chars)
    print(json.dumps(index.manifest.statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
