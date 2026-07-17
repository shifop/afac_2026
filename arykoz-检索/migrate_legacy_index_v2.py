#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast migration of a complete legacy BIndex pickle to IndexV2.

The migration uses the legacy chunk offsets to reconstruct the complete source
text, then applies the corrected V2 chunker and structure extractors.  It keeps
legacy BM25/entity/identity assets where they remain valuable and drops the
retired V1/V2 scorer annotations.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from build_index_v2 import extract_structured_fields, extract_values
from chunker_v2 import DomainChunker
from field_ontology import CLAUSE_TERMS, EXACT_FIELD_ALIASES, RETRIEVAL_FIELD_ALIASES
from index_models import (ChunkV2, DocMetaV2, EntityEntryV2, IndexManifest,
                          IndexV2, TableAtom)
from legacy_index import RuntimeIndex
from text_utils import (canonical_title, extract_number_strings, extract_regulatory_identity,
                        stable_hash, tokenize)


def _meta_v2(doc_id: str, meta, text: str, runtime: RuntimeIndex) -> DocMetaV2:
    domain = str(getattr(meta, "domain", "") or "")
    title = str(getattr(meta, "title", "") or doc_id)
    aliases = list(getattr(meta, "aliases", []) or [])
    family_title = title
    extraction_variant = "legacy_reconstructed"
    if domain == "regulatory":
        identity = extract_regulatory_identity(text, doc_id=doc_id, fallback_title=title)
        title = str(identity["title"])
        aliases = list(dict.fromkeys(list(identity["aliases"])))
        family_title = str(identity["family_title"])
        extraction_variant = str(identity["role"])
    elif domain == "research":
        # Legacy report headers are frequently generic ("证券研究报告",
        # "公司研究") and must not merge unrelated reports into one family.
        family_title = doc_id
    path = str(getattr(meta, "filepath", "") or "")
    return DocMetaV2(
        doc_id=doc_id,
        domain=domain,
        title=title,
        filepath=path,
        doc_subtype=str(getattr(meta, "doc_subtype", "") or ""),
        year=getattr(meta, "year", None),
        issuer=getattr(meta, "issuer", None),
        key_entities=list(getattr(meta, "key_entities", []) or []),
        aliases=aliases,
        total_chars=len(text),
        source_hash=runtime.source_hash(doc_id),
        document_family_id=canonical_title(family_title) or doc_id,
        extraction_variant=extraction_variant,
    )


def migrate(legacy_path: str, output_path: str, *, export_dir: str = "",
            min_chars: int = 180, max_chars: int = 2400):
    runtime = RuntimeIndex.load(legacy_path)
    index = IndexV2(manifest=IndexManifest(created_at=datetime.now().isoformat()))
    chunker = DomainChunker(min_chars=min_chars, max_chars=max_chars)
    ratios: Dict[str, float] = {}
    bm25_doc_ids = defaultdict(list)
    bm25_doc_lengths = defaultdict(list)
    bm25_doc_freqs = defaultdict(Counter)
    bm25_postings = defaultdict(lambda: defaultdict(list))

    for doc_id, old_meta in runtime.meta_table.items():
        doc_id = str(doc_id)
        text = runtime.get_doc_text(doc_id)
        if not text:
            raise RuntimeError(f"empty reconstructed document: {doc_id}")
        expected = int(getattr(old_meta, "total_chars", 0) or 0)
        if expected:
            ratios[doc_id] = round(len(text) / expected, 6)
        meta = _meta_v2(doc_id, old_meta, text, runtime)
        index.meta_table[doc_id] = meta
        index.manifest.source_hashes[doc_id] = meta.source_hash

        # Build persistent BM25 postings while the full text is already in
        # memory.  This prevents a costly first-query rebuild over the complete
        # regulatory corpus.
        counts = Counter(token for token in tokenize(text, ()) if len(token) >= 2)
        bm25_doc_ids[meta.domain].append(doc_id)
        bm25_doc_lengths[meta.domain].append(max(1, sum(counts.values())))
        bm25_doc_freqs[meta.domain].update(counts.keys())
        for token, frequency in counts.items():
            bm25_postings[meta.domain][token].append((doc_id, int(frequency)))

        drafts = chunker.split(text, meta.domain, doc_id)
        chunks: List[ChunkV2] = []
        for seq, draft in enumerate(drafts, 1):
            chunk_id = f"{doc_id}#C{seq}"
            keywords = []
            for standard, aliases in {**RETRIEVAL_FIELD_ALIASES, **EXACT_FIELD_ALIASES}.items():
                if any(alias in draft.text for alias in [standard] + aliases):
                    keywords.append(standard)
            chunk = ChunkV2(
                chunk_id=chunk_id, doc_id=doc_id, chunk_type=draft.chunk_type,
                title=draft.title, level=draft.level, char_start=draft.char_start,
                char_end=draft.char_end, text=draft.text,
                section_path=list(draft.section_path),
                values=extract_number_strings(draft.text)[:50],
                keywords=keywords, metric_hint=",".join(keywords[:12]),
                source_hash=stable_hash(draft.text), metadata=dict(draft.metadata),
            )
            chunks.append(chunk)
            index.chunk_by_id[chunk_id] = chunk
            index.value_index.extend(extract_values(chunk))
            for terms in CLAUSE_TERMS.values():
                for term in terms:
                    if term in chunk.text:
                        index.clause_kw_index.setdefault(term, []).append((doc_id, chunk_id, 1.0))
        index.chunk_index[doc_id] = chunks

        for table_seq, table in enumerate(chunker.extract_tables(text, meta.domain, drafts), 1):
            table_id = f"{doc_id}#T{table_seq}"
            for row_seq, (row_header, cells) in enumerate(table.rows, 1):
                key = f"{table_id}:R{row_seq}"
                index.tables[key] = TableAtom(
                    table_id=table_id, doc_id=doc_id, section_path=list(table.section_path),
                    table_title=table.title, unit_context=table.unit_context,
                    row_header=row_header, column_headers=list(table.column_headers),
                    cells={str(i): value for i, value in enumerate(cells)},
                    char_start=table.char_start, char_end=table.char_end, raw_text=table.raw_text,
                )
        index.structured_fields[doc_id] = extract_structured_fields(meta, text)
        identity = dict(runtime.doc_identities.get(doc_id, {}) or {})
        identity.setdefault("title", meta.title)
        identity.setdefault("aliases", list(meta.aliases))
        if meta.issuer:
            identity.setdefault("issuer", meta.issuer)
        index.doc_identities[doc_id] = identity

    # Preserve the complete legacy entity lexicon without rescanning every text.
    for name, entry in runtime.entity_lexicon.items():
        doc_ids = [str(x) for x in getattr(entry, "doc_ids", []) or [] if str(x) in index.meta_table]
        if not doc_ids:
            continue
        index.entity_lexicon[str(name)] = EntityEntryV2(
            entity=str(getattr(entry, "entity", name) or name),
            entity_type=str(getattr(entry, "entity_type", "entity") or "entity"),
            domain=str(getattr(entry, "domain", "") or index.meta_table[doc_ids[0]].domain),
            doc_ids=doc_ids,
            aliases=list(dict.fromkeys([str(name)] + [str(x) for x in getattr(entry, "aliases", []) or []])),
            chunk_ids=[],
        )
    # Persist query-time BM25 postings generated from the reconstructed full
    # texts.  Legacy df/length statistics are not sufficient for fast online
    # scoring because they do not contain per-document term frequencies.
    for domain, doc_ids in bm25_doc_ids.items():
        lengths = bm25_doc_lengths[domain]
        index.bm25[domain] = {
            "doc_ids": list(doc_ids),
            "doc_freqs": dict(bm25_doc_freqs[domain]),
            "doc_lengths": list(lengths),
            "avgdl": sum(lengths) / max(1, len(lengths)),
            "postings": dict(bm25_postings[domain]),
            "tokenizer": "char_bigram_v1",
        }

    max_chunk = max((len(chunk.text) for chunk in index.chunk_by_id.values()), default=0)
    if max_chunk > max_chars:
        raise AssertionError(f"chunk invariant failed: max={max_chunk} limit={max_chars}")
    index.manifest.statistics = {
        "documents": len(index.meta_table),
        "chunks": len(index.chunk_by_id),
        "values": len(index.value_index),
        "tables": len(index.tables),
        "max_chunk_chars": max_chunk,
        "domains": dict(Counter(meta.domain for meta in index.meta_table.values())),
        "bm25_posting_terms": {domain: len(stats.get("postings", {})) for domain, stats in index.bm25.items()},
        "migration_source": "complete_legacy_index",
        "legacy_v1v2_scorer_retained": False,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "legacy_path": os.path.abspath(legacy_path),
        "output_path": os.path.abspath(output_path),
        "min_length_ratio": min(ratios.values()) if ratios else None,
        "max_length_ratio": max(ratios.values()) if ratios else None,
        "statistics": index.manifest.statistics,
    }
    output.with_suffix(".migration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if export_dir:
        export = Path(export_dir)
        export.mkdir(parents=True, exist_ok=True)
        (export / "manifest.json").write_text(
            json.dumps(asdict(index.manifest), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (export / "migration_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return index, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_index")
    parser.add_argument("--output", default="b_index_v2.pkl")
    parser.add_argument("--export-dir", default="")
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--max-chars", type=int, default=2400)
    args = parser.parse_args()
    _, report = migrate(args.legacy_index, args.output, export_dir=args.export_dir,
                        min_chars=args.min_chars, max_chars=args.max_chars)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
