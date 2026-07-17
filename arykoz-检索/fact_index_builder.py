#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an adjudicable FactIndex directly from IndexV2 assets.

The retired V1/V2 annotation bridge is intentionally not used.  Facts are
created only from source-backed structured fields, table atoms, and value
records carrying a metric and raw evidence.
"""
from __future__ import annotations

import os
import pickle
import re
from typing import Any, Iterable, List, Optional, Tuple

from legacy_index import RuntimeIndex
from lightweight_fact_index import FactIndex, FactRecord


def _entities(runtime: RuntimeIndex, doc_id: str) -> List[str]:
    meta = runtime.meta_table.get(doc_id)
    values: List[str] = []
    if meta is not None:
        values.extend([
            str(getattr(meta, "issuer", "") or ""),
            *[str(x) for x in getattr(meta, "key_entities", []) or []],
        ])
    # The legacy entity entry may have a misleading single domain, but doc_ids
    # still provide a reliable link to the document.
    for standard, entry in runtime.entity_lexicon.items():
        if doc_id in [str(x) for x in getattr(entry, "doc_ids", []) or []]:
            values.append(str(standard))
    values.append(doc_id)
    if meta is not None:
        values.append(str(getattr(meta, "title", "") or ""))
    identity = runtime.doc_identities.get(doc_id, {}) or {}
    for key in ("issuer", "product", "short"):
        value = identity.get(key)
        if isinstance(value, list):
            values.extend(str(x) for x in value)
        elif value:
            values.append(str(value))
    out, seen = [], set()
    for value in values:
        value = value.strip()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out[:4]


def _year(runtime: RuntimeIndex, doc_id: str, text: str = "") -> Optional[str]:
    match = re.search(r"(?<!\d)(20\d{2})(?:\s*年|年度)?", str(text or ""))
    if match:
        return f"{match.group(1)}年"
    meta = runtime.meta_table.get(doc_id)
    year = getattr(meta, "year", None) if meta else None
    return f"{year}年" if year else None


def _number(value: str) -> Tuple[Optional[float], Optional[str]]:
    text = str(value or "").replace(",", "").replace("，", "").replace(" ", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None, None
    try:
        number = float(match.group(0))
    except ValueError:
        return None, None
    units = ["万亿元", "千亿元", "亿元", "万元", "千元", "元/股", "元每股", "元", "%", "％", "倍", "工作日", "个月", "日", "天", "年"]
    unit = next((item for item in units if item in text), None)
    multipliers = {"万亿元": 1e12, "千亿元": 1e11, "亿元": 1e8, "万元": 1e4, "千元": 1e3}
    if unit in multipliers:
        number *= multipliers[unit]
    if unit == "％":
        unit = "%"
    return number, unit


def _add_unique(index: FactIndex, fact: FactRecord, seen: set) -> None:
    key = (
        fact.entity, fact.field, fact.time or "", fact.value_original,
        fact.source_doc, fact.char_start, fact.table_row_header, fact.table_col_header,
    )
    if key in seen:
        return
    seen.add(key)
    index.add(fact)


def build_fact_index_from_retriever(index_path: str, *, cache_path: str = "",
                                    force_rebuild: bool = False) -> FactIndex:
    cache_path = cache_path or f"{index_path}.fact_index.pkl"
    if not force_rebuild and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, FactIndex):
                return cached
        except Exception:
            pass

    runtime = RuntimeIndex.load(index_path)
    fact_index = FactIndex()
    seen = set()
    entity_map = {str(doc_id): _entities(runtime, str(doc_id)) for doc_id in runtime.meta_table}

    # 1. Structured key-value fields: highest-quality deterministic facts.
    for doc_id, field_map in runtime.structured_fields.items():
        for field, rows in (field_map or {}).items():
            for row in rows or []:
                raw = str(row.get("raw_evidence", "") or "")
                value = str(row.get("value", "") or "")
                if not raw or not value:
                    continue
                normalized, unit = _number(value)
                for entity in entity_map.get(str(doc_id), [str(doc_id)]):
                    _add_unique(fact_index, FactRecord(
                        entity=entity, field=str(field), value_original=value,
                        value_normalized=normalized, unit=unit,
                        time=_year(runtime, str(doc_id), raw), source_doc=str(doc_id),
                        char_start=row.get("char_start"), char_end=row.get("char_end"),
                        tool="index_v2_structured", confidence="HIGH",
                        raw_evidence=raw, modality="fact", fact_type="atomic_fact",
                        resolve_status="ADJUDICABLE",
                    ), seen)

    # 2. Table atoms: bind row header, column header, unit, and cell value.
    for table in runtime.tables.values():
        doc_id = str(getattr(table, "doc_id", "") or "")
        row_header = str(getattr(table, "row_header", "") or "").strip()
        raw = str(getattr(table, "raw_text", "") or "")
        headers = [str(x) for x in getattr(table, "column_headers", []) or []]
        cells = dict(getattr(table, "cells", {}) or {})
        if not doc_id or not row_header or not raw:
            continue
        for key, cell in cells.items():
            value = str(cell or "").strip()
            if not value:
                continue
            try:
                idx = int(key)
            except (TypeError, ValueError):
                idx = -1
            header = headers[idx] if 0 <= idx < len(headers) else ""
            normalized, unit = _number(value)
            unit_context = str(getattr(table, "unit_context", "") or "")
            if not unit and unit_context:
                _, unit = _number("1" + unit_context)
            time = _year(runtime, doc_id, header + " " + raw)
            for entity in entity_map.get(doc_id, [doc_id]):
                _add_unique(fact_index, FactRecord(
                    entity=entity, field=row_header, value_original=value,
                    value_normalized=normalized, unit=unit, time=time,
                    source_doc=doc_id,
                    char_start=getattr(table, "char_start", None),
                    char_end=getattr(table, "char_end", None),
                    tool="index_v2_table", confidence="HIGH",
                    raw_evidence=raw[:4000], table_row_header=row_header,
                    table_col_header=header, modality="fact", fact_type="atomic_fact",
                    resolve_status="ADJUDICABLE" if header or time else "TABLE_AMBIGUOUS",
                ), seen)

    # 3. Metric-bound value records.  Records without a field or source text are
    # retrieval hints only and cannot adjudicate.
    for value in runtime.value_index:
        doc_id = str(getattr(value, "doc_id", "") or "")
        field = str(getattr(value, "metric", "") or "").strip()
        original = str(getattr(value, "value_raw", "") or "").strip()
        raw = str(getattr(value, "raw_text", "") or "").strip()
        if not doc_id or not field or not original:
            continue
        normalized = getattr(value, "value_norm", None)
        unit = str(getattr(value, "unit_norm", "") or "") or None
        if normalized is None:
            normalized, detected = _number(original)
            unit = unit or detected
        char_start = getattr(value, "char_pos", None)
        char_end = (int(char_start) + len(raw)) if char_start is not None and raw else None
        status = "ADJUDICABLE" if raw and char_start is not None else "RETRIEVAL_HINT_ONLY"
        for entity in entity_map.get(doc_id, [doc_id]):
            _add_unique(fact_index, FactRecord(
                entity=entity, field=field, value_original=original,
                value_normalized=normalized, unit=unit,
                time=str(getattr(value, "time_hint", "") or "") or _year(runtime, doc_id, raw),
                source_doc=doc_id, char_start=char_start, char_end=char_end,
                tool="index_v2_value", confidence="HIGH" if status == "ADJUDICABLE" else "MEDIUM",
                raw_evidence=raw or None,
                table_row_header=str(getattr(value, "row_header", "") or "") or None,
                table_col_header=str(getattr(value, "col_header", "") or "") or None,
                modality="fact", fact_type="atomic_fact", resolve_status=status,
            ), seen)

    if cache_path:
        try:
            with open(cache_path, "wb") as handle:
                pickle.dump(fact_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass
    return fact_index


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path")
    parser.add_argument("--output", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_fact_index_from_retriever(
        args.index_path, cache_path=args.output, force_rebuild=args.force,
    )
    print(json.dumps(result.stats(), ensure_ascii=False, indent=2))
