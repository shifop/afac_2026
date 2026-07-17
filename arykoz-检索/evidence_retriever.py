#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-claim evidence retrieval and domain-aware evidence expansion."""
from __future__ import annotations

import re
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from field_ontology import CLAUSE_TERMS, aliases_for_field
from option_query_planner import OptionQueryPlanner, QueryVariant
from retrieval_models import EvidenceSpanV2, claim_attr, unique_strings
from text_utils import centered_window, match_positions, normalize_text


class ClaimEvidenceRetriever:
    def __init__(self, runtime_index):
        self.index = runtime_index
        self.query_planner = OptionQueryPlanner()

    def _identity_terms(self, doc_id: str) -> List[str]:
        terms: List[str] = []
        meta = self.index.meta_table.get(str(doc_id))
        if meta is not None:
            terms.extend([str(getattr(meta, "title", "") or ""), str(getattr(meta, "issuer", "") or "")])
            terms.extend(str(x) for x in getattr(meta, "aliases", []) or [])
            terms.extend(str(x) for x in getattr(meta, "key_entities", []) or [])
        identity = self.index.doc_identities.get(str(doc_id), {})
        for value in identity.values():
            if isinstance(value, list):
                terms.extend(str(x) for x in value)
            elif value:
                terms.append(str(value))
        return unique_strings(terms)

    def _number_raw(self, number: Any) -> str:
        if isinstance(number, dict):
            return str(number.get("raw", "") or "")
        return str(getattr(number, "raw", "") or "")

    def claim_terms(self, claim: Any, extra_terms: Sequence[str] = ()) -> List[str]:
        terms = []
        terms.extend(claim_attr(claim, "subject_candidates", []) or [])
        for field in claim_attr(claim, "field_candidates", []) or []:
            terms.extend(aliases_for_field(str(field), adjudication=False))
        terms.extend(claim_attr(claim, "years", []) or [])
        terms.extend(self._number_raw(number) for number in claim_attr(claim, "numbers", []) or [])
        terms.extend(claim_attr(claim, "keywords", []) or [])
        terms.extend(extra_terms)
        raw = str(claim_attr(claim, "raw", "") or "")
        for category_terms in CLAUSE_TERMS.values():
            terms.extend(term for term in category_terms if term in raw)
        return unique_strings(sorted((term for term in terms if len(str(term)) >= 2), key=len, reverse=True))[:40]

    def _score(self, text: str, claim: Any, terms: Sequence[str], *, base: float = 0.0) -> Tuple[float, List[str]]:
        matched = [term for term in terms if term in text]
        score = base + sum(min(6.0, 1.0 + len(term) / 4.0) for term in matched)
        subjects = claim_attr(claim, "subject_candidates", []) or []
        fields = claim_attr(claim, "field_candidates", []) or []
        years = claim_attr(claim, "years", []) or []
        if subjects and any(subject in text for subject in subjects):
            score += 6.0
        if fields and any(alias in text for field in fields for alias in aliases_for_field(field, adjudication=False)):
            score += 6.0
        if years and any(year in text for year in years):
            score += 4.0
        numbers = [self._number_raw(number).replace(" ", "") for number in claim_attr(claim, "numbers", []) or []]
        normalized = text.replace(" ", "")
        if numbers and any(number and number in normalized for number in numbers):
            score += 15.0
        option_norm = normalize_text(str(claim_attr(claim, "raw", "") or ""))
        if len(option_norm) >= 8 and option_norm in normalize_text(text):
            score += 80.0
        return score, matched

    def _structured_spans(self, claim: Any, doc_id: str, terms: Sequence[str]) -> List[EvidenceSpanV2]:
        spans: List[EvidenceSpanV2] = []
        fields = claim_attr(claim, "field_candidates", []) or []
        structured = self.index.structured_fields.get(doc_id, {})
        for field in fields:
            aliases = aliases_for_field(field, adjudication=False)
            for key, rows in structured.items():
                if not any(alias in key or key in alias for alias in aliases):
                    continue
                for idx, row in enumerate(rows):
                    text = str(row.get("raw_evidence", "") or row.get("value", ""))
                    score, matched = self._score(text, claim, terms, base=30.0)
                    spans.append(EvidenceSpanV2(
                        evidence_id=f"structured:{doc_id}:{key}:{idx}", doc_id=doc_id,
                        text=text, score=score, source="structured_field",
                        char_start=row.get("char_start"), char_end=row.get("char_end"),
                        matched_terms=matched, source_hash=self.index.source_hash(doc_id),
                        metadata={"field": key, "value": row.get("value", "")},
                    ))
        return spans

    def _table_spans(self, claim: Any, doc_id: str, terms: Sequence[str], *, limit: int = 24) -> List[EvidenceSpanV2]:
        candidates = []
        fields = claim_attr(claim, "field_candidates", []) or []
        years = claim_attr(claim, "years", []) or []
        for key, table in self.index.tables_by_doc.get(doc_id, []):
            row = str(getattr(table, "row_header", "") or "")
            raw = str(getattr(table, "raw_text", "") or "")
            if fields and not any(alias in row or alias in raw for field in fields for alias in aliases_for_field(field, adjudication=False)):
                continue
            if years and not any(year in raw or year in " ".join(getattr(table, "column_headers", []) or []) for year in years):
                continue
            score, matched = self._score(raw, claim, terms, base=25.0)
            candidates.append((score, str(key), table, matched))
        candidates.sort(key=lambda row: (-row[0], row[1]))
        spans: List[EvidenceSpanV2] = []
        for score, key, table, matched in candidates[:limit]:
            row = str(getattr(table, "row_header", "") or "")
            raw = str(getattr(table, "raw_text", "") or "")
            spans.append(EvidenceSpanV2(
                evidence_id=f"table:{key}", doc_id=doc_id, text=raw[:5000], score=score,
                source="table_atom", chunk_id=None,
                char_start=int(getattr(table, "char_start", 0)), char_end=int(getattr(table, "char_end", 0)),
                matched_terms=matched, section_path=list(getattr(table, "section_path", []) or []),
                table_id=str(getattr(table, "table_id", key)), row_header=row,
                column_headers=list(getattr(table, "column_headers", []) or []),
                unit_context=str(getattr(table, "unit_context", "") or ""),
                source_hash=self.index.source_hash(doc_id),
            ))
        return spans

    def _expand(self, doc_id: str, start: int, end: int, domain: str) -> Tuple[int, int, str]:
        full = self.index.get_doc_text(doc_id)
        if not full:
            return start, end, ""
        before, after = 450, 900
        if domain == "financial_reports":
            before, after = 900, 1600
        elif domain in {"regulatory", "financial_contracts", "insurance"}:
            before, after = 650, 1300
        position = max(start, min(len(full) - 1, start + max(0, end - start) // 2)) if full else 0
        return centered_window(full, position, before=before, after=after)

    def _exact_option_span(self, claim: Any, doc_id: str, domain: str, terms: Sequence[str]) -> Optional[EvidenceSpanV2]:
        raw = str(claim_attr(claim, "raw", "") or "").strip()
        if len(raw) < 8:
            return None
        full = self.index.get_doc_text(doc_id)
        if not full:
            return None
        # Ignore whitespace differences introduced by PDF extraction. Keep the
        # pattern bounded to avoid an expensive fuzzy search.
        pieces = [re.escape(piece) for piece in re.split(r"\s+", raw) if piece]
        if not pieces:
            return None
        pattern = re.compile(r"\s*".join(pieces))
        match = pattern.search(full)
        if not match:
            compact = raw.replace(" ", "")
            position = full.find(compact)
            if position < 0:
                return None
            start0, end0 = position, position + len(compact)
        else:
            start0, end0 = match.start(), match.end()
        start, end, text = self._expand(doc_id, start0, end0, domain)
        score, matched = self._score(text, claim, terms, base=100.0)
        return EvidenceSpanV2(
            evidence_id=f"exact:{doc_id}:{claim_attr(claim, 'claim_id', '')}", doc_id=doc_id,
            text=text, score=score, source="normalized_exact_claim",
            char_start=start, char_end=end, matched_terms=matched,
            source_hash=self.index.source_hash(doc_id),
        )

    @staticmethod
    def _variant_score(text: str, variant: QueryVariant) -> float:
        compact = text.replace("％", "%").replace(" ", "")
        hits = 0.0
        for term in variant.terms:
            normalized = str(term).replace("％", "%").replace(" ", "")
            if not normalized:
                continue
            if normalized in compact:
                hits += min(7.0, 1.5 + len(normalized) / 3.5)
        query_compact = variant.text.replace("％", "%").replace(" ", "")
        if len(query_compact) >= 8 and query_compact in compact:
            hits += 30.0
        return hits

    def _query_anchor_spans(self, claim: Any, doc_id: str, domain: str,
                            variants: Sequence[QueryVariant], *, limit: int = 8) -> List[EvidenceSpanV2]:
        """Create compact exact windows around high-value query anchors.

        This is a lexical IndexV2 operation, not a generic full-document retry.
        It is especially useful when a formula/table clause is outside the
        highest scoring legacy chunk window.
        """
        full = self.index.get_doc_text(doc_id)
        if not full:
            return []
        terms: List[Tuple[float, str, str]] = []
        for variant in variants:
            for term in variant.terms:
                term = str(term or "").strip()
                if len(term) < 2:
                    continue
                priority = float(variant.weight)
                if re.search(r"\d|%|％", term):
                    priority += 1.5
                if term in {"身故给付比例", "基本保险金额", "个人账户价值", "保单账户价值", "较大值",
                            "保存期限", "差异报告", "市场规模", "复合增长率", "兑付日", "资产减值补偿"}:
                    priority += 2.0
                terms.append((priority, term, variant.query_id))
        terms.sort(key=lambda row: (-row[0], -len(row[1]), row[1]))
        rows: List[Tuple[float, int, int, str, List[str], List[str]]] = []
        seen_positions = set()
        for priority, term, query_id in terms[:24]:
            cursor = 0
            occurrences = 0
            while occurrences < 4:
                pos = full.find(term, cursor)
                if pos < 0:
                    break
                cursor = pos + max(1, len(term))
                occurrences += 1
                bucket = pos // 240
                if bucket in seen_positions:
                    continue
                seen_positions.add(bucket)
                before, after = (520, 980) if domain in {"regulatory", "financial_contracts", "insurance"} else (420, 820)
                start, end, text = centered_window(full, pos, before=before, after=after)
                matched_queries = []
                score = 30.0 + 8.0 * priority
                matched_terms: List[str] = []
                for variant in variants:
                    vscore = self._variant_score(text, variant)
                    if vscore > 0:
                        score += variant.weight * vscore
                        matched_queries.append(variant.query_id)
                        matched_terms.extend(t for t in variant.terms if t in text)
                rows.append((score, start, end, text, unique_strings(matched_terms), unique_strings(matched_queries)))
        rows.sort(key=lambda row: (-row[0], row[1]))
        out: List[EvidenceSpanV2] = []
        occupied: List[Tuple[int, int]] = []
        for serial, (score, start, end, text, matched_terms, matched_queries) in enumerate(rows, 1):
            if any(max(start, a) < min(end, b) and min(end, b) - max(start, a) > 0.72 * min(end - start, b - a)
                   for a, b in occupied):
                continue
            occupied.append((start, end))
            out.append(EvidenceSpanV2(
                evidence_id=f"query_anchor:{doc_id}:{claim_attr(claim, 'claim_id', '')}:{start}",
                doc_id=doc_id, text=text, score=score, source="b1_lite_query_anchor",
                char_start=start, char_end=end, matched_terms=matched_terms[:16],
                source_hash=self.index.source_hash(doc_id),
                metadata={"query_channels": matched_queries, "query_anchor": True},
            ))
            if len(out) >= limit:
                break
        return out

    def _chunk_spans(self, claim: Any, doc_id: str, terms: Sequence[str], domain: str,
                     *, variants: Sequence[QueryVariant] = (), limit: int = 28) -> List[EvidenceSpanV2]:
        # Rank each chunk under independent option/claim query channels and
        # fuse the ranks with weighted reciprocal-rank fusion.
        raw_candidates = []
        chunks = list(self.index.get_chunks(doc_id))
        for serial, chunk in enumerate(chunks):
            body = str(getattr(chunk, "text", "") or "")
            base_score, matched = self._score(body, claim, terms)
            variant_scores = {v.query_id: self._variant_score(body, v) for v in variants}
            if base_score <= 0 and not any(score > 0 for score in variant_scores.values()):
                continue
            raw_candidates.append({
                "serial": serial, "chunk": chunk, "body": body,
                "base": base_score, "matched": matched, "variant_scores": variant_scores,
            })

        rank_maps: Dict[str, Dict[int, int]] = {}
        for variant in variants:
            ranked = sorted(
                (row for row in raw_candidates if row["variant_scores"].get(variant.query_id, 0.0) > 0),
                key=lambda row: (-row["variant_scores"][variant.query_id], row["serial"]),
            )
            rank_maps[variant.query_id] = {row["serial"]: rank for rank, row in enumerate(ranked, 1)}

        candidates = []
        for row in raw_candidates:
            rrf = 0.0
            channels: List[str] = []
            for variant in variants:
                rank = rank_maps.get(variant.query_id, {}).get(row["serial"] )
                if rank is None:
                    continue
                rrf += variant.weight / (40.0 + rank)
                channels.append(variant.query_id)
            fused = float(row["base"]) + 220.0 * rrf
            candidates.append((fused, row["serial"], row["chunk"], row["matched"], channels, rrf))
        candidates.sort(key=lambda row: (-row[0], row[1]))

        spans: List[EvidenceSpanV2] = []
        full = self.index.get_doc_text(doc_id)
        for score, _, chunk, matched, channels, rrf in candidates[:limit]:
            body = str(getattr(chunk, "text", "") or "")
            chunk_start = int(getattr(chunk, "char_start", 0) or 0)
            chunk_end = int(getattr(chunk, "char_end", chunk_start + len(body)) or chunk_start + len(body))
            if len(body) > 5000:
                positions = match_positions(body, matched or terms[:12], per_term_limit=4)[:8]
                for window_serial, (position, term) in enumerate(positions, 1):
                    absolute = chunk_start + position
                    start, end, text = centered_window(full or body, absolute if full else position,
                                                       before=650, after=1250)
                    local_score, local_matched = self._score(text, claim, terms, base=score * 0.1)
                    spans.append(EvidenceSpanV2(
                        evidence_id=f"window:{doc_id}:{getattr(chunk, 'chunk_id', '')}:{window_serial}",
                        doc_id=doc_id, chunk_id=str(getattr(chunk, "chunk_id", "")), text=text,
                        score=local_score, source="legacy_huge_chunk_window",
                        char_start=start, char_end=end, matched_terms=local_matched,
                        section_path=list(getattr(chunk, "section_path", []) or [str(getattr(chunk, "title", ""))]),
                        source_hash=self.index.source_hash(doc_id),
                        metadata={"anchor_term": term, "legacy_chunk_chars": len(body),
                                  "query_channels": channels, "weighted_rrf": rrf},
                    ))
                continue
            start, end, text = self._expand(doc_id, chunk_start, chunk_end, domain)
            if not text:
                text, start, end = body, chunk_start, chunk_end
            local_score, local_matched = self._score(text, claim, terms, base=score * 0.2)
            spans.append(EvidenceSpanV2(
                evidence_id=f"chunk:{doc_id}:{getattr(chunk, 'chunk_id', '')}", doc_id=doc_id,
                chunk_id=str(getattr(chunk, "chunk_id", "")), text=text, score=local_score,
                source="chunk_expanded", char_start=start, char_end=end,
                matched_terms=local_matched,
                section_path=list(getattr(chunk, "section_path", []) or [str(getattr(chunk, "title", ""))]),
                source_hash=self.index.source_hash(doc_id),
                metadata={"query_channels": channels, "weighted_rrf": rrf},
            ))
        return spans

    def _full_scan(self, claim: Any, doc_ids: Sequence[str], terms: Sequence[str]) -> List[EvidenceSpanV2]:
        spans = []
        # For literal absence use the most specific field/keyword, not generic subject words.
        fields = [alias for field in claim_attr(claim, "field_candidates", []) or [] for alias in aliases_for_field(field, adjudication=False)]
        search_terms = unique_strings(fields + list(claim_attr(claim, "keywords", []) or []) + list(terms))
        search_terms = sorted((term for term in search_terms if len(term) >= 2), key=len, reverse=True)
        for did in doc_ids:
            full = self.index.get_doc_text(did)
            found = None
            for term in search_terms[:20]:
                position = full.find(term)
                if position >= 0:
                    found = (position, term)
                    break
            if found:
                start, end, text = centered_window(full, found[0], before=600, after=1000)
                spans.append(EvidenceSpanV2(
                    evidence_id=f"fullscan:found:{did}", doc_id=did, text=text, score=60.0,
                    source="full_document_scan", char_start=start, char_end=end,
                    matched_terms=[found[1]], source_hash=self.index.source_hash(did),
                    metadata={"full_scan_completed": True, "literal_found": True, "term": found[1]},
                ))
            else:
                spans.append(EvidenceSpanV2(
                    evidence_id=f"fullscan:absent:{did}", doc_id=did,
                    text=f"[FULL_SCAN_ABSENT] 已完整扫描文档，未发现目标术语：{' / '.join(search_terms[:8])}",
                    score=45.0, source="full_document_scan", char_start=0, char_end=len(full),
                    matched_terms=[], source_hash=self.index.source_hash(did),
                    metadata={"full_scan_completed": True, "literal_found": False, "terms": search_terms[:20]},
                ))
        return spans

    @staticmethod
    def _dedupe(spans: Iterable[EvidenceSpanV2]) -> List[EvidenceSpanV2]:
        out = []
        seen = set()
        for span in sorted(spans, key=lambda item: (-item.score, item.doc_id, item.char_start or -1)):
            key = (span.doc_id, span.char_start, span.char_end, normalize_text(span.text[:300]))
            if key in seen:
                continue
            seen.add(key)
            out.append(span)
        return out

    def retrieve_claim(self, claim: Any, routed_doc_ids: Sequence[str], domain: str, *,
                       max_spans: int = 8, required_docs: Sequence[str] = (),
                       extra_terms: Sequence[str] = (), min_distinct_docs: int = 1,
                       question: str = "", option_text: str = "") -> List[EvidenceSpanV2]:
        claim_docs = unique_strings(claim_attr(claim, "doc_hints", []) or [])
        scoped_docs = list(required_docs) + claim_docs
        target_docs = unique_strings(scoped_docs if scoped_docs else list(routed_doc_ids))
        target_docs = [did for did in target_docs if did in self.index.meta_table and str(getattr(self.index.meta_table[did], "domain", "")) == domain]
        terms = self.claim_terms(claim, extra_terms)
        variants = self.query_planner.build(claim, domain, question=question, option_text=option_text)
        if bool(claim_attr(claim, "literal_absence", False)):
            spans = self._full_scan(claim, claim_docs or target_docs, terms)
            for span in spans:
                span.metadata.setdefault("doc_identity_terms", self._identity_terms(span.doc_id))
            return spans

        spans: List[EvidenceSpanV2] = []
        for did in target_docs:
            exact = self._exact_option_span(claim, did, domain, terms)
            if exact is not None:
                spans.append(exact)
            spans.extend(self._structured_spans(claim, did, terms))
            spans.extend(self._table_spans(claim, did, terms))
            spans.extend(self._query_anchor_spans(claim, did, domain, variants, limit=6))
            spans.extend(self._chunk_spans(claim, did, terms, domain, variants=variants))
        spans = self._dedupe(spans)
        for span in spans:
            span.metadata.setdefault("doc_identity_terms", self._identity_terms(span.doc_id))

        # Per-document quota prevents one long document from filling all slots.
        selected = []
        quota_docs = claim_docs or list(required_docs)
        if not quota_docs and min_distinct_docs > 1:
            # Reserve one evidence window from the strongest distinct documents
            # for comparison/universal claims even when B-list provides no IDs.
            for did in unique_strings(span.doc_id for span in spans)[:min_distinct_docs]:
                selected.extend([span for span in spans if span.doc_id == did][:1])
        else:
            for did in quota_docs:
                selected.extend([span for span in spans if span.doc_id == did][:2])
        # Reserve the strongest evidence for each independent query purpose.
        # This is the evidence-level counterpart of per-option document quota.
        for variant in variants:
            hit = next((span for span in spans if variant.query_id in (span.metadata or {}).get("query_channels", [])), None)
            if hit is not None and hit not in selected:
                selected.append(hit)
            if len(selected) >= max_spans:
                break
        for span in spans:
            if span not in selected:
                selected.append(span)
            if len(selected) >= max_spans:
                break
        out = self._dedupe(selected)[:max_spans]
        for span in out:
            span.metadata.setdefault("query_variant_count", len(variants))
            span.metadata.setdefault("query_variants", [v.to_dict() for v in variants])
        return out
