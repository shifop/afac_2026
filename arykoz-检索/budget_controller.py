#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic evidence-memory budget controller.

The controller trims redundant evidence after sufficiency checking.  It always
preserves one strong span per claim and one span per explicitly required
document before filling the remaining character budget by score.  It never
uses an answer or confidence threshold to decide correctness.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Mapping, Sequence, Tuple

from evidence_card import EvidenceCardCompressor
from retrieval_models import EvidenceObligation, EvidenceSpanV2


class EvidenceBudgetController:
    def __init__(self, *, chars_per_token: float = 1.5, max_span_chars: int = 3600,
                 minimum_total_chars: int = 6000):
        self.chars_per_token = max(0.5, float(chars_per_token))
        self.max_span_chars = max(800, int(max_span_chars))
        self.minimum_total_chars = max(2000, int(minimum_total_chars))
        self.card_compressor = EvidenceCardCompressor(max_chars=min(1400, self.max_span_chars))

    def _bounded(self, span: EvidenceSpanV2) -> EvidenceSpanV2:
        if len(span.text) <= self.max_span_chars:
            return span
        text = span.text[:self.max_span_chars]
        # Prefer a complete sentence or line near the boundary.
        cut = max(text.rfind("。"), text.rfind("\n"), text.rfind("；"))
        if cut >= int(self.max_span_chars * 0.65):
            text = text[:cut + 1]
        metadata = dict(span.metadata)
        metadata["budget_truncated"] = True
        metadata["original_span_chars"] = len(span.text)
        end = span.char_end
        if span.char_start is not None:
            end = int(span.char_start) + len(text)
        return replace(span, text=text, char_end=end, metadata=metadata)

    @staticmethod
    def _required_docs(obligations: Sequence[EvidenceObligation]) -> List[str]:
        out: List[str] = []
        seen = set()
        for obligation in obligations:
            for doc_id in obligation.required_docs:
                if doc_id and doc_id not in seen:
                    out.append(doc_id)
                    seen.add(doc_id)
        return out

    def trim(self, obligations_by_claim: Mapping[str, Sequence[EvidenceObligation]],
             claim_evidence: Mapping[str, Sequence[EvidenceSpanV2]], *,
             token_budget: int, claims_by_id: Mapping[str, object] | None = None
             ) -> Tuple[Dict[str, List[EvidenceSpanV2]], Dict[str, object]]:
        char_budget = max(self.minimum_total_chars, int(max(1, token_budget) * self.chars_per_token))
        claims_by_id = dict(claims_by_id or {})
        bounded: Dict[str, List[EvidenceSpanV2]] = {}
        card_audits = []
        for claim_id, spans in claim_evidence.items():
            compressed, audit = self.card_compressor.compress_many(
                [self._bounded(span) for span in spans], claim=claims_by_id.get(claim_id)
            )
            bounded[claim_id] = compressed
            card_audits.append(audit)

        selected_keys = set()
        selected: Dict[str, List[EvidenceSpanV2]] = {claim_id: [] for claim_id in bounded}
        used_chars = 0

        def add(claim_id: str, span: EvidenceSpanV2) -> None:
            nonlocal used_chars
            key = (claim_id, span.evidence_id, span.doc_id, span.char_start, span.char_end)
            if key in selected_keys:
                return
            selected_keys.add(key)
            selected.setdefault(claim_id, []).append(span)
            used_chars += len(span.text)

        # Mandatory reservation: one strongest span per claim.
        for claim_id, spans in bounded.items():
            if spans:
                add(claim_id, max(spans, key=lambda row: row.score))

        # Mandatory reservation: every explicitly required document receives a
        # span when available, so cross-document claims survive budget trimming.
        for claim_id, obligations in obligations_by_claim.items():
            spans = bounded.get(claim_id, [])
            for doc_id in self._required_docs(obligations):
                candidates = [span for span in spans if span.doc_id == doc_id]
                if candidates:
                    add(claim_id, max(candidates, key=lambda row: row.score))

        candidates = []
        for claim_id, spans in bounded.items():
            for span in spans:
                candidates.append((float(span.score), claim_id, span))
        candidates.sort(key=lambda row: (-row[0], row[1], row[2].doc_id, row[2].char_start or -1))

        for _, claim_id, span in candidates:
            key = (claim_id, span.evidence_id, span.doc_id, span.char_start, span.char_end)
            if key in selected_keys:
                continue
            if used_chars + len(span.text) > char_budget:
                continue
            add(claim_id, span)

        for claim_id in selected:
            selected[claim_id].sort(key=lambda row: (-row.score, row.doc_id, row.char_start or -1))

        before_chars = sum(len(span.text) for spans in claim_evidence.values() for span in spans)
        after_chars = sum(len(span.text) for spans in selected.values() for span in spans)
        card_before = sum(int(row.get("chars_before", 0)) for row in card_audits)
        card_after = sum(int(row.get("chars_after", 0)) for row in card_audits)
        audit = {
            "token_budget": int(token_budget),
            "char_budget": char_budget,
            "chars_before": before_chars,
            "chars_after": after_chars,
            "estimated_tokens_after": int(round(after_chars / self.chars_per_token)),
            "mandatory_exceeded_budget": after_chars > char_budget,
            "spans_before": sum(len(spans) for spans in claim_evidence.values()),
            "spans_after": sum(len(spans) for spans in selected.values()),
            "evidence_cards": {
                "chars_before": card_before,
                "chars_after": card_after,
                "chars_saved": max(0, card_before - card_after),
                "duplicates_removed": sum(int(row.get("duplicates_removed", 0)) for row in card_audits),
                "spans_before": sum(int(row.get("spans_before", 0)) for row in card_audits),
                "spans_after": sum(int(row.get("spans_after", 0)) for row in card_audits),
            },
        }
        return selected, audit
