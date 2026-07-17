#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Index-aware correction of retrieval subjects and multi-document duties.

This module does not parse claims independently.  It only resolves named
entities already present in the option/question against the runtime index and
uses them to correct retrieval obligations when compiler candidates are noisy
or pronominal.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from retrieval_models import EvidenceObligation, RetrievalPlan, claim_attr, unique_strings


_MULTI_REFERENCE = re.compile(r"两家|两份|双方|二者|前者|后者|均|分别|相比|对比")
_COMPARISON = re.compile(r"高于|低于|大于|小于|超过|不足|不低于|不超过|早于|晚于|优于|不及|相比|对比|>|<")
_GENERIC = {"公司", "该公司", "两家公司", "发行人", "文档", "两份文档", "本产品", "该产品", "上市公司"}


class IndexAwarePlanEnricher:
    def __init__(self, runtime_index):
        self.index = runtime_index
        self.aliases_by_domain: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._build_aliases()

    def _build_aliases(self) -> None:
        rows: Dict[str, Dict[str, str]] = defaultdict(dict)
        for standard, entry in self.index.entity_lexicon.items():
            # Legacy EntityEntry.domain can reflect only the first document even
            # when the same entity appears across several domains.  Derive the
            # actual domains from doc_ids instead of trusting that single field.
            domains = {
                str(getattr(self.index.meta_table.get(str(doc_id)), "domain", "") or "")
                for doc_id in getattr(entry, "doc_ids", []) or []
                if str(doc_id) in self.index.meta_table
            }
            if not domains:
                domains = {str(getattr(entry, "domain", "") or "")}
            aliases = [str(standard)] + [str(x) for x in getattr(entry, "aliases", []) or []]
            for domain in domains:
                for alias in aliases:
                    alias = alias.strip()
                    if len(alias) >= 2 and alias not in _GENERIC:
                        rows[domain].setdefault(alias, str(standard))
        for doc_id, meta in self.index.meta_table.items():
            domain = str(getattr(meta, "domain", "") or "")
            canonical = str(getattr(meta, "issuer", "") or "") or str(getattr(meta, "title", "") or doc_id)
            aliases = [canonical, str(getattr(meta, "title", "") or "")]
            aliases.extend(str(x) for x in getattr(meta, "aliases", []) or [])
            identity = self.index.doc_identities.get(str(doc_id), {}) or {}
            for value in identity.values():
                if isinstance(value, list):
                    aliases.extend(str(x) for x in value)
                elif value:
                    aliases.append(str(value))
            for alias in aliases:
                alias = alias.strip()
                if len(alias) >= 2 and alias not in _GENERIC:
                    rows[domain].setdefault(alias, canonical)
        for domain, mapping in rows.items():
            self.aliases_by_domain[domain] = sorted(mapping.items(), key=lambda item: (-len(item[0]), item[0]))

    def entities_in_text(self, text: str, domain: str) -> List[str]:
        text = str(text or "")
        hits: List[Tuple[int, int, str]] = []
        occupied: List[Tuple[int, int]] = []
        for alias, canonical in self.aliases_by_domain.get(domain, []):
            start = text.find(alias)
            while start >= 0:
                end = start + len(alias)
                if not any(start < old_end and end > old_start for old_start, old_end in occupied):
                    hits.append((start, end, canonical or alias))
                    occupied.append((start, end))
                    break
                start = text.find(alias, start + 1)
        hits.sort(key=lambda row: row[0])
        return unique_strings(row[2] for row in hits)

    def enrich(self, plan: RetrievalPlan) -> RetrievalPlan:
        # Rebuild subject groups from index-resolved mentions.  Do not retain
        # noisy compiler inheritance such as a question subject copied into an
        # unrelated option.
        new_groups: List[List[str]] = []
        for option_claims in plan.claims_by_option.values():
            for claim in option_claims:
                claim_id = str(claim_attr(claim, "claim_id", ""))
                raw = str(claim_attr(claim, "raw", "") or "")
                entities = self.entities_in_text(raw, plan.domain)
                if _MULTI_REFERENCE.search(raw):
                    entities = unique_strings(entities + self.entities_in_text(plan.question, plan.domain))
                if not entities:
                    fallback = []
                    for value in claim_attr(claim, "subject_candidates", []) or []:
                        value = str(value or "").strip()
                        if value and value in raw and value not in _GENERIC and len(value) <= 40:
                            fallback.append(value)
                    entities = unique_strings(fallback)
                if entities:
                    new_groups.append(entities)
                obligations = plan.obligations.get(claim_id, [])
                for obligation in obligations:
                    if entities:
                        obligation.required_subjects = list(entities)
                if len(entities) >= 2 and (_COMPARISON.search(raw) or _MULTI_REFERENCE.search(raw)):
                    if not any(ob.obligation_type in {"CROSS_DOC_PAIR", "UNIVERSAL_MULTI_DOC"} for ob in obligations):
                        obligations.append(EvidenceObligation(
                            obligation_id=f"{claim_id}:OE1",
                            claim_id=claim_id,
                            obligation_type="CROSS_DOC_PAIR",
                            required_subjects=list(entities),
                            required_fields=unique_strings(claim_attr(claim, "field_candidates", []) or []),
                            required_years=unique_strings(claim_attr(claim, "years", []) or []),
                            min_distinct_docs=2,
                        ))
        plan.subject_groups = [group for group in new_groups if group]
        return plan
