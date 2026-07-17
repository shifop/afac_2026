#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AFAC retrieval V2 public facade.

- Reads the current complete legacy pickle immediately.
- Reads IndexV2 after rebuild with no API change.
- Uses V6.1 ClaimCompiler as the only formal Claim parser.
- Provides native retrieve_v2() and legacy retrieve() compatibility.
- Does not use the retired V1/V2 scorer chain.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

from budget_controller import EvidenceBudgetController
from claim_planner import RetrievalPlanBuilder, SharedClaimCompiler
from document_router import DocumentRouter
from evidence_retriever import ClaimEvidenceRetriever
from legacy_index import RuntimeIndex
from plan_enricher import IndexAwarePlanEnricher
from retrieval_models import RetrievalResult, RetrievalResultV2, claim_attr
from sufficiency_controller import SufficiencyController


class DocumentRetriever:
    def __init__(self, index_path: Optional[str] = None, *, compiler: Any = None,
                 fact_index: Any = None):
        if not index_path:
            try:
                import b_config as cfg
                index_path = getattr(cfg, "INDEX_V2_PKL", "") or getattr(cfg, "INDEX_PKL", "")
            except Exception:
                index_path = ""
        if not index_path:
            raise ValueError("index_path is required")
        self.index_path = str(index_path)
        self.index = RuntimeIndex.load(self.index_path)
        self.shared_compiler = SharedClaimCompiler(compiler=compiler, fact_index=fact_index)
        self.plan_builder = RetrievalPlanBuilder(self.shared_compiler)
        self.plan_enricher = IndexAwarePlanEnricher(self.index)
        self.router = DocumentRouter(self.index)
        self.evidence_retriever = ClaimEvidenceRetriever(self.index)
        self.sufficiency = SufficiencyController()
        self.budget = EvidenceBudgetController()
        self.last_v2_result: Optional[RetrievalResultV2] = None

    def retrieve_v2(self, *, qid: str, question: str, options: Mapping[str, str], domain: str,
                    answer_format: str = "single", ordered_doc_ids: Sequence[str] = (),
                    hard_doc_ids: Sequence[str] = (),
                    claims_by_option: Optional[Dict[str, List[Any]]] = None,
                    top_k_docs: int = 5, top_k_claim_spans: int = 8,
                    token_budget: int = 10000) -> RetrievalResultV2:
        plan = self.plan_builder.build(
            qid=qid, question=question, options=options, domain=domain,
            answer_format=answer_format, ordered_doc_ids=ordered_doc_ids,
            hard_doc_ids=hard_doc_ids, claims_by_option=claims_by_option,
        )
        plan = self.plan_enricher.enrich(plan)
        route = self.router.route(plan, top_k_docs=top_k_docs)
        route_audit = [dict(route.audit)]
        claim_evidence: Dict[str, List[Any]] = {}
        claim_lookup: Dict[str, Any] = {}
        for option_claims in plan.claims_by_option.values():
            for claim in option_claims:
                claim_id = str(claim_attr(claim, "claim_id", ""))
                claim_lookup[claim_id] = claim
                obligations = plan.obligations.get(claim_id, [])
                min_docs = max((ob.min_distinct_docs for ob in obligations), default=1)
                needs_multi = min_docs > 1
                explicit_required_docs = []
                obligation_terms = []
                for obligation in obligations:
                    explicit_required_docs.extend(obligation.required_docs)
                    obligation_terms.extend(obligation.required_subjects)
                    obligation_terms.extend(obligation.required_fields)
                    obligation_terms.extend(obligation.required_years)
                initial_required_docs = list(dict.fromkeys(
                    (plan.ordered_doc_ids if needs_multi and plan.ordered_doc_ids else []) + explicit_required_docs
                ))
                option_key = str(claim_attr(claim, "option_key", "") or claim_id.split("_", 1)[0])
                claim_evidence[claim_id] = self.evidence_retriever.retrieve_claim(
                    claim, route.doc_ids, domain, max_spans=top_k_claim_spans,
                    required_docs=initial_required_docs,
                    extra_terms=list(dict.fromkeys(obligation_terms)),
                    min_distinct_docs=min_docs,
                    question=question,
                    option_text=str(options.get(option_key, "")),
                )

        status, missing, confidence = self.sufficiency.evaluate(plan.obligations, claim_evidence)
        retrieval_rounds = 1

        # Bounded evidence-gap loop.  Each additional round expands document
        # routing as well as the local claim terms; it does not merely rescan
        # the same Top-K documents.
        for round_index in range(1, max(1, plan.max_evidence_rounds)):
            if not missing:
                break
            retrieval_rounds += 1
            expanded_route = self.router.route(
                plan, top_k_docs=top_k_docs,
                expansion=max(2, top_k_docs * round_index),
            )
            route = expanded_route
            route_audit.append(dict(expanded_route.audit))
            gap_requests = self.sufficiency.gap_request(missing)
            for claim_id, request in gap_requests.items():
                claim = claim_lookup.get(claim_id)
                if claim is None:
                    continue
                min_docs = max((ob.min_distinct_docs for ob in plan.obligations.get(claim_id, [])), default=1)
                option_key = str(claim_attr(claim, "option_key", "") or claim_id.split("_", 1)[0])
                extra = self.evidence_retriever.retrieve_claim(
                    claim, route.doc_ids, domain,
                    max_spans=top_k_claim_spans + 2 * round_index,
                    required_docs=request["required_docs"],
                    extra_terms=request["extra_terms"],
                    min_distinct_docs=min_docs,
                    question=question,
                    option_text=str(options.get(option_key, "")),
                )
                claim_evidence[claim_id] = self.evidence_retriever._dedupe(
                    claim_evidence.get(claim_id, []) + extra
                )[:top_k_claim_spans + 2 * round_index]
            status, missing, confidence = self.sufficiency.evaluate(plan.obligations, claim_evidence)

        missing_before_budget = list(missing)
        confidence_before_budget = confidence
        claim_evidence, budget_audit = self.budget.trim(
            plan.obligations, claim_evidence, token_budget=token_budget,
            claims_by_id=claim_lookup,
        )
        # The final result reports sufficiency of the exact evidence memory that
        # will be handed to V6.1, not the larger temporary candidate pool.
        status, missing, confidence = self.sufficiency.evaluate(plan.obligations, claim_evidence)
        fallback_level = 0 if retrieval_rounds == 1 and not missing else (1 if not missing else 2)

        option_evidence: Dict[str, List[Any]] = defaultdict(list)
        for option_key, claims in plan.claims_by_option.items():
            for claim in claims:
                option_evidence[option_key].extend(
                    claim_evidence.get(str(claim_attr(claim, "claim_id", "")), [])
                )
            option_evidence[option_key] = self.evidence_retriever._dedupe(
                option_evidence[option_key]
            )[:12]

        all_final_spans = [span for spans in claim_evidence.values() for span in spans]
        query_anchor_spans = sum(1 for span in all_final_spans if (span.metadata or {}).get("query_anchor"))
        weighted_rrf_spans = sum(1 for span in all_final_spans if (span.metadata or {}).get("weighted_rrf"))
        query_variant_counts = [
            int((span.metadata or {}).get("query_variant_count", 0) or 0)
            for span in all_final_spans
        ]
        b1_lite_audit = {
            "option_claim_query_decomposition": True,
            "weighted_rrf": True,
            "query_variants_routed": sum(int((row or {}).get("query_variant_count", 0) or 0) for row in route_audit),
            "query_variants_in_evidence": max(query_variant_counts, default=0),
            "query_anchor_spans": query_anchor_spans,
            "weighted_rrf_spans": weighted_rrf_spans,
            "gap_retry_rounds": max(0, retrieval_rounds - 1),
            "evidence_cards": dict((budget_audit or {}).get("evidence_cards", {}) or {}),
        }

        result = RetrievalResultV2(
            doc_ids=route.doc_ids, doc_scores=route.doc_scores,
            doc_reasons=route.doc_reasons, option_evidence=dict(option_evidence),
            claim_evidence=claim_evidence, obligation_status=status,
            missing_obligations=missing, confidence=confidence,
            fallback_level=fallback_level,
            audit={
                "schema_version": self.index.schema_version,
                "index_path": self.index_path,
                "route_rounds": route_audit,
                "retrieval_rounds": retrieval_rounds,
                "token_budget": token_budget,
                "budget": budget_audit,
                "claims": len(claim_lookup),
                "obligations": sum(len(rows) for rows in plan.obligations.values()),
                "missing_before_budget": missing_before_budget,
                "confidence_before_budget": confidence_before_budget,
                "stage_b1_lite": b1_lite_audit,
            },
        )
        self.last_v2_result = result
        return result

    def compact_augmented_evidence(self, retrieval: RetrievalResultV2, *,
                                   claims_by_option: Mapping[str, Sequence[Any]],
                                   max_claim_spans: int = 8, max_option_spans: int = 12
                                   ) -> Dict[str, int]:
        """Compress evidence added after retrieve_v2 (targeted retry/ledger).

        Targeted retry runs in the integration layer because it needs original
        document paths.  This second quote-safe pass ensures those newly added
        long windows do not bypass the B1-Lite evidence-card budget.
        """
        claim_lookup: Dict[str, Any] = {}
        for claims in claims_by_option.values():
            for claim in claims:
                claim_lookup[str(claim_attr(claim, "claim_id", ""))] = claim
        totals = {
            "chars_before": 0, "chars_after": 0, "chars_saved": 0,
            "duplicates_removed": 0, "spans_before": 0, "spans_after": 0,
        }
        for claim_id, spans in list(retrieval.claim_evidence.items()):
            compressed, audit = self.budget.card_compressor.compress_many(
                list(spans or []), claim=claim_lookup.get(claim_id)
            )
            retrieval.claim_evidence[claim_id] = compressed[:max_claim_spans]
            for key in totals:
                totals[key] += int(audit.get(key, 0) or 0)
        rebuilt: Dict[str, List[Any]] = defaultdict(list)
        for option_key, claims in claims_by_option.items():
            for claim in claims:
                rebuilt[str(option_key)].extend(
                    retrieval.claim_evidence.get(str(claim_attr(claim, "claim_id", "")), [])
                )
            rebuilt[str(option_key)] = self.evidence_retriever._dedupe(rebuilt[str(option_key)])[:max_option_spans]
        retrieval.option_evidence = dict(rebuilt)
        stage = retrieval.audit.setdefault("stage_b1_lite", {})
        stage["post_augmentation_evidence_cards"] = totals
        return totals

    def retrieve(self, question: str, options: dict, domain: str,
                 answer_format: str = "single", top_k_docs: int = 5,
                 top_k_chunks: int = 10, token_budget: int = 10000) -> RetrievalResult:
        qid = "retrieval_" + hashlib.sha1(
            f"{domain}\n{question}\n{options}".encode("utf-8", errors="ignore")
        ).hexdigest()[:12]
        result = self.retrieve_v2(
            qid=qid, question=question, options=options, domain=domain,
            answer_format=answer_format, top_k_docs=top_k_docs,
            top_k_claim_spans=max(4, min(top_k_chunks, 12)), token_budget=token_budget,
        )
        return result.to_legacy(max_chunks=top_k_chunks)


__all__ = ["DocumentRetriever", "RetrievalResult", "RetrievalResultV2"]
