#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explainable document routing with independent subject quotas and rank fusion."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from field_ontology import aliases_for_field
from option_query_planner import OptionQueryPlanner
from retrieval_models import RetrievalPlan, claim_attr, unique_strings
from text_utils import extract_years, normalize_text, rrf_score, tokenize


@dataclass
class RouteResult:
    doc_ids: List[str]
    doc_scores: Dict[str, float]
    doc_reasons: Dict[str, List[str]]
    audit: Dict[str, Any] = field(default_factory=dict)


class DocumentRouter:
    def __init__(self, runtime_index):
        self.index = runtime_index
        self._domain_stats: Dict[str, Dict[str, Any]] = {}
        self._term_count_cache: Dict[Tuple[str, str], int] = {}
        self._identity_terms: Dict[str, List[Tuple[str, str]]] = {}
        self.query_planner = OptionQueryPlanner()
        self._financial_doc_alias_cache: Dict[str, List[str]] = {}

    @staticmethod
    def _insurance_identity_forms(value: str) -> List[str]:
        """Normalize common product short names without a hardcoded doc map.

        B-list questions often use market abbreviations (家财险、食责险、重疾险)
        while the clauses contain the formal product name.  These are lexical
        identity aliases, not relationship expansion.
        """
        raw = normalize_text(str(value or ""))
        if not raw:
            return []
        forms = [raw]
        company_prefixes = [
            "中国平安财产保险股份有限公司", "中国平安人寿保险股份有限公司",
            "中国平安", "平安产险", "平安人寿", "平安养老", "平安健康", "平安",
            "众安在线财产保险股份有限公司", "众安在线", "众安保险", "众安",
            "中国人寿保险股份有限公司", "中国人寿", "国寿",
            "中国太平洋财产保险股份有限公司", "中国太平洋人寿保险股份有限公司",
            "太平洋保险", "太保",
        ]
        stripped = raw
        changed = True
        while changed:
            changed = False
            for prefix in company_prefixes:
                prefix_n = normalize_text(prefix)
                if stripped.startswith(prefix_n) and len(stripped) > len(prefix_n):
                    stripped = stripped[len(prefix_n):]
                    changed = True
                    break
        if stripped and stripped not in forms:
            forms.append(stripped)
        replacements = {
            "食责险": "食品安全责任保险",
            "家财险": "家庭财产保险",
            "重疾险": "重大疾病保险",
            "医疗险": "医疗保险",
            "意外险": "意外伤害保险",
            "年金险": "年金保险",
            "责任险": "责任保险",
            "车险": "机动车保险",
        }
        for base in list(forms):
            expanded = base
            for short, full in replacements.items():
                expanded = expanded.replace(short, full)
            if expanded and expanded not in forms:
                forms.append(expanded)
        return [form for form in forms if len(form) >= 3]

    def _claim_identity_docs(self, query: str, domain: str) -> List[str]:
        """Find document identities explicitly named inside one atomic claim."""
        query_text = normalize_text(str(query or ""))
        if not query_text:
            return []
        query_forms = self._insurance_identity_forms(query_text) if domain == "insurance" else [query_text]
        scored: List[Tuple[int, str]] = []
        for did in self.index.all_doc_ids(domain):
            meta = self.index.meta_table[did]
            values = [str(getattr(meta, "title", "") or ""), str(getattr(meta, "issuer", "") or "")]
            values.extend(str(x) for x in getattr(meta, "aliases", []) or [])
            identity = self.index.doc_identities.get(did, {})
            for value in identity.values():
                if isinstance(value, list):
                    values.extend(str(x) for x in value)
                elif value:
                    values.append(str(value))
            best = 0
            for value in values:
                value_forms = self._insurance_identity_forms(value) if domain == "insurance" else [normalize_text(value)]
                for qform in query_forms:
                    for vform in value_forms:
                        if len(vform) < 3:
                            continue
                        if vform in qform or qform in vform:
                            best = max(best, min(len(vform), len(qform)))
            if best >= 3:
                scored.append((best, did))
        scored.sort(key=lambda item: (-item[0], self._document_preference(item[1])))
        return unique_strings(did for _, did in scored)


    @staticmethod
    def _annual_year(meta: Any, doc_id: str) -> int:
        raw = str(getattr(meta, "year", "") or "")
        match = re.search(r"20\d{2}", raw) or re.search(r"20\d{2}", str(doc_id))
        return int(match.group(0)) if match else 0

    def _financial_doc_aliases(self, doc_id: str) -> List[str]:
        """Return validated issuer aliases for one annual report.

        Legacy IndexV2 metadata uses English shorthands for a few reports
        (for example ``chinamobile`` and ``cmb``).  The entity lexicon may also
        contain polluted cross-document links, so an alias is accepted only
        when it occurs in the report's opening text or is already present in
        the document metadata.
        """
        if doc_id in self._financial_doc_alias_cache:
            return self._financial_doc_alias_cache[doc_id]
        meta = self.index.meta_table[doc_id]
        values: List[str] = [
            str(getattr(meta, "title", "") or ""),
            str(getattr(meta, "issuer", "") or ""),
        ]
        values.extend(str(x) for x in getattr(meta, "aliases", []) or [])
        values.extend(str(x) for x in getattr(meta, "key_entities", []) or [])
        identity = self.index.doc_identities.get(doc_id, {}) or {}
        for value in identity.values():
            if isinstance(value, list):
                values.extend(str(x) for x in value)
            elif value:
                values.append(str(value))
        opening = normalize_text(self.index.get_doc_text(doc_id)[:16000])
        for standard, entry in self.index.entity_lexicon.items():
            linked = {str(x) for x in getattr(entry, "doc_ids", []) or []}
            if doc_id not in linked:
                continue
            candidates = [str(standard)] + [str(x) for x in getattr(entry, "aliases", []) or []]
            # Validate lexicon aliases against the actual report opening. This
            # prevents a polluted lexicon edge from creating a hard route.
            if any(len(normalize_text(alias)) >= 3 and normalize_text(alias) in opening for alias in candidates):
                values.extend(candidates)
        out: List[str] = []
        generic = {"公司", "集团", "股份有限公司", "年度报告", "report", "annual"}
        legal_noise = ("公司法", "证券法", "管理办法", "披露办法", "信披办法", "会计准则", "条例", "规定")
        for value in values:
            cleaned = re.sub(r"20\d{2}年?年度报告.*$", "", str(value or ""), flags=re.I).strip()
            norm = normalize_text(cleaned)
            if len(norm) < 3 or norm.lower() in generic:
                continue
            if re.fullmatch(r"20\d{2}(?:年度|年)?", norm) or re.fullmatch(r"\d{6}", norm):
                continue
            if any(term in cleaned for term in legal_noise):
                continue
            if norm not in [normalize_text(x) for x in out]:
                out.append(cleaned)
        self._financial_doc_alias_cache[doc_id] = out
        return out

    def _financial_report_scope(self, question: str) -> Dict[str, Any]:
        """Infer an answer-blind annual-report scope from the question only.

        A uniquely named issuer is a hard document identity.  Restricting the
        route to that issuer prevents facts from another company's annual
        report from entering FactIndex trend checks.  Explicit years are
        honored; ``连续两年/近两年`` selects the latest two reports for the
        issuer.  Multiple named issuers are kept independently.
        """
        annual_docs = [
            did for did in self.index.all_doc_ids("financial_reports")
            if str(getattr(self.index.meta_table[did], "doc_subtype", "") or "") in {"annual", ""}
            and str(did).startswith("annual_")
        ]
        qnorm = normalize_text(question)
        matched: Dict[str, Dict[str, Any]] = {}
        for did in annual_docs:
            aliases = self._financial_doc_aliases(did)
            hits = [alias for alias in aliases if len(normalize_text(alias)) >= 3 and normalize_text(alias) in qnorm]
            if not hits:
                continue
            # Use the stable annual-report id stem as the issuer group. This
            # keeps 2024/2025 reports together even when one metadata title is
            # reconstructed from an English ticker.
            visible = max(hits, key=lambda x: len(normalize_text(x)))
            group_key = re.sub(r"_20\d{2}_report$", "", str(did))
            row = matched.setdefault(group_key, {"alias": visible, "docs": []})
            if len(normalize_text(visible)) > len(normalize_text(str(row.get("alias", "")))):
                row["alias"] = visible
            row["docs"].append(did)

        if not matched:
            return {"active": False, "doc_ids": [], "issuers": [], "years": [], "reason": "no_named_issuer"}

        explicit_years = sorted({int(y) for y in extract_years(question)})
        latest_two = bool(re.search(r"连续\s*两年|近\s*两年|两年(?:的)?年度报告|前后\s*两年", question))
        scoped: List[str] = []
        issuer_rows: List[Dict[str, Any]] = []
        one_global_year = explicit_years[0] if len(explicit_years) == 1 else 0
        for row in matched.values():
            docs = sorted(set(row["docs"]), key=lambda did: (self._annual_year(self.index.meta_table[did], did), did))
            alias = str(row["alias"])
            local_years: List[int] = []
            # Prefer the nearest year when two named issuers carry distinct
            # years (宁德时代2024与中国移动2025). For a single issuer, retain
            # both explicitly named comparison years.
            apos = qnorm.find(normalize_text(alias))
            if apos >= 0 and len(matched) > 1 and len(explicit_years) > 1:
                year_positions = [(abs(m.start() - apos), int(m.group(0))) for m in re.finditer(r"20\d{2}", qnorm)]
                if year_positions:
                    local_years = [min(year_positions)[1]]
            elif apos >= 0:
                nearby = qnorm[max(0, apos - 8): apos + len(normalize_text(alias)) + 28]
                local_years = [int(y) for y in re.findall(r"20\d{2}", nearby)]
            wanted = sorted(set(local_years)) or ([one_global_year] if one_global_year else explicit_years)
            selected = [did for did in docs if not wanted or self._annual_year(self.index.meta_table[did], did) in wanted]
            if latest_two and not explicit_years:
                selected = docs[-2:]
            if not selected:
                selected = docs
            scoped.extend(selected)
            issuer_rows.append({
                "issuer_alias": alias,
                "available_docs": docs,
                "wanted_years": wanted,
                "selected_docs": selected,
            })
        scoped = unique_strings(scoped)
        return {
            "active": True,
            "doc_ids": scoped,
            "issuers": issuer_rows,
            "years": explicit_years,
            "latest_two": latest_two,
            "reason": "question_named_annual_report_issuer",
        }


    def _document_preference(self, doc_id: str) -> Tuple[int, int, str]:
        """Prefer authoritative/normative variants within one document family."""
        meta = self.index.meta_table[str(doc_id)]
        role = str(getattr(meta, "extraction_variant", "") or "")
        role_rank = {
            "normative_text": 0,
            "enforcement_decision": 0,
            "legacy_reconstructed": 1,
            "mineru": 2,
            "text": 2,
            "revision_explanation": 5,
        }.get(role, 2)
        did = str(doc_id)
        if did.startswith("strict_v3_"):
            source_rank = 0
        elif did.startswith("csrc_") and "_att1" in did:
            source_rank = 0
        elif re.match(r"^csrc_\d+$", did):
            source_rank = 0
        elif did.startswith("csrc_penalty_"):
            source_rank = 4
        elif did.startswith("csrc_") and "_att" not in did:
            source_rank = 1
        elif did.startswith("regulatory_mineru_"):
            source_rank = 3
        elif "_att2" in did or "_att3" in did or "_att4" in did:
            source_rank = 4
        else:
            source_rank = 2
        return role_rank, source_rank, did

    def _build_identity_terms(self, domain: str) -> List[Tuple[str, str]]:
        if domain in self._identity_terms:
            return self._identity_terms[domain]
        rows: List[Tuple[str, str]] = []
        for did in self.index.all_doc_ids(domain):
            meta = self.index.meta_table[did]
            values = [str(getattr(meta, "title", "") or "")]
            values.extend(str(x) for x in getattr(meta, "aliases", []) or [])
            for value in values:
                cleaned = str(value or "").strip()
                if len(cleaned) >= 4:
                    rows.append((cleaned, did))
        # Longest aliases first; exact visible law titles should dominate generic
        # abbreviations such as “公司法”.
        rows.sort(key=lambda item: (-len(item[0]), self._document_preference(item[1])))
        self._identity_terms[domain] = rows
        return rows

    def _named_document_terms(self, plan: RetrievalPlan) -> List[str]:
        if plan.domain != "regulatory":
            return []
        question = str(plan.question or "")
        terms = [x.strip() for x in re.findall(r"《([^》]{4,120})》", question)]
        normalized_question = normalize_text(question)
        # Also detect unquoted exact titles/aliases.  Only the question is used
        # here so deliberately wrong option titles cannot become hard routes.
        for alias, _ in self._build_identity_terms(plan.domain):
            normalized_alias = normalize_text(alias)
            if len(normalized_alias) >= 4 and normalized_alias in normalized_question:
                terms.append(alias)
        return unique_strings(terms)

    def _ensure_stats(self, domain: str) -> Dict[str, Any]:
        """Return a persisted BM25 inverted index for one domain.

        IndexV2 stores per-token postings and document lengths.  Legacy indexes
        may lack postings; for those only, a compatibility rebuild is performed
        once and cached.  Production B-list runs should use IndexV2 so the first
        query never scans the complete corpus.
        """
        if domain in self._domain_stats:
            return self._domain_stats[domain]

        stored = dict(self.index.bm25.get(domain, {}) or {})
        postings = stored.get("postings") or {}
        doc_ids = [str(x) for x in stored.get("doc_ids", []) or self.index.all_doc_ids(domain)]
        if postings:
            raw_lengths = stored.get("doc_lengths", []) or []
            if isinstance(raw_lengths, dict):
                lengths = {str(k): max(1, int(v)) for k, v in raw_lengths.items()}
            else:
                lengths = {
                    did: max(1, int(raw_lengths[i])) if i < len(raw_lengths) else 1
                    for i, did in enumerate(doc_ids)
                }
            stats = {
                "doc_ids": doc_ids,
                "postings": postings,
                "lengths": lengths,
                "avgdl": float(stored.get("avgdl", 0.0) or (sum(lengths.values()) / max(1, len(lengths)))),
                "source": "persisted_index",
            }
            self._domain_stats[domain] = stats
            return stats

        # Compatibility only: old pickles contain df/lengths but not term
        # frequencies.  Build once, record the source in audit, and strongly
        # prefer migrating with migrate_legacy_index_v2.py before production.
        postings_fallback: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        lengths: Dict[str, int] = {}
        for did in doc_ids:
            tokens = [token for token in tokenize(self.index.get_doc_text(did), ()) if len(token) >= 2]
            counts = Counter(tokens)
            lengths[did] = max(1, sum(counts.values()))
            for token, frequency in counts.items():
                postings_fallback[token].append((did, int(frequency)))
        stats = {
            "doc_ids": doc_ids,
            "postings": dict(postings_fallback),
            "lengths": lengths,
            "avgdl": sum(lengths.values()) / max(1, len(lengths)),
            "source": "legacy_runtime_rebuild",
        }
        self._domain_stats[domain] = stats
        return stats

    def _identity_docs(self, term: str, domain: str) -> List[str]:
        term = str(term or "").strip()
        if len(term) < 2:
            return []
        out = []
        for did in self.index.all_doc_ids(domain):
            meta = self.index.meta_table[did]
            values = [str(getattr(meta, "title", "") or ""), str(getattr(meta, "issuer", "") or "")]
            values.extend(str(x) for x in getattr(meta, "aliases", []) or [])
            identity = self.index.doc_identities.get(did, {})
            for value in identity.values():
                if isinstance(value, list):
                    values.extend(str(x) for x in value)
                elif value:
                    values.append(str(value))
            if any(term == value or term in value or value in term for value in values if len(value) >= 2):
                out.append(did)
        return sorted(unique_strings(out), key=self._document_preference)

    def _entity_docs(self, term: str, domain: str) -> List[str]:
        out = []
        for standard, entry in self.index.entity_lexicon.items():
            aliases = [str(standard)] + [str(x) for x in getattr(entry, "aliases", []) or []]
            if any(term == alias or term in alias or alias in term for alias in aliases if len(alias) >= 2):
                out.extend(str(x) for x in getattr(entry, "doc_ids", []) or [] if str(x) in self.index.meta_table)
        return [did for did in unique_strings(out) if str(getattr(self.index.meta_table[did], "domain", "")) == domain]

    def _bm25_rank(self, query: str, domain: str, limit: int) -> List[Tuple[str, float]]:
        stats = self._ensure_stats(domain)
        query_tokens = [token for token in tokenize(str(query or ""), ()) if len(token) >= 2]
        query_tokens = list(dict.fromkeys(query_tokens))[:96]
        scores: Dict[str, float] = defaultdict(float)
        n_docs = len(stats["doc_ids"])
        k1, b = 1.5, 0.75
        for token in query_tokens:
            posting = stats["postings"].get(token, [])
            if not posting:
                continue
            df = len(posting)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for did, freq in posting:
                dl = stats["lengths"].get(did, 1)
                denom = freq + k1 * (1.0 - b + b * dl / max(stats["avgdl"], 1.0))
                scores[did] += idf * freq * (k1 + 1.0) / denom
        rows = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return rows[:limit]

    def route(self, plan: RetrievalPlan, *, top_k_docs: int = 5, expansion: int = 0) -> RouteResult:
        domain_docs = set(self.index.all_doc_ids(plan.domain))
        financial_scope = {"active": False, "doc_ids": []}
        if plan.domain == "financial_reports":
            financial_scope = self._financial_report_scope(str(plan.question or ""))
            if financial_scope.get("active") and financial_scope.get("doc_ids"):
                domain_docs &= set(str(x) for x in financial_scope["doc_ids"])
        channel_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
        channel_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        reasons: Dict[str, List[str]] = defaultdict(list)
        query_best: Dict[str, str] = {}
        hard = [did for did in plan.hard_doc_ids if did in domain_docs]
        authoritative_ordered = [did for did in plan.ordered_doc_ids if did in domain_docs]
        if authoritative_ordered:
            return RouteResult(
                doc_ids=authoritative_ordered,
                doc_scores={did: 1000.0 for did in authoritative_ordered},
                doc_reasons={did: ["authoritative_ordered_doc"] for did in authoritative_ordered},
                audit={"dynamic_k": len(authoritative_ordered), "required_count": len(authoritative_ordered),
                       "channels": {"authoritative_ordered_doc": len(authoritative_ordered)},
                       "hard_docs": authoritative_ordered, "bm25_source": "authoritative_docs",
                       "financial_report_scope": financial_scope},
            )

        # Hard document references are never subject to Top-K competition.
        for did in hard:
            channel_ranks["hard"][did] = 1
            reasons[did].append("hard_doc_reference")

        title_terms = self._named_document_terms(plan)
        for term in title_terms:
            docs = [did for did in self._identity_docs(term, plan.domain) if did in domain_docs]
            for rank, did in enumerate(docs, 1):
                channel_ranks[f"title:{term}"][did] = rank
                reasons[did].append(f"title:{term}")

        claims = [claim for rows in plan.claims_by_option.values() for claim in rows]
        subjects = unique_strings(subject for group in plan.subject_groups for subject in group)
        fields = unique_strings(
            field for claim in claims for field in (claim_attr(claim, "field_candidates", []) or [])
        )
        years = unique_strings(year for claim in claims for year in (claim_attr(claim, "years", []) or []))

        for subject in subjects:
            docs = [did for did in unique_strings(self._identity_docs(subject, plan.domain) + self._entity_docs(subject, plan.domain)) if did in domain_docs]
            for rank, did in enumerate(docs, 1):
                channel_ranks[f"subject:{subject}"][did] = rank
                reasons[did].append(f"subject:{subject}")

        for field in fields:
            aliases = aliases_for_field(field, adjudication=False)
            docs = [did for did in domain_docs if any(alias in self.index.metrics_by_doc.get(did, set()) for alias in aliases)]
            for rank, did in enumerate(sorted(docs), 1):
                channel_ranks[f"field:{field}"][did] = rank
                reasons[did].append(f"field:{field}")

        for year in years:
            docs = []
            for did in domain_docs:
                meta = self.index.meta_table[did]
                meta_year = str(getattr(meta, "year", "") or "")
                if year in meta_year or year in str(getattr(meta, "title", "")) or year in did:
                    docs.append(did)
            for rank, did in enumerate(sorted(docs), 1):
                channel_ranks[f"year:{year}"][did] = rank
                reasons[did].append(f"year:{year}")

        # The question and every option/atomic claim are independent channels.
        # Stage B1-Lite decomposes each claim into a bounded set of lexical
        # variants and fuses them with weighted reciprocal-rank fusion.
        claim_identity_best: Dict[str, str] = {}
        claim_variant_best: Dict[str, List[str]] = defaultdict(list)
        query_variant_audit: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        queries: List[Tuple[str, str, float, str]] = [("question", plan.question, 0.8, "question") ]
        for claim in claims:
            claim_id = str(claim_attr(claim, "claim_id", ""))
            base_channel = f"claim:{claim_id}"
            variants = self.query_planner.build(claim, plan.domain, question=plan.question)
            query_variant_audit[claim_id] = [variant.to_dict() for variant in variants]
            for variant in variants:
                queries.append((f"{base_channel}:{variant.purpose}", variant.text, variant.weight, base_channel))

        for channel, query, weight, base_channel in queries:
            channel_weights[channel] = float(weight)
            identity_docs = [did for did in self._claim_identity_docs(query, plan.domain) if did in domain_docs]
            if identity_docs:
                identity_channel = f"identity:{channel}"
                channel_weights[identity_channel] = max(1.2, float(weight) + 0.25)
                claim_identity_best.setdefault(base_channel, identity_docs[0])
                for rank, did in enumerate(identity_docs, 1):
                    channel_ranks[identity_channel][did] = rank
                    reasons[did].append(identity_channel)
            rows = [(did, score) for did, score in self._bm25_rank(query, plan.domain, max(20, top_k_docs * 5)) if did in domain_docs]
            if rows:
                query_best[channel] = rows[0][0]
                if base_channel.startswith("claim:"):
                    claim_variant_best[base_channel].append(rows[0][0])
            for rank, (did, _) in enumerate(rows, 1):
                channel_ranks[channel][did] = rank
                reasons[did].append(channel)

        scores: Dict[str, float] = {}
        for did in domain_docs:
            weighted = [
                (channel_weights.get(name, 1.0), rank_map[did])
                for name, rank_map in channel_ranks.items() if did in rank_map
            ]
            if not weighted:
                continue
            scores[did] = sum(weight / (60.0 + rank) for weight, rank in weighted)
            title_hits = sum(1 for term in title_terms if f"title:{term}" in reasons.get(did, []))
            if title_hits:
                scores[did] += 5.0 * title_hits
            if did in hard:
                scores[did] += 1000.0

        required_count = max(1, len(hard), len(title_terms), len(plan.subject_groups), max(
            (ob.min_distinct_docs for obs in plan.obligations.values() for ob in obs), default=1
        ))
        dynamic_k = min(len(domain_docs), max(top_k_docs + expansion, required_count * 2 + 2, 5))

        # Collapse extraction duplicates and revision notes to one representative
        # per canonical document family before Top-K selection.  The legacy
        # corpus may contain csrc, strict-v3 and MinerU copies of the same law;
        # allowing all copies to compete wastes the route budget and suppresses
        # the second law required by cross-document questions.
        family_members: Dict[str, List[str]] = defaultdict(list)
        for did in scores:
            family_members[self.index.family_by_doc.get(did, did)].append(did)
        family_rows: List[Tuple[float, str]] = []
        family_representative: Dict[str, str] = {}
        for family, members in family_members.items():
            best_score = max(scores[did] for did in members)
            near_best = [did for did in members if scores[did] >= best_score * 0.65]
            representative = min(near_best, key=self._document_preference)
            # Rank the family by its strongest retrieval signal, but return the
            # authoritative representative when scores are effectively tied.
            scores[representative] = best_score
            family_representative[family] = representative
            family_rows.append((best_score, representative))
        ranked = [did for _, did in sorted(
            family_rows, key=lambda item: (-item[0], self._document_preference(item[1]))
        )]
        selected: List[str] = list(hard)
        # Reserve the best authoritative variant for every explicitly named law.
        for term in title_terms:
            candidates = [did for did in ranked if f"title:{term}" in reasons.get(did, [])]
            candidates.sort(key=lambda did: (self._document_preference(did), -scores.get(did, 0.0)))
            candidate = next((did for did in candidates if did not in selected), None)
            if candidate:
                selected.append(candidate)

        # Reserve the best document family for every atomic claim.  This is the
        # document-level counterpart of per-claim evidence quotas and prevents
        # rank fusion across four options from suppressing a document that is
        # decisive for only one option.
        reserve_channels = sorted(claim_variant_best)
        if str(plan.answer_format or "").lower() in {"tf", "judge"}:
            reserve_channels = ["question"] + reserve_channels
        for channel in reserve_channels:
            if channel == "question":
                raw_doc = query_best.get("question")
            else:
                raw_doc = claim_identity_best.get(channel)
                if not raw_doc:
                    candidates = claim_variant_best.get(channel, [])
                    raw_doc = candidates[0] if candidates else ""
            if not raw_doc or raw_doc not in domain_docs:
                continue
            family = self.index.family_by_doc.get(raw_doc, raw_doc)
            candidate = family_representative.get(family, raw_doc)
            if candidate not in selected:
                selected.append(candidate)
                reasons[candidate].append(f"quota:{channel}")

        # Reserve one document for each independently named subject group.
        for group in plan.subject_groups:
            candidate = next((did for did in ranked if did not in selected and any(
                f"subject:{subject}" in reasons.get(did, []) for subject in group
            )), None)
            if candidate:
                selected.append(candidate)

        family_counts = Counter(self.index.family_by_doc.get(did, did) for did in selected)
        for did in ranked:
            if did in selected:
                continue
            family = self.index.family_by_doc.get(did, did)
            if family_counts[family] >= 1 and did not in hard:
                continue
            selected.append(did)
            family_counts[family] += 1
            if len(selected) >= dynamic_k:
                break

        selected = [did for did in unique_strings(selected) if did in domain_docs]
        return RouteResult(
            doc_ids=selected,
            doc_scores={did: float(scores.get(did, 0.0)) for did in selected},
            doc_reasons={did: unique_strings(reasons.get(did, [])) for did in selected},
            audit={
                "dynamic_k": dynamic_k,
                "required_count": required_count,
                "channels": {name: len(rows) for name, rows in channel_ranks.items()},
                "hard_docs": hard,
                "named_document_terms": title_terms,
                "query_best": query_best,
                "claim_identity_best": claim_identity_best,
                "claim_variant_best": dict(claim_variant_best),
                "query_variants": dict(query_variant_audit),
                "query_variant_count": sum(len(rows) for rows in query_variant_audit.values()),
                "weighted_rrf": True,
                "bm25_source": self._ensure_stats(plan.domain).get("source", "unknown"),
                "financial_report_scope": financial_scope,
            },
        )
