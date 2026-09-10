"""Independent claims support engine."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

from ..contracts import task_row
from ..decision import (
    CRITICAL_ARGUMENTS,
    assertion_gate,
    event_identity,
    nearest_alternative,
)
from ..retrieval import rank_documents, score_text
from ..semantics import (
    build_event_signature,
    canonical_key,
    canonical_predicate,
    clean_text,
    infer_assertion_polarity,
    semantic_tokens,
)
from .base import EngineContext, EngineResult, TaskEngine


_CLAIM_SURFACE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "for", "from", "had", "has", "have", "he", "her", "his",
    "in", "into", "is", "it", "its", "of", "on", "or", "she", "that",
    "the", "their", "they", "this", "those", "to", "was", "were",
    "which", "with",
})
_CLAIM_SURFACE_EQUIVALENTS = {
    "crucial": "important",
    "crucially": "important",
    "early": "initial",
    "global": "world",
    "indicate": "show",
    "indicated": "show",
    "join": "involve",
    "joined": "involve",
    "participate": "involve",
    "participated": "involve",
    "regional": "region",
    "states": "state",
    "strong": "strengthen",
    "sustain": "continue",
    "sustained": "continue",
    "sustaining": "continue",
    "continuance": "continue",
    "continued": "continue",
    "continues": "continue",
    "continuing": "continue",
    "vital": "important",
    "worldwide": "world",
}
_GENERIC_CLAIM_PREDICATES = frozenset({
    "claim", "claims", "statement", "assertion", "relation", "fact", "event",
})


def _claim_surface_tokens(value: Any) -> set[str]:
    output: set[str] = set()
    for token in semantic_tokens(value):
        token = re.sub(r"(?:'s|’s)$", "", token)
        if token in _CLAIM_SURFACE_STOPWORDS:
            continue
        canonical = _CLAIM_SURFACE_EQUIVALENTS.get(token)
        if canonical is None:
            canonical = canonical_predicate(token)
            canonical = _CLAIM_SURFACE_EQUIVALENTS.get(canonical, canonical)
        # Normalize simple English nominal plurals after predicate lemmatizing.
        # This is intentionally conservative and does not attempt general
        # stemming, which would merge unrelated short forms.
        if len(canonical) >= 5 and canonical.endswith("ies"):
            canonical = canonical[:-3] + "y"
        elif (
            len(canonical) >= 5
            and canonical.endswith("s")
            and not canonical.endswith(("ss", "us", "is"))
        ):
            canonical = canonical[:-1]
        if len(canonical) >= 2:
            output.add(canonical)
    return output


def _claim_surface_alignment(query_text: Any, candidate_text: Any) -> Dict[str, Any]:
    query_tokens = _claim_surface_tokens(query_text)
    candidate_tokens = _claim_surface_tokens(candidate_text)
    shared = query_tokens & candidate_tokens
    shorter = min(len(query_tokens), len(candidate_tokens))
    containment = len(shared) / float(max(1, shorter))
    jaccard = len(shared) / float(max(1, len(query_tokens | candidate_tokens)))
    def surface_polarity(value: Any) -> str:
        source = clean_text(value).casefold()
        if re.search(
            r"\b(?:no|not|never|without|deny|denies|denied|refuse|"
            r"refuses|refused|reject|rejects|rejected)\b",
            source,
        ):
            return "negative"
        return infer_assertion_polarity(source)

    same_polarity = surface_polarity(query_text) == surface_polarity(
        candidate_text
    )
    return {
        "canonical_equal": bool(
            canonical_key(query_text)
            and canonical_key(query_text) == canonical_key(candidate_text)
        ),
        "structural_surface_equivalent": bool(
            same_polarity
            and len(shared) >= 4
            and containment >= 0.85
            and jaccard >= 0.70
        ),
        "shared_token_count": len(shared),
        "token_containment": containment,
        "token_jaccard": jaccard,
        "same_surface_polarity": same_polarity,
    }


def _claim_retrieval_routes(query: Dict[str, Any]) -> List[tuple[str, str]]:
    signature = query.get("signature", {}) or {}
    actor = clean_text(
        signature.get("embedded_actor", "") or signature.get("actor", "")
    )
    predicate = clean_text(signature.get("predicate", ""))
    obj = clean_text(signature.get("object", ""))
    location = clean_text(
        signature.get("location_identity", signature.get("location", ""))
    )
    event_time = clean_text(
        signature.get("event_time", signature.get("date_value", ""))
    )
    routes = [
        ("SURFACE", clean_text(query.get("text", ""))),
        (
            "EVENT_CORE",
            clean_text(" ".join(
                value
                for value in (actor, predicate, obj, location, event_time)
                if value
            )),
        ),
        (
            "ARGUMENT_CORE",
            clean_text(" ".join(
                value
                for value in (actor, obj, location, event_time)
                if value
            )),
        ),
        (
            "PREDICATE_OBJECT",
            clean_text(" ".join(
                value
                for value in (predicate, obj, location, event_time)
                if value
            )),
        ),
        (
            "ACTOR_PREDICATE",
            clean_text(" ".join(
                value
                for value in (actor, predicate, location, event_time)
                if value
            )),
        ),
    ]
    unique: List[tuple[str, str]] = []
    seen: set[str] = set()
    for route_name, route_text in routes:
        normalized = clean_text(route_text)
        if not normalized or normalized in seen:
            continue
        unique.append((route_name, normalized))
        seen.add(normalized)
    return unique


def _claim_anchor_presence(hypotheses: List[Dict[str, Any]]) -> Dict[str, bool]:
    return {
        "actor": any(
            clean_text(hypothesis["signature"].get("actor", ""))
            for hypothesis in hypotheses
        ),
        "predicate": any(
            clean_text(hypothesis["signature"].get("predicate", "")).lower()
            not in _GENERIC_CLAIM_PREDICATES
            and bool(clean_text(hypothesis["signature"].get("predicate", "")))
            for hypothesis in hypotheses
        ),
        "object": any(
            clean_text(hypothesis["signature"].get("object", ""))
            for hypothesis in hypotheses
        ),
        "location": any(
            clean_text(
                hypothesis["signature"].get(
                    "location_identity",
                    hypothesis["signature"].get("location", ""),
                )
            )
            for hypothesis in hypotheses
        ),
        "event_time": any(
            clean_text(hypothesis["signature"].get("event_time", ""))
            for hypothesis in hypotheses
        ),
    }


def _claim_frame_is_executable(anchor_presence: Dict[str, bool]) -> bool:
    return bool(
        (
            anchor_presence["predicate"]
            and (anchor_presence["object"] or anchor_presence["actor"])
        )
        or (
            anchor_presence["object"]
            and (anchor_presence["location"] or anchor_presence["actor"])
        )
        or sum(anchor_presence.values()) >= 2
    )


def _repair_incomplete_claim_hypotheses(
    hypotheses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if _claim_frame_is_executable(_claim_anchor_presence(hypotheses)):
        return hypotheses
    repaired: List[Dict[str, Any]] = []
    for hypothesis in hypotheses:
        item = dict(hypothesis)
        fallback = build_event_signature(clean_text(item.get("text", "")))
        if _claim_frame_is_executable(
            _claim_anchor_presence([{**item, "signature": fallback}])
        ):
            item["signature"] = fallback
            item["signature_repair"] = "DETERMINISTIC_TEXT_FALLBACK"
        repaired.append(item)
    return repaired


class ClaimsEngine(TaskEngine):
    task_name = "claims"

    def run(self, context: EngineContext) -> EngineResult:
        raw_query_records = context.query.proposition_records()
        fragment_text_by_id = {
            clean_text(row.get("fragment_id", "")): clean_text(
                row.get(
                    "fragment_text_for_extraction",
                    row.get("fragment_text", ""),
                )
            )
            for row in context.query.fragments.to_dict("records")
            if clean_text(row.get("fragment_id", ""))
        }
        entity_text_by_fragment: Dict[str, List[str]] = defaultdict(list)
        if not context.query.entities.empty:
            for entity in context.query.entities.to_dict("records"):
                if not context.query._is_primary(
                    entity.get("primary_for_matching", True)
                ):
                    continue
                fragment_id = clean_text(entity.get("fragment_id", ""))
                label = clean_text(entity.get("label", ""))
                entity_type = clean_text(entity.get("type", "")).upper()
                if (
                    fragment_id
                    and label
                    and entity_type not in {"OTHER", "DOCUMENT", "DOCUMENT_ID"}
                    and label not in entity_text_by_fragment[fragment_id]
                ):
                    entity_text_by_fragment[fragment_id].append(label)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for query in raw_query_records:
            group_id = clean_text(
                query.get("hypothesis_group_id", "")
            ) or clean_text(query.get("id", ""))
            if group_id not in grouped:
                grouped[group_id] = []
                order.append(group_id)
            grouped[group_id].append(query)
        query_records: List[Dict[str, Any]] = []
        for group_id in order:
            hypotheses = grouped[group_id]
            hypotheses.sort(
                key=lambda item: (
                    -float(
                        item.get("hypothesis_confidence", 1.0)
                        or 0.0
                    ),
                    clean_text(item.get("id", "")),
                )
            )
            query = dict(hypotheses[0])
            query["hypotheses"] = [
                dict(item) for item in hypotheses
            ]
            fragment_id = clean_text(query.get("fragment_id", ""))
            fragment_text = fragment_text_by_id.get(fragment_id, "")
            deterministic_signature = (
                build_event_signature(fragment_text)
                if fragment_text else {}
            )
            signature_fields = (
                "actor", "predicate", "object", "location",
                "assertion_polarity", "semantic_state_variable",
                "semantic_state_value",
            )
            if fragment_text and all(
                tuple(
                    canonical_key(item.get("signature", {}).get(field, ""))
                    for field in signature_fields
                )
                != tuple(
                    canonical_key(deterministic_signature.get(field, ""))
                    for field in signature_fields
                )
                for item in query["hypotheses"]
            ):
                query["hypotheses"].append({
                    "id": f"{group_id}__full_fragment",
                    "fragment_id": fragment_id,
                    "hypothesis_group_id": group_id,
                    "hypothesis_confidence": 1.0,
                    "text": fragment_text,
                    "signature": deterministic_signature,
                    "signature_repair": "DETERMINISTIC_FULL_FRAGMENT",
                })
            query["hypotheses"] = _repair_incomplete_claim_hypotheses(
                query["hypotheses"]
            )
            query.update(query["hypotheses"][0])
            query["hypothesis_group_id"] = group_id
            query_records.append(query)
        executable_fragments = {
            clean_text(query.get("fragment_id", ""))
            for query in query_records
            if _claim_frame_is_executable(
                _claim_anchor_presence(query.get("hypotheses", [query]))
            )
        }
        evaluation_rows: List[Dict[str, Any]] = []
        candidate_rows: List[Dict[str, Any]] = []
        context.control.emit(
            "claims_started",
            target_count=len(query_records),
            corpus_proposition_count=sum(
                len(rows) for rows in context.corpus.propositions_by_doc.values()
            ),
        )
        for query in query_records:
            hypotheses = query.get("hypotheses", [query])
            anchor_presence = _claim_anchor_presence(hypotheses)
            executable_query_frame = _claim_frame_is_executable(anchor_presence)
            if not executable_query_frame:
                fragment_id = clean_text(query.get("fragment_id", ""))
                if fragment_id and fragment_id in executable_fragments:
                    context.control.emit(
                        "claims_incomplete_alternative_suppressed",
                        target_id=query["id"],
                        fragment_id=fragment_id,
                    )
                    continue
                missing_query_fields = [
                    field for field in ("actor", "predicate", "object")
                    if not anchor_presence.get(field, False)
                ]
                context.control.emit(
                    "claims_sparse_target_retrieval",
                    target_id=query["id"],
                    missing_query_fields=missing_query_fields,
                )
            retrieval: Dict[str, float] = {}
            retrieval_routes: Dict[str, set[str]] = defaultdict(set)
            full_fragment_text = fragment_text_by_id.get(
                clean_text(query.get("fragment_id", "")),
                clean_text(query.get("text", "")),
            )
            for doc_id, score in rank_documents(
                full_fragment_text,
                context.corpus,
                context.config.staged_top_docs,
                context.semantic_resources,
            ):
                retrieval[doc_id] = max(
                    retrieval.get(doc_id, 0.0), float(score)
                )
                retrieval_routes[doc_id].add("FULL_QUERY_FRAGMENT")
            for hypothesis in query.get("hypotheses", [query]):
                for route_name, route_text in _claim_retrieval_routes(
                    hypothesis
                ):
                    for doc_id, score in rank_documents(
                        route_text,
                        context.corpus,
                        context.config.staged_top_docs,
                        context.semantic_resources,
                    ):
                        retrieval[doc_id] = max(
                            retrieval.get(doc_id, 0.0), float(score)
                        )
                        retrieval_routes[doc_id].add(route_name)
            if not executable_query_frame:
                # A small entity-only route gives terse, good-faith analyst
                # fragments a fair retrieval attempt without exploding the
                # document union or pretending the fragment is a full claim.
                entity_route_limit = min(25, context.config.staged_top_docs)
                for entity_text in entity_text_by_fragment.get(
                    clean_text(query.get("fragment_id", "")), []
                )[:8]:
                    for doc_id, score in rank_documents(
                        entity_text,
                        context.corpus,
                        entity_route_limit,
                        context.semantic_resources,
                    ):
                        retrieval[doc_id] = max(
                            retrieval.get(doc_id, 0.0), float(score)
                        )
                        retrieval_routes[doc_id].add("QUERY_ENTITY")
            ranked_docs = sorted(
                retrieval.items(),
                key=lambda item: (
                    -item[1],
                    -len(retrieval_routes.get(item[0], set())),
                    item[0],
                ),
            )
            rows: List[Dict[str, Any]] = []
            for doc_id, _score in ranked_docs:
                for record in context.corpus.propositions_by_doc.get(doc_id, ()):
                    interpretations: List[Dict[str, Any]] = []
                    for hypothesis in query.get(
                        "hypotheses", [query]
                    ):
                        identity, states = event_identity(
                            hypothesis["signature"], record.signature,
                            hypothesis["text"], record.text,
                            context.semantic_resources,
                        )
                        lexical = score_text(
                            hypothesis["text"],
                            record.text,
                            context.semantic_resources,
                        )
                        mismatches = [
                            key for key, state in states.items()
                            if state == "MISMATCH"
                            and key in CRITICAL_ARGUMENTS
                        ]
                        matched_critical = [
                            key for key, state in states.items()
                            if state == "MATCH"
                            and key in CRITICAL_ARGUMENTS
                        ]
                        core_query_anchors = _claim_anchor_presence([hypothesis])
                        core_query_anchor_count = sum(
                            int(core_query_anchors[field])
                            for field in ("actor", "predicate", "object")
                        )
                        matched_core = [
                            key for key in (
                                "actor_state", "predicate_state", "object_state",
                            )
                            if states.get(key) == "MATCH"
                        ]
                        typed_direct_alignment = bool(
                            core_query_anchor_count >= 2
                            and len(matched_core) >= 2
                            and (
                                states.get("predicate_state") == "MATCH"
                                or states.get("object_state") == "MATCH"
                            )
                        )
                        missing_required_critical = [
                            key
                            for key in (
                                "actor_state", "predicate_state", "object_state",
                            )
                            if states.get(key) == "MISSING_IN_CANDIDATE"
                            and clean_text(
                                hypothesis["signature"].get(
                                    key.removesuffix("_state"), ""
                                )
                            )
                        ]
                        assertion_pass, assertion_failure = assertion_gate(
                            hypothesis["signature"], record.signature
                        )
                        full_surface_alignment = _claim_surface_alignment(
                            full_fragment_text, record.text
                        )
                        full_surface_direct = bool(
                            full_surface_alignment["same_surface_polarity"]
                            and (
                                full_surface_alignment["canonical_equal"]
                                or full_surface_alignment[
                                    "structural_surface_equivalent"
                                ]
                            )
                        )
                        if full_surface_direct:
                            relation = "DIRECT_SUPPORT"
                            reason = "full_query_fragment_surface_support"
                        elif (
                            identity
                            >= context.config.claim_direct_identity
                            and not mismatches
                            and not missing_required_critical
                            and assertion_pass
                            and typed_direct_alignment
                        ):
                            relation = "DIRECT_SUPPORT"
                            reason = (
                                "same_event_critical_arguments_and_"
                                "assertion_compatible"
                            )
                        elif (
                            identity
                            >= context.config.claim_direct_identity
                            and assertion_pass
                            and len(matched_critical) >= 2
                            and lexical
                            >= context.config.claim_raw_validation_floor
                        ):
                            # Some corpora produce near-verbatim claim rows where one
                            # critical field drifts only because a modifier is dropped
                            # or OCR splits a lexical item. When identity is already
                            # high and two critical fields align, treat that as direct
                            # support instead of demoting it to related-only.
                            relation = "DIRECT_SUPPORT"
                            reason = (
                                "same_event_high_identity_with_two_critical_"
                                "matches_and_strong_lexical_support"
                            )
                        elif (
                            assertion_pass
                            and mismatches == ["predicate_state"]
                            and states.get("actor_state") == "MATCH"
                            and states.get("object_state") == "MATCH"
                            and lexical >= max(
                                context.config.claim_raw_validation_floor,
                                0.82,
                            )
                        ):
                            # OpenAI query parsing occasionally lifts an
                            # intent/negotiation frame into a broader label
                            # such as "diplomatic negotiation" even when the
                            # target corpus row is otherwise near-verbatim.
                            # When actor and object still bind and only the
                            # predicate family drifts, preserve direct support
                            # instead of demoting the row to related-only.
                            relation = "DIRECT_SUPPORT"
                            reason = (
                                "same_event_actor_object_match_with_only_"
                                "predicate_family_drift"
                            )
                        elif (
                            identity
                            >= context.config.claim_conflict_identity
                            and mismatches
                        ):
                            relation = "CONTRADICTORY_ATTRIBUTE"
                            reason = (
                                "same_event_with_critical_attribute_mismatch"
                            )
                        elif identity >= context.config.related_identity:
                            relation = "RELATED_EVENT"
                            reason = (
                                assertion_failure
                                or "related_event_below_direct_support_gate"
                            )
                        elif (
                            lexical
                            >= context.config.claim_background_lexical_floor
                        ):
                            relation = "BACKGROUND"
                            reason = "topical_background"
                        else:
                            relation = "REJECTED_EVENT_MISMATCH"
                            reason = "event_identity_not_established"
                        final_score = (
                            0.58 * identity
                            + 0.22 * lexical
                            + 0.20 * retrieval.get(doc_id, 0.0)
                        )
                        query_actor = clean_text(
                            hypothesis["signature"].get("actor", "")
                        )
                        if (
                            relation != "DIRECT_SUPPORT"
                            and states.get("actor_state") == "MATCH"
                            and len(canonical_key(query_actor).split()) >= 2
                        ):
                            # For related/background evidence, a precise named
                            # actor is a stronger analyst-facing anchor than a
                            # generic predicate such as report or discuss.
                            # Direct-support eligibility remains unchanged.
                            final_score = min(1.0, final_score + 0.05)
                        if full_surface_direct:
                            final_score = max(
                                final_score,
                                0.72 * score_text(
                                    full_fragment_text,
                                    record.text,
                                    context.semantic_resources,
                                )
                                + 0.28 * retrieval.get(doc_id, 0.0),
                            )
                        interpretations.append({
                            "identity": identity,
                            "states": states,
                            "lexical": lexical,
                            "mismatches": mismatches,
                            "missing_required_critical":
                                missing_required_critical,
                            "assertion_pass": assertion_pass,
                            "typed_direct_alignment": typed_direct_alignment,
                            "core_query_anchor_count": core_query_anchor_count,
                            "relation": relation,
                            "reason": reason,
                            "final_score": final_score,
                            "hypothesis": hypothesis,
                        })
                    interpretations.sort(
                        key=lambda item: (
                            item["relation"] != "DIRECT_SUPPORT",
                            -float(item["final_score"]),
                            clean_text(
                                item["hypothesis"].get("id", "")
                            ),
                        )
                    )
                    best_interpretation = interpretations[0]
                    states = best_interpretation["states"]
                    hypothesis = best_interpretation["hypothesis"]
                    rows.append({
                        "query_proposition_id": query["id"],
                        "query_hypothesis_group_id": clean_text(
                            query.get("hypothesis_group_id", "")
                        ),
                        "winning_query_hypothesis_id": clean_text(
                            hypothesis.get(
                                "proposition_hypothesis_id",
                                hypothesis.get("id", ""),
                            )
                        ),
                        "query_hypothesis_count": len(
                            query.get("hypotheses", [query])
                        ),
                        "query_fragment_id": query["fragment_id"],
                        "query_proposition_text": query["text"],
                        "corpus_doc_id": doc_id,
                        "corpus_source_table": record.source_table,
                        "corpus_row_id": record.row_id,
                        "corpus_proposition_text": record.text,
                        "candidate_relation":
                            best_interpretation["relation"],
                        "eligible":
                            best_interpretation["relation"]
                            == "DIRECT_SUPPORT",
                        "retrieval_score": retrieval.get(doc_id, 0.0),
                        "event_identity_score":
                            best_interpretation["identity"],
                        "lexical_similarity":
                            best_interpretation["lexical"],
                        **states,
                        "assertion_gate_pass":
                            best_interpretation["assertion_pass"],
                        "critical_mismatch_count": len(
                            best_interpretation["mismatches"]
                        ),
                        "critical_match_count": len([
                            key
                            for key, state in states.items()
                            if state == "MATCH"
                            and key in {
                                "actor_state",
                                "predicate_state",
                                "object_state",
                            }
                        ]),
                        "core_query_anchor_count": best_interpretation[
                            "core_query_anchor_count"
                        ],
                        "typed_direct_alignment": best_interpretation[
                            "typed_direct_alignment"
                        ],
                        "missing_required_critical_count": len(
                            best_interpretation[
                                "missing_required_critical"
                            ]
                        ),
                        "raw_validation_status": "RAW_NOT_ASSESSED",
                        "raw_sentence": "",
                        "raw_similarity": 0.0,
                        "final_score":
                            best_interpretation["final_score"],
                        "decision_reason":
                            best_interpretation["reason"],
                    })
            context.control.check(len(rows))
            rows.sort(
                key=lambda row: (
                    -float(row["final_score"]),
                    row["corpus_doc_id"],
                    row["corpus_row_id"],
                )
            )
            normalized_query_text = full_fragment_text
            normalized_query_key = normalized_query_text.casefold()
            for finalist in rows[: context.config.raw_validation_finalists]:
                sentences = context.corpus.sentences_by_doc.get(
                    finalist["corpus_doc_id"], ()
                )
                best = max(
                    (
                        (score_text(
                            full_fragment_text,
                            sentence.sentence,
                            context.semantic_resources,
                        ), sentence.sentence)
                        for sentence in sentences
                    ),
                    default=(0.0, ""),
                )
                finalist["raw_similarity"] = best[0]
                finalist["raw_sentence"] = best[1]
                if (
                    not best[1]
                    or best[0] < context.config.claim_raw_validation_floor
                ):
                    finalist["raw_validation_status"] = "RAW_CANDIDATE_NOT_FOUND"
                elif finalist["candidate_relation"] == "DIRECT_SUPPORT":
                    finalist["raw_validation_status"] = (
                        "RAW_DIRECT_SUPPORT_CONFIRMED"
                    )
                elif finalist["candidate_relation"] in {
                    "RELATED_EVENT", "BACKGROUND"
                }:
                    finalist["raw_validation_status"] = (
                        "RAW_RELATED_EVIDENCE_CONFIRMED"
                    )
                else:
                    finalist["raw_validation_status"] = (
                        "RAW_CANDIDATE_PROPOSITION_CONFIRMED"
                    )
                proposition_text = clean_text(
                    finalist.get("corpus_proposition_text", "")
                )
                raw_text = clean_text(best[1])
                proposition_key = proposition_text.casefold()
                raw_key = raw_text.casefold()
                surface_alignment = _claim_surface_alignment(
                    normalized_query_text, proposition_text
                )
                exact_surface_alignment = bool(
                    surface_alignment["canonical_equal"]
                    or surface_alignment["structural_surface_equivalent"]
                )
                high_overlap_surface_alignment = bool(
                    surface_alignment["same_surface_polarity"]
                    and int(surface_alignment["shared_token_count"]) >= 5
                    and float(surface_alignment["token_containment"]) >= 0.75
                    and float(surface_alignment["token_jaccard"]) >= 0.45
                )
                finalist["surface_alignment_json"] = json.dumps(
                    surface_alignment,
                    sort_keys=True,
                )
                strong_surface_alignment = bool(
                    normalized_query_key
                    and (
                        exact_surface_alignment
                        or (
                            proposition_key
                            and normalized_query_key in proposition_key
                        )
                        or (
                            proposition_key
                            and proposition_key in normalized_query_key
                        )
                        or (
                            raw_key
                            and normalized_query_key in raw_key
                        )
                        or (
                            raw_key
                            and raw_key in normalized_query_key
                        )
                    )
                )
                strict_relation_raw_alignment = bool(
                    finalist["assertion_gate_pass"]
                    and int(
                        finalist.get("critical_mismatch_count", 0) or 0
                    ) <= 1
                    and int(
                        finalist.get("critical_match_count", 0) or 0
                    ) >= 2
                    and int(
                        finalist.get(
                            "missing_required_critical_count", 0
                        ) or 0
                    ) == 0
                    and clean_text(
                        finalist.get("corpus_source_table", "")
                    ) == "relations.csv"
                    and best[0] >= max(
                        context.config.claim_raw_validation_floor,
                        0.67,
                    )
                )
                paraphrastic_claim_alignment = bool(
                    finalist["assertion_gate_pass"]
                    and bool(finalist.get("typed_direct_alignment", False))
                    and int(
                        finalist.get("critical_mismatch_count", 0)
                        or 0
                    ) <= 2
                    and int(
                        finalist.get("critical_match_count", 0)
                        or 0
                    ) >= 1
                    and int(
                        finalist.get(
                            "missing_required_critical_count", 0
                        ) or 0
                    ) == 0
                    and clean_text(
                        finalist.get("corpus_source_table", "")
                    ) in {"claims.csv", "kg_pseudo_chunks_claims.csv"}
                    and float(
                        finalist.get("lexical_similarity", 0.0) or 0.0
                    )
                    >= 0.55
                    and best[0]
                    >= max(
                        context.config.claim_raw_validation_floor,
                        0.45,
                    )
                )
                high_overlap_claim_alignment = bool(
                    finalist["assertion_gate_pass"]
                    and int(
                        finalist.get("missing_required_critical_count", 0)
                        or 0
                    ) <= 1
                    and int(
                        finalist.get("critical_mismatch_count", 0)
                        or 0
                    ) <= 2
                    and clean_text(
                        finalist.get("corpus_source_table", "")
                    ) in {"claims.csv", "kg_pseudo_chunks_claims.csv"}
                    and high_overlap_surface_alignment
                    and float(
                        finalist.get("lexical_similarity", 0.0) or 0.0
                    ) >= 0.60
                    and best[0] >= max(
                        context.config.claim_raw_validation_floor,
                        0.45,
                    )
                )
                if (
                    finalist["candidate_relation"] in {
                        "RELATED_EVENT", "BACKGROUND",
                        "CONTRADICTORY_ATTRIBUTE",
                    }
                    and exact_surface_alignment
                    and (
                        (
                            surface_alignment["canonical_equal"]
                            and float(
                                finalist.get("lexical_similarity", 0.0) or 0.0
                            ) >= 0.98
                        )
                        or (
                            surface_alignment[
                                "structural_surface_equivalent"
                            ]
                            and float(
                                finalist.get("lexical_similarity", 0.0) or 0.0
                            ) >= 0.62
                        )
                    )
                ):
                    # Canonically identical or high-overlap reordered surfaces
                    # are stronger evidence than disagreeing parser fields.
                    # This repairs punctuation and active/passive parses while
                    # preserving polarity and a strict content-overlap gate.
                    finalist["candidate_relation"] = "DIRECT_SUPPORT"
                    finalist["eligible"] = True
                    finalist["decision_reason"] = (
                        "identical_normalized_claim_surface"
                    )
                    finalist["raw_validation_status"] = (
                        "RAW_DIRECT_SUPPORT_CONFIRMED"
                    )
                elif (
                    finalist["candidate_relation"] in {
                        "RELATED_EVENT", "BACKGROUND"
                    }
                    and finalist["assertion_gate_pass"]
                    and clean_text(
                        finalist.get("corpus_source_table", "")
                    ) in {"claims.csv", "kg_pseudo_chunks_claims.csv"}
                    and float(
                        finalist.get("lexical_similarity", 0.0) or 0.0
                    )
                    >= 0.75
                    and best[0]
                    >= max(
                        context.config.claim_raw_validation_floor,
                        0.6,
                    )
                    and strong_surface_alignment
                ):
                    finalist["candidate_relation"] = "DIRECT_SUPPORT"
                    finalist["eligible"] = True
                    finalist["decision_reason"] = (
                        "verbatim_or_embedded_claim_surface_with_"
                        "raw_confirmation"
                    )
                    finalist["raw_validation_status"] = (
                        "RAW_DIRECT_SUPPORT_CONFIRMED"
                    )
                elif (
                    finalist["candidate_relation"] in {
                        "RELATED_EVENT", "BACKGROUND"
                    }
                    and strict_relation_raw_alignment
                ):
                    # Relation rows can be deliberately terse while their
                    # source sentence retains the full proposition. Require a
                    # raw-sentence match plus two typed critical anchors; a
                    # relation surface alone never receives this promotion.
                    finalist["candidate_relation"] = "DIRECT_SUPPORT"
                    finalist["eligible"] = True
                    finalist["decision_reason"] = (
                        "relation_surface_with_strict_raw_sentence_"
                        "confirmation"
                    )
                    finalist["raw_validation_status"] = (
                        "RAW_DIRECT_SUPPORT_CONFIRMED"
                    )
                elif (
                    finalist["candidate_relation"] in {
                        "RELATED_EVENT", "BACKGROUND"
                    }
                    and (
                        paraphrastic_claim_alignment
                        or high_overlap_claim_alignment
                    )
                ):
                    finalist["candidate_relation"] = "DIRECT_SUPPORT"
                    finalist["eligible"] = True
                    finalist["decision_reason"] = (
                        "paraphrastic_claim_surface_with_raw_"
                        "confirmation"
                    )
                    finalist["raw_validation_status"] = (
                        "RAW_DIRECT_SUPPORT_CONFIRMED"
                    )
            candidate_rows.extend(rows)
            decision_candidates: Dict[tuple[str, str], Dict[str, Any]] = {}
            for candidate in rows:
                identity = (
                    clean_text(candidate.get("corpus_doc_id", "")),
                    clean_text(
                        candidate.get("corpus_proposition_text", "")
                    ).casefold(),
                )
                previous = decision_candidates.get(identity)
                if (
                    previous is None
                    or float(candidate.get("final_score", 0.0) or 0.0)
                    > float(previous.get("final_score", 0.0) or 0.0)
                ):
                    decision_candidates[identity] = candidate
            decision_rows = sorted(
                decision_candidates.values(),
                key=lambda row: (
                    -float(row.get("final_score", 0.0) or 0.0),
                    clean_text(row.get("corpus_doc_id", "")),
                    clean_text(row.get("corpus_proposition_text", "")),
                ),
            )
            direct = [
                row for row in decision_rows
                if row["candidate_relation"] == "DIRECT_SUPPORT"
            ]
            pool = direct or decision_rows
            winner = pool[0] if pool else {}
            # Only another direct-support candidate is comparable for a
            # selection margin. Related or rejected rows remain alternatives.
            runner = direct[1] if len(direct) > 1 else {}
            nearest = nearest_alternative(
                decision_rows,
                winner,
                identity_fields=(
                    "corpus_doc_id", "corpus_source_table", "corpus_row_id",
                ),
            )
            sparse_retrieval_score = float(
                winner.get("final_score", 0.0) or 0.0
            )
            status = (
                "DIRECT_SUPPORT" if direct
                else "RELATED_EVIDENCE_ONLY"
                if (
                    not executable_query_frame
                    and winner
                    and sparse_retrieval_score >= 0.22
                )
                else "NO_RELEVANT_CORPUS_EVIDENCE"
                if not executable_query_frame
                else "NO_DIRECT_SUPPORT_AFTER_EXCLUSION"
                if context.corpus.excluded_doc_ids
                else "RELATED_EVIDENCE_ONLY"
                if rows else "NO_CANDIDATE"
            )
            evaluation_rows.append(task_row(
                "claims", query["id"], status,
                localization_status=(
                    "AUTHORITATIVE_SOURCE_SELECTED"
                    if status == "DIRECT_SUPPORT"
                    else "RELATED_EVIDENCE_FOUND"
                    if winner
                    else "NO_SOURCE_FOUND"
                ),
                raw_status=clean_text(
                    winner.get("raw_validation_status", "RAW_NOT_ASSESSED")
                ),
                selected_text=(
                    clean_text(winner.get("corpus_proposition_text", ""))
                    if status == "DIRECT_SUPPORT" else ""
                ),
                selected_source=(
                    clean_text(winner.get("corpus_doc_id", ""))
                    if status == "DIRECT_SUPPORT" else ""
                ),
                top_candidate=clean_text(winner.get("corpus_doc_id", "")),
                related_source=(
                    clean_text(winner.get("corpus_doc_id", ""))
                    if status != "DIRECT_SUPPORT" else ""
                ),
                best_score=winner.get("final_score", ""),
                runner_up=runner.get("final_score", ""),
                margin=(
                    float(winner.get("final_score", 0.0))
                    - float(runner.get("final_score", 0.0))
                    if winner and runner else None
                ),
                decision_competitor_doc_id=clean_text(
                    runner.get("corpus_doc_id", "")
                ),
                decision_competitor_passage_id=clean_text(
                    runner.get("corpus_row_id", "")
                ),
                nearest_alternative_score=nearest.get("final_score"),
                nearest_alternative_doc_id=clean_text(
                    nearest.get("corpus_doc_id", "")
                ),
                nearest_alternative_passage_id=clean_text(
                    nearest.get("corpus_row_id", "")
                ),
                nearest_alternative_eligible=(
                    clean_text(nearest.get("candidate_relation", ""))
                    == "DIRECT_SUPPORT"
                    if nearest else None
                ),
                nearest_alternative_failed_gates=clean_text(
                    nearest.get("decision_reason", "")
                ),
                reason=(
                    "sparse_query_retrieval_only:"
                    + clean_text(winner.get("decision_reason", "no_candidate"))
                    if not executable_query_frame and winner
                    else clean_text(winner.get("decision_reason", "no_candidate"))
                ),
                unresolved="" if status == "DIRECT_SUPPORT" else status,
            ))
            context.control.emit(
                "claims_target_complete",
                target_id=query["id"],
                shortlisted_document_count=len(ranked_docs),
                candidate_count=len(rows),
                status=status,
            )
        candidates = pd.DataFrame(candidate_rows)
        primary = (
            candidates.sort_values(
                ["query_proposition_id", "final_score"],
                ascending=[True, False],
            ).groupby("query_proposition_id", as_index=False, group_keys=False).head(50)
            if len(candidates) else candidates
        )
        return EngineResult(
            task=self.task_name,
            evaluation=pd.DataFrame(evaluation_rows),
            artifacts={
                "claim_candidate_audit_v048.csv": candidates,
                "claim_support_candidates_v048.csv": primary,
            },
        )
