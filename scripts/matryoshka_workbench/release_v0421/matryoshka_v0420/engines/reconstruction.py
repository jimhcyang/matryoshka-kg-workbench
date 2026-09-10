"""Independent single/collective reconstruction engine with provenance."""
from __future__ import annotations

from collections import defaultdict
import json
import math
import re
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import pandas as pd

from ..contracts import task_row
from ..decision import assertion_gate, event_identity, nearest_alternative
from ..decision_profile import DEFAULT_ENGINE_THRESHOLDS
from ..retrieval import containment, rank_documents, score_text
from ..semantic_resources import SemanticResources
from ..semantics import (
    build_event_signature,
    canonical_key,
    clean_text,
    compare_argument,
    compare_dates,
    normalize_numeric_arguments,
    semantic_tokens,
)
from .base import EngineContext, EngineResult, TaskEngine

PASSAGE_COLUMNS = [
    "fragment_id", "doc_id", "evidence_set", "pseudo_component_ids",
    "pseudo_source_row_ids", "pseudo_evidence",
    "matched_proposition_text", "matched_proposition_source_table",
    "matched_proposition_row_id",
    "pseudo_component_count", "pseudo_component_cards_json", "raw_chunk_ids",
    "raw_passage_count",
    "raw_chunk_cards_json", "previous_raw_chunk", "selected_raw_chunk",
    "next_raw_chunk", "raw_passage", "raw_status", "provenance_route",
]


def _reconstruction_retrieval_routes(
    query: Mapping[str, Any],
) -> List[tuple[str, str]]:
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

def _expanded_token_set(
    value: Any,
    resources: SemanticResources | None = None,
) -> Set[str]:
    if resources is None:
        return set(semantic_tokens(value))
    return set(resources.expanded_tokens(value))


_QUALIFIER_RE = re.compile(
    r"\b(?:because\s+of|because|due\s+to|owing\s+to|as\s+a\s+result\s+of|"
    r"before|after)\s+(?P<qualifier>[^.;]+)",
    flags=re.IGNORECASE,
)


def _qualifier_coverage(
    query_text: str,
    candidate_text: str,
    resources: SemanticResources,
) -> float:
    """Measure causal/temporal detail without making it an eligibility gate."""
    qualifiers = [
        clean_text(match.group("qualifier"))
        for match in _QUALIFIER_RE.finditer(clean_text(query_text))
        if clean_text(match.group("qualifier"))
    ]
    if not qualifiers:
        return 1.0
    candidate_tokens = resources.expanded_tokens(candidate_text)
    return min(
        containment(resources.expanded_tokens(value), candidate_tokens)
        for value in qualifiers
    )


def _fact_set(
    signature: Mapping[str, Any],
    text: str,
    resources: SemanticResources | None = None,
) -> Set[str]:
    facts: Set[str] = set()
    actor = clean_text(signature.get("actor", ""))
    embedded = clean_text(signature.get("embedded_actor", ""))
    if embedded:
        actor = embedded
    predicate_value = clean_text(signature.get("predicate", ""))
    object_value = clean_text(signature.get("object", ""))
    duplicate_predicate_object = bool(
        predicate_value
        and object_value
        and canonical_key(predicate_value) == canonical_key(object_value)
    )
    values = {
        "actor": actor,
        "predicate": predicate_value,
        "object": "" if duplicate_predicate_object else object_value,
        "location_identity": signature.get(
            "location_identity", signature.get("location", "")
        ),
        "event_time": signature.get(
            "event_time", signature.get("date_value", "")
        ),
        "semantic_state_variable": signature.get("semantic_state_variable", ""),
        "semantic_state_value": signature.get("semantic_state_value", ""),
    }
    for key, value in values.items():
        normalized = clean_text(value) if key == "event_time" else canonical_key(value)
        if normalized:
            facts.add(key + ":" + normalized)
        if resources is not None and clean_text(value):
            for concept in sorted(resources.concepts(value, "alias")):
                facts.add(key + "_alias:" + concept)
            for concept in sorted(resources.concepts(value, "event")):
                facts.add(key + "_event:" + concept)
    for key in ("packet_count", "total_weight_grams"):
        value = signature.get(key)
        if value is not None and not (
            isinstance(value, float) and math.isnan(value)
        ):
            facts.add(key + ":" + str(float(value)))
    if not facts:
        facts.update(
            "token:" + token
            for token in _expanded_token_set(text, resources)
            if len(token) >= 5
        )
    return facts


def _fact_dimension(fact: str) -> str:
    """Return the semantic field represented by one fact encoding.

    Canonical, alias, and event-concept encodings are alternative evidence for
    one field. Counting them independently makes a well-grounded entity look
    incomplete merely because one representation was absent.
    """
    kind = fact.partition(":")[0]
    for suffix in ("_alias", "_event"):
        if kind.endswith(suffix):
            return kind[:-len(suffix)]
    return kind


def _covered_dimensions(facts: Set[str]) -> Set[str]:
    return {_fact_dimension(fact) for fact in facts}


def _coverage_ratio(covered: Set[str], required: Set[str]) -> float:
    required_dimensions = _covered_dimensions(required)
    if not required_dimensions:
        return 0.0
    return len(
        _covered_dimensions(covered) & required_dimensions
    ) / len(required_dimensions)


def _covered(
    facts: Set[str],
    text: str,
    signature: Mapping[str, Any],
    literal_containment_floor: float = DEFAULT_ENGINE_THRESHOLDS[
        "reconstruction_literal_containment_floor"
    ],
    resources: SemanticResources | None = None,
) -> Set[str]:
    tokens = _expanded_token_set(text, resources)
    result: Set[str] = set()
    for fact in facts:
        kind, _, value = fact.partition(":")
        if kind == "token":
            if value in tokens:
                result.add(fact)
        elif kind.endswith("_alias"):
            field = kind[:-6]
            if clean_text(signature.get(field, "")) and resources is not None:
                if value in resources.concepts(signature.get(field, ""), "alias"):
                    result.add(fact)
        elif kind.endswith("_event"):
            field = kind[:-6]
            if clean_text(signature.get(field, "")) and resources is not None:
                if value in resources.concepts(signature.get(field, ""), "event"):
                    result.add(fact)
        elif kind in {"packet_count", "total_weight_grams"}:
            try:
                if abs(float(signature.get(kind)) - float(value)) <= 1e-6:
                    result.add(fact)
            except (TypeError, ValueError):
                pass
        elif kind == "event_time":
            if compare_dates(
                value, signature.get("event_time", signature.get("date_value", text))
            ) == "MATCH":
                result.add(fact)
        else:
            candidate_value = clean_text(signature.get(kind, ""))
            structured = (
                compare_argument(value, candidate_value) == "MATCH"
            )
            if resources is not None and not structured and candidate_value:
                # This is the same alias/event concept test used by
                # ``event_identity`` for one semantic field, without paying
                # for an entire eight-field identity calculation per fact.
                structured = bool(
                    resources.concepts(value, "alias")
                    & resources.concepts(candidate_value, "alias")
                ) or bool(
                    resources.concepts(value, "event")
                    & resources.concepts(candidate_value, "event")
                )
            literal = bool(
                _expanded_token_set(value, resources)
                and containment(_expanded_token_set(value, resources), tokens)
                >= literal_containment_floor
            )
            if structured or literal:
                result.add(fact)
    return result


def _localized_chunks(
    query_text: str,
    matched_proposition_text: str,
    sentences: Sequence[Any],
    resources: SemanticResources | None = None,
    limit: int = 3,
) -> List[Tuple[float, Any]]:
    """Choose distinct passages, retaining deterministic neighbor context."""
    ranked = sorted(
        (
            (
                max(
                    score_text(query_text, sentence.sentence, resources),
                    score_text(
                        matched_proposition_text,
                        sentence.sentence,
                        resources,
                    ) if matched_proposition_text else 0.0,
                ),
                sentence,
            )
            for sentence in sentences
        ),
        key=lambda item: (-item[0], item[1].chunk_id, item[1].sentence_index),
    )
    selected: List[Tuple[float, Any]] = []
    seen_passages: Set[Tuple[str, int, str]] = set()
    for score, sentence in ranked:
        identity = (
            clean_text(sentence.chunk_id),
            int(sentence.sentence_index),
            canonical_key(sentence.sentence),
        )
        if identity in seen_passages:
            continue
        selected.append((score, sentence))
        seen_passages.add(identity)
        if len(selected) >= limit:
            break
    return selected


def _passage_row(
    fragment_id: str,
    doc_id: str,
    query_text: str,
    context: EngineContext,
    evidence_set: str,
    *,
    matched_proposition_text: str = "",
    matched_proposition_source_table: str = "",
    matched_proposition_row_id: str = "",
) -> Dict[str, Any]:
    provenance = context.corpus.provenance_by_doc.get(doc_id, ())
    sentences = context.corpus.sentences_by_doc.get(doc_id, ())
    component_rows = [
        {
            "component_id": record.component_id,
            "source_table": record.source_table,
            "source_row_id": record.source_row_id,
            "text": record.component_text,
        }
        for record in provenance
        if clean_text(record.component_text)
    ]
    provenance_route = "pseudo_component_to_source_row_to_raw_chunk"
    # Some historical extraction folders contain the task propositions but
    # omit the component bridge, or contain bridge rows with blank text.
    # Proposition rows are already non-lossy task pseudo-evidence, so expose
    # them explicitly rather than rendering an empty pseudo card.
    if not component_rows:
        component_rows = [
            {
                "component_id": (
                    f"{record.source_table}:{record.row_id}"
                ),
                "source_table": record.source_table,
                "source_row_id": record.row_id,
                "text": record.text,
            }
            for record in context.corpus.propositions_by_doc.get(doc_id, ())
            if clean_text(record.text)
        ]
        provenance_route = "task_proposition_to_raw_chunk"
    # Evidence cards should expose the components nearest to the target, not
    # an entire document-sized pseudo surface.  The latter made passage
    # localization drift toward unrelated but globally frequent sentences.
    component_rows.sort(
        key=lambda row: (
            -max(
                score_text(query_text, row["text"], context.semantic_resources),
                score_text(
                    matched_proposition_text,
                    row["text"],
                    context.semantic_resources,
                ) if matched_proposition_text else 0.0,
            ),
            row["source_table"],
            row["source_row_id"],
        )
    )
    component_rows = component_rows[:12]
    pseudo_text = clean_text(" ".join(
        row["text"] for row in component_rows if clean_text(row["text"])
    ))
    localized = _localized_chunks(
        query_text,
        matched_proposition_text,
        sentences,
        context.semantic_resources,
    )
    raw_cards = [
        {
            "rank": rank,
            "score": score,
            "chunk_id": item.chunk_id,
            "sentence_index": item.sentence_index,
            "previous": item.left_neighbor,
            "selected": item.sentence,
            "next": item.right_neighbor,
        }
        for rank, (score, item) in enumerate(localized, start=1)
    ]
    return {
        "fragment_id": fragment_id,
        "doc_id": doc_id,
        "evidence_set": evidence_set,
        "pseudo_component_ids": ";".join(
            record.component_id for record in provenance
        ),
        "pseudo_source_row_ids": ";".join(
            record.source_row_id for record in provenance
        ),
        "pseudo_evidence": pseudo_text,
        "matched_proposition_text": matched_proposition_text,
        "matched_proposition_source_table":
            matched_proposition_source_table,
        "matched_proposition_row_id": matched_proposition_row_id,
        "pseudo_component_count": len(component_rows),
        "pseudo_component_cards_json": json.dumps(
            component_rows, ensure_ascii=False
        ),
        "raw_chunk_ids": ";".join(
            item.chunk_id for _score, item in localized
        ),
        "raw_passage_count": len(localized),
        "raw_chunk_cards_json": json.dumps(raw_cards, ensure_ascii=False),
        "previous_raw_chunk": localized[0][1].left_neighbor if localized else "",
        "selected_raw_chunk": localized[0][1].sentence if localized else "",
        "next_raw_chunk": localized[0][1].right_neighbor if localized else "",
        "raw_passage": " [...] ".join(
            clean_text(" ".join(
                part for part in (
                    item.left_neighbor, item.sentence, item.right_neighbor,
                ) if part
            ))
            for _score, item in localized
        ),
        "raw_status": (
            "RAW_PROVENANCE_LINKED"
            if component_rows and pseudo_text and localized
            else "RAW_WITHIN_DOCUMENT_LOCALIZED"
            if localized else "RAW_NOT_AVAILABLE"
        ),
        "provenance_route": (
            provenance_route
            if component_rows and pseudo_text and localized
            else "within_document_localized_fallback"
        ),
    }


class ReconstructionEngine(TaskEngine):
    task_name = "reconstruction"

    def run(self, context: EngineContext) -> EngineResult:
        query_records = context.query.proposition_records()
        query_by_fragment: Dict[str, List[Dict[str, Any]]] = {}
        for record in query_records:
            query_by_fragment.setdefault(record["fragment_id"], []).append(record)
        evidence_rows: List[Dict[str, Any]] = []
        evaluation_rows: List[Dict[str, Any]] = []
        minimal_rows: List[Dict[str, Any]] = []
        extended_rows: List[Dict[str, Any]] = []
        passage_rows: List[Dict[str, Any]] = []
        rejected_passage_rows: List[Dict[str, Any]] = []
        context.control.emit(
            "reconstruction_started",
            target_count=len(context.query.fragments),
        )
        for fragment_index, fragment in context.query.fragments.iterrows():
            fragment_id = clean_text(fragment.get("fragment_id", "")) or (
                f"U{fragment_index + 1}"
            )
            query_text = clean_text(
                fragment.get(
                    "fragment_text_for_extraction",
                    fragment.get("fragment_text", ""),
                )
            )
            fragment_queries = query_by_fragment.get(fragment_id, [])
            full_fragment_query = {
                "id": fragment_id,
                "fragment_id": fragment_id,
                "text": query_text,
                "signature": build_event_signature(query_text),
            }
            if not fragment_queries:
                fragment_queries = [full_fragment_query]
                required_fact_queries = fragment_queries
            elif query_text and all(
                tuple(
                    canonical_key(query.get("signature", {}).get(field, ""))
                    for field in (
                        "actor", "predicate", "object", "location",
                        "assertion_polarity", "semantic_state_variable",
                        "semantic_state_value",
                    )
                )
                != tuple(
                    canonical_key(
                        full_fragment_query["signature"].get(field, "")
                    )
                    for field in (
                        "actor", "predicate", "object", "location",
                        "assertion_polarity", "semantic_state_variable",
                        "semantic_state_value",
                    )
                )
                for query in fragment_queries
            ):
                # OpenAI parsing may shorten or mis-attach a proposition. Keep
                # that structured parse, but also evaluate a deterministic
                # signature of the complete analyst fragment. This is required
                # even when both records retain identical surface text: an LLM
                # can still attach the wrong actor, predicate, or object.
                fragment_queries = [*fragment_queries, full_fragment_query]
                required_fact_queries = [full_fragment_query]
            else:
                required_fact_queries = fragment_queries
            facts: Set[str] = set()
            for query in required_fact_queries:
                facts.update(
                    _fact_set(
                        query["signature"],
                        query["text"],
                        context.semantic_resources,
                    )
                )
            typed_facts = {fact for fact in facts if not fact.startswith("token:")}
            retrieval_scores: Dict[str, float] = {}
            retrieval_routes: Dict[str, set[str]] = defaultdict(set)
            for doc_id, score in rank_documents(
                query_text,
                context.corpus,
                context.config.staged_top_docs,
                context.semantic_resources,
            ):
                retrieval_scores[doc_id] = max(
                    retrieval_scores.get(doc_id, 0.0), float(score)
                )
                retrieval_routes[doc_id].add("FULL_QUERY_FRAGMENT")
            for query in fragment_queries:
                for route_name, route_text in _reconstruction_retrieval_routes(
                    query
                ):
                    for doc_id, score in rank_documents(
                        route_text,
                        context.corpus,
                        context.config.staged_top_docs,
                        context.semantic_resources,
                    ):
                        retrieval_scores[doc_id] = max(
                            retrieval_scores.get(doc_id, 0.0),
                            float(score),
                        )
                        retrieval_routes[doc_id].add(route_name)
            ranked_docs = sorted(
                retrieval_scores.items(),
                key=lambda item: (
                    -item[1],
                    -len(retrieval_routes.get(item[0], set())),
                    item[0],
                ),
            )
            candidates: List[Dict[str, Any]] = []
            records_processed = 0
            total_ranked_docs = len(ranked_docs)
            for doc_index, (doc_id, retrieval_score) in enumerate(
                ranked_docs, start=1
            ):
                document_text = context.corpus.document_text.get(doc_id, "")
                # A concatenated document is a retrieval surface, not one
                # grammatical proposition.  Parsing it as a giant event costs
                # tens of seconds on large corpora and can manufacture a weak
                # actor/predicate from an unrelated early clause.  Structured
                # event evidence is evaluated below from proposition records.
                # Retain only a lightweight document scan for typed values
                # whose weak coverage cannot be established from containment.
                needs_typed_document_signature = any(
                    fact.partition(":")[0] in {
                        "event_time", "packet_count", "total_weight_grams",
                    }
                    for fact in facts
                )
                if needs_typed_document_signature:
                    numeric = normalize_numeric_arguments(document_text)
                    document_signature = {
                        "event_time": numeric.get("date_value", ""),
                        "date_value": numeric.get("date_value", ""),
                        "packet_count": numeric.get("packet_count"),
                        "total_weight_grams": numeric.get("total_weight_grams"),
                    }
                else:
                    document_signature = {}
                document_weak = _covered(
                    facts,
                    document_text,
                    document_signature,
                    context.config.reconstruction_literal_containment_floor,
                    context.semantic_resources,
                )
                aligned_strong: Set[str] = set()
                best_identity = 0.0
                best_states: Dict[str, str] = {}
                best_query_signature: Mapping[str, Any] = {}
                best_candidate_signature: Mapping[str, Any] = document_signature
                best_assertion_pass = False
                best_assertion_failure = ""
                best_record_text = ""
                best_record_source_table = ""
                best_record_row_id = ""
                best_surface_score = 0.0
                best_qualifier_coverage = 1.0
                best_alignment_key: tuple[float, ...] = (-1.0,)
                records = context.corpus.propositions_by_doc.get(doc_id, ())
                records_processed += len(records)
                for record in records:
                    record_covered = _covered(
                        facts,
                        record.text,
                        record.signature,
                        context.config.reconstruction_literal_containment_floor,
                        context.semantic_resources,
                    )
                    for query in fragment_queries:
                        identity, states = event_identity(
                            query["signature"], record.signature,
                            query["text"], record.text,
                            context.semantic_resources,
                        )
                        assertion_pass, assertion_failure = assertion_gate(
                            query["signature"], record.signature
                        )
                        record_critical_coverage = _coverage_ratio(
                            record_covered & typed_facts,
                            typed_facts,
                        )
                        qualifier_coverage = _qualifier_coverage(
                            query_text, record.text,
                            context.semantic_resources,
                        )
                        alignment_key = (
                            float(assertion_pass),
                            float(
                                identity
                                >= context.config.reconstruction_identity_floor
                            ),
                            record_critical_coverage,
                            qualifier_coverage,
                            identity,
                            score_text(
                                query["text"],
                                record.text,
                                context.semantic_resources,
                            ),
                        )
                        if alignment_key > best_alignment_key:
                            best_alignment_key = alignment_key
                            aligned_strong = set(record_covered)
                            best_identity = identity
                            best_states = states
                            best_query_signature = query["signature"]
                            best_candidate_signature = record.signature
                            best_assertion_pass = assertion_pass
                            best_assertion_failure = assertion_failure
                            best_record_text = record.text
                            best_record_source_table = record.source_table
                            best_record_row_id = record.row_id
                            best_surface_score = alignment_key[-1]
                            best_qualifier_coverage = qualifier_coverage
                if not records:
                    for query in fragment_queries:
                        identity, states = event_identity(
                            query["signature"], document_signature,
                            query["text"], document_text,
                            context.semantic_resources,
                        )
                        assertion_pass, assertion_failure = assertion_gate(
                            query["signature"], document_signature
                        )
                        if identity > best_identity:
                            best_identity = identity
                            best_states = states
                            best_query_signature = query["signature"]
                            best_assertion_pass = assertion_pass
                            best_assertion_failure = assertion_failure
                            aligned_strong = set(document_weak)
                            best_record_text = document_text
                # Document-wide containment is retained only as diagnostic
                # weak evidence. It cannot assemble an event by finding its
                # actor, predicate, and object in unrelated propositions.
                weak = document_weak - aligned_strong
                covered = set(aligned_strong)
                critical_strong = aligned_strong & typed_facts
                critical_coverage = _coverage_ratio(
                    critical_strong,
                    typed_facts,
                )
                weighted_coverage = _coverage_ratio(
                    aligned_strong,
                    facts,
                )
                assertion_pass = best_assertion_pass
                assertion_failure = best_assertion_failure
                event_pass = (
                    best_identity >= context.config.reconstruction_identity_floor
                )
                critical_mismatches = {
                    field
                    for field in (
                        "actor_state", "predicate_state", "object_state",
                        "location_state", "date_state", "packet_count_state",
                        "total_weight_grams_state", "vehicle_state",
                    )
                    if best_states.get(field) == "MISMATCH"
                }
                matched_core_fields = sum(
                    best_states.get(field) == "MATCH"
                    for field in (
                        "actor_state", "predicate_state", "object_state"
                    )
                )
                semantic_reparse_rescue = False
                if (
                    assertion_pass
                    and event_pass
                    and critical_coverage
                    >= context.config.reconstruction_critical_floor
                    and critical_mismatches
                    and critical_mismatches
                    <= {"actor_state", "predicate_state", "object_state"}
                    and best_surface_score >= 0.60
                    and best_record_text
                ):
                    candidate_reparse = build_event_signature(best_record_text)
                    for query in fragment_queries:
                        query_reparse = build_event_signature(query["text"])
                        reparse_identity, reparse_states = event_identity(
                            query_reparse,
                            candidate_reparse,
                            query["text"],
                            best_record_text,
                            context.semantic_resources,
                        )
                        reparse_mismatches = {
                            field for field in (
                                "actor_state", "predicate_state", "object_state",
                                "location_state", "date_state",
                                "packet_count_state",
                                "total_weight_grams_state", "vehicle_state",
                            )
                            if reparse_states.get(field) == "MISMATCH"
                        }
                        reparse_assertion_pass, _ = assertion_gate(
                            query_reparse, candidate_reparse
                        )
                        if (
                            reparse_assertion_pass
                            and reparse_identity >= 0.72
                            and not reparse_mismatches
                        ):
                            semantic_reparse_rescue = True
                            best_identity = max(best_identity, reparse_identity)
                            best_states = reparse_states
                            critical_mismatches = set()
                            matched_core_fields = sum(
                                reparse_states.get(field) == "MATCH"
                                for field in (
                                    "actor_state", "predicate_state",
                                    "object_state",
                                )
                            )
                            break
                semantic_one_field_rescue = bool(
                    assertion_pass
                    and event_pass
                    and critical_coverage
                    >= context.config.reconstruction_critical_floor
                    and len(critical_mismatches) == 1
                    and critical_mismatches
                    <= {"actor_state", "predicate_state", "object_state"}
                    and matched_core_fields >= 2
                    and best_surface_score >= 0.62
                )
                independent_role_anchor = any(
                    best_states.get(field) == "MATCH"
                    for field in (
                        "location_state", "date_state", "packet_count_state",
                        "total_weight_grams_state", "vehicle_state",
                    )
                )
                semantic_role_shift_rescue = bool(
                    assertion_pass
                    and event_pass
                    and critical_coverage
                    >= context.config.reconstruction_critical_floor
                    and critical_mismatches == {"actor_state", "object_state"}
                    and best_states.get("predicate_state") == "MATCH"
                    and independent_role_anchor
                    and best_surface_score >= 0.72
                )
                untyped_semantic_rescue = bool(
                    not typed_facts
                    and assertion_pass
                    and event_pass
                    and weighted_coverage >= 0.66
                    and best_identity >= 0.80
                    and best_surface_score >= 0.68
                )
                single_eligible = bool(
                    assertion_pass
                    and event_pass
                    and (
                        not critical_mismatches
                        or semantic_reparse_rescue
                        or semantic_one_field_rescue
                        or semantic_role_shift_rescue
                    )
                    and (
                        critical_coverage
                        >= context.config.reconstruction_critical_floor
                        or untyped_semantic_rescue
                    )
                )
                component_eligible = bool(
                    assertion_pass
                    and event_pass
                    and (
                        not critical_mismatches
                        or semantic_reparse_rescue
                        or semantic_one_field_rescue
                        or semantic_role_shift_rescue
                    )
                )
                candidates.append({
                    "fragment_id": fragment_id,
                    "doc_id": doc_id,
                    "retrieval_score": retrieval_score,
                    "event_identity_score": best_identity,
                    "critical_fact_coverage": critical_coverage,
                    "weighted_fact_coverage": weighted_coverage,
                    "assertion_gate_pass": assertion_pass,
                    "event_identity_gate_pass": event_pass,
                    "semantic_one_field_rescue_pass":
                        semantic_one_field_rescue,
                    "semantic_reparse_rescue_pass": semantic_reparse_rescue,
                    "untyped_semantic_rescue_pass": untyped_semantic_rescue,
                    "semantic_role_shift_rescue_pass":
                        semantic_role_shift_rescue,
                    "matched_surface_score": best_surface_score,
                    "qualifier_coverage_score": best_qualifier_coverage,
                    "candidate_gate_pass": component_eligible,
                    "collective_component_eligible": component_eligible,
                    "single_document_eligible": single_eligible,
                    "eligible": single_eligible,
                    "final_score": (
                        0.15 * retrieval_score
                        + 0.25 * critical_coverage
                        + 0.20 * best_identity
                        + 0.25 * best_surface_score
                        + 0.15 * best_qualifier_coverage
                    ),
                    "strong_facts_json": json.dumps(sorted(aligned_strong)),
                    "aligned_facts_json": json.dumps(sorted(aligned_strong)),
                    "weak_facts_json": json.dumps(sorted(weak)),
                    "covered_facts_json": json.dumps(sorted(covered)),
                    "missing_critical_facts_json": json.dumps(
                        sorted(typed_facts - covered)
                    ),
                    "event_gate_states_json": json.dumps(best_states, sort_keys=True),
                    "matched_proposition_text": best_record_text,
                    "matched_proposition_source_table":
                        best_record_source_table,
                    "matched_proposition_row_id": best_record_row_id,
                    "failed_gates": ";".join(
                        value for value in (
                            "" if assertion_pass else assertion_failure,
                            "" if event_pass else "EVENT_IDENTITY",
                            ""
                            if not critical_mismatches
                            or semantic_reparse_rescue
                            or semantic_one_field_rescue
                            or semantic_role_shift_rescue
                            else "CRITICAL_ARGUMENT_MISMATCH",
                            ""
                            if critical_coverage
                            >= context.config.reconstruction_critical_floor
                            or untyped_semantic_rescue
                            else "CRITICAL_COVERAGE",
                        )
                        if value
                    ),
                })
                if (
                    doc_index == total_ranked_docs
                    or doc_index % 25 == 0
                ):
                    context.control.emit(
                        "reconstruction_candidate_progress",
                        target_id=fragment_id,
                        docs_processed=doc_index,
                        total_docs=total_ranked_docs,
                        records_processed=records_processed,
                        current_doc_id=doc_id,
                    )
                    context.control.check(len(candidates))
            context.control.check(len(candidates))
            candidates.sort(
                key=lambda row: (
                    -float(row["single_document_eligible"]),
                    -float(row["final_score"]),
                    row["doc_id"],
                )
            )
            evidence_rows.extend(candidates)

            selected: List[Dict[str, Any]] = []
            single = next(
                (row for row in candidates if row["single_document_eligible"]),
                None,
            )
            if single:
                selected = [single]
                reconstruction_status = "RECONSTRUCTABLE_SINGLE_DOCUMENT"
            else:
                covered_collective: Set[str] = set()
                for candidate in candidates:
                    if not candidate["candidate_gate_pass"]:
                        continue
                    candidate_facts = set(
                        json.loads(candidate["covered_facts_json"])
                    )
                    new_critical = (
                        candidate_facts - covered_collective
                    ) & typed_facts
                    if not new_critical:
                        continue
                    selected.append(candidate)
                    covered_collective.update(candidate_facts)
                    collective_coverage = _coverage_ratio(
                        covered_collective & typed_facts,
                        typed_facts,
                    )
                    if (
                        collective_coverage
                        >= context.config.reconstruction_critical_floor
                        or len(selected) >= context.config.max_reconstruction_docs
                    ):
                        break
                collective_coverage = _coverage_ratio(
                    set().union(*(
                        set(json.loads(row["covered_facts_json"]))
                        for row in selected
                    )) & typed_facts,
                    typed_facts,
                ) if selected else 0.0
                reconstruction_status = (
                    "RECONSTRUCTABLE_COLLECTIVE"
                    if selected
                    and collective_coverage
                    >= context.config.reconstruction_critical_floor
                    else "NOT_RECONSTRUCTABLE_FROM_ELIGIBLE_EVIDENCE"
                )
                if not reconstruction_status.startswith("RECONSTRUCTABLE"):
                    selected = []

            cumulative: Set[str] = set()
            selected_doc_ids = {row["doc_id"] for row in selected}
            for order, candidate in enumerate(selected, start=1):
                candidate_facts = set(json.loads(candidate["covered_facts_json"]))
                new_facts = candidate_facts - cumulative
                cumulative.update(candidate_facts)
                minimal_rows.append({
                    "selection_order": order,
                    "fragment_id": fragment_id,
                    "doc_id": candidate["doc_id"],
                    "marginal_weighted_gain": _coverage_ratio(
                        new_facts, facts
                    ),
                    "new_critical_fact_count": len(
                        _covered_dimensions(new_facts & typed_facts)
                    ),
                    "cumulative_weighted_coverage": _coverage_ratio(
                        cumulative, facts
                    ),
                    "single_or_collective": (
                        "SINGLE" if len(selected) == 1 else "COLLECTIVE"
                    ),
                    "set_membership": "MINIMAL_EXPLANATORY_SET",
                    "selection_reason": "adds_required_critical_fact",
                })
                passage_rows.append(_passage_row(
                    fragment_id, candidate["doc_id"], query_text, context,
                    "SELECTED_RECONSTRUCTION",
                    matched_proposition_text=clean_text(
                        candidate.get("matched_proposition_text", "")
                    ),
                    matched_proposition_source_table=clean_text(
                        candidate.get(
                            "matched_proposition_source_table", ""
                        )
                    ),
                    matched_proposition_row_id=clean_text(
                        candidate.get("matched_proposition_row_id", "")
                    ),
                ))
            for candidate in candidates:
                if (
                    candidate["candidate_gate_pass"]
                    and candidate["doc_id"] not in selected_doc_ids
                ):
                    extended_rows.append({
                        "fragment_id": fragment_id,
                        "doc_id": candidate["doc_id"],
                        "final_score": candidate["final_score"],
                        "critical_fact_coverage": candidate["critical_fact_coverage"],
                        "weighted_fact_coverage": candidate["weighted_fact_coverage"],
                        "set_membership": "EXTENDED_SUPPORT",
                        "selection_reason": "eligible_but_not_required_by_minimal_set",
                    })
            top = candidates[0] if candidates else {}
            comparable = [
                candidate for candidate in candidates[1:]
                if bool(candidate.get("candidate_gate_pass"))
                == bool(top.get("candidate_gate_pass"))
                and bool(candidate.get("single_document_eligible"))
                == bool(top.get("single_document_eligible"))
            ]
            competitor = comparable[0] if comparable else {}
            nearest = nearest_alternative(
                candidates,
                top,
                identity_fields=("doc_id",),
            )
            nearest_rejected = next(
                (
                    candidate for candidate in candidates
                    if candidate["doc_id"] not in selected_doc_ids
                ),
                None,
            )
            if nearest_rejected:
                rejected = _passage_row(
                    fragment_id, nearest_rejected["doc_id"], query_text,
                    context, "NEAREST_REJECTED_EVIDENCE",
                    matched_proposition_text=clean_text(
                        nearest_rejected.get(
                            "matched_proposition_text", ""
                        )
                    ),
                    matched_proposition_source_table=clean_text(
                        nearest_rejected.get(
                            "matched_proposition_source_table", ""
                        )
                    ),
                    matched_proposition_row_id=clean_text(
                        nearest_rejected.get(
                            "matched_proposition_row_id", ""
                        )
                    ),
                )
                rejected.update({
                    "final_score": nearest_rejected["final_score"],
                    "failed_gates": nearest_rejected["failed_gates"],
                    "event_identity_score":
                        nearest_rejected["event_identity_score"],
                    "critical_fact_coverage":
                        nearest_rejected["critical_fact_coverage"],
                })
                rejected_passage_rows.append(rejected)
            resolved = reconstruction_status.startswith("RECONSTRUCTABLE")
            selected_source = selected[0]["doc_id"] if selected else ""
            selected_text = " | ".join(
                clean_text(row.get("matched_proposition_text", ""))
                for row in selected
                if clean_text(row.get("matched_proposition_text", ""))
            )
            evaluation_rows.append(task_row(
                "reconstruction", fragment_id, reconstruction_status,
                localization_status=(
                    "SOURCE_LOCALIZED" if top else "NO_CANDIDATE"
                ),
                raw_status=(
                    "RAW_PROVENANCE_LINKED" if resolved else "RAW_NOT_ASSESSED"
                ),
                reconstructability=(
                    "RECONSTRUCTABLE" if resolved else "NOT_RECONSTRUCTABLE"
                ),
                selected_text=selected_text,
                selected_source=selected_source,
                top_candidate=clean_text(top.get("doc_id", "")),
                best_score=top.get("final_score", ""),
                runner_up=competitor.get("final_score"),
                margin=(
                    float(top.get("final_score", 0.0))
                    - float(competitor.get("final_score", 0.0))
                    if top and competitor else None
                ),
                decision_competitor_doc_id=clean_text(
                    competitor.get("doc_id", "")
                ),
                nearest_alternative_score=nearest.get("final_score"),
                nearest_alternative_doc_id=clean_text(
                    nearest.get("doc_id", "")
                ),
                nearest_alternative_eligible=(
                    bool(nearest.get("candidate_gate_pass"))
                    if nearest else None
                ),
                nearest_alternative_failed_gates=clean_text(
                    nearest.get("failed_gates", "")
                ),
                reason=(
                    "critical_fact_event_and_assertion_gates_passed"
                    if resolved
                    else "no_single_or_collective_eligible_evidence_set"
                ),
                unresolved="" if resolved else reconstruction_status,
                failed="" if resolved else clean_text(
                    top.get(
                        "failed_gates",
                        "CRITICAL_COVERAGE;EVENT_IDENTITY;ASSERTION",
                    )
                ),
            ))
            context.control.emit(
                "reconstruction_target_complete",
                target_id=fragment_id,
                candidate_count=len(candidates),
                selected_document_count=len(selected),
                status=reconstruction_status,
            )
        return EngineResult(
            task=self.task_name,
            evaluation=pd.DataFrame(evaluation_rows),
            artifacts={
                "fragment_document_evidence_v048.csv": pd.DataFrame(evidence_rows),
                "minimal_reconstruction_set_v048.csv": pd.DataFrame(minimal_rows),
                "extended_supporting_set_v048.csv": pd.DataFrame(extended_rows),
                "fragment_source_passages_v048.csv":
                    pd.DataFrame(passage_rows, columns=PASSAGE_COLUMNS),
                "nearest_rejected_evidence_v048.csv":
                    pd.DataFrame(rejected_passage_rows),
            },
        )
