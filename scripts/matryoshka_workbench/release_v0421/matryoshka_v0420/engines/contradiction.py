"""Contradiction, temporal-update, and uncertainty evidence engine.

The v0.4.20 engine uses independent retrieval routes, candidate-local critical
anchors, an explicit same-event gate for uncertainty, and sentence-level raw
propositions from retrieved documents. It produces one coherent decision per
query/document plus a query-level grouped assessment; it does not adjudicate
historical truth.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from ..contracts import task_row
from ..corpus import PropositionRecord
from ..decision import event_identity
from ..decision_profile import DEFAULT_ENGINE_THRESHOLDS
from ..retrieval import containment, rank_documents, score_text
from ..semantic_resources import (
    SemanticResources,
    load_semantic_resources,
    shared_concepts,
)
from ..semantics import (
    PREDICATE_PHRASE_LEMMAS,
    PREDICATE_LEMMAS,
    build_event_signature,
    canonical_key,
    clean_text,
    compare_argument,
    compare_dates,
    compare_numeric,
    infer_assertion_polarity,
    infer_epistemic_state,
    infer_simple_clause_arguments,
    resolve_epistemic_scope,
    semantic_tokens,
)
from .base import EngineContext, EngineResult, TaskEngine


_EVENT_IDENTITY_DIMENSIONS = (
    "actor", "predicate", "object", "location_identity", "event_time",
    "packet_count", "total_weight_grams", "vehicle_identity",
)
_NUMERIC_DIMENSIONS = frozenset({
    "packet_count", "total_weight_grams",
})
_DISCRIMINATIVE_EVENT_DIMENSIONS = frozenset(
    set(_EVENT_IDENTITY_DIMENSIONS) - {"predicate", "object"}
)

_RAW_CONFLICT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "his", "in", "into", "is", "it", "its",
    "may", "might", "of", "on", "or", "she", "that", "the", "their",
    "there", "they", "this", "those", "to", "was", "were", "will",
    "with", "would",
})
_RAW_CONFLICT_TOKEN_EQUIVALENTS = {
    "classified": "secret",
    "confidential": "secret",
    "discontinued": "stop",
    "discontinue": "stop",
    "first": "initial",
    "ignore": "overlook",
    "ignored": "overlook",
    "overlooked": "overlook",
    "respected": "respect",
    "spanish": "spain",
    "spain's": "spain",
    "venerable": "respect",
    "scheduled": "plan",
    "schedule": "plan",
    "undisclosed": "secret",
}
_RAW_UNCERTAINTY_NEGATION_PHRASES = (
    "could not confirm",
    "cannot confirm",
    "not confirmed",
    "not known",
    "no evidence",
    "no information",
    "no news",
)

_RAW_MONTH_NUMBERS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8,
    "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_RAW_MONTH_DATE_RE = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?|tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
    r"(?:(?P<day>\d{1,2})(?:,\s*|\s+)(?P<year>\d{4})|"
    r"(?P<year_only>\d{4})|(?P<day_only>\d{1,2}))\b",
    re.IGNORECASE,
)
_RAW_DAY_MONTH_DATE_RE = re.compile(
    r"\b(?P<day>\d{1,2})\s+"
    r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?|tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
    r"(?:,?\s+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_RAW_STANDALONE_MONTH_RE = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?|tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\b",
    re.IGNORECASE,
)
_RAW_SLASH_DATE_RE = re.compile(
    r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{2,4})\b"
)
_RAW_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_RAW_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![A-Za-z0-9])"
)


def _normalize_concern_relation_signature(
    text: Any,
    signature: Mapping[str, Any],
) -> Dict[str, Any]:
    """Normalize active and inverted concern frames to one event shape."""
    source = clean_text(text)
    normalized = dict(signature or {})
    if not source:
        return normalized
    active = re.match(
        r"^(?P<actor>.+?)\s+(?:is|are|was|were)\s+(?:not\s+)?"
        r"(?:[A-Za-z]+ly\s+){0,2}concerned\s+(?:about|over|with)\s+"
        r"(?P<object>.+?)[.!?]?$",
        source,
        flags=re.I,
    )
    inverted = re.match(
        r"^(?P<object>.+?)\s+(?:is|are|was|were)\s+(?:not\s+)?"
        r"(?:of\s+)?(?:[A-Za-z]+\s+){0,2}concern\s+(?:to|for)\s+"
        r"(?P<actor>.+?)[.!?]?$",
        source,
        flags=re.I,
    )
    match = active or inverted
    if not match:
        return normalized
    actor = re.sub(
        r"^(?:the\s+)", "", clean_text(match.group("actor")), flags=re.I
    )
    obj = re.sub(
        r"^(?:the\s+)", "", clean_text(match.group("object")), flags=re.I
    )
    if not actor or not obj:
        return normalized
    normalized.update({
        "actor": actor,
        "agent": actor,
        "embedded_actor": actor,
        "predicate": "concern",
        "predicate_surface": "concern",
        "object": obj,
        "object_surface": obj,
        "patient_or_theme": obj,
        "role_normalization_status": "CONCERN_RELATION_NORMALIZED",
    })
    return normalized


def _explicit_structure_is_authoritative(
    text: Any,
    signature: Mapping[str, Any],
) -> bool:
    """Return whether deterministic parsing found an explicit relation frame.

    Parser alternatives remain useful for ambiguous prose. They must not
    override an unambiguous surface construction such as ``agent of`` or
    ``affiliated with`` by reducing it to a generic copular predicate.
    """
    if clean_text(signature.get("role_normalization_status", "")) == (
        "CONCERN_RELATION_NORMALIZED"
    ):
        return True
    source = clean_text(text)
    return any(pattern.search(source) for pattern, _lemma in PREDICATE_PHRASE_LEMMAS)


def _dimension_state(
    dimension: str,
    query_value: Any,
    candidate_value: Any,
    resources: SemanticResources | None = None,
) -> str:
    if dimension in _NUMERIC_DIMENSIONS:
        return compare_numeric(query_value, candidate_value)
    if dimension == "event_time":
        return compare_dates(query_value, candidate_value)
    state = compare_argument(query_value, candidate_value)
    if state != "MISMATCH":
        return state
    if resources is not None:
        left_aliases = resources.concepts(query_value, "alias")
        right_aliases = resources.concepts(candidate_value, "alias")
        shared_aliases = left_aliases & right_aliases
        if shared_aliases:
            if dimension != "object" or _alias_dominates_argument_pair(
                query_value,
                candidate_value,
                shared_aliases,
                resources,
            ):
                return "MATCH"
        if (
            dimension in {"predicate", "object"}
            and shared_concepts(
                resources, query_value, candidate_value, "event"
            )
        ):
            return "MATCH"
        if (
            dimension == "actor"
            and left_aliases
            and right_aliases
        ):
            # Two different canonical IDs in an explicitly configured
            # entity resource are affirmative evidence of distinct actors.
            return "MISMATCH"
    if dimension == "actor":
        # Open-world surface inequality is not proof of distinct identity:
        # a phrase may name an organization, a unit, a representative, or a
        # generic class at different levels of granularity.
        return "AMBIGUOUS_IDENTITY"
    return state


def _alias_dominates_argument_pair(
    query_value: Any,
    candidate_value: Any,
    shared_aliases: Sequence[str],
    resources: SemanticResources,
) -> bool:
    """Require a shared entity alias to explain both complete arguments.

    Alias overlap is sufficient for entity-valued objects such as ``USSR`` and
    ``Soviet Union``. It is not sufficient when the shared entity is merely a
    participant inside two different event clauses.
    """
    stopwords = {
        "a", "an", "and", "as", "at", "be", "been", "being", "by",
        "for", "from", "in", "is", "not", "of", "on", "the", "to",
        "was", "were", "with",
    }

    def content_tokens(value: Any) -> set[str]:
        return {
            PREDICATE_LEMMAS.get(token, token)
            for token in semantic_tokens(value, preserve_negation=False)
            if token not in stopwords
        }

    def covered(value: Any) -> bool:
        value_tokens = content_tokens(value)
        if not value_tokens:
            return False
        for concept in shared_aliases:
            for phrase in resources.alias_groups.get(concept, ()):
                phrase_tokens = content_tokens(phrase)
                if not phrase_tokens:
                    continue
                overlap = value_tokens & phrase_tokens
                if overlap and len(overlap) / len(value_tokens) >= 0.80:
                    return True
        return False

    return covered(query_value) and covered(candidate_value)


def dimension_exclusion_event_binding(
    query_hypotheses: Sequence[Mapping[str, Any]],
    candidate_hypotheses: Sequence[Mapping[str, Any]],
    *,
    contradiction_dimension: str,
    identity_floor: float,
    minimum_margin: float,
    additional_excluded_dimensions: Sequence[str] = (),
    resources: SemanticResources | None = None,
) -> Dict[str, Any]:
    """Bind an event without using the attribute currently under dispute.

    Every query/candidate hypothesis pair is scored on the remaining available
    typed dimensions.  A disputed dimension can become contradiction evidence
    only when the best event hypothesis clears both an identity floor and the
    margin over the nearest competing hypothesis. At least two remaining typed
    dimensions must independently match; merely having two dimensions present
    while only one matches is insufficient. No named dimension or fixed anchor
    combination is required.
    """
    excluded = clean_text(contradiction_dimension)
    excluded_dimensions = {
        excluded,
        *(
            clean_text(dimension)
            for dimension in additional_excluded_dimensions
            if clean_text(dimension)
        ),
    }
    dimensions = [
        dimension for dimension in _EVENT_IDENTITY_DIMENSIONS
        if dimension not in excluded_dimensions
    ]
    scored: List[Dict[str, Any]] = []
    for query_index, query in enumerate(query_hypotheses):
        for candidate_index, candidate in enumerate(candidate_hypotheses):
            states: Dict[str, str] = {}
            used: List[str] = []
            match_points = 0.0
            for dimension in dimensions:
                query_value = query.get(dimension)
                if query_value is None or not clean_text(query_value):
                    continue
                state = _dimension_state(
                    dimension,
                    query_value,
                    candidate.get(dimension),
                    resources,
                )
                states[dimension] = state
                used.append(dimension)
                if state == "MATCH":
                    match_points += 1.0
                elif state in {"MISSING_IN_CANDIDATE", "UNPARSEABLE"}:
                    match_points += 0.0
            identity_score = (
                match_points / len(used) if used else 0.0
            )
            try:
                query_confidence = float(
                    query.get("hypothesis_confidence", 1.0)
                )
            except (TypeError, ValueError):
                query_confidence = 1.0
            try:
                candidate_confidence = float(
                    candidate.get("hypothesis_confidence", 1.0)
                )
            except (TypeError, ValueError):
                candidate_confidence = 1.0
            joint_confidence = (
                max(0.0, min(1.0, query_confidence))
                * max(0.0, min(1.0, candidate_confidence))
            )
            # Confidence ranks otherwise comparable interpretations; it does
            # not veto an event identity established by independent typed
            # dimensions. A low-confidence deterministic fallback is still a
            # real user-query hypothesis, not weak evidence about history.
            matched_dimension_count = sum(
                state == "MATCH" for state in states.values()
            )
            rank_score = (
                identity_score
                + 0.01 * matched_dimension_count
                + 0.001 * joint_confidence
            )
            scored.append({
                "query_hypothesis_index": query_index,
                "candidate_hypothesis_index": candidate_index,
                "identity_score": identity_score,
                "rank_score": rank_score,
                "joint_hypothesis_confidence": joint_confidence,
                "raw_identity_score": identity_score,
                "states": states,
                "identity_dimensions_used": used,
                "query": query,
                "candidate": candidate,
            })
    scored.sort(
        key=lambda item: (
            -item["rank_score"],
            item["query_hypothesis_index"],
            item["candidate_hypothesis_index"],
        )
    )
    if not scored:
        return {
            "binding_state": "AMBIGUOUS_EVENT",
            "comparison_state": "NOT_APPLICABLE",
            "identity_score": 0.0,
            "runner_up_score": 0.0,
            "margin": 0.0,
            "identity_dimensions_used": [],
            "reason": "no_event_hypotheses",
        }
    winner = scored[0]
    matched_dimensions = [
        dimension
        for dimension, state in winner["states"].items()
        if state == "MATCH"
    ]
    # All binding properties must be derived from the selected winning
    # hypothesis.  The loop-local count belongs to the last evaluated pair,
    # which need not be the winner when Parser supplies alternatives.
    matched_dimension_count = len(matched_dimensions)
    has_discriminative_match = bool(
        set(matched_dimensions) & _DISCRIMINATIVE_EVENT_DIMENSIONS
    )
    # A contradiction may exclude only its disputed dimension.  Any other
    # discriminative identity field explicitly supplied by the query must be
    # observable in the candidate before the candidate can certify that it is
    # the same event.  Missing evidence is not evidence of contradiction.
    missing_remaining_discriminators = sorted(
        dimension
        for dimension, state in winner["states"].items()
        if dimension in _DISCRIMINATIVE_EVENT_DIMENSIONS
        and state in {"MISSING_IN_CANDIDATE", "UNPARSEABLE"}
    )
    # Alternative typed interpretations in one query hypothesis group are
    # not independent historical events. Only another candidate event
    # interpretation is a binding competitor.
    competitors = [
        item for item in scored[1:]
        if item["candidate_hypothesis_index"]
        != winner["candidate_hypothesis_index"]
    ]
    runner_up_score = (
        float(competitors[0]["rank_score"]) if competitors else 0.0
    )
    margin = float(winner["rank_score"]) - runner_up_score
    uniquely_bound = (
        float(winner["identity_score"]) >= float(identity_floor)
        and (
            not competitors
            or margin >= float(minimum_margin)
        )
        and len(winner["identity_dimensions_used"]) >= 2
        and matched_dimension_count >= 2
        and has_discriminative_match
        and not missing_remaining_discriminators
    )
    comparison = _dimension_state(
        excluded,
        winner["query"].get(excluded),
        winner["candidate"].get(excluded),
        resources,
    )
    return {
        "binding_state": (
            "UNIQUELY_BOUND" if uniquely_bound else "AMBIGUOUS_EVENT"
        ),
        "comparison_state": comparison,
        "contradiction_dimension": excluded,
        "identity_score": float(winner["identity_score"]),
        "runner_up_score": runner_up_score,
        "margin": margin,
        "identity_floor": float(identity_floor),
        "minimum_margin": float(minimum_margin),
        "identity_dimensions_used":
            list(winner["identity_dimensions_used"]),
        "matched_identity_dimensions": matched_dimensions,
        "has_discriminative_identity_match": has_discriminative_match,
        "missing_remaining_discriminators":
            missing_remaining_discriminators,
        "excluded_dimensions": sorted(excluded_dimensions),
        "joint_hypothesis_confidence":
            float(winner["joint_hypothesis_confidence"]),
        "dimension_states": dict(winner["states"]),
        "winning_query_hypothesis_index":
            winner["query_hypothesis_index"],
        "winning_candidate_hypothesis_index":
            winner["candidate_hypothesis_index"],
        "hypothesis_pair_count": len(scored),
        "reason": (
            "event_bound_after_excluding_disputed_dimension"
            if uniquely_bound
            else (
                "query_discriminator_missing_or_unparseable_in_candidate"
                if missing_remaining_discriminators
                else "remaining_evidence_does_not_uniquely_bind_event"
            )
        ),
    }


def _dimension_exclusion_conflict(
    query_hypotheses: Sequence[Mapping[str, Any]],
    candidate_hypotheses: Sequence[Mapping[str, Any]],
    *,
    identity_floor: float,
    minimum_margin: float,
    resources: SemanticResources | None = None,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Return a conflict only when one excluded dimension binds uniquely."""
    audits: List[Dict[str, Any]] = []
    comparison_states: Dict[str, List[str]] = defaultdict(list)
    for query in query_hypotheses:
        for candidate in candidate_hypotheses:
            for dimension in _EVENT_IDENTITY_DIMENSIONS:
                query_value = query.get(dimension)
                if query_value is None or not clean_text(query_value):
                    continue
                comparison_states[dimension].append(
                    _dimension_state(
                        dimension,
                        query_value,
                        candidate.get(dimension),
                        resources,
                    )
                )
    disputed_dimensions = {
        dimension
        for dimension, states in comparison_states.items()
        if "MISMATCH" in states and "MATCH" not in states
    }
    for dimension in _EVENT_IDENTITY_DIMENSIONS:
        if not any(
            hypothesis.get(dimension) is not None
            and clean_text(hypothesis.get(dimension))
            for hypothesis in query_hypotheses
        ):
            continue
        result = dimension_exclusion_event_binding(
            query_hypotheses,
            candidate_hypotheses,
            contradiction_dimension=dimension,
            identity_floor=identity_floor,
            minimum_margin=minimum_margin,
            additional_excluded_dimensions=sorted(
                disputed_dimensions - {dimension}
            ),
            resources=resources,
        )
        audits.append(result)
    bound_mismatches = [
        result for result in audits
        if result["binding_state"] == "UNIQUELY_BOUND"
        and result["comparison_state"] == "MISMATCH"
    ]
    bound_mismatches.sort(
        key=lambda result: (
            -float(result["identity_score"]),
            -float(result["margin"]),
            result["contradiction_dimension"],
        )
    )
    if not bound_mismatches:
        return "", [], {
            "contract": "dimension_exclusion_event_binding",
            "state": (
                "AMBIGUOUS_OR_NO_CONFLICT"
                if any(
                    result["comparison_state"] == "MISMATCH"
                    for result in audits
                )
                else "NO_MISMATCH"
            ),
            "dimensions": audits,
        }
    fields = [
        result["contradiction_dimension"]
        for result in bound_mismatches
    ]
    if any(field in _NUMERIC_DIMENSIONS for field in fields):
        conflict_class = "DIRECT_NUMERICAL_CONTRADICTION"
    elif "location_identity" in fields:
        conflict_class = "LOCATION_CONTRADICTION"
    else:
        conflict_class = "ATTRIBUTE_CONTRADICTION"
    return conflict_class, fields, {
        "contract": "dimension_exclusion_event_binding",
        "state": "UNIQUELY_BOUND_CONFLICT",
        "dimensions": audits,
        "conflict_dimensions": fields,
    }


def _opposition_precedes_argument_conflict(
    *,
    explicit_conflict: str,
    conflict_fields: Sequence[str],
    query_polarity: str,
    candidate_polarity: str,
    critical_anchor_gate_pass: bool,
) -> bool:
    """Prefer event-level polarity when only actor granularity differs.

    A source may identify a delegation while another identifies its named
    representative. That actor granularity difference must not replace an
    otherwise fully bound positive/negative event opposition. Numeric,
    temporal, location, vehicle, and object conflicts remain more specific
    and retain precedence.
    """
    return bool(
        explicit_conflict == "ATTRIBUTE_CONTRADICTION"
        and set(conflict_fields) == {"actor"}
        and clean_text(query_polarity)
        and clean_text(candidate_polarity)
        and clean_text(query_polarity) != clean_text(candidate_polarity)
        and critical_anchor_gate_pass
    )


def _evidence_role_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a stable, corpus-independent summary of one evidence role."""
    if not row:
        return {}
    return {
        "class": clean_text(row.get("primary_class", "")),
        "source_doc_id": clean_text(row.get("corpus_doc_id", "")),
        "score": row.get("final_score", ""),
        "raw_certification_status": clean_text(
            row.get("raw_validation_status", "RAW_NOT_ASSESSED")
        ),
        "decision_reason": clean_text(row.get("decision_reason", "")),
    }


def _primary_evidence(
    *,
    side_a: Mapping[str, Any],
    change: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    qualification: Mapping[str, Any],
    related_different: Mapping[str, Any],
    background: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Select the principal finding without collapsing complementary roles.

    An explicit contradiction, opposition, or update is the principal finding
    even when direct support for the query is absent.  Other evidence classes
    remain independently exported and are never forced into a single
    cross-class winner/runner-up competition.
    """
    return (
        change
        or side_a
        or uncertainty
        or qualification
        or related_different
        or background
        or {}
    )


def _evidence_role_contract(
    *,
    pattern: str,
    side_a: Mapping[str, Any],
    change: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    qualification: Mapping[str, Any],
    related_different: Mapping[str, Any],
    background: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the query-level contradiction contract from independent roles."""
    primary = _primary_evidence(
        side_a=side_a,
        change=change,
        uncertainty=uncertainty,
        qualification=qualification,
        related_different=related_different,
        background=background,
    )
    evidence_roles = {
        "side_a_support": _evidence_role_payload(side_a),
        "contradiction_or_update": _evidence_role_payload(change),
        "uncertainty": _evidence_role_payload(uncertainty),
        "qualification_or_scope": _evidence_role_payload(qualification),
        "related_different_event":
            _evidence_role_payload(related_different),
        "background_only": _evidence_role_payload(background),
    }
    has_evidence_finding = any(bool(row) for row in (
        side_a, change, uncertainty, qualification, related_different,
    ))
    return {
        "primary_finding": clean_text(
            primary.get("primary_class", "")
        ),
        "primary_finding_source": clean_text(
            primary.get("corpus_doc_id", "")
        ),
        "primary_finding_raw_status": clean_text(
            primary.get("raw_validation_status", "RAW_NOT_ASSESSED")
        ),
        "primary_finding_score": primary.get("final_score", ""),
        "evidence_pattern": pattern,
        "side_a_source": clean_text(side_a.get("corpus_doc_id", "")),
        "contradiction_or_update_source": clean_text(
            change.get("corpus_doc_id", "")
        ),
        "uncertainty_source": clean_text(
            uncertainty.get("corpus_doc_id", "")
        ),
        "qualification_or_scope_source": clean_text(
            qualification.get("corpus_doc_id", "")
        ),
        "related_different_event_source": clean_text(
            related_different.get("corpus_doc_id", "")
        ),
        "background_only_source": clean_text(
            background.get("corpus_doc_id", "")
        ),
        "evidence_roles_json": json.dumps(
            evidence_roles, sort_keys=True
        ),
        "has_evidence_finding": has_evidence_finding,
        "primary": primary,
    }


def _explicit_conflict(
    query_signature: Mapping[str, Any],
    candidate_signature: Mapping[str, Any],
    states: Mapping[str, str],
) -> Tuple[str, List[str]]:
    """Classify explicit argument conflicts after same-event gating.

    Mismatches are never used to establish event identity.  They become
    contradiction evidence only when the surrounding participant, predicate,
    object, date, or discriminative numeric anchors already bind the two
    propositions to the same event.
    """
    query_date = clean_text(
        query_signature.get("date_value", query_signature.get("event_time", ""))
    )
    candidate_date = clean_text(
        candidate_signature.get(
            "date_value",
            candidate_signature.get("event_time", ""),
        )
    )
    predicate_object_bound = (
        states.get("predicate_state") == "MATCH"
        and states.get("object_state") == "MATCH"
    )
    actor_bound = states.get("actor_state") == "MATCH"
    date_bound = states.get("date_state") == "MATCH"
    location_bound = states.get("location_state") == "MATCH"
    vehicle_bound = states.get("vehicle_state") == "MATCH"
    numeric_bound = any(
        states.get(name) == "MATCH"
        for name in ("packet_count_state", "total_weight_grams_state")
    )
    event_specific_match = any(
        (
            actor_bound,
            date_bound,
            location_bound,
            vehicle_bound,
            numeric_bound,
        )
    )
    numeric = [
        name for name in ("packet_count_state", "total_weight_grams_state")
        if states.get(name) == "MISMATCH"
    ]
    if numeric and predicate_object_bound and event_specific_match:
        return "DIRECT_NUMERICAL_CONTRADICTION", numeric
    if (
        states.get("date_state") == "MISMATCH"
        and query_date
        and candidate_date
        and (
            predicate_object_bound
            or (
                actor_bound
                and states.get("predicate_state") == "MATCH"
            )
            or actor_bound
            or location_bound
            or vehicle_bound
            or numeric_bound
        )
    ):
        return "ATTRIBUTE_CONTRADICTION", ["date_state"]
    attribute_binding = any(
        states.get(name) == "MATCH"
        for name in (
            "actor_state", "date_state", "location_state", "packet_count_state",
            "total_weight_grams_state",
        )
    )
    if (
        states.get("vehicle_state") == "MISMATCH"
        and predicate_object_bound
        and attribute_binding
    ):
        return "ATTRIBUTE_CONTRADICTION", ["vehicle_state"]
    location_binding = any(
        states.get(name) == "MATCH"
        for name in (
            "actor_state", "date_state", "packet_count_state",
            "total_weight_grams_state", "vehicle_state",
        )
    )
    if (
        states.get("location_state") == "MISMATCH"
        and predicate_object_bound
        and location_binding
        and clean_text(query_signature.get("location_identity", ""))
        and clean_text(candidate_signature.get("location_identity", ""))
    ):
        return "LOCATION_CONTRADICTION", ["location_state"]
    return "", []


def _conflict_anchor_contract_satisfied(
    explicit_conflict: str,
    conflict_fields: Sequence[str],
    states: Mapping[str, str],
    critical: Mapping[str, Any],
    *,
    event_score: float,
    anchor_score: float,
) -> bool:
    """Permit contradiction dimensions to bypass the generic date gate.

    The generic critical-anchor gate intentionally blocks candidates when a
    query-specified date is absent or mismatched. That is correct for support
    evidence, but it hides genuine contradiction rows when the date itself is
    the contradicted attribute. This helper keeps the contradiction contract
    conservative by still requiring participant and object binding plus one
    other non-disputed discriminative match.
    """
    if not explicit_conflict:
        return False
    disputed_fields = set(conflict_fields)
    if not bool(critical.get("participant_anchor_pass", False)):
        if not (
            disputed_fields == {"date_state"}
            and bool(critical.get("event_object_anchor_pass", False))
            and anchor_score >= 0.75
            and event_score >= 0.50
        ):
            return False
    if "object_state" not in disputed_fields and not bool(
        critical.get("event_object_anchor_pass", False)
    ):
        date_bound_without_object = bool(
            disputed_fields == {"date_state"}
            and states.get("actor_state") == "MATCH"
            and states.get("predicate_state") == "MATCH"
            and anchor_score >= 0.55
            and event_score >= 0.48
        )
        if not date_bound_without_object:
            return False
    if disputed_fields not in ({"date_state"}, {"object_state"}):
        if states.get("predicate_state") != "MATCH":
            return False
        if states.get("object_state") != "MATCH":
            return False
    stable_core_matches = sum(
        states.get(name) == "MATCH"
        for name in ("actor_state", "predicate_state", "object_state")
    )
    if disputed_fields != {"date_state"} and stable_core_matches < 2:
        return False
    remaining_specific_matches = [
        name
        for name in (
            "actor_state",
            "date_state",
            "location_state",
            "vehicle_state",
            "packet_count_state",
            "total_weight_grams_state",
        )
        if name not in disputed_fields
        and states.get(name) == "MATCH"
    ]
    if disputed_fields == {"date_state"}:
        return bool(remaining_specific_matches) or (
            anchor_score >= 0.75 and event_score >= 0.50
        )
    if disputed_fields == {"object_state"}:
        return bool(
            states.get("actor_state") == "MATCH"
            and states.get("predicate_state") == "MATCH"
            and remaining_specific_matches
        )
    return bool(remaining_specific_matches)


def _polarity_anchor_contract_satisfied(
    states: Mapping[str, str],
    *,
    event_score: float,
    anchor_score: float,
    identity_floor: float,
) -> bool:
    """Allow clean polarity opposition despite actor-granularity drift."""
    full_event_frame_match = all(
        states.get(field) == "MATCH"
        for field in ("actor_state", "predicate_state", "object_state")
    )
    if full_event_frame_match:
        # Three independently normalized role matches establish the event
        # even when a concise paraphrase lowers bag-of-words overlap. Keep a
        # modest lexical/anchor floor to reject accidental generic matches.
        return bool(
            anchor_score >= 0.30
            and event_score >= max(0.60, float(identity_floor) - 0.08)
        )
    object_not_required = states.get("object_state") == "NOT_APPLICABLE"
    if (
        object_not_required
        and states.get("actor_state") == "MATCH"
        and states.get("predicate_state") == "MATCH"
        and any(
            states.get(field) == "MATCH"
            for field in ("date_state", "location_state", "vehicle_state")
        )
    ):
        return bool(
            anchor_score >= 0.50
            and event_score >= max(float(identity_floor), 0.60)
        )
    return bool(
        states.get("predicate_state") == "MATCH"
        and states.get("object_state") == "MATCH"
        and anchor_score >= 0.85
        and event_score >= max(float(identity_floor), 0.78)
    )


def _related_different_event(
    query_signature: Mapping[str, Any],
    candidate_signature: Mapping[str, Any],
    states: Mapping[str, str],
) -> bool:
    """Identify a close event-family neighbor that is not the queried event."""
    stable_matches = sum(
        states.get(name) == "MATCH"
        for name in (
            "actor_state", "predicate_state", "object_state", "location_state",
        )
    )
    discriminators = (
        "date_state", "packet_count_state", "total_weight_grams_state",
        "vehicle_state", "location_state",
    )
    distinguishes = any(
        states.get(name) in {
            "MISMATCH", "MISSING_IN_CANDIDATE", "UNPARSEABLE",
        }
        for name in discriminators
    )
    # A query-specified discriminator missing from the candidate is evidence
    # of a related but non-certifying event, not evidence of contradiction.
    query_has_discriminator = bool(
        clean_text(query_signature.get("event_time", ""))
        or query_signature.get("packet_count") is not None
        or query_signature.get("total_weight_grams") is not None
        or clean_text(query_signature.get("vehicle_identity", ""))
        or clean_text(query_signature.get("location_identity", ""))
    )
    candidate_has_event = bool(
        clean_text(candidate_signature.get("predicate", ""))
        and clean_text(candidate_signature.get("object", ""))
    )
    return (
        stable_matches >= 2
        and candidate_has_event
        and query_has_discriminator
        and distinguishes
    )


def _uncertainty_same_event_gate(
    states: Mapping[str, str],
    critical: Mapping[str, Any],
    *,
    event_score: float,
    anchor_score: float,
    identity_floor: float,
    anchor_floor: float,
) -> Tuple[bool, str]:
    """Require event identity before assigning the uncertainty evidence role.

    An uncertainty cue is not itself evidence that a proposition concerns the
    queried event. Hard disagreement on an event discriminator therefore
    blocks the uncertainty role even when the candidate is lexically close.
    Missing event time remains eligible only through the critical-anchor
    exception implemented in :func:`_critical_anchors`.
    """
    discriminator_fields = (
        "date_state", "packet_count_state", "total_weight_grams_state",
        "vehicle_state", "location_state",
    )
    mismatches = [
        field for field in discriminator_fields
        if states.get(field) == "MISMATCH"
    ]
    if mismatches:
        return False, "discriminative_event_mismatch:" + ",".join(mismatches)
    if not bool(critical.get("critical_anchor_gate_pass", False)):
        return False, "critical_anchor_gate_failed"
    core_matches = sum(
        states.get(field) == "MATCH"
        for field in (
            "actor_state", "predicate_state", "object_state",
            "location_state",
        )
    )
    if core_matches < 1:
        # Some extraction pipelines retain an uncertainty assertion only as
        # raw source prose. Permit that source-independent condition only when
        # all typed critical anchors bind. A complete typed contract is
        # independently sufficient and is intentionally not vetoed by the
        # general lexical floor.
        full_critical_anchor_contract = all((
            bool(critical.get("participant_anchor_required", False)),
            bool(critical.get("participant_anchor_pass", False)),
            bool(critical.get("event_object_anchor_required", False)),
            bool(critical.get("event_object_anchor_pass", False)),
            bool(critical.get("event_time_anchor_required", False)),
            bool(critical.get("event_time_anchor_pass", False)),
            critical.get("event_time_comparison") == "MATCH",
        ))
        if full_critical_anchor_contract:
            return True, "same_event_bound_by_full_critical_anchor_contract"
        return False, "no_matching_core_event_argument"
    if event_score < identity_floor and anchor_score < anchor_floor:
        return False, "same_event_identity_and_anchor_floors_not_met"
    return True, "same_event_eligible_for_uncertainty"


def _resource_or_core(
    resources: SemanticResources | None,
) -> SemanticResources:
    return resources if resources is not None else load_semantic_resources()


def _expanded_tokens(
    value: Any, resources: SemanticResources | None = None
) -> set[str]:
    return set(_resource_or_core(resources).expanded_tokens(value))


def _literal_anchor_tokens(value: Any) -> set[str]:
    return {
        token
        for token in semantic_tokens(value, preserve_negation=False)
        if len(token) >= 2
    }


def _anchor_score(
    query_text: str,
    candidate_text: str,
    resources: SemanticResources | None = None,
) -> float:
    generic = {
        "there", "would", "will", "have", "been", "for", "the", "a", "an",
        "no", "not", "report", "reported", "said", "claim",
    }
    anchors = _expanded_tokens(query_text, resources) - generic
    return (
        containment(anchors, _expanded_tokens(candidate_text, resources))
        if anchors else 0.0
    )


def _raw_conflict_tokens(value: Any) -> set[str]:
    """Normalize only high-confidence lexical variants for raw conflict gates."""
    source = clean_text(value).casefold()
    normalized: set[str] = set()
    for token in semantic_tokens(source, preserve_negation=False):
        if token in _RAW_CONFLICT_STOPWORDS or token.isdigit():
            continue
        # Tokens cannot contain a multiword predicate frame. Direct dictionary
        # lookup is equivalent here and avoids scanning every phrase regex for
        # every candidate token in a large corpus.
        canonical = PREDICATE_LEMMAS.get(token, token)
        canonical = _RAW_CONFLICT_TOKEN_EQUIVALENTS.get(
            canonical, _RAW_CONFLICT_TOKEN_EQUIVALENTS.get(token, canonical)
        )
        if len(canonical) >= 2:
            normalized.add(canonical)
    # Add narrow phrase concepts where token-by-token normalization would be
    # ambiguous. These bind event identity only; a separate explicit
    # date/number/polarity difference is still required for contradiction.
    phrase_concepts = (
        (
            r"\b(?:formal\s+)?(?:diplomatic\s+|economic\s+)?"
            r"relation(?:ship)?s?\b",
            "formal_relation",
        ),
        (
            r"\b(?:meet(?:s|ing)?|satisf(?:y|ies|ied|ying))\b[^.;]{0,55}"
            r"\bclaims?\b",
            "satisfy_claim",
        ),
        (
            r"\b(?:act\s+positively|(?:take\s+)?(?:some\s+)?positive\s+action)\b",
            "positive_action",
        ),
        (
            r"\b(?:no\s+progress|nothing\s+(?:is\s+)?accomplished|"
            r"lack\s+of\s+progress)\b",
            "lack_progress",
        ),
    )
    for pattern, concept in phrase_concepts:
        if re.search(pattern, source):
            normalized.add(concept)
    return normalized


def _raw_explicit_negation(value: Any) -> bool:
    """Detect proposition negation while excluding pure knowledge uncertainty."""
    source = clean_text(value).casefold()
    if not source:
        return False
    # Quotation punctuation must not sever the grammatical scope of a
    # negator, as in ``not 'on the books'``. Preserve apostrophes inside words
    # while removing quote marks used as delimiters.
    source = re.sub(r"[\"“”`]+", " ", source)
    source = re.sub(r"(?<![a-z])['’]+|['’]+(?![a-z])", " ", source)
    source = clean_text(source)
    for phrase in _RAW_UNCERTAINTY_NEGATION_PHRASES:
        source = source.replace(phrase, " ")
    if re.search(
        r"\b(?:deny|denies|denied|refuse|refuses|refused|reject|rejects|"
        r"rejected)\b",
        source,
    ):
        return True
    if re.search(r"\b(?:no longer|not on|without)\b", source):
        return True
    # Preserve proposition-level negation expressed through a bare ``no``
    # noun phrase, while the uncertainty phrases above remain excluded. This
    # covers clauses such as ``no demonstrations were called`` without making
    # every occurrence of ``no`` a contradiction signal.
    if re.search(
        r"\bno\s+(?:[a-z][a-z'-]*\s+){0,5}"
        r"(?:be|been|being|is|are|was|were|occur|occurred|happen|happened|"
        r"call|called|report|reported|find|found|provide|provided|make|made)\b",
        source,
    ):
        return True
    scoped = re.finditer(
        r"\b(?:not|never|neither|nor)\b\s+"
        r"(?:(?:have|has|had|be|been|being|is|are|was|were|to)\s+){0,3}"
        r"(?P<head>[a-z][a-z'-]*)",
        source,
    )
    nonsemantic_heads = {
        "clear", "confirmed", "evidence", "information", "known",
        "news", "we", "they", "he", "she", "i", "you",
    }
    return any(
        match.group("head") not in nonsemantic_heads
        for match in scoped
    )


def _raw_best_local_surface(query_text: Any, candidate_text: Any) -> str:
    """Select the candidate sentence/clause closest to the raw query.

    Chunk-backed evidence can contain several independent facts. Conflict
    dimensions and negation must be read from the matching local fact rather
    than leaking from a later sentence in the same chunk.
    """
    candidate = clean_text(candidate_text)
    query_tokens = _raw_conflict_tokens(query_text)
    if not candidate or not query_tokens:
        return candidate
    surfaces = [
        clean_text(value)
        for value in re.split(
            r"(?<=[.!?;])(?:[\"'”’]+)?\s+|"
            r"\s+(?=(?:and|but)\s+(?:added|noted|reported|said|stated)\b)",
            candidate,
            flags=re.IGNORECASE,
        )
        if clean_text(value)
    ]
    if len(surfaces) <= 1:
        return candidate

    def local_score(surface: str) -> tuple[float, int, float]:
        surface_tokens = _raw_conflict_tokens(surface)
        shared = query_tokens & surface_tokens
        containment = len(shared) / float(max(1, min(
            len(query_tokens), len(surface_tokens)
        )))
        query_coverage = len(shared) / float(max(1, len(query_tokens)))
        return (query_coverage, len(shared), containment)

    best = max(surfaces, key=local_score)
    return best if local_score(best)[1] >= 2 else candidate


@lru_cache(maxsize=65_536)
def _raw_surface_dimension_sets(
    normalized_text: str,
) -> Tuple[frozenset[str], frozenset[str]]:
    """Return every explicit date and non-date number on one text surface."""
    dates: set[str] = set()

    def replace_month_date(match: re.Match[str]) -> str:
        month_key = match.group("month").casefold().rstrip(".")
        month = _RAW_MONTH_NUMBERS.get(month_key)
        if month is None:
            month = _RAW_MONTH_NUMBERS.get(month_key[:3], 0)
        day = match.group("day") or match.group("day_only") or ""
        year = match.group("year") or match.group("year_only") or ""
        if year and day:
            dates.add(f"{int(year):04d}-{month:02d}-{int(day):02d}")
        elif year:
            dates.add(f"{int(year):04d}-{month:02d}")
        elif day:
            dates.add(f"{month:02d}-{int(day):02d}")
        return " " * len(match.group(0))

    def replace_slash_date(match: re.Match[str]) -> str:
        year = int(match.group("year"))
        if year < 100:
            year += 1900 if year >= 50 else 2000
        dates.add(
            f"{year:04d}-{int(match.group('month')):02d}-"
            f"{int(match.group('day')):02d}"
        )
        return " " * len(match.group(0))

    def replace_day_month_date(match: re.Match[str]) -> str:
        month_key = match.group("month").casefold().rstrip(".")
        month = _RAW_MONTH_NUMBERS.get(month_key)
        if month is None:
            month = _RAW_MONTH_NUMBERS.get(month_key[:3], 0)
        day = int(match.group("day"))
        year = match.group("year") or ""
        if year:
            dates.add(f"{int(year):04d}-{month:02d}-{day:02d}")
        else:
            dates.add(f"{month:02d}-{day:02d}")
        return " " * len(match.group(0))

    without_dates = _RAW_DAY_MONTH_DATE_RE.sub(
        replace_day_month_date, normalized_text
    )
    without_dates = _RAW_MONTH_DATE_RE.sub(replace_month_date, without_dates)
    without_dates = _RAW_SLASH_DATE_RE.sub(replace_slash_date, without_dates)

    def replace_standalone_month(match: re.Match[str]) -> str:
        month_key = match.group("month").casefold().rstrip(".")
        month = _RAW_MONTH_NUMBERS.get(month_key)
        if month is None:
            month = _RAW_MONTH_NUMBERS.get(month_key[:3], 0)
        dates.add(f"M-{month:02d}")
        return " " * len(match.group(0))

    without_dates = _RAW_STANDALONE_MONTH_RE.sub(
        replace_standalone_month, without_dates
    )

    def replace_year(match: re.Match[str]) -> str:
        dates.add(match.group(0))
        return " " * len(match.group(0))

    without_dates = _RAW_YEAR_RE.sub(replace_year, without_dates)
    numbers = {
        match.group(0).replace(",", "")
        for match in _RAW_NUMBER_RE.finditer(without_dates)
    }
    return frozenset(dates), frozenset(numbers)


def _raw_surface_dimensions(value: Any) -> Tuple[frozenset[str], frozenset[str]]:
    return _raw_surface_dimension_sets(clean_text(value).casefold())


def _raw_temporal_identity(signature: Mapping[str, Any], text: Any) -> str:
    normalized = clean_text(signature.get("event_time", ""))
    if normalized:
        return normalized
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", clean_text(text))))
    return years[0] if len(years) == 1 else ""


def _raw_local_conflict(
    query_text: Any,
    candidate_text: Any,
    query_signature: Mapping[str, Any],
    candidate_signature: Mapping[str, Any],
    *,
    query_similarity: float,
    anchor_score: float,
    typed_actor_match: bool = False,
    typed_predicate_match: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Recover explicit clause-local polarity or time conflicts.

    This is a fallback for parser noise, not a looser contradiction rule. A
    conflict is accepted only when the two surfaces share a substantial event
    core or when typed actor/predicate anchors independently bind them.
    """
    candidate_local_text = _raw_best_local_surface(query_text, candidate_text)
    query_tokens = _raw_conflict_tokens(query_text)
    candidate_tokens = _raw_conflict_tokens(candidate_local_text)
    shared = query_tokens & candidate_tokens
    shorter = min(len(query_tokens), len(candidate_tokens))
    token_containment = len(shared) / float(max(1, shorter))
    actor_match = typed_actor_match or compare_argument(
        query_signature.get("embedded_actor", "")
        or query_signature.get("actor", ""),
        candidate_signature.get("embedded_actor", "")
        or candidate_signature.get("actor", ""),
    ) == "MATCH"
    predicate_match = typed_predicate_match or compare_argument(
        query_signature.get("predicate", ""),
        candidate_signature.get("predicate", ""),
    ) == "MATCH"
    strong_surface_core = bool(
        len(shared) >= 3
        and token_containment >= 0.68
        and (query_similarity >= 0.42 or anchor_score >= 0.60)
    )
    typed_surface_core = bool(
        actor_match
        and predicate_match
        and len(shared) >= 2
        and token_containment >= 0.50
        and query_similarity >= 0.30
    )
    compact_typed_core = bool(
        actor_match
        and len(shared) >= 2
        and token_containment >= 0.66
        and query_similarity >= 0.50
    )
    query_dates, query_numbers = _raw_surface_dimensions(query_text)
    candidate_dates, candidate_numbers = _raw_surface_dimensions(
        candidate_local_text
    )
    date_conflict = bool(
        query_dates
        and candidate_dates
        and (
            query_dates.isdisjoint(candidate_dates)
            or (
                len(query_dates) == len(candidate_dates)
                and query_dates != candidate_dates
            )
        )
    )
    numeric_conflict = bool(
        query_numbers
        and candidate_numbers
        and (
            query_numbers.isdisjoint(candidate_numbers)
            or (
                len(query_numbers) == len(candidate_numbers)
                and query_numbers != candidate_numbers
            )
        )
    )
    shared_dimension_value = bool(
        (query_dates & candidate_dates)
        or (query_numbers & candidate_numbers)
    )
    # A shared actor alone does not bind two numerical or temporal statements
    # to the same event. Require a predicate or a strong lexical event core.
    dimension_surface_core = bool(
        (date_conflict or numeric_conflict)
        and (
            (
                predicate_match
                and len(shared) >= 4
                and token_containment >= 0.42
                and query_similarity >= 0.30
            )
            or (
                shared_dimension_value
                and len(shared) >= 3
                and token_containment >= 0.45
                and query_similarity >= 0.30
            )
            or (
                len(shared) >= 5
                and token_containment >= 0.60
                and query_similarity >= 0.38
                and (actor_match or anchor_score >= 0.66)
            )
            or (
                # Parser roles can be unreliable for nominal schedule and
                # deadline statements. A date disagreement plus six shared
                # event terms is an independent, conservative event binding.
                len(shared) >= 6
                and token_containment >= 0.50
                and query_similarity >= 0.42
                and anchor_score >= 0.50
            )
            or (
                # Phrase concepts can independently bind compact paraphrases
                # even when parser roles drift. The explicit conflicting
                # dimension remains mandatory outside this event-core gate.
                len(shared) >= 6
                and token_containment >= 0.75
                and query_similarity >= 0.28
            )
            or (
                {"positive_action", "lack_progress"}.issubset(shared)
                and len(shared) >= 5
                and query_similarity >= 0.28
            )
        )
    )
    query_time = _raw_temporal_identity(query_signature, query_text)
    candidate_time = _raw_temporal_identity(candidate_signature, candidate_text)
    temporal_surface_core = bool(
        query_time
        and candidate_time
        and query_time != candidate_time
        and len(shared) >= 5
        and token_containment >= 0.60
        and query_similarity >= 0.35
    )
    same_event = any((
        strong_surface_core,
        typed_surface_core,
        compact_typed_core,
        temporal_surface_core,
        dimension_surface_core,
    ))
    audit = {
        "shared_token_count": len(shared),
        "token_containment": round(token_containment, 6),
        "actor_match": actor_match,
        "predicate_match": predicate_match,
        "query_dates": sorted(query_dates),
        "candidate_dates": sorted(candidate_dates),
        "query_numbers": sorted(query_numbers),
        "candidate_numbers": sorted(candidate_numbers),
        "candidate_local_surface": candidate_local_text,
        "date_set_conflict": date_conflict,
        "numeric_set_conflict": numeric_conflict,
        "dimension_surface_core_pass": dimension_surface_core,
        "same_event_gate_pass": same_event,
    }
    if not same_event:
        return "", audit

    if date_conflict or (
        query_time and candidate_time and query_time != candidate_time
    ):
        audit.update({
            "query_time": query_time,
            "candidate_time": candidate_time,
        })
        return "RAW_LOCAL_TEMPORAL_CONFLICT", audit

    if numeric_conflict:
        return "RAW_LOCAL_NUMERIC_CONFLICT", audit

    query_negative = _raw_explicit_negation(query_text)
    candidate_negative = _raw_explicit_negation(candidate_local_text)
    audit.update({
        "query_explicit_negation": query_negative,
        "candidate_explicit_negation": candidate_negative,
    })
    if query_negative != candidate_negative:
        return "RAW_LOCAL_POLARITY_CONFLICT", audit
    return "", audit


def _critical_anchors(
    query_signature: Mapping[str, Any],
    candidate_signature: Mapping[str, Any],
    query_text: str,
    candidate_text: str,
    resources: SemanticResources | None = None,
    *,
    participant_containment_floor: float = DEFAULT_ENGINE_THRESHOLDS[
        "contradiction_participant_containment_floor"
    ],
    object_containment_floor: float = DEFAULT_ENGINE_THRESHOLDS[
        "contradiction_object_containment_floor"
    ],
) -> Dict[str, Any]:
    resources = _resource_or_core(resources)
    query_actor = clean_text(
        query_signature.get("embedded_actor", "")
        or query_signature.get("actor", "")
    )
    candidate_actor = clean_text(
        candidate_signature.get("embedded_actor", "")
        or candidate_signature.get("actor", "")
    )
    q_actor_tokens = _expanded_tokens(query_actor, resources)
    c_actor_tokens = _expanded_tokens(
        candidate_actor + " " + candidate_text, resources
    )
    participant_concepts = shared_concepts(
        resources,
        query_actor or query_text,
        candidate_actor + " " + candidate_text,
        "alias",
    )
    participant_required = bool(q_actor_tokens)
    q_actor_literal = _literal_anchor_tokens(query_actor)
    c_actor_literal = _literal_anchor_tokens(
        candidate_actor + " " + candidate_text
    )
    participant_literal_pass = (
        containment(q_actor_literal, c_actor_literal)
        >= participant_containment_floor
        if q_actor_literal else False
    )
    participant_argument_pass = (
        compare_argument(query_actor, candidate_actor) == "MATCH"
        if query_actor and candidate_actor else False
    )
    participant_pass = (
        bool(participant_concepts)
        or participant_argument_pass
        or participant_literal_pass
        or containment(q_actor_tokens, c_actor_tokens)
        >= participant_containment_floor
        if participant_required else True
    )

    query_object = clean_text(query_signature.get("object", ""))
    object_required = bool(query_object)
    object_tokens = _expanded_tokens(query_object, resources)
    candidate_object_tokens = _expanded_tokens(
        clean_text(candidate_signature.get("object", "")) + " " + candidate_text,
        resources,
    )
    event_family_concepts = (
        shared_concepts(
            resources,
            query_object or query_text,
            clean_text(candidate_signature.get("object", ""))
            + " " + candidate_text,
            "event",
        )
        | shared_concepts(
            resources,
            query_object or query_text,
            clean_text(candidate_signature.get("object", ""))
            + " " + candidate_text,
            "alias",
        )
    )
    object_literal_tokens = _literal_anchor_tokens(query_object)
    candidate_object_literal_tokens = _literal_anchor_tokens(
        clean_text(candidate_signature.get("object", "")) + " " + candidate_text
    )
    object_literal_pass = (
        containment(object_literal_tokens, candidate_object_literal_tokens)
        >= object_containment_floor
        if object_literal_tokens else False
    )
    object_argument_pass = (
        compare_argument(
            query_object, clean_text(candidate_signature.get("object", ""))
        ) == "MATCH"
        if query_object and clean_text(candidate_signature.get("object", ""))
        else False
    )
    object_pass = (
        bool(event_family_concepts)
        or object_argument_pass
        or object_literal_pass
        or containment(object_tokens, candidate_object_tokens)
        >= object_containment_floor
        if object_required else True
    )

    query_date = clean_text(query_signature.get("event_time", ""))
    candidate_date = clean_text(candidate_signature.get("event_time", ""))
    if not candidate_date:
        report_date = clean_text(candidate_signature.get("report_time", ""))
        weekday_match = next(
            (
                (index, name)
                for index, name in enumerate(
                    ("monday", "tuesday", "wednesday", "thursday",
                     "friday", "saturday", "sunday")
                )
                if name in candidate_text.lower()
            ),
            None,
        )
        try:
            if weekday_match and report_date:
                base = date.fromisoformat(report_date[:10])
                delta = (weekday_match[0] - base.weekday()) % 7
                candidate_date = str(base + timedelta(days=delta or 7))
        except ValueError:
            candidate_date = ""
    date_required = bool(query_date)
    date_comparison = compare_dates(query_date, candidate_date)
    uncertainty = (
        infer_epistemic_state(candidate_text)
        == "uncertain_or_insufficient_information"
    )
    date_pass = (
        date_comparison == "MATCH"
        or (
            date_required
            and not candidate_date
            and uncertainty
            and participant_pass
            and object_pass
        )
    ) if date_required else True
    if (
        date_required and not candidate_date and date_pass and uncertainty
    ):
        date_comparison = (
            "MISSING_IN_CANDIDATE_ACCEPTED_FOR_UNCERTAINTY"
        )
    passed = participant_pass and object_pass and date_pass
    return {
        "participant_anchor_required": participant_required,
        "participant_anchor_pass": participant_pass,
        "participant_resource_concepts_json": json.dumps(
            sorted(participant_concepts)
        ),
        "event_object_anchor_required": object_required,
        "event_object_anchor_pass": object_pass,
        "event_object_resource_concepts_json": json.dumps(
            sorted(event_family_concepts)
        ),
        "event_time_anchor_required": date_required,
        "event_time_anchor_pass": date_pass,
        "event_time_comparison": date_comparison,
        # Compatibility aliases for established audit readers.
        "memorial_anchor_required": object_required,
        "memorial_anchor_pass": object_pass,
        "event_date_anchor_required": date_required,
        "event_date_anchor_pass": date_pass,
        "event_date_comparison": date_comparison,
        "critical_anchor_gate_pass": passed,
        "semantic_resource_fingerprint": resources.fingerprint,
    }


def _retrieval_routes(query: Mapping[str, Any]) -> List[Tuple[str, str]]:
    signature = _normalize_concern_relation_signature(
        query.get("text", ""), query["signature"]
    )
    actor = clean_text(
        signature.get("embedded_actor", "") or signature.get("actor", "")
    )
    obj = clean_text(signature.get("object", ""))
    event_time = clean_text(signature.get("event_time", ""))
    core = clean_text(" ".join([actor, obj, event_time]))
    routes = [
        ("SIDE_A_SUPPORT", clean_text(query["text"])),
        (
            "SIDE_B_OR_UPDATE",
            clean_text(
                " ".join([
                    core, "opposite changed updated present absent",
                ])
            ),
        ),
        (
            "UNCERTAINTY",
            clean_text(
                " ".join([
                    core, "unknown whether uncertain not confirmed insufficient",
                ])
            ),
        ),
        (
            "QUALIFICATION_OR_SCOPE",
            clean_text(
                " ".join([
                    core, "qualification scope partial conditional approximate",
                ])
            ),
        ),
        ("EVENT_CORE", core),
    ]
    return [(name, text) for name, text in routes if text]


def _event_family_candidates(
    query: Mapping[str, Any],
    context: EngineContext,
    per_class_quota: int = 30,
) -> List[Tuple[str, float, str]]:
    """High-recall proposition route with independent evidence-class quotas.

    Whole-document lexical rank can bury a short update or uncertainty
    sentence inside a long cable.  This route scores assembled propositions
    against the actor/object/date event core, then retains a bounded quota for
    each semantic-state family.  It is corpus-agnostic and uses no document
    identifiers or expected answers.
    """
    signature = query["signature"]
    core = clean_text(" ".join([
        clean_text(
            signature.get("embedded_actor", "") or signature.get("actor", "")
        ),
        clean_text(signature.get("object", "")),
        clean_text(signature.get("event_time", "")),
    ]))
    core_tokens = _expanded_tokens(core, context.semantic_resources)
    if not core_tokens:
        return []
    exhaustive = (
        context.config.raw_use_mode == "discover"
        or context.config.retrieval_mode == "exhaustive"
    )
    # First use the query-independent document token index to bound the
    # proposition-level scan. Four evidence families with a twofold overfetch
    # require at most eight per-class quotas; two staged document budgets
    # provide the same protection when that is larger. Small corpora remain
    # exhaustive, while large corpora no longer re-expand every proposition
    # for every contradiction query.
    document_overfetch = max(
        context.config.staged_top_docs * 2,
        per_class_quota * 8,
    )
    document_candidates: List[Tuple[str, float]] = []
    for doc_id, tokens in context.corpus.document_tokens.items():
        score = containment(core_tokens, tokens)
        if score >= context.config.contradiction_route_relevance_floor:
            document_candidates.append((doc_id, float(score)))
    document_candidates.sort(key=lambda item: (-item[1], item[0]))
    if not exhaustive:
        document_candidates = document_candidates[:document_overfetch]

    buckets: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
    for doc_id, _document_score in document_candidates:
        records = context.corpus.propositions_by_doc.get(doc_id, ())
        best_score = 0.0
        best_family = "QUALIFICATION"
        for record in records:
            record_tokens = _expanded_tokens(
                record.text, context.semantic_resources
            )
            score = containment(core_tokens, record_tokens)
            if score < best_score:
                continue
            epistemic = resolve_epistemic_scope(
                record.text,
                record.signature.get("epistemic_state", ""),
            )["epistemic_state"]
            state = clean_text(
                record.signature.get("semantic_state_value", "")
            )
            if epistemic == "uncertain_or_insufficient_information":
                family = "UNCERTAINTY"
            elif state == "present":
                family = "SIDE_B_OR_UPDATE"
            elif state == "absent":
                family = "SIDE_A_SUPPORT"
            else:
                family = "QUALIFICATION"
            best_score = score
            best_family = family
        if best_score >= context.config.contradiction_route_relevance_floor:
            buckets[best_family].append(
                (doc_id, float(best_score), "EVENT_FAMILY_" + best_family)
            )
    selected: List[Tuple[str, float, str]] = []
    for family in sorted(buckets):
        selected.extend(sorted(
            buckets[family], key=lambda item: (-item[1], item[0])
        )[:per_class_quota])
    return selected


def _proposition_family(record: PropositionRecord) -> str:
    scoped = resolve_epistemic_scope(
        record.text, record.signature.get("epistemic_state", "")
    )
    if scoped["epistemic_state"] == "uncertain_or_insufficient_information":
        return "UNCERTAINTY"
    state = clean_text(record.signature.get("semantic_state_value", ""))
    if state == "present":
        return "PRESENT"
    if state == "absent":
        return "ABSENT"
    polarity = clean_text(record.signature.get("assertion_polarity", ""))
    if polarity in {"negative", "positive"}:
        return "POLARITY_" + polarity.upper()
    return "QUALIFICATION"


def _retrieved_document_propositions(
    context: EngineContext,
    doc_id: str,
    query: Mapping[str, Any],
) -> Tuple[Sequence[PropositionRecord], List[Dict[str, Any]]]:
    """Return bounded, class-preserving propositions plus a budget audit.

    Canonical pseudo-table propositions and raw source sentences have separate
    per-class quotas.  The budget depends only on semantic class and relevance;
    it never depends on a corpus name, document identifier, case identifier, or
    expected answer.
    """
    canonical = list(context.corpus.propositions_by_doc.get(doc_id, ()))
    seen_text = {
        clean_text(record.text).casefold()
        for record in canonical
        if clean_text(record.text)
    }
    report_time = clean_text(
        context.corpus.document_dates.get(doc_id, "")
    )
    raw_records: List[PropositionRecord] = []
    for sentence in context.corpus.sentences_by_doc.get(doc_id, ()):
        text = clean_text(sentence.sentence)
        normalized = text.casefold()
        if not text or normalized in seen_text:
            continue
        arguments = infer_simple_clause_arguments(text)
        signature = build_event_signature(
            text,
            actor=arguments.get("actor", ""),
            predicate=arguments.get("predicate", ""),
            obj=arguments.get("object", ""),
            location=arguments.get("location", ""),
            report_time=report_time,
        )
        raw_records.append(PropositionRecord(
            doc_id=doc_id,
            source_table="chunks.csv",
            row_id=(
                f"{sentence.chunk_id}::sentence_"
                f"{sentence.sentence_index}"
            ),
            text=text,
            signature=signature,
            report_time=report_time,
        ))
        seen_text.add(normalized)

    selected: List[PropositionRecord] = []
    audit_rows: List[Dict[str, Any]] = []
    query_surfaces = {
        clean_text(query.get("text", "")),
        *(
            clean_text(hypothesis.get("text", ""))
            for hypothesis in query.get("hypotheses", ())
        ),
    }
    query_surfaces.discard("")
    for representation, records, quota in (
        (
            "canonical",
            canonical,
            context.config.contradiction_propositions_per_class_per_document,
        ),
        (
            "raw_sentence",
            raw_records,
            context.config.contradiction_raw_sentences_per_class_per_document,
        ),
    ):
        buckets: Dict[str, List[Tuple[float, PropositionRecord]]] = defaultdict(list)
        for record in records:
            relevance = max(
                (
                    _anchor_score(
                        surface, record.text, context.semantic_resources
                    )
                    for surface in query_surfaces
                ),
                default=0.0,
            )
            buckets[_proposition_family(record)].append((relevance, record))
        for family, values in sorted(buckets.items()):
            ranked = sorted(
                values,
                key=lambda item: (
                    -item[0],
                    item[1].source_table,
                    item[1].row_id,
                ),
            )
            kept = ranked[: max(0, quota)]
            selected.extend(record for _score, record in kept)
            audit_rows.append({
                "query_proposition_id": query["id"],
                "corpus_doc_id": doc_id,
                "representation": representation,
                "semantic_family": family,
                "available_count": len(ranked),
                "quota": quota,
                "selected_count": len(kept),
                "dropped_count": max(0, len(ranked) - len(kept)),
                "top_relevance_score": kept[0][0] if kept else "",
                "selection_contract": (
                    "bounded_best_per_document_representation_and_semantic_class"
                ),
            })
    selected.sort(key=lambda record: (
        record.source_table == "chunks.csv",
        record.source_table,
        record.row_id,
    ))
    return selected, audit_rows


class ContradictionEngine(TaskEngine):
    task_name = "contradictions"

    def run(self, context: EngineContext) -> EngineResult:
        evaluation_rows: List[Dict[str, Any]] = []
        all_rows: List[Dict[str, Any]] = []
        coverage_rows: List[Dict[str, Any]] = []
        budget_rows: List[Dict[str, Any]] = []
        document_class_rows: List[Dict[str, Any]] = []
        collapse_rows: List[Dict[str, Any]] = []
        raw_queries = context.query.proposition_records()
        fragment_text_by_id = {
            clean_text(row.get("fragment_id", "")): clean_text(
                row.get("fragment_text", "")
            )
            for row in context.query.fragments.to_dict("records")
            if clean_text(row.get("fragment_id", ""))
        }
        fragment_signature_by_id = {
            fragment_id: _normalize_concern_relation_signature(
                fragment_text,
                build_event_signature(fragment_text),
            )
            for fragment_id, fragment_text in fragment_text_by_id.items()
        }
        grouped_queries: Dict[str, List[Dict[str, Any]]] = {}
        group_order: List[str] = []
        for query in raw_queries:
            group_id = clean_text(
                query.get("hypothesis_group_id", "")
            ) or clean_text(query.get("id", ""))
            if group_id not in grouped_queries:
                grouped_queries[group_id] = []
                group_order.append(group_id)
            grouped_queries[group_id].append(query)
        queries: List[Dict[str, Any]] = []
        for group_id in group_order:
            hypotheses = grouped_queries[group_id]
            hypotheses.sort(
                key=lambda item: (
                    -float(item.get("hypothesis_confidence", 1.0) or 0.0),
                    clean_text(item.get("proposition_hypothesis_id", "")),
                    clean_text(item.get("id", "")),
                )
            )
            primary_query = dict(hypotheses[0])
            primary_query["hypothesis_group_id"] = group_id
            retained_hypotheses = [dict(item) for item in hypotheses]
            fragment_id = clean_text(primary_query.get("fragment_id", ""))
            fragment_text = fragment_text_by_id.get(fragment_id, "")
            deterministic_signature = fragment_signature_by_id.get(
                fragment_id, {}
            )
            deterministic_is_authoritative = (
                fragment_text
                and _explicit_structure_is_authoritative(
                    fragment_text, deterministic_signature
                )
            )
            if deterministic_is_authoritative:
                deterministic_predicate = canonical_key(
                    deterministic_signature.get("predicate", "")
                )
                compatible_hypotheses = [
                    item for item in retained_hypotheses
                    if canonical_key(
                        item.get("signature", {}).get("predicate", "")
                    ) == deterministic_predicate
                ]
                # Keep parser detail only when it respects the explicit
                # relation stated by the analyst. The complete-fragment
                # hypothesis below remains the authoritative fallback.
                retained_hypotheses = compatible_hypotheses
            if fragment_text and all(
                tuple(
                    canonical_key(item.get("signature", {}).get(field, ""))
                    for field in (
                        "actor", "predicate", "object", "location",
                        "assertion_polarity", "semantic_state_variable",
                        "semantic_state_value",
                    )
                )
                != tuple(
                    canonical_key(deterministic_signature.get(field, ""))
                    for field in (
                        "actor", "predicate", "object", "location",
                        "assertion_polarity", "semantic_state_variable",
                        "semantic_state_value",
                    )
                )
                for item in retained_hypotheses
            ):
                # Bind candidates against a deterministic signature of the
                # complete analyst fragment. Identical surface text is not
                # sufficient: a parser may still attach the wrong event
                # arguments or polarity.
                retained_hypotheses.append({
                    "id": f"{group_id}__full_fragment",
                    "fragment_id": fragment_id,
                    "hypothesis_group_id": group_id,
                    "hypothesis_confidence": 1.0,
                    "text": fragment_text,
                    "signature": deterministic_signature,
                })
            if not retained_hypotheses and fragment_text:
                retained_hypotheses.append({
                    "id": f"{group_id}__full_fragment",
                    "fragment_id": fragment_id,
                    "hypothesis_group_id": group_id,
                    "hypothesis_confidence": 1.0,
                    "text": fragment_text,
                    "signature": deterministic_signature,
                })
            primary_query["hypotheses"] = retained_hypotheses
            queries.append(primary_query)
        context.control.emit(
            "contradictions_started",
            target_count=len(queries),
            corpus_proposition_count=sum(
                len(rows) for rows in context.corpus.propositions_by_doc.values()
            ),
        )
        for query in queries:
            retrieval_scores: Dict[str, float] = {}
            retrieval_routes: Dict[str, set[str]] = defaultdict(set)
            retrieval_route_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
            full_fragment_text = fragment_text_by_id.get(
                clean_text(query.get("fragment_id", "")),
                clean_text(query.get("text", "")),
            )
            for route_rank, (doc_id, score) in enumerate(
                rank_documents(
                    full_fragment_text,
                    context.corpus,
                    context.config.staged_top_docs,
                    context.semantic_resources,
                ),
                start=1,
            ):
                retrieval_scores[doc_id] = max(
                    retrieval_scores.get(doc_id, 0.0), float(score)
                )
                retrieval_routes[doc_id].add("FULL_QUERY_FRAGMENT")
                retrieval_route_ranks["FULL_QUERY_FRAGMENT"][doc_id] = (
                    route_rank
                )
            for hypothesis in query.get("hypotheses", [query]):
                for route_name, route_text in _retrieval_routes(
                    hypothesis
                ):
                    ranked_route = rank_documents(
                        route_text,
                        context.corpus,
                        context.config.staged_top_docs,
                        context.semantic_resources,
                    )
                    for route_rank, (doc_id, score) in enumerate(
                        ranked_route, start=1
                    ):
                        retrieval_scores[doc_id] = max(
                            retrieval_scores.get(doc_id, 0.0),
                            float(score),
                        )
                        retrieval_routes[doc_id].add(route_name)
                        retrieval_route_ranks[route_name][doc_id] = min(
                            retrieval_route_ranks[route_name].get(
                                doc_id, route_rank
                            ),
                            route_rank,
                        )
                event_route_positions: Dict[str, int] = defaultdict(int)
                for doc_id, score, route_name in _event_family_candidates(
                    hypothesis, context
                ):
                    event_route_positions[route_name] += 1
                    route_rank = event_route_positions[route_name]
                    retrieval_scores[doc_id] = max(
                        retrieval_scores.get(doc_id, 0.0),
                        float(score),
                    )
                    retrieval_routes[doc_id].add(route_name)
                    retrieval_route_ranks[route_name][doc_id] = min(
                        retrieval_route_ranks[route_name].get(
                            doc_id, route_rank
                        ),
                        route_rank,
                    )

            exhaustive = (
                context.config.raw_use_mode == "discover"
                or context.config.retrieval_mode == "exhaustive"
            )
            route_union_cap = max(
                context.config.staged_top_docs,
                len(retrieval_route_ranks),
            )
            if not exhaustive and len(retrieval_scores) > route_union_cap:
                route_count = max(1, len(retrieval_route_ranks))
                per_route_floor = max(
                    1,
                    route_union_cap // (2 * route_count),
                )
                protected_docs: set[str] = set()
                for route_name in sorted(retrieval_route_ranks):
                    ranked_docs = sorted(
                        retrieval_route_ranks[route_name],
                        key=lambda doc_id: (
                            retrieval_route_ranks[route_name][doc_id],
                            -retrieval_scores.get(doc_id, 0.0),
                            doc_id,
                        ),
                    )
                    protected_docs.update(ranked_docs[:per_route_floor])
                consensus_docs = sorted(
                    retrieval_scores,
                    key=lambda doc_id: (
                        -len(retrieval_routes.get(doc_id, ())),
                        -retrieval_scores.get(doc_id, 0.0),
                        min(
                            (
                                ranks[doc_id]
                                for ranks in retrieval_route_ranks.values()
                                if doc_id in ranks
                            ),
                            default=10**9,
                        ),
                        doc_id,
                    ),
                )
                retained_docs = set(protected_docs)
                for doc_id in consensus_docs:
                    if len(retained_docs) >= route_union_cap:
                        break
                    retained_docs.add(doc_id)
                retrieval_scores = {
                    doc_id: score
                    for doc_id, score in retrieval_scores.items()
                    if doc_id in retained_docs
                }
                retrieval_routes = defaultdict(
                    set,
                    {
                        doc_id: routes
                        for doc_id, routes in retrieval_routes.items()
                        if doc_id in retained_docs
                    },
                )

            interim: List[Dict[str, Any]] = []
            for doc_id in sorted(retrieval_scores):
                records, budget_audit = _retrieved_document_propositions(
                    context, doc_id, query
                )
                budget_rows.extend(budget_audit)
                for record in records:
                    hypothesis_bindings = []
                    for hypothesis in query.get(
                        "hypotheses", [query]
                    ):
                        hypothesis_identity, hypothesis_states = (
                            event_identity(
                                hypothesis["signature"],
                                record.signature,
                                hypothesis["text"],
                                record.text,
                                context.semantic_resources,
                            )
                        )
                        hypothesis_anchor = _anchor_score(
                            hypothesis["text"],
                            record.text,
                            context.semantic_resources,
                        )
                        hypothesis_critical = _critical_anchors(
                            hypothesis["signature"],
                            record.signature,
                            hypothesis["text"],
                            record.text,
                            context.semantic_resources,
                            participant_containment_floor=(
                                context.config
                                .contradiction_participant_containment_floor
                            ),
                            object_containment_floor=(
                                context.config
                                .contradiction_object_containment_floor
                            ),
                        )
                        hypothesis_bindings.append({
                            "query": hypothesis,
                            "identity": hypothesis_identity,
                            "states": hypothesis_states,
                            "anchor": hypothesis_anchor,
                            "critical": hypothesis_critical,
                            "score": (
                                0.62 * hypothesis_identity
                                + 0.38 * hypothesis_anchor
                            ),
                        })
                    hypothesis_bindings.sort(
                        key=lambda item: (
                            -float(item["score"]),
                            clean_text(item["query"].get("id", "")),
                        )
                    )
                    active_binding = hypothesis_bindings[0]
                    active_query = active_binding["query"]
                    identity = active_binding["identity"]
                    states = active_binding["states"]
                    anchor = active_binding["anchor"]
                    critical = active_binding["critical"]
                    event_score = 0.62 * identity + 0.38 * anchor
                    q_state = clean_text(
                        active_query["signature"].get(
                            "semantic_state_value", ""
                        )
                    )
                    c_state = clean_text(
                        record.signature.get("semantic_state_value", "")
                    )
                    q_var = clean_text(
                        active_query["signature"].get(
                            "semantic_state_variable", ""
                        )
                    )
                    c_var = clean_text(
                        record.signature.get("semantic_state_variable", "")
                    )
                    q_pol = clean_text(
                        active_query["signature"].get(
                            "assertion_polarity", ""
                        )
                    )
                    scoped_epistemic = resolve_epistemic_scope(
                        record.text,
                        record.signature.get("epistemic_state", ""),
                    )
                    epistemic = clean_text(
                        scoped_epistemic["epistemic_state"]
                    )
                    approximation_fields = list(
                        scoped_epistemic["numeric_approximation_fields"]
                    )
                    c_pol = clean_text(
                        record.signature.get("assertion_polarity", "")
                    )
                    if (
                        c_pol == "uncertain"
                        and epistemic == "asserted"
                        and approximation_fields
                    ):
                        c_pol = infer_assertion_polarity(record.text)
                    local_conflict_type = ""
                    local_conflict_audit: Dict[str, Any] = {
                        "screened_for_raw_local_conflict": False,
                        "screen_reason": "no_explicit_dimension_difference",
                    }
                    local_query_text = fragment_text_by_id.get(
                        clean_text(active_query.get("fragment_id", "")),
                        active_query["text"],
                    )
                    query_dimensions = _raw_surface_dimensions(local_query_text)
                    candidate_dimensions = _raw_surface_dimensions(record.text)
                    has_dimension_difference = bool(
                        (
                            query_dimensions[0]
                            and candidate_dimensions[0]
                            and query_dimensions[0] != candidate_dimensions[0]
                        )
                        or (
                            query_dimensions[1]
                            and candidate_dimensions[1]
                            and query_dimensions[1] != candidate_dimensions[1]
                        )
                    )
                    plausible_dimension_binding = bool(
                        states.get("actor_state") == "MATCH"
                        or states.get("predicate_state") == "MATCH"
                        or event_score >= 0.25
                        or anchor >= 0.30
                    )
                    if has_dimension_difference and plausible_dimension_binding:
                        local_conflict_type, local_conflict_audit = (
                            _raw_local_conflict(
                                local_query_text,
                                record.text,
                                fragment_signature_by_id.get(
                                    clean_text(
                                        active_query.get("fragment_id", "")
                                    ),
                                    active_query["signature"],
                                ),
                                record.signature,
                                query_similarity=score_text(
                                    local_query_text,
                                    record.text,
                                    context.semantic_resources,
                                ),
                                anchor_score=anchor,
                                typed_actor_match=(
                                    states.get("actor_state") == "MATCH"
                                ),
                                typed_predicate_match=(
                                    states.get("predicate_state") == "MATCH"
                                ),
                            )
                        )
                        local_conflict_audit[
                            "screened_for_raw_local_conflict"
                        ] = True
                        local_conflict_audit["screen_reason"] = (
                            "explicit_date_or_numeric_set_difference"
                        )
                    elif has_dimension_difference:
                        local_conflict_audit["screen_reason"] = (
                            "dimension_difference_without_event_binding"
                        )
                    (
                        explicit_conflict,
                        conflict_fields,
                        dimension_binding_audit,
                    ) = _dimension_exclusion_conflict(
                        [
                            hypothesis["signature"]
                            for hypothesis in query.get(
                                "hypotheses", [query]
                            )
                        ],
                        [record.signature],
                        identity_floor=(
                            context.config.contradiction_identity_floor
                        ),
                        minimum_margin=context.config.minimum_margin,
                        resources=context.semantic_resources,
                    )
                    fallback_conflict, fallback_fields = _explicit_conflict(
                        active_query["signature"],
                        record.signature,
                        states,
                    )
                    if not explicit_conflict and fallback_conflict:
                        explicit_conflict = fallback_conflict
                        conflict_fields = fallback_fields
                        dimension_binding_audit = {
                            "contract": "explicit_argument_conflict",
                            "state": "CONSERVATIVE_CONFLICT",
                            "conflict_dimensions": list(fallback_fields),
                        }
                    conflict_anchor_contract_pass = (
                        _conflict_anchor_contract_satisfied(
                            explicit_conflict,
                            conflict_fields,
                            states,
                            critical,
                            event_score=event_score,
                            anchor_score=anchor,
                        )
                    )
                    polarity_anchor_contract_pass = (
                        _polarity_anchor_contract_satisfied(
                            states,
                            event_score=event_score,
                            anchor_score=anchor,
                            identity_floor=(
                                context.config.contradiction_identity_floor
                            ),
                        )
                    )
                    related_different_event = (
                        _related_different_event(
                            active_query["signature"], record.signature, states
                        )
                        if not (
                            explicit_conflict and conflict_anchor_contract_pass
                        )
                        else False
                    )
                    opposition_precedes_conflict = (
                        _opposition_precedes_argument_conflict(
                            explicit_conflict=explicit_conflict,
                            conflict_fields=conflict_fields,
                            query_polarity=q_pol,
                            candidate_polarity=c_pol,
                            critical_anchor_gate_pass=bool(
                                critical["critical_anchor_gate_pass"]
                            ),
                        )
                    )
                    uncertainty_gate_pass = False
                    uncertainty_gate_reason = "not_an_uncertainty_candidate"
                    if (
                        explicit_conflict
                        and not opposition_precedes_conflict
                        and epistemic
                        != "uncertain_or_insufficient_information"
                        and (
                            bool(critical["critical_anchor_gate_pass"])
                            or conflict_anchor_contract_pass
                        )
                    ):
                        primary = explicit_conflict
                        reason_prefix = (
                            "same_event_dimension_exclusion_conflict"
                            if dimension_binding_audit.get("contract")
                            == "dimension_exclusion_event_binding"
                            else "same_event_explicit_argument_conflict"
                        )
                        reason = (
                            reason_prefix + ":" + ",".join(conflict_fields)
                        )
                    elif (
                        q_pol
                        and c_pol
                        and q_pol != c_pol
                        and polarity_anchor_contract_pass
                    ):
                        primary = "DIRECT_OPPOSITION_SIDE_B"
                        reason = (
                            "same_event_incompatible_polarity_with_"
                            "complete_role_anchor_contract"
                        )
                    elif local_conflict_type:
                        # Raw date/number extraction is a rescue route for
                        # parser omissions. It must not replace a stronger
                        # structured argument or polarity decision.
                        primary = {
                            "RAW_LOCAL_TEMPORAL_CONFLICT": (
                                "ATTRIBUTE_CONTRADICTION"
                            ),
                            "RAW_LOCAL_NUMERIC_CONFLICT": (
                                "DIRECT_NUMERICAL_CONTRADICTION"
                            ),
                            "RAW_LOCAL_POLARITY_CONFLICT": (
                                "DIRECT_OPPOSITION_SIDE_B"
                            ),
                        }.get(local_conflict_type, "DIRECT_OPPOSITION_SIDE_B")
                        reason = local_conflict_type.casefold()
                    elif not critical["critical_anchor_gate_pass"]:
                        if (
                            q_var and c_var and q_var == c_var and q_state and c_state
                            and q_state != c_state
                            and polarity_anchor_contract_pass
                        ):
                            primary = "DIRECT_OPPOSITION_SIDE_B"
                            reason = (
                                "same_event_incompatible_state_with_"
                                "high_identity_anchor_contract"
                            )
                        elif (
                            q_pol and c_pol and q_pol != c_pol
                            and polarity_anchor_contract_pass
                        ):
                            primary = "DIRECT_OPPOSITION_SIDE_B"
                            reason = (
                                "same_event_incompatible_polarity_with_"
                                "high_identity_anchor_contract"
                            )
                        else:
                            primary = (
                                "BACKGROUND"
                                if retrieval_scores.get(doc_id, 0.0)
                                >= context.config.contradiction_background_retrieval_floor
                                else "REJECTED_EVENT_MISMATCH"
                            )
                            reason = "critical_participant_event_or_date_anchor_failed"
                    elif epistemic == "uncertain_or_insufficient_information":
                        (
                            uncertainty_gate_pass,
                            uncertainty_gate_reason,
                        ) = _uncertainty_same_event_gate(
                            states,
                            critical,
                            event_score=event_score,
                            anchor_score=anchor,
                            identity_floor=(
                                context.config.contradiction_identity_floor
                            ),
                            anchor_floor=(
                                context.config.contradiction_anchor_floor
                            ),
                        )
                        if uncertainty_gate_pass:
                            primary = (
                                "UNCERTAINTY_OR_INSUFFICIENT_INFORMATION"
                            )
                            reason = (
                                "same_event_source_reports_knowledge_"
                                "uncertainty"
                            )
                        elif related_different_event:
                            primary = "RELATED_DIFFERENT_EVENT"
                            reason = (
                                "uncertainty_candidate_failed_same_event_gate:"
                                + uncertainty_gate_reason
                            )
                        else:
                            primary = "REJECTED_EVENT_MISMATCH"
                            reason = (
                                "uncertainty_candidate_failed_same_event_gate:"
                                + uncertainty_gate_reason
                            )
                    elif related_different_event:
                        primary = "RELATED_DIFFERENT_EVENT"
                        reason = (
                            "event_family_matches_but_query_discriminator_"
                            "is_missing_or_different"
                        )
                    elif (
                        event_score < context.config.contradiction_identity_floor
                        or anchor < context.config.contradiction_anchor_floor
                    ):
                        primary = "REJECTED_EVENT_MISMATCH"
                        reason = "same_event_identity_floor_not_met"
                    elif q_var and c_var and q_var == c_var and q_state and c_state:
                        if q_state == c_state:
                            primary = "DIRECT_SUPPORT_SIDE_A"
                            reason = "same_event_compatible_semantic_state"
                        else:
                            primary = "OPPOSING_STATE_PENDING_TIME"
                            reason = "same_event_incompatible_semantic_state"
                    elif q_pol and c_pol and q_pol == c_pol:
                        primary = "DIRECT_SUPPORT_SIDE_A"
                        reason = "same_event_compatible_polarity"
                    elif q_pol and c_pol and q_pol != c_pol:
                        primary = "OPPOSING_STATE_PENDING_TIME"
                        reason = "same_event_incompatible_polarity"
                    else:
                        primary = "QUALIFICATION"
                        reason = "same_event_related_evidence"
                    interim.append({
                        "query_proposition_id": query["id"],
                        "query_hypothesis_group_id": clean_text(
                            query.get("hypothesis_group_id", "")
                        ),
                        "query_hypothesis_count": len(
                            query.get("hypotheses", [query])
                        ),
                        "winning_query_hypothesis_id": clean_text(
                            active_query.get(
                                "proposition_hypothesis_id",
                                active_query.get("id", ""),
                            )
                        ),
                        "query_fragment_id": query["fragment_id"],
                        "query_proposition_text": query["text"],
                        "query_event_time":
                            clean_text(
                                active_query["signature"].get(
                                    "event_time", ""
                                )
                            ),
                        "corpus_doc_id": doc_id,
                        "corpus_source_table": record.source_table,
                        "corpus_row_id": record.row_id,
                        "corpus_proposition_text": record.text,
                        "primary_class": primary,
                        "retrieval_routes": ";".join(
                            sorted(retrieval_routes.get(doc_id, set()))
                        ),
                        "retrieval_score": retrieval_scores.get(doc_id, 0.0),
                        "event_identity_score": event_score,
                        "critical_anchor_score": anchor,
                        **critical,
                        **states,
                        "conflict_fields_json": json.dumps(conflict_fields),
                        "dimension_exclusion_binding_json": json.dumps(
                            dimension_binding_audit,
                            sort_keys=True,
                            default=str,
                        ),
                        "conflict_anchor_contract_pass":
                            conflict_anchor_contract_pass,
                        "polarity_anchor_contract_pass":
                            polarity_anchor_contract_pass,
                        "query_vehicle": clean_text(
                            active_query["signature"].get(
                                "vehicle_surface", ""
                            )
                        ),
                        "corpus_vehicle": clean_text(
                            record.signature.get("vehicle_surface", "")
                        ),
                        "query_state_variable": q_var,
                        "query_state_value": q_state,
                        "corpus_state_variable": c_var,
                        "corpus_state_value": c_state,
                        "query_polarity": q_pol,
                        "corpus_polarity": c_pol,
                        "epistemic_state": epistemic,
                        "epistemic_scope_reason": clean_text(
                            scoped_epistemic["epistemic_scope_reason"]
                        ),
                        "numeric_approximation_fields_json": json.dumps(
                            approximation_fields, sort_keys=True
                        ),
                        "uncertainty_same_event_gate_pass":
                            uncertainty_gate_pass,
                        "uncertainty_same_event_gate_reason":
                            uncertainty_gate_reason,
                        "event_time":
                            clean_text(record.signature.get("event_time", "")),
                        "report_time": record.report_time
                        or context.corpus.document_dates.get(doc_id, ""),
                        "raw_validation_status": "RAW_NOT_ASSESSED",
                        "raw_sentence": "",
                        "raw_similarity": 0.0,
                        "raw_local_conflict_audit_json": json.dumps(
                            local_conflict_audit,
                            sort_keys=True,
                        ),
                        "raw_local_conflict_type": local_conflict_type,
                        "final_score": (
                            0.65 * event_score
                            + 0.25 * retrieval_scores.get(doc_id, 0.0)
                            + 0.10 * float(
                                critical["critical_anchor_gate_pass"]
                                or bool(local_conflict_type)
                            )
                        ),
                        "decision_reason": reason,
                    })

            context.control.check(len(interim))
            interim.sort(
                key=lambda row: (-float(row["final_score"]), row["corpus_doc_id"])
            )
            for finalist in interim[: context.config.raw_validation_finalists]:
                sentences = context.corpus.sentences_by_doc.get(
                    finalist["corpus_doc_id"], ()
                )
                score, sentence = max(
                    (
                        (
                            score_text(
                                finalist["corpus_proposition_text"],
                                item.sentence,
                                context.semantic_resources,
                            ),
                            item.sentence,
                        )
                        for item in sentences
                    ),
                    default=(0.0, ""),
                )
                finalist["raw_similarity"] = score
                finalist["raw_sentence"] = sentence
                finalist["raw_query_similarity"] = 0.0
                if (
                    not sentence
                    or score < context.config.contradiction_raw_validation_floor
                ):
                    finalist["raw_validation_status"] = "RAW_CANDIDATE_NOT_FOUND"
                elif finalist["primary_class"] == "DIRECT_SUPPORT_SIDE_A":
                    finalist["raw_validation_status"] = (
                        "RAW_DIRECT_SUPPORT_CONFIRMED"
                    )
                else:
                    finalist["raw_validation_status"] = (
                        "RAW_CANDIDATE_PROPOSITION_CONFIRMED"
                    )
                if sentence:
                    raw_bindings: List[Dict[str, Any]] = []
                    raw_signature = _normalize_concern_relation_signature(
                        sentence,
                        build_event_signature(sentence),
                    )
                    for hypothesis in query.get("hypotheses", [query]):
                        local_query_text = fragment_text_by_id.get(
                            clean_text(hypothesis.get("fragment_id", "")),
                            hypothesis["text"],
                        )
                        query_signature = _normalize_concern_relation_signature(
                            hypothesis.get("text", ""),
                            hypothesis["signature"],
                        )
                        local_query_signature = dict(query_signature)
                        fragment_signature = fragment_signature_by_id.get(
                            clean_text(hypothesis.get("fragment_id", "")),
                            {},
                        )
                        for field in ("event_time", "date_value"):
                            if not clean_text(
                                local_query_signature.get(field, "")
                            ):
                                local_query_signature[field] = clean_text(
                                    fragment_signature.get(field, "")
                                )
                        raw_identity, raw_states = event_identity(
                            query_signature,
                            raw_signature,
                            hypothesis["text"],
                            sentence,
                            context.semantic_resources,
                        )
                        raw_anchor = _anchor_score(
                            hypothesis["text"],
                            sentence,
                            context.semantic_resources,
                        )
                        raw_critical = _critical_anchors(
                            query_signature,
                            raw_signature,
                            hypothesis["text"],
                            sentence,
                            context.semantic_resources,
                            participant_containment_floor=(
                                context.config
                                .contradiction_participant_containment_floor
                            ),
                            object_containment_floor=(
                                context.config
                                .contradiction_object_containment_floor
                            ),
                        )
                        raw_query_similarity = score_text(
                            hypothesis["text"],
                            sentence,
                            context.semantic_resources,
                        )
                        raw_bindings.append({
                            "query_text": hypothesis["text"],
                            "local_query_text": local_query_text,
                            "query_signature": query_signature,
                            "local_query_signature": local_query_signature,
                            "raw_signature": raw_signature,
                            "identity": raw_identity,
                            "states": raw_states,
                            "anchor": raw_anchor,
                            "critical": raw_critical,
                            "query_similarity": raw_query_similarity,
                            "event_score": (
                                0.62 * raw_identity + 0.38 * raw_anchor
                            ),
                        })
                    raw_bindings.sort(
                        key=lambda item: (
                            -float(item["event_score"]),
                            -float(item["query_similarity"]),
                        )
                    )
                    raw_binding = raw_bindings[0]
                    finalist["raw_query_similarity"] = raw_binding[
                        "query_similarity"
                    ]
                    finalist["raw_event_identity_score"] = raw_binding[
                        "event_score"
                    ]
                    finalist["raw_anchor_score"] = raw_binding["anchor"]
                    finalist["raw_critical_anchor_gate_pass"] = raw_binding[
                        "critical"
                    ]["critical_anchor_gate_pass"]
                    raw_query_polarity = clean_text(
                        raw_binding["query_signature"].get(
                            "assertion_polarity", ""
                        )
                    )
                    raw_corpus_polarity = clean_text(
                        raw_binding["raw_signature"].get(
                            "assertion_polarity", ""
                        )
                    )
                    raw_opposition_confirmed = bool(
                        raw_query_polarity
                        and raw_corpus_polarity
                        and raw_query_polarity != raw_corpus_polarity
                        and raw_binding["critical"][
                            "critical_anchor_gate_pass"
                        ]
                        and raw_binding["anchor"]
                        >= context.config.contradiction_anchor_floor
                        and raw_binding["query_similarity"] >= 0.50
                        and (
                            raw_binding["event_score"]
                            >= context.config.contradiction_identity_floor
                            or (
                                raw_binding["query_similarity"] >= 0.52
                                and raw_binding["anchor"] >= 0.66
                            )
                            )
                        )
                    raw_local_conflict_type, raw_local_conflict_audit = (
                        _raw_local_conflict(
                            raw_binding["local_query_text"],
                            clean_text(" ".join((
                                finalist.get(
                                    "corpus_proposition_text", ""
                                ),
                                sentence,
                            ))),
                            raw_binding["local_query_signature"],
                            {
                                **raw_binding["raw_signature"],
                                "event_time": clean_text(
                                    raw_binding["raw_signature"].get(
                                        "event_time", ""
                                    )
                                ) or clean_text(
                                    finalist.get("event_time", "")
                                ),
                            },
                            query_similarity=float(
                                raw_binding["query_similarity"]
                            ),
                            anchor_score=float(raw_binding["anchor"]),
                            typed_actor_match=(
                                finalist.get("actor_state") == "MATCH"
                            ),
                            typed_predicate_match=(
                                finalist.get("predicate_state") == "MATCH"
                            ),
                        )
                    )
                    finalist["raw_local_conflict_type"] = (
                        raw_local_conflict_type
                    )
                    finalist["raw_local_conflict_audit_json"] = json.dumps(
                        raw_local_conflict_audit,
                        sort_keys=True,
                    )
                    existing_explicit_conflict = finalist[
                        "primary_class"
                    ] in {
                        "ATTRIBUTE_CONTRADICTION",
                        "DIRECT_NUMERICAL_CONTRADICTION",
                        "DIRECT_OPPOSITION_SIDE_B",
                        "LOCATION_CONTRADICTION",
                    }
                    if raw_opposition_confirmed and not existing_explicit_conflict:
                        finalist.update(raw_binding["states"])
                        finalist.update(raw_binding["critical"])
                        finalist["event_identity_score"] = raw_binding[
                            "event_score"
                        ]
                        finalist["critical_anchor_score"] = raw_binding[
                            "anchor"
                        ]
                        finalist["query_polarity"] = raw_query_polarity
                        finalist["corpus_polarity"] = raw_corpus_polarity
                        finalist["primary_class"] = (
                            "DIRECT_OPPOSITION_SIDE_B"
                        )
                        finalist["decision_reason"] = (
                            "raw_sentence_same_event_opposite_polarity"
                        )
                        finalist["raw_validation_status"] = (
                            "RAW_DIRECT_OPPOSITION_CONFIRMED"
                        )
                        finalist["final_score"] = max(
                            float(finalist.get("final_score", 0.0) or 0.0),
                            0.65 * raw_binding["event_score"]
                            + 0.25 * float(
                                finalist.get("retrieval_score", 0.0) or 0.0
                            )
                            + 0.10,
                        )
                    elif raw_local_conflict_type and (
                        not existing_explicit_conflict
                        or (
                            raw_local_conflict_type
                            == "RAW_LOCAL_POLARITY_CONFLICT"
                            and finalist["primary_class"]
                            == "ATTRIBUTE_CONTRADICTION"
                            and set(json.loads(
                                finalist.get("conflict_fields_json", "[]")
                                or "[]"
                            )) <= {"predicate"}
                        )
                    ):
                        finalist.update(raw_binding["states"])
                        finalist["event_identity_score"] = raw_binding[
                            "event_score"
                        ]
                        finalist["critical_anchor_score"] = raw_binding[
                            "anchor"
                        ]
                        finalist["primary_class"] = {
                            "RAW_LOCAL_TEMPORAL_CONFLICT": (
                                "ATTRIBUTE_CONTRADICTION"
                            ),
                            "RAW_LOCAL_NUMERIC_CONFLICT": (
                                "DIRECT_NUMERICAL_CONTRADICTION"
                            ),
                            "RAW_LOCAL_POLARITY_CONFLICT": (
                                "DIRECT_OPPOSITION_SIDE_B"
                            ),
                        }.get(
                            raw_local_conflict_type,
                            "DIRECT_OPPOSITION_SIDE_B",
                        )
                        finalist["decision_reason"] = (
                            raw_local_conflict_type.casefold()
                        )
                        finalist["raw_validation_status"] = (
                            "RAW_DIRECT_OPPOSITION_CONFIRMED"
                        )
                        finalist["final_score"] = max(
                            float(finalist.get("final_score", 0.0) or 0.0),
                            0.65 * raw_binding["event_score"]
                            + 0.25 * float(
                                finalist.get("retrieval_score", 0.0) or 0.0
                            )
                            + 0.10,
                        )

            side_a_dates = [
                clean_text(row["report_time"])
                for row in interim
                if row["primary_class"] == "DIRECT_SUPPORT_SIDE_A"
                and clean_text(row["report_time"])
            ]
            latest_side_a = max(side_a_dates) if side_a_dates else ""
            for row in interim:
                if row["primary_class"] == "OPPOSING_STATE_PENDING_TIME":
                    if latest_side_a and clean_text(row["report_time"]) > latest_side_a:
                        row["primary_class"] = "TEMPORAL_UPDATE_OR_SOURCE_REVISION"
                        row["decision_reason"] = (
                            "later_report_changes_state_for_same_event_date"
                        )
                    else:
                        row["primary_class"] = "DIRECT_OPPOSITION_SIDE_B"
                        row["decision_reason"] = (
                            "same_event_incompatible_state_without_later_report"
                        )

            # Preserve the best proposition per document and evidence class
            # before reducing each document to one primary class.
            document_class_best: Dict[
                Tuple[str, str], Dict[str, Any]
            ] = {}
            for row in sorted(
                interim,
                key=lambda item: (
                    -float(item["final_score"]),
                    clean_text(item["corpus_source_table"]),
                    clean_text(item["corpus_row_id"]),
                ),
            ):
                key = (
                    clean_text(row["corpus_doc_id"]),
                    clean_text(row["primary_class"]),
                )
                document_class_best.setdefault(key, row)
            for (doc_id, evidence_class), row in sorted(
                document_class_best.items()
            ):
                document_class_rows.append({
                    **row,
                    "precollapse_document_class": evidence_class,
                    "precollapse_contract": (
                        "best_candidate_per_document_and_evidence_class"
                    ),
                })

            # One coherent primary decision per document.  All direct conflict
            # forms are equally evidentiary and must compete on their grounded
            # alignment score.  Giving ATTRIBUTE_CONTRADICTION precedence over
            # DIRECT_OPPOSITION allowed a partial proposition to displace an
            # exact, higher-scoring polarity conflict from the same document.
            # Evidence-bearing classes still outrank contextual/background rows.
            class_priority = {
                "DIRECT_SUPPORT_SIDE_A": 0,
                "DIRECT_NUMERICAL_CONTRADICTION": 0,
                "ATTRIBUTE_CONTRADICTION": 0,
                "LOCATION_CONTRADICTION": 0,
                "DIRECT_OPPOSITION_SIDE_B": 0,
                "TEMPORAL_UPDATE_OR_SOURCE_REVISION": 1,
                "UNCERTAINTY_OR_INSUFFICIENT_INFORMATION": 3,
                "SCOPE_OR_CARDINALITY_DIFFERENCE": 4,
                "QUALIFICATION": 4,
                "RELATED_DIFFERENT_EVENT": 5,
                "BACKGROUND": 8,
                "REJECTED_EVENT_MISMATCH": 9,
            }
            interim.sort(key=lambda row: (
                class_priority.get(row["primary_class"], 7),
                -float(row["final_score"]),
                row["corpus_doc_id"],
            ))
            per_doc: Dict[str, Dict[str, Any]] = {}
            for row in interim:
                per_doc.setdefault(row["corpus_doc_id"], row)
            for (doc_id, evidence_class), row in sorted(
                document_class_best.items()
            ):
                selected = per_doc.get(doc_id, {})
                collapse_rows.append({
                    "query_proposition_id": query["id"],
                    "corpus_doc_id": doc_id,
                    "candidate_class": evidence_class,
                    "candidate_source_table": row["corpus_source_table"],
                    "candidate_row_id": row["corpus_row_id"],
                    "candidate_score": row["final_score"],
                    "selected_document_class": clean_text(
                        selected.get("primary_class", "")
                    ),
                    "selected_source_table": clean_text(
                        selected.get("corpus_source_table", "")
                    ),
                    "selected_row_id": clean_text(
                        selected.get("corpus_row_id", "")
                    ),
                    "selected": (
                        clean_text(selected.get("primary_class", ""))
                        == evidence_class
                    ),
                    "collapse_reason": (
                        "lowest_class_priority_then_highest_score"
                    ),
                })
            rows = list(per_doc.values())
            all_rows.extend(rows)
            by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_class[row["primary_class"]].append(row)
            for values in by_class.values():
                values.sort(
                    key=lambda row: (-float(row["final_score"]), row["corpus_doc_id"])
                )

            expected_classes = {
                "SIDE_A": ["DIRECT_SUPPORT_SIDE_A"],
                "SIDE_B_OR_UPDATE": [
                    "DIRECT_OPPOSITION_SIDE_B",
                    "TEMPORAL_UPDATE_OR_SOURCE_REVISION",
                    "DIRECT_NUMERICAL_CONTRADICTION",
                    "ATTRIBUTE_CONTRADICTION",
                    "LOCATION_CONTRADICTION",
                ],
                "UNCERTAINTY": [
                    "UNCERTAINTY_OR_INSUFFICIENT_INFORMATION"
                ],
                "QUALIFICATION_OR_SCOPE": [
                    "QUALIFICATION", "SCOPE_OR_CARDINALITY_DIFFERENCE",
                    "RELATED_DIFFERENT_EVENT",
                ],
                "BACKGROUND": ["BACKGROUND", "REJECTED_EVENT_MISMATCH"],
            }
            for evidence_class, labels in expected_classes.items():
                found = [
                    row for label in labels for row in by_class.get(label, [])
                ]
                coverage_rows.append({
                    "query_proposition_id": query["id"],
                    "evidence_class": evidence_class,
                    "retrieval_status": (
                        "RETRIEVED_AND_CLASSIFIED"
                        if found else "NOT_RETRIEVED_OR_REJECTED"
                    ),
                    "document_count": len(found),
                    "top_doc_id": (
                        max(found, key=lambda row: float(row["final_score"]))[
                            "corpus_doc_id"
                        ]
                        if found else ""
                    ),
                    "reason": (
                        "class_has_eligible_document"
                        if found
                        else "no_document_survived_class_specific_gates"
                    ),
                })

            side_a = (by_class.get("DIRECT_SUPPORT_SIDE_A") or [{}])[0]
            changes = (
                by_class.get("TEMPORAL_UPDATE_OR_SOURCE_REVISION", [])
                + by_class.get("DIRECT_OPPOSITION_SIDE_B", [])
                + by_class.get("DIRECT_NUMERICAL_CONTRADICTION", [])
                + by_class.get("ATTRIBUTE_CONTRADICTION", [])
                + by_class.get("LOCATION_CONTRADICTION", [])
            )
            changes.sort(
                key=lambda row: (
                    # Eligible conflict findings first compete on hard event
                    # anchors. A softer aggregate score must not let a generic
                    # government/actor overlap displace a candidate that binds
                    # more of the query's actual event structure.
                    -float(row.get("critical_anchor_score", 0.0) or 0.0),
                    -float(row.get("retrieval_score", 0.0) or 0.0),
                    -float(row.get("raw_query_similarity", 0.0) or 0.0),
                    -float(row.get("final_score", 0.0) or 0.0),
                    row.get("corpus_doc_id", ""),
                )
            )
            change = changes[0] if changes else {}
            uncertainty = (
                by_class.get("UNCERTAINTY_OR_INSUFFICIENT_INFORMATION") or [{}]
            )[0]
            qualification = (
                by_class.get("QUALIFICATION", [])
                + by_class.get("SCOPE_OR_CARDINALITY_DIFFERENCE", [])
            )
            qualification.sort(
                key=lambda row: -float(row.get("final_score", 0.0))
            )
            qualification_row = qualification[0] if qualification else {}
            related_different = (
                by_class.get("RELATED_DIFFERENT_EVENT") or [{}]
            )[0]
            background_candidates = (
                by_class.get("BACKGROUND", [])
                + by_class.get("REJECTED_EVENT_MISMATCH", [])
            )
            background_candidates.sort(
                key=lambda row: -float(row.get("final_score", 0.0))
            )
            background = (
                background_candidates[0] if background_candidates else {}
            )
            parts = []
            if side_a:
                parts.append("SIDE_A_SUPPORT")
            if change:
                parts.append(clean_text(change.get("primary_class", "")))
            if uncertainty:
                parts.append("UNCERTAINTY_EVIDENCE")
            if qualification_row:
                parts.append("QUALIFICATION_OR_SCOPE")
            if related_different:
                parts.append("RELATED_DIFFERENT_EVENT")
            pattern = " -> ".join(parts) or "NO_RELEVANT_EVIDENCE"
            role_contract = _evidence_role_contract(
                pattern=pattern,
                side_a=side_a,
                change=change,
                uncertainty=uncertainty,
                qualification=qualification_row,
                related_different=related_different,
                background=background,
            )
            primary = role_contract.pop("primary")
            has_evidence_finding = bool(
                role_contract.pop("has_evidence_finding")
            )
            # Evidence classes are complementary, not contestants in one
            # winner/runner race.  Export per-class scores and leave the
            # cross-class margin explicitly not applicable.
            runner_score = ""
            best_score = primary.get("final_score", "")
            margin = None
            alternatives = [
                row for row in sorted(
                    rows,
                    key=lambda item: (
                        -float(item.get("final_score", 0.0) or 0.0),
                        clean_text(item.get("corpus_doc_id", "")),
                        clean_text(item.get("corpus_row_id", "")),
                    ),
                )
                if (
                    clean_text(row.get("corpus_doc_id", "")),
                    clean_text(row.get("corpus_row_id", "")),
                ) != (
                    clean_text(primary.get("corpus_doc_id", "")),
                    clean_text(primary.get("corpus_row_id", "")),
                )
            ]
            nearest = alternatives[0] if alternatives else {}
            evaluation_rows.append(task_row(
                "contradictions", query["id"], pattern,
                localization_status=(
                    "AUTHORITATIVE_SOURCE_SELECTED"
                    if primary and primary.get("primary_class") not in {
                        "BACKGROUND", "REJECTED_EVENT_MISMATCH",
                    } else "RELATED_EVIDENCE_FOUND"
                    if rows else "NO_SOURCE_FOUND"
                ),
                raw_status=clean_text(
                    primary.get(
                        "raw_validation_status",
                        "RAW_NOT_FOUND_FOR_SELECTED_EVIDENCE",
                    )
                ),
                # These legacy fields retain their v0.4.6 meanings for pair
                # reconciliation. Matcher v0.4.20 retains the explicit
                # v0.4.9-schema fields below as the authoritative
                # contradiction-result contract.
                selected_text=clean_text(
                    primary.get("corpus_proposition_text", "")
                ),
                selected_source=clean_text(side_a.get("corpus_doc_id", "")),
                top_candidate=clean_text(primary.get("corpus_doc_id", "")),
                related_source=clean_text(change.get("corpus_doc_id", "")),
                background_source=clean_text(
                    background.get("corpus_doc_id", "")
                ),
                **role_contract,
                best_score=best_score,
                runner_up=runner_score,
                margin=margin,
                decision_competitor_scope=(
                    "not_applicable_complementary_evidence_classes"
                ),
                nearest_alternative_score=nearest.get("final_score"),
                nearest_alternative_doc_id=clean_text(
                    nearest.get("corpus_doc_id", "")
                ),
                nearest_alternative_passage_id=clean_text(
                    nearest.get("corpus_row_id", "")
                ),
                nearest_alternative_eligible=(
                    clean_text(nearest.get("primary_class", ""))
                    not in {"BACKGROUND", "REJECTED_EVENT_MISMATCH"}
                    if nearest else None
                ),
                nearest_alternative_failed_gates=clean_text(
                    nearest.get("failed_gates", "")
                ),
                reason=(
                    "grouped_same_event_support_update_uncertainty_and_scope"
                ),
                unresolved=(
                    "" if has_evidence_finding
                    else "NO_ELIGIBLE_CONTRADICTION_EVIDENCE"
                ),
                failed=(
                    "" if has_evidence_finding
                    else "CONTRADICTION_EVIDENCE_GATE"
                ),
            ))
            context.control.emit(
                "contradictions_target_complete",
                target_id=query["id"],
                candidate_count=len(interim),
                document_decision_count=len(rows),
                retrieval_route_count=len(_retrieval_routes(query)),
                status=pattern,
            )

        audit = pd.DataFrame(all_rows)
        if len(audit):
            audit["eligible"] = ~audit[
                "primary_class"
            ].fillna("").astype(str).isin({
                "BACKGROUND", "REJECTED_EVENT_MISMATCH",
            })
        primary_parts = []
        if len(audit):
            for label in (
                "DIRECT_SUPPORT_SIDE_A",
                "TEMPORAL_UPDATE_OR_SOURCE_REVISION",
                "DIRECT_OPPOSITION_SIDE_B",
                "DIRECT_NUMERICAL_CONTRADICTION",
                "ATTRIBUTE_CONTRADICTION", "LOCATION_CONTRADICTION",
                "UNCERTAINTY_OR_INSUFFICIENT_INFORMATION",
                "QUALIFICATION", "SCOPE_OR_CARDINALITY_DIFFERENCE",
                "RELATED_DIFFERENT_EVENT",
                "BACKGROUND", "REJECTED_EVENT_MISMATCH",
            ):
                subset = audit[
                    audit["primary_class"].astype(str).eq(label)
                ].sort_values("final_score", ascending=False).head(10)
                if len(subset):
                    primary_parts.append(subset)
        primary = (
            pd.concat(primary_parts, ignore_index=True)
            if primary_parts else pd.DataFrame()
        )
        return EngineResult(
            task=self.task_name,
            evaluation=pd.DataFrame(evaluation_rows),
            artifacts={
                "contradiction_candidate_audit_v048.csv": audit,
                "contradiction_candidates_v048.csv": primary,
                "contradiction_retrieval_coverage_v048.csv":
                    pd.DataFrame(coverage_rows),
                "contradiction_document_class_trace_v0420.csv":
                    pd.DataFrame(document_class_rows),
                "contradiction_collapse_trace_v0420.csv":
                    pd.DataFrame(collapse_rows),
                "candidate_budget_audit_v0420.csv":
                    pd.DataFrame(budget_rows),
            },
            metadata={
                "semantic_resource_fingerprint":
                    context.semantic_resources.fingerprint,
                "candidate_budget_contract":
                    "bounded_per_document_representation_and_semantic_class",
                "canonical_per_class_per_document":
                    context.config
                    .contradiction_propositions_per_class_per_document,
                "raw_per_class_per_document":
                    context.config
                    .contradiction_raw_sentences_per_class_per_document,
            },
        )
