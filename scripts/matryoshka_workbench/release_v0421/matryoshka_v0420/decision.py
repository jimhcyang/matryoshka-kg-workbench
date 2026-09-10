"""Shared noncompensatory semantic gates and deterministic ranking."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .retrieval import containment, jaccard, score_text
from .semantic_resources import SemanticResources, shared_concepts
from .semantics import (
    canonical_key,
    canonical_location,
    clean_text,
    compare_argument,
    compare_dates,
    compare_numeric,
    semantic_tokens,
)


CRITICAL_ARGUMENTS = {
    "predicate_state", "object_state", "location_state", "date_state",
    "packet_count_state", "total_weight_grams_state", "vehicle_state",
}


def _semantic_argument_state(
    query_value: Any,
    candidate_value: Any,
    resources: Optional[SemanticResources] = None,
) -> str:
    state = compare_argument(query_value, candidate_value)
    if state == "MATCH" or resources is None:
        return state
    query_text = clean_text(query_value)
    candidate_text = clean_text(candidate_value)
    if not query_text or not candidate_text:
        return state
    if shared_concepts(resources, query_text, candidate_text, "alias"):
        return "MATCH"
    if shared_concepts(resources, query_text, candidate_text, "event"):
        return "MATCH"
    return state


def event_identity(
    query_signature: Mapping[str, Any],
    candidate_signature: Mapping[str, Any],
    query_text: str,
    candidate_text: str,
    resources: Optional[SemanticResources] = None,
) -> Tuple[float, Dict[str, str]]:
    states = {
        "actor_state": _semantic_argument_state(
            query_signature.get("actor", ""), candidate_signature.get("actor", "")
            , resources
        ),
        "predicate_state": _semantic_argument_state(
            query_signature.get("predicate", ""), candidate_signature.get("predicate", "")
            , resources
        ),
        "object_state": _semantic_argument_state(
            query_signature.get("object", ""), candidate_signature.get("object", "")
            , resources
        ),
        "location_state": _semantic_argument_state(
            query_signature.get("location_identity", query_signature.get("location", "")),
            candidate_signature.get(
                "location_identity", candidate_signature.get("location", "")
            ),
            resources,
        ),
        "date_state": compare_dates(
            query_signature.get("date_value", query_signature.get("event_time", "")),
            candidate_signature.get(
                "date_value", candidate_signature.get("event_time", "")
            ),
        ),
        "packet_count_state": compare_numeric(
            query_signature.get("packet_count"),
            candidate_signature.get("packet_count"),
            tolerance=0.0,
        ),
        "total_weight_grams_state": compare_numeric(
            query_signature.get("total_weight_grams"),
            candidate_signature.get("total_weight_grams"),
            tolerance=1.0,
        ),
        "vehicle_state": _semantic_argument_state(
            query_signature.get("vehicle_identity", ""),
            candidate_signature.get("vehicle_identity", ""),
            resources,
        ),
    }
    query_state_variable = clean_text(
        query_signature.get("semantic_state_variable", "")
    )
    candidate_state_variable = clean_text(
        candidate_signature.get("semantic_state_variable", "")
    )
    query_state_value = clean_text(
        query_signature.get("semantic_state_value", "")
    )
    candidate_state_value = clean_text(
        candidate_signature.get("semantic_state_value", "")
    )
    if (
        query_state_variable
        and query_state_variable == candidate_state_variable
        and query_state_value
        and query_state_value == candidate_state_value
    ):
        # Equivalent normalized states carry the proposition complement. This
        # makes "did not participate" comparable to "did not join" and
        # "public knowledge" comparable to "no longer secret" without
        # treating arbitrary antonyms as synonyms.
        states["object_state"] = "MATCH"
    weights = {
        "actor_state": 0.18,
        "predicate_state": 0.22,
        "object_state": 0.18,
        "location_state": 0.12,
        "date_state": 0.12,
        "packet_count_state": 0.08,
        "total_weight_grams_state": 0.10,
        "vehicle_state": 0.10,
    }
    available = 0.0
    matched = 0.0
    for key, weight in weights.items():
        state = states[key]
        if state in {"MISSING_IN_QUERY", "NOT_APPLICABLE"}:
            continue
        available += weight
        if state == "MATCH":
            matched += weight
        elif state in {"MISSING_IN_CANDIDATE", "UNPARSEABLE"}:
            matched += 0.20 * weight
    lexical = score_text(query_text, candidate_text, resources)
    structural = matched / available if available else lexical
    return float(0.72 * structural + 0.28 * lexical), states


def assertion_gate(
    query_signature: Mapping[str, Any],
    candidate_signature: Mapping[str, Any],
) -> Tuple[bool, str]:
    candidate_epistemic = clean_text(candidate_signature.get("epistemic_state", ""))
    query_uncertain = any(
        "uncertain" in clean_text(query_signature.get(field, "")).casefold()
        for field in ("epistemic_state", "modality", "stance", "speaker_stance")
    )
    if (
        candidate_epistemic == "uncertain_or_insufficient_information"
        and not query_uncertain
    ):
        return False, "UNCERTAIN_CANDIDATE"
    query_variable = clean_text(
        query_signature.get("semantic_state_variable", "")
    )
    candidate_variable = clean_text(
        candidate_signature.get("semantic_state_variable", "")
    )
    query_state = clean_text(query_signature.get("semantic_state_value", ""))
    candidate_state = clean_text(candidate_signature.get("semantic_state_value", ""))
    if (
        query_variable
        and query_variable == candidate_variable
        and query_state
        and candidate_state
    ):
        if query_state != candidate_state:
            return False, "SEMANTIC_STATE_MISMATCH"
        # Surface polarity can differ when two phrases encode the same state,
        # for example "public" versus "no longer secret".
        return True, ""
    query_polarity = clean_text(query_signature.get("assertion_polarity", ""))
    candidate_polarity = clean_text(candidate_signature.get("assertion_polarity", ""))
    if (
        query_polarity
        and candidate_polarity
        and query_polarity != candidate_polarity
        and "uncertain" not in {query_polarity, candidate_polarity}
    ):
        return False, "ASSERTION_POLARITY_MISMATCH"
    return True, ""


def canonical_value(value: str) -> str:
    return canonical_key(value)


_GENERIC_VALUE_PLACEHOLDERS = {
    "unknown",
    "unk",
    "none",
    "na",
    "n a",
    "n/a",
    "not known",
    "not reported",
    "not specified",
    "unnamed",
    "unidentified",
}

_PERSON_DISQUALIFY_TOKENS = {
    "agency",
    "airbase",
    "battalion",
    "base",
    "board",
    "brigade",
    "center",
    "centre",
    "city",
    "command",
    "committee",
    "conference",
    "department",
    "district",
    "division",
    "embassy",
    "event",
    "force",
    "forces",
    "government",
    "house",
    "military",
    "ministry",
    "mission",
    "meeting",
    "office",
    "party",
    "province",
    "regiment",
    "report",
    "reports",
    "school",
    "squadron",
    "states",
    "study",
    "summit",
    "training",
    "union",
}

_PERSON_ROLE_HEAD_RE = re.compile(
    r"^(?:(?:the|a|an|acting|assistant|associate|deputy|former|interim|"
    r"senior|special)\s+)*(?:"
    r"ambassador|bishop|chair|chairman|chairperson|chief|colonel|"
    r"commander|commissioner|consul|director|general|governor|head|"
    r"judge|leader|minister|president|prime\s+minister|professor|"
    r"representative|secretary|spokesman|spokesperson"
    r")\b"
)


def _plausible_person_surface(value: str) -> bool:
    cleaned = clean_text(value)
    key = canonical_key(cleaned)
    if not key or key in _GENERIC_VALUE_PLACEHOLDERS:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", cleaned)
    if not words:
        return False
    # Extractors legitimately represent an unnamed office-holder by a role
    # surface such as "Chief of the Iran Fisheries Company".  Those values
    # are person identities for typed-slot recovery even though an
    # organization token appears inside the title.  Keep the rule bounded to
    # explicit role heads rather than accepting arbitrary long noun phrases.
    if len(words) <= 12 and _PERSON_ROLE_HEAD_RE.match(key):
        return True
    if len(words) > 5:
        return False
    lowered = {word.lower() for word in words}
    if lowered & _PERSON_DISQUALIFY_TOKENS:
        return False
    if len(words) == 1 and words[0].isupper():
        return words[0] not in {"USIS", "USAID", "CIA", "FBI", "UN", "US", "UK"}
    return all(word.isupper() or word[:1].isupper() for word in words)


def type_compatible(expected: str, actual: str, value: str) -> bool:
    expected = clean_text(expected).upper()
    actual = clean_text(actual).upper()
    if not expected or expected in {"ANY", "UNKNOWN"}:
        return True
    if expected in {"ORGANIZATION", "ORG"}:
        return actual in {"ORGANIZATION", "ORG"} or any(
            token in canonical_key(value).split()
            for token in ("police", "ministry", "directorate", "embassy")
        )
    if expected == "PERSON":
        plausible = _plausible_person_surface(value)
        return (actual == "PERSON" and plausible) or (
            actual == "UNKNOWN" and plausible
        )
    if expected == "LOCATION":
        if actual in {
            "ORGANIZATION",
            "ORG",
            "GOVERNMENT",
            "MILITARY",
            "POLITICAL_PARTY",
        }:
            return False
        return actual in {"LOCATION", "GPE", "COUNTRY"} or any(
            token in canonical_key(value).split()
            for token in ("province", "city", "district")
        )
    if expected in {"COUNTRY", "LOCATION_OR_COUNTRY"}:
        if actual in {
            "ORGANIZATION",
            "ORG",
            "GOVERNMENT",
            "MILITARY",
            "POLITICAL_PARTY",
        }:
            return False
        return actual in {"LOCATION", "GPE", "COUNTRY"} or any(
            token in canonical_key(value).split()
            for token in ("province", "city", "district", "country", "state")
        )
    return actual == expected


def deterministic_rank(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_field: str = "final_score",
    identity_fields: Sequence[str] = ("doc_id",),
) -> List[Dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(bool(row.get("eligible", True))),
            -float(row.get(score_field, 0.0) or 0.0),
            *(clean_text(row.get(field, "")) for field in identity_fields),
        ),
    )


def winner_with_margin(
    rows: Sequence[Mapping[str, Any]],
    minimum_score: float,
    minimum_margin: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[float], bool]:
    eligible = [dict(row) for row in rows if bool(row.get("eligible", False))]
    ranked = deterministic_rank(eligible)
    winner = ranked[0] if ranked else {}
    runner = ranked[1] if len(ranked) > 1 else {}
    margin: Optional[float] = (
        float(winner.get("final_score", 0.0))
        - float(runner.get("final_score", 0.0))
        if winner and runner
        else None
    )
    resolved = bool(
        winner
        and float(winner.get("final_score", 0.0)) >= minimum_score
        and (not runner or float(margin or 0.0) >= minimum_margin)
    )
    return winner, runner, margin, resolved


def nearest_alternative(
    rows: Sequence[Mapping[str, Any]],
    winner: Mapping[str, Any],
    *,
    identity_fields: Sequence[str] = ("doc_id",),
) -> Dict[str, Any]:
    """Return the strongest distinct non-selected candidate of any eligibility.

    This is intentionally separate from the decision competitor.  A rejected
    candidate is useful audit evidence but must never be substituted into the
    mathematical winner-margin contract.
    """
    winner_identity = tuple(
        clean_text(winner.get(field, "")) for field in identity_fields
    )
    for row in deterministic_rank(
        rows, identity_fields=identity_fields
    ):
        identity = tuple(
            clean_text(row.get(field, "")) for field in identity_fields
        )
        if not winner or identity != winner_identity:
            return row
    return {}
