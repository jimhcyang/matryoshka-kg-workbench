"""Predicate-local, noncompensatory value-redaction engine.

Candidate identity, clause role, event arguments, and type are separate gates.
Document-level topical similarity cannot compensate for failure of a critical
candidate-local gate.
"""
from __future__ import annotations

import re
import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import pandas as pd

from ..contracts import task_row
from ..decision import (
    canonical_value, deterministic_rank, event_identity,
    nearest_alternative, type_compatible, winner_with_margin,
)
from ..retrieval import (
    canonical_tokens,
    containment,
    jaccard,
    rank_documents_tokens,
)
from ..semantics import (
    PREDICATE_LEMMAS, build_event_signature, canonical_key,
    canonical_location, canonical_predicate, clean_text, parse_number_phrase,
    semantic_tokens,
)
from .base import EngineContext, EngineResult, TaskEngine


_VR_TRANSPORT_DIRECTION_PREDICATES = frozenset({
    "direct",
    "send",
    "sent",
    "taken",
    "transport",
    "move",
    "relocate",
    "relocation",
    "ship",
    "travel",
    "visit",
    "visited",
    "visiting",
    "visits",
    "deliver",
    "transfer",
    "leave",
    "left",
    "depart",
    "departed",
    "head",
    "headed",
    "go",
    "went",
})

_VR_WEAK_GOVERNING_PREDICATES = frozenset({
    "a",
    "an",
    "the",
    "had",
    "has",
    "have",
    "am",
    "are",
    "be",
    "been",
    "being",
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
    "is",
    "was",
    "were",
    "and",
    "apparently",
    "example",
    "relationship",
    "s",
    "p",
    "v",
    "redact",
    "as",
    "at",
    "on",
    "in",
    "of",
    "for",
    "from",
    "to",
    "with",
    "by",
    "around",
    "that",
})

_VR_TOKEN_ALIASES = {
    "aim": "intent",
    "aimed": "intent",
    "aiming": "intent",
    "aims": "intent",
    "affect": "effect",
    "affected": "effect",
    "affecting": "effect",
    "approaches": "approach",
    "build": "construct",
    "building": "construct",
    "built": "construct",
    "cause": "effect",
    "caused": "effect",
    "causing": "effect",
    "concern": "intent",
    "concerned": "intent",
    "concerns": "intent",
    "co-operation": "cooperation",
    "conjecture": "speculation",
    "constructed": "construct",
    "constructing": "construct",
    "construction": "construct",
    "delay": "delay",
    "delayed": "delay",
    "delaying": "delay",
    "delays": "delay",
    "direction": "direct",
    "directions": "direct",
    "directed": "direct",
    "directing": "direct",
    "order": "direct",
    "ordered": "direct",
    "ordering": "direct",
    "orders": "direct",
    "bound": "destination",
    "destined": "destination",
    "destination": "destination",
    "destinations": "destination",
    "head": "destination",
    "headed": "destination",
    "heading": "destination",
    "heads": "destination",
    "depart": "leave",
    "departed": "leave",
    "departing": "leave",
    "departs": "leave",
    "go": "leave",
    "goes": "leave",
    "going": "leave",
    "went": "leave",
    "prepare": "ready",
    "prepared": "ready",
    "prepares": "ready",
    "preparing": "ready",
    "organization": "group",
    "organizations": "group",
    "groups": "group",
    "discontent": "unrest",
    "dissatisfaction": "unrest",
    "impact": "effect",
    "impacted": "effect",
    "impacting": "effect",
    "grow": "increase",
    "growing": "increase",
    "increased": "increase",
    "increases": "increase",
    "increasing": "increase",
    "instruct": "direct",
    "instructed": "direct",
    "instructing": "direct",
    "instruction": "direct",
    "instructions": "direct",
    "outreach": "approach",
    "observe": "report",
    "observed": "report",
    "observes": "report",
    "observing": "report",
    "observation": "report",
    "observations": "report",
    "perspective": "view",
    "perspectives": "view",
    "postpone": "delay",
    "postponed": "delay",
    "postponing": "delay",
    "postponement": "delay",
    "appoint": "appoint",
    "appointed": "appoint",
    "appointing": "appoint",
    "appointment": "appoint",
    "appointments": "appoint",
    "nominate": "appoint",
    "nominated": "appoint",
    "nominating": "appoint",
    "nomination": "appoint",
    "nominations": "appoint",
    "deliberation": "discussion",
    "deliberations": "discussion",
    "discussion": "discussion",
    "discussions": "discussion",
    "talk": "discussion",
    "talks": "discussion",
    "retake": "recapture",
    "retaken": "recapture",
    "retook": "recapture",
    "speculative": "speculation",
    "trouble": "effect",
    "troubles": "effect",
    "unrest": "unrest",
    "views": "view",
    "repatriation": "return",
    "repatriated": "return",
    "returning": "return",
    "returned": "return",
    "returns": "return",
    "relevance": "relevant",
    "reporting": "report",
    "reports": "report",
    "obtained": "acquire",
    "obtain": "acquire",
    "obtaining": "acquire",
    "acquired": "acquire",
    "acquires": "acquire",
    "components": "component",
    "parts": "component",
    "papers": "document",
    "documents": "document",
    "vessel": "ship",
    "ships": "ship",
    "defeated": "defeat",
    "defeating": "defeat",
    "overthrew": "overthrow",
    "overthrown": "overthrow",
    "overthrows": "overthrow",
    "launched": "launch",
    "launching": "launch",
    "flew": "fly",
    "flown": "fly",
    "flying": "fly",
    "airlifting": "airlift",
    "airlifted": "airlift",
    "imprisoned": "imprison",
    "imprisonment": "imprison",
    "jail": "imprison",
    "jailed": "imprison",
    "jailing": "imprison",
    "arrived": "arrive",
    "arrival": "arrive",
    "leaving": "leave",
    "left": "leave",
    "decided": "decide",
    "decision": "decide",
    "decisions": "decide",
    "differ": "identity_compare",
    "differed": "identity_compare",
    "difference": "identity_compare",
    "differences": "identity_compare",
    "different": "identity_compare",
    "inhabitant": "resident",
    "inhabitants": "resident",
    "residents": "resident",
    "same": "identity_compare",
    "force": "force",
    "forces": "force",
    "troop": "force",
    "troops": "force",
    "trip": "travel",
    "trips": "travel",
    "traveled": "travel",
    "travelled": "travel",
    "travels": "travel",
}

_VR_SCALAR_SURFACE_RE = re.compile(
    r"\b(?:\d[\d,]*(?:\.\d+)?|zero|one|two|three|four|five|six|seven|"
    r"eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred|thousand)\b",
    flags=re.I,
)

_VR_GENERIC_ANCHOR_STOPWORDS = frozenset({
    "redact",
    "slot",
    "type",
    "person",
    "organization",
    "location",
    "country",
    "military",
    "unknown",
    "entity",
    "generic",
    "value",
})

_VR_RETRIEVAL_STOPWORDS = _VR_GENERIC_ANCHOR_STOPWORDS | frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "his", "i",
    "in", "is", "it", "its", "may", "might", "not", "of", "on", "or",
    "our", "s", "she", "that", "the", "their", "them", "they", "this",
    "to", "u", "was", "we", "were", "will", "with", "would", "you",
})

_VR_CONTAINER_OF_PATTERN = re.compile(
    r"\b(?:component|part|subsidiary|branch)\s+of\b",
    flags=re.I,
)

# A Stage C redaction companion row is direct, provenance-bearing evidence for
# the visible slot neighborhood.  This route is deliberately stricter than an
# ordinary semantic candidate: it requires at least 20% local-window alignment
# (0.18 is the maximum local bonus), a strong retrieved document, and a score
# above the normal ambiguous-candidate zone before it may override noisy
# parser-inferred predicate, date, or grammatical-role gates.
_VR_DIRECT_CONTEXT_BONUS_FLOOR = 0.18 * 0.20
_VR_DIRECT_CONTEXT_STRONG_FRAME_BONUS_FLOOR = 0.18 * 0.10
_VR_DIRECT_CONTEXT_RETRIEVAL_FLOOR = 0.45
_VR_DIRECT_CONTEXT_SCORE_FLOOR = 0.60
_VR_DIRECT_CONTEXT_STRONG_FRAME_SCORE_FLOOR = 0.65


def _vr_direct_context_is_locally_grounded(
    *,
    event_score: float,
    event_floor: float,
    local_context_bonus: float,
    retrieval_score: float,
    local_argument_frame_pass: bool,
) -> bool:
    """Keep pseudo-chunk rescue local to the candidate's actual event frame.

    Pseudo-chunk rows intentionally carry broad context.  Retrieval relevance
    alone therefore cannot certify a value: the candidate must occupy the
    local argument frame, except for an almost exact event-and-context match,
    and must have either event agreement or strong two-sided local context.
    """
    frame_is_grounded = bool(
        local_argument_frame_pass
        or (
            event_score >= 0.80
            and local_context_bonus >= 0.15
        )
        or (
            # A near-exact event frame can survive harmless rewriting or a
            # sentence split at an abbreviation (for example "U.S."). Two
            # thirds of the maximum local-context bonus is sufficient only
            # at this substantially stronger event threshold.
            event_score >= 0.90
            and local_context_bonus >= 0.12
        )
    )
    context_is_grounded = bool(
        event_score >= event_floor
        or local_context_bonus >= 0.06
        or (
            retrieval_score >= 0.52
            and local_context_bonus >= 0.045
        )
    )
    return frame_is_grounded and context_is_grounded


def _expanded_tokens(
    value: Any,
    context: EngineContext,
) -> Set[str]:
    return set(context.semantic_resources.expanded_tokens(value))


def _normalized_expanded_tokens(
    value: Any,
    context: EngineContext,
) -> Set[str]:
    normalized: Set[str] = set()
    for token in canonical_tokens(context.semantic_resources.expanded_tokens(value)):
        token_text = str(token)
        if ":" in token_text:
            normalized.add(token_text)
            continue
        normalized.add(_VR_TOKEN_ALIASES.get(token_text, token_text))
    return normalized


def _vr_retrieval_tokens(
    value: Any,
    context: EngineContext,
) -> Set[str]:
    """Return content-bearing tokens for first-stage VR document retrieval."""
    return {
        token
        for token in _normalized_expanded_tokens(value, context)
        if ":" in token
        or (len(token) >= 3 and token not in _VR_RETRIEVAL_STOPWORDS)
    }


def _token_score(
    query_tokens: Set[str],
    candidate_tokens: Set[str],
) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    return 0.62 * containment(query_tokens, candidate_tokens) + 0.38 * jaccard(
        query_tokens,
        candidate_tokens,
    )


def _clean_slot_signature_value(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if canonical_key(text) in {"slot", "[slot]"}:
        return ""
    return text


def _vr_context_has_substantive_surface(text: str) -> bool:
    tokens = [
        token
        for token in semantic_tokens(text)
        if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    ]
    return len(tokens) >= 2


def _vr_anchor_support_ratio(
    anchor_text: str,
    support_text: str,
    context: EngineContext,
) -> float:
    anchor_tokens = {
        token
        for token in _normalized_expanded_tokens(anchor_text, context)
        if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    }
    support_tokens = {
        token
        for token in _normalized_expanded_tokens(support_text, context)
        if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    }
    if not anchor_tokens or not support_tokens:
        return 0.0
    return float(containment(anchor_tokens, support_tokens))


def _vr_object_anchor_looks_clause_like(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return False
    content_tokens = [
        token
        for token in semantic_tokens(cleaned)
        if len(token) >= 3 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    ]
    if len(content_tokens) >= 7:
        return True
    return bool(re.search(
        r"\b(?:and|that|while|which|who|whom|within|following|including|opined|proposal|proposals)\b",
        cleaned,
        flags=re.I,
    ))


def _vr_slot_looks_enumerative(slot: Mapping[str, Any]) -> bool:
    before = clean_text(slot.get("local_context_before", ""))
    after = clean_text(slot.get("local_context_after", ""))
    if re.search(r"\b(?:including|among|amongst)\b", before, flags=re.I):
        return True
    if (
        re.search(
            r"\b(?:consisting|composed|comprised|made\s+up)\s+of\s*$",
            before,
            flags=re.I,
        )
        and re.search(r"(?:,|\b(?:and|or)\b)", after, flags=re.I)
    ):
        return True
    if re.search(
        r"\b(?:and|or)(?:\s+(?:a|an|the))?\s*$",
        before,
        flags=re.I,
    ):
        return True
    if (
        re.search(
            r"\b(?:include|includes|including|with|alongside|together with)\s*$",
            before,
            flags=re.I,
        )
        and re.search(r"^\s*,?\s*[^.]{0,140}\b(?:and|or)\b", after, flags=re.I)
    ):
        return True
    if re.search(r"\bwas\s*$", before, flags=re.I) and re.match(
        r"^\s*,?\s*(?:identified|described|known|reported|regarded|named)\b",
        after,
        flags=re.I,
    ):
        return True
    if before.count(",") >= 2 and re.search(
        r",\s*(?:and|or)(?:\s+the)?\s*$",
        before,
        flags=re.I,
    ):
        return True
    if before.count(",") >= 2 and re.match(
        r"^\s*,?\s*(?:and|or)\b",
        after,
        flags=re.I,
    ):
        return True
    return False


def _vr_visible_enumeration_members(slot: Mapping[str, Any]) -> List[str]:
    """Extract visible list members immediately surrounding a missing value."""
    if not _vr_slot_looks_enumerative(slot):
        return []
    left = clean_text(slot.get("local_context_before", ""))
    if left.count(",") < 2:
        return []
    parts = [clean_text(part) for part in left.split(",")]
    if parts:
        lead = parts[0]
        boundaries = list(re.finditer(
            r"\b(?:including|among|amongst|in|from|at|of)\b",
            lead,
            flags=re.I,
        ))
        if boundaries:
            lead = clean_text(lead[boundaries[-1].end():])
        parts[0] = lead
    members: List[str] = []
    seen: Set[str] = set()
    for part in parts[-10:]:
        candidate = re.sub(
            r"^(?:and|or|also|the|a|an)\s+", "", part, flags=re.I
        )
        candidate = re.sub(
            r"\s+(?:province|provinces|district|districts|city|cities|"
            r"country|countries|area|areas)$",
            "",
            candidate,
            flags=re.I,
        )
        words = semantic_tokens(candidate)
        key = canonical_key(candidate)
        if not key or not 1 <= len(words) <= 5:
            continue
        if len(words) == 1:
            only_word = next(iter(words))
            if (
                len(only_word) < 4
                or only_word in _VR_GENERIC_ANCHOR_STOPWORDS
            ):
                continue
        if key not in seen:
            seen.add(key)
            members.append(candidate)
    return members if len(members) >= 4 else []


def _vr_local_enumeration_context_pass(
    members: List[str],
    value: str,
    evidence: str,
    document_text: str,
) -> bool:
    """Require several visible list peers near an enumerative candidate."""
    if not members:
        return True
    sources = [clean_text(evidence)]
    document = clean_text(document_text)
    candidate = clean_text(value)
    if document and candidate:
        pattern = re.compile(r"\b" + re.escape(candidate) + r"\b", flags=re.I)
        for match in list(pattern.finditer(document))[:12]:
            sources.append(document[
                max(0, match.start() - 700):min(len(document), match.end() + 700)
            ])
    required = min(3, max(2, math.ceil(len(members) * 0.35)))
    for source in sources:
        normalized = canonical_key(source)
        hits = sum(
            bool(re.search(r"\b" + re.escape(canonical_key(member)) + r"\b", normalized))
            for member in members
        )
        if hits >= required:
            return True
    return False


def _vr_infer_generic_object_anchor(slot: Mapping[str, Any]) -> str:
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type not in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}:
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    right = clean_text(slot.get("local_context_after", ""))
    if not re.match(r"^\s*(?:by|from|against)\b", right, flags=re.I):
        return ""
    match = re.search(
        r"\b(?:the\s+)?([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,3})\s+of\s*$",
        left,
        flags=re.I,
    )
    if not match:
        return ""
    phrase = clean_text(match.group(1))
    phrase = re.sub(
        r"^(?:following|after|amid|during|while|in\s+wake\s+of)\s+",
        "",
        phrase,
        flags=re.I,
    )
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.I)
    if len(semantic_tokens(phrase)) > 4:
        return ""
    return phrase


def _vr_infer_person_title_anchor(slot: Mapping[str, Any]) -> str:
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type != "PERSON":
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    # A blank immediately following a communication verb is its recipient,
    # not part of the preceding official's title (for example,
    # ``the ambassador informed [PERSON] that ...``).
    if re.search(
        r"\b(?:told|asked|informed|briefed|advised|warned|notified|urged)\s*$",
        left,
        flags=re.I,
    ):
        return ""
    match = re.search(
        r"\b("
        r"(?:(?:[A-Za-z][A-Za-z-]*\s+){0,2}"
        r"(?:m\.?p\.?|member\s+of\s+parliament|parliamentarian))"
        r"|"
        r"(?:somali\s+president|president|foreign\s+minister|fonmin|"
        r"prime\s+minister|minister|ambassador|chairman|director|secretary|"
        r"general|colonel|major|captain)(?:\s+[A-Za-z-]+){0,2}"
        r")\s*$",
        left,
        flags=re.I,
    )
    if match:
        title = clean_text(match.group(1))
        title_words = canonical_key(title).split()
        # A coordinated object such as "the agriculture minister and [SLOT]"
        # is not a title-plus-name construction.  The former regex admitted
        # the conjunction as one of the title's free trailing words and then
        # incorrectly forced the missing value through a subject-identity
        # gate.
        if title_words and title_words[-1] not in {
            "a", "an", "and", "or", "the", "told", "asked", "informed",
            "briefed", "advised", "warned", "notified", "urged",
        }:
            return title
    right = clean_text(slot.get("local_context_after", ""))
    appositive = re.match(
        r"^\s*,?\s*((?:acting\s+|deputy\s+|assistant\s+)?"
        r"(?:ambassador|chair(?:man|woman|person)?|chief|commander|director|"
        r"governor|head|leader|minister|president|representative|secretary)"
        r"\s+(?:at|for|from|in|of|to)\s+[^,;:.]{2,100})\s*[,;:]",
        right,
        flags=re.I,
    )
    return clean_text(appositive.group(1)) if appositive else ""


def _vr_infer_person_object_trigger(slot: Mapping[str, Any]) -> str:
    """Infer a bounded patient/object role from the words before a PERSON slot."""
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type != "PERSON":
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    if re.search(
        r"\b(?:nominate(?:d|s|ing)?|appoint(?:ed|s|ing)?|install(?:ed|s|ing)?|select(?:ed|s|ing)?|"
        r"name(?:d|s|ing)?)\s*$",
        left,
        flags=re.I,
    ):
        return "nominate"
    if re.search(
        r"\b(?:affect(?:ed|s|ing)?|impact(?:ed|s|ing)?|troubl(?:e|ed|es|ing)|"
        r"threaten(?:ed|s|ing)?|harm(?:ed|s|ing)?|endanger(?:ed|s|ing)?)\s*$",
        left,
        flags=re.I,
    ):
        return "affect"
    if re.search(
        r"\b(?:death|murder|killing|assassination)\s+of\s*$",
        left,
        flags=re.I,
    ):
        return "death"
    right = clean_text(slot.get("local_context_after", ""))
    if re.match(
        r"^\s*(?:'s|’s)\s+(?:death|murder|killing|assassination)\b",
        right,
        flags=re.I,
    ):
        return "death"
    return ""


def _vr_local_person_object_pass(
    trigger: str,
    candidate_value: str,
    evidence: str,
) -> bool:
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not trigger or not candidate or not source:
        return False
    candidate_pattern = re.escape(candidate)
    if trigger == "nominate":
        role_modifier = (
            r"(?:(?:the|a|an|new|acting|deputy|assistant|interim|former|"
            r"ambassador|admiral|general|colonel|major|captain|minister|"
            r"president|secretary)\s+){0,3}"
        )
        patterns = (
            r"\b(?:nominate(?:d|s|ing)?|appoint(?:ed|s|ing)?|install(?:ed|s|ing)?|"
            r"select(?:ed|s|ing)?|name(?:d|s|ing)?)\s+" + role_modifier
            + candidate_pattern + r"\b",
            r"\b(?:nomination|appointment|installation|selection|naming)\s+of\s+"
            + role_modifier + candidate_pattern + r"\b",
            r"\b" + candidate_pattern
            + r"\b\s+(?:was|is|had been|has been)\s+"
            r"(?:nominated|appointed|installed|selected|named)\b",
        )
    elif trigger == "affect":
        patterns = (
            r"\b(?:affect(?:ed|s|ing)?|impact(?:ed|s|ing)?|"
            r"threaten(?:ed|s|ing)?|harm(?:ed|s|ing)?|endanger(?:ed|s|ing)?)"
            r"\s+(?:\w+[ -]){0,3}" + candidate_pattern + r"\b",
            r"\b(?:cause|caused|causes|causing)\s+(?:\w+\s+){0,2}"
            r"(?:trouble|harm|difficulty|problems?)\s+to\s+"
            r"(?:\w+[ -]){0,3}" + candidate_pattern + r"\b",
            r"\b" + candidate_pattern
            + r"\b\s+(?:was|is|could be|may be|might be)\s+"
            r"(?:affected|impacted|threatened|harmed|endangered)\b",
        )
    else:
        patterns = (
            r"\b(?:death|murder|killing|assassination)\s+of\s+"
            r"(?:[A-Za-z][A-Za-z.'-]*[\s,]+){0,5}"
            + candidate_pattern + r"\b",
            r"\b" + candidate_pattern
            + r"(?:'s)?\s+(?:death|murder|killing|assassination)\b",
            r"\b(?:murdered|killed|assassinated)\s+(?:\w+[ -]){0,3}"
            + candidate_pattern + r"\b",
            r"\b(?:death|murder|killing|assassination)\s+of\s+"
            r"(?:[A-Za-z][A-Za-z.'-]*[\s,]+){0,8}"
            + candidate_pattern + r"\b",
        )
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.I):
            if trigger == "affect" and re.match(
                r"^\s*(?:'s|’s)\b",
                source[match.end():],
                flags=re.I,
            ):
                # "impact of Khrushchev's survey" affects the survey, not
                # Khrushchev.  A possessive owner is not the patient slot.
                continue
            return True
    return False


def _vr_infer_adjacent_scalar_anchor(slot: Mapping[str, Any]) -> str:
    """Capture a count or compact report code immediately after a location slot."""
    if clean_text(slot.get("slot_frame", "")) != "generic_value_slot":
        return ""
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type not in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}:
        return ""
    right = clean_text(slot.get("local_context_after", ""))
    match = _VR_SCALAR_SURFACE_RE.match(right)
    return clean_text(match.group(0)) if match else ""


def _vr_local_adjacent_scalar_pass(
    candidate_value: str,
    scalar_anchor: str,
    evidence: str,
) -> bool:
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    expected = parse_number_phrase(scalar_anchor)
    if not candidate or not source or expected is None:
        return False
    for candidate_match in re.finditer(
        r"\b" + re.escape(candidate) + r"\b",
        source,
        flags=re.I,
    ):
        after = source[candidate_match.end():candidate_match.end() + 56]
        scalar_match = _VR_SCALAR_SURFACE_RE.search(after)
        if not scalar_match:
            continue
        intervening = semantic_tokens(after[:scalar_match.start()])
        if len(intervening) > 3:
            continue
        observed = parse_number_phrase(scalar_match.group(0))
        if observed is not None and abs(observed - expected) < 1e-9:
            return True
    return False


def _vr_infer_person_recipient_trigger(slot: Mapping[str, Any]) -> str:
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type != "PERSON":
        return ""
    if clean_text(slot.get("slot_frame", "")) != "generic_value_slot":
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    match = re.search(
        r"\b(told|asked|informed|briefed|advised|warned|notified|urged)\s*$",
        left,
        flags=re.I,
    )
    return clean_text(match.group(1)) if match else ""


def _vr_infer_copular_complement_trigger(slot: Mapping[str, Any]) -> bool:
    if clean_text(slot.get("slot_frame", "")) != "generic_value_slot":
        return False
    left = clean_text(slot.get("local_context_before", ""))
    right = clean_text(slot.get("local_context_after", ""))
    return bool(
        re.search(r"\b(?:was|were|is|are)\s*$", left, flags=re.I)
        and re.match(
            r"^\s*,?\s*(?:identified|described|known|reported|named)\b",
            right,
            flags=re.I,
        )
    )


def _vr_infer_container_of_trigger(slot: Mapping[str, Any]) -> str:
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type != "ORGANIZATION":
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    match = re.search(r"\b(?:component|part|subsidiary|branch)\s+of\s*$", left, flags=re.I)
    return "container_of" if match else ""


def _vr_infer_membership_container_trigger(slot: Mapping[str, Any]) -> str:
    if clean_text(slot.get("slot_frame", "")) != "generic_value_slot":
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    match = re.search(
        r"\bAmong\s+the\s+(.+?)\s+(?:was|were)\s*,?\s*$",
        left,
        flags=re.I,
    )
    if not match:
        return ""
    container = clean_text(match.group(1))
    tokens = [token for token in semantic_tokens(container) if len(token) >= 4]
    if not tokens:
        return ""
    return container


def _vr_infer_location_prepositional_trigger(
    slot: Mapping[str, Any],
) -> Tuple[str, str]:
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type not in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}:
        return "", ""
    left = clean_text(slot.get("local_context_before", ""))
    match = re.search(
        r"(.{0,180}?)\b(to|from|in|at|on|of)\b(?:\s+the)?\s*$",
        left,
        flags=re.I,
    )
    if not match:
        return "", ""
    phrase = clean_text(match.group(1))
    prep = clean_text(match.group(2)).lower()
    phrase = re.sub(r"^(?:for|and|but|or)\s+", "", phrase, flags=re.I)
    ordered_tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'().-]*", phrase)
        if token
        and canonical_key(token) not in _VR_GENERIC_ANCHOR_STOPWORDS
        and canonical_key(token) not in _VR_WEAK_GOVERNING_PREDICATES
    ]
    # The grammatical head immediately before the preposition is the stable
    # role constraint.  Pulling several preceding modifiers into this trigger
    # makes harmless paraphrases and OCR plurals fail (for example,
    # ``jungles of`` versus ``in the jungles of``).
    compact_phrase = clean_text(" ".join(ordered_tokens[-1:])) if ordered_tokens else ""
    if not compact_phrase and ordered_tokens:
        compact_phrase = clean_text(" ".join(ordered_tokens))
    if not compact_phrase:
        return "", ""
    return compact_phrase, prep


def _vr_infer_leading_locative_trigger(
    slot: Mapping[str, Any],
) -> Tuple[str, str]:
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type not in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}:
        return "", ""
    left = clean_text(slot.get("local_context_before", ""))
    if not left:
        return "", ""
    match = re.match(
        r"^(?P<prep>in|at|from|to|on)\s+(?P<phrase>.+)$",
        left,
        flags=re.I,
    )
    if match:
        prep = clean_text(match.group("prep")).lower()
        phrase = clean_text(match.group("phrase"))
        # A sentence-initial locative must not be projected onto a later list
        # slot (for example, ``In April ..., and [LOCATION], Darlac ...``).
        if re.search(r"[,;:]", phrase) or len(semantic_tokens(phrase)) > 8:
            return "", ""
    else:
        bare = re.fullmatch(r"(?P<prep>in|at|from|to|on)", left, flags=re.I)
        if not bare:
            return "", ""
        prep = clean_text(bare.group("prep")).lower()
        right = re.sub(
            r"^[\s,;:()\-]+", "", clean_text(slot.get("local_context_after", ""))
        )
        phrase = clean_text(
            re.split(r"[,;:.]", right, maxsplit=1)[0]
        )
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'().-]*", phrase)
        if token
        and canonical_key(token) not in _VR_GENERIC_ANCHOR_STOPWORDS
        and canonical_key(token) not in _VR_WEAK_GOVERNING_PREDICATES
    ]
    if not tokens:
        return "", ""
    selected = tokens[-4:] if match else tokens[:6]
    return clean_text(" ".join(selected)), prep


def _vr_infer_possessive_trigger(slot: Mapping[str, Any]) -> str:
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type not in {
        "PERSON", "ORGANIZATION", "GOVERNMENT", "MILITARY",
        "LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY", "GPE",
    }:
        return ""
    right = clean_text(slot.get("local_context_after", ""))
    match = re.match(
        r"^\s*(?:'s|’s)\s+(?P<head>[^,;:.!?]{1,80})",
        right,
        flags=re.I,
    )
    if not match:
        return ""
    head = clean_text(match.group("head"))
    return clean_text(" ".join(head.split()[:5]))


def _vr_infer_attribution_trigger(slot: Mapping[str, Any]) -> str:
    """Identify a blank that names the source of the visible assertion."""
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type not in {"PERSON", "ORGANIZATION", "GOVERNMENT"}:
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    match = re.search(
        r"\b(?P<trigger>according\s+to|as\s+reported\s+by|"
        r"as\s+stated\s+by|on\s+the\s+account\s+of|per)\s*$",
        left,
        flags=re.I,
    )
    return canonical_key(match.group("trigger")) if match else ""


def _vr_local_attribution_pass(
    candidate_value: str,
    evidence: str,
) -> bool:
    """Verify that a candidate is locally presented as an information source."""
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not candidate or not source:
        return False
    escaped = re.escape(candidate)
    patterns = (
        r"\b(?:according\s+to|as\s+reported\s+by|as\s+stated\s+by|per)\s+"
        + escaped + r"\b",
        r"\bconversation\b[^.!?]{0,100}\bwith\s+" + escaped + r"\b",
        r"\b" + escaped
        + r"\b(?:\s+(?:personally|privately|also|later|reportedly|then)){0,3}"
        + r"\s+(?:reported|said|stated|told|informed|observed)\b",
    )
    return any(re.search(pattern, source, flags=re.I) for pattern in patterns)


def _vr_infer_list_head_trigger(slot: Mapping[str, Any]) -> bool:
    left = clean_text(slot.get("local_context_before", ""))
    return bool(re.search(
        r"\b(?:consisting|composed|comprised|made\s+up)\s+of\s*$",
        left,
        flags=re.I,
    ))


def _vr_local_list_head_pass(
    value: str,
    evidence: str,
    document_text: str,
) -> bool:
    candidate = clean_text(value)
    if not candidate:
        return False
    pattern = re.compile(
        r"\b(?:consisting|composed|comprised|made\s+up)\s+of\s+"
        + re.escape(candidate)
        + r"\b",
        flags=re.I,
    )
    return any(
        bool(source and pattern.search(clean_text(source)))
        for source in (evidence, document_text)
    )


def _vr_local_possessive_pass(
    value: str,
    trigger: str,
    evidence: str,
    context: EngineContext,
) -> bool:
    candidate = clean_text(value)
    source = clean_text(evidence)
    trigger_tokens = {
        token
        for token in _normalized_expanded_tokens(trigger, context)
        if len(canonical_key(token)) >= 3
    }
    if not candidate or not source or not trigger_tokens:
        return False
    candidate_re = re.compile(r"\b" + re.escape(candidate) + r"\b", flags=re.I)
    for match in candidate_re.finditer(source):
        local = source[match.end():min(len(source), match.end() + 90)]
        local_tokens = {
            token
            for token in _normalized_expanded_tokens(local, context)
            if len(canonical_key(token)) >= 3
        }
        direct_possessive = bool(re.match(r"^\s*(?:'s|’s)\b", local, flags=re.I))
        if direct_possessive and (
            containment(trigger_tokens, local_tokens) >= 0.45
            or bool(trigger_tokens & local_tokens)
        ):
            return True
        before = source[max(0, match.start() - 90):match.start()]
        inverse = re.search(r"\bof\s*$", before, flags=re.I)
        if inverse:
            before_tokens = {
                token
                for token in _normalized_expanded_tokens(before, context)
                if len(canonical_key(token)) >= 3
            }
            if (
                containment(trigger_tokens, before_tokens) >= 0.45
                or bool(trigger_tokens & before_tokens)
            ):
                return True
    return False


def _vr_infer_location_movement_origin_trigger(
    slot: Mapping[str, Any],
) -> str:
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type not in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}:
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    right = clean_text(slot.get("local_context_after", ""))
    match = re.search(
        r"\b(?P<verb>left|leave|leaves|leaving|departed|departs|departing)\s*$",
        left,
        flags=re.I,
    )
    if not match or not re.match(
        r"^\s*(?:for|to|toward|towards|via)\b",
        right,
        flags=re.I,
    ):
        return ""
    return clean_text(match.group("verb"))


def _vr_infer_location_movement_destination_trigger(
    slot: Mapping[str, Any],
) -> str:
    """Recover a destination frame when the location blank ends the clause."""
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type not in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}:
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    match = re.search(
        r"\b(?P<verb>leave|leaves|left|leaving|depart|departs|departed|"
        r"departing|send|sends|sent|move|moves|moved|moving|transport|"
        r"transports|transported|transporting|travel|travels|traveled|"
        r"travelled|traveling|travelling|head|heads|headed|heading)\b"
        r"(?:\s+[A-Za-z0-9'-]+){0,5}\s+"
        r"(?:for|to|toward|towards|via)(?:\s+the)?\s*$",
        left,
        flags=re.I,
    )
    return clean_text(match.group("verb")) if match else ""


def _vr_local_location_movement_origin_pass(
    trigger: str,
    value: str,
    evidence: str,
) -> bool:
    predicate = _vr_predicate_family(trigger)
    candidate = clean_text(value)
    source = clean_text(evidence)
    if not predicate or not candidate or not source:
        return False
    return bool(re.search(
        r"\b" + _predicate_pattern(predicate) + r"\b\s+(?:the\s+)?"
        + re.escape(candidate)
        + r"\b\s+(?:for|to|toward|towards|via)\b",
        source,
        flags=re.I,
    ))


def _vr_local_location_movement_destination_pass(
    trigger: str,
    value: str,
    evidence: str,
) -> bool:
    predicate = _vr_predicate_family(trigger)
    candidate = clean_text(value)
    source = clean_text(evidence)
    if not predicate or not candidate or not source:
        return False
    return bool(re.search(
        r"\b" + _predicate_pattern(predicate) + r"\b"
        r"(?:\s+[A-Za-z0-9'-]+){0,6}\s+"
        r"(?:for|to|toward|towards|via)(?:\s+the)?\s+"
        + re.escape(candidate) + r"\b",
        source,
        flags=re.I,
    ))


def _vr_infer_person_infinitive_trigger(slot: Mapping[str, Any]) -> bool:
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type != "PERSON":
        return False
    if clean_text(slot.get("slot_frame", "")) != "generic_value_slot":
        return False
    left = clean_text(slot.get("local_context_before", ""))
    right = clean_text(slot.get("local_context_after", ""))
    return bool(
        re.search(r"\bfor\s*$", left, flags=re.I)
        and re.match(r"^\s*to\s+\w+", right, flags=re.I)
    )


def _vr_infer_generic_subject_role(slot: Mapping[str, Any]) -> str:
    if clean_text(slot.get("slot_frame", "")) != "generic_value_slot":
        return ""
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}:
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    right = clean_text(slot.get("local_context_after", ""))
    if re.search(r"\b(?:in|at|from|to|near|by|with|for|of)\s*$", left, flags=re.I):
        return ""
    if re.match(
        r"^\s*(?:\([A-Za-z]+\)\s*)?"
        r"(?:is|are|was|were|be|been|being)\b[^.!?]{0,80}\b"
        r"(?:taken|moved|sent|transported|directed|transferred|"
        r"delivered|seized|confiscated|arrested|appointed)\b",
        right,
        flags=re.I,
    ):
        # Passive/patient constructions like "was among the detainees taken to"
        # or "was moved from" should not require the hidden value to surface
        # as a direct grammatical subject of the content verb.
        return ""
    if re.match(
        r"^\s*(?:\([A-Za-z]+\)\s*)?"
        r"(?:has|have|had)\s+been\s+"
        r"(?:moved|sent|transported|directed|transferred|delivered|"
        r"seized|confiscated|arrested|appointed)\b",
        right,
        flags=re.I,
    ):
        return ""
    if re.match(
        r"^\s*(?:\([A-Za-z]+\)\s*)?(?:is|are|was|were|has|have|had|claimed|claims|reported|reports|said|says|told|tells|stated|states|noted|notes|expected|expects)\b",
        right,
        flags=re.I,
    ):
        return "grammatical_subject"
    return ""


def _vr_object_anchor_detached_from_governing_predicate(
    slot: Mapping[str, Any],
    query_signature: Mapping[str, Any],
    context: EngineContext,
) -> bool:
    object_text = _clean_slot_signature_value(
        query_signature.get("object", "") or query_signature.get("patient_or_theme", "")
    )
    right_context = clean_text(slot.get("local_context_after", ""))
    governing_surface = clean_text(slot.get("governing_predicate", ""))
    if not object_text or not right_context or not governing_surface:
        return False
    predicate_match = re.search(
        r"\b" + re.escape(governing_surface) + r"\b",
        right_context,
        flags=re.I,
    )
    if not predicate_match:
        return False
    before = right_context[:predicate_match.start()]
    after = right_context[predicate_match.end():]
    before_support = _vr_anchor_support_ratio(object_text, before, context)
    after_support = _vr_anchor_support_ratio(object_text, after, context)
    if before_support < 0.45 or after_support >= before_support:
        return False
    if re.search(r"[,;:]|\b(?:and|but|while|although|though)\b", before, flags=re.I):
        return True
    return len(before.split()) >= 6


def _vr_signature_fields_to_suppress(
    slot: Mapping[str, Any],
    query_signature: Mapping[str, Any],
    context: EngineContext,
) -> Set[str]:
    slot_frame = clean_text(slot.get("slot_frame", ""))
    governing_predicate = _vr_predicate_family_for_frame(
        slot.get("governing_predicate", ""), slot_frame
    )
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    left_context = clean_text(slot.get("local_context_before", ""))
    right_context = clean_text(slot.get("local_context_after", ""))
    right_has_surface = _vr_context_has_substantive_surface(right_context)
    object_text = _clean_slot_signature_value(
        query_signature.get("object", "") or query_signature.get("patient_or_theme", "")
    )
    actor_text = _clean_slot_signature_value(
        query_signature.get("actor", "") or query_signature.get("agent", "")
    )
    suppressed: Set[str] = set()

    if canonical_key(actor_text) in {
        "i", "you", "he", "she", "it", "we", "they",
    }:
        # Unresolved pronouns are not stable cross-document identity anchors.
        # Predicate, role, relation, and local context continue to gate the
        # candidate independently.
        suppressed.update({"actor", "agent", "embedded_actor"})

    if slot_frame == "generic_value_slot":
        if (
            expected_type in {"PERSON", "ORGANIZATION", "GOVERNMENT"}
            and re.match(
                r"^\s*(?:located|situated|based|headquartered)\b",
                right_context,
                flags=re.I,
            )
        ):
            suppressed.update({
                "predicate",
                "predicate_surface",
                "object",
                "object_surface",
                "patient_or_theme",
            })
        if (
            expected_type == "ORGANIZATION"
            and _vr_infer_container_of_trigger(slot)
        ):
            suppressed.update({
                "predicate",
                "predicate_surface",
                "object",
                "object_surface",
                "patient_or_theme",
                "recipient",
                "destination",
                "event_time",
            })
        if not governing_predicate:
            suppressed.update({
                "predicate",
                "predicate_surface",
                "object",
                "object_surface",
                "patient_or_theme",
                "recipient",
                "destination",
                "event_time",
            })
        else:
            if right_has_surface and _vr_anchor_support_ratio(
                governing_predicate,
                right_context,
                context,
            ) < 0.45:
                suppressed.update({"predicate", "predicate_surface"})
            if _vr_slot_looks_enumerative(slot):
                suppressed.update({"object", "object_surface", "patient_or_theme"})
            elif (
                expected_type in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}
                and governing_predicate in {"follow", "following"}
            ):
                suppressed.update({"object", "object_surface", "patient_or_theme"})
            elif object_text and (
                _vr_object_anchor_looks_clause_like(object_text)
                or (
                    right_has_surface
                    and _vr_anchor_support_ratio(object_text, right_context, context) < 0.34
                )
            ):
                suppressed.update({"object", "object_surface", "patient_or_theme"})
        if actor_text and canonical_key(actor_text).find("slot") >= 0:
            suppressed.update({"actor", "agent", "embedded_actor"})

    if (
        slot_frame == "entity_as_grammatical_subject"
        and object_text
        and right_has_surface
        and _vr_anchor_support_ratio(object_text, right_context, context) < 0.34
    ):
        suppressed.update({"object", "object_surface", "patient_or_theme"})
    if _vr_object_anchor_detached_from_governing_predicate(
        slot,
        query_signature,
        context,
    ):
        suppressed.update({"object", "object_surface", "patient_or_theme"})
    if _vr_infer_container_of_trigger(slot):
        suppressed.update({"object", "object_surface", "patient_or_theme"})
    if (
        slot_frame == "entity_as_grammatical_subject"
        and actor_text
        and right_has_surface
        and _vr_anchor_support_ratio(actor_text, right_context, context) < 0.28
        and expected_type in {"MILITARY", "LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}
    ):
        suppressed.update({"actor", "agent", "embedded_actor"})
    return suppressed


def _vr_predicate_family(value: Any) -> str:
    """Collapse near-equivalent transport/direction predicates for VR only.

    Value redaction often relies on short local context windows.  Rephrased
    prompts such as "sent to" versus corpus evidence phrased as "directed to"
    should still be treated as the same event family, but that broadening is
    intentionally isolated to this engine.
    """
    surface = clean_text(value)
    if re.search(r"(?:'s|’s)\b", surface, flags=re.I):
        return ""
    predicate = canonical_predicate(surface)
    if predicate in _VR_WEAK_GOVERNING_PREDICATES:
        return ""
    if predicate in _VR_TRANSPORT_DIRECTION_PREDICATES:
        return "transport"
    if predicate in {
        "differ", "differs", "differed", "differing", "different", "same",
    }:
        return "identity_compare"
    if predicate in {
        "decide", "decides", "decided", "deciding", "decision", "decisions",
    }:
        return "decide"
    return predicate


def _vr_predicate_family_for_frame(value: Any, slot_frame: Any) -> str:
    """Interpret direction verbs according to the blank's grammatical role."""
    if (
        clean_text(slot_frame) == "agentive_by_slot"
        and canonical_predicate(value) == "direct"
    ):
        return "direct"
    return _vr_predicate_family(value)


def _vr_fragment_predicate_family(fragment_text: str) -> str:
    """Recover a reliable predicate when a slot parser lands on punctuation glue."""
    for surface in re.findall(r"[A-Za-z][A-Za-z'-]*", clean_text(fragment_text)):
        family = _vr_predicate_family(surface)
        if family in {"transport", "decide", "identity_compare"}:
            return family
        if surface.casefold() in PREDICATE_LEMMAS and family:
            return family
    return ""


def _vr_multi_slot_sequence_pass(
    slot: Mapping[str, Any],
    candidate_value: str,
    evidence: str,
) -> Optional[bool]:
    """Align adjacent redaction markers with ordered conjunction/list values."""
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not candidate or not source:
        return None
    before = clean_text(slot.get("local_context_before", ""))
    after = clean_text(slot.get("local_context_after", ""))
    marker_pattern = r"\[REDACT:[^\]]+\]"
    checks: List[bool] = []
    prior_markers = list(re.finditer(marker_pattern, before, flags=re.I))
    if prior_markers:
        tail = clean_text(before[prior_markers[-1].end():])
        connector = re.search(r"(?:^|[,;])?\s*(and|or)\s*$", tail, flags=re.I)
        if connector:
            checks.append(bool(re.search(
                r"\b" + re.escape(connector.group(1))
                + r"\s+(?:the\s+)?" + re.escape(candidate) + r"\b",
                source,
                flags=re.I,
            )))
    next_marker = re.search(marker_pattern, after, flags=re.I)
    if next_marker:
        prefix = clean_text(after[:next_marker.start()])
        connector = re.search(r"^\s*(and|or)\b", prefix, flags=re.I)
        if connector:
            checks.append(bool(re.search(
                r"\b" + re.escape(candidate)
                + r"\s+" + re.escape(connector.group(1)) + r"\b",
                source,
                flags=re.I,
            )))
    return all(checks) if checks else None


def _vr_local_slot_probe_text(
    slot: Mapping[str, Any],
    *,
    left_words: int = 8,
    right_words: int = 8,
) -> str:
    left = clean_text(slot.get("local_context_before", ""))
    right = clean_text(slot.get("local_context_after", ""))
    # Candidate certification must describe the blank's event, not a complete
    # earlier clause that happens to share the prompt.  Retrieval remains broad;
    # only this local event-overlap gate is scoped after the nearest boundary.
    left_boundaries = list(re.finditer(r"[.;!?]\s*", left))
    if left_boundaries:
        left = clean_text(left[left_boundaries[-1].end():])
    right = re.split(
        r"[,;]\s*(?=(?:caus(?:e|ed|es|ing)|result(?:ed|s|ing)?|"
        r"leav(?:e|es|ing)|which|where|while)\b)",
        right,
        maxsplit=1,
        flags=re.I,
    )[0]
    left_tail = " ".join(left.split()[-left_words:]) if left else ""
    right_head = " ".join(right.split()[:right_words]) if right else ""
    return clean_text(" ".join(part for part in (left_tail, right_head) if part))


def _vr_authoritative_slot_context(
    slot: Mapping[str, Any],
    fragment_text: str,
) -> Dict[str, Any]:
    """Re-derive marker sides from the authoritative fragment when possible."""
    normalized = dict(slot)
    source = clean_text(fragment_text)
    marker = clean_text(
        slot.get("slot_marker", "") or slot.get("normalized_marker", "")
    )
    if not source or not marker:
        return normalized
    marker_start = source.casefold().find(marker.casefold())
    if marker_start < 0:
        return normalized
    normalized["local_context_before"] = clean_text(source[:marker_start])
    normalized["local_context_after"] = clean_text(
        source[marker_start + len(marker):]
    )
    return normalized


def _vr_event_anchor_tokens(
    slot: Mapping[str, Any],
    slot_query_text: str,
    context: EngineContext,
) -> Set[str]:
    """Use the clause around the blank for candidate-local event identity.

    Document retrieval intentionally uses several broad query views. Candidate
    certification is narrower: a value in one clause must not be penalized by
    unrelated tokens from another sentence in the analyst prompt.
    """
    ignored = {
        "redact", "type", "person", "organization", "location", "slot",
    }

    def materialize(value: str) -> Set[str]:
        return {
            token
            for token in _normalized_expanded_tokens(value, context)
            if len(token) >= 4
            and token not in ignored
            and token not in _VR_GENERIC_ANCHOR_STOPWORDS
        }

    local_tokens = materialize(_vr_local_slot_probe_text(slot))
    return local_tokens if len(local_tokens) >= 3 else materialize(slot_query_text)


def _structured_slot_probe(
    slot: Mapping[str, Any],
    query_signature: Mapping[str, Any],
) -> str:
    parts: List[str] = []
    slot_frame = clean_text(slot.get("slot_frame", ""))
    governing_normalized = _vr_predicate_family_for_frame(
        slot.get("governing_predicate_normalized", ""), slot_frame
    )
    governing_surface = _vr_predicate_family_for_frame(
        slot.get("governing_predicate", ""), slot_frame
    )
    for value in (
        governing_normalized,
        governing_surface,
        query_signature.get("actor", ""),
        query_signature.get("patient_or_theme", ""),
        query_signature.get("object", ""),
        query_signature.get("recipient", ""),
        query_signature.get("destination", ""),
        query_signature.get("location_identity", ""),
        query_signature.get("location", ""),
        query_signature.get("event_time", ""),
        slot.get("modified_head", ""),
    ):
        cleaned = _clean_slot_signature_value(value)
        if cleaned:
            parts.append(cleaned)
    required_relation = clean_text(slot.get("required_relation", ""))
    if required_relation == "identity_appositive_to_title":
        parts.append("title identity")
    elif required_relation == "administratively_contained_by":
        parts.append("administrative container")
    return clean_text(" ".join(parts))


def _sanitize_value_redaction_query_signature(
    slot: Mapping[str, Any],
    query_signature: Mapping[str, Any],
    context: EngineContext,
) -> Dict[str, Any]:
    """Drop parser-invented structure for weak generic slots.

    For generic value redaction, the parser can infer a clause-like signature
    from a short context window even when the prompt itself does not contain a
    trustworthy predicate/object frame. Those inferred fields can dominate both
    document reranking and candidate event gates, which hurts semantically
    equivalent rephrasings. We keep explicit anchors such as date/location when
    present, but clear the noisy argument frame when the slot is generic and no
    governing predicate was supplied.
    """
    normalized = dict(query_signature or {})
    slot_frame = clean_text(slot.get("slot_frame", ""))
    for field in (
        "actor", "agent", "embedded_actor", "predicate",
        "predicate_surface", "object", "object_surface",
        "patient_or_theme", "recipient", "destination", "location",
        "location_identity", "event_time",
    ):
        if field in normalized and not _clean_slot_signature_value(
            normalized.get(field, "")
        ):
            normalized[field] = ""
    normalized["predicate"] = _vr_predicate_family_for_frame(
        normalized.get("predicate", ""), slot_frame
    )
    governing_predicate = _vr_predicate_family_for_frame(
        slot.get("governing_predicate", ""), slot_frame
    )
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    inferred_object = _vr_infer_generic_object_anchor(slot)
    if slot_frame == "possessive_entity_slot":
        for field in (
            "predicate", "predicate_surface", "object", "object_surface",
            "patient_or_theme", "recipient", "destination",
        ):
            normalized[field] = ""
    suppressed_fields = _vr_signature_fields_to_suppress(
        slot,
        normalized,
        context,
    )
    if _vr_slot_looks_enumerative(slot):
        # A trailing list head (for example, "..., and [LOCATION]
        # Provinces") is not a governing predicate or event object.
        suppressed_fields.update({
            "predicate", "predicate_surface", "object", "object_surface",
            "patient_or_theme", "recipient", "destination", "event_time",
        })
    if slot_frame != "generic_value_slot" and not suppressed_fields:
        return normalized
    if (
        slot_frame == "generic_value_slot"
        and not governing_predicate
        and not suppressed_fields
    ):
        suppressed_fields = {
            "predicate",
            "predicate_surface",
            "object",
            "object_surface",
            "patient_or_theme",
            "recipient",
            "destination",
            "event_time",
        }
    for field in suppressed_fields:
        normalized[field] = ""
    suppress_inferred_object = (
        slot_frame == "generic_value_slot"
        and expected_type in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}
        and governing_predicate in {"follow", "following"}
    )
    suppress_predicate = suppress_inferred_object
    if suppress_predicate:
        normalized["predicate"] = ""
        normalized["predicate_surface"] = ""
    if (
        inferred_object
        and not suppress_inferred_object
        and not _clean_slot_signature_value(normalized.get("object", ""))
    ):
        normalized["object"] = inferred_object
        normalized["object_surface"] = inferred_object
        normalized["patient_or_theme"] = inferred_object
    return normalized


def _signature_anchor_strength(signature: Mapping[str, Any]) -> int:
    return sum(
        bool(
            _clean_slot_signature_value(
                signature.get(field, "")
            )
        )
        for field in (
            "actor",
            "predicate",
            "object",
            "patient_or_theme",
            "recipient",
            "destination",
            "location_identity",
            "location",
            "event_time",
        )
    )


def _sanitize_value_redaction_required_fields(
    slot: Mapping[str, Any],
    required_fields: Optional[Set[str]],
    query_signature: Mapping[str, Any],
    context: EngineContext,
) -> Optional[Set[str]]:
    if not required_fields:
        return None
    slot_frame = clean_text(slot.get("slot_frame", ""))
    governing_predicate = _vr_predicate_family_for_frame(
        slot.get("governing_predicate", ""), slot_frame
    )
    normalized = {clean_text(value) for value in required_fields if clean_text(value)}
    if _vr_slot_looks_enumerative(slot):
        return None
    suppressed_fields = _vr_signature_fields_to_suppress(
        slot,
        query_signature,
        context,
    )
    inferred_object = _vr_infer_generic_object_anchor(slot)
    sanitized_predicate = _vr_predicate_family_for_frame(
        query_signature.get("predicate", ""), slot_frame
    )
    sanitized_object = _clean_slot_signature_value(
        query_signature.get("object", "") or query_signature.get("patient_or_theme", "")
    )
    sanitized_event_time = _clean_slot_signature_value(
        query_signature.get("event_time", "")
    )
    if "predicate" in suppressed_fields:
        normalized.discard("predicate")
    if suppressed_fields.intersection({"object", "object_surface", "patient_or_theme"}):
        normalized.discard("object")
        if inferred_object:
            normalized.add("object")
    if "recipient" in suppressed_fields:
        normalized.discard("recipient")
    if "destination" in suppressed_fields:
        normalized.discard("destination")
    if "event_time" in suppressed_fields:
        normalized.discard("date")
    if not sanitized_predicate:
        normalized.discard("predicate")
    if not sanitized_object:
        normalized.discard("object")
        if inferred_object:
            normalized.add("object")
    if not sanitized_event_time:
        normalized.discard("date")
    if slot_frame == "generic_value_slot" and not governing_predicate:
        normalized -= {
            "predicate",
            "object",
            "recipient",
            "destination",
            "event_time",
            "date",
        }
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    left_context = clean_text(slot.get("local_context_before", ""))
    right_context = clean_text(slot.get("local_context_after", ""))
    combined_context = f"{left_context} {right_context}".upper()
    normalized_governing = governing_predicate
    if (
        slot_frame == "generic_value_slot"
        and expected_type in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}
    ):
        # Multi-slot locative clauses and preposition-led date tails often
        # produce a noisy "object" anchor even when the missing value is
        # simply one side of a diplomatic or transport relation. That object
        # gate overconstrains historically good VR cases, especially in viet.
        if "[REDACT:" in combined_context and "object" in normalized and not inferred_object:
            normalized.discard("object")
        if normalized_governing in {"from", "to", "in", "at", "on", "with", "for", "of"}:
            normalized.discard("predicate")
        if (
            normalized_governing in {"follow", "following"}
            and "object" in normalized
        ):
            normalized.discard("object")
            normalized.discard("predicate")
        if (
            "object" in normalized
            and re.search(
                r"\b(?:recognize|maintain|support|deal|relations?|treaty)\b",
                combined_context,
                flags=re.I,
            )
            and re.search(r"\bAS PART OF\b|\bTREATY WITH\b|\bRELATIONS WITH\b", combined_context)
        ):
            normalized.discard("object")
    if (
        slot_frame == "generic_value_slot"
        and expected_type in {"ORGANIZATION", "PERSON"}
        and _vr_infer_authority_trigger(slot)
    ):
        # In "under the guidance of [ORG]", the blank is the governing
        # authority.  A pronoun such as "it" is the action's object and is not
        # an independently useful anchor for identifying that authority.
        normalized.discard("object")
    return normalized or None


def _vr_source_table_priority(source_table: str) -> int:
    return {
        "kg_pseudo_chunks_redaction.csv": 4,
        "relations.csv": 3,
        "entities.csv": 2,
        "claims.csv": 1,
    }.get(clean_text(source_table), 0)


def _vr_slot_anchor_tokens(
    slot: Mapping[str, Any],
    slot_query_text: str,
    fragment_query_text: str,
    query_signature: Mapping[str, Any],
    context: EngineContext,
) -> Set[str]:
    local_probe_text = _vr_local_slot_probe_text(slot)
    generic_slot = clean_text(slot.get("slot_frame", "")) == "generic_value_slot"
    parts: List[str] = [
        local_probe_text or slot_query_text,
        fragment_query_text if not generic_slot else "",
        clean_text(slot.get("modified_head", "")),
        clean_text(slot.get("governing_predicate", "")),
        _structured_slot_probe(slot, query_signature),
    ]
    for field in (
        "actor",
        "patient_or_theme",
        "object",
        "recipient",
        "destination",
        "location_identity",
        "location",
        "event_time",
        "predicate",
    ):
        parts.append(_clean_slot_signature_value(query_signature.get(field, "")))
    tokens: Set[str] = set()
    for part in parts:
        for token in _normalized_expanded_tokens(part, context):
            if len(token) < 3:
                continue
            if token in _VR_GENERIC_ANCHOR_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def _vr_value_record_hint_score(
    value_record: Any,
    anchor_tokens: Set[str],
    predicate_tokens: Set[str],
    context: EngineContext,
) -> float:
    evidence_text = clean_text(
        " ".join(
            part
            for part in (
                clean_text(getattr(value_record, "evidence", "")),
                clean_text(getattr(value_record, "relation", "")),
                clean_text(getattr(value_record, "relation_side", "")),
            )
            if part
        )
    )
    evidence_tokens = _normalized_expanded_tokens(evidence_text, context)
    evidence_score = _token_score(anchor_tokens, evidence_tokens)
    predicate_score = containment(predicate_tokens, evidence_tokens) if predicate_tokens else 0.0
    priority_bonus = 0.025 * _vr_source_table_priority(
        clean_text(getattr(value_record, "source_table", ""))
    )
    return float(evidence_score + 0.30 * predicate_score + priority_bonus)


def _vr_ordered_context_tokens(text: str) -> List[str]:
    ordered: List[str] = []
    for match in re.finditer(
        r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?",
        clean_text(text),
    ):
        key = canonical_key(match.group(0))
        normalized = _VR_TOKEN_ALIASES.get(key, key)
        if (
            not normalized
            or len(normalized) < 3
            or normalized in _VR_GENERIC_ANCHOR_STOPWORDS
            or normalized in _VR_WEAK_GOVERNING_PREDICATES
        ):
            continue
        ordered.append(normalized)
    return ordered


def _vr_candidate_local_context_bonus(
    slot: Mapping[str, Any],
    candidate_value: str,
    evidence: str,
) -> float:
    left_anchor = set(
        _vr_ordered_context_tokens(
            clean_text(slot.get("local_context_before", ""))
        )[-4:]
    )
    right_anchor = set(
        _vr_ordered_context_tokens(
            clean_text(slot.get("local_context_after", ""))
        )[:4]
    )
    if not left_anchor and not right_anchor:
        return 0.0
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not candidate or not source:
        return 0.0
    candidate_re = re.compile(r"\b" + re.escape(candidate) + r"\b", flags=re.I)
    best_score = 0.0
    for match in candidate_re.finditer(source):
        before = source[max(0, match.start() - 96):match.start()]
        after = source[match.end():min(len(source), match.end() + 96)]
        before_tokens = set(_vr_ordered_context_tokens(before)[-6:])
        after_tokens = set(_vr_ordered_context_tokens(after)[:6])
        left_score = containment(left_anchor, before_tokens) if left_anchor else 0.0
        right_score = containment(right_anchor, after_tokens) if right_anchor else 0.0
        score = (
            0.40 * left_score + 0.60 * right_score
            if left_anchor and right_anchor
            else max(left_score, right_score)
        )
        if score > best_score:
            best_score = score
    return 0.18 * float(best_score)


def _vr_candidate_visible_in_query_context(
    slot: Mapping[str, Any],
    candidate_value: str,
    modified_head: str,
    context: EngineContext,
) -> bool:
    candidate = clean_text(candidate_value)
    if not candidate:
        return False
    candidate_re = re.compile(r"\b" + re.escape(candidate) + r"\b", flags=re.I)
    initialism = "".join(
        word[0].upper()
        for word in re.findall(r"[A-Za-z]+", candidate)
        if word.casefold() not in {"a", "an", "and", "of", "the"}
    )
    initialism_re = (
        re.compile(r"\b" + re.escape(initialism) + r"\b")
        if 2 <= len(initialism) <= 6
        else None
    )
    for source in (
        clean_text(slot.get("local_context_before", "")),
        clean_text(slot.get("local_context_after", "")),
        clean_text(modified_head),
    ):
        if not source:
            continue
        if candidate_re.search(source):
            return True
        if initialism_re and initialism_re.search(source):
            return True
        candidate_aliases = context.semantic_resources.concepts(
            candidate, "alias"
        )
        if candidate_aliases and candidate_aliases.intersection(
            context.semantic_resources.concepts(source, "alias")
        ):
            return True
    return False


def _vr_candidate_query_visibility_penalty(
    slot: Mapping[str, Any],
    candidate_value: str,
    modified_head: str,
    context: EngineContext,
) -> float:
    if not _vr_candidate_visible_in_query_context(
        slot,
        candidate_value,
        modified_head,
        context,
    ):
        return 0.0
    # A visible value can legitimately be repeated, so this is a bounded
    # ranking penalty rather than a hard rejection. In list-like slots it is
    # usually a distractor sibling and receives the stronger form.
    return 0.12 if _vr_slot_looks_enumerative(slot) else 0.06


def _vr_rejected_candidate_surface(value: str) -> bool:
    key = canonical_key(value)
    if not key:
        return True
    return key in {
        "none", "not applicable", "redacted", "slot", "someone",
        "something", "unknown", "unidentified", "unspecified",
    }


def _compact_value_records_for_doc(
    value_records: Tuple[Any, ...],
    *,
    slot: Mapping[str, Any],
    slot_query_text: str,
    fragment_query_text: str,
    query_signature: Mapping[str, Any],
    expected: str,
    context: EngineContext,
) -> List[Any]:
    if not value_records:
        return []
    broad_expected = expected in {
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "COUNTRY",
        "LOCATION_OR_COUNTRY",
        "MILITARY",
        "UNKNOWN",
    }
    generic_slot = clean_text(slot.get("slot_frame", "")) == "generic_value_slot"
    large_corpus = len(context.corpus.doc_ids) >= 100
    # Small corpora can afford every provenance surface for the same value.
    # Keeping only two can discard the one surface that binds the candidate to
    # the queried event even though all documents are being searched.
    per_identity_cap = 2 if large_corpus else len(value_records)
    if generic_slot and expected in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}:
        per_identity_cap = 4 if large_corpus else 5
    if broad_expected or generic_slot:
        canonical_value_cap = 24 if large_corpus else 36
    else:
        canonical_value_cap = 32 if large_corpus else 48
    anchor_tokens = _vr_slot_anchor_tokens(
        slot,
        slot_query_text,
        fragment_query_text,
        query_signature,
        context,
    )
    predicate_tokens = _normalized_expanded_tokens(
        clean_text(slot.get("governing_predicate", "")),
        context,
    )
    grouped: Dict[str, List[Tuple[float, int, int, Any]]] = defaultdict(list)
    protected_identities: Set[str] = set()
    for value_record in value_records:
        value = clean_text(getattr(value_record, "value", ""))
        if not value:
            continue
        if not type_compatible(expected, getattr(value_record, "value_type", ""), value):
            continue
        identity = canonical_value(value)
        hint_score = _vr_value_record_hint_score(
            value_record,
            anchor_tokens,
            predicate_tokens,
            context,
        )
        evidence_text = clean_text(getattr(value_record, "evidence", ""))
        local_context_bonus = _vr_candidate_local_context_bonus(
            slot,
            value,
            evidence_text,
        )
        hint_score += local_context_bonus
        local_argument_frame_pass = bool(
            evidence_text
            and _vr_local_argument_frame_pass(
                slot,
                query_signature,
                value,
                evidence_text,
                context,
            )
        )
        if local_argument_frame_pass:
            hint_score += 0.18
            # Never discard an identity whose provenance row occupies the
            # query's local slot frame merely because a long document has
            # more than the generic canonical-value cap. This is a recall
            # guard before full non-compensatory scoring, not an acceptance
            # shortcut.
            if local_context_bonus >= 0.09:
                protected_identities.add(identity)
        grouped[identity].append(
            (
                hint_score,
                _vr_source_table_priority(clean_text(getattr(value_record, "source_table", ""))),
                len(clean_text(getattr(value_record, "evidence", ""))),
                value_record,
            )
        )
    if not grouped:
        return []
    ranked_groups: List[Tuple[float, int, str, List[Any]]] = []
    for identity, scored_records in grouped.items():
        if generic_slot:
            scored_records.sort(
                key=lambda item: (-item[0], -item[1], item[2], str(item[3]))
            )
        else:
            scored_records.sort(
                key=lambda item: (-item[0], -item[1], -item[2], str(item[3]))
            )
        kept_records = [item[3] for item in scored_records[:per_identity_cap]]
        ranked_groups.append(
            (
                scored_records[0][0],
                scored_records[0][1],
                identity,
                kept_records,
            )
        )
    ranked_groups.sort(key=lambda item: (-item[0], -item[1], item[2]))
    if len(ranked_groups) > canonical_value_cap:
        best_hint = ranked_groups[0][0]
        kept_groups: List[Tuple[float, int, str, List[Any]]] = []
        for group in ranked_groups:
            if len(kept_groups) < canonical_value_cap:
                kept_groups.append(group)
                continue
            if group[2] in protected_identities:
                kept_groups.append(group)
                continue
            if best_hint > 0.0 and group[0] >= best_hint * 0.92:
                kept_groups.append(group)
        ranked_groups = kept_groups
    compacted: List[Any] = []
    for _score, _priority, _identity, records in ranked_groups:
        compacted.extend(records)
    return compacted


def _value_redaction_rank_documents(
    slot_query_text: str,
    fragment_query_text: str,
    slot: Mapping[str, Any],
    query_signature: Mapping[str, Any],
    context: EngineContext,
) -> List[tuple[str, float]]:
    document_limit = (
        len(context.corpus.doc_ids)
        if context.config.raw_use_mode == "discover"
        or context.config.retrieval_mode == "exhaustive"
        else context.config.staged_top_docs
    )
    normalized_signature = _sanitize_value_redaction_query_signature(
        slot,
        query_signature,
        context,
    )
    local_probe_text = _vr_local_slot_probe_text(slot)
    shortlist_limit = max(document_limit * 3, 120)
    # Keep the complete slot context as an independent retrieval route.  The
    # local probe is ranked separately below; replacing the full probe with it
    # can discard the rare right-side anchors that identify the source doc.
    raw_probe_text = slot_query_text or fragment_query_text
    structured_probe = _structured_slot_probe(slot, normalized_signature)
    left_context = clean_text(slot.get("local_context_before", ""))
    right_context = clean_text(slot.get("local_context_after", ""))
    head_probe = clean_text(slot.get("modified_head", ""))
    probe_cache: Dict[tuple[str, ...], List[tuple[str, float]]] = {}

    def _rank_probe_tokens(tokens: Set[str]) -> List[tuple[str, float]]:
        if not tokens:
            return []
        key = tuple(sorted(tokens))
        cached = probe_cache.get(key)
        if cached is not None:
            return cached
        ranked = rank_documents_tokens(
            tokens,
            context.corpus,
            shortlist_limit,
        )
        probe_cache[key] = ranked
        return ranked

    raw_query_tokens = _vr_retrieval_tokens(raw_probe_text, context)
    local_query_tokens = _vr_retrieval_tokens(local_probe_text, context)
    structured_tokens = _vr_retrieval_tokens(structured_probe, context)
    fragment_tokens = _vr_retrieval_tokens(fragment_query_text, context)
    left_tokens = _vr_retrieval_tokens(left_context, context)
    right_tokens = _vr_retrieval_tokens(right_context, context)
    head_tokens = _vr_retrieval_tokens(head_probe, context)

    raw_ranked = _rank_probe_tokens(raw_query_tokens)
    structured_ranked = _rank_probe_tokens(structured_tokens)
    fragment_ranked = (
        _rank_probe_tokens(fragment_tokens)
        if clean_text(fragment_query_text)
        and clean_text(fragment_query_text) != clean_text(slot_query_text)
        else []
    )
    left_ranked = _rank_probe_tokens(left_tokens)
    right_ranked = _rank_probe_tokens(right_tokens)
    head_ranked = _rank_probe_tokens(head_tokens)
    local_ranked = (
        _rank_probe_tokens(local_query_tokens)
        if local_probe_text and clean_text(local_probe_text) != clean_text(raw_probe_text)
        else []
    )
    raw_map = dict(raw_ranked)
    structured_map = dict(structured_ranked)
    fragment_map = dict(fragment_ranked)
    left_map = dict(left_ranked)
    right_map = dict(right_ranked)
    head_map = dict(head_ranked)
    local_map = dict(local_ranked)
    route_rankings = (
        raw_ranked,
        structured_ranked,
        fragment_ranked,
        left_ranked,
        right_ranked,
        head_ranked,
        local_ranked,
    )
    candidate_doc_ids = {
        doc_id
        for ranked in route_rankings
        for doc_id, _score in ranked
    }
    signature_ready = any(
        _clean_slot_signature_value(normalized_signature.get(field, ""))
        for field in (
            "actor",
            "patient_or_theme",
            "object",
            "recipient",
            "destination",
            "location_identity",
            "location",
            "event_time",
            "predicate",
        )
    )
    weak_generic_signature = (
        clean_text(slot.get("slot_frame", "")) == "generic_value_slot"
        and not clean_text(normalized_signature.get("predicate", ""))
    )
    cheap_components: Dict[str, tuple[float, float, float, float, float]] = {}
    cheap_rank: List[tuple[str, float]] = []
    for doc_id in sorted(candidate_doc_ids):
        # Apply the same VR paraphrase normalization to both sides.  Previously
        # only the query was mapped (for example ``postpone`` -> ``delay``), so
        # an otherwise exact document using the alternate surface could fall
        # outside the bounded retrieval set.
        doc_tokens = {
            token if ":" in token else _VR_TOKEN_ALIASES.get(token, token)
            for token in canonical_tokens(
                context.corpus.document_tokens.get(doc_id, frozenset())
            )
        }
        raw_score = max(
            float(raw_map.get(doc_id, 0.0) or 0.0),
            float(fragment_map.get(doc_id, 0.0) or 0.0),
            _token_score(raw_query_tokens, doc_tokens),
        )
        local_score = max(
            float(local_map.get(doc_id, 0.0) or 0.0),
            _token_score(local_query_tokens, doc_tokens),
        ) if local_probe_text else 0.0
        structured_score = max(
            float(structured_map.get(doc_id, 0.0) or 0.0),
            _token_score(structured_tokens, doc_tokens),
        )
        left_score = float(left_map.get(doc_id, 0.0) or 0.0)
        right_score = float(right_map.get(doc_id, 0.0) or 0.0)
        head_score = float(head_map.get(doc_id, 0.0) or 0.0)
        context_score = 0.0
        if left_score and right_score:
            context_score = 0.6 * min(left_score, right_score) + 0.4 * max(
                left_score,
                right_score,
            )
        else:
            context_score = max(left_score, right_score)
        if weak_generic_signature:
            cheap_score = (
                0.32 * max(raw_score, structured_score)
                + 0.18 * local_score
                + 0.25 * context_score
                + 0.05 * head_score
            )
        else:
            cheap_score = (
                0.35 * max(raw_score, structured_score)
                + 0.20 * context_score
                + 0.10 * head_score
            )
        cheap_components[doc_id] = (
            raw_score, local_score, structured_score, context_score, head_score
        )
        cheap_rank.append((doc_id, cheap_score))

    # Event-identity comparison is substantially more expensive than token
    # retrieval.  Preserve deep coverage from every independent route, then
    # spend proposition work only on the strongest combined candidates.
    rerank_pool_limit = max(document_limit * 4, 240)
    if len(candidate_doc_ids) > rerank_pool_limit:
        route_depth = min(shortlist_limit, max(32, document_limit // 2))
        rerank_doc_ids = {
            doc_id
            for ranked in route_rankings
            for doc_id, _score in ranked[:route_depth]
        }
        cheap_rank.sort(key=lambda item: (-item[1], item[0]))
        for doc_id, _score in cheap_rank:
            if len(rerank_doc_ids) >= rerank_pool_limit:
                break
            rerank_doc_ids.add(doc_id)
        context.control.emit(
            "value_redaction_document_rerank_pool",
            retrieved_document_count=len(candidate_doc_ids),
            rerank_document_count=len(rerank_doc_ids),
            route_recall_depth=route_depth,
        )
    else:
        rerank_doc_ids = candidate_doc_ids

    reranked: List[tuple[str, float]] = []
    for doc_id in sorted(rerank_doc_ids):
        (
            raw_score,
            local_score,
            structured_score,
            context_score,
            head_score,
        ) = cheap_components[doc_id]
        proposition_score = 0.0
        if signature_ready:
            for record in context.corpus.propositions_by_doc.get(doc_id, ()):
                identity, _states = event_identity(
                    normalized_signature,
                    record.signature,
                    slot_query_text or fragment_query_text or structured_probe,
                    record.text,
                    context.semantic_resources,
                )
                if identity > proposition_score:
                    proposition_score = identity
        if weak_generic_signature:
            final_score = (
                0.32 * max(raw_score, structured_score)
                + 0.18 * local_score
                + 0.25 * context_score
                + 0.05 * head_score
                + 0.20 * proposition_score
            )
        else:
            final_score = (
                0.35 * max(raw_score, structured_score)
                + 0.20 * context_score
                + 0.10 * head_score
                + 0.35 * proposition_score
            )
        # A noisy extracted proposition must not erase strong agreement from
        # the complete query and its two-sided slot context.  This independent
        # retrieval score preserves semantically direct documents while the
        # candidate-level role and event gates still decide the actual fill.
        retrieval_only_score = (
            0.70 * max(raw_score, structured_score)
            + 0.30 * context_score
        )
        final_score = max(final_score, retrieval_only_score)
        reranked.append((doc_id, float(final_score)))
    reranked.sort(key=lambda item: (-item[1], item[0]))
    return reranked[: max(1, document_limit)]


def _bounded_context(text: str, needles: List[str], radius: int = 260) -> str:
    source = clean_text(text)
    lower = source.lower()
    spans = []
    for needle in needles:
        value = clean_text(needle).lower()
        if not value:
            continue
        position = lower.find(value)
        if position >= 0:
            spans.append((
                max(0, position - radius),
                min(len(source), position + len(value) + radius),
            ))
    if not spans:
        return source[: radius * 2]
    return " [...] ".join(source[start:end] for start, end in sorted(spans)[:4])


def _candidate_local_clause(
    document_text: str,
    record_evidence: str,
    value: str,
    predicate: str,
    preferred_terms: Optional[List[str]] = None,
) -> str:
    """Return the shortest clause containing value and governing predicate.

    This prevents an entity in a reporting clause from inheriting a match from
    a different embedded clause elsewhere in the same document.
    """
    preferred_tokens = {
        _VR_TOKEN_ALIASES.get(token, token)
        for term in (preferred_terms or [])
        for token in semantic_tokens(term)
        if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    }
    query_has_number = any(
        re.search(r"\b\d+(?:\.\d+)?\b", clean_text(term))
        for term in (preferred_terms or [])
    )
    preferred_numbers = {
        match.group(0)
        for term in (preferred_terms or [])
        for match in re.finditer(r"\b\d+(?:\.\d+)?\b", clean_text(term))
    }

    def anchor_overlap(clause: str) -> float:
        if not preferred_tokens:
            return 0.0
        clause_tokens = {
            _VR_TOKEN_ALIASES.get(token, token)
            for token in semantic_tokens(clause)
            if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
        }
        if not clause_tokens:
            return 0.0
        return float(containment(preferred_tokens, clause_tokens))

    value_re = re.compile(r"\b" + re.escape(value) + r"\b", flags=re.I)
    predicate_re = re.compile(r"\b" + _predicate_pattern(predicate) + r"\b", flags=re.I)
    def candidate_windows(source: str) -> Tuple[List[str], List[str]]:
        exact: List[str] = []
        value_only: List[str] = []
        text = clean_text(source)
        if not text:
            return exact, value_only
        for match in value_re.finditer(text):
            lower_bound = max(0, match.start() - 420)
            upper_bound = min(len(text), match.end() + 520)
            left_candidates = [
                text.rfind(mark, lower_bound, match.start())
                for mark in (". ", "? ", "! ", "; ")
            ]
            left = max([position for position in left_candidates if position >= 0], default=lower_bound)
            if left > lower_bound:
                left += 2
            right_candidates = [
                position
                for mark in (". ", "? ", "! ", "; ")
                for position in [text.find(mark, match.end(), upper_bound)]
                if position >= 0
            ]
            right = min(right_candidates, default=upper_bound)
            if right < upper_bound:
                right += 1
            window = clean_text(text[left:right])
            if not window:
                continue
            value_only.append(window)
            if not predicate or predicate_re.search(window):
                exact.append(window)
        return exact, value_only

    def best_clause(clauses: List[str]) -> str:
        # Prefer anchor-rich evidence, then the shortest sufficient clause.
        return max(
            clauses,
            key=lambda clause: (
                containment(
                    preferred_numbers,
                    set(re.findall(r"\b\d+(?:\.\d+)?\b", clause)),
                ) if preferred_numbers else 0.0,
                anchor_overlap(clause),
                bool(
                    query_has_number
                    and re.search(r"\b\d+(?:\.\d+)?\b", clause)
                ),
                -len(semantic_tokens(clause)),
                -len(clause),
            ),
        )

    # Compare all candidate-local windows instead of trusting whichever
    # extracted representation happened to be visited first.  Every document
    # window still has to contain the candidate and governing predicate in the
    # same bounded clause, so this does not permit cross-event borrowing.
    record_exact, record_value_only = candidate_windows(record_evidence)
    document_exact, document_value_only = candidate_windows(document_text)
    exact = list(dict.fromkeys(record_exact + document_exact))
    if exact:
        return best_clause(exact)
    value_only = list(dict.fromkeys(record_value_only + document_value_only))
    if value_only:
        return best_clause(value_only) if preferred_tokens else min(value_only, key=len)
    return _bounded_context(
        clean_text(record_evidence + " " + document_text),
        [value, predicate, *(preferred_terms or [])],
        radius=260,
    )


def _title_identity_pass(
    value: str,
    title: str,
    document_text: str,
    candidate_evidence: str,
    containment_floor: float,
    strict_local: bool = False,
) -> bool:
    """Verify a PERSON name adjacent to or locally supported by a title."""
    title_words = [
        word for word in semantic_tokens(title)
        if len(word) >= 3 and word not in {"with", "from", "that", "this"}
    ]
    title_tokens = {
        token for token in title_words
        if len(token) >= 4 and token not in {"with", "from", "that", "this"}
    }
    distinctive_title_tokens = title_tokens - {
        "acting", "assistant", "chairman", "chairperson", "chairwoman",
        "chief", "commander", "deputy", "director", "governor", "head",
        "leader", "minister", "official", "president", "representative",
        "secretary",
    }
    title_key = canonical_key(title)
    if re.search(
        r"\b(?:mp|member of parliament|parliamentarian)\b",
        title_key,
    ):
        descriptor_tokens = [
            token
            for token in semantic_tokens(title)
            if token not in {
                "mp", "member", "of", "parliament", "parliamentarian",
            }
        ]
        descriptor = (
            r"(?:"
            + r"\s+".join(re.escape(token) for token in descriptor_tokens)
            + r"\s+)?"
            if descriptor_tokens
            else ""
        )
        parliamentary_title = (
            descriptor
            + r"(?:m\.?p\.?|member\s+of\s+parliament|parliamentarian)"
        )
        candidate = re.escape(clean_text(value))
        adjacency = re.compile(
            r"(?:\b" + parliamentary_title + r"\s+" + candidate + r"\b|"
            r"\b" + candidate + r"\b\s*[,;(]\s*" + parliamentary_title + r"\b)",
            flags=re.I,
        )
        return any(
            bool(adjacency.search(clean_text(source)))
            for source in (candidate_evidence, document_text)
            if clean_text(source)
        )
    if not title_tokens:
        return False
    title_pattern = r"\b" + r"\s+".join(re.escape(word) for word in title_words) + r"\b"
    value_re = re.compile(r"\b" + re.escape(value) + r"\b", flags=re.I)
    sources = (candidate_evidence,) if strict_local else (candidate_evidence, document_text)
    for source in sources:
        text = clean_text(source)
        if not text:
            continue
        if strict_local:
            for match in value_re.finditer(text):
                before = text[max(0, match.start() - 140):match.start()]
                after = text[match.end():min(len(text), match.end() + 180)]
                if (
                    distinctive_title_tokens
                    and re.match(r"^\s*[,;:(-]", after)
                    and containment(
                        distinctive_title_tokens,
                        semantic_tokens(after),
                    ) >= max(0.60, containment_floor)
                ):
                    return True
                if (
                    distinctive_title_tokens
                    and re.search(
                        r"\b(?:m\.?p\.?|member\s+of\s+parliament|"
                        r"parliamentarian|ambassador|chair(?:man|woman|person)?|chief|"
                        r"commander|director|governor|head|leader|minister|"
                        r"president|representative|secretary)\b[^,;:.]{0,100}$",
                        before,
                        flags=re.I,
                    )
                    and containment(
                        distinctive_title_tokens,
                        semantic_tokens(before),
                    ) >= max(0.60, containment_floor)
                ):
                    return True
            continue
        if re.search(
            title_pattern
            + r"(?:[\s,;:()/-]+\w+){0,4}[\s,;:()/-]+"
            + re.escape(value)
            + r"\b",
            text,
            flags=re.I,
        ):
            return True
        # Value-before-title is only reliable when it is an explicit appositive
        # or parenthetical, not a loose narrative span like
        # "Asnaj expressed fear that Somali President...".
        if re.search(
            r"\b"
            + re.escape(value)
            + r"\b(?:\s*,\s*|\s*\(\s*|\s*-\s*|\s+the\s+)"
            + title_pattern,
            text,
            flags=re.I,
        ):
            return True
        for match in value_re.finditer(text):
            direct_appositive = text[
                match.end():min(len(text), match.end() + 180)
            ]
            if (
                distinctive_title_tokens
                and re.match(r"^\s*[,;:(-]", direct_appositive)
                and containment(
                    distinctive_title_tokens,
                    semantic_tokens(direct_appositive),
                ) >= max(0.60, containment_floor)
            ):
                return True
            nearby_windows = (
                text[max(0, match.start() - 36):match.start()],
                text[match.end():min(len(text), match.end() + 36)],
            )
            if any(
                containment(title_tokens, semantic_tokens(window)) >= containment_floor
                for window in nearby_windows
            ):
                return True
    return False


def _vr_local_argument_frame_pass(
    slot: Mapping[str, Any],
    query_signature: Mapping[str, Any],
    candidate_value: str,
    evidence: str,
    context: EngineContext,
) -> bool:
    slot_frame = clean_text(slot.get("slot_frame", ""))
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if slot_frame != "generic_value_slot":
        return True
    if expected_type not in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}:
        return True
    destination_trigger = _vr_infer_location_movement_destination_trigger(slot)
    if destination_trigger:
        return _vr_local_location_movement_destination_pass(
            destination_trigger,
            candidate_value,
            evidence,
        )
    inferred_object = _vr_infer_generic_object_anchor(slot)
    object_anchor = inferred_object or _clean_slot_signature_value(
        query_signature.get("object", "") or query_signature.get("patient_or_theme", "")
    )
    right_context = clean_text(slot.get("local_context_after", ""))
    prep_match = re.match(
        r"^\s*(by|from|against|on)\b",
        right_context,
        flags=re.I,
    )
    required_prep = clean_text(prep_match.group(1)) if prep_match else ""
    if not object_anchor and not required_prep:
        return True
    candidate_text = clean_text(candidate_value)
    source = clean_text(evidence)
    if not candidate_text or not source:
        return False
    candidate_re = re.compile(r"\b" + re.escape(candidate_text) + r"\b", flags=re.I)
    object_terms = [
        token
        for token in _normalized_expanded_tokens(object_anchor, context)
        if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    ]
    object_tokens = set(object_terms)
    anchor_pattern = (
        re.compile(
            r"\b(?:"
            + "|".join(re.escape(token) for token in sorted(object_tokens))
            + r")\b",
            flags=re.I,
        )
        if object_tokens
        else None
    )
    prep_pattern = (
        re.compile(r"\b" + re.escape(required_prep) + r"\b", flags=re.I)
        if required_prep
        else None
    )

    def token_gap(text: str) -> int:
        return len([token for token in semantic_tokens(text) if token])

    for match in candidate_re.finditer(source):
        before_window = source[max(0, match.start() - 120):match.start()]
        after_window = source[match.end():min(len(source), match.end() + 120)]
        if not object_tokens:
            if not prep_pattern or prep_pattern.search(after_window[:72]):
                return True
            continue
        # Case 1: "... defeat of SOMALIA by ..."
        if anchor_pattern:
            for anchor_match in reversed(list(anchor_pattern.finditer(before_window))):
                if token_gap(before_window[anchor_match.end():]) > 6:
                    continue
                if not prep_pattern or prep_pattern.search(after_window[:72]):
                    return True
            # Case 2: "SOMALIA will be defeated by ..."
            for anchor_match in anchor_pattern.finditer(after_window):
                if token_gap(after_window[:anchor_match.start()]) > 6:
                    continue
                tail = after_window[anchor_match.end():anchor_match.end() + 72]
                if not prep_pattern or prep_pattern.search(tail):
                    return True
    return False


def _vr_local_recipient_pass(
    trigger_verb: str,
    candidate_value: str,
    evidence: str,
) -> bool:
    verb = clean_text(trigger_verb)
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not verb or not candidate or not source:
        return False
    candidate_re = re.compile(r"\b" + re.escape(candidate) + r"\b", flags=re.I)
    if canonical_key(verb) in {
        "tell", "told", "ask", "asked", "inform", "informed", "brief",
        "briefed", "advise", "advised", "warn", "warned", "notify",
        "notified", "urge", "urged",
    }:
        verb_re = re.compile(
            r"\b(?:tell|tells|told|ask|asks|asked|inform|informs|informed|"
            r"brief|briefs|briefed|advise|advises|advised|warn|warns|warned|"
            r"notify|notifies|notified|urge|urges|urged)\b",
            flags=re.I,
        )
    else:
        verb_re = re.compile(r"\b" + re.escape(verb) + r"\b", flags=re.I)
    for match in candidate_re.finditer(source):
        before = source[max(0, match.start() - 140):match.start()]
        after = source[match.end():min(len(source), match.end() + 140)]
        verb_matches = list(verb_re.finditer(before))
        if not verb_matches:
            continue
        if re.search(r"\bthat\b", before, flags=re.I):
            continue
        intervening = semantic_tokens(before[verb_matches[-1].end():])
        # The recipient relation is licensed by local attachment to the
        # communication verb. A following ``that`` clause is common but not
        # required ("told X at a hearing it was too late").
        if len(intervening) <= 4 and (
            re.search(r"\bthat\b", after, flags=re.I)
            or len(intervening) <= 2
        ):
            return True
    return False


def _vr_local_copular_complement_pass(
    candidate_value: str,
    evidence: str,
) -> bool:
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not candidate or not source:
        return False
    return bool(re.search(
        r"\b(?:was|were|is|are)\b"
        r"(?:\s+[A-Za-z0-9'\"().-]+){0,6}\s+"
        + re.escape(candidate)
        + r"\b(?:\s*,\s*|\s+)(?:identified|described|known|reported|named)\b",
        source,
        flags=re.I,
    ))


def _vr_local_prepositional_complement_pass(
    trigger_phrase: str,
    candidate_value: str,
    evidence: str,
) -> bool:
    trigger = clean_text(trigger_phrase)
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not trigger or not candidate or not source:
        return False
    if trigger == "container_of":
        pattern = _VR_CONTAINER_OF_PATTERN.pattern
    else:
        pattern = re.escape(trigger)
    return bool(re.search(
        pattern
        + r"(?:\s+[A-Za-z0-9'\"().-]+){0,4}\s+"
        + re.escape(candidate)
        + r"\b",
        source,
        flags=re.I,
    ))


def _vr_local_membership_container_pass(
    container_text: str,
    candidate_value: str,
    evidence: str,
    context: EngineContext,
) -> bool:
    container = clean_text(container_text)
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not container or not candidate or not source:
        return False
    candidate_match = re.search(r"\b" + re.escape(candidate) + r"\b", source, flags=re.I)
    if not candidate_match:
        return False
    container_tokens = {
        token
        for token in _normalized_expanded_tokens(container, context)
        if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    }
    if not container_tokens:
        return False
    before = source[max(0, candidate_match.start() - 160):candidate_match.start()]
    before_tokens = {
        token
        for token in _normalized_expanded_tokens(before, context)
        if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    }
    if containment(container_tokens, before_tokens) < 0.6:
        return False
    if re.search(r"\bincluding\b", before[-48:], flags=re.I):
        return True
    after = source[candidate_match.end():min(len(source), candidate_match.end() + 80)]
    if re.search(r"^\s*,?\s*(?:who\s+was\s+)?identified\b", after, flags=re.I):
        return True
    return False


def _vr_local_location_prepositional_pass(
    trigger_phrase: str,
    prep: str,
    candidate_value: str,
    evidence: str,
    context: EngineContext,
) -> bool:
    phrase = clean_text(trigger_phrase)
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not phrase or not prep or not candidate or not source:
        return False
    phrase_tokens = {
        token
        for token in _normalized_expanded_tokens(phrase, context)
        if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    }
    candidate_re = re.compile(r"\b" + re.escape(candidate) + r"\b", flags=re.I)
    for candidate_match in candidate_re.finditer(source):
        before = source[max(0, candidate_match.start() - 120):candidate_match.start()]
        # The explicit preposition immediately binding the candidate is a
        # local grammatical fact.  Do not reject it merely because a semantic
        # rewrite changed the wording in the earlier part of the clause.
        if re.search(
            r"\b" + re.escape(prep) + r"\b(?:\s+the)?\s*$",
            before,
            flags=re.I,
        ):
            return True
        before_tokens = {
            token
            for token in _normalized_expanded_tokens(before, context)
            if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
        }
        if phrase_tokens and containment(phrase_tokens, before_tokens) < 0.5:
            continue
        # Semantic rewrites may change a locative/target preposition while
        # preserving the anchored event (for example "in" versus "against").
        # Permit that drift only when the surrounding phrase is strongly
        # aligned and another explicit preposition binds the candidate.
        phrase_alignment = (
            containment(phrase_tokens, before_tokens)
            if phrase_tokens else 0.0
        )
        if phrase_alignment >= 0.75 and re.search(
            r"\b(?:in|at|on|against|within|across|throughout|near|by|"
            r"of|from|to|for|with)\b"
            r"(?:\s+the)?\s*$",
            before,
            flags=re.I,
        ):
            return True


def _vr_infer_authority_trigger(slot: Mapping[str, Any]) -> str:
    """Return a local authority phrase immediately governing an entity slot."""
    expected_type = clean_text(
        slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
    ).upper()
    if expected_type not in {"ORGANIZATION", "PERSON"}:
        return ""
    left = clean_text(slot.get("local_context_before", ""))
    match = re.search(
        r"\b(?:under|with|at|on)\s+(?:the\s+)?"
        r"(?P<authority>(?:administrative\s+)?(?:guidance|direction|"
        r"authority|supervision|orders?))\s+of\s*$",
        left,
        flags=re.I,
    )
    return clean_text(match.group("authority")) if match else ""


def _vr_local_authority_pass(
    authority_trigger: str,
    candidate_value: str,
    evidence: str,
) -> bool:
    """Verify that a candidate locally fills an authority-of construction."""
    trigger = clean_text(authority_trigger)
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not trigger or not candidate or not source:
        return False
    return bool(re.search(
        r"\b(?:under|with|at|on)\s+(?:the\s+)?"
        r"(?:administrative\s+)?"
        + re.escape(trigger)
        + r"\s+of\s+(?:the\s+)?"
        + re.escape(candidate)
        + r"\b",
        source,
        flags=re.I,
    ))
    return False


def _vr_local_leading_locative_phrase_pass(
    trigger_phrase: str,
    prep: str,
    candidate_value: str,
    evidence: str,
    context: EngineContext,
) -> bool:
    phrase = clean_text(trigger_phrase)
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not phrase or not prep or not candidate or not source:
        return False
    phrase_tokens = {
        token
        for token in _normalized_expanded_tokens(phrase, context)
        if len(token) >= 4 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    }
    candidate_tokens = {
        token
        for token in _normalized_expanded_tokens(candidate, context)
        if len(token) >= 2 and token not in _VR_GENERIC_ANCHOR_STOPWORDS
    }
    if not candidate_tokens:
        return False
    source_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'().-]*", source)
    normalized_source = [
        _VR_TOKEN_ALIASES.get(canonical_key(token), canonical_key(token))
        for token in source_tokens
        if canonical_key(token)
    ]
    candidate_seq = [
        _VR_TOKEN_ALIASES.get(canonical_key(token), canonical_key(token))
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'().-]*", candidate)
        if canonical_key(token)
    ]
    if not candidate_seq:
        return False
    for index in range(len(normalized_source) - len(candidate_seq) + 1):
        if normalized_source[index : index + len(candidate_seq)] != candidate_seq:
            continue
        window_start = max(0, index - 6)
        before = normalized_source[window_start:index]
        if before and before[0] == canonical_key(prep):
            before = before[1:]
        if phrase_tokens and containment(phrase_tokens, set(before)) < 0.5:
            continue
        if window_start > 0 and normalized_source[window_start - 1] != canonical_key(prep):
            continue
        if window_start == 0 and canonical_key(prep) not in normalized_source[:index]:
            continue
        return True
    return False


def _vr_local_person_infinitive_subject_pass(
    candidate_value: str,
    evidence: str,
) -> bool:
    candidate = clean_text(candidate_value)
    source = clean_text(evidence)
    if not candidate or not source:
        return False
    return bool(re.search(
        r"\bfor\s+"
        + re.escape(candidate)
        + r"\b\s+to\b",
        source,
        flags=re.I,
    ))


def _vr_person_like_unknown_surface(value: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", clean_text(value))
    if not words or len(words) > 5:
        return False
    reject_tokens = {
        "board", "committee", "government", "ministry", "embassy",
        "department", "office", "union", "school", "study", "house",
        "agency", "airbase", "base", "investigation", "operation",
        "report", "states", "conference", "event", "meeting", "summit",
    }
    lowered = {word.lower() for word in words}
    if lowered & reject_tokens:
        return False
    if len(words) == 1 and words[0].isupper():
        return words[0] not in {"USIS", "USAID", "CIA", "FBI", "UN", "US", "UK"}
    return all(word.isupper() or word[:1].isupper() for word in words)


def _subject_before_predicate(value: str, predicate: str, clause: str) -> bool:
    """Require a direct subject-to-predicate span with no reporting boundary."""
    value_matches = list(re.finditer(r"\b" + re.escape(value) + r"\b", clause, flags=re.I))
    predicate_matches = list(re.finditer(
        r"\b" + _predicate_pattern(predicate) + r"\b",
        clause,
        flags=re.I,
    ))
    if not value_matches or not predicate_matches:
        return False
    for predicate_match in predicate_matches:
        prior = [
            match for match in value_matches
            if match.end() <= predicate_match.start()
        ]
        for match in reversed(prior):
            span = clause[match.end():predicate_match.start()]
            if len(span) > 90:
                continue
            if re.search(r"[.;:!?]", span):
                continue
            if len([token for token in semantic_tokens(span) if token]) > 3:
                continue
            # ``KND informed ... that National Police seized``: KND is not the
            # grammatical subject of seized, even though both occur in one sentence.
            reporting_boundary = re.search(
                r"\b(?:reported|informed|said|told|stated|claimed|claims|noted|notes)\b|\bthat\b",
                span,
                flags=re.I,
            )
            if reporting_boundary is None:
                return True
    return False


def _embedded_subject_before_copular(
    value: str,
    clause: str,
) -> bool:
    candidate = clean_text(value)
    source = clean_text(clause)
    if not candidate or not source:
        return False
    return bool(re.search(
        r"\bthat\s+" + re.escape(candidate) + r"\b\s+(?:is|are|was|were)\b",
        source,
        flags=re.I,
    ))


def _role_predicate_surface(
    slot: Mapping[str, Any],
    normalized_predicate: str,
) -> str:
    if clean_text(normalized_predicate):
        return clean_text(normalized_predicate)
    raw = clean_text(
        slot.get("governing_predicate_surface", "")
        or slot.get("governing_predicate", "")
    )
    return canonical_predicate(raw)


def _critical_event_gates(
    query_signature: Dict[str, Any],
    candidate_signature: Dict[str, Any],
    query_text: str,
    evidence: str,
    resources: Any = None,
    required_fields: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    score, states = event_identity(
        query_signature,
        candidate_signature,
        query_text,
        evidence,
        resources,
    )
    inferred_required = {
        "predicate_state": bool(clean_text(query_signature.get("predicate", ""))),
        "object_state": bool(clean_text(query_signature.get("object", ""))),
        "location_state": bool(clean_text(
            query_signature.get("location_identity", query_signature.get("location", ""))
        )),
        "date_state": bool(clean_text(query_signature.get("event_time", ""))),
        "packet_count_state": query_signature.get("packet_count") is not None,
        "total_weight_grams_state":
            query_signature.get("total_weight_grams") is not None,
    }
    required = {
        key: (
            value
            if required_fields is None
            else key.replace("_state", "") in required_fields
        )
        for key, value in inferred_required.items()
    }
    gate_pass = {
        key: (not required[key] or states.get(key) == "MATCH")
        for key in required
    }
    return {
        "event_identity_score": score,
        **states,
        **{key.replace("_state", "_gate_required"): value for key, value in required.items()},
        **{key.replace("_state", "_gate_pass"): value for key, value in gate_pass.items()},
        "critical_event_gate_pass": all(gate_pass.values()),
    }


def _predicate_pattern(predicate: str) -> str:
    """Return surface forms for one canonical predicate.

    Canonicalization is used for semantic comparison, but a canonical lemma
    must never be searched as though it were the original surface verb.
    """
    variants = {
        "be": r"(?:is|are|was|were|be|been|being)",
        "have": r"(?:has|have|had)",
        "claim": r"(?:claim(?:ed|s|ing)?)",
        "report": r"(?:report(?:ed|s)?|inform(?:ed|s)?|say|said|tell|told|stat(?:e|ed|es))",
        "seize": r"(?:seiz(?:e|ed|es)|seizure|confiscat(?:e|ed|es))",
        "arrest": r"(?:arrest(?:ed|s)?)",
        "detain": r"(?:detain(?:ed|s|ing)?|imprison(?:ed|s|ing)?|incarcerat(?:e|ed|es|ing))",
        "transport": (
            r"(?:transport(?:ed|s|ing)?|"
            r"direct(?:ed|s|ing)?|"
            r"send|sent|sends|sending|"
            r"move(?:d|s|ing)?|"
            r"ship(?:ped|s|ping)?|"
            r"travel(?:ed|led|s|ing)?|"
            r"come|comes|came|coming|"
            r"arriv(?:e|ed|es|ing)|"
            r"leave|leaves|left|leaving|"
            r"depart(?:ed|s|ing)?|"
            r"head(?:ed|s|ing)?|"
            r"go|goes|going|went|"
            r"deliver(?:ed|s|ing)?|"
            r"transfer(?:red|s|ring)?)"
        ),
        "direct": r"(?:direct(?:ed|s|ing|ion|ions)?)",
        "attend": r"(?:attend(?:ed|s|ing)?|represent(?:ed|s|ing)?)",
        "decide": r"(?:decid(?:e|ed|es|ing)|(?:made?|make|makes|making)\s+(?:a\s+)?decision|decision(?:s)?)",
        "identity_compare": r"(?:differ(?:ed|s|ing)?|different|not\s+(?:the\s+)?same)",
        "aim": (
            r"(?:aim(?:ed|s|ing)?|intend(?:ed|s|ing)?|seek|seeks|sought|"
            r"(?:is|are|was|were|be|been|being)\s+(?:particularly\s+)?"
            r"concerned\s+to)"
        ),
    }
    if predicate in variants:
        return variants[predicate]
    forms = sorted(
        {
            surface
            for surface, lemma in PREDICATE_LEMMAS.items()
            if lemma == predicate and clean_text(surface)
        },
        key=len,
        reverse=True,
    )
    if forms:
        return r"(?:" + "|".join(re.escape(form) for form in forms) + r")"
    return re.escape(predicate) + r"(?:d|ed|s|ing)?"


def _candidate_event_signature(
    evidence: str,
    governing_predicate: str,
) -> tuple[Dict[str, Any], bool]:
    """Build a local signature without inventing an absent predicate.

    Production entity/proposition records can prefix the natural-language
    proposition with relation labels and document metadata.  The generic
    clause parser is deliberately start-anchored, so those prefixes can hide
    an otherwise explicit action such as ``National Police ... seized
    object``.  Once the governing predicate is independently observed on the
    candidate-local surface, it is safe to supply its canonical lemma as an
    extraction anchor.  The returned surface flag remains a separate,
    noncompensatory eligibility gate.
    """
    surface_pass = (
        not governing_predicate
        or bool(re.search(
            r"\b" + _predicate_pattern(governing_predicate) + r"\b",
            evidence,
            flags=re.I,
        ))
    )
    signature = build_event_signature(
        evidence,
        predicate=governing_predicate if surface_pass else "",
    )
    if surface_pass and governing_predicate:
        # Nested-clause detection can otherwise replace an observed governing
        # verb ("directed ... to buy") with the subordinate verb ("buy").
        signature["predicate"] = governing_predicate
        signature["predicate_surface"] = governing_predicate
    return signature, surface_pass


def _candidate_identity_tokens(value: Any) -> Set[str]:
    tokens: Set[str] = set()
    for token in canonical_value(clean_text(value)).split():
        if len(token) >= 3:
            tokens.add(token)
        elif len(token) == 2 and token.isalpha():
            tokens.add(token)
    return tokens


def _same_family_margin_duplicate(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_type = clean_text(left.get("candidate_type", "")).upper()
    right_type = clean_text(right.get("candidate_type", "")).upper()
    geographic_types = {"COUNTRY", "GPE", "LOCATION", "LOCATION_OR_COUNTRY"}
    if left_type != right_type and not (
        left_type in geographic_types and right_type in geographic_types
    ):
        return False
    left_tokens = _candidate_identity_tokens(left.get("candidate", ""))
    right_tokens = _candidate_identity_tokens(right.get("candidate", ""))
    if not left_tokens or not right_tokens or left_tokens == right_tokens:
        return False
    smaller, larger = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    if not smaller < larger:
        return False
    if bool(left.get("eligible", False)) != bool(right.get("eligible", False)):
        return False
    same_document = clean_text(left.get("doc_id", "")) == clean_text(
        right.get("doc_id", "")
    )
    score_gap = abs(
        float(left.get("final_score", 0.0) or 0.0)
        - float(right.get("final_score", 0.0) or 0.0)
    )
    same_evidence = clean_text(left.get("evidence", "")) and (
        clean_text(left.get("evidence", ""))
        == clean_text(right.get("evidence", ""))
    )
    if (
        not same_document
        and left_type == right_type == "PERSON"
        and len(smaller) == 1
        and len(larger) <= 4
        and same_evidence
        and score_gap <= 0.035
    ):
        # Duplicate documents can expose a full name and its surname as two
        # candidates. Identical evidence and a near tie make them one answer
        # family rather than a meaningful decision competitor.
        return True
    if not same_document:
        person_title_glue = {
            "acting", "ambassador", "chairman", "chief", "colonel",
            "deputy", "director", "dr", "foreign", "general", "governor",
            "leader", "minister", "mr", "mrs", "president", "prime",
            "professor", "secretary", "sir", "vice",
        }
        if (
            left_type == right_type == "PERSON"
            and len(smaller) >= 1
            and (larger - smaller) <= person_title_glue
            and score_gap <= 0.035
        ):
            # A titled and untitled mention of the same retained name is one
            # answer family, even when duplicate source documents expose the
            # two surfaces separately.
            return True
        # Cross-document mentions may use a short official name and its full
        # constitutional form, e.g. "Russian Government" versus
        # "Government of the Russian Federation".  Collapse only when the
        # shorter form carries a shared distinctive token and every added
        # token is institutional glue; arbitrary nested names remain distinct.
        institutional_glue = {
            "administration", "democratic", "federal", "federation",
            "government", "kingdom", "national", "of", "people",
            "peoples", "republic", "state", "states", "the",
        }
        distinctive_shared = smaller - institutional_glue
        return bool(
            len(smaller) >= 2
            and distinctive_shared
            and (larger - smaller) <= institutional_glue
        )
    if score_gap > 1e-9:
        if not same_evidence or score_gap > 0.035:
            return False
    # Collapse tied same-document variants like "government" versus
    # "Russian government" or "Page" versus "Carter Page" so the generic
    # alias does not zero out winner margin against the more specific surface.
    return smaller < larger


def _dedupe_margin_competitors(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(bool(row.get("eligible", True))),
            -float(row.get("final_score", 0.0) or 0.0),
            -len(_candidate_identity_tokens(row.get("candidate", ""))),
            -len(clean_text(row.get("candidate", ""))),
            clean_text(row.get("doc_id", "")),
            clean_text(row.get("candidate", "")),
        ),
    )
    kept: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []
    for row in ranked:
        duplicate = any(
            _same_family_margin_duplicate(row, prior)
            for prior in family_rows
        )
        family_rows.append(dict(row))
        if duplicate:
            continue
        kept.append(dict(row))
    return kept


class ValueRedactionEngine(TaskEngine):
    task_name = "redaction"

    def run(self, context: EngineContext) -> EngineResult:
        slots = context.query.value_slots
        fragment_text = {
            clean_text(row.get("fragment_id", "")): clean_text(
                row.get("fragment_text_for_extraction", row.get("fragment_text", ""))
            )
            for _, row in context.query.fragments.iterrows()
        }
        audit_rows: List[Dict[str, Any]] = []
        audit_summary_rows: List[Dict[str, Any]] = []
        coverage_rows: List[Dict[str, Any]] = []
        evaluation_rows: List[Dict[str, Any]] = []
        context.control.emit("value_redaction_started", target_count=len(slots))
        for index, slot_row in slots.iterrows():
            slot = dict(slot_row)
            slot_id = clean_text(slot.get("slot_label", "")) or f"R{index + 1}"
            fragment_query_text = fragment_text.get(
                clean_text(slot.get("fragment_id", "")), ""
            )
            slot = _vr_authoritative_slot_context(slot, fragment_query_text)
            expected = clean_text(
                slot.get("expected_type_hint", slot.get("explicit_type_hint", ""))
            )
            effective_expected = expected
            if clean_text(expected).upper() in {"LOCATION", "COUNTRY"}:
                effective_expected = "LOCATION_OR_COUNTRY"
            slot_frame = clean_text(slot.get("slot_frame", ""))
            required_role = clean_text(slot.get("grammatical_role", ""))
            if (
                clean_text(effective_expected).upper()
                in {"LOCATION", "COUNTRY", "LOCATION_OR_COUNTRY"}
                and required_role == "grammatical_subject"
                and (
                    slot_frame == "generic_value_slot"
                    or re.search(
                        r"\b(?:to|from|in|at|on|near|by|of)\s*$",
                        clean_text(slot.get("local_context_before", "")),
                        flags=re.I,
                    )
                    or _vr_infer_location_movement_origin_trigger(slot)
                )
            ):
                # Generic location slots are usually prepositional objects or
                # list members. A parser-level subject label is not reliable
                # enough here to exclude an otherwise exact local fill.
                required_role = ""
            raw_governing_predicate = clean_text(
                slot.get("governing_predicate", "")
            )
            predicate = _vr_predicate_family_for_frame(
                raw_governing_predicate, slot_frame
            )
            required_relation = clean_text(slot.get("required_relation", ""))
            modified_head = clean_text(slot.get("modified_head", ""))
            inferred_subject_identity = False
            strict_title_identity = False
            if (
                not modified_head
                and slot_frame == "generic_value_slot"
            ):
                inferred_title = _vr_infer_person_title_anchor(slot)
                if inferred_title:
                    modified_head = inferred_title
                    strict_title_identity = True
                    if not required_role or required_role == "unknown":
                        required_role = "subject_identity"
                        inferred_subject_identity = True
            if not required_role or required_role == "unknown":
                inferred_subject_role = _vr_infer_generic_subject_role(slot)
                if inferred_subject_role:
                    required_role = inferred_subject_role
            fragment_predicate = _vr_fragment_predicate_family(fragment_query_text)
            if not predicate and slot_frame != "possessive_entity_slot":
                predicate = fragment_predicate
                if predicate and raw_governing_predicate:
                    # The parser occasionally attaches a nearby preposition
                    # (for example, "of") to a coordinated blank.  A reliable
                    # full-fragment predicate should drive the same gates as a
                    # parser-supplied governing predicate.
                    slot["governing_predicate"] = predicate
                    slot["governing_predicate_normalized"] = predicate
            slot_query_text = clean_text(" ".join((
                clean_text(slot.get("local_context_before", "")),
                clean_text(slot.get("local_context_after", "")),
            ))) or fragment_query_text
            try:
                parsed_signature = json.loads(clean_text(
                    slot.get("slot_event_signature_json", "")
                ))
            except json.JSONDecodeError:
                parsed_signature = {}
            rebuilt_signature = build_event_signature(
                slot_query_text,
                predicate=predicate,
            )
            query_signature = (
                parsed_signature
                if isinstance(parsed_signature, dict) and parsed_signature
                else rebuilt_signature
            )
            if isinstance(query_signature, dict):
                query_signature = dict(query_signature)
            if isinstance(rebuilt_signature, dict) and rebuilt_signature:
                if _signature_anchor_strength(rebuilt_signature) > _signature_anchor_strength(query_signature):
                    for field, value in rebuilt_signature.items():
                        if field not in query_signature or not _clean_slot_signature_value(
                            query_signature.get(field, "")
                        ):
                            query_signature[field] = value
            if isinstance(query_signature, dict):
                query_signature = dict(query_signature)
                query_signature["predicate"] = (
                    _vr_predicate_family_for_frame(
                        query_signature.get("predicate", ""), slot_frame
                    )
                    or predicate
                )
            try:
                parsed_required = json.loads(clean_text(
                    slot.get("required_anchor_fields_json", "")
                ))
            except json.JSONDecodeError:
                parsed_required = []
            required_anchor_fields = {
                (
                    "date" if clean_text(value) == "event_time"
                    else clean_text(value)
                )
                for value in parsed_required
                if clean_text(value)
            } or None
            required_anchor_fields = _sanitize_value_redaction_required_fields(
                slot,
                required_anchor_fields,
                query_signature,
                context,
            )
            if required_relation:
                # The modified location head and its administrative container
                # are deliberately different arguments.  Their relationship
                # is enforced by relation_gate_pass below.
                query_signature = dict(query_signature)
                query_signature["location"] = ""
                query_signature["location_identity"] = ""
            query_signature = _sanitize_value_redaction_query_signature(
                slot,
                query_signature,
                context,
            )
            matching_predicate = _vr_predicate_family_for_frame(
                query_signature.get("predicate", ""), slot_frame
            )
            ranked_docs = _value_redaction_rank_documents(
                slot_query_text,
                fragment_query_text,
                slot,
                query_signature,
                context,
            )
            candidates: List[Dict[str, Any]] = []
            query_anchor_tokens = _vr_event_anchor_tokens(
                slot,
                slot_query_text,
                context,
            )
            generic_person_title_anchor = (
                modified_head
                if (
                    slot_frame == "generic_value_slot"
                    and expected.upper() == "PERSON"
                    and modified_head
                )
                else ""
            )
            generic_person_recipient_trigger = (
                _vr_infer_person_recipient_trigger(slot)
                if (
                    slot_frame == "generic_value_slot"
                    and expected.upper() == "PERSON"
                )
                else ""
            )
            generic_attribution_trigger = _vr_infer_attribution_trigger(slot)
            generic_person_object_trigger = _vr_infer_person_object_trigger(slot)
            generic_copular_complement_trigger = (
                _vr_infer_copular_complement_trigger(slot)
            )
            generic_container_of_trigger = (
                _vr_infer_container_of_trigger(slot)
            )
            generic_membership_container_trigger = (
                _vr_infer_membership_container_trigger(slot)
            )
            (
                generic_location_prepositional_phrase,
                generic_location_preposition,
            ) = _vr_infer_location_prepositional_trigger(slot)
            if not generic_location_prepositional_phrase:
                (
                    generic_location_prepositional_phrase,
                    generic_location_preposition,
                ) = _vr_infer_leading_locative_trigger(slot)
            generic_person_infinitive_trigger = (
                _vr_infer_person_infinitive_trigger(slot)
            )
            generic_possessive_trigger = _vr_infer_possessive_trigger(slot)
            generic_authority_trigger = _vr_infer_authority_trigger(slot)
            generic_list_head_trigger = _vr_infer_list_head_trigger(slot)
            generic_location_movement_origin_trigger = (
                _vr_infer_location_movement_origin_trigger(slot)
            )
            generic_location_movement_destination_trigger = (
                _vr_infer_location_movement_destination_trigger(slot)
            )
            generic_enumeration_members = _vr_visible_enumeration_members(slot)
            generic_adjacent_scalar_anchor = _vr_infer_adjacent_scalar_anchor(slot)
            if _vr_slot_looks_enumerative(slot):
                generic_location_prepositional_phrase = ""
                generic_location_preposition = ""
            role_predicate = _role_predicate_surface(
                slot,
                matching_predicate or predicate,
            )
            preferred_clause_terms: List[str] = [
                slot_query_text,
                fragment_query_text,
                _vr_local_slot_probe_text(slot),
                modified_head,
                clean_text(query_signature.get("object", "")),
                clean_text(query_signature.get("patient_or_theme", "")),
                clean_text(query_signature.get("location_identity", "")),
                clean_text(query_signature.get("location", "")),
                clean_text(query_signature.get("event_time", "")),
                clean_text(query_signature.get("predicate", "")),
            ]
            clause_cache: Dict[tuple[str, str, str, str, str], str] = {}
            signature_cache: Dict[
                str, tuple[Dict[str, Any], bool]
            ] = {}
            for doc_index, (doc_id, retrieval_score) in enumerate(ranked_docs, start=1):
                if doc_index == 1 or doc_index % 25 == 0:
                    context.control.emit(
                        "value_redaction_slot_progress",
                        slot_id=slot_id,
                        docs_processed=doc_index,
                        docs_total=len(ranked_docs),
                        candidates_so_far=len(candidates),
                    )
                document_text = context.corpus.document_text.get(doc_id, "")
                relation_text = " ".join(
                    record.text
                    for record in context.corpus.propositions_by_doc.get(doc_id, ())
                    if record.source_table == "relations.csv"
                )
                doc_value_records = _compact_value_records_for_doc(
                    context.corpus.values_by_doc.get(doc_id, ()),
                    slot=slot,
                    slot_query_text=slot_query_text,
                    fragment_query_text=fragment_query_text,
                    query_signature=query_signature,
                    expected=effective_expected,
                    context=context,
                )
                for value_record in doc_value_records:
                    value = value_record.value
                    if _vr_rejected_candidate_surface(value):
                        continue
                    type_pass = type_compatible(
                        effective_expected, value_record.value_type, value
                    )
                    if (
                        type_pass
                        and generic_person_title_anchor
                        and clean_text(value_record.value_type).upper() != "PERSON"
                    ):
                        type_pass = False
                    if (
                        type_pass
                        and expected.upper() == "PERSON"
                        and clean_text(value_record.value_type).upper() == "UNKNOWN"
                        and not _vr_person_like_unknown_surface(value)
                    ):
                        type_pass = False
                    # Type is a cheap, independent prefilter.  Expensive local
                    # clause construction is never performed for incompatible
                    # values.
                    if not type_pass:
                        continue
                    clause_key = (
                        doc_id,
                        canonical_value(value),
                        matching_predicate or predicate,
                        clean_text(value_record.source_table),
                        clean_text(value_record.row_id),
                    )
                    evidence = clause_cache.get(clause_key, "")
                    if not evidence:
                        evidence = _candidate_local_clause(
                            document_text,
                            value_record.evidence,
                            value,
                            matching_predicate,
                            preferred_clause_terms,
                        )
                        clause_cache[clause_key] = evidence
                    # Candidate-local identity prevents an entity mentioned in
                    # one part of a document from inheriting an unrelated
                    # event match elsewhere in that document.
                    event_score = containment(
                        query_anchor_tokens,
                        _normalized_expanded_tokens(evidence, context),
                    )
                    local_argument_frame_pass = _vr_local_argument_frame_pass(
                        slot,
                        query_signature,
                        value,
                        evidence,
                        context,
                    )
                    cached_signature = signature_cache.get(evidence)
                    if cached_signature is None:
                        cached_signature = _candidate_event_signature(
                            evidence, matching_predicate
                        )
                        signature_cache[evidence] = cached_signature
                    (
                        candidate_signature,
                        predicate_surface_pass,
                    ) = cached_signature
                    gates = _critical_event_gates(
                        query_signature, candidate_signature, slot_query_text,
                        evidence,
                        context.semantic_resources,
                        required_anchor_fields,
                    )
                    self_fill = bool(
                        modified_head
                        and canonical_location(modified_head)
                        == canonical_location(value)
                    )
                    local_context_bonus = _vr_candidate_local_context_bonus(
                        slot,
                        value,
                        evidence,
                    )
                    multi_slot_sequence_pass = _vr_multi_slot_sequence_pass(
                        slot,
                        value,
                        evidence,
                    )
                    enumeration_context_pass = _vr_local_enumeration_context_pass(
                        generic_enumeration_members,
                        value,
                        evidence,
                        document_text,
                    )
                    visible_query_penalty = _vr_candidate_query_visibility_penalty(
                        slot,
                        value,
                        modified_head,
                        context,
                    )
                    adjacent_scalar_pass = bool(
                        generic_adjacent_scalar_anchor
                        and _vr_local_adjacent_scalar_pass(
                            value,
                            generic_adjacent_scalar_anchor,
                            evidence,
                        )
                    )
                    role_pass = True
                    actual_role = "not_applicable"
                    if type_pass and generic_person_title_anchor:
                        title_pass = _title_identity_pass(
                            value,
                            generic_person_title_anchor,
                            document_text,
                            evidence,
                            context.config.value_redaction_title_containment_floor,
                            strict_local=strict_title_identity,
                        )
                        role_pass = title_pass
                        actual_role = (
                            "subject_identity"
                            if role_pass else "identity_role_not_observed"
                        )
                    elif type_pass and generic_list_head_trigger:
                        role_pass = _vr_local_list_head_pass(
                            value,
                            evidence,
                            document_text,
                        )
                        actual_role = (
                            "list_head_identity"
                            if role_pass else "list_head_identity_not_observed"
                        )
                    elif type_pass and generic_attribution_trigger:
                        role_pass = _vr_local_attribution_pass(value, evidence)
                        actual_role = (
                            "attribution_source_identity"
                            if role_pass else "attribution_source_not_observed"
                        )
                    elif type_pass and generic_person_recipient_trigger:
                        role_pass = _vr_local_recipient_pass(
                            generic_person_recipient_trigger,
                            value,
                            evidence,
                        )
                        actual_role = (
                            "recipient_identity"
                            if role_pass else "recipient_role_not_observed"
                        )
                    elif type_pass and generic_person_object_trigger:
                        role_pass = _vr_local_person_object_pass(
                            generic_person_object_trigger,
                            value,
                            evidence,
                        )
                        actual_role = (
                            "patient_or_object_identity"
                            if role_pass else "patient_or_object_not_observed"
                        )
                    elif type_pass and generic_person_infinitive_trigger:
                        role_pass = _vr_local_person_infinitive_subject_pass(
                            value,
                            evidence,
                        )
                        actual_role = (
                            "infinitive_subject_identity"
                            if role_pass else "infinitive_subject_not_observed"
                        )
                    elif type_pass and generic_possessive_trigger:
                        role_pass = _vr_local_possessive_pass(
                            value,
                            generic_possessive_trigger,
                            evidence,
                            context,
                        )
                        actual_role = (
                            "possessive_identity"
                            if role_pass else "possessive_identity_not_observed"
                        )
                    elif type_pass and generic_authority_trigger:
                        role_pass = _vr_local_authority_pass(
                            generic_authority_trigger,
                            value,
                            evidence,
                        )
                        actual_role = (
                            "governing_authority_identity"
                            if role_pass else "governing_authority_not_observed"
                        )
                    elif (
                        type_pass
                        and generic_location_movement_origin_trigger
                    ):
                        role_pass = _vr_local_location_movement_origin_pass(
                            generic_location_movement_origin_trigger,
                            value,
                            evidence,
                        )
                        actual_role = (
                            "movement_origin_identity"
                            if role_pass else "movement_origin_not_observed"
                        )
                    elif (
                        type_pass
                        and generic_location_movement_destination_trigger
                    ):
                        role_pass = _vr_local_location_movement_destination_pass(
                            generic_location_movement_destination_trigger,
                            value,
                            evidence,
                        )
                        actual_role = (
                            "movement_destination_identity"
                            if role_pass else "movement_destination_not_observed"
                        )
                    elif type_pass and generic_container_of_trigger:
                        role_pass = _vr_local_prepositional_complement_pass(
                            generic_container_of_trigger,
                            value,
                            evidence,
                        )
                        actual_role = (
                            "prepositional_complement_identity"
                            if role_pass else "prepositional_complement_not_observed"
                        )
                    elif type_pass and generic_membership_container_trigger:
                        role_pass = _vr_local_membership_container_pass(
                            generic_membership_container_trigger,
                            value,
                            evidence,
                            context,
                        )
                        actual_role = (
                            "membership_identity"
                            if role_pass else "membership_not_observed"
                        )
                    elif type_pass and generic_adjacent_scalar_anchor:
                        role_pass = adjacent_scalar_pass
                        actual_role = (
                            "adjacent_scalar_identity"
                            if role_pass else "adjacent_scalar_not_observed"
                        )
                    elif type_pass and generic_location_prepositional_phrase:
                        role_pass = (
                            _vr_local_location_prepositional_pass(
                                generic_location_prepositional_phrase,
                                generic_location_preposition,
                                value,
                                evidence,
                                context,
                            )
                            or _vr_local_leading_locative_phrase_pass(
                                generic_location_prepositional_phrase,
                                generic_location_preposition,
                                value,
                                evidence,
                                context,
                            )
                        )
                        actual_role = (
                            "location_prepositional_identity"
                            if role_pass else "location_prepositional_not_observed"
                        )
                    elif type_pass and generic_copular_complement_trigger:
                        role_pass = _vr_local_copular_complement_pass(
                            value,
                            evidence,
                        )
                        actual_role = (
                            "copular_complement_identity"
                            if role_pass else "copular_complement_not_observed"
                        )
                    elif type_pass and required_role == "grammatical_subject" and role_predicate:
                        if (
                            clean_text(slot.get("slot_frame", "")) == "generic_value_slot"
                            and re.search(r"\bthat\s*$", clean_text(slot.get("local_context_before", "")), flags=re.I)
                            and re.match(r"^\s*(?:is|are|was|were)\b", clean_text(slot.get("local_context_after", "")), flags=re.I)
                        ):
                            role_pass = _embedded_subject_before_copular(
                                value,
                                evidence,
                            )
                            actual_role = (
                                "embedded_clause_subject"
                                if role_pass else "embedded_clause_subject_not_observed"
                            )
                        else:
                            role_pass = _subject_before_predicate(
                                value,
                                role_predicate,
                                evidence,
                            )
                            actual_role = (
                                "grammatical_subject"
                                if role_pass else "role_not_observed"
                            )
                    elif type_pass and (
                        required_role == "subject_identity"
                        or strict_title_identity
                    ):
                        title_pass = _title_identity_pass(
                            value,
                            modified_head,
                            document_text,
                            evidence,
                            context.config.value_redaction_title_containment_floor,
                        )
                        if inferred_subject_identity or strict_title_identity:
                            role_pass = title_pass
                        else:
                            role_pass = title_pass or (
                                bool(role_predicate)
                                and _subject_before_predicate(
                                    value,
                                    role_predicate,
                                    evidence,
                                )
                            )
                        actual_role = (
                            "subject_identity"
                            if role_pass else "identity_role_not_observed"
                        )
                    elif required_role:
                        actual_role = required_role if type_pass else "type_incompatible"
                    relation_pass = True
                    if (
                        type_pass
                        and required_relation == "identity_appositive_to_title"
                    ):
                        relation_pass = _title_identity_pass(
                            value,
                            modified_head,
                            document_text,
                            evidence,
                            context.config.value_redaction_title_containment_floor,
                        ) or (
                            bool(matching_predicate or predicate)
                            and _subject_before_predicate(
                                value, matching_predicate or predicate, evidence
                            )
                        )
                    elif type_pass and required_relation:
                        normalized_relation = canonical_location(relation_text)
                        relation_pass = bool(
                            canonical_location(modified_head)
                            and canonical_location(modified_head) in normalized_relation
                            and canonical_location(value) in normalized_relation
                            and any(
                                token in canonical_key(relation_text)
                                for token in (
                                    "contain", "province", "administrative",
                                    "located in",
                                )
                            )
                        )
                        if not relation_pass:
                            relation_pass = bool(re.search(
                                re.escape(modified_head)
                                + r"\s*\(\s*"
                                + re.escape(value)
                                + r"\s*\)",
                                evidence,
                                flags=re.I,
                            ))
                    event_pass = (
                        max(
                            event_score,
                            float(gates.get("event_identity_score", 0.0) or 0.0),
                        ) >= context.config.value_redaction_event_floor
                        and predicate_surface_pass
                        and local_argument_frame_pass
                        and bool(gates["critical_event_gate_pass"])
                    )
                    score = (
                        0.55 * event_score
                        + 0.15 * retrieval_score
                        + 0.15 * float(role_pass)
                        + 0.15 * float(relation_pass)
                    )
                    score = max(
                        0.0,
                        score
                        + local_context_bonus
                        + (0.12 if multi_slot_sequence_pass is True else 0.0)
                        + (0.10 if generic_enumeration_members and enumeration_context_pass else 0.0)
                        + (0.14 if adjacent_scalar_pass else 0.0)
                        - visible_query_penalty,
                    )
                    if (
                        clean_text(value_record.source_table)
                        == "kg_pseudo_chunks_redaction.csv"
                        and generic_person_object_trigger
                        and role_pass
                    ):
                        # An explicit local patient pattern (for example,
                        # "murder of <person>") is more reliable than a noisy
                        # event signature extracted from a reporting clause.
                        # Keep retrieval dominant, but retain event agreement
                        # as an independent quarter of the score.
                        score = max(
                            score,
                            0.75 * retrieval_score + 0.25 * event_score,
                        )
                    title_context_certification_pass = bool(
                        clean_text(value_record.source_table)
                        == "kg_pseudo_chunks_redaction.csv"
                        and strict_title_identity
                        and role_pass
                        and type_pass
                        and relation_pass
                        and not self_fill
                        and retrieval_score >= 0.30
                        and event_score >= 0.30
                        and local_context_bonus >= 0.045
                        and score >= 0.65
                        and all(
                            not gates.get(field + "_gate_required")
                            or gates.get(field + "_gate_pass")
                            for field in (
                                "date", "packet_count",
                                "total_weight_grams",
                            )
                        )
                    )
                    direct_context_certification_pass = bool(
                        title_context_certification_pass
                        or (
                            clean_text(value_record.source_table)
                            == "kg_pseudo_chunks_redaction.csv"
                            and type_pass
                            and relation_pass
                            and not self_fill
                            and enumeration_context_pass
                            and _vr_direct_context_is_locally_grounded(
                                event_score=event_score,
                                event_floor=(
                                    context.config.value_redaction_event_floor
                                ),
                                local_context_bonus=local_context_bonus,
                                retrieval_score=retrieval_score,
                                local_argument_frame_pass=(
                                    local_argument_frame_pass
                                ),
                            )
                            # Movement-destination prompts often retrieve a
                            # document containing several unrelated journeys.
                            # Require two-sided local context before a Stage C
                            # row may rescue noisy event parsing in this frame.
                            and (
                                not generic_location_movement_destination_trigger
                                or local_context_bonus >= 0.09
                            )
                            # Direct-context rescue repairs noisy event
                            # extraction. It must not overrule an explicit,
                            # locally testable grammatical-role mismatch.
                            and not (
                                not role_pass
                                and any((
                                    generic_person_title_anchor,
                                    generic_attribution_trigger,
                                    generic_person_recipient_trigger,
                                    generic_person_object_trigger,
                                    generic_person_infinitive_trigger,
                                    generic_possessive_trigger,
                                    generic_authority_trigger,
                                    generic_location_movement_origin_trigger,
                                    generic_location_movement_destination_trigger,
                                    generic_container_of_trigger,
                                    generic_membership_container_trigger,
                                    generic_adjacent_scalar_anchor,
                                    generic_location_prepositional_phrase,
                                    generic_copular_complement_trigger,
                                ))
                            )
                            and not _vr_candidate_visible_in_query_context(
                                slot,
                                value,
                                modified_head,
                                context,
                            )
                            # A provenance-bearing companion row may repair
                            # noisy predicate/role parsing, but never an
                            # explicit date or quantity mismatch.
                            and all(
                                not gates.get(field + "_gate_required")
                                or gates.get(field + "_gate_pass")
                                for field in (
                                    "date", "packet_count",
                                    "total_weight_grams",
                                )
                            )
                            and retrieval_score >= _VR_DIRECT_CONTEXT_RETRIEVAL_FLOOR
                            and (
                                (
                                    local_context_bonus
                                    >= _VR_DIRECT_CONTEXT_BONUS_FLOOR
                                    and score >= _VR_DIRECT_CONTEXT_SCORE_FLOOR
                                )
                                or (
                                    local_context_bonus
                                    >= _VR_DIRECT_CONTEXT_STRONG_FRAME_BONUS_FLOOR
                                    and predicate_surface_pass
                                    and role_pass
                                    and score
                                    >= _VR_DIRECT_CONTEXT_STRONG_FRAME_SCORE_FLOOR
                                )
                            )
                        )
                        or (
                            clean_text(value_record.source_table)
                            == "kg_pseudo_chunks_redaction.csv"
                            and bool(generic_person_object_trigger)
                            and role_pass
                            and type_pass
                            and relation_pass
                            and not self_fill
                            and _vr_direct_context_is_locally_grounded(
                                event_score=event_score,
                                event_floor=(
                                    context.config.value_redaction_event_floor
                                ),
                                local_context_bonus=local_context_bonus,
                                retrieval_score=retrieval_score,
                                local_argument_frame_pass=(
                                    local_argument_frame_pass
                                ),
                            )
                            and retrieval_score
                            >= _VR_DIRECT_CONTEXT_RETRIEVAL_FLOOR
                            and all(
                                not gates.get(field + "_gate_required")
                                or gates.get(field + "_gate_pass")
                                for field in (
                                    "date", "packet_count",
                                    "total_weight_grams",
                                )
                            )
                        )
                        or (
                            # A possessive PERSON/organization slot backed by
                            # a Stage C row is an explicit local grammatical
                            # fact. Permit this narrow route to repair a noisy
                            # object signature when the source-local frame and
                            # surface predicate are both independently present.
                            clean_text(value_record.source_table)
                            == "kg_pseudo_chunks_redaction.csv"
                            and bool(generic_possessive_trigger)
                            and role_pass
                            and type_pass
                            and relation_pass
                            and not self_fill
                            and local_argument_frame_pass
                            and predicate_surface_pass
                            and local_context_bonus >= 0.06
                            and score >= 0.70
                            and all(
                                not gates.get(field + "_gate_required")
                                or gates.get(field + "_gate_pass")
                                for field in (
                                    "date", "packet_count",
                                    "total_weight_grams",
                                )
                            )
                        )
                    )
                    # Local argument-frame certification is stronger than a
                    # broad pseudo-chunk topical match.  Its bounded advantage
                    # is smaller than any primary score component, but large
                    # enough to prevent an unrelated near-tie from suppressing
                    # a source-local, fully gated fill.
                    grounding_certification_bonus = (
                        0.04 if direct_context_certification_pass else 0.0
                    )
                    score += grounding_certification_bonus
                    failures = []
                    if not type_pass:
                        failures.append("TYPE")
                    if not role_pass and not direct_context_certification_pass:
                        failures.append("ROLE")
                    if not relation_pass:
                        failures.append("REQUIRED_RELATION")
                    if self_fill:
                        failures.append("SELF_FILL")
                    if not enumeration_context_pass:
                        failures.append("ENUMERATION_CONTEXT")
                    if not event_pass and not direct_context_certification_pass:
                        failures.append("EVENT_IDENTITY")
                    if (
                        not local_argument_frame_pass
                        and not direct_context_certification_pass
                    ):
                        failures.append("LOCAL_ARGUMENT_FRAME")
                    if (
                        not predicate_surface_pass
                        and not direct_context_certification_pass
                    ):
                        failures.append("GOVERNING_PREDICATE")
                    for gate_name in (
                        "predicate", "object", "location", "date",
                        "packet_count", "total_weight_grams",
                    ):
                        if (
                            gates.get(gate_name + "_gate_required")
                            and not gates.get(gate_name + "_gate_pass")
                            and not direct_context_certification_pass
                        ):
                            failures.append(gate_name.upper() + "_ANCHOR")
                    eligible = not failures
                    candidates.append({
                        "slot_id": slot_id,
                        "canonical_candidate_id": "value:" + canonical_value(value),
                        "candidate": value,
                        "candidate_type": value_record.value_type,
                        "doc_id": doc_id,
                        "source_table": value_record.source_table,
                        "source_row_id": value_record.row_id,
                        "expected_role": required_role,
                        "actual_role": actual_role,
                        "required_relation": required_relation,
                        "retrieval_score": retrieval_score,
                        "event_identity_score": event_score,
                        "local_argument_frame_pass": local_argument_frame_pass,
                        "governing_predicate": matching_predicate or predicate,
                        "predicate_surface_gate_pass": predicate_surface_pass,
                        **gates,
                        "type_gate_pass": type_pass,
                        "role_gate_pass": role_pass,
                        "relation_gate_pass": relation_pass,
                        "self_fill_rejected": self_fill,
                        "event_identity_gate_pass": event_pass,
                        "direct_context_certification_pass": (
                            direct_context_certification_pass
                        ),
                        "grounding_certification_bonus": (
                            grounding_certification_bonus
                        ),
                        "title_context_certification_pass": (
                            title_context_certification_pass
                        ),
                        "eligible": eligible,
                        "final_score": score,
                        "local_context_bonus": local_context_bonus,
                        "multi_slot_sequence_pass": multi_slot_sequence_pass,
                        "adjacent_scalar_pass": adjacent_scalar_pass,
                        "enumeration_context_pass": enumeration_context_pass,
                        "visible_query_penalty": visible_query_penalty,
                        "failed_gates": ";".join(failures),
                        "evidence": clean_text(evidence)[:1600],
                    })
            context.control.check(len(candidates))
            canonical: Dict[str, Dict[str, Any]] = {}
            representation_counts: Dict[str, int] = {}
            supporting_docs: Dict[str, set[str]] = {}
            for candidate in candidates:
                identity = canonical_value(candidate["candidate"])
                representation_counts[identity] = (
                    representation_counts.get(identity, 0) + 1
                )
                supporting_docs.setdefault(identity, set()).add(
                    clean_text(candidate.get("doc_id", ""))
                )
                previous = canonical.get(identity)
                if previous is None or (
                    bool(candidate["eligible"]),
                    bool(candidate["direct_context_certification_pass"]),
                    float(candidate["event_identity_score"]),
                    float(candidate["local_context_bonus"]),
                    bool(candidate["local_argument_frame_pass"]),
                    float(candidate["final_score"]),
                ) > (
                    bool(previous["eligible"]),
                    bool(previous["direct_context_certification_pass"]),
                    float(previous["event_identity_score"]),
                    float(previous["local_context_bonus"]),
                    bool(previous["local_argument_frame_pass"]),
                    float(previous["final_score"]),
                ):
                    canonical[identity] = candidate
            canonical_audit = []
            for identity, candidate in canonical.items():
                summary = dict(candidate)
                summary["representation_count"] = representation_counts[identity]
                summary["supporting_document_count"] = len(
                    supporting_docs.get(identity, set())
                )
                canonical_audit.append(summary)
            audit_rows.extend(canonical_audit)
            audit_summary_rows.append({
                "slot_id": slot_id,
                "shortlisted_document_count": len(ranked_docs),
                "representation_count": len(candidates),
                "canonical_candidate_count": len(canonical_audit),
                "eligible_canonical_candidate_count": sum(
                    bool(row.get("eligible", False))
                    for row in canonical_audit
                ),
                "audit_contract": (
                    "ONE_BEST_ROW_PER_CANONICAL_VALUE_WITH_REPRESENTATION_COUNTS"
                ),
            })
            decision_candidates = _dedupe_margin_competitors(
                list(canonical.values())
            )
            winner, runner, margin, resolved = winner_with_margin(
                decision_candidates,
                context.config.minimum_score,
                context.config.minimum_margin,
            )
            nearest = nearest_alternative(
                decision_candidates,
                winner,
                identity_fields=("canonical_candidate_id",),
            )
            status = (
                "RESOLVED" if resolved
                else "UNRESOLVED_INSUFFICIENT_MARGIN" if winner
                else "UNRESOLVED_NO_ELIGIBLE_SAME_EVENT_CANDIDATE"
            )
            evaluation_rows.append(task_row(
                "redaction", slot_id, status,
                localization_status=(
                    "AUTHORITATIVE_SOURCE_SELECTED"
                    if resolved
                    else "RELATED_EVIDENCE_FOUND"
                    if winner
                    else "NO_SOURCE_FOUND"
                ),
                selected_text=clean_text(winner.get("candidate", "")) if resolved else "",
                selected_source=clean_text(winner.get("doc_id", "")) if resolved else "",
                top_candidate=clean_text(winner.get("doc_id", "")),
                related_source=(
                    clean_text(winner.get("doc_id", "")) if winner and not resolved else ""
                ),
                best_score=winner.get("final_score", ""),
                runner_up=runner.get("final_score", ""),
                margin=margin,
                decision_competitor_doc_id=clean_text(
                    runner.get("doc_id", "")
                ),
                decision_competitor_passage_id=clean_text(
                    runner.get("source_row_id", "")
                ),
                nearest_alternative_score=nearest.get("final_score"),
                nearest_alternative_doc_id=clean_text(
                    nearest.get("doc_id", "")
                ),
                nearest_alternative_passage_id=clean_text(
                    nearest.get("source_row_id", "")
                ),
                nearest_alternative_eligible=(
                    bool(nearest.get("eligible"))
                    if nearest else None
                ),
                nearest_alternative_failed_gates=clean_text(
                    nearest.get("failed_gates", "")
                ),
                reason="bounded_noncompensatory_type_role_relation_event_gates",
                unresolved="" if resolved else status,
                failed="" if resolved else clean_text(
                    winner.get("failed_gates", "NO_ELIGIBLE_CANDIDATE")
                ),
            ))
            coverage_rows.append({
                "slot_id": slot_id,
                "fragment_id": clean_text(slot.get("fragment_id", "")),
                "expected_type": expected,
                "slot_frame": clean_text(slot.get("slot_frame", "")),
                "governing_predicate": predicate,
                "evaluation_row_present": True,
                "candidate_count": len(candidates),
                "canonical_candidate_count": len(canonical),
                "eligible_candidate_count": sum(
                    bool(row.get("eligible", False))
                    for row in canonical.values()
                ),
                "selected_candidate": (
                    clean_text(winner.get("candidate", "")) if resolved else ""
                ),
                "selected_source_doc_id": (
                    clean_text(winner.get("doc_id", "")) if resolved else ""
                ),
                "semantic_status": status,
            })
            context.control.emit(
                "value_redaction_target_complete",
                target_id=slot_id,
                shortlisted_document_count=len(ranked_docs),
                representation_count=len(candidates),
                canonical_candidate_count=len(canonical),
                status=status,
            )
        return EngineResult(
            task=self.task_name,
            evaluation=pd.DataFrame(evaluation_rows),
            artifacts={
                "redaction_candidate_audit_v048.csv":
                    pd.DataFrame(audit_rows),
                "redaction_audit_summary_v048.csv":
                    pd.DataFrame(audit_summary_rows),
                "redaction_slot_coverage_v048.csv":
                    pd.DataFrame(coverage_rows),
            },
            metadata={
                "audit_summary_rows": len(audit_summary_rows),
                "declared_slot_count": len(slots),
                "evaluation_slot_count": len(evaluation_rows),
                "all_declared_slots_reported": len(slots) == len(evaluation_rows),
            },
        )
