#!/usr/bin/env python3
"""Task-specific, corpus-agnostic benchmark case selection.

Generators produce broad catalogs. This module applies only structural quality
rules and ranking. It deliberately avoids document IDs, entity names, and
remembered benchmark sentences so the policy transfers to unseen corpora.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from task_build_rules import TaskBuildBehavior


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
REDACTION_SLOT_RE = re.compile(r"\[REDACT:R\d+[^\]]*\]", flags=re.IGNORECASE)
SENTENCE_SLOT_RE = re.compile(
    r"\[REDACT_SENT:S1\s+approx_tokens=(\d+)\]", flags=re.IGNORECASE
)
PRONOUN_LEAD_RE = re.compile(r"^(?:he|she|they|it|there)\b", flags=re.IGNORECASE)
FIRST_PERSON_RE = re.compile(
    r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves)\b", flags=re.IGNORECASE
)
VAGUE_REFERENCE_RE = re.compile(
    r"\b(?:somebody|someone|something|anything|the former|the latter)\b",
    flags=re.IGNORECASE,
)
ABSTRACT_LEAD_RE = re.compile(
    r"^(?:there\s+(?:is|are)|(?:an?|the|his|her|their|its)\s+"
    r"(?:(?:main|primary|central|underlying)\s+)?"
    r"(?:principle|need|goal|theme|strategy|effort|concern|priority|focus))\b",
    flags=re.IGNORECASE,
)
CLAIM_PREDICATE_RE = re.compile(
    r"\b(?:"
    r"is|are|was|were|has|have|had|will|would|can|could|may|might|must|should|"
    r"want(?:s|ed)?|intend(?:s|ed)?|plan(?:s|ned)?|seek(?:s|ing|sought)?|"
    r"support(?:s|ed|ing)?|oppose(?:s|d|ing)?|deny|denies|denied|"
    r"believe(?:s|d)?|expect(?:s|ed)?|report(?:s|ed)?|say|says|said|"
    r"build(?:s|ing|built)?|develop(?:s|ed|ing)?|meet(?:s|ing)?|"
    r"join(?:s|ed|ing)?|leave(?:s|left|leaving)?|provide(?:s|d|ing)?|"
    r"receive(?:s|d|ing)?|give(?:s|n|giving)?|make(?:s|made|making)?"
    r")\b",
    flags=re.IGNORECASE,
)
CONTRADICTION_CUE_RE = re.compile(
    r"\b(?:"
    r"not|never|no|none|without|unlikely|impossible|false|deny|denies|denied|"
    r"refuse(?:s|d)?|oppose(?:s|d)?|reject(?:s|ed)?|failed?|ceased?|"
    r"instead|rather than|only|before|after|earlier|later|"
    r"wants?|wanted|supports?|supported|prepared|ready|willing|determined|"
    r"expected|concerned|doubt(?:s|ed|ful)?"
    r")\b",
    flags=re.IGNORECASE,
)
CONTEXT_DEPENDENT_LEAD_RE = re.compile(
    r"^(?:however|but|finally|meanwhile|in related developments?|"
    r"it is unclear whether|there (?:are|were) reports? that|"
    r"according to (?:more recent )?reporting)\b",
    flags=re.IGNORECASE,
)
BOILERPLATE_RE = re.compile(
    r"\b(?:above-referenced|unless stated otherwise|duration of the authorities|"
    r"facts and circumstances in support of this application|"
    r"additional information of the same type|according to information on its website)\b",
    flags=re.IGNORECASE,
)
CLAIM_RELATION_ONLY_KINDS = {
    "claim_relation_query",
    "claim_pair_relation_surface",
    "claim_relation_evidence_surface",
}
SUBJECTLESS_PREDICATE_LEAD_RE = re.compile(
    r"^(?:rejected|approved|accepted|denied|reported|stated|said|urged|"
    r"requested|ordered|announced|believed|expected|claimed)\b",
    flags=re.IGNORECASE,
)
UNRESOLVED_TRUTH_PLACEHOLDER_RE = re.compile(
    r"_{3,}|\[(?:REDACT|SLOT)\b|<\s*(?:REDACTED|MISSING)\s*>|"
    r"\b(?:candidate|political party|source|sub-source)\s*#\d+\b",
    flags=re.IGNORECASE,
)
MALFORMED_CONNECTIVE_RE = re.compile(
    r"\b(?:by|from|with|to|of)\s+that\s+(?:it|he|she|they|we)\b",
    flags=re.IGNORECASE,
)
VALUE_FRAGMENT_LEAD_RE = re.compile(
    r"^(?:"
    r"and|or|but|although|though|while|because|as|by|for|from|in|into|"
    r"of|on|to|with|between|after|before|during|that|which|who|whom|whose|"
    r"he|she|they|it|we|i|"
    r"appears?|seems?|den(?:y|ies|ied|ying)|reports?|reported|states?|stated|"
    r"says?|said|believes?|believed|meets?|met|travels?|travelled|traveled"
    r")\b",
    flags=re.IGNORECASE,
)
MALFORMED_VALUE_SLOT_GRAMMAR_RE = re.compile(
    r"(?:"
    r"\bincluding\s+\[REDACT:R\d+[^\]]*\]\s+and\s+"
    r"(?:is|are|was|were|has|have|had)\b|"
    r"\[REDACT:R\d+[^\]]*\],\s+"
    r"(?:must|will|would|can|could|is|are|was|were|has|have|had)\b|"
    r",\s*[A-Z]\.?$"
    r")",
    flags=re.IGNORECASE,
)


def _words(value: Any) -> list[str]:
    return WORD_RE.findall(str(value or ""))


def _word_count(case: dict[str, Any]) -> int:
    return int(case.get("word_count", 0) or len(_words(case.get("query", ""))))


def _int(case: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(case.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _float(case: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(case.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _bool(case: dict[str, Any], key: str) -> bool:
    value = case.get(key, False)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _has_immediate_phrase_repetition(value: Any) -> bool:
    tokens = [token.casefold() for token in _words(value)]
    for width in range(2, min(6, len(tokens) // 2 + 1)):
        for index in range(0, len(tokens) - 2 * width + 1):
            if tokens[index:index + width] == tokens[
                index + width:index + 2 * width
            ]:
                return True
    return False


def _case_surface_is_usable(case: dict[str, Any]) -> bool:
    query = " ".join(str(case.get("query", "") or "").split())
    truth = " ".join(str(case.get("ground_truth", "") or "").split())
    return bool(
        query
        and truth
        and not UNRESOLVED_TRUTH_PLACEHOLDER_RE.search(truth)
        and not MALFORMED_CONNECTIVE_RE.search(query)
        and not _has_immediate_phrase_repetition(query)
    )


def _note_fields(case: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in str(case.get("notes", "") or "").split(";"):
        key, separator, value = part.partition("=")
        if separator:
            fields[key.strip().casefold()] = value.strip()
    return fields


def _normalized_corpus_code(value: Any) -> str:
    code = str(value or "").strip().casefold()
    return {"vietnam": "viet", "canberra": "canb"}.get(code, code)


def value_redaction_corpus_code(
    case: dict[str, Any], corpus_code: str | None = None
) -> str:
    """Return only an explicit corpus label; never infer it from cable names."""
    return _normalized_corpus_code(corpus_code or case.get("corpus", ""))


def claim_corpus_code(case: dict[str, Any], corpus_code: str | None = None) -> str:
    return _normalized_corpus_code(corpus_code or case.get("corpus", ""))


def claim_corpus_priority_adjustment(
    case: dict[str, Any], *, corpus_code: str | None = None
) -> float:
    del case, corpus_code
    return 0.0


def contradiction_corpus_priority_adjustment(
    case: dict[str, Any], *, corpus_code: str | None = None
) -> float:
    del case, corpus_code
    return 0.0


def value_redaction_corpus_priority_adjustment(
    case: dict[str, Any], *, corpus_code: str | None = None
) -> float:
    del case, corpus_code
    return 0.0


def _canonical_query_key(text: Any) -> str:
    normalized = " ".join(str(text or "").split()).casefold()
    token_key = " ".join(re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)?", normalized))
    return token_key or normalized


def leading_slot_surface(case: dict[str, Any]) -> bool:
    query = " ".join(str(case.get("query", "") or "").split())
    return bool(re.match(r"^(?:the\s+)?\[REDACT:R1", query, flags=re.IGNORECASE))


def value_redaction_residual_content_count(case: dict[str, Any]) -> int:
    stripped = REDACTION_SLOT_RE.sub(" ", str(case.get("query", "") or ""))
    stopwords = {
        "about", "after", "against", "also", "among", "because", "being",
        "between", "could", "during", "from", "government", "have", "having",
        "into", "likely", "might", "other", "perhaps", "report", "reports",
        "said", "some", "such", "that", "their", "them", "they", "this",
        "those", "through", "under", "which", "while", "with", "would",
    }
    return len(
        {
            token.casefold()
            for token in _words(stripped)
            if len(token) >= 4 and token.casefold() not in stopwords
        }
    )


def value_redaction_has_strong_residual_context(
    case: dict[str, Any], *, small_corpus: bool
) -> bool:
    del small_corpus
    if _bool(case, "clean_direct_slot"):
        return True
    residual = value_redaction_residual_content_count(case)
    return residual >= 5 or (
        residual >= 3
        and (
            _bool(case, "has_date_or_number")
            or _int(case, "visible_anchor_count") >= 1
            or _int(case, "local_support_hits") >= 1
            or _int(case, "role_words") >= 3
        )
    )


def _answer_leaks_into_query(case: dict[str, Any]) -> bool:
    query_text = REDACTION_SLOT_RE.sub(" ", str(case.get("query", "") or ""))
    query_key = f" {_canonical_query_key(query_text)} "
    answer = str(case.get("ground_truth", "") or "").strip()
    if not answer:
        return False
    for value in re.split(r"\s*(?:\|\||;;)\s*", answer):
        answer_key = _canonical_query_key(value)
        if len(answer_key) >= 4 and f" {answer_key} " in query_key:
            return True
    return False


def _value_answer_is_clause_fragment(case: dict[str, Any]) -> bool:
    """Reject extraction spans that are clauses mislabeled as entities."""
    answer = str(case.get("ground_truth", "") or "").strip()
    for part in re.split(r"\s*\|\s*", answer):
        _slot, separator, value_with_doc = part.partition("=")
        value = value_with_doc.rsplit("@", 1)[0].strip() if separator else ""
        if value and VALUE_FRAGMENT_LEAD_RE.search(value):
            return True
    return False


def _single_value_answer(case: dict[str, Any]) -> str:
    """Return the answer surface for a one-slot value-redaction case."""
    answer = str(case.get("ground_truth", "") or "").strip()
    if not answer or "|" in answer:
        return ""
    _slot, separator, value_with_doc = answer.partition("=")
    return value_with_doc.rsplit("@", 1)[0].strip() if separator else answer


def _ambiguous_value_list(case: dict[str, Any]) -> bool:
    query = " ".join(str(case.get("query", "") or "").split())
    comma_count = query.count(",")
    if _int(case, "slot_count", 1) != 1 or comma_count < 2:
        return False
    marker = REDACTION_SLOT_RE.search(query)
    if not marker:
        return False
    before = query[: marker.start()]
    after = query[marker.end():]
    coordinate_tail = bool(re.search(
            r"\b(?:and|or)(?:\s+the)?\s*$",
            before,
            flags=re.IGNORECASE,
        ))
    return bool(
        coordinate_tail
        or (
            comma_count >= 4
            and (after.count(",") >= 2 or before.count(",") >= 3)
        )
    )


def value_redaction_case_has_clean_surface(
    case: dict[str, Any], *, small_corpus: bool
) -> bool:
    query = " ".join(str(case.get("query", "") or "").split())
    markers = REDACTION_SLOT_RE.findall(query)
    if not query or not markers:
        return False
    slot_count = _int(case, "slot_count", len(markers))
    if slot_count != len(markers) or not 1 <= slot_count <= 3:
        return False
    word_count = _word_count(case)
    if word_count < 5 or word_count > 88:
        return False
    if (
        BOILERPLATE_RE.search(query)
        or MALFORMED_VALUE_SLOT_GRAMMAR_RE.search(query)
        or re.search(
            r"[.!?]\s*\[REDACT:R\d+[^\]]*\]\s+"
            r"[A-Za-z][A-Za-z.\-]+,\s+(?:and|then|but)\b",
            query,
            flags=re.IGNORECASE,
        )
        or _answer_leaks_into_query(case)
        or _value_answer_is_clause_fragment(case)
        or _ambiguous_value_list(case)
    ):
        return False
    if re.search(r"\[REDACT:[^\]]+\]\s+(?:he|she|they|it)\b", query, flags=re.IGNORECASE):
        return False

    entity_type = str(case.get("entity_type", "") or "").strip().casefold()
    visible = _int(case, "visible_anchor_count")
    support = _int(case, "local_support_hits")
    same_type = _int(case, "same_type_hits")
    dated = _bool(case, "has_date_or_number")
    clean_direct = _bool(case, "clean_direct_slot")
    residual = value_redaction_residual_content_count(case)
    strong = value_redaction_has_strong_residual_context(case, small_corpus=small_corpus)
    note_fields = _note_fields(case)
    acronym_like = _bool(case, "acronym_like") or note_fields.get(
        "acronym_like", ""
    ).casefold() in {"1", "true", "yes", "y"}
    answer_words = _words(_single_value_answer(case))
    if (
        entity_type == "person"
        and acronym_like
        and len(answer_words) == 1
        and len(answer_words[0]) <= 4
        and visible == 0
        and not dated
    ):
        # Short all-caps person tokens without an independent visible anchor
        # are usually initials, OCR abbreviations, or unresolved local names.
        return False
    if slot_count == 1 and leading_slot_surface(case) and not (
        clean_direct or visible >= 1 or support >= 2 or dated or residual >= 6
    ):
        return False
    if entity_type in {"location", "country", "person"} and same_type >= 4:
        if not (clean_direct or visible >= 1 or support >= 2 or dated):
            return False
    if slot_count >= 2 and not (strong or support >= 2 or visible >= 1 or dated):
        return False
    if CONTEXT_DEPENDENT_LEAD_RE.search(query) and not (
        clean_direct or visible >= 1 or support >= 2 or dated or residual >= 7
    ):
        return False
    return strong or residual >= 4


def value_redaction_case_is_run_eligible(
    case: dict[str, Any], *, small_corpus: bool
) -> bool:
    if not value_redaction_case_has_clean_surface(case, small_corpus=small_corpus):
        return False
    return _float(case, "selector_score") >= 14.0


def value_redaction_passage_preference_tier(
    case: dict[str, Any], *, small_corpus: bool, pair_like_claim_corpus: bool
) -> int:
    del pair_like_claim_corpus
    kind = str(case.get("query_kind", "") or "").strip().casefold()
    strong = value_redaction_has_strong_residual_context(case, small_corpus=small_corpus)
    clean = _bool(case, "clean_direct_slot")
    if kind == "pair_sentence_slot" and (strong or clean):
        return 0
    if kind == "entity_slot" and (strong or clean):
        return 1
    if kind == "entity_slot_minimal" and strong:
        return 2
    if "small_corpus_emergency" in kind:
        return 4
    return 3


def claim_case_is_top10_eligible(
    case: dict[str, Any], *, small_corpus: bool, pair_like_claim_corpus: bool
) -> bool:
    del small_corpus, pair_like_claim_corpus
    query = " ".join(str(case.get("query", "") or "").split())
    if not query or not 4 <= _word_count(case) <= 65:
        return False
    if PRONOUN_LEAD_RE.search(query) or FIRST_PERSON_RE.search(query) or VAGUE_REFERENCE_RE.search(query):
        return False
    if BOILERPLATE_RE.search(query):
        return False
    fields = _note_fields(case)
    kind = str(case.get("query_kind", "") or "").strip().casefold()
    if kind == "claim_structured_query":
        actor = fields.get("actor", "").casefold()
        if not actor or not fields.get("predicate") or not fields.get("object"):
            return False
        if actor in {"he", "she", "they", "it", "we", "unspecified"}:
            return False
    if kind in CLAIM_RELATION_ONLY_KINDS:
        if SUBJECTLESS_PREDICATE_LEAD_RE.search(query):
            return False
        if _float(case, "selector_score") < 22.0:
            return False
    return bool(CLAIM_PREDICATE_RE.search(query) or fields.get("predicate"))


def contradiction_case_is_top10_eligible(
    case: dict[str, Any], *, small_corpus: bool, pair_like_claim_corpus: bool
) -> bool:
    del small_corpus, pair_like_claim_corpus
    query = " ".join(str(case.get("query", "") or "").split())
    if not query or not 4 <= _word_count(case) <= 70:
        return False
    if PRONOUN_LEAD_RE.search(query) or VAGUE_REFERENCE_RE.search(query) or BOILERPLATE_RE.search(query):
        return False
    kind = str(case.get("query_kind", "") or "").strip().casefold()
    generated_kind = kind in {
        "relation_contradiction_query", "claim_case_contradiction",
        "reconstruction_seed_contradiction", "relation_contradiction",
    }
    return generated_kind or bool(CONTRADICTION_CUE_RE.search(query)) or bool(_note_fields(case).get("strategy"))


def reconstruction_case_is_top10_eligible(
    case: dict[str, Any], *, small_corpus: bool, pair_like_claim_corpus: bool
) -> bool:
    del small_corpus, pair_like_claim_corpus
    query = " ".join(str(case.get("query", "") or "").split())
    if not query or not 4 <= _word_count(case) <= 70:
        return False
    if PRONOUN_LEAD_RE.search(query) or VAGUE_REFERENCE_RE.search(query) or BOILERPLATE_RE.search(query):
        return False
    kind = str(case.get("query_kind", "") or "").strip().casefold()
    if kind == "reconstruction_source_excerpt" and _word_count(case) > 52:
        return False
    return bool(CLAIM_PREDICATE_RE.search(query) or _note_fields(case).get("predicate"))


def _split_sentence_redaction_query(query: str) -> tuple[str, str, int]:
    match = SENTENCE_SLOT_RE.search(query)
    if not match:
        return query, "", 0
    return query[: match.start()].strip(), query[match.end() :].strip(), int(match.group(1))


def sentence_redaction_case_is_top10_eligible(
    case: dict[str, Any], *, small_corpus: bool, pair_like_claim_corpus: bool
) -> bool:
    query = " ".join(str(case.get("query", "") or "").split())
    left, right, approx_tokens = _split_sentence_redaction_query(query)
    if not right or approx_tokens <= 0:
        return False
    if approx_tokens > (48 if pair_like_claim_corpus else 60):
        return False
    if len(_words(left)) < 6 or len(_words(right)) < 6:
        return False
    ground_truth = " ".join(str(case.get("ground_truth", "") or "").split())
    if not 4 <= len(_words(ground_truth)) <= 65:
        return False
    if BOILERPLATE_RE.search(query) or BOILERPLATE_RE.search(ground_truth):
        return False
    fields = _note_fields(case)
    entity_hits = int(float(fields.get("entity_hits", 0) or 0))
    claim_overlap = int(float(fields.get("claim_overlap", 0) or 0))
    event_overlap = int(float(fields.get("event_overlap", 0) or 0))
    strong_support = entity_hits >= 2 or claim_overlap + event_overlap >= 2
    balanced_small_corpus_support = bool(
        small_corpus
        and entity_hits >= 1
        and len(_words(left)) >= 10
        and len(_words(right)) >= 10
    )
    if not strong_support and not balanced_small_corpus_support:
        return False
    if re.match(r"^(?:and|but|yes|no|third|finally)\b", right, flags=re.IGNORECASE):
        if len(_words(right)) < 12:
            return False
    return True


def selected_backfill_quality_eligible(
    task_name: str,
    case: dict[str, Any],
    *,
    small_corpus: bool,
    pair_like_claim_corpus: bool,
) -> bool:
    if not _case_surface_is_usable(case):
        return False
    if task_name == "claims":
        return claim_case_is_top10_eligible(case, small_corpus=small_corpus, pair_like_claim_corpus=pair_like_claim_corpus)
    if task_name == "contradictions":
        return contradiction_case_is_top10_eligible(case, small_corpus=small_corpus, pair_like_claim_corpus=pair_like_claim_corpus)
    if task_name == "reconstruction":
        return reconstruction_case_is_top10_eligible(case, small_corpus=small_corpus, pair_like_claim_corpus=pair_like_claim_corpus)
    if task_name == "value_redaction":
        return value_redaction_case_is_run_eligible(case, small_corpus=small_corpus)
    if task_name == "sentence_redaction":
        return sentence_redaction_case_is_top10_eligible(case, small_corpus=small_corpus, pair_like_claim_corpus=pair_like_claim_corpus)
    return True


def claim_case_backfill_penalty(
    case: dict[str, Any], *, small_corpus: bool, pair_like_claim_corpus: bool,
    corpus_code: str | None = None,
) -> float:
    del corpus_code
    query = " ".join(str(case.get("query", "") or "").split())
    kind = str(case.get("query_kind", "") or "").casefold()
    penalty = 0.0
    if not claim_case_is_top10_eligible(case, small_corpus=small_corpus, pair_like_claim_corpus=pair_like_claim_corpus):
        penalty += 100.0
    if ABSTRACT_LEAD_RE.search(query):
        penalty += 18.0
    if kind in CLAIM_RELATION_ONLY_KINDS:
        penalty += 18.0
    if kind in {"claim_small_emergency_surface", "claim_reconstruction_backfill"}:
        penalty += 12.0
    if _word_count(case) > 45:
        penalty += min((_word_count(case) - 45) * 0.5, 10.0)
    fields = _note_fields(case)
    if fields.get("actor") and fields.get("predicate") and fields.get("object"):
        penalty -= 10.0
    return penalty


def contradiction_case_backfill_penalty(
    case: dict[str, Any], *, small_corpus: bool, pair_like_claim_corpus: bool
) -> float:
    query = " ".join(str(case.get("query", "") or "").split())
    kind = str(case.get("query_kind", "") or "").casefold()
    penalty = 0.0
    if not contradiction_case_is_top10_eligible(case, small_corpus=small_corpus, pair_like_claim_corpus=pair_like_claim_corpus):
        penalty += 100.0
    if not CONTRADICTION_CUE_RE.search(query) and kind == "structured_claim_contradiction":
        penalty += 20.0
    if kind in {"relation_contradiction_query", "claim_case_contradiction"}:
        penalty -= 6.0
    if ABSTRACT_LEAD_RE.search(query):
        penalty += 12.0
    return penalty


def reconstruction_case_backfill_penalty(
    case: dict[str, Any], *, small_corpus: bool, pair_like_claim_corpus: bool
) -> float:
    query = " ".join(str(case.get("query", "") or "").split())
    kind = str(case.get("query_kind", "") or "").casefold()
    penalty = 0.0
    if not reconstruction_case_is_top10_eligible(case, small_corpus=small_corpus, pair_like_claim_corpus=pair_like_claim_corpus):
        penalty += 100.0
    if kind == "reconstruction_source_excerpt":
        penalty += 12.0
    if ABSTRACT_LEAD_RE.search(query):
        penalty += 12.0
    if _word_count(case) > 45:
        penalty += min((_word_count(case) - 45) * 0.5, 10.0)
    return penalty


def value_redaction_case_backfill_penalty(
    case: dict[str, Any], *, small_corpus: bool, corpus_code: str | None = None
) -> float:
    del corpus_code
    penalty = 0.0
    if not value_redaction_case_is_run_eligible(case, small_corpus=small_corpus):
        penalty += 100.0
    kind = str(case.get("query_kind", "") or "").casefold()
    slots = _int(case, "slot_count", 1)
    visible = _int(case, "visible_anchor_count")
    support = _int(case, "local_support_hits")
    residual = value_redaction_residual_content_count(case)
    if kind == "pair_sentence_slot":
        penalty -= 6.0
    elif kind == "entity_slot":
        penalty -= 3.0
    elif kind == "entity_slot_minimal":
        penalty += 5.0
    elif "small_corpus_emergency" in kind:
        penalty += 12.0
    if slots > 1:
        penalty += 4.0 * (slots - 1)
    if visible <= 0 and support <= 0:
        penalty += 12.0
    if residual < 5:
        penalty += 8.0
    if _bool(case, "clean_direct_slot"):
        penalty -= 8.0
    if _bool(case, "has_date_or_number"):
        penalty -= 3.0
    if leading_slot_surface(case):
        penalty += 6.0
    if CONTEXT_DEPENDENT_LEAD_RE.search(str(case.get("query", "") or "")):
        penalty += 8.0
    return penalty


def sentence_redaction_case_backfill_penalty(
    case: dict[str, Any], *, small_corpus: bool, pair_like_claim_corpus: bool
) -> float:
    penalty = 0.0
    if not sentence_redaction_case_is_top10_eligible(case, small_corpus=small_corpus, pair_like_claim_corpus=pair_like_claim_corpus):
        penalty += 100.0
    left, right, approx_tokens = _split_sentence_redaction_query(
        " ".join(str(case.get("query", "") or "").split())
    )
    penalty += min(abs(len(_words(left)) - len(_words(right))) * 0.15, 8.0)
    if approx_tokens > 40:
        penalty += (approx_tokens - 40) * 0.5
    fields = _note_fields(case)
    entity_hits = int(float(fields.get("entity_hits", 0) or 0))
    claim_overlap = int(float(fields.get("claim_overlap", 0) or 0))
    event_overlap = int(float(fields.get("event_overlap", 0) or 0))
    if entity_hits < 2 and claim_overlap + event_overlap < 2:
        penalty += 12.0
    penalty -= min(claim_overlap + event_overlap, 3) * 1.5
    return penalty


def _adjust_cases_for_penalty(
    records: list[dict[str, Any]], *,
    penalty_fn: Callable[[dict[str, Any]], float] | None,
) -> tuple[list[dict[str, Any]], str]:
    if penalty_fn is None:
        return records, "selector_score"
    adjusted: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["_selection_score"] = _float(item, "selector_score") - float(penalty_fn(record))
        adjusted.append(item)
    return adjusted, "_selection_score"


def _seeded_tiebreak(value: Any, seed: int) -> int:
    """Return a stable seed-dependent rank used only after quality fields."""
    payload = f"{int(seed)}\x1f{value}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def quality_fill_unique_query_cases(
    records: list[dict[str, Any]], target_count: int, *, max_cases_per_doc: int,
    score_field: str = "selector_score", doc_first_pass: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if target_count <= 0 or not records:
        return []
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_doc.setdefault(str(record.get("doc_id", "")), []).append(record)
    for bucket in by_doc.values():
        bucket.sort(
            key=lambda item: (
                _float(item, score_field, _float(item, "selector_score")),
                _word_count(item),
                _seeded_tiebreak(item.get("case_id", item.get("query", "")), seed),
                str(item.get("case_id", "")),
            ),
            reverse=True,
        )
    selected: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    per_doc_counts: dict[str, int] = {doc_id: 0 for doc_id in by_doc}

    def try_add(record: dict[str, Any]) -> bool:
        doc_id = str(record.get("doc_id", ""))
        if per_doc_counts.get(doc_id, 0) >= max_cases_per_doc:
            return False
        query_key = _canonical_query_key(record.get("query", ""))
        if query_key and query_key in seen_queries:
            return False
        selected.append(record)
        per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
        if query_key:
            seen_queries.add(query_key)
        return True

    if doc_first_pass:
        doc_order = sorted(
            by_doc,
            key=lambda doc_id: (
                _float(by_doc[doc_id][0], score_field, _float(by_doc[doc_id][0], "selector_score")),
                _word_count(by_doc[doc_id][0]),
                _seeded_tiebreak(doc_id, seed),
                doc_id,
            ),
            reverse=True,
        )
        for doc_id in doc_order:
            if len(selected) >= target_count:
                break
            try_add(by_doc[doc_id][0])
    leftovers = [record for bucket in by_doc.values() for record in bucket]
    leftovers.sort(
        key=lambda item: (
            _float(item, score_field, _float(item, "selector_score")),
            _word_count(item),
            _seeded_tiebreak(item.get("case_id", item.get("query", "")), seed),
            str(item.get("case_id", "")),
        ),
        reverse=True,
    )
    for record in leftovers:
        if len(selected) >= target_count:
            break
        try_add(record)
    return selected[:target_count]


def fill_preferred_then_backfill(
    preferred_cases: list[dict[str, Any]], fallback_cases: list[dict[str, Any]], *,
    target_count: int, max_cases_per_doc: int,
    backfill_penalty_fn: Callable[[dict[str, Any]], float] | None = None,
    doc_first_pass: bool = True, seed: int = 0,
) -> list[dict[str, Any]]:
    adjusted, score_field = _adjust_cases_for_penalty(preferred_cases, penalty_fn=backfill_penalty_fn)
    selected = quality_fill_unique_query_cases(
        adjusted, target_count, max_cases_per_doc=max_cases_per_doc,
        score_field=score_field, doc_first_pass=doc_first_pass, seed=seed,
    )
    if len(selected) >= target_count:
        return selected[:target_count]
    selected_ids = {str(case.get("case_id", "")).strip() for case in selected}
    selected_queries = {_canonical_query_key(case.get("query", "")) for case in selected}
    fallback = [
        case for case in fallback_cases
        if str(case.get("case_id", "")).strip() not in selected_ids
        and _canonical_query_key(case.get("query", "")) not in selected_queries
    ]
    adjusted, score_field = _adjust_cases_for_penalty(fallback, penalty_fn=backfill_penalty_fn)
    selected.extend(
        quality_fill_unique_query_cases(
            adjusted, target_count - len(selected), max_cases_per_doc=max_cases_per_doc,
            score_field=score_field, doc_first_pass=doc_first_pass, seed=seed,
        )
    )
    return selected[:target_count]


def build_catalog_cases_for_task(
    *, task_name: str, strict_task_cases: list[dict[str, Any]], requested: int,
    collector_counts: dict[str, int], sampling_with_doc_cap: dict[str, dict[str, Any]],
    max_cases_per_doc: int, seed: int, tier_order: dict[str, int],
    build_behavior: TaskBuildBehavior,
    collect_claim_cases_local: Callable[[dict[str, Any] | None], list[dict[str, Any]]] | None,
    collect_contradiction_cases_local: Callable[[dict[str, Any] | None], list[dict[str, Any]]] | None,
    catalog_runner_cfg: dict[str, Any],
    merge_ranked_candidates: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    catalog = list(strict_task_cases)
    target = int(collector_counts.get(task_name, requested))
    doc_cap = int(sampling_with_doc_cap.get(task_name, {}).get("max_cases_per_doc", max_cases_per_doc))
    if build_behavior.catalog_merge_mode == "claims_relaxed_merge" and collect_claim_cases_local:
        catalog = merge_ranked_candidates(
            [strict_task_cases, collect_claim_cases_local(catalog_runner_cfg)],
            target_count=target, max_cases_per_doc=doc_cap, score_key="selector_score",
            seed=seed + tier_order.get(task_name, 0),
        )
    elif build_behavior.catalog_merge_mode == "contradictions_relaxed_merge" and collect_contradiction_cases_local:
        catalog = merge_ranked_candidates(
            [strict_task_cases, collect_contradiction_cases_local(catalog_runner_cfg)],
            target_count=target, max_cases_per_doc=doc_cap, score_key="selector_score",
            seed=seed + tier_order.get(task_name, 0),
        )
    return catalog


def _task_penalty(
    task_name: str, *, small_corpus: bool, pair_like_claim_corpus: bool,
    corpus_code: str,
) -> Callable[[dict[str, Any]], float] | None:
    if task_name == "claims":
        return lambda case: claim_case_backfill_penalty(
            case, small_corpus=small_corpus,
            pair_like_claim_corpus=pair_like_claim_corpus, corpus_code=corpus_code,
        )
    if task_name == "contradictions":
        return lambda case: contradiction_case_backfill_penalty(
            case, small_corpus=small_corpus,
            pair_like_claim_corpus=pair_like_claim_corpus,
        )
    if task_name == "reconstruction":
        return lambda case: reconstruction_case_backfill_penalty(
            case, small_corpus=small_corpus,
            pair_like_claim_corpus=pair_like_claim_corpus,
        )
    if task_name == "value_redaction":
        return lambda case: value_redaction_case_backfill_penalty(
            case, small_corpus=small_corpus, corpus_code=corpus_code,
        )
    if task_name == "sentence_redaction":
        return lambda case: sentence_redaction_case_backfill_penalty(
            case, small_corpus=small_corpus,
            pair_like_claim_corpus=pair_like_claim_corpus,
        )
    return None


def select_run_cases_for_task(
    *, task_name: str, requested: int,
    strict_task_cases: list[dict[str, Any]], catalog_cases: list[dict[str, Any]],
    sampling_with_doc_cap: dict[str, dict[str, Any]], max_cases_per_doc: int,
    seed: int, tier_order: dict[str, int], build_behavior: TaskBuildBehavior,
    small_corpus: bool, pair_like_claim_corpus: bool, corpus_code: str,
    unique_query_round_robin: Callable[..., list[dict[str, Any]]],
    rebalance_value_redaction_cases: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    del unique_query_round_robin
    selection_seed = seed + tier_order.get(task_name, 0)
    source = catalog_cases if build_behavior.selection_source == "catalog" else strict_task_cases
    if requested <= 0:
        return list(source)
    doc_cap = int(sampling_with_doc_cap.get(task_name, {}).get("max_cases_per_doc", max_cases_per_doc))
    if small_corpus or pair_like_claim_corpus:
        doc_cap = max(doc_cap, requested)
    penalty_fn = _task_penalty(
        task_name, small_corpus=small_corpus,
        pair_like_claim_corpus=pair_like_claim_corpus, corpus_code=corpus_code,
    )
    preferred = [
        case for case in source
        if selected_backfill_quality_eligible(
            task_name, case, small_corpus=small_corpus,
            pair_like_claim_corpus=pair_like_claim_corpus,
        )
    ]
    fallback = [
        case for case in catalog_cases
        if selected_backfill_quality_eligible(
            task_name, case, small_corpus=small_corpus,
            pair_like_claim_corpus=pair_like_claim_corpus,
        )
    ]
    adjusted, score_field = _adjust_cases_for_penalty(preferred, penalty_fn=penalty_fn)
    if task_name == "value_redaction" and len(adjusted) > requested:
        rebalance_input: list[dict[str, Any]] = []
        for case in adjusted:
            item = dict(case)
            item["selector_score"] = _float(item, score_field, _float(item, "selector_score"))
            rebalance_input.append(item)
        selected = rebalance_value_redaction_cases(
            rebalance_input, requested, max_cases_per_doc=doc_cap,
            max_slot_count=int(sampling_with_doc_cap.get(task_name, {}).get("max_slot_count", 3)),
            slot_count_decay=float(sampling_with_doc_cap.get(task_name, {}).get("slot_count_decay", 0.5)),
        )
    else:
        selected = quality_fill_unique_query_cases(
            adjusted, requested, max_cases_per_doc=doc_cap,
            score_field=score_field, doc_first_pass=True, seed=selection_seed,
        )
    if len(selected) < requested and build_behavior.allow_selected_backfill_from_catalog:
        selected = fill_preferred_then_backfill(
            selected, fallback, target_count=min(requested, len(selected) + len(fallback)),
            max_cases_per_doc=doc_cap, backfill_penalty_fn=penalty_fn,
            doc_first_pass=True, seed=selection_seed,
        )
    return selected[:requested]
