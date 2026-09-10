#!/usr/bin/env python3
"""Task-specific rules for selected-slice backfill from admissible catalogs."""

from __future__ import annotations

from typing import Any


CLAIMS_BACKFILL_QUERY_KINDS = {
    "claim_pair_direct_surface",
    "claim_pair_parser_surface",
    "claim_pair_repeated_surface",
    "claim_direct_minimal",
    "claim_small_direct_surface",
    "claim_small_emergency_surface",
    "claim_reconstruction_backfill",
    "claim_structured_query",
}

CONTRADICTIONS_BACKFILL_QUERY_KINDS = {
    "claim_case_contradiction",
    "reconstruction_seed_contradiction",
    "relation_contradiction",
}

RECONSTRUCTION_BACKFILL_QUERY_KINDS = {
    "reconstruction_claim_surface",
    "reconstruction_event_surface",
    "reconstruction_source_excerpt",
    "reconstruction_relation_surface",
}

VALUE_REDACTION_BACKFILL_QUERY_KINDS = {
    "pair_sentence_slot",
    "entity_slot",
    "entity_slot_minimal",
    "entity_slot_small_corpus_emergency",
}


def catalog_case_is_selected_backfill_eligible(
    task_name: str,
    case: dict[str, Any],
    *,
    small_corpus: bool,
    pair_like_claim_corpus: bool,
) -> bool:
    query_kind = str(case.get("query_kind", "") or "").strip().casefold()
    selector_score = float(case.get("selector_score", 0.0) or 0.0)
    word_count = int(case.get("word_count", 0) or 0)
    query_text = str(case.get("query", "") or "")

    if not query_text or word_count <= 0:
        return False

    if task_name == "claims":
        if query_kind not in CLAIMS_BACKFILL_QUERY_KINDS:
            return False
        return selector_score >= (20.0 if small_corpus else 90.0)

    if task_name == "contradictions":
        if query_kind not in CONTRADICTIONS_BACKFILL_QUERY_KINDS:
            return False
        return selector_score >= (14.0 if small_corpus else 85.0)

    if task_name == "reconstruction":
        if query_kind not in RECONSTRUCTION_BACKFILL_QUERY_KINDS:
            return False
        if pair_like_claim_corpus and small_corpus:
            return selector_score >= 70.0
        return selector_score >= (110.0 if small_corpus else 150.0)

    if task_name == "value_redaction":
        if query_kind not in VALUE_REDACTION_BACKFILL_QUERY_KINDS:
            return False
        notes = str(case.get("notes", "") or "").casefold()
        if query_kind == "pair_sentence_slot":
            return selector_score >= (38.0 if small_corpus else 48.0)
        if query_kind == "entity_slot":
            return selector_score >= (28.0 if small_corpus else 40.0)
        if query_kind == "entity_slot_minimal":
            if pair_like_claim_corpus:
                return selector_score >= 42.0 and "fallback=claim_surface" not in notes
            if small_corpus:
                return selector_score >= 48.0
            return selector_score >= 42.0
        if query_kind == "entity_slot_small_corpus_emergency":
            return selector_score >= (16.0 if small_corpus else 28.0)

    return False
