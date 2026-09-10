#!/usr/bin/env python3
"""
query_semantic_parser_v0_3_22_zgx.py

On-the-fly semantic parser for redaction and document-reconstruction query texts.

v0.4.10 coordinated Requirements v1.10 Final parser correction
---------------------------
1. Makes primary relation output corpus-agnostic and safe by default:
   heuristic co-occurrence relations are debug hints, not asserted semantic
   relations, and are not written as primary matching rows unless explicitly
   requested with --allow-heuristic-relations-primary.
2. Adds query_propositions.csv as a normalized proposition layer for claim
   search and contradiction discovery.  Propositions are derived from primary
   claims and asserted primary relations, not from heuristic co-occurrence rows.
3. Adds polarity/modality normalization for claims and relations so downstream
   contradiction logic can compare positive, negative, and uncertain evidence
   across different corpora.
4. Keeps fallback_only as the recommended policy for entities/claims/events,
   but treats relations more strictly because non-asserted co-occurrence edges
   are not safe for contradiction or claim matching.
5. Preserves debug files for all heuristic and OpenAI rows for audit.
6. Filters redaction placeholders out of primary entity rows.  Redaction slots
   remain only in query_redaction_slots.csv, where they belong.
7. Makes heuristic claim promotion explicit. In OpenAI mode, OpenAI claims are
   preferred; a deterministic claim is promoted only for a fragment where the
   model emitted no claim, unless --allow-heuristic-claims-primary requests the
   broader configured policy.
8. Filters redaction placeholders out of event rows as well as entity rows.
   Placeholder-bearing event rows are audit/debug artifacts, not primary matching
   evidence.
9. Makes heuristic event promotion explicit. In OpenAI mode, heuristic event rows
   are kept out of primary query_events.csv by default unless
   --allow-heuristic-events-primary is passed.
10. Adds explicit sentence/span redaction syntax for semantic reconstruction:
    [REDACT_SENT:S1 type=CLAIM approx_tokens=10] and
    [REDACT_SPAN:S1 type=ANY approx_tokens=12]. These are written only to
    query_sentence_redaction_slots.csv.
11. Adds redaction-format cleanup for common shorthand markers such as
    [REDACT type=PERSON] and [REDACT_SENT type=CLAIM]. The parser records the
    normalized marker and repair flag so downstream code can still audit what
    happened.

Outputs:
    query_fragments.csv
    query_entities.csv
    query_claims.csv
    query_events.csv
    query_relations.csv
    query_propositions.csv
    query_redaction_slots.csv
    query_sentence_redaction_slots.csv
    query_parse_manifest.json

Typical usage in Colab after setting OPENAI_API_KEY:
    python query_semantic_parser_v0_3_20_zgx.py \
      --query-file /content/drive/MyDrive/kg_run_3000docs/2doc_reconstruction.txt \
      --output-dir /content/drive/MyDrive/kg_run_3000docs/query_parse_v0318 \
      --use-openai \
      --openai-model gpt-4o
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

# Requirements v1.9 deliberately centralizes meaning-bearing normalization.
# Keeping these helpers in one module prevents the parser and matcher from
# disagreeing about punctuation-only context, number roles, dates, or event
# state.  Keep matryoshka_v19_common.py beside this script.
from .semantics import (
    NORMALIZATION_VERSION,
    STRONG_CLAUSE_NORMALIZATION_STATUSES,
    build_event_signature,
    canonical_object_head,
    canonical_predicate,
    clean_text as v19_clean_text,
    has_semantic_tokens,
    infer_value_slot_frame,
    normalize_escaped_layout,
    infer_simple_clause_arguments,
    normalize_numeric_arguments,
    normalize_sentence_slot_context,
    semantic_tokens,
    sha256_file,
    stable_hash as v19_stable_hash,
    utc_now_iso,
    write_csv_with_schema,
)

VERSION = "v0.4.10"
REQUIREMENTS_VERSION = (
    "Matryoshka Requirements v1.4 + Addendum v1.4.2 Revision 1"
)
SCHEMA_VERSION = "query_parse_schema_v0_4_7"
PROMPT_VERSION = "query_semantic_prompt_v0_4_10"
HEURISTIC_POLICY_VERSION = "heuristic_policy_v0_4_11"
OPENAI_TELEMETRY_COLUMNS = [
    "fragment_id", "model", "attempt_number", "outcome",
    "api_elapsed_seconds", "timeout_seconds", "retry_scheduled",
    "rate_limit_observed", "deterministic_fallback_state",
]
_OPENAI_TELEMETRY_ROWS: List[Dict[str, Any]] = []

# DID markers are analyst/evaluation references. They are captured as metadata,
# but intentionally not used for chunking or semantic extraction.
DID_RE = re.compile(r"\bDID_([A-Za-z0-9_\-]+)\b")

# Non-DID cable/document IDs, if present, may still be useful as references.
DOC_ID_RE = re.compile(r"\b(?:19\d{2})[A-Z]{2,}[A-Z0-9]*\d{3,}\b")

# Explicit redaction syntax used in the current experiments, e.g.
# [REDACT:R1 type=PERSON], [REDACT:R5 type=POSITION]
EXPLICIT_REDACTION_RE = re.compile(
    r"\[\s*REDACT\s*:\s*(?P<slot_label>R\d+)\s*(?:type\s*=\s*(?P<type>[A-Z_][A-Z0-9_\-/ ]*))?\s*\]",
    re.IGNORECASE,
)

# v0.3.19: common shorthand value redaction syntax.  Earlier experiments used
# [REDACT type=PERSON], which the strict v0.3.18 regex did not count as a slot.
# We now accept it and normalize it to [REDACT:Rn type=PERSON] while recording
# the repair in query_redaction_slots.csv.
SHORTHAND_REDACTION_RE = re.compile(
    r"\[\s*REDACT\s+(?:type\s*=\s*(?P<type>[A-Z_][A-Z0-9_\-/ ]*))?\s*\]",
    re.IGNORECASE,
)

# v0.3.19: sentence/span-level redaction.  This is deliberately separate from
# value redaction.  A sentence redaction means: find a semantically supported
# proposition/sentence candidate, not an entity/value fill.
SENTENCE_REDACTION_RE = re.compile(
    r"\[\s*(?P<kind>REDACT_SENT|REDACT_SPAN)\s*(?::\s*(?P<slot_label>S\d+))?"
    r"(?:\s+type\s*=\s*(?P<type>[A-Z_][A-Z0-9_\-/ ]*))?"
    r"(?:\s+approx_tokens\s*=\s*(?P<approx_tokens>\d+))?\s*\]",
    re.IGNORECASE,
)

# Generic fallback redaction markers.
GENERIC_REDACTION_RE = re.compile(
    r"(\[\s*REDACTED\s*\]|\[\s*redacted\s*\]|\[\s*R\d+\s*\]|_{3,}|█+|\*{3,}|<\s*redacted\s*>)",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.]?\s+\d{1,2}(?:,\s*\d{4})?"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b19\d{2}\b",
    re.IGNORECASE,
)

# Polarity and uncertainty patterns are used to build the generic proposition
# layer.  They are intentionally domain-agnostic: the same code should work on
# diplomatic cables, legal memoranda, news archives, corporate documents, etc.
NEGATION_RE = re.compile(
    r"\b(no|not|never|neither|nor|without|denied|denies|deny|failed to|did not|cannot|can't|won't|would not|refused|rejected|opposed|lack|lacks|lacked|lacking)\b",
    re.IGNORECASE,
)
UNCERTAIN_RE = re.compile(
    r"\b(alleged|reportedly|possibly|may|might|could|unclear|unknown|uncertain|appears|seems|suggests|rumor|rumour|rumored|rumoured|likely|unlikely)\b",
    re.IGNORECASE,
)

NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:[A-Z]\.\s*)?[A-Z][a-z]+){1,4}\b")
SINGLE_CAP_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{3,}\b")

EVENT_VERBS = {
    "met", "meet", "meeting", "discussed", "discussion", "visited", "visit", "appointed",
    "resigned", "announced", "reported", "warned", "asked", "requested", "supported",
    "opposed", "denied", "held", "hosted", "sent", "received", "instructed", "agreed",
    "rejected", "protested", "vigil", "prayer", "transferred", "nominated", "selected",
    "consulted", "called", "raised", "informed", "stated", "said", "told", "proposed",
}

# Stoplists are intentionally conservative. The goal is not perfect entity
# extraction in heuristic mode; the goal is to avoid obvious garbage entities.
LOWER_STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "for", "on", "with", "by", "from", "as", "at",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "a", "an", "it", "its", "his", "her", "their", "our", "they", "we", "i", "you", "he", "she",
    "have", "has", "had", "having", "some", "any", "all", "both", "either", "neither", "not",
    "will", "would", "can", "could", "may", "might", "must", "should", "shall", "do", "does", "did",
    "if", "then", "than", "there", "here", "when", "where", "which", "who", "whom", "what", "why", "how",
}
UPPER_STOPWORDS = {
    "REDACT", "REDACTED", "PERSON", "POSITION", "LOCATION", "ORGANIZATION", "ORG", "R", "TYPE",
    "DID", "DOC", "DOCUMENT", "QUERY", "FRAGMENT", "UNIT", "TEXT", "PARAGRAPH", "SECTION",
    "THE", "AND", "OR", "OF", "TO", "IN", "FOR", "ON", "WITH", "BY", "FROM", "AS", "AT",
    "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "THIS", "THAT", "THESE", "THOSE", "A", "AN",
    "IT", "ITS", "HIS", "HER", "THEIR", "OUR", "THEY", "WE", "HE", "SHE", "HAVE", "HAS", "HAD",
    "SOME", "ANY", "ALL", "BOTH", "EITHER", "NEITHER", "NOT", "WILL", "WOULD", "CAN", "COULD",
    "MAY", "MIGHT", "MUST", "SHOULD", "SHALL", "DO", "DOES", "DID", "IF", "THEN", "THAN",
    "THERE", "HERE", "WHEN", "WHERE", "WHICH", "WHO", "WHOM", "WHAT", "WHY", "HOW", "QUESTION", "MENTIONED", "CONCERNED", "REPRESENTATIVES",
    "REGARDING", "REGARDS", "SUBJECT", "REF", "REFTEL", "INFO", "STATE", "DEPT", "EMBASSY",
}

FRAGMENT_COLUMNS = [
    "fragment_id", "fragment_index", "fragment_text", "fragment_text_for_extraction",
    "fragment_text_normalized_redactions",
    "did_markers_json", "expected_doc_ids_in_text_json",
    "redaction_marker_count", "value_redaction_marker_count", "sentence_redaction_marker_count",
    "has_sentence_redaction", "char_count",
    "sentence_count", "fragment_source", "parser_note",
]
ENTITY_COLUMNS = [
    "query_entity_id", "fragment_id", "label", "canonical_key", "type", "role_or_context",
    "evidence_text", "extraction_basis", "parser_source", "confidence", "primary_for_matching",
]
CLAIM_COLUMNS = [
    "query_claim_id", "fragment_id", "claim_text", "speaker", "target", "stance",
    "evidence_text", "extraction_basis", "parser_source", "confidence", "primary_for_matching",
]
EVENT_COLUMNS = [
    "query_event_id", "fragment_id", "event_text", "event_type", "location", "date_or_temporal_context",
    "evidence_text", "extraction_basis", "parser_source", "confidence", "primary_for_matching",
]
RELATION_COLUMNS = [
    "query_relation_id", "fragment_id", "subject_entity", "predicate", "object_entity", "qualifier",
    "evidence_text", "extraction_basis", "parser_source", "confidence", "primary_for_matching",
]
REDACTION_SLOT_COLUMNS = [
    "query_redaction_slot_id", "fragment_id", "slot_index_global", "slot_index_in_fragment",
    "slot_label", "slot_marker", "normalized_marker", "marker_start", "marker_end",
    "explicit_type_hint", "expected_type_hint",
    "local_context_before", "local_context_after",
    # v0.3.20 canonical slot grammar.  These are constraints for the matcher,
    # not preselected answers.
    "slot_frame", "grammatical_role", "governing_predicate",
    "governing_predicate_surface", "governing_predicate_normalized",
    "clause_level",
    "slot_clause_level", "slot_event_signature_json",
    "required_anchor_fields_json",
    "required_relation", "modified_head", "administrative_container_expected",
    "frame_trigger",
    "extraction_basis", "parser_source", "confidence", "primary_for_matching",
    "redaction_format_repaired",
]

# v0.3.19 output contract for sentence/span redactions.  This file is read by
# matcher v0.3.30.  Keep it separate from query_redaction_slots.csv because the
# downstream task is different: semantic sentence reconstruction rather than
# short value/entity filling.
SENTENCE_REDACTION_SLOT_COLUMNS = [
    "query_sentence_redaction_slot_id", "fragment_id", "slot_index_global", "slot_index_in_fragment",
    "sentence_slot_label", "slot_marker", "normalized_marker", "marker_start", "marker_end",
    "slot_kind", "expected_semantic_type", "approx_tokens",
    "local_context_before", "local_context_after", "sentence_before", "sentence_after",
    # Original and normalized fields are both retained.  This makes automatic
    # boundary repair auditable and permits clean/punctuation variants to share
    # one normalized context hash.
    "original_sentence_before", "original_sentence_after",
    "normalized_sentence_before", "normalized_sentence_after",
    "left_context_semantic_token_count", "right_context_semantic_token_count",
    "boundary_repair_applied", "boundary_repair_reason", "context_mode_detected",
    "normalized_slot_context_sha256",
    "paragraph_context", "extraction_basis", "parser_source", "confidence",
    "primary_for_matching", "redaction_format_repaired",
]

# Normalized proposition rows are the preferred interface for claim search and
# contradiction discovery.  A proposition can come from either a claim row or an
# asserted relation row.  Heuristic co-occurrence relations should not become
# propositions because they do not assert facts.
PROPOSITION_COLUMNS = [
    "query_proposition_id", "fragment_id", "proposition_source", "source_row_id",
    "hypothesis_group_id", "proposition_hypothesis_id", "hypothesis_type",
    "subject_semantic_role", "hypothesis_confidence",
    "hypothesis_provenance_json",
    "proposition_text", "normalized_proposition", "canonical_subject", "canonical_predicate", "canonical_object",
    "canonical_location", "object_surface", "predicate_surface",
    "reporter", "embedded_actor", "agent", "patient_or_theme",
    "recipient", "destination", "voice", "role_normalization_status",
    "clause_level",
    "reporting_event_signature_json", "embedded_assertion_signature_json",
    "event_time", "event_time_precision", "report_time",
    "assertion_polarity", "polarity", "modality", "speaker_stance", "stance",
    "epistemic_state", "semantic_state_variable", "semantic_state_value",
    "packet_count", "weight_kg", "weight_g", "total_weight_grams", "date_value",
    "event_signature_json", "event_signature_sha256",
    "representation_enrichment_json",
    "evidence_text", "extraction_basis", "parser_source", "confidence",
    "primary_for_matching", "contradiction_key",
]

EVENT_SIGNATURE_COLUMNS = [
    "query_event_signature_id", "fragment_id", "source_kind", "source_row_id",
    "proposition_text", "actor", "agent", "patient_or_theme",
    "recipient", "destination", "voice", "role_normalization_status",
    "reporter", "embedded_actor",
    "predicate", "predicate_surface", "object", "object_surface",
    "location", "location_identity", "event_time",
    "event_time_precision", "report_time", "source_attribution", "stance", "modality",
    "assertion_polarity", "epistemic_state", "semantic_state_variable", "semantic_state_value",
    "packet_count", "weight_kg", "weight_g", "total_weight_grams", "date_value",
    "event_signature_json", "event_signature_sha256", "primary_for_matching",
]

TASK_PREFLIGHT_COLUMNS = [
    "task", "target_id", "status", "reason", "repair_applied",
    "left_semantic_tokens", "right_semantic_tokens", "normalized_context_sha256",
    "cross_representation_agreement_count", "resolved_conflict_count",
    "unresolved_conflict_count",
]

REPRESENTATION_RECONCILIATION_COLUMNS = [
    "fragment_id", "target_id", "source_sentence",
    "representation_target_ids_json", "representation_sources_json",
    "representation_count", "agreement_count", "resolved_conflict_count",
    "unresolved_conflict_count", "actor", "patient_or_theme",
    "predicate", "recipient", "destination", "polarity", "event_time",
    "enriched_fields_json", "enrichment_sources_json",
    "reconciliation_status", "reconciliation_reason",
]



@dataclass
class QueryParseResult:
    output_dir: str
    query_file: str
    parser_mode: str
    fragment_count: int
    entity_count: int
    claim_count: int
    event_count: int
    relation_count: int
    proposition_count: int
    redaction_slot_count: int
    sentence_redaction_slot_count: int
    output_files: Dict[str, str]
    warnings: List[str]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any, max_len: Optional[int] = None) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if max_len and len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def canonical_key(value: Any) -> str:
    text = clean_text(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_id(parts: Iterable[Any], prefix: str) -> str:
    material = "||".join("" if p is None else str(p) for p in parts)
    return prefix + "_" + hashlib.sha1(material.encode("utf-8", errors="ignore")).hexdigest()[:16]


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame with stable columns, even if it is empty."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(columns))
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    # Keep known columns first but preserve extra columns for debugging.
    extras = [c for c in out.columns if c not in columns]
    return out[list(columns) + extras]


def sentence_split(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    # Conservative split: not perfect, but avoids splitting document IDs.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text)
    return [p.strip() for p in parts if p.strip()]


def remove_reference_markers_for_extraction(text: str) -> str:
    """Remove DID markers and redaction marker internals before heuristic extraction."""
    no_did = DID_RE.sub(" ", text)
    # Keep a placeholder for redactions so sentence context still shows a gap,
    # but remove R1/type/PERSON internals that were polluting entity extraction.
    no_redact = EXPLICIT_REDACTION_RE.sub(" [REDACTION_SLOT] ", no_did)
    no_redact = SHORTHAND_REDACTION_RE.sub(" [REDACTION_SLOT] ", no_redact)
    no_redact = SENTENCE_REDACTION_RE.sub(" [SENTENCE_REDACTION_SLOT] ", no_redact)
    no_redact = GENERIC_REDACTION_RE.sub(" [REDACTION_SLOT] ", no_redact)
    return clean_text(no_redact)


def extract_did_metadata(text: str) -> Tuple[List[str], List[str]]:
    markers = [m.group(0) for m in DID_RE.finditer(text)]
    # If DID_1977STATE252921 appears, expected_doc_ids stores 1977STATE252921.
    expected_docs = [m.group(1) for m in DID_RE.finditer(text)]
    return sorted(set(markers)), sorted(set(expected_docs))


def _collect_non_overlapping_redaction_spans(text: str) -> Tuple[List[Tuple[int, int]], int, int]:
    """Return all redaction spans plus value/sentence counts.

    v0.3.19 uses this in fragment metadata.  Sentence redactions are counted
    separately because they route to a different matcher task.
    """
    spans: List[Tuple[int, int]] = []
    value_count = 0
    sentence_count = 0
    def add(span: Tuple[int, int], kind: str) -> None:
        nonlocal value_count, sentence_count
        if any(max(span[0], s[0]) < min(span[1], s[1]) for s in spans):
            return
        spans.append(span)
        if kind == "sentence":
            sentence_count += 1
        else:
            value_count += 1
    for m in SENTENCE_REDACTION_RE.finditer(text):
        add(m.span(), "sentence")
    for m in EXPLICIT_REDACTION_RE.finditer(text):
        add(m.span(), "value")
    for m in SHORTHAND_REDACTION_RE.finditer(text):
        add(m.span(), "value")
    for m in GENERIC_REDACTION_RE.finditer(text):
        add(m.span(), "value")
    spans.sort()
    return spans, value_count, sentence_count


def count_redaction_markers(text: str) -> int:
    spans, _, _ = _collect_non_overlapping_redaction_spans(text)
    return len(spans)


def normalize_redaction_markers_for_fragment(text: str) -> str:
    """Return a human-readable fragment text with repaired marker syntax.

    This is for audit/UI display only.  The original fragment_text is preserved.
    """
    counters = {"R": 0, "S": 0}
    def repl_sent(m: re.Match) -> str:
        label = clean_text(m.groupdict().get("slot_label", "")).upper()
        if not label:
            counters["S"] += 1
            label = f"S{counters['S']}"
        typ = clean_text(m.groupdict().get("type", "ANY")).upper() or "ANY"
        toks = clean_text(m.groupdict().get("approx_tokens", ""))
        return f"[REDACT_SENT:{label} type={typ}" + (f" approx_tokens={toks}" if toks else "") + "]"
    out = SENTENCE_REDACTION_RE.sub(repl_sent, text)
    def repl_short(m: re.Match) -> str:
        counters["R"] += 1
        typ = clean_text(m.groupdict().get("type", "UNKNOWN")).upper() or "UNKNOWN"
        return f"[REDACT:R{counters['R']} type={typ}]"
    out = SHORTHAND_REDACTION_RE.sub(repl_short, out)
    return clean_text(out)


def split_long_paragraph(paragraph: str, max_sentences_per_fragment: int, max_chars_per_fragment: int) -> List[str]:
    """Split only long paragraphs; DID markers are not boundaries."""
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars_per_fragment:
        return [paragraph]
    sents = sentence_split(paragraph)
    if not sents:
        return [paragraph]
    chunks = []
    current: List[str] = []
    for sent in sents:
        candidate = " ".join(current + [sent]).strip()
        if current and (len(candidate) > max_chars_per_fragment or len(current) >= max_sentences_per_fragment):
            chunks.append(" ".join(current).strip())
            current = [sent]
        else:
            current.append(sent)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def split_query_into_fragments(
    text: str,
    max_sentences_per_fragment: int = 6,
    max_chars_per_fragment: int = 2200,
) -> pd.DataFrame:
    """Split query into semantic fragments.

    Design rule for v0.3.14:
      - Blank lines / paragraphs define primary fragments.
      - DID_... markers are captured as metadata only.
      - DID_... markers are NOT used as chunk boundaries.
      - Long paragraphs may be split into sentence windows for LLM context size.
    """
    text, escaped_layout_repairs = normalize_escaped_layout(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    rows: List[Dict[str, Any]] = []
    frag_index = 1
    for para_index, para in enumerate(paragraphs, start=1):
        subfrags = split_long_paragraph(para, max_sentences_per_fragment, max_chars_per_fragment)
        for sub_index, frag_text in enumerate(subfrags, start=1):
            did_markers, expected_docs = extract_did_metadata(frag_text)
            _all_redaction_spans, _value_redaction_count, _sentence_redaction_count = _collect_non_overlapping_redaction_spans(frag_text)
            rows.append({
                "fragment_id": f"U{frag_index}",
                "fragment_index": frag_index,
                "fragment_text": frag_text,
                "fragment_text_for_extraction": remove_reference_markers_for_extraction(frag_text),
                "fragment_text_normalized_redactions": normalize_redaction_markers_for_fragment(frag_text),
                "did_markers_json": json.dumps(did_markers),
                "expected_doc_ids_in_text_json": json.dumps(expected_docs),
                "redaction_marker_count": len(_all_redaction_spans),
                "value_redaction_marker_count": _value_redaction_count,
                "sentence_redaction_marker_count": _sentence_redaction_count,
                "has_sentence_redaction": bool(_sentence_redaction_count),
                "char_count": len(frag_text),
                "sentence_count": len(sentence_split(frag_text)),
                "fragment_source": f"paragraph_{para_index}" + (f"_part_{sub_index}" if len(subfrags) > 1 else ""),
                "parser_note": (
                    "paragraph_first; DID markers metadata only; "
                    "value/sentence redactions separated; "
                    f"escaped_layout_repairs={escaped_layout_repairs}"
                ),
            })
            frag_index += 1
    return ensure_columns(pd.DataFrame(rows), FRAGMENT_COLUMNS)


def is_bad_entity_label(label: str) -> bool:
    if not label:
        return True
    key = canonical_key(label)
    if not key or key in LOWER_STOPWORDS:
        return True
    up = label.upper().strip()
    if up in UPPER_STOPWORDS:
        return True
    # Remove isolated R1/R2 marker pieces.
    if re.fullmatch(r"R\d+", up):
        return True
    # Avoid extracting generic placeholder.
    if key in {"redaction slot", "redacted", "blank", "person", "position", "location"}:
        return True
    return False


def is_redaction_placeholder_label(label: Any) -> bool:
    """Return True when an entity label is actually a redaction marker.

    This is a corpus-agnostic hygiene rule.  A redaction marker such as
    ``[REDACT:R1 type=PERSON]`` is not an entity; it is a slot to be solved.
    The authoritative representation for such markers is
    ``query_redaction_slots.csv``.  Letting placeholders into
    ``query_entities.csv`` pollutes downstream entity matching and can create
    circular redaction candidates.
    """
    raw = clean_text(label)
    if not raw:
        return False
    up = raw.upper().strip()
    key = canonical_key(raw)

    if EXPLICIT_REDACTION_RE.fullmatch(raw) or SHORTHAND_REDACTION_RE.fullmatch(raw) or SENTENCE_REDACTION_RE.fullmatch(raw) or GENERIC_REDACTION_RE.fullmatch(raw):
        return True
    if "[REDACT" in up or "REDACTION_SLOT" in up or "SENTENCE_REDACTION_SLOT" in up or "<REDACT" in up:
        return True
    if re.fullmatch(r"R\d+", up):
        return True
    if key in {
        "redact", "redacted", "redaction", "redaction slot",
        "redact r1", "redact r2", "redact r3", "redact r4", "redact r5", "redact r6", "redact r7",
        "person", "position", "location", "organization", "org",
    }:
        return True
    # Canonical keys from labels such as "REDACT R1 type PERSON".
    if re.fullmatch(r"redact r\d+( type [a-z0-9_ /-]+)?", key):
        return True
    return False


def filter_redaction_placeholder_entities(df: pd.DataFrame, source_note: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split entity rows into clean rows and redaction-placeholder rows.

    The returned ``suppressed`` DataFrame is intended for audit/debug output.
    It should not be used as primary matching evidence.  This function is
    applied before primary row selection, so placeholder rows do not block
    fallback behavior.  For example, if OpenAI returns only ``[REDACT:R1]`` as
    an entity for a fragment, those rows are removed first and the selector may
    still fall back to heuristic entities if policy allows.
    """
    df = ensure_columns(df, ENTITY_COLUMNS)
    if df.empty or "label" not in df.columns:
        return df, pd.DataFrame(columns=list(ENTITY_COLUMNS) + ["suppression_reason", "suppression_source_note"])

    mask = df["label"].map(is_redaction_placeholder_label)
    kept = df[~mask].copy()
    suppressed = df[mask].copy()
    if len(suppressed):
        suppressed["primary_for_matching"] = False
        suppressed["suppression_reason"] = "redaction_placeholder_entity_label"
        suppressed["suppression_source_note"] = source_note
    return ensure_columns(kept, ENTITY_COLUMNS), suppressed


def row_contains_redaction_placeholder(row: Mapping[str, Any], fields: Sequence[str]) -> bool:
    """Return True if any named field contains a redaction placeholder.

    v0.3.18 generalizes the v0.3.18 entity hygiene rule.  A row such as
    ``[REDACTION_SLOT] is scheduled to replace [REDACTION_SLOT]`` may look like
    an event-like sentence to the heuristic parser, but it is not useful primary
    semantic evidence.  It should be retained for debugging only.
    """
    for field in fields:
        # ``row`` may be a plain dict or a pandas Series. Both expose ``get``;
        # do not require ``isinstance(row, Mapping)`` because Series is not a
        # Mapping subclass in all pandas versions.
        value = clean_text(row.get(field, "")) if hasattr(row, "get") else ""
        if not value:
            continue
        up = value.upper()
        if EXPLICIT_REDACTION_RE.search(value) or SHORTHAND_REDACTION_RE.search(value) or SENTENCE_REDACTION_RE.search(value) or GENERIC_REDACTION_RE.search(value):
            return True
        if "[REDACT" in up or "REDACTION_SLOT" in up or "SENTENCE_REDACTION_SLOT" in up or "<REDACT" in up:
            return True
    return False


def filter_redaction_placeholder_events(df: pd.DataFrame, source_note: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split event rows into clean rows and redaction-placeholder rows.

    Redaction placeholders are structural slots, not historical events.  This
    function prevents rows containing ``[REDACTION_SLOT]`` or
    ``[REDACT:R1 type=...]`` from entering primary query_events.csv.  Suppressed
    rows are still written to query_events_suppressed_debug.csv so developers can
    see exactly what was removed.
    """
    df = ensure_columns(df, EVENT_COLUMNS)
    if df.empty:
        return df, pd.DataFrame(columns=list(EVENT_COLUMNS) + ["suppression_reason", "suppression_source_note"])

    mask = df.apply(
        lambda row: row_contains_redaction_placeholder(
            row, ["event_text", "event_type", "location", "date_or_temporal_context", "evidence_text", "extraction_basis"]
        ),
        axis=1,
    )
    kept = df[~mask].copy()
    suppressed = df[mask].copy()
    if len(suppressed):
        suppressed["primary_for_matching"] = False
        suppressed["suppression_reason"] = "redaction_placeholder_event_row"
        suppressed["suppression_source_note"] = source_note
    return ensure_columns(kept, EVENT_COLUMNS), suppressed


def infer_entity_type(label: str) -> str:
    key = canonical_key(label)
    if DOC_ID_RE.fullmatch(label):
        return "DOCUMENT_ID"
    if key in {"iran", "turkey", "somalia", "united states", "washington", "tehran", "israel", "iraq", "syria", "egypt"}:
        return "LOCATION_OR_COUNTRY"
    if key in {"ambassador", "minister", "secretary", "general", "governor", "deputy", "advisor"}:
        return "POSITION"
    if any(w in key for w in ["department", "embassy", "ministry", "cathedral", "university", "committee", "council", "government", "office"]):
        return "ORGANIZATION"
    if re.fullmatch(r"[A-Z][a-z]+(?:\s+(?:[A-Z]\.\s*)?[A-Z][a-z]+){1,4}", label):
        return "PERSON_OR_ORG"
    if re.fullmatch(r"[A-Z]{3,}", label):
        return "ACRONYM_OR_ORG"
    return "ENTITY"


def extract_heuristic_entities(fragment_id: str, text_for_extraction: str) -> List[Dict[str, Any]]:
    labels: List[str] = []
    # Doc IDs can be useful if they appear directly, but DID_ markers were already removed.
    labels.extend(DOC_ID_RE.findall(text_for_extraction))
    labels.extend(NAME_RE.findall(text_for_extraction))
    labels.extend(SINGLE_CAP_RE.findall(text_for_extraction))
    labels.extend(ACRONYM_RE.findall(text_for_extraction))

    # Drop single-word pieces that are merely components of a longer extracted phrase,
    # unless they are useful standalone geopolitical/person anchors such as Iran/Tehran/Vance.
    multiword_keys = {canonical_key(x) for x in labels if len(canonical_key(x).split()) > 1}
    useful_singletons = {"iran", "iraq", "egypt", "turkey", "tehran", "vance", "shah", "zahedi", "afshar"}

    rows = []
    seen = set()
    for label in sorted(labels, key=lambda x: text_for_extraction.find(x)):
        label = clean_text(label)
        key = canonical_key(label)
        if key in seen or is_bad_entity_label(label):
            continue
        if len(key.split()) == 1 and key not in useful_singletons:
            if any(key in mw.split() for mw in multiword_keys):
                continue
        seen.add(key)
        rows.append({
            "query_entity_id": stable_id([VERSION, fragment_id, label], "qe"),
            "fragment_id": fragment_id,
            "label": label,
            "canonical_key": key,
            "type": infer_entity_type(label),
            "role_or_context": "",
            "evidence_text": label,
            "extraction_basis": "heuristic_explicit",
            "parser_source": "heuristic",
            "confidence": 0.50,
            "primary_for_matching": False,
        })
    return rows


def extract_heuristic_claims(fragment_id: str, text_for_extraction: str) -> List[Dict[str, Any]]:
    rows = []
    for sent in sentence_split(text_for_extraction):
        token_count = len(re.findall(r"\w+", sent))
        if token_count < 7:
            continue
        rows.append({
            "query_claim_id": stable_id([VERSION, fragment_id, sent], "qc"),
            "fragment_id": fragment_id,
            "claim_text": sent,
            "speaker": "",
            "target": "",
            "stance": "unknown",
            "evidence_text": sent,
            "extraction_basis": "heuristic_sentence_claim",
            "parser_source": "heuristic",
            "confidence": 0.40,
            "primary_for_matching": False,
        })
    return rows


def extract_heuristic_events(fragment_id: str, text_for_extraction: str) -> List[Dict[str, Any]]:
    rows = []
    for sent in sentence_split(text_for_extraction):
        key_words = canonical_key(sent).split()
        if not any(v in key_words or v in canonical_key(sent) for v in EVENT_VERBS):
            continue
        dates = DATE_RE.findall(sent)
        rows.append({
            "query_event_id": stable_id([VERSION, fragment_id, sent], "qev"),
            "fragment_id": fragment_id,
            "event_text": sent,
            "event_type": "heuristic_event",
            "location": "",
            "date_or_temporal_context": "; ".join(dates),
            "evidence_text": sent,
            "extraction_basis": "heuristic_event_verb",
            "parser_source": "heuristic",
            "confidence": 0.40,
            "primary_for_matching": False,
        })
    return rows


def extract_heuristic_relations(fragment_id: str, text_for_extraction: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    labels = [e["label"] for e in entities if e.get("label") and e.get("type") != "DOCUMENT_ID"]
    for sent in sentence_split(text_for_extraction):
        present = [label for label in labels if re.search(r"\b" + re.escape(label) + r"\b", sent)]
        if len(present) < 2:
            continue
        # Keep these explicitly low-confidence. Real relation extraction should come from OpenAI mode.
        for a, b in zip(present, present[1:]):
            rows.append({
                "query_relation_id": stable_id([VERSION, fragment_id, sent, a, b], "qr"),
                "fragment_id": fragment_id,
                "subject_entity": a,
                "predicate": "co_occurs_in_query_context",
                "object_entity": b,
                "qualifier": "candidate_relation_not_asserted_fact",
                "evidence_text": sent,
                "extraction_basis": "heuristic_cooccurrence",
                "parser_source": "heuristic",
                "confidence": 0.25,
                "primary_for_matching": False,
            })
    return rows


def infer_slot_type_from_context(before: str, after: str) -> str:
    context = canonical_key(before[-160:] + " " + after[:160])
    if any(w in context for w in ["ambassador", "minister", "general", "mr", "mrs", "dr", "shah", "secretary", "governor"]):
        return "PERSON_OR_POSITION"
    if any(w in context for w in ["city", "country", "province", "embassy in", "capital", "from", "to"]):
        return "LOCATION"
    if any(w in context for w in ["ministry", "department", "organization", "party", "council", "government"]):
        return "ORGANIZATION"
    return "UNKNOWN"


def iter_redaction_matches(text: str) -> List[Tuple[re.Match, str]]:
    """Return value/entity redaction matches only.

    Sentence redactions are intentionally excluded here and handled by
    iter_sentence_redaction_matches().  This keeps query_redaction_slots.csv
    backward-compatible for ordinary value redaction.
    """
    matches: List[Tuple[re.Match, str]] = []
    occupied: List[Tuple[int, int]] = []
    # Reserve sentence spans so generic fallback markers inside them never
    # become value slots.
    for m in SENTENCE_REDACTION_RE.finditer(text):
        occupied.append(m.span())
    for m in EXPLICIT_REDACTION_RE.finditer(text):
        matches.append((m, "explicit"))
        occupied.append(m.span())
    for m in SHORTHAND_REDACTION_RE.finditer(text):
        s = m.span()
        if not any(max(s[0], o[0]) < min(s[1], o[1]) for o in occupied):
            matches.append((m, "shorthand"))
            occupied.append(s)
    for m in GENERIC_REDACTION_RE.finditer(text):
        s = m.span()
        if not any(max(s[0], o[0]) < min(s[1], o[1]) for o in occupied):
            matches.append((m, "generic"))
    matches.sort(key=lambda x: x[0].start())
    return matches


def iter_sentence_redaction_matches(text: str) -> List[Tuple[re.Match, str]]:
    """Return sentence/span redaction matches.

    Accepted syntax:
      [REDACT_SENT:S1 type=CLAIM approx_tokens=10]
      [REDACT_SPAN:S1 type=ANY approx_tokens=12]
      [REDACT_SENT type=CLAIM]

    The last form is repaired to an S-numbered normalized marker.
    """
    matches = [(m, "sentence") for m in SENTENCE_REDACTION_RE.finditer(text)]
    matches.sort(key=lambda x: x[0].start())
    return matches


def _neighbor_sentence(text: str, start: int, end: int, side: str) -> str:
    """Return the nearest *meaningful* visible sentence around a marker.

    v0.3.19 used Python string truthiness and could therefore return ``"."`` as
    a right-hand sentence.  v0.3.20 delegates to the shared boundary normalizer
    so punctuation-only pieces are ignored deterministically.
    """
    normalized = normalize_sentence_slot_context(text, start, end)
    key = "normalized_sentence_before" if side == "before" else "normalized_sentence_after"
    return v19_clean_text(normalized.get(key, ""))


def _value_slot_signature_context(
    before: str,
    after: str,
    frame: Dict[str, Any],
) -> str:
    """Scope a value slot signature to its nearest governing clause."""
    clause_before = before
    # A value slot begins a new local event after a sentence or semicolon. Do
    # not let actors, places, quantities, or dates from the preceding event
    # become noncompensatory anchors for the missing value's event.
    clause_boundaries = list(re.finditer(r"[.;!?]\s*", clause_before))
    if clause_boundaries:
        clause_before = clause_before[clause_boundaries[-1].end():]
    # Coordinated analyst prompts often place the blank in a later event:
    # ``X proposed ..., while Y increased in [LOCATION]``.  Bind the slot to
    # that nearest explicit contrast clause rather than leaking the actor and
    # object from the completed event to its left.  Comma-less ``and`` is not
    # treated as a boundary because it frequently coordinates arguments within
    # one event rather than introducing a new event.
    contrast_boundaries = list(re.finditer(
        r"(?:,|;)\s*(?:while|whereas|but)\s+",
        clause_before,
        flags=re.I,
    ))
    if contrast_boundaries:
        clause_before = clause_before[contrast_boundaries[-1].end():]
    reporting_boundaries = list(re.finditer(
        r"\bthat\b|,\s*(?:claiming|reporting|stating|noting|saying)\b",
        clause_before,
        flags=re.I,
    ))
    if reporting_boundaries:
        clause_before = clause_before[reporting_boundaries[-1].end():]

    governing = clean_text(frame.get("governing_predicate", ""))
    slot_frame = clean_text(frame.get("slot_frame", ""))
    if (
        governing
        and slot_frame in {
            "entity_as_grammatical_subject",
            "organization_as_grammatical_subject",
        }
        and re.match(
            r"^\s*to\s+" + re.escape(governing) + r"\b",
            after,
            flags=re.I,
        )
    ):
        # The missing value starts an infinitival clause.  Material before the
        # slot belongs to the controlling event (``allowed X to conduct``), not
        # to X's action.  Parse the local proposition as ``SLOT conduct ...``.
        local_after = re.sub(r"^\s*to\s+", "", after, count=1, flags=re.I)
        return clean_text("SLOT " + local_after)
    if governing and slot_frame in {
        "travel_destination_slot",
        "counterparty_slot",
        "agentive_by_slot",
        "possessive_entity_slot",
    }:
        predicate_matches = list(re.finditer(
            r"\b" + re.escape(governing) + r"\b",
            clause_before,
            flags=re.I,
        ))
        if predicate_matches:
            prefix = clause_before[:predicate_matches[-1].start()]
            boundaries = list(re.finditer(
                r"(?:[.;:!?]\s*|,\s*|\b(?:that|when|while|after|before|"
                r"because|although|though|but|and)\b\s*)",
                prefix,
                flags=re.I,
            ))
            if boundaries:
                clause_before = clause_before[boundaries[-1].end():]
    return clean_text(clause_before + " SLOT " + after)

def extract_redaction_slots(fragments: pd.DataFrame) -> pd.DataFrame:
    """Extract ordinary value/entity redaction slots.

    v0.3.19 adds redaction format repair.  The matcher should read
    normalized_marker for display/audit, but slot_label and expected_type_hint
    remain the stable downstream fields.
    """
    rows = []
    global_index = 1
    for _, frag in fragments.iterrows():
        text = str(frag.get("fragment_text", ""))
        local_index = 1
        for match, kind in iter_redaction_matches(text):
            start, end = match.span()
            before = text[max(0, start - 220):start].strip()
            after = text[end:min(len(text), end + 220)].strip()
            explicit_label = ""
            explicit_type = ""
            repaired = False
            if kind == "explicit":
                explicit_label = clean_text(match.groupdict().get("slot_label", "")).upper()
                explicit_type = clean_text(match.groupdict().get("type", "")).upper()
                normalized_marker = f"[REDACT:{explicit_label} type={explicit_type}]" if explicit_type else f"[REDACT:{explicit_label}]"
            elif kind == "shorthand":
                explicit_type = clean_text(match.groupdict().get("type", "")).upper()
                repaired = True
                explicit_label = f"R{global_index}"
                normalized_marker = f"[REDACT:{explicit_label} type={explicit_type or 'UNKNOWN'}]"
            elif re.fullmatch(r"\[\s*R\d+\s*\]", match.group(0), flags=re.I):
                explicit_label = re.sub(r"[^R0-9]", "", match.group(0).upper())
                normalized_marker = f"[REDACT:{explicit_label}]"
            else:
                repaired = kind == "generic"
                explicit_label = f"R{global_index}"
                normalized_marker = f"[REDACT:{explicit_label} type=UNKNOWN]"
            slot_label = explicit_label or f"R{global_index}"
            # Infer a generic clause/relation frame from the visible syntax.
            # The frame is an auditable matcher constraint, not a hidden answer.
            frame = infer_value_slot_frame(before, after, explicit_type or infer_slot_type_from_context(before, after))
            slot_context = clean_text(before + " " + after)
            # A reporting preamble or an earlier sibling clause must not
            # become the event frame for the missing value.
            signature_context = _value_slot_signature_context(
                before, after, frame
            )
            slot_signature = build_event_signature(
                signature_context,
                predicate=frame.get("governing_predicate", ""),
            )
            slot_frame = clean_text(frame.get("slot_frame", ""))
            # A passive clause can parse the redaction marker itself as the
            # patient/object (for example, ``[LOCATION] was shelled``).  The
            # hidden value is the candidate being evaluated, not an observable
            # event anchor, so never serialize that placeholder as evidence.
            if canonical_key(slot_signature.get("object", "")) in {
                "slot", "redaction slot",
            }:
                for field in ("object", "object_surface", "patient_or_theme"):
                    slot_signature[field] = ""
            if (
                slot_frame == "possessive_entity_slot"
                and clean_text(slot_signature.get("role_normalization_status", ""))
                == "COPULAR_FALLBACK"
            ):
                # The copula in ``troops were in [PERSON]'s forces`` describes
                # membership, not an event governed by the missing possessor.
                for field in (
                    "predicate", "predicate_surface", "object",
                    "object_surface", "patient_or_theme",
                ):
                    slot_signature[field] = ""
            required_anchor_fields = [
                field for field in (
                    "predicate", "object", "location", "event_time",
                    "packet_count", "total_weight_grams",
                )
                if slot_signature.get(field)
            ]
            local_before_norm = clean_text(before).casefold()
            role_normalization_status = clean_text(
                slot_signature.get("role_normalization_status", "")
            )
            if slot_frame == "possessive_entity_slot":
                # The blank is the possessor. A fallback copula inferred from
                # the surrounding sentence is not its governing predicate.
                required_anchor_fields = [
                    field for field in required_anchor_fields
                    if field == "event_time"
                ]
            elif slot_frame in {
                "travel_destination_slot",
                "counterparty_slot",
                "agentive_by_slot",
            }:
                # These frames represent a missing value that is itself the
                # destination/counterparty/agentive filler.  Requiring the
                # generic clause object as a noncompensatory gate causes
                # exact-support candidates to be rejected even when the
                # governing predicate and local clause match cleanly.
                required_anchor_fields = [
                    field for field in required_anchor_fields
                    if field in {"predicate", "event_time"}
                ] or ["predicate"]
            if (
                slot_frame == "event_argument_slot"
                and explicit_type in {"LOCATION", "COUNTRY"}
            ):
                # The hidden value itself fills the event location. Requiring a
                # different visible location extracted from the query would
                # reject the correct candidate by construction.
                required_anchor_fields = [
                    field for field in required_anchor_fields
                    if field != "location"
                ]
            temporal_markers = {
                canonical_key(value)
                for value in re.findall(
                    r"\b(?:18|19|20)\d{2}\b|"
                    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
                    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
                    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b|"
                    r"\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
                    r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
                    r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                    signature_context,
                    flags=re.I,
                )
                if canonical_key(value)
            }
            if len(temporal_markers) > 1:
                # The scalar date comparator represents one event date. A
                # multi-date context is therefore useful retrieval evidence but
                # cannot be a single noncompensatory candidate gate.
                required_anchor_fields = [
                    field for field in required_anchor_fields
                    if field != "event_time"
                ]
            if (
                slot_frame == "generic_value_slot"
                and explicit_type in {"LOCATION", "COUNTRY"}
                and "object" in required_anchor_fields
                and re.search(
                    r"(?:"
                    r"(?:ambassador|representative|delegation|official|minister)\s+to|"
                    r"(?:companies|firm|factory|station|mission|office|units?)\s+in|"
                    r"(?:plans?|access|sources?)\s+to|"
                    r"\b(?:to|in|at|from|near|within|outside|inside|between|across|"
                    r"toward|towards|into|onto|through|via)\s*$"
                    r")",
                    local_before_norm,
                    flags=re.I,
                )
            ):
                # A location slot embedded in a prepositional shell is often
                # not the semantic object of the governing clause. Requiring
                # the heuristic fallback object as a noncompensatory anchor
                # turns historically clean recoveries into false negatives.
                # The local shell itself is the reliable cue here; some
                # fragments still produce a weak or empty clause-level role
                # normalization status even though the missing value is clearly
                # a destination/location adjunct rather than a clause object.
                required_anchor_fields = [
                    field for field in required_anchor_fields
                    if field in {"predicate", "event_time"}
                ] or ["predicate"]
            if (
                slot_frame == "generic_value_slot"
                and explicit_type in {"LOCATION", "COUNTRY"}
                and "object" in required_anchor_fields
                and clean_text(
                    frame.get("governing_predicate_normalized", "")
                    or frame.get("governing_predicate", "")
                ).casefold()
                in {"direct", "travel", "transport", "send", "sent", "move", "moved"}
                and re.search(
                    r"\b(?:to|into|toward|towards|from|in|at)\b"
                    r"[^.!?]{0,140}"
                    r"(?:,\s*[^.!?]{1,50}){1,8}"
                    r"\s*(?:and|or)(?:\s+the)?\s*$",
                    local_before_norm,
                    flags=re.I,
                )
            ):
                # Enumerated destination lists ending at the slot marker are
                # location-filler queries, not object-filler queries. Keeping
                # the fallback clause object as a required anchor overconstrains
                # otherwise correct recoveries such as destination-country
                # lists in procurement or travel reporting.
                required_anchor_fields = [
                    field for field in required_anchor_fields
                    if field in {"predicate", "event_time"}
                ] or ["predicate"]
            # A reporting-clause subject is scoped to the act of reporting;
            # dates and quantities in the embedded assertion are not required
            # anchors for the reporter slot.
            if canonical_predicate(frame.get("governing_predicate", "")) == "report":
                required_anchor_fields = ["predicate"]
            rows.append({
                "query_redaction_slot_id": stable_id([VERSION, frag.get("fragment_id"), global_index, slot_label, match.group(0), start], "qrs"),
                "fragment_id": frag.get("fragment_id"),
                "slot_index_global": global_index,
                "slot_index_in_fragment": local_index,
                "slot_label": slot_label,
                "slot_marker": match.group(0),
                "normalized_marker": normalized_marker,
                "marker_start": start,
                "marker_end": end,
                "explicit_type_hint": explicit_type,
                "expected_type_hint": explicit_type or infer_slot_type_from_context(before, after),
                "local_context_before": before,
                "local_context_after": after,
                **frame,
                "slot_clause_level": frame.get("clause_level", "main_clause"),
                "slot_event_signature_json": json.dumps(
                    slot_signature, sort_keys=True, ensure_ascii=False,
                    default=str,
                ),
                "required_anchor_fields_json": json.dumps(
                    required_anchor_fields
                ),
                "extraction_basis": "explicit_redaction_marker" if kind == "explicit" else "repaired_shorthand_redaction_marker" if kind == "shorthand" else "generic_redaction_marker",
                "parser_source": "heuristic",
                "confidence": 0.95 if explicit_type and kind == "explicit" else 0.85 if explicit_type else 0.65,
                "primary_for_matching": True,
                "redaction_format_repaired": bool(repaired),
            })
            local_index += 1
            global_index += 1
    return ensure_columns(pd.DataFrame(rows), REDACTION_SLOT_COLUMNS)


def extract_sentence_redaction_slots(fragments: pd.DataFrame) -> pd.DataFrame:
    """Extract sentence/span redaction slots with canonical boundary repair.

    The parser never invents the missing sentence.  It only produces a typed
    hole, meaningful neighboring context, a repair audit, and a stable context
    hash for controlled baseline/holdout comparisons.
    """
    rows: List[Dict[str, Any]] = []
    global_index = 1
    for _, frag in fragments.iterrows():
        text = str(frag.get("fragment_text", ""))
        local_index = 1
        for match, _kind in iter_sentence_redaction_matches(text):
            start, end = match.span()
            gd = match.groupdict()
            label = clean_text(gd.get("slot_label", "")).upper()
            repaired_marker = False
            if not label:
                label = f"S{global_index}"
                repaired_marker = True
            sem_type = clean_text(gd.get("type", "ANY")).upper() or "ANY"
            approx = clean_text(gd.get("approx_tokens", ""))
            approx_val: Any = int(approx) if approx.isdigit() else ""
            normalized_marker = f"[REDACT_SENT:{label} type={sem_type}" + (
                f" approx_tokens={approx_val}" if approx_val != "" else ""
            ) + "]"

            local_before = text[max(0, start - 500):start].strip()
            local_after = text[end:min(len(text), end + 500)].strip()
            boundary = normalize_sentence_slot_context(text, start, end)
            rows.append({
                "query_sentence_redaction_slot_id": stable_id(
                    [VERSION, frag.get("fragment_id"), global_index, label,
                     boundary.get("normalized_slot_context_sha256", "")], "qsrs"
                ),
                "fragment_id": frag.get("fragment_id"),
                "slot_index_global": global_index,
                "slot_index_in_fragment": local_index,
                "sentence_slot_label": label,
                "slot_marker": match.group(0),
                "normalized_marker": normalized_marker,
                "marker_start": start,
                "marker_end": end,
                "slot_kind": "sentence_redaction",
                "expected_semantic_type": sem_type,
                "approx_tokens": approx_val,
                "local_context_before": local_before,
                "local_context_after": local_after,
                "sentence_before": boundary.get("normalized_sentence_before", ""),
                "sentence_after": boundary.get("normalized_sentence_after", ""),
                **boundary,
                "paragraph_context": clean_text(text, 1600),
                "extraction_basis": (
                    "sentence_redaction_marker_with_boundary_repair"
                    if boundary.get("boundary_repair_applied")
                    else "explicit_sentence_redaction_marker"
                ),
                "parser_source": "heuristic_boundary_contract",
                "confidence": 0.98 if sem_type else 0.80,
                "primary_for_matching": True,
                "redaction_format_repaired": bool(repaired_marker),
            })
            local_index += 1
            global_index += 1
    return ensure_columns(pd.DataFrame(rows), SENTENCE_REDACTION_SLOT_COLUMNS)

def heuristic_parse(text: str, max_chars_per_fragment: int = 2200) -> Dict[str, pd.DataFrame]:
    fragments = split_query_into_fragments(text, max_chars_per_fragment=max_chars_per_fragment)
    entity_rows: List[Dict[str, Any]] = []
    claim_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []
    for _, frag in fragments.iterrows():
        fid = frag["fragment_id"]
        ftext = frag["fragment_text_for_extraction"]
        ents = extract_heuristic_entities(fid, ftext)
        entity_rows.extend(ents)
        claim_rows.extend(extract_heuristic_claims(fid, ftext))
        event_rows.extend(extract_heuristic_events(fid, ftext))
        relation_rows.extend(extract_heuristic_relations(fid, ftext, ents))
    return {
        "fragments": fragments,
        "entities": ensure_columns(pd.DataFrame(entity_rows), ENTITY_COLUMNS),
        "claims": ensure_columns(pd.DataFrame(claim_rows), CLAIM_COLUMNS),
        "events": ensure_columns(pd.DataFrame(event_rows), EVENT_COLUMNS),
        "relations": ensure_columns(pd.DataFrame(relation_rows), RELATION_COLUMNS),
        "redaction_slots": extract_redaction_slots(fragments),
        "sentence_redaction_slots": extract_sentence_redaction_slots(fragments),
    }


def openai_fragment_prompt(fragment_id: str, fragment_text_for_extraction: str, redaction_slots: List[Dict[str, Any]]) -> str:
    """Prompt for per-fragment semantic extraction.

    DID markers and marker internals have already been removed from the text.
    The redaction_slots list is provided separately so the model can use context
    without treating R1/type/PERSON as normal entities.
    """
    return f"""
You are extracting conservative semantic evidence from one query fragment of a historical diplomatic document.
The fragment may be paraphrased, redacted, incomplete, or stitched from another document.

Rules:
- Do NOT fill redactions.
- Do NOT invent names, dates, or relations that are not supported by the fragment.
- Separate explicit evidence from inferred/uncertain evidence.
- Extract only meaningful entities, claims, events, and relations.
- Redaction slot metadata is provided separately; do not treat slot labels like R1 or type words like PERSON as entities.
- Do not return redaction markers/placeholders such as [REDACT:R1 type=PERSON] as entities.
- Return ONLY valid JSON. No markdown.

JSON schema:
{{
  "entities": [{{"label":"...", "type":"PERSON|ORGANIZATION|LOCATION|COUNTRY|POSITION|DOCUMENT|EVENT_TOPIC|OTHER", "role_or_context":"...", "evidence_text":"...", "extraction_basis":"explicit|inferred|uncertain", "confidence":0.0}}],
  "claims": [{{"claim_text":"...", "speaker":"", "target":"", "stance":"affirmed|denied|uncertain|unknown", "evidence_text":"...", "extraction_basis":"explicit|inferred|uncertain", "confidence":0.0}}],
  "events": [{{"event_text":"...", "event_type":"...", "location":"", "date_or_temporal_context":"", "evidence_text":"...", "extraction_basis":"explicit|inferred|uncertain", "confidence":0.0}}],
  "relations": [{{"subject_entity":"...", "predicate":"...", "object_entity":"...", "qualifier":"", "evidence_text":"...", "extraction_basis":"explicit|inferred|uncertain", "confidence":0.0}}]
}}

fragment_id: {fragment_id}
redaction_slots_json: {json.dumps(redaction_slots, ensure_ascii=False)}
fragment_text:
{fragment_text_for_extraction}
""".strip()


def clean_json_from_llm(content: str) -> Dict[str, Any]:
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content, flags=re.I).strip()
        content = re.sub(r"```$", "", content).strip()
    return json.loads(content)


def get_openai_client(api_key_env: str = "OPENAI_API_KEY") -> Tuple[Any, Optional[str]]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return None, f"Environment variable {api_key_env} is not set."
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=api_key), None
    except Exception as exc:
        return None, f"Could not import/use openai package: {exc}"


def parse_fragment_with_openai(
    client: Any,
    fragment_id: str,
    fragment_text_for_extraction: str,
    redaction_slots: List[Dict[str, Any]],
    model: str,
    timeout_seconds: float = 120.0,
    max_retries: int = 1,
) -> Tuple[Dict[str, pd.DataFrame], Optional[str]]:
    prompt = openai_fragment_prompt(fragment_id, fragment_text_for_extraction, redaction_slots)
    obj: Dict[str, Any] = {}
    last_error = ""
    for attempt in range(1, max(0, int(max_retries)) + 2):
        api_started = time.perf_counter()
        outcome = "FAILED"
        rate_limited = False
        try:
            # Chat Completions is used for compatibility with supported SDKs.
            request_kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative information extraction "
                            "system. Output valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "timeout": max(5.0, float(timeout_seconds)),
            }
            if not str(model).startswith("gpt-5"):
                request_kwargs["temperature"] = 0
            try:
                response = client.chat.completions.create(
                    **request_kwargs
                )
            except TypeError:
                request_kwargs.pop("timeout", None)
                response = client.chat.completions.create(
                    **request_kwargs
                )
            content = response.choices[0].message.content or "{}"
            obj = clean_json_from_llm(content)
            outcome = "SUCCESS"
        except Exception as exc:
            last_error = clean_text(str(exc))[:500]
            lower = (
                type(exc).__name__ + " " + last_error
            ).lower()
            rate_limited = (
                "rate" in lower and "limit" in lower
            ) or "429" in lower
            outcome = (
                "RATE_LIMITED"
                if rate_limited
                else "TIMEOUT"
                if "timeout" in lower
                else "FAILED"
            )
        retry_scheduled = (
            outcome != "SUCCESS"
            and attempt <= max(0, int(max_retries))
        )
        _OPENAI_TELEMETRY_ROWS.append({
            "fragment_id": fragment_id,
            "model": model,
            "attempt_number": attempt,
            "outcome": outcome,
            "api_elapsed_seconds":
                time.perf_counter() - api_started,
            "timeout_seconds": max(5.0, float(timeout_seconds)),
            "retry_scheduled": retry_scheduled,
            "rate_limit_observed": rate_limited,
            "deterministic_fallback_state": (
                "NOT_REQUIRED"
                if outcome == "SUCCESS"
                else "PENDING_AFTER_RETRYABLE_FAILURE"
                if retry_scheduled
                else "FALLBACK_REQUIRED"
            ),
        })
        if outcome == "SUCCESS":
            break
    if not obj:
        return {}, (
            f"OpenAI parse failed for {fragment_id}: "
            + (last_error or "empty response")
        )

    outputs: Dict[str, List[Dict[str, Any]]] = {"entities": [], "claims": [], "events": [], "relations": []}

    for item in obj.get("entities", []) or []:
        label = clean_text(item.get("label"))
        if is_bad_entity_label(label):
            continue
        outputs["entities"].append({
            "query_entity_id": stable_id([VERSION, "openai", fragment_id, label, item.get("type"), item.get("evidence_text")], "qe"),
            "fragment_id": fragment_id,
            "label": label,
            "canonical_key": canonical_key(label),
            "type": clean_text(item.get("type")) or infer_entity_type(label),
            "role_or_context": clean_text(item.get("role_or_context")),
            "evidence_text": clean_text(item.get("evidence_text")) or label,
            "extraction_basis": clean_text(item.get("extraction_basis")) or "openai_unspecified",
            "parser_source": "openai",
            "confidence": item.get("confidence", 0.70),
            "primary_for_matching": True,
        })

    for item in obj.get("claims", []) or []:
        claim = clean_text(item.get("claim_text"))
        if not claim:
            continue
        outputs["claims"].append({
            "query_claim_id": stable_id([VERSION, "openai", fragment_id, claim, item.get("speaker"), item.get("target")], "qc"),
            "fragment_id": fragment_id,
            "claim_text": claim,
            "speaker": clean_text(item.get("speaker")),
            "target": clean_text(item.get("target")),
            "stance": clean_text(item.get("stance")) or "unknown",
            "evidence_text": clean_text(item.get("evidence_text")) or claim,
            "extraction_basis": clean_text(item.get("extraction_basis")) or "openai_unspecified",
            "parser_source": "openai",
            "confidence": item.get("confidence", 0.65),
            "primary_for_matching": True,
        })

    for item in obj.get("events", []) or []:
        ev = clean_text(item.get("event_text"))
        if not ev:
            continue
        outputs["events"].append({
            "query_event_id": stable_id([VERSION, "openai", fragment_id, ev, item.get("event_type")], "qev"),
            "fragment_id": fragment_id,
            "event_text": ev,
            "event_type": clean_text(item.get("event_type")) or "event",
            "location": clean_text(item.get("location")),
            "date_or_temporal_context": clean_text(item.get("date_or_temporal_context")),
            "evidence_text": clean_text(item.get("evidence_text")) or ev,
            "extraction_basis": clean_text(item.get("extraction_basis")) or "openai_unspecified",
            "parser_source": "openai",
            "confidence": item.get("confidence", 0.65),
            "primary_for_matching": True,
        })

    for item in obj.get("relations", []) or []:
        subj = clean_text(item.get("subject_entity"))
        pred = clean_text(item.get("predicate"))
        obj_ent = clean_text(item.get("object_entity"))
        if not (subj and pred and obj_ent):
            continue
        outputs["relations"].append({
            "query_relation_id": stable_id([VERSION, "openai", fragment_id, subj, pred, obj_ent, item.get("evidence_text")], "qr"),
            "fragment_id": fragment_id,
            "subject_entity": subj,
            "predicate": pred,
            "object_entity": obj_ent,
            "qualifier": clean_text(item.get("qualifier")),
            "evidence_text": clean_text(item.get("evidence_text")),
            "extraction_basis": clean_text(item.get("extraction_basis")) or "openai_unspecified",
            "parser_source": "openai",
            "confidence": item.get("confidence", 0.60),
            "primary_for_matching": True,
        })

    return {
        "entities": ensure_columns(pd.DataFrame(outputs["entities"]), ENTITY_COLUMNS),
        "claims": ensure_columns(pd.DataFrame(outputs["claims"]), CLAIM_COLUMNS),
        "events": ensure_columns(pd.DataFrame(outputs["events"]), EVENT_COLUMNS),
        "relations": ensure_columns(pd.DataFrame(outputs["relations"]), RELATION_COLUMNS),
    }, None


def parse_with_openai_per_fragment(
    heuristic: Dict[str, pd.DataFrame],
    model: str,
    api_key_env: str = "OPENAI_API_KEY",
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    _OPENAI_TELEMETRY_ROWS.clear()
    warnings: List[str] = []
    client, err = get_openai_client(api_key_env)
    if client is None:
        fragments = heuristic.get("fragments", pd.DataFrame())
        for _, fragment in fragments.iterrows():
            _OPENAI_TELEMETRY_ROWS.append({
                "fragment_id": clean_text(
                    fragment.get("fragment_id", "")
                ),
                "model": model,
                "attempt_number": 0,
                "outcome": "CREDENTIAL_OR_CLIENT_UNAVAILABLE",
                "api_elapsed_seconds": 0.0,
                "timeout_seconds": 120.0,
                "retry_scheduled": False,
                "rate_limit_observed": False,
                "deterministic_fallback_state": "FALLBACK_REQUIRED",
            })
        return {}, [err or "OpenAI client unavailable."]

    fragments = heuristic.get("fragments", pd.DataFrame())
    slots = heuristic.get("redaction_slots", pd.DataFrame())
    all_rows: Dict[str, List[pd.DataFrame]] = {"entities": [], "claims": [], "events": [], "relations": []}

    for _, frag in fragments.iterrows():
        fid = str(frag.get("fragment_id"))
        ftext = str(frag.get("fragment_text_for_extraction", ""))
        frag_slots: List[Dict[str, Any]] = []
        if slots is not None and not slots.empty and "fragment_id" in slots.columns:
            frag_slots = slots[slots["fragment_id"].astype(str) == fid].to_dict(orient="records")
        parsed, err = parse_fragment_with_openai(client, fid, ftext, frag_slots, model)
        if err:
            warnings.append(err)
            continue
        for key in all_rows:
            if key in parsed and not parsed[key].empty:
                all_rows[key].append(parsed[key])

    return {
        key: ensure_columns(pd.concat(frames, ignore_index=True, sort=False),
                            {"entities": ENTITY_COLUMNS, "claims": CLAIM_COLUMNS, "events": EVENT_COLUMNS, "relations": RELATION_COLUMNS}[key])
        if frames else ensure_columns(pd.DataFrame(), {"entities": ENTITY_COLUMNS, "claims": CLAIM_COLUMNS, "events": EVENT_COLUMNS, "relations": RELATION_COLUMNS}[key])
        for key, frames in all_rows.items()
    }, warnings


def dedupe_rows(df: pd.DataFrame, id_col: str, semantic_cols: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in semantic_cols:
        if col not in out.columns:
            out[col] = ""
    out["_dedupe_key"] = out[list(semantic_cols)].astype(str).agg("||".join, axis=1).map(canonical_key)
    # Prefer OpenAI rows over heuristic rows when semantic key collides.
    if "parser_source" in out.columns:
        out["_source_priority"] = out["parser_source"].map(lambda x: 0 if str(x) == "openai" else 1)
        out = out.sort_values(["_dedupe_key", "_source_priority"])
    out = out.drop_duplicates("_dedupe_key", keep="first").drop(columns=[c for c in ["_dedupe_key", "_source_priority"] if c in out.columns])
    # Rebuild IDs if a dedupe/concat created duplicates.
    seen: Dict[str, int] = {}
    ids: List[str] = []
    for i, row in out.reset_index(drop=True).iterrows():
        base = clean_text(row.get(id_col)) or stable_id([VERSION, id_col, i], id_col[:2] or "id")
        n = seen.get(base, 0)
        seen[base] = n + 1
        ids.append(base if n == 0 else stable_id([base, n, i], id_col[:2] or "id"))
    out[id_col] = ids
    return out.reset_index(drop=True)



def set_primary_flag(df: pd.DataFrame, value: bool) -> pd.DataFrame:
    """Return a copy with primary_for_matching set consistently.

    v0.3.16 uses this flag as the contract with the future semantic matcher:
    downstream code should normally read only rows where primary_for_matching is
    true.  Debug rows are still written, but they should not drive candidate
    scoring unless explicitly requested.
    """
    out = ensure_columns(df, df.columns.tolist() if df is not None and not df.empty else []) if df is not None else pd.DataFrame()
    out = out.copy()
    out["primary_for_matching"] = bool(value)
    return out


def add_fallback_basis(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    """Annotate heuristic rows that were promoted because OpenAI had no row.

    We do not hide the fact that these rows are heuristic; instead we mark their
    extraction_basis so the analyst and matcher can tell why they entered the
    primary table.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "extraction_basis" not in out.columns:
        out["extraction_basis"] = ""
    out["extraction_basis"] = out["extraction_basis"].map(clean_text).map(lambda x: f"{x};{reason}" if x else reason)
    return out


def _primary_flag(value: Any) -> bool:
    """Interpret primary_for_matching consistently across bool/string CSV rows."""
    if pd.isna(value):
        return True
    if isinstance(value, bool):
        return value
    key = canonical_key(value)
    if key in {"", "1", "true", "t", "yes", "y"}:
        return True
    if key in {"0", "false", "f", "no", "n"}:
        return False
    return bool(value)


def keep_primary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows explicitly marked primary for matching."""
    if df is None or df.empty or "primary_for_matching" not in df.columns:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    mask = out["primary_for_matching"].map(_primary_flag)
    return out.loc[mask].reset_index(drop=True)


def suppress_redundant_primary_relations(df: pd.DataFrame) -> pd.DataFrame:
    """Demote short same-fragment relation subspans when a fuller relation exists.

    OpenAI sometimes emits both the full asserted relation and a shorter
    qualifier-bearing prefix from the same fragment. The shorter row is useful
    for audit, but if it becomes a separate proposition it can create an extra
    benchmark target with only background-level evidence. We therefore retain
    the row while marking it non-primary for matcher construction.
    """
    if df is None or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = ensure_columns(df, RELATION_COLUMNS).copy()
    if "primary_for_matching" not in out.columns or len(out) < 2:
        return out

    primary_indices = [
        idx for idx, value in out["primary_for_matching"].items()
        if _primary_flag(value)
    ]
    if len(primary_indices) < 2:
        return out

    metadata: Dict[int, Dict[str, Any]] = {}
    for idx in primary_indices:
        row = out.loc[idx]
        evidence_text = clean_text(row.get("evidence_text", "")) or " ".join(
            part for part in [
                clean_text(row.get("subject_entity", "")),
                clean_text(row.get("predicate", "")),
                clean_text(row.get("object_entity", "")),
            ]
            if part
        )
        metadata[idx] = {
            "fragment_id": clean_text(row.get("fragment_id", "")),
            "evidence_text": evidence_text,
            "evidence_key": canonical_key(evidence_text),
            "evidence_tokens": set(semantic_tokens(evidence_text)),
            "anchor_tokens": set(
                semantic_tokens(
                    " ".join(
                        part
                        for part in [
                            clean_text(row.get("subject_entity", "")),
                            clean_text(row.get("object_entity", "")),
                            clean_text(row.get("qualifier", "")),
                        ]
                        if part
                    )
                )
            ),
            "qualifier": clean_text(row.get("qualifier", "")),
            "subject_key": canonical_key(row.get("subject_entity", "")),
            "predicate_key": canonical_key(row.get("predicate", "")),
            "object_key": canonical_key(row.get("object_entity", "")),
        }

    demoted: Set[int] = set()
    for short_idx in primary_indices:
        short_meta = metadata.get(short_idx, {})
        if not short_meta.get("fragment_id") or not short_meta.get("evidence_key"):
            continue
        for long_idx in primary_indices:
            if short_idx == long_idx:
                continue
            long_meta = metadata.get(long_idx, {})
            if short_meta["fragment_id"] != long_meta.get("fragment_id"):
                continue
            if not long_meta.get("evidence_key"):
                continue
            if len(long_meta["evidence_tokens"]) < len(short_meta["evidence_tokens"]) + 3:
                continue
            if short_meta["evidence_key"] == long_meta["evidence_key"]:
                continue
            strict_subspan = short_meta["evidence_key"] in long_meta["evidence_key"]
            lexical_subset = bool(short_meta["anchor_tokens"]) and short_meta["anchor_tokens"] <= long_meta["evidence_tokens"]
            shared_endpoint = (
                short_meta["subject_key"]
                and short_meta["subject_key"] in long_meta["evidence_key"]
            ) or (
                short_meta["object_key"]
                and short_meta["object_key"] in long_meta["evidence_key"]
            )
            infinitive_tail = bool(
                re.match(
                    r"^(?:to|in order to)\s+[a-z]",
                    short_meta["evidence_text"],
                    flags=re.I,
                )
            )
            if infinitive_tail:
                same_subject = bool(
                    short_meta["subject_key"]
                    and short_meta["subject_key"] == long_meta.get("subject_key", "")
                )
                longer_noninfinitive = not re.match(
                    r"^(?:to|in order to)\s+[a-z]",
                    long_meta["evidence_text"],
                    flags=re.I,
                )
                if same_subject and longer_noninfinitive:
                    demoted.add(short_idx)
                    break
            if not ((strict_subspan or lexical_subset) and shared_endpoint):
                continue
            if not short_meta["qualifier"] and len(short_meta["anchor_tokens"]) >= 4:
                continue
            demoted.add(short_idx)
            break

    if not demoted:
        return out

    out.loc[list(demoted), "primary_for_matching"] = False
    if "extraction_basis" not in out.columns:
        out["extraction_basis"] = ""
    out.loc[list(demoted), "extraction_basis"] = out.loc[list(demoted), "extraction_basis"].map(clean_text).map(
        lambda value: f"{value};suppressed_redundant_relation_subspan" if value else "suppressed_redundant_relation_subspan"
    )
    return out


def select_primary_semantic_rows(
    key: str,
    fragments: pd.DataFrame,
    openai_df: pd.DataFrame,
    heuristic_df: pd.DataFrame,
    heuristic_policy: str,
    id_col: str,
    semantic_cols: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    """Select rows for a primary semantic table under the requested policy.

    Policies:
    - fallback_only (recommended): for each fragment/table, use OpenAI rows when
      they exist. Use heuristic rows only for that fragment/table when OpenAI
      returned zero rows. This keeps primary tables clean while preserving some
      recall for fragments the LLM did not parse.
    - openai_only: use OpenAI rows only. This is cleanest but can lose recall.
    - merge: reproduce v0.3.14-style behavior by including both OpenAI and
      heuristic rows in the primary CSV. OpenAI rows are flagged primary=True;
      heuristic rows are flagged primary=False. This is kept only for backward
      compatibility/debugging.
    """
    openai_df = ensure_columns(openai_df, columns)
    heuristic_df = ensure_columns(heuristic_df, columns)

    if heuristic_policy == "openai_only":
        return ensure_columns(dedupe_rows(set_primary_flag(openai_df, True), id_col, semantic_cols), columns)

    if heuristic_policy == "merge":
        frames = []
        if not openai_df.empty:
            frames.append(set_primary_flag(openai_df, True))
        if not heuristic_df.empty:
            frames.append(set_primary_flag(heuristic_df, False))
        merged = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        return ensure_columns(dedupe_rows(merged, id_col, semantic_cols), columns)

    if heuristic_policy != "fallback_only":
        raise ValueError(f"Unknown heuristic_policy: {heuristic_policy}")

    frames: List[pd.DataFrame] = []
    fragment_ids = fragments["fragment_id"].astype(str).tolist() if fragments is not None and not fragments.empty else []
    for fid in fragment_ids:
        o = openai_df[openai_df["fragment_id"].astype(str) == fid] if "fragment_id" in openai_df.columns else pd.DataFrame()
        h = heuristic_df[heuristic_df["fragment_id"].astype(str) == fid] if "fragment_id" in heuristic_df.columns else pd.DataFrame()
        if len(o):
            frames.append(set_primary_flag(o, True))
        elif len(h):
            # Heuristic rows are primary only because no OpenAI row exists for
            # this fragment/table. The annotation lets later stages downweight
            # or review these rows separately.
            frames.append(set_primary_flag(add_fallback_basis(h, "fallback_no_openai_rows_for_fragment_table"), True))
    selected = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return ensure_columns(dedupe_rows(selected, id_col, semantic_cols), columns)


def infer_polarity_from_text(text: str, stance: str = "") -> str:
    """Infer a conservative polarity label for claim/contradiction matching.

    This is not a theorem prover.  It is a corpus-agnostic normalization layer
    that lets downstream code group propositions into positive/negative/
    uncertain buckets for candidate contradiction discovery.
    """
    joined = f"{stance} {text}".strip()
    stance_key = canonical_key(stance)
    if stance_key in {"negative", "neg", "denied", "deny", "denies", "false", "opposes", "opposed", "rejected"}:
        return "negative"
    if stance_key in {"uncertain", "unknown", "possible", "possible_or_uncertain", "rumor", "rumour"}:
        return "uncertain"
    if NEGATION_RE.search(joined):
        return "negative"
    if UNCERTAIN_RE.search(joined):
        return "uncertain"
    if stance_key in {"positive", "affirmed", "affirm", "true", "support", "supports", "supported"}:
        return "positive"
    return "positive" if clean_text(text) else "uncertain"


def infer_modality_from_text(text: str, extraction_basis: str = "", stance: str = "") -> str:
    """Return explicit / inferred / uncertain / heuristic modality."""
    basis_key = canonical_key(extraction_basis)
    stance_key = canonical_key(stance)
    joined = f"{text} {extraction_basis} {stance}".strip()
    if "heuristic" in basis_key:
        return "heuristic"
    if "uncertain" in basis_key or stance_key in {"uncertain", "unknown"} or UNCERTAIN_RE.search(joined):
        return "uncertain"
    if "inferred" in basis_key or "infer" in basis_key:
        return "inferred"
    if "explicit" in basis_key:
        return "explicit"
    return "unspecified"


def normalized_relation_text(subject: str, predicate: str, obj: str) -> str:
    """Canonical relation signature for matching query relations to corpus relations."""
    return " | ".join([canonical_key(subject), canonical_key(predicate), canonical_key(obj)]).strip(" |")


def normalized_claim_text(claim: str) -> str:
    """Canonical claim-text signature used when no SPO parse is available."""
    return canonical_key(claim)


def build_query_propositions(
    claims: pd.DataFrame,
    relations: pd.DataFrame,
    events: Optional[pd.DataFrame] = None,
    entities: Optional[pd.DataFrame] = None,
    fragments: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build canonical propositions for claims and contradiction tasks.

    v0.3.20 permits audited event-to-proposition projection when a declarative
    query has no native claim.  v0.4.10 also permits a conservative,
    deterministic fragment-to-proposition fallback when a short declarative
    clause has no claim, asserted relation, or event row.  It writes semantic
    number/date/state fields so the matcher does not have to reverse-engineer
    them differently per task.
    """
    rows: List[Dict[str, Any]] = []
    claims = keep_primary_rows(claims if isinstance(claims, pd.DataFrame) else pd.DataFrame())
    relations = keep_primary_rows(relations if isinstance(relations, pd.DataFrame) else pd.DataFrame())
    events = keep_primary_rows(events if isinstance(events, pd.DataFrame) else pd.DataFrame())
    entities = entities if isinstance(entities, pd.DataFrame) else pd.DataFrame()
    fragments = fragments if isinstance(fragments, pd.DataFrame) else pd.DataFrame()

    fragment_text_map: Dict[str, str] = {}
    if len(fragments):
        for _, row in fragments.iterrows():
            fragment_text_map[clean_text(row.get("fragment_id", ""))] = clean_text(
                row.get("fragment_text_for_extraction", row.get("fragment_text", ""))
            )

    entity_map: Dict[str, List[Dict[str, Any]]] = {}
    if len(entities):
        for fid, sub in entities.groupby(entities.get("fragment_id", pd.Series(dtype=str)).astype(str), dropna=False):
            entity_map[clean_text(fid)] = sub.to_dict("records")

    event_map: Dict[str, List[Dict[str, Any]]] = {}
    if len(events):
        for _, event in events.iterrows():
            fid = clean_text(event.get("fragment_id", ""))
            event_text = clean_text(event.get("event_text", ""))
            evidence_text = clean_text(event.get("evidence_text", ""))
            event_map.setdefault(fid, []).append({
                "source_row_id": clean_text(
                    event.get("query_event_id", "")
                ),
                "event_text": event_text,
                "evidence_text": evidence_text,
                "event_type": clean_text(event.get("event_type", "")),
                "location": clean_text(event.get("location", "")),
                "event_time": clean_text(
                    event.get("date_or_temporal_context", "")
                ),
                "tokens": set(
                    semantic_tokens(" ".join(
                        value for value in (event_text, evidence_text)
                        if value
                    ))
                ),
            })

    def compatible_event_hint(
        fid: str,
        proposition_text: str,
        predicate: str,
    ) -> Dict[str, Any]:
        """Return one unambiguous event row that overlaps this proposition.

        OpenAI commonly emits a concise claim and a separate event row carrying
        its date or location.  The prior parser ignored the event row whenever
        a claim existed.  This routine performs a conservative cross-table join:
        it requires substantial lexical/event overlap, rejects reporting-clause
        inheritance from an unrelated embedded event, and refuses ambiguous
        conflicting values.
        """
        proposition_tokens = set(semantic_tokens(proposition_text))
        if not proposition_tokens:
            return {}
        predicate_key = canonical_predicate(predicate)
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for event in event_map.get(fid, []):
            event_tokens = set(event.get("tokens", set()))
            if not event_tokens:
                continue
            overlap_count = len(proposition_tokens & event_tokens)
            overlap = overlap_count / max(
                1, min(len(proposition_tokens), len(event_tokens))
            )
            event_predicate = canonical_predicate(event.get("event_type", ""))
            if (
                predicate_key in {"report", "inform"}
                and event_predicate
                and event_predicate not in {"report", "inform"}
            ):
                continue
            event_key = canonical_key(event.get("event_text", ""))
            proposition_key = canonical_key(proposition_text)
            phrase_match = bool(
                event_key
                and (
                    event_key in proposition_key
                    or all(
                        token in proposition_tokens
                        for token in semantic_tokens(
                            event.get("event_text", "")
                        )
                    )
                )
            )
            if not phrase_match and not (
                overlap_count >= 2 and overlap >= 0.50
            ):
                continue
            score = overlap + (0.25 if phrase_match else 0.0)
            candidates.append((score, event))
        if not candidates:
            return {}
        candidates.sort(
            key=lambda item: (
                -item[0],
                clean_text(item[1].get("source_row_id", "")),
            )
        )
        best_score, best = candidates[0]
        tied = [
            event for score, event in candidates
            if abs(score - best_score) < 1e-9
        ]
        for field in ("event_time", "location"):
            values = {
                canonical_key(event.get(field, ""))
                for event in tied
                if clean_text(event.get(field, ""))
            }
            if len(values) > 1:
                return {}
        return best

    def infer_actor(fid: str, proposition_text: str) -> str:
        # Prefer explicitly typed organizations/persons that occur in the text.
        candidates = entity_map.get(fid, [])
        lowered = proposition_text.lower()
        for item in candidates:
            label = clean_text(item.get("label", ""))
            typ = clean_text(item.get("type", "")).upper()
            if label and label.lower() in lowered and typ in {"ORGANIZATION", "ORG", "PERSON"}:
                return label
        return ""

    def is_generic_discourse_actor(value: str) -> bool:
        actor_key = canonical_key(value)
        if not actor_key:
            return False
        head = actor_key.split()[0]
        return head in {
            "announcement",
            "announcements",
            "statement",
            "statements",
            "report",
            "reports",
            "source",
            "sources",
            "comment",
            "comments",
            "message",
            "messages",
            "telegram",
            "telegrams",
        }

    def append_row(
        fid: str,
        source_kind: str,
        source_row_id: Any,
        proposition_text: str,
        subject: str = "",
        predicate: str = "",
        obj: str = "",
        location: str = "",
        event_time: str = "",
        stance: str = "",
        modality: str = "",
        evidence_text: str = "",
        extraction_basis: str = "",
        parser_source: str = "",
        confidence: Any = "",
        claim_speaker: str = "",
        claim_target: str = "",
    ) -> None:
        proposition_text = clean_text(proposition_text)
        if not proposition_text:
            return
        raw_subject = clean_text(subject)
        raw_predicate_surface = clean_text(predicate)
        raw_object_surface = clean_text(obj)
        evidence_surface = clean_text(evidence_text)
        inferred_clause = infer_simple_clause_arguments(proposition_text)
        supplemental_clause = (
            infer_simple_clause_arguments(evidence_surface)
            if evidence_surface
            and canonical_key(evidence_surface) != canonical_key(proposition_text)
            else {}
        )
        suppress_entity_subject_backfill = bool(
            source_kind == "claim"
            and clean_text(
                inferred_clause.get("role_normalization_status", "")
            ) == "NOMINAL_CLAIM_FALLBACK"
            and not clean_text(inferred_clause.get("actor", ""))
        )
        subject = (
            raw_subject
            or clean_text(inferred_clause.get("actor", ""))
            or clean_text(supplemental_clause.get("actor", ""))
            or (
                ""
                if suppress_entity_subject_backfill
                else infer_actor(fid, proposition_text)
            )
        )
        if (
            source_kind == "claim"
            and not subject
            and clean_text(claim_speaker)
            and not clean_text(supplemental_clause.get("actor", ""))
        ):
            subject = clean_text(claim_speaker)
        predicate_surface = (
            raw_predicate_surface
            or clean_text(inferred_clause.get("predicate", ""))
            or clean_text(supplemental_clause.get("predicate", ""))
        )
        object_surface = (
            raw_object_surface
            or clean_text(inferred_clause.get("object", ""))
            or clean_text(supplemental_clause.get("object", ""))
        )
        if (
            source_kind == "claim"
            and not object_surface
            and clean_text(claim_target)
        ):
            object_surface = clean_text(claim_target)
        if source_kind == "claim":
            weak_primary_predicate = clean_text(
                inferred_clause.get("predicate", "")
            ) in {"", "be"}
            strong_supplemental_predicate = clean_text(
                supplemental_clause.get("predicate", "")
            ) not in {"", "be"}
            if strong_supplemental_predicate and weak_primary_predicate:
                predicate_surface = clean_text(
                    supplemental_clause.get("predicate", "")
                )
                if clean_text(supplemental_clause.get("object", "")):
                    object_surface = clean_text(
                        supplemental_clause.get("object", "")
                    )
                if clean_text(supplemental_clause.get("actor", "")):
                    subject = clean_text(
                        supplemental_clause.get("actor", "")
                    )
        if source_kind == "relation":
            inferred_actor = clean_text(inferred_clause.get("actor", ""))
            inferred_predicate = clean_text(
                inferred_clause.get("predicate", "")
            )
            inferred_object = clean_text(inferred_clause.get("object", ""))
            suspicious_raw_relation = bool(
                (" " in raw_predicate_surface.strip())
                or (
                    raw_subject
                    and raw_object_surface
                    and canonical_key(raw_subject)
                    in canonical_key(raw_object_surface)
                )
                or (
                    raw_subject
                    and inferred_actor
                    and canonical_key(raw_subject)
                    != canonical_key(inferred_actor)
                )
            )
            inferred_strength = sum(
                bool(clean_text(value))
                for value in (
                    inferred_actor,
                    inferred_predicate,
                    inferred_object,
                )
            )
            if suspicious_raw_relation and inferred_strength >= 2:
                subject = inferred_actor or subject
                predicate_surface = (
                    inferred_predicate or predicate_surface
                )
                object_surface = inferred_object or object_surface
        strong_clause = next(
            (
                clause for clause in (inferred_clause, supplemental_clause)
                if clean_text(
                    clause.get("role_normalization_status", "")
                ) in STRONG_CLAUSE_NORMALIZATION_STATUSES
            ),
            {},
        )
        if strong_clause:
            # These narrow grammatical frames are more trustworthy than a
            # generic LLM event label such as ``security measure`` or a
            # relation that drops the embedded confirmation target.  This is
            # symmetric: the same normalizer builds corpus proposition
            # signatures, so no benchmark answer or document ID is involved.
            subject = clean_text(strong_clause.get("actor", ""))
            predicate_surface = clean_text(
                strong_clause.get("predicate", "")
            ) or predicate_surface
            object_surface = clean_text(
                strong_clause.get("object", "")
            ) or object_surface
            location = clean_text(
                strong_clause.get("location", "")
            )
        if (
            source_kind == "claim"
            and not clean_text(claim_speaker)
            and is_generic_discourse_actor(subject)
        ):
            subject = ""
        predicate = canonical_predicate(predicate_surface)
        obj = canonical_object_head(object_surface)
        location = (
            clean_text(location)
            or clean_text(inferred_clause.get("location", ""))
            or clean_text(supplemental_clause.get("location", ""))
        )
        event_time = clean_text(event_time)
        enrichment: Dict[str, Any] = {}
        event_hint = compatible_event_hint(
            fid, proposition_text, predicate
        )
        if event_hint:
            if not event_time and clean_text(event_hint.get("event_time", "")):
                event_time = clean_text(event_hint.get("event_time", ""))
                enrichment["event_time"] = event_time
            if not location and clean_text(event_hint.get("location", "")):
                location = clean_text(event_hint.get("location", ""))
                enrichment["location"] = location
            if enrichment:
                enrichment["source_table"] = "query_events.csv"
                enrichment["source_row_id"] = clean_text(
                    event_hint.get("source_row_id", "")
                )
                enrichment["contract"] = (
                    "unambiguous_fragment_event_overlap"
                )
        fragment_temporal_context = fragment_text_map.get(fid, "")
        signature_source = proposition_text
        if (
            source_kind == "claim"
            and evidence_surface
        ):
            generic_supplemental_actor = (
                is_generic_discourse_actor(
                    clean_text(supplemental_clause.get("actor", ""))
                )
            )
            primary_clause_score = sum(
                bool(clean_text(inferred_clause.get(field, "")))
                for field in ("actor", "predicate", "object", "location")
            )
            supplemental_clause_score = sum(
                bool(clean_text(supplemental_clause.get(field, "")))
                for field in ("actor", "predicate", "object", "location")
            )
            if (
                supplemental_clause_score > primary_clause_score
                or (
                    not clean_text(inferred_clause.get("predicate", ""))
                    and supplemental_clause_score >= primary_clause_score
                )
                or (
                    clean_text(inferred_clause.get("predicate", "")) == "be"
                    and clean_text(supplemental_clause.get("predicate", ""))
                    and clean_text(supplemental_clause.get("predicate", ""))
                    != "be"
                )
                or (
                    not clean_text(inferred_clause.get("object", ""))
                    and clean_text(supplemental_clause.get("object", ""))
                )
                or (
                    clean_text(claim_speaker)
                    and clean_text(supplemental_clause.get("actor", ""))
                    and clean_text(supplemental_clause.get("actor", "")).casefold()
                    == clean_text(claim_speaker).casefold()
                    and clean_text(inferred_clause.get("actor", "")).casefold()
                    != clean_text(claim_speaker).casefold()
                )
            ):
                signature_source = evidence_surface
            if (
                generic_supplemental_actor
                and not clean_text(claim_speaker)
                and clean_text(inferred_clause.get("predicate", ""))
                and clean_text(inferred_clause.get("object", ""))
            ):
                signature_source = proposition_text
        # An LLM may abridge a single-event claim (for example, dropping its
        # location/date/quantities).  Rehydrate from the immutable fragment
        # only when the fragment contains one governing event family.  A
        # multi-event fragment is deliberately not inherited wholesale,
        # preventing dates and quantities from leaking into sibling clauses.
        action_families = {
            canonical_predicate(item)
            for item in re.findall(
                r"\b(?:seiz(?:e|ed|es)|confiscat(?:e|ed|es|ing)|arrest(?:ed|s)?|"
                r"transport(?:ed|s|ing)?|attend(?:ed|s|ing)?|represent(?:ed|s|ing)?|"
                r"report(?:ed|s)?|inform(?:ed|s)?|turn(?:ed)?\s+over)\b",
                fragment_temporal_context,
                flags=re.I,
            )
        }
        action_families.discard("")
        if (
            fragment_temporal_context
            and len(action_families) == 1
            and predicate in action_families
            and canonical_key(proposition_text) in canonical_key(fragment_temporal_context)
        ):
            signature_source = fragment_temporal_context
        signature = build_event_signature(
            signature_source,
            actor=subject,
            predicate=predicate,
            obj=obj,
            location=location,
            event_time=event_time,
            stance=stance,
            modality=modality,
        )
        signature_json = json.dumps(signature, sort_keys=True, ensure_ascii=False, default=str)
        # Reporting and embedded assertion are distinct dimensions.  The
        # canonical matching actor is the embedded actor where one is known.
        reporter = clean_text(signature.get("reporter", ""))
        embedded_actor = clean_text(signature.get("embedded_actor", ""))
        canonical_subject = embedded_actor or clean_text(signature.get("actor", "")) or subject
        predicate = clean_text(signature.get("predicate", "")) or predicate
        obj = clean_text(signature.get("object", "")) or obj
        location = clean_text(signature.get("location", "")) or location
        reporting_signature = (
            build_event_signature(
                f"{reporter} reported an embedded assertion",
                actor=reporter,
                predicate="report",
                obj="embedded assertion",
                source_attribution=reporter,
            )
            if reporter else {}
        )
        polarity = clean_text(signature.get("assertion_polarity", "uncertain"))
        normalized = canonical_key(proposition_text)
        rows.append({
            "query_proposition_id": stable_id(
                [VERSION, source_kind, fid, source_row_id, signature_json], "qp"
            ),
            "fragment_id": fid,
            "proposition_source": source_kind,
            "source_row_id": clean_text(source_row_id),
            "proposition_text": proposition_text,
            "normalized_proposition": normalized,
            "canonical_subject": canonical_subject,
            "canonical_predicate": clean_text(predicate) or "claim",
            "canonical_object": clean_text(obj),
            "canonical_location": clean_text(location),
            "object_surface": object_surface,
            "predicate_surface": predicate_surface,
            "reporter": reporter,
            "embedded_actor": embedded_actor,
            "agent": clean_text(signature.get("agent", "")),
            "patient_or_theme": clean_text(
                signature.get("patient_or_theme", "")
            ),
            "recipient": clean_text(signature.get("recipient", "")),
            "destination": clean_text(signature.get("destination", "")),
            "voice": clean_text(signature.get("voice", "unknown")),
            "role_normalization_status": clean_text(
                signature.get(
                    "role_normalization_status", "NOT_APPLICABLE"
                )
            ),
            "clause_level": "embedded_assertion" if reporter else "main_clause",
            "reporting_event_signature_json": json.dumps(
                reporting_signature, sort_keys=True, ensure_ascii=False, default=str
            ) if reporting_signature else "",
            "embedded_assertion_signature_json": signature_json,
            "event_time": clean_text(signature.get("event_time", "")),
            "event_time_precision": clean_text(signature.get("event_time_precision", "")),
            "report_time": clean_text(signature.get("report_time", "")),
            "assertion_polarity": polarity,
            "polarity": polarity,
            "modality": clean_text(modality) or "unspecified",
            "speaker_stance": clean_text(stance),
            "stance": clean_text(stance),
            "epistemic_state": clean_text(signature.get("epistemic_state", "")),
            "semantic_state_variable": clean_text(signature.get("semantic_state_variable", "")),
            "semantic_state_value": clean_text(signature.get("semantic_state_value", "")),
            "packet_count": signature.get("packet_count"),
            "weight_kg": signature.get("weight_kg"),
            "weight_g": signature.get("weight_g"),
            "total_weight_grams": signature.get("total_weight_grams"),
            "date_value": clean_text(signature.get("date_value", "")),
            "event_signature_json": signature_json,
            "event_signature_sha256": v19_stable_hash(signature),
            "representation_enrichment_json": json.dumps(
                enrichment, sort_keys=True, ensure_ascii=False
            ) if enrichment else "",
            "evidence_text": clean_text(evidence_text) or proposition_text,
            "extraction_basis": extraction_basis,
            "parser_source": parser_source,
            "confidence": confidence,
            "primary_for_matching": True,
            "contradiction_key": " | ".join([
                canonical_key(canonical_subject), canonical_key(predicate), canonical_key(obj),
                canonical_key(location), clean_text(signature.get("event_time", "")),
                clean_text(signature.get("semantic_state_variable", "")),
            ]).strip(" |"),
        })

    for _, row in claims.iterrows():
        fid = clean_text(row.get("fragment_id", ""))
        append_row(
            fid, "claim", row.get("query_claim_id", ""),
            clean_text(row.get("claim_text", row.get("evidence_text", ""))),
            stance=clean_text(row.get("stance", "")),
            modality=infer_modality_from_text(
                row.get("claim_text", ""), row.get("extraction_basis", ""), row.get("stance", "")
            ),
            evidence_text=row.get("evidence_text", ""),
            extraction_basis=clean_text(row.get("extraction_basis", "")),
            parser_source=clean_text(row.get("parser_source", "")),
            confidence=row.get("confidence", ""),
            claim_speaker=clean_text(row.get("speaker", "")),
            claim_target=clean_text(row.get("target", "")),
        )

    for _, row in relations.iterrows():
        fid = clean_text(row.get("fragment_id", ""))
        subject = clean_text(row.get("subject_entity", ""))
        predicate = clean_text(row.get("predicate", ""))
        obj = clean_text(row.get("object_entity", ""))
        proposition_text = clean_text(row.get("evidence_text", "")) or f"{subject} {predicate} {obj}"
        append_row(
            fid, "relation", row.get("query_relation_id", ""), proposition_text,
            subject=subject, predicate=predicate, obj=obj,
            stance=clean_text(row.get("qualifier", "")),
            modality=infer_modality_from_text(proposition_text, row.get("extraction_basis", ""), row.get("qualifier", "")),
            evidence_text=row.get("evidence_text", ""), extraction_basis=clean_text(row.get("extraction_basis", "")),
            parser_source=clean_text(row.get("parser_source", "")), confidence=row.get("confidence", ""),
        )

    def relation_is_redundant_with_claim(
        relation_row: Mapping[str, Any],
        same_fragment_claims: Sequence[Mapping[str, Any]],
    ) -> bool:
        relation_text = canonical_key(relation_row.get("proposition_text", ""))
        relation_subject = canonical_key(
            relation_row.get("canonical_subject", "")
        )
        relation_predicate = canonical_predicate(
            relation_row.get("canonical_predicate", "")
        )
        relation_object = canonical_key(
            relation_row.get("canonical_object", "")
        )
        generic_discourse_subjects = {
            "announcement",
            "announcements",
            "comment",
            "commentator",
            "commentators",
            "communique",
            "editorial",
            "message",
            "note",
            "press",
            "press report",
            "report",
            "reports",
            "source",
            "sources",
            "statement",
        }
        generic_reporting_predicates = {
            "claim",
            "inform",
            "press",
            "report",
            "say",
            "statement",
        }
        for claim_row in same_fragment_claims:
            claim_text = canonical_key(claim_row.get("proposition_text", ""))
            claim_predicate = canonical_predicate(
                claim_row.get("canonical_predicate", "")
            )
            claim_object = canonical_key(
                claim_row.get("canonical_object", "")
            )
            same_predicate = bool(
                relation_predicate
                and claim_predicate
                and relation_predicate == claim_predicate
            )
            same_object = bool(
                relation_object
                and claim_object
                and (
                    relation_object in claim_object
                    or claim_object in relation_object
                )
            )
            contained_surface = bool(
                relation_text
                and claim_text
                and (
                    relation_text in claim_text
                    or claim_text in relation_text
                )
            )
            if same_predicate and same_object:
                return True
            if contained_surface and same_predicate:
                return True
            if (
                contained_surface
                and relation_predicate in generic_reporting_predicates
            ):
                return True
            if (
                relation_subject in generic_discourse_subjects
                and (same_predicate or same_object or contained_surface)
            ):
                return True
        return False

    if rows:
        claim_rows_by_fragment: Dict[str, List[Dict[str, Any]]] = {}
        for item in rows:
            if clean_text(item.get("proposition_source", "")) != "claim":
                continue
            claim_rows_by_fragment.setdefault(
                clean_text(item.get("fragment_id", "")), []
            ).append(item)
        if claim_rows_by_fragment:
            rows = [
                item
                for item in rows
                if not (
                    clean_text(item.get("proposition_source", "")) == "relation"
                    and relation_is_redundant_with_claim(
                        item,
                        claim_rows_by_fragment.get(
                            clean_text(item.get("fragment_id", "")),
                            [],
                        ),
                    )
                )
            ]

    # Project a declarative event only when the fragment has no native claim or
    # relation proposition.  This is the C1 route and avoids duplicate votes.
    represented_fragments = {clean_text(item.get("fragment_id", "")) for item in rows}
    for _, row in events.iterrows():
        fid = clean_text(row.get("fragment_id", ""))
        if fid in represented_fragments:
            continue
        full_text = fragment_text_map.get(fid, "")
        event_text = clean_text(row.get("event_text", row.get("evidence_text", "")))
        proposition_text = full_text or clean_text(row.get("evidence_text", "")) or event_text
        append_row(
            fid, "event_projection", row.get("query_event_id", ""), proposition_text,
            predicate=clean_text(row.get("event_type", "event")), obj=event_text,
            location=clean_text(row.get("location", "")),
            event_time=clean_text(row.get("date_or_temporal_context", "")),
            modality=infer_modality_from_text(proposition_text, row.get("extraction_basis", ""), ""),
            evidence_text=row.get("evidence_text", ""), extraction_basis="event_to_proposition_projection",
            parser_source=clean_text(row.get("parser_source", "")), confidence=row.get("confidence", ""),
        )
        represented_fragments.add(fid)

    # v0.4.10 deterministic short-clause fallback.
    #
    # The prior heuristic claim extractor required seven tokens and its event
    # trigger list did not include all inflected predicates understood by the
    # shared clause analyzer.  A valid statement such as
    # A short actor/action/object statement could therefore produce entities
    # but no executable proposition, blocking contradiction preflight.
    #
    # This fallback is intentionally conservative and corpus-independent:
    #   * it runs only when the fragment has no proposition from a claim,
    #     asserted relation, or event;
    #   * it rejects questions and redaction placeholders;
    #   * it requires the shared deterministic analyzer to find an actor,
    #     predicate, and object/location;
    #   * it records explicit fallback provenance in the proposition row.
    #
    # The matcher contract and output schema are unchanged.
    represented_fragments = {
        clean_text(item.get("fragment_id", "")) for item in rows
    }
    if len(fragments):
        for _, fragment in fragments.iterrows():
            fid = clean_text(fragment.get("fragment_id", ""))
            if not fid or fid in represented_fragments:
                continue
            fragment_text = clean_text(
                fragment.get(
                    "fragment_text_for_extraction",
                    fragment.get("fragment_text", ""),
                )
            )
            if not fragment_text:
                continue
            if any(
                marker in fragment_text
                for marker in (
                    "[REDACTION_SLOT]",
                    "[SENTENCE_REDACTION_SLOT]",
                    "[REDACT",
                )
            ):
                continue
            for sentence_index, sentence in enumerate(
                sentence_split(fragment_text), start=1
            ):
                sentence = clean_text(sentence)
                if not sentence or "?" in sentence:
                    continue
                if re.match(
                    r"^(?:who|what|when|where|why|how|which|whose|whom|"
                    r"did|do|does|is|are|was|were|can|could|may|might|"
                    r"should|would|will)\b",
                    sentence,
                    flags=re.I,
                ):
                    continue
                if len(re.findall(r"\b\w+\b", sentence)) < 3:
                    continue
                clause = infer_simple_clause_arguments(sentence)
                actor = clean_text(clause.get("actor", ""))
                predicate = canonical_predicate(clause.get("predicate", ""))
                obj = clean_text(clause.get("object", ""))
                location = clean_text(clause.get("location", ""))
                agentless_passive = bool(
                    not actor
                    and clean_text(clause.get("voice", "")) == "passive"
                    and clean_text(clause.get("patient_or_theme", ""))
                )
                if not location:
                    trailing_location = re.search(
                        r"\b(?:at|in|near)\s+(?:the\s+)?"
                        r"(?P<location>[A-Z][A-Za-z0-9'().&/\- ]{1,70}?)"
                        r"[\s,.;:]*$",
                        sentence,
                    )
                    if trailing_location:
                        location = clean_text(
                            trailing_location.group("location")
                        )
                if (
                    (not actor and not agentless_passive)
                    or not predicate
                    or not (obj or location)
                ):
                    continue
                append_row(
                    fid,
                    "declarative_fragment_projection",
                    stable_id(
                        [VERSION, fid, sentence_index, sentence],
                        "qdfp",
                    ),
                    sentence,
                    subject=actor,
                    predicate=predicate,
                    obj=obj,
                    location=location,
                    modality=infer_modality_from_text(
                        sentence,
                        "deterministic_clause_to_proposition_fallback",
                        "",
                    ),
                    evidence_text=sentence,
                    extraction_basis=(
                        "deterministic_clause_to_proposition_fallback"
                    ),
                    parser_source="heuristic_declarative_fallback",
                    confidence=0.45,
                )
            if any(
                clean_text(item.get("fragment_id", "")) == fid
                for item in rows
            ):
                represented_fragments.add(fid)

    # PRD-316--318: preserve typed role alternatives instead of allowing one
    # surface relation to become the only event representation.  This uses
    # entity types and compatible same-fragment event evidence only; no name,
    # location, document, corpus, or event-specific vocabulary participates.
    hypothesis_rows: List[Dict[str, Any]] = []
    for base in rows:
        fid = clean_text(base.get("fragment_id", ""))
        source_kind = clean_text(base.get("proposition_source", ""))
        source_row_id = clean_text(base.get("source_row_id", ""))
        group_id = stable_id(
            [VERSION, "hypothesis_group", fid, source_kind, source_row_id],
            "qhg",
        )
        subject = clean_text(base.get("canonical_subject", ""))
        subject_types = sorted({
            clean_text(item.get("type", "")).upper()
            for item in entity_map.get(fid, [])
            if clean_text(item.get("label", "")).casefold()
            == subject.casefold()
            and clean_text(item.get("type", ""))
        })
        subject_role = subject_types[0] if len(subject_types) == 1 else (
            "AMBIGUOUS" if len(subject_types) > 1 else "UNKNOWN"
        )
        try:
            source_confidence = float(base.get("confidence", 0.5))
        except (TypeError, ValueError):
            source_confidence = 0.5
        source_confidence = max(0.0, min(1.0, source_confidence))
        base = {
            **base,
            "hypothesis_group_id": group_id,
            "proposition_hypothesis_id": stable_id(
                [group_id, "direct", base.get("event_signature_sha256", "")],
                "qph",
            ),
            "hypothesis_type": "DIRECT_TYPED_INTERPRETATION",
            "subject_semantic_role": subject_role,
            "hypothesis_confidence": source_confidence,
            "hypothesis_provenance_json": json.dumps(
                [{
                    "source_table": (
                        "query_relations.csv"
                        if source_kind == "relation"
                        else "query_claims.csv"
                        if source_kind == "claim"
                        else "query_events.csv"
                        if source_kind == "event_projection"
                        else "query_fragments.csv"
                    ),
                    "source_row_id": source_row_id,
                    "interpretation": "direct",
                }],
                sort_keys=True,
            ),
        }
        hypothesis_rows.append(base)
        inferred_nominal = infer_simple_clause_arguments(
            base.get("proposition_text", "")
        )
        nominal_modifier = clean_text(
            inferred_nominal.get("nominal_modifier", "")
        )
        if (
            nominal_modifier
            and canonical_key(nominal_modifier)
            != canonical_key(base.get("canonical_object", ""))
        ):
            for role in ("location", "actor"):
                signature = build_event_signature(
                    base.get("proposition_text", ""),
                    actor=(
                        nominal_modifier
                        if role == "actor"
                        else base.get("canonical_subject", "")
                    ),
                    predicate=base.get("canonical_predicate", ""),
                    obj=base.get("canonical_object", ""),
                    location=(
                        nominal_modifier
                        if role == "location"
                        else base.get("canonical_location", "")
                    ),
                    event_time=base.get("event_time", ""),
                    stance=base.get("stance", ""),
                    modality=base.get("modality", ""),
                )
                if role == "location":
                    signature["location"] = nominal_modifier
                    signature["location_identity"] = canonical_key(
                        nominal_modifier
                    )
                else:
                    signature["actor"] = nominal_modifier
                    signature["agent"] = nominal_modifier
                signature_json = json.dumps(
                    signature,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                interpretation = (
                    "NOMINAL_MODIFIER_"
                    + role.upper()
                    + "_INTERPRETATION"
                )
                alternative = {
                    **base,
                    "query_proposition_id": stable_id(
                        [
                            VERSION, source_kind, fid, source_row_id,
                            interpretation,
                        ],
                        "qp",
                    ),
                    "proposition_hypothesis_id": stable_id(
                        [group_id, interpretation],
                        "qph",
                    ),
                    "hypothesis_type": interpretation,
                    "subject_semantic_role": role.upper(),
                    "hypothesis_confidence": source_confidence * 0.75,
                    "hypothesis_provenance_json": json.dumps(
                        [{
                            "source_table": (
                                "query_claims.csv"
                                if source_kind == "claim"
                                else "query_fragments.csv"
                            ),
                            "source_row_id": source_row_id,
                            "interpretation":
                                "ambiguous_nominal_event_modifier",
                            "semantic_role": role,
                        }],
                        sort_keys=True,
                    ),
                    "canonical_subject": (
                        nominal_modifier
                        if role == "actor"
                        else base.get("canonical_subject", "")
                    ),
                    "canonical_location": (
                        nominal_modifier
                        if role == "location"
                        else base.get("canonical_location", "")
                    ),
                    "agent": (
                        nominal_modifier if role == "actor" else ""
                    ),
                    "role_normalization_status":
                        "AMBIGUOUS_TYPED_HYPOTHESES_PRESERVED",
                    "event_signature_json": signature_json,
                    "event_signature_sha256":
                        v19_stable_hash(signature),
                    "representation_enrichment_json": json.dumps({
                        "contract":
                            "ambiguous_nominal_modifier_alternatives",
                        "modifier": nominal_modifier,
                        "semantic_role": role,
                    }, sort_keys=True),
                }
                hypothesis_rows.append(alternative)
        if (
            source_kind != "relation"
            or not set(subject_types) & {"LOCATION", "FACILITY"}
        ):
            continue
        proposition_tokens = set(
            semantic_tokens(base.get("proposition_text", ""))
        )
        compatible_events = []
        for event in event_map.get(fid, []):
            event_tokens = set(event.get("tokens", set()))
            if not event_tokens or not proposition_tokens:
                continue
            shared = proposition_tokens & event_tokens
            if len(shared) < 1:
                continue
            compatible_events.append(event)
        for event in compatible_events:
            event_predicate = (
                canonical_predicate(event.get("event_type", ""))
                or canonical_predicate(base.get("canonical_predicate", ""))
            )
            event_object = (
                canonical_object_head(event.get("event_text", ""))
                or clean_text(base.get("canonical_object", ""))
            )
            signature = build_event_signature(
                base.get("proposition_text", ""),
                actor="",
                predicate=event_predicate,
                obj=event_object,
                location=subject,
                event_time=event.get("event_time", ""),
            )
            # The alternative represents an event with a locative adjunct;
            # a surface location must not be silently promoted to actor.
            signature["actor"] = ""
            signature["agent"] = ""
            signature["location"] = subject
            signature["location_identity"] = canonical_key(subject)
            signature_json = json.dumps(
                signature, sort_keys=True, ensure_ascii=False, default=str
            )
            alternative = {
                **base,
                "query_proposition_id": stable_id(
                    [
                        VERSION, source_kind, fid, source_row_id,
                        "event_locative", event.get("source_row_id", ""),
                    ],
                    "qp",
                ),
                "proposition_hypothesis_id": stable_id(
                    [
                        group_id, "event_locative",
                        event.get("source_row_id", ""),
                    ],
                    "qph",
                ),
                "hypothesis_type": "EVENT_LOCATIVE_INTERPRETATION",
                "subject_semantic_role": "EVENT",
                "hypothesis_confidence": source_confidence * 0.90,
                "hypothesis_provenance_json": json.dumps(
                    [
                        {
                            "source_table": "query_relations.csv",
                            "source_row_id": source_row_id,
                            "interpretation": "surface_relation",
                        },
                        {
                            "source_table": "query_events.csv",
                            "source_row_id": clean_text(
                                event.get("source_row_id", "")
                            ),
                            "interpretation": "compatible_typed_event",
                        },
                    ],
                    sort_keys=True,
                ),
                "canonical_subject": clean_text(
                    event.get("event_text", "")
                ),
                "canonical_predicate": event_predicate,
                "canonical_object": event_object,
                "canonical_location": subject,
                "agent": "",
                "role_normalization_status":
                    "AMBIGUOUS_TYPED_HYPOTHESES_PRESERVED",
                "event_signature_json": signature_json,
                "event_signature_sha256": v19_stable_hash(signature),
                "representation_enrichment_json": json.dumps({
                    "contract":
                        "typed_same_fragment_event_locative_alternative",
                    "source_table": "query_events.csv",
                    "source_row_id": clean_text(
                        event.get("source_row_id", "")
                    ),
                }, sort_keys=True),
            }
            hypothesis_rows.append(alternative)

    out = ensure_columns(
        pd.DataFrame(hypothesis_rows), PROPOSITION_COLUMNS
    )
    if len(out):
        # Deduplicate only identical hypotheses; alternative typed
        # interpretations sharing a fragment are intentionally retained.
        out = out.drop_duplicates(
            [
                "fragment_id", "event_signature_sha256",
                "hypothesis_type", "source_row_id",
            ],
            keep="first",
        ).reset_index(drop=True)
    return out

def parse_query_text(
    text: str,
    use_openai: bool = False,
    openai_model: str = "gpt-4o",
    api_key_env: str = "OPENAI_API_KEY",
    max_chars_per_fragment: int = 2200,
    heuristic_policy: str = "fallback_only",
    allow_heuristic_relations_primary: bool = False,
    allow_heuristic_claims_primary: bool = False,
    allow_heuristic_events_primary: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], str, List[str]]:
    """Parse a query text into deterministic fragments and semantic rows.

    v0.3.18 deliberately separates three concepts that were mixed in v0.3.14:

    1. Deterministic structure: fragments and explicit redaction slots.
       These are always produced locally and are authoritative.

    2. Primary semantic evidence: rows intended for matching. In OpenAI mode
       with fallback_only, these are OpenAI rows unless OpenAI produced no rows
       for a fragment/table, in which case heuristic rows are promoted with an
       explicit fallback annotation.

    3. Debug evidence: full heuristic and full OpenAI outputs, written to
       *_debug.csv files by write_parse_outputs. These are for audit only and
       should not be used by the matcher unless deliberately configured.
    """
    warnings: List[str] = []
    heuristic = heuristic_parse(text, max_chars_per_fragment=max_chars_per_fragment)

    # Heuristic-only mode remains useful for smoke tests and when API access is
    # unavailable. In that case the heuristic rows are the only available rows,
    # so they are marked as primary while retaining parser_source=heuristic.
    if not use_openai:
        parsed = {k: v for k, v in heuristic.items()}
        clean_entities, suppressed_entities = filter_redaction_placeholder_entities(parsed.get("entities", pd.DataFrame()), "heuristic_only_primary")
        parsed["entities"] = set_primary_flag(clean_entities, True)
        parsed["entity_suppressed_debug"] = suppressed_entities
        parsed["claims"] = set_primary_flag(parsed["claims"], True)
        clean_events, suppressed_events = filter_redaction_placeholder_events(parsed.get("events", pd.DataFrame()), "heuristic_only_primary")
        parsed["events"] = set_primary_flag(clean_events, True) if allow_heuristic_events_primary or not use_openai else ensure_columns(pd.DataFrame(), EVENT_COLUMNS)
        parsed["event_suppressed_debug"] = suppressed_events
        parsed["redaction_slots"] = set_primary_flag(parsed["redaction_slots"], True)
        parsed["sentence_redaction_slots"] = set_primary_flag(parsed.get("sentence_redaction_slots", pd.DataFrame()), True)
        # Heuristic co-occurrence relations are not asserted facts.  Keep them
        # out of primary matching by default, even in heuristic-only mode.
        if allow_heuristic_relations_primary:
            parsed["relations"] = set_primary_flag(parsed["relations"], True)
        else:
            parsed["relations"] = ensure_columns(pd.DataFrame(), RELATION_COLUMNS)
        parsed["propositions"] = build_query_propositions(parsed["claims"], parsed["relations"], parsed.get("events"), parsed.get("entities"), parsed.get("fragments"))
        parsed["heuristic_debug"] = {k: heuristic[k] for k in ["entities", "claims", "events", "relations"]}
        parsed["openai_debug"] = {k: ensure_columns(pd.DataFrame(), {"entities": ENTITY_COLUMNS, "claims": CLAIM_COLUMNS, "events": EVENT_COLUMNS, "relations": RELATION_COLUMNS}[k]) for k in ["entities", "claims", "events", "relations"]}
        return parsed, "heuristic", warnings

    openai_rows, openai_warnings = parse_with_openai_per_fragment(heuristic, model=openai_model, api_key_env=api_key_env)
    warnings.extend(openai_warnings)

    openai_has_any = bool(openai_rows) and any(not openai_rows.get(k, pd.DataFrame()).empty for k in ["entities", "claims", "events", "relations"])
    if not openai_has_any:
        warnings.append("OpenAI parsing unavailable or returned no semantic rows; using heuristic parse only.")
        parsed = {k: v for k, v in heuristic.items()}
        clean_entities, suppressed_entities = filter_redaction_placeholder_entities(parsed.get("entities", pd.DataFrame()), "openai_unavailable_heuristic_primary")
        parsed["entities"] = set_primary_flag(clean_entities, True)
        parsed["entity_suppressed_debug"] = suppressed_entities
        parsed["claims"] = set_primary_flag(parsed["claims"], True)
        clean_events, suppressed_events = filter_redaction_placeholder_events(parsed.get("events", pd.DataFrame()), "openai_unavailable_heuristic_primary")
        parsed["events"] = set_primary_flag(clean_events, True)
        parsed["event_suppressed_debug"] = suppressed_events
        parsed["redaction_slots"] = set_primary_flag(parsed["redaction_slots"], True)
        parsed["sentence_redaction_slots"] = set_primary_flag(parsed.get("sentence_redaction_slots", pd.DataFrame()), True)
        if allow_heuristic_relations_primary:
            parsed["relations"] = set_primary_flag(parsed["relations"], True)
        else:
            parsed["relations"] = ensure_columns(pd.DataFrame(), RELATION_COLUMNS)
        parsed["propositions"] = build_query_propositions(parsed["claims"], parsed["relations"], parsed.get("events"), parsed.get("entities"), parsed.get("fragments"))
        parsed["heuristic_debug"] = {k: heuristic[k] for k in ["entities", "claims", "events", "relations"]}
        parsed["openai_debug"] = {k: ensure_columns(pd.DataFrame(), {"entities": ENTITY_COLUMNS, "claims": CLAIM_COLUMNS, "events": EVENT_COLUMNS, "relations": RELATION_COLUMNS}[k]) for k in ["entities", "claims", "events", "relations"]}
        return parsed, f"heuristic_openai_unavailable:{openai_model}", warnings

    # Deterministic fragments and redaction slots remain authoritative. The LLM
    # never creates its own fragments or redaction slots.
    combined: Dict[str, Any] = {
        "fragments": heuristic["fragments"],
        "redaction_slots": set_primary_flag(heuristic["redaction_slots"], True),
        "sentence_redaction_slots": set_primary_flag(heuristic.get("sentence_redaction_slots", pd.DataFrame()), True),
        "heuristic_debug": {k: set_primary_flag(heuristic[k], False) for k in ["entities", "claims", "events", "relations"]},
        "openai_debug": {k: set_primary_flag(openai_rows.get(k, pd.DataFrame()), True) for k in ["entities", "claims", "events", "relations"]},
        "entity_suppressed_debug": pd.DataFrame(),
        "event_suppressed_debug": pd.DataFrame(),
    }

    for key, cols, id_col, sem_cols in [
        ("entities", ENTITY_COLUMNS, "query_entity_id", ["fragment_id", "label", "type"]),
        ("claims", CLAIM_COLUMNS, "query_claim_id", ["fragment_id", "claim_text"]),
        ("events", EVENT_COLUMNS, "query_event_id", ["fragment_id", "event_text", "event_type"]),
        ("relations", RELATION_COLUMNS, "query_relation_id", ["fragment_id", "subject_entity", "predicate", "object_entity"]),
    ]:
        openai_df = openai_rows.get(key, pd.DataFrame())
        heuristic_df = heuristic.get(key, pd.DataFrame())

        if key == "entities":
            # v0.3.18/v0.3.18 hygiene: OpenAI may sometimes echo redaction placeholders
            # as entities, e.g. "[REDACT:R1 type=PERSON]".  Those rows belong
            # only in query_redaction_slots.csv, not in query_entities.csv.
            # We filter before primary selection so placeholders do not count as
            # real OpenAI evidence and do not suppress fallback behavior.
            openai_df, openai_suppressed = filter_redaction_placeholder_entities(openai_df, "openai_pre_primary_selection")
            heuristic_df, heuristic_suppressed = filter_redaction_placeholder_entities(heuristic_df, "heuristic_pre_primary_selection")
            suppressed_frames = [df for df in [combined.get("entity_suppressed_debug", pd.DataFrame()), openai_suppressed, heuristic_suppressed] if df is not None and len(df)]
            combined["entity_suppressed_debug"] = pd.concat(suppressed_frames, ignore_index=True, sort=False) if suppressed_frames else pd.DataFrame()

        if key == "events":
            # v0.3.18 hygiene: event-like rows that contain redaction slots are
            # not historical events. They belong only in debug output and should
            # not drive reconstruction, claim support, or contradiction matching.
            openai_df, openai_event_suppressed = filter_redaction_placeholder_events(openai_df, "openai_pre_primary_selection")
            heuristic_df, heuristic_event_suppressed = filter_redaction_placeholder_events(heuristic_df, "heuristic_pre_primary_selection")
            suppressed_frames = [
                df for df in [combined.get("event_suppressed_debug", pd.DataFrame()), openai_event_suppressed, heuristic_event_suppressed]
                if df is not None and len(df)
            ]
            combined["event_suppressed_debug"] = pd.concat(suppressed_frames, ignore_index=True, sort=False) if suppressed_frames else pd.DataFrame()

        # Corpus-agnostic safety rule: a heuristic relation is only a
        # co-occurrence hint unless a real extractor asserted a predicate.
        # Therefore, relations default to openai_only primary selection even
        # when entities/claims/events use fallback_only.  The user can override
        # with --allow-heuristic-relations-primary for experiments.
        effective_policy = heuristic_policy
        if key == "relations" and not allow_heuristic_relations_primary:
            effective_policy = "openai_only"
        # Prefer OpenAI claims, but do not preflight-block a declarative query
        # solely because the model emitted zero claim rows. In that narrow case,
        # fallback_only promotes the deterministic claim for the empty fragment;
        # it never merges heuristic rows into a fragment with an OpenAI claim.
        if key == "claims" and not allow_heuristic_claims_primary:
            effective_policy = "fallback_only"
        # v0.3.18 applies the same high-precision default to heuristic events
        # when OpenAI is available. Heuristic event sentences are useful for
        # debugging and recall experiments, but they are often too broad for
        # primary semantic matching unless explicitly requested.
        if key == "events" and not allow_heuristic_events_primary:
            effective_policy = "openai_only"
        combined[key] = select_primary_semantic_rows(
            key=key,
            fragments=combined["fragments"],
            openai_df=openai_df,
            heuristic_df=heuristic_df,
            heuristic_policy=effective_policy,
            id_col=id_col,
            semantic_cols=sem_cols,
            columns=cols,
        )
        if key == "relations":
            combined[key] = suppress_redundant_primary_relations(
                combined[key]
            )

    combined["propositions"] = build_query_propositions(combined["claims"], combined["relations"], combined.get("events"), combined.get("entities"), combined.get("fragments"))
    return combined, f"openai_{heuristic_policy}_fixed_fragments:{openai_model}", warnings



def build_query_event_signatures(parsed: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten canonical proposition signatures for matcher/cache inspection."""
    rows: List[Dict[str, Any]] = []
    propositions = parsed.get("propositions", pd.DataFrame())
    for _, row in propositions.iterrows():
        signature_text = clean_text(row.get("event_signature_json", ""))
        try:
            signature = json.loads(signature_text) if signature_text else {}
        except json.JSONDecodeError:
            signature = {}
        rows.append({
            "query_event_signature_id": "qes:" + v19_stable_hash([
                row.get("fragment_id", ""), row.get("query_proposition_id", ""), signature
            ], 20),
            "fragment_id": clean_text(row.get("fragment_id", "")),
            "source_kind": clean_text(row.get("proposition_source", "")),
            "source_row_id": clean_text(row.get("source_row_id", "")),
            "proposition_text": clean_text(row.get("proposition_text", "")),
            **{key: signature.get(key, "") for key in [
                "actor", "agent", "patient_or_theme", "recipient",
                "destination", "voice", "role_normalization_status",
                "reporter", "embedded_actor", "predicate",
                "predicate_surface", "object", "object_surface",
                "location", "location_identity", "event_time",
                "event_time_precision", "report_time", "source_attribution", "stance",
                "modality", "assertion_polarity", "epistemic_state",
                "semantic_state_variable", "semantic_state_value", "packet_count",
                "weight_kg", "weight_g", "total_weight_grams", "date_value",
            ]},
            "event_signature_json": json.dumps(signature, sort_keys=True, ensure_ascii=False, default=str),
            "event_signature_sha256": v19_stable_hash(signature),
            "primary_for_matching": _primary_flag(
                row.get("primary_for_matching", True)
            ),
        })
    return ensure_columns(pd.DataFrame(rows), EVENT_SIGNATURE_COLUMNS)


def _role_equivalence_key(value: Any) -> str:
    """Normalize harmless OCR, acronym, and auxiliary role variation."""
    text = clean_text(value)
    text = re.sub(r"(?<=[A-Za-z])-\s*(?=[A-Za-z])", "", text)
    text = re.sub(r"\s*\([A-Z][A-Z0-9.\-]{1,12}\)\s*$", "", text)
    text = re.sub(
        r"\b(?:has|have|had|is|are|was|were|be|been|being)\s*$",
        "",
        text,
        flags=re.I,
    )
    return canonical_key(text)


def build_representation_reconciliation(
    parsed: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Reconcile proposition views by source sentence and typed role.

    Reconciliation is field-level and semantic.  Punctuation, OCR hyphenation,
    a trailing auxiliary, or a parenthetical acronym cannot manufacture a hard
    conflict.  Event-table enrichment provenance is preserved separately.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def reconciliation_anchor(
        signature: Mapping[str, Any],
        proposition_text: str,
        source_sentence: str,
    ) -> str:
        actor = _role_equivalence_key(
            signature.get("embedded_actor", "") or signature.get("actor", "")
        )
        predicate = canonical_predicate(signature.get("predicate", ""))
        obj = _role_equivalence_key(
            signature.get("object", "") or signature.get("patient_or_theme", "")
        )
        location = _role_equivalence_key(
            signature.get("location_identity", "") or signature.get("location", "")
        )
        event_time = canonical_key(
            signature.get("event_time", "") or signature.get("date_value", "")
        )
        polarity = canonical_key(signature.get("assertion_polarity", ""))
        anchored_count = sum(
            bool(value)
            for value in (actor, predicate, obj, location, event_time)
        )
        if anchored_count >= 2:
            return "sig||" + "||".join(
                (
                    actor,
                    predicate,
                    obj,
                    location,
                    event_time,
                    polarity,
                )
            )
        fallback_text = proposition_text or source_sentence
        return "text||" + canonical_key(fallback_text)

    propositions = parsed.get("propositions", pd.DataFrame())
    for _, row in propositions.iterrows():
        sentence = clean_text(
            row.get("evidence_text", row.get("proposition_text", ""))
        )
        try:
            signature = json.loads(
                clean_text(row.get("event_signature_json", ""))
            )
        except (TypeError, json.JSONDecodeError):
            signature = {}
        proposition_text = clean_text(
            row.get("proposition_text", row.get("normalized_proposition", ""))
        )
        key = (
            clean_text(row.get("fragment_id", "")),
            reconciliation_anchor(signature, proposition_text, sentence),
        )
        groups.setdefault(key, []).append({
            "target_id": clean_text(row.get("query_proposition_id", "")),
            "source": clean_text(row.get("proposition_source", "")),
            "source_sentence": sentence,
            "proposition_text": proposition_text or sentence,
            "signature": signature,
            "enrichment": clean_text(
                row.get("representation_enrichment_json", "")
            ),
        })
    rows: List[Dict[str, Any]] = []
    fields = (
        "actor", "patient_or_theme", "predicate", "recipient",
        "destination", "assertion_polarity", "event_time",
    )
    for (fragment_id, _sentence_key), records in sorted(groups.items()):
        sentence = records[0]["source_sentence"]
        proposition_text = next(
            (
                clean_text(record.get("proposition_text", ""))
                for record in records
                if clean_text(record.get("proposition_text", ""))
            ),
            sentence,
        )
        canonical = build_event_signature(proposition_text or sentence)
        disagreements = 0
        resolved_conflicts = 0
        for field in fields:
            observed_raw = {
                clean_text(record["signature"].get(field, ""))
                for record in records
                if clean_text(record["signature"].get(field, ""))
            }
            observed = {
                _role_equivalence_key(value)
                for value in observed_raw
                if _role_equivalence_key(value)
            }
            if len(observed) > 1:
                disagreements += 1
            elif len(observed_raw) > 1 and len(observed) == 1:
                resolved_conflicts += 1
        passive_actor_conflict = bool(
            canonical.get("voice") == "passive"
            and canonical.get("patient_or_theme")
            and canonical.get("actor")
            and canonical_key(canonical.get("actor"))
            == canonical_key(canonical.get("patient_or_theme"))
        )
        unresolved = disagreements + int(passive_actor_conflict)
        representation_count = len(records)
        reference = records[0]["signature"]
        compatible_count = 0
        for record in records[1:]:
            compatible = True
            for field in fields:
                left = clean_text(reference.get(field, ""))
                right = clean_text(record["signature"].get(field, ""))
                if (
                    left and right
                    and _role_equivalence_key(left)
                    != _role_equivalence_key(right)
                ):
                    compatible = False
                    break
            compatible_count += int(compatible)
        enrichment_payloads: List[Dict[str, Any]] = []
        for record in records:
            if not record["enrichment"]:
                continue
            try:
                enrichment_payloads.append(
                    json.loads(record["enrichment"])
                )
            except json.JSONDecodeError:
                continue
        enriched_fields = sorted({
            field
            for payload in enrichment_payloads
            for field in ("event_time", "location")
            if clean_text(payload.get(field, ""))
        })
        enrichment_sources = sorted({
            clean_text(payload.get("source_row_id", ""))
            for payload in enrichment_payloads
            if clean_text(payload.get("source_row_id", ""))
        })
        for field in ("event_time", "location"):
            if clean_text(canonical.get(field, "")):
                continue
            canonical[field] = next(
                (
                    clean_text(payload.get(field, ""))
                    for payload in enrichment_payloads
                    if clean_text(payload.get(field, ""))
                ),
                "",
            )
        rows.append({
            "fragment_id": fragment_id,
            "target_id": records[0]["target_id"],
            "source_sentence": sentence,
            "representation_target_ids_json": json.dumps(
                sorted({
                    record["target_id"] for record in records
                    if record["target_id"]
                }),
                ensure_ascii=False,
            ),
            "representation_sources_json": json.dumps(
                sorted({
                    record["source"] for record in records
                    if record["source"]
                }),
                ensure_ascii=False,
            ),
            "representation_count": representation_count,
            "agreement_count": compatible_count,
            "resolved_conflict_count": resolved_conflicts,
            "unresolved_conflict_count": unresolved,
            "actor": clean_text(canonical.get("actor", "")),
            "patient_or_theme": clean_text(
                canonical.get("patient_or_theme", "")
            ),
            "predicate": clean_text(canonical.get("predicate", "")),
            "recipient": clean_text(canonical.get("recipient", "")),
            "destination": clean_text(canonical.get("destination", "")),
            "polarity": clean_text(
                canonical.get("assertion_polarity", "")
            ),
            "event_time": clean_text(canonical.get("event_time", "")),
            "enriched_fields_json": json.dumps(enriched_fields),
            "enrichment_sources_json": json.dumps(enrichment_sources),
            "reconciliation_status": (
                "UNRESOLVED_ROLE_CONFLICT"
                if unresolved else "RECONCILED"
            ),
            "reconciliation_reason": (
                "incompatible_typed_roles_across_representations"
                if unresolved else
                "typed_roles_agree_under_shared_normalization"
            ),
        })
    return ensure_columns(
        pd.DataFrame(rows), REPRESENTATION_RECONCILIATION_COLUMNS
    )


def build_query_task_preflight(parsed: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Create parser-side task-input validation before the matcher loads corpus."""
    rows: List[Dict[str, Any]] = []
    reconciliation = parsed.get(
        "representation_reconciliation", pd.DataFrame()
    )

    def reconciliation_counts(
        fragment_id: str,
        *,
        target_id: str = "",
        target_predicate: str = "",
    ) -> Dict[str, int]:
        subset = reconciliation
        if (
            isinstance(subset, pd.DataFrame)
            and not subset.empty
            and "fragment_id" in subset.columns
        ):
            subset = subset[
                subset["fragment_id"].fillna("").astype(str).eq(fragment_id)
            ]
        if (
            isinstance(subset, pd.DataFrame)
            and not subset.empty
            and target_id
            and "representation_target_ids_json" in subset.columns
        ):
            subset = subset[
                subset["representation_target_ids_json"]
                .fillna("").astype(str)
                .map(lambda value: target_id in value)
            ]
        elif (
            isinstance(subset, pd.DataFrame)
            and not subset.empty
            and target_predicate
            and "predicate" in subset.columns
        ):
            predicate_key = canonical_predicate(target_predicate)
            subset = subset[
                subset["predicate"].fillna("").astype(str)
                .map(canonical_predicate).eq(predicate_key)
            ]
        else:
            # A fragment-level disagreement is not automatically relevant to a
            # missing sentence or to an unrelated target in that fragment.
            subset = subset.iloc[0:0] if isinstance(
                subset, pd.DataFrame
            ) else subset
        if not isinstance(subset, pd.DataFrame) or subset.empty:
            return {
                "cross_representation_agreement_count": 0,
                "resolved_conflict_count": 0,
                "unresolved_conflict_count": 0,
            }
        return {
            "cross_representation_agreement_count": int(
                pd.to_numeric(
                    subset["agreement_count"], errors="coerce"
                ).fillna(0).sum()
            ),
            "resolved_conflict_count": int(
                pd.to_numeric(
                    subset["resolved_conflict_count"], errors="coerce"
                ).fillna(0).sum()
            ),
            "unresolved_conflict_count": int(
                pd.to_numeric(
                    subset["unresolved_conflict_count"], errors="coerce"
                ).fillna(0).sum()
            ),
        }
    for _, slot in parsed.get("sentence_redaction_slots", pd.DataFrame()).iterrows():
        left_count = int(slot.get("left_context_semantic_token_count", 0) or 0)
        right_count = int(slot.get("right_context_semantic_token_count", 0) or 0)
        if left_count and right_count:
            status = "EXECUTABLE_AFTER_REPAIR" if bool(slot.get("boundary_repair_applied", False)) else "EXECUTABLE"
            reason = clean_text(slot.get("boundary_repair_reason", "")) or "valid_two_sided_context"
        elif left_count or right_count:
            status = "EXECUTABLE_ONE_SIDED_IF_EXPLICITLY_ENABLED"
            reason = "one meaningful context side is present"
        else:
            status = "BLOCKED_MISSING_CONTEXT"
            reason = "no meaningful sentence context on either side"
        rows.append({
            "task": "sentence_redaction",
            "target_id": clean_text(slot.get("sentence_slot_label", "")),
            "status": status,
            "reason": reason,
            "repair_applied": bool(slot.get("boundary_repair_applied", False)),
            "left_semantic_tokens": left_count,
            "right_semantic_tokens": right_count,
            "normalized_context_sha256": clean_text(slot.get("normalized_slot_context_sha256", "")),
            **reconciliation_counts(
                clean_text(slot.get("fragment_id", "")),
            ),
        })
    for _, slot in parsed.get("redaction_slots", pd.DataFrame()).iterrows():
        target_predicate = clean_text(
            slot.get("governing_predicate_normalized", "")
        ) or clean_text(slot.get("governing_predicate", ""))
        rows.append({
            "task": "redaction", "target_id": clean_text(slot.get("slot_label", "")),
            "status": "EXECUTABLE", "reason": clean_text(slot.get("slot_frame", "generic_value_slot")),
            "repair_applied": bool(slot.get("redaction_format_repaired", False)),
            "left_semantic_tokens": len(semantic_tokens(slot.get("local_context_before", ""))),
            "right_semantic_tokens": len(semantic_tokens(slot.get("local_context_after", ""))),
            "normalized_context_sha256": "",
            **reconciliation_counts(
                clean_text(slot.get("fragment_id", "")),
                target_predicate=target_predicate,
            ),
        })
    if len(parsed.get("propositions", pd.DataFrame())):
        for _, prop in parsed["propositions"].iterrows():
            rows.append({
                "task": "claims_or_contradictions",
                "target_id": clean_text(prop.get("query_proposition_id", "")),
                "status": "EXECUTABLE", "reason": clean_text(prop.get("proposition_source", "")),
                "repair_applied": False, "left_semantic_tokens": "", "right_semantic_tokens": "",
                "normalized_context_sha256": "",
                **reconciliation_counts(
                    clean_text(prop.get("fragment_id", "")),
                    target_id=clean_text(
                        prop.get("query_proposition_id", "")
                    ),
                ),
            })
    frame = ensure_columns(pd.DataFrame(rows), TASK_PREFLIGHT_COLUMNS)
    if len(frame):
        conflicts = pd.to_numeric(
            frame["unresolved_conflict_count"], errors="coerce"
        ).fillna(0).gt(0)
        frame.loc[conflicts, "status"] = (
            "BLOCKED_UNRESOLVED_DISCRIMINATIVE_ROLE_CONFLICT"
        )
        frame.loc[conflicts, "reason"] = (
            "claim_relation_event_role_reconciliation_failed"
        )
    return frame

def write_parse_outputs(parsed: Dict[str, pd.DataFrame], output_dir: Path, query_file: str, parser_mode: str, warnings: List[str]) -> QueryParseResult:
    """Write primary parse outputs plus v0.3.18 debug tables.

    Primary CSVs have the historical filenames used by downstream notebooks:
      query_entities.csv, query_claims.csv, query_events.csv, query_relations.csv

    In v0.3.18 these primary CSVs are intentionally cleaner. In OpenAI mode with
    fallback_only, heuristic rows are absent from these primary tables unless the
    OpenAI extractor produced zero rows for that specific fragment/table.

    Full heuristic and OpenAI rows are still written to *_debug.csv files so a
    developer can audit what was suppressed or promoted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "fragments": output_dir / "query_fragments.csv",
        "entities": output_dir / "query_entities.csv",
        "claims": output_dir / "query_claims.csv",
        "events": output_dir / "query_events.csv",
        "relations": output_dir / "query_relations.csv",
        "propositions": output_dir / "query_propositions.csv",
        "redaction_slots": output_dir / "query_redaction_slots.csv",
        "sentence_redaction_slots": output_dir / "query_sentence_redaction_slots.csv",
        "event_signatures": output_dir / "query_event_signatures.csv",
        "representation_reconciliation":
            output_dir / "query_representation_reconciliation.csv",
        "task_preflight": output_dir / "query_task_preflight.csv",
    }
    # Derived canonical tables are rebuilt from the final primary rows so they
    # remain consistent in heuristic, OpenAI, and fallback modes.
    parsed["event_signatures"] = build_query_event_signatures(parsed)
    parsed["representation_reconciliation"] = (
        build_representation_reconciliation(parsed)
    )
    parsed["task_preflight"] = build_query_task_preflight(parsed)

    column_map = {
        "fragments": FRAGMENT_COLUMNS,
        "entities": ENTITY_COLUMNS,
        "claims": CLAIM_COLUMNS,
        "events": EVENT_COLUMNS,
        "relations": RELATION_COLUMNS,
        "propositions": PROPOSITION_COLUMNS,
        "redaction_slots": REDACTION_SLOT_COLUMNS,
        "sentence_redaction_slots": SENTENCE_REDACTION_SLOT_COLUMNS,
        "event_signatures": EVENT_SIGNATURE_COLUMNS,
        "representation_reconciliation":
            REPRESENTATION_RECONCILIATION_COLUMNS,
        "task_preflight": TASK_PREFLIGHT_COLUMNS,
    }
    for key, path in files.items():
        ensure_columns(parsed.get(key, pd.DataFrame()), column_map[key]).to_csv(path, index=False)

    # v0.3.18 debug/audit outputs. These are not primary matching inputs.
    debug_files: Dict[str, Path] = {}

    suppressed_entities_path = output_dir / "query_entities_suppressed_debug.csv"
    ensure_columns(parsed.get("entity_suppressed_debug", pd.DataFrame()), ENTITY_COLUMNS).to_csv(suppressed_entities_path, index=False)
    debug_files["entities_suppressed_debug"] = suppressed_entities_path

    suppressed_events_path = output_dir / "query_events_suppressed_debug.csv"
    ensure_columns(parsed.get("event_suppressed_debug", pd.DataFrame()), list(EVENT_COLUMNS) + ["suppression_reason", "suppression_source_note"]).to_csv(suppressed_events_path, index=False)
    debug_files["events_suppressed_debug"] = suppressed_events_path
    for debug_group, prefix in [("heuristic_debug", "heuristic"), ("openai_debug", "openai")]:
        group = parsed.get(debug_group, {})
        if not isinstance(group, dict):
            group = {}
        for key, cols in [("entities", ENTITY_COLUMNS), ("claims", CLAIM_COLUMNS), ("events", EVENT_COLUMNS), ("relations", RELATION_COLUMNS)]:
            path = output_dir / f"query_{key}_{prefix}_debug.csv"
            ensure_columns(group.get(key, pd.DataFrame()), cols).to_csv(path, index=False)
            debug_files[f"{key}_{prefix}_debug"] = path

    # Compact count table for quick sanity checks inside Colab.
    count_rows: List[Dict[str, Any]] = []
    for key in ["entities", "claims", "events", "relations"]:
        primary_df = ensure_columns(parsed.get(key, pd.DataFrame()), column_map[key])
        h_df = ensure_columns(parsed.get("heuristic_debug", {}).get(key, pd.DataFrame()) if isinstance(parsed.get("heuristic_debug"), dict) else pd.DataFrame(), column_map[key])
        o_df = ensure_columns(parsed.get("openai_debug", {}).get(key, pd.DataFrame()) if isinstance(parsed.get("openai_debug"), dict) else pd.DataFrame(), column_map[key])
        count_rows.append({
            "table": key,
            "primary_rows": int(len(primary_df)),
            "primary_openai_rows": int((primary_df.get("parser_source", pd.Series(dtype=str)).astype(str) == "openai").sum()) if len(primary_df) else 0,
            "primary_heuristic_rows": int((primary_df.get("parser_source", pd.Series(dtype=str)).astype(str) == "heuristic").sum()) if len(primary_df) else 0,
            "heuristic_debug_rows": int(len(h_df)),
            "openai_debug_rows": int(len(o_df)),
            "suppressed_redaction_placeholder_rows": (
                int(len(parsed.get("entity_suppressed_debug", pd.DataFrame()))) if key == "entities"
                else int(len(parsed.get("event_suppressed_debug", pd.DataFrame()))) if key == "events"
                else 0
            ),
        })
    counts_path = output_dir / "query_parse_primary_debug_counts.csv"
    pd.DataFrame(count_rows).to_csv(counts_path, index=False)
    telemetry_path = output_dir / "query_openai_telemetry_v0410.csv"
    ensure_columns(
        pd.DataFrame(_OPENAI_TELEMETRY_ROWS),
        OPENAI_TELEMETRY_COLUMNS,
    ).to_csv(telemetry_path, index=False)

    result = QueryParseResult(
        output_dir=str(output_dir),
        query_file=query_file,
        parser_mode=parser_mode,
        fragment_count=int(len(parsed.get("fragments", pd.DataFrame()))),
        entity_count=int(len(parsed.get("entities", pd.DataFrame()))),
        claim_count=int(len(parsed.get("claims", pd.DataFrame()))),
        event_count=int(len(parsed.get("events", pd.DataFrame()))),
        relation_count=int(len(parsed.get("relations", pd.DataFrame()))),
        proposition_count=int(len(parsed.get("propositions", pd.DataFrame()))),
        redaction_slot_count=int(len(parsed.get("redaction_slots", pd.DataFrame()))),
        sentence_redaction_slot_count=int(len(parsed.get("sentence_redaction_slots", pd.DataFrame()))),
        output_files={k: str(v) for k, v in files.items()},
        warnings=warnings,
    )
    result.output_files.update({k: str(v) for k, v in debug_files.items()})
    result.output_files["primary_debug_counts"] = str(counts_path)
    result.output_files["openai_telemetry"] = str(telemetry_path)

    manifest = asdict(result)
    manifest["created_at"] = now_utc_iso()
    manifest["version"] = VERSION
    manifest["parser_implementation_version"] = VERSION
    manifest["fragmentation_policy"] = "paragraph_first; DID markers captured as metadata only; DID markers not used for chunking"
    manifest["openai_key_policy"] = "Read API key from environment variable; do not hard-code secrets in this file."
    manifest["heuristic_policy_contract"] = (
        "fallback_only means primary CSVs for entities/claims/events use OpenAI rows when available and heuristic rows only "
        "when OpenAI produced zero rows for that fragment/table. Full heuristic rows are written to *_heuristic_debug.csv."
    )
    manifest["relation_policy_contract"] = (
        "By default, heuristic co-occurrence relations are debug-only and do not enter query_relations.csv as primary rows. "
        "Pass --allow-heuristic-relations-primary only for recall experiments, not for claim/contradiction matching."
    )
    manifest["entity_hygiene_contract"] = (
        "Redaction placeholders are never primary entities. Value redactions are stored in query_redaction_slots.csv; sentence/span redactions are stored in query_sentence_redaction_slots.csv. If placeholders are suppressed from entity output, they are written to query_entities_suppressed_debug.csv."
    )
    manifest["sentence_redaction_contract"] = (
        "v0.3.20 supports [REDACT_SENT:S1 type=CLAIM approx_tokens=10] and [REDACT_SPAN:S1 type=ANY approx_tokens=12]. These rows are parsed only as missing sentence/proposition slots; the parser never reconstructs the missing sentence."
    )
    manifest["supports_sentence_redaction"] = True
    manifest["sentence_redaction_slot_count"] = int(len(parsed.get("sentence_redaction_slots", pd.DataFrame())))
    manifest["redaction_format_repairs_count"] = int(
        pd.to_numeric(ensure_columns(parsed.get("redaction_slots", pd.DataFrame()), REDACTION_SLOT_COLUMNS).get("redaction_format_repaired", pd.Series(dtype=bool)), errors="coerce").fillna(0).astype(bool).sum()
        + pd.to_numeric(ensure_columns(parsed.get("sentence_redaction_slots", pd.DataFrame()), SENTENCE_REDACTION_SLOT_COLUMNS).get("redaction_format_repaired", pd.Series(dtype=bool)), errors="coerce").fillna(0).astype(bool).sum()
    )
    manifest["claim_contradiction_contract"] = (
        "Use query_propositions.csv for generic claim search and contradiction discovery. It is derived from primary claims and asserted primary relations only. "
        "In OpenAI mode, OpenAI claim rows are primary. A deterministic sentence claim is promoted only for a fragment where OpenAI emitted no claim; --allow-heuristic-claims-primary enables the broader configured policy."
    )
    manifest["event_hygiene_contract"] = (
        "Redaction-placeholder event rows are never primary events. They are written to query_events_suppressed_debug.csv. "
        "In OpenAI mode, heuristic event rows are excluded from primary query_events.csv by default unless --allow-heuristic-events-primary is passed."
    )
    manifest["matcher_contract"] = "Downstream semantic matcher should use primary CSV rows, preferably filtering primary_for_matching == true; use query_propositions.csv for claim/contradiction workflows."
    manifest["warning"] = "Semantic parses are candidate evidence and should be audited before final use."
    manifest["requirements_version"] = REQUIREMENTS_VERSION
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["normalization_version"] = NORMALIZATION_VERSION
    manifest["prompt_version"] = PROMPT_VERSION
    manifest["heuristic_policy_version"] = HEURISTIC_POLICY_VERSION
    manifest["model"] = parser_mode.split(":", 1)[-1] if "openai" in parser_mode else "deterministic_heuristic"
    manifest["query_sha256"] = sha256_file(Path(query_file))
    manifest["release_stage"] = "coordinated_release"
    manifest["table_sha256"] = {
        key: sha256_file(path) for key, path in files.items() if path.exists()
    }
    manifest["query_parse_sha256"] = v19_stable_hash({
        "query_sha256": manifest["query_sha256"],
        "parser_version": VERSION,
        "model": manifest["model"],
        "prompt_version": PROMPT_VERSION,
        "heuristic_policy": HEURISTIC_POLICY_VERSION,
        "normalization_version": manifest["normalization_version"],
        "schema_version": SCHEMA_VERSION,
        "parser_mode": parser_mode,
        "tables": manifest["table_sha256"],
    })
    manifest["frozen_parse_contract"] = {
        "reusable_without_parser": True,
        "immutable_tables": sorted(manifest["table_sha256"]),
        "fingerprint_fields": [
            "query_sha256", "parser_version", "model", "prompt_version",
            "heuristic_policy", "normalization_version", "schema_version",
            "table_sha256",
        ],
    }
    manifest["sentence_redaction_contract"] = (
        "v0.3.20 preserves original context, ignores punctuation-only neighbors, "
        "writes normalized context/hash, and never reconstructs the missing sentence."
    )
    manifest["claim_contradiction_contract"] = (
        "query_propositions.csv and query_event_signatures.csv contain canonical event/state, "
        "numeric, date, polarity, epistemic, and provenance fields."
    )
    manifest_path = output_dir / "query_parse_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result.output_files["manifest"] = str(manifest_path)
    return result

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse query documents into semantic fragments and KG-like rows.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI extraction per deterministic fragment.")
    parser.add_argument("--openai-model", default="gpt-4o")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the OpenAI API key.")
    parser.add_argument(
        "--heuristic-policy",
        choices=["fallback_only", "openai_only", "merge"],
        default="fallback_only",
        help=(
            "How to use heuristic rows when --use-openai is set. "
            "fallback_only is recommended: use OpenAI rows when present and heuristic rows only for fragment/table gaps. "
            "openai_only suppresses heuristic rows completely. merge reproduces v0.3.14-style combined outputs."
        ),
    )
    parser.add_argument(
        "--allow-heuristic-relations-primary",
        action="store_true",
        help=(
            "Allow heuristic co-occurrence relations to enter query_relations.csv as primary rows. "
            "Default is safer: keep such rows in query_relations_heuristic_debug.csv only."
        ),
    )
    parser.add_argument(
        "--allow-heuristic-claims-primary",
        action="store_true",
        help=(
            "Allow heuristic sentence-level claims to enter query_claims.csv/query_propositions.csv in OpenAI mode. "
            "Default is safer for claim search and contradiction discovery: use OpenAI claims only when OpenAI is available."
        ),
    )
    parser.add_argument(
        "--allow-heuristic-events-primary",
        action="store_true",
        help=(
            "Allow heuristic event rows to enter query_events.csv in OpenAI mode. "
            "Default is safer: use OpenAI event rows only and keep heuristic events in query_events_heuristic_debug.csv; "
            "redaction-placeholder events are still suppressed."
        ),
    )
    parser.add_argument("--max-chars-per-fragment", type=int, default=2200, help="Split only paragraphs longer than this size.")
    parser.add_argument("--debug-tracebacks", action="store_true", help="Print exception tracebacks for debugging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    query_path = Path(args.query_file)
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")
    text = query_path.read_text(encoding="utf-8", errors="ignore")
    try:
        parsed, parser_mode, warnings = parse_query_text(
            text,
            use_openai=args.use_openai,
            openai_model=args.openai_model,
            api_key_env=args.openai_api_key_env,
            max_chars_per_fragment=args.max_chars_per_fragment,
            heuristic_policy=args.heuristic_policy,
            allow_heuristic_relations_primary=args.allow_heuristic_relations_primary,
            allow_heuristic_claims_primary=args.allow_heuristic_claims_primary,
            allow_heuristic_events_primary=args.allow_heuristic_events_primary,
        )
    except Exception as exc:
        if args.debug_tracebacks:
            traceback.print_exc()
        raise exc
    result = write_parse_outputs(parsed, Path(args.output_dir), str(query_path), parser_mode, warnings)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
