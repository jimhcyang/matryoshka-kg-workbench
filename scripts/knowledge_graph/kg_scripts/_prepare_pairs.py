#!/usr/bin/env python3
"""
Stage A0: prepare a redaction-aware project from CIA transcription pairs.

The aligned bracketed difference file is retained as a debug artifact and
redaction reference, but the prepared unredacted/redacted corpora are built
from the literal `document/*.transcription.txt` variants so redacted runs do
not accidentally inherit unredacted text.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


PAIR_METADATA_COLUMNS = [
    "collection",
    "record_id",
    "date_display",
    "source_pdf_url",
    "date_compact",
    "cia_document_id",
    "document_url",
    "meta_path",
    "title",
    "year",
    "paired_flag",
]

PAGE_TAG_RE = re.compile(r"(\[PAGE\s+\d+\])", flags=re.IGNORECASE)
RED_TAG_RE = re.compile(
    r"\[\[CLEAN_REDACTION_(\d+)\]\](.*?)\[\[/CLEAN_REDACTION_\1\]\]",
    flags=re.DOTALL | re.IGNORECASE,
)
SENTINEL_RE = re.compile(r"__CLEAN_REDACTION_(\d{4,})__")
MONTH_TOKEN = r"(?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)"
SERIAL_HEADER_RE = re.compile(r"^((?:[IVXLCM]+|[A-Z]|\d+)[.)])\s+(.+)$", re.IGNORECASE)
SPACED_ALL_CAPS_RE = re.compile(r"\b(?:[A-Z]\s+){2,}[A-Z]\b")

DEFAULT_GAP_MARKER = "[REDACTED]"

EXPLICIT_DROP_LINE_PATTERNS = [
    re.compile(r"^\[PAGE\s+\d+\]$", re.IGNORECASE),
    re.compile(r"^CENTRAL\s+INTELLIGENCE\s+BULLETIN$", re.IGNORECASE),
    re.compile(r"^CURRENT\s+INTELLIGENCE\s+BULLETIN$", re.IGNORECASE),
    re.compile(r"^CENTRAL\s+INTELLIGENCE\s+BULLETIN\s+PAGE\s+\d+$", re.IGNORECASE),
    re.compile(r"^CURRENT\s+INTELLIGENCE\s+BULLETIN\s+PAGE\s+\d+$", re.IGNORECASE),
    re.compile(r"^CENTRAL$", re.IGNORECASE),
    re.compile(r"^INTELLIGENCE$", re.IGNORECASE),
    re.compile(r"^BULLETIN$", re.IGNORECASE),
    re.compile(r"^DAILY\s+BRIEF$", re.IGNORECASE),
    re.compile(r"^DAILY\s+BRIEF\s+[ivxlcdm]+$", re.IGNORECASE),
    re.compile(r"^DAILY$", re.IGNORECASE),
    re.compile(r"^BRIEF$", re.IGNORECASE),
    re.compile(r"^TOP\s+SECRET$", re.IGNORECASE),
    re.compile(r"^SECRET$", re.IGNORECASE),
    re.compile(r"^CONFIDENTIAL$", re.IGNORECASE),
    re.compile(r"^COPY\s+NO\.?(?:\s+.*)?$", re.IGNORECASE),
    re.compile(rf"^\d{{1,2}}\s+{MONTH_TOKEN}\s+\d{{2}}\s+CENTRAL\s+INTELLIGENCE\s+BULLETIN\s+PAGE\s+\d+$", re.IGNORECASE),
    re.compile(rf"^\d{{1,2}}\s+{MONTH_TOKEN}\s+\d{{2}}\s+DAILY\s+BRIEF\s+[ivxlcdm]+$", re.IGNORECASE),
    re.compile(rf"^\d{{1,2}}\s+{MONTH_TOKEN}\s+\d{{2}}\s+DAILY\s+BRIEF$", re.IGNORECASE),
    re.compile(rf"^\d{{1,2}}\s+{MONTH_TOKEN}\s+\d{{4}}$", re.IGNORECASE),
    re.compile(rf"^\d{{1,2}}\s+{MONTH_TOKEN}\s+\d{{2}}$", re.IGNORECASE),
    re.compile(rf"^\d{{1,2}}\s+{MONTH_TOKEN}\s+DAILY\s+BRIEF(?:\s+[ivxlcdm]+)?$", re.IGNORECASE),
    re.compile(rf"^\d{{1,2}}\s+{MONTH_TOKEN}\s+\d{{4}}\s+DAILY\s+BRIEF(?:\s+[ivxlcdm]+)?$", re.IGNORECASE),
    re.compile(rf"^\d{{1,2}}\s+{MONTH_TOKEN}\s+\d{{2}}\s+CENTRAL\s+INTELLIGENCE\s+BULLETIN$", re.IGNORECASE),
    re.compile(r"^NEXT\s+\d+\s+PAGE\(S\)\s+IN\s+DOCUMENT\s+EXEMPT$", re.IGNORECASE),
    re.compile(r"^STATE\s+DEPT\.?\s+REVIEW\s+COMPLETED\.?$", re.IGNORECASE),
    re.compile(r"^DIA\s+AND\s+DOS\s+REVIEW\(S\)\s+COMPLETED\.?$", re.IGNORECASE),
    re.compile(r"^UNCLASSIFIED$", re.IGNORECASE),
    re.compile(r"^PAGE\s+\d+$", re.IGNORECASE),
    re.compile(r"^[ivxlcdm]{1,6}$", re.IGNORECASE),
    re.compile(r"^\(MAP\)$", re.IGNORECASE),
    re.compile(r"^25X\d+[A-Z]?$", re.IGNORECASE),
    re.compile(r"^STATUTE\s+MILES.*$", re.IGNORECASE),
    re.compile(r"^\d+$"),
]

INLINE_PAGE_REFERENCE_RE = re.compile(
    r"\(\s*(?:Backup,\s*)?Page\s*[\.:]?\s*(?:\d+|[ivxlcdm]+)\s*\)",
    re.IGNORECASE,
)
INLINE_NO_BACKUP_RE = re.compile(r"\(\s*No\s+Backup\s*\)", re.IGNORECASE)
INLINE_MAP_REFERENCE_RE = re.compile(r"\(\s*Map\s*\)", re.IGNORECASE)
MAJOR_SECTION_RE = re.compile(r"^LATE\s+ITEM$", re.IGNORECASE)
NONSERIAL_MAJOR_SECTION_PATTERNS = [
    re.compile(r"^(?:THE\s+)?COMMUNIST\s+BLOC$", re.IGNORECASE),
    re.compile(r"^ASIA[-\s]+AFRICA$", re.IGNORECASE),
    re.compile(r"^(?:THE\s+)?WEST$", re.IGNORECASE),
    re.compile(r"^SPECIAL\s+ITEM$", re.IGNORECASE),
    re.compile(r"^WATCH\s+COMMITTEE\s+CONCLUSIONS$", re.IGNORECASE),
]
TAIL_START_PATTERNS = [
    re.compile(r"^THE\s+PRESIDENT$", re.IGNORECASE),
    re.compile(r"^THE\s+VICE\s+PRESIDENT$", re.IGNORECASE),
]
MAP_CUE_PATTERNS = [
    re.compile(r"^SELECTED\s+ROADS$", re.IGNORECASE),
    re.compile(r"^STATUTE\s+MILES(?:\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^MAP$", re.IGNORECASE),
    re.compile(r"^LEGEND$", re.IGNORECASE),
    re.compile(r"^BOUNDARY\s+LINE\b.*$", re.IGNORECASE),
    re.compile(r"^PROPOSED\s+BOUNDARY\s+AGREEMENT\b.*$", re.IGNORECASE),
]
CONTENTS_HEADING_RE = re.compile(r"^CONTENTS$", re.IGNORECASE)
CONTENTS_PREFIX_RE = re.compile(r"^(?:Copy(?:\s+No\.)?\s+\S+\s+)?CONTENTS\b", re.IGNORECASE)
TRAILING_SOURCE_METADATA_RE = re.compile(
    r"\bCurrent/Central Intelligence Bulletin Collection\b.*$",
    re.IGNORECASE,
)
MAP_PAGE_PREFIX_RE = re.compile(
    r"^.*?\b(?:CENTRAL|CURRENT)\s+INTELLIGENCE\s+BULLETIN\s+Map\s+Page\b\s*",
    re.IGNORECASE,
)
BULLETIN_PAGE_PREFIX_RE = re.compile(
    r"^\d{1,2}\s+[A-Z][a-z]{2}\s+\d{2}\s+(?:CENTRAL|CURRENT)\s+INTELLIGENCE\s+BULLETIN\s+Page\s+\w+\s*",
    re.IGNORECASE,
)
MAP_PAGE_HEADING_TITLE_RE = re.compile(
    r"^.*?\b(?:CENTRAL|CURRENT)\s+INTELLIGENCE\s+BULLETIN\s+Map\s+Page\s+(.+)$",
    re.IGNORECASE,
)
MAP_TRAILING_TOPIC_RE = re.compile(
    r".*\b(Situation in [A-Z][A-Za-z' -]+|Sino-Indian Border Dispute Remains Deadlocked)\s*$"
)
MAP_TRAILING_TITLE_RE = re.compile(
    r".*\b([A-Z][A-Za-z'&/-]*(?:\s+(?:in|of|the|and|to|for|on|with|against|over|under|from|at|[A-Z][A-Za-z'&/-]*)){1,10})\s*$"
)
ESTIMATE_TITLE_RE = re.compile(
    r"^Estimate of the World Situation:.*?\b([A-Z][a-z0-9'()/,&.-]*(?:\s+[A-Z][a-z0-9'()/,&.-]*){2,18})\s+(The\b.*)$"
)
FRONT_MATTER_PREFIX_PATTERNS = [
    re.compile(
        r"^The Daily Brief of the (?:CENTRAL|CURRENT) INTELLIGENCE BULLETIN is produced\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^WARNING This document contains classified information\b",
        re.IGNORECASE,
    ),
]
MAP_LEGEND_PREFIX_RES = [
    re.compile(
        r"^Road Trail(?: [A-Za-z][A-Za-z'/-]*)*(?: 0 25 50 75 100 Miles 0 25 50 75 100 Kilometers)?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^LAOS Government forces Government Antigovernment Antigovernment concentrations Kong Le/Pathet Lao forces Main route number\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^Boundary representation is not necessarily authoritative\.?\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^\(Charts\)\s*", re.IGNORECASE),
    re.compile(
        r"^(?:Railroad\s+)?Road Trail(?:\s+Boundaries are not necessarily those recognized by the U\.S\. Government\.)?\s*",
        re.IGNORECASE,
    ),
]
MAP_LEGEND_SUFFIX_RES = [
    re.compile(
        r"\bGovernment forces Government Antigovernment(?: Kong Le/Pathet Lao forces)? Main route number Road Trail Antigovernment concentrations\s*$",
        re.IGNORECASE,
    ),
]
MAP_LEGEND_SIGNAL_PATTERNS = [
    re.compile(r"\bRoad Trail\b", re.IGNORECASE),
    re.compile(r"\bRailroad\b", re.IGNORECASE),
    re.compile(r"\bSelected road\b", re.IGNORECASE),
    re.compile(r"\bSelected railroad\b", re.IGNORECASE),
    re.compile(r"\bSelected airfield\b", re.IGNORECASE),
    re.compile(r"\bGovernment forces\b", re.IGNORECASE),
    re.compile(r"\bAntigovernment\b", re.IGNORECASE),
    re.compile(r"\bKong Le\s*[-/]\s*Pathet Lao\b", re.IGNORECASE),
    re.compile(r"\bRoute number\b", re.IGNORECASE),
    re.compile(r"\bApproximate area nominally controlled by\b", re.IGNORECASE),
    re.compile(r"\bUnited Nations Forces\b", re.IGNORECASE),
    re.compile(r"\bChinese Nationalist Irregulars\b", re.IGNORECASE),
    re.compile(r"\bMap Page\b", re.IGNORECASE),
    re.compile(r"\bBoundaries are not necessarily those recognized by the U\.S\. Government\b", re.IGNORECASE),
]
TEXT_REPAIR_PREFIX_RES = [
    (
        ESTIMATE_TITLE_RE,
        lambda match: f"{norm_space(match.group(1))} {norm_space(match.group(2))}",
    ),
    (
        re.compile(r"^North Vietnam South Vietnam\s+(North Vietnam\b.*)$", re.IGNORECASE),
        lambda match: norm_space(match.group(1)),
    ),
]
MAP_ANCHOR_RE = re.compile(
    r"(\*?\s*(?:The\s+Situation in [A-Z][A-Za-z' -]+|Situation in [A-Z][A-Za-z' -]+|[A-Z][A-Za-z0-9'()/,&.-]*(?:\s+[A-Z][A-Za-z0-9'()/,&.-]*){0,10}):\s+)"
)
PURE_MAP_TEXT_PATTERNS = [
    re.compile(r"Government Antigovernment concentrations", re.IGNORECASE),
    re.compile(r"0 25 50 75 100 Miles 0 25 50 75 100 Kilometers", re.IGNORECASE),
    re.compile(r"Boundary representation is not necessarily authoritative", re.IGNORECASE),
]
CONTINUATION_HEADING_PATTERNS = [
    re.compile(r"^However,", re.IGNORECASE),
    re.compile(r"^school of the National University", re.IGNORECASE),
]
CONTINUATION_TEXT_PATTERNS = [
    re.compile(r"^(?:Hence|However|Nevertheless|Meanwhile|Accordingly|Therefore|Thus|Moreover|Elsewhere|At midday)\b", re.IGNORECASE),
]
LONG_SEGMENT_AUDIT_THRESHOLD = 2500

TITLE_CONNECTORS = {
    "a",
    "an",
    "and",
    "at",
    "against",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "over",
    "the",
    "to",
    "under",
    "with",
}

CARTOGRAPHIC_CUE_WORDS = {
    "airfield",
    "airfields",
    "boundary",
    "boundaries",
    "chart",
    "charts",
    "kilometer",
    "kilometers",
    "legend",
    "map",
    "mile",
    "miles",
    "railroad",
    "railroads",
    "road",
    "roads",
    "route",
    "routes",
    "scale",
    "selected",
    "statute",
    "trail",
}

GENERIC_MARKER_WORDS = {
    "contents",
    "summary",
}

REDUCTION_KEEP_REDACTION_COLUMNS = [
    "doc_id",
    "paragraph_id",
    "heading",
    "redaction_id",
    "clean_redaction_id",
    "decision",
    "row_kind",
    "page_tag",
    "aligned_page",
    "segment_id",
    "text",
    "redacted_text",
    "token_count",
    "word_count",
    "char_count",
    "char_start_0based",
    "char_end_0based",
    "unred_source_page",
    "red_source_page",
    "left_context",
    "right_context",
    "local_context",
    "bridge_context",
    "source_unredacted_text",
    "source_redacted_text",
    "redaction_kind",
    "recovery_keep",
    "recovery_reason",
]

REDACTION_REVIEW_COLUMNS = [
    "redaction_id",
    "decision",
    "redaction_kind",
    "word_count",
    "char_count",
    "redacted_text",
    "query_context",
    "doc_id",
    "paragraph_id",
    "heading",
    "recovery_reason",
]


def iter_progress(iterable, total: Optional[int] = None, desc: str = ""):
    if tqdm is None:
        return iterable
    kwargs = {"desc": desc}
    if total is not None:
        kwargs["total"] = total
    return tqdm(iterable, **kwargs)


def safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def norm_space(value: object) -> str:
    return re.sub(r"\s+", " ", safe_str(value)).strip()


REDACTION_TOKEN_RE = re.compile(r"\S+")
REDACTION_ALPHA_RE = re.compile(r"[A-Za-z]")
REDACTION_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*\s+")
REDACTION_NOISE_RE = re.compile(
    r"(?:"
    r"^copy\s+no\b|"
    r"^contents?$|"
    r"^summary$|"
    r"^watch committee conclusions$|"
    r"^page\s+\d+$|"
    r"^top secret$|"
    r"^secret$|"
    r"^confidential$|"
    r"^statute miles\b|"
    r"^route number\b|"
    r"^road trail\b|"
    r"^legend$|"
    r"^map$|"
    r"^chart(?:s)?$|"
    r"^daily brief$|"
    r"^central intelligence bulletin$|"
    r"^current intelligence bulletin$|"
    r"^source description$|"
    r"^leeann flynn hall$|"
    r"^electronic surveillance and physical search\b|"
    r"^signed\b|"
    r"^certification$|"
    r"^declaration$|"
    r"^verification$"
    r")",
    re.IGNORECASE,
)
REDACTION_EXPLICIT_DROP_PATTERNS = [
    re.compile(r"^\s*S//\s*$", re.IGNORECASE),
    re.compile(r"^\s*IN RE CARTER W\.?\s*PAGE,?\s*A U\.S\. PERSON\s*$", re.IGNORECASE),
    re.compile(r"\bFacilities Used by Page\b", re.IGNORECASE),
    re.compile(r"\bSource Descriptive\b", re.IGNORECASE),
    re.compile(r"\bPrevious Applications None\b", re.IGNORECASE),
    re.compile(r"\b1804\(c\)\s+and\s+1823\(c\)\b", re.IGNORECASE),
    re.compile(r"^\s*also\s+\(d\)\)\s*$", re.IGNORECASE),
    re.compile(r"\bSubject to Sequestration\b", re.IGNORECASE),
]


def redaction_token_matches(text: str) -> List[re.Match[str]]:
    return list(REDACTION_TOKEN_RE.finditer(text))


def diff_redaction_spans(unredacted: str, redacted: str) -> List[Dict[str, object]]:
    u_matches = redaction_token_matches(unredacted)
    r_matches = redaction_token_matches(redacted)
    if not u_matches:
        return []
    u_tokens = [match.group(0) for match in u_matches]
    r_tokens = [match.group(0) for match in r_matches]
    matcher = difflib.SequenceMatcher(a=r_tokens, b=u_tokens, autojunk=False)
    spans: List[Dict[str, object]] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag not in {"insert", "replace"} or j1 >= j2:
            continue
        char_start = u_matches[j1].start()
        char_end = u_matches[j2 - 1].end()
        deleted_text = norm_space(unredacted[char_start:char_end])
        if not deleted_text:
            continue
        spans.append(
            {
                "token_start": j1,
                "token_end": j2,
                "char_start": char_start,
                "char_end": char_end,
                "text": deleted_text,
            }
        )
    return spans


def redaction_sentence_bounds(text: str, char_start: int, char_end: int) -> Tuple[int, int]:
    left = 0
    for match in REDACTION_SENTENCE_END_RE.finditer(text[:char_start]):
        left = match.end()
    right = len(text)
    next_match = REDACTION_SENTENCE_END_RE.search(text[char_end:])
    if next_match:
        right = char_end + next_match.end()
    return left, right


def redaction_previous_sentence(text: str, char_start: int) -> str:
    prev_end = 0
    prev_start = 0
    for match in REDACTION_SENTENCE_END_RE.finditer(text[:char_start]):
        prev_start = prev_end
        prev_end = match.end()
    return norm_space(text[prev_start:prev_end])


def redaction_next_sentence(text: str, char_end: int) -> str:
    remainder = text[char_end:]
    match = REDACTION_SENTENCE_END_RE.search(remainder)
    if match:
        return norm_space(remainder[: match.end()])
    return norm_space(remainder)


def redaction_context_slice(text: str, char_start: int, char_end: int, radius: int = 220) -> Tuple[str, str]:
    left_start = max(0, char_start - radius)
    while left_start > 0 and not text[left_start - 1].isspace():
        left_start -= 1
    right_end = min(len(text), char_end + radius)
    while right_end < len(text) and not text[right_end - 1].isspace():
        right_end += 1
    left = norm_space(text[left_start:char_start])
    right = norm_space(text[char_end:right_end])
    return left, right


def classify_redaction_span(
    text: str,
    left_context: str,
    right_context: str,
    bridge_context: str,
) -> Tuple[str, bool, str]:
    cleaned = norm_space(text)
    if not cleaned:
        return "discard", False, "empty"
    if not REDACTION_ALPHA_RE.search(cleaned):
        return "discard", False, "no_alpha"
    if REDACTION_NOISE_RE.search(cleaned):
        return "discard", False, "boilerplate_or_noise"
    for pattern in REDACTION_EXPLICIT_DROP_PATTERNS:
        if pattern.search(cleaned):
            return "discard", False, "boilerplate_or_heading"

    combined_context = norm_space(f"{left_context} {right_context}")
    if not combined_context or norm_space(bridge_context) == "[REDACTED]":
        return "discard", False, "no_usable_context"

    word_count = len(cleaned.split())
    if word_count <= 3 and re.search(r"[\d()§/]", cleaned):
        return "discard", False, "citation_fragment"
    if word_count <= 8 and re.match(r"^(?:[IVXLC]+|\d+|[A-Z])[\.)]\s+", cleaned):
        return "discard", False, "heading_fragment"
    sentence_like = (
        cleaned[:1].isupper()
        and cleaned.endswith((".", "!", "?"))
        and word_count >= 6
    )
    bridge_ready = bool(left_context) and bool(right_context) and bool(bridge_context)

    if sentence_like and bridge_ready and word_count <= 80:
        return "sentence", True, "sentence_like_gap"
    if word_count <= 10:
        return "value", True, "short_value_like_gap"
    if sentence_like:
        return "sentence", True, "sentence_like_gap"
    return "paragraph", True, "longer_passage_gap"


def build_redaction_rows_from_source_segments(
    doc_id: str,
    source_segments: pd.DataFrame,
) -> pd.DataFrame:
    if len(source_segments) == 0:
        return pd.DataFrame(columns=REDUCTION_KEEP_REDACTION_COLUMNS)

    rows: List[Dict[str, object]] = []
    redaction_counter = 0
    for _, row in source_segments.iterrows():
        paragraph_id = norm_space(row.get("paragraph_id", ""))
        heading = norm_space(row.get("heading", ""))
        unredacted_text = safe_str(row.get("unredacted_text", ""))
        redacted_text = safe_str(row.get("redacted_text", ""))
        if not unredacted_text or norm_space(unredacted_text) == norm_space(redacted_text):
            continue
        for span in diff_redaction_spans(unredacted_text, redacted_text):
            redaction_counter += 1
            left_bound, right_bound = redaction_sentence_bounds(
                unredacted_text,
                int(span["char_start"]),
                int(span["char_end"]),
            )
            left_context = norm_space(unredacted_text[left_bound:int(span["char_start"])])
            right_context = norm_space(unredacted_text[int(span["char_end"]):right_bound])
            if len((left_context + " " + right_context).split()) < 3:
                left_context, right_context = redaction_context_slice(
                    unredacted_text,
                    int(span["char_start"]),
                    int(span["char_end"]),
                )
            local_context = norm_space(
                f"{left_context} [REDACTED] {right_context}".strip()
            )
            previous_sentence = redaction_previous_sentence(unredacted_text, int(span["char_start"]))
            next_sentence = redaction_next_sentence(unredacted_text, int(span["char_end"]))
            bridge_context = norm_space(
                f"{previous_sentence} [REDACTED] {next_sentence}".strip()
            )
            if len(previous_sentence.split()) < 3 or len(next_sentence.split()) < 3:
                bridge_left, bridge_right = redaction_context_slice(
                    unredacted_text,
                    int(span["char_start"]),
                    int(span["char_end"]),
                )
                bridge_context = norm_space(
                    f"{bridge_left} [REDACTED] {bridge_right}".strip()
                )
            redaction_kind, recovery_keep, recovery_reason = classify_redaction_span(
                safe_str(span["text"]),
                left_context,
                right_context,
                bridge_context,
            )
            text_value = norm_space(span["text"])
            rows.append(
                {
                    "doc_id": doc_id,
                    "paragraph_id": paragraph_id,
                    "heading": heading,
                    "redaction_id": f"r{redaction_counter:04d}",
                    "clean_redaction_id": f"{redaction_counter:04d}",
                    "decision": "keep" if recovery_keep else "discard",
                    "row_kind": "paragraph_diff",
                    "page_tag": "",
                    "aligned_page": "",
                    "segment_id": paragraph_id,
                    "text": text_value,
                    "redacted_text": text_value,
                    "token_count": len(text_value.split()),
                    "word_count": len(text_value.split()),
                    "char_count": len(text_value),
                    "char_start_0based": int(span["char_start"]),
                    "char_end_0based": int(span["char_end"]),
                    "unred_source_page": "",
                    "red_source_page": "",
                    "left_context": left_context,
                    "right_context": right_context,
                    "local_context": local_context,
                    "bridge_context": bridge_context,
                    "source_unredacted_text": norm_space(unredacted_text),
                    "source_redacted_text": norm_space(redacted_text),
                    "redaction_kind": redaction_kind,
                    "recovery_keep": "true" if recovery_keep else "false",
                    "recovery_reason": recovery_reason,
                }
            )

    if not rows:
        return pd.DataFrame(columns=REDUCTION_KEEP_REDACTION_COLUMNS)
    return pd.DataFrame(rows, columns=REDUCTION_KEEP_REDACTION_COLUMNS)


def collapse_spaced_all_caps(text: object) -> str:
    value = safe_str(text)
    if not value:
        return ""
    return SPACED_ALL_CAPS_RE.sub(lambda match: match.group(0).replace(" ", ""), value)


def title_like_token(token: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", token)
    if not letters:
        return False
    if letters.lower() in TITLE_CONNECTORS:
        return True
    return letters[0].isupper() and letters[1:] == letters[1:].lower()


def starts_title_phrase(token: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", token)
    if not letters:
        return False
    if letters.lower() in TITLE_CONNECTORS:
        return False
    return letters[0].isupper() and letters[1:] == letters[1:].lower()


def cue_token(token: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", token).lower()
    return letters in CARTOGRAPHIC_CUE_WORDS


def uppercase_or_numeric_ratio(text: str) -> float:
    tokens = re.findall(r"[A-Za-z0-9&'/.-]+", norm_space(text))
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        letters = re.sub(r"[^A-Za-z]", "", token)
        has_digit = any(char.isdigit() for char in token)
        if has_digit:
            noisy += 1
            continue
        if letters and letters.upper() == letters:
            noisy += 1
    return noisy / len(tokens)


def longest_uppercase_numeric_run(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9&'/.-]+", norm_space(text))
    best = 0
    current = 0
    for token in tokens:
        letters = re.sub(r"[^A-Za-z]", "", token)
        if any(char.isdigit() for char in token) or (letters and letters.upper() == letters):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def looks_like_numeric_scale_legend(text: str) -> bool:
    value = norm_space(text)
    if not value or re.search(r"[.!?]", value):
        return False
    numbers = [int(match) for match in re.findall(r"\b\d+\b", value)]
    if len(numbers) < 3:
        return False
    lower = value.lower()
    if not any(term in lower for term in ("mile", "miles", "kilometer", "kilometers", "route", "legend", "scale")):
        return False
    monotonic = all(left <= right for left, right in zip(numbers, numbers[1:]))
    return monotonic or uppercase_or_numeric_ratio(value) >= 0.7


def looks_like_dense_allcaps_noise(text: str) -> bool:
    value = norm_space(text)
    if not value or re.search(r"[.!?]", value):
        return False
    tokens = re.findall(r"[A-Za-z0-9&'/.-]+", value)
    if len(tokens) < 5:
        return False
    return (
        uppercase_or_numeric_ratio(value) >= 0.72
        and longest_uppercase_numeric_run(value) >= 5
    )


def looks_like_tabular_noise(text: str) -> bool:
    value = norm_space(text)
    if not value:
        return False
    tokens = re.findall(r"[A-Za-z0-9&'/.-]+", value)
    if len(tokens) < 8:
        return False
    digit_tokens = sum(1 for token in tokens if any(char.isdigit() for char in token))
    if digit_tokens < 4:
        return False
    sentence_breaks = len(re.findall(r"[.!?][\"')\]]?(?=\s+[A-Z]|\s*$)", value))
    label_tokens = sum(1 for token in tokens if starts_title_phrase(token) or all_caps_marker_token(token))
    if sentence_breaks >= 2 and digit_tokens / len(tokens) < 0.35:
        return False
    return digit_tokens / len(tokens) >= 0.2 and label_tokens >= 4


def all_caps_marker_token(token: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", token)
    return bool(letters) and len(letters) >= 3 and letters.upper() == letters


def strip_generic_noise_prefix(text: str) -> str:
    value = norm_space(collapse_spaced_all_caps(text))
    if not value:
        return ""

    summary_list_match = re.match(
        r"^(?:\d{1,2}[./-][A-Za-z]+\s+\d{4}\s+)?(?:[A-Z][A-Z&/-]*\s+){1,6}SUMMARY\s+(\d+\.\s+.+)$",
        value,
        re.IGNORECASE,
    )
    if summary_list_match:
        return norm_space(summary_list_match.group(1))

    tokens = value.split()
    if len(tokens) >= 2 and re.fullmatch(r"(?:[IVXLCM]+|[A-Z]|\d+)[.)]", tokens[0], re.IGNORECASE):
        second_word = re.sub(r"[^A-Za-z]", "", tokens[1]).lower()
        if second_word in GENERIC_MARKER_WORDS:
            if len(tokens) > 2:
                return " ".join(tokens[2:]).strip()
            return ""
    if len(tokens) >= 2:
        cue_len = 0
        for token in tokens[:6]:
            if cue_token(token):
                cue_len += 1
                continue
            break
        if cue_len >= 2 and cue_len < len(tokens):
            return " ".join(tokens[cue_len:]).strip()

    if tokens:
        lead_word = re.sub(r"[^A-Za-z]", "", tokens[0]).lower()
        if lead_word in GENERIC_MARKER_WORDS:
            if len(tokens) > 1:
                return " ".join(tokens[1:]).strip()
            return ""
    if len(tokens) >= 3 and all_caps_marker_token(tokens[0]):
        if re.fullmatch(r"\d+[.)]?", tokens[1]):
            return " ".join(tokens[1:]).strip()
        for idx in range(1, min(len(tokens), 8)):
            if starts_title_phrase(tokens[idx]):
                return " ".join(tokens[idx:]).strip()
    if len(tokens) < 6:
        return value

    start = 0
    while start < len(tokens) and not starts_title_phrase(tokens[start]):
        start += 1
    if start >= len(tokens):
        return value

    for idx in range(start, len(tokens) - 2):
        if not starts_title_phrase(tokens[idx]):
            continue
        title_len = 1
        for j in range(idx + 1, len(tokens)):
            if title_like_token(tokens[j]):
                title_len += 1
                continue
            break

        if title_len < 2:
            continue

        prefix = " ".join(tokens[:idx]).strip(" ,;:-")
        suffix = " ".join(tokens[idx:]).strip()
        if not prefix or not suffix:
            continue
        if len(prefix.split()) < 4 or len(suffix.split()) < 3:
            continue
        if not (
            looks_like_numeric_scale_legend(prefix)
            or looks_like_dense_allcaps_noise(prefix)
            or (
                uppercase_or_numeric_ratio(prefix) >= 0.6
                and uppercase_token_ratio(prefix) >= 0.55
            )
        ):
            continue
        return suffix

    return value


def normalize_date(*candidates: object) -> str:
    for candidate in candidates:
        text = safe_str(candidate).strip()
        if not text:
            continue
        parsed = pd.to_datetime(text, errors="coerce")
        if not pd.isna(parsed):
            return parsed.strftime("%Y-%m-%d")
    return safe_str(candidates[0]).strip() if candidates else ""


def parse_pair_metadata(meta_path: Path) -> Dict[str, str]:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    csv_row = payload.get("csv_row") or []

    row_map: Dict[str, str] = {}
    for idx, name in enumerate(PAIR_METADATA_COLUMNS):
        row_map[name] = safe_str(csv_row[idx]) if idx < len(csv_row) else ""

    row_map["pair_key"] = safe_str(payload.get("pair_key")) or meta_path.parent.parent.name
    row_map["model"] = safe_str(payload.get("model"))
    return row_map


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def split_pages(text: str) -> List[Tuple[Optional[int], str]]:
    src = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    parts = PAGE_TAG_RE.split(src)
    if len(parts) <= 1:
        return [(None, src)]

    pages: List[Tuple[Optional[int], str]] = []
    pending_page: Optional[int] = None
    for part in parts:
        if not part:
            continue
        page_match = re.fullmatch(r"\[PAGE\s+(\d+)\]", part.strip(), flags=re.IGNORECASE)
        if page_match:
            pending_page = int(page_match.group(1))
            continue
        pages.append((pending_page, part))
    return pages


def replace_redactions_with_sentinels(text: str) -> Tuple[str, Dict[str, str]]:
    redaction_map: Dict[str, str] = {}

    def _repl(match: re.Match[str]) -> str:
        redaction_id = f"{int(match.group(1)):04d}"
        redaction_map[redaction_id] = match.group(2).strip()
        return f"__CLEAN_REDACTION_{redaction_id}__"

    return RED_TAG_RE.sub(_repl, text), redaction_map


def looks_like_repeated_boilerplate(line: str, repeated_edge_lines: set[str]) -> bool:
    text = norm_space(line)
    if not text:
        return False
    if looks_like_major_section_header(text):
        return False
    return text in repeated_edge_lines


def is_explicit_drop_line(line: str) -> bool:
    text = norm_space(line)
    if not text:
        return False
    return any(pattern.match(text) for pattern in EXPLICIT_DROP_LINE_PATTERNS)


def compute_repeated_edge_lines(pages: Sequence[Tuple[Optional[int], str]]) -> set[str]:
    edge_counter: Counter[str] = Counter()
    for _page_no, body in pages:
        lines = [norm_space(line) for line in body.splitlines()]
        lines = [line for line in lines if line]
        edge_candidates = lines[:4] + lines[-4:]
        for line in edge_candidates:
            if len(line) <= 90:
                edge_counter[line] += 1

    if not edge_counter:
        return set()

    threshold = max(2, int(round(len(pages) * 0.15)))
    return {
        line
        for line, count in edge_counter.items()
        if count >= threshold
        and not SENTINEL_RE.search(line)
    }


def clean_page_lines(body: str, repeated_edge_lines: set[str]) -> List[str]:
    cleaned: List[str] = []
    prior_blank = True
    for raw_line in body.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            if not prior_blank:
                cleaned.append("")
            prior_blank = True
            continue
        if is_explicit_drop_line(line) or looks_like_repeated_boilerplate(line, repeated_edge_lines):
            continue
        cleaned.append(line)
        prior_blank = False
    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    cleaned = drop_cover_title_blocks(cleaned)
    return cleaned


def drop_cover_title_blocks(lines: Sequence[str]) -> List[str]:
    if not lines:
        return []
    sequences = [
        ["CENTRAL", "INTELLIGENCE", "BULLETIN"],
        ["CURRENT", "INTELLIGENCE", "BULLETIN"],
    ]
    cleaned: List[str] = []
    idx = 0
    while idx < len(lines):
        matched = False
        for seq in sequences:
            window = lines[idx : idx + len(seq)]
            if len(window) == len(seq) and [norm_space(part).upper() for part in window] == seq:
                matched = True
                idx += len(seq)
                while idx < len(lines) and not norm_space(lines[idx]):
                    idx += 1
                break
        if matched:
            continue
        cleaned.append(lines[idx])
        idx += 1
    return cleaned


def looks_like_map_or_figure_page(lines: Sequence[str]) -> bool:
    nonblank = [norm_space(line) for line in lines if norm_space(line)]
    if len(nonblank) < 4:
        return False

    map_cue_count = sum(
        1
        for line in nonblank
        if any(pattern.fullmatch(line) for pattern in MAP_CUE_PATTERNS)
    )
    if map_cue_count == 0:
        return False

    short_label_count = 0
    sentence_like_count = 0
    for line in nonblank:
        tokens = re.findall(r"[A-Za-z][A-Za-z&'/.-]*", line)
        token_count = len(tokens)
        has_terminal_punct = bool(re.search(r"[.!?][\"')\]]?$", line))
        if token_count >= 8 or has_terminal_punct:
            sentence_like_count += 1
        if (
            token_count <= 4
            and len(line) <= 32
            and not has_terminal_punct
            and uppercase_token_ratio(line) >= 0.6
        ):
            short_label_count += 1

    if map_cue_count > 0:
        return short_label_count >= 4 and sentence_like_count <= 1

    return len(nonblank) >= 7 and short_label_count >= 5 and sentence_like_count <= 1


def join_hyphenated_lines(left: str, right: str) -> str:
    left_text = safe_str(left).strip()
    right_text = safe_str(right).strip()
    if left_text.endswith("-") and right_text:
        first_alpha = next((char for char in right_text if char.isalpha()), "")
        if first_alpha and first_alpha.islower():
            return left_text[:-1] + right_text
    return f"{left_text} {right_text}".strip()


def strip_inline_page_references(text: str) -> str:
    cleaned = safe_str(text)
    cleaned = INLINE_PAGE_REFERENCE_RE.sub("", cleaned)
    cleaned = INLINE_NO_BACKUP_RE.sub("", cleaned)
    cleaned = INLINE_MAP_REFERENCE_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def uppercase_token_ratio(text: str) -> float:
    tokens = re.findall(r"[A-Za-z][A-Za-z&'/.-]*", norm_space(text))
    if not tokens:
        return 0.0
    uppercase_like = 0
    for token in tokens:
        letters = re.sub(r"[^A-Za-z]", "", token)
        if letters and letters.upper() == letters:
            uppercase_like += 1
    return uppercase_like / len(tokens)


def looks_like_major_section_header(text: str) -> bool:
    value = norm_space(text)
    if not value:
        return False
    if MAJOR_SECTION_RE.fullmatch(value):
        return True
    if any(pattern.fullmatch(value) for pattern in NONSERIAL_MAJOR_SECTION_PATTERNS):
        return True

    match = SERIAL_HEADER_RE.fullmatch(value)
    if not match:
        return False

    body = match.group(2).strip()
    body_core = body.rstrip(":").strip()
    if not body_core:
        return False
    if body_core.endswith((".", "?", "!")):
        return False
    if len(body_core.split()) > 12:
        return False

    ratio = uppercase_token_ratio(body_core)
    if ratio >= 0.6:
        return True
    return body_core.istitle() and len(body_core.split()) <= 6


def looks_like_title_candidate(text: str) -> bool:
    value = norm_space(text)
    if not value or looks_like_major_section_header(value):
        return False
    if value.endswith("."):
        return False
    if len(value) > 120:
        return False
    if value.endswith(":"):
        return True
    alpha = re.sub(r"[^A-Za-z]", "", value)
    if alpha and alpha.upper() == alpha and len(alpha) >= 4:
        return True
    tokens = re.findall(r"[A-Za-z][A-Za-z'&/.-]*", value)
    if not tokens or len(tokens) > 12:
        return False
    allowed_lower = {"a", "an", "and", "at", "by", "de", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
    titleish = 0
    for token in tokens:
        normalized = re.sub(r"[^A-Za-z]", "", token)
        if not normalized:
            continue
        if normalized.lower() in allowed_lower or normalized[0].isupper():
            titleish += 1
    return (titleish / len(tokens)) >= 0.85


def starts_lowercaseish(text: str) -> bool:
    value = strip_inline_page_references(norm_space(text))
    if not value:
        return False
    for char in value:
        if char.isalpha():
            return char.islower()
    return False


def ends_with_terminal_punct(text: str) -> bool:
    value = strip_inline_page_references(norm_space(text))
    if not value:
        return False
    return bool(re.search(r"[.!?][\"')\]]?$", value))


def should_continue_across_blank(previous_text: str, next_text: str) -> bool:
    if not previous_text or not next_text:
        return False
    if looks_like_major_section_header(next_text) or looks_like_title_candidate(next_text):
        return False
    prev = strip_inline_page_references(previous_text)
    if not prev:
        return True
    if starts_lowercaseish(next_text):
        return True
    if prev.endswith((":", ";", ",", "-", "/", "(")):
        return True
    if not ends_with_terminal_punct(prev):
        return True
    return False


def is_document_tail_start(text: str) -> bool:
    value = norm_space(text)
    if not value:
        return False
    return any(pattern.match(value) for pattern in TAIL_START_PATTERNS)


def is_heading_like(paragraph: str) -> bool:
    text = norm_space(paragraph)
    if not text:
        return False
    return looks_like_major_section_header(text) or looks_like_title_candidate(text)


def looks_like_orphan_page_title(lines: Sequence[str]) -> bool:
    nonblank = [norm_space(line) for line in lines if norm_space(line)]
    if len(nonblank) != 1:
        return False
    text = nonblank[0]
    if looks_like_major_section_header(text):
        return False
    if not looks_like_title_candidate(text):
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z'&/.-]*", text)
    return 1 < len(tokens) <= 8


def render_variant_text(backbone_text: str, redaction_map: Dict[str, str], variant: str, gap_marker: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        redaction_id = match.group(1)
        if variant == "unredacted":
            return redaction_map.get(redaction_id, "")
        if variant == "redacted":
            return gap_marker
        raise ValueError(f"Unknown variant: {variant}")

    rendered = SENTINEL_RE.sub(_repl, backbone_text)
    rendered = re.sub(r"\s+", " ", rendered).strip()
    rendered = re.sub(rf"\s+{re.escape(gap_marker)}", f" {gap_marker}", rendered)
    rendered = re.sub(rf"{re.escape(gap_marker)}\s+", f"{gap_marker} ", rendered)
    return rendered.strip()


def render_raw_document_variant(raw_text: str, variant: str, gap_marker: str = DEFAULT_GAP_MARKER) -> str:
    def _repl(match: re.Match[str]) -> str:
        inner = match.group(2)
        if variant == "unredacted":
            return inner
        if variant == "redacted":
            return gap_marker
        raise ValueError(f"Unknown variant: {variant}")

    rendered = RED_TAG_RE.sub(_repl, raw_text)
    rendered = rendered.replace("\r\n", "\n").replace("\r", "\n")
    return rendered


def render_variant_fragment(backbone_text: str, redaction_map: Dict[str, str], variant: str, gap_marker: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        redaction_id = match.group(1)
        if variant == "unredacted":
            return redaction_map.get(redaction_id, "")
        if variant == "redacted":
            return gap_marker
        raise ValueError(f"Unknown variant: {variant}")

    rendered = SENTINEL_RE.sub(_repl, backbone_text)
    rendered = rendered.replace("\r\n", "\n").replace("\r", "\n")
    rendered = re.sub(r"[ \t]+", " ", rendered).strip()
    return rendered


def split_embedded_serial_heading(text: str) -> Tuple[str, str]:
    value = norm_space(text)
    match = SERIAL_HEADER_RE.match(value)
    if not match:
        return "", value

    prefix = match.group(1)
    rest = match.group(2).strip()
    tokens = rest.split()
    header_tokens: List[str] = []
    split_at = 0
    for idx, token in enumerate(tokens):
        normalized = re.sub(r"[^A-Za-z&/-]", "", token)
        if not normalized:
            break
        if normalized.upper() == normalized and len(normalized) >= 1:
            header_tokens.append(token)
            split_at = idx + 1
            continue
        break

    if not header_tokens or split_at >= len(tokens):
        return "", value

    header = f"{prefix} {' '.join(header_tokens)}".strip()
    content = " ".join(tokens[split_at:]).strip()
    if not content:
        return "", value
    return header, content


def expand_parallel_fragments(
    backbone_text: str,
    unredacted_text: str,
    redacted_text: str,
) -> List[Dict[str, object]]:
    unred_lines = unredacted_text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    red_lines = redacted_text.replace("\r\n", "\n").replace("\r", "\n").splitlines()

    if len(unred_lines) <= 1:
        return [
            {
                "backbone_text": norm_space(backbone_text),
                "unredacted_text": norm_space(unredacted_text),
                "redacted_text": norm_space(redacted_text),
            }
        ]

    rows: List[Dict[str, object]] = []
    first_redacted = norm_space(red_lines[0] if red_lines else redacted_text)
    first_nonempty_idx = next(
        (idx for idx, line in enumerate(unred_lines) if norm_space(line)),
        0,
    )
    carrier_idx = first_nonempty_idx
    if looks_like_major_section_header(norm_space(unred_lines[first_nonempty_idx])):
        carrier_idx = next(
            (
                idx
                for idx in range(first_nonempty_idx + 1, len(unred_lines))
                if norm_space(unred_lines[idx])
            ),
            first_nonempty_idx,
        )
    for idx, unred_line in enumerate(unred_lines):
        rows.append(
            {
                "backbone_text": norm_space(backbone_text) if idx == carrier_idx else "",
                "unredacted_text": norm_space(unred_line),
                "redacted_text": first_redacted if idx == carrier_idx else "",
            }
        )
    return rows


def build_clean_line_rows(
    pages: Sequence[Tuple[Optional[int], str]],
    repeated_edge_lines: set[str],
    redaction_map: Dict[str, str],
    gap_marker: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for page_no, page_body in pages:
        page_lines = clean_page_lines(page_body, repeated_edge_lines)
        if not page_lines:
            continue
        if looks_like_map_or_figure_page(page_lines):
            continue
        if looks_like_orphan_page_title(page_lines):
            continue
        if rows and not rows[-1].get("is_blank", False):
            rows.append({"is_blank": True, "page_no": page_no})

        for line in page_lines:
            if line == "":
                rows.append({"is_blank": True, "page_no": page_no})
                continue

            backbone = norm_space(line)
            unredacted = render_variant_fragment(backbone, redaction_map, "unredacted", gap_marker)
            redacted = render_variant_fragment(backbone, redaction_map, "redacted", gap_marker)
            for fragment in expand_parallel_fragments(backbone, unredacted, redacted):
                fragment_unredacted = norm_space(fragment.get("unredacted_text", ""))
                if not fragment_unredacted:
                    rows.append({"is_blank": True, "page_no": page_no})
                    continue

                split_header, split_content = split_embedded_serial_heading(fragment_unredacted)
                if split_header:
                    rows.append(
                        {
                            "is_blank": False,
                            "page_no": page_no,
                            "backbone_text": "",
                            "unredacted_text": split_header,
                            "redacted_text": "",
                        }
                    )
                    rows.append(
                        {
                            "is_blank": False,
                            "page_no": page_no,
                            "backbone_text": norm_space(fragment.get("backbone_text", "")),
                            "unredacted_text": split_content,
                            "redacted_text": norm_space(fragment.get("redacted_text", "")),
                        }
                    )
                    continue

                rows.append(
                    {
                        "is_blank": False,
                        "page_no": page_no,
                        "backbone_text": norm_space(fragment.get("backbone_text", "")),
                        "unredacted_text": fragment_unredacted,
                        "redacted_text": norm_space(fragment.get("redacted_text", "")),
                    }
                )
    return rows


def _combine_buffer(buffer: Sequence[Dict[str, object]], key: str) -> str:
    if not buffer:
        return ""
    joined = safe_str(buffer[0].get(key, ""))
    for row in buffer[1:]:
        joined = join_hyphenated_lines(joined, safe_str(row.get(key, "")))
    joined = strip_inline_page_references(joined)
    return norm_space(joined)


def paragraphize(lines: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    paragraphs: List[Dict[str, object]] = []
    buffer: List[Dict[str, object]] = []
    pending_blank = False

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        joined_backbone = _combine_buffer(buffer, "backbone_text")
        joined_unredacted = _combine_buffer(buffer, "unredacted_text")
        joined_redacted = _combine_buffer(buffer, "redacted_text")
        page_numbers = [
            int(row.get("page_no"))
            for row in buffer
            if row.get("page_no") is not None and safe_str(row.get("page_no")).strip()
        ]
        if joined_unredacted:
            paragraphs.append(
                {
                    "backbone_text": joined_backbone,
                    "unredacted_text": joined_unredacted,
                    "redacted_text": joined_redacted,
                    "is_heading": is_heading_like(joined_unredacted),
                    "page_start": min(page_numbers) if page_numbers else None,
                    "page_end": max(page_numbers) if page_numbers else None,
                }
            )
        buffer = []

    for row in lines:
        if row.get("is_blank", False):
            pending_blank = True
            continue

        line_unredacted = strip_inline_page_references(safe_str(row.get("unredacted_text", "")))
        if is_document_tail_start(line_unredacted):
            flush()
            break
        if is_heading_like(line_unredacted):
            flush()
            paragraphs.append(
                {
                    "backbone_text": norm_space(strip_inline_page_references(row.get("backbone_text", ""))),
                    "unredacted_text": norm_space(line_unredacted),
                    "redacted_text": norm_space(strip_inline_page_references(row.get("redacted_text", ""))),
                    "is_heading": True,
                    "page_start": row.get("page_no"),
                    "page_end": row.get("page_no"),
                }
            )
            pending_blank = False
            continue
        if not buffer:
            buffer.append(row)
            pending_blank = False
            continue
        previous_unredacted = safe_str(buffer[-1].get("unredacted_text", ""))
        if pending_blank and not should_continue_across_blank(previous_unredacted, line_unredacted):
            flush()
        buffer.append(row)
        pending_blank = False
    flush()
    return paragraphs


def build_clean_paragraphs_from_plain_transcription(raw_text: str, gap_marker: str) -> List[Dict[str, object]]:
    pages = split_pages(raw_text.replace("\r\n", "\n").replace("\r", "\n"))
    repeated_edge_lines = compute_repeated_edge_lines(pages)
    clean_line_rows = build_clean_line_rows(
        pages=pages,
        repeated_edge_lines=repeated_edge_lines,
        redaction_map={},
        gap_marker=gap_marker,
    )
    return paragraphize(clean_line_rows)


def normalize_match_text(text: str) -> str:
    value = strip_inline_page_references(norm_space(text)).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def paragraph_match_score(left: str, right: str) -> float:
    left_norm = normalize_match_text(left)
    right_norm = normalize_match_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0

    left_tokens = left_norm.split()
    right_tokens = right_norm.split()
    shorter_text, longer_text = (
        (left_norm, right_norm)
        if len(left_tokens) <= len(right_tokens)
        else (right_norm, left_norm)
    )
    if min(len(left_tokens), len(right_tokens)) >= 10 and shorter_text in longer_text:
        return 1.0

    left_set = set(left_tokens)
    right_set = set(right_tokens)
    overlap = len(left_set & right_set) / max(1, min(len(left_set), len(right_set)))
    ratio = difflib.SequenceMatcher(a=left_norm, b=right_norm, autojunk=False).ratio()

    if left_tokens[:8] and left_tokens[:8] == right_tokens[:8]:
        return max(ratio, overlap, 0.92)
    return max(ratio, overlap * 0.9)


def is_recoverable_paragraph(text: str) -> bool:
    value = norm_space(text)
    if not value:
        return False
    if is_heading_like(value):
        return True
    return len(value.split()) >= 12


def should_log_omission_text(text: str) -> bool:
    value = norm_space(text)
    if not value:
        return False
    if is_explicit_drop_line(value):
        return False
    if CONTENTS_PREFIX_RE.match(value):
        return False
    if any(pattern.match(value) for pattern in FRONT_MATTER_PREFIX_PATTERNS):
        return False
    if MAP_PAGE_PREFIX_RE.match(value):
        return False
    if re.search(r"\bBoundary line\b", value, re.IGNORECASE) and re.search(
        r"\bProposed boundary agreement\b", value, re.IGNORECASE
    ):
        return False
    if looks_like_numeric_scale_legend(value):
        return False
    if looks_like_dense_allcaps_noise(value):
        return False
    if looks_like_tabular_noise(value):
        return False
    if sum(1 for pattern in MAP_LEGEND_SIGNAL_PATTERNS if pattern.search(value)) >= 2:
        return False

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9&'/.-]*", value)
    token_count = len(tokens)
    terminal_punct = bool(re.search(r"[.!?:][\"')\]]?$", value))

    if looks_like_major_section_header(value) and token_count <= 6 and not terminal_punct:
        return False
    if token_count <= 4 and uppercase_token_ratio(value) >= 0.75 and not terminal_punct:
        return False
    if token_count <= 4 and looks_like_title_candidate(value) and not terminal_punct:
        return False
    if re.match(r"^\d+\.\s+", value) and token_count <= 18:
        return False
    if re.match(r"^[A-Z]\.\s+", value) and token_count <= 24:
        return False
    if is_heading_like(value) and not terminal_punct:
        return False
    if is_heading_like(value) and (token_count <= 8 or value.endswith(":")):
        return False

    return token_count >= 6 or terminal_punct


def split_leading_topic_label(text: str) -> Tuple[str, str]:
    value = norm_space(text)
    match = re.match(r"^([^:]{1,80}:)\s+(.+)$", value)
    if not match:
        return "", value
    title = norm_space(match.group(1))
    remainder = norm_space(match.group(2))
    if not remainder:
        return "", value
    if looks_like_major_section_header(title):
        return "", value
    if len(title.split()) > 8 or len(remainder.split()) < 8:
        return "", value
    return title, remainder


def infer_heading_from_text(text: str) -> str:
    value = norm_space(text)
    if not value:
        return ""

    match = re.match(r"^\*?\s*(?:\d+\.\s*)?([^:]{1,80}):\s+", value)
    if not match:
        return ""

    heading = norm_space(match.group(1))
    if not heading:
        return ""
    if is_explicit_drop_line(heading):
        return ""
    return heading


def strip_trailing_source_metadata(text: str) -> str:
    cleaned = TRAILING_SOURCE_METADATA_RE.sub("", safe_str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def looks_like_map_noise_prefix(text: str) -> bool:
    value = norm_space(text)
    if not value:
        return False
    if looks_like_numeric_scale_legend(value):
        return True
    if looks_like_dense_allcaps_noise(value):
        return True
    if looks_like_tabular_noise(value):
        return True
    if any(pattern.search(value) for pattern in PURE_MAP_TEXT_PATTERNS):
        return True
    if any(pattern.match(value) for pattern in MAP_CUE_PATTERNS):
        return True
    if sum(1 for pattern in MAP_LEGEND_SIGNAL_PATTERNS if pattern.search(value)) >= 2:
        return True
    token_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9&'/.-]*", value))
    return token_count <= 20 and not re.search(r"[.!?]", value) and uppercase_token_ratio(value) >= 0.55


def clean_segment_text_for_output(text: str) -> str:
    value = strip_trailing_source_metadata(strip_inline_page_references(norm_space(collapse_spaced_all_caps(text))))
    if not value:
        return ""
    if any(pattern.match(value) for pattern in FRONT_MATTER_PREFIX_PATTERNS):
        return ""
    if CONTENTS_PREFIX_RE.match(value):
        return "CONTENTS"

    value = MAP_PAGE_PREFIX_RE.sub("", value).strip()
    value = BULLETIN_PAGE_PREFIX_RE.sub("", value).strip()

    prior = None
    while prior != value:
        prior = value
        for pattern in MAP_LEGEND_PREFIX_RES:
            value = pattern.sub("", value, count=1).strip()
        anchor_match = MAP_ANCHOR_RE.search(value)
        if anchor_match and anchor_match.start() > 0:
            prefix = value[: anchor_match.start()].strip()
            if looks_like_map_noise_prefix(prefix):
                value = value[anchor_match.start() :].strip()

    for pattern, replacement in TEXT_REPAIR_PREFIX_RES:
        match = pattern.match(value)
        if match:
            value = replacement(match).strip()
            break

    for pattern in MAP_LEGEND_SUFFIX_RES:
        value = pattern.sub("", value).strip()

    prior = None
    while prior != value:
        prior = value
        value = strip_generic_noise_prefix(value)
    value = re.sub(r"^[\W_]+", "", value).strip()
    value = re.sub(r"^\*\s*", "", value).strip()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_segment_heading(heading: str, unredacted_text: str, redacted_text: str) -> str:
    value = strip_trailing_source_metadata(strip_inline_page_references(norm_space(collapse_spaced_all_caps(heading))))
    value = re.sub(r"^\*\s*", "", value).strip()
    prior = None
    while prior != value:
        prior = value
        value = strip_generic_noise_prefix(value)
    serial_match = SERIAL_HEADER_RE.fullmatch(value)
    if serial_match:
        body_letters = re.sub(r"[^A-Za-z ]", "", serial_match.group(2)).strip().lower()
        if body_letters in GENERIC_MARKER_WORDS:
            value = ""
    if re.search(r"\bSUMMARY\b", value, re.IGNORECASE) and uppercase_or_numeric_ratio(value) >= 0.35:
        value = ""
    if CONTENTS_PREFIX_RE.match(value):
        value = "CONTENTS"
    else:
        value = BULLETIN_PAGE_PREFIX_RE.sub("", value).strip()
        if value in {"Road Trail", "Railroad Road Trail"}:
            value = ""
        map_page_match = MAP_PAGE_HEADING_TITLE_RE.match(value)
        if map_page_match:
            value = norm_space(map_page_match.group(1))
        elif MAP_PAGE_PREFIX_RE.match(value):
            value = ""
        trailing_topic = MAP_TRAILING_TOPIC_RE.match(value or norm_space(heading))
        if trailing_topic and len((value or norm_space(heading))) - len(trailing_topic.group(1)) >= 12:
            value = norm_space(trailing_topic.group(1))
        if not value:
            trailing_title = MAP_TRAILING_TITLE_RE.match(norm_space(heading))
            if trailing_title and looks_like_map_noise_prefix(norm_space(heading)):
                value = norm_space(trailing_title.group(1))
        if not value and looks_like_map_noise_prefix(norm_space(heading)):
            value = ""
    if value.startswith("Estimate of the World Situation:"):
        match = ESTIMATE_TITLE_RE.match(unredacted_text) or ESTIMATE_TITLE_RE.match(redacted_text)
        value = norm_space(match.group(1)) if match else ""
    if any(pattern.match(value) for pattern in MAP_CUE_PATTERNS) or looks_like_map_noise_prefix(value):
        value = ""
    if value and not CONTENTS_HEADING_RE.fullmatch(value):
        value = re.sub(r"^\(Charts\)\s*", "", value, flags=re.IGNORECASE).strip()
    if not value:
        value = infer_heading_from_text(unredacted_text) or infer_heading_from_text(redacted_text)
    if not value and CONTENTS_PREFIX_RE.match(norm_space(unredacted_text)):
        value = "CONTENTS"
    return norm_space(value)


def is_pure_nonprose_segment(heading: str, text: str) -> bool:
    value = norm_space(text)
    heading_value = norm_space(heading)
    if not value:
        return True
    token_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9&'/.-]*", value))
    if CONTENTS_PREFIX_RE.match(heading_value) or CONTENTS_PREFIX_RE.match(value):
        return True
    if value == heading_value and looks_like_major_section_header(value):
        return True
    if looks_like_numeric_scale_legend(value):
        return True
    if looks_like_tabular_noise(value):
        return True
    if any(pattern.search(value) for pattern in PURE_MAP_TEXT_PATTERNS):
        if token_count <= 40 or not re.search(r"[.!?]", value):
            return True
    if looks_like_map_noise_prefix(value):
        return True
    return False


def is_probable_continuation_segment(heading: str, text: str) -> bool:
    heading_value = norm_space(heading)
    text_value = norm_space(text)
    if not text_value:
        return False
    if heading_value.endswith("-"):
        return True
    if heading_value and heading_value[:1].islower():
        return True
    if any(pattern.match(heading_value) for pattern in CONTINUATION_HEADING_PATTERNS):
        return True
    if not heading_value and any(pattern.match(text_value) for pattern in CONTINUATION_TEXT_PATTERNS):
        return True
    if starts_lowercaseish(text_value):
        return True
    return False


def append_segment_text(left: str, right: str) -> str:
    left_value = norm_space(left)
    right_value = norm_space(right)
    if not left_value:
        return right_value
    if not right_value:
        return left_value
    return join_hyphenated_lines(left_value, right_value)


def normalize_source_segments(source_segments: pd.DataFrame) -> pd.DataFrame:
    if len(source_segments) == 0:
        return source_segments.copy()

    normalized_rows: List[Dict[str, object]] = []
    for _, row in source_segments.iterrows():
        row_copy = dict(row)
        unredacted_text = clean_segment_text_for_output(safe_str(row_copy.get("unredacted_text", "")))
        redacted_text = clean_segment_text_for_output(safe_str(row_copy.get("redacted_text", "")))
        heading = clean_segment_heading(
            safe_str(row_copy.get("heading", "")),
            unredacted_text,
            redacted_text,
        )
        if heading and (heading in {"Road Trail", "Railroad Road Trail"} or looks_like_map_noise_prefix(heading)):
            inferred_heading = infer_heading_from_text(unredacted_text) or infer_heading_from_text(redacted_text)
            if inferred_heading:
                heading = inferred_heading

        display_text = unredacted_text or redacted_text
        if is_pure_nonprose_segment(heading, display_text):
            continue

        row_copy["unredacted_text"] = unredacted_text
        row_copy["redacted_text"] = redacted_text
        row_copy["heading"] = heading
        row_copy["char_count"] = len(display_text)
        row_copy["word_count"] = len(display_text.split())

        if (
            normalized_rows
            and safe_str(normalized_rows[-1].get("doc_id", "")) == safe_str(row_copy.get("doc_id", ""))
            and is_probable_continuation_segment(heading, display_text)
        ):
            previous = normalized_rows[-1]
            addon_unredacted = strip_leading_title(unredacted_text, heading) if heading else unredacted_text
            addon_redacted = strip_leading_title(redacted_text, heading) if heading else redacted_text
            previous["unredacted_text"] = append_segment_text(
                safe_str(previous.get("unredacted_text", "")),
                addon_unredacted,
            )
            previous["redacted_text"] = append_segment_text(
                safe_str(previous.get("redacted_text", "")),
                addon_redacted,
            )
            previous["redaction_count"] = int(previous.get("redaction_count", 0) or 0) + int(
                row_copy.get("redaction_count", 0) or 0
            )
            display = norm_space(previous.get("unredacted_text", "")) or norm_space(previous.get("redacted_text", ""))
            previous["char_count"] = len(display)
            previous["word_count"] = len(display.split())
            continue

        normalized_rows.append(row_copy)

    for idx, row in enumerate(normalized_rows, start=1):
        row["paragraph_id"] = f"seg{idx:04d}"
    return pd.DataFrame(normalized_rows)


def build_long_segment_audit(source_segments: pd.DataFrame) -> pd.DataFrame:
    if len(source_segments) == 0:
        return pd.DataFrame(
            columns=[
                "doc_id",
                "paragraph_id",
                "heading",
                "char_count",
                "word_count",
                "source_path",
                "audit_reason",
                "preview",
            ]
        )

    audit = source_segments[
        pd.to_numeric(source_segments["char_count"], errors="coerce").fillna(0) > LONG_SEGMENT_AUDIT_THRESHOLD
    ].copy()
    if len(audit) == 0:
        audit["audit_reason"] = []
        audit["preview"] = []
        return audit.reindex(
            columns=[
                "doc_id",
                "paragraph_id",
                "heading",
                "char_count",
                "word_count",
                "source_path",
                "audit_reason",
                "preview",
            ]
        )
    audit["audit_reason"] = "long_segment"
    audit["preview"] = audit["unredacted_text"].map(lambda value: norm_space(value)[:400])
    return audit.loc[
        :,
        [
            "doc_id",
            "paragraph_id",
            "heading",
            "char_count",
            "word_count",
            "source_path",
            "audit_reason",
            "preview",
        ],
    ].reset_index(drop=True)


def _match_threshold(text: str) -> float:
    value = norm_space(text)
    if looks_like_major_section_header(value):
        return 0.98
    if looks_like_title_candidate(value):
        token_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9&'/.-]*", value))
        if token_count <= 4:
            return 0.98
        return 0.88
    return 0.72


def _can_append_to_existing_match(red_text: str, unred_text: str, score: float) -> bool:
    red_tokens = normalize_match_text(red_text).split()
    unred_tokens = normalize_match_text(unred_text).split()
    if not red_tokens or not unred_tokens:
        return False
    if score >= 0.92:
        return True
    if score < 0.88:
        return False
    return len(red_tokens) <= int(len(unred_tokens) * 1.25) + 4


def _append_match_text(existing_text: str, new_text: str) -> str:
    existing = norm_space(existing_text)
    new = norm_space(new_text)
    if not existing:
        return new
    if not new or new in existing:
        return existing
    if existing in new:
        return new
    return f"{existing} {new}".strip()


def build_shared_units_from_unredacted_paragraphs(
    unredacted_paragraphs: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    shared_units = merge_small_topic_tail_units(build_units_from_paragraphs(unredacted_paragraphs))
    return merge_leading_serial_overview_run(shared_units)


def project_redacted_text_onto_shared_units(
    shared_units: Sequence[Dict[str, object]],
    canonical_redacted_text: str,
    gap_marker: str,
) -> List[Dict[str, object]]:
    projected_units = [dict(unit) for unit in shared_units]
    unredacted_body_tokens, segment_spans = build_body_and_segment_spans(projected_units, "unredacted")
    canonical_redacted_clean_text = build_clean_debug_text(canonical_redacted_text)
    redacted_body_tokens = canonical_redacted_clean_text.split()
    projected_texts = project_variant_texts_onto_shared_segments(
        unredacted_body=unredacted_body_tokens,
        redacted_body=redacted_body_tokens,
        segment_spans=segment_spans,
    )
    for unit, (redacted_text, redaction_count) in zip(projected_units, projected_texts):
        unit["redacted_text"] = redacted_text
        unit["redaction_count"] = redaction_count
    return projected_units


def find_redacted_only_omissions(
    doc_id: str,
    unredacted_units: Sequence[Dict[str, object]],
    redacted_paragraphs: Sequence[Dict[str, object]],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    targets: List[Dict[str, str]] = []
    for unit in unredacted_units:
        text = norm_space(unit.get("unredacted_text", ""))
        if not text:
            continue
        heading = norm_space(unit.get("segment_title", "")) or norm_space(unit.get("section_heading", ""))
        targets.append({"text": text, "heading": heading})

    omitted_rows: List[Dict[str, object]] = []
    nonempty_redacted_count = 0

    for red_idx, red_row in enumerate(redacted_paragraphs):
        red_text = norm_space(red_row.get("unredacted_text", ""))
        if not red_text:
            continue
        nonempty_redacted_count += 1

        threshold = _match_threshold(red_text)
        best_score = 0.0
        best_heading = ""
        for target in targets:
            score = paragraph_match_score(target["text"], red_text)
            if score > best_score:
                best_score = score
                best_heading = target["heading"]

        if best_score >= threshold:
            continue

        if is_recoverable_paragraph(red_text) and should_log_omission_text(red_text):
            omitted_rows.append(
                {
                    "doc_id": doc_id,
                    "redacted_paragraph_index": red_idx + 1,
                    "page_start": red_row.get("page_start"),
                    "page_end": red_row.get("page_end"),
                    "is_heading": bool(red_row.get("is_heading", False)),
                    "match_threshold": round(float(threshold), 4),
                    "best_match_score": round(float(best_score), 4),
                    "best_match_heading": best_heading,
                    "omission_reason": "redacted_only_unmatched",
                    "text": red_text,
                }
            )

    stats = {
        "matched_redacted_paragraphs": max(0, nonempty_redacted_count - len(omitted_rows)),
        "omitted_redacted_only_paragraphs": len(omitted_rows),
    }
    return pd.DataFrame(omitted_rows), stats


def _merge_title_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, str]:
    return {
        "backbone_text": " ".join(norm_space(row.get("backbone_text", "")) for row in rows if norm_space(row.get("backbone_text", ""))).strip(),
        "unredacted_text": " ".join(norm_space(row.get("unredacted_text", "")) for row in rows if norm_space(row.get("unredacted_text", ""))).strip(),
        "redacted_text": " ".join(norm_space(row.get("redacted_text", "")) for row in rows if norm_space(row.get("redacted_text", ""))).strip(),
    }


def split_off_leading_contents_page(
    paragraphs: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not paragraphs:
        return [], []

    first_text = norm_space(paragraphs[0].get("unredacted_text", ""))
    if first_text.upper() != "CONTENTS":
        return [], [dict(row) for row in paragraphs]

    contents_page = paragraphs[0].get("page_start")
    collected: List[Dict[str, object]] = []
    remaining_start = 0
    for idx, row in enumerate(paragraphs):
        text = norm_space(row.get("unredacted_text", ""))
        if not text:
            continue
        if row.get("page_start") != contents_page:
            remaining_start = idx
            break
        collected.append(dict(row))
    else:
        remaining_start = len(paragraphs)

    if len(collected) <= 1:
        return [], [dict(row) for row in paragraphs]

    merged_texts = _merge_title_rows(collected)
    overview_unit = {
        "section_id": "",
        "section_heading": "",
        "paragraph_id": "",
        "segment_id": "",
        "segment_order": 0,
        "segment_kind": "section_overview",
        "segment_title": "CONTENTS",
        "is_heading": False,
        "backbone_text": merged_texts["backbone_text"],
        "unredacted_text": merged_texts["unredacted_text"],
        "redacted_text": merged_texts["redacted_text"],
    }
    remaining = [dict(row) for row in paragraphs[remaining_start:]]
    return [overview_unit], remaining


def build_units_from_paragraphs(paragraphs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    prefix_units, working_paragraphs = split_off_leading_contents_page(paragraphs)
    units: List[Dict[str, object]] = list(prefix_units)
    current_section_id = ""
    current_section_heading = ""
    section_counter = 0
    current_title: Optional[Dict[str, str]] = None
    current_content: List[Dict[str, object]] = []
    deferred_title: Optional[Dict[str, str]] = None
    segment_counter = 0
    current_section_emitted = False

    def emit_empty_section_if_needed() -> None:
        nonlocal segment_counter, current_section_emitted
        if not current_section_heading or current_section_emitted:
            return
        segment_counter += 1
        units.append(
            {
                "section_id": current_section_id,
                "section_heading": current_section_heading,
                "paragraph_id": f"seg{segment_counter:04d}",
                "segment_id": f"seg{segment_counter:04d}",
                "segment_order": segment_counter,
                "segment_kind": "section_overview",
                "segment_title": "",
                "is_heading": False,
                "backbone_text": current_section_heading,
                "unredacted_text": current_section_heading,
                "redacted_text": current_section_heading,
            }
        )
        current_section_emitted = True

    def flush() -> None:
        nonlocal current_title, current_content, segment_counter, current_section_emitted
        content_backbone = " ".join(norm_space(part.get("backbone_text", "")) for part in current_content if norm_space(part.get("backbone_text", ""))).strip()
        content_unredacted = " ".join(norm_space(part.get("unredacted_text", "")) for part in current_content if norm_space(part.get("unredacted_text", ""))).strip()
        content_redacted = " ".join(norm_space(part.get("redacted_text", "")) for part in current_content if norm_space(part.get("redacted_text", ""))).strip()
        title_backbone = norm_space(current_title.get("backbone_text", "")) if current_title else ""
        title_unredacted = norm_space(current_title.get("unredacted_text", "")) if current_title else ""
        title_redacted = norm_space(current_title.get("redacted_text", "")) if current_title else ""

        if not content_unredacted:
            current_title = None
            current_content = []
            return
        segment_counter += 1
        if title_unredacted:
            is_section_opening_block = not current_section_emitted
            heading_prefix = f"{current_section_heading} " if current_section_heading and is_section_opening_block else ""
            backbone_core = f"{title_backbone} {content_backbone}".strip() if title_backbone else content_backbone
            unredacted_core = f"{title_unredacted} {content_unredacted}".strip()
            redacted_core = f"{title_redacted} {content_redacted}".strip() if title_redacted else content_redacted
            backbone_text = f"{heading_prefix}{backbone_core}".strip()
            unredacted_text = f"{heading_prefix}{unredacted_core}".strip()
            redacted_text = f"{heading_prefix}{redacted_core}".strip() if redacted_core else heading_prefix.strip()
            segment_kind = "section_overview" if is_section_opening_block else "topic_block"
        else:
            backbone_text = f"{current_section_heading} {content_backbone}".strip() if current_section_heading else content_backbone
            unredacted_text = f"{current_section_heading} {content_unredacted}".strip() if current_section_heading else content_unredacted
            redacted_text = f"{current_section_heading} {content_redacted}".strip() if current_section_heading else content_redacted
            segment_kind = "section_overview"
        units.append(
            {
                "section_id": current_section_id,
                "section_heading": current_section_heading,
                "paragraph_id": f"seg{segment_counter:04d}",
                "segment_id": f"seg{segment_counter:04d}",
                "segment_order": segment_counter,
                "segment_kind": segment_kind,
                "segment_title": title_unredacted,
                "is_heading": False,
                "backbone_text": backbone_text,
                "unredacted_text": unredacted_text,
                "redacted_text": redacted_text,
            }
        )
        current_title = None
        current_content = []
        current_section_emitted = True

    idx = 0
    while idx < len(working_paragraphs):
        paragraph_row = working_paragraphs[idx]
        paragraph = norm_space(paragraph_row.get("unredacted_text", ""))
        if not paragraph:
            idx += 1
            continue
        if is_document_tail_start(paragraph):
            flush()
            emit_empty_section_if_needed()
            break
        if looks_like_major_section_header(paragraph):
            flush()
            emit_empty_section_if_needed()
            deferred_title = None
            section_counter += 1
            current_section_id = f"s{section_counter:03d}"
            current_section_heading = paragraph
            current_section_emitted = False
            idx += 1
            continue
        if looks_like_title_candidate(paragraph):
            title_lines = [paragraph_row]
            lookahead = idx + 1
            while lookahead < len(working_paragraphs):
                candidate_row = working_paragraphs[lookahead]
                candidate = norm_space(candidate_row.get("unredacted_text", ""))
                if not candidate or looks_like_major_section_header(candidate) or is_document_tail_start(candidate):
                    break
                if not looks_like_title_candidate(candidate):
                    break
                title_lines.append(candidate_row)
                lookahead += 1
            gathered_title = _merge_title_rows(title_lines)
            gathered_title_unredacted = norm_space(gathered_title.get("unredacted_text", ""))
            next_paragraph = ""
            next_idx = lookahead
            while next_idx < len(working_paragraphs):
                candidate = norm_space(working_paragraphs[next_idx].get("unredacted_text", ""))
                if candidate:
                    next_paragraph = candidate
                    break
                next_idx += 1

            if len(title_lines) > 1 or gathered_title_unredacted.endswith(":") or not current_content:
                flush()
                current_title = gathered_title
                idx = lookahead
                continue

            if next_paragraph and starts_lowercaseish(next_paragraph):
                deferred_title = gathered_title
                idx = lookahead
                continue

            flush()
            current_title = gathered_title
            idx = lookahead
            continue

        if deferred_title and not starts_lowercaseish(paragraph):
            flush()
            current_title = deferred_title
            deferred_title = None

        current_content.append(paragraph_row)
        idx += 1

    flush()
    emit_empty_section_if_needed()
    return units


def looks_like_decorative_allcaps_title(title: str) -> bool:
    value = norm_space(title)
    if not value:
        return False
    if value.endswith(":"):
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z'&/.-]*", value)
    if not tokens or len(tokens) > 5:
        return False
    return uppercase_token_ratio(value) >= 0.95


def strip_leading_title(text: str, title: str) -> str:
    text_value = norm_space(text)
    title_value = norm_space(title)
    if not text_value or not title_value:
        return text_value
    prefix = f"{title_value} "
    if text_value.startswith(prefix):
        return text_value[len(prefix):].strip()
    if text_value == title_value:
        return ""
    return text_value


def merge_small_topic_tail_units(units: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    merged = [dict(unit) for unit in units]
    for idx, unit in enumerate(merged, start=1):
        seg_id = f"seg{idx:04d}"
        unit["segment_order"] = idx
        unit["paragraph_id"] = seg_id
        unit["segment_id"] = seg_id
    return merged


def _serial_marker(text: str) -> str:
    match = SERIAL_HEADER_RE.fullmatch(norm_space(text))
    if not match:
        return ""
    return match.group(1).rstrip(".)").upper()


def _serial_family(marker: str) -> str:
    if not marker:
        return ""
    if marker.isdigit():
        return "digit"
    if re.fullmatch(r"[IVXLCM]+", marker, re.IGNORECASE):
        return "roman"
    if re.fullmatch(r"[A-Z]", marker, re.IGNORECASE):
        return "alpha"
    return ""


def merge_leading_serial_overview_run(units: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if len(units) < 2:
        return [dict(unit) for unit in units]

    merged = [dict(unit) for unit in units]
    first_marker = _serial_marker(safe_str(merged[0].get("section_heading", "")))
    first_family = _serial_family(first_marker)

    if first_marker and first_marker in {"I", "1", "A"}:
        run_indices: List[int] = []
        saw_multiple_serial_sections = False
        for idx, unit in enumerate(merged):
            if safe_str(unit.get("segment_kind", "")) != "section_overview":
                break
            marker = _serial_marker(safe_str(unit.get("section_heading", "")))
            if marker:
                if _serial_family(marker) != first_family:
                    break
                if idx > 0 and marker == first_marker:
                    break
                if idx > 0:
                    saw_multiple_serial_sections = True
                run_indices.append(idx)
                continue

            if not saw_multiple_serial_sections:
                break
            word_count = len(norm_space(unit.get("unredacted_text", "")).split())
            if word_count > 90:
                break
            run_indices.append(idx)
    else:
        first_heading = norm_space(merged[0].get("section_heading", ""))
        if not looks_like_major_section_header(first_heading):
            return merged
        run_indices = []
        seen_headings: set[str] = set()
        for idx, unit in enumerate(merged):
            if safe_str(unit.get("segment_kind", "")) != "section_overview":
                break
            heading = norm_space(unit.get("section_heading", ""))
            if not looks_like_major_section_header(heading):
                break
            heading_key = heading.upper()
            if heading_key in seen_headings:
                break
            word_count = len(norm_space(unit.get("unredacted_text", "")).split())
            if word_count > 140:
                break
            seen_headings.add(heading_key)
            run_indices.append(idx)

    if len(run_indices) < 2:
        return merged

    run_units = [merged[idx] for idx in run_indices]
    run_word_counts = [len(norm_space(unit.get("unredacted_text", "")).split()) for unit in run_units]
    if any(count > 140 for count in run_word_counts) or sum(run_word_counts) > 700:
        return merged

    lead = dict(run_units[0])
    for text_key in ("backbone_text", "unredacted_text", "redacted_text"):
        values = [norm_space(unit.get(text_key, "")) for unit in run_units if norm_space(unit.get(text_key, ""))]
        lead[text_key] = " ".join(values).strip()

    new_units = [lead] + merged[len(run_indices):]
    for idx, unit in enumerate(new_units, start=1):
        seg_id = f"seg{idx:04d}"
        unit["segment_order"] = idx
        unit["paragraph_id"] = seg_id
        unit["segment_id"] = seg_id
    return new_units


def build_units_from_plain_transcription(raw_text: str, gap_marker: str) -> List[Dict[str, object]]:
    paragraphs = build_clean_paragraphs_from_plain_transcription(raw_text, gap_marker)
    units = build_units_from_paragraphs(paragraphs)
    return merge_small_topic_tail_units(units)


def build_clean_debug_text(raw_text: str) -> str:
    pages = split_pages(raw_text.replace("\r\n", "\n").replace("\r", "\n"))
    repeated_edge_lines = compute_repeated_edge_lines(pages)
    clean_line_rows: List[Dict[str, object]] = []

    for _page_no, page_body in pages:
        page_lines = clean_page_lines(page_body, repeated_edge_lines)
        if not page_lines:
            continue
        if looks_like_map_or_figure_page(page_lines):
            continue
        if looks_like_orphan_page_title(page_lines):
            continue
        if clean_line_rows and not clean_line_rows[-1].get("is_blank", False):
            clean_line_rows.append({"is_blank": True})

        for line in page_lines:
            if line == "":
                clean_line_rows.append({"is_blank": True})
                continue

            plain_text = norm_space(line)
            split_header, split_content = split_embedded_serial_heading(plain_text)
            if split_header:
                clean_line_rows.append(
                    {
                        "is_blank": False,
                        "backbone_text": "",
                        "unredacted_text": split_header,
                        "redacted_text": split_header,
                    }
                )
                clean_line_rows.append(
                    {
                        "is_blank": False,
                        "backbone_text": split_content,
                        "unredacted_text": split_content,
                        "redacted_text": split_content,
                    }
                )
                continue

            clean_line_rows.append(
                {
                    "is_blank": False,
                    "backbone_text": plain_text,
                    "unredacted_text": plain_text,
                    "redacted_text": plain_text,
                }
            )

    paragraphs = paragraphize(clean_line_rows)
    values = [
        norm_space(row.get("unredacted_text", ""))
        for row in paragraphs
        if norm_space(row.get("unredacted_text", ""))
    ]
    return "\n\n".join(values).strip()


def build_body_and_segment_spans(units: Sequence[Dict[str, object]], variant: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    text_key = f"{variant}_text"
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for unit in units:
        unit_tokens = norm_space(unit.get(text_key, "")).split()
        start = len(tokens)
        end = start + len(unit_tokens)
        spans.append((start, end))
        tokens.extend(unit_tokens)
    return tokens, spans


def _has_meaningful_gap(tokens: Sequence[str]) -> bool:
    return any(re.search(r"[A-Za-z0-9]", token) for token in tokens)


def _normalize_projected_text(text: str) -> str:
    value = safe_str(text).replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def project_variant_texts_onto_shared_segments(
    unredacted_body: Sequence[str],
    redacted_body: Sequence[str],
    segment_spans: Sequence[Tuple[int, int]],
) -> List[Tuple[str, int]]:
    matcher = difflib.SequenceMatcher(a=unredacted_body, b=redacted_body, autojunk=False)
    blocks = matcher.get_matching_blocks()
    projected: List[Tuple[str, int]] = []

    for seg_start, seg_end in segment_spans:
        if seg_end <= seg_start:
            projected.append(("", 0))
            continue

        pieces: List[str] = []
        cursor = seg_start
        gap_count = 0

        for block in blocks:
            block_start = block.a
            block_end = block.a + block.size
            if block_start >= seg_end:
                break
            if block_end <= seg_start or block.size == 0:
                continue

            overlap_start = max(seg_start, block_start)
            overlap_end = min(seg_end, block_end)
            if overlap_end <= overlap_start:
                continue

            if overlap_start > cursor and _has_meaningful_gap(unredacted_body[cursor:overlap_start]):
                gap_count += 1

            red_start = block.b + (overlap_start - block_start)
            red_end = red_start + (overlap_end - overlap_start)
            fragment = list(redacted_body[red_start:red_end])
            if fragment:
                pieces.extend(fragment)
            cursor = overlap_end

        if cursor < seg_end and _has_meaningful_gap(unredacted_body[cursor:seg_end]):
            gap_count += 1

        projected.append((_normalize_projected_text(" ".join(piece for piece in pieces if piece)), gap_count))

    return projected


def build_source_segments(
    doc_id: str,
    date_value: str,
    subject: str,
    source_path: str,
    units: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, unit in enumerate(units, start=1):
        unredacted_text = norm_space(unit.get("unredacted_text", ""))
        redacted_text = norm_space(unit.get("redacted_text", ""))
        if not unredacted_text and not redacted_text:
            continue
        paragraph_id = safe_str(unit.get("paragraph_id", f"seg{idx:04d}"))
        display_text = unredacted_text or redacted_text
        heading = norm_space(unit.get("segment_title", "")) or norm_space(unit.get("section_heading", ""))
        if not heading:
            heading = infer_heading_from_text(unredacted_text) or infer_heading_from_text(redacted_text)
        rows.append(
            {
                "doc_id": doc_id,
                "date": date_value,
                "subject": subject,
                "source_path": source_path,
                "paragraph_id": paragraph_id,
                "heading": heading,
                "redaction_count": int(unit.get("redaction_count", 0) or 0),
                "char_count": len(display_text),
                "word_count": len(display_text.split()),
                "unredacted_text": unredacted_text,
                "redacted_text": redacted_text,
            }
        )

    return normalize_source_segments(pd.DataFrame(rows))


def build_redaction_span_index(
    doc_id: str,
    kept_redactions: pd.DataFrame,
    source_segments: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if source_segments is not None and len(source_segments):
        derived = build_redaction_rows_from_source_segments(doc_id, source_segments)
        if len(derived):
            return derived
    if len(kept_redactions) == 0:
        return pd.DataFrame(columns=REDUCTION_KEEP_REDACTION_COLUMNS)

    kept_redactions = kept_redactions.copy()
    kept_redactions["doc_id"] = doc_id
    if "text" in kept_redactions.columns:
        kept_redactions["text"] = kept_redactions["text"].map(norm_space)

    for column in REDUCTION_KEEP_REDACTION_COLUMNS:
        if column not in kept_redactions.columns:
            kept_redactions[column] = ""
    return kept_redactions.loc[:, REDUCTION_KEEP_REDACTION_COLUMNS].copy()


def build_redaction_review_table(redaction_df: pd.DataFrame) -> pd.DataFrame:
    if len(redaction_df) == 0:
        return pd.DataFrame(columns=REDACTION_REVIEW_COLUMNS)

    review = redaction_df.copy()
    for column in REDACTION_REVIEW_COLUMNS:
        if column not in review.columns and column != "query_context":
            review[column] = ""

    def sentence_spans(text: str) -> List[Tuple[int, int]]:
        if not text:
            return []
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for match in REDACTION_SENTENCE_END_RE.finditer(text):
            end = match.end()
            spans.append((cursor, end))
            cursor = end
        if cursor < len(text):
            spans.append((cursor, len(text)))
        cleaned: List[Tuple[int, int]] = []
        for start, end in spans:
            if norm_space(text[start:end]):
                cleaned.append((start, end))
        return cleaned

    def build_review_context_from_source(row: pd.Series, min_side_tokens: int = 10) -> str:
        source_text = safe_str(row.get("source_unredacted_text", ""))
        if not source_text:
            return ""
        try:
            char_start = int(float(row.get("char_start_0based", 0)))
            char_end = int(float(row.get("char_end_0based", 0)))
        except Exception:
            return ""
        if char_start < 0 or char_end <= char_start or char_end > len(source_text):
            return ""

        spans = sentence_spans(source_text)
        if not spans:
            left_context, right_context = redaction_context_slice(source_text, char_start, char_end, radius=320)
            return norm_space(f"{left_context} [REDACTED] {right_context}".strip())

        containing_idx = None
        for idx, (start, end) in enumerate(spans):
            if char_start >= start and char_end <= end:
                containing_idx = idx
                break
        if containing_idx is None:
            left_context, right_context = redaction_context_slice(source_text, char_start, char_end, radius=320)
            return norm_space(f"{left_context} [REDACTED] {right_context}".strip())

        left_idx = containing_idx
        right_idx = containing_idx

        def side_token_counts(li: int, ri: int) -> Tuple[int, int]:
            window_start = spans[li][0]
            window_end = spans[ri][1]
            left_text = norm_space(source_text[window_start:char_start])
            right_text = norm_space(source_text[char_end:window_end])
            return len(left_text.split()), len(right_text.split())

        left_tokens, right_tokens = side_token_counts(left_idx, right_idx)
        while left_tokens < min_side_tokens and left_idx > 0:
            left_idx -= 1
            left_tokens, right_tokens = side_token_counts(left_idx, right_idx)
        while right_tokens < min_side_tokens and right_idx < len(spans) - 1:
            right_idx += 1
            left_tokens, right_tokens = side_token_counts(left_idx, right_idx)

        window_start = spans[left_idx][0]
        window_end = spans[right_idx][1]
        left_text = norm_space(source_text[window_start:char_start])
        right_text = norm_space(source_text[char_end:window_end])
        context = norm_space(f"{left_text} [REDACTED] {right_text}".strip())
        if "[REDACTED]" in context and (left_text or right_text):
            return context
        return ""

    def choose_review_context(row: pd.Series) -> str:
        source_window = build_review_context_from_source(row)
        if source_window:
            return source_window
        kind = norm_space(row.get("redaction_kind", "")).lower()
        local_context = norm_space(row.get("local_context", ""))
        bridge_context = norm_space(row.get("bridge_context", ""))
        if kind == "value":
            return bridge_context or local_context
        return bridge_context or local_context

    review["query_context"] = review.apply(choose_review_context, axis=1)

    numeric_front = ["word_count", "char_count"]
    for column in numeric_front:
        review[column] = pd.to_numeric(review[column], errors="coerce").fillna(0).astype(int)

    def natural_sort_key(value: object) -> tuple:
        parts = re.findall(r"\d+|[^\d]+", safe_str(value))
        key = []
        for part in parts:
            if part.isdigit():
                key.append((0, int(part)))
            else:
                key.append((1, part.casefold()))
        return tuple(key)

    review["decision_sort"] = review["decision"].map({"keep": 0, "discard": 1}).fillna(9)
    review["doc_sort"] = review["doc_id"].map(natural_sort_key)
    review["paragraph_sort"] = review["paragraph_id"].map(natural_sort_key)
    review["redaction_sort"] = review["redaction_id"].map(natural_sort_key)
    review = review.sort_values(
        ["doc_sort", "paragraph_sort", "redaction_sort", "decision_sort"],
        ascending=[True, True, True, True],
        kind="stable",
    )
    return review.loc[:, REDACTION_REVIEW_COLUMNS].reset_index(drop=True)


def build_document_body(source_segments: pd.DataFrame, variant: str) -> str:
    return build_document_body_with_options(source_segments, variant=variant, preserve_empty=False)


def build_document_body_with_options(
    source_segments: pd.DataFrame,
    variant: str,
    preserve_empty: bool,
) -> str:
    text_col = f"{variant}_text"
    if text_col not in source_segments.columns:
        raise KeyError(f"Missing source segment text column: {text_col}")
    values = [norm_space(value) for value in source_segments[text_col].tolist()]
    if preserve_empty:
        return "\n\n".join(values)
    visible = [value for value in values if value]
    return "\n\n".join(visible).strip()


def write_variant_document(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_debug_source_copy(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_original_transcription(pair_dir: Path, prepared_dir: Path, doc_id: str, variant: str) -> None:
    source_path = pair_dir / "document" / f"{variant}.transcription.txt"
    if not source_path.exists():
        return
    target_path = prepared_dir / "original_sources" / f"raw_{variant}_rendered" / f"{doc_id}.txt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        source_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n"),
        encoding="utf-8",
    )


def copy_source_pdfs(pair_dir: Path, prepared_dir: Path, doc_id: str) -> None:
    source_pdf_dir = pair_dir / "source_pdfs"
    if not source_pdf_dir.exists():
        return
    target_dir = prepared_dir / "original_sources" / "source_pdfs" / doc_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(source_pdf_dir.glob("*.pdf")):
        shutil.copy2(pdf_path, target_dir / pdf_path.name)


def build_document_record(
    pair_dir: Path,
    prepared_dir: Path,
    transcription_variant: str,
    gap_marker: str,
) -> Optional[Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, str]]]:
    difference_path = pair_dir / "difference" / "unredacted_bracketed.filtered.aligned.txt"
    metadata_path = pair_dir / "document" / "pair_metadata.json"
    kept_redactions_path = pair_dir / "difference" / "kept_redactions.jsonl"
    page_alignment_path = pair_dir / "difference" / "page_alignment.json"
    raw_unredacted_path = pair_dir / "document" / "unredacted.transcription.txt"
    raw_redacted_path = pair_dir / "document" / "redacted.transcription.txt"

    if not difference_path.exists() or not metadata_path.exists():
        return None

    metadata = parse_pair_metadata(metadata_path)
    doc_id = metadata["pair_key"] or pair_dir.name
    source_id = metadata["cia_document_id"] or doc_id.split("_", 1)[-1]
    subject = metadata["title"] or metadata["collection"] or source_id
    date_value = normalize_date(metadata["date_compact"], metadata["date_display"])

    aligned_raw_text = difference_path.read_text(encoding="utf-8", errors="ignore")
    text_with_sentinels, redaction_map = replace_redactions_with_sentinels(aligned_raw_text)

    variant_raw_map = {
        "unredacted": raw_unredacted_path.read_text(encoding="utf-8", errors="ignore")
        if raw_unredacted_path.exists()
        else render_raw_document_variant(aligned_raw_text, "unredacted"),
        "redacted": raw_redacted_path.read_text(encoding="utf-8", errors="ignore")
        if raw_redacted_path.exists()
        else render_raw_document_variant(aligned_raw_text, "redacted"),
    }
    canonical_redacted_text = render_raw_document_variant(aligned_raw_text, "redacted", gap_marker="")

    variant_source_path_map = {
        "unredacted": raw_unredacted_path if raw_unredacted_path.exists() else difference_path,
        "redacted": raw_redacted_path if raw_redacted_path.exists() else difference_path,
    }
    unredacted_paragraphs = build_clean_paragraphs_from_plain_transcription(
        variant_raw_map["unredacted"], gap_marker
    )
    redacted_paragraphs = build_clean_paragraphs_from_plain_transcription(
        variant_raw_map["redacted"], gap_marker
    )
    shared_units = build_shared_units_from_unredacted_paragraphs(unredacted_paragraphs)
    shared_units = project_redacted_text_onto_shared_units(
        shared_units=shared_units,
        canonical_redacted_text=canonical_redacted_text,
        gap_marker=gap_marker,
    )
    omitted_redacted_only_df, paragraph_merge_stats = find_redacted_only_omissions(
        doc_id=doc_id,
        unredacted_units=shared_units,
        redacted_paragraphs=redacted_paragraphs,
    )
    if not shared_units:
        return None
    unmatched_unredacted_segments = sum(
        1
        for unit in shared_units
        if is_recoverable_paragraph(norm_space(unit.get("unredacted_text", "")))
        and not norm_space(unit.get("redacted_text", ""))
    )

    source_segments = build_source_segments(
        doc_id=doc_id,
        date_value=date_value,
        subject=subject,
        source_path=str(variant_source_path_map[transcription_variant]),
        units=shared_units,
    )
    variant_bodies = {
        "unredacted": build_document_body_with_options(source_segments, variant="unredacted", preserve_empty=False),
        "redacted": build_document_body_with_options(source_segments, variant="redacted", preserve_empty=True),
    }
    document_body = variant_bodies[transcription_variant]
    if not document_body:
        return None

    kept_redactions = pd.DataFrame(load_jsonl(kept_redactions_path))
    kept_redactions = build_redaction_span_index(
        doc_id,
        kept_redactions,
        source_segments=source_segments,
    )

    page_alignment: object = []
    if page_alignment_path.exists():
        try:
            page_alignment = json.loads(page_alignment_path.read_text(encoding="utf-8"))
        except Exception:
            page_alignment = []

    summary = {
        "doc_id": doc_id,
        "source_id": source_id,
        "subject": subject,
        "date": date_value,
        "transcription_variant": transcription_variant,
        "segment_count": int(len(source_segments)),
        "segments_with_redactions": int((source_segments["redaction_count"] > 0).sum()),
        "mean_segment_words": round(float(pd.to_numeric(source_segments["word_count"], errors="coerce").fillna(0).mean()), 2) if len(source_segments) else 0.0,
        "max_segment_chars": int(pd.to_numeric(source_segments["char_count"], errors="coerce").fillna(0).max()) if len(source_segments) else 0,
        "kept_redaction_rows": int(len(kept_redactions)),
        "page_alignment_rows": int(len(page_alignment) if isinstance(page_alignment, list) else len(page_alignment.get("rows", []))),
        "source_path": str(variant_source_path_map[transcription_variant]),
        "matched_redacted_paragraphs": int(paragraph_merge_stats["matched_redacted_paragraphs"]),
        "omitted_redacted_only_paragraphs": int(paragraph_merge_stats["omitted_redacted_only_paragraphs"]),
        "unmatched_unredacted_paragraphs": int(unmatched_unredacted_segments),
    }

    record = {
        "doc_id": doc_id,
        "id": source_id,
        "subject": subject,
        "date": date_value,
        "body": document_body,
        "collection": metadata["collection"],
        "record_id": metadata["record_id"],
        "date_display": metadata["date_display"],
        "date_compact": metadata["date_compact"],
        "document_title": metadata["title"],
        "year": metadata["year"],
        "document_url": metadata["document_url"],
        "source_pdf_url": metadata["source_pdf_url"],
        "pair_key": doc_id,
        "pair_model": metadata["model"],
        "transcription_variant": transcription_variant,
        "transcription_path": str(variant_source_path_map[transcription_variant]),
        "metadata_path": str(metadata_path),
        "tag": "",
        "concepts": "",
    }

    write_debug_source_copy(
        prepared_dir / "original_sources" / "aligned_bracketed" / f"{doc_id}.txt",
        aligned_raw_text,
    )
    write_debug_source_copy(
        prepared_dir / "original_sources" / "backbone_with_sentinels" / f"{doc_id}.txt",
        build_clean_debug_text(text_with_sentinels),
    )
    copy_original_transcription(pair_dir, prepared_dir, doc_id, "unredacted")
    copy_original_transcription(pair_dir, prepared_dir, doc_id, "redacted")
    write_debug_source_copy(
        prepared_dir / "original_sources" / "aligned_unredacted_rendered" / f"{doc_id}.txt",
        build_clean_debug_text(render_raw_document_variant(aligned_raw_text, "unredacted")),
    )
    write_debug_source_copy(
        prepared_dir / "original_sources" / "aligned_redacted_rendered" / f"{doc_id}.txt",
        build_clean_debug_text(canonical_redacted_text),
    )
    copy_source_pdfs(pair_dir, prepared_dir, doc_id)
    return record, source_segments, kept_redactions, omitted_redacted_only_df, summary, variant_bodies


def build_dataframe(records: List[Dict[str, object]]) -> pd.DataFrame:
    if not records:
        raise ValueError("No aligned bracketed source documents were prepared.")

    df = pd.DataFrame(records)
    df["date_sort"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date_sort", "doc_id"], na_position="last").drop(columns=["date_sort"])

    ordered_columns = [
        "doc_id",
        "id",
        "subject",
        "date",
        "body",
        "collection",
        "record_id",
        "date_display",
        "date_compact",
        "document_title",
        "year",
        "document_url",
        "source_pdf_url",
        "pair_key",
        "pair_model",
        "transcription_variant",
        "transcription_path",
        "metadata_path",
        "tag",
        "concepts",
    ]
    return df[ordered_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a redaction-aware KG project from aligned bracketed sources."
    )
    parser.add_argument(
        "--corpus-root",
        default="cib_postprocessed",
        help="Root folder containing per-document postprocessed corpus directories.",
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Output project directory.",
    )
    parser.add_argument(
        "--csv-name",
        default="input_documents.csv",
        help="Name of the prepared document CSV to write in the project directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for a small test run.",
    )
    parser.add_argument(
        "--transcription-variant",
        choices=["unredacted", "redacted"],
        default="unredacted",
        help="Which aligned variant body should populate input_documents.csv.",
    )
    parser.add_argument(
        "--gap-marker",
        default=DEFAULT_GAP_MARKER,
        help="Placeholder inserted into the redacted variant for kept redaction spans.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_root = Path(args.corpus_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    prepared_dir = project_dir / "prepared"
    variant_dir = prepared_dir / "variant_documents"

    if not corpus_root.exists():
        raise FileNotFoundError(f"Corpus root not found: {corpus_root}")

    project_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    source_segments_frames: List[pd.DataFrame] = []
    kept_redaction_frames: List[pd.DataFrame] = []
    omitted_redacted_only_frames: List[pd.DataFrame] = []
    doc_summaries: List[Dict[str, object]] = []

    doc_dirs = [path for path in sorted(corpus_root.iterdir()) if path.is_dir()]
    if args.limit is not None:
        doc_dirs = doc_dirs[: args.limit]

    for doc_dir in iter_progress(
        doc_dirs,
        total=len(doc_dirs),
        desc=f"A0 prepare ({args.transcription_variant})",
    ):
        built = build_document_record(
            pair_dir=doc_dir,
            prepared_dir=prepared_dir,
            transcription_variant=args.transcription_variant,
            gap_marker=args.gap_marker,
        )
        if built is None:
            continue
        record, source_segments, kept_redactions, omitted_redacted_only, doc_summary, variant_bodies = built
        records.append(record)
        source_segments_frames.append(source_segments)
        kept_redaction_frames.append(kept_redactions)
        omitted_redacted_only_frames.append(omitted_redacted_only)
        doc_summaries.append(doc_summary)

        doc_id = safe_str(record["doc_id"])
        write_variant_document(
            variant_dir / "unredacted" / f"{doc_id}.txt",
            variant_bodies.get("unredacted", ""),
        )
        write_variant_document(
            variant_dir / "redacted" / f"{doc_id}.txt",
            variant_bodies.get("redacted", ""),
        )

    input_df = build_dataframe(records)
    input_csv = project_dir / args.csv_name
    input_df.to_csv(input_csv, index=False)

    source_segments_df = (
        pd.concat(source_segments_frames, ignore_index=True)
        if source_segments_frames
        else pd.DataFrame()
    )
    kept_redactions_df = (
        pd.concat(kept_redaction_frames, ignore_index=True)
        if kept_redaction_frames
        else pd.DataFrame()
    )
    omitted_redacted_only_df = (
        pd.concat(omitted_redacted_only_frames, ignore_index=True)
        if omitted_redacted_only_frames
        else pd.DataFrame(
            columns=[
                "doc_id",
                "redacted_paragraph_index",
                "page_start",
                "page_end",
                "is_heading",
                "match_threshold",
                "best_match_score",
                "best_match_heading",
                "omission_reason",
                "text",
            ]
        )
    )
    source_segments_df.to_csv(prepared_dir / "source_segments.csv", index=False)
    kept_redactions_df.to_csv(prepared_dir / "redaction_span_index.csv", index=False)
    build_redaction_review_table(kept_redactions_df).to_csv(
        prepared_dir / "redaction_review.csv",
        index=False,
    )
    omitted_redacted_only_df.to_csv(prepared_dir / "redacted_only_omissions.csv", index=False)
    build_long_segment_audit(source_segments_df).to_csv(prepared_dir / "long_segment_audit.csv", index=False)
    pd.DataFrame(doc_summaries).to_csv(prepared_dir / "document_prepare_summary.csv", index=False)

    summary = {
        "corpus_root": str(corpus_root),
        "project_dir": str(project_dir),
        "input_csv": str(input_csv),
        "transcription_variant": args.transcription_variant,
        "gap_marker": args.gap_marker,
        "documents_written": int(len(input_df)),
        "source_segment_rows": int(len(source_segments_df)),
        "redaction_span_rows": int(len(kept_redactions_df)),
        "redacted_only_omission_rows": int(len(omitted_redacted_only_df)),
        "long_segment_audit_rows": int((pd.to_numeric(source_segments_df.get("char_count", pd.Series(dtype=float)), errors="coerce").fillna(0) > LONG_SEGMENT_AUDIT_THRESHOLD).sum()) if len(source_segments_df) else 0,
        "prepared_dir": str(prepared_dir),
    }
    (project_dir / "input_build_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
