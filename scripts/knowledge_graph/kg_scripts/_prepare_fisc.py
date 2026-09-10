#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from . import _prepare_pairs as base

ORIGINAL_NORMALIZE_SOURCE_SEGMENTS = base.normalize_source_segments
ORIGINAL_FIND_REDACTED_ONLY_OMISSIONS = base.find_redacted_only_omissions
ORIGINAL_BUILD_REDACTION_SPAN_INDEX = base.build_redaction_span_index


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'/.-]*")
LEGAL_CITE_RE = re.compile(r"\b\d+\s+u\.?s\.?c\.?\b", re.I)
DOC_STAMP_RE = re.compile(r"^[A-Z0-9-]{8,}$")
MASK_RE = re.compile(r"^(?:X{2,}|\[REDACTED\]+)$", re.I)
FISC_CONTROL_RE = re.compile(r"^(?:[A-Z]{1,3}(?://[A-Z]{1,4}){0,3}|[A-Z]{1,3}/[A-Z]{1,4})$")
FISC_PAGE_STAMP_RE = re.compile(r"\b\d{1,3}-cv-\d+\(fbi\)\s*-\s*\d+\b", re.I)
FISC_REDACTION_CODE_RE = re.compile(r"(?:\(\s*b\s*\)\s*\(\s*[\w-]+\s*\))+|\bb\d+(?:-\d+)?(?:\s+b\d+(?:-\d+)?)+(?:\s*,\s*\d+)*\b", re.I)
FISC_HEADING_PREFIX_RE = re.compile(r"^(?:\d+\s+)?(?:\([A-ZX/.-]+\)\s*)+(?=[A-Z])")
FISC_INLINE_REDACTED_RE = re.compile(r"\[\s*REDACTED\s*\]", re.I)
FISC_ARTICLE_FRAGMENT_RE = re.compile(r"^[A-Z]-\s*Article\)\.?\s*", re.I)
FISC_CODE_TOKEN_RE = re.compile(r"^b[0-9a-z]+-\d+$", re.I)
FISC_USC_CITE_RE = re.compile(r"(?:\b\d+\s+)?u\.?s\.?c\.?\s*(?:§+)?", re.I)
FISC_SECTION_CITE_RE = re.compile(r"\b\d+[A-Za-z]?(?:\([A-Za-z0-9.-]+\)){1,6}\b")
FISC_DOCKET_RE = re.compile(r"\bDocket Number:\s*\d[\d-]*\b", re.I)
FISC_UNCLASSIFIED_BANNER_RE = re.compile(r"All Information is considered unclassified except where otherwise shown", re.I)
FISC_DECLASSIFIED_RE = re.compile(r"\bDeclas+ified by .*?(?:\d{1,2}/\d{1,2}/\d{2,4})?\b", re.I)
FISC_CLASSIFICATION_RE = re.compile(r"\b(?:top\s+secret(?:/?/?noforn)?|xx\s+secret|secretm|secref|senket|secret)\b", re.I)
FISC_CERT_RE = re.compile(r"\bFISC,?\s*(?:I\s+hereby\s+)?certify.*?(?:orig(?:inal|in))\.?", re.I)
FISC_JUDGE_LINE_RE = re.compile(r"\bJudge,\s+United States Foreign Intelligence Surveillance Court\b", re.I)
FISC_XMASK_RE = re.compile(r"\bX{3,}\b", re.I)
FISC_AUTH_EXPIRY_RE = re.compile(
    r"This authorization regarding .*? expires at .*?(?:day of [A-Za-z]+(?:,\s*\d{4})?\.?|$)",
    re.I,
)
FISC_CLERK_CERT_RE = re.compile(r"I,\s*Lee ?Ann Flynn Hall,\s*Clerk,\s*FISC,.*?original\.?", re.I)
FISC_SENSITIVE_INFO_RE = re.compile(r"Sensitive Information\s*[§%]?", re.I)
FISC_SERIAL_CLASSIFICATION_PREFIX_RE = re.compile(
    r"^(?:(?:[IVXLCM]+|\d+|[A-Z])[\].)]\s*)?(?:\(\s*[A-Z][A-Z/.-]*\s*\)\s*)+",
    re.I,
)
FISC_NUMBERED_CITE_PREFIX_RE = re.compile(
    r"^(?:\[?\s*)?(?:\d+[A-Za-z]?(?:\([A-Za-z0-9./-]+\)){1,8}\]?\s*)+",
    re.I,
)
FISC_SENATE_STAMP_RE = re.compile(r"\bSENATE-FISA\d{4}-\d+\b", re.I)
FISC_SOURCE_DESCRIPTION_RE = re.compile(
    r"^(?:\d+\s*)?(?:\(?\s*)?Source Description(?:\s*\[[^\]]+\])?(?:\s*\)?\s*)?",
    re.I,
)
FISC_LEEANN_PREFIX_RE = re.compile(
    r"^I,\s*Lee ?Ann Flynn Hall,\s*Clerk,?\s*(?:FISC,?\s*)?(?:certify.*?original\.?\s*)?",
    re.I,
)
FISC_SIGNATURE_LINE_RE = re.compile(
    r"^(?:[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+\s+)?"
    r"(?:(?:Acting|Deputy)\s+)?Attorney General of the United States\s+Date\s*",
    re.I,
)
FISC_DATE_PREFIX_RE = re.compile(r"^(?:Date\b\s*)+", re.I)
FISC_PUNCT_PREFIX_RE = re.compile(r"^(?:[-–—=:;,.()\[\]/\\]|_{2,}|\*{2,})+\s*")
FISC_PROCEDURAL_DROP_PATTERNS = [
    re.compile(r"\bI declare under penalty of perjury\b", re.I),
    re.compile(r"\bExecuted pursuant to Title 28\b", re.I),
    re.compile(r"\bI,\s*Lee ?Ann Flynn Hall,\s*Clerk\b", re.I),
    re.compile(r"\bWHEREFORE,\s*the United States submits\b", re.I),
    re.compile(r"\bThe authorities approved are for the period indicated below\b", re.I),
    re.compile(r"\bThe FBI has reviewed this verified application for accuracy\b", re.I),
    re.compile(r"\bI find that this application regarding Carter W\. Page satisfies the criteria and requirements\b", re.I),
    re.compile(r"\bAs to all information acquired through the authorities requested herein\b", re.I),
    re.compile(r"\bPursuant to Executive Order 13787\b", re.I),
    re.compile(r"\bThe certification of the Assistant to the President for National Security Affairs\b", re.I),
]
FISC_SPAN_DROP_PATTERNS = [
    re.compile(r"\bNon-SES PII\b", re.I),
    re.compile(r"\bSSA PII\b", re.I),
    re.compile(r"\bClerk,\s*FISC\b", re.I),
    re.compile(r"\bAttorney General of the United States\b", re.I),
    re.compile(r"\bNational Intelligence\s+\d{1,2}/\d{1,2}/\d{2,4}\b", re.I),
    re.compile(r"\bPrincipal Deputy Director of National Intelligence\b", re.I),
    re.compile(r"\bSecretary of State\b", re.I),
    re.compile(r"\bDeputy Secretary of State\b", re.I),
]
FISC_PROCEDURAL_SIGNAL_RE = re.compile(
    r"(?:"
    r"premises or property to be searched|"
    r"information, material, or property to be seized|"
    r"normal investigative techniques|"
    r"facts concerning previous applications|"
    r"duration of the authorities requested|"
    r"electronic surveillance shall be directed|"
    r"order and warrant|"
    r"foreign intelligence surveillance act|"
    r"minimization procedures|"
    r"probable cause to believe|"
    r"full consideration having been given|"
    r"authorities requested herein|"
    r"identity of federal officer making application"
    r")",
    re.I,
)
FISC_PROCEDURAL_HEADING_PATTERNS = [
    re.compile(r"^Date$", re.I),
    re.compile(r"^Approval$", re.I),
    re.compile(r"^Wherefore$", re.I),
    re.compile(r"^Source Description$", re.I),
    re.compile(r"^and\s+\d+[A-Za-z]?(?:\([A-Za-z0-9./-]+\)){1,8}\]?$", re.I),
]
FISC_ALWAYS_DISCARD_PATTERNS = [
    re.compile(r"^\s*S//\s*$", re.I),
    re.compile(r"^\s*(?:U|S|TS|NF|NOFORN|S/NF|S//NF|TS//NF)\s*$", re.I),
]
FISC_NOISEY_PREFIX_TOKEN_RE = re.compile(
    r"^(?:"
    r"(?:\d+[A-Za-z]?(?:\([A-Za-z0-9./-]+\)){1,8})|"
    r"(?:SENATE-FISA\d{4}-\d+)|"
    r"(?:[A-Z]{1,4}//[A-Z]{1,4})|"
    r"(?:[A-Z])|"
    r"(?:X{2,})|"
    r"(?:Date)|"
    r"(?:Source)|"
    r"(?:Description)|"
    r"(?:U\.?S\.?C\.?)"
    r")$",
    re.I,
)
FISC_MEANINGFUL_START_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))(?:\d+\.\s+)?(?:The|As|Wherefore|Further|United|I|This)\b"
)
FISC_SIGNATURE_HINTS = {
    "director",
    "deputy",
    "secretary",
    "assistant",
    "attorney",
    "president",
    "clerk",
    "fisc",
    "intelligence",
    "national",
    "federal",
    "bureau",
    "agency",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "into",
    "is",
    "it",
    "its",
    "new",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "under",
    "was",
    "were",
    "which",
    "who",
    "with",
}

FISC_INCOMPLETE_TAIL_TOKENS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "under",
    "with",
}


def normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def lexical_tokens(text: str) -> List[str]:
    return [normalize_token(token) for token in TOKEN_RE.findall(base.norm_space(text)) if normalize_token(token)]


def content_tokens(text: str) -> List[str]:
    return [token for token in lexical_tokens(text) if len(token) >= 3 and token not in STOPWORDS]


def clean_fisc_match_text(text: str) -> str:
    value = base.norm_space(text)
    if not value:
        return ""
    value = FISC_PAGE_STAMP_RE.sub(" ", value)
    value = FISC_REDACTION_CODE_RE.sub(" ", value)
    value = FISC_USC_CITE_RE.sub(" ", value)
    value = FISC_SECTION_CITE_RE.sub(" ", value)
    value = FISC_DOCKET_RE.sub(" ", value)
    value = FISC_UNCLASSIFIED_BANNER_RE.sub(" ", value)
    value = FISC_DECLASSIFIED_RE.sub(" ", value)
    value = FISC_CERT_RE.sub(" ", value)
    value = FISC_JUDGE_LINE_RE.sub(" ", value)
    value = FISC_CLASSIFICATION_RE.sub(" ", value)
    value = FISC_XMASK_RE.sub(" ", value)
    value = re.sub(r"\bb[0-9a-z]+-\d+\b", " ", value, flags=re.I)
    value = re.sub(r"(?:,\s*\d+)+", " ", value)
    value = FISC_SENSITIVE_INFO_RE.sub(" ", value)
    value = FISC_AUTH_EXPIRY_RE.sub(" ", value)
    value = FISC_CLERK_CERT_RE.sub(" ", value)
    value = re.sub(r"\b(?:and\s+){2,}", "and ", value, flags=re.I)
    value = re.sub(r"^(?:and\s+)+", "", value, flags=re.I)
    value = re.sub(r"\s+\)\s+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_fisc_heading(text: str) -> str:
    value = clean_fisc_match_text(text)
    if not value:
        return ""
    value = re.sub(r"^(?:\(?\d+\)?[.)]?\s*)+", "", value)
    value = re.sub(r"^(?:[A-Z]\.\s*\(\s*\)\s*)+", "", value)
    value = FISC_HEADING_PREFIX_RE.sub("", value)
    value = FISC_SERIAL_CLASSIFICATION_PREFIX_RE.sub("", value)
    value = re.sub(r"^(?:and\s+)?(?:\d+[A-Za-z]?(?:\([A-Za-z0-9./-]+\)){1,8}\]?\s*)+", "", value, flags=re.I)
    value = FISC_NUMBERED_CITE_PREFIX_RE.sub("", value)
    value = FISC_SENATE_STAMP_RE.sub(" ", value)
    value = FISC_DATE_PREFIX_RE.sub("", value)
    value = FISC_SOURCE_DESCRIPTION_RE.sub("", value)
    value = FISC_INLINE_REDACTED_RE.sub("", value)
    value = re.sub(r"\bX{2,}\b", "", value)
    value = re.sub(r"\bPRIMARY ORDER AND WARRANT\b", "", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" -:;,.")
    if not value:
        return ""
    if any(pattern.fullmatch(value) for pattern in FISC_PROCEDURAL_HEADING_PATTERNS):
        return ""
    heading_tokens = lexical_tokens(value)
    if not heading_tokens:
        return ""
    if all(FISC_NOISEY_PREFIX_TOKEN_RE.fullmatch(token) for token in value.split()):
        return ""
    if len(heading_tokens) <= 2 and all(token in {"secret", "sensitive", "information", "docket", "number"} for token in heading_tokens):
        return ""
    if len(heading_tokens) <= 3 and any(ch.isdigit() for ch in value):
        return ""
    if value.lower() in {"source description", "sub-source", "sub-source #1"}:
        return value.title().replace("Source", "Source")
    return value


def strip_fisc_noise_prefixes(text: str) -> str:
    value = base.norm_space(text)
    if not value:
        return ""
    previous = None
    while value and value != previous:
        previous = value
        value = FISC_PUNCT_PREFIX_RE.sub("", value)
        value = FISC_LEEANN_PREFIX_RE.sub("", value)
        value = FISC_SIGNATURE_LINE_RE.sub("", value)
        value = FISC_DATE_PREFIX_RE.sub("", value)
        value = FISC_SENATE_STAMP_RE.sub(" ", value)
        value = re.sub(r"^(?:[A-Z]\.\s*\(\s*\)\s*)+", "", value)
        value = re.sub(r"^(?:and\s+)?(?:\d+[A-Za-z]?(?:\([A-Za-z0-9./-]+\)){1,8}\]?\s*)+", "", value, flags=re.I)
        value = FISC_SERIAL_CLASSIFICATION_PREFIX_RE.sub("", value)
        value = FISC_NUMBERED_CITE_PREFIX_RE.sub("", value)
        value = FISC_SOURCE_DESCRIPTION_RE.sub("", value)
        value = re.sub(r"^(?:\d+\s*)?(?:\(\s*X\s*\)\s*)+", "", value, flags=re.I)
        value = re.sub(r"^\d+\s+(?=[A-Za-z])", "", value)
        value = re.sub(r"\bPRIMARY ORDER AND WARRANT\b", "", value, flags=re.I)
        value = re.sub(r"\s+", " ", value).strip(" -:;,.()[]")
    return value


def clean_fisc_segment_text(text: str) -> str:
    value = clean_fisc_match_text(text)
    if not value:
        return ""
    value = strip_fisc_noise_prefixes(value)
    value = FISC_ARTICLE_FRAGMENT_RE.sub("", value)
    value = re.sub(r"^(?:\(\d+\)\([A-Z/.-]+\)\s*)+", "", value)
    value = re.sub(r"^\d+\s+(?=[A-Z][a-z])", "", value)
    value = re.sub(r"^Date\b", "", value).strip()
    tokens = value.split()
    prefix_probe = tokens[: min(20, len(tokens))]
    prefix_noise = 0.0
    if prefix_probe:
        prefix_noise = sum(
            1
            for token in prefix_probe
            if (
                token.lower() == "and"
                or token.isdigit()
                or token.upper() == "XXXXXXXX"
                or token.upper() == "XXXXX"
                or "u.s.c" in token.lower()
                or token.startswith("§")
                or FISC_SECTION_CITE_RE.fullmatch(token)
            )
        ) / len(prefix_probe)
    if prefix_noise >= 0.35:
        meaningful_start = FISC_MEANINGFUL_START_RE.search(value)
        if meaningful_start and meaningful_start.start() > 0:
            value = value[meaningful_start.start() :].strip()

    suffix_match = re.search(r"(?:\d+\.\s+The authorities approved\b|As to all information acquired\b).*", value)
    if suffix_match:
        value = suffix_match.group(0).strip()

    value = re.sub(r"\s+", " ", value).strip()
    return value


def looks_like_fisc_procedural_boilerplate(text: str, heading: str = "") -> bool:
    value = base.norm_space(text)
    title = base.norm_space(heading)
    if not value:
        return True
    if any(pattern.search(value) for pattern in FISC_PROCEDURAL_DROP_PATTERNS):
        return True
    if title and any(pattern.fullmatch(title) for pattern in FISC_PROCEDURAL_HEADING_PATTERNS):
        return True
    return False


def looks_like_fisc_signature_block(text: str) -> bool:
    value = base.norm_space(text)
    if not value:
        return False
    terminal_punct = bool(re.search(r"[.!?][\"')\]]?$", value))
    if "Attorney General of the United States" in value and not terminal_punct:
        return True
    if "Clerk, FISC" in value:
        return True
    return False


def looks_like_fisc_redaction_noise(text: str) -> bool:
    value = strip_fisc_noise_prefixes(clean_fisc_match_text(text))
    if not value:
        return True
    if looks_like_fisc_noise_text(value):
        return True
    if looks_like_fisc_procedural_boilerplate(value):
        return True
    if looks_like_fisc_signature_block(value):
        return True
    if any(pattern.search(value) for pattern in FISC_SPAN_DROP_PATTERNS):
        return True
    pieces = value.split()
    if not pieces:
        return True
    cite_like = sum(
        1
        for piece in pieces
        if (
            FISC_SECTION_CITE_RE.fullmatch(piece)
            or "u.s.c" in piece.lower()
            or piece.startswith("§")
            or FISC_SENATE_STAMP_RE.search(piece)
        )
    )
    if cite_like / max(1, len(pieces)) >= 0.3:
        return True
    alpha_tokens = [token for token in lexical_tokens(value) if re.search(r"[a-z]", token)]
    if not alpha_tokens:
        return True
    return False


def _fisc_bridge_ready(text: str) -> bool:
    value = base.norm_space(text)
    if "[REDACTED]" not in value:
        return False
    left, _, right = value.partition("[REDACTED]")
    return len(content_tokens(left)) >= 3 and len(content_tokens(right)) >= 3


def _fisc_context_is_substantive(row: pd.Series) -> bool:
    contexts = [
        clean_fisc_match_text(base.safe_str(row.get("local_context", ""))),
        clean_fisc_match_text(base.safe_str(row.get("bridge_context", ""))),
        clean_fisc_match_text(base.safe_str(row.get("source_unredacted_text", ""))),
    ]
    for context in contexts:
        if len(content_tokens(context)) >= 4 and not looks_like_fisc_signature_block(context):
            return True
    return False


def _fisc_has_incomplete_tail(text: str) -> bool:
    tokens = lexical_tokens(text)
    if not tokens:
        return True
    return tokens[-1].casefold() in FISC_INCOMPLETE_TAIL_TOKENS


def fisc_reclassify_redaction_index(redaction_df: pd.DataFrame) -> pd.DataFrame:
    if len(redaction_df) == 0:
        return redaction_df.copy()

    rows: List[Dict[str, object]] = []
    for _, row in redaction_df.iterrows():
        row_copy = dict(row)
        original_text = base.norm_space(row_copy.get("text", "") or row_copy.get("redacted_text", ""))
        cleaned = strip_fisc_noise_prefixes(original_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")
        heading = clean_fisc_heading(base.safe_str(row_copy.get("heading", "")))
        context_substantive = _fisc_context_is_substantive(pd.Series(row_copy))
        bridge_ready = _fisc_bridge_ready(base.safe_str(row_copy.get("bridge_context", "")))
        content_count = len(content_tokens(cleaned))
        word_count = len(cleaned.split())
        procedural = looks_like_fisc_procedural_boilerplate(cleaned, heading)
        signatureish = looks_like_fisc_signature_block(cleaned)
        incomplete_tail = _fisc_has_incomplete_tail(cleaned)
        procedural_signal = bool(
            FISC_PROCEDURAL_SIGNAL_RE.search(cleaned)
            or FISC_PROCEDURAL_SIGNAL_RE.search(heading)
            or LEGAL_CITE_RE.search(cleaned)
            or FISC_SECTION_CITE_RE.search(cleaned)
        )

        decision = base.safe_str(row_copy.get("decision", ""))
        kind = base.safe_str(row_copy.get("redaction_kind", ""))
        reason = base.safe_str(row_copy.get("recovery_reason", ""))

        hard_discard = any(pattern.search(cleaned) for pattern in FISC_ALWAYS_DISCARD_PATTERNS)
        if (
            not cleaned
            or hard_discard
            or (looks_like_fisc_noise_text(cleaned) and content_count <= 1)
            or (content_count == 0 and word_count <= 8)
        ):
            decision = "discard"
            kind = "discard"
            reason = reason or "fisc_noise"
        else:
            should_override_discard = (
                decision != "keep"
                and content_count >= 2
                and context_substantive
            )
            keep_candidate = should_override_discard or decision == "keep"
            sentence_candidate = (
                keep_candidate
                and bridge_ready
                and context_substantive
                and 11 <= word_count <= 60
                and not procedural
                and not procedural_signal
                and not signatureish
                and not incomplete_tail
            )
            value_candidate = (
                keep_candidate
                and word_count <= 10
                and not signatureish
            )
            if keep_candidate:
                decision = "keep"
                if sentence_candidate:
                    kind = "sentence"
                    reason = (
                        "sentence_like_gap"
                        if not should_override_discard
                        else "contextual_sentence_keep"
                    )
                elif value_candidate:
                    kind = "value"
                    reason = (
                        "short_value_like_gap"
                        if not should_override_discard
                        else "contextual_value_keep"
                    )
                else:
                    kind = "paragraph"
                    reason = (
                        "longer_passage_gap"
                        if not should_override_discard
                        else "contextual_paragraph_keep"
                    )
            else:
                decision = "discard"
                kind = "discard"
                reason = reason or "discarded"

        row_copy["text"] = cleaned or original_text
        row_copy["redacted_text"] = cleaned or original_text
        row_copy["token_count"] = len((cleaned or original_text).split())
        row_copy["word_count"] = len((cleaned or original_text).split())
        row_copy["char_count"] = len(cleaned or original_text)
        row_copy["heading"] = heading or base.safe_str(row_copy.get("heading", ""))
        row_copy["decision"] = decision
        row_copy["redaction_kind"] = kind
        row_copy["recovery_keep"] = "true" if decision == "keep" else "false"
        row_copy["recovery_reason"] = reason
        rows.append(row_copy)

    return pd.DataFrame(rows, columns=base.REDUCTION_KEEP_REDACTION_COLUMNS)


def looks_like_signature_roster(text: str) -> bool:
    value = base.norm_space(text)
    if not value:
        return False
    tokens = lexical_tokens(value)
    if len(tokens) < 12:
        return False
    hint_count = sum(1 for token in tokens if token in FISC_SIGNATURE_HINTS)
    capitalized_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+\b", value)
    return hint_count >= 5 and len(capitalized_words) >= 3


def infer_fisc_heading_from_text(text: str) -> str:
    value = base.norm_space(text)
    if not value:
        return ""
    first_sentence, _, _rest = value.partition(". ")
    sentence = first_sentence.rstrip(".")
    if not sentence:
        return ""
    words = sentence.split()
    if len(words) > 12:
        return ""
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    if len(alpha_words) < 3:
        return ""
    upper_like = sum(1 for word in alpha_words if word[:1].isupper() or word.isupper())
    if upper_like / max(1, len(alpha_words)) < 0.7:
        return ""
    return sentence


def fast_paragraph_match_score(left: str, right: str) -> float:
    left_norm = base.normalize_match_text(clean_fisc_match_text(left))
    right_norm = base.normalize_match_text(clean_fisc_match_text(right))
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

    prefix_matches = 0
    prefix_cap = min(12, len(left_tokens), len(right_tokens))
    for idx in range(prefix_cap):
        if left_tokens[idx] != right_tokens[idx]:
            break
        prefix_matches += 1
    prefix_ratio = prefix_matches / max(1, prefix_cap)

    left_bigrams = set(zip(left_tokens, left_tokens[1:]))
    right_bigrams = set(zip(right_tokens, right_tokens[1:]))
    if left_bigrams and right_bigrams:
        bigram_overlap = len(left_bigrams & right_bigrams) / max(1, min(len(left_bigrams), len(right_bigrams)))
    else:
        bigram_overlap = 0.0

    return max(overlap, 0.75 * overlap + 0.15 * prefix_ratio + 0.10 * bigram_overlap, bigram_overlap)


def looks_like_fisc_noise_text(text: str) -> bool:
    value = base.norm_space(text)
    if not value:
        return True
    tokens = lexical_tokens(value)
    if not tokens:
        return True
    if len(tokens) <= 2 and all(token in STOPWORDS for token in tokens):
        return True
    if len(tokens) <= 2 and all(FISC_CONTROL_RE.fullmatch(part.upper()) for part in value.split()):
        return True
    if FISC_UNCLASSIFIED_BANNER_RE.search(value):
        return True
    if FISC_DOCKET_RE.search(value):
        return True
    if FISC_DECLASSIFIED_RE.search(value):
        return True
    if FISC_CERT_RE.search(value):
        return True
    if FISC_JUDGE_LINE_RE.search(value):
        return True
    if DOC_STAMP_RE.fullmatch(value) and any(ch.isdigit() for ch in value):
        return True
    if MASK_RE.fullmatch(value):
        return True
    if FISC_XMASK_RE.fullmatch(value):
        return True
    if len(tokens) <= 3 and LEGAL_CITE_RE.search(value):
        return True
    if len(tokens) <= 3 and any(part.lower().startswith("senate-fisa") for part in value.split()):
        return True
    if len(tokens) == 1 and tokens[0] in {"date", "page", "investigation", "is", "as", "or", "and", "in"}:
        return True
    if len(tokens) <= 3 and all(token in {"secret", "xx", "secref", "secretm", "senket", "sensitive", "information"} for token in tokens):
        return True
    code_ratio = sum(1 for part in value.split() if FISC_CODE_TOKEN_RE.fullmatch(part)) / max(1, len(value.split()))
    if code_ratio >= 0.35:
        return True
    return False


def should_merge_bridge_fragment(text: str) -> bool:
    value = base.norm_space(text)
    if not value:
        return False
    tokens = lexical_tokens(value)
    if not tokens or len(tokens) > 3:
        return False
    if looks_like_fisc_noise_text(value):
        return False
    if tokens[0] in STOPWORDS:
        return True
    return value[:1].islower()


def fisc_normalize_source_segments(source_segments: pd.DataFrame) -> pd.DataFrame:
    normalized = ORIGINAL_NORMALIZE_SOURCE_SEGMENTS(source_segments)
    if len(normalized) == 0:
        return normalized

    out_rows: List[Dict[str, object]] = []
    for _, row in normalized.iterrows():
        row_copy = dict(row)
        unredacted_text = clean_fisc_segment_text(base.safe_str(row_copy.get("unredacted_text", "")))
        redacted_text = clean_fisc_segment_text(base.safe_str(row_copy.get("redacted_text", "")))
        heading = clean_fisc_heading(base.safe_str(row_copy.get("heading", "")))
        display_text = unredacted_text or redacted_text

        if looks_like_fisc_noise_text(display_text):
            continue
        if looks_like_fisc_procedural_boilerplate(display_text, heading):
            continue
        if looks_like_fisc_signature_block(display_text):
            continue
        if looks_like_signature_roster(display_text):
            continue
        if not heading:
            heading = infer_fisc_heading_from_text(display_text)
        if looks_like_fisc_procedural_boilerplate(display_text, heading):
            continue

        if out_rows and base.safe_str(out_rows[-1].get("doc_id", "")) == base.safe_str(row_copy.get("doc_id", "")):
            if should_merge_bridge_fragment(display_text):
                prev = out_rows[-1]
                prev["unredacted_text"] = base.append_segment_text(base.safe_str(prev.get("unredacted_text", "")), unredacted_text)
                prev["redacted_text"] = base.append_segment_text(base.safe_str(prev.get("redacted_text", "")), redacted_text)
                prev["redaction_count"] = int(prev.get("redaction_count", 0) or 0) + int(row_copy.get("redaction_count", 0) or 0)
                merged_display = base.norm_space(prev.get("unredacted_text", "")) or base.norm_space(prev.get("redacted_text", ""))
                prev["char_count"] = len(merged_display)
                prev["word_count"] = len(merged_display.split())
                continue
            if (
                not heading
                and not base.safe_str(out_rows[-1].get("heading", ""))
                and len(display_text.split()) <= 40
                and int(out_rows[-1].get("word_count", 0) or 0) <= 40
            ):
                prev = out_rows[-1]
                prev["unredacted_text"] = base.append_segment_text(base.safe_str(prev.get("unredacted_text", "")), unredacted_text)
                prev["redacted_text"] = base.append_segment_text(base.safe_str(prev.get("redacted_text", "")), redacted_text)
                prev["redaction_count"] = int(prev.get("redaction_count", 0) or 0) + int(row_copy.get("redaction_count", 0) or 0)
                merged_display = base.norm_space(prev.get("unredacted_text", "")) or base.norm_space(prev.get("redacted_text", ""))
                prev["char_count"] = len(merged_display)
                prev["word_count"] = len(merged_display.split())
                continue
            if (
                heading
                and out_rows[-1].get("heading") == heading
                and len(display_text.split()) <= 4
                and not re.search(r"[.!?]$", base.safe_str(out_rows[-1].get("unredacted_text", "")))
            ):
                prev = out_rows[-1]
                prev["unredacted_text"] = base.append_segment_text(base.safe_str(prev.get("unredacted_text", "")), unredacted_text)
                prev["redacted_text"] = base.append_segment_text(base.safe_str(prev.get("redacted_text", "")), redacted_text)
                prev["redaction_count"] = int(prev.get("redaction_count", 0) or 0) + int(row_copy.get("redaction_count", 0) or 0)
                merged_display = base.norm_space(prev.get("unredacted_text", "")) or base.norm_space(prev.get("redacted_text", ""))
                prev["char_count"] = len(merged_display)
                prev["word_count"] = len(merged_display.split())
                continue

        row_copy["heading"] = heading
        row_copy["unredacted_text"] = unredacted_text
        row_copy["redacted_text"] = redacted_text
        row_copy["char_count"] = len(display_text)
        row_copy["word_count"] = len(display_text.split())
        out_rows.append(row_copy)

    for idx, row in enumerate(out_rows, start=1):
        row["paragraph_id"] = f"seg{idx:04d}"
    return pd.DataFrame(out_rows)


def candidate_target_indices(
    red_text: str,
    targets: Sequence[Dict[str, object]],
    inverted_index: Dict[str, List[int]],
    *,
    max_candidates: int = 28,
) -> List[int]:
    tokens = content_tokens(red_text)
    overlap_scores: Dict[int, int] = defaultdict(int)
    for token in tokens:
        for idx in inverted_index.get(token, []):
            overlap_scores[idx] += 1
    if overlap_scores:
        scored = sorted(
            overlap_scores.items(),
            key=lambda item: (
                -item[1],
                abs(int(targets[item[0]]["token_count"]) - len(tokens)),
                item[0],
            ),
        )
        return [idx for idx, _ in scored[:max_candidates]]

    red_len = max(1, len(content_tokens(red_text)))
    scored = sorted(
        range(len(targets)),
        key=lambda idx: (
            abs(int(targets[idx]["token_count"]) - red_len),
            idx,
        ),
    )
    return scored[: min(max_candidates, len(scored))]


def fisc_find_redacted_only_omissions(
    doc_id: str,
    unredacted_units: Sequence[Dict[str, object]],
    redacted_paragraphs: Sequence[Dict[str, object]],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    targets: List[Dict[str, object]] = []
    inverted_index: Dict[str, List[int]] = defaultdict(list)
    for unit in unredacted_units:
        text = clean_fisc_match_text(base.norm_space(unit.get("unredacted_text", "")))
        if not text:
            continue
        heading = base.norm_space(unit.get("segment_title", "")) or base.norm_space(unit.get("section_heading", ""))
        row = {
            "text": text,
            "heading": heading,
            "token_count": len(content_tokens(text)),
        }
        target_idx = len(targets)
        targets.append(row)
        for token in set(content_tokens(text)):
            inverted_index[token].append(target_idx)

    omitted_rows: List[Dict[str, object]] = []
    nonempty_redacted_count = 0

    for red_idx, red_row in enumerate(redacted_paragraphs):
        red_text = clean_fisc_match_text(base.norm_space(red_row.get("unredacted_text", "")))
        if not red_text:
            continue
        nonempty_redacted_count += 1

        if looks_like_fisc_noise_text(red_text):
            continue

        threshold = base._match_threshold(red_text)
        best_score = 0.0
        best_heading = ""
        shortlist = candidate_target_indices(red_text, targets, inverted_index)
        for target_idx in shortlist:
            target = targets[target_idx]
            score = fast_paragraph_match_score(str(target["text"]), red_text)
            if score > best_score:
                best_score = score
                best_heading = str(target["heading"])
            if best_score >= 0.995:
                break

        if best_score >= threshold:
            continue

        if base.is_recoverable_paragraph(red_text) and base.should_log_omission_text(red_text):
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


def fisc_build_redaction_span_index(
    doc_id: str,
    kept_redactions: pd.DataFrame,
    source_segments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base_index = ORIGINAL_BUILD_REDACTION_SPAN_INDEX(
        doc_id,
        kept_redactions,
        source_segments=source_segments,
    )
    if len(base_index) == 0:
        return base_index
    return fisc_reclassify_redaction_index(base_index)


def main() -> None:
    base.find_redacted_only_omissions = fisc_find_redacted_only_omissions
    base.normalize_source_segments = fisc_normalize_source_segments
    base.build_redaction_span_index = fisc_build_redaction_span_index
    base.LONG_SEGMENT_AUDIT_THRESHOLD = 4500
    base.main()


if __name__ == "__main__":
    main()
