#!/usr/bin/env python3
"""Preparation logic for single-source text corpora like Vietnam and Iran.

These corpora are already plain text and do not have real redacted/unredacted
pair differences. We still materialize them into the pair-shaped postprocessed
contract so the rest of the pipeline can stay shared, but the preparation logic
must be simpler than the CIA/FISC path:

- do not infer CIA-style headings from all-caps cable text
- preserve content aggressively
- split numbered cable sections into separate units
- chunk very long sections into sentence windows
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from kg_scripts import _prepare_pairs as base
from kg_scripts._common import norm_space, safe_str


LONG_SEGMENT_AUDIT_THRESHOLD = 2000

SERIAL_START_RE = re.compile(r"^(?:\(?\d{1,3}[.)]|[A-Z][.)]|--)\s+")
TRAILING_SIGNATURE_RE = re.compile(r"^[A-Z][A-Z .'-]{1,40}$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
QUOTE_MARKER_RE = re.compile(r"\b(?:QTE|NQTE|UNQTE|QUOTE|END TEXT)\b", re.IGNORECASE)
ROUTE_HEADER_INLINE_RE = re.compile(
    r"-{4,}\s*\d{6}Z.*?\b(?:CONFIDENTIAL|SECRET|UNCLASSIFIED|C\s*O\s*N\s*F\s*I\s*D\s*E\s*N\s*T\s*I\s*A\s*L)\b.*?\bSECTION\s+\d+\s+OF\s+\d+\b\s+[A-Z]+\s+\d+\b",
    re.IGNORECASE,
)
REPEATED_SINGLE_LETTERS_RE = re.compile(r"(?:\b[A-Z]\b\s+){5,}[A-Z]\b")
TRAILING_ATTACHED_SIGNATURE_RE = re.compile(r"([.!?])([A-Z][A-Z]{3,})$")


def _normalize_lines(raw_text: str) -> List[str]:
    text = raw_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    pages = base.split_pages(text)
    lines: List[str] = []
    for _, page_body in pages:
        for raw_line in page_body.splitlines():
            line = raw_line.strip()
            if not line:
                lines.append("")
                continue
            if base.is_explicit_drop_line(line):
                continue
            lines.append(line)
    return lines


def _looks_like_signature(line: str) -> bool:
    value = norm_space(line)
    if not value or value.endswith(":"):
        return False
    tokens = re.findall(r"[A-Z][A-Z.'-]*", value)
    if not tokens or len(tokens) > 4:
        return False
    return bool(TRAILING_SIGNATURE_RE.fullmatch(value))


def _strip_serial_prefix(text: str) -> str:
    value = norm_space(text)
    value = SERIAL_START_RE.sub("", value)
    return value.strip()


def _strip_quote_markers(text: str) -> str:
    value = QUOTE_MARKER_RE.sub(" ", safe_str(text))
    value = re.sub(r"\s+", " ", value).strip(" ,;:-")
    return value


def _cleanup_generic_text(text: str) -> str:
    value = safe_str(text)
    value = ROUTE_HEADER_INLINE_RE.sub(" ", value)
    value = REPEATED_SINGLE_LETTERS_RE.sub(" ", value)
    value = TRAILING_ATTACHED_SIGNATURE_RE.sub(r"\1", value)
    value = _strip_quote_markers(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _join_generic_lines(left: str, right: str) -> str:
    left_text = safe_str(left).strip()
    right_text = safe_str(right).strip()
    if left_text.endswith("-") and right_text:
        first_alpha = next((char for char in right_text if char.isalpha()), "")
        if first_alpha:
            return left_text[:-1] + right_text
    return f"{left_text} {right_text}".strip()


def _join_lines(lines: Sequence[str]) -> str:
    if not lines:
        return ""
    trimmed = [line.strip() for line in lines if norm_space(line)]
    if not trimmed:
        return ""
    joined = trimmed[0]
    for line in trimmed[1:]:
        joined = _join_generic_lines(joined, line)
    joined = norm_space(joined)
    joined = _strip_serial_prefix(joined)
    joined = _cleanup_generic_text(joined)
    return joined


def _build_blocks(lines: Sequence[str]) -> List[str]:
    blocks: List[str] = []
    current: List[str] = []

    def flush(*, is_final: bool = False) -> None:
        nonlocal current
        candidate = list(current)
        if is_final and len(candidate) > 1 and _looks_like_signature(candidate[-1]):
            candidate = candidate[:-1]
        text = _join_lines(candidate)
        if text:
            blocks.append(text)
        current = []

    for line in lines:
        if not norm_space(line):
            flush()
            continue
        if SERIAL_START_RE.match(line) and current:
            flush()
        current.append(line)
    flush(is_final=True)
    return blocks


def _split_into_sentences(text: str) -> List[str]:
    value = norm_space(text)
    if not value:
        return []
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(value) if part.strip()]
    if len(parts) <= 1:
        return [value]
    return parts


def _chunk_block(text: str, *, max_words: int = 110, max_chars: int = 750) -> List[str]:
    value = norm_space(text)
    if not value:
        return []
    if len(value.split()) <= max_words and len(value) <= max_chars:
        return [value]

    sentences = _split_into_sentences(value)
    if len(sentences) <= 1:
        words = value.split()
        if len(words) <= max_words:
            return [value]
        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + max_words)
            chunks.append(" ".join(words[start:end]).strip())
            start = end
        return chunks

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    current_chars = 0
    for sentence in sentences:
        sent_words = len(sentence.split())
        sent_chars = len(sentence)
        if current and (current_words + sent_words > max_words or current_chars + sent_chars > max_chars):
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0
            current_chars = 0
        current.append(sentence)
        current_words += sent_words
        current_chars += sent_chars + 1
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def build_generic_units(raw_text: str) -> List[Dict[str, object]]:
    lines = _normalize_lines(raw_text)
    blocks = _build_blocks(lines)
    units: List[Dict[str, object]] = []
    seg_idx = 0
    for block in blocks:
        for chunk in _chunk_block(block):
            chunk_text = norm_space(chunk)
            if not chunk_text:
                continue
            seg_idx += 1
            units.append(
                {
                    "section_id": "",
                    "section_heading": "",
                    "paragraph_id": f"seg{seg_idx:04d}",
                    "segment_id": f"seg{seg_idx:04d}",
                    "segment_order": seg_idx,
                    "segment_kind": "text_block",
                    "segment_title": "",
                    "is_heading": False,
                    "backbone_text": chunk_text,
                    "unredacted_text": chunk_text,
                    "redacted_text": chunk_text,
                    "redaction_count": 0,
                }
            )
    return units


def build_source_segments(
    doc_id: str,
    date_value: str,
    subject: str,
    source_path: str,
    units: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, unit in enumerate(units, start=1):
        text = _cleanup_generic_text(norm_space(unit.get("unredacted_text", "")))
        if not text:
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "date": date_value,
                "subject": subject,
                "source_path": source_path,
                "paragraph_id": safe_str(unit.get("paragraph_id", f"seg{idx:04d}")),
                "heading": "",
                "redaction_count": 0,
                "char_count": len(text),
                "word_count": len(text.split()),
                "unredacted_text": text,
                "redacted_text": text,
            }
        )
    return normalize_source_segments(pd.DataFrame(rows))


def normalize_source_segments(source_segments: pd.DataFrame) -> pd.DataFrame:
    if len(source_segments) == 0:
        return source_segments.copy()

    out_rows: List[Dict[str, object]] = []
    for _, row in source_segments.iterrows():
        row_copy = dict(row)
        text = _cleanup_generic_text(norm_space(row_copy.get("unredacted_text", "")))
        if not text:
            continue
        row_copy["unredacted_text"] = text
        row_copy["redacted_text"] = text
        row_copy["char_count"] = len(text)
        row_copy["word_count"] = len(text.split())

        if out_rows:
            prev = out_rows[-1]
            prev_text = safe_str(prev.get("unredacted_text", ""))
            if (
                safe_str(prev.get("doc_id", "")) == safe_str(row_copy.get("doc_id", ""))
                and len(text.split()) <= 14
                and (
                    prev_text.endswith(":")
                    or prev_text.endswith(",")
                    or not re.search(r"[.!?]$", prev_text)
                )
            ):
                merged = f"{prev_text} {text}".strip()
                prev["unredacted_text"] = merged
                prev["redacted_text"] = merged
                prev["char_count"] = len(merged)
                prev["word_count"] = len(merged.split())
                continue

        out_rows.append(row_copy)

    for idx, row in enumerate(out_rows, start=1):
        row["paragraph_id"] = f"seg{idx:04d}"
    return pd.DataFrame(out_rows)


def build_document_record(
    pair_dir: Path,
    prepared_dir: Path,
    transcription_variant: str,
    gap_marker: str,
) -> Optional[Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, str]]]:
    metadata_path = pair_dir / "document" / "pair_metadata.json"
    raw_unredacted_path = pair_dir / "document" / "unredacted.transcription.txt"
    raw_redacted_path = pair_dir / "document" / "redacted.transcription.txt"

    if not metadata_path.exists() or not raw_unredacted_path.exists():
        return None

    metadata = base.parse_pair_metadata(metadata_path)
    doc_id = metadata["pair_key"] or pair_dir.name
    source_id = metadata["cia_document_id"] or doc_id
    subject = metadata["title"] or metadata["collection"] or source_id
    date_value = base.normalize_date(metadata["date_compact"], metadata["date_display"])

    variant_raw_map = {
        "unredacted": raw_unredacted_path.read_text(encoding="utf-8", errors="ignore"),
        "redacted": raw_redacted_path.read_text(encoding="utf-8", errors="ignore")
        if raw_redacted_path.exists()
        else raw_unredacted_path.read_text(encoding="utf-8", errors="ignore"),
    }

    units = build_generic_units(variant_raw_map["unredacted"])
    if not units:
        return None

    variant_source_path = raw_unredacted_path if transcription_variant == "unredacted" else raw_redacted_path
    source_segments = build_source_segments(
        doc_id=doc_id,
        date_value=date_value,
        subject=subject,
        source_path=str(variant_source_path),
        units=units,
    )
    if len(source_segments) == 0:
        return None

    variant_bodies = {
        "unredacted": base.build_document_body_with_options(source_segments, variant="unredacted", preserve_empty=False),
        "redacted": base.build_document_body_with_options(source_segments, variant="redacted", preserve_empty=True),
    }
    document_body = variant_bodies[transcription_variant]
    if not document_body:
        return None

    kept_redactions = pd.DataFrame(columns=base.REDUCTION_KEEP_REDACTION_COLUMNS)
    omitted_redacted_only_df = pd.DataFrame(
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

    summary = {
        "doc_id": doc_id,
        "source_id": source_id,
        "subject": subject,
        "date": date_value,
        "transcription_variant": transcription_variant,
        "segment_count": int(len(source_segments)),
        "segments_with_redactions": 0,
        "mean_segment_words": round(float(pd.to_numeric(source_segments["word_count"], errors="coerce").fillna(0).mean()), 2),
        "max_segment_chars": int(pd.to_numeric(source_segments["char_count"], errors="coerce").fillna(0).max()),
        "kept_redaction_rows": 0,
        "page_alignment_rows": 0,
        "source_path": str(variant_source_path),
        "matched_redacted_paragraphs": 0,
        "omitted_redacted_only_paragraphs": 0,
        "unmatched_unredacted_paragraphs": 0,
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
        "transcription_path": str(variant_source_path),
        "metadata_path": str(metadata_path),
        "tag": "",
        "concepts": "",
    }

    base.write_debug_source_copy(
        prepared_dir / "original_sources" / "aligned_unredacted_rendered" / f"{doc_id}.txt",
        variant_raw_map["unredacted"],
    )
    base.write_debug_source_copy(
        prepared_dir / "original_sources" / "aligned_redacted_rendered" / f"{doc_id}.txt",
        variant_raw_map["redacted"],
    )
    base.copy_original_transcription(pair_dir, prepared_dir, doc_id, "unredacted")
    base.copy_original_transcription(pair_dir, prepared_dir, doc_id, "redacted")
    base.copy_source_pdfs(pair_dir, prepared_dir, doc_id)
    return record, source_segments, kept_redactions, omitted_redacted_only_df, summary, variant_bodies


build_dataframe = base.build_dataframe
build_redaction_review_table = base.build_redaction_review_table
build_long_segment_audit = base.build_long_segment_audit
write_variant_document = base.write_variant_document
