#!/usr/bin/env python3
"""Materialize a single-source text corpus into pair-style prepared folders.

This adapter is for corpora like Vietnam and Iran that already exist as clean
text tables rather than redacted/unredacted PDF pairs. It emits the same folder
shape that `kg_scripts/stage_a.py` expects from CIB/FISC/Canberra:

<prepared-root>/<doc_id>/
  document/
    pair_metadata.json
    prompt.txt
    unredacted.transcription.txt
    redacted.transcription.txt
  difference/
    unredacted_bracketed.filtered.aligned.txt
    unredacted.aligned.txt
    redacted.aligned.txt
    original.unredacted.aligned.txt
    original.redacted.aligned.txt
    page_alignment.json
    original.page_alignment.json
    original.difference.summary.json
    postprocess_summary.json
    kept_redactions.jsonl

The redacted and unredacted variants are intentionally identical here. True
redaction tasks for these corpora are synthetic downstream tasks, not literal
pair diffs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def normalize_space(text: object) -> str:
    return " ".join(str(text or "").split())


def detect_column(df: pd.DataFrame, preferred: list[str], *, required: bool = True) -> Optional[str]:
    lowered = {str(col).strip().lower(): str(col) for col in df.columns}
    for candidate in preferred:
        hit = lowered.get(candidate.lower())
        if hit:
            return hit
    if required:
        raise ValueError(
            f"Could not find any of {preferred} in source columns: {list(df.columns)}"
        )
    return None


def normalize_date_fields(value: object) -> tuple[str, str, str]:
    text = normalize_space(value)
    if not text:
        return "", "", ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        year = year_match.group(0) if year_match else ""
        return text, text, year
    return parsed.strftime("%Y-%m-%d"), parsed.strftime("%Y%m%d"), parsed.strftime("%Y")


def clean_body_text(text: object) -> str:
    value = str(text or "").replace("\ufeff", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]

    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
            continue
        blank_run = 0
        cleaned.append(stripped)

    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    out = "\n".join(cleaned).strip()
    if not out:
        return ""
    return out + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_csv_row(
    *,
    collection_name: str,
    record_id: str,
    date_display: str,
    date_compact: str,
    doc_id: str,
    document_url: str,
    meta_path: str,
    title: str,
    year: str,
    source_url: str,
) -> list[str]:
    return [
        collection_name,
        record_id,
        date_display,
        source_url,
        date_compact,
        doc_id,
        document_url,
        meta_path,
        title,
        year,
        "0",
    ]


def build_pair_metadata(
    *,
    row: dict[str, Any],
    source_csv: Path,
    collection_name: str,
    doc_id: str,
    record_id: str,
    title: str,
    date_display: str,
    date_compact: str,
    year: str,
    document_url: str,
    source_url: str,
    row_index_1based: int,
) -> dict[str, Any]:
    return {
        "pair_key": doc_id,
        "model": "local_single_source_pair_materializer",
        "csv_row": build_csv_row(
            collection_name=collection_name,
            record_id=record_id,
            date_display=date_display,
            date_compact=date_compact,
            doc_id=doc_id,
            document_url=document_url,
            meta_path=str(source_csv),
            title=title,
            year=year,
            source_url=source_url,
        ),
        "row_index_1based": row_index_1based,
        "source_csv": str(source_csv),
        "source_kind": "single_source_text_table",
        "source_row": row,
    }


def build_postprocess_summary(
    *,
    doc_id: str,
    subject: str,
    text: str,
    collection_name: str,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "subject": subject,
        "collection": collection_name,
        "pair_mode": "synthetic_single_source",
        "segment_basis": "source_text_table",
        "char_count": len(text),
        "word_count": len(text.split()),
        "redaction_rows": 0,
    }


def materialize_document(
    *,
    doc_root: Path,
    doc_id: str,
    title: str,
    body_text: str,
    pair_metadata: dict[str, Any],
    postprocess_summary: dict[str, Any],
) -> None:
    document_dir = doc_root / "document"
    difference_dir = doc_root / "difference"

    write_json(document_dir / "pair_metadata.json", pair_metadata)
    write_text(document_dir / "prompt.txt", f"{title}\n")
    write_text(document_dir / "unredacted.transcription.txt", body_text)
    write_text(document_dir / "redacted.transcription.txt", body_text)

    for name in (
        "unredacted_bracketed.filtered.aligned.txt",
        "unredacted.aligned.txt",
        "redacted.aligned.txt",
        "original.unredacted.aligned.txt",
        "original.redacted.aligned.txt",
    ):
        write_text(difference_dir / name, body_text)
    write_json(difference_dir / "page_alignment.json", [])
    write_json(difference_dir / "original.page_alignment.json", [])
    write_json(difference_dir / "original.difference.summary.json", {"doc_id": doc_id, "redaction_rows": 0})
    write_json(difference_dir / "postprocess_summary.json", postprocess_summary)
    write_text(difference_dir / "kept_redactions.jsonl", "")


def materialize_corpus(
    *,
    source_csv: Path,
    prepared_root: Path,
    collection_name: str,
    doc_id_column: Optional[str],
    subject_column: Optional[str],
    date_column: Optional[str],
    body_column: Optional[str],
    record_id_column: Optional[str],
    document_url_column: Optional[str],
    source_url_column: Optional[str],
    limit: Optional[int],
    overwrite: bool,
) -> dict[str, Any]:
    df = pd.read_csv(source_csv, low_memory=False)
    if limit is not None:
        df = df.head(limit).copy()

    doc_col = doc_id_column or detect_column(df, ["doc_id", "id"])
    subject_col = subject_column or detect_column(df, ["subject", "title"])
    date_col = date_column or detect_column(df, ["date", "date_display"], required=False)
    body_col = body_column or detect_column(df, ["body", "text", "content"])
    record_col = record_id_column or detect_column(df, ["id", "record_id", "doc_id"], required=False)
    document_url_col = document_url_column or detect_column(df, ["document_url", "url"], required=False)
    source_url_col = source_url_column or detect_column(df, ["source_pdf_url", "source_url", "pdf_url"], required=False)

    if prepared_root.exists() and overwrite:
        shutil.rmtree(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    written_ids: list[str] = []
    skipped_ids: list[str] = []

    for idx, row in df.iterrows():
        row_dict = {
            str(key): ("" if pd.isna(value) else value)
            for key, value in row.to_dict().items()
        }
        doc_id = normalize_space(row_dict.get(doc_col, ""))
        if not doc_id:
            skipped_ids.append(f"<row_{idx + 1}>")
            continue

        title = normalize_space(row_dict.get(subject_col, "")) or doc_id
        record_id = normalize_space(row_dict.get(record_col, "")) or doc_id
        body_text = clean_body_text(row_dict.get(body_col, ""))
        if not body_text:
            skipped_ids.append(doc_id)
            continue

        date_display, date_compact, year = normalize_date_fields(row_dict.get(date_col, "")) if date_col else ("", "", "")
        document_url = normalize_space(row_dict.get(document_url_col, "")) if document_url_col else ""
        source_url = normalize_space(row_dict.get(source_url_col, "")) if source_url_col else ""

        pair_metadata = build_pair_metadata(
            row=row_dict,
            source_csv=source_csv,
            collection_name=collection_name,
            doc_id=doc_id,
            record_id=record_id,
            title=title,
            date_display=date_display,
            date_compact=date_compact,
            year=year,
            document_url=document_url,
            source_url=source_url,
            row_index_1based=idx + 2,
        )
        summary = build_postprocess_summary(
            doc_id=doc_id,
            subject=title,
            text=body_text,
            collection_name=collection_name,
        )
        materialize_document(
            doc_root=prepared_root / doc_id,
            doc_id=doc_id,
            title=title,
            body_text=body_text,
            pair_metadata=pair_metadata,
            postprocess_summary=summary,
        )
        written_ids.append(doc_id)

    manifest = {
        "status": "ok",
        "source_csv": str(source_csv),
        "prepared_root": str(prepared_root),
        "collection_name": collection_name,
        "row_count": int(len(df)),
        "written_count": int(len(written_ids)),
        "skipped_count": int(len(skipped_ids)),
        "written_doc_ids": written_ids,
        "skipped_doc_ids": skipped_ids,
        "columns": list(df.columns),
    }
    write_json(prepared_root / "prepare_single_source_summary.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--doc-id-column", default=None)
    parser.add_argument("--subject-column", default=None)
    parser.add_argument("--date-column", default=None)
    parser.add_argument("--body-column", default=None)
    parser.add_argument("--record-id-column", default=None)
    parser.add_argument("--document-url-column", default=None)
    parser.add_argument("--source-url-column", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = materialize_corpus(
        source_csv=Path(args.source_csv).expanduser().resolve(),
        prepared_root=Path(args.prepared_root).expanduser().resolve(),
        collection_name=args.collection_name,
        doc_id_column=args.doc_id_column,
        subject_column=args.subject_column,
        date_column=args.date_column,
        body_column=args.body_column,
        record_id_column=args.record_id_column,
        document_url_column=args.document_url_column,
        source_url_column=args.source_url_column,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
