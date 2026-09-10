#!/usr/bin/env python3
"""Materialize the Canberra cables as a pair-style prepared corpus root.

The kg_scripts extraction stage expects the same directory shape used by the CIB/FISC
prepared corpora. Canberra only has single-source raw cables, so this script
creates identical redacted/unredacted transcription files per cable and minimal
alignment/debug artifacts required by Stage A.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_body_text(document: dict[str, Any]) -> str:
    body_lines = document.get("body_lines") or []
    if isinstance(body_lines, list):
        cleaned = [str(line).strip() for line in body_lines if normalize_space(str(line))]
        if cleaned:
            return "\n".join(cleaned).strip() + "\n"
    body_text = str(document.get("body_text") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if body_text:
        return body_text + "\n"
    raise ValueError(f"Document {document.get('doc_id', '<unknown>')} has no body text")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_pair_metadata(document: dict[str, Any], source_json: Path) -> dict[str, Any]:
    doc_id = normalize_space(document.get("doc_id", ""))
    date_value = normalize_space(document.get("date", ""))
    subject = normalize_space(document.get("subject", "")) or doc_id
    year = date_value[:4] if len(date_value) >= 4 else ""
    csv_row = [
        "Canberra Mosaic Corpus",
        doc_id,
        date_value,
        "",
        date_value,
        doc_id,
        "",
        str(source_json),
        subject,
        year,
        "",
    ]
    return {
        "pair_key": doc_id,
        "model": "local_canberra_pair_materializer",
        "csv_row": csv_row,
        "source_json": str(source_json),
        "classification": normalize_space(document.get("classification", "")),
        "signature": normalize_space(document.get("signature", "")),
        "handling": normalize_space(document.get("handling", "")),
        "notes": normalize_space(document.get("notes", "")),
    }


def build_postprocess_summary(doc_id: str, subject: str, text: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "subject": subject,
        "pair_mode": "synthetic_single_source",
        "segment_basis": "raw_cable_text",
        "char_count": len(text),
        "word_count": len(text.split()),
        "redaction_rows": 0,
    }


def materialize_document(document: dict[str, Any], source_json: Path, out_root: Path, overwrite: bool) -> None:
    doc_id = normalize_space(document.get("doc_id", ""))
    if not doc_id:
        raise ValueError("Encountered Canberra document without doc_id")

    body_text = normalize_body_text(document)
    subject = normalize_space(document.get("subject", "")) or doc_id
    doc_root = out_root / doc_id
    if doc_root.exists() and overwrite:
        shutil.rmtree(doc_root)
    doc_root.mkdir(parents=True, exist_ok=True)

    pair_metadata = build_pair_metadata(document, source_json)
    summary = build_postprocess_summary(doc_id, subject, body_text)

    document_dir = doc_root / "document"
    difference_dir = doc_root / "difference"

    write_json(document_dir / "pair_metadata.json", pair_metadata)
    write_text(document_dir / "prompt.txt", f"{subject}\n")
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
    write_json(difference_dir / "postprocess_summary.json", summary)
    write_text(difference_dir / "kept_redactions.jsonl", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-json",
        default=str(PACKAGE_ROOT / "data" / "raw" / "canberra" / "canberra_cables.json"),
        help="Path to the Canberra cable JSON source.",
    )
    parser.add_argument(
        "--prepared-root",
        default=str(PACKAGE_ROOT / "data" / "prepared" / "canberra"),
        help="Output prepared root compatible with kg_scripts Stage A.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_json = Path(args.source_json).expanduser().resolve()
    out_root = Path(args.prepared_root).expanduser().resolve()
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    documents = payload.get("documents") or []
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"No documents found in {source_json}")

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for document in documents:
        materialize_document(document, source_json, out_root, overwrite=False)

    summary = {
        "status": "ok",
        "source_json": str(source_json),
        "prepared_root": str(out_root),
        "document_count": len(documents),
        "document_ids": [normalize_space(doc.get("doc_id", "")) for doc in documents],
    }
    write_json(out_root / "prepare_canberra_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
