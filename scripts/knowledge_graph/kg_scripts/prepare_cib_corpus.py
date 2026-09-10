#!/usr/bin/env python3
"""Prepare CIB redacted/unredacted PDF pairs for the packaged KG pipeline.

This is the CIB analogue of `prepare_fisc_corpus.py`, but it reuses the
packaged pair-transcription and postprocessing helpers directly.

Starting from:

- `data/raw/cib/cibcia.csv`
- `data/raw/cib/redacted_pdfs/`
- `data/raw/cib/unredacted_pdfs/`

it produces:

1. a transcription staging root with per-pair text and alignment artifacts; and
2. a final pair-style prepared corpus root consumed by `kg_scripts/run_pipeline.py`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upstream_pair_processing.openai_pdf_pipeline import (
    TRANSCRIPTION_PROMPT,
    transcribe_pairs,
)
from upstream_pair_processing.postprocess_redaction_labels import run_postprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-root",
        default=str(PACKAGE_ROOT / "data" / "raw" / "cib"),
        help="Folder containing cibcia.csv plus redacted_pdfs/ and unredacted_pdfs/.",
    )
    parser.add_argument(
        "--staging-root",
        default=str(PACKAGE_ROOT / "data" / "intermediate" / "cib" / "transcription_staging"),
        help="Intermediate transcription/alignment staging root.",
    )
    parser.add_argument(
        "--prepared-root",
        default=str(PACKAGE_ROOT / "data" / "prepared" / "cib"),
        help="Final prepared root compatible with kg_scripts Stage A.",
    )
    parser.add_argument("--pair-key", default=None, help="Optional single pair key for testing.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap for smoke runs.")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Reuse an existing staging root and run only the postprocess step.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the staging/prepared roots before rebuilding.",
    )
    parser.add_argument(
        "--cleanup-staging",
        action="store_true",
        help="Delete the transcription staging root after successful postprocess.",
    )
    args = parser.parse_args()

    docs_root = Path(args.docs_root).expanduser().resolve()
    staging_root = Path(args.staging_root).expanduser().resolve()
    prepared_root = Path(args.prepared_root).expanduser().resolve()

    if args.overwrite:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        if prepared_root.exists():
            shutil.rmtree(prepared_root)

    if not args.skip_transcription:
        transcribe_pairs(
            docs_root=docs_root,
            out_root=staging_root,
            model=args.openai_model,
            prompt=TRANSCRIPTION_PROMPT,
            csv_name="cibcia.csv",
            redacted_dir_name="redacted_pdfs",
            unredacted_dir_name="unredacted_pdfs",
            pair_key=args.pair_key,
            max_pairs=args.max_pairs,
            skip_existing=not bool(args.overwrite),
        )

    run_postprocess(
        in_root=staging_root,
        out_root=prepared_root,
        pair_key=args.pair_key,
        max_pairs=args.max_pairs,
        copy_source_pdfs=True,
        min_tokens=2,
        full_page_coverage=0.95,
        low_alignment_score=0.25,
        min_full_page_tokens=50,
        ocr_similarity_threshold=0.50,
        short_ocr_similarity_threshold=0.50,
        alternate_near_full_page_coverage=0.90,
        progress_every=50,
    )

    if args.cleanup_staging and staging_root.exists():
        shutil.rmtree(staging_root)

    print("Prepared CIB corpus written to:", prepared_root)


if __name__ == "__main__":
    main()
