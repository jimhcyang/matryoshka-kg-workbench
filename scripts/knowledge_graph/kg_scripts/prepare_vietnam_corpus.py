#!/usr/bin/env python3
"""Convenience wrapper for materializing the Vietnam corpus as pair-style prepared text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_scripts.prepare_single_source_corpus import main as prepare_single_source_main

PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        default=str(PACKAGE_ROOT / "data" / "raw" / "vietnam" / "source_input.csv"),
        help="Vietnam source CSV with doc_id/subject/date/body columns.",
    )
    parser.add_argument(
        "--prepared-root",
        default=str(PACKAGE_ROOT / "data" / "prepared" / "vietnam"),
        help="Output prepared root compatible with kg_scripts Stage A.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    prepare_single_source_main(
        [
            "--source-csv",
            args.source_csv,
            "--prepared-root",
            args.prepared_root,
            "--collection-name",
            "Vietnam Diplomatic Archive",
            "--doc-id-column",
            "doc_id",
            "--subject-column",
            "subject",
            "--date-column",
            "date",
            "--body-column",
            "body",
            "--record-id-column",
            "id",
            *(["--limit", str(args.limit)] if args.limit is not None else []),
            *(["--overwrite"] if args.overwrite else []),
        ]
    )


if __name__ == "__main__":
    main()
