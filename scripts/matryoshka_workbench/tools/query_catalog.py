#!/usr/bin/env python3
"""Helpers for exporting retained query catalogs during benchmark builds."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

QUERY_CATALOG_FIELDNAMES = [
    "corpus",
    "corpus_full",
    "task_family",
    "legacy_case_id",
    "run_case_id",
    "selected_for_run",
    "selection_rank",
    "catalog_rank",
    "doc_id",
    "paragraph_id",
    "query_kind",
    "selector_score",
    "word_count",
    "query_target_words",
    "query_text",
    "ground_truth",
    "source_context",
    "notes",
]


def write_query_catalog_files(
    summary_root: Path,
    catalog_rows: list[dict[str, Any]],
) -> Path:
    summary_root.mkdir(parents=True, exist_ok=True)
    catalog_path = summary_root / "final_query_catalog.csv"
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUERY_CATALOG_FIELDNAMES)
        writer.writeheader()
        for row in catalog_rows:
            writer.writerow({key: row.get(key, "") for key in QUERY_CATALOG_FIELDNAMES})
    return catalog_path
