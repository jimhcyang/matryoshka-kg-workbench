#!/usr/bin/env python3
"""Task-level build behavior for benchmark generation.

This keeps task-specific catalog/selection policy out of the main builder so
changes to one task do not silently alter another.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskBuildBehavior:
    catalog_merge_mode: str = "strict_only"
    selection_source: str = "strict"
    selection_strategy: str = "round_robin"
    allow_selected_backfill_from_catalog: bool = False


DEFAULT_TASK_BUILD_BEHAVIORS: dict[str, TaskBuildBehavior] = {
    "reconstruction": TaskBuildBehavior(
        catalog_merge_mode="strict_only",
        selection_source="strict",
        selection_strategy="round_robin",
        allow_selected_backfill_from_catalog=True,
    ),
    "claims": TaskBuildBehavior(
        catalog_merge_mode="claims_relaxed_merge",
        selection_source="strict",
        selection_strategy="round_robin",
        allow_selected_backfill_from_catalog=True,
    ),
    "value_redaction": TaskBuildBehavior(
        catalog_merge_mode="strict_only",
        selection_source="catalog",
        selection_strategy="value_redaction_rebalance",
        allow_selected_backfill_from_catalog=True,
    ),
    "sentence_redaction": TaskBuildBehavior(
        catalog_merge_mode="strict_only",
        selection_source="strict",
        selection_strategy="round_robin",
        allow_selected_backfill_from_catalog=False,
    ),
    "contradictions": TaskBuildBehavior(
        catalog_merge_mode="contradictions_relaxed_merge",
        selection_source="strict",
        selection_strategy="round_robin",
        allow_selected_backfill_from_catalog=True,
    ),
}


def resolve_task_build_behavior(
    task_name: str,
    *,
    enable_small_corpus_parser_vet: bool,
) -> TaskBuildBehavior:
    base = DEFAULT_TASK_BUILD_BEHAVIORS.get(task_name, TaskBuildBehavior())
    if not enable_small_corpus_parser_vet:
        return TaskBuildBehavior(
            catalog_merge_mode="strict_only",
            selection_source=base.selection_source,
            selection_strategy=base.selection_strategy,
            allow_selected_backfill_from_catalog=base.allow_selected_backfill_from_catalog,
        )
    return base
