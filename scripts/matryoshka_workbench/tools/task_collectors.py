#!/usr/bin/env python3
"""Collector construction helpers for benchmark task families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


CollectorFn = Callable[[], list[dict[str, Any]]]


@dataclass
class TaskCollectorBundle:
    collectors: dict[str, CollectorFn]
    collect_claim_cases_local: Callable[[dict[str, Any] | None], list[dict[str, Any]]] | None
    collect_contradiction_cases_local: Callable[[dict[str, Any] | None], list[dict[str, Any]]] | None
    get_reconstruction_seed_cases: Callable[[], list[dict[str, Any]]]


def build_task_collectors(
    *,
    task_counts: dict[str, int],
    collector_counts: dict[str, int],
    sampling_with_doc_cap: dict[str, dict[str, Any]],
    seed: int,
    runner_cfg: dict[str, Any],
    entities: pd.DataFrame,
    chunks: pd.DataFrame,
    para_richness: dict[str, Any],
    component_index: dict[str, Any],
    doc_chunks_by_doc: dict[str, Any],
    chunk_text_lookup: dict[str, str],
    claim_generation_views: pd.DataFrame,
    reconstruction_views: pd.DataFrame,
    relations: pd.DataFrame,
    claim_rows_cache: list[dict[str, Any]] | None,
    claim_profiles_cache: list[dict[str, Any]] | None,
    claim_peer_rows_cache: list[list[dict[str, Any]]] | None,
    choose_reconstruction_cases: Callable[..., list[dict[str, Any]]],
    choose_claim_cases: Callable[..., list[dict[str, Any]]],
    choose_synthetic_value_redaction_cases: Callable[..., list[dict[str, Any]]],
    choose_sentence_redaction_cases: Callable[..., list[dict[str, Any]]],
    choose_contradiction_cases: Callable[..., list[dict[str, Any]]],
) -> TaskCollectorBundle:
    collectors: dict[str, CollectorFn] = {}
    reconstruction_case_cache: list[dict[str, Any]] | None = None

    if "reconstruction" in task_counts:
        def get_reconstruction_seed_cases() -> list[dict[str, Any]]:
            nonlocal reconstruction_case_cache
            if reconstruction_case_cache is None:
                reconstruction_case_cache = choose_reconstruction_cases(
                    reconstruction_views,
                    claim_generation_views,
                    relations,
                    chunk_text_lookup,
                    doc_chunks_by_doc,
                    collector_counts["reconstruction"],
                    sampling_with_doc_cap["reconstruction"],
                    seed + 1,
                    runner_cfg=runner_cfg,
                )
            return reconstruction_case_cache

        collectors["reconstruction"] = lambda: get_reconstruction_seed_cases()
    else:
        def get_reconstruction_seed_cases() -> list[dict[str, Any]]:
            return []

    collect_claim_cases_local = None
    if "claims" in task_counts:
        claim_rows_cache = claim_rows_cache or claim_generation_views.to_dict("records")

        def collect_claim_cases_local(active_runner_cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
            return choose_claim_cases(
                claim_generation_views,
                relations,
                chunk_text_lookup,
                collector_counts["claims"],
                sampling_with_doc_cap["claims"],
                seed + 2,
                runner_cfg=active_runner_cfg,
                claim_rows=claim_rows_cache,
                claim_profiles=claim_profiles_cache,
                peer_rows_by_idx=claim_peer_rows_cache,
                reconstruction_views=reconstruction_views,
                reconstruction_seed_cases=get_reconstruction_seed_cases(),
            )

        collectors["claims"] = lambda: collect_claim_cases_local(runner_cfg)

    if "value_redaction" in task_counts:
        collectors["value_redaction"] = lambda: choose_synthetic_value_redaction_cases(
            entities,
            doc_chunks_by_doc,
            para_richness,
            component_index,
            collector_counts["value_redaction"],
            sampling_with_doc_cap["value_redaction"],
            seed + 3,
            runner_cfg=runner_cfg,
        )

    if "sentence_redaction" in task_counts:
        collectors["sentence_redaction"] = lambda: choose_sentence_redaction_cases(
            chunks,
            component_index,
            para_richness,
            collector_counts["sentence_redaction"],
            sampling_with_doc_cap["sentence_redaction"],
            seed + 4,
            runner_cfg=runner_cfg,
        )

    collect_contradiction_cases_local = None
    if "contradictions" in task_counts:
        claim_rows_cache = claim_rows_cache or claim_generation_views.to_dict("records")

        def collect_contradiction_cases_local(active_runner_cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
            return choose_contradiction_cases(
                claim_generation_views,
                relations,
                chunk_text_lookup,
                collector_counts["contradictions"],
                sampling_with_doc_cap["contradictions"],
                seed + 5,
                runner_cfg=active_runner_cfg,
                claim_rows=claim_rows_cache,
                claim_profiles=claim_profiles_cache,
                peer_rows_by_idx=claim_peer_rows_cache,
                reconstruction_views=reconstruction_views,
                reconstruction_seed_cases=get_reconstruction_seed_cases(),
            )

        collectors["contradictions"] = lambda: collect_contradiction_cases_local(runner_cfg)

    return TaskCollectorBundle(
        collectors=collectors,
        collect_claim_cases_local=collect_claim_cases_local,
        collect_contradiction_cases_local=collect_contradiction_cases_local,
        get_reconstruction_seed_cases=get_reconstruction_seed_cases,
    )
