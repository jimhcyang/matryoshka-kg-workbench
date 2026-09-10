#!/usr/bin/env python3
"""Generic task catalog and selected-slice pipeline for benchmark builds."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from task_build_rules import resolve_task_build_behavior
from task_case_selection import build_catalog_cases_for_task, select_run_cases_for_task

BUILD_HEARTBEAT_SECONDS = 300.0


def _trim_strict_case_pool(
    cases: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or len(cases) <= limit:
        return list(cases)
    return sorted(
        cases,
        key=lambda case: (
            float(case.get("selector_score", 0.0) or 0.0),
            int(case.get("word_count", 0) or 0),
            str(case.get("case_id", "")),
        ),
        reverse=True,
    )[:limit]


def _run_with_heartbeat(
    label: str,
    fn: Callable[[], Any],
    *,
    interval_seconds: float = BUILD_HEARTBEAT_SECONDS,
) -> Any:
    start = time.monotonic()
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(interval_seconds):
            elapsed = time.monotonic() - start
            print(f"{label}: still running after {elapsed:.1f}s", flush=True)

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop_event.set()
        thread.join(timeout=0.1)


def process_task_catalogs(
    *,
    collectors: dict[str, Callable[[], list[dict[str, Any]]]],
    task_counts: dict[str, int],
    collector_counts: dict[str, int],
    sampling_with_doc_cap: dict[str, dict[str, Any]],
    seed: int,
    tier_order: dict[str, int],
    progress_prefix: str,
    enable_small_corpus_parser_vet: bool,
    small_corpus: bool,
    pair_like_claim_corpus: bool,
    max_cases_per_doc: int,
    corpus_code: str,
    config_corpus: str,
    fisc_claimlike_acceptable_sources: list[str],
    task_prefix_map: dict[str, str],
    runner_defaults: dict[str, Any],
    task_runner_overrides: dict[str, dict[str, Any]],
    collect_claim_cases_local: Callable[[dict[str, Any] | None], list[dict[str, Any]]] | None,
    collect_contradiction_cases_local: Callable[[dict[str, Any] | None], list[dict[str, Any]]] | None,
    catalog_runner_cfg: dict[str, Any],
    merge_ranked_candidates: Callable[..., list[dict[str, Any]]],
    rebalance_value_redaction_cases: Callable[..., list[dict[str, Any]]],
    unique_query_round_robin: Callable[..., list[dict[str, Any]]],
    normalize_space: Callable[[Any], str],
    build_expected: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    actual_counts: dict[str, int] = {}
    admissible_counts: dict[str, int] = {}
    task_stage_counts: dict[str, dict[str, Any]] = {}
    task_build_behaviors: dict[str, dict[str, Any]] = {}
    order_counter = 100

    for task_name, collect in collectors.items():
        task_started = time.monotonic()
        requested = task_counts.get(task_name, 0)
        build_behavior = resolve_task_build_behavior(
            task_name,
            enable_small_corpus_parser_vet=enable_small_corpus_parser_vet,
        )
        task_build_behaviors[task_name] = {
            "catalog_merge_mode": build_behavior.catalog_merge_mode,
            "selection_source": build_behavior.selection_source,
            "selection_strategy": build_behavior.selection_strategy,
            "allow_selected_backfill_from_catalog": build_behavior.allow_selected_backfill_from_catalog,
        }
        if progress_prefix:
            print(
                f"{progress_prefix}{task_name}: generating catalog target {collector_counts.get(task_name, requested)} for final top {requested if requested > 0 else 'ALL'}...",
                flush=True,
            )

        strict_task_cases = _run_with_heartbeat(
            f"{progress_prefix}{task_name}: collecting strict candidates",
            collect,
        )
        selection_pool_limit = int(collector_counts.get(task_name, requested))
        if (
            build_behavior.selection_source == "strict"
            and selection_pool_limit > 0
            and len(strict_task_cases) > selection_pool_limit
        ):
            strict_task_cases = _trim_strict_case_pool(
                strict_task_cases,
                selection_pool_limit,
            )
        catalog_cases = _run_with_heartbeat(
            f"{progress_prefix}{task_name}: building admissible catalog",
            lambda: build_catalog_cases_for_task(
                task_name=task_name,
                strict_task_cases=strict_task_cases,
                requested=requested,
                collector_counts=collector_counts,
                sampling_with_doc_cap=sampling_with_doc_cap,
                max_cases_per_doc=max_cases_per_doc,
                seed=seed,
                tier_order=tier_order,
                build_behavior=build_behavior,
                collect_claim_cases_local=collect_claim_cases_local,
                collect_contradiction_cases_local=collect_contradiction_cases_local,
                catalog_runner_cfg=catalog_runner_cfg,
                merge_ranked_candidates=merge_ranked_candidates,
            ),
        )

        task_cases = _run_with_heartbeat(
            f"{progress_prefix}{task_name}: selecting final cases",
            lambda: select_run_cases_for_task(
                task_name=task_name,
                requested=requested,
                strict_task_cases=strict_task_cases,
                catalog_cases=catalog_cases,
                sampling_with_doc_cap=sampling_with_doc_cap,
                max_cases_per_doc=max_cases_per_doc,
                seed=seed,
                tier_order=tier_order,
                build_behavior=build_behavior,
                small_corpus=small_corpus,
                pair_like_claim_corpus=pair_like_claim_corpus,
                corpus_code=corpus_code,
                unique_query_round_robin=unique_query_round_robin,
                rebalance_value_redaction_cases=rebalance_value_redaction_cases,
            ),
        )

        catalog_snapshot = [dict(case) for case in catalog_cases]
        selected_legacy_ids = [
            normalize_space(case.get("case_id", ""))
            for case in task_cases
            if normalize_space(case.get("case_id", ""))
        ]
        selected_rank_by_legacy = {
            legacy_id: index
            for index, legacy_id in enumerate(selected_legacy_ids, start=1)
        }

        admissible_counts[task_name] = len(catalog_cases)
        actual_counts[task_name] = len(task_cases)
        task_stage_counts[task_name] = {
            "requested_count": requested,
            "catalog_target_count": int(collector_counts.get(task_name, requested)),
            "admissible_count": len(catalog_cases),
            "admissible_rate_vs_catalog_target": (
                round(len(catalog_cases) / max(int(collector_counts.get(task_name, requested)), 1), 4)
            ),
            "selected_count": len(task_cases),
            "selection_rate_vs_catalog_target": (
                round(len(task_cases) / max(int(collector_counts.get(task_name, requested)), 1), 4)
            ),
            "selection_rate_vs_admissible": (
                round(len(task_cases) / max(len(catalog_cases), 1), 4)
            ),
        }
        if progress_prefix:
            print(
                f"{progress_prefix}{task_name}: admissible {len(catalog_cases)} from target {collector_counts.get(task_name, requested)}; selected {len(task_cases)} in {time.monotonic() - task_started:.1f}s",
                flush=True,
            )

        selected_run_case_ids: dict[str, str] = {}
        for index, case in enumerate(task_cases, start=1):
            legacy_case_id = normalize_space(case.get("case_id", ""))
            case["legacy_case_id"] = case["case_id"]
            case["case_id"] = f"{task_prefix_map[task_name]}{index - 1:02d}_{corpus_code}"
            if legacy_case_id:
                selected_run_case_ids[legacy_case_id] = case["case_id"]
            task_runner_cfg = dict(runner_defaults)
            task_runner_cfg.update(task_runner_overrides.get(task_name, {}))
            case["corpus"] = corpus_code
            case["tier"] = 2
            case["order"] = order_counter
            case["enabled"] = True
            case["mode"] = task_runner_cfg["mode"]
            case["target_doc_ids"] = list(case.get("target_doc_ids", []) or [case["doc_id"]])
            case["required_doc_ids"] = list(case.get("required_doc_ids", []) or [case["doc_id"]])
            case["staged_top_docs"] = task_runner_cfg["staged_top_docs"]
            case["retrieval_mode"] = task_runner_cfg["retrieval_mode"]
            case["raw_use_mode"] = task_runner_cfg["raw_use_mode"]
            case["max_elapsed_seconds"] = task_runner_cfg["max_elapsed_seconds"]
            if (
                fisc_claimlike_acceptable_sources
                and case["task_family"] in {"claims", "reconstruction", "contradictions"}
            ):
                case["acceptable_source_ids_any"] = list(fisc_claimlike_acceptable_sources)
            case["expected"] = build_expected(case)
            order_counter += 1

        for rank, catalog_case in enumerate(catalog_snapshot, start=1):
            legacy_case_id = normalize_space(catalog_case.get("case_id", ""))
            catalog_rows.append(
                {
                    "corpus": corpus_code,
                    "corpus_full": config_corpus,
                    "task_family": task_name,
                    "legacy_case_id": legacy_case_id,
                    "run_case_id": selected_run_case_ids.get(legacy_case_id, ""),
                    "selected_for_run": bool(legacy_case_id in selected_run_case_ids),
                    "selection_rank": selected_rank_by_legacy.get(legacy_case_id, ""),
                    "catalog_rank": rank,
                    "doc_id": str(catalog_case.get("doc_id", "")),
                    "paragraph_id": str(catalog_case.get("paragraph_id", "")),
                    "query_kind": str(catalog_case.get("query_kind", "")),
                    "selector_score": float(catalog_case.get("selector_score", 0.0) or 0.0),
                    "word_count": int(catalog_case.get("word_count", 0) or 0),
                    "query_target_words": int(catalog_case.get("query_target_words", 0) or 0),
                    "query_text": str(catalog_case.get("query", "")),
                    "ground_truth": str(catalog_case.get("ground_truth", "")),
                    "source_context": str(catalog_case.get("source_context", "")),
                    "notes": str(catalog_case.get("notes", "")),
                }
            )

        cases.extend(task_cases)

    return {
        "cases": cases,
        "catalog_rows": catalog_rows,
        "actual_counts": actual_counts,
        "admissible_counts": admissible_counts,
        "task_stage_counts": task_stage_counts,
        "task_build_behaviors": task_build_behaviors,
    }
