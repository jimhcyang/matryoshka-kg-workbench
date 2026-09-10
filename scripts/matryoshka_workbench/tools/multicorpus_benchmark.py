#!/usr/bin/env python3
"""Build and run a combined multi-corpus Matryoshka benchmark."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus_benchmark import (
    DEFAULT_CONFIG,
    KG_DEMO_ROOT,
    RUNNER_RELATIVE,
    atomic_write_json,
    authoritative_decision_source,
    build_cases,
    build_runner_groups,
    create_case_aliases,
    detect_run_layout,
    derive_effective_batch_status,
    find_digest_artifact,
    first_eval_in_dir,
    first_nonempty,
    format_expected_targets,
    load_case_manifest_map,
    materialize_case_sidecars,
    load_config,
    load_json,
    normalize_space,
    read_summary_row,
    read_text_if_exists,
    resolve_path,
    resolve_path_preserve_symlink,
    run_streaming_group,
    stringify_expected_targets,
    write_plan_subset,
    canonicalize_corpus_name,
)
from query_catalog import write_query_catalog_files
from summarize_benchmark_results import summarize_experiment
from task_registry import CORPUS_ALIASES, DEFAULT_CORPORA, TASK_PREFIX_MAP, short_corpus_code


DEFAULT_EXTRACTION_PROFILE = "vietnam"


@dataclass
class MultiPaths:
    output_root: Path
    experiment_root: Path
    config_root: Path
    query_manifest_root: Path
    query_file_root: Path
    run_root: Path
    summary_root: Path


def ensure_dirs(paths: MultiPaths) -> None:
    for directory in (
        paths.output_root,
        paths.experiment_root,
        paths.config_root,
        paths.query_manifest_root,
        paths.query_file_root,
        paths.run_root,
        paths.summary_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

def experiment_paths(output_root: Path, experiment_name: str) -> MultiPaths:
    experiment_root = output_root / experiment_name
    return MultiPaths(
        output_root=output_root,
        experiment_root=experiment_root,
        config_root=experiment_root / "configs",
        query_manifest_root=experiment_root / "query_manifests",
        query_file_root=experiment_root / "query_files",
        run_root=experiment_root / "runs",
        summary_root=experiment_root / "summaries",
    )


def load_base_config(config_path: Path) -> dict[str, Any]:
    base = load_json(config_path)
    base["_config_path"] = str(config_path.resolve())
    base["python_executable"] = str(
        resolve_path_preserve_symlink(base.get("python_executable", sys.executable))
    )
    base["output_root"] = str((KG_DEMO_ROOT / str(base["output_root"])).resolve())
    return base


def load_multicorpus_configs(
    config_path: Path,
    corpora: list[str],
    *,
    base_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    per_corpus_configs: dict[str, dict[str, Any]] = {}
    for corpus in corpora:
        config = load_config(config_path, corpus)
        if base_config:
            if "runner" in base_config:
                config["runner"] = {
                    **dict(config.get("runner", {})),
                    **dict(base_config.get("runner", {})),
                }
            for passthrough_key in ("python_executable", "candidate_pool_multiplier"):
                if passthrough_key in base_config:
                    config[passthrough_key] = base_config[passthrough_key]
        per_corpus_configs[corpus] = config
    return per_corpus_configs


def build_multicorpus_cases(
    config_path: Path,
    corpora: list[str],
    *,
    base_config: dict[str, Any] | None = None,
    per_task_override: int | None,
    tasks_override: list[str] | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    combined_cases: list[dict[str, Any]] = []
    combined_catalog_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"corpora": {}, "total_cases": 0}
    per_corpus_configs = load_multicorpus_configs(
        config_path,
        corpora,
        base_config=base_config,
    )
    global_order = 100

    total_corpora = len(corpora)
    for corpus_index, corpus in enumerate(corpora, start=1):
        print(
            f"[build {corpus_index}/{total_corpora}] {short_corpus_code(corpus)}: loading corpus config",
            flush=True,
        )
        config = per_corpus_configs[corpus]
        cases, case_diag, case_catalog_rows = build_cases(
            config,
            per_task_override=per_task_override,
            tasks_override=tasks_override,
            progress_prefix=f"[build {corpus_index}/{total_corpora} {short_corpus_code(corpus)}] ",
        )
        diagnostics["corpora"][corpus] = case_diag
        combined_catalog_rows.extend(case_catalog_rows)
        print(
            f"[build {corpus_index}/{total_corpora}] {short_corpus_code(corpus)}: total kept {len(cases)} cases",
            flush=True,
        )
        for case in cases:
            case["extraction_profile"] = corpus
            case["corpus_full"] = corpus
            case["corpus"] = short_corpus_code(corpus)
            case["order"] = global_order
            global_order += 1
            combined_cases.append(case)

    diagnostics["total_cases"] = len(combined_cases)
    combined_cases.sort(key=lambda item: item["order"])
    return combined_cases, diagnostics, per_corpus_configs, combined_catalog_rows


def write_combined_cases(
    base_config: dict[str, Any],
    paths: MultiPaths,
    cases: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    catalog_rows: list[dict[str, Any]],
    corpora: list[str],
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    if overwrite:
        if paths.experiment_root.exists():
            shutil.rmtree(paths.experiment_root)
        ensure_dirs(paths)
        for directory in (paths.query_manifest_root, paths.query_file_root, paths.summary_root):
            directory.mkdir(parents=True, exist_ok=True)
    else:
        ensure_dirs(paths)

    plan_cases: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    for case in cases:
        task_dir = paths.query_file_root / case["task_family"] / case["corpus"]
        task_dir.mkdir(parents=True, exist_ok=True)
        query_path = task_dir / f"{case['case_id']}.txt"
        raw_query = str(case["query"]).strip()
        expected_targets = format_expected_targets(case)
        answer_targets = case.get("answer_targets")
        if not isinstance(answer_targets, dict) or not answer_targets:
            answer_targets = expected_targets
        query_path.write_text(raw_query + "\n", encoding="utf-8")
        plan_cases.append(
            {
                "id": case["case_id"],
                "tier": case["tier"],
                "order": case["order"],
                "mode": case["mode"],
                "task": case["runner_task"],
                "query": raw_query,
                "target_doc_ids": case["target_doc_ids"],
                "required_doc_ids": case["required_doc_ids"],
                "staged_top_docs": case["staged_top_docs"],
                "retrieval_mode": case["retrieval_mode"],
                "raw_use_mode": case["raw_use_mode"],
                "max_elapsed_seconds": case["max_elapsed_seconds"],
                "expected": case["expected"],
                "enabled": True,
                "extraction_profile": case["extraction_profile"],
            }
        )
        manifest_rows.append(
            {
                "case_id": case["case_id"],
                "legacy_case_id": case.get("legacy_case_id", ""),
                "corpus": case["corpus"],
                "corpus_full": case["corpus_full"],
                "task_family": case["task_family"],
                "runner_task": case["runner_task"],
                "doc_id": case["doc_id"],
                "paragraph_id": case["paragraph_id"],
                "query_kind": case["query_kind"],
                "ground_truth": case["ground_truth"],
                "notes": case.get("notes", ""),
                "query_file": str(query_path.relative_to(paths.experiment_root)),
                "target_doc_ids": ",".join(case["target_doc_ids"]),
                "source_context": case.get("source_context", ""),
            }
        )
        answer_rows.append(
            {
                "case_id": case["case_id"],
                "corpus": case["corpus"],
                "corpus_full": case["corpus_full"],
                "task_family": case["task_family"],
                "runner_task": case["runner_task"],
                "doc_id": case["doc_id"],
                "paragraph_id": case["paragraph_id"],
                "query_kind": case["query_kind"],
                "query_word_count": len(raw_query.split()),
                "query_target_words": case.get("query_target_words", ""),
                "source_context_word_count": len(str(case.get("source_context", "")).split()),
                "ground_truth": case["ground_truth"],
                "expected_targets": stringify_expected_targets(answer_targets),
                "source_context": case.get("source_context", ""),
                "query_text": raw_query,
                "notes": case.get("notes", ""),
                "query_file": str(query_path.relative_to(paths.experiment_root)),
            }
        )

    plan_payload = {
        "schema_version": "1.0.18",
        "purpose": "Combined multi-corpus benchmark generated from processed extraction bundles.",
        "notes": [
            f"Config path: {base_config['_config_path']}",
            f"Corpora: {', '.join(corpora)}",
            f"Experiment: {paths.experiment_root.name}",
            "Cases are extraction-driven and use named extraction profiles for corpus routing.",
        ],
        "cases": plan_cases,
    }
    plan_path = paths.query_manifest_root / "benchmark_plan.json"
    atomic_write_json(plan_path, plan_payload)

    manifest_path = paths.query_manifest_root / "case_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [
            "case_id", "legacy_case_id", "corpus", "corpus_full", "task_family",
            "runner_task", "doc_id", "paragraph_id", "query_kind", "ground_truth",
            "notes", "query_file", "target_doc_ids", "source_context",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    answer_key_path = paths.query_manifest_root / "query_answer_key.csv"
    with answer_key_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(answer_rows[0].keys()) if answer_rows else [
            "case_id", "corpus", "corpus_full", "task_family", "runner_task",
            "doc_id", "paragraph_id", "query_kind", "query_word_count",
            "query_target_words", "source_context_word_count",
            "ground_truth", "expected_targets", "source_context", "query_text",
            "notes", "query_file",
        ])
        writer.writeheader()
        writer.writerows(answer_rows)

    stage_count_rows: list[dict[str, Any]] = []
    for corpus in corpora:
        corpus_diag = diagnostics.get("corpora", {}).get(corpus, {})
        context = corpus_diag.get("corpus_context", {})
        stage_counts = corpus_diag.get("task_stage_counts", {})
        for task_name, task_stage in stage_counts.items():
            stage_count_rows.append(
                {
                    "corpus": short_corpus_code(corpus),
                    "corpus_full": corpus,
                    "task_family": task_name,
                    "requested_count": task_stage.get("requested_count", 0),
                    "catalog_target_count": task_stage.get("catalog_target_count", 0),
                    "admissible_count": task_stage.get("admissible_count", 0),
                    "admissible_rate_vs_catalog_target": task_stage.get("admissible_rate_vs_catalog_target", 0),
                    "selected_count": task_stage.get("selected_count", 0),
                    "selection_rate_vs_catalog_target": task_stage.get("selection_rate_vs_catalog_target", 0),
                    "selection_rate_vs_admissible": task_stage.get("selection_rate_vs_admissible", 0),
                    "corpus_doc_count": context.get("corpus_doc_count", 0),
                    "claim_row_count": context.get("claim_row_count", 0),
                    "pair_like_claim_corpus": context.get("pair_like_claim_corpus", False),
                    "max_cases_per_doc": context.get("max_cases_per_doc", 0),
                    "size_bucket": corpus_diag.get("task_generation_policies", {}).get(task_name, {}).get("size_bucket", ""),
                }
            )
    stage_count_path = paths.summary_root / "build_stage_counts.csv"
    with stage_count_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "corpus",
                "corpus_full",
                "task_family",
                "requested_count",
                "catalog_target_count",
                "admissible_count",
                "admissible_rate_vs_catalog_target",
                "selected_count",
                "selection_rate_vs_catalog_target",
                "selection_rate_vs_admissible",
                "corpus_doc_count",
                "claim_row_count",
                "pair_like_claim_corpus",
                "max_cases_per_doc",
                "size_bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(stage_count_rows)

    write_query_catalog_files(paths.summary_root, catalog_rows)
    atomic_write_json(paths.summary_root / "build_diagnostics.json", diagnostics)
    return plan_path, manifest_path


def snapshot_config(
    base_config: dict[str, Any],
    per_corpus_configs: dict[str, dict[str, Any]],
    paths: MultiPaths,
) -> Path:
    ensure_dirs(paths)
    snapshot_path = paths.config_root / "resolved_multicorpus_config.json"
    payload = {
        "base_config": base_config,
        "corpora": {
            corpus: {
                "profile_path": config.get("_profile_path", ""),
                "manifest_path": config.get("_manifest_path", ""),
                "processed_source_tables": config["source_tables"],
                "matryoshka_root": config["matryoshka_root"],
                "decision_profile": config["decision_profile"],
                "corpus_entity_resource": config["corpus_entity_resource"],
            }
            for corpus, config in per_corpus_configs.items()
        },
    }
    atomic_write_json(snapshot_path, payload)
    return snapshot_path


def load_snapshot_config(paths: MultiPaths) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    snapshot_path = paths.config_root / "resolved_multicorpus_config.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Missing config snapshot: {snapshot_path}. Run the build step first."
        )

    payload = load_json(snapshot_path)
    base_config = dict(payload.get("base_config", {}))
    raw_corpora = payload.get("corpora", {})
    if not isinstance(raw_corpora, dict) or not raw_corpora:
        raise ValueError(f"Invalid or empty corpora config snapshot: {snapshot_path}")

    per_corpus_configs: dict[str, dict[str, Any]] = {}
    for corpus, config in raw_corpora.items():
        if not isinstance(config, dict):
            continue
        normalized = dict(config)
        if "source_tables" not in normalized and "processed_source_tables" in normalized:
            normalized["source_tables"] = dict(normalized.get("processed_source_tables", {}))
        per_corpus_configs[str(corpus)] = normalized

    if not per_corpus_configs:
        raise ValueError(f"No usable corpus configs found in snapshot: {snapshot_path}")

    return base_config, per_corpus_configs


def run_batch(
    base_config: dict[str, Any],
    per_corpus_configs: dict[str, dict[str, Any]],
    paths: MultiPaths,
    plan_path: Path,
    *,
    overwrite: bool,
) -> Path:
    if DEFAULT_EXTRACTION_PROFILE in per_corpus_configs:
        first_corpus = DEFAULT_EXTRACTION_PROFILE
    else:
        first_corpus = next(iter(sorted(per_corpus_configs)))
    first_config = per_corpus_configs[first_corpus]
    runner = resolve_path(first_config["matryoshka_root"]) / RUNNER_RELATIVE
    python_executable = resolve_path_preserve_symlink(
        base_config.get("python_executable", sys.executable)
    )
    if overwrite and paths.run_root.exists():
        shutil.rmtree(paths.run_root)
    paths.run_root.mkdir(parents=True, exist_ok=True)

    runner_cfg = dict(base_config["runner"])
    plan_payload = load_json(plan_path)
    groups = build_runner_groups(plan_payload, runner_cfg)
    total_cases = sum(len(group_cases) for _, group_cases, _ in groups)
    print(f"Benchmark cases scheduled: {total_cases}")
    manifest_map = load_case_manifest_map(paths.query_manifest_root / "case_manifest.csv")
    merged_rows: list[dict[str, Any]] = []
    group_had_failures = False

    shared_prefix = [
        str(python_executable),
        "-u",
        str(runner),
        "--code-dir",
        str(resolve_path(first_config["matryoshka_root"])),
        "--extraction-dir",
        str(resolve_path(first_config["source_tables"]["chunks"]).parent),
        "--output-root",
        str(paths.run_root),
        "--python-executable",
        str(python_executable),
        "--decision-profile",
        str(resolve_path(first_config["decision_profile"])),
    ]
    for corpus, config in per_corpus_configs.items():
        shared_prefix.extend(
            [
                "--extraction-profile",
                f"{corpus}={resolve_path(config['source_tables']['chunks']).parent}",
            ]
        )
        corpus_entity_resource = normalize_space(
            config.get("corpus_entity_resource", "")
        )
        if corpus_entity_resource:
            shared_prefix.extend(
                [
                    "--corpus-entity-resource-profile",
                    f"{corpus}={resolve_path(corpus_entity_resource)}",
                ]
            )

    completed_offset = 0
    for group_name, group_cases, use_openai in groups:
        group_plan_path = paths.query_manifest_root / f"benchmark_plan__{group_name}.json"
        write_plan_subset(plan_payload, group_cases, group_plan_path)
        cmd = list(shared_prefix)
        cmd.extend(["--plan", str(group_plan_path)])
        if use_openai:
            cmd.append("--use-openai-parser")
        if runner_cfg.get("use_llm_digest"):
            cmd.append("--use-llm-digest")
        if runner_cfg.get("llm_model"):
            cmd.extend(["--llm-model", str(runner_cfg["llm_model"])])
        if runner_cfg.get("openai_api_key_env"):
            cmd.extend(["--openai-api-key-env", str(runner_cfg["openai_api_key_env"])])
        if runner_cfg.get("minimum_score") is not None:
            cmd.extend(["--minimum-score", str(runner_cfg["minimum_score"])])
        if runner_cfg.get("minimum_margin") is not None:
            cmd.extend(["--minimum-margin", str(runner_cfg["minimum_margin"])])
        cmd.extend(
            [
                "--contradiction-propositions-per-class-per-document",
                str(runner_cfg["contradiction_propositions_per_class_per_document"]),
                "--contradiction-raw-sentences-per-class-per-document",
                str(runner_cfg["contradiction_raw_sentences_per_class_per_document"]),
            ]
        )
        if runner_cfg.get("stop_on_failure"):
            cmd.append("--stop-on-failure")
        return_code = run_streaming_group(
            cmd,
            offset=completed_offset,
            global_total=total_cases,
            group_cases=group_cases,
            run_root=paths.run_root,
            manifest_map=manifest_map,
        )
        if return_code not in {0, 2}:
            raise subprocess.CalledProcessError(return_code, cmd)
        if return_code == 2:
            group_had_failures = True
        completed_offset += len(group_cases)
        group_batch_summary = paths.run_root / "batch_summary.csv"
        if group_batch_summary.exists():
            copy_path = paths.run_root / f"batch_summary__{group_name}.csv"
            shutil.copy2(group_batch_summary, copy_path)
            with copy_path.open(newline="", encoding="utf-8") as handle:
                merged_rows.extend(list(csv.DictReader(handle)))
        materialize_case_sidecars(paths.run_root, group_cases, manifest_map)

    if merged_rows:
        fieldnames: list[str] = []
        for row in merged_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with (paths.run_root / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)
    if group_had_failures:
        print("Batch runner completed with case-level benchmark failures. Artifacts were still written.")
    create_case_aliases(paths.run_root, plan_payload.get("cases", []))
    materialize_case_sidecars(
        paths.run_root,
        list(plan_payload.get("cases", [])),
        manifest_map,
    )
    return paths.run_root


def summarize_multicorpus_runs(paths: MultiPaths) -> Path:
    manifest_path = paths.query_manifest_root / "case_manifest.csv"
    batch_summary_path = paths.run_root / "batch_summary.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing case manifest: {manifest_path}")
    if not batch_summary_path.exists():
        raise FileNotFoundError(f"Missing batch summary: {batch_summary_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with batch_summary_path.open(newline="", encoding="utf-8") as handle:
        batch_rows = list(csv.DictReader(handle))
    batch_by_id = {str(row.get("id", "")): row for row in batch_rows}

    output_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        case_id = str(row.get("case_id", ""))
        batch_info = batch_by_id.get(case_id, {})
        run_root = Path(str(batch_info.get("run_root", ""))) if batch_info.get("run_root") else None
        run_layout, primary_root, holdout_root = detect_run_layout(run_root)
        primary_eval = first_eval_in_dir(primary_root) if primary_root else None
        holdout_eval = first_eval_in_dir(holdout_root) if holdout_root else None
        primary_row = read_summary_row(primary_eval)
        holdout_row = read_summary_row(holdout_eval)
        query_path = paths.experiment_root / str(row.get("query_file", ""))
        query_text = read_text_if_exists(query_path)
        primary_digest = find_digest_artifact(primary_root)
        holdout_digest = find_digest_artifact(holdout_root)
        effective_batch_status = derive_effective_batch_status(
            str(row.get("task_family", "")),
            batch_info.get("status", ""),
            primary_row,
            holdout_row,
            run_layout,
        )
        # The batch runner uses a generic FAIL exit code for a blocked preflight.
        # Preserve the coordinated pipeline's more specific terminal state in reports.
        if run_root:
            status_path = run_root / "coordinated_status_v048.json"
            if status_path.exists():
                try:
                    coordinated_status = load_json(status_path)
                except (OSError, ValueError, TypeError):
                    coordinated_status = {}
                terminal_status = normalize_space(coordinated_status.get("status", "")).upper()
                if terminal_status == "PREFLIGHT_BLOCKED":
                    effective_batch_status = terminal_status
        output_rows.append(
            {
                "corpus": row.get("corpus", ""),
                "corpus_full": row.get("corpus_full", ""),
                "task_family": row.get("task_family", ""),
                "case_id": case_id,
                "doc_id": row.get("doc_id", ""),
                "paragraph_id": row.get("paragraph_id", ""),
                "query_kind": row.get("query_kind", ""),
                "batch_status": effective_batch_status,
                "run_layout": run_layout,
                "semantic_status": first_nonempty(
                    primary_row.get("semantic_status", ""),
                    primary_row.get("task_execution_status", ""),
                ),
                "baseline_semantic_status": first_nonempty(
                    primary_row.get("semantic_status", ""),
                    primary_row.get("task_execution_status", ""),
                ),
                "holdout_semantic_status": first_nonempty(
                    holdout_row.get("semantic_status", ""),
                    holdout_row.get("task_execution_status", ""),
                ),
                "decision_band": primary_row.get("operational_decision_band", ""),
                "baseline_selected_source_doc_id": primary_row.get("selected_source_doc_id", ""),
                "holdout_selected_source_doc_id": holdout_row.get("selected_source_doc_id", ""),
                "baseline_decision_source_doc_id": authoritative_decision_source(
                    row.get("task_family", ""), primary_row
                ),
                "holdout_decision_source_doc_id": authoritative_decision_source(
                    row.get("task_family", ""), holdout_row
                ),
                "baseline_authoritative_source_doc_id": primary_row.get("authoritative_source_doc_id", ""),
                "holdout_authoritative_source_doc_id": holdout_row.get("authoritative_source_doc_id", ""),
                "baseline_primary_finding_source_doc_id": primary_row.get("primary_finding_source_doc_id", ""),
                "holdout_primary_finding_source_doc_id": holdout_row.get("primary_finding_source_doc_id", ""),
                "baseline_all_decision_source_doc_ids": primary_row.get("all_decision_source_doc_ids", ""),
                "holdout_all_decision_source_doc_ids": holdout_row.get("all_decision_source_doc_ids", ""),
                "baseline_top_candidate_doc_id": primary_row.get("top_candidate_doc_id", ""),
                "holdout_top_candidate_doc_id": holdout_row.get("top_candidate_doc_id", ""),
                "outcome_reason": first_nonempty(
                    primary_row.get("decision_reason", ""),
                    primary_row.get("unresolved_reason", ""),
                    primary_row.get("failed_gates", ""),
                    batch_info.get("issues", ""),
                ),
                "ground_truth": row.get("ground_truth", ""),
                "query_text": query_text,
                "query_file_abs": str(query_path.resolve()) if query_path.exists() else "",
                "run_root_abs": str(run_root.resolve()) if run_root else "",
                "streamlit_open_path_abs": (
                    str(run_root.resolve()) if run_root and run_layout == "pair"
                    else str(primary_root.resolve()) if primary_root else ""
                ),
                "streamlit_holdout_path_abs": str(holdout_root.resolve()) if holdout_root else "",
                "primary_digest_artifact_abs": str(primary_digest.resolve()) if primary_digest else "",
                "holdout_digest_artifact_abs": str(holdout_digest.resolve()) if holdout_digest else "",
            }
        )

    index_path = paths.summary_root / "streamlit_run_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()) if output_rows else [
            "corpus", "corpus_full", "task_family", "case_id", "doc_id", "paragraph_id",
            "query_kind", "batch_status", "run_layout", "semantic_status",
            "baseline_semantic_status", "holdout_semantic_status", "decision_band",
            "baseline_selected_source_doc_id", "holdout_selected_source_doc_id",
            "baseline_decision_source_doc_id", "holdout_decision_source_doc_id",
            "baseline_authoritative_source_doc_id", "holdout_authoritative_source_doc_id",
            "baseline_primary_finding_source_doc_id", "holdout_primary_finding_source_doc_id",
            "baseline_all_decision_source_doc_ids", "holdout_all_decision_source_doc_ids",
            "baseline_top_candidate_doc_id", "holdout_top_candidate_doc_id",
            "outcome_reason", "ground_truth", "query_text", "query_file_abs", "run_root_abs",
            "streamlit_open_path_abs", "streamlit_holdout_path_abs", "primary_digest_artifact_abs",
            "holdout_digest_artifact_abs",
        ])
        writer.writeheader()
        writer.writerows(output_rows)

    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "run", "summarize", "all"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-name", default="multicorpus_pilot")
    parser.add_argument(
        "--corpora",
        default="vietnam,iran,fisc,cib,canberra",
        help=f"Comma-separated corpus list. Supported names: {', '.join(sorted(CORPUS_ALIASES))}",
    )
    parser.add_argument("--per-task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-openai-parser", action="store_true")
    parser.add_argument("--use-llm-digest", action="store_true")
    parser.add_argument(
        "--model-preset",
        choices=["4o", "5.6"],
        default="",
        help="Convenience preset for runtime OpenAI models.",
    )
    parser.add_argument(
        "--parser-model",
        default="",
        help="Override the runtime query parser model.",
    )
    parser.add_argument(
        "--llm-model",
        default="",
        help="Override the runtime digest/adjudication model.",
    )
    parser.add_argument(
        "--runner-mode",
        choices=["single", "pair"],
        default="",
        help="Override runner mode. `pair` executes baseline plus holdout for every case.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    corpora = [
        canonicalize_corpus_name(item)
        for item in args.corpora.split(",")
        if normalize_space(item)
    ]
    base_config = load_base_config(config_path)
    if args.seed is not None:
        base_config["seed"] = int(args.seed)
    if args.use_openai_parser:
        base_config["runner"]["use_openai_parser"] = True
    if args.use_llm_digest:
        base_config["runner"]["use_llm_digest"] = True
    if args.model_preset:
        if args.model_preset == "4o":
            base_config["runner"]["parser_model"] = "gpt-4o"
            base_config["runner"]["llm_model"] = "gpt-4o"
        elif args.model_preset == "5.6":
            base_config["runner"]["parser_model"] = "gpt-5.6-sol"
            base_config["runner"]["llm_model"] = "gpt-5.6-sol"
    if args.parser_model:
        base_config["runner"]["parser_model"] = args.parser_model
    if args.llm_model:
        base_config["runner"]["llm_model"] = args.llm_model
    if args.runner_mode:
        base_config["runner"]["mode"] = args.runner_mode
    output_root = resolve_path(base_config["output_root"])
    paths = experiment_paths(output_root, args.experiment_name)
    tasks_override = [value.strip() for value in args.tasks.split(",") if value.strip()] or None

    cases: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    per_corpus_configs: dict[str, dict[str, Any]] = {}

    if args.command in {"build", "all"}:
        cases, diagnostics, per_corpus_configs, catalog_rows = build_multicorpus_cases(
            config_path,
            corpora,
            base_config=base_config,
            per_task_override=args.per_task,
            tasks_override=tasks_override,
        )
        ensure_dirs(paths)
        plan_path, manifest_path = write_combined_cases(
            base_config,
            paths,
            cases,
            diagnostics,
            catalog_rows,
            corpora,
            overwrite=args.overwrite,
        )
        snapshot_config(base_config, per_corpus_configs, paths)
        print(f"Plan written to: {plan_path}")
        print(f"Case manifest written to: {manifest_path}")
        if args.command == "build":
            return

    if args.command in {"run", "all"}:
        plan_path = paths.query_manifest_root / "benchmark_plan.json"
        if not plan_path.exists():
            raise FileNotFoundError(
                f"Missing benchmark plan: {plan_path}. Run the build step first."
            )
        if not per_corpus_configs:
            try:
                snapshot_base_config, per_corpus_configs = load_snapshot_config(paths)
                if snapshot_base_config:
                    for key in ("python_executable", "output_root"):
                        if key in snapshot_base_config and key not in base_config:
                            base_config[key] = snapshot_base_config[key]
            except FileNotFoundError:
                per_corpus_configs = load_multicorpus_configs(
                    config_path,
                    corpora,
                    base_config=base_config,
                )
                ensure_dirs(paths)
                snapshot_config(base_config, per_corpus_configs, paths)
        run_batch(base_config, per_corpus_configs, paths, plan_path, overwrite=args.overwrite)
        index_path = summarize_multicorpus_runs(paths)
        print(f"Streamlit run index written to: {index_path}")
        if args.command == "all":
            visual_root = summarize_experiment(paths.experiment_root)
            print(f"Visual summary written to: {visual_root}")
        if args.command == "run":
            return

    if args.command == "summarize":
        index_path = summarize_multicorpus_runs(paths)
        print(f"Streamlit run index written to: {index_path}")


if __name__ == "__main__":
    main()
