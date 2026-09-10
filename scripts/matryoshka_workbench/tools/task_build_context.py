#!/usr/bin/env python3
"""Shared build-context bootstrap for benchmark generation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from task_policies import (
    SMALL_CORPUS_DOC_THRESHOLD,
    TaskPolicyContext,
    resolve_task_generation_policies,
)


def prepare_build_context(
    config: dict[str, Any],
    *,
    per_task_override: int | None,
    tasks_override: list[str] | None,
    maybe_bool: Callable[[Any], bool],
    normalize_space: Callable[[Any], str],
    read_tables: Callable[[dict[str, Any]], dict[str, pd.DataFrame]],
    build_doc_sentences: Callable[[pd.DataFrame], dict[str, list[dict[str, Any]]]],
    build_chunk_text_lookup: Callable[[pd.DataFrame], dict[tuple[str, str], str]],
    build_doc_para_richness: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict[tuple[str, str], dict[str, int]]],
    build_paragraph_component_index: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict[tuple[str, str], dict[str, list[str]]]],
) -> dict[str, Any]:
    seed = int(config["seed"])
    sampling = dict(config["sampling"])
    runner_cfg = dict(config.get("runner", {}))
    runner_cfg["__matryoshka_root"] = config.get("matryoshka_root", "")
    runner_cfg["__python_executable"] = config.get("python_executable", sys.executable)
    runner_cfg["__source_extraction_root"] = str(
        Path(config["source_tables"]["chunks"]).resolve().parent
    )
    runner_cfg["__corpus_entity_resource"] = config.get("corpus_entity_resource", "")
    runner_cfg["__decision_profile"] = config.get("decision_profile", "")
    max_cases_per_doc = int(sampling.get("max_cases_per_doc", 2))
    candidate_pool_multiplier = int(config.get("candidate_pool_multiplier", 5))
    base_task_counts = {
        key: int(value)
        for key, value in dict(config["task_counts"]).items()
    }
    task_counts = {
        key: int(per_task_override if per_task_override is not None else value)
        for key, value in base_task_counts.items()
    }
    policy_task_counts = {
        key: int(base_task_counts.get(key, 0) if task_counts.get(key, 0) <= 0 else task_counts.get(key, 0))
        for key in base_task_counts
    }
    if tasks_override:
        allowed = set(tasks_override)
        task_counts = {key: value for key, value in task_counts.items() if key in allowed}
        policy_task_counts = {key: value for key, value in policy_task_counts.items() if key in allowed}
    needs_redaction_support = bool(
        {"value_redaction", "sentence_redaction"} & set(task_counts)
    )
    tables = read_tables(config)
    chunks = tables["chunks"]
    claims = tables["claims"]
    entities = tables["entities"]
    redaction_views = tables["kg_pseudo_chunks_redaction"]
    claim_views = tables["kg_pseudo_chunks_claims"]
    reconstruction_views = tables["kg_pseudo_chunks"]
    doc_chunks_by_doc = build_doc_sentences(chunks)
    chunk_text_lookup = build_chunk_text_lookup(chunks)
    corpus_doc_count = max(1, int(chunks["doc_id"].nunique()) if "doc_id" in chunks.columns else 1)
    if corpus_doc_count <= 8 and max_cases_per_doc < 3:
        max_cases_per_doc = 3
    if needs_redaction_support:
        para_richness = build_doc_para_richness(
            claims,
            tables["events"],
            tables["relations"],
        )
        component_index = build_paragraph_component_index(
            entities,
            tables["events"],
            claims,
        )
    else:
        para_richness = {}
        component_index = {}
    use_raw_claim_table_for_generation = maybe_bool(
        config.get("use_raw_claim_table_for_generation", False)
    )
    claim_generation_views = claims if use_raw_claim_table_for_generation else claim_views
    claim_doc_ids = {
        normalize_space(doc_id)
        for doc_id in claims.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    }
    pair_like_claim_corpus = bool(claim_doc_ids) and all(
        doc_id.startswith("Pair_") for doc_id in claim_doc_ids
    )
    fisc_claimlike_acceptable_sources = (
        sorted(claim_doc_ids)
        if str(config.get("corpus", "")).casefold() == "fisc" and pair_like_claim_corpus
        else []
    )
    runner_cfg["__corpus_doc_count"] = corpus_doc_count
    runner_cfg["__claim_row_count"] = len(claims)
    runner_cfg["__corpus_name"] = config["corpus"]
    runner_cfg["__pair_like_claim_corpus"] = pair_like_claim_corpus
    runner_cfg["__fisc_claimlike_acceptable_sources"] = list(
        fisc_claimlike_acceptable_sources
    )
    enable_small_corpus_parser_vet = (
        corpus_doc_count <= SMALL_CORPUS_DOC_THRESHOLD
    )
    runner_cfg["__strict_small_corpus_vet"] = enable_small_corpus_parser_vet
    policy_context = TaskPolicyContext(
        corpus=str(config.get("corpus", "")),
        corpus_doc_count=corpus_doc_count,
        claim_row_count=len(claims),
        pair_like_claim_corpus=pair_like_claim_corpus,
        candidate_pool_multiplier=candidate_pool_multiplier,
    )
    task_generation_policies = resolve_task_generation_policies(
        config,
        policy_task_counts,
        context=policy_context,
    )
    claim_parser_vet_max_rows = int(
        task_generation_policies.get("claims", {}).get(
            "build_parser_vet_max_claim_rows",
            0,
        )
        or 0
    )
    large_claim_parser_vet_disabled = (
        claim_parser_vet_max_rows > 0
        and len(claims) > claim_parser_vet_max_rows
        and not pair_like_claim_corpus
    )
    # Build-time claim vetting stays local-only, but very large corpora should
    # not shell out to the parser for dozens of candidates during catalog build.
    runner_cfg["__disable_claim_build_parser_vet"] = (
        large_claim_parser_vet_disabled and not enable_small_corpus_parser_vet
    )
    runner_cfg["__disable_contradiction_build_parser_vet"] = not enable_small_corpus_parser_vet
    # Build-time candidate vetting stays local-only. Runtime can still use the
    # OpenAI parser, but catalog generation should not depend on network calls.
    runner_cfg["build_vet_claim_queries_with_pair_runner"] = False
    runner_cfg["build_vet_contradiction_queries_with_pair_runner"] = False
    runner_cfg["build_parser_vet_case_budget"] = max(
        int(runner_cfg.get("build_parser_vet_case_budget", 0) or 0),
        max(
            int(task_generation_policies.get("claims", {}).get("parser_vet_budget", 0) or 0),
            int(task_generation_policies.get("contradictions", {}).get("parser_vet_budget", 0) or 0),
            1,
        ),
    )
    runner_cfg["build_pair_vet_case_budget"] = max(
        int(runner_cfg.get("build_pair_vet_case_budget", 0) or 0),
        max(
            int(task_generation_policies.get("claims", {}).get("pair_vet_budget", 0) or 0),
            int(task_generation_policies.get("contradictions", {}).get("pair_vet_budget", 0) or 0),
            1,
        ),
    )
    collector_counts = {
        task_name: int(policy.get("catalog_target", requested_count) or requested_count)
        for task_name, requested_count in policy_task_counts.items()
        for policy in [task_generation_policies.get(task_name, {})]
    }
    sampling_with_doc_cap: dict[str, dict[str, Any]] = {}
    for key, value in sampling.items():
        if not isinstance(value, dict):
            continue
        policy = task_generation_policies.get(key, {})
        policy_sampling_fields = {
            policy_key: policy_value
            for policy_key, policy_value in policy.items()
            if policy_key
            not in {
                "requested_count",
                "size_bucket",
                "catalog_target",
                "parser_vet_budget",
                "pair_vet_budget",
            }
        }
        sampling_with_doc_cap[key] = {
            **policy_sampling_fields,
            **value,
            "max_cases_per_doc": max_cases_per_doc,
            "_catalog_target": int(policy.get("catalog_target", policy_task_counts.get(key, 0)) or 0),
            "_catalog_size_bucket": str(policy.get("size_bucket", "")),
        }
    catalog_runner_cfg = dict(runner_cfg)
    catalog_runner_cfg["__strict_small_corpus_vet"] = enable_small_corpus_parser_vet
    diagnostics: dict[str, Any] = {
        "corpus_context": {
            "corpus": str(config.get("corpus", "")),
            "corpus_doc_count": corpus_doc_count,
            "claim_row_count": len(claims),
            "pair_like_claim_corpus": pair_like_claim_corpus,
            "candidate_pool_multiplier": candidate_pool_multiplier,
            "max_cases_per_doc": max_cases_per_doc,
        },
        "requested_counts": task_counts,
        "policy_request_counts": policy_task_counts,
        "catalog_target_counts": collector_counts,
        "task_generation_policies": task_generation_policies,
        "task_build_behaviors": {},
        "actual_counts": {},
        "admissible_counts": {},
        "task_stage_counts": {},
    }
    return {
        "seed": seed,
        "sampling": sampling,
        "runner_cfg": runner_cfg,
        "max_cases_per_doc": max_cases_per_doc,
        "candidate_pool_multiplier": candidate_pool_multiplier,
        "task_counts": task_counts,
        "policy_task_counts": policy_task_counts,
        "tables": tables,
        "chunks": chunks,
        "claims": claims,
        "entities": entities,
        "redaction_views": redaction_views,
        "claim_views": claim_views,
        "reconstruction_views": reconstruction_views,
        "doc_chunks_by_doc": doc_chunks_by_doc,
        "chunk_text_lookup": chunk_text_lookup,
        "corpus_doc_count": corpus_doc_count,
        "para_richness": para_richness,
        "component_index": component_index,
        "claim_generation_views": claim_generation_views,
        "pair_like_claim_corpus": pair_like_claim_corpus,
        "fisc_claimlike_acceptable_sources": fisc_claimlike_acceptable_sources,
        "enable_small_corpus_parser_vet": enable_small_corpus_parser_vet,
        "task_generation_policies": task_generation_policies,
        "collector_counts": collector_counts,
        "sampling_with_doc_cap": sampling_with_doc_cap,
        "catalog_runner_cfg": catalog_runner_cfg,
        "diagnostics": diagnostics,
    }
