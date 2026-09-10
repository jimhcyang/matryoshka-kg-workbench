#!/usr/bin/env python3
"""Task-specific benchmark generation policies.

This module isolates query-catalog sizing and related build policies from the
task selectors in ``corpus_benchmark.py`` so per-task tuning does not require
editing the monolithic builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_UNIFORM_CATALOG_TARGET = 400


@dataclass(frozen=True)
class TaskPolicyContext:
    corpus: str
    corpus_doc_count: int
    claim_row_count: int
    pair_like_claim_corpus: bool
    candidate_pool_multiplier: int


DEFAULT_TASK_GENERATION_POLICIES: dict[str, dict[str, Any]] = {
    "reconstruction": {
        "catalog_multiplier": 18,
        "catalog_min_small": 24,
        "catalog_min_medium": 120,
        "catalog_min_large": 400,
        "catalog_max": 1000,
    },
    "claims": {
        "catalog_multiplier": 18,
        "catalog_min_small": 24,
        "catalog_min_medium": 140,
        "catalog_min_large": 320,
        "catalog_max": 800,
        "max_claim_rows_for_minimal_scan": 1600,
        "max_minimal_probe_rows": 600,
        "max_fast_seed_rows": 900,
        "max_ranked_candidate_rows": 1400,
        "max_candidate_case_budget": 600,
        "parser_vet_budget_multiplier": 20,
        "parser_vet_budget_cap": 320,
        "pair_vet_budget_multiplier": 8,
        "pair_vet_budget_cap": 80,
    },
    "value_redaction": {
        "catalog_multiplier": 20,
        "catalog_min_small": 24,
        "catalog_min_medium": 140,
        "catalog_min_large": 480,
        "catalog_max": 1200,
    },
    "sentence_redaction": {
        "catalog_multiplier": 20,
        "catalog_min_small": 24,
        "catalog_min_medium": 180,
        "catalog_min_large": 480,
        "catalog_max": 1200,
    },
    "contradictions": {
        "catalog_multiplier": 18,
        "catalog_min_small": 24,
        "catalog_min_medium": 140,
        "catalog_min_large": 320,
        "catalog_max": 800,
        "max_claim_rows_for_minimal_scan": 1600,
        "max_seed_claim_catalog": 320,
        "max_ranked_candidate_rows": 1400,
        "max_candidate_case_budget": 600,
        "parser_vet_budget_multiplier": 20,
        "parser_vet_budget_cap": 320,
        "pair_vet_budget_multiplier": 8,
        "pair_vet_budget_cap": 80,
    },
}

SMALL_CORPUS_DOC_THRESHOLD = 50


def corpus_size_bucket(context: TaskPolicyContext) -> str:
    if context.corpus_doc_count <= SMALL_CORPUS_DOC_THRESHOLD:
        return "small"
    return "large"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _catalog_floor(policy: Mapping[str, Any], bucket: str) -> int:
    return int(policy.get(f"catalog_min_{bucket}", 0) or 0)


def resolve_task_generation_policy(
    config: Mapping[str, Any],
    task_name: str,
    *,
    requested_count: int,
    context: TaskPolicyContext,
) -> dict[str, Any]:
    base_policy = dict(DEFAULT_TASK_GENERATION_POLICIES.get(task_name, {}))
    config_policies = config.get("task_generation_policies", {})
    if isinstance(config_policies, Mapping):
        override = config_policies.get(task_name, {})
        if isinstance(override, Mapping):
            base_policy = _deep_merge(base_policy, override)

    sampling = config.get("sampling", {})
    if isinstance(sampling, Mapping):
        task_sampling = sampling.get(task_name, {})
        if isinstance(task_sampling, Mapping):
            inline_override = {
                key: task_sampling[key]
                for key in (
                    "catalog_multiplier",
                    "catalog_min_small",
                    "catalog_min_medium",
                    "catalog_min_large",
                    "catalog_max",
                    "parser_vet_budget_multiplier",
                    "parser_vet_budget_cap",
                    "pair_vet_budget_multiplier",
                    "pair_vet_budget_cap",
                )
                if key in task_sampling
            }
            if inline_override:
                base_policy = _deep_merge(base_policy, inline_override)

    bucket = corpus_size_bucket(context)
    uniform_target = int(
        config.get("catalog_target_uniform", DEFAULT_UNIFORM_CATALOG_TARGET)
        or DEFAULT_UNIFORM_CATALOG_TARGET
    )
    uniform_target = max(1, uniform_target)
    target = max(int(requested_count), uniform_target)
    cap = max(int(requested_count), uniform_target)
    base_policy["catalog_max"] = cap

    parser_vet_multiplier = int(base_policy.get("parser_vet_budget_multiplier", 8) or 8)
    parser_vet_cap = int(base_policy.get("parser_vet_budget_cap", 0) or 0)
    parser_vet_budget = max(requested_count, requested_count * max(parser_vet_multiplier, 1))
    if parser_vet_cap > 0:
        parser_vet_budget = min(parser_vet_budget, parser_vet_cap)

    pair_vet_multiplier = int(base_policy.get("pair_vet_budget_multiplier", 2) or 2)
    pair_vet_cap = int(base_policy.get("pair_vet_budget_cap", 0) or 0)
    pair_vet_budget = max(requested_count, requested_count * max(pair_vet_multiplier, 1))
    if pair_vet_cap > 0:
        pair_vet_budget = min(pair_vet_budget, pair_vet_cap)

    return {
        **base_policy,
        "requested_count": requested_count,
        "size_bucket": bucket,
        "catalog_target_uniform": uniform_target,
        "catalog_target": target,
        "parser_vet_budget": parser_vet_budget,
        "pair_vet_budget": pair_vet_budget,
    }


def resolve_task_generation_policies(
    config: Mapping[str, Any],
    task_counts: Mapping[str, int],
    *,
    context: TaskPolicyContext,
) -> dict[str, dict[str, Any]]:
    return {
        task_name: resolve_task_generation_policy(
            config,
            task_name,
            requested_count=int(requested_count),
            context=context,
        )
        for task_name, requested_count in task_counts.items()
        if int(requested_count) > 0
    }
