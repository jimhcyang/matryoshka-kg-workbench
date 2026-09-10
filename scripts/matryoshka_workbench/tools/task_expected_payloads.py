#!/usr/bin/env python3
"""Expected payload builders per task family."""

from __future__ import annotations

from typing import Any, Callable


def _claimlike_source_ids(case: dict[str, Any]) -> list[str]:
    acceptable = [
        str(value).strip()
        for value in case.get("acceptable_source_ids_any", [])
        if str(value).strip()
    ]
    if case.get("task_family") in {"claims", "reconstruction", "contradictions"} and acceptable:
        return []
    return [case["doc_id"]]


def build_expected_payload(
    case: dict[str, Any],
    *,
    format_expected_targets: Callable[[dict[str, Any]], dict[str, Any]],
    maybe_bool: Callable[[Any], bool],
) -> dict[str, Any]:
    if case["task_family"] == "value_redaction":
        expected_targets = format_expected_targets(case)
        answer_targets = case.get("answer_targets")
        if not isinstance(answer_targets, dict) or not answer_targets:
            answer_targets = expected_targets
        if not expected_targets:
            target_expectation = {
                "source_id": case["doc_id"],
            }
            if maybe_bool(case.get("strict_selected_value_match", False)):
                target_expectation["selected_value"] = case["ground_truth"]
            expected_targets = {"R1": target_expectation}
        return {
            "source_ids": [case["doc_id"]],
            "acceptable_source_ids_any": list(case.get("acceptable_source_ids_any", [])),
            "targets": expected_targets,
            "require_visualization": True,
        }
    if case["task_family"] == "sentence_redaction":
        return {
            "source_ids": [case["doc_id"]],
            "acceptable_source_ids_any": list(case.get("acceptable_source_ids_any", [])),
            "targets": {
                "S1": {
                    "source_id": case["doc_id"],
                    "selected_value": case["ground_truth"],
                }
            },
            "require_visualization": True,
        }
    if case["task_family"] == "claims":
        return {
            "source_ids": _claimlike_source_ids(case),
            "acceptable_source_ids_any": list(case.get("acceptable_source_ids_any", [])),
            "require_visualization": True,
        }
    if case["task_family"] == "reconstruction":
        return {
            "source_ids": _claimlike_source_ids(case),
            "acceptable_source_ids_any": list(case.get("acceptable_source_ids_any", [])),
            "require_visualization": True,
            "require_nonblank_pseudo_text": True,
        }
    if case["task_family"] == "contradictions":
        return {
            "source_ids": _claimlike_source_ids(case),
            "acceptable_source_ids_any": list(case.get("acceptable_source_ids_any", [])),
            "require_visualization": True,
        }
    return {}
