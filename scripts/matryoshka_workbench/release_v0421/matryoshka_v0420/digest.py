"""Deterministic and Optional LLM digest contracts for v0.4.13.

The deterministic digest is complete for every task and target.  Optional LLM
prose receives only the bounded evidence-unit payload and is accepted only
when API execution, deterministic consistency, explanatory completeness,
target coverage, and evidence grounding all pass independently.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .contracts import (
    DIGEST_BUILDER_VERSION,
    MATCHER_VERSION,
    REQUIREMENTS_VERSION,
)
from .semantics import atomic_write_json, clean_text, stable_hash, utc_now_iso


LLM_INPUT_SCHEMA = "matryoshka_llm_digest_input_v0_4_13"
LLM_STATE_SCHEMA = "matryoshka_llm_digest_state_v0_4_13"

GATE_DEFINITIONS = {
    "EVENT_IDENTITY": (
        "The candidate does not sufficiently establish the same event; inspect "
        "actor, predicate, object, date, location, quantity, and participants."
    ),
    "CRITICAL_COVERAGE": (
        "The candidate is related but does not establish enough required facts."
    ),
    "ASSERTION": (
        "The candidate does not assert the proposition with compatible "
        "polarity, modality, or epistemic commitment."
    ),
    "TARGET_COVERAGE": (
        "One or more requested target IDs are missing, duplicated, or added."
    ),
    "EVIDENCE_GROUNDING": (
        "A cited source or passage is not present in the bounded authoritative "
        "payload."
    ),
    "EXPLANATORY_COMPLETENESS": (
        "A required result, why, evidence, alternatives, limitation, or pair "
        "transition explanation is missing."
    ),
}


def _task_name(task: str) -> str:
    return {
        "reconstruction": "Reconstruction",
        "redaction": "Value redaction",
        "sentence_redaction": "Sentence redaction",
        "claims": "Claims",
        "contradictions": "Contradiction",
    }.get(task, task.replace("_", " ").title())


def _result_sentence(unit: Mapping[str, Any]) -> str:
    task = clean_text(unit.get("task", ""))
    finding = unit.get("deterministic_finding", {})
    status = clean_text(finding.get("semantic_status", "")) or "NO_STATUS"
    selected_text = clean_text(finding.get("selected_value_or_text", ""))
    selected_source = clean_text(finding.get("selected_source_doc_id", ""))
    primary = clean_text(finding.get("primary_finding", ""))
    primary_source = clean_text(
        finding.get("primary_finding_source_doc_id", "")
    )
    if task == "redaction":
        return (
            f"The slot is `{status}`"
            + (f" with proposed value **{selected_text}**" if selected_text else "")
            + (f" from `{selected_source}`" if selected_source else "")
            + "."
        )
    if task == "sentence_redaction":
        return (
            f"The sentence target is `{status}`"
            + (f" with proposed text: {selected_text}" if selected_text else "")
            + (f" from `{selected_source}`" if selected_source else "")
            + "."
        )
    if task == "claims":
        return (
            f"The claim is `{status}`"
            + (f" using `{selected_source}`" if selected_source else "")
            + "."
        )
    if task == "contradictions":
        return (
            f"The evidence pattern is `{status}`"
            + (f"; primary finding `{primary}`" if primary else "")
            + (f" from `{primary_source}`" if primary_source else "")
            + "."
        )
    return (
        f"The fragment is `{status}`"
        + (f" from `{selected_source}`" if selected_source else "")
        + "."
    )


def _fragment_line(fragment: Mapping[str, Any]) -> str:
    source = clean_text(fragment.get("source_doc_id", "")) or "unknown source"
    passage = clean_text(fragment.get("passage_id", "")) or "unknown passage"
    role = clean_text(fragment.get("evidence_role", "")) or "evidence"
    excerpt = clean_text(fragment.get("excerpt", ""))
    score = fragment.get("final_score", "")
    score_text = f"; score {score:.4f}" if isinstance(score, float) else ""
    return (
        f"- **{role.replace('_', ' ').title()}** "
        f"[source: {source}] [passage: {passage}]{score_text}: "
        f"{excerpt or 'No bounded excerpt was available.'}"
    )


def deterministic_task_digest(payload: Mapping[str, Any]) -> str:
    task = clean_text(payload.get("task", ""))
    lines = [
        f"# Matryoshka {_task_name(task)} Deterministic Digest",
        "",
        f"- Matcher: `{MATCHER_VERSION}`",
        f"- Digest Builder: `{DIGEST_BUILDER_VERSION}`",
        f"- Requirements: `{REQUIREMENTS_VERSION}`",
        f"- Pair role: `{clean_text(payload.get('pair_role', 'standalone'))}`",
        f"- Pair status: `{clean_text(payload.get('pair_status', 'STANDALONE'))}`",
        f"- Target coverage: `{clean_text(payload.get('target_coverage_state', ''))}` "
        f"({payload.get('emitted_target_count', 0)}/"
        f"{payload.get('requested_target_count', 0)})",
        f"- Query clause coverage: `"
        f"{clean_text(payload.get('query_clause_coverage_state', 'NOT_ASSESSED'))}` "
        f"({payload.get('covered_clause_count', payload.get('executed_clause_count', 0))}/"
        f"{payload.get('query_clause_count', 0)} covered; "
        f"{payload.get('executed_target_clause_count', payload.get('executed_clause_count', 0))} target-executed; "
        f"{payload.get('context_consumed_clause_count', 0)} context-consumed; "
        f"{payload.get('review_clause_count', 0)} review; "
        f"{payload.get('blocked_clause_count', 0)} blocked)",
        "",
        "## Overall conclusion",
        "",
        (
            "Every requested target has a deterministic evidence unit, and "
            "every substantive query clause was either executed as a target "
            "or consumed as task context."
            if payload.get("target_coverage_state") == "PASS"
            and payload.get("query_clause_coverage_state") in {
                "PASS", "NOT_ASSESSED", "NOT_APPLICABLE"
            }
            else "Target evidence units were emitted, but query clause "
                 "coverage is partial; review the clause execution ledger "
                 "before interpreting this as a full-query result."
            if payload.get("target_coverage_state") == "PASS"
            else "Target coverage failed; inspect the missing, duplicate, or "
                 "unexpected target lists before interpretation."
        ),
        "",
        "## Target index",
        "",
    ]
    for unit in payload.get("units", []):
        target_id = clean_text(unit.get("target_id", ""))
        lines.append(f"- `{target_id}`")
    for unit in payload.get("units", []):
        target_id = clean_text(unit.get("target_id", ""))
        target = unit.get("target", {})
        finding = unit.get("deterministic_finding", {})
        fragments = unit.get("evidence_fragments", [])
        lines.extend([
            "",
            f"## TARGET {target_id}",
            "",
            f"### Result",
            _result_sentence(unit),
            "",
            "### Target context",
            clean_text(target.get("target_text", ""))
            or clean_text(target.get("left_context", ""))
            + " [TARGET] "
            + clean_text(target.get("right_context", ""))
            or "No bounded target text was available.",
            "",
            "### Why",
            clean_text(finding.get("decision_reason", ""))
            or clean_text(finding.get("unresolved_reason", ""))
            or "The authoritative task evaluation did not provide a reason.",
            "",
            "### Evidence",
        ])
        if fragments:
            lines.extend(_fragment_line(fragment) for fragment in fragments)
        else:
            lines.append("- No qualifying evidence fragment was available.")
        alternatives = [
            item for item in fragments
            if clean_text(item.get("evidence_role", ""))
            in {"alternative", "partial", "related", "background", "rejected"}
        ]
        lines.extend(["", "### Alternatives"])
        if alternatives:
            lines.extend(_fragment_line(item) for item in alternatives)
        else:
            lines.append("- No additional bounded alternative was exported.")
        limitations = [
            clean_text(value) for value in unit.get("limitations", [])
            if clean_text(value)
        ]
        lines.extend(["", "### Limitations"])
        if limitations:
            lines.extend(f"- {value}" for value in limitations)
        else:
            lines.append("- No task-level limitation was recorded.")
    lines.extend([
        "",
        "Deterministic evaluations, evidence units, and audit tables are "
        "authoritative. Optional LLM prose cannot change them.",
    ])
    return "\n".join(lines)


def pair_evidence_payload(
    *,
    pair_id: str,
    pair_status: str,
    baseline: Mapping[str, Any],
    holdout: Mapping[str, Any],
    comparison: pd.DataFrame,
) -> Dict[str, Any]:
    baseline_map = {
        clean_text(unit.get("target_id", "")): unit
        for unit in baseline.get("units", [])
    }
    holdout_map = {
        clean_text(unit.get("target_id", "")): unit
        for unit in holdout.get("units", [])
    }
    comparison_map = {
        clean_text(row.get("target_id", "")): row
        for row in comparison.to_dict("records")
    }
    requested = list(dict.fromkeys(
        list(baseline.get("requested_target_ids", []))
        + list(holdout.get("requested_target_ids", []))
    ))
    targets = []
    for target_id in requested:
        baseline_unit = dict(baseline_map.get(target_id, {}))
        holdout_unit = dict(holdout_map.get(target_id, {}))
        if baseline_unit:
            baseline_unit["pair_status"] = pair_status
        if holdout_unit:
            holdout_unit["pair_status"] = pair_status
        targets.append({
            "target_id": target_id,
            "baseline": baseline_unit,
            "holdout": holdout_unit,
            "transition": comparison_map.get(target_id, {}),
            "canonical_outcome": {
                "baseline_status": clean_text(
                    baseline_unit.get(
                        "deterministic_finding", {}
                    ).get("semantic_status", "")
                ),
                "holdout_status": clean_text(
                    holdout_unit.get(
                        "deterministic_finding", {}
                    ).get("semantic_status", "")
                ),
                "transition": clean_text(
                    comparison_map.get(target_id, {}).get("change", "")
                ),
            },
        })
    emitted = [clean_text(item.get("target_id", "")) for item in targets]
    missing = sorted(set(requested) - set(emitted))
    return {
        "schema_version": "matryoshka_pair_evidence_units_v0_4_13",
        "llm_input_schema_version": LLM_INPUT_SCHEMA,
        "matcher_version": MATCHER_VERSION,
        "digest_builder_version": DIGEST_BUILDER_VERSION,
        "generated_at": utc_now_iso(),
        "pair_id": pair_id,
        "pair_status": pair_status,
        "task": clean_text(baseline.get("task", ""))
        or clean_text(holdout.get("task", "")),
        "requested_target_ids": requested,
        "emitted_target_ids": emitted,
        "target_coverage_state": "PASS" if not missing else "FAIL",
        "query_clause_coverage_state": clean_text(
            baseline.get("query_clause_coverage_state", "NOT_ASSESSED")
        ),
        "query_clause_count": int(
            baseline.get("query_clause_count", 0) or 0
        ),
        "executed_clause_count": int(
            baseline.get("executed_clause_count", 0) or 0
        ),
        "executed_target_clause_count": int(
            baseline.get(
                "executed_target_clause_count",
                baseline.get("executed_clause_count", 0),
            ) or 0
        ),
        "context_consumed_clause_count": int(
            baseline.get("context_consumed_clause_count", 0) or 0
        ),
        "covered_clause_count": int(
            baseline.get(
                "covered_clause_count",
                baseline.get("executed_clause_count", 0),
            ) or 0
        ),
        "review_clause_count": int(
            baseline.get("review_clause_count", 0) or 0
        ),
        "blocked_clause_count": int(
            baseline.get("blocked_clause_count", 0) or 0
        ),
        "review_clause_ids": baseline.get("review_clause_ids", []),
        "blocked_clause_ids": baseline.get("blocked_clause_ids", []),
        "missing_target_ids": missing,
        "gate_definitions": GATE_DEFINITIONS,
        "targets": targets,
    }


def deterministic_pair_digest(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Matryoshka Controlled-Pair Deterministic Digest",
        "",
        f"- Matcher: `{MATCHER_VERSION}`",
        f"- Digest Builder: `{DIGEST_BUILDER_VERSION}`",
        f"- Task: `{clean_text(payload.get('task', ''))}`",
        f"- Pair status: `{clean_text(payload.get('pair_status', ''))}`",
        f"- Target coverage: `{clean_text(payload.get('target_coverage_state', ''))}`",
        f"- Query clause coverage: `"
        f"{clean_text(payload.get('query_clause_coverage_state', 'NOT_ASSESSED'))}` "
        f"({payload.get('covered_clause_count', payload.get('executed_clause_count', 0))}/"
        f"{payload.get('query_clause_count', 0)} covered; "
        f"{payload.get('executed_target_clause_count', payload.get('executed_clause_count', 0))} target-executed; "
        f"{payload.get('context_consumed_clause_count', 0)} context-consumed)",
        "",
        "## Overall conclusion",
        "",
        (
            "The baseline and holdout are presented as one controlled pair. "
            "Each requested target includes the evidence used in both arms and "
            "the deterministic transition."
        ),
        "",
        "## Target index",
        "",
    ]
    for item in payload.get("targets", []):
        lines.append(f"- `{clean_text(item.get('target_id', ''))}`")
    for item in payload.get("targets", []):
        target_id = clean_text(item.get("target_id", ""))
        transition = item.get("transition", {})
        canonical = item.get("canonical_outcome", {})
        lines.extend([
            "",
            f"## TARGET {target_id}",
            "",
            "### What changed",
            (
                "Canonical Result: "
                f"baseline=`{clean_text(canonical.get('baseline_status', ''))}`; "
                f"holdout=`{clean_text(canonical.get('holdout_status', ''))}`; "
                f"transition=`{clean_text(canonical.get('transition', ''))}`."
            ),
            (
                f"`{clean_text(transition.get('change', 'NO_CHANGE_RECORDED'))}`: "
                f"{clean_text(transition.get('analyst_interpretation', ''))}"
            ),
        ])
        for label, key in (("Baseline", "baseline"), ("Holdout", "holdout")):
            unit = item.get(key, {})
            lines.extend([
                "",
                f"### {label} result",
                _result_sentence(unit) if unit else "Target missing from this arm.",
                "",
                f"### {label} evidence",
            ])
            fragments = unit.get("evidence_fragments", []) if unit else []
            if fragments:
                lines.extend(_fragment_line(fragment) for fragment in fragments)
            else:
                lines.append("- No bounded evidence fragment was exported.")
            limitations = unit.get("limitations", []) if unit else []
            lines.extend(["", f"### {label} limitations"])
            if limitations:
                lines.extend(f"- {clean_text(value)}" for value in limitations)
            else:
                lines.append("- No task-level limitation was recorded.")
    lines.extend([
        "",
        "Deterministic pair comparison, evidence units, and audit tables are "
        "authoritative. Optional LLM prose cannot change them.",
    ])
    return "\n".join(lines)


def llm_input(
    payload: Mapping[str, Any],
    deterministic_digest: str,
) -> Dict[str, Any]:
    return {
        "schema_version": LLM_INPUT_SCHEMA,
        "generated_at": utc_now_iso(),
        "authority": (
            "Deterministic fields and evidence fragments are authoritative. "
            "The LLM may explain but must not change them."
        ),
        "required_sections_per_target": [
            "Result", "Why", "Evidence", "Alternatives", "Limitations",
        ],
        "required_pair_sections": [
            "Baseline result", "Holdout result", "What changed",
        ] if "targets" in payload else [],
        "canonical_outcomes": {
            clean_text(item.get("target_id", "")):
                dict(item.get("canonical_outcome", {}))
            for item in payload.get("targets", [])
            if clean_text(item.get("target_id", ""))
        },
        "gate_definitions": GATE_DEFINITIONS,
        "deterministic_digest": deterministic_digest,
        "evidence_payload": dict(payload),
    }


def llm_prompt(input_payload: Mapping[str, Any]) -> str:
    return (
        "You explain a deterministic Matryoshka result. Do not change any "
        "status, selected value, source, score, gate, or pair transition. "
        "Do not introduce a fact or citation absent from the JSON.\n"
        "Use exactly one heading '## TARGET <id>' for every requested target "
        "and no other TARGET headings. Within every target use headings "
        "'### Result', '### Why', '### Evidence', '### Alternatives', and "
        "'### Limitations'. For a controlled pair also explain baseline, "
        "holdout, and what changed. Reproduce each target's baseline status, "
        "holdout status, and transition exactly as supplied in "
        "canonical_outcomes. Cite evidence with the exact forms "
        "[source: <source_doc_id>] and [passage: <passage_id>]. Interpret "
        "scores only with their comparison or gate. If evidence is absent, say "
        "so. Begin with a short plain-language overall conclusion.\n\n"
        + json.dumps(input_payload, ensure_ascii=False, indent=2, default=str)
    )


def _canonical_citations(unit: Mapping[str, Any]) -> str:
    citations: List[str] = []
    seen = set()
    for fragment in unit.get("evidence_fragments", []):
        source = clean_text(fragment.get("source_doc_id", ""))
        passage = clean_text(fragment.get("passage_id", ""))
        if not source or not passage:
            continue
        citation = f"[source: {source}] [passage: {passage}]"
        if citation not in seen:
            citations.append(citation)
            seen.add(citation)
    return "; ".join(citations) or (
        "No evidence-bearing citation is available for this arm."
    )


def _finding_summary(unit: Mapping[str, Any]) -> str:
    finding = unit.get("deterministic_finding", {})
    values = [
        clean_text(finding.get("semantic_status", "")),
        clean_text(finding.get("selected_value_or_text", "")),
        clean_text(finding.get("selected_source_doc_id", "")),
        clean_text(finding.get("primary_finding", "")),
        clean_text(finding.get("primary_finding_source_doc_id", "")),
    ]
    return "; ".join(value for value in values if value) or "UNRESOLVED"


def render_pair_explanation(
    evidence_payload: Mapping[str, Any],
) -> str:
    """Render a structurally valid pair explanation from saved evidence.

    This deterministic renderer is also the bounded repair path for an LLM
    response that is consistent but incomplete.  It cannot invent claims or
    citations because all content is copied from the canonical pair payload.
    """
    pieces = [
        "# Matryoshka controlled-pair explanation",
        (
            "This explanation is rendered from authoritative deterministic "
            "pair evidence and cannot alter its decisions."
        ),
    ]
    for item in evidence_payload.get("targets", []):
        target_id = clean_text(item.get("target_id", ""))
        baseline = item.get("baseline", {})
        holdout = item.get("holdout", {})
        outcome = item.get("canonical_outcome", {})
        baseline_summary = _finding_summary(baseline)
        holdout_summary = _finding_summary(holdout)
        transition = clean_text(outcome.get("transition", "")) or (
            "NO_DECLARED_TRANSITION"
        )
        citations = "; ".join(
            value for value in (
                _canonical_citations(baseline),
                _canonical_citations(holdout),
            )
            if not value.startswith("No evidence-bearing")
        ) or "No evidence-bearing citation is available for this target."
        pieces.extend([
            f"## TARGET {target_id}",
            "### Baseline result",
            baseline_summary,
            "### Holdout result",
            holdout_summary,
            "### What changed",
            (
                f"{transition}; baseline="
                f"{clean_text(outcome.get('baseline_status', ''))}; "
                f"holdout={clean_text(outcome.get('holdout_status', ''))}."
            ),
            "### Result",
            (
                f"Baseline {baseline_summary}. Holdout "
                f"{holdout_summary}. Transition {transition}."
            ),
            "### Why",
            (
                "The result follows the saved deterministic arm findings "
                "and controlled exclusion comparison."
            ),
            "### Evidence",
            citations,
            "### Alternatives",
            (
                "Candidate alternatives remain recorded in the authoritative "
                "task audit and evidence-unit artifacts."
            ),
            "### Limitations",
            (
                "This explanation does not add evidence and cannot override "
                "the deterministic pair decision."
            ),
        ])
    return "\n\n".join(pieces) + "\n"


def _expected_targets(payload: Mapping[str, Any]) -> List[str]:
    if "requested_target_ids" in payload:
        return [
            clean_text(value)
            for value in payload.get("requested_target_ids", [])
            if clean_text(value)
        ]
    evidence = payload.get("evidence_payload", {})
    return [
        clean_text(value)
        for value in evidence.get("requested_target_ids", [])
        if clean_text(value)
    ]


def _units(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if "units" in payload:
        return list(payload.get("units", []))
    if "targets" in payload:
        flattened: List[Mapping[str, Any]] = []
        for item in payload.get("targets", []):
            for key in ("baseline", "holdout"):
                unit = item.get(key, {})
                if unit:
                    flattened.append(unit)
        return flattened
    evidence = payload.get("evidence_payload", {})
    return _units(evidence)


def _target_block(text: str, target_id: str) -> str:
    match = re.search(
        rf"(?ims)^##\s+TARGET\s+{re.escape(target_id)}\s*$"
        rf"(.*?)(?=^##\s+TARGET\s+|\Z)",
        text or "",
    )
    return match.group(1) if match else ""


def _target_subsection(block: str, heading: str) -> str:
    """Return one level-three section without text from adjacent pair arms."""
    match = re.search(
        rf"(?ims)^###\s+{re.escape(heading)}\s*$"
        rf"(.*?)(?=^###\s+|\Z)",
        block or "",
    )
    return match.group(1) if match else ""


def _status_is_unresolved(status: str) -> bool:
    upper = clean_text(status).upper()
    return any(
        token in upper
        for token in (
            "UNRESOLVED", "NOT_RECONSTRUCTABLE", "NO_DIRECT",
            "NO_SUPPORTED", "NO_CANDIDATE", "BLOCKED",
            "INSUFFICIENT", "RELATED_EVIDENCE_ONLY",
        )
    )


def _outcome_contradictions(
    block: str,
    canonical_statuses: Sequence[str],
) -> List[str]:
    """Find plain-language claims that reverse deterministic resolution."""
    # Canonical tokens themselves do not match the grammatical contradiction
    # patterns below (for example, ``holdout=UNRESOLVED`` is not
    # ``the holdout is resolved``), so retain the full block.  Removing a short
    # token such as RESOLVED would also erase the very contradiction we need to
    # detect in explanatory prose.
    scrubbed = block
    unresolved = any(
        _status_is_unresolved(status) for status in canonical_statuses
    )
    if unresolved and re.search(
        r"\b(?:is|was|became|remains?|leading\s+to\s+a)\s+resolved\b"
        r"|\bsuccessfully\s+(?:resolved|recovered|filled|reconstructed)\b"
        r"|\b(?:is|was|became)\s+reconstructable\b"
        r"|\bconfirmed\s+(?:the\s+)?(?:answer|fill|resolution)\b",
        scrubbed,
        flags=re.I,
    ):
        return ["LLM_OUTCOME_CONTRADICTION"]
    if (
        canonical_statuses
        and all(not _status_is_unresolved(status) for status in canonical_statuses)
        and re.search(
            r"\b(?:remains?|is|was)\s+unresolved\b"
            r"|\bno\s+(?:supported\s+)?(?:answer|fill|resolution)\b",
            scrubbed,
            flags=re.I,
        )
    ):
        return ["LLM_OUTCOME_CONTRADICTION"]
    return []


def validate_llm_explanation(
    *,
    text: str,
    evidence_payload: Mapping[str, Any],
    request_state: str,
    model: str = "",
    error_type: str = "",
    error_detail: str = "",
) -> Dict[str, Any]:
    expected = _expected_targets(evidence_payload)
    headings = re.findall(
        r"(?im)^##\s+TARGET\s+([^\s#]+)\s*$", text or ""
    )
    missing = sorted(set(expected) - set(headings))
    unexpected = sorted(set(headings) - set(expected))
    duplicate = sorted({value for value in headings if headings.count(value) > 1})
    api_success = request_state in {
        "GENERATED_PENDING_VALIDATION", "GENERATED",
    } and bool(clean_text(text))
    api_state = (
        "SUCCESS" if api_success else request_state
        if request_state else "NOT_REQUESTED"
    )
    if not api_success:
        return {
            "schema_version": LLM_STATE_SCHEMA,
            "state": api_state,
            "api_call_state": api_state,
            "consistency_state": "NOT_RUN",
            "consistency_validation_state": "NOT_RUN",
            "explanatory_completeness_state": "NOT_RUN",
            "target_coverage_state": "NOT_RUN",
            "evidence_grounding_state": "NOT_RUN",
            "model": model,
            "error_type": error_type,
            "error_detail": clean_text(error_detail)[:500],
            "retryable": api_state in {
                "CALL_FAILED", "API_KEY_UNAVAILABLE", "EMPTY_RESPONSE",
                "RATE_LIMITED",
            },
            "reasons": (
                [clean_text(error_detail)[:500]]
                if clean_text(error_detail) else []
            ),
            "missing_target_ids": [],
            "unexpected_target_ids": [],
            "duplicate_target_ids": [],
            "response_sha256": "",
            "attempt_timestamp": utc_now_iso(),
            "authoritative_fallback": "deterministic_digest",
        }
    target_state = (
        "PASS" if api_success and not missing and not unexpected and not duplicate
        else "NOT_RUN" if not api_success else "FAIL"
    )
    lower = (text or "").lower()
    consistency_reasons: List[str] = []
    for unit in _units(evidence_payload):
        target_id = clean_text(unit.get("target_id", ""))
        finding = unit.get("deterministic_finding", {})
        status = clean_text(finding.get("semantic_status", ""))
        if target_id in expected and status and (
            status.lower() not in lower
            and status.lower().replace("_", " ") not in lower
        ):
            consistency_reasons.append("missing_status:" + target_id)
        for field in (
            "selected_value_or_text", "selected_source_doc_id",
            "primary_finding", "primary_finding_source_doc_id",
        ):
            value = clean_text(finding.get(field, ""))
            if value and value.lower() not in lower:
                consistency_reasons.append(
                    f"missing_{field}:" + target_id
                )
    if "targets" in evidence_payload:
        for item in evidence_payload.get("targets", []):
            target_id = clean_text(item.get("target_id", ""))
            block = _target_block(text, target_id)
            outcome = item.get("canonical_outcome", {})
            required = (
                ("baseline_status", clean_text(
                    outcome.get("baseline_status", "")
                )),
                ("holdout_status", clean_text(
                    outcome.get("holdout_status", "")
                )),
                ("transition", clean_text(
                    outcome.get("transition", "")
                )),
            )
            for dimension, value in required:
                if value and (
                    value.lower() not in block.lower()
                    and value.lower().replace("_", " ")
                    not in block.lower()
                ):
                    consistency_reasons.append(
                        f"missing_{dimension}:{target_id}"
                    )
            # A pair target deliberately contains two different outcomes.  A
            # target-wide scan treats a true baseline resolution as a
            # contradiction whenever the holdout is unresolved (and vice
            # versa).  Validate each arm only against its own canonical state.
            for section_name, status in (
                ("Baseline result", required[0][1]),
                ("Holdout result", required[1][1]),
            ):
                section = _target_subsection(block, section_name)
                if section and status:
                    consistency_reasons.extend(
                        reason + ":" + target_id + ":" + section_name
                        for reason in _outcome_contradictions(
                            section, [status]
                        )
                    )
    else:
        for unit in evidence_payload.get("units", []):
            target_id = clean_text(unit.get("target_id", ""))
            status = clean_text(
                unit.get("deterministic_finding", {}).get(
                    "semantic_status", ""
                )
            )
            consistency_reasons.extend(
                reason + ":" + target_id
                for reason in _outcome_contradictions(
                    _target_block(text, target_id), [status] if status else []
                )
            )
    consistency_state = (
        "PASS" if api_success and not consistency_reasons
        else "NOT_RUN" if not api_success else "FAIL"
    )
    allowed_sources = set()
    allowed_passages = set()
    units_by_target: Dict[str, List[Mapping[str, Any]]] = {}
    for unit in _units(evidence_payload):
        target_id = clean_text(unit.get("target_id", ""))
        units_by_target.setdefault(target_id, []).append(unit)
        for fragment in unit.get("evidence_fragments", []):
            source = clean_text(fragment.get("source_doc_id", ""))
            passage = clean_text(fragment.get("passage_id", ""))
            if source:
                allowed_sources.add(source)
            if passage:
                allowed_passages.add(passage)
    cited_sources = set(re.findall(r"\[source:\s*([^\]]+)\]", text or "", re.I))
    cited_passages = set(re.findall(r"\[passage:\s*([^\]]+)\]", text or "", re.I))
    invented_sources = sorted(cited_sources - allowed_sources)
    invented_passages = sorted(cited_passages - allowed_passages)
    grounding_reasons = [
        "invented_source:" + value for value in invented_sources
    ] + [
        "invented_passage:" + value for value in invented_passages
    ]
    for target_id in expected:
        has_fragments = any(
            unit.get("evidence_fragments", [])
            for unit in units_by_target.get(target_id, [])
        )
        if has_fragments:
            block_match = re.search(
                rf"(?ims)^##\s+TARGET\s+{re.escape(target_id)}\s*$"
                rf"(.*?)(?=^##\s+TARGET\s+|\Z)",
                text or "",
            )
            block = block_match.group(1) if block_match else ""
            if "[source:" not in block.lower() or "[passage:" not in block.lower():
                grounding_reasons.append("missing_grounded_citation:" + target_id)
    grounding_state = (
        "PASS" if api_success and not grounding_reasons
        else "NOT_RUN" if not api_success else "FAIL"
    )
    completeness_reasons: List[str] = []
    pair_payload = "targets" in evidence_payload
    for target_id in expected:
        block_match = re.search(
            rf"(?ims)^##\s+TARGET\s+{re.escape(target_id)}\s*$"
            rf"(.*?)(?=^##\s+TARGET\s+|\Z)",
            text or "",
        )
        block = block_match.group(1) if block_match else ""
        for heading in ("Result", "Why", "Evidence", "Alternatives", "Limitations"):
            if not re.search(
                rf"(?im)^###\s+{heading}\s*$", block
            ):
                completeness_reasons.append(
                    "missing_section:" + target_id + ":" + heading
                )
        if pair_payload:
            for heading in (
                "Baseline result", "Holdout result", "What changed"
            ):
                if not _target_subsection(block, heading):
                    completeness_reasons.append(
                        "missing_section:" + target_id + ":" + heading
                    )
    completeness_state = (
        "PASS" if api_success and not completeness_reasons
        else "NOT_RUN" if not api_success else "FAIL"
    )
    all_pass = all(
        value == "PASS"
        for value in (
            consistency_state, completeness_state, target_state,
            grounding_state,
        )
    ) and api_state == "SUCCESS"
    if all_pass:
        state = "VALIDATED_EXPLANATORY"
    elif api_state != "SUCCESS":
        state = api_state
    elif consistency_state == "PASS" and (
        completeness_state != "PASS" or target_state != "PASS"
    ):
        state = "CONSISTENT_BUT_NOT_EXPLANATORY"
    else:
        state = "REJECTED"
    return {
        "schema_version": LLM_STATE_SCHEMA,
        "state": state,
        "api_call_state": api_state,
        "consistency_state": consistency_state,
        "consistency_validation_state": consistency_state,
        "explanatory_completeness_state": completeness_state,
        "target_coverage_state": target_state,
        "evidence_grounding_state": grounding_state,
        "model": model,
        "error_type": error_type,
        "error_detail": clean_text(error_detail)[:500],
        "retryable": api_state in {
            "CALL_FAILED", "API_KEY_UNAVAILABLE", "EMPTY_RESPONSE",
            "RATE_LIMITED",
        },
        "reasons": (
            consistency_reasons + completeness_reasons + grounding_reasons
            + ["missing_target:" + value for value in missing]
            + ["unexpected_target:" + value for value in unexpected]
            + ["duplicate_target:" + value for value in duplicate]
        ),
        "missing_target_ids": missing,
        "unexpected_target_ids": unexpected,
        "duplicate_target_ids": duplicate,
        "response_sha256": stable_hash(text) if text else "",
        "attempt_timestamp": utc_now_iso(),
        "authoritative_fallback": "deterministic_digest",
    }


def _llm_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    pieces = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
    return "\n\n".join(pieces)


def generate_optional_llm(
    *,
    enabled: bool,
    evidence_payload: Mapping[str, Any],
    deterministic_digest: str,
    model: str,
    api_key_env: str,
    timeout_seconds: float,
    max_retries: int,
    max_output_tokens: int,
    output_dir: Path,
    pair_level: bool = False,
) -> Dict[str, Any]:
    key_name = clean_text(api_key_env) or "OPENAI_API_KEY"
    input_payload = llm_input(evidence_payload, deterministic_digest)
    input_name = (
        "pair_llm_digest_input_v0413.json"
        if pair_level else "llm_digest_input_v0413.json"
    )
    output_name = (
        "semantic_match_pair_digest_llm_v0413.md"
        if pair_level else "semantic_match_task_digest_llm_v0413.md"
    )
    state_name = (
        "pair_llm_digest_state_v0413.json"
        if pair_level else "llm_digest_state_v0413.json"
    )
    atomic_write_json(output_dir / input_name, input_payload)
    request_state = "NOT_REQUESTED"
    response_text = ""
    error_type = ""
    error_detail = ""
    if enabled:
        api_key = os.getenv(key_name, "").strip()
        if not api_key:
            request_state = "API_KEY_UNAVAILABLE"
            error_type = "missing_api_key"
            error_detail = f"Environment variable {key_name} is empty."
        else:
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=api_key,
                    timeout=max(5.0, float(timeout_seconds)),
                    max_retries=max(0, int(max_retries)),
                )
                kwargs = {
                    "model": model,
                    "input": llm_prompt(input_payload),
                    "max_output_tokens": max(512, int(max_output_tokens)),
                }
                try:
                    response = client.responses.create(**kwargs)
                except TypeError:
                    kwargs.pop("max_output_tokens", None)
                    response = client.responses.create(**kwargs)
                response_text = _llm_text(response)
                request_state = (
                    "GENERATED_PENDING_VALIDATION"
                    if response_text else "EMPTY_RESPONSE"
                )
            except Exception as exc:
                error_text = clean_text(str(exc))
                error_type = type(exc).__name__
                lower = error_text.lower()
                request_state = (
                    "QUOTA_EXHAUSTED"
                    if (
                        "insufficient_quota" in lower
                        or "quota" in lower
                    )
                    else "RATE_LIMITED"
                    if (
                        "ratelimit" in error_type.lower()
                        or "rate limit" in lower
                        or "429" in lower
                    )
                    else "AUTHENTICATION_FAILED"
                    if (
                        "authentication" in error_type.lower()
                        or "incorrect api key" in lower
                        or "invalid api key" in lower
                        or "401" in lower
                    )
                    else "CALL_FAILED"
                )
                secret = api_key
                error_detail = (
                    error_text.replace(secret, "[REDACTED]")
                    if secret else error_text
                )
    state = validate_llm_explanation(
        text=response_text,
        evidence_payload=evidence_payload,
        request_state=request_state,
        model=model,
        error_type=error_type,
        error_detail=error_detail,
    )
    initial_state = dict(state)
    repair_eligible = (
        pair_level
        and request_state == "GENERATED_PENDING_VALIDATION"
        and state.get("consistency_state") == "PASS"
        and state.get("target_coverage_state") == "PASS"
        and (
            state.get("explanatory_completeness_state") != "PASS"
            or state.get("evidence_grounding_state") != "PASS"
        )
    )
    if repair_eligible:
        repaired_text = render_pair_explanation(evidence_payload)
        repaired_state = validate_llm_explanation(
            text=repaired_text,
            evidence_payload=evidence_payload,
            request_state="GENERATED_PENDING_VALIDATION",
            model=model,
        )
        repaired_state.update({
            "repair_applied": True,
            "repair_method":
                "deterministic_structured_pair_renderer_from_saved_evidence",
            "repair_attempt_count": 1,
            "initial_state": initial_state.get("state", ""),
            "initial_reasons": list(initial_state.get("reasons", [])),
            "parser_rerun": False,
            "matcher_rerun": False,
        })
        state = repaired_state
        response_text = repaired_text
    else:
        state.update({
            "repair_applied": False,
            "repair_method": "",
            "repair_attempt_count": 0,
            "initial_state": initial_state.get("state", ""),
            "initial_reasons": list(initial_state.get("reasons", [])),
            "parser_rerun": False,
            "matcher_rerun": False,
        })
    state["prompt_version"] = "llm_digest_prompt_v0_4_13"
    state["credential_source"] = "environment:" + key_name
    state["pair_level"] = pair_level
    if response_text:
        (output_dir / output_name).write_text(
            response_text, encoding="utf-8"
        )
    atomic_write_json(output_dir / state_name, state)
    return state


def write_task_digest_bundle(
    output_dir: Path,
    evidence_payload: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    digest = deterministic_task_digest(evidence_payload)
    (output_dir / "semantic_match_task_digest_v0413.md").write_text(
        digest, encoding="utf-8"
    )
    input_payload = llm_input(evidence_payload, digest)
    atomic_write_json(output_dir / "llm_digest_input_v0413.json", input_payload)
    return digest, input_payload


def write_pair_digest_bundle(
    *,
    root: Path,
    pair_id: str,
    pair_status: str,
    baseline_dir: Path,
    holdout_dir: Path,
    comparison: pd.DataFrame,
) -> Tuple[Dict[str, Any], str]:
    baseline_path = baseline_dir / "task_evidence_units_v0413.json"
    holdout_path = holdout_dir / "task_evidence_units_v0413.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    for arm_payload, arm_dir in (
        (baseline, baseline_dir),
        (holdout, holdout_dir),
    ):
        arm_payload["pair_status"] = pair_status
        for unit in arm_payload.get("units", []):
            unit["pair_status"] = pair_status
        atomic_write_json(
            arm_dir / "task_evidence_units_v0413.json", arm_payload
        )
        arm_digest = deterministic_task_digest(arm_payload)
        (arm_dir / "semantic_match_task_digest_v0413.md").write_text(
            arm_digest, encoding="utf-8"
        )
        (arm_dir / "semantic_match_task_digest_v048.md").write_text(
            arm_digest, encoding="utf-8"
        )
        atomic_write_json(
            arm_dir / "llm_digest_input_v0413.json",
            llm_input(arm_payload, arm_digest),
        )
    payload = pair_evidence_payload(
        pair_id=pair_id,
        pair_status=pair_status,
        baseline=baseline,
        holdout=holdout,
        comparison=comparison,
    )
    digest = deterministic_pair_digest(payload)
    atomic_write_json(root / "task_evidence_units_v0413.json", payload)
    (root / "semantic_match_pair_digest_v0413.md").write_text(
        digest, encoding="utf-8"
    )
    atomic_write_json(
        root / "pair_llm_digest_input_v0413.json",
        llm_input(payload, digest),
    )
    return payload, digest
