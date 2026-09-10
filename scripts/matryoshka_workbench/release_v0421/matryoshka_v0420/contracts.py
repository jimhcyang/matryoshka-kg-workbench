"""Version, schema, output, digest, and controlled-pair contracts."""
from __future__ import annotations

import json
import hashlib
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .semantics import (
    atomic_write_json,
    clean_text,
    sha256_file,
    stable_hash,
    utc_now_iso,
    write_csv_with_schema,
)

RELEASE_VERSION = (
    "2026.08-parser-v0.4.15-matcher-v0.4.20-digest-v0.4.14-"
    "workbench-v3.19-automation-v1.0.18-requirements-v1.5.4"
)
PARSER_VERSION = "v0.4.15"
MATCHER_VERSION = "v0.4.20"
DIGEST_BUILDER_VERSION = "v0.4.14"
WORKBENCH_VERSION = "v3.19"
AUTOMATION_VERSION = "v1.0.18"
REQUIREMENTS_VERSION = (
    "Matryoshka Requirements v1.4 + Addenda v1.4.1, v1.4.2 Revision 1, "
    "Decision Profile Addendum v1.5, Independent Event Binding Addendum "
    "v1.5.2, Complete Event Identity Addendum v1.5.3, and Final Freeze "
    "Invariants Addendum v1.5.4"
)
# Stable v048 physical filenames remain readable for downstream compatibility.
# New target-evidence, digest, and pair contracts use the v0.4.13 schema.
OUTPUT_SCHEMA_VERSION = "matryoshka_output_schema_v0_4_13"
FINAL_MANIFEST_SCHEMA = "matryoshka_final_output_manifest_v0_4_20"
FINAL_MANIFEST_FILENAME = "final_output_manifest_v0420.json"

PAIR_STATUSES = {
    "CONTROLLED_PAIR",
    "PAIR_INCOMPLETE",
    "QUERY_MISMATCH_NOT_COMPARABLE",
    "CONFIGURATION_MISMATCH_NOT_COMPARABLE",
    "STANDALONE",
}

TASK_EVALUATION_COLUMNS = [
    "coordinated_release_version", "matcher_version", "requirements_version",
    "task", "target_id", "task_execution_status", "localization_status",
    "semantic_status", "raw_certification_status", "reconstructability_status",
    "pair_comparability_status", "selected_value_or_text",
    "selected_source_doc_id", "authoritative_source_doc_id",
    "top_candidate_doc_id",
    "top_related_evidence_doc_id", "top_background_doc_id",
    "primary_finding", "primary_finding_source_doc_id",
    "primary_finding_raw_certification_status", "primary_finding_score",
    "evidence_pattern", "side_a_support_source_doc_id",
    "contradiction_or_update_source_doc_id",
    "uncertainty_source_doc_id",
    "qualification_or_scope_source_doc_id",
    "related_different_event_source_doc_id",
    "background_only_source_doc_id", "evidence_roles_json",
    "best_score", "runner_up_score",
    "nearest_alternative_score", "nearest_alternative_doc_id",
    "nearest_alternative_passage_id", "nearest_alternative_eligible",
    "nearest_alternative_failed_gates",
    "decision_competitor_score", "decision_competitor_doc_id",
    "decision_competitor_passage_id", "decision_competitor_scope",
    "winner_margin", "margin_status", "decision_reason",
    "unresolved_reason", "failed_gates", "query_parse_sha256",
    "configuration_sha256", "pair_invariant_sha256",
]

PAIR_COMPARISON_COLUMNS = [
    "pair_id", "pair_status", "task", "target_id", "baseline_status",
    "baseline_value", "baseline_source_doc_id", "holdout_status",
    "holdout_value", "holdout_source_doc_id", "change",
    "baseline_authoritative_source_doc_id",
    "holdout_authoritative_source_doc_id",
    "baseline_primary_finding",
    "baseline_primary_finding_source_doc_id",
    "holdout_primary_finding",
    "holdout_primary_finding_source_doc_id",
    "removed_evidence_roles", "retained_evidence_roles",
    "added_evidence_roles",
    "authoritative_source_targeted_by_holdout",
    "primary_finding_source_targeted_by_holdout",
    "analyst_interpretation", "query_parse_sha256",
    "pair_invariant_sha256", "holdout_relevance_status",
    "excluded_doc_ids", "targeted_exclusion_ids",
    "pair_consistency_state", "reconciled_at",
]


def _finalizable_files(
    output_dir: Path,
    *,
    excluded_names: Sequence[str] = (),
) -> List[Path]:
    excluded = set(excluded_names) | {
        FINAL_MANIFEST_FILENAME,
        "final_output_manifest_validation_v0420.json",
    }
    return sorted(
        path for path in output_dir.rglob("*")
        if path.is_file()
        and path.relative_to(output_dir).as_posix() not in excluded
        and not any(part.startswith(".") for part in path.parts)
    )


def finalize_output_manifest(
    output_dir: Path,
    *,
    excluded_names: Sequence[str] = (),
) -> Dict[str, Any]:
    """Seal all final bytes after every post-processing mutation is complete."""
    root = output_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            "Output directory not found for finalization: " + str(root)
        )
    entries = []
    for path in _finalizable_files(
        root, excluded_names=excluded_names
    ):
        relative = path.relative_to(root).as_posix()
        entries.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "schema_version": FINAL_MANIFEST_SCHEMA,
        "generated_at": utc_now_iso(),
        "finalization_contract": (
            "all_mutations_complete_before_final_byte_authentication"
        ),
        "file_count": len(entries),
        "files": entries,
        "inventory_sha256": stable_hash(entries),
    }
    atomic_write_json(root / FINAL_MANIFEST_FILENAME, manifest)
    return manifest


def validate_output_manifest(
    output_dir: Path,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate existence, byte size, and SHA-256 for a sealed directory."""
    root = output_dir.expanduser().resolve()
    payload = (
        dict(manifest)
        if manifest is not None
        else safe_json(root / FINAL_MANIFEST_FILENAME)
    )
    missing: List[str] = []
    mismatched: List[str] = []
    checked = 0
    for item in payload.get("files", []):
        relative = clean_text(item.get("path", ""))
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            mismatched.append(relative or "<invalid-path>")
            continue
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        checked += 1
        expected_size = int(item.get("size_bytes", -1))
        expected_hash = clean_text(item.get("sha256", ""))
        if (
            path.stat().st_size != expected_size
            or sha256_file(path) != expected_hash
        ):
            mismatched.append(relative)
    declared_paths = {
        clean_text(item.get("path", ""))
        for item in payload.get("files", [])
        if clean_text(item.get("path", ""))
    }
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in _finalizable_files(root)
    }
    unexpected = sorted(actual_paths - declared_paths)
    result = {
        "schema_version":
            "matryoshka_final_output_validation_v0_4_17",
        "state": (
            "PASS" if payload.get("schema_version")
            == FINAL_MANIFEST_SCHEMA
            and not missing and not mismatched and not unexpected
            else "FAIL"
        ),
        "declared_file_count": len(payload.get("files", [])),
        "checked_file_count": checked,
        "missing_files": sorted(missing),
        "mismatched_files": sorted(mismatched),
        "unexpected_files": unexpected,
        "validated_at": utc_now_iso(),
    }
    return result


def validate_zip_against_output_manifest(
    zip_path: Path,
    run_root: Path,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Revalidate sealed bytes inside a completed run ZIP."""
    root = run_root.expanduser().resolve()
    payload = (
        dict(manifest)
        if manifest is not None
        else safe_json(root / FINAL_MANIFEST_FILENAME)
    )
    prefix = root.name.rstrip("/") + "/"
    missing: List[str] = []
    mismatched: List[str] = []
    unsafe: List[str] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = set(archive.namelist())
        for name in names:
            parts = Path(name).parts
            if Path(name).is_absolute() or ".." in parts:
                unsafe.append(name)
        for item in payload.get("files", []):
            relative = clean_text(item.get("path", ""))
            member = prefix + relative
            if member not in names:
                missing.append(relative)
                continue
            data = archive.read(member)
            if (
                len(data) != int(item.get("size_bytes", -1))
                or hashlib.sha256(data).hexdigest()
                != clean_text(item.get("sha256", ""))
            ):
                mismatched.append(relative)
        if prefix + FINAL_MANIFEST_FILENAME not in names:
            missing.append(FINAL_MANIFEST_FILENAME)
    return {
        "schema_version":
            "matryoshka_zip_manifest_validation_v0_4_17",
        "state": (
            "PASS"
            if payload.get("schema_version") == FINAL_MANIFEST_SCHEMA
            and not missing and not mismatched and not unsafe
            else "FAIL"
        ),
        "missing_files": sorted(missing),
        "mismatched_files": sorted(mismatched),
        "unsafe_members": sorted(unsafe),
        "validated_at": utc_now_iso(),
    }


def refresh_logical_output_manifest(output_dir: Path) -> Dict[str, Any]:
    """Refresh a compatibility logical manifest from current physical bytes."""
    root = output_dir.expanduser().resolve()
    path = root / "logical_output_manifest_v048.json"
    payload = safe_json(path)
    for entry in payload.get("outputs", {}).values():
        if not isinstance(entry, dict):
            continue
        filename = clean_text(entry.get("physical_filename", ""))
        physical = root / filename
        entry.update({
            "exists": physical.is_file(),
            "size_bytes": (
                physical.stat().st_size if physical.is_file() else 0
            ),
            "sha256": (
                sha256_file(physical) if physical.is_file() else ""
            ),
        })
    payload["generated_at"] = utc_now_iso()
    payload["refresh_contract"] = (
        "all_declared_physical_artifacts_rehashed_after_postprocessing"
    )
    atomic_write_json(path, payload)
    return payload


def safe_csv(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(columns or []))
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        frame = pd.DataFrame(columns=list(columns or []))
    for column in columns or []:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def safe_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def configuration_hash(configuration: Mapping[str, Any]) -> str:
    return stable_hash(dict(sorted(configuration.items())))


def pair_invariant_configuration(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    ignored = {
        "exclude_doc_ids", "exclude_doc_ids_file", "target_doc_ids",
        "target_doc_policy", "pair_role", "paired_run_dir", "output_dir",
    }
    return {
        key: value for key, value in sorted(configuration.items())
        if key not in ignored
    }


def pair_invariant_hash(configuration: Mapping[str, Any]) -> str:
    return stable_hash(pair_invariant_configuration(configuration))


def task_row(
    task: str,
    target_id: str,
    semantic_status: str,
    *,
    localization_status: str = "NO_FINAL_SOURCE",
    raw_status: str = "RAW_NOT_ASSESSED",
    reconstructability: str = "NOT_APPLICABLE",
    selected_text: str = "",
    selected_source: str = "",
    top_candidate: str = "",
    related_source: str = "",
    background_source: str = "",
    primary_finding: str = "",
    primary_finding_source: str = "",
    primary_finding_raw_status: str = "",
    primary_finding_score: Any = "",
    evidence_pattern: str = "",
    side_a_source: str = "",
    contradiction_or_update_source: str = "",
    uncertainty_source: str = "",
    qualification_or_scope_source: str = "",
    related_different_event_source: str = "",
    background_only_source: str = "",
    evidence_roles_json: str = "",
    best_score: Any = "",
    runner_up: Any = "",
    margin: Any = None,
    nearest_alternative_score: Any = None,
    nearest_alternative_doc_id: str = "",
    nearest_alternative_passage_id: str = "",
    nearest_alternative_eligible: Any = None,
    nearest_alternative_failed_gates: str = "",
    decision_competitor_score: Any = None,
    decision_competitor_doc_id: str = "",
    decision_competitor_passage_id: str = "",
    decision_competitor_scope: str = "",
    margin_status: str = "",
    reason: str = "",
    unresolved: str = "",
    failed: str = "",
) -> Dict[str, Any]:
    authoritative_source = (
        primary_finding_source
        if task == "contradictions"
        else selected_source
    )
    competitor = (
        decision_competitor_score
        if decision_competitor_score not in ("", None)
        else runner_up if runner_up not in ("", None) else None
    )
    if competitor is None:
        margin = None
        margin_status = (
            margin_status
            or "NOT_APPLICABLE_NO_COMPARABLE_COMPETITOR"
        )
    else:
        margin_status = margin_status or "APPLICABLE_COMPARABLE_COMPETITOR"
    return {
        "coordinated_release_version": RELEASE_VERSION,
        "matcher_version": MATCHER_VERSION,
        "requirements_version": REQUIREMENTS_VERSION,
        "task": task,
        "target_id": target_id,
        "task_execution_status": "EXECUTED",
        "localization_status": localization_status,
        "semantic_status": semantic_status,
        "raw_certification_status": raw_status,
        "reconstructability_status": reconstructability,
        "pair_comparability_status": "STANDALONE",
        "selected_value_or_text": selected_text,
        "selected_source_doc_id": selected_source,
        "authoritative_source_doc_id": authoritative_source,
        "top_candidate_doc_id": top_candidate,
        "top_related_evidence_doc_id": related_source,
        "top_background_doc_id": background_source,
        "primary_finding": primary_finding,
        "primary_finding_source_doc_id": primary_finding_source,
        "primary_finding_raw_certification_status":
            primary_finding_raw_status,
        "primary_finding_score": primary_finding_score,
        "evidence_pattern": evidence_pattern,
        "side_a_support_source_doc_id": side_a_source,
        "contradiction_or_update_source_doc_id":
            contradiction_or_update_source,
        "uncertainty_source_doc_id": uncertainty_source,
        "qualification_or_scope_source_doc_id":
            qualification_or_scope_source,
        "related_different_event_source_doc_id":
            related_different_event_source,
        "background_only_source_doc_id": background_only_source,
        "evidence_roles_json": evidence_roles_json,
        "best_score": best_score,
        "runner_up_score": competitor,
        "nearest_alternative_score": nearest_alternative_score,
        "nearest_alternative_doc_id": nearest_alternative_doc_id,
        "nearest_alternative_passage_id": nearest_alternative_passage_id,
        "nearest_alternative_eligible": nearest_alternative_eligible,
        "nearest_alternative_failed_gates":
            nearest_alternative_failed_gates,
        "decision_competitor_score": competitor,
        "decision_competitor_doc_id": decision_competitor_doc_id,
        "decision_competitor_passage_id": decision_competitor_passage_id,
        "decision_competitor_scope": (
            decision_competitor_scope
            or (
                "distinct_eligible_candidate_same_task_decision"
                if competitor is not None else
                "no_distinct_comparable_candidate"
            )
        ),
        "winner_margin": margin,
        "margin_status": margin_status,
        "decision_reason": reason,
        "unresolved_reason": unresolved,
        "failed_gates": failed,
        "query_parse_sha256": "",
        "configuration_sha256": "",
        "pair_invariant_sha256": "",
    }


def authoritative_digest(
    evaluation: pd.DataFrame,
    pair_role: str = "standalone",
    pair_status: str = "STANDALONE",
) -> str:
    lines = [
        "# Matryoshka v0.4 Authoritative Task Digest",
        "",
        f"- Matcher: `{MATCHER_VERSION}`",
        f"- Requirements: `{REQUIREMENTS_VERSION}`",
        f"- Pair role: `{pair_role}`",
        f"- Pair status: `{pair_status}`",
        "",
    ]
    for _, row in evaluation.iterrows():
        lines.extend([
            f"## TARGET {clean_text(row.get('target_id', ''))}",
            f"- Task: `{clean_text(row.get('task', ''))}`",
            f"- Semantic status: `{clean_text(row.get('semantic_status', ''))}`",
            f"- Selected value/text: {clean_text(row.get('selected_value_or_text', '')) or 'not selected'}",
            f"- Selected source: `{clean_text(row.get('selected_source_doc_id', '')) or 'not selected'}`",
            f"- Raw certification: `{clean_text(row.get('raw_certification_status', ''))}`",
            f"- Reason: {clean_text(row.get('decision_reason', '')) or clean_text(row.get('unresolved_reason', ''))}",
            "",
        ])
        if clean_text(row.get("task", "")) == "contradictions":
            lines.extend([
                f"- Primary finding: `{clean_text(row.get('primary_finding', '')) or 'not selected'}`",
                f"- Primary finding source: `{clean_text(row.get('primary_finding_source_doc_id', '')) or 'not selected'}`",
                f"- Evidence pattern: `{clean_text(row.get('evidence_pattern', '')) or clean_text(row.get('semantic_status', ''))}`",
                f"- Side-A support source: `{clean_text(row.get('side_a_support_source_doc_id', '')) or 'not found'}`",
                f"- Contradiction/update source: `{clean_text(row.get('contradiction_or_update_source_doc_id', '')) or 'not found'}`",
                f"- Uncertainty source: `{clean_text(row.get('uncertainty_source_doc_id', '')) or 'not found'}`",
                f"- Qualification/scope source: `{clean_text(row.get('qualification_or_scope_source_doc_id', '')) or 'not found'}`",
                f"- Related-different-event source: `{clean_text(row.get('related_different_event_source_doc_id', '')) or 'not found'}`",
                "",
            ])
    lines.append(
        "Deterministic tables and this digest are authoritative; optional LLM prose cannot change them."
    )
    return "\n".join(lines)


def authoritative_pair_digest(
    baseline: pd.DataFrame,
    holdout: pd.DataFrame,
    comparison: pd.DataFrame,
    pair_status: str,
) -> str:
    """Produce one deterministic, pair-aware digest used by both sides."""
    lines = [
        "# Matryoshka v0.4.14 Authoritative Controlled-Pair Digest",
        "",
        f"- Matcher: `{MATCHER_VERSION}`",
        f"- Requirements: `{REQUIREMENTS_VERSION}`",
        f"- Final pair status: `{pair_status}`",
        "",
    ]
    for role, frame in (("Baseline", baseline), ("Holdout", holdout)):
        lines.append(f"## {role}")
        lines.append("")
        for _, row in frame.iterrows():
            target = clean_text(row.get("target_id", ""))
            lines.extend([
                f"### {role} TARGET {target}",
                f"- Task: `{clean_text(row.get('task', ''))}`",
                f"- Semantic status: `{clean_text(row.get('semantic_status', ''))}`",
                f"- Selected value/text: {clean_text(row.get('selected_value_or_text', '')) or 'not selected'}",
                f"- Selected source: `{clean_text(row.get('selected_source_doc_id', '')) or 'not selected'}`",
                f"- Primary finding: `{clean_text(row.get('primary_finding', '')) or 'not selected'}`",
                f"- Primary finding source: `{clean_text(row.get('primary_finding_source_doc_id', '')) or 'not selected'}`",
                f"- Raw certification: `{clean_text(row.get('raw_certification_status', ''))}`",
                f"- Unresolved reason: {clean_text(row.get('unresolved_reason', '')) or 'none'}",
                "",
            ])
    lines.extend(["## Pair transitions", ""])
    for _, row in comparison.iterrows():
        lines.extend([
            f"### TARGET {clean_text(row.get('target_id', ''))}",
            f"- Change: `{clean_text(row.get('change', ''))}`",
            f"- Baseline primary finding: `{clean_text(row.get('baseline_primary_finding', '')) or 'not selected'}`",
            f"- Baseline primary source: `{clean_text(row.get('baseline_primary_finding_source_doc_id', '')) or 'not selected'}`",
            f"- Holdout primary finding: `{clean_text(row.get('holdout_primary_finding', '')) or 'not selected'}`",
            f"- Holdout primary source: `{clean_text(row.get('holdout_primary_finding_source_doc_id', '')) or 'not selected'}`",
            f"- Removed evidence roles: `{clean_text(row.get('removed_evidence_roles', '')) or 'none'}`",
            f"- Retained evidence roles: `{clean_text(row.get('retained_evidence_roles', '')) or 'none'}`",
            f"- Added evidence roles: `{clean_text(row.get('added_evidence_roles', '')) or 'none'}`",
            f"- Interpretation: {clean_text(row.get('analyst_interpretation', ''))}",
            "",
        ])
    lines.append(
        "Deterministic tables and this digest are authoritative; optional LLM prose cannot change them."
    )
    return "\n".join(lines)


def validate_llm_digest(
    text: str,
    evaluation: pd.DataFrame,
    request_state: str,
    model: str = "",
    error_type: str = "",
) -> Dict[str, Any]:
    reasons: List[str] = []
    if request_state in {
        "NOT_REQUESTED", "API_KEY_UNAVAILABLE", "CALL_FAILED",
        "AUTHENTICATION_FAILED",
    }:
        state = request_state
    elif not clean_text(text):
        state = "EMPTY_RESPONSE"
        reasons.append("empty_response")
    else:
        lower = text.lower()
        for _, row in evaluation.iterrows():
            target = clean_text(row.get("target_id", ""))
            status = clean_text(row.get("semantic_status", ""))
            if f"target {target}".lower() not in lower:
                reasons.append("missing_target:" + target)
            if status and status.lower() not in lower and status.lower().replace("_", " ") not in lower:
                reasons.append("missing_status:" + target)
            for field, label in (
                ("selected_value_or_text", "selected_value"),
                ("selected_source_doc_id", "selected_source"),
                ("raw_certification_status", "raw_status"),
                ("unresolved_reason", "unresolved_reason"),
            ):
                expected = clean_text(row.get(field, ""))
                if (
                    expected
                    and expected.lower() not in lower
                    and expected.lower().replace("_", " ") not in lower
                ):
                    reasons.append(f"missing_{label}:" + target)
        state = "VALIDATED" if not reasons else "REJECTED"
    return {
        "state": state,
        "request_state": request_state,
        "model": model,
        "error_type": error_type,
        "retryable": state in {
            "CALL_FAILED", "API_KEY_UNAVAILABLE", "EMPTY_RESPONSE"
        },
        "validation_result": "PASS" if state == "VALIDATED" else "NOT_ACCEPTED",
        "reasons": reasons,
        "response_sha256": stable_hash(text) if text else "",
        "attempt_timestamp": utc_now_iso(),
        "authoritative_fallback": "deterministic_digest",
    }


def _pair_authoritative_source(row: Mapping[str, Any]) -> str:
    """Return the source that owns the task's authoritative finding.

    Contradiction outputs may intentionally have no Side-A selected source.
    Their primary finding source is therefore authoritative for controlled-pair
    reconciliation.  Other tasks retain the selected-source contract.
    """
    canonical = clean_text(row.get("authoritative_source_doc_id", ""))
    if canonical:
        return canonical
    task = clean_text(row.get("task", "")).lower()
    if task == "contradictions":
        return clean_text(
            row.get("primary_finding_source_doc_id", "")
        ) or clean_text(row.get("selected_source_doc_id", ""))
    return clean_text(row.get("selected_source_doc_id", ""))


def _evidence_role_sources(row: Mapping[str, Any]) -> Dict[str, str]:
    """Read the normalized role-to-source map from one evaluation row."""
    raw = row.get("evidence_roles_json", "")
    try:
        decoded = json.loads(raw) if clean_text(raw) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded = {}
    if not isinstance(decoded, dict):
        return {}
    roles: Dict[str, str] = {}
    for role, payload in decoded.items():
        if not isinstance(payload, dict):
            continue
        source = clean_text(payload.get("source_doc_id", "")).upper()
        if source:
            roles[clean_text(role).lower()] = source
    return roles


def _authoritative_sources(row: Mapping[str, Any]) -> set[str]:
    """Return every source exposed by an authoritative evidence role."""
    sources = {
        _pair_authoritative_source(row).upper(),
        clean_text(row.get("selected_source_doc_id", "")).upper(),
        clean_text(row.get("primary_finding_source_doc_id", "")).upper(),
    }
    sources.update(_evidence_role_sources(row).values())
    return {source for source in sources if source}


def _pair_change(
    baseline: Mapping[str, Any],
    holdout: Mapping[str, Any],
) -> Tuple[str, str]:
    baseline_status = clean_text(baseline.get("semantic_status", ""))
    holdout_status = clean_text(holdout.get("semantic_status", ""))
    unresolved = any(
        token in holdout_status
        for token in ("UNRESOLVED", "NOT_RECONSTRUCTABLE", "NO_DIRECT", "NO_SUPPORTED", "BLOCKED")
    )
    if baseline_status != holdout_status and unresolved:
        return "RESOLUTION_LOSS", "Declared-source removal caused loss of authoritative resolution."
    baseline_source = _pair_authoritative_source(baseline)
    holdout_source = _pair_authoritative_source(holdout)
    task = clean_text(baseline.get("task", "")).lower()
    if task == "contradictions" and baseline_status != holdout_status:
        return (
            "EVIDENCE_PATTERN_CHANGE",
            f"Semantic evidence pattern changed from {baseline_status} to "
            f"{holdout_status}.",
        )
    if baseline_source and holdout_source and baseline_source != holdout_source:
        return "SOURCE_SUBSTITUTION", "Holdout selected a different source."
    if clean_text(baseline.get("selected_value_or_text", "")) != clean_text(
        holdout.get("selected_value_or_text", "")
    ):
        return "VALUE_CHANGE", "Selected authoritative value/text changed."
    if baseline_status != holdout_status:
        # A semantic transition is material even when both sides still expose
        # related evidence.  Contradiction pairs commonly move from support +
        # update to opposition-only after the supporting source is excluded.
        if baseline_source and not holdout_source:
            return (
                "SUPPORT_REMOVED_OPPOSITION_REMAINS",
                "The selected support source was removed while opposing or "
                "updating evidence remains.",
            )
        return (
            "EVIDENCE_PATTERN_CHANGE",
            f"Semantic evidence pattern changed from {baseline_status} to "
            f"{holdout_status}.",
        )
    try:
        if float(holdout.get("best_score", 0) or 0) < float(baseline.get("best_score", 0) or 0):
            return "CONFIDENCE_LOSS", "Task-local score decreased after exclusion."
    except (TypeError, ValueError):
        pass
    return "NO_MATERIAL_CHANGE", "No authoritative change detected."


def reconcile_pair(baseline_dir: Path, holdout_dir: Path, pair_id: str) -> Dict[str, Any]:
    baseline_manifest = safe_json(baseline_dir / "semantic_match_run_manifest_v048.json")
    holdout_manifest = safe_json(holdout_dir / "semantic_match_run_manifest_v048.json")
    if not baseline_manifest or not holdout_manifest:
        return {"pair_status": "PAIR_INCOMPLETE"}
    if baseline_manifest.get("query_parse_sha256") != holdout_manifest.get("query_parse_sha256"):
        pair_status = "QUERY_MISMATCH_NOT_COMPARABLE"
    elif baseline_manifest.get("pair_invariant_sha256") != holdout_manifest.get("pair_invariant_sha256"):
        pair_status = "CONFIGURATION_MISMATCH_NOT_COMPARABLE"
    else:
        pair_status = "CONTROLLED_PAIR"
    baseline = safe_csv(baseline_dir / "task_evaluation_v048.csv", TASK_EVALUATION_COLUMNS)
    holdout = safe_csv(holdout_dir / "task_evaluation_v048.csv", TASK_EVALUATION_COLUMNS)
    baseline_map = {
        (clean_text(row["task"]), clean_text(row["target_id"])): row
        for _, row in baseline.iterrows()
    }
    holdout_map = {
        (clean_text(row["task"]), clean_text(row["target_id"])): row
        for _, row in holdout.iterrows()
    }
    rows: List[Dict[str, Any]] = []
    excluded_doc_ids = {
        clean_text(value).upper()
        for value in holdout_manifest.get("effective_excluded_doc_ids", [])
        if clean_text(value)
    }
    for task, target in sorted(set(baseline_map) | set(holdout_map)):
        before = baseline_map.get((task, target), {})
        after = holdout_map.get((task, target), {})
        if pair_status == "CONTROLLED_PAIR":
            change, interpretation = _pair_change(before, after)
        else:
            change, interpretation = "NOT_COMPARABLE", "Controlled-pair contract failed: " + pair_status
        baseline_source = clean_text(
            _pair_authoritative_source(before)
        ).upper()
        holdout_source = clean_text(
            _pair_authoritative_source(after)
        ).upper()
        baseline_primary_source = clean_text(
            before.get("primary_finding_source_doc_id", "")
        ).upper()
        baseline_roles = _evidence_role_sources(before)
        holdout_roles = _evidence_role_sources(after)
        removed_roles = sorted(
            role for role, source in baseline_roles.items()
            if holdout_roles.get(role) != source
        )
        retained_roles = sorted(
            role for role, source in baseline_roles.items()
            if holdout_roles.get(role) == source
        )
        added_roles = sorted(
            role for role, source in holdout_roles.items()
            if baseline_roles.get(role) != source
        )
        authoritative_targeted = bool(
            baseline_source and baseline_source in excluded_doc_ids
        )
        primary_targeted: Any = (
            bool(
                baseline_primary_source
                and baseline_primary_source in excluded_doc_ids
            )
            if task == "contradictions"
            else None
        )
        leaked_sources = sorted(
            _authoritative_sources(after) & excluded_doc_ids
        )
        if leaked_sources:
            relevance = "EXCLUSION_LEAK"
            change = "INVALID_EXCLUDED_SOURCE_SELECTED"
            interpretation = (
                "Holdout retained an authoritative evidence source declared "
                "excluded; pair output must not be interpreted."
            )
        elif authoritative_targeted:
            relevance = "BASELINE_SOURCE_TARGETED"
        elif baseline_source:
            relevance = "BASELINE_SOURCE_NOT_EXCLUDED"
        else:
            relevance = "NO_BASELINE_AUTHORITATIVE_SOURCE"
        rows.append({
            "pair_id": pair_id,
            "pair_status": pair_status,
            "task": task,
            "target_id": target,
            "baseline_status": clean_text(before.get("semantic_status", "")),
            "baseline_value": clean_text(before.get("selected_value_or_text", "")),
            "baseline_source_doc_id": baseline_source,
            "holdout_status": clean_text(after.get("semantic_status", "")),
            "holdout_value": clean_text(after.get("selected_value_or_text", "")),
            "holdout_source_doc_id": holdout_source,
            "baseline_authoritative_source_doc_id": baseline_source,
            "holdout_authoritative_source_doc_id": holdout_source,
            "change": change,
            "baseline_primary_finding": clean_text(
                before.get("primary_finding", "")
            ),
            "baseline_primary_finding_source_doc_id":
                baseline_primary_source,
            "holdout_primary_finding": clean_text(
                after.get("primary_finding", "")
            ),
            "holdout_primary_finding_source_doc_id": clean_text(
                after.get("primary_finding_source_doc_id", "")
            ).upper(),
            "removed_evidence_roles": ";".join(removed_roles),
            "retained_evidence_roles": ";".join(retained_roles),
            "added_evidence_roles": ";".join(added_roles),
            "authoritative_source_targeted_by_holdout":
                authoritative_targeted,
            "primary_finding_source_targeted_by_holdout":
                primary_targeted,
            "analyst_interpretation": interpretation,
            "query_parse_sha256": clean_text(baseline_manifest.get("query_parse_sha256", "")),
            "pair_invariant_sha256": clean_text(baseline_manifest.get("pair_invariant_sha256", "")),
            "holdout_relevance_status": relevance,
            "excluded_doc_ids": ";".join(sorted(excluded_doc_ids)),
            "targeted_exclusion_ids": ";".join(
                sorted(
                    source for source in excluded_doc_ids
                    if source in _authoritative_sources(before)
                )
            ),
            "pair_consistency_state": (
                "PASS"
                if (
                    relevance != "BASELINE_SOURCE_TARGETED"
                    or authoritative_targeted
                )
                else "FAIL_AUTHORITY_TARGETING_DISAGREEMENT"
            ),
            "reconciled_at": utc_now_iso(),
        })
    comparison = pd.DataFrame(rows)
    pair_digest = authoritative_pair_digest(
        baseline, holdout, comparison, pair_status
    )
    pair_input = {
        "pair_id": pair_id,
        "pair_status": pair_status,
        "baseline": baseline.to_dict("records"),
        "holdout": holdout.to_dict("records"),
        "comparison": comparison.to_dict("records"),
    }
    task_specific_names = {
        "reconstruction": "reconstruction_evaluation_v048.csv",
        "redaction": "redaction_evaluation_v048.csv",
        "sentence_redaction": "sentence_redaction_evaluation_v048.csv",
        "claims": "claim_evaluation_v048.csv",
        "contradictions": "contradiction_evaluation_v048.csv",
    }
    for directory, role in ((baseline_dir, "baseline"), (holdout_dir, "holdout")):
        write_csv_with_schema(directory / "pair_comparison_v048.csv", comparison, PAIR_COMPARISON_COLUMNS)
        evaluation = safe_csv(directory / "task_evaluation_v048.csv", TASK_EVALUATION_COLUMNS)
        evaluation["pair_comparability_status"] = pair_status
        write_csv_with_schema(directory / "task_evaluation_v048.csv", evaluation, TASK_EVALUATION_COLUMNS)
        manifest = safe_json(directory / "semantic_match_run_manifest_v048.json")
        manifest["pair_status"] = pair_status
        atomic_write_json(directory / "semantic_match_run_manifest_v048.json", manifest)
        (directory / "semantic_match_task_digest_v048.md").write_text(
            pair_digest, encoding="utf-8"
        )
        task_name = clean_text(manifest.get("task_mode", ""))
        task_filename = task_specific_names.get(task_name, "")
        if task_filename:
            write_csv_with_schema(
                directory / task_filename,
                evaluation,
                TASK_EVALUATION_COLUMNS,
            )
        contract_path = directory / "matcher_output_contract_v048.json"
        contract = safe_json(contract_path)
        contract["pair_status"] = pair_status
        contract["pair_id"] = pair_id
        atomic_write_json(contract_path, contract)
        atomic_write_json(
            directory / "llm_digest_input_v048.json",
            {**pair_input, "view_role": role},
        )
        logical_path = directory / "logical_output_manifest_v048.json"
        logical = safe_json(logical_path)
        for entry in logical.get("outputs", {}).values():
            if not isinstance(entry, dict):
                continue
            physical = directory / clean_text(
                entry.get("physical_filename", "")
            )
            entry["exists"] = physical.is_file()
            entry["size_bytes"] = (
                physical.stat().st_size if physical.is_file() else 0
            )
            entry["sha256"] = (
                sha256_file(physical) if physical.is_file() else ""
            )
        logical["pair_status"] = pair_status
        logical["pair_id"] = pair_id
        atomic_write_json(logical_path, logical)
        manifest = safe_json(directory / "semantic_match_run_manifest_v048.json")
        manifest["output_inventory"] = output_inventory(directory)
        atomic_write_json(
            directory / "semantic_match_run_manifest_v048.json", manifest
        )
    return {"pair_status": pair_status, "row_count": len(rows)}


def output_inventory(output_dir: Path) -> Dict[str, Any]:
    return {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
        and path.name not in {
            "semantic_match_run_manifest_v048.json",
            "logical_output_manifest_v048.json",
        }
    }
