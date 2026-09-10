#!/usr/bin/env python3
"""Matryoshka Semantic Evidence Workbench v3.19."""
from __future__ import annotations

import json
import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from matryoshka_v0420.workbench_runtime import (
    collision_safe_run_name,
    resolve_credential_source,
    summarize_llm_validation_failures,
    terminal_status_for_exited_process,
)
from matryoshka_v0420.contracts import (
    validate_output_manifest,
    validate_zip_against_output_manifest,
)
from matryoshka_v0420.decision_profile import operational_band

APP_VERSION = "v3.19"
ROOT = Path(__file__).resolve().parent


def shared_workbench_cache_root(base: Path) -> Path:
    override = os.environ.get("MATRYOSHKA_CACHE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    resolved = base.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "matryoshka_workbench":
            return candidate / ".matryoshka_cache"
    return resolved / ".matryoshka_cache"


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def safe_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def safe_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, nrows=nrows)
    except Exception:
        return pd.DataFrame()


def safe_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    except Exception:
        pass
    return pd.DataFrame(rows)


def safe_path(value: str, label: str, must_exist: bool = False) -> Tuple[Optional[Path], str]:
    text = str(value or "").strip().strip("\"'")
    if not text:
        return None, label + " is empty."
    if len(text) > 4096 or any(
        len(part.encode("utf-8")) > 240 for part in Path(text).parts
    ):
        return None, label + " is not a valid filesystem path; query text may have been pasted here."
    try:
        path = Path(text).expanduser().resolve()
    except OSError as exc:
        return None, label + " could not be resolved: " + str(exc)
    if must_exist and not path.is_dir():
        return None, label + " is not an existing directory: " + str(path)
    return path, ""


def python_environment_preflight(
    python_executable: str,
    code_dir: Path,
) -> Tuple[Dict[str, Any], str]:
    """Validate the exact interpreter used by child processes."""
    coordinator = code_dir / "run_coordinated_v0_4_21.py"
    if not coordinator.is_file():
        return {}, "Coordinator is missing: " + str(coordinator)
    try:
        completed = subprocess.run(
            [
                python_executable, "-c",
                (
                    "import json,sys,pandas,numpy;"
                    "print(json.dumps({'python':sys.version.split()[0],"
                    "'pandas':pandas.__version__,'numpy':numpy.__version__}))"
                ),
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, "Python preflight could not run: " + clean_text(exc)
    if completed.returncode:
        return {}, (
            "Selected Python is missing required packages. Install with: "
            f"{python_executable} -m pip install -r {code_dir / 'requirements.txt'}"
            "\n\n" + clean_text(completed.stderr or completed.stdout)
        )
    try:
        versions = json.loads(completed.stdout)
        major, minor = (
            int(value) for value in versions["python"].split(".")[:2]
        )
    except (ValueError, KeyError, json.JSONDecodeError):
        return {}, "Python preflight returned an unreadable version response."
    if (major, minor) < (3, 10):
        return versions, "Python 3.10 or newer is required."
    return versions, ""


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def zip_tree_to_file(root: Path) -> Path:
    manifest_path = root / "final_output_manifest_v0420.json"
    if manifest_path.is_file():
        directory_validation = validate_output_manifest(root)
        if directory_validation["state"] != "PASS":
            raise RuntimeError(
                "ZIP blocked: final output manifest does not match "
                "current run bytes."
            )
    descriptor, name = tempfile.mkstemp(
        prefix=root.name + "_", suffix=".zip"
    )
    os.close(descriptor)
    target = Path(name)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root.parent))
    if manifest_path.is_file():
        zip_validation = validate_zip_against_output_manifest(
            target, root
        )
        if zip_validation["state"] != "PASS":
            target.unlink(missing_ok=True)
            raise RuntimeError(
                "ZIP blocked: archived bytes failed final manifest "
                "validation."
            )
    return target


def complete_run_root(output_dir: Path) -> Path:
    """Return the coordinated root for single or baseline/holdout output."""
    parent = output_dir.parent
    return parent.parent if parent.name in {"baseline", "holdout"} else parent


def complete_run_self_check(run_root: Path) -> List[str]:
    """List missing artifacts that make a run ZIP incomplete."""
    missing: List[str] = []
    required_root = [
        "query_parse/query_parse_manifest.json",
        "query_parse/query_fragments.csv",
        "coordinated_status_v048.json",
        "coordinated_release_manifest_v048.json",
    ]
    for relative in required_root:
        if not (run_root / relative).is_file():
            missing.append(relative)
    outputs = running_outputs(run_root)
    if not outputs:
        missing.append("semantic_match_output")
    for output in outputs:
        for name in (
            "task_evaluation_v048.csv",
            "semantic_match_run_manifest_v048.json",
            "task_evidence_units_v0413.json",
            "semantic_match_task_digest_v0413.md",
            "llm_digest_state_v0413.json",
            "task_visualization_v048.html",
            "task_visualization_data_v048.json",
        ):
            if not (output / name).is_file():
                missing.append(str((output / name).relative_to(run_root)))
    if len(outputs) == 2:
        for output in outputs:
            if not (output / "pair_comparison_v048.csv").is_file():
                missing.append(str(
                    (output / "pair_comparison_v048.csv").relative_to(run_root)
                ))
    return missing


def show_table(frame: pd.DataFrame, key: str, limit: int = 100) -> None:
    if frame.empty:
        st.info("No rows.")
        return
    st.dataframe(frame.head(limit), width="stretch", hide_index=True)
    if len(frame) > limit:
        st.caption(f"Showing {limit:,} of {len(frame):,} rows.")


def show_csv_page(
    path: Path,
    key: str,
    *,
    page_size: int = 100,
) -> None:
    """Display a bounded CSV page without loading a large audit into memory."""
    if not path.is_file() or path.stat().st_size == 0:
        st.info("No rows.")
        return
    size_mb = path.stat().st_size / (1024 * 1024)
    st.caption(
        f"Non-authoritative audit · {size_mb:,.1f} MB · "
        f"loaded in bounded pages of {page_size:,} rows"
    )
    page = int(st.number_input(
        "Audit page",
        min_value=1,
        value=1,
        step=1,
        key=key + "_page",
    ))
    start = (page - 1) * page_size
    try:
        frame = pd.read_csv(
            path,
            low_memory=False,
            skiprows=range(1, start + 1) if start else None,
            nrows=page_size,
        )
    except Exception as exc:
        st.error("Could not read this audit page: " + clean_text(exc))
        return
    show_table(frame, key + "_rows", page_size)
    if len(frame) < page_size:
        st.caption("End of audit.")


def show_grouped_csv_page(
    path: Path,
    key: str,
    target_column: str,
    *,
    page_size: int = 100,
) -> None:
    """Page a large audit within one analyst-selected target."""
    if not path.is_file() or path.stat().st_size == 0:
        st.info("No rows.")
        return
    targets: List[str] = []
    try:
        for chunk in pd.read_csv(
            path, usecols=[target_column], chunksize=50000,
            low_memory=False,
        ):
            targets.extend(
                clean_text(value)
                for value in chunk[target_column].dropna().unique()
                if clean_text(value)
            )
    except Exception:
        show_csv_page(path, key, page_size=page_size)
        return
    targets = sorted(set(targets))
    if not targets:
        show_csv_page(path, key, page_size=page_size)
        return
    selected = st.selectbox(
        "Target", targets, key=key + "_target"
    )
    page = int(st.number_input(
        "Target audit page", min_value=1, value=1, step=1,
        key=key + "_page",
    ))
    start = (page - 1) * page_size
    pieces: List[pd.DataFrame] = []
    matched = 0
    for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
        subset = chunk[
            chunk[target_column].fillna("").astype(str).eq(selected)
        ]
        if subset.empty:
            continue
        if matched + len(subset) <= start:
            matched += len(subset)
            continue
        local_start = max(0, start - matched)
        pieces.append(subset.iloc[local_start:])
        matched += len(subset)
        if sum(len(piece) for piece in pieces) >= page_size:
            break
    frame = (
        pd.concat(pieces, ignore_index=True).head(page_size)
        if pieces else pd.DataFrame()
    )
    st.caption(
        f"Target-balanced audit view · {selected} · "
        f"{path.stat().st_size / (1024 * 1024):,.1f} MB"
    )
    show_table(frame, key + "_rows", page_size)


def balanced_csv_sample(
    path: Path,
    target_column: str,
    per_target: int = 30,
) -> pd.DataFrame:
    """Return a bounded sample for every target, not global first-N rows."""
    samples: Dict[str, List[pd.DataFrame]] = {}
    counts: Dict[str, int] = {}
    try:
        for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
            if target_column not in chunk.columns:
                return chunk.head(per_target)
            for target, subset in chunk.groupby(target_column, dropna=False):
                key = clean_text(target)
                remaining = per_target - counts.get(key, 0)
                if remaining <= 0:
                    continue
                samples.setdefault(key, []).append(subset.head(remaining))
                counts[key] = counts.get(key, 0) + min(remaining, len(subset))
    except Exception:
        return pd.DataFrame()
    pieces = [
        piece for target in sorted(samples)
        for piece in samples[target]
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def credential_available(
    source: str,
    environment_name: str,
    session_key: str,
) -> bool:
    return bool(
        credential_source_status(
            source, environment_name, session_key
        )["populated"]
    )


def streamlit_secret_value(name: str) -> str:
    try:
        return clean_text(st.secrets.get(name, ""))
    except (AttributeError, KeyError, RuntimeError, FileNotFoundError):
        return ""


def credential_source_status(
    source: str,
    environment_name: str,
    session_key: str,
) -> Dict[str, Any]:
    """Resolve one explicit, session-scoped credential source."""
    name = clean_text(environment_name) or "OPENAI_API_KEY"
    return resolve_credential_source(
        source,
        name,
        session_key,
        environment=os.environ,
        streamlit_secrets={name: streamlit_secret_value(name)},
    )


def resolve_output(path: Path) -> Path:
    if (path / "task_evaluation_v048.csv").exists():
        return path
    baseline = path / "baseline" / "semantic_match_output"
    holdout = path / "holdout" / "semantic_match_output"
    if baseline.is_dir():
        return baseline
    if holdout.is_dir():
        return holdout
    nested = path / "semantic_match_output"
    return nested if nested.is_dir() else path


def infer_case_id_from_run_root(run_root: Path) -> str:
    name = run_root.name
    if "_" not in name:
        return name
    prefix, suffix = name.split("_", 1)
    return suffix if prefix.isdigit() else name


def load_case_metadata(run_root: Path) -> Dict[str, Any]:
    sidecar = safe_json(run_root / "case_metadata_v1.json")
    if sidecar:
        return sidecar
    query_text = clean_text((run_root / "query_input.txt").read_text(
        encoding="utf-8", errors="ignore"
    )) if (run_root / "query_input.txt").is_file() else ""
    experiment_root = run_root.parent.parent if run_root.parent.name == "runs" else None
    manifest = (
        safe_csv(experiment_root / "query_manifests" / "case_manifest.csv")
        if experiment_root else pd.DataFrame()
    )
    case_id = infer_case_id_from_run_root(run_root)
    if not manifest.empty and "case_id" in manifest.columns:
        subset = manifest[manifest["case_id"].astype(str) == case_id]
        if not subset.empty:
            row = subset.iloc[0].to_dict()
            query_rel = clean_text(row.get("query_file", ""))
            query_path = experiment_root / query_rel if query_rel else None
            if not query_text and query_path and query_path.is_file():
                query_text = clean_text(query_path.read_text(encoding="utf-8", errors="ignore"))
            return {
                "schema_version": "kg_demo_case_metadata_v1",
                "case_id": case_id,
                "legacy_case_id": clean_text(row.get("legacy_case_id", "")),
                "corpus": clean_text(row.get("corpus", "")),
                "task_family": clean_text(row.get("task_family", "")),
                "runner_task": clean_text(row.get("runner_task", "")),
                "doc_id": clean_text(row.get("doc_id", "")),
                "paragraph_id": clean_text(row.get("paragraph_id", "")),
                "query_kind": clean_text(row.get("query_kind", "")),
                "ground_truth": clean_text(row.get("ground_truth", "")),
                "source_context": clean_text(row.get("source_context", "")),
                "notes": clean_text(row.get("notes", "")),
                "query_text": query_text,
            }
    return {"query_text": query_text, "case_id": case_id}


def show_evidence_cards(
    evidence: Dict[str, Any],
    key_prefix: str,
) -> None:
    """Render one complete, bounded evidence card per requested target."""
    if not evidence:
        st.info("No v0.4.13 evidence-unit payload was found.")
        return
    requested = int(evidence.get("requested_target_count", 0) or 0)
    emitted = int(evidence.get("emitted_target_count", 0) or 0)
    coverage = clean_text(evidence.get("target_coverage_state", "UNKNOWN"))
    columns = st.columns(3)
    columns[0].metric("Requested targets", requested)
    columns[1].metric("Rendered targets", emitted)
    columns[2].metric("Coverage", coverage)
    if coverage != "PASS":
        st.error(
            "Target coverage failed. Missing: "
            + ", ".join(evidence.get("missing_target_ids", []))
        )
    for position, unit in enumerate(evidence.get("units", []), start=1):
        target_id = clean_text(unit.get("target_id", ""))
        target = unit.get("target", {})
        finding = unit.get("deterministic_finding", {})
        status = clean_text(finding.get("semantic_status", "NO_STATUS"))
        with st.expander(
            f"{position}. {target_id} · {status}",
            expanded=position == 1,
        ):
            st.markdown("#### Result")
            selected = clean_text(
                finding.get("selected_value_or_text", "")
            )
            source = clean_text(
                finding.get("selected_source_doc_id", "")
            )
            st.write(
                f"**{status}**"
                + (f" · **Selected:** {selected}" if selected else "")
                + (f" · **Source:** `{source}`" if source else "")
            )


            st.markdown("#### Target context")
            st.write(
                clean_text(target.get("target_text", ""))
                or (
                    clean_text(target.get("left_context", ""))
                    + " [TARGET] "
                    + clean_text(target.get("right_context", ""))
                )
                or "No bounded target context was available."
            )
            st.markdown("#### Why")
            st.write(
                clean_text(finding.get("decision_reason", ""))
                or clean_text(finding.get("unresolved_reason", ""))
                or "No additional deterministic reason was recorded."
            )
            st.markdown("#### Evidence fragments")
            fragments = unit.get("evidence_fragments", [])
            if not fragments:
                st.info("No bounded evidence fragment was available.")
            for fragment_number, fragment in enumerate(
                fragments, start=1
            ):
                role = clean_text(fragment.get("evidence_role", "evidence"))
                fragment_source = clean_text(
                    fragment.get("source_doc_id", "")
                )
                passage = clean_text(fragment.get("passage_id", ""))
                st.markdown(
                    f"**{fragment_number}. {role.replace('_', ' ').title()}** "
                    f"· source `{fragment_source or 'unknown'}` "
                    f"· passage `{passage or 'unknown'}`"
                )
                st.write(
                    clean_text(fragment.get("excerpt", ""))
                    or "No bounded excerpt was available."
                )
                context = fragment.get("context", {})
                if any(clean_text(context.get(key, "")) for key in (
                    "previous", "selected", "next", "pseudo_excerpt"
                )):
                    with st.expander(
                        "Raw/pseudo context", expanded=False
                    ):
                        for label, context_key in (
                            ("Previous", "previous"),
                            ("Selected", "selected"),
                            ("Next", "next"),
                            ("Pseudo", "pseudo_excerpt"),
                        ):
                            value = clean_text(context.get(context_key, ""))
                            if value:
                                st.markdown(f"**{label}:** {value}")
            st.markdown("#### Alternatives and limitations")
            limitations = [
                clean_text(value)
                for value in unit.get("limitations", [])
                if clean_text(value)
            ]
            if limitations:
                for value in limitations:
                    st.write("- " + value)
            else:
                st.write("No task-level limitation was recorded.")


def show_query_clause_coverage(run_root: Path, key_prefix: str) -> None:
    """Expose full-query coverage separately from emitted-target coverage."""
    ledger = safe_csv(
        run_root / "query_clause_execution_ledger_v0415.csv"
    )
    if ledger.empty:
        return
    substantive = ledger[
        ledger.get("is_substantive", pd.Series(dtype=bool)).map(
            lambda value: str(value).strip().lower()
            in {"1", "true", "yes", "y"}
        )
    ]
    if substantive.empty:
        return
    states = substantive.get(
        "execution_state", pd.Series(dtype=str)
    ).astype(str)
    target_executed = int(
        states.isin({"EXECUTED", "EXECUTED_TARGET"}).sum()
    )
    context_consumed = int(states.eq("CONTEXT_CONSUMED").sum())
    covered = target_executed + context_consumed
    total = int(len(substantive))
    if covered == total:
        st.success(
            f"Full query clause coverage: {covered}/{total} covered "
            f"({target_executed} target-executed; "
            f"{context_consumed} context-consumed)."
        )
    else:
        st.warning(
            "Query clause coverage is partial: "
            f"{covered}/{total} substantive clauses covered. "
            "The result must not be interpreted as covering the entire query."
        )
    columns = [
        column for column in (
            "source_clause_id", "clause_text", "readiness_state",
            "execution_state", "reason",
        ) if column in substantive.columns
    ]
    show_table(
        substantive[columns] if columns else substantive,
        key_prefix + "_clause_coverage",
        100,
    )


def show_output(
    output_dir: Path,
    key_prefix: str,
    *,
    code_dir: Optional[Path] = None,
    python_executable: str = sys.executable,
    llm_model: str = "gpt-4o-mini",
    key_environment: str = "OPENAI_API_KEY",
    session_key: str = "",
    credential_source: str = "Environment variable",
) -> None:
    output_dir = resolve_output(output_dir)
    evaluation = safe_csv(output_dir / "task_evaluation_v048.csv")
    if evaluation.empty:
        st.warning("No completed v0.4 task evaluation was found at " + str(output_dir))
        return
    manifest = safe_json(output_dir / "semantic_match_run_manifest_v048.json")
    evidence = safe_json(output_dir / "task_evidence_units_v0413.json")
    run_root = complete_run_root(output_dir)
    case_meta = load_case_metadata(run_root)
    tasks = set(evaluation.get("task", pd.Series(dtype=str)).astype(str))
    tabs: List[Tuple[str, Any]] = []

    def overview() -> None:
        query_text = clean_text(case_meta.get("query_text", ""))
        if query_text:
            st.markdown("### Original query")
            st.code(query_text)
        meta_rows = [
            {
                "field": "Case ID",
                "value": clean_text(case_meta.get("case_id", "")),
            },
            {
                "field": "Corpus",
                "value": clean_text(case_meta.get("corpus", "")),
            },
            {
                "field": "Task",
                "value": clean_text(case_meta.get("task_family", "")),
            },
            {
                "field": "Source document",
                "value": clean_text(case_meta.get("doc_id", "")),
            },
            {
                "field": "Paragraph",
                "value": clean_text(case_meta.get("paragraph_id", "")),
            },
            {
                "field": "Query kind",
                "value": clean_text(case_meta.get("query_kind", "")),
            },
            {
                "field": "Ground truth",
                "value": clean_text(case_meta.get("ground_truth", "")),
            },
        ]
        meta_rows = [row for row in meta_rows if row["value"]]
        if meta_rows:
            st.markdown("### Case metadata")
            st.dataframe(pd.DataFrame(meta_rows), width="stretch", hide_index=True)
        if clean_text(case_meta.get("source_context", "")):
            st.markdown("### Source context")
            st.write(clean_text(case_meta.get("source_context", "")))
        show_query_clause_coverage(
            run_root, key_prefix + "_overview"
        )
        digest = output_dir / "semantic_match_task_digest_v0413.md"
        if not digest.is_file():
            digest = output_dir / "semantic_match_task_digest_v048.md"
        if digest.exists():
            st.markdown(digest.read_text(encoding="utf-8", errors="ignore"))
        show_table(evaluation, key_prefix + "_evaluation")
        pair_comparison = safe_csv(
            output_dir / "pair_comparison_v048.csv"
        )
        if not pair_comparison.empty:
            st.markdown("### Controlled-pair transition")
            pair_columns = [
                column for column in (
                    "target_id", "change",
                    "baseline_primary_finding",
                    "baseline_primary_finding_source_doc_id",
                    "holdout_primary_finding",
                    "holdout_primary_finding_source_doc_id",
                    "removed_evidence_roles",
                    "retained_evidence_roles",
                    "added_evidence_roles",
                    "authoritative_source_targeted_by_holdout",
                    "primary_finding_source_targeted_by_holdout",
                    "holdout_relevance_status",
                ) if column in pair_comparison.columns
            ]
            show_table(
                pair_comparison[pair_columns]
                if pair_columns else pair_comparison,
                key_prefix + "_pair_transition",
            )
        pair_llm_state = run_root / "pair_llm_digest_state_v0413.json"
        task_llm_state = output_dir / "llm_digest_state_v0413.json"
        llm_state_path = (
            pair_llm_state if pair_llm_state.is_file() else task_llm_state
        )
        llm = safe_json(llm_state_path)
        pair_evidence = safe_json(
            run_root / "pair_evidence_units_v0413.json"
        )
        digest_evidence = pair_evidence or evidence
        st.markdown("### Optional LLM status")
        state = clean_text(llm.get("state", "NOT_REQUESTED"))
        dimensions = {
            "API": llm.get("api_call_state", "NOT_ASSESSED"),
            "Consistency": llm.get("consistency_state", "NOT_ASSESSED"),
            "Explanation": llm.get(
                "explanatory_completeness_state", "NOT_ASSESSED"
            ),
            "Coverage": llm.get(
                "target_coverage_state", "NOT_ASSESSED"
            ),
            "Grounding": llm.get(
                "evidence_grounding_state", "NOT_ASSESSED"
            ),
        }
        st.dataframe(
            pd.DataFrame([
                {"validation dimension": key, "state": value}
                for key, value in dimensions.items()
            ]),
            width="stretch",
            hide_index=True,
        )
        failure_summary = summarize_llm_validation_failures(
            llm, digest_evidence
        )
        if failure_summary["failed_target_count"]:
            st.warning(
                "Pair digest validation failures: "
                f'{failure_summary["failed_target_count"]} of '
                f'{failure_summary["requested_target_count"]} targets.'
            )
            st.dataframe(
                pd.DataFrame([
                    {
                        "target_id": target_id,
                        "validation_failures": "; ".join(reasons),
                    }
                    for target_id, reasons in failure_summary[
                        "per_target_reasons"
                    ].items()
                ]),
                width="stretch",
                hide_index=True,
            )
        if failure_summary["global_reasons"]:
            st.caption(
                "Global validation failures: "
                + "; ".join(failure_summary["global_reasons"])
            )
        if state == "VALIDATED_EXPLANATORY":
            st.success("VALIDATED_EXPLANATORY")
            llm_path = (
                run_root / "semantic_match_pair_digest_llm_v0413.md"
                if pair_llm_state.is_file()
                else output_dir / "semantic_match_digest_llm_v0413.md"
            )
            if llm_path.exists():
                st.markdown(llm_path.read_text(encoding="utf-8", errors="ignore"))
        else:
            llm_state = state
            pair_incomplete = clean_text(
                manifest.get("pair_status", "")
            ) == "PAIR_INCOMPLETE"
            llm_deferred = (
                llm_state == "NOT_REQUESTED" and pair_incomplete
            )
            if llm_deferred:
                llm_state = "DEFERRED_UNTIL_PAIR_RECONCILIATION"
            if llm_state == "REJECTED":
                st.warning(
                    "Optional LLM digest rejected. Deterministic results "
                    "remain authoritative."
                )
            elif llm_state == "AUTHENTICATION_FAILED":
                st.error(
                    "Optional LLM authentication failed for the explicitly "
                    "selected credential source. Authentication errors are "
                    "not retried until the credential changes."
                )
            else:
                st.info(llm_state)
            if code_dir and not llm_deferred and st.button(
                "Retry only the Optional LLM digest",
                key=key_prefix + "_retry_llm",
            ):
                environment = os.environ.copy()
                credential = credential_source_status(
                    credential_source,
                    key_environment,
                    session_key,
                )
                if not credential["populated"]:
                    st.error(
                        "Optional LLM preflight failed: "
                        + credential["source_name"]
                        + " is empty. Each independently launched process "
                        "or browser session must receive its selected "
                        "credential source."
                    )
                    return
                if credential_source != "Environment variable":
                    environment[key_environment] = credential["value"]
                command = [
                    python_executable,
                    str(code_dir / "retry_llm_digest_v0_4_14.py"),
                    "--output-dir", str(
                        run_root if pair_llm_state.is_file() else output_dir
                    ),
                    "--llm-model", llm_model,
                    "--openai-api-key-env", key_environment,
                ]
                completed = subprocess.run(
                    command,
                    cwd=str(code_dir),
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                if completed.returncode == 0:
                    st.success("Optional LLM digest validated.")
                    st.rerun()
                else:
                    refreshed = safe_json(
                        llm_state_path
                    )
                    st.error(
                        "Digest retry was not accepted: "
                        + clean_text(refreshed.get("state", completed.stderr))
                    )
    tabs.append(("Overview", overview))

    def evidence_cards() -> None:
        st.markdown("### Target evidence cards")
        show_evidence_cards(evidence, key_prefix + "_evidence_units")

    tabs.append(("Evidence", evidence_cards))

    if "reconstruction" in tasks:
        def reconstruction() -> None:
            show_evidence_cards(evidence, key_prefix + "_reconstruction_cards")
            st.markdown("### Minimal explanatory set")
            show_table(
                safe_csv(output_dir / "minimal_reconstruction_set_v048.csv"),
                key_prefix + "_minimal",
            )
            st.markdown("### Strong, weak, and missing fact evidence")
            show_table(
                safe_csv(output_dir / "fragment_document_evidence_v048.csv"),
                key_prefix + "_evidence",
            )
            st.markdown("### Localized raw provenance")
            passages = safe_csv(
                output_dir / "fragment_source_passages_v048.csv"
            )
            show_table(passages, key_prefix + "_passages")
            if not passages.empty:
                choices = [
                    f"{clean_text(row.get('fragment_id', ''))} / "
                    f"{clean_text(row.get('doc_id', ''))}"
                    for _, row in passages.iterrows()
                ]
                selected = st.selectbox(
                    "Fragment/document evidence card",
                    choices,
                    key=key_prefix + "_evidence_card",
                )
                row = passages.iloc[choices.index(selected)]
                st.markdown("**KG-derived pseudo evidence**")
                st.write(clean_text(row.get("pseudo_evidence", "")) or "Unavailable")
                component_cards = clean_text(
                    row.get("pseudo_component_cards_json", "")
                )
                if component_cards:
                    try:
                        st.dataframe(
                            pd.DataFrame(json.loads(component_cards)),
                            width="stretch", hide_index=True,
                        )
                    except Exception:
                        pass
                raw_cards = clean_text(row.get("raw_chunk_cards_json", ""))
                if raw_cards:
                    try:
                        for card in json.loads(raw_cards):
                            with st.expander(
                                f"Raw chunk {card.get('rank')} · "
                                f"{card.get('chunk_id')}",
                                expanded=card.get("rank") == 1,
                            ):
                                st.markdown("**Previous**")
                                st.write(clean_text(card.get("previous")) or "Unavailable")
                                st.markdown("**Selected**")
                                st.write(clean_text(card.get("selected")) or "Unavailable")
                                st.markdown("**Next**")
                                st.write(clean_text(card.get("next")) or "Unavailable")
                    except Exception:
                        st.write(clean_text(row.get("raw_passage", "")))
            st.markdown("### Nearest rejected evidence")
            show_table(
                safe_csv(output_dir / "nearest_rejected_evidence_v048.csv"),
                key_prefix + "_rejected",
            )
        tabs.append(("Task-specific", reconstruction))
    if "redaction" in tasks:
        def redaction() -> None:
            show_evidence_cards(evidence, key_prefix + "_redaction_cards")
            show_grouped_csv_page(
                output_dir / "redaction_candidate_audit_v048.csv",
                key_prefix + "_redaction_audit",
                "slot_id",
            )
        tabs.append(("Task-specific", redaction))
    if "sentence_redaction" in tasks:
        def sentence_redaction() -> None:
            show_evidence_cards(evidence, key_prefix + "_sentence_cards")
            show_csv_page(
                output_dir / "sentence_redaction_boundary_audit_v048.csv",
                key_prefix + "_sentence_audit",
            )
        tabs.append(("Task-specific", sentence_redaction))
    if "claims" in tasks:
        def claims() -> None:
            show_evidence_cards(evidence, key_prefix + "_claims_cards")
            show_table(
                safe_csv(output_dir / "claim_support_candidates_v048.csv"),
                key_prefix + "_claims",
            )
        tabs.append(("Task-specific", claims))
    if "contradictions" in tasks:
        def contradiction() -> None:
            show_evidence_cards(
                evidence, key_prefix + "_contradiction_cards"
            )
            st.caption(
                "Evidence relationships only; this is not automatic historical-truth adjudication."
            )
            role_columns = [
                column for column in (
                    "target_id", "primary_finding",
                    "primary_finding_source_doc_id",
                    "primary_finding_raw_certification_status",
                    "evidence_pattern",
                    "side_a_support_source_doc_id",
                    "contradiction_or_update_source_doc_id",
                    "uncertainty_source_doc_id",
                    "qualification_or_scope_source_doc_id",
                    "related_different_event_source_doc_id",
                    "background_only_source_doc_id",
                ) if column in evaluation.columns
            ]
            st.markdown("### Authoritative evidence-role summary")
            show_table(
                evaluation[role_columns] if role_columns else evaluation,
                key_prefix + "_contradiction_roles",
            )
            st.markdown("### Candidate-level evidence")
            show_table(
                safe_csv(output_dir / "contradiction_candidates_v048.csv"),
                key_prefix + "_contradiction",
            )
        tabs.append(("Task-specific", contradiction))

    def visualizations() -> None:
        st.caption(
            "Task-local decision visualization; deterministic tables remain authoritative."
        )
        if "reconstruction" in tasks:
            frame = safe_csv(
                output_dir / "fragment_document_evidence_v048.csv", 2000
            )
            columns = [
                column for column in (
                    "fragment_id", "doc_id", "final_score",
                    "critical_fact_coverage", "eligible",
                ) if column in frame.columns
            ]
            show_table(frame[columns] if columns else frame, key_prefix + "_viz_recon", 200)
            if {"doc_id", "final_score"} <= set(frame.columns):
                chart = frame.nlargest(30, "final_score").set_index("doc_id")[
                    ["final_score"]
                ]
                st.bar_chart(chart)
        elif "redaction" in tasks:
            frame = balanced_csv_sample(
                output_dir / "redaction_candidate_audit_v048.csv",
                "slot_id", 40,
            )
            columns = [
                column for column in (
                    "slot_id", "candidate", "doc_id", "expected_role",
                    "actual_role", "event_identity_score", "final_score",
                    "eligible", "failed_gates",
                ) if column in frame.columns
            ]
            show_table(frame[columns] if columns else frame, key_prefix + "_viz_red", 100)
            if {"candidate", "final_score"} <= set(frame.columns):
                chart = frame.nlargest(30, "final_score").set_index("candidate")[
                    ["final_score"]
                ]
                st.bar_chart(chart)
        elif "sentence_redaction" in tasks:
            frame = safe_csv(
                output_dir / "sentence_redaction_boundary_audit_v048.csv", 500
            )
            columns = [
                column for column in (
                    "slot_id", "doc_id", "left_neighbor",
                    "selected_sentence", "right_neighbor",
                    "semantic_bridge_score", "event_identity_score",
                    "adjacency_gate_pass", "eligible", "final_score",
                    "failed_gates",
                ) if column in frame.columns
            ]
            st.markdown("### Boundary and hard-gate view")
            show_table(frame[columns] if columns else frame, key_prefix + "_viz_sentence", 50)
            if {"doc_id", "final_score"} <= set(frame.columns):
                st.bar_chart(
                    frame.nlargest(30, "final_score").set_index("doc_id")[
                        ["final_score"]
                    ]
                )
        elif "claims" in tasks:
            frame = safe_csv(
                output_dir / "claim_support_candidates_v048.csv", 1000
            )
            argument_columns = [
                column for column in frame.columns
                if column.endswith("_comparison")
                or column in {
                    "corpus_doc_id", "candidate_relation",
                    "event_identity_score", "final_score",
                    "raw_validation_status",
                }
            ]
            st.markdown("### Proposition argument comparison")
            show_table(
                frame[argument_columns] if argument_columns else frame,
                key_prefix + "_viz_claims",
                100,
            )
            if "final_score" in frame.columns and not frame.empty:
                chart = frame.copy()
                chart["final_score"] = pd.to_numeric(
                    chart["final_score"], errors="coerce"
                )
                chart = chart.dropna(subset=["final_score"]).nlargest(
                    30, "final_score"
                )
                if not chart.empty:
                    doc_column = (
                        "corpus_doc_id"
                        if "corpus_doc_id" in chart.columns
                        else "doc_id"
                        if "doc_id" in chart.columns
                        else ""
                    )
                    labels = (
                        chart[doc_column].astype(str)
                        if doc_column
                        else chart.index.astype(str)
                    )
                    chart = chart.assign(
                        candidate_label=[
                            f"{label} · {position + 1}"
                            for position, label in enumerate(labels)
                        ]
                    )
                    eligible = chart.get(
                        "eligible", pd.Series(False, index=chart.index)
                    ).map(
                        lambda value: str(value).strip().lower()
                        in {"1", "true", "yes", "y"}
                    )
                    plotted = pd.DataFrame({
                        "candidate_label": chart["candidate_label"].to_numpy(),
                        "eligible": chart["final_score"].where(
                            eligible
                        ).to_numpy(),
                        "ineligible_or_review": chart["final_score"].where(
                            ~eligible
                        ).to_numpy(),
                    }).set_index("candidate_label")
                    st.markdown(
                        "### Claims candidate score and eligibility"
                    )
                    st.caption(
                        "Scores support ranking; eligibility remains governed "
                        "by the recorded semantic gates."
                    )
                    st.bar_chart(plotted)
            comparison_fields = [
                column for column in frame.columns
                if column.endswith("_comparison")
            ]
            if comparison_fields and not frame.empty:
                missing_typed_roles = frame[comparison_fields].astype(str).apply(
                    lambda row: row.str.contains(
                        "MISSING_IN_CANDIDATE", case=False, regex=False
                    ).any(),
                    axis=1,
                )
                raw_related = frame.get(
                    "raw_validation_status",
                    pd.Series("", index=frame.index),
                ).astype(str).str.contains(
                    "RELATED|CONFIRMED", case=False, regex=True
                )
                representation_gap = frame[missing_typed_roles & raw_related]
                if not representation_gap.empty:
                    st.warning(
                        "Source representation gap: raw text is related, but "
                        "one or more typed proposition roles are missing. "
                        "This is an extraction/pseudo-proposition review item; "
                        "the Workbench does not lower Matcher gates or promote "
                        "raw prose automatically."
                    )
        elif "contradictions" in tasks:
            coverage = safe_csv(
                output_dir / "contradiction_retrieval_coverage_v048.csv"
            )
            candidates = safe_csv(
                output_dir / "contradiction_candidates_v048.csv", 1000
            )
            st.markdown("### Evidence-class coverage")
            show_table(coverage, key_prefix + "_viz_con_coverage", 100)
            if (
                not candidates.empty
                and "primary_class" in candidates.columns
            ):
                counts = (
                    candidates["primary_class"]
                    .value_counts()
                    .rename_axis("evidence_class")
                    .to_frame("documents")
                )
                st.bar_chart(counts)
            timeline_columns = [
                column for column in (
                    "report_time", "event_time", "corpus_doc_id",
                    "primary_class", "final_score",
                ) if column in candidates.columns
            ]
            st.markdown("### Report-time and event-time evidence lanes")
            show_table(
                candidates[timeline_columns]
                if timeline_columns else candidates,
                key_prefix + "_viz_con_timeline",
                100,
            )
    tabs.append(("Visualizations", visualizations))

    def diagnostics() -> None:
        run_root = complete_run_root(output_dir)
        coordinated_manifest = safe_json(
            run_root / "coordinated_release_manifest_v048.json"
        )
        query_parse_dir = run_root / "query_parse"
        parser_manifest = safe_json(
            query_parse_dir / "query_parse_manifest.json"
        )
        st.markdown("### Parser configuration and eligibility")
        st.json({
            "parser_version": coordinated_manifest.get(
                "parser_version", parser_manifest.get("version", "")
            ),
            "parser_model": coordinated_manifest.get(
                "parser_model", parser_manifest.get("model", "")
            ),
            "prompt_version": parser_manifest.get("prompt_version", ""),
            "schema_version": parser_manifest.get("schema_version", ""),
            "query_sha256": parser_manifest.get("query_sha256", ""),
            "query_parse_sha256": parser_manifest.get(
                "query_parse_sha256", ""
            ),
        })
        st.warning(
            "query_representation_reconciliation.csv is diagnostic-only. "
            "Downstream semantic decisions use query_propositions.csv and "
            "query_event_signatures.csv."
        )
        st.markdown("### Coordinator Parser compatibility checks")
        authoritative_validation = coordinated_manifest.get(
            "authoritative_query_input_validation", {}
        ) or safe_json(
            run_root / "authoritative_query_input_validation_v0415.json"
        )
        schema_compatibility = coordinated_manifest.get(
            "query_schema_compatibility", {}
        ) or safe_json(
            run_root / "query_schema_compatibility_v0415_v0420.json"
        )
        st.json({
            "authoritative_query_inputs": authoritative_validation,
            "schema_and_preflight_adapter": schema_compatibility,
        })
        st.caption(
            "The adapter changes no proposition or event-signature bytes. "
            "Any preflight-only downgrade is counted and reason-coded here."
        )
        show_table(
            safe_csv(
                query_parse_dir / "query_parse_primary_debug_counts.csv"
            ),
            key_prefix + "_parser_primary_counts",
            100,
        )
        show_table(
            safe_csv(query_parse_dir / "query_task_preflight.csv"),
            key_prefix + "_parser_preflight",
            100,
        )
        with st.expander("Parser propositions used for semantic decisions"):
            show_table(
                safe_csv(query_parse_dir / "query_propositions.csv"),
                key_prefix + "_parser_propositions",
                250,
            )
        with st.expander("Parser event signatures used for semantic decisions"):
            show_table(
                safe_csv(query_parse_dir / "query_event_signatures.csv"),
                key_prefix + "_parser_event_signatures",
                250,
            )
        with st.expander("Diagnostic-only representation reconciliation"):
            st.caption(
                "Audit aid only; these role columns are not Matcher, Digest, "
                "or Automation decision inputs."
            )
            show_table(
                safe_csv(
                    query_parse_dir
                    / "query_representation_reconciliation.csv"
                ),
                key_prefix + "_parser_reconciliation_diagnostic_only",
                250,
            )
        st.markdown("### Effective decision profile")
        resolved_profile = safe_json(
            output_dir / "resolved_decision_profile_v0420.json"
        )
        if resolved_profile:
            st.json(resolved_profile)
        else:
            st.info("No resolved v0.4.20 decision profile is available yet.")
        show_table(
            safe_csv(output_dir / "decision_profile_audit_v0420.csv"),
            key_prefix + "_decision_profile_audit",
            100,
        )
        st.markdown("### Advisory decision bands")
        st.caption(
            "These bands do not alter semantic eligibility, ranking, or scores."
        )
        decision_display = safe_csv(
            output_dir / "decision_display_v0420.csv"
        )
        if not decision_display.empty:
            policy = {
                "lower_threshold": float(st.session_state.get(
                    "v315_display_lower", 0.42
                )),
                "upper_threshold": float(st.session_state.get(
                    "v315_display_upper", 0.62
                )),
                "minimum_margin": float(st.session_state.get(
                    "v315_display_margin", 0.03
                )),
            }
            mode = str(st.session_state.get(
                "v315_display_mode", "score_only"
            ))
            decision_display["workbench_operational_band"] = [
                operational_band(
                    eligible=clean_text(row.get(
                        "semantic_eligibility_state", ""
                    )) == "ELIGIBLE",
                    score=row.get("score", ""),
                    margin=row.get("winner_margin", ""),
                    runner_present=clean_text(
                        row.get("runner_up_score", "")
                    ) != "",
                    mode=mode,
                    display_policy=policy,
                )
                for row in decision_display.to_dict("records")
            ]
            decision_display["workbench_policy_mode"] = mode
            decision_display["workbench_policy_lower"] = policy[
                "lower_threshold"
            ]
            decision_display["workbench_policy_upper"] = policy[
                "upper_threshold"
            ]
            decision_display["workbench_policy_margin"] = policy[
                "minimum_margin"
            ]
        show_table(
            decision_display, key_prefix + "_decision_display", 250
        )
        st.markdown("### Progress ledger")
        show_table(
            safe_jsonl(output_dir / "progress_v048.jsonl"),
            key_prefix + "_progress",
            250,
        )
        st.markdown("### Source load and exclusion audit")
        show_table(
            safe_csv(output_dir / "exclusion_audit_v048.csv"),
            key_prefix + "_load_audit",
        )
        st.markdown("### Performance")
        show_table(
            safe_csv(output_dir / "performance_timing_v048.csv"),
            key_prefix + "_timing",
        )
        with st.expander("Native v0.4 run manifest"):
            st.json(manifest)
        with st.expander("Optional LLM diagnostics"):
            st.json(safe_json(output_dir / "llm_digest_state_v048.json"))
    tabs.append(("Diagnostics", diagnostics))

    def files() -> None:
        inventory = manifest.get("output_inventory", {})
        st.json(inventory)
        run_root = complete_run_root(output_dir)
        coordinated = safe_json(
            run_root / "coordinated_status_v048.json"
        )
        complete_states = {"COMPLETED", "COMPLETED_WITH_WARNINGS"}
        incomplete = complete_run_self_check(run_root)
        complete_ready = (
            coordinated.get("status") in complete_states
            if coordinated else not incomplete
        )
        zip_state_key = key_prefix + "_prepared_zip"
        if not complete_ready:
            st.warning(
                "This coordinated run is still active or incomplete. Any ZIP "
                "created now is an in-progress snapshot, not a complete run."
            )
        if st.button(
            "Prepare complete run ZIP",
            key=key_prefix + "_prepare_download",
            disabled=not complete_ready,
        ):
            if incomplete:
                st.error(
                    "ZIP blocked because the coordinated run is incomplete: "
                    + ", ".join(incomplete)
                )
            else:
                with st.spinner("Preparing disk-backed ZIP..."):
                    st.session_state[zip_state_key] = str(
                        zip_tree_to_file(run_root)
                    )
        prepared = Path(st.session_state.get(zip_state_key, ""))
        if prepared.is_file():
            with prepared.open("rb") as handle:
                st.download_button(
                    "Download complete run ZIP",
                    data=handle,
                    file_name=complete_run_root(output_dir).name + ".zip",
                    mime="application/zip",
                    key=key_prefix + "_download",
                )
        snapshot_state_key = key_prefix + "_prepared_snapshot_zip"
        if not complete_ready and st.button(
            "Prepare in-progress snapshot ZIP",
            key=key_prefix + "_prepare_snapshot",
        ):
            with st.spinner("Preparing explicitly labelled snapshot ZIP..."):
                st.session_state[snapshot_state_key] = str(
                    zip_tree_to_file(run_root)
                )
        snapshot = Path(
            st.session_state.get(snapshot_state_key, "")
        )
        if not complete_ready and snapshot.is_file():
            with snapshot.open("rb") as handle:
                st.download_button(
                    "Download in-progress snapshot ZIP",
                    data=handle,
                    file_name=run_root.name + "_IN_PROGRESS_SNAPSHOT.zip",
                    mime="application/zip",
                    key=key_prefix + "_snapshot_download",
                )
        visualization = output_dir / "task_visualization_v048.html"
        if visualization.is_file():
            with visualization.open("rb") as handle:
                st.download_button(
                    "Download standalone visualization",
                    data=handle,
                    file_name=visualization.name,
                    mime="text/html",
                    key=key_prefix + "_visualization_download",
                )
    tabs.append(("Files", files))

    rendered = st.tabs([name for name, _function in tabs])
    for tab, (_name, function) in zip(rendered, tabs):
        with tab:
            function()


def running_outputs(run_root: Path) -> List[Path]:
    candidates = [
        run_root / "semantic_match_output",
        run_root / "baseline" / "semantic_match_output",
        run_root / "holdout" / "semantic_match_output",
    ]
    return [path for path in candidates if path.is_dir()]


st.set_page_config(
    page_title="Matryoshka Semantic Evidence Workbench v3.19",
    layout="wide",
)
st.title("Matryoshka Semantic Evidence Workbench v3.19")
st.caption(
    "Parser v0.4.15 / Matcher v0.4.20 / Workbench v3.19 — "
    "Requirements v1.4 + Addenda v1.4.2 Revision 1 and v1.5"
)

with st.sidebar:
    st.header("Native v0.4 configuration")
    code_dir_text = st.text_input("Code directory", value=str(ROOT))
    extraction_dir_text = st.text_input("Extraction directory", value="")
    corpus_entity_resource_text = st.text_input(
        "Corpus entity/alias resource (optional)",
        value="",
        help=(
            "Explicit path to a versioned corpus resource v1. Names and "
            "locations are data here; they are never hard-coded in Matcher."
        ),
    )
    semantic_resources_text = st.text_input(
        "Additional semantic resource (optional)",
        value="",
    )
    decision_profile_text = st.text_input(
        "Decision profile (optional)",
        value="",
        help=(
            "Leave blank for the packaged v0.4.16-compatible defaults. The "
            "profile may change thresholds, but cannot change scoring weights."
        ),
    )
    with st.expander("Display-only decision policy"):
        display_mode = st.selectbox(
            "Decision display mode",
            ["score_only", "three_zone", "binary_threshold"],
            help="Presentation only; it never changes retrieval or semantic scores.",
            key="v315_display_mode",
        )
        display_lower = st.number_input(
            "Display lower threshold", min_value=0.0, max_value=1.0,
            value=0.42, step=0.01,
            key="v315_display_lower",
        )
        display_upper = st.number_input(
            "Display upper threshold", min_value=0.0, max_value=1.0,
            value=0.62, step=0.01,
            key="v315_display_upper",
        )
        display_margin = st.number_input(
            "Display minimum winner margin", min_value=0.0, max_value=1.0,
            value=0.03, step=0.01,
            key="v315_display_margin",
        )
    output_base_text = st.text_input("Run output base", value="")
    python_executable = st.text_input("Python executable", value=sys.executable)
    task = st.selectbox(
        "Task",
        [
            "auto", "reconstruction", "redaction", "sentence_redaction",
            "claims", "contradictions",
        ],
    )
    experiment = st.radio(
        "Experiment",
        ["Single run", "Controlled baseline + holdout"],
    )
    target_doc_ids = st.text_area("Controlled holdout document IDs", height=80)
    explicit_exclusions = st.text_area("Additional exclusions", height=80)
    staged_top_docs = st.number_input(
        "Maximum candidate documents per target",
        min_value=1, max_value=10000, value=150,
    )
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        ["staged", "exhaustive"],
        help="Exhaustive is an explicit audit mode and can be substantially slower.",
    )
    raw_use_mode = st.selectbox(
        "Raw sentence mode",
        ["validate", "discover"],
        help="Discover permits corpus-wide raw sentence discovery; validate stays within staged documents.",
    )
    max_candidates = st.number_input(
        "Candidate safety limit per target",
        min_value=100, max_value=1000000, value=100000,
    )
    max_elapsed = st.number_input(
        "Maximum matcher seconds (0 = unlimited)",
        min_value=0.0, value=0.0,
    )
    reuse_source_cache = st.checkbox(
        "Reuse validated source-table cache across runs",
        value=True,
        help=(
            "Avoids reparsing the same extraction CSV files for each test. "
            "The cache is rejected automatically when source sizes or mtimes change."
        ),
    )
    st.header("OpenAI")
    key_environment = st.text_input(
        "API-key environment variable", value="OPENAI_API_KEY"
    )
    credential_source = st.radio(
        "Credential source",
        ["Environment variable", "Streamlit secret", "Session-only key"],
        help=(
            "The environment source is never silently replaced by a session "
            "field. A session key is used only when explicitly selected."
        ),
    )
    session_key = ""
    if credential_source == "Session-only key":
        session_key = st.text_input("Session-only API key", type="password")
    selected_credential = credential_source_status(
        credential_source, key_environment, session_key
    )
    selected_credential_available = bool(
        selected_credential["populated"]
    )
    if selected_credential_available:
        st.success(
            selected_credential["source_name"] + " is populated."
        )
    else:
        st.info(selected_credential["source_name"] + " is empty.")
    use_openai_parser = st.checkbox("Use OpenAI parser", value=False)
    parser_model = st.text_input(
        "Parser model",
        value="gpt-4o",
        help=(
            "GPT-4o is the recommended quality default. GPT-4o-mini remains "
            "available for lower-cost experiments; the selected value is "
            "recorded in the coordinated and Parser manifests."
        ),
    )
    use_llm_digest = st.checkbox("Generate optional LLM digest", value=False)
    llm_model = st.text_input("LLM digest model", value="gpt-4o-mini")
    run_label = st.text_input(
        "Run label",
        value="",
        help="Optional filesystem-safe label recorded before launch.",
    )

query_text = st.text_area(
    "Query",
    height=280,
    placeholder="Paste the reconstruction, redaction, claim, or contradiction query.",
)

terminal_states = {
    "COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED", "PARSER_FAILED",
    "PREFLIGHT_BLOCKED", "PAIR_RECONCILIATION_FAILED", "CANCELLED",
    "PROCESS_EXITED_NONZERO", "PROCESS_EXITED_WITHOUT_TERMINAL_STATUS",
}
prior_root_text = st.session_state.get("v315_run_root", "")
prior_status = (
    safe_json(Path(prior_root_text) / "coordinated_status_v048.json")
    if prior_root_text and Path(prior_root_text).is_dir() else {}
)
active_nonterminal = bool(
    prior_root_text
    and Path(prior_root_text).is_dir()
    and prior_status.get("status") not in terminal_states
)
launch = st.button(
    "Launch native coordinated run",
    type="primary",
    disabled=active_nonterminal,
    help="The current run must reach a terminal state before another launch."
    if active_nonterminal else None,
)
if launch:
    errors: List[str] = []
    code_dir, error = safe_path(code_dir_text, "Code directory", True)
    if error:
        errors.append(error)
    corpus_entity_resource = None
    if clean_text(corpus_entity_resource_text):
        corpus_entity_resource = Path(
            corpus_entity_resource_text
        ).expanduser().resolve()
        if not corpus_entity_resource.is_file():
            errors.append(
                "Corpus entity/alias resource is not a file: "
                + str(corpus_entity_resource)
            )
    semantic_resources = None
    if clean_text(semantic_resources_text):
        semantic_resources = Path(
            semantic_resources_text
        ).expanduser().resolve()
        if not semantic_resources.is_file():
            errors.append(
                "Additional semantic resource is not a file: "
                + str(semantic_resources)
            )
    decision_profile = None
    if clean_text(decision_profile_text):
        decision_profile = Path(
            decision_profile_text
        ).expanduser().resolve()
        if not decision_profile.is_file():
            errors.append(
                "Decision profile is not a file: " + str(decision_profile)
            )
    if float(display_lower) > float(display_upper):
        errors.append(
            "Display lower threshold may not exceed the upper threshold."
        )
    extraction_dir, error = safe_path(
        extraction_dir_text, "Extraction directory", True
    )
    if error:
        errors.append(error)
    output_base, error = safe_path(output_base_text, "Run output base", False)
    if error:
        errors.append(error)
    if not clean_text(query_text):
        errors.append("Query is empty.")
    if (
        experiment == "Controlled baseline + holdout"
        and not clean_text(target_doc_ids)
    ):
        errors.append("A controlled pair requires target document IDs.")
    if (
        (use_openai_parser or use_llm_digest)
        and not selected_credential_available
    ):
        errors.append(
            "OpenAI preflight failed: "
            + selected_credential["source_name"]
            + " is empty. Each independently launched process or browser "
            "session must receive its selected credential source."
        )
    if not errors and code_dir:
        environment_versions, environment_error = python_environment_preflight(
            python_executable, code_dir
        )
        if environment_error:
            errors.append(environment_error)
    if errors:
        for error in errors:
            st.error(error)
    else:
        assert code_dir and extraction_dir and output_base
        output_base.mkdir(parents=True, exist_ok=True)
        run_name = collision_safe_run_name(clean_text(run_label))
        run_root = output_base / run_name
        run_root.mkdir(parents=True, exist_ok=False)
        query_file = run_root / "query_input.txt"
        query_file.write_text(query_text, encoding="utf-8")
        launch_fingerprint = hashlib.sha256(json.dumps({
            "query": clean_text(query_text),
            "task": task,
            "experiment": experiment,
            "targets": clean_text(target_doc_ids),
            "exclusions": clean_text(explicit_exclusions),
            "retrieval_mode": retrieval_mode,
            "raw_use_mode": raw_use_mode,
            "corpus_entity_resource":
                str(corpus_entity_resource or ""),
            "semantic_resources": str(semantic_resources or ""),
            "decision_profile": str(decision_profile or ""),
            "display_policy": {
                "mode": display_mode,
                "lower_threshold": float(display_lower),
                "upper_threshold": float(display_upper),
                "minimum_margin": float(display_margin),
            },
        }, sort_keys=True).encode("utf-8")).hexdigest()
        (run_root / "launch_fingerprint_v048.txt").write_text(
            launch_fingerprint + "\n", encoding="utf-8"
        )
        (run_root / "workbench_display_policy_v0314.json").write_text(
            json.dumps({
                "schema_version": "matryoshka_workbench_display_policy_v1",
                "mode": display_mode,
                "lower_threshold": float(display_lower),
                "upper_threshold": float(display_upper),
                "minimum_margin": float(display_margin),
                "advisory_only": True,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        if credential_source != "Environment variable":
            environment[key_environment] = selected_credential["value"]
        command = [
            python_executable,
            str(code_dir / "run_coordinated_v0_4_21.py"),
            "pair" if experiment.startswith("Controlled") else "single",
            "--query-file", str(query_file),
            "--extraction-dir", str(extraction_dir),
            "--task", task,
            "--run-root", str(run_root),
            "--code-dir", str(code_dir),
            "--python-executable", python_executable,
            "--staged-top-docs", str(int(staged_top_docs)),
            "--retrieval-mode", retrieval_mode,
            "--raw-use-mode", raw_use_mode,
            "--max-candidates-per-target", str(int(max_candidates)),
            "--max-elapsed-seconds", str(float(max_elapsed)),
            "--openai-api-key-env", key_environment,
            "--parser-model", parser_model,
            "--llm-model", llm_model,
        ]
        if reuse_source_cache:
            cache_dir = shared_workbench_cache_root(output_base)
            cache_name = (
                hashlib.sha256(
                    str(extraction_dir).encode("utf-8")
                ).hexdigest()[:16]
                + f"_{task}_source_tables_v0420.pkl"
            )
            feature_name = (
                hashlib.sha256(
                    (str(extraction_dir) + "|" + task).encode("utf-8")
                ).hexdigest()[:16]
                + f"_{task}_features_v0420.pkl"
            )
            command += [
                "--shared-source-cache", str(cache_dir / cache_name),
                "--task-feature-cache", str(cache_dir / feature_name),
            ]
        if corpus_entity_resource is not None:
            command += [
                "--corpus-entity-resource",
                str(corpus_entity_resource),
            ]
        if semantic_resources is not None:
            command += [
                "--semantic-resources", str(semantic_resources)
            ]
        if decision_profile is not None:
            command += ["--decision-profile", str(decision_profile)]
        if clean_text(target_doc_ids):
            command += ["--target-doc-ids", target_doc_ids]
        if clean_text(explicit_exclusions):
            command += ["--exclude-doc-ids", explicit_exclusions]
        if use_openai_parser:
            command.append("--use-openai-parser")
        if use_llm_digest:
            command.append("--use-llm-digest")
        (run_root / "workbench_command.txt").write_text(
            command_text(command) + "\n", encoding="utf-8"
        )
        stdout_handle = (run_root / "workbench_stdout.log").open(
            "w", encoding="utf-8"
        )
        stderr_handle = (run_root / "workbench_stderr.log").open(
            "w", encoding="utf-8"
        )
        process = subprocess.Popen(
            command,
            cwd=str(code_dir),
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        stdout_handle.close()
        stderr_handle.close()
        st.session_state["v315_process"] = process
        st.session_state["v315_run_root"] = str(run_root)
        st.success("Run launched. Status will refresh automatically.")

active_root_text = st.session_state.get("v315_run_root", "")
if active_root_text:
    active_root = Path(active_root_text)
    st.divider()
    st.subheader("Active coordinated run")
    st.code(str(active_root))
    process = st.session_state.get("v315_process")
    status = safe_json(active_root / "coordinated_status_v048.json")
    if isinstance(process, subprocess.Popen):
        recovered = terminal_status_for_exited_process(
            process.poll(),
            status,
            (
                (active_root / "workbench_stderr.log").read_text(
                    encoding="utf-8", errors="ignore"
                )[-4000:]
                if (active_root / "workbench_stderr.log").is_file()
                else ""
            ),
        )
        if recovered:
            recovered["completed_at"] = datetime.now().astimezone().isoformat()
            status_path = active_root / "coordinated_status_v048.json"
            temporary = status_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(recovered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, status_path)
            status = recovered
    if not active_root.exists():
        st.error("RUN_ROOT_MOVED: the active run directory is missing or was renamed.")
    if status.get("status") in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}:
        st.success("Run finished. All result tabs and downloads are ready.")
    elif status.get("status") in terminal_states:
        st.error("Run stopped with status: " + clean_text(status.get("status")))
    else:
        st.info(
            "Run in progress: " + clean_text(status.get("status", "STARTING"))
        )
        command_record = (
            active_root / "workbench_command.txt"
        ).read_text(
            encoding="utf-8", errors="ignore"
        ) if (active_root / "workbench_command.txt").is_file() else ""
        if (
            "--use-llm-digest" in command_record
            and " pair " in " " + command_record + " "
        ):
            st.info(
                "Optional LLM digest: "
                "DEFERRED_UNTIL_PAIR_RECONCILIATION. Deterministic baseline "
                "and holdout results must finish first."
            )
    st.json(status)
    coordinated_preflight = safe_csv(
        active_root / "coordinated_preflight_v048.csv"
    )
    if not coordinated_preflight.empty:
        st.markdown("#### Parse and task preflight")
        show_table(
            coordinated_preflight,
            "active_coordinated_preflight",
            100,
        )
    controls = st.columns(3)
    if controls[0].button("Refresh status"):
        st.rerun()
    if controls[1].button("Cancel safely"):
        (active_root / "CANCEL").write_text(
            "Cancellation requested from Workbench v3.19\n",
            encoding="utf-8",
        )
        for output in running_outputs(active_root):
            (output / "CANCEL").write_text(
                "Cancellation requested from Workbench v3.19\n",
                encoding="utf-8",
            )
        st.warning("Cooperative cancellation requested; use Refresh status.")
    if controls[2].button("Clear active run from view"):
        st.session_state.pop("v315_process", None)
        st.session_state.pop("v315_run_root", None)
        st.rerun()
    if status.get("status") not in terminal_states:
        snapshot_key = "v315_active_snapshot_zip"
        st.warning(
            "A ZIP prepared before the terminal status is an in-progress "
            "snapshot and must not be treated as a complete run."
        )
        if st.button(
            "Prepare in-progress snapshot ZIP",
            key="v315_prepare_active_snapshot",
        ):
            with st.spinner("Preparing explicitly labelled snapshot ZIP..."):
                st.session_state[snapshot_key] = str(
                    zip_tree_to_file(active_root)
                )
        snapshot_path = Path(st.session_state.get(snapshot_key, ""))
        if snapshot_path.is_file():
            with snapshot_path.open("rb") as handle:
                st.download_button(
                    "Download in-progress snapshot ZIP",
                    data=handle,
                    file_name=active_root.name + "_IN_PROGRESS_SNAPSHOT.zip",
                    mime="application/zip",
                    key="v315_download_active_snapshot",
                )
    auto_refresh = st.checkbox(
        "Auto-refresh while running",
        value=True,
        key="v315_auto_refresh",
        help="Refreshes every two seconds until a terminal status is written.",
    )
    for output in running_outputs(active_root):
        preflight = safe_csv(output / "task_preflight_v048.csv")
        if not preflight.empty:
            st.markdown("#### Preflight")
            show_table(preflight, "active_preflight_" + output.parent.name, 50)
        progress = safe_jsonl(output / "progress_v048.jsonl")
        if len(progress):
            st.markdown("#### " + str(output.relative_to(active_root)))
            if "stage" in progress.columns:
                st.info(
                    "Current matcher phase: "
                    + clean_text(progress.iloc[-1].get("stage", ""))
                )
            show_table(progress.tail(20), "active_" + output.name, 20)
    if status.get("status") in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}:
        outputs = running_outputs(active_root)
        if len(outputs) == 2:
            st.markdown("### Controlled pair — side-by-side")
            baseline_column, holdout_column = st.columns(2)
            with baseline_column:
                st.markdown("#### Baseline")
                show_output(
                    outputs[0], "active_baseline", code_dir=Path(code_dir_text),
                    python_executable=python_executable, llm_model=llm_model,
                    key_environment=key_environment, session_key=session_key,
                    credential_source=credential_source,
                )
            with holdout_column:
                st.markdown("#### Holdout")
                show_output(
                    outputs[1], "active_holdout", code_dir=Path(code_dir_text),
                    python_executable=python_executable, llm_model=llm_model,
                    key_environment=key_environment, session_key=session_key,
                    credential_source=credential_source,
                )
        elif outputs:
            show_output(
                outputs[0], "active_single", code_dir=Path(code_dir_text),
                python_executable=python_executable, llm_model=llm_model,
                key_environment=key_environment, session_key=session_key,
                credential_source=credential_source,
            )
    elif auto_refresh and status.get("status") not in terminal_states:
        time.sleep(2)
        st.rerun()

st.divider()
st.subheader("Open an existing v0.4 output")
existing_text = st.text_input("Existing run or semantic_match_output directory")
if clean_text(existing_text):
    existing, error = safe_path(existing_text, "Existing output", True)
    if error:
        st.error(error)
    elif existing:
        resolved_existing = resolve_output(existing)
        existing_root = complete_run_root(resolved_existing)
        existing_outputs = running_outputs(existing_root)
        if len(existing_outputs) == 2:
            st.markdown("### Controlled pair — side-by-side")
            baseline_column, holdout_column = st.columns(2)
            with baseline_column:
                st.markdown("#### Baseline")
                show_output(
                    existing_outputs[0], "existing_baseline",
                    code_dir=Path(code_dir_text),
                    python_executable=python_executable,
                    llm_model=llm_model,
                    key_environment=key_environment,
                    session_key=session_key,
                    credential_source=credential_source,
                )
            with holdout_column:
                st.markdown("#### Holdout")
                show_output(
                    existing_outputs[1], "existing_holdout",
                    code_dir=Path(code_dir_text),
                    python_executable=python_executable,
                    llm_model=llm_model,
                    key_environment=key_environment,
                    session_key=session_key,
                    credential_source=credential_source,
                )
        else:
            show_output(
                existing, "existing", code_dir=Path(code_dir_text),
                python_executable=python_executable, llm_model=llm_model,
                key_environment=key_environment, session_key=session_key,
                credential_source=credential_source,
            )
