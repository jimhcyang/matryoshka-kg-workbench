"""Native Matcher v0.4 orchestrator.

This module coordinates services and engines.  It contains no task-specific
semantic ranking logic and never invokes a v0.3 matcher.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
)

import pandas as pd

from .contracts import (
    DIGEST_BUILDER_VERSION,
    MATCHER_VERSION,
    OUTPUT_SCHEMA_VERSION,
    PARSER_VERSION,
    RELEASE_VERSION,
    REQUIREMENTS_VERSION,
    TASK_EVALUATION_COLUMNS,
    WORKBENCH_VERSION,
    authoritative_digest,
    configuration_hash,
    finalize_output_manifest,
    output_inventory,
    pair_invariant_hash,
    reconcile_pair,
    refresh_logical_output_manifest,
    safe_json,
    validate_output_manifest,
    validate_llm_digest,
)
from .corpus import CorpusIndex
from .decision_profile import (
    DEFAULT_ENGINE_THRESHOLDS,
    DEFAULT_DISPLAY_POLICY,
    load_decision_profile,
    operational_band,
)
from .digest import generate_optional_llm, write_task_digest_bundle
from .evidence import build_task_evidence_units, write_evidence_outputs
from .engines import ENGINE_TYPES
from .engines.base import EngineConfig, EngineContext
from .progress import CancelledError, RunControl, WorkLimitError
from .query import QueryPackage
from .semantic_resources import (
    load_corpus_entity_resource,
    load_semantic_resources,
)
from .semantics import (
    atomic_write_json,
    clean_text,
    sha256_file,
    utc_now_iso,
    write_csv_with_schema,
)


ARTIFACT_EMPTY_COLUMNS: Dict[str, List[str]] = {
    "fragment_document_evidence_v048.csv": [
        "fragment_id", "doc_id", "retrieval_score", "event_identity_score",
        "critical_fact_coverage", "assertion_gate_pass", "eligible", "final_score",
    ],
    "minimal_reconstruction_set_v048.csv": [
        "selection_order", "fragment_id", "doc_id", "marginal_weighted_gain",
        "new_critical_fact_count", "cumulative_weighted_coverage",
        "single_or_collective", "set_membership", "selection_reason",
    ],
    "extended_supporting_set_v048.csv": [
        "fragment_id", "doc_id", "final_score", "critical_fact_coverage",
        "weighted_fact_coverage", "set_membership", "selection_reason",
    ],
    "fragment_source_passages_v048.csv": [
        "fragment_id", "doc_id", "pseudo_component_ids",
        "pseudo_source_row_ids", "pseudo_evidence", "pseudo_component_count",
        "pseudo_component_cards_json", "raw_chunk_ids", "raw_passage_count",
        "raw_chunk_cards_json",
        "previous_raw_chunk", "selected_raw_chunk", "next_raw_chunk",
        "raw_passage", "raw_status", "provenance_route",
    ],
    "nearest_rejected_evidence_v048.csv": [
        "fragment_id", "doc_id", "evidence_set", "final_score",
        "failed_gates", "event_identity_score", "critical_fact_coverage",
        "pseudo_evidence", "raw_passage",
    ],
    "redaction_candidate_audit_v048.csv": [
        "slot_id", "canonical_candidate_id", "candidate", "candidate_type",
        "doc_id", "source_table", "source_row_id", "eligible", "final_score",
        "failed_gates", "evidence",
    ],
    "redaction_audit_summary_v048.csv": [
        "slot_id", "shortlisted_document_count", "representation_count",
        "canonical_candidate_count", "eligible_canonical_candidate_count",
        "audit_contract",
    ],
    "redaction_slot_coverage_v048.csv": [
        "slot_id", "fragment_id", "expected_type", "slot_frame",
        "governing_predicate", "evaluation_row_present", "candidate_count",
        "canonical_candidate_count", "eligible_candidate_count",
        "selected_candidate", "selected_source_doc_id", "semantic_status",
    ],
    "sentence_redaction_boundary_audit_v048.csv": [
        "slot_id", "doc_id", "raw_chunk_id", "selected_sentence",
        "left_neighbor", "right_neighbor", "eligible", "final_score",
    ],
    "claim_candidate_audit_v048.csv": [
        "query_proposition_id", "corpus_doc_id", "candidate_relation",
        "eligible", "event_identity_score", "raw_validation_status",
        "final_score",
    ],
    "claim_support_candidates_v048.csv": [
        "query_proposition_id", "corpus_doc_id", "candidate_relation",
        "event_identity_score", "raw_validation_status", "final_score",
    ],
    "contradiction_candidate_audit_v048.csv": [
        "query_proposition_id", "corpus_doc_id", "primary_class",
        "eligible", "event_identity_score", "raw_validation_status",
        "final_score",
        "epistemic_state", "epistemic_scope_reason",
        "numeric_approximation_fields_json",
        "uncertainty_same_event_gate_pass",
        "uncertainty_same_event_gate_reason",
    ],
    "contradiction_candidates_v048.csv": [
        "query_proposition_id", "corpus_doc_id", "primary_class",
        "event_identity_score", "raw_validation_status", "final_score",
        "epistemic_state", "epistemic_scope_reason",
        "numeric_approximation_fields_json",
        "uncertainty_same_event_gate_pass",
        "uncertainty_same_event_gate_reason",
    ],
    "contradiction_retrieval_coverage_v048.csv": [
        "query_proposition_id", "evidence_class", "retrieval_status",
        "document_count", "top_doc_id", "reason",
    ],
    "contradiction_document_class_trace_v0420.csv": [
        "query_proposition_id", "corpus_doc_id", "primary_class",
        "corpus_source_table", "corpus_row_id", "final_score",
        "precollapse_document_class", "precollapse_contract",
    ],
    "contradiction_collapse_trace_v0420.csv": [
        "query_proposition_id", "corpus_doc_id", "candidate_class",
        "candidate_source_table", "candidate_row_id", "candidate_score",
        "selected_document_class", "selected_source_table",
        "selected_row_id", "selected", "collapse_reason",
    ],
    "candidate_budget_audit_v0420.csv": [
        "query_proposition_id", "corpus_doc_id", "representation",
        "semantic_family", "available_count", "quota", "selected_count",
        "dropped_count", "top_relevance_score", "selection_contract",
    ],
    "audit_export_summary_v0420.csv": [
        "task", "artifact", "target_id", "original_row_count",
        "exported_row_count", "omitted_row_count",
        "row_limit_per_target", "export_contract",
    ],
}


def _write_task_visualization(
    output_dir: Path,
    task: str,
    evaluation: pd.DataFrame,
    artifacts: Dict[str, pd.DataFrame],
) -> None:
    """Export a portable, deterministic target-balanced HTML visualization."""
    audit_name = {
        "reconstruction": "fragment_document_evidence_v048.csv",
        "redaction": "redaction_candidate_audit_v048.csv",
        "sentence_redaction": "sentence_redaction_boundary_audit_v048.csv",
        "claims": "claim_support_candidates_v048.csv",
        "contradictions": "contradiction_candidates_v048.csv",
    }[task]
    target_column = {
        "reconstruction": "fragment_id",
        "redaction": "slot_id",
        "sentence_redaction": "slot_id",
        "claims": "query_proposition_id",
        "contradictions": "query_proposition_id",
    }[task]
    audit = artifacts.get(audit_name, pd.DataFrame()).copy()
    selected_parts: List[pd.DataFrame] = []
    if not audit.empty and target_column in audit.columns:
        for _target, subset in audit.groupby(target_column, dropna=False):
            if "final_score" in subset.columns:
                subset = subset.sort_values(
                    "final_score", ascending=False, kind="stable"
                )
            selected_parts.append(subset.head(12))
    balanced = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts else audit.head(60)
    )
    payload = {
        "schema": "matryoshka_task_visualization_v0_4_13",
        "task": task,
        "authoritative": False,
        "note": "Display aid only; deterministic CSV outputs are authoritative.",
        "evaluation": evaluation.to_dict("records"),
        "target_balanced_candidates": balanced.to_dict("records"),
    }
    atomic_write_json(
        output_dir / "task_visualization_data_v048.json", payload
    )
    table = balanced.fillna("").astype(str)
    headers = "".join(
        f"<th>{html.escape(column)}</th>" for column in table.columns
    )
    body = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(value)}</td>" for value in row
        ) + "</tr>"
        for row in table.itertuples(index=False, name=None)
    )
    status_cards = "".join(
        "<article><h2>Target "
        + html.escape(clean_text(row.get("target_id", "")))
        + "</h2><p><b>"
        + html.escape(clean_text(
            row.get("primary_finding", "")
            or row.get("semantic_status", "")
        ))
        + "</b></p><p>"
        + html.escape(clean_text(
            row.get("selected_value_or_text", "")
            or row.get("primary_finding_source_doc_id", "")
            or row.get("selected_source_doc_id", "")
            or row.get("unresolved_reason", "")
        ))
        + "</p></article>"
        for row in evaluation.to_dict("records")
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Matryoshka {html.escape(task)} visualization</title>
<style>
body{{font:15px system-ui;margin:2rem;color:#18202a}}
section{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}}
article{{border:1px solid #ccd5df;border-radius:10px;padding:1rem;background:#f7f9fb}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #dbe2e8;padding:.45rem;text-align:left;vertical-align:top}}
th{{position:sticky;top:0;background:#26384a;color:white}}.scroll{{overflow:auto;max-height:70vh}}
</style></head><body>
<h1>Matryoshka {html.escape(task)} decision view</h1>
<p>Display aid only. Deterministic CSV outputs remain authoritative.</p>
<section>{status_cards}</section>
<h2>Target-balanced candidate evidence</h2>
<div class="scroll"><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>
</body></html>"""
    (output_dir / "task_visualization_v048.html").write_text(
        document, encoding="utf-8"
    )


def _parse_doc_ids(*values: str) -> Set[str]:
    import re
    output: Set[str] = set()
    for value in values:
        output.update(
            token.strip().upper()
            for token in re.split(r"[,;\s]+", value or "")
            if token.strip()
        )
    return output


def _preflight(task: str, query: QueryPackage) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if task == "redaction":
        if query.value_slots.empty:
            rows.append({
                "task": task, "target_id": "",
                "status": "BLOCKED_NO_VALUE_SLOTS",
                "reason": "query_redaction_slots.csv contains no slots",
            })
        else:
            for _, row in query.value_slots.iterrows():
                rows.append({
                    "task": task,
                    "target_id": clean_text(row.get("slot_label", "")),
                    "status": "READY",
                    "reason": "native value-slot contract available",
                })
    elif task == "sentence_redaction":
        if query.sentence_slots.empty:
            rows.append({
                "task": task, "target_id": "",
                "status": "BLOCKED_NO_SENTENCE_SLOTS",
                "reason": "query_sentence_redaction_slots.csv contains no slots",
            })
        else:
            for _, row in query.sentence_slots.iterrows():
                rows.append({
                    "task": task,
                    "target_id": clean_text(
                        row.get("sentence_slot_label", "")
                    ),
                    "status": "READY",
                    "reason": "native sentence-slot contract available",
                })
    elif task in {"claims", "contradictions"}:
        if query.propositions.empty:
            rows.append({
                "task": task, "target_id": "",
                "status": "BLOCKED_NO_PROPOSITIONS",
                "reason": "query_propositions.csv contains no propositions",
            })
        else:
            for query_record in query.proposition_records():
                rows.append({
                    "task": task,
                    "target_id": query_record["id"],
                    "status": "READY",
                    "reason": "native canonical proposition available",
                })
            # Parser review and target-local block states remain visible to
            # the analyst, but they are not semantic inputs and do not abort
            # executable sibling targets.
            for target_id in query.blocked_primary_target_ids:
                rows.append({
                    "task": task,
                    "target_id": target_id,
                    "status": "UNRESOLVED_TARGET_NOT_EXECUTED",
                    "reason": "target_local_parser_preflight_block",
                })
            for target_id in query.review_target_ids:
                rows.append({
                    "task": task,
                    "target_id": target_id,
                    "status": "REVIEW_ONLY_NOT_EXECUTED",
                    "reason": "primary_for_matching_false",
                })
    elif task == "reconstruction":
        if query.fragments.empty:
            rows.append({
                "task": task, "target_id": "",
                "status": "BLOCKED_NO_FRAGMENTS",
                "reason": "query_fragments.csv contains no fragments",
            })
        else:
            for _, row in query.fragments.iterrows():
                rows.append({
                    "task": task,
                    "target_id": clean_text(row.get("fragment_id", "")),
                    "status": "READY",
                    "reason": "native query fragment available",
                })
    return rows


def _auto_task(query: QueryPackage) -> str:
    if not query.sentence_slots.empty:
        return "sentence_redaction"
    if not query.value_slots.empty:
        return "redaction"
    if not query.propositions.empty:
        return "claims"
    return "reconstruction"


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


def _generate_llm_digest(
    args: argparse.Namespace,
    evidence_payload: Mapping[str, Any],
    deterministic: str,
    output_dir: Path,
    *,
    pair_level: bool = False,
) -> Dict[str, Any]:
    """Compatibility entry point for digest-only retry and the coordinator."""
    return generate_optional_llm(
        enabled=bool(args.use_llm_digest),
        evidence_payload=evidence_payload,
        deterministic_digest=deterministic,
        model=args.llm_model,
        api_key_env=args.openai_api_key_env,
        timeout_seconds=args.llm_timeout_seconds,
        max_retries=args.llm_max_retries,
        max_output_tokens=args.llm_max_output_tokens,
        output_dir=output_dir,
        pair_level=pair_level,
    )


def _bounded_audit_exports(
    task: str,
    artifacts: Mapping[str, pd.DataFrame],
    evaluation: pd.DataFrame,
    row_limit_per_target: int,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Bound persisted audit rows without changing engine decisions.

    Engines and evidence projection retain their complete in-memory audits.
    The standard run ZIP persists selected/eligible/boundary rows plus a
    score-balanced sample.  This prevents one diagnostic CSV from dominating
    an otherwise small run.
    """
    audit_contract = {
        "reconstruction": (
            "fragment_document_evidence_v048.csv", "fragment_id"
        ),
        "redaction": ("redaction_candidate_audit_v048.csv", "slot_id"),
        "sentence_redaction": (
            "sentence_redaction_boundary_audit_v048.csv", "slot_id"
        ),
        "claims": (
            "claim_candidate_audit_v048.csv", "query_proposition_id"
        ),
        "contradictions": (
            "contradiction_candidate_audit_v048.csv",
            "query_proposition_id",
        ),
    }
    persisted = {
        name: frame.copy() for name, frame in artifacts.items()
    }
    summary_rows: List[Dict[str, Any]] = []
    if task not in audit_contract:
        return persisted, pd.DataFrame(summary_rows)
    filename, target_column = audit_contract[task]
    audit = artifacts.get(filename, pd.DataFrame())
    limit = max(25, int(row_limit_per_target))
    exported_parts: List[pd.DataFrame] = []
    evaluation_by_target = {
        clean_text(row.get("target_id", "")): row
        for row in evaluation.to_dict("records")
    }
    if audit.empty or target_column not in audit.columns:
        summary_rows.extend({
            "task": task,
            "artifact": filename,
            "target_id": target_id,
            "original_row_count": 0,
            "exported_row_count": 0,
            "omitted_row_count": 0,
            "row_limit_per_target": limit,
            "export_contract":
                "zero_candidate_target_accounted_explicitly",
        } for target_id in sorted(evaluation_by_target))
        return persisted, pd.DataFrame(summary_rows)
    for target_value, subset in audit.groupby(
        target_column, dropna=False, sort=True
    ):
        target_id = clean_text(target_value)
        group = subset.copy()
        finding = evaluation_by_target.get(target_id, {})
        important_docs = {
            clean_text(finding.get(field, ""))
            for field in (
                "selected_source_doc_id",
                "authoritative_source_doc_id",
                "primary_finding_source_doc_id",
                "nearest_alternative_doc_id",
                "decision_competitor_doc_id",
            )
            if clean_text(finding.get(field, ""))
        }
        doc_column = next(
            (
                column for column in (
                    "doc_id", "corpus_doc_id", "selected_doc_id"
                )
                if column in group.columns
            ),
            "",
        )
        score_column = next(
            (
                column for column in (
                    "final_score", "score", "retrieval_score"
                )
                if column in group.columns
            ),
            "",
        )
        class_column = next(
            (
                column for column in (
                    "primary_class", "candidate_relation",
                    "failed_gates", "semantic_status"
                )
                if column in group.columns
            ),
            "",
        )
        def ranked_indices(frame: pd.DataFrame) -> List[Any]:
            if score_column and score_column in frame.columns:
                frame = frame.sort_values(
                    score_column, ascending=False, kind="stable"
                )
            return frame.index.tolist()

        chosen_indices: List[Any] = []
        # The row that explains the decision is never displaced by a large
        # population of merely eligible rows.
        if "selected" in group.columns:
            chosen_indices.extend(ranked_indices(
                group[group["selected"].fillna(False).astype(bool)]
            ))
        if important_docs and doc_column:
            chosen_indices.extend(ranked_indices(
                group[
                    group[doc_column].fillna("").astype(str)
                    .isin(important_docs)
                ]
            ))
        if "eligible" in group.columns:
            chosen_indices.extend(ranked_indices(
                group[group["eligible"].fillna(False).astype(bool)]
            ))
        if class_column:
            for _class, class_rows in group.groupby(
                class_column, dropna=False, sort=True
            ):
                if score_column:
                    class_rows = class_rows.sort_values(
                        score_column, ascending=False, kind="stable"
                    )
                chosen_indices.extend(class_rows.head(12).index.tolist())
        remaining = group.drop(
            index=list(dict.fromkeys(chosen_indices)),
            errors="ignore",
        )
        if score_column:
            remaining = remaining.sort_values(
                score_column, ascending=False, kind="stable"
            )
        chosen_indices.extend(
            remaining.head(max(0, limit - len(set(chosen_indices))))
            .index.tolist()
        )
        chosen_indices = list(dict.fromkeys(chosen_indices))[:limit]
        exported = group.loc[chosen_indices]
        exported_parts.append(exported)
        summary_rows.append({
            "task": task,
            "artifact": filename,
            "target_id": target_id,
            "original_row_count": int(len(group)),
            "exported_row_count": int(len(exported)),
            "omitted_row_count": int(len(group) - len(exported)),
            "row_limit_per_target": limit,
            "export_contract": (
                "important_decision_rows_then_class_balanced_top_scores"
            ),
        })
    persisted[filename] = pd.concat(
        exported_parts, ignore_index=True
    ) if exported_parts else audit.iloc[0:0].copy()
    summarized_targets = {
        clean_text(row.get("target_id", ""))
        for row in summary_rows
    }
    summary_rows.extend({
        "task": task,
        "artifact": filename,
        "target_id": target_id,
        "original_row_count": 0,
        "exported_row_count": 0,
        "omitted_row_count": 0,
        "row_limit_per_target": limit,
        "export_contract":
            "zero_candidate_target_accounted_explicitly",
    } for target_id in sorted(
        set(evaluation_by_target) - summarized_targets
    ))
    return persisted, pd.DataFrame(summary_rows)


def _candidate_accounting(
    task: str,
    evaluation: pd.DataFrame,
    artifacts: Mapping[str, pd.DataFrame],
    persisted_artifacts: Mapping[str, pd.DataFrame],
    engine_elapsed: float,
) -> pd.DataFrame:
    audit_contract = {
        "reconstruction": (
            "fragment_document_evidence_v048.csv", "fragment_id"
        ),
        "redaction": ("redaction_candidate_audit_v048.csv", "slot_id"),
        "sentence_redaction": (
            "sentence_redaction_boundary_audit_v048.csv", "slot_id"
        ),
        "claims": (
            "claim_candidate_audit_v048.csv", "query_proposition_id"
        ),
        "contradictions": (
            "contradiction_candidate_audit_v048.csv",
            "query_proposition_id",
        ),
    }
    filename, target_column = audit_contract[task]
    audit = artifacts.get(filename, pd.DataFrame())
    rows: List[Dict[str, Any]] = []
    for finding in evaluation.to_dict("records"):
        target_id = clean_text(finding.get("target_id", ""))
        subset = audit
        if (
            not subset.empty
            and target_column in subset.columns
        ):
            subset = subset[
                subset[target_column].fillna("").astype(str).eq(target_id)
            ]
        rows.append({
            "task": task,
            "target_id": target_id,
            "audit_source": filename,
            "candidate_population_contract":
                "task_native_audit_rows_before_evidence_projection",
            "candidate_row_count": int(len(subset)),
            "eligible_candidate_count": int(
                subset.get("eligible", pd.Series(dtype=bool))
                .fillna(False).astype(bool).sum()
            ) if not subset.empty and "eligible" in subset.columns else 0,
            "exported_audit_row_count": int(len(
                persisted_artifacts.get(filename, pd.DataFrame())
                .loc[
                    lambda frame: (
                        frame[target_column].fillna("").astype(str)
                        .eq(target_id)
                        if (
                            not frame.empty
                            and target_column in frame.columns
                        )
                        else pd.Series(False, index=frame.index)
                    )
                ]
            )),
            "decision_competitor_present": clean_text(
                finding.get("decision_competitor_score", "")
            ) != "",
            "nearest_alternative_present": clean_text(
                finding.get("nearest_alternative_score", "")
            ) != "",
            "engine_elapsed_seconds": float(engine_elapsed),
            "candidate_limit_state": "WITHIN_CONFIGURED_LIMIT",
        })
    return pd.DataFrame(rows)


def _candidate_route_document_accounting(
    task: str,
    artifacts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate complete in-memory candidates by target, route, and document."""
    audit_contract = {
        "reconstruction":
            ("fragment_document_evidence_v048.csv", "fragment_id"),
        "redaction": ("redaction_candidate_audit_v048.csv", "slot_id"),
        "sentence_redaction":
            ("sentence_redaction_boundary_audit_v048.csv", "slot_id"),
        "claims":
            ("claim_candidate_audit_v048.csv", "query_proposition_id"),
        "contradictions": (
            "contradiction_candidate_audit_v048.csv",
            "query_proposition_id",
        ),
    }
    filename, target_column = audit_contract[task]
    audit = artifacts.get(filename, pd.DataFrame()).copy()
    columns = [
        "task", "target_id", "retrieval_route", "doc_id",
        "candidate_row_count", "unique_candidate_count",
        "deduplicated_row_count", "eligible_candidate_count",
        "selected_candidate_count", "accounting_scope",
    ]
    if audit.empty or target_column not in audit.columns:
        return pd.DataFrame(columns=columns)
    doc_column = next(
        (
            column for column in (
                "corpus_doc_id", "doc_id", "selected_doc_id",
            )
            if column in audit.columns
        ),
        "",
    )
    route_column = next(
        (
            column for column in (
                "retrieval_routes", "provenance_route",
                "candidate_relation", "primary_class",
            )
            if column in audit.columns
        ),
        "",
    )
    identity_columns = [
        column for column in (
            "corpus_source_table", "corpus_row_id", "source_table",
            "source_row_id", "raw_chunk_id", "candidate",
            "selected_sentence", "pseudo_component_ids",
        )
        if column in audit.columns
    ]
    working = pd.DataFrame({
        "target_id": audit[target_column].fillna("").astype(str),
        "retrieval_route": (
            audit[route_column].fillna("").astype(str)
            if route_column else pd.Series(
                ["TASK_NATIVE"] * len(audit), index=audit.index
            )
        ),
        "doc_id": (
            audit[doc_column].fillna("").astype(str)
            if doc_column else pd.Series(
                [""] * len(audit), index=audit.index
            )
        ),
        "eligible": (
            audit["eligible"].fillna(False).astype(bool)
            if "eligible" in audit.columns
            else pd.Series([False] * len(audit), index=audit.index)
        ),
        "selected": (
            audit["selected"].fillna(False).astype(bool)
            if "selected" in audit.columns
            else pd.Series([False] * len(audit), index=audit.index)
        ),
    })
    if identity_columns:
        working["candidate_identity"] = audit[
            identity_columns
        ].fillna("").astype(str).agg("|".join, axis=1)
    else:
        working["candidate_identity"] = audit.index.astype(str)
    rows = []
    for (target, route, doc_id), group in working.groupby(
        ["target_id", "retrieval_route", "doc_id"],
        dropna=False,
        sort=True,
    ):
        unique_count = int(group["candidate_identity"].nunique())
        rows.append({
            "task": task,
            "target_id": clean_text(target),
            "retrieval_route": clean_text(route) or "TASK_NATIVE",
            "doc_id": clean_text(doc_id),
            "candidate_row_count": int(len(group)),
            "unique_candidate_count": unique_count,
            "deduplicated_row_count": int(len(group) - unique_count),
            "eligible_candidate_count": int(group["eligible"].sum()),
            "selected_candidate_count": int(group["selected"].sum()),
            "accounting_scope":
                "complete_in_memory_pre_export_audit",
        })
    return pd.DataFrame(rows, columns=columns)


def _normalize_score_and_authority_contract(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    frame = evaluation.copy()
    for index, row in frame.iterrows():
        task = clean_text(row.get("task", "")).lower()
        authority = (
            clean_text(row.get("primary_finding_source_doc_id", ""))
            if task == "contradictions"
            else clean_text(row.get("selected_source_doc_id", ""))
        )
        frame.at[index, "authoritative_source_doc_id"] = authority
        best_text = clean_text(row.get("best_score", ""))
        competitor_text = clean_text(
            row.get("decision_competitor_score", "")
        ) or clean_text(row.get("runner_up_score", ""))
        if not competitor_text:
            frame.at[index, "decision_competitor_score"] = None
            frame.at[index, "runner_up_score"] = None
            frame.at[index, "winner_margin"] = None
            frame.at[index, "margin_status"] = (
                "NOT_APPLICABLE_NO_COMPARABLE_COMPETITOR"
            )
            continue
        try:
            best = float(best_text)
            competitor = float(competitor_text)
        except (TypeError, ValueError):
            frame.at[index, "winner_margin"] = None
            frame.at[index, "margin_status"] = (
                "INVALID_NON_NUMERIC_SCORE_CONTRACT"
            )
            continue
        frame.at[index, "decision_competitor_score"] = competitor
        frame.at[index, "runner_up_score"] = competitor
        frame.at[index, "winner_margin"] = best - competitor
        frame.at[index, "margin_status"] = (
            "APPLICABLE_COMPARABLE_COMPETITOR"
        )
    return frame


def _write_logical_manifest(
    output_dir: Path,
    task_specific_name: str,
    task_artifacts: Iterable[str],
) -> None:
    specifications = [
        ("task_evaluation", "task_evaluation_v048.csv", True),
        ("task_specific_evaluation", task_specific_name, True),
        ("deterministic_digest", "semantic_match_task_digest_v048.md", True),
        (
            "deterministic_digest_v0413",
            "semantic_match_task_digest_v0413.md",
            True,
        ),
        ("task_evidence_units", "task_evidence_units_v0413.json", True),
        (
            "task_evidence_units_schema",
            "task_evidence_units_schema_v0413.json",
            True,
        ),
        (
            "task_evidence_units_summary",
            "task_evidence_units_v0413.csv",
            True,
        ),
        (
            "reconstruction_candidate_passages",
            "reconstruction_candidate_passages_v0413.csv",
            True,
        ),
        ("run_manifest", "semantic_match_run_manifest_v048.json", True),
        (
            "semantic_resource_manifest",
            "semantic_resource_manifest_v0420.json",
            True,
        ),
        (
            "corpus_entity_resource_manifest",
            "corpus_entity_resource_manifest_v0420.json",
            True,
        ),
        (
            "resolved_decision_profile",
            "resolved_decision_profile_v0420.json",
            True,
        ),
        (
            "decision_profile_audit",
            "decision_profile_audit_v0420.csv",
            True,
        ),
        (
            "decision_display",
            "decision_display_v0420.csv",
            True,
        ),
        ("output_schema_registry", "output_schema_registry_v048.json", True),
        ("task_preflight", "task_preflight_v048.csv", True),
        ("exclusion_audit", "exclusion_audit_v048.csv", False),
        ("performance_timing", "performance_timing_v048.csv", False),
        (
            "candidate_accounting",
            "candidate_accounting_v0420.csv",
            True,
        ),
        (
            "candidate_route_document_accounting",
            "candidate_route_document_accounting_v0420.csv",
            True,
        ),
        (
            "audit_export_summary",
            "audit_export_summary_v0420.csv",
            True,
        ),
        ("progress", "progress_v048.jsonl", False),
        ("pair_comparison", "pair_comparison_v048.csv", True),
        ("llm_digest_state", "llm_digest_state_v048.json", False),
        ("llm_digest_state_v0413", "llm_digest_state_v0413.json", False),
        ("optional_llm_digest", "semantic_match_digest_llm_v048.md", False),
        (
            "optional_llm_digest_v0413",
            "semantic_match_task_digest_llm_v0413.md",
            False,
        ),
        ("llm_digest_input_v0413", "llm_digest_input_v0413.json", False),
        ("task_visualization", "task_visualization_v048.html", False),
        ("task_visualization_data", "task_visualization_data_v048.json", False),
    ]
    specifications.extend(
        ("task_audit:" + filename, filename, False)
        for filename in sorted(task_artifacts)
    )
    entries: Dict[str, Dict[str, Any]] = {}
    for logical_name, filename, authoritative in specifications:
        path = output_dir / filename
        entries[logical_name] = {
            "physical_filename": filename,
            "authoritative": authoritative,
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256_file(path) if path.is_file() else "",
        }
    atomic_write_json(
        output_dir / "logical_output_manifest_v048.json",
        {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "outputs": entries,
            "compatibility_outputs": [],
        },
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native Matryoshka Semantic Matcher v0.4.20"
    )
    parser.add_argument("--extraction-dir", required=True)
    parser.add_argument("--query-parse-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--task-mode",
        default="auto",
        choices=[
            "auto", "reconstruction", "redaction", "sentence_redaction",
            "claims", "contradictions",
        ],
    )
    parser.add_argument("--exclude-doc-ids", default="")
    parser.add_argument("--exclude-doc-ids-file", default="")
    parser.add_argument("--target-doc-ids", default="")
    parser.add_argument(
        "--target-doc-policy", choices=["warn", "exclude", "include"], default="warn"
    )
    parser.add_argument("--pair-id", default="")
    parser.add_argument(
        "--pair-role",
        choices=["standalone", "baseline", "holdout"],
        default="standalone",
    )
    parser.add_argument("--paired-run-dir", default="")
    parser.add_argument(
        "--defer-pair-finalization",
        action="store_true",
        help=(
            "Leave pair reconciliation and final-byte sealing to the "
            "coordinator. Required for coordinated pairs."
        ),
    )
    parser.add_argument("--staged-top-docs", type=int, default=150)
    parser.add_argument(
        "--retrieval-mode",
        choices=["staged", "exhaustive"],
        default="staged",
    )
    parser.add_argument(
        "--raw-use-mode",
        choices=["validate", "discover"],
        default="validate",
    )
    parser.add_argument("--raw-validation-finalists", type=int, default=20)
    parser.add_argument(
        "--audit-row-limit-per-target",
        type=int,
        default=250,
        help=(
            "Maximum persisted rows per target for large task-native audit "
            "CSVs. Complete in-memory audits still drive decisions and "
            "candidate accounting."
        ),
    )
    parser.add_argument(
        "--semantic-resources",
        default="",
        help=(
            "Optional versioned semantic-resource JSON merged with the "
            "packaged core resource. Selection never depends on corpus names."
        ),
    )
    parser.add_argument(
        "--corpus-entity-resource",
        default="",
        help=(
            "Optional versioned corpus entity/alias resource v1. It is "
            "selected only by this explicit path, participates in cache "
            "identity, and may not contain answer-key fields."
        ),
    )
    parser.add_argument(
        "--contradiction-propositions-per-class-per-document",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--contradiction-raw-sentences-per-class-per-document",
        type=int,
        default=4,
    )
    parser.add_argument("--max-candidates-per-target", type=int, default=100000)
    parser.add_argument("--max-elapsed-seconds", type=float, default=0.0)
    parser.add_argument(
        "--shared-source-cache",
        default="",
        help=(
            "Trusted coordinator-generated v0.4.20 source-table cache. "
            "Exclusions are still applied before task indexes are built."
        ),
    )
    parser.add_argument(
        "--task-feature-cache",
        default="",
        help=(
            "Validated query-independent v0.4.20 task feature cache. "
            "Arm exclusions are applied while the cached view is created."
        ),
    )
    parser.add_argument(
        "--decision-profile",
        default="",
        help=(
            "Optional matryoshka_decision_profile_v1 JSON. Thresholds may be "
            "overridden; score-construction weights remain internal and fixed."
        ),
    )
    parser.add_argument(
        "--minimum-score", type=float, default=None,
        help="Optional compatibility override applied after the decision profile.",
    )
    parser.add_argument(
        "--minimum-margin", type=float, default=None,
        help="Optional compatibility override applied after the decision profile.",
    )
    parser.add_argument("--use-llm-digest", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-max-output-tokens", type=int, default=1600)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(
            "Unsupported arguments (nothing is silently ignored): "
            + " ".join(unknown)
        )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    control = RunControl(
        output_dir,
        max_candidates_per_target=args.max_candidates_per_target,
        max_elapsed_seconds=args.max_elapsed_seconds,
    )
    control.emit("run_started", matcher_version=MATCHER_VERSION)
    status = "FAILED"
    try:
        decision_profile = load_decision_profile(
            Path(args.decision_profile)
            if clean_text(args.decision_profile)
            else None,
            minimum_score_override=args.minimum_score,
            minimum_margin_override=args.minimum_margin,
        )
        decision_manifest = decision_profile.manifest()
        atomic_write_json(
            output_dir / "resolved_decision_profile_v0420.json",
            decision_manifest,
        )
        decision_audit = pd.DataFrame([
            {
                "parameter": name,
                "effective_value": value,
                "packaged_default_value": DEFAULT_ENGINE_THRESHOLDS[name],
                "source": decision_profile.threshold_sources[name],
                "changed_from_packaged_default": (
                    float(value) != float(DEFAULT_ENGINE_THRESHOLDS[name])
                ),
                "parameter_kind": "engine_threshold",
            }
            for name, value in sorted(decision_profile.thresholds.items())
        ] + [
            {
                "parameter": name,
                "effective_value": value,
                "packaged_default_value": DEFAULT_DISPLAY_POLICY[name],
                "source": "decision_profile_display_policy",
                "changed_from_packaged_default": (
                    float(value) != float(DEFAULT_DISPLAY_POLICY[name])
                ),
                "parameter_kind": "display_policy",
            }
            for name, value in sorted(decision_profile.display_policy.items())
        ])
        write_csv_with_schema(
            output_dir / "decision_profile_audit_v0420.csv",
            decision_audit,
            list(decision_audit.columns),
        )
        query = QueryPackage.load(
            Path(args.query_parse_dir), task=args.task_mode
        )
        task = _auto_task(query) if args.task_mode == "auto" else args.task_mode
        corpus_entity_resource = (
            load_corpus_entity_resource(
                Path(args.corpus_entity_resource)
            )
            if clean_text(args.corpus_entity_resource)
            else None
        )
        semantic_resources = load_semantic_resources(
            Path(args.semantic_resources)
            if clean_text(args.semantic_resources)
            else None,
            corpus_entity_resource=corpus_entity_resource,
        )
        atomic_write_json(
            output_dir / "semantic_resource_manifest_v0420.json",
            semantic_resources.manifest(),
        )
        corpus_resource_manifest = (
            corpus_entity_resource.manifest()
            if corpus_entity_resource is not None
            else {
                "schema_version":
                    "matryoshka_corpus_entity_alias_resource_v1",
                "load_status": "NOT_CONFIGURED",
                "configured_path": "",
                "size_bytes": 0,
                "sha256": "",
                "fingerprint": "",
                "entity_count": 0,
            }
        )
        atomic_write_json(
            output_dir / "corpus_entity_resource_manifest_v0420.json",
            corpus_resource_manifest,
        )
        excluded = _parse_doc_ids(args.exclude_doc_ids)
        if args.exclude_doc_ids_file:
            exclusion_path = Path(args.exclude_doc_ids_file)
            if not exclusion_path.is_file():
                raise FileNotFoundError(
                    "Exclusion file not found: " + str(exclusion_path)
                )
            excluded.update(_parse_doc_ids(exclusion_path.read_text()))
        targets = _parse_doc_ids(args.target_doc_ids)
        if args.target_doc_policy == "exclude":
            excluded.update(targets)

        configuration = {
            **vars(args),
            "task_mode": task,
            "exclude_doc_ids": sorted(excluded),
            "target_doc_ids": sorted(targets),
            "semantic_resource_fingerprint": semantic_resources.fingerprint,
            "corpus_entity_resource_fingerprint":
                clean_text(corpus_resource_manifest.get(
                    "fingerprint", ""
                )),
            "decision_profile": decision_manifest,
        }
        config_hash = configuration_hash(configuration)
        invariant_hash = pair_invariant_hash(configuration)
        preflight = _preflight(task, query)
        write_csv_with_schema(
            output_dir / "task_preflight_v048.csv",
            pd.DataFrame(preflight),
            ["task", "target_id", "status", "reason"],
        )
        blocked = [
            row for row in preflight
            if clean_text(row.get("status", "")).startswith("BLOCKED")
        ]
        if blocked:
            raise ValueError(
                "Task preflight blocked: "
                + "; ".join(row["status"] for row in blocked)
            )

        corpus_started = time.perf_counter()
        corpus = CorpusIndex.build(
            Path(args.extraction_dir),
            excluded,
            control,
            task=task,
            shared_source_cache=(
                Path(args.shared_source_cache)
                if clean_text(args.shared_source_cache)
                else None
            ),
            task_feature_cache=(
                Path(args.task_feature_cache)
                if clean_text(args.task_feature_cache)
                else None
            ),
            semantic_resource_fingerprint=semantic_resources.fingerprint,
            semantic_resources=semantic_resources,
        )
        corpus_elapsed = time.perf_counter() - corpus_started
        corpus.audit_frame().to_csv(
            output_dir / "exclusion_audit_v048.csv", index=False
        )
        engine_config = EngineConfig(
            **dict(decision_profile.thresholds),
            staged_top_docs=(
                len(corpus.doc_ids)
                if args.retrieval_mode == "exhaustive"
                else args.staged_top_docs
            ),
            retrieval_mode=args.retrieval_mode,
            raw_use_mode=args.raw_use_mode,
            raw_validation_finalists=args.raw_validation_finalists,
            contradiction_propositions_per_class_per_document=max(
                1,
                args.contradiction_propositions_per_class_per_document,
            ),
            contradiction_raw_sentences_per_class_per_document=max(
                0,
                args.contradiction_raw_sentences_per_class_per_document,
            ),
        )
        engine_started = time.perf_counter()
        engine = ENGINE_TYPES[task]()
        result = engine.run(EngineContext(
            corpus=corpus,
            query=query,
            output_dir=output_dir,
            control=control,
            config=engine_config,
            semantic_resources=semantic_resources,
        ))
        engine_elapsed = time.perf_counter() - engine_started
        evaluation = _normalize_score_and_authority_contract(
            result.evaluation
        )
        persisted_artifacts, audit_export_summary = (
            _bounded_audit_exports(
                task,
                result.artifacts,
                evaluation,
                args.audit_row_limit_per_target,
            )
        )
        persisted_artifacts["audit_export_summary_v0420.csv"] = (
            audit_export_summary
        )
        for filename, frame in persisted_artifacts.items():
            columns = (
                list(frame.columns)
                if len(frame.columns)
                else ARTIFACT_EMPTY_COLUMNS.get(filename, ["schema_empty"])
            )
            write_csv_with_schema(output_dir / filename, frame, columns)
        for column in TASK_EVALUATION_COLUMNS:
            if column not in evaluation.columns:
                evaluation[column] = ""
        evaluation["query_parse_sha256"] = query.parse_sha256
        evaluation["configuration_sha256"] = config_hash
        evaluation["pair_invariant_sha256"] = invariant_hash
        pair_status = (
            "STANDALONE" if args.pair_role == "standalone" else "PAIR_INCOMPLETE"
        )
        evaluation["pair_comparability_status"] = pair_status
        display_rows: List[Dict[str, Any]] = []
        for row in evaluation.to_dict("records"):
            score = row.get("best_score", "")
            runner = row.get("runner_up_score", "")
            margin = row.get("winner_margin", "")
            semantic_status = clean_text(row.get("semantic_status", ""))
            task_status = clean_text(row.get("task_execution_status", ""))
            failed_gates = clean_text(row.get("failed_gates", ""))
            eligible = not (
                "NO_ELIGIBLE" in semantic_status.upper()
                or "REJECT" in semantic_status.upper()
                or "FAILED" in task_status.upper()
                or bool(failed_gates)
            )
            display_rows.append({
                "task": task,
                "target_id": clean_text(row.get("target_id", "")),
                "semantic_eligibility_state": (
                    "ELIGIBLE" if eligible
                    else "INELIGIBLE_SEMANTIC_MISMATCH"
                ),
                "score": score,
                "runner_up_score": runner,
                "winner_margin": margin,
                "operational_decision_band": operational_band(
                    eligible=eligible,
                    score=score,
                    margin=margin,
                    runner_present=clean_text(runner) != "",
                    mode=decision_profile.decision_mode,
                    display_policy=decision_profile.display_policy,
                ),
                "decision_profile_id": decision_profile.profile_id,
                "decision_profile_sha256":
                    decision_profile.configured_sha256,
                "decision_mode": decision_profile.decision_mode,
                "display_lower_threshold":
                    decision_profile.display_policy["lower_threshold"],
                "display_upper_threshold":
                    decision_profile.display_policy["upper_threshold"],
                "display_minimum_margin":
                    decision_profile.display_policy["minimum_margin"],
                "advisory_only": True,
            })
        decision_display = pd.DataFrame(display_rows)
        write_csv_with_schema(
            output_dir / "decision_display_v0420.csv",
            decision_display,
            list(decision_display.columns) or [
                "task", "target_id", "operational_decision_band"
            ],
        )
        write_csv_with_schema(
            output_dir / "task_evaluation_v048.csv",
            evaluation,
            TASK_EVALUATION_COLUMNS,
        )
        task_specific_name = {
            "reconstruction": "reconstruction_evaluation_v048.csv",
            "redaction": "redaction_evaluation_v048.csv",
            "sentence_redaction": "sentence_redaction_evaluation_v048.csv",
            "claims": "claim_evaluation_v048.csv",
            "contradictions": "contradiction_evaluation_v048.csv",
        }[task]
        write_csv_with_schema(
            output_dir / task_specific_name,
            evaluation,
            TASK_EVALUATION_COLUMNS,
        )
        candidate_accounting = _candidate_accounting(
            task,
            evaluation,
            result.artifacts,
            persisted_artifacts,
            engine_elapsed,
        )
        write_csv_with_schema(
            output_dir / "candidate_accounting_v0420.csv",
            candidate_accounting,
            list(candidate_accounting.columns) or [
                "task", "target_id", "candidate_row_count",
            ],
        )
        route_document_accounting = (
            _candidate_route_document_accounting(
                task, result.artifacts
            )
        )
        write_csv_with_schema(
            output_dir
            / "candidate_route_document_accounting_v0420.csv",
            route_document_accounting,
            list(route_document_accounting.columns),
        )
        evidence_started = time.perf_counter()
        evidence_payload = build_task_evidence_units(
            task=task,
            query=query,
            evaluation=evaluation,
            artifacts=result.artifacts,
            pair_role=args.pair_role,
            pair_status=pair_status,
            max_alternatives=3,
        )
        evidence_outputs = write_evidence_outputs(
            output_dir, evidence_payload
        )
        evidence_elapsed = time.perf_counter() - evidence_started
        _write_task_visualization(
            output_dir, task, evaluation, result.artifacts
        )
        digest_started = time.perf_counter()
        digest, _digest_input = write_task_digest_bundle(
            output_dir, evidence_payload
        )
        digest_elapsed = time.perf_counter() - digest_started
        # Retain the established physical digest for downstream readers while
        # making the v0.4.13 file the authoritative explanatory contract.
        (output_dir / "semantic_match_task_digest_v048.md").write_text(
            digest, encoding="utf-8"
        )
        llm_started = time.perf_counter()
        llm_state = _generate_llm_digest(
            args, evidence_payload, digest, output_dir
        )
        llm_elapsed = time.perf_counter() - llm_started
        # Compatibility state mirrors the new five-dimensional state.
        atomic_write_json(output_dir / "llm_digest_state_v048.json", llm_state)

        timings = pd.DataFrame([
            {
                "stage": "corpus_single_load_and_index",
                "elapsed_seconds": corpus_elapsed,
                "cache_state": (
                    "task_feature_cache"
                    if clean_text(args.task_feature_cache)
                    else "shared_source_cache"
                    if clean_text(args.shared_source_cache)
                    else "native_single_load"
                ),
            },
            {
                "stage": "task_engine",
                "elapsed_seconds": engine_elapsed,
                "cache_state": "immutable_index",
            },
            {
                "stage": "task_evidence_units",
                "elapsed_seconds": evidence_elapsed,
                "cache_state": evidence_payload["target_coverage_state"],
            },
            {
                "stage": "deterministic_digest",
                "elapsed_seconds": digest_elapsed,
                "cache_state": "authoritative",
            },
            {
                "stage": "llm_digest",
                "elapsed_seconds": llm_elapsed,
                "cache_state": llm_state["state"],
            },
            {
                "stage": "total_v0413",
                "elapsed_seconds": time.perf_counter() - started,
                "cache_state": "native",
            },
        ])
        timings.to_csv(output_dir / "performance_timing_v048.csv", index=False)
        schema_registry = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "authoritative_outputs": {
                "task_evaluation": {
                    "filename": "task_evaluation_v048.csv",
                    "required_columns": TASK_EVALUATION_COLUMNS,
                },
                "task_specific_evaluation": {
                    "filename": task_specific_name,
                    "required_columns": TASK_EVALUATION_COLUMNS,
                },
                "deterministic_digest":
                    "semantic_match_task_digest_v0413.md",
                "task_evidence_units": {
                    "filename": "task_evidence_units_v0413.json",
                    "schema":
                        "matryoshka_task_evidence_units_v0_4_13",
                },
                "run_manifest": "semantic_match_run_manifest_v048.json",
                "pair_comparison": "pair_comparison_v048.csv",
                "resolved_decision_profile":
                    "resolved_decision_profile_v0420.json",
                "decision_profile_audit":
                    "decision_profile_audit_v0420.csv",
                "decision_display": "decision_display_v0420.csv",
            },
            "task_artifacts": {
                filename: list(frame.columns)
                if len(frame.columns)
                else ARTIFACT_EMPTY_COLUMNS.get(filename, ["schema_empty"])
                for filename, frame in sorted(result.artifacts.items())
            },
        }
        atomic_write_json(
            output_dir / "output_schema_registry_v048.json", schema_registry
        )
        _write_logical_manifest(output_dir, task_specific_name, result.artifacts)
        manifest = {
            "coordinated_release_version": RELEASE_VERSION,
            "parser_implementation_version": PARSER_VERSION,
            "matcher_implementation_version": MATCHER_VERSION,
            "digest_builder_implementation_version":
                DIGEST_BUILDER_VERSION,
            "workbench_implementation_version": WORKBENCH_VERSION,
            "requirements_version": REQUIREMENTS_VERSION,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "architecture": (
                "native_v0420_primary_only_target_local_preflight"
            ),
            "legacy_matcher_invoked": False,
            "task_mode": task,
            "engine_metadata": dict(result.metadata),
            "query_parse_dir": str(query.root),
            "query_parse_sha256": query.parse_sha256,
            "configuration": configuration,
            "configuration_sha256": config_hash,
            "pair_invariant_sha256": invariant_hash,
            "pair_id": args.pair_id,
            "pair_role": args.pair_role,
            "pair_status": pair_status,
            "corpus_fingerprint": corpus.fingerprint,
            "semantic_resources": semantic_resources.manifest(),
            "corpus_entity_resource": corpus_resource_manifest,
            "decision_profile": decision_manifest,
            "source_load_statistics": {
                table: dict(statistics)
                for table, statistics in corpus.load_statistics.items()
            },
            "effective_excluded_doc_ids": sorted(excluded),
            "task_evidence_unit_inventory": {
                name: path.name
                for name, path in evidence_outputs.items()
            },
            "requested_target_count":
                evidence_payload["requested_target_count"],
            "emitted_target_count":
                evidence_payload["emitted_target_count"],
            "target_coverage_state":
                evidence_payload["target_coverage_state"],
            "llm_digest_state": llm_state,
            "completed_at": utc_now_iso(),
        }
        atomic_write_json(
            output_dir / "semantic_match_run_manifest_v048.json", manifest
        )
        status = "COMPLETED"
        contract = {
            "status": status,
            "matcher_version": MATCHER_VERSION,
            "requirements_version": REQUIREMENTS_VERSION,
            "resolved_output_dir": str(output_dir),
            "task_evaluation_rows": len(evaluation),
            "pair_status": pair_status,
            "required_artifacts": [
                "task_evaluation_v048.csv",
                "semantic_match_run_manifest_v048.json",
                "semantic_match_task_digest_v0413.md",
                "task_evidence_units_v0413.json",
                "task_evidence_units_schema_v0413.json",
                "output_schema_registry_v048.json",
                "semantic_resource_manifest_v0420.json",
                "corpus_entity_resource_manifest_v0420.json",
                "candidate_route_document_accounting_v0420.csv",
                "resolved_decision_profile_v0420.json",
                "decision_profile_audit_v0420.csv",
                "decision_display_v0420.csv",
            ],
        }
        atomic_write_json(output_dir / "matcher_output_contract_v048.json", contract)
        control.emit("run_completed", task=task, row_count=len(evaluation))

        if (
            args.pair_role in {"baseline", "holdout"}
            and args.paired_run_dir
            and Path(args.paired_run_dir).is_dir()
            and not args.defer_pair_finalization
        ):
            paired = Path(args.paired_run_dir).resolve()
            baseline_dir = output_dir if args.pair_role == "baseline" else paired
            holdout_dir = paired if args.pair_role == "baseline" else output_dir
            if (
                (baseline_dir / "semantic_match_run_manifest_v048.json").exists()
                and (holdout_dir / "semantic_match_run_manifest_v048.json").exists()
            ):
                pair_result = reconcile_pair(
                    baseline_dir,
                    holdout_dir,
                    args.pair_id
                    or f"pair-{query.parse_sha256[:12]}",
                )
                contract["pair_status"] = pair_result["pair_status"]
                atomic_write_json(
                    output_dir / "matcher_output_contract_v048.json", contract
                )

        contract["finalization_state"] = (
            "DEFERRED_TO_COORDINATOR"
            if args.defer_pair_finalization
            else "READY_FOR_FINAL_BYTE_SEAL"
        )
        contract["final_output_manifest"] = (
            "" if args.defer_pair_finalization
            else "final_output_manifest_v0420.json"
        )
        atomic_write_json(
            output_dir / "matcher_output_contract_v048.json", contract
        )
        final_manifest = safe_json(
            output_dir / "semantic_match_run_manifest_v048.json"
        )
        final_manifest["finalization_contract"] = (
            "deferred_to_coordinator"
            if args.defer_pair_finalization
            else "matcher_single_run_final_byte_seal"
        )
        final_manifest["final_output_manifest"] = (
            "" if args.defer_pair_finalization
            else "final_output_manifest_v0420.json"
        )
        final_manifest["output_inventory"] = output_inventory(output_dir)
        atomic_write_json(
            output_dir / "semantic_match_run_manifest_v048.json", final_manifest
        )
        _write_logical_manifest(output_dir, task_specific_name, result.artifacts)
        refresh_logical_output_manifest(output_dir)
        if not args.defer_pair_finalization:
            sealed = finalize_output_manifest(output_dir)
            validation = validate_output_manifest(output_dir, sealed)
            atomic_write_json(
                output_dir
                / "final_output_manifest_validation_v0420.json",
                validation,
            )
            if validation["state"] != "PASS":
                raise RuntimeError(
                    "Final output manifest failed byte authentication."
                )
        print(json.dumps(contract, indent=2))
        return 0
    except CancelledError as exc:
        status = "CANCELLED"
        control.emit("run_cancelled", error=str(exc))
        atomic_write_json(
            output_dir / "matcher_output_contract_v048.json",
            {"status": status, "error": str(exc), "matcher_version": MATCHER_VERSION},
        )
        return 130
    except (WorkLimitError, Exception) as exc:
        status = "FAILED_BOUNDED" if isinstance(exc, WorkLimitError) else "FAILED"
        traceback_text = traceback.format_exc()
        control.emit(
            "run_failed", status=status, error_type=type(exc).__name__, error=str(exc)
        )
        atomic_write_json(
            output_dir / "matcher_output_contract_v048.json",
            {
                "status": status,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback_text,
                "matcher_version": MATCHER_VERSION,
            },
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
