"""Five-task, fragment-grounded evidence-unit contract for v0.4.13.

The task engines remain authoritative for matching and classification.  This
module projects their native audit tables into one bounded, provenance-rich
schema used by deterministic digests, Optional LLM input, Workbench cards, and
automation.  It never changes a score, gate, selected source, selected value,
or semantic status.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .contracts import (
    DIGEST_BUILDER_VERSION,
    MATCHER_VERSION,
    OUTPUT_SCHEMA_VERSION,
)
from .semantics import atomic_write_json, clean_text, stable_hash, utc_now_iso


EVIDENCE_UNITS_SCHEMA = "matryoshka_task_evidence_units_v0_4_13"
EVIDENCE_FRAGMENT_SCHEMA = "matryoshka_evidence_fragment_v0_4_13"
EXCERPT_MAX_CHARACTERS = 1600
EXCERPT_CLIPPING_STRATEGY = (
    "matched_statement_first_then_symmetric_local_context"
)

TASK_AUDIT_CONTRACT: Dict[str, Tuple[str, str]] = {
    "reconstruction": ("fragment_source_passages_v048.csv", "fragment_id"),
    "redaction": ("redaction_candidate_audit_v048.csv", "slot_id"),
    "sentence_redaction": (
        "sentence_redaction_boundary_audit_v048.csv", "slot_id",
    ),
    "claims": ("claim_candidate_audit_v048.csv", "query_proposition_id"),
    "contradictions": (
        "contradiction_candidate_audit_v048.csv", "query_proposition_id",
    ),
}

ROLE_PRIORITY = {
    "selected": 0,
    "supporting": 1,
    "refuting": 2,
    "corroborating": 3,
    "partial": 4,
    "related": 5,
    "alternative": 6,
    "background": 7,
    "rejected": 8,
}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).lower() in {"1", "true", "yes", "y", "pass"}


def _number(value: Any) -> Any:
    if value is None or clean_text(value) == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return clean_text(value)


def _first(row: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = clean_text(row.get(name, ""))
        if value:
            return value
    return ""


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = clean_text(value)
    if not text:
        return default
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default
    return parsed


def _target_context(task: str, query: Any, target_id: str) -> Dict[str, Any]:
    """Return a task-specific, bounded target description from the frozen parse."""
    frame: pd.DataFrame
    id_columns: Sequence[str]
    if task == "reconstruction":
        frame = query.fragments
        id_columns = ("fragment_id",)
    elif task == "redaction":
        frame = query.value_slots
        id_columns = ("slot_label", "slot_id")
    elif task == "sentence_redaction":
        frame = query.sentence_slots
        id_columns = ("sentence_slot_label", "slot_id")
    else:
        frame = query.propositions
        id_columns = ("query_proposition_id", "proposition_id", "local_id")
    if frame is None or frame.empty:
        return {"target_id": target_id, "target_text": ""}
    selected = pd.DataFrame()
    for column in id_columns:
        if column in frame.columns:
            selected = frame[
                frame[column].fillna("").astype(str).eq(target_id)
            ]
            if len(selected):
                break
    if selected.empty:
        return {"target_id": target_id, "target_text": ""}
    row = selected.iloc[0].to_dict()
    common = {
        "target_id": target_id,
        "fragment_id": _first(row, ("fragment_id",)),
    }
    if task == "reconstruction":
        common.update({
            "target_text": _first(
                row, ("fragment_text", "query_fragment_text", "text")
            ),
        })
    elif task == "redaction":
        common.update({
            "target_text": _first(
                row, ("slot_frame", "fragment_text", "redacted_text")
            ),
            "left_context": _first(
                row,
                (
                    "local_context_before", "left_context",
                    "context_before", "before_context",
                ),
            ),
            "right_context": _first(
                row,
                (
                    "local_context_after", "right_context",
                    "context_after", "after_context",
                ),
            ),
            "expected_type": _first(
                row,
                (
                    "expected_type_hint", "explicit_type_hint",
                    "expected_type", "slot_type", "value_type",
                ),
            ),
            "explicit_type": _first(
                row, ("explicit_type_hint", "expected_type_hint")
            ),
            "grammatical_role": _first(
                row, ("grammatical_role",)
            ),
            "slot_frame": _first(row, ("slot_frame",)),
            "required_relation": _first(row, ("required_relation",)),
            "governing_predicate": _first(
                row, ("governing_predicate", "predicate")
            ),
            "governing_predicate_surface": _first(
                row,
                (
                    "governing_predicate_surface",
                    "governing_predicate",
                ),
            ),
            "governing_predicate_normalized": _first(
                row,
                (
                    "governing_predicate_normalized",
                    "governing_predicate",
                ),
            ),
            "required_anchor_fields": _json_value(
                row.get("required_anchor_fields_json", ""), []
            ),
            "slot_event_signature": _json_value(
                row.get("slot_event_signature_json", ""), {}
            ),
        })
    elif task == "sentence_redaction":
        common.update({
            "target_text": _first(
                row, ("slot_frame", "fragment_text", "redacted_text")
            ),
            "left_context": _first(
                row,
                (
                    "normalized_sentence_before", "sentence_before",
                    "left_context", "context_before",
                ),
            ),
            "right_context": _first(
                row,
                (
                    "normalized_sentence_after", "sentence_after",
                    "right_context", "context_after",
                ),
            ),
            "original_left_sentence": _first(
                row,
                ("original_sentence_before", "sentence_before"),
            ),
            "original_right_sentence": _first(
                row,
                ("original_sentence_after", "sentence_after"),
            ),
            "normalized_left_sentence": _first(
                row,
                ("normalized_sentence_before", "sentence_before"),
            ),
            "normalized_right_sentence": _first(
                row,
                ("normalized_sentence_after", "sentence_after"),
            ),
            "context_mode": _first(
                row, ("context_mode_detected",)
            ),
            "approx_tokens": _number(row.get("approx_tokens", "")),
            "expected_semantic_type": _first(
                row, ("expected_semantic_type",)
            ),
            "semantic_anchors": _json_value(
                row.get("semantic_anchors_json", ""), {}
            ),
        })
    else:
        common.update({
            "target_text": _first(
                row,
                (
                    "normalized_proposition_text", "proposition_text",
                    "claim_text", "text",
                ),
            ),
            "actor": _first(row, ("actor", "subject")),
            "predicate": _first(row, ("predicate", "relation")),
            "object": _first(row, ("object", "target")),
            "location": _first(row, ("location",)),
            "date": _first(row, ("date", "report_time")),
            "polarity": _first(row, ("polarity",)),
            "modality": _first(row, ("modality",)),
        })
    return common


def requested_target_ids(task: str, query: Any) -> List[str]:
    if task == "reconstruction":
        frame, columns = query.fragments, ("fragment_id",)
    elif task == "redaction":
        frame, columns = query.value_slots, ("slot_label", "slot_id")
    elif task == "sentence_redaction":
        frame, columns = (
            query.sentence_slots,
            ("sentence_slot_label", "slot_id"),
        )
    else:
        frame, columns = (
            query.propositions,
            ("query_proposition_id", "proposition_id", "local_id"),
        )
    if frame is None or frame.empty:
        return []
    column = next((name for name in columns if name in frame.columns), "")
    if not column:
        return []
    if task in {"claims", "contradictions"} and (
        "hypothesis_group_id" in frame.columns
    ):
        ordered: List[str] = []
        working = frame.copy()
        working["_hypothesis_confidence"] = pd.to_numeric(
            working.get(
                "hypothesis_confidence",
                pd.Series([1.0] * len(working)),
            ),
            errors="coerce",
        ).fillna(0.0)
        for _group, subset in working.groupby(
            "hypothesis_group_id", dropna=False, sort=False
        ):
            subset = subset.sort_values(
                ["_hypothesis_confidence", column],
                ascending=[False, True],
                kind="stable",
            )
            value = clean_text(subset.iloc[0].get(column, ""))
            if value:
                ordered.append(value)
        return ordered
    return [
        clean_text(value)
        for value in frame[column].tolist()
        if clean_text(value)
    ]


def _document_id(row: Mapping[str, Any]) -> str:
    return _first(row, ("doc_id", "corpus_doc_id", "candidate_doc_id")).upper()


def _passage_id(task: str, row: Mapping[str, Any], doc_id: str) -> str:
    if task == "sentence_redaction":
        chunk = _first(row, ("raw_chunk_id", "chunk_id"))
        sentence = _first(row, ("sentence_index",))
        return f"{doc_id}::{chunk or 'raw'}::sentence_{sentence or '0'}"
    if task == "reconstruction":
        cards = _json_value(row.get("raw_chunk_cards_json", ""), [])
        if cards and isinstance(cards[0], dict):
            chunk = clean_text(cards[0].get("chunk_id", "")) or "raw"
            sentence = clean_text(cards[0].get("sentence_index", "")) or "0"
            return f"{doc_id}::{chunk}::sentence_{sentence}"
        chunk = _first(row, ("raw_chunk_ids", "raw_chunk_id"))
        return f"{doc_id}::{chunk or 'passage'}"
    source_table = _first(
        row, ("source_table", "corpus_source_table")
    ) or "evidence"
    source_row = _first(
        row, ("source_row_id", "corpus_row_id", "row_id")
    ) or stable_hash(dict(row))[:12]
    return f"{doc_id}::{source_table}::{source_row}"


def _clip_excerpt(value: Any) -> str:
    text = clean_text(value)
    if len(text) <= EXCERPT_MAX_CHARACTERS:
        return text
    marker = " … [CLIPPED] … "
    available = EXCERPT_MAX_CHARACTERS - len(marker)
    left = available // 2
    return text[:left].rstrip() + marker + text[-(available - left):].lstrip()


def _excerpt(task: str, row: Mapping[str, Any]) -> str:
    if task == "reconstruction":
        return _clip_excerpt(_first(
            row,
            (
                "selected_raw_chunk", "raw_passage", "pseudo_evidence",
                "candidate_local_context",
            ),
        ))
    if task == "redaction":
        return _clip_excerpt(
            _first(row, ("evidence", "candidate_local_context"))
        )
    if task == "sentence_redaction":
        return _clip_excerpt(_first(
            row, ("candidate_local_context", "selected_sentence")
        ))
    return _clip_excerpt(_first(
        row,
        (
            "raw_sentence", "corpus_proposition_text", "candidate_relation",
            "evidence",
        ),
    ))


def _evidence_role(
    task: str,
    row: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> str:
    doc_id = _document_id(row)
    selected_source = clean_text(
        evaluation.get("selected_source_doc_id", "")
    ).upper()
    selected_text = clean_text(
        evaluation.get("selected_value_or_text", "")
    ).lower()
    if task == "reconstruction":
        evidence_set = clean_text(row.get("evidence_set", "")).lower()
        if doc_id and doc_id == selected_source:
            return "selected"
        if "minimal" in evidence_set or "collective" in evidence_set:
            return "supporting"
        if "extended" in evidence_set:
            return "corroborating"
        if "reject" in evidence_set:
            return "rejected"
        return "partial"
    if task == "redaction":
        candidate = clean_text(row.get("candidate", "")).lower()
        if (
            selected_source
            and doc_id == selected_source
            and selected_text
            and candidate == selected_text
        ):
            return "selected"
        return "alternative" if _truthy(row.get("eligible", False)) else "rejected"
    if task == "sentence_redaction":
        selected_sentence = clean_text(row.get("selected_sentence", "")).lower()
        if (
            selected_source
            and doc_id == selected_source
            and selected_text
            and selected_sentence == selected_text
        ):
            return "selected"
        return "alternative" if _truthy(row.get("eligible", False)) else "rejected"
    if task == "claims":
        relation = clean_text(row.get("candidate_relation", "")).upper()
        if relation == "DIRECT_SUPPORT":
            return "selected" if doc_id == selected_source else "supporting"
        if "CONTRADICT" in relation:
            return "refuting"
        if "RELATED" in relation:
            return "related"
        if "BACKGROUND" in relation:
            return "background"
        return "rejected"
    evidence_class = clean_text(row.get("primary_class", "")).upper()
    if evidence_class == "DIRECT_SUPPORT_SIDE_A":
        return "supporting"
    if any(
        token in evidence_class
        for token in ("OPPOSITION", "CONTRADICTION")
    ):
        return "refuting"
    if evidence_class in {
        "TEMPORAL_UPDATE_OR_SOURCE_REVISION", "QUALIFICATION",
        "SCOPE_OR_CARDINALITY_DIFFERENCE",
        "UNCERTAINTY_OR_INSUFFICIENT_INFORMATION",
    }:
        return "corroborating"
    if evidence_class == "RELATED_DIFFERENT_EVENT":
        return "related"
    if evidence_class == "BACKGROUND":
        return "background"
    return "rejected"


def _fragment(
    task: str,
    target_id: str,
    row: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> Dict[str, Any]:
    doc_id = _document_id(row)
    role = _evidence_role(task, row, evaluation)
    passed_gates = [
        key[:-10].upper()
        for key, value in row.items()
        if key.endswith("_gate_pass") and _truthy(value)
    ]
    failed_gates = [
        value for value in clean_text(row.get("failed_gates", "")).split(";")
        if value
    ]
    failed_gates.extend(
        key[:-10].upper()
        for key, value in row.items()
        if key.endswith("_gate_pass") and not _truthy(value)
    )
    matched = {
        key: clean_text(value)
        for key, value in row.items()
        if key.endswith("_state") and clean_text(value) in {"MATCH", "COMPATIBLE"}
    }
    missing = {
        key: clean_text(value)
        for key, value in row.items()
        if key.endswith("_state")
        and clean_text(value) in {"MISSING", "MISMATCH", "INCOMPATIBLE"}
    }
    covered_facts = _json_value(
        row.get("covered_facts_json", ""), []
    )
    missing_facts = _json_value(
        row.get("missing_critical_facts_json", ""), []
    )
    if covered_facts:
        matched["covered_facts"] = covered_facts
    if missing_facts:
        missing["critical_facts"] = missing_facts
    fragment = {
        "schema_version": EVIDENCE_FRAGMENT_SCHEMA,
        "target_id": target_id,
        "source_doc_id": doc_id,
        "passage_id": _passage_id(task, row, doc_id),
        "excerpt": _excerpt(task, row),
        "origin": (
            "raw_and_pseudo"
            if task == "reconstruction"
            and _first(row, ("raw_passage", "selected_raw_chunk"))
            and _first(row, ("pseudo_evidence",))
            else "raw"
            if task == "sentence_redaction"
            or _first(row, ("raw_sentence",))
            else "pseudo_or_structured"
        ),
        "evidence_role": role,
        "authoritative": role in {"selected", "supporting", "refuting"},
        "candidate_value_or_text": _clip_excerpt(_first(
            row,
            (
                "candidate", "selected_sentence", "corpus_proposition_text",
                "matched_proposition_text", "pseudo_evidence",
            ),
        )),
        "source_table": _first(
            row,
            (
                "source_table", "corpus_source_table",
                "matched_proposition_source_table",
            ),
        ),
        "source_row_id": _first(
            row,
            (
                "source_row_id", "corpus_row_id", "row_id",
                "matched_proposition_row_id",
            ),
        ),
        "retrieval_score": _number(row.get("retrieval_score", "")),
        "event_identity_score": _number(
            row.get("event_identity_score", "")
        ),
        "critical_fact_coverage": _number(
            row.get("critical_fact_coverage", "")
        ),
        "final_score": _number(row.get("final_score", "")),
        "passed_gates": sorted(set(passed_gates)),
        "failed_gates": sorted(set(failed_gates)),
        "matched_facts": matched,
        "missing_or_conflicting_facts": missing,
        "decision_reason": _first(
            row, ("decision_reason", "candidate_relation", "primary_class")
        ),
        "context": {
            "previous": _first(
                row, ("previous_raw_chunk", "left_neighbor")
            ),
            "selected": _first(
                row, ("selected_raw_chunk", "selected_sentence")
            ),
            "next": _first(row, ("next_raw_chunk", "right_neighbor")),
            "pseudo_excerpt": _clip_excerpt(
                _first(row, ("pseudo_evidence",))
            ),
        },
        "excerpt_character_count": len(_excerpt(task, row)),
        "excerpt_character_limit": EXCERPT_MAX_CHARACTERS,
        "excerpt_clipping_strategy": EXCERPT_CLIPPING_STRATEGY,
    }
    fragment["fragment_sha256"] = stable_hash(fragment)
    return fragment


def _bounded_fragments(
    fragments: Sequence[Dict[str, Any]],
    max_alternatives: int,
) -> List[Dict[str, Any]]:
    deduplicated: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for fragment in fragments:
        key = (
            clean_text(fragment.get("source_doc_id", "")).upper(),
            clean_text(fragment.get("excerpt", "")).casefold(),
        )
        previous = deduplicated.get(key)
        if previous is None or (
            -ROLE_PRIORITY.get(
                clean_text(fragment.get("evidence_role", "")), 99
            ),
            float(fragment.get("final_score", 0.0) or 0.0),
        ) > (
            -ROLE_PRIORITY.get(
                clean_text(previous.get("evidence_role", "")), 99
            ),
            float(previous.get("final_score", 0.0) or 0.0),
        ):
            deduplicated[key] = fragment
    ordered = sorted(
        deduplicated.values(),
        key=lambda item: (
            ROLE_PRIORITY.get(clean_text(item.get("evidence_role", "")), 99),
            -float(item.get("final_score", 0.0) or 0.0),
            clean_text(item.get("source_doc_id", "")),
            clean_text(item.get("passage_id", "")),
        ),
    )
    selected = [
        item for item in ordered
        if item["evidence_role"] in {"selected", "supporting", "refuting"}
    ][:3]
    alternatives = [
        item for item in ordered if item not in selected
    ][:max(0, max_alternatives)]
    result = selected + alternatives
    if not result and ordered:
        result = ordered[:1]
    return result


def build_task_evidence_units(
    *,
    task: str,
    query: Any,
    evaluation: pd.DataFrame,
    artifacts: Mapping[str, pd.DataFrame],
    pair_role: str,
    pair_status: str,
    max_alternatives: int = 3,
) -> Dict[str, Any]:
    requested = requested_target_ids(task, query)
    audit_name, target_column = TASK_AUDIT_CONTRACT[task]
    audit = artifacts.get(audit_name, pd.DataFrame())
    if task == "reconstruction" and not audit.empty:
        evidence = artifacts.get(
            "fragment_document_evidence_v048.csv", pd.DataFrame()
        )
        if (
            not evidence.empty
            and {"fragment_id", "doc_id"}.issubset(evidence.columns)
        ):
            audit = audit.merge(
                evidence,
                on=["fragment_id", "doc_id"],
                how="left",
                suffixes=("", "_evidence"),
            )
    evaluation_map = {
        clean_text(row.get("target_id", "")): row
        for row in evaluation.to_dict("records")
    }
    # The frozen parse is authoritative for requested coverage.  Include an
    # evaluation-only target defensively so defects are visible rather than
    # silently discarded.
    ordered_ids = list(dict.fromkeys(
        requested + [
            clean_text(value)
            for value in evaluation.get("target_id", pd.Series(dtype=str))
            if clean_text(value)
        ]
    ))
    units: List[Dict[str, Any]] = []
    for target_id in ordered_ids:
        finding = evaluation_map.get(target_id, {})
        subset = pd.DataFrame()
        if not audit.empty and target_column in audit.columns:
            subset = audit[
                audit[target_column].fillna("").astype(str).eq(target_id)
            ]
        fragments = [
            _fragment(task, target_id, row, finding)
            for row in subset.to_dict("records")
        ]
        bounded = _bounded_fragments(fragments, max_alternatives)
        unresolved = _first(
            finding, ("unresolved_reason", "failed_gates")
        )
        units.append({
            "schema_version": EVIDENCE_UNITS_SCHEMA,
            "task": task,
            "target_id": target_id,
            "target": _target_context(task, query, target_id),
            "pair_role": pair_role,
            "pair_status": pair_status,
            "deterministic_finding": {
                "semantic_status": clean_text(
                    finding.get("semantic_status", "")
                ),
                "selected_value_or_text": clean_text(
                    finding.get("selected_value_or_text", "")
                ),
                "selected_source_doc_id": clean_text(
                    finding.get("selected_source_doc_id", "")
                ),
                "authoritative_source_doc_id": clean_text(
                    finding.get("authoritative_source_doc_id", "")
                ),
                "primary_finding": clean_text(
                    finding.get("primary_finding", "")
                ),
                "primary_finding_source_doc_id": clean_text(
                    finding.get("primary_finding_source_doc_id", "")
                ),
                "best_score": _number(finding.get("best_score", "")),
                "runner_up_score": _number(
                    finding.get("runner_up_score", "")
                ),
                "nearest_alternative_score": _number(
                    finding.get("nearest_alternative_score", "")
                ),
                "nearest_alternative_doc_id": clean_text(
                    finding.get("nearest_alternative_doc_id", "")
                ),
                "nearest_alternative_passage_id": clean_text(
                    finding.get("nearest_alternative_passage_id", "")
                ),
                "nearest_alternative_eligible": (
                    _truthy(finding.get("nearest_alternative_eligible"))
                    if clean_text(
                        finding.get("nearest_alternative_eligible", "")
                    ) else None
                ),
                "nearest_alternative_failed_gates": [
                    value for value in clean_text(
                        finding.get(
                            "nearest_alternative_failed_gates", ""
                        )
                    ).split(";") if value
                ],
                "decision_competitor_score": _number(
                    finding.get("decision_competitor_score", "")
                ),
                "decision_competitor_doc_id": clean_text(
                    finding.get("decision_competitor_doc_id", "")
                ),
                "decision_competitor_passage_id": clean_text(
                    finding.get("decision_competitor_passage_id", "")
                ),
                "winner_margin": _number(
                    finding.get("winner_margin", "")
                ),
                "margin_status": clean_text(
                    finding.get("margin_status", "")
                ),
                "decision_reason": clean_text(
                    finding.get("decision_reason", "")
                ),
                "unresolved_reason": clean_text(
                    finding.get("unresolved_reason", "")
                ),
                "failed_gates": [
                    value for value in clean_text(
                        finding.get("failed_gates", "")
                    ).split(";") if value
                ],
            },
            "evidence_fragments": bounded,
            "available_fragment_count": len(fragments),
            "exported_fragment_count": len(bounded),
            "limitations": (
                [unresolved] if unresolved else []
            ) + (
                ["NO_EVIDENCE_FRAGMENT_AVAILABLE"] if not bounded else []
            ),
        })
    emitted = [unit["target_id"] for unit in units]
    missing = sorted(set(requested) - set(emitted))
    unexpected = sorted(set(emitted) - set(requested))
    duplicate = sorted({
        value for value in emitted if emitted.count(value) > 1
    })
    coverage_state = (
        "PASS" if not missing and not unexpected and not duplicate else "FAIL"
    )
    return {
        "schema_version": EVIDENCE_UNITS_SCHEMA,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "matcher_version": MATCHER_VERSION,
        "digest_builder_version": DIGEST_BUILDER_VERSION,
        "generated_at": utc_now_iso(),
        "task": task,
        "pair_role": pair_role,
        "pair_status": pair_status,
        "requested_target_ids": requested,
        "emitted_target_ids": emitted,
        "requested_target_count": len(requested),
        "emitted_target_count": len(emitted),
        "target_coverage_state": coverage_state,
        "missing_target_ids": missing,
        "unexpected_target_ids": unexpected,
        "duplicate_target_ids": duplicate,
        "units": units,
    }


def evidence_units_summary(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for unit in payload.get("units", []):
        finding = unit.get("deterministic_finding", {})
        roles = [
            clean_text(item.get("evidence_role", ""))
            for item in unit.get("evidence_fragments", [])
        ]
        rows.append({
            "task": clean_text(unit.get("task", "")),
            "target_id": clean_text(unit.get("target_id", "")),
            "target_text": clean_text(
                unit.get("target", {}).get("target_text", "")
            ),
            "semantic_status": clean_text(
                finding.get("semantic_status", "")
            ),
            "selected_value_or_text": clean_text(
                finding.get("selected_value_or_text", "")
            ),
            "selected_source_doc_id": clean_text(
                finding.get("selected_source_doc_id", "")
            ),
            "evidence_fragment_count": len(
                unit.get("evidence_fragments", [])
            ),
            "evidence_roles": ";".join(roles),
            "limitations": ";".join(unit.get("limitations", [])),
        })
    return pd.DataFrame(rows)


def reconstruction_candidate_passages(
    payload: Mapping[str, Any],
) -> pd.DataFrame:
    columns = [
        "query_fragment_id", "query_fragment_text", "candidate_doc_id",
        "candidate_evidence_id", "evidence_status",
        "single_document_eligible", "collective_component_eligible",
        "pseudo_component_ids", "pseudo_excerpt", "raw_chunk_id",
        "raw_sentence_index", "previous_raw_sentence",
        "selected_raw_sentence", "next_raw_sentence",
        "matched_critical_facts_json", "missing_critical_facts_json",
        "event_gate_states_json", "retrieval_score",
        "event_identity_score", "critical_fact_coverage", "final_score",
        "passed_gates", "failed_gates", "decision_reason",
    ]
    rows: List[Dict[str, Any]] = []
    if clean_text(payload.get("task", "")) != "reconstruction":
        return pd.DataFrame(columns=columns)
    for unit in payload.get("units", []):
        target = unit.get("target", {})
        finding = unit.get("deterministic_finding", {})
        for fragment in unit.get("evidence_fragments", []):
            context = fragment.get("context", {})
            role = clean_text(fragment.get("evidence_role", ""))
            rows.append({
                "query_fragment_id": clean_text(unit.get("target_id", "")),
                "query_fragment_text": clean_text(
                    target.get("target_text", "")
                ),
                "candidate_doc_id": clean_text(
                    fragment.get("source_doc_id", "")
                ),
                "candidate_evidence_id": clean_text(
                    fragment.get("passage_id", "")
                ),
                "evidence_status": {
                    "selected": "SELECTED_RECONSTRUCTION",
                    "supporting": "COLLECTIVE_COMPONENT",
                    "corroborating": "CORROBORATING_EVIDENCE",
                    "partial": "PARTIAL_RELATED_EVIDENCE",
                    "related": "PARTIAL_RELATED_EVIDENCE",
                    "rejected": "NEAREST_REJECTED",
                }.get(role, role.upper()),
                "single_document_eligible": (
                    role == "selected"
                    and bool(finding.get("selected_source_doc_id"))
                ),
                "collective_component_eligible": role == "supporting",
                "pseudo_component_ids": "",
                "pseudo_excerpt": clean_text(
                    context.get("pseudo_excerpt", "")
                ),
                "raw_chunk_id": clean_text(
                    fragment.get("passage_id", "")
                ).split("::")[1]
                if "::" in clean_text(fragment.get("passage_id", ""))
                else "",
                "raw_sentence_index": clean_text(
                    fragment.get("passage_id", "")
                ).split("sentence_")[-1]
                if "sentence_" in clean_text(
                    fragment.get("passage_id", "")
                ) else "",
                "previous_raw_sentence": clean_text(
                    context.get("previous", "")
                ),
                "selected_raw_sentence": clean_text(
                    context.get("selected", "")
                ) or clean_text(fragment.get("excerpt", "")),
                "next_raw_sentence": clean_text(context.get("next", "")),
                "matched_critical_facts_json": json.dumps(
                    fragment.get("matched_facts", {}),
                    sort_keys=True,
                ),
                "missing_critical_facts_json": json.dumps(
                    fragment.get("missing_or_conflicting_facts", {}),
                    sort_keys=True,
                ),
                "event_gate_states_json": json.dumps({
                    "passed": fragment.get("passed_gates", []),
                    "failed": fragment.get("failed_gates", []),
                }, sort_keys=True),
                "retrieval_score": fragment.get("retrieval_score", ""),
                "event_identity_score": fragment.get(
                    "event_identity_score", ""
                ),
                "critical_fact_coverage": fragment.get(
                    "critical_fact_coverage", ""
                ),
                "final_score": fragment.get("final_score", ""),
                "passed_gates": ";".join(
                    fragment.get("passed_gates", [])
                ),
                "failed_gates": ";".join(
                    fragment.get("failed_gates", [])
                ),
                "decision_reason": clean_text(
                    fragment.get("decision_reason", "")
                ),
            })
    return pd.DataFrame(rows, columns=columns)


def schema_document() -> Dict[str, Any]:
    return {
        "schema_version": EVIDENCE_UNITS_SCHEMA,
        "common_required_fields": [
            "task", "target_id", "target", "pair_role", "pair_status",
            "deterministic_finding", "evidence_fragments", "limitations",
        ],
        "fragment_required_fields": [
            "source_doc_id", "passage_id", "excerpt", "origin",
            "evidence_role", "authoritative", "final_score", "passed_gates",
            "failed_gates", "decision_reason",
        ],
        "task_target_units": {
            "reconstruction": "query_fragment",
            "redaction": "value_redaction_slot",
            "sentence_redaction": "sentence_target",
            "claims": "claim_or_proposition",
            "contradictions": "proposition_or_event_comparison",
        },
    }


def write_evidence_outputs(
    output_dir: Path,
    payload: Mapping[str, Any],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "task_evidence_units_v0413.json"
    schema_path = output_dir / "task_evidence_units_schema_v0413.json"
    csv_path = output_dir / "task_evidence_units_v0413.csv"
    atomic_write_json(json_path, dict(payload))
    atomic_write_json(schema_path, schema_document())
    evidence_units_summary(payload).to_csv(csv_path, index=False)
    outputs = {
        "task_evidence_units": json_path,
        "task_evidence_units_schema": schema_path,
        "task_evidence_units_summary": csv_path,
    }
    passages = reconstruction_candidate_passages(payload)
    if clean_text(payload.get("task", "")) == "reconstruction":
        passage_path = (
            output_dir / "reconstruction_candidate_passages_v0413.csv"
        )
        passages.to_csv(passage_path, index=False)
        outputs["reconstruction_candidate_passages"] = passage_path
    return outputs
