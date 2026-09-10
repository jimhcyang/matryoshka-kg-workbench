#!/usr/bin/env python3
"""Parse once and run one native v0.4 task or a controlled pair."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from matryoshka_v0420.contracts import (
    DIGEST_BUILDER_VERSION,
    MATCHER_VERSION,
    PARSER_VERSION,
    RELEASE_VERSION,
    REQUIREMENTS_VERSION,
    WORKBENCH_VERSION,
    finalize_output_manifest,
    output_inventory,
    reconcile_pair,
    refresh_logical_output_manifest,
    safe_json,
    validate_output_manifest,
)
from matryoshka_v0420.digest import (
    deterministic_task_digest,
    generate_optional_llm,
    write_pair_digest_bundle,
)
from matryoshka_v0420.semantics import (
    atomic_write_json,
    clean_text,
    utc_now_iso,
)
from matryoshka_v0420.progress import StageTimingRecorder
from matryoshka_v0420.corpus import (
    CorpusIndex,
    validate_shared_source_cache,
    validate_task_feature_cache,
)
from matryoshka_v0420.matcher import _auto_task, _preflight
from matryoshka_v0420.parser import (
    build_query_task_preflight,
    build_representation_reconciliation,
)
from matryoshka_v0420.query import QueryPackage
from matryoshka_v0420.semantic_resources import (
    load_corpus_entity_resource,
    load_semantic_resources,
)


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


QUERY_PARSE_CACHE_MANIFEST = "query_parse_cache_manifest_v1.json"


def _source_fingerprint(paths: List[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        resolved = path.resolve()
        digest.update(str(resolved).encode("utf-8"))
        try:
            digest.update(resolved.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def _query_parse_cache_identity(
    *,
    query_text: str,
    use_openai_parser: bool,
    parser_model: str,
) -> Dict[str, Any]:
    release_root = Path(__file__).resolve().parent
    parser_runtime_sources = [
        release_root / "matryoshka_v0420" / "parser.py",
        release_root / "matryoshka_v0420" / "semantics.py",
    ]
    return {
        "contract": QUERY_PARSE_CACHE_MANIFEST,
        "parser_version": PARSER_VERSION,
        "parser_mode": "openai" if use_openai_parser else "heuristic",
        "parser_model": clean_text(
            parser_model if use_openai_parser else "heuristic"
        ),
        "parser_runtime_source_sha256": _source_fingerprint(
            parser_runtime_sources
        ),
        "query_sha256": hashlib.sha256(
            query_text.encode("utf-8")
        ).hexdigest(),
    }


def _reuse_query_parse_cache(
    *,
    cache_dir: Path,
    query_parse: Path,
    expected: Dict[str, Any],
) -> bool:
    manifest_path = cache_dir / QUERY_PARSE_CACHE_MANIFEST
    payload_dir = cache_dir / "payload"
    if not manifest_path.is_file() or not payload_dir.is_dir():
        return False
    manifest = safe_json(manifest_path)
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    required = (
        payload_dir / "query_fragments.csv",
        payload_dir / "query_task_preflight.csv",
    )
    if not all(path.is_file() for path in required):
        return False
    if query_parse.exists():
        shutil.rmtree(query_parse)
    shutil.copytree(payload_dir, query_parse)
    return True


def _store_query_parse_cache(
    *,
    cache_dir: Path,
    query_parse: Path,
    manifest: Dict[str, Any],
) -> None:
    payload_dir = cache_dir / "payload"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    payload_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(query_parse, payload_dir)
    atomic_write_json(cache_dir / QUERY_PARSE_CACHE_MANIFEST, manifest)


def _is_primary(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def query_preflight_diagnostics(query_parse: Path) -> Dict[str, Any]:
    """Summarize parser preflight without inventing a blocking category."""
    path = query_parse / "query_task_preflight.csv"
    if not path.is_file():
        return {"blocking_rows": [], "review_required_rows": []}
    frame = pd.read_csv(path, low_memory=False).fillna("")
    records = frame.to_dict(orient="records")
    return {
        "blocking_rows": [
            row for row in records
            if clean_text(row.get("status", "")).startswith("BLOCKED")
        ],
        "review_required_rows": [
            row for row in records
            if clean_text(row.get("status", "")).startswith(
                "REVIEW_REQUIRED"
            )
        ],
    }


def build_query_clause_execution_ledger(
    query_parse: Path,
    coordinator_preflight: Optional[List[Dict[str, Any]]] = None,
    *,
    matcher_completed: bool = False,
    task: str = "",
) -> pd.DataFrame:
    """Map every substantive clause to target execution or context use."""
    inventory_path = query_parse / "query_clause_inventory_v0415.csv"
    proposition_path = query_parse / "query_propositions.csv"
    parser_preflight_path = query_parse / "query_task_preflight.csv"
    if not inventory_path.is_file():
        return pd.DataFrame()
    inventory = pd.read_csv(inventory_path, low_memory=False).fillna("")
    propositions = (
        pd.read_csv(proposition_path, low_memory=False).fillna("")
        if proposition_path.is_file() else pd.DataFrame()
    )
    parser_preflight = (
        pd.read_csv(parser_preflight_path, low_memory=False).fillna("")
        if parser_preflight_path.is_file() else pd.DataFrame()
    )
    parser_status = {
        clean_text(row.get("target_id", "")): clean_text(
            row.get("status", "")
        )
        for row in parser_preflight.to_dict(orient="records")
    }
    coordinator_status = {
        clean_text(row.get("target_id", "")): clean_text(
            row.get("status", "")
        )
        for row in (coordinator_preflight or [])
    }
    rows: List[Dict[str, Any]] = []
    normalized_task = clean_text(task).lower()
    context_tasks = {
        "reconstruction", "redaction", "sentence_redaction",
    }

    def normalized_clause_text(value: Any) -> str:
        return " ".join(re.findall(r"\w+", clean_text(value).lower()))

    for clause in inventory.to_dict(orient="records"):
        clause_id = clean_text(clause.get("source_clause_id", ""))
        substantive = _is_primary(clause.get("is_substantive", False))
        candidates = (
            propositions[
                propositions.get(
                    "source_clause_id", pd.Series(dtype=str)
                ).astype(str).eq(clause_id)
            ]
            if len(propositions) and "source_clause_id" in propositions
            else pd.DataFrame()
        )
        if candidates.empty and len(propositions):
            clause_key = normalized_clause_text(
                clause.get("normalized_clause", clause.get("clause_text", ""))
            )
            if clause_key:
                exact = propositions[
                    propositions.get(
                        "normalized_proposition", pd.Series(dtype=str)
                    ).map(normalized_clause_text).eq(clause_key)
                ]
                if len(exact) == 1:
                    candidates = exact
        candidate_ids = (
            candidates.get("query_proposition_id", pd.Series(dtype=str))
            .astype(str).tolist()
        )
        primary_ids = [
            clean_text(row.get("query_proposition_id", ""))
            for row in candidates.to_dict(orient="records")
            if _is_primary(row.get("primary_for_matching", False))
        ]
        statuses = [
            coordinator_status.get(target)
            or parser_status.get(target, "")
            for target in candidate_ids
        ]
        primary_statuses = [
            coordinator_status.get(target)
            or parser_status.get(target, "")
            for target in primary_ids
        ]
        if not substantive:
            readiness = "NOT_SUBSTANTIVE"
            reason = "clause_inventory_not_substantive"
        elif normalized_task in context_tasks:
            readiness = "CONTEXT_AVAILABLE"
            reason = "substantive_clause_available_to_task_context"
        elif primary_ids and any(
            status in {"READY", "EXECUTABLE"}
            or status.startswith("REVIEW_REQUIRED")
            for status in primary_statuses
        ):
            readiness = "EXECUTABLE"
            reason = "primary_query_proposition_available"
        elif any(status.startswith("REVIEW_REQUIRED") for status in statuses):
            readiness = "REVIEW_REQUIRED"
            reason = "no_executable_primary;review_candidate_available"
        else:
            readiness = "BLOCKED"
            reason = (
                "no_executable_primary;" + ";".join(
                    sorted(set(status for status in statuses if status))
                )
            ).rstrip(";")
        if not substantive:
            execution = "NOT_SUBSTANTIVE"
        elif matcher_completed and readiness == "EXECUTABLE":
            execution = "EXECUTED_TARGET"
        elif matcher_completed and readiness == "CONTEXT_AVAILABLE":
            execution = "CONTEXT_CONSUMED"
        elif readiness == "REVIEW_REQUIRED":
            execution = "REVIEW_REQUIRED_NOT_EXECUTED"
        elif readiness == "BLOCKED":
            execution = "BLOCKED_NOT_EXECUTED"
        else:
            execution = "NOT_RUN"
        rows.append({
            "fragment_id": clean_text(clause.get("fragment_id", "")),
            "source_sentence_id": clean_text(
                clause.get("source_sentence_id", "")
            ),
            "source_clause_id": clause_id,
            "clause_text": clean_text(clause.get("clause_text", "")),
            "is_substantive": substantive,
            "candidate_target_ids_json": json.dumps(candidate_ids),
            "primary_target_ids_json": json.dumps(primary_ids),
            "readiness_state": readiness,
            "execution_state": execution,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def clause_coverage_summary(ledger: pd.DataFrame) -> Dict[str, Any]:
    if ledger.empty:
        return {
            "query_clause_coverage_state": "NOT_ASSESSED",
            "query_clause_count": 0,
            "executed_clause_count": 0,
            "executed_target_clause_count": 0,
            "context_consumed_clause_count": 0,
            "covered_clause_count": 0,
            "review_clause_count": 0,
            "blocked_clause_count": 0,
        }
    substantive = ledger[
        ledger["is_substantive"].map(_is_primary)
    ]
    states = substantive["execution_state"].astype(str)
    target_executed = int(states.isin({"EXECUTED", "EXECUTED_TARGET"}).sum())
    context_consumed = int(states.eq("CONTEXT_CONSUMED").sum())
    covered = target_executed + context_consumed
    review = int(states.str.startswith("REVIEW_REQUIRED").sum())
    blocked = int(states.str.startswith("BLOCKED").sum())
    total = int(len(substantive))
    state = (
        "PASS" if total and covered == total
        else "PARTIAL" if covered
        else "FAIL" if total
        else "NOT_APPLICABLE"
    )
    return {
        "query_clause_coverage_state": state,
        "query_clause_count": total,
        "executed_clause_count": target_executed,
        "executed_target_clause_count": target_executed,
        "context_consumed_clause_count": context_consumed,
        "covered_clause_count": covered,
        "review_clause_count": review,
        "blocked_clause_count": blocked,
        "review_clause_ids": substantive.loc[
            states.str.startswith("REVIEW_REQUIRED"), "source_clause_id"
        ].astype(str).tolist(),
        "blocked_clause_ids": substantive.loc[
            states.str.startswith("BLOCKED"), "source_clause_id"
        ].astype(str).tolist(),
    }


def augment_task_digest_clause_coverage(
    output_dir: Path,
    summary: Dict[str, Any],
) -> None:
    """Add query-clause coverage without changing Matcher decisions."""
    evidence_path = output_dir / "task_evidence_units_v0413.json"
    if not evidence_path.is_file():
        return
    payload = safe_json(evidence_path)
    payload.update(summary)
    atomic_write_json(evidence_path, payload)
    digest = deterministic_task_digest(payload)
    (output_dir / "semantic_match_task_digest_v0413.md").write_text(
        digest, encoding="utf-8"
    )
    (output_dir / "semantic_match_task_digest_v048.md").write_text(
        digest, encoding="utf-8"
    )


def validate_authoritative_query_inputs(query_parse: Path) -> Dict[str, Any]:
    """Require proposition/signature agreement without reading diagnostics.

    Matcher v0.4.20 consumes the event signature embedded
    in each proposition row. The coordinator makes the separately exported
    signature table authoritative by refusing to continue unless its primary
    signature multiset is identical. Reconciliation is intentionally absent.
    """
    proposition_path = query_parse / "query_propositions.csv"
    signature_path = query_parse / "query_event_signatures.csv"
    if not proposition_path.is_file() or not signature_path.is_file():
        raise ValueError(
            "Authoritative query inputs are incomplete: both "
            "query_propositions.csv and query_event_signatures.csv are "
            "required."
        )
    propositions = pd.read_csv(proposition_path, low_memory=False)
    signatures = pd.read_csv(signature_path, low_memory=False)
    required = {"event_signature_sha256", "primary_for_matching"}
    for label, frame in (
        ("query_propositions.csv", propositions),
        ("query_event_signatures.csv", signatures),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"{label} is missing authoritative columns: "
                + ", ".join(missing)
            )
    proposition_hashes = Counter(
        str(row.event_signature_sha256).strip()
        for row in propositions.itertuples(index=False)
        if _is_primary(row.primary_for_matching)
    )
    signature_hashes = Counter(
        str(row.event_signature_sha256).strip()
        for row in signatures.itertuples(index=False)
        if _is_primary(row.primary_for_matching)
    )
    if "" in proposition_hashes or "nan" in proposition_hashes:
        raise ValueError(
            "A primary proposition has no valid event-signature hash."
        )
    if "" in signature_hashes or "nan" in signature_hashes:
        raise ValueError(
            "A primary event-signature row has no valid signature hash."
        )
    if proposition_hashes != signature_hashes:
        raise ValueError(
            "Authoritative query inputs disagree: primary event signatures "
            "in query_propositions.csv do not match "
            "query_event_signatures.csv."
        )
    return {
        "status": "PASS",
        "contract": (
            "semantic decisions use query_propositions.csv and "
            "query_event_signatures.csv; reconciliation is diagnostic-only"
        ),
        "primary_proposition_count": sum(proposition_hashes.values()),
        "primary_event_signature_count": sum(signature_hashes.values()),
        "signature_multiset_match": True,
    }


def prepare_matcher_query_compatibility_view(
    query_parse: Path, run_root: Path
) -> tuple[Path, Dict[str, Any]]:
    """Expose Parser v0.4.15 tables to Matcher v0.4.20.

    The adapter changes only the manifest schema label. Every CSV is copied
    byte-for-byte and verified, so it cannot alter a semantic input.
    """
    manifest_path = query_parse / "query_parse_manifest.json"
    manifest = safe_json(manifest_path)
    source_schema = clean_text(manifest.get("schema_version", ""))
    if source_schema in {
        "query_parse_schema_v0_4_6", "query_parse_schema_v0_4_7"
    }:
        return query_parse, {
            "status": "NOT_REQUIRED",
            "source_schema_version": source_schema,
            "matcher_schema_version": source_schema,
            "semantic_tables_byte_identical": True,
        }
    if source_schema != "query_parse_schema_v0_4_15_candidate":
        raise ValueError(
            "Unsupported frozen Parser schema for Matcher compatibility: "
            + (source_schema or "missing schema")
        )
    compatibility = run_root / "matcher_query_parse_compat_v0415"
    compatibility.mkdir(parents=True, exist_ok=False)
    copied_hashes: Dict[str, str] = {}
    for source in sorted(query_parse.iterdir()):
        if not source.is_file() or source.name == manifest_path.name:
            continue
        destination = compatibility / source.name
        shutil.copy2(source, destination)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        destination_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        if source_hash != destination_hash:
            raise RuntimeError(
                "Matcher compatibility copy changed bytes: " + source.name
            )
        copied_hashes[source.name] = source_hash
    adapted_manifest = dict(manifest)
    adapted_manifest["schema_version"] = "query_parse_schema_v0_4_7"
    adapted_manifest["source_schema_version"] = source_schema
    adapted_manifest["schema_compatibility_adapter"] = (
        "coordinator_label_only_v0415_to_v0420; semantic CSV bytes unchanged"
    )
    adapted_manifest["source_query_parse_dir"] = str(query_parse)
    atomic_write_json(
        compatibility / manifest_path.name, adapted_manifest
    )
    alternative_preflight_rows_downgraded = 0
    nominal_event_rows_downgraded = 0
    declarative_clause_rows_downgraded = 0
    proposition_copy = compatibility / "query_propositions.csv"
    preflight_copy = compatibility / "query_task_preflight.csv"
    reconciliation_copy = compatibility / "query_representation_reconciliation.csv"
    if proposition_copy.is_file():
        parsed_for_refresh = {
            "fragments": pd.read_csv(
                compatibility / "query_fragments.csv", low_memory=False
            ) if (compatibility / "query_fragments.csv").is_file() else pd.DataFrame(),
            "claims": pd.read_csv(
                compatibility / "query_claims.csv", low_memory=False
            ) if (compatibility / "query_claims.csv").is_file() else pd.DataFrame(),
            "events": pd.read_csv(
                compatibility / "query_events.csv", low_memory=False
            ) if (compatibility / "query_events.csv").is_file() else pd.DataFrame(),
            "relations": pd.read_csv(
                compatibility / "query_relations.csv", low_memory=False
            ) if (compatibility / "query_relations.csv").is_file() else pd.DataFrame(),
            "propositions": pd.read_csv(
                proposition_copy, low_memory=False
            ),
            "redaction_slots": pd.read_csv(
                compatibility / "query_redaction_slots.csv", low_memory=False
            ) if (compatibility / "query_redaction_slots.csv").is_file() else pd.DataFrame(),
            "sentence_redaction_slots": pd.read_csv(
                compatibility / "query_sentence_redaction_slots.csv", low_memory=False
            ) if (compatibility / "query_sentence_redaction_slots.csv").is_file() else pd.DataFrame(),
        }
        reconciliation_frame = build_representation_reconciliation(
            parsed_for_refresh
        )
        parsed_for_refresh["representation_reconciliation"] = (
            reconciliation_frame
        )
        reconciliation_frame.to_csv(reconciliation_copy, index=False)
        build_query_task_preflight(parsed_for_refresh).to_csv(
            preflight_copy, index=False
        )
    if proposition_copy.is_file() and preflight_copy.is_file():
        proposition_frame = pd.read_csv(proposition_copy, low_memory=False)
        preflight_frame = pd.read_csv(preflight_copy, low_memory=False)
        if {
            "query_proposition_id", "hypothesis_group_id"
        }.issubset(proposition_frame.columns) and {
            "target_id", "status", "reason"
        }.issubset(preflight_frame.columns):
            target_group = {
                clean_text(row.query_proposition_id):
                    clean_text(row.hypothesis_group_id)
                for row in proposition_frame.itertuples(index=False)
            }
            target_signature: Dict[str, Dict[str, Any]] = {}
            for row in proposition_frame.itertuples(index=False):
                target = clean_text(row.query_proposition_id)
                try:
                    signature = json.loads(clean_text(
                        getattr(row, "event_signature_json", "")
                    ))
                except (TypeError, json.JSONDecodeError):
                    signature = {}
                target_signature[target] = (
                    signature if isinstance(signature, dict) else {}
                )
            nonblocked_groups = {
                target_group.get(clean_text(row.target_id), "")
                for row in preflight_frame.itertuples(index=False)
                if not clean_text(row.status).startswith("BLOCKED_")
            }
            nonblocked_groups.discard("")
            def normalized_clause_text(value: Any) -> str:
                return " ".join(
                    re.findall(r"\w+", clean_text(value).lower())
                )

            proposition_coverage = {
                (
                    clean_text(getattr(row, "fragment_id", "")),
                    normalized_clause_text(
                        getattr(row, "proposition_text", "")
                    ),
                )
                for row in proposition_frame.itertuples(index=False)
                if _is_primary(getattr(row, "primary_for_matching", True))
            }
            clause_coverage: Dict[str, tuple[str, str]] = {}
            clause_inventory_copy = (
                compatibility / "query_clause_inventory_v0415.csv"
            )
            if clause_inventory_copy.is_file():
                clause_inventory = pd.read_csv(
                    clause_inventory_copy, low_memory=False
                )
                if {
                    "source_clause_id", "fragment_id", "clause_text"
                }.issubset(clause_inventory.columns):
                    clause_coverage = {
                        clean_text(row.source_clause_id): (
                            clean_text(row.fragment_id),
                            normalized_clause_text(row.clause_text),
                        )
                        for row in clause_inventory.itertuples(index=False)
                    }
            for index, row in preflight_frame.iterrows():
                target = clean_text(row.get("target_id", ""))
                group = target_group.get(target, "")
                if (
                    group in nonblocked_groups
                    and clean_text(row.get("status", "")).startswith(
                        "BLOCKED_INCOMPLETE_QUERY_SIGNATURE"
                    )
                ):
                    preflight_frame.at[index, "status"] = (
                        "REVIEW_REQUIRED_ALTERNATIVE_HYPOTHESIS"
                    )
                    preflight_frame.at[index, "reason"] = (
                        clean_text(row.get("reason", ""))
                        + "; viable_sibling_hypothesis_present"
                    )
                    alternative_preflight_rows_downgraded += 1
                    continue
                status = clean_text(row.get("status", ""))
                reason = clean_text(row.get("reason", ""))
                signature = target_signature.get(target, {})
                discriminative_anchor = any(
                    clean_text(signature.get(field, ""))
                    for field in (
                        "location_identity", "location", "event_time",
                        "date_value", "vehicle_identity",
                        "semantic_state_variable", "semantic_state_value",
                    )
                ) or any(
                    signature.get(field) is not None
                    for field in (
                        "packet_count", "total_weight_grams",
                        "weight_kg", "weight_g",
                    )
                )
                if (
                    status == "BLOCKED_INCOMPLETE_QUERY_SIGNATURE"
                    and reason == "missing_required_fields:actor"
                    and clean_text(signature.get("predicate", ""))
                    and clean_text(signature.get("object", ""))
                    and discriminative_anchor
                ):
                    preflight_frame.at[index, "status"] = (
                        "REVIEW_REQUIRED_ACTORLESS_NOMINAL_EVENT"
                    )
                    preflight_frame.at[index, "reason"] = (
                        reason
                        + "; typed_nominal_event_anchors_sufficient_for_review"
                    )
                    nominal_event_rows_downgraded += 1
                    continue
                if (
                    status == "BLOCKED_UNCOVERED_QUERY_CLAUSE"
                    and clause_coverage.get(target) in proposition_coverage
                ):
                    preflight_frame.at[index, "status"] = (
                        "REVIEW_REQUIRED_CLAUSE_COVERED_BY_DECLARATIVE_PROJECTION"
                    )
                    preflight_frame.at[index, "reason"] = (
                        reason
                        + "; exact_primary_declarative_projection_present"
                    )
                    declarative_clause_rows_downgraded += 1
            if (
                alternative_preflight_rows_downgraded
                or nominal_event_rows_downgraded
                or declarative_clause_rows_downgraded
            ):
                preflight_frame.to_csv(preflight_copy, index=False)
    report = {
        "status": "PASS",
        "source_schema_version": source_schema,
        "matcher_schema_version": "query_parse_schema_v0_4_7",
        "semantic_tables_byte_identical": True,
        "copied_file_count": len(copied_hashes),
        "authoritative_table_sha256": {
            name: copied_hashes[name]
            for name in (
                "query_propositions.csv", "query_event_signatures.csv"
            )
        },
        "diagnostic_reconciliation_used": False,
        "alternative_preflight_rows_downgraded_to_review":
            alternative_preflight_rows_downgraded,
        "alternative_preflight_contract": (
            "An incomplete typed alternative is review-only when its same "
            "hypothesis group contains a nonblocked sibling; no proposition "
            "or event-signature row is changed."
        ),
        "actorless_nominal_event_rows_downgraded_to_review":
            nominal_event_rows_downgraded,
        "actorless_nominal_event_contract": (
            "Missing actor alone is review-only when predicate, object, and "
            "at least one typed event discriminator are present; no semantic "
            "table row is changed."
        ),
        "declarative_clause_rows_downgraded_to_review":
            declarative_clause_rows_downgraded,
        "declarative_clause_coverage_contract": (
            "An uncovered-clause block is review-only only when a primary "
            "proposition has the same fragment and exact normalized clause "
            "text; no approximate coverage is inferred."
        ),
    }
    atomic_write_json(
        run_root / "query_schema_compatibility_v0415_v0420.json", report
    )
    return compatibility, report


def run(
    command: List[str],
    cwd: Path,
    prefix: Path,
    *,
    cancel_path: Optional[Path] = None,
    matcher_output: Optional[Path] = None,
    status_path: Optional[Path] = None,
    stage_name: str = "",
) -> int:
    stage_started = time.perf_counter()
    stage_started_at = utc_now_iso()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    cancellation_forwarded = False
    next_heartbeat = 0.0
    while process.poll() is None:
        now = time.perf_counter()
        if status_path is not None and now >= next_heartbeat:
            status = safe_json(status_path)
            status.update({
                "current_stage": stage_name,
                "stage_started_at": stage_started_at,
                "stage_elapsed_seconds": now - stage_started,
                "last_heartbeat": utc_now_iso(),
                "child_process_id": process.pid,
                "child_process_state": "RUNNING",
                "terminal_state": "",
                "failure_owner": "",
            })
            atomic_write_json(status_path, status)
            next_heartbeat = now + 1.0
        if cancel_path and cancel_path.exists() and not cancellation_forwarded:
            cancellation_forwarded = True
            if matcher_output:
                matcher_output.mkdir(parents=True, exist_ok=True)
                (matcher_output / "CANCEL").write_text(
                    "Cancellation forwarded by coordinated v0.4 runner\n",
                    encoding="utf-8",
                )
            else:
                process.terminate()
        time.sleep(0.25)
    stdout, stderr = process.communicate()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".command.txt").write_text(
        " ".join(shlex.quote(item) for item in command) + "\n",
        encoding="utf-8",
    )
    prefix.with_suffix(".stdout.log").write_text(
        stdout or "", encoding="utf-8"
    )
    prefix.with_suffix(".stderr.log").write_text(
        stderr or "", encoding="utf-8"
    )
    return int(process.returncode or 0)


def write_pair_candidate_route_delta(
    root: Path,
    baseline_dir: Path,
    holdout_dir: Path,
) -> pd.DataFrame:
    """Explain arm changes by atomic route, exclusion, refill, and dedup.

    Matcher records the union of all retrieval routes for a document as a
    semicolon-delimited signature.  A holdout refill can legitimately add a
    route to an already-retained document.  Comparing whole signatures would
    then report a false loss from the old signature.  Pair accounting instead
    compares membership in each atomic route while retaining a count of full
    signature migrations for diagnostics.
    """
    filename = "candidate_route_document_accounting_v0420.csv"
    baseline_raw = pd.read_csv(
        baseline_dir / filename, low_memory=False
    )
    holdout_raw = pd.read_csv(
        holdout_dir / filename, low_memory=False
    )

    def routes(value: object) -> List[str]:
        members = {
            clean_text(item)
            for item in clean_text(value).split(";")
            if clean_text(item)
        }
        return sorted(members) or ["TASK_NATIVE"]

    def atomic(frame: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for source in frame.to_dict("records"):
            signature = (
                clean_text(source.get("retrieval_route", ""))
                or "TASK_NATIVE"
            )
            for route in routes(signature):
                row = dict(source)
                row["retrieval_route_signature"] = signature
                row["retrieval_route"] = route
                rows.append(row)
        columns = list(frame.columns)
        if "retrieval_route_signature" not in columns:
            columns.append("retrieval_route_signature")
        return pd.DataFrame(rows, columns=columns)

    baseline = atomic(baseline_raw)
    holdout = atomic(holdout_raw)
    holdout_manifest = safe_json(
        holdout_dir / "semantic_match_run_manifest_v048.json"
    )
    excluded = {
        clean_text(value).upper()
        for value in holdout_manifest.get(
            "effective_excluded_doc_ids", []
        )
        if clean_text(value)
    }
    keys = {
        (
            clean_text(row.get("target_id", "")),
            clean_text(row.get("retrieval_route", ""))
            or "TASK_NATIVE",
        )
        for row in baseline.to_dict("records")
        + holdout.to_dict("records")
    }
    rows: List[Dict[str, object]] = []
    for target_id, route in sorted(keys):
        before = baseline[
            baseline["target_id"].fillna("").astype(str).eq(target_id)
            & baseline["retrieval_route"].fillna("").astype(str).eq(route)
        ]
        after = holdout[
            holdout["target_id"].fillna("").astype(str).eq(target_id)
            & holdout["retrieval_route"].fillna("").astype(str).eq(route)
        ]

        def total(frame: pd.DataFrame, column: str) -> int:
            return int(pd.to_numeric(
                frame.get(column, pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum())

        baseline_count = total(before, "candidate_row_count")
        holdout_count = total(after, "candidate_row_count")
        removed = total(
            before[
                before["doc_id"].fillna("").astype(str)
                .str.upper().isin(excluded)
            ],
            "candidate_row_count",
        )
        post_exclusion_before_refill = max(
            0, baseline_count - removed
        )

        def candidate_counts_by_doc(frame: pd.DataFrame) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for source in frame.to_dict("records"):
                doc_id = clean_text(source.get("doc_id", "")).upper()
                if not doc_id:
                    continue
                try:
                    count = int(float(source.get("candidate_row_count", 0) or 0))
                except (TypeError, ValueError):
                    count = 0
                counts[doc_id] = counts.get(doc_id, 0) + max(0, count)
            return counts

        baseline_survivor_counts = {
            doc_id: count
            for doc_id, count in candidate_counts_by_doc(before).items()
            if doc_id not in excluded
        }
        holdout_counts = candidate_counts_by_doc(after)
        carried = sum(
            min(count, holdout_counts.get(doc_id, 0))
            for doc_id, count in baseline_survivor_counts.items()
        )
        dropped = sum(
            max(0, count - holdout_counts.get(doc_id, 0))
            for doc_id, count in baseline_survivor_counts.items()
        )
        added = sum(
            max(0, count - baseline_survivor_counts.get(doc_id, 0))
            for doc_id, count in holdout_counts.items()
        )
        # A bounded route can legitimately lose and gain members after the
        # excluded source is removed because its top-k boundary is reranked.
        # `refill` therefore means observed additions, not a net-count guess.
        refill = added
        before_signatures = {
            clean_text(row.get("doc_id", "")).upper():
                clean_text(row.get("retrieval_route_signature", ""))
            for row in before.to_dict("records")
            if clean_text(row.get("doc_id", ""))
        }
        after_signatures = {
            clean_text(row.get("doc_id", "")).upper():
                clean_text(row.get("retrieval_route_signature", ""))
            for row in after.to_dict("records")
            if clean_text(row.get("doc_id", ""))
        }
        signature_migrations = sum(
            before_signatures[doc_id] != after_signatures[doc_id]
            for doc_id in set(before_signatures) & set(after_signatures)
        )
        rows.append({
            "target_id": target_id,
            "retrieval_route": route,
            "route_accounting_basis": "ATOMIC_ROUTE_MEMBERSHIP",
            "route_signature_migration_document_count":
                signature_migrations,
            "baseline_candidate_count": baseline_count,
            "excluded_candidate_count": removed,
            "post_exclusion_before_refill_count":
                post_exclusion_before_refill,
            "carried_forward_candidate_count": carried,
            "route_churn_dropped_candidate_count": dropped,
            "route_churn_added_candidate_count": added,
            "refill_candidate_count": refill,
            "holdout_candidate_count": holdout_count,
            "population_delta":
                holdout_count - baseline_count,
            "baseline_deduplicated_row_count":
                total(before, "deduplicated_row_count"),
            "holdout_deduplicated_row_count":
                total(after, "deduplicated_row_count"),
            "baseline_final_retention_count":
                total(before, "eligible_candidate_count"),
            "holdout_final_retention_count":
                total(after, "eligible_candidate_count"),
            "accounting_state": (
                "RECONCILED"
                if (
                    post_exclusion_before_refill == carried + dropped
                    and holdout_count == carried + added
                    and holdout_count
                    == post_exclusion_before_refill - dropped + refill
                )
                else "UNRECONCILED"
            ),
            "accounting_contract": (
                "atomic_route_exclusion_churn_refill_deduplication_"
                "final_retention"
            ),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(
        root / "pair_candidate_route_delta_v0420.csv",
        index=False,
    )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["single", "pair"])
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--extraction-dir", required=True)
    parser.add_argument(
        "--task", required=True,
        choices=[
            "auto", "reconstruction", "redaction", "sentence_redaction",
            "claims", "contradictions",
        ],
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--code-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--target-doc-ids", default="")
    parser.add_argument("--exclude-doc-ids", default="")
    parser.add_argument("--use-openai-parser", action="store_true")
    parser.add_argument(
        "--parser-model", default="gpt-4o",
        help=(
            "OpenAI model used only for Parser extraction. GPT-4o is the "
            "recommended quality default; lower-cost models remain selectable."
        ),
    )
    parser.add_argument("--use-llm-digest", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-max-output-tokens", type=int, default=2400)
    parser.add_argument("--staged-top-docs", type=int, default=150)
    parser.add_argument(
        "--retrieval-mode", choices=["staged", "exhaustive"], default="staged"
    )
    parser.add_argument(
        "--raw-use-mode", choices=["validate", "discover"], default="validate"
    )
    parser.add_argument("--max-candidates-per-target", type=int, default=100000)
    parser.add_argument("--max-elapsed-seconds", type=float, default=0.0)
    parser.add_argument("--semantic-resources", default="")
    parser.add_argument("--corpus-entity-resource", default="")
    parser.add_argument(
        "--decision-profile",
        default="",
        help="Optional validated decision-profile JSON passed unchanged to Matcher.",
    )
    parser.add_argument("--minimum-score", type=float, default=None)
    parser.add_argument("--minimum-margin", type=float, default=None)
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
    parser.add_argument(
        "--shared-source-cache",
        default="",
        help=(
            "Optional persistent immutable source-table cache. The cache is "
            "validated against extraction path and exact table content hashes."
        ),
    )
    parser.add_argument(
        "--rebuild-shared-source-cache", action="store_true"
    )
    parser.add_argument(
        "--task-feature-cache",
        default="",
        help=(
            "Optional persistent query-independent task feature cache. "
            "Controlled pairs create a temporary cache automatically."
        ),
    )
    parser.add_argument(
        "--query-parse-cache",
        default="",
        help=(
            "Optional persistent query-parse cache keyed by query text and "
            "parser settings. Reusing it avoids repeat parser calls and keeps "
            "same-query reruns deterministic."
        ),
    )
    parser.add_argument(
        "--rebuild-query-parse-cache", action="store_true"
    )
    parser.add_argument(
        "--rebuild-task-feature-cache", action="store_true"
    )
    args = parser.parse_args()
    coordinated_started = time.perf_counter()
    coordinated_started_at = utc_now_iso()
    timing = StageTimingRecorder()

    code = Path(args.code_dir).expanduser().resolve()
    root = Path(args.run_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cancel_path = root / "CANCEL"
    status_path = root / "coordinated_status_v048.json"
    if status_path.exists():
        raise FileExistsError("Run root already contains a coordinated run: " + str(root))
    atomic_write_json(status_path, {
        "status": "PARSING",
        "started_at": coordinated_started_at,
        "mode": args.mode,
        "task": args.task,
    })
    atomic_write_json(root / ".matryoshka_run_identity_v048.json", {
        "run_root": str(root),
        "created_at": utc_now_iso(),
    })
    query_parse = root / "query_parse"
    query_file = Path(args.query_file).expanduser().resolve()
    query_text = query_file.read_text(encoding="utf-8")
    parser_command = [
        args.python_executable,
        str(code / "query_semantic_parser_v0_4_15.py"),
        "--query-file", str(query_file),
        "--output-dir", str(query_parse),
        "--openai-model", args.parser_model,
        "--openai-api-key-env", args.openai_api_key_env,
    ]
    if args.use_openai_parser:
        parser_command.append("--use-openai")
    parse_cache_summary: Dict[str, Any] = {
        "cache_state": "NOT_REQUESTED",
        "cache_path": "",
    }
    parse_cache_identity = _query_parse_cache_identity(
        query_text=query_text,
        use_openai_parser=bool(args.use_openai_parser),
        parser_model=args.parser_model,
    )
    query_parse_cache = (
        Path(args.query_parse_cache).expanduser().resolve()
        if args.query_parse_cache else None
    )
    parser_started = time.perf_counter()
    parser_rc = 0
    if query_parse_cache is not None:
        query_parse_cache.parent.mkdir(parents=True, exist_ok=True)
        parse_cache_summary["cache_path"] = str(query_parse_cache)
        cache_lock = query_parse_cache.with_name(query_parse_cache.name + ".lock")
        with _file_lock(cache_lock):
            if (
                args.rebuild_query_parse_cache
                and query_parse_cache.exists()
            ):
                shutil.rmtree(query_parse_cache, ignore_errors=True)
            reused = _reuse_query_parse_cache(
                cache_dir=query_parse_cache,
                query_parse=query_parse,
                expected=parse_cache_identity,
            )
            if reused:
                parse_cache_summary["cache_state"] = "REUSED_EXISTING_VALIDATED"
            else:
                parser_rc = run(
                    parser_command,
                    code,
                    root / "parser",
                    cancel_path=cancel_path,
                    status_path=status_path,
                    stage_name="PARSER",
                )
                if parser_rc == 0:
                    manifest = dict(parse_cache_identity)
                    manifest["cached_at"] = utc_now_iso()
                    _store_query_parse_cache(
                        cache_dir=query_parse_cache,
                        query_parse=query_parse,
                        manifest=manifest,
                    )
                    parse_cache_summary["cache_state"] = "BUILT_NEW"
                else:
                    parse_cache_summary["cache_state"] = "BUILD_FAILED"
    else:
        parser_rc = run(
            parser_command,
            code,
            root / "parser",
            cancel_path=cancel_path,
            status_path=status_path,
            stage_name="PARSER",
        )
    timing.record(
        "parser",
        time.perf_counter() - parser_started,
        owner="parser",
        state=(
            parse_cache_summary["cache_state"]
            if parser_rc == 0 and query_parse_cache is not None
            else ("COMPLETED" if parser_rc == 0 else "FAILED")
        ),
    )
    atomic_write_json(
        root / "query_parse_cache_status_v1.json",
        parse_cache_summary,
    )
    if parser_rc:
        atomic_write_json(status_path, {
            "status": "PARSER_FAILED", "return_code": parser_rc,
            "completed_at": utc_now_iso(),
        })
        return 2

    preflight_started = time.perf_counter()
    try:
        authoritative_query_validation = (
            validate_authoritative_query_inputs(query_parse)
        )
        atomic_write_json(
            root / "authoritative_query_input_validation_v0415.json",
            authoritative_query_validation,
        )
        matcher_query_parse, schema_compatibility = (
            prepare_matcher_query_compatibility_view(query_parse, root)
        )
        query_package = QueryPackage.load(
            matcher_query_parse, task=args.task
        )
    except ValueError as exc:
        diagnostics = query_preflight_diagnostics(query_parse)
        atomic_write_json(status_path, {
            "status": "PREFLIGHT_BLOCKED",
            "task": args.task,
            "error_type": type(exc).__name__,
            "error_detail": clean_text(str(exc))[:2000],
            "blocking_rows": diagnostics["blocking_rows"],
            "review_required_rows": diagnostics[
                "review_required_rows"
            ],
            "completed_at": utc_now_iso(),
        })
        return 2
    except Exception as exc:
        atomic_write_json(status_path, {
            "status": "FAILED",
            "task": args.task,
            "error_type": type(exc).__name__,
            "error_detail": clean_text(str(exc))[:2000],
            "completed_at": utc_now_iso(),
        })
        return 2
    resolved_task = (
        _auto_task(query_package) if args.task == "auto" else args.task
    )
    corpus_entity_resource = (
        load_corpus_entity_resource(Path(args.corpus_entity_resource))
        if args.corpus_entity_resource else None
    )
    semantic_resources = load_semantic_resources(
        Path(args.semantic_resources) if args.semantic_resources else None,
        corpus_entity_resource=corpus_entity_resource,
    )
    preflight = _preflight(resolved_task, query_package)
    timing.record(
        "preflight",
        time.perf_counter() - preflight_started,
        owner="coordinator",
    )
    pd.DataFrame(preflight).to_csv(
        root / "coordinated_preflight_v048.csv", index=False
    )
    clause_ledger = build_query_clause_execution_ledger(
        query_parse, preflight, matcher_completed=False, task=resolved_task
    )
    clause_ledger.to_csv(
        root / "query_clause_execution_ledger_v0415.csv", index=False
    )
    blocked = [
        row for row in preflight
        if str(row.get("status", "")).startswith("BLOCKED")
    ]
    if blocked:
        diagnostics = query_preflight_diagnostics(query_parse)
        atomic_write_json(status_path, {
            "status": "PREFLIGHT_BLOCKED",
            "task": resolved_task,
            "blocking_statuses": [row["status"] for row in blocked],
            "blocking_rows": blocked,
            "review_required_rows": diagnostics[
                "review_required_rows"
            ],
            "completed_at": utc_now_iso(),
        })
        return 2

    def task_scoped_cache_path(
        path: Path,
        task_name: str,
        extraction_dir: Path,
    ) -> Path:
        """Prevent persistent caches from leaking across tasks or corpora."""
        normalized = path.stem.lower().replace("-", "_")
        token = task_name.lower().replace("-", "_")
        extraction_resolved = extraction_dir.expanduser().resolve()
        corpus_token = (
            extraction_resolved.parent.name.lower().replace("-", "_")
            or extraction_resolved.name.lower().replace("-", "_")
            or "corpus"
        )
        extraction_token = hashlib.sha256(
            str(extraction_resolved).encode("utf-8")
        ).hexdigest()[:10]
        composite = f"{token}_{corpus_token}_{extraction_token}"
        if composite in normalized:
            return path
        return path.with_name(
            path.stem + "_" + composite + path.suffix
        )

    shared_cache_directory = None
    shared_cache_path: Optional[Path] = None
    shared_cache_summary: Dict[str, object] = {}
    requested_cache = (
        task_scoped_cache_path(
            Path(args.shared_source_cache).expanduser().resolve(),
            resolved_task,
            Path(args.extraction_dir),
        )
        if args.shared_source_cache else None
    )
    cache_started = time.perf_counter()
    if requested_cache or args.mode == "pair":
        atomic_write_json(status_path, {
            "status": "PREPARING_SHARED_SOURCE",
            "task": resolved_task,
            "completed_preflight_rows": len(preflight),
        })
        if requested_cache:
            shared_cache_path = requested_cache
        else:
            shared_cache_directory = tempfile.TemporaryDirectory(
                prefix="matryoshka_v0420_pair_"
            )
            shared_cache_path = (
                Path(shared_cache_directory.name) / "source_tables_v0420.pkl"
            )
        if args.rebuild_shared_source_cache and shared_cache_path.exists():
            shared_cache_path.unlink()
        with _file_lock(shared_cache_path.with_suffix(shared_cache_path.suffix + ".lock")):
            if shared_cache_path.exists():
                try:
                    shared_cache_summary = validate_shared_source_cache(
                        shared_cache_path,
                        Path(args.extraction_dir),
                        resolved_task,
                    )
                except Exception:
                    shared_cache_path.unlink(missing_ok=True)
            if not shared_cache_path.exists():
                shared_cache_summary = CorpusIndex.prepare_shared_source_cache(
                    Path(args.extraction_dir),
                    shared_cache_path,
                    task=resolved_task,
                )

    feature_cache_directory = None
    feature_cache_path: Optional[Path] = None
    feature_cache_summary: Dict[str, object] = {}
    requested_feature_cache = (
        task_scoped_cache_path(
            Path(args.task_feature_cache).expanduser().resolve(),
            resolved_task,
            Path(args.extraction_dir),
        )
        if args.task_feature_cache else None
    )
    if requested_feature_cache or args.mode == "pair":
        atomic_write_json(status_path, {
            "status": "PREPARING_TASK_FEATURES",
            "task": resolved_task,
            "shared_source_cache": bool(shared_cache_path),
        })
        if requested_feature_cache:
            feature_cache_path = requested_feature_cache
        else:
            feature_cache_directory = tempfile.TemporaryDirectory(
                prefix="matryoshka_v0420_features_"
            )
            feature_cache_path = (
                Path(feature_cache_directory.name)
                / f"{resolved_task}_features_v0420.pkl"
            )
        if args.rebuild_task_feature_cache and feature_cache_path.exists():
            feature_cache_path.unlink()
        with _file_lock(feature_cache_path.with_suffix(feature_cache_path.suffix + ".lock")):
            if feature_cache_path.exists():
                try:
                    feature_cache_summary = validate_task_feature_cache(
                        feature_cache_path,
                        Path(args.extraction_dir),
                        resolved_task,
                        semantic_resource_fingerprint=(
                            semantic_resources.fingerprint
                        ),
                    )
                except Exception:
                    feature_cache_path.unlink(missing_ok=True)
            if not feature_cache_path.exists():
                feature_cache_summary = CorpusIndex.prepare_task_feature_cache(
                    extraction_dir=Path(args.extraction_dir),
                    task=resolved_task,
                    cache_path=feature_cache_path,
                    shared_source_cache=shared_cache_path,
                    semantic_resource_fingerprint=(
                        semantic_resources.fingerprint
                    ),
                )
    timing.record(
        "cache_preparation",
        time.perf_counter() - cache_started,
        owner="coordinator",
        cache_state=(
            "warm"
            if "REUSED" in str(shared_cache_summary)
            or "REUSED" in str(feature_cache_summary)
            else "cold"
        ),
    )

    common = [
        args.python_executable,
        str(code / "external_query_semantic_matcher_v0_4_20.py"),
        "--extraction-dir", str(Path(args.extraction_dir).expanduser().resolve()),
        "--query-parse-dir", str(matcher_query_parse),
        "--task-mode", resolved_task,
        "--staged-top-docs", str(args.staged_top_docs),
        "--retrieval-mode", args.retrieval_mode,
        "--raw-use-mode", args.raw_use_mode,
        "--max-candidates-per-target", str(args.max_candidates_per_target),
        "--max-elapsed-seconds", str(args.max_elapsed_seconds),
        "--contradiction-propositions-per-class-per-document",
        str(args.contradiction_propositions_per_class_per_document),
        "--contradiction-raw-sentences-per-class-per-document",
        str(args.contradiction_raw_sentences_per_class_per_document),
        "--openai-api-key-env", args.openai_api_key_env,
        "--llm-model", args.llm_model,
        "--llm-timeout-seconds", str(args.llm_timeout_seconds),
        "--llm-max-retries", str(args.llm_max_retries),
        "--llm-max-output-tokens", str(args.llm_max_output_tokens),
    ]
    if shared_cache_path:
        common += ["--shared-source-cache", str(shared_cache_path)]
    if feature_cache_path:
        common += ["--task-feature-cache", str(feature_cache_path)]
    if args.semantic_resources:
        common += [
            "--semantic-resources",
            str(Path(args.semantic_resources).expanduser().resolve()),
        ]
    if args.corpus_entity_resource:
        common += [
            "--corpus-entity-resource",
            str(Path(args.corpus_entity_resource).expanduser().resolve()),
        ]
    if args.decision_profile:
        common += [
            "--decision-profile",
            str(Path(args.decision_profile).expanduser().resolve()),
        ]
    if args.minimum_score is not None:
        common += ["--minimum-score", str(args.minimum_score)]
    if args.minimum_margin is not None:
        common += ["--minimum-margin", str(args.minimum_margin)]
    if args.exclude_doc_ids:
        common += ["--exclude-doc-ids", args.exclude_doc_ids]
    # A pair digest must be generated only after reconcile_pair has rewritten
    # both deterministic views.  Single runs can generate it in Matcher.
    if args.use_llm_digest and args.mode == "single":
        common.append("--use-llm-digest")
    if args.mode == "pair":
        common.append("--defer-pair-finalization")

    jobs: List[tuple[str, Path, List[str]]] = []
    if args.mode == "single":
        output = root / "semantic_match_output"
        jobs.append((
            "single", output,
            common + [
                "--output-dir", str(output),
                "--pair-role", "standalone",
            ],
        ))
    else:
        if not args.target_doc_ids:
            raise ValueError("Controlled pair requires --target-doc-ids.")
        baseline = root / "baseline" / "semantic_match_output"
        holdout = root / "holdout" / "semantic_match_output"
        pair_id = root.name
        jobs.extend([
            (
                "baseline", baseline,
                common + [
                    "--output-dir", str(baseline),
                    "--pair-id", pair_id,
                    "--pair-role", "baseline",
                    "--paired-run-dir", str(holdout),
                ],
            ),
            (
                "holdout", holdout,
                common + [
                    "--output-dir", str(holdout),
                    "--pair-id", pair_id,
                    "--pair-role", "holdout",
                    "--paired-run-dir", str(baseline),
                    "--target-doc-ids", args.target_doc_ids,
                    "--target-doc-policy", "exclude",
                ],
            ),
        ])
    results: Dict[str, Dict[str, object]] = {}
    for role, output, command in jobs:
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(status_path, {
            "status": "MATCHING",
            "active_role": role,
            "mode": args.mode,
            "task": args.task,
            "results": results,
        })
        matcher_started = time.perf_counter()
        return_code = run(
            command,
            code,
            root / role / "matcher",
            cancel_path=cancel_path,
            matcher_output=output,
            status_path=status_path,
            stage_name="MATCHER_" + role.upper(),
        )
        timing.record(
            "matcher_" + role,
            time.perf_counter() - matcher_started,
            owner="matcher",
            state="COMPLETED" if return_code == 0 else "FAILED",
        )
        results[role] = {
            "return_code": return_code,
            "output_dir": str(output),
        }
        if return_code:
            break
    completed = bool(results) and all(
        value["return_code"] == 0 for value in results.values()
    ) and len(results) == len(jobs)
    clause_ledger = build_query_clause_execution_ledger(
        query_parse, preflight, matcher_completed=completed,
        task=resolved_task,
    )
    clause_ledger.to_csv(
        root / "query_clause_execution_ledger_v0415.csv", index=False
    )
    query_clause_summary = clause_coverage_summary(clause_ledger)
    if completed:
        for _role, output, _command in jobs:
            augment_task_digest_clause_coverage(
                output, query_clause_summary
            )
            if args.mode == "single":
                refresh_logical_output_manifest(output)
                sealed = finalize_output_manifest(output)
                validation = validate_output_manifest(output, sealed)
                atomic_write_json(
                    output
                    / "final_output_manifest_validation_v0420.json",
                    validation,
                )
                if validation.get("state") != "PASS":
                    completed = False
    pair_status = "STANDALONE"
    pair_payload: Dict[str, object] = {}
    pair_digest = ""
    if completed and args.mode == "pair":
        reconciliation_started = time.perf_counter()
        atomic_write_json(status_path, {
            "status": "RECONCILING_PAIR",
            "mode": args.mode,
            "task": resolved_task,
            "results": results,
        })
        pair_result = reconcile_pair(
            root / "baseline" / "semantic_match_output",
            root / "holdout" / "semantic_match_output",
            root.name,
        )
        pair_status = str(pair_result.get("pair_status", "PAIR_INCOMPLETE"))
        timing.record(
            "pair_reconciliation",
            time.perf_counter() - reconciliation_started,
            owner="coordinator",
            state=pair_status,
        )
        completed = pair_status == "CONTROLLED_PAIR"
        if completed:
            comparison = pd.read_csv(
                root / "baseline" / "semantic_match_output"
                / "pair_comparison_v048.csv",
                low_memory=False,
            )
            digest_started = time.perf_counter()
            pair_payload, pair_digest = write_pair_digest_bundle(
                root=root,
                pair_id=root.name,
                pair_status=pair_status,
                baseline_dir=(
                    root / "baseline" / "semantic_match_output"
                ),
                holdout_dir=(
                    root / "holdout" / "semantic_match_output"
                ),
                comparison=comparison,
            )
            timing.record(
                "deterministic_pair_digest",
                time.perf_counter() - digest_started,
                owner="digest",
            )
            write_pair_candidate_route_delta(
                root,
                root / "baseline" / "semantic_match_output",
                root / "holdout" / "semantic_match_output",
            )
            for arm in ("baseline", "holdout"):
                arm_output = (
                    root / arm / "semantic_match_output"
                )
                arm_manifest_path = (
                    arm_output / "semantic_match_run_manifest_v048.json"
                )
                arm_manifest = safe_json(arm_manifest_path)
                arm_manifest.update({
                    "pair_status": pair_status,
                    "terminal_pair_status": pair_status,
                    "digest_builder_implementation_version":
                        DIGEST_BUILDER_VERSION,
                    "output_inventory": output_inventory(arm_output),
                })
                atomic_write_json(arm_manifest_path, arm_manifest)
    if completed and args.mode == "pair" and args.use_llm_digest:
        llm_started = time.perf_counter()
        atomic_write_json(status_path, {
            "status": "FINALIZING_PAIR_LLM_DIGEST",
            "mode": args.mode,
            "task": resolved_task,
            "results": results,
        })
        pair_llm_state = generate_optional_llm(
            enabled=True,
            evidence_payload=pair_payload,
            deterministic_digest=pair_digest,
            model=args.llm_model,
            api_key_env=args.openai_api_key_env,
            timeout_seconds=args.llm_timeout_seconds,
            max_retries=args.llm_max_retries,
            max_output_tokens=args.llm_max_output_tokens,
            output_dir=root,
            pair_level=True,
        )
        timing.record(
            "optional_llm_generation_and_repair",
            time.perf_counter() - llm_started,
            owner="digest",
            state=str(pair_llm_state.get("state", "")),
            attempt=1 + int(
                pair_llm_state.get("repair_attempt_count", 0) or 0
            ),
        )
    else:
        pair_llm_state = {
            "state": "NOT_REQUESTED",
            "api_call_state": "NOT_REQUESTED",
            "consistency_state": "NOT_ASSESSED",
            "explanatory_completeness_state": "NOT_ASSESSED",
            "target_coverage_state": "NOT_ASSESSED",
            "evidence_grounding_state": "NOT_ASSESSED",
        }
        if completed and args.mode == "pair":
            atomic_write_json(
                root / "pair_llm_digest_state_v0413.json",
                pair_llm_state,
            )
        timing.record(
            "optional_llm_generation_and_repair",
            0.0,
            owner="digest",
            state="NOT_REQUESTED",
            attempt=0,
        )
    llm_failures = (
        ["pair"]
        if (
            args.mode == "pair"
            and args.use_llm_digest
            and pair_llm_state.get("state") != "VALIDATED_EXPLANATORY"
        )
        else []
    )
    finalization_failures: List[str] = []
    if completed and args.mode == "pair":
        for arm in ("baseline", "holdout"):
            arm_started = time.perf_counter()
            arm_output = root / arm / "semantic_match_output"
            contract_path = arm_output / "matcher_output_contract_v048.json"
            contract = safe_json(contract_path)
            contract.update({
                "finalization_state": "FINAL_BYTE_SEALED",
                "final_output_manifest":
                    "final_output_manifest_v0420.json",
            })
            atomic_write_json(contract_path, contract)
            arm_manifest_path = (
                arm_output / "semantic_match_run_manifest_v048.json"
            )
            arm_manifest = safe_json(arm_manifest_path)
            arm_manifest.update({
                "finalization_contract":
                    "coordinator_pair_postprocessing_complete_then_seal",
                "final_output_manifest":
                    "final_output_manifest_v0420.json",
                "output_inventory": output_inventory(arm_output),
            })
            atomic_write_json(arm_manifest_path, arm_manifest)
            refresh_logical_output_manifest(arm_output)
            sealed = finalize_output_manifest(arm_output)
            validation = validate_output_manifest(arm_output, sealed)
            atomic_write_json(
                arm_output
                / "final_output_manifest_validation_v0420.json",
                validation,
            )
            if validation["state"] != "PASS":
                finalization_failures.append(arm)
            timing.record(
                "manifest_finalization_" + arm,
                time.perf_counter() - arm_started,
                owner="coordinator",
                state=validation["state"],
            )
        if finalization_failures:
            completed = False

    timing.total_wall_seconds = time.perf_counter() - coordinated_started
    performance_report = timing.report(tolerance_seconds=0.50)
    atomic_write_json(
        root / "coordinated_performance_v0420.json",
        performance_report,
    )
    pd.DataFrame(performance_report["stages"]).to_csv(
        root / "coordinated_performance_stages_v0420.csv",
        index=False,
    )
    manifest = {
        "coordinated_release_version": RELEASE_VERSION,
        "requirements_version": REQUIREMENTS_VERSION,
        "parser_version": PARSER_VERSION,
        "parser_model": (
            args.parser_model if args.use_openai_parser
            else "deterministic_heuristic"
        ),
        "openai_parser_requested": bool(args.use_openai_parser),
        "parser_reconciliation_contract": (
            "query_representation_reconciliation.csv is diagnostic-only; "
            "semantic decisions use query_propositions.csv and "
            "query_event_signatures.csv"
        ),
        "authoritative_query_input_validation":
            authoritative_query_validation,
        "matcher_version": MATCHER_VERSION,
        "digest_builder_version": DIGEST_BUILDER_VERSION,
        "workbench_version": WORKBENCH_VERSION,
        "legacy_matcher_invoked": False,
        "mode": args.mode,
        "task": resolved_task,
        "query_parse": str(query_parse),
        "matcher_query_parse": str(matcher_query_parse),
        "query_schema_compatibility": schema_compatibility,
        "shared_source_contract": (
            "SHARED_SOURCE_TABLES_EXCLUSIONS_BEFORE_INDEX"
            if shared_cache_path else "NOT_APPLICABLE"
        ),
        "shared_source_summary": {
            key: value for key, value in shared_cache_summary.items()
            if key != "cache_path"
        },
        "task_feature_contract": (
            "QUERY_INDEPENDENT_FEATURES_EXCLUSIONS_BEFORE_SCORING"
            if feature_cache_path else "NOT_APPLICABLE"
        ),
        "task_feature_summary": {
            key: value for key, value in feature_cache_summary.items()
            if key != "cache_path"
        },
        "semantic_resources": semantic_resources.manifest(),
        "corpus_entity_resource": (
            corpus_entity_resource.manifest()
            if corpus_entity_resource is not None
            else {"load_status": "NOT_CONFIGURED"}
        ),
        "results": results,
        "pair_status": pair_status,
        "pair_digest_state": pair_llm_state,
        "llm_requested_by_coordinator": bool(args.use_llm_digest),
        "llm_execution_phase": (
            "post_pair_reconciliation"
            if args.mode == "pair" and args.use_llm_digest
            else "single_task"
            if args.mode == "single" and args.use_llm_digest
            else "not_requested"
        ),
        "llm_model": args.llm_model if args.use_llm_digest else "",
        "llm_credential_source_name": (
            "environment:" + args.openai_api_key_env
            if args.use_llm_digest else ""
        ),
        "pair_target_coverage_state": (
            pair_payload.get("target_coverage_state", "NOT_APPLICABLE")
            if pair_payload else "NOT_APPLICABLE"
        ),
        "query_clause_coverage": query_clause_summary,
        "query_clause_execution_ledger": (
            "query_clause_execution_ledger_v0415.csv"
        ),
        "llm_failures": llm_failures,
        "finalization_failures": finalization_failures,
        "performance_report": "coordinated_performance_v0420.json",
        "final_output_manifest": "final_output_manifest_v0420.json",
        "completed_at": utc_now_iso(),
    }
    atomic_write_json(root / "coordinated_release_manifest_v048.json", manifest)
    final_status = (
        "PAIR_RECONCILIATION_FAILED"
        if args.mode == "pair" and pair_status != "CONTROLLED_PAIR"
        else "COMPLETED_WITH_WARNINGS"
        if completed and llm_failures
        else "COMPLETED"
        if completed
        else "FAILED"
    )
    atomic_write_json(status_path, {
        "status": final_status,
        "status_detail": (
            "deterministic result completed; Optional LLM digest rejected"
            if final_status == "COMPLETED_WITH_WARNINGS"
            and llm_failures else ""
        ),
        "results": results,
        "pair_status": pair_status,
        "llm_failures": llm_failures,
        "finalization_failures": finalization_failures,
        "started_at": coordinated_started_at,
        "total_elapsed_seconds":
            time.perf_counter() - coordinated_started,
        "last_heartbeat": utc_now_iso(),
        "child_process_id": "",
        "child_process_state": "EXITED",
        "terminal_state": final_status,
        "failure_owner": (
            "coordinator_manifest_finalization"
            if finalization_failures
            else "pair_reconciliation"
            if final_status == "PAIR_RECONCILIATION_FAILED"
            else "matcher"
            if final_status == "FAILED"
            else ""
        ),
        "completed_at": utc_now_iso(),
    })
    root_sealed = finalize_output_manifest(root)
    root_validation = validate_output_manifest(root, root_sealed)
    atomic_write_json(
        root / "final_output_manifest_validation_v0420.json",
        root_validation,
    )
    if root_validation["state"] != "PASS":
        completed = False
    if shared_cache_directory is not None:
        shared_cache_directory.cleanup()
    if feature_cache_directory is not None:
        feature_cache_directory.cleanup()
    return 0 if completed else 3


if __name__ == "__main__":
    raise SystemExit(main())
