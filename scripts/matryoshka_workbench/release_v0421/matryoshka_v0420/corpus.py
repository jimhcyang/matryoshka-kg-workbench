"""Single-load immutable corpus index used by every task engine.

Task engines never open CSV files.  All source I/O, exclusions, normalization,
tokenization, proposition assembly, value extraction, raw sentence indexing,
and provenance indexing happen exactly once here.
"""
from __future__ import annotations

import contextlib
import json
import os
import pickle
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from .progress import RunControl
from .semantic_resources import SemanticResources
from .semantics import (
    NORMALIZATION_VERSION,
    atomic_write_json,
    build_event_signature,
    canonical_key,
    clean_text,
    infer_simple_clause_arguments,
    semantic_tokens,
    sha256_file,
    stable_hash,
)


ALL_SOURCE_TABLES = [
    "chunks.csv", "entities.csv", "claims.csv", "events.csv",
    "relations.csv", "kg_pseudo_chunks.csv",
    "kg_pseudo_chunks_reconstruction.csv",
    "kg_pseudo_chunks_redaction.csv",
    "kg_pseudo_chunks_claims.csv",
    "kg_pseudo_chunks_contradictions.csv",
    "kg_pseudo_chunk_components.csv",
]

TASK_SOURCE_TABLES = {
    "reconstruction": [
        "chunks.csv", "entities.csv", "claims.csv", "events.csv", "relations.csv",
        "kg_pseudo_chunks.csv", "kg_pseudo_chunks_reconstruction.csv",
        "kg_pseudo_chunk_components.csv",
    ],
    "redaction": [
        "entities.csv", "claims.csv", "events.csv",
        "relations.csv", "kg_pseudo_chunks.csv",
        "kg_pseudo_chunks_redaction.csv",
        "kg_pseudo_chunk_components.csv",
    ],
    "sentence_redaction": [
        "chunks.csv", "entities.csv", "claims.csv", "events.csv",
        "relations.csv", "kg_pseudo_chunks.csv",
        "kg_pseudo_chunk_components.csv",
    ],
    "claims": [
        "chunks.csv", "entities.csv", "claims.csv", "events.csv", "relations.csv",
        "kg_pseudo_chunks.csv",
        "kg_pseudo_chunks_claims.csv",
        "kg_pseudo_chunk_components.csv",
    ],
    "contradictions": [
        "chunks.csv", "entities.csv", "claims.csv", "events.csv", "relations.csv",
        "kg_pseudo_chunks.csv",
        "kg_pseudo_chunks_contradictions.csv",
        "kg_pseudo_chunk_components.csv",
    ],
    "all": ALL_SOURCE_TABLES,
}

SHARED_SOURCE_CACHE_VERSION = "shared_source_tables_v0_4_17"
TASK_FEATURE_CACHE_VERSION = "task_features_v0_4_17"
CACHE_METADATA_VERSION = "matryoshka_cache_metadata_v1"

TEXT_COLUMN_PRIORITY = (
    "body", "chunk_text", "text", "raw_text", "claim_text",
    "event_text", "evidence_text", "evidence", "component_text",
    "pseudo_view_text", "pseudo_chunk_text", "proposition_text",
)


def feature_implementation_fingerprint() -> str:
    """Identify code that materializes query-independent cached features."""
    package = Path(__file__).resolve().parent
    paths = (
        Path(__file__).resolve(),
        package / "semantics.py",
        package / "semantic_resources.py",
    )
    return stable_hash({
        path.name: sha256_file(path)
        for path in paths
    })


def source_tables_for_task(task: str) -> List[str]:
    return list(TASK_SOURCE_TABLES.get(task, ALL_SOURCE_TABLES))


def _cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _write_cache_metadata(
    cache_path: Path,
    *,
    cache_kind: str,
    payload_metadata: Mapping[str, Any],
) -> None:
    """Write a small validation header beside a potentially huge pickle."""
    target = cache_path.expanduser().resolve()
    stat = target.stat()
    atomic_write_json(
        _cache_metadata_path(target),
        {
            "schema_version": CACHE_METADATA_VERSION,
            "cache_kind": cache_kind,
            "cache_size_bytes": int(stat.st_size),
            "cache_mtime_ns": int(stat.st_mtime_ns),
            "payload_metadata": dict(payload_metadata),
        },
    )


def _read_cache_metadata(
    cache_path: Path,
    *,
    cache_kind: str,
) -> Optional[Dict[str, Any]]:
    """Return trusted sidecar metadata when it still describes the pickle."""
    target = cache_path.expanduser().resolve()
    metadata_path = _cache_metadata_path(target)
    if not metadata_path.is_file() or not target.is_file():
        return None
    try:
        envelope = json.loads(metadata_path.read_text(encoding="utf-8"))
        stat = target.stat()
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(envelope, dict):
        return None
    if envelope.get("schema_version") != CACHE_METADATA_VERSION:
        return None
    if envelope.get("cache_kind") != cache_kind:
        return None
    if int(envelope.get("cache_size_bytes", -1)) != int(stat.st_size):
        return None
    if int(envelope.get("cache_mtime_ns", -1)) != int(stat.st_mtime_ns):
        return None
    payload_metadata = envelope.get("payload_metadata")
    return payload_metadata if isinstance(payload_metadata, dict) else None


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def _atomic_write_pickle(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a pickle without sharing a temporary filename.

    Coordinated runs lock cache preparation, but direct matcher invocations or
    interrupted analyst sessions can overlap.  A unique sibling prevents one
    writer from renaming another writer's temporary file.
    """
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(target.parent),
            prefix=target.name + ".tmp.",
            delete=False,
        ) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_name = handle.name
        os.replace(temporary_name, target)
        temporary_name = ""
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def _source_signature_cache_path(root: Path) -> Path:
    return root / ".source_table_signature_cache_v0420.json"


def _load_source_signature_cache(root: Path) -> Dict[str, Dict[str, Any]]:
    path = _source_signature_cache_path(root)
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in entries.items()
        if isinstance(value, dict)
    }


def _store_source_signature_cache(
    root: Path,
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = {
        "version": "source_table_signature_cache_v0420",
        "entries": {
            str(key): {
                "size_bytes": int(value.get("size_bytes", 0)),
                "mtime_ns": int(value.get("mtime_ns", 0)),
                "sha256": clean_text(value.get("sha256", "")),
            }
            for key, value in entries.items()
        },
    }
    with contextlib.suppress(Exception):
        atomic_write_json(_source_signature_cache_path(root), payload)


def _doc_column(frame: pd.DataFrame) -> str:
    return next(
        (name for name in ("doc_id", "source_id", "document_id") if name in frame.columns),
        "",
    )


def _text_column(frame: pd.DataFrame) -> str:
    return next(
        (
            name for name in TEXT_COLUMN_PRIORITY
            if name in frame.columns
        ),
        "",
    )


def _row_text(
    row: Mapping[str, Any], columns: Sequence[str]
) -> Tuple[str, str]:
    """Return the first nonblank compatible text field for one row."""
    for column in columns:
        value = clean_text(row.get(column, ""))
        if value:
            return value, column
    return "", ""


def _tuple_value(
    row: Tuple[Any, ...],
    column_index: Mapping[str, int],
    column: str,
) -> Any:
    """Fast positional accessor for ``itertuples(name=None)`` rows."""
    index = column_index.get(column)
    if index is None:
        return ""
    return row[index]


def _json_string_list(value: Any) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [
        clean_text(item)
        for item in payload
        if clean_text(item)
    ]


def _sentences(text: str) -> List[str]:
    import re
    raw_parts = [
        clean_text(part)
        for part in re.split(r"(?<=[.!?])\s+|\n+", clean_text(text))
        if clean_text(part)
    ]
    # Numbered cable paragraphs commonly appear as ``1. TEXT 2. TEXT``.
    # A naive sentence splitter turns ``1.``, ``2.``, and ``3.`` into
    # standalone sentences.  They are structural separators, not semantic
    # neighbors.  Removing the marker while retaining the following sentence
    # gives the same adjacency graph as an unnumbered rendering and prevents a
    # marker from breaking sentence-redaction certification.
    marker = re.compile(
        r"^(?:\(?\d{1,3}\)?[.)]|\(?[ivxlcdm]{1,8}\)?[.)]|[A-Z][.)])$",
        flags=re.I,
    )
    repaired: List[str] = []
    pending_marker = False
    for part in raw_parts:
        if marker.fullmatch(part):
            pending_marker = True
            continue
        # ``pending_marker`` is intentionally not prepended: exact selected
        # sentence text must remain source content without the list label.
        repaired.append(part)
        pending_marker = False
    return repaired


def source_table_signature(root: Path, table_names: Sequence[str]) -> str:
    """Fingerprint exact source-table contents.

    File size and modification time are insufficient cache identities: content
    can change while size remains constant and timestamps can be preserved.
    """
    cache = _load_source_signature_cache(root)
    cache_changed = False
    rows = []
    for name in table_names:
        path = root / name
        stat = path.stat() if path.is_file() else None
        sha256 = ""
        if stat:
            cached = cache.get(name, {})
            if (
                int(cached.get("size_bytes", -1)) == int(stat.st_size)
                and int(cached.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
                and clean_text(cached.get("sha256", ""))
            ):
                sha256 = clean_text(cached.get("sha256", ""))
            else:
                sha256 = sha256_file(path)
                cache[name] = {
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "sha256": sha256,
                }
                cache_changed = True
        rows.append({
            "name": name,
            "exists": bool(stat),
            "size_bytes": stat.st_size if stat else 0,
            "sha256": sha256 if stat else "",
        })
    if cache_changed:
        _store_source_signature_cache(root, cache)
    return stable_hash(rows)


def validate_shared_source_cache(
    cache_path: Path,
    extraction_dir: Path,
    task: str,
) -> Dict[str, Any]:
    """Raise ValueError when a shared source cache is stale or incompatible."""
    target = cache_path.expanduser().resolve()
    root = extraction_dir.expanduser().resolve()
    payload = _read_cache_metadata(target, cache_kind="shared_source")
    if payload is None:
        with target.open("rb") as handle:
            full_payload = pickle.load(handle)
        payload = {
            "version": full_payload.get("version"),
            "task": full_payload.get("task", "all"),
            "table_names": full_payload.get("table_names", []),
            "table_signatures": full_payload.get("table_signatures", {}),
            "table_count": len(full_payload.get("tables", {})),
        } if isinstance(full_payload, dict) else {}
        if payload:
            _write_cache_metadata(
                target,
                cache_kind="shared_source",
                payload_metadata=payload,
            )
    if not isinstance(payload, dict) or payload.get(
        "version"
    ) != SHARED_SOURCE_CACHE_VERSION:
        raise ValueError("Unsupported shared source cache contract.")
    # The stored path is audit metadata only. Exact source-table hashes below
    # decide whether a cache remains valid after a package move or rename.
    table_names = source_tables_for_task(task)
    cache_task = payload.get("task", "all")
    cached_names = set(payload.get("table_names", []))
    if cache_task not in {task, "all"} or not set(table_names).issubset(cached_names):
        raise ValueError(
            "Shared source cache belongs to a different task profile."
        )
    signatures = payload.get("table_signatures", {})
    if any(
        signatures.get(name) != source_table_signature(root, [name])
        for name in table_names
    ):
        raise ValueError(
            "Shared source cache is stale because extraction tables changed."
        )
    return {
        "cache_path": str(target),
        "cache_state": "REUSED_EXISTING_VALIDATED",
        "task": cache_task,
        "table_count": int(payload.get("table_count", 0)),
    }


def validate_task_feature_cache(
    cache_path: Path,
    extraction_dir: Path,
    task: str,
    semantic_resource_fingerprint: str = "",
) -> Dict[str, Any]:
    """Raise ValueError when a task feature cache is stale or incompatible."""
    target = cache_path.expanduser().resolve()
    root = extraction_dir.expanduser().resolve()
    payload = _read_cache_metadata(target, cache_kind="task_feature")
    if payload is None:
        with target.open("rb") as handle:
            full_payload = pickle.load(handle)
        payload = {
            "version": full_payload.get("version"),
            "task": full_payload.get("task"),
            "normalization_version": full_payload.get("normalization_version"),
            "feature_implementation_fingerprint": full_payload.get(
                "feature_implementation_fingerprint", ""
            ),
            "source_table_signature": full_payload.get("source_table_signature", ""),
            "semantic_resource_fingerprint": full_payload.get(
                "semantic_resource_fingerprint", ""
            ),
            "document_count": int(full_payload.get("document_count", 0)),
        } if isinstance(full_payload, dict) else {}
        if payload:
            _write_cache_metadata(
                target,
                cache_kind="task_feature",
                payload_metadata=payload,
            )
    if not isinstance(payload, dict) or payload.get(
        "version"
    ) != TASK_FEATURE_CACHE_VERSION:
        raise ValueError("Unsupported task feature cache contract.")
    # Do not bind a query-independent cache to an absolute installation path.
    # Source signatures, implementation hashes, and resource fingerprints are
    # the authoritative compatibility checks.
    if payload.get("task") != task:
        raise ValueError("Task feature cache belongs to a different task.")
    if payload.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("Task feature cache normalization version is stale.")
    if payload.get(
        "feature_implementation_fingerprint", ""
    ) != feature_implementation_fingerprint():
        raise ValueError(
            "Task feature cache is stale because feature-building code changed."
        )
    expected_source_signature = source_table_signature(
        root,
        source_tables_for_task(task),
    )
    if payload.get("source_table_signature") != expected_source_signature:
        raise ValueError(
            "Task feature cache is stale because extraction tables changed."
        )
    if payload.get(
        "semantic_resource_fingerprint", ""
    ) != semantic_resource_fingerprint:
        raise ValueError(
            "Task feature cache is stale because semantic resources changed."
        )
    return {
        "cache_path": str(target),
        "cache_state": "REUSED_EXISTING_VALIDATED",
        "task": task,
        "document_count": int(payload.get("document_count", 0)),
        "source_table_signature": expected_source_signature,
        "semantic_resource_fingerprint": semantic_resource_fingerprint,
        "normalization_version": NORMALIZATION_VERSION,
    }


@dataclass(frozen=True)
class PropositionRecord:
    doc_id: str
    source_table: str
    row_id: str
    text: str
    signature: Mapping[str, Any]
    report_time: str


@dataclass(frozen=True)
class ValueRecord:
    doc_id: str
    source_table: str
    row_id: str
    value: str
    value_type: str
    evidence: str
    relation: str = ""
    relation_side: str = ""


@dataclass(frozen=True)
class SentenceRecord:
    doc_id: str
    chunk_id: str
    sentence_index: int
    left_neighbor: str
    sentence: str
    right_neighbor: str
    source_text_column: str = ""


@dataclass(frozen=True)
class ProvenanceRecord:
    doc_id: str
    component_id: str
    source_table: str
    source_row_id: str
    component_text: str


@dataclass(frozen=True)
class CorpusIndex:
    extraction_dir: Path
    excluded_doc_ids: frozenset[str]
    requested_task: str
    document_text: Mapping[str, str]
    document_tokens: Mapping[str, frozenset[str]]
    document_dates: Mapping[str, str]
    propositions_by_doc: Mapping[str, Tuple[PropositionRecord, ...]]
    values_by_doc: Mapping[str, Tuple[ValueRecord, ...]]
    sentences_by_doc: Mapping[str, Tuple[SentenceRecord, ...]]
    provenance_by_doc: Mapping[str, Tuple[ProvenanceRecord, ...]]
    fingerprint: str
    load_statistics: Mapping[str, Mapping[str, Any]]

    @property
    def doc_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self.document_text))

    @classmethod
    def build(
        cls,
        extraction_dir: Path,
        excluded_doc_ids: Iterable[str] = (),
        control: Optional[RunControl] = None,
        task: str = "all",
        shared_source_cache: Optional[Path] = None,
        task_feature_cache: Optional[Path] = None,
        semantic_resource_fingerprint: str = "",
        semantic_resources: Optional[SemanticResources] = None,
    ) -> "CorpusIndex":
        root = extraction_dir.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError("Extraction directory not found: " + str(root))
        excluded = frozenset(clean_text(value).upper() for value in excluded_doc_ids if clean_text(value))
        table_names = source_tables_for_task(task)
        expected_source_signature = source_table_signature(root, table_names)
        feature_cache_path = (
            task_feature_cache.expanduser().resolve()
            if task_feature_cache else None
        )
        if feature_cache_path and feature_cache_path.is_file():
            if control:
                control.emit(
                    "task_feature_cache_validation_started",
                    task=task,
                    cache_path=str(feature_cache_path),
                )
            try:
                return cls._from_task_feature_cache(
                    cache_path=feature_cache_path,
                    extraction_dir=root,
                    expected_source_signature=expected_source_signature,
                    task=task,
                    excluded=excluded,
                    control=control,
                    expected_semantic_resource_fingerprint=(
                        semantic_resource_fingerprint
                    ),
                )
            except ValueError as exc:
                if control:
                    control.emit(
                        "task_feature_cache_invalidated",
                        task=task,
                        cache_path=str(feature_cache_path),
                        reason=str(exc),
                    )
                with contextlib.suppress(OSError):
                    feature_cache_path.unlink()
        cached_tables: Dict[str, pd.DataFrame] = {}
        cache_path = (
            shared_source_cache.expanduser().resolve()
            if shared_source_cache else None
        )
        if cache_path and cache_path.is_file():
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict) or payload.get(
                "version"
            ) != SHARED_SOURCE_CACHE_VERSION:
                raise ValueError("Unsupported shared source cache contract.")
            # The package may have moved. Content signatures below, rather
            # than the recorded audit path, determine compatibility.
            cache_task = payload.get("task", "all")
            cached_names = set(payload.get("table_names", []))
            if cache_task not in {task, "all"} or not set(
                table_names
            ).issubset(cached_names):
                raise ValueError(
                    "Shared source cache belongs to a different task profile."
                )
            signatures = payload.get("table_signatures", {})
            if any(
                signatures.get(name)
                != source_table_signature(root, [name])
                for name in table_names
            ):
                raise ValueError(
                    "Shared source cache is stale because extraction tables "
                    "changed. Rebuild the cache."
                )
            cached_tables = payload.get("tables", {})
            if not isinstance(cached_tables, dict):
                raise ValueError("Shared source cache has no table mapping.")

        tables: Dict[str, pd.DataFrame] = {}
        statistics: Dict[str, Dict[str, Any]] = {}
        for name in table_names:
            cache_hit = name in cached_tables
            frame = (
                cached_tables[name].copy(deep=False)
                if cache_hit else _safe_csv(root / name)
            )
            before = len(frame)
            doc_col = _doc_column(frame)
            removed = 0
            if doc_col:
                normalized = frame[doc_col].fillna("").astype(str).str.upper()
                removed = int(normalized.isin(excluded).sum())
                frame = frame.loc[~normalized.isin(excluded)].copy()
                frame[doc_col] = frame[doc_col].fillna("").astype(str).str.upper()
            tables[name] = frame
            statistics[name] = {
                "read_count": 0 if cache_hit else 1,
                "shared_cache_read_count": 1 if cache_hit else 0,
                "source_state": "SHARED_SOURCE_CACHE" if cache_hit else "CSV",
                "rows_before_exclusion": before,
                "rows_removed": removed,
                "rows_after_exclusion": len(frame),
            }
            if control:
                control.emit(
                    "corpus_table_loaded",
                    source_table=name,
                    rows_before=before,
                    rows_removed=removed,
                    rows_after=len(frame),
                )
                control.check()
        for name in ALL_SOURCE_TABLES:
            tables.setdefault(name, pd.DataFrame())

        text_parts: Dict[str, List[str]] = defaultdict(list)
        seen_text_parts: Dict[str, Set[str]] = defaultdict(set)
        dates: Dict[str, str] = {}
        total_source_rows = sum(len(frame) for frame in tables.values())
        processed_source_rows = 0
        if control:
            control.emit(
                "corpus_document_assembly_started",
                source_row_count=total_source_rows,
            )
        raw_surface_columns = {"body", "chunk_text", "raw_text"}
        for name, frame in tables.items():
            doc_col = _doc_column(frame)
            if not doc_col:
                processed_source_rows += len(frame)
                continue
            if name == "kg_pseudo_chunk_components.csv":
                # Component rows are a provenance bridge, not a retrieval
                # surface. Index them later for passage provenance when the
                # task needs it, but do not bloat document retrieval text with
                # hundreds of thousands of near-duplicate component snippets.
                processed_source_rows += len(frame)
                if control and len(frame):
                    control.emit(
                        "corpus_document_assembly_skipped_table",
                        source_table=name,
                        table_row_count=len(frame),
                        reason="provenance_only_component_bridge",
                    )
                    control.check()
                continue
            candidate_text_columns = [
                column for column in (
                    "body", "chunk_text", "text", "raw_text", "claim_text",
                    "event_text", "evidence_text", "evidence", "source",
                    "relation", "target", "qualifier", "component_text",
                    "pseudo_view_text", "pseudo_chunk_text", "proposition_text",
                    "semantic_signature_text", "candidate_label",
                    "candidate_text", "candidate_role_or_context",
                    "candidate_role_context", "canonical_subject",
                    "canonical_predicate", "canonical_object",
                    "canonical_location", "object_or_target",
                    "supporting_entities", "supporting_relations",
                )
                if column in frame.columns
            ]
            if task == "redaction":
                # For value redaction, first-stage document retrieval should
                # lean on structured semantic surfaces more than raw passage
                # text so rephrased prompts are not dominated by literal chunk
                # overlap. Candidate-local certification can still inspect
                # source-side evidence later through values_by_doc.
                candidate_text_columns = [
                    column
                    for column in candidate_text_columns
                    if column not in raw_surface_columns
                ]
            for row_number, row in enumerate(
                frame.itertuples(index=False), start=1
            ):
                data = row._asdict()
                doc_id = clean_text(data.get(doc_col, "")).upper()
                if not doc_id:
                    continue
                doc_parts = text_parts[doc_id]
                doc_seen = seen_text_parts[doc_id]
                for column in candidate_text_columns:
                    value = clean_text(data.get(column, ""))
                    if not value or value in doc_seen:
                        continue
                    doc_parts.append(value)
                    doc_seen.add(value)
                if clean_text(data.get("date", "")):
                    dates[doc_id] = clean_text(data.get("date", ""))
                if control and row_number % 100000 == 0:
                    control.emit(
                        "corpus_document_assembly_progress",
                        source_table=name,
                        table_rows_processed=row_number,
                        table_row_count=len(frame),
                        source_rows_processed=(
                            processed_source_rows + row_number
                        ),
                        source_row_count=total_source_rows,
                    )
                    control.check()
            processed_source_rows += len(frame)

        document_text = {
            doc_id: clean_text(" ".join(parts))
            for doc_id, parts in text_parts.items()
        }
        document_tokens: Dict[str, frozenset[str]] = {}
        for document_number, (doc_id, text) in enumerate(
            document_text.items(), start=1
        ):
            document_tokens[doc_id] = frozenset(
                semantic_resources.expanded_tokens(text)
                if semantic_resources is not None
                else semantic_tokens(text)
            )
            if control and document_number % 250 == 0:
                control.emit(
                    "corpus_document_tokenization_progress",
                    documents_processed=document_number,
                    document_count=len(document_text),
                )
                control.check()
        if control:
            control.emit(
                "corpus_document_assembly_ready",
                document_count=len(document_text),
                source_rows_processed=processed_source_rows,
            )
            control.check()

        needs_propositions = task in {
            "all", "claims", "contradictions", "reconstruction", "redaction"
        }
        needs_values = task in {"all", "redaction"}
        needs_sentences = task in {
            "all", "claims", "contradictions", "reconstruction",
            "sentence_redaction",
        }
        needs_provenance = task in {"all", "reconstruction"}

        propositions: Dict[str, List[PropositionRecord]] = defaultdict(list)
        if needs_propositions:
            if control:
                control.emit("corpus_proposition_index_started")
            cls._index_claims(tables["claims.csv"], propositions)
            cls._index_events(tables["events.csv"], propositions)
            cls._index_relations(tables["relations.csv"], propositions)
            # Task-specific pseudo views are proposition sources, not merely
            # document-retrieval text.  This matters when an extraction's
            # canonical claims table omits a proposition that remains present
            # in the non-lossy task view (for example, an uncertainty clause).
            # Indexing the explicit proposition_text keeps the correction
            # corpus-independent and preserves its actual source provenance.
            pseudo_table = {
                "claims": "kg_pseudo_chunks_claims.csv",
                "contradictions": "kg_pseudo_chunks_contradictions.csv",
                "reconstruction": "kg_pseudo_chunks_reconstruction.csv",
                "redaction": "kg_pseudo_chunks_redaction.csv",
            }.get(task, "")
            if pseudo_table:
                cls._index_task_pseudo_propositions(
                    tables[pseudo_table],
                    pseudo_table,
                    dates,
                    propositions,
                )
        for doc_id, records in propositions.items():
            canonical_texts = {
                clean_text(record.text).lower()
                for record in records
                if not record.source_table.startswith("kg_pseudo_chunks_")
            }
            unique: Dict[Tuple[str, str], PropositionRecord] = {}
            for record in records:
                text_key = clean_text(record.text).lower()
                if (
                    record.source_table.startswith("kg_pseudo_chunks_")
                    and text_key in canonical_texts
                ):
                    # Pseudo views remain important when they preserve a
                    # proposition omitted by the canonical extraction tables.
                    # When the exact same proposition text is already present
                    # from a primary source table, keeping both only doubles
                    # candidate work without adding semantic coverage.
                    continue
                unique.setdefault((record.source_table, text_key), record)
            propositions[doc_id] = list(unique.values())
        if control and needs_propositions:
            control.emit(
                "corpus_proposition_index_ready",
                proposition_count=sum(
                    len(rows) for rows in propositions.values()
                ),
            )
            control.check()

        values: Dict[str, List[ValueRecord]] = defaultdict(list)
        if needs_values:
            if control:
                control.emit("corpus_value_index_started")
            cls._index_values(tables, values)
            if control:
                control.emit(
                    "corpus_value_index_ready",
                    value_count=sum(len(rows) for rows in values.values()),
                )
                control.check()
        sentences: Dict[str, List[SentenceRecord]] = defaultdict(list)
        if needs_sentences:
            if control:
                control.emit("corpus_sentence_index_started")
            cls._index_sentences(tables["chunks.csv"], sentences)
            if control:
                control.emit(
                    "corpus_sentence_index_ready",
                    sentence_count=sum(
                        len(rows) for rows in sentences.values()
                    ),
                )
                control.check()
        provenance: Dict[str, List[ProvenanceRecord]] = defaultdict(list)
        if needs_provenance:
            if control:
                control.emit("corpus_provenance_index_started")
            cls._index_provenance(
                tables["kg_pseudo_chunk_components.csv"], provenance, tables
            )
            if control:
                control.emit(
                    "corpus_provenance_index_ready",
                    provenance_count=sum(
                        len(rows) for rows in provenance.values()
                    ),
                )
                control.check()

        fingerprint_payload = {
            "excluded_doc_ids": sorted(excluded),
            "tables": {
                name: {
                    "rows": len(frame),
                    "columns": list(frame.columns),
                }
                for name, frame in tables.items()
            },
            "document_count": len(document_text),
            "index_version": "corpus_index_v0_4_17",
            "requested_task": task,
            "source_table_signature": expected_source_signature,
            "semantic_resource_fingerprint": semantic_resource_fingerprint,
            "normalization_version": NORMALIZATION_VERSION,
        }
        fingerprint = stable_hash(fingerprint_payload)
        if control:
            control.emit(
                "corpus_index_ready",
                document_count=len(document_text),
                proposition_count=sum(len(rows) for rows in propositions.values()),
                value_count=sum(len(rows) for rows in values.values()),
                sentence_count=sum(len(rows) for rows in sentences.values()),
                provenance_count=sum(len(rows) for rows in provenance.values()),
                corpus_fingerprint=fingerprint,
            )

        index = cls(
            extraction_dir=root,
            excluded_doc_ids=excluded,
            requested_task=task,
            document_text=MappingProxyType(document_text),
            document_tokens=MappingProxyType(document_tokens),
            document_dates=MappingProxyType(dates),
            propositions_by_doc=MappingProxyType({
                key: tuple(value) for key, value in propositions.items()
            }),
            values_by_doc=MappingProxyType({
                key: tuple(value) for key, value in values.items()
            }),
            sentences_by_doc=MappingProxyType({
                key: tuple(value) for key, value in sentences.items()
            }),
            provenance_by_doc=MappingProxyType({
                key: tuple(value) for key, value in provenance.items()
            }),
            fingerprint=fingerprint,
            load_statistics=MappingProxyType(statistics),
        )
        # A direct matcher invocation can encounter a cache invalidated by a
        # code or extraction update.  Persist the rebuilt, unexcluded baseline
        # index so subsequent cases do not repeat the full corpus build.  A
        # holdout index is intentionally never written as the shared baseline.
        if feature_cache_path and not excluded:
            cls._write_task_feature_cache(
                index=index,
                cache_path=feature_cache_path,
                source_signature=expected_source_signature,
                semantic_resource_fingerprint=semantic_resource_fingerprint,
                control=control,
            )
        return index

    @staticmethod
    def prepare_shared_source_cache(
        extraction_dir: Path,
        cache_path: Path,
        control: Optional[RunControl] = None,
        task: str = "all",
    ) -> Dict[str, Any]:
        """Read source CSVs once for a controlled pair.

        Each Matcher still applies its own exclusions before constructing any
        profile or index.  The cache shares only immutable source tables, which
        satisfies exclusion timing while avoiding a second full CSV parse.
        """
        root = extraction_dir.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError("Extraction directory not found: " + str(root))
        table_names = source_tables_for_task(task)
        tables: Dict[str, pd.DataFrame] = {}
        for name in table_names:
            tables[name] = _safe_csv(root / name)
            if control:
                control.emit(
                    "shared_source_table_loaded",
                    source_table=name,
                    row_count=len(tables[name]),
                )
                control.check()
        payload = {
            "version": SHARED_SOURCE_CACHE_VERSION,
            "extraction_dir": str(root),
            "task": task,
            "normalization_version": NORMALIZATION_VERSION,
            "source_table_signature": source_table_signature(root, table_names),
            "table_signatures": {
                name: source_table_signature(root, [name])
                for name in table_names
            },
            "table_names": table_names,
            "tables": tables,
        }
        target = cache_path.expanduser().resolve()
        _atomic_write_pickle(target, payload)
        _write_cache_metadata(
            target,
            cache_kind="shared_source",
            payload_metadata={
                "version": payload["version"],
                "task": payload["task"],
                "table_names": payload["table_names"],
                "table_signatures": payload["table_signatures"],
                "table_count": len(tables),
            },
        )
        return {
            "cache_path": str(target),
            "table_count": len(tables),
            "row_count": sum(len(frame) for frame in tables.values()),
            "task": task,
            "source_table_signature": payload["source_table_signature"],
        }

    @classmethod
    def prepare_task_feature_cache(
        cls,
        extraction_dir: Path,
        task: str,
        cache_path: Path,
        shared_source_cache: Optional[Path] = None,
        control: Optional[RunControl] = None,
        semantic_resource_fingerprint: str = "",
    ) -> Dict[str, Any]:
        """Create a query-independent feature cache for one task profile."""
        root = extraction_dir.expanduser().resolve()
        signature = source_table_signature(root, source_tables_for_task(task))
        if control:
            control.emit(
                "task_feature_cache_build_started",
                task=task,
                source_table_signature=signature,
            )
        index = cls.build(
            extraction_dir=root,
            excluded_doc_ids=(),
            control=control,
            task=task,
            shared_source_cache=shared_source_cache,
            task_feature_cache=None,
            semantic_resource_fingerprint=semantic_resource_fingerprint,
        )

        return cls._write_task_feature_cache(
            index=index,
            cache_path=cache_path,
            source_signature=signature,
            semantic_resource_fingerprint=semantic_resource_fingerprint,
            control=control,
        )

    @classmethod
    def _write_task_feature_cache(
        cls,
        *,
        index: "CorpusIndex",
        cache_path: Path,
        source_signature: str,
        semantic_resource_fingerprint: str,
        control: Optional[RunControl] = None,
    ) -> Dict[str, Any]:
        """Persist one query-independent, unexcluded task index."""
        if index.excluded_doc_ids:
            raise ValueError(
                "A task feature cache cannot be written from an excluded index."
            )

        def proposition_payload(record: PropositionRecord) -> Dict[str, Any]:
            return {
                "doc_id": record.doc_id,
                "source_table": record.source_table,
                "row_id": record.row_id,
                "text": record.text,
                "signature": dict(record.signature),
                "report_time": record.report_time,
            }

        payload: Dict[str, Any] = {
            "version": TASK_FEATURE_CACHE_VERSION,
            "feature_implementation_fingerprint":
                feature_implementation_fingerprint(),
            "extraction_dir": str(index.extraction_dir),
            "task": index.requested_task,
            "normalization_version": NORMALIZATION_VERSION,
            "source_table_signature": source_signature,
            "semantic_resource_fingerprint": semantic_resource_fingerprint,
            "document_text": dict(index.document_text),
            "document_tokens": {
                key: sorted(value)
                for key, value in index.document_tokens.items()
            },
            "document_dates": dict(index.document_dates),
            "propositions_by_doc": {
                key: [proposition_payload(record) for record in value]
                for key, value in index.propositions_by_doc.items()
            },
            "values_by_doc": {
                key: [record.__dict__ for record in value]
                for key, value in index.values_by_doc.items()
            },
            "sentences_by_doc": {
                key: [record.__dict__ for record in value]
                for key, value in index.sentences_by_doc.items()
            },
            "provenance_by_doc": {
                key: [record.__dict__ for record in value]
                for key, value in index.provenance_by_doc.items()
            },
            "document_count": len(index.document_text),
        }
        target = cache_path.expanduser().resolve()
        _atomic_write_pickle(target, payload)
        _write_cache_metadata(
            target,
            cache_kind="task_feature",
            payload_metadata={
                "version": payload["version"],
                "task": payload["task"],
                "normalization_version": payload["normalization_version"],
                "feature_implementation_fingerprint": payload[
                    "feature_implementation_fingerprint"
                ],
                "source_table_signature": payload["source_table_signature"],
                "semantic_resource_fingerprint": payload[
                    "semantic_resource_fingerprint"
                ],
                "document_count": payload["document_count"],
            },
        )
        if control:
            control.emit(
                "task_feature_cache_ready",
                task=index.requested_task,
                cache_path=str(target),
                document_count=payload["document_count"],
            )
        return {
            "cache_path": str(target),
            "task": index.requested_task,
            "document_count": payload["document_count"],
            "source_table_signature": source_signature,
            "semantic_resource_fingerprint": semantic_resource_fingerprint,
            "normalization_version": NORMALIZATION_VERSION,
        }

    @classmethod
    def _from_task_feature_cache(
        cls,
        cache_path: Path,
        extraction_dir: Path,
        expected_source_signature: str,
        task: str,
        excluded: frozenset[str],
        control: Optional[RunControl],
        expected_semantic_resource_fingerprint: str = "",
    ) -> "CorpusIndex":
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get(
            "version"
        ) != TASK_FEATURE_CACHE_VERSION:
            raise ValueError("Unsupported task feature cache contract.")
        # The recorded extraction path is informational. A moved package may
        # reuse this cache only when every content/code/resource check passes.
        if payload.get("task") != task:
            raise ValueError("Task feature cache belongs to a different task.")
        if payload.get("normalization_version") != NORMALIZATION_VERSION:
            raise ValueError("Task feature cache normalization version is stale.")
        if payload.get(
            "feature_implementation_fingerprint", ""
        ) != feature_implementation_fingerprint():
            raise ValueError(
                "Task feature cache is stale because feature-building code "
                "changed."
            )
        if payload.get("source_table_signature") != expected_source_signature:
            raise ValueError(
                "Task feature cache is stale because extraction tables changed."
            )
        if payload.get(
            "semantic_resource_fingerprint", ""
        ) != expected_semantic_resource_fingerprint:
            raise ValueError(
                "Task feature cache is stale because semantic resources changed."
            )

        def allowed(key: str) -> bool:
            return clean_text(key).upper() not in excluded

        document_text = {
            key: value for key, value in payload["document_text"].items()
            if allowed(key)
        }
        document_tokens = {
            key: frozenset(value)
            for key, value in payload["document_tokens"].items()
            if allowed(key)
        }
        document_dates = {
            key: value for key, value in payload["document_dates"].items()
            if allowed(key)
        }
        propositions = {
            key: tuple(
                PropositionRecord(
                    doc_id=record["doc_id"],
                    source_table=record["source_table"],
                    row_id=record["row_id"],
                    text=record["text"],
                    signature=MappingProxyType(dict(record["signature"])),
                    report_time=record["report_time"],
                )
                for record in values
            )
            for key, values in payload["propositions_by_doc"].items()
            if allowed(key)
        }
        indexed_values = {
            key: tuple(ValueRecord(**record) for record in values)
            for key, values in payload["values_by_doc"].items()
            if allowed(key)
        }
        sentences = {
            key: tuple(SentenceRecord(**record) for record in values)
            for key, values in payload["sentences_by_doc"].items()
            if allowed(key)
        }
        provenance = {
            key: tuple(ProvenanceRecord(**record) for record in values)
            for key, values in payload["provenance_by_doc"].items()
            if allowed(key)
        }
        rows_before = int(payload.get("document_count", len(document_text)))
        rows_after = len(document_text)
        statistics = {
            "task_feature_cache": {
                "read_count": 0,
                "shared_cache_read_count": 0,
                "source_state": "TASK_FEATURE_CACHE",
                "rows_before_exclusion": rows_before,
                "rows_removed": rows_before - rows_after,
                "rows_after_exclusion": rows_after,
            }
        }
        fingerprint = stable_hash({
            "version": TASK_FEATURE_CACHE_VERSION,
            "task": task,
            "source_table_signature": expected_source_signature,
            "semantic_resource_fingerprint":
                expected_semantic_resource_fingerprint,
            "normalization_version": NORMALIZATION_VERSION,
            "feature_implementation_fingerprint":
                feature_implementation_fingerprint(),
            "excluded_doc_ids": sorted(excluded),
            "document_count": rows_after,
        })
        if control:
            control.emit(
                "task_feature_cache_hit",
                task=task,
                cache_path=str(cache_path),
                rows_before=rows_before,
                rows_removed=rows_before - rows_after,
                rows_after=rows_after,
                corpus_fingerprint=fingerprint,
            )
            control.check()
        return cls(
            extraction_dir=extraction_dir,
            excluded_doc_ids=excluded,
            requested_task=task,
            document_text=MappingProxyType(document_text),
            document_tokens=MappingProxyType(document_tokens),
            document_dates=MappingProxyType(document_dates),
            propositions_by_doc=MappingProxyType(propositions),
            values_by_doc=MappingProxyType(indexed_values),
            sentences_by_doc=MappingProxyType(sentences),
            provenance_by_doc=MappingProxyType(provenance),
            fingerprint=fingerprint,
            load_statistics=MappingProxyType(statistics),
        )

    @staticmethod
    def _index_claims(
        frame: pd.DataFrame,
        output: Dict[str, List[PropositionRecord]],
    ) -> None:
        if "doc_id" not in frame.columns:
            return
        column_index = {
            column: index for index, column in enumerate(frame.columns)
        }
        inference_cache: Dict[str, Dict[str, str]] = {}
        seen_texts: Dict[str, Set[str]] = defaultdict(set)
        for index, row in enumerate(
            frame.itertuples(index=False, name=None)
        ):
            doc_id = clean_text(
                _tuple_value(row, column_index, "doc_id")
            ).upper()
            text = clean_text(
                _tuple_value(row, column_index, "claim_text")
            )
            if not doc_id or not text:
                continue
            text_key = text.casefold()
            if text_key in seen_texts[doc_id]:
                continue
            seen_texts[doc_id].add(text_key)
            inferred = inference_cache.get(text)
            if inferred is None:
                inferred = infer_simple_clause_arguments(text)
                inference_cache[text] = inferred
            signature = build_event_signature(
                text,
                actor=inferred["actor"],
                predicate=inferred["predicate"],
                obj=inferred["object"],
                location=inferred["location"],
                report_time=clean_text(
                    _tuple_value(row, column_index, "date")
                ),
                source_attribution=clean_text(
                    _tuple_value(row, column_index, "speaker")
                ),
                stance=clean_text(
                    _tuple_value(row, column_index, "stance")
                ),
                modality=clean_text(
                    _tuple_value(row, column_index, "modality")
                ),
                inferred_arguments=inferred,
            )
            output[doc_id].append(PropositionRecord(
                doc_id, "claims.csv",
                clean_text(_tuple_value(row, column_index, "local_id") or index),
                text,
                MappingProxyType(signature),
                clean_text(_tuple_value(row, column_index, "date")),
            ))

    @staticmethod
    def _index_task_pseudo_propositions(
        frame: pd.DataFrame,
        source_table: str,
        document_dates: Mapping[str, str],
        output: Dict[str, List[PropositionRecord]],
    ) -> None:
        """Index explicit task-view propositions with stable provenance.

        Builders normally emit ``proposition_text``.  Older compatible
        builders may expose only a pseudo-view field; in that case sentence
        splitting prevents a document summary from becoming one over-broad
        proposition.  No document identifiers, expected answers, or
        corpus-specific terms are used.
        """
        doc_col = _doc_column(frame)
        if not doc_col:
            return
        explicit_text_column = next(
            (
                name for name in (
                    "proposition_text", "component_text",
                    "evidence_text", "claim_text",
                    "proposition", "reconstruction_text",
                )
                if name in frame.columns
            ),
            "",
        )
        fallback_text_column = next(
            (
                name for name in (
                    "pseudo_view_text", "pseudo_chunk_text", "text",
                )
                if name in frame.columns
            ),
            "",
        )
        if not explicit_text_column and not fallback_text_column:
            return
        row_id_column = next(
            (
                name for name in (
                    "pseudo_view_id", "pseudo_chunk_id", "component_id",
                    "source_row_id", "local_id",
                )
                if name in frame.columns
            ),
            "",
        )
        inference_cache: Dict[str, Dict[str, str]] = {}
        seen_texts: Dict[str, Set[str]] = defaultdict(set)
        column_index = {
            column: index for index, column in enumerate(frame.columns)
        }
        for index, row in enumerate(
            frame.itertuples(index=False, name=None)
        ):
            doc_id = clean_text(
                _tuple_value(row, column_index, doc_col)
            ).upper()
            if not doc_id:
                continue
            raw_text = clean_text(
                _tuple_value(row, column_index, explicit_text_column)
                if explicit_text_column else ""
            )
            used_fallback = False
            if not raw_text:
                raw_text = clean_text(
                    _tuple_value(row, column_index, fallback_text_column)
                    if fallback_text_column else ""
                )
                used_fallback = True
            if not raw_text:
                continue
            reconstruction_claims = (
                _json_string_list(
                    _tuple_value(
                        row,
                        column_index,
                        "reconstruction_claims_json",
                    )
                )
                if source_table == "kg_pseudo_chunks_reconstruction.csv"
                else []
            )
            if reconstruction_claims:
                texts = reconstruction_claims
                used_fallback = False
            else:
                texts = _sentences(raw_text) if used_fallback else [raw_text]
            report_time = (
                clean_text(_tuple_value(row, column_index, "date"))
                or clean_text(document_dates.get(doc_id, ""))
            )
            base_row_id = clean_text(
                _tuple_value(row, column_index, row_id_column)
                if row_id_column else index
            )
            for sentence_index, text in enumerate(texts):
                text = clean_text(text)
                if not text:
                    continue
                text_key = text.casefold()
                if text_key in seen_texts[doc_id]:
                    continue
                seen_texts[doc_id].add(text_key)
                canonical_subject = clean_text(
                    _tuple_value(row, column_index, "canonical_subject")
                )
                row_speaker = clean_text(
                    _tuple_value(row, column_index, "speaker")
                )
                canonical_predicate = clean_text(
                    _tuple_value(row, column_index, "canonical_predicate")
                )
                normalized_relation = clean_text(
                    _tuple_value(row, column_index, "normalized_relation")
                )
                predicate_or_relation = clean_text(
                    _tuple_value(row, column_index, "predicate_or_relation")
                )
                canonical_object = clean_text(
                    _tuple_value(row, column_index, "canonical_object")
                )
                object_or_target = clean_text(
                    _tuple_value(row, column_index, "object_or_target")
                )
                row_target = clean_text(
                    _tuple_value(row, column_index, "target")
                )
                canonical_location = clean_text(
                    _tuple_value(row, column_index, "canonical_location")
                )
                candidate_role_or_context = clean_text(
                    _tuple_value(
                        row,
                        column_index,
                        "candidate_role_or_context",
                    )
                )
                requires_inference = not (
                    canonical_subject or row_speaker
                ) or not (
                    canonical_predicate
                    or normalized_relation
                    or predicate_or_relation
                ) or not (
                    canonical_object
                    or object_or_target
                    or row_target
                ) or not (
                    canonical_location
                    or (
                        source_table == "kg_pseudo_chunks_redaction.csv"
                        and candidate_role_or_context
                    )
                )
                inferred = inference_cache.get(text)
                if inferred is None:
                    inferred = (
                        infer_simple_clause_arguments(text)
                        if requires_inference
                        else {
                        "actor": "",
                        "predicate": "",
                        "object": "",
                        "location": "",
                        }
                    )
                    inference_cache[text] = inferred
                actor = clean_text(
                    canonical_subject or row_speaker or inferred["actor"]
                ) or inferred["actor"]
                predicate = clean_text(
                    canonical_predicate
                    or normalized_relation
                    or predicate_or_relation
                    or inferred["predicate"]
                ) or inferred["predicate"]
                obj = clean_text(
                    canonical_object
                    or object_or_target
                    or row_target
                    or inferred["object"]
                ) or inferred["object"]
                location = clean_text(
                    canonical_location
                    or (
                        candidate_role_or_context
                        if source_table == "kg_pseudo_chunks_redaction.csv"
                        else ""
                    )
                    or inferred["location"]
                ) or inferred["location"]
                event_time = clean_text(
                    _tuple_value(
                        row,
                        column_index,
                        "date_or_temporal_context",
                    )
                    or _tuple_value(row, column_index, "event_time")
                    or _tuple_value(row, column_index, "date")
                )
                signature = build_event_signature(
                    text,
                    actor=actor,
                    predicate=predicate,
                    obj=obj,
                    location=location,
                    event_time=event_time,
                    report_time=report_time,
                    stance=clean_text(
                        _tuple_value(row, column_index, "stance")
                    ),
                    modality=clean_text(
                        _tuple_value(row, column_index, "modality")
                    ),
                    inferred_arguments=inferred,
                )
                row_id = (
                    f"{base_row_id}::sentence_{sentence_index}"
                    if used_fallback else base_row_id
                )
                output[doc_id].append(PropositionRecord(
                    doc_id,
                    source_table,
                    row_id,
                    text,
                    MappingProxyType(signature),
                    report_time,
                ))

    @staticmethod
    def _index_events(
        frame: pd.DataFrame,
        output: Dict[str, List[PropositionRecord]],
    ) -> None:
        if "doc_id" not in frame.columns:
            return
        column_index = {
            column: index for index, column in enumerate(frame.columns)
        }
        inference_cache: Dict[str, Dict[str, str]] = {}
        seen_texts: Dict[str, Set[str]] = defaultdict(set)
        for index, row in enumerate(
            frame.itertuples(index=False, name=None)
        ):
            doc_id = clean_text(
                _tuple_value(row, column_index, "doc_id")
            ).upper()
            text = (
                clean_text(
                    _tuple_value(row, column_index, "evidence_text")
                )
                or clean_text(
                    _tuple_value(row, column_index, "event_text")
                )
                or clean_text(
                    _tuple_value(row, column_index, "label")
                )
                or clean_text(
                    _tuple_value(row, column_index, "description")
                )
            )
            if not doc_id or not text:
                continue
            text_key = text.casefold()
            if text_key in seen_texts[doc_id]:
                continue
            seen_texts[doc_id].add(text_key)
            inferred = inference_cache.get(text)
            if inferred is None:
                inferred = infer_simple_clause_arguments(text)
                inference_cache[text] = inferred
            signature = build_event_signature(
                text,
                actor=inferred["actor"],
                predicate=clean_text(
                    _tuple_value(row, column_index, "event_type")
                    or _tuple_value(row, column_index, "type")
                    or inferred["predicate"]
                ),
                obj=inferred["object"],
                location=clean_text(
                    _tuple_value(row, column_index, "location")
                    or inferred["location"]
                ),
                event_time=clean_text(
                    _tuple_value(row, column_index, "date_or_temporal_context")
                ),
                report_time=clean_text(
                    _tuple_value(row, column_index, "date")
                ),
                inferred_arguments=inferred,
            )
            output[doc_id].append(PropositionRecord(
                doc_id, "events.csv",
                clean_text(_tuple_value(row, column_index, "local_id") or index),
                text,
                MappingProxyType(signature),
                clean_text(_tuple_value(row, column_index, "date")),
            ))

    @staticmethod
    def _index_relations(
        frame: pd.DataFrame,
        output: Dict[str, List[PropositionRecord]],
    ) -> None:
        if "doc_id" not in frame.columns:
            return
        column_index = {
            column: index for index, column in enumerate(frame.columns)
        }
        inference_cache: Dict[str, Dict[str, str]] = {}
        seen_texts: Dict[str, Set[str]] = defaultdict(set)
        for index, row in enumerate(
            frame.itertuples(index=False, name=None)
        ):
            doc_id = clean_text(
                _tuple_value(row, column_index, "doc_id")
            ).upper()
            text = clean_text(
                _tuple_value(row, column_index, "evidence")
            ) or clean_text(" ".join(
                clean_text(_tuple_value(row, column_index, column))
                for column in ("source", "relation", "target", "qualifier")
            ))
            if not doc_id or not text:
                continue
            text_key = text.casefold()
            if text_key in seen_texts[doc_id]:
                continue
            seen_texts[doc_id].add(text_key)
            inferred = inference_cache.get(text)
            if inferred is None:
                inferred = infer_simple_clause_arguments(text)
                inference_cache[text] = inferred
            # Some extraction relation labels are intentionally broad.  When
            # the evidence exposes an unambiguous argument-reversing frame,
            # retain the evidence semantics rather than overwriting them with
            # the coarse source/relation/target tuple.
            prefer_inferred_roles = inferred.get(
                "role_normalization_status", ""
            ) in {
                "RECEIVED_DIRECTION_NORMALIZED",
            }
            signature = build_event_signature(
                text,
                actor=(
                    inferred["actor"]
                    if prefer_inferred_roles
                    else clean_text(_tuple_value(row, column_index, "source"))
                ),
                predicate=(
                    inferred["predicate"]
                    if prefer_inferred_roles
                    else clean_text(_tuple_value(row, column_index, "relation"))
                ),
                obj=(
                    inferred["object"]
                    if prefer_inferred_roles
                    else clean_text(_tuple_value(row, column_index, "target"))
                ),
                event_time=clean_text(
                    _tuple_value(row, column_index, "qualifier")
                ),
                report_time=clean_text(
                    _tuple_value(row, column_index, "date")
                ),
                inferred_arguments=inferred,
            )
            output[doc_id].append(PropositionRecord(
                doc_id, "relations.csv",
                clean_text(_tuple_value(row, column_index, "local_id") or index),
                text,
                MappingProxyType(signature),
                clean_text(_tuple_value(row, column_index, "date")),
            ))

    @staticmethod
    def _index_values(
        tables: Mapping[str, pd.DataFrame],
        output: Dict[str, List[ValueRecord]],
    ) -> None:
        entities = tables["entities.csv"]
        entity_types_by_doc: Dict[Tuple[str, str], str] = {}
        entity_types_global: Dict[str, Dict[str, int]] = defaultdict(dict)

        def _remember_entity_type(doc_id: str, value: str, raw_type: Any) -> None:
            entity_type = clean_text(raw_type).upper()
            if not doc_id or not value or not entity_type or entity_type == "UNKNOWN":
                return
            key = canonical_key(value)
            if not key:
                return
            entity_types_by_doc[(doc_id, key)] = entity_type
            bucket = entity_types_global.setdefault(key, {})
            bucket[entity_type] = int(bucket.get(entity_type, 0)) + 1

        def _resolve_entity_type(doc_id: str, value: str) -> str:
            key = canonical_key(value)
            if not key:
                return "UNKNOWN"
            local = entity_types_by_doc.get((doc_id, key), "")
            if local:
                return local
            counts = entity_types_global.get(key, {})
            if not counts:
                return "UNKNOWN"
            ranked = sorted(
                counts.items(),
                key=lambda item: (-int(item[1]), item[0]),
            )
            return clean_text(ranked[0][0]).upper() or "UNKNOWN"

        if "doc_id" in entities.columns:
            columns = {
                column: index for index, column in enumerate(entities.columns)
            }
            value_column = next(
                (name for name in ("label", "entity", "name") if name in columns),
                "",
            )
            type_column = next(
                (name for name in ("type", "entity_type") if name in columns),
                "",
            )
            for index, row in enumerate(
                entities.itertuples(index=False, name=None)
            ):
                value = clean_text(
                    _tuple_value(row, columns, value_column)
                    if value_column else ""
                )
                doc_id = clean_text(
                    _tuple_value(row, columns, "doc_id")
                ).upper()
                entity_type = clean_text(
                    _tuple_value(row, columns, type_column)
                    if type_column else "UNKNOWN"
                ).upper()
                if value and doc_id:
                    _remember_entity_type(doc_id, value, entity_type)
                if value and doc_id:
                    output[doc_id].append(ValueRecord(
                        doc_id, "entities.csv",
                        clean_text(
                            _tuple_value(row, columns, "local_id") or index
                        ), value,
                        entity_type,
                        clean_text(
                            _tuple_value(row, columns, "evidence")
                            or _tuple_value(row, columns, "context")
                        ),
                    ))
        claims = tables["claims.csv"]
        if "doc_id" in claims.columns:
            columns = {
                column: index for index, column in enumerate(claims.columns)
            }
            for index, row in enumerate(
                claims.itertuples(index=False, name=None)
            ):
                value = clean_text(_tuple_value(row, columns, "speaker"))
                doc_id = clean_text(
                    _tuple_value(row, columns, "doc_id")
                ).upper()
                if value and doc_id:
                    output[doc_id].append(ValueRecord(
                        doc_id, "claims.csv",
                        clean_text(
                            _tuple_value(row, columns, "local_id") or index
                        ), value, "PERSON",
                        clean_text(
                            _tuple_value(row, columns, "claim_text")
                        ),
                    ))
        relations = tables["relations.csv"]
        if "doc_id" in relations.columns:
            columns = {
                column: index for index, column in enumerate(relations.columns)
            }
            for index, row in enumerate(
                relations.itertuples(index=False, name=None)
            ):
                doc_id = clean_text(
                    _tuple_value(row, columns, "doc_id")
                ).upper()
                evidence = clean_text(
                    _tuple_value(row, columns, "evidence")
                )
                relation = clean_text(
                    _tuple_value(row, columns, "relation")
                )
                for side in ("source", "target"):
                    value = clean_text(_tuple_value(row, columns, side))
                    if value and doc_id:
                        resolved_type = _resolve_entity_type(doc_id, value)
                        output[doc_id].append(ValueRecord(
                            doc_id, "relations.csv",
                            clean_text(
                                _tuple_value(row, columns, "local_id") or index
                            ),
                            value,
                            resolved_type,
                            evidence, relation, side,
                        ))
        pseudo_redactions = tables.get("kg_pseudo_chunks_redaction.csv", pd.DataFrame())
        if len(pseudo_redactions) and "doc_id" in pseudo_redactions.columns:
            columns = {
                column: index
                for index, column in enumerate(pseudo_redactions.columns)
            }
            for index, row in enumerate(
                pseudo_redactions.itertuples(index=False, name=None)
            ):
                doc_id = clean_text(
                    _tuple_value(row, columns, "doc_id")
                ).upper()
                value = clean_text(
                    _tuple_value(row, columns, "candidate_label")
                    or _tuple_value(row, columns, "candidate_text")
                )
                if value and doc_id:
                    output[doc_id].append(ValueRecord(
                        doc_id,
                        "kg_pseudo_chunks_redaction.csv",
                        clean_text(
                            _tuple_value(row, columns, "pseudo_view_id")
                            or _tuple_value(row, columns, "source_row_id")
                            or index
                        ),
                        value,
                        clean_text(
                            _tuple_value(
                                row, columns, "candidate_type_normalized"
                            )
                            or _tuple_value(row, columns, "candidate_type")
                            or "UNKNOWN"
                        ).upper(),
                        clean_text(
                            _tuple_value(row, columns, "pseudo_view_text")
                            or _tuple_value(
                                row, columns, "candidate_role_context"
                            )
                        ),
                    ))
        for doc_id, records in output.items():
            unique: Dict[Tuple[str, str, str], ValueRecord] = {}
            for record in records:
                key = (
                    record.value.lower(),
                    record.value_type,
                    record.evidence.lower(),
                )
                unique.setdefault(key, record)
            output[doc_id] = list(unique.values())

    @staticmethod
    def _index_sentences(
        frame: pd.DataFrame,
        output: Dict[str, List[SentenceRecord]],
    ) -> None:
        doc_col = _doc_column(frame)
        text_columns = [
            column for column in TEXT_COLUMN_PRIORITY
            if column in frame.columns
        ]
        if not doc_col or not text_columns:
            return
        chunk_id_col = next(
            (
                column for column in (
                    "chunk_id", "paragraph_id", "source_row_id", "local_id"
                )
                if column in frame.columns
            ),
            "",
        )
        for row_index, row in frame.iterrows():
            doc_id = clean_text(row.get(doc_col, "")).upper()
            text, source_text_column = _row_text(row, text_columns)
            parts = _sentences(text)
            chunk_id = clean_text(row.get(chunk_id_col, row_index)) if chunk_id_col else str(row_index)
            for index, sentence in enumerate(parts):
                output[doc_id].append(SentenceRecord(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    sentence_index=index,
                    left_neighbor=parts[index - 1] if index else "",
                    sentence=sentence,
                    right_neighbor=parts[index + 1] if index + 1 < len(parts) else "",
                    source_text_column=source_text_column,
                ))

    @staticmethod
    def _index_provenance(
        frame: pd.DataFrame,
        output: Dict[str, List[ProvenanceRecord]],
        tables: Optional[Mapping[str, pd.DataFrame]] = None,
    ) -> None:
        if "doc_id" not in frame.columns:
            return
        for index, row in frame.iterrows():
            doc_id = clean_text(row.get("doc_id", "")).upper()
            if not doc_id:
                continue
            source_table = clean_text(row.get("source_table", ""))
            # Component builders have historically emitted both ``claims``
            # and ``claims.csv``.  Normalize at the provenance boundary so a
            # present source row cannot silently produce an empty card.
            source_table_key = (
                source_table
                if source_table.lower().endswith(".csv")
                else source_table + ".csv"
            )
            source_row_id = clean_text(row.get("source_row_id", ""))
            component_text = clean_text(row.get("component_text", ""))
            if not component_text and tables and source_table_key in tables:
                source = tables[source_table_key]
                matched = pd.DataFrame()
                for id_column in (
                    "local_id", "source_row_id", "row_id", "component_id",
                ):
                    if id_column in source.columns and source_row_id:
                        matched = source[
                            source[id_column].fillna("").astype(str).eq(source_row_id)
                        ]
                        if len(matched):
                            break
                if matched.empty and source_row_id.isdigit():
                    source_index = int(source_row_id)
                    if 0 <= source_index < len(source):
                        matched = source.iloc[[source_index]]
                if len(matched):
                    source_row = matched.iloc[0]
                    component_text = clean_text(" ".join(
                        clean_text(source_row.get(column, ""))
                        for column in (
                            "component_text", "pseudo_view_text",
                            "pseudo_chunk_text", "claim_text", "event_text",
                            "evidence_text", "evidence", "source", "relation",
                            "target", "qualifier", "body", "chunk_text", "text",
                        )
                        if column in source.columns
                        and clean_text(source_row.get(column, ""))
                    ))
            output[doc_id].append(ProvenanceRecord(
                doc_id=doc_id,
                component_id=clean_text(row.get("component_id", index)),
                source_table=source_table_key,
                source_row_id=source_row_id,
                component_text=component_text,
            ))

    def audit_frame(self) -> pd.DataFrame:
        rows = []
        for table, stats in self.load_statistics.items():
            rows.append({
                "source_table": table,
                "requested_excluded_doc_ids": ";".join(sorted(self.excluded_doc_ids)),
                "rows_before": stats["rows_before_exclusion"],
                "rows_removed": stats["rows_removed"],
                "rows_after": stats["rows_after_exclusion"],
                "source_read_count": stats["read_count"],
            })
        return pd.DataFrame(rows)
