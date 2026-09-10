#!/usr/bin/env python3
"""
Consolidated Stage A pipeline.

This script intentionally collapses the prepare + extract side of the current
pipeline into one user-facing entrypoint:

1. Read paired postprocessed documents.
2. Build redaction-aware paragraph/segment rows.
3. Write prepared artifacts and variant documents.
4. Run KG extraction over prepared chunk rows.
5. Build one base entity profile per normalized spelling.

It does not do sense splitting or canonical merging. At this stage, the same
normalized spelling is treated as the same base entity profile.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_scripts import _extract_core as extract_stage
from kg_scripts import _prepare_fisc as prep_fisc
from kg_scripts import _prepare_generic as prep_generic
from kg_scripts import _prepare_pairs as prep_base
from kg_scripts import _context_profiles as profile_context
from kg_scripts._common import (
    OpenAI,
    append_jsonl,
    iter_progress,
    load_jsonl,
    normalize_label_basic,
    norm_space,
    safe_str,
    stable_base_entity_key,
)


DEFAULT_GAP_MARKER = "[REDACTED]"
ORIGINAL_PREP_NORMALIZE = prep_base.normalize_source_segments
ORIGINAL_PREP_OMISSIONS = prep_base.find_redacted_only_omissions
ORIGINAL_PREP_REDACTION_SPAN_INDEX = prep_base.build_redaction_span_index
ORIGINAL_LONG_SEGMENT_AUDIT_THRESHOLD = prep_base.LONG_SEGMENT_AUDIT_THRESHOLD


def normalize_dataset_mode_name(dataset_mode: str) -> str:
    value = safe_str(dataset_mode).strip().lower()
    if value in {"generic", ""}:
        return "generic"
    if value == "cib":
        return "cib"
    if value == "fisc":
        return "fisc"
    return value


def detect_dataset_mode(corpus_root: Path) -> str:
    doc_dirs = [path for path in corpus_root.iterdir() if path.is_dir()]
    if any(path.name.startswith("Pair_") for path in doc_dirs):
        return "fisc"
    for doc_dir in doc_dirs[:3]:
        meta_path = doc_dir / "document" / "pair_metadata.json"
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if safe_str(payload.get("source_kind")) == "single_source_text_table":
            return "generic"
        model = safe_str(payload.get("model"))
        if model in {"local_single_source_pair_materializer", "local_canberra_pair_materializer"}:
            return "generic"
    return "cib"


def configure_prepare_namespace(dataset_mode: str):
    dataset_mode = normalize_dataset_mode_name(dataset_mode)
    if dataset_mode == "generic":
        return prep_generic
    prep_base.normalize_source_segments = ORIGINAL_PREP_NORMALIZE
    prep_base.find_redacted_only_omissions = ORIGINAL_PREP_OMISSIONS
    prep_base.build_redaction_span_index = ORIGINAL_PREP_REDACTION_SPAN_INDEX
    prep_base.LONG_SEGMENT_AUDIT_THRESHOLD = ORIGINAL_LONG_SEGMENT_AUDIT_THRESHOLD
    if dataset_mode == "fisc":
        prep_base.normalize_source_segments = prep_fisc.fisc_normalize_source_segments
        prep_base.find_redacted_only_omissions = prep_fisc.fisc_find_redacted_only_omissions
        prep_base.build_redaction_span_index = prep_fisc.fisc_build_redaction_span_index
        prep_base.LONG_SEGMENT_AUDIT_THRESHOLD = 4500
    return prep_base


def build_prepare_outputs(
    *,
    corpus_root: Path,
    project_dir: Path,
    transcription_variant: str,
    gap_marker: str,
    limit: Optional[int],
    dataset_mode: str,
) -> Dict[str, object]:
    prepare_mod = configure_prepare_namespace(dataset_mode)
    prepared_dir = project_dir / "prepared"
    variant_dir = prepared_dir / "variant_documents"

    project_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    source_segments_frames: List[pd.DataFrame] = []
    kept_redaction_frames: List[pd.DataFrame] = []
    omitted_redacted_only_frames: List[pd.DataFrame] = []
    doc_summaries: List[Dict[str, object]] = []

    doc_dirs = [path for path in sorted(corpus_root.iterdir()) if path.is_dir()]
    if limit is not None:
        doc_dirs = doc_dirs[:limit]

    for doc_dir in iter_progress(
        doc_dirs,
        total=len(doc_dirs),
        desc=f"Stage A prepare ({transcription_variant}, {dataset_mode})",
    ):
        built = prepare_mod.build_document_record(
            pair_dir=doc_dir,
            prepared_dir=prepared_dir,
            transcription_variant=transcription_variant,
            gap_marker=gap_marker,
        )
        if built is None:
            continue
        record, source_segments, kept_redactions, omitted_redacted_only, doc_summary, variant_bodies = built
        records.append(record)
        source_segments_frames.append(source_segments)
        kept_redaction_frames.append(kept_redactions)
        omitted_redacted_only_frames.append(omitted_redacted_only)
        doc_summaries.append(doc_summary)

        doc_id = safe_str(record["doc_id"])
        prepare_mod.write_variant_document(
            variant_dir / "unredacted" / f"{doc_id}.txt",
            variant_bodies.get("unredacted", ""),
        )
        prepare_mod.write_variant_document(
            variant_dir / "redacted" / f"{doc_id}.txt",
            variant_bodies.get("redacted", ""),
        )

    input_df = prepare_mod.build_dataframe(records)
    input_csv = project_dir / "input_documents.csv"
    input_df.to_csv(input_csv, index=False)

    source_segments_df = pd.concat(source_segments_frames, ignore_index=True) if source_segments_frames else pd.DataFrame()
    kept_redactions_df = pd.concat(kept_redaction_frames, ignore_index=True) if kept_redaction_frames else pd.DataFrame()
    omitted_redacted_only_df = (
        pd.concat(omitted_redacted_only_frames, ignore_index=True)
        if omitted_redacted_only_frames
        else pd.DataFrame(
            columns=[
                "doc_id",
                "redacted_paragraph_index",
                "page_start",
                "page_end",
                "is_heading",
                "match_threshold",
                "best_match_score",
                "best_match_heading",
                "omission_reason",
                "text",
            ]
        )
    )

    source_segments_df.to_csv(prepared_dir / "source_segments.csv", index=False)
    kept_redactions_df.to_csv(prepared_dir / "redaction_span_index.csv", index=False)
    prepare_mod.build_redaction_review_table(kept_redactions_df).to_csv(
        prepared_dir / "redaction_review.csv",
        index=False,
    )
    omitted_redacted_only_df.to_csv(prepared_dir / "redacted_only_omissions.csv", index=False)
    prepare_mod.build_long_segment_audit(source_segments_df).to_csv(
        prepared_dir / "long_segment_audit.csv",
        index=False,
    )
    pd.DataFrame(doc_summaries).to_csv(prepared_dir / "document_prepare_summary.csv", index=False)

    summary = {
        "corpus_root": str(corpus_root),
        "project_dir": str(project_dir),
        "input_csv": str(input_csv),
        "transcription_variant": transcription_variant,
        "gap_marker": gap_marker,
        "dataset_mode": dataset_mode,
        "documents_written": int(len(input_df)),
        "source_segment_rows": int(len(source_segments_df)),
        "redaction_span_rows": int(len(kept_redactions_df)),
        "redacted_only_omission_rows": int(len(omitted_redacted_only_df)),
        "long_segment_audit_rows": int(
            (
                pd.to_numeric(
                    source_segments_df.get("char_count", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0)
                > prepare_mod.LONG_SEGMENT_AUDIT_THRESHOLD
            ).sum()
        )
        if len(source_segments_df)
        else 0,
        "prepared_dir": str(prepared_dir),
    }
    (project_dir / "input_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _attach_source_id(df: pd.DataFrame, source_id_map: Mapping[str, str]) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    out = df.copy()
    out["source_id"] = out["doc_id"].astype(str).map(source_id_map).fillna(out["doc_id"].astype(str))
    return out


def _relation_endpoint_label_maps(entities: pd.DataFrame) -> Tuple[Dict[Tuple[str, str, str, str], str], Dict[Tuple[str, str, str, str], str]]:
    local_lookup: Dict[Tuple[str, str, str, str], str] = {}
    label_lookup: Dict[Tuple[str, str, str, str], str] = {}
    for _, row in entities.iterrows():
        key_prefix = (
            safe_str(row.get("doc_id", "")),
            safe_str(row.get("paragraph_id", "")),
            safe_str(row.get("chunk_id", "")),
        )
        local_id = safe_str(row.get("local_id", ""))
        normalized = safe_str(row.get("normalized_label", "")) or normalize_label_basic(row.get("label", ""))
        if local_id and normalized:
            local_lookup[key_prefix + (local_id,)] = normalized
        if normalized:
            label_lookup[key_prefix + (normalized,)] = normalized
    return local_lookup, label_lookup


def _resolve_relation_endpoint(
    row: pd.Series,
    field_name: str,
    local_lookup: Mapping[Tuple[str, str, str, str], str],
    label_lookup: Mapping[Tuple[str, str, str, str], str],
) -> str:
    doc_id = safe_str(row.get("doc_id", ""))
    paragraph_id = safe_str(row.get("paragraph_id", ""))
    chunk_id = safe_str(row.get("chunk_id", ""))
    raw_value = safe_str(row.get(field_name, ""))
    if not raw_value:
        return ""
    if (doc_id, paragraph_id, chunk_id, raw_value) in local_lookup:
        return local_lookup[(doc_id, paragraph_id, chunk_id, raw_value)]
    normalized = normalize_label_basic(raw_value)
    if not normalized:
        return ""
    return label_lookup.get((doc_id, paragraph_id, chunk_id, normalized), normalized)


def _merge_chunk_context(frame: pd.DataFrame, chunk_context: pd.DataFrame) -> pd.DataFrame:
    if len(frame) == 0:
        out = frame.copy()
        for required in ["doc_id", "paragraph_id", "chunk_id", "heading", "chunk_text", "paragraph_text"]:
            if required not in out.columns:
                out[required] = pd.Series(dtype=str)
        return out
    out = frame.copy()
    for col in ["chunk_text", "paragraph_text", "heading"]:
        if col in out.columns:
            out.drop(columns=[col], inplace=True)
    for required in ["doc_id", "paragraph_id", "chunk_id"]:
        if required not in out.columns:
            out[required] = ""
    return out.merge(chunk_context, on=["doc_id", "paragraph_id", "chunk_id"], how="left")


def build_base_entity_profiles(
    mention_catalog: pd.DataFrame,
    label_catalog: pd.DataFrame,
    entity_inventory: pd.DataFrame,
    relations: pd.DataFrame,
) -> pd.DataFrame:
    if len(label_catalog) == 0:
        return pd.DataFrame(
            columns=[
                "base_entity_key",
                "normalized_label",
                "mention_count",
                "doc_count",
                "paragraph_count",
                "primary_type",
                "type_distribution",
                "surface_forms",
                "representative_contexts",
                "cooccurring_entities",
                "relation_neighbor_entities",
                "relation_type_distribution",
                "profile_summary",
                "is_generic",
                "looks_like_acronym",
            ]
        )

    label_catalog = label_catalog.copy()
    label_catalog["base_entity_key"] = label_catalog["normalized_label"].map(stable_base_entity_key)

    inventory_lookup = entity_inventory.set_index("normalized_label").to_dict(orient="index") if len(entity_inventory) else {}
    mentions_by_label = {label: group.copy() for label, group in mention_catalog.groupby("normalized_label", dropna=False)}

    relation_neighbors: Dict[str, Counter] = defaultdict(Counter)
    relation_types: Dict[str, Counter] = defaultdict(Counter)
    if len(relations) and len(mention_catalog):
        local_lookup, label_lookup = _relation_endpoint_label_maps(
            mention_catalog.rename(columns={"raw_label": "label", "source_type": "type"})
        )
        for _, row in relations.iterrows():
            source_label = _resolve_relation_endpoint(row, "source", local_lookup, label_lookup)
            target_label = _resolve_relation_endpoint(row, "target", local_lookup, label_lookup)
            relation_type = safe_str(row.get("relation", ""))
            if source_label and target_label and source_label != target_label:
                relation_neighbors[source_label][target_label] += 1
                relation_neighbors[target_label][source_label] += 1
                if relation_type:
                    relation_types[source_label][relation_type] += 1
                    relation_types[target_label][relation_type] += 1

    rows: List[Dict[str, object]] = []
    for _, profile in label_catalog.iterrows():
        normalized_label = safe_str(profile.get("normalized_label", ""))
        inventory_row = inventory_lookup.get(normalized_label, {})
        mention_group = mentions_by_label.get(normalized_label, pd.DataFrame())

        type_counter = Counter(
            safe_str(value)
            for value in mention_group.get("source_type", pd.Series(dtype=str)).astype(str).tolist()
            if safe_str(value)
        )
        type_distribution = "; ".join(
            f"{label}:{count}" for label, count in type_counter.most_common()
        )
        cooccurring_entities = safe_str(profile.get("top_co_mentions", ""))
        relation_neighbor_entities = "; ".join(
            f"{label}:{count}" for label, count in relation_neighbors.get(normalized_label, Counter()).most_common(12)
        )
        relation_type_distribution = "; ".join(
            f"{label}:{count}" for label, count in relation_types.get(normalized_label, Counter()).most_common()
        )
        contexts = safe_str(profile.get("example_contexts", ""))
        surface_forms = safe_str(profile.get("example_mentions", ""))
        primary_type = safe_str(profile.get("primary_type", ""))
        mention_count = int(profile.get("mention_count", 0) or 0)
        doc_count = int(profile.get("unique_docs", 0) or 0)
        paragraph_count = int(profile.get("unique_paragraphs", 0) or 0)

        summary_bits = [
            f"{normalized_label} appears in {mention_count} mentions across {doc_count} documents.",
            f"Primary type: {primary_type}." if primary_type else "",
            f"Common surface forms: {surface_forms}." if surface_forms else "",
            f"Frequent co-occurring entities: {cooccurring_entities}." if cooccurring_entities else "",
            f"Relation neighborhood: {relation_neighbor_entities}." if relation_neighbor_entities else "",
        ]

        rows.append(
            {
                "base_entity_key": safe_str(profile.get("base_entity_key", "")),
                "normalized_label": normalized_label,
                "mention_count": mention_count,
                "doc_count": doc_count,
                "paragraph_count": paragraph_count,
                "primary_type": primary_type,
                "type_distribution": type_distribution,
                "surface_forms": surface_forms,
                "representative_contexts": contexts,
                "cooccurring_entities": cooccurring_entities,
                "relation_neighbor_entities": relation_neighbor_entities,
                "relation_type_distribution": relation_type_distribution,
                "profile_summary": " ".join(bit for bit in summary_bits if bit).strip(),
                "is_generic": bool(inventory_row.get("is_generic", False)),
                "looks_like_acronym": bool(inventory_row.get("acronym", False)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["mention_count", "doc_count", "normalized_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def run_extraction(
    *,
    project_dir: Path,
    transcription_variant: str,
    openai_model: str,
    max_new_chunks: Optional[int],
) -> Dict[str, object]:
    prepared_dir = project_dir / "prepared"
    source_segments_path = prepared_dir / "source_segments.csv"
    if not source_segments_path.exists():
        raise FileNotFoundError(f"Missing prepared source segments: {source_segments_path}")

    segments = pd.read_csv(source_segments_path, low_memory=False)
    chunk_df = extract_stage.chunk_rows_from_segments(segments, variant=transcription_variant)
    if len(chunk_df) == 0:
        raise ValueError("No non-empty chunk rows were generated from source segments.")

    extraction_dir = project_dir / "extraction"
    run_dir = project_dir / "extraction_run"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "chunk_results.jsonl"
    error_path = run_dir / "chunk_errors.jsonl"
    completed = extract_stage.checkpoint_keys(checkpoint_path)
    pending_mask = [extract_stage.chunk_key(row) not in completed for _, row in chunk_df.iterrows()]
    pending_df = chunk_df[pending_mask].copy()
    if max_new_chunks is not None:
        pending_df = pending_df.head(max_new_chunks).copy()

    total_for_progress = len(chunk_df)
    print(
        json.dumps(
            {
                "variant": transcription_variant,
                "total_chunks": int(len(chunk_df)),
                "completed_chunks": int(len(completed)),
                "pending_chunks": int(len(pending_df)),
            },
            indent=2,
        )
    )

    if len(pending_df):
        if OpenAI is None:
            raise ImportError("OpenAI package is not installed. Run: pip install openai")
        api_key = safe_str(__import__("os").environ.get("OPENAI_API_KEY", ""))
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=2)
        iterator = iter_progress(
            pending_df.iterrows(),
            total=total_for_progress,
            initial=min(len(completed), total_for_progress),
            desc="Stage A extraction",
        )
        for _, row in iterator:
            try:
                record = extract_stage.process_chunk(client, row, model=openai_model)
                append_jsonl(checkpoint_path, record)
            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "doc_id": safe_str(row.get("doc_id", "")),
                        "paragraph_id": safe_str(row.get("paragraph_id", "")),
                        "chunk_id": safe_str(row.get("chunk_id", "")),
                        "error_type": type(exc).__name__,
                        "error": safe_str(exc),
                        "chunk_text": safe_str(row.get("chunk_text", "")),
                    },
                )
                print(
                    f"Skipping Stage A chunk {row.get('doc_id', '')} {row.get('paragraph_id', '')} {row.get('chunk_id', '')}: "
                    f"{type(exc).__name__}: {exc}"
                )

    results = load_jsonl(checkpoint_path)
    chunks_df, entities_df, events_df, claims_df, relations_df = extract_stage.flatten_results(results)

    input_docs = pd.read_csv(project_dir / "input_documents.csv", low_memory=False)
    source_id_map = (
        dict(zip(input_docs["doc_id"].astype(str), input_docs["id"].astype(str)))
        if "doc_id" in input_docs.columns and "id" in input_docs.columns
        else {}
    )
    chunks_df = _attach_source_id(chunks_df, source_id_map)
    entities_df = _attach_source_id(entities_df, source_id_map)
    events_df = _attach_source_id(events_df, source_id_map)
    claims_df = _attach_source_id(claims_df, source_id_map)
    relations_df = _attach_source_id(relations_df, source_id_map)

    mention_catalog = profile_context.build_mention_catalog(entities_df, events_df, claims_df, chunks_df, input_docs)
    label_catalog = profile_context.build_label_catalog_from_mentions(mention_catalog)
    entity_inventory = profile_context.build_entity_inventory(label_catalog)
    base_entity_profiles = build_base_entity_profiles(mention_catalog, label_catalog, entity_inventory, relations_df)

    entity_enrichment_cols = [
        "mention_key",
        "raw_label",
        "normalized_label",
        "canonical_label_current",
        "doc_subject",
        "doc_date",
        "context_window_text",
        "context_text",
        "sibling_entities",
        "sibling_events",
        "sibling_claims",
        "family",
        "looks_like_acronym",
    ]
    enriched_entities = entities_df.merge(
        mention_catalog.loc[:, [col for col in entity_enrichment_cols if col in mention_catalog.columns]],
        on="mention_key",
        how="left",
    )
    enriched_entities["normalized_label"] = enriched_entities.get("normalized_label", "").astype(str)
    enriched_entities["base_entity_key"] = enriched_entities["normalized_label"].map(
        lambda value: stable_base_entity_key(value) if safe_str(value) else ""
    )
    enriched_entities["base_entity_label"] = enriched_entities["normalized_label"]
    enriched_entities["context_window_text"] = enriched_entities.get("context_window_text", "").fillna("")
    enriched_entities["context_text"] = enriched_entities.get("context_text", "").fillna("")

    chunk_context_cols = [
        "doc_id",
        "paragraph_id",
        "chunk_id",
        "heading",
        "chunk_text",
        "paragraph_text",
    ]
    chunk_context = chunks_df.loc[:, [col for col in chunk_context_cols if col in chunks_df.columns]].drop_duplicates()
    events_df = _merge_chunk_context(events_df, chunk_context)
    claims_df = _merge_chunk_context(claims_df, chunk_context)
    relations_df = _merge_chunk_context(relations_df, chunk_context)

    chunks_df.to_csv(extraction_dir / "chunks.csv", index=False)
    enriched_entities.to_csv(extraction_dir / "entities.csv", index=False)
    events_df.to_csv(extraction_dir / "events.csv", index=False)
    claims_df.to_csv(extraction_dir / "claims.csv", index=False)
    relations_df.to_csv(extraction_dir / "relations.csv", index=False)
    base_entity_profiles.to_csv(extraction_dir / "base_entity_profiles.csv", index=False)

    summary = {
        "variant": transcription_variant,
        "chunks_csv": str(extraction_dir / "chunks.csv"),
        "entities_rows": int(len(enriched_entities)),
        "events_rows": int(len(events_df)),
        "claims_rows": int(len(claims_df)),
        "relations_rows": int(len(relations_df)),
        "chunk_rows": int(len(chunks_df)),
        "checkpoint_rows": int(len(results)),
        "base_entity_profile_rows": int(len(base_entity_profiles)),
    }
    (extraction_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidated Stage A prepare + extract pipeline.")
    parser.add_argument("--corpus-root", required=True, help="Postprocessed corpus root.")
    parser.add_argument("--project-dir", required=True, help="Output project directory.")
    parser.add_argument(
        "--dataset-mode",
        choices=["auto", "generic", "cib", "fisc"],
        default="auto",
        help="Corpus-specific preparation rules.",
    )
    parser.add_argument(
        "--transcription-variant",
        choices=["unredacted", "redacted"],
        default="unredacted",
        help="Which variant to extract from.",
    )
    parser.add_argument("--gap-marker", default=DEFAULT_GAP_MARKER, help="Marker used for retained redaction gaps.")
    parser.add_argument("--limit", type=int, default=None, help="Optional document limit for smoke runs.")
    parser.add_argument("--max-new-chunks", type=int, default=None, help="Optional cap on newly processed chunks.")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_root = Path(args.corpus_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not corpus_root.exists():
        raise FileNotFoundError(f"Corpus root not found: {corpus_root}")

    dataset_mode = args.dataset_mode
    if dataset_mode == "auto":
        dataset_mode = detect_dataset_mode(corpus_root)
    else:
        dataset_mode = normalize_dataset_mode_name(dataset_mode)

    prepare_summary = build_prepare_outputs(
        corpus_root=corpus_root,
        project_dir=project_dir,
        transcription_variant=args.transcription_variant,
        gap_marker=args.gap_marker,
        limit=args.limit,
        dataset_mode=dataset_mode,
    )
    extraction_summary = run_extraction(
        project_dir=project_dir,
        transcription_variant=args.transcription_variant,
        openai_model=args.openai_model,
        max_new_chunks=args.max_new_chunks,
    )
    print(
        json.dumps(
            {
                "stage": "A",
                "dataset_mode": dataset_mode,
                "prepare": prepare_summary,
                "extraction": extraction_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
