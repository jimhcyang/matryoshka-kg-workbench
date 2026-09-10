"""
Stage A1: build mention-level evidence tables and label-level catalogs.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ._common import (
    BASE_ENTITY_INVENTORY_FILENAME,
    MENTION_PROFILE_FILENAME,
    profile_output_dir,
    first_existing_col,
    infer_structural_family,
    iter_progress,
    label_ambiguity_score,
    load_entities,
    looks_like_acronym,
    mean_pairwise_jaccard,
    norm_space,
    normalize_label_basic,
    raw_extraction_dir,
    safe_str,
    token_set,
    type_set_from_text,
    _sample_text_list,
)


def _sentence_split(text: object) -> List[str]:
    value = norm_space(text)
    if not value:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])|\n{2,}", value)
    out = [norm_space(part) for part in parts if norm_space(part)]
    return out if out else [value]


def _local_context_window(paragraph_text: object, raw_label: str, fallback_text: object) -> str:
    paragraph = norm_space(paragraph_text)
    fallback = norm_space(fallback_text)
    if not paragraph:
        return fallback

    sentences = _sentence_split(paragraph)
    if len(sentences) <= 2:
        return paragraph

    raw_label_norm = normalize_label_basic(raw_label)
    if not raw_label_norm:
        return fallback or paragraph

    hit_indices: List[int] = []
    for idx, sentence in enumerate(sentences):
        sentence_norm = normalize_label_basic(sentence)
        if raw_label_norm and raw_label_norm in sentence_norm:
            hit_indices.append(idx)

    if not hit_indices:
        return fallback or paragraph

    left = max(0, min(hit_indices) - 1)
    right = min(len(sentences), max(hit_indices) + 2)
    return " ".join(sentences[left:right]).strip()


def build_entity_inventory(label_catalog: pd.DataFrame) -> pd.DataFrame:
    if len(label_catalog) == 0:
        return pd.DataFrame(
            columns=[
                "normalized_label",
                "canonical_seed_label",
                "mention_count",
                "doc_count",
                "entity_type",
                "surface_forms",
                "sample_context",
                "is_generic",
                "acronym",
            ]
        )

    inventory = label_catalog.copy()
    inventory["is_generic"] = inventory.apply(
        lambda row: infer_structural_family(
            safe_str(row.get("normalized_label", "")),
            type_set_from_text(row.get("source_types", "")),
        )
        in {"government", "location", "organization", "role"},
        axis=1,
    )
    inventory["acronym"] = inventory["normalized_label"].map(looks_like_acronym)
    inventory = inventory.rename(
        columns={
            "canonical_label_current": "canonical_seed_label",
            "unique_docs": "doc_count",
            "primary_type": "entity_type",
            "example_mentions": "surface_forms",
            "context_text": "sample_context",
        }
    )
    keep_cols = [
        "normalized_label",
        "canonical_seed_label",
        "mention_count",
        "doc_count",
        "entity_type",
        "surface_forms",
        "sample_context",
        "is_generic",
        "acronym",
    ]
    return inventory.loc[:, keep_cols].reset_index(drop=True)


def load_stage_a_tables(project_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    extraction_dir = raw_extraction_dir(project_dir)
    entities = load_entities(project_dir)

    def maybe_read_csv(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
        return pd.DataFrame()

    events = maybe_read_csv(extraction_dir / "events.csv")
    claims = maybe_read_csv(extraction_dir / "claims.csv")
    chunks = maybe_read_csv(extraction_dir / "chunks.csv")
    input_docs = maybe_read_csv(project_dir / "input_documents.csv")
    return entities, events, claims, chunks, input_docs


def build_mention_catalog(
    entities: pd.DataFrame,
    events: pd.DataFrame,
    claims: pd.DataFrame,
    chunks: pd.DataFrame,
    input_docs: pd.DataFrame,
) -> pd.DataFrame:
    label_col = first_existing_col(entities, ["label", "name", "entity", "text"])
    type_col = first_existing_col(entities, ["type", "entity_type", "broad_type"])
    desc_col = first_existing_col(entities, ["description", "summary", "evidence"])
    if label_col is None:
        raise KeyError(f"Could not find entity label column in entities. Columns: {entities.columns.tolist()}")

    chunk_lookup_cols = [
        col
        for col in [
            "doc_id",
            "paragraph_id",
            "chunk_id",
            "heading",
            "subject",
            "source_path",
            "chunk_text",
            "paragraph_text",
            "source_variant",
            "redaction_count",
        ]
        if col in chunks.columns
    ]
    chunk_lookup: Dict[Tuple[str, str, str], dict] = {}
    if len(chunks):
        for _, row in chunks.loc[:, chunk_lookup_cols].drop_duplicates(subset=["doc_id", "paragraph_id", "chunk_id"]).iterrows():
            key = (
                safe_str(row.get("doc_id", "")),
                safe_str(row.get("paragraph_id", "")),
                safe_str(row.get("chunk_id", "")),
            )
            chunk_lookup[key] = row.to_dict()

    input_subject_map: Dict[str, str] = {}
    input_date_map: Dict[str, str] = {}
    input_path_map: Dict[str, str] = {}
    if len(input_docs) and "doc_id" in input_docs.columns:
        subject_col = first_existing_col(input_docs, ["subject", "document_title", "title"])
        date_col = first_existing_col(input_docs, ["date", "date_display"])
        path_col = first_existing_col(input_docs, ["transcription_path", "source_path", "body_path"])
        if subject_col is not None:
            input_subject_map = dict(zip(input_docs["doc_id"].astype(str), input_docs[subject_col].map(norm_space)))
        if date_col is not None:
            input_date_map = dict(zip(input_docs["doc_id"].astype(str), input_docs[date_col].map(norm_space)))
        if path_col is not None:
            input_path_map = dict(zip(input_docs["doc_id"].astype(str), input_docs[path_col].map(norm_space)))

    chunk_key_cols = ["doc_id", "paragraph_id", "chunk_id"]
    event_label_col = first_existing_col(events, ["label", "event_label", "event_text_for_cluster"])
    event_desc_col = first_existing_col(events, ["description", "summary"])
    claim_text_col = first_existing_col(claims, ["claim_text", "label", "text"])

    events_by_chunk: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    if len(events) and event_label_col is not None:
        for _, row in iter_progress(events.iterrows(), total=len(events), desc="A1 event context"):
            key = tuple(safe_str(row.get(col, "")) for col in chunk_key_cols)
            label = norm_space(row.get(event_label_col, ""))
            desc = norm_space(row.get(event_desc_col, "")) if event_desc_col else ""
            text = " - ".join(part for part in [label, desc] if part)
            if text:
                events_by_chunk[key].append(text)

    claims_by_chunk: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    if len(claims) and claim_text_col is not None:
        for _, row in iter_progress(claims.iterrows(), total=len(claims), desc="A1 claim context"):
            key = tuple(safe_str(row.get(col, "")) for col in chunk_key_cols)
            text = norm_space(row.get(claim_text_col, ""))
            if text:
                claims_by_chunk[key].append(text)

    entity_labels_by_chunk: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for _, row in iter_progress(entities.iterrows(), total=len(entities), desc="A1 entity context"):
        key = tuple(safe_str(row.get(col, "")) for col in chunk_key_cols)
        raw_label = norm_space(row.get(label_col, ""))
        if raw_label:
            entity_labels_by_chunk[key].append(raw_label)

    rows: List[Dict[str, object]] = []
    for idx, row in iter_progress(entities.iterrows(), total=len(entities), desc="A1 mention catalog"):
        raw_label = norm_space(row.get(label_col, ""))
        normalized_label = normalize_label_basic(raw_label)
        if not normalized_label:
            continue

        doc_id = safe_str(row.get("doc_id", ""))
        paragraph_id = safe_str(row.get("paragraph_id", ""))
        chunk_id = safe_str(row.get("chunk_id", ""))
        mention_key = safe_str(row.get("mention_key", ""))
        source_type = norm_space(row.get(type_col, "")) if type_col else ""
        description = norm_space(row.get(desc_col, "")) if desc_col else ""
        chunk_key = (doc_id, paragraph_id, chunk_id)
        chunk_meta = chunk_lookup.get(chunk_key, {})

        sibling_entities = [
            value
            for value in _sample_text_list(entity_labels_by_chunk.get(chunk_key, []), limit=12)
            if normalize_label_basic(value) != normalized_label
        ]
        sibling_events = _sample_text_list(events_by_chunk.get(chunk_key, []), limit=6)
        sibling_claims = _sample_text_list(claims_by_chunk.get(chunk_key, []), limit=4)
        subject = input_subject_map.get(
            doc_id,
            safe_str(row.get("subject", "")) or safe_str(chunk_meta.get("subject", "")),
        )
        date = input_date_map.get(doc_id, safe_str(row.get("date", "")))
        source_path = input_path_map.get(
            doc_id,
            safe_str(row.get("source_path", "")) or safe_str(chunk_meta.get("source_path", "")),
        )
        chunk_text = norm_space(chunk_meta.get("chunk_text", ""))
        paragraph_text = norm_space(chunk_meta.get("paragraph_text", "")) or chunk_text
        heading = safe_str(chunk_meta.get("heading", row.get("heading", "")))
        context_window_text = _local_context_window(
            paragraph_text or chunk_text,
            raw_label,
            chunk_text or paragraph_text,
        )

        rows.append(
            {
                "mention_key": mention_key or f"{doc_id}::{paragraph_id}::{chunk_id}::m{idx}",
                "doc_id": doc_id,
                "paragraph_id": paragraph_id,
                "chunk_id": chunk_id,
                "heading": heading,
                "raw_label": raw_label,
                "normalized_label": normalized_label,
                "canonical_label_current": normalized_label,
                "source_type": source_type,
                "description": description,
                "doc_subject": subject,
                "doc_date": date,
                "source_path": source_path,
                "source_variant": safe_str(chunk_meta.get("source_variant", row.get("source_variant", ""))),
                "redaction_count": int(chunk_meta.get("redaction_count", 0) or 0),
                "chunk_text": chunk_text,
                "paragraph_text": paragraph_text,
                "context_window_text": context_window_text,
                "sibling_entities": "; ".join(sibling_entities),
                "sibling_events": "; ".join(sibling_events),
                "sibling_claims": "; ".join(sibling_claims),
                "context_text": context_window_text or chunk_text or paragraph_text,
                "context_token_count": len(token_set(context_window_text or chunk_text or paragraph_text)),
                "family": infer_structural_family(normalized_label, {normalize_label_basic(source_type)} if source_type else set()),
                "looks_like_acronym": looks_like_acronym(normalized_label),
            }
        )

    return pd.DataFrame(rows)


def build_label_catalog_from_mentions(mention_catalog: pd.DataFrame) -> pd.DataFrame:
    if len(mention_catalog) == 0:
        return pd.DataFrame(
            columns=[
                "normalized_label",
                "canonical_label_current",
                "mention_count",
                "unique_docs",
                "unique_paragraphs",
                "source_types",
                "primary_type",
                "example_mentions",
                "example_descriptions",
                "example_contexts",
                "top_co_mentions",
                "context_text",
                "context_coherence",
                "label_ambiguity",
            ]
        )

    chunk_groups: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for _, row in iter_progress(mention_catalog.iterrows(), total=len(mention_catalog), desc="A1 chunk groups"):
        chunk_groups[
            (
                safe_str(row.get("doc_id", "")),
                safe_str(row.get("paragraph_id", "")),
                safe_str(row.get("chunk_id", "")),
            )
        ].append(safe_str(row.get("normalized_label", "")))

    co_mentions: Dict[str, Counter] = defaultdict(Counter)
    for labels in chunk_groups.values():
        unique_labels = sorted(set(label for label in labels if label))
        for left_idx, left in enumerate(unique_labels):
            for right_idx, right in enumerate(unique_labels):
                if left_idx == right_idx:
                    continue
                co_mentions[left][right] += 1

    grouped_rows: List[Dict[str, object]] = []
    grouped_mentions = mention_catalog.groupby("normalized_label")
    total = mention_catalog["normalized_label"].nunique()
    for normalized_label, group in iter_progress(grouped_mentions, total=total, desc="A1 label catalog"):
        example_mentions = _sample_text_list(group["raw_label"].tolist(), limit=10)
        example_descriptions = _sample_text_list(group["description"].tolist(), limit=8)
        example_contexts = _sample_text_list(group["context_text"].tolist(), limit=3)
        type_counter = Counter(value for value in group["source_type"].astype(str) if value)
        source_types = sorted(type_counter.keys())
        top_co_mentions = [label for label, _count in co_mentions.get(normalized_label, Counter()).most_common(12)]
        context_token_sets = [token_set(text) for text in example_contexts if token_set(text)]
        context_coherence = mean_pairwise_jaccard(context_token_sets[:12])
        ambiguity = label_ambiguity_score(
            normalized_label=normalized_label,
            source_types=source_types,
            context_token_sets=context_token_sets,
            top_co_mentions=top_co_mentions,
        )
        context_bits = [
            normalized_label,
            " ".join(source_types[:8]),
            " ".join(top_co_mentions[:10]),
            " ".join(example_descriptions[:6]),
            " ".join(example_contexts[:2]),
        ]
        grouped_rows.append(
            {
                "normalized_label": normalized_label,
                "canonical_label_current": normalized_label,
                "mention_count": int(len(group)),
                "unique_docs": int(group["doc_id"].astype(str).nunique()),
                "unique_paragraphs": int(group[["doc_id", "paragraph_id"]].astype(str).drop_duplicates().shape[0]),
                "source_types": "; ".join(source_types[:10]),
                "primary_type": type_counter.most_common(1)[0][0] if type_counter else "",
                "example_mentions": "; ".join(example_mentions),
                "example_descriptions": "; ".join(example_descriptions),
                "example_contexts": " || ".join(example_contexts),
                "top_co_mentions": "; ".join(top_co_mentions),
                "context_text": " || ".join(bit for bit in context_bits if bit).strip(),
                "context_coherence": round(context_coherence, 4),
                "label_ambiguity": round(ambiguity, 4),
            }
        )

    return pd.DataFrame(grouped_rows).sort_values(
        ["mention_count", "unique_docs", "normalized_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def run_profile_catalog(
    project_dir: Path,
    entity_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    _ = entity_dir
    project_dir = Path(project_dir).expanduser().resolve()
    out_dir = profile_output_dir(project_dir)

    entities, events, claims, chunks, input_docs = load_stage_a_tables(project_dir)
    mention_catalog = build_mention_catalog(entities, events, claims, chunks, input_docs)
    label_catalog = build_label_catalog_from_mentions(mention_catalog)
    entity_inventory = build_entity_inventory(label_catalog)

    out_dir.mkdir(parents=True, exist_ok=True)
    mention_catalog.to_csv(out_dir / MENTION_PROFILE_FILENAME, index=False)
    label_catalog.to_csv(out_dir / "label_catalog.csv", index=False)
    entity_inventory.to_csv(out_dir / BASE_ENTITY_INVENTORY_FILENAME, index=False)

    print("Profile mention rows:", len(mention_catalog))
    print("Profile label rows:", len(label_catalog))
    return {
        "mention_catalog": mention_catalog,
        "label_catalog": label_catalog,
        "entity_inventory": entity_inventory,
    }
