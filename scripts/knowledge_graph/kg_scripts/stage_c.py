#!/usr/bin/env python3
"""
Stage C: task materialization.

Builds task-facing corpus views from canonicalized extraction outputs:

- tasks/pseudo_chunks.csv
- tasks/pseudo_chunk_components.csv
- tasks/reconstruction_view.csv
- tasks/short_redaction_recovery_view.csv
- tasks/sentence_redaction_recovery_view.csv
- tasks/claim_support_view.csv
- tasks/contradiction_tension_view.csv
- tasks/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_scripts._common import norm_space, safe_str


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
GAP_MARKER_RE = re.compile(r"\[(?:REDACTED|REDACTION)\]")


def choose_extraction_dir(project_dir: Path, extraction_dir_arg: Optional[str]) -> Path:
    if extraction_dir_arg:
        return Path(extraction_dir_arg).expanduser().resolve()
    canonicalized = project_dir / "extraction_canonicalized"
    if (canonicalized / "entities.csv").exists():
        return canonicalized
    return project_dir / "extraction"


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_base_chunks(project_dir: Path, extraction_dir: Path) -> pd.DataFrame:
    preferred = extraction_dir / "chunks.csv"
    if preferred.exists():
        return load_table(preferred)
    return load_table(project_dir / "extraction" / "chunks.csv")


def sentence_split(text: object) -> List[str]:
    value = safe_str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        return []
    parts = [norm_space(part) for part in SENTENCE_SPLIT_RE.split(value) if norm_space(part)]
    if parts:
        return parts
    return [norm_space(value)]


def stable_id(prefix: str, *parts: object) -> str:
    import hashlib

    payload = "||".join(norm_space(part) for part in parts)
    return f"{prefix}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12].upper()}"


def group_components(df: pd.DataFrame, text_col: str, label_col: str) -> Dict[Tuple[str, str], List[dict]]:
    out: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    if len(df) == 0:
        return out
    for _, row in df.iterrows():
        key = (safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))
        out[key].append(
            {
                "row_id": safe_str(row.get("local_id", "")),
                "label": safe_str(row.get(label_col, "")),
                "text": safe_str(row.get(text_col, "")),
                "type": safe_str(row.get("type", "")) or safe_str(row.get("stance", "")),
                "chunk_id": safe_str(row.get("chunk_id", "")),
                "source_table": text_col,
                "source_id": safe_str(row.get("source_id", "")),
                "date": safe_str(row.get("date", "")),
                "subject": safe_str(row.get("subject", "")),
                "base_entity_key": safe_str(row.get("base_entity_key", "")),
                "speaker": safe_str(row.get("speaker", "")),
                "target": safe_str(row.get("target", "")),
                "stance": safe_str(row.get("stance", "")),
            }
        )
    return out


def anchor_tokens(text: object) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", safe_str(text))
    keep: List[str] = []
    for token in tokens:
        if len(token) >= 4 or token.isupper():
            keep.append(token)
    seen = set()
    ordered: List[str] = []
    for token in keep:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(token)
    return ordered


def polarity_label(text: object) -> str:
    value = safe_str(text)
    if not value:
        return "unspecified"
    if re.search(r"\b(no|not|never|deny|denied|denies|without|cannot|can't|won't|would not|did not)\b", value, re.I):
        return "negative"
    return "positive"


def write_matcher_compatibility_outputs(
    *,
    project_dir: Path,
    extraction_dir: Path,
    base_chunks: pd.DataFrame,
    segments: pd.DataFrame,
    pseudo_rows: List[Dict[str, object]],
    component_rows: List[Dict[str, object]],
    reconstruction_rows: List[Dict[str, object]],
    short_redaction_rows: List[Dict[str, object]],
    claim_support_rows: List[Dict[str, object]],
    contradiction_rows: List[Dict[str, object]],
) -> Dict[str, int]:
    extraction_dir.mkdir(parents=True, exist_ok=True)

    if len(base_chunks):
        base_chunks.to_csv(extraction_dir / "chunks.csv", index=False)

    segment_lookup = {
        (safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", ""))): row
        for _, row in segments.iterrows()
    }

    pseudo_chunk_rows: List[Dict[str, object]] = []
    pseudo_component_rows: List[Dict[str, object]] = []
    reconstruction_view_rows: List[Dict[str, object]] = []
    redaction_view_rows: List[Dict[str, object]] = []
    claims_view_rows: List[Dict[str, object]] = []
    contradiction_view_rows: List[Dict[str, object]] = []

    components_by_pc: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in component_rows:
        components_by_pc[safe_str(item.get("pseudo_chunk_id", ""))].append(item)

    for row in pseudo_rows:
        pseudo_chunk_id = safe_str(row.get("pseudo_chunk_id", ""))
        doc_id = safe_str(row.get("doc_id", ""))
        paragraph_id = safe_str(row.get("paragraph_id", ""))
        meta = segment_lookup.get((doc_id, paragraph_id), {})
        entity_labels = [part.strip() for part in safe_str(row.get("entity_labels", "")).split(";") if part.strip()]
        event_labels = [part.strip() for part in safe_str(row.get("event_labels", "")).split(";") if part.strip()]
        claim_texts = [part.strip() for part in safe_str(row.get("claim_texts", "")).split("||") if part.strip()]
        relation_triples_text = [part.strip() for part in safe_str(row.get("relation_triples", "")).split("||") if part.strip()]
        related_components = components_by_pc.get(pseudo_chunk_id, [])
        canonical_keys = [
            safe_str(item.get("base_entity_key", ""))
            for item in related_components
            if safe_str(item.get("component_kind", "")) == "entity" and safe_str(item.get("base_entity_key", ""))
        ]
        semantic_signature_text = "\n".join(
            [
                f"DOC={doc_id}",
                f"SUBJECT={safe_str(meta.get('subject', row.get('heading', '')))}",
                f"ENTITIES={'; '.join(entity_labels)}",
                f"EVENTS={'; '.join(event_labels)}",
                f"CLAIMS={'; '.join(claim_texts[:6])}",
                f"RELATIONS={'; '.join(relation_triples_text[:8])}",
                f"RARE_ANCHORS={'; '.join(anchor_tokens(' '.join(entity_labels + event_labels + claim_texts + relation_triples_text))[:8])}",
            ]
        )
        pseudo_chunk_text = "\n".join(
            [
                f"Document: {doc_id} | Paragraph: {paragraph_id} | Date: {safe_str(meta.get('date', ''))} | Subject: {safe_str(meta.get('subject', row.get('heading', '')))}",
                f"Entities: {'; '.join(entity_labels) if entity_labels else '[none]'}",
                f"Claims: {'; '.join(claim_texts[:6]) if claim_texts else '[none]'}",
                f"Events: {'; '.join(event_labels) if event_labels else '[none]'}",
                f"Relations: {'; '.join(relation_triples_text[:8]) if relation_triples_text else '[none]'}",
            ]
        )
        structured_components = []
        for comp in related_components:
            structured_components.append(
                {
                    "component_kind": safe_str(comp.get("component_kind", "")),
                    "component_id": safe_str(comp.get("component_id", "")),
                    "label": safe_str(comp.get("component_label", "")),
                    "type": safe_str(comp.get("component_type", "")),
                    "text": safe_str(comp.get("component_text", "")),
                    "source_table": safe_str(comp.get("source_table", "")),
                    "source_row_id": safe_str(comp.get("source_row_id", "")),
                    "base_entity_key": safe_str(comp.get("base_entity_key", "")),
                }
            )
        pseudo_chunk_rows.append(
            {
                "pseudo_chunk_id": pseudo_chunk_id,
                "doc_id": doc_id,
                "source_id": safe_str(meta.get("source_id", "")),
                "paragraph_id": paragraph_id,
                "date": safe_str(meta.get("date", "")),
                "subject": safe_str(meta.get("subject", row.get("heading", ""))),
                "source_row_number": int(meta.get("source_row_number", 0) or 0),
                "in_kg_universe": True,
                "pseudo_chunk_text": pseudo_chunk_text,
                "source_tables_used": ";".join(sorted({safe_str(item.get("source_table", "")) for item in related_components if safe_str(item.get("source_table", ""))})),
                "entity_labels": json.dumps(entity_labels, ensure_ascii=False),
                "claim_texts": json.dumps(claim_texts, ensure_ascii=False),
                "event_texts": json.dumps(event_labels, ensure_ascii=False),
                "relation_triples_text": json.dumps(relation_triples_text, ensure_ascii=False),
                "canonical_entity_keys": json.dumps(canonical_keys, ensure_ascii=False),
                "structured_components": json.dumps(structured_components, ensure_ascii=False),
                "num_entities": len(entity_labels),
                "num_claims": len(claim_texts),
                "num_events": len(event_labels),
                "num_relations": len(relation_triples_text),
                "source_type": "kg_pseudo_chunk",
                "confidence_type": "structured_semantic",
                "evidence_basis": "derived_from_structured_kg_rows",
                "exact_text_match": False,
                "raw_chunk_used": False,
                "pseudo_chunk_used": True,
                "is_raw_chunk": False,
                "is_pseudo_chunk": True,
                "not_original_text": True,
                "schema_version": "kg_scripts_pseudo_evidence_v1",
                "generic_anchor_count": max(0, len(anchor_tokens(safe_str(meta.get('subject', '')))) // 2),
                "distinctive_anchor_count": len(anchor_tokens(pseudo_chunk_text)),
                "semantic_signature_text": semantic_signature_text,
            }
        )

        emitted_redaction_keys = set()
        for comp in related_components:
            if safe_str(comp.get("component_kind", "")) != "entity":
                continue
            candidate_label = safe_str(comp.get("component_label", ""))
            evidence_text = (
                safe_str(comp.get("component_text", ""))
                or pseudo_chunk_text
            )
            if not candidate_label or not evidence_text:
                continue
            redaction_key = (
                pseudo_chunk_id,
                safe_str(comp.get("base_entity_key", "")),
                candidate_label,
                evidence_text,
            )
            if redaction_key in emitted_redaction_keys:
                continue
            emitted_redaction_keys.add(redaction_key)
            redaction_view_rows.append(
                {
                    "pseudo_view_id": stable_id(
                        "RVIEW",
                        pseudo_chunk_id,
                        comp.get("component_id", ""),
                        candidate_label,
                    ),
                    "pseudo_view_type": "entity_candidate_context",
                    "pseudo_chunk_id": pseudo_chunk_id,
                    "doc_id": doc_id,
                    "source_id": safe_str(meta.get("source_id", "")),
                    "paragraph_id": paragraph_id,
                    "date": safe_str(meta.get("date", "")),
                    "subject": safe_str(meta.get("subject", row.get("heading", ""))),
                    "candidate_label": candidate_label,
                    "candidate_text": candidate_label,
                    "candidate_type": safe_str(comp.get("component_type", "")) or "UNKNOWN",
                    "candidate_type_normalized": (
                        safe_str(comp.get("component_type", "")) or "UNKNOWN"
                    ).upper(),
                    "candidate_role_or_context": evidence_text,
                    "candidate_role_context": evidence_text,
                    "candidate_source_table": safe_str(comp.get("source_table", "")),
                    "candidate_source_confidence": 0.7,
                    "candidate_doc_id": doc_id,
                    "candidate_base_entity_key": safe_str(comp.get("base_entity_key", "")),
                    "candidate_evidence_json": json.dumps(
                        {
                            "component_text": safe_str(comp.get("component_text", "")),
                            "pseudo_chunk_text": pseudo_chunk_text,
                            "component_id": safe_str(comp.get("component_id", "")),
                        },
                        ensure_ascii=False,
                    ),
                    "pseudo_view_text": evidence_text,
                    "evidence_class": "entity_local_context",
                    "source_type": "structured_entity_context",
                    "confidence_type": "component_level",
                    "exact_text_match": False,
                    "raw_chunk_used": False,
                    "pseudo_chunk_used": True,
                    "do_not_assume_exact_unredacted_copy": False,
                }
            )

    for comp in component_rows:
        component_text = safe_str(comp.get("component_text", "")) or safe_str(
            comp.get("component_label", "")
        )
        pseudo_component_rows.append(
            {
                "pseudo_chunk_id": safe_str(comp.get("pseudo_chunk_id", "")),
                "component_id": safe_str(comp.get("component_id", "")) or stable_id("COMP", comp.get("pseudo_chunk_id", ""), comp.get("component_kind", ""), comp.get("component_label", ""), comp.get("chunk_id", "")),
                "source_table": safe_str(comp.get("source_table", "")),
                "source_row_id": safe_str(comp.get("source_row_id", "")),
                "doc_id": safe_str(comp.get("doc_id", "")),
                "source_id": safe_str(comp.get("source_id", "")),
                "paragraph_id": safe_str(comp.get("paragraph_id", "")),
                "date": safe_str(comp.get("date", "")),
                "subject": safe_str(comp.get("subject", "")),
                "field_name": safe_str(comp.get("component_kind", "")),
                "field_value": safe_str(comp.get("component_label", "")),
                "confidence": 0.7,
                "structured_component_json": json.dumps(
                    {
                        "component_kind": safe_str(comp.get("component_kind", "")),
                        "component_label": safe_str(comp.get("component_label", "")),
                        "component_type": safe_str(comp.get("component_type", "")),
                        "component_text": component_text,
                        "source_table": safe_str(comp.get("source_table", "")),
                        "source_row_id": safe_str(comp.get("source_row_id", "")),
                    },
                    ensure_ascii=False,
                ),
                "evidence_text": component_text,
                "component_text": component_text,
                "entity_labels": safe_str(comp.get("component_label", "")),
            }
        )

    for row in reconstruction_rows:
        doc_id = safe_str(row.get("doc_id", ""))
        paragraph_id = safe_str(row.get("paragraph_id", ""))
        meta = segment_lookup.get((doc_id, paragraph_id), {})
        reconstruction_view_rows.append(
            {
                "pseudo_view_id": safe_str(row.get("pseudo_view_id", "")),
                "pseudo_view_type": "document_reconstruction",
                "pseudo_chunk_id": safe_str(row.get("pseudo_chunk_id", "")),
                "doc_id": doc_id,
                "source_id": safe_str(meta.get("source_id", "")),
                "paragraph_id": paragraph_id,
                "date": safe_str(meta.get("date", "")),
                "subject": safe_str(meta.get("subject", row.get("heading", ""))),
                "source_row_number": int(meta.get("source_row_number", 0) or 0),
                "in_kg_universe": True,
                "pseudo_view_text": f"Reconstruction semantic signature.\n{safe_str(row.get('reconstruction_text', ''))}",
                "reconstruction_signature_text": f"ENTITIES={safe_str(row.get('entity_labels', ''))}\nEVENTS={safe_str(row.get('event_labels', ''))}\nCLAIMS={safe_str(row.get('claim_texts', ''))}",
                "reconstruction_entities_json": json.dumps([part.strip() for part in safe_str(row.get("entity_labels", "")).split(";") if part.strip()], ensure_ascii=False),
                "reconstruction_events_json": json.dumps([part.strip() for part in safe_str(row.get("event_labels", "")).split(";") if part.strip()], ensure_ascii=False),
                "reconstruction_claims_json": json.dumps([part.strip() for part in safe_str(row.get("claim_texts", "")).split("||") if part.strip()], ensure_ascii=False),
                "reconstruction_relations_json": json.dumps([], ensure_ascii=False),
                "generic_anchor_count": 0,
                "distinctive_anchor_count": len(anchor_tokens(safe_str(row.get("reconstruction_text", "")))),
                "text_reconstructibility_score": 1.0 if safe_str(row.get("reconstruction_text", "")) else 0.0,
                "text_reconstructibility_label": "strong" if safe_str(row.get("reconstruction_text", "")) else "none",
                "entity_semantic_reconstructibility_score": 1.0 if safe_str(row.get("entity_labels", "")) else 0.0,
                "entity_semantic_reconstructibility_label": "strong" if safe_str(row.get("entity_labels", "")) else "none",
                "claim_event_reconstructibility_score": 1.0 if safe_str(row.get("claim_texts", "")) or safe_str(row.get("event_labels", "")) else 0.0,
                "claim_event_reconstructibility_label": "strong" if safe_str(row.get("claim_texts", "")) or safe_str(row.get("event_labels", "")) else "none",
                "overall_reconstructibility_score": 1.0 if safe_str(row.get("reconstruction_text", "")) else 0.0,
                "overall_reconstructibility_label": "strong" if safe_str(row.get("reconstruction_text", "")) else "none",
                "reconstructibility_basis": "raw_chunk_and_structured_kg",
                "source_type": "raw_chunk+kg_pseudo_chunk",
                "confidence_type": "direct_text+structured_semantic",
                "exact_text_match": True,
                "raw_chunk_used": True,
                "pseudo_chunk_used": True,
            }
        )

    for row in short_redaction_rows:
        doc_id = safe_str(row.get("doc_id", ""))
        paragraph_id = safe_str(row.get("paragraph_id", ""))
        meta = segment_lookup.get((doc_id, paragraph_id), {})
        redaction_view_rows.append(
            {
                "pseudo_view_id": safe_str(row.get("pseudo_view_id", "")),
                "pseudo_view_type": "redaction_slot",
                "pseudo_chunk_id": stable_id("PC", doc_id, paragraph_id),
                "doc_id": doc_id,
                "source_id": safe_str(meta.get("source_id", "")),
                "paragraph_id": paragraph_id,
                "date": safe_str(meta.get("date", "")),
                "subject": safe_str(meta.get("subject", row.get("heading", ""))),
                "candidate_label": safe_str(row.get("redacted_text_truth", "")),
                "candidate_text": safe_str(row.get("redacted_text_truth", "")),
                "candidate_type": "UNKNOWN",
                "candidate_type_normalized": "UNKNOWN",
                "candidate_role_or_context": safe_str(row.get("local_context", "")),
                "candidate_role_context": safe_str(row.get("local_context", "")),
                "candidate_source_table": "redaction_span_index",
                "candidate_source_confidence": 1.0,
                "candidate_doc_id": doc_id,
                "candidate_evidence_json": json.dumps(
                    {
                        "left_context": safe_str(row.get("left_context", "")),
                        "right_context": safe_str(row.get("right_context", "")),
                        "truth": safe_str(row.get("redacted_text_truth", "")),
                    },
                    ensure_ascii=False,
                ),
                "pseudo_view_text": safe_str(row.get("local_context", "")),
                "evidence_class": "ground_truth_redaction_slot",
                "source_type": "prepared_redaction_span",
                "confidence_type": "ground_truth",
                "exact_text_match": True,
                "raw_chunk_used": True,
                "pseudo_chunk_used": False,
                "do_not_assume_exact_unredacted_copy": False,
            }
        )

    for row in claim_support_rows:
        claim_text = safe_str(row.get("claim_text", ""))
        claims_view_rows.append(
            {
                "pseudo_view_id": safe_str(row.get("pseudo_view_id", "")),
                "pseudo_view_type": "claim_search",
                "doc_id": safe_str(row.get("doc_id", "")),
                "source_id": "",
                "paragraph_id": safe_str(row.get("paragraph_id", "")),
                "date": "",
                "subject": safe_str(row.get("heading", "")),
                "claim_text": claim_text,
                "proposition_text": claim_text,
                "speaker": safe_str(row.get("speaker", "")),
                "target": safe_str(row.get("target", "")),
                "canonical_subject": safe_str(row.get("speaker", "")),
                "canonical_predicate": "",
                "canonical_object": safe_str(row.get("target", "")),
                "predicate_or_relation": "",
                "object_or_target": safe_str(row.get("target", "")),
                "stance": safe_str(row.get("stance", "")),
                "polarity": polarity_label(claim_text),
                "claim_anchors_json": json.dumps(anchor_tokens(claim_text)[:12], ensure_ascii=False),
                "pseudo_view_text": f"Claim: {claim_text}\nStance: {safe_str(row.get('stance', ''))}\nContext: {safe_str(row.get('context_text', ''))[:280]}",
                "source_type": "structured_claim",
                "confidence_type": "claim_level",
                "exact_text_match": False,
                "raw_chunk_used": False,
                "pseudo_chunk_used": True,
            }
        )

    for row in contradiction_rows:
        proposition = safe_str(row.get("assertion_text", ""))
        contradiction_view_rows.append(
            {
                "pseudo_view_id": safe_str(row.get("pseudo_view_id", "")),
                "pseudo_view_type": "contradiction_discovery",
                "doc_id": safe_str(row.get("doc_id", "")),
                "source_id": "",
                "paragraph_id": safe_str(row.get("paragraph_id", "")),
                "date": "",
                "proposition": proposition,
                "proposition_text": proposition,
                "normalized_proposition": norm_space(proposition).lower(),
                "canonical_subject": safe_str(row.get("speaker", "")),
                "normalized_relation": "",
                "canonical_object": safe_str(row.get("target", "")),
                "polarity": safe_str(row.get("polarity", "")),
                "date_or_temporal_context": "",
                "pseudo_view_text": f"Proposition: {proposition}\nSpeaker: {safe_str(row.get('speaker', ''))}\nTarget: {safe_str(row.get('target', ''))}",
                "source_type": "structured_claim",
                "confidence_type": "claim_level",
                "exact_text_match": False,
                "raw_chunk_used": False,
                "pseudo_chunk_used": True,
            }
        )

    compatibility_outputs = {
        "kg_pseudo_chunks.csv": pd.DataFrame(pseudo_chunk_rows),
        "kg_pseudo_chunk_components.csv": pd.DataFrame(pseudo_component_rows),
        "kg_pseudo_chunks_reconstruction.csv": pd.DataFrame(reconstruction_view_rows),
        "kg_pseudo_chunks_redaction.csv": pd.DataFrame(redaction_view_rows),
        "kg_pseudo_chunks_claims.csv": pd.DataFrame(claims_view_rows),
        "kg_pseudo_chunks_contradictions.csv": pd.DataFrame(contradiction_view_rows),
    }
    for filename, frame in compatibility_outputs.items():
        frame.to_csv(extraction_dir / filename, index=False)

    return {name: int(len(frame)) for name, frame in compatibility_outputs.items()}


def build_task_views(project_dir: Path, extraction_dir: Path, min_redaction_words: int, max_redaction_words: int) -> Dict[str, object]:
    tasks_dir = project_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    base_chunks = load_base_chunks(project_dir, extraction_dir)
    entities = load_table(extraction_dir / "entities.csv")
    events = load_table(extraction_dir / "events.csv")
    claims = load_table(extraction_dir / "claims.csv")
    relations = load_table(extraction_dir / "relations.csv")
    segments = load_table(project_dir / "prepared" / "source_segments.csv")
    redaction_spans = load_table(project_dir / "prepared" / "redaction_span_index.csv")

    entity_by_para = group_components(entities, "context_window_text", "normalized_label" if "normalized_label" in entities.columns else "label")
    event_by_para = group_components(events, "description", "label")
    claim_by_para = group_components(claims, "claim_text", "claim_text")

    relation_by_para: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    if len(relations):
        for _, row in relations.iterrows():
            key = (safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))
            relation_by_para[key].append(
                {
                    "row_id": safe_str(row.get("local_id", "")),
                    "source": safe_str(row.get("source", "")),
                    "target": safe_str(row.get("target", "")),
                    "relation": safe_str(row.get("relation", "")),
                    "evidence": safe_str(row.get("evidence", "")),
                    "chunk_id": safe_str(row.get("chunk_id", "")),
                    "source_id": safe_str(row.get("source_id", "")),
                    "date": safe_str(row.get("date", "")),
                    "subject": safe_str(row.get("subject", "")),
                }
            )

    pseudo_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []
    reconstruction_rows: List[Dict[str, object]] = []

    for _, row in segments.iterrows():
        doc_id = safe_str(row.get("doc_id", ""))
        paragraph_id = safe_str(row.get("paragraph_id", ""))
        key = (doc_id, paragraph_id)
        heading = safe_str(row.get("heading", ""))
        unredacted_text = safe_str(row.get("unredacted_text", ""))
        redacted_text = safe_str(row.get("redacted_text", ""))
        pseudo_chunk_id = stable_id("PC", doc_id, paragraph_id)
        ents = entity_by_para.get(key, [])
        evts = event_by_para.get(key, [])
        cls = claim_by_para.get(key, [])
        rels = relation_by_para.get(key, [])

        pseudo_rows.append(
            {
                "pseudo_chunk_id": pseudo_chunk_id,
                "doc_id": doc_id,
                "paragraph_id": paragraph_id,
                "heading": heading,
                "redaction_count": int(row.get("redaction_count", 0) or 0),
                "char_count": int(row.get("char_count", 0) or 0),
                "word_count": int(row.get("word_count", 0) or 0),
                "unredacted_text": unredacted_text,
                "redacted_text": redacted_text,
                "entity_labels": "; ".join(sorted({safe_str(item.get('label', '')) for item in ents if safe_str(item.get('label', ''))})),
                "event_labels": "; ".join(sorted({safe_str(item.get('label', '')) for item in evts if safe_str(item.get('label', ''))})),
                "claim_texts": " || ".join([safe_str(item.get("text", "")) for item in cls[:6]]),
                "relation_triples": " || ".join(
                    f"{safe_str(item.get('source', ''))} --{safe_str(item.get('relation', ''))}--> {safe_str(item.get('target', ''))}"
                    for item in rels[:8]
                ),
            }
        )

        for item in ents:
            component_rows.append(
                {
                    "pseudo_chunk_id": pseudo_chunk_id,
                    "component_kind": "entity",
                    "component_id": safe_str(item.get("row_id", "")),
                    "component_label": safe_str(item.get("label", "")),
                    "component_type": safe_str(item.get("type", "")),
                    "component_text": safe_str(item.get("text", "")),
                    "doc_id": doc_id,
                    "paragraph_id": paragraph_id,
                    "chunk_id": safe_str(item.get("chunk_id", "")),
                    "source_table": "entities",
                    "source_row_id": safe_str(item.get("row_id", "")),
                    "source_id": safe_str(item.get("source_id", "")),
                    "date": safe_str(item.get("date", row.get("date", ""))),
                    "subject": safe_str(item.get("subject", row.get("subject", heading))),
                    "base_entity_key": safe_str(item.get("base_entity_key", "")),
                }
            )
        for item in evts:
            component_rows.append(
                {
                    "pseudo_chunk_id": pseudo_chunk_id,
                    "component_kind": "event",
                    "component_id": safe_str(item.get("row_id", "")),
                    "component_label": safe_str(item.get("label", "")),
                    "component_text": safe_str(item.get("text", "")),
                    "doc_id": doc_id,
                    "paragraph_id": paragraph_id,
                    "chunk_id": safe_str(item.get("chunk_id", "")),
                    "source_table": "events",
                    "source_row_id": safe_str(item.get("row_id", "")),
                    "source_id": safe_str(item.get("source_id", "")),
                    "date": safe_str(item.get("date", row.get("date", ""))),
                    "subject": safe_str(item.get("subject", row.get("subject", heading))),
                }
            )
        for item in cls:
            component_rows.append(
                {
                    "pseudo_chunk_id": pseudo_chunk_id,
                    "component_kind": "claim",
                    "component_id": safe_str(item.get("row_id", "")),
                    "component_label": safe_str(item.get("label", "")),
                    "component_text": safe_str(item.get("text", "")),
                    "doc_id": doc_id,
                    "paragraph_id": paragraph_id,
                    "chunk_id": safe_str(item.get("chunk_id", "")),
                    "source_table": "claims",
                    "source_row_id": safe_str(item.get("row_id", "")),
                    "source_id": safe_str(item.get("source_id", "")),
                    "date": safe_str(item.get("date", row.get("date", ""))),
                    "subject": safe_str(item.get("subject", row.get("subject", heading))),
                }
            )
        for item in rels:
            component_rows.append(
                {
                    "pseudo_chunk_id": pseudo_chunk_id,
                    "component_kind": "relation",
                    "component_id": safe_str(item.get("row_id", "")) or stable_id("REL", doc_id, paragraph_id, item.get("source", ""), item.get("relation", ""), item.get("target", "")),
                    "component_label": f"{safe_str(item.get('source', ''))} --{safe_str(item.get('relation', ''))}--> {safe_str(item.get('target', ''))}",
                    "component_text": safe_str(item.get("evidence", "")),
                    "doc_id": doc_id,
                    "paragraph_id": paragraph_id,
                    "chunk_id": safe_str(item.get("chunk_id", "")),
                    "source_table": "relations",
                    "source_row_id": safe_str(item.get("row_id", "")),
                    "source_id": safe_str(item.get("source_id", "")),
                    "date": safe_str(item.get("date", row.get("date", ""))),
                    "subject": safe_str(item.get("subject", row.get("subject", heading))),
                }
            )

        reconstruction_rows.append(
            {
                "pseudo_view_id": stable_id("RECON", doc_id, paragraph_id),
                "pseudo_chunk_id": pseudo_chunk_id,
                "doc_id": doc_id,
                "paragraph_id": paragraph_id,
                "heading": heading,
                "entity_labels": "; ".join(sorted({safe_str(item.get('label', '')) for item in ents if safe_str(item.get('label', ''))})),
                "event_labels": "; ".join(sorted({safe_str(item.get('label', '')) for item in evts if safe_str(item.get('label', ''))})),
                "claim_texts": " || ".join([safe_str(item.get("text", "")) for item in cls[:4]]),
                "reconstruction_text": unredacted_text,
                "proposition_text": unredacted_text,
            }
        )

    short_redaction_rows: List[Dict[str, object]] = []
    sentence_redaction_rows: List[Dict[str, object]] = []
    if len(redaction_spans):
        has_preclassified = {
            "redaction_kind",
            "recovery_keep",
            "local_context",
            "bridge_context",
        }.issubset(set(redaction_spans.columns))
        if has_preclassified:
            for _, row in redaction_spans.iterrows():
                keep_value = safe_str(row.get("recovery_keep", "")).lower()
                if keep_value not in {"1", "true", "yes"}:
                    continue
                kind = safe_str(row.get("redaction_kind", "")).lower()
                text = safe_str(row.get("redacted_text", "")) or safe_str(row.get("text", ""))
                word_count = int(row.get("word_count", 0) or 0) or len(
                    [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)]
                )
                if kind == "value":
                    if word_count < min_redaction_words or word_count > max_redaction_words:
                        continue
                    short_redaction_rows.append(
                        {
                            "pseudo_view_id": stable_id("RED", row.get("doc_id", ""), row.get("paragraph_id", ""), row.get("redaction_id", "")),
                            "doc_id": safe_str(row.get("doc_id", "")),
                            "paragraph_id": safe_str(row.get("paragraph_id", "")),
                            "redaction_id": safe_str(row.get("redaction_id", "")),
                            "heading": safe_str(row.get("heading", "")),
                            "word_count": word_count,
                            "redacted_text_truth": text,
                            "left_context": safe_str(row.get("left_context", "")),
                            "right_context": safe_str(row.get("right_context", "")),
                            "local_context": safe_str(row.get("local_context", "")),
                        }
                    )
                elif kind == "sentence":
                    sentence_redaction_rows.append(
                        {
                            "pseudo_view_id": stable_id("SRED", row.get("doc_id", ""), row.get("paragraph_id", ""), row.get("redaction_id", "")),
                            "doc_id": safe_str(row.get("doc_id", "")),
                            "paragraph_id": safe_str(row.get("paragraph_id", "")),
                            "heading": safe_str(row.get("heading", "")),
                            "sentence_index": 0,
                            "redacted_sentence": safe_str(row.get("bridge_context", "")),
                            "unredacted_sentence_truth": text,
                        }
                    )
        else:
            for _, row in redaction_spans.iterrows():
                text = safe_str(row.get("redacted_text", ""))
                word_count = len([token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)])
                if word_count < min_redaction_words or word_count > max_redaction_words:
                    continue
                short_redaction_rows.append(
                    {
                        "pseudo_view_id": stable_id("RED", row.get("doc_id", ""), row.get("paragraph_id", ""), row.get("redaction_id", "")),
                        "doc_id": safe_str(row.get("doc_id", "")),
                        "paragraph_id": safe_str(row.get("paragraph_id", "")),
                        "redaction_id": safe_str(row.get("redaction_id", "")),
                        "heading": safe_str(row.get("heading", "")),
                        "word_count": word_count,
                        "redacted_text_truth": text,
                        "left_context": safe_str(row.get("left_context", "")),
                        "right_context": safe_str(row.get("right_context", "")),
                        "local_context": f"{safe_str(row.get('left_context', ''))} [REDACTED] {safe_str(row.get('right_context', ''))}".strip(),
                    }
                )

    if not sentence_redaction_rows:
        for _, row in segments.iterrows():
            doc_id = safe_str(row.get("doc_id", ""))
            paragraph_id = safe_str(row.get("paragraph_id", ""))
            heading = safe_str(row.get("heading", ""))
            unredacted_sentences = sentence_split(row.get("unredacted_text", ""))
            redacted_sentences = sentence_split(row.get("redacted_text", ""))
            max_len = max(len(unredacted_sentences), len(redacted_sentences))
            for idx in range(max_len):
                unred = unredacted_sentences[idx] if idx < len(unredacted_sentences) else ""
                red = redacted_sentences[idx] if idx < len(redacted_sentences) else ""
                if GAP_MARKER_RE.search(red) or (unred and red and unred != red):
                    sentence_redaction_rows.append(
                        {
                            "pseudo_view_id": stable_id("SRED", doc_id, paragraph_id, idx),
                            "doc_id": doc_id,
                            "paragraph_id": paragraph_id,
                            "heading": heading,
                            "sentence_index": idx,
                            "redacted_sentence": red,
                            "unredacted_sentence_truth": unred,
                        }
                    )

    claim_support_rows: List[Dict[str, object]] = []
    if len(claims):
        for _, row in claims.iterrows():
            doc_id = safe_str(row.get("doc_id", ""))
            paragraph_id = safe_str(row.get("paragraph_id", ""))
            key = (doc_id, paragraph_id)
            claim_support_rows.append(
                {
                    "pseudo_view_id": stable_id("CLAIM", doc_id, paragraph_id, row.get("local_id", ""), row.get("claim_text", "")),
                    "doc_id": doc_id,
                    "paragraph_id": paragraph_id,
                    "heading": safe_str(row.get("heading", "")),
                    "claim_text": safe_str(row.get("claim_text", "")),
                    "speaker": safe_str(row.get("speaker", "")),
                    "target": safe_str(row.get("target", "")),
                    "stance": safe_str(row.get("stance", "")),
                    "supporting_entities": "; ".join(sorted({safe_str(item.get('label', '')) for item in entity_by_para.get(key, []) if safe_str(item.get('label', ''))})),
                    "supporting_relations": " || ".join(
                        f"{safe_str(item.get('source', ''))} --{safe_str(item.get('relation', ''))}--> {safe_str(item.get('target', ''))}"
                        for item in relation_by_para.get(key, [])[:6]
                    ),
                    "context_text": safe_str(row.get("paragraph_text", "")) or safe_str(row.get("chunk_text", "")),
                }
            )

    contradiction_rows: List[Dict[str, object]] = []
    negation_re = re.compile(r"\b(no|not|never|deny|denied|denies|without|cannot|can't|won't|would not|did not)\b", re.I)
    if len(claim_support_rows):
        for row in claim_support_rows:
            claim_text = safe_str(row.get("claim_text", ""))
            contradiction_rows.append(
                {
                    "pseudo_view_id": stable_id("CONTRA", row.get("doc_id", ""), row.get("paragraph_id", ""), claim_text),
                    "doc_id": safe_str(row.get("doc_id", "")),
                    "paragraph_id": safe_str(row.get("paragraph_id", "")),
                    "heading": safe_str(row.get("heading", "")),
                    "assertion_text": claim_text,
                    "speaker": safe_str(row.get("speaker", "")),
                    "target": safe_str(row.get("target", "")),
                    "polarity": "negative" if negation_re.search(claim_text) else "positive_or_unspecified",
                    "supporting_entities": safe_str(row.get("supporting_entities", "")),
                    "supporting_relations": safe_str(row.get("supporting_relations", "")),
                }
            )

    outputs = {
        "pseudo_chunks.csv": pd.DataFrame(pseudo_rows),
        "pseudo_chunk_components.csv": pd.DataFrame(component_rows),
        "reconstruction_view.csv": pd.DataFrame(reconstruction_rows),
        "short_redaction_recovery_view.csv": pd.DataFrame(short_redaction_rows),
        "sentence_redaction_recovery_view.csv": pd.DataFrame(sentence_redaction_rows),
        "claim_support_view.csv": pd.DataFrame(claim_support_rows),
        "contradiction_tension_view.csv": pd.DataFrame(contradiction_rows),
    }
    for filename, frame in outputs.items():
        frame.to_csv(tasks_dir / filename, index=False)

    compat_counts = write_matcher_compatibility_outputs(
        project_dir=project_dir,
        extraction_dir=extraction_dir,
        base_chunks=base_chunks,
        segments=segments,
        pseudo_rows=pseudo_rows,
        component_rows=component_rows,
        reconstruction_rows=reconstruction_rows,
        short_redaction_rows=short_redaction_rows,
        claim_support_rows=claim_support_rows,
        contradiction_rows=contradiction_rows,
    )

    summary = {
        "project_dir": str(project_dir),
        "extraction_dir": str(extraction_dir),
        "pseudo_chunk_rows": int(len(outputs["pseudo_chunks.csv"])),
        "component_rows": int(len(outputs["pseudo_chunk_components.csv"])),
        "reconstruction_rows": int(len(outputs["reconstruction_view.csv"])),
        "short_redaction_rows": int(len(outputs["short_redaction_recovery_view.csv"])),
        "sentence_redaction_rows": int(len(outputs["sentence_redaction_recovery_view.csv"])),
        "claim_support_rows": int(len(outputs["claim_support_view.csv"])),
        "contradiction_rows": int(len(outputs["contradiction_tension_view.csv"])),
        "matcher_compatibility_rows": compat_counts,
    }
    (tasks_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C task materialization.")
    parser.add_argument("--project-dir", required=True, help="Project directory created by Stage A.")
    parser.add_argument("--extraction-dir", default=None, help="Optional extraction directory override.")
    parser.add_argument("--min-redaction-words", type=int, default=1)
    parser.add_argument("--max-redaction-words", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    extraction_dir = choose_extraction_dir(project_dir, args.extraction_dir)
    build_task_views(project_dir, extraction_dir, args.min_redaction_words, args.max_redaction_words)


if __name__ == "__main__":
    main()
