#!/usr/bin/env python3
"""
Stage B: graph materialization from canonicalized structured extraction tables.

This stage intentionally stays narrower than the old notebook-era graph build.
It writes a clean, provenance-preserving KG substrate:

- graph/nodes.csv
- graph/edges.csv
- graph/provenance_links.csv
- graph/summary.json

It does not attempt hierarchy construction. That belongs later if needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_scripts._common import normalize_label_basic, norm_space, safe_str


def stable_hash(*parts: object, length: int = 12) -> str:
    payload = "||".join(norm_space(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length].upper()


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


def make_entity_node_id(label: str) -> str:
    return f"ENTITY:{normalize_label_basic(label).replace(' ', '_')}"


def make_document_node_id(doc_id: str) -> str:
    return f"DOCUMENT:{doc_id}"


def make_segment_node_id(doc_id: str, paragraph_id: str) -> str:
    return f"SEGMENT:{doc_id}::{paragraph_id}"


def make_event_node_id(row: Mapping[str, object]) -> str:
    local_id = safe_str(row.get("local_id", ""))
    if local_id:
        return f"EVENT:{safe_str(row.get('doc_id', ''))}::{safe_str(row.get('paragraph_id', ''))}::{local_id}"
    return f"EVENT:{stable_hash(row.get('doc_id', ''), row.get('paragraph_id', ''), row.get('label', ''), row.get('description', ''))}"


def make_claim_node_id(row: Mapping[str, object]) -> str:
    local_id = safe_str(row.get("local_id", ""))
    if local_id:
        return f"CLAIM:{safe_str(row.get('doc_id', ''))}::{safe_str(row.get('paragraph_id', ''))}::{local_id}"
    return f"CLAIM:{stable_hash(row.get('doc_id', ''), row.get('paragraph_id', ''), row.get('claim_text', ''))}"


def build_entity_nodes(entities: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str, str], str], Dict[Tuple[str, str], str]]:
    if len(entities) == 0:
        return pd.DataFrame(columns=["node_id", "node_type", "label"]), {}, {}

    normalized_col = "normalized_label" if "normalized_label" in entities.columns else "label"
    type_col = "type" if "type" in entities.columns else None

    rows: List[Dict[str, object]] = []
    mention_lookup: Dict[Tuple[str, str, str, str], str] = {}
    paragraph_label_lookup: Dict[Tuple[str, str], str] = {}

    grouped = entities.groupby(normalized_col, dropna=False)
    for label, group in grouped:
        label_text = norm_space(label)
        if not label_text:
            continue
        node_id = make_entity_node_id(label_text)
        type_counter = Counter(group[type_col].fillna("").astype(str).tolist()) if type_col else Counter()
        doc_count = int(group["doc_id"].astype(str).nunique()) if "doc_id" in group.columns else 0
        mention_count = int(len(group))
        rows.append(
            {
                "node_id": node_id,
                "node_type": "Entity",
                "label": label_text,
                "primary_type": type_counter.most_common(1)[0][0] if type_counter else "",
                "type_distribution": "; ".join(f"{key}:{count}" for key, count in type_counter.most_common()),
                "mention_count": mention_count,
                "doc_count": doc_count,
                "surface_forms": "; ".join(sorted({norm_space(value) for value in group.get('label', pd.Series(dtype=str)).astype(str).tolist() if norm_space(value)})[:12]),
            }
        )
        for _, row in group.iterrows():
            mention_lookup[
                (
                    safe_str(row.get("doc_id", "")),
                    safe_str(row.get("paragraph_id", "")),
                    safe_str(row.get("chunk_id", "")),
                    safe_str(row.get("local_id", "")),
                )
            ] = node_id
            paragraph_label_lookup[
                (
                    safe_str(row.get("doc_id", "")),
                    normalize_label_basic(row.get(normalized_col, "")),
                )
            ] = node_id
    return pd.DataFrame(rows), mention_lookup, paragraph_label_lookup


def build_document_and_segment_nodes(source_segments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    doc_rows: List[Dict[str, object]] = []
    seg_rows: List[Dict[str, object]] = []
    seen_docs: set[str] = set()
    for _, row in source_segments.iterrows():
        doc_id = safe_str(row.get("doc_id", ""))
        paragraph_id = safe_str(row.get("paragraph_id", ""))
        if not doc_id or not paragraph_id:
            continue
        if doc_id not in seen_docs:
            doc_rows.append(
                {
                    "node_id": make_document_node_id(doc_id),
                    "node_type": "Document",
                    "label": doc_id,
                    "date": safe_str(row.get("date", "")),
                    "subject": safe_str(row.get("subject", "")),
                }
            )
            seen_docs.add(doc_id)
        seg_rows.append(
            {
                "node_id": make_segment_node_id(doc_id, paragraph_id),
                "node_type": "Segment",
                "label": paragraph_id,
                "doc_id": doc_id,
                "paragraph_id": paragraph_id,
                "heading": safe_str(row.get("heading", "")),
                "char_count": int(row.get("char_count", 0) or 0),
                "word_count": int(row.get("word_count", 0) or 0),
            }
        )
    return pd.DataFrame(doc_rows), pd.DataFrame(seg_rows)


def resolve_relation_entity(
    value: object,
    row: Mapping[str, object],
    mention_lookup: Mapping[Tuple[str, str, str, str], str],
    paragraph_label_lookup: Mapping[Tuple[str, str], str],
) -> str:
    raw_value = safe_str(value)
    if not raw_value:
        return ""
    direct = mention_lookup.get(
        (
            safe_str(row.get("doc_id", "")),
            safe_str(row.get("paragraph_id", "")),
            safe_str(row.get("chunk_id", "")),
            raw_value,
        )
    )
    if direct:
        return direct
    normalized = normalize_label_basic(raw_value)
    return paragraph_label_lookup.get((safe_str(row.get("doc_id", "")), normalized), make_entity_node_id(normalized))


def build_graph(project_dir: Path, extraction_dir: Path) -> Dict[str, object]:
    graph_dir = project_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    entities = load_table(extraction_dir / "entities.csv")
    events = load_table(extraction_dir / "events.csv")
    claims = load_table(extraction_dir / "claims.csv")
    relations = load_table(extraction_dir / "relations.csv")
    source_segments = load_table(project_dir / "prepared" / "source_segments.csv")

    entity_nodes, mention_lookup, paragraph_label_lookup = build_entity_nodes(entities)
    document_nodes, segment_nodes = build_document_and_segment_nodes(source_segments)

    event_rows: List[Dict[str, object]] = []
    claim_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    provenance_rows: List[Dict[str, object]] = []

    for _, row in events.iterrows():
        node_id = make_event_node_id(row)
        label = safe_str(row.get("label", "")) or safe_str(row.get("description", ""))
        event_rows.append(
            {
                "node_id": node_id,
                "node_type": "Event",
                "label": label,
                "event_type": safe_str(row.get("type", "")),
                "doc_id": safe_str(row.get("doc_id", "")),
                "paragraph_id": safe_str(row.get("paragraph_id", "")),
                "chunk_id": safe_str(row.get("chunk_id", "")),
            }
        )
        provenance_rows.append(
            {
                "node_id": node_id,
                "source_table": "events",
                "doc_id": safe_str(row.get("doc_id", "")),
                "paragraph_id": safe_str(row.get("paragraph_id", "")),
                "chunk_id": safe_str(row.get("chunk_id", "")),
                "source_local_id": safe_str(row.get("local_id", "")),
                "evidence_text": safe_str(row.get("description", "")),
            }
        )
        seg_id = make_segment_node_id(safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))
        edge_rows.append({"source_node_id": seg_id, "target_node_id": node_id, "edge_type": "SEGMENT_HAS_EVENT", "weight": 1.0, "doc_id": safe_str(row.get("doc_id", "")), "paragraph_id": safe_str(row.get("paragraph_id", ""))})

    for _, row in claims.iterrows():
        node_id = make_claim_node_id(row)
        label = safe_str(row.get("claim_text", "")) or safe_str(row.get("label", ""))
        claim_rows.append(
            {
                "node_id": node_id,
                "node_type": "Claim",
                "label": label,
                "stance": safe_str(row.get("stance", "")),
                "doc_id": safe_str(row.get("doc_id", "")),
                "paragraph_id": safe_str(row.get("paragraph_id", "")),
                "chunk_id": safe_str(row.get("chunk_id", "")),
            }
        )
        provenance_rows.append(
            {
                "node_id": node_id,
                "source_table": "claims",
                "doc_id": safe_str(row.get("doc_id", "")),
                "paragraph_id": safe_str(row.get("paragraph_id", "")),
                "chunk_id": safe_str(row.get("chunk_id", "")),
                "source_local_id": safe_str(row.get("local_id", "")),
                "evidence_text": label,
            }
        )
        seg_id = make_segment_node_id(safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))
        edge_rows.append({"source_node_id": seg_id, "target_node_id": node_id, "edge_type": "SEGMENT_HAS_CLAIM", "weight": 1.0, "doc_id": safe_str(row.get("doc_id", "")), "paragraph_id": safe_str(row.get("paragraph_id", ""))})
        speaker = safe_str(row.get("speaker", ""))
        target = safe_str(row.get("target", ""))
        if speaker:
            edge_rows.append({"source_node_id": resolve_relation_entity(speaker, row, mention_lookup, paragraph_label_lookup), "target_node_id": node_id, "edge_type": "CLAIM_SPEAKER", "weight": 1.0, "doc_id": safe_str(row.get("doc_id", "")), "paragraph_id": safe_str(row.get("paragraph_id", ""))})
        if target:
            edge_rows.append({"source_node_id": node_id, "target_node_id": resolve_relation_entity(target, row, mention_lookup, paragraph_label_lookup), "edge_type": "CLAIM_TARGET", "weight": 1.0, "doc_id": safe_str(row.get("doc_id", "")), "paragraph_id": safe_str(row.get("paragraph_id", ""))})

    for _, row in entities.iterrows():
        normalized_label = safe_str(row.get("normalized_label", "")) or safe_str(row.get("label", ""))
        entity_node_id = make_entity_node_id(normalized_label)
        seg_id = make_segment_node_id(safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))
        edge_rows.append({"source_node_id": seg_id, "target_node_id": entity_node_id, "edge_type": "SEGMENT_HAS_ENTITY", "weight": 1.0, "doc_id": safe_str(row.get("doc_id", "")), "paragraph_id": safe_str(row.get("paragraph_id", ""))})
        provenance_rows.append(
            {
                "node_id": entity_node_id,
                "source_table": "entities",
                "doc_id": safe_str(row.get("doc_id", "")),
                "paragraph_id": safe_str(row.get("paragraph_id", "")),
                "chunk_id": safe_str(row.get("chunk_id", "")),
                "source_local_id": safe_str(row.get("local_id", "")),
                "evidence_text": safe_str(row.get("context_window_text", "")) or safe_str(row.get("description", "")),
            }
        )

    for _, row in relations.iterrows():
        source_node = resolve_relation_entity(row.get("source", ""), row, mention_lookup, paragraph_label_lookup)
        target_node = resolve_relation_entity(row.get("target", ""), row, mention_lookup, paragraph_label_lookup)
        if not source_node or not target_node:
            continue
        edge_rows.append(
            {
                "source_node_id": source_node,
                "target_node_id": target_node,
                "edge_type": safe_str(row.get("relation", "")) or "RELATED",
                "weight": float(row.get("confidence", 1.0) or 1.0),
                "doc_id": safe_str(row.get("doc_id", "")),
                "paragraph_id": safe_str(row.get("paragraph_id", "")),
                "evidence_text": safe_str(row.get("evidence", "")),
            }
        )

    for _, row in segment_nodes.iterrows():
        edge_rows.append(
            {
                "source_node_id": make_document_node_id(safe_str(row.get("doc_id", ""))),
                "target_node_id": safe_str(row.get("node_id", "")),
                "edge_type": "DOCUMENT_HAS_SEGMENT",
                "weight": 1.0,
                "doc_id": safe_str(row.get("doc_id", "")),
                "paragraph_id": safe_str(row.get("paragraph_id", "")),
            }
        )

    nodes_df = pd.concat(
        [
            document_nodes,
            segment_nodes,
            entity_nodes,
            pd.DataFrame(event_rows),
            pd.DataFrame(claim_rows),
        ],
        ignore_index=True,
    ).fillna("")
    edges_df = pd.DataFrame(edge_rows).fillna("")
    provenance_df = pd.DataFrame(provenance_rows).fillna("")

    nodes_df = nodes_df.drop_duplicates(subset=["node_id"]).reset_index(drop=True)
    edges_df = edges_df[(edges_df["source_node_id"] != "") & (edges_df["target_node_id"] != "")].reset_index(drop=True)
    provenance_df = provenance_df[provenance_df["node_id"] != ""].reset_index(drop=True)

    nodes_df.to_csv(graph_dir / "nodes.csv", index=False)
    edges_df.to_csv(graph_dir / "edges.csv", index=False)
    provenance_df.to_csv(graph_dir / "provenance_links.csv", index=False)

    summary = {
        "project_dir": str(project_dir),
        "extraction_dir": str(extraction_dir),
        "node_count": int(len(nodes_df)),
        "edge_count": int(len(edges_df)),
        "provenance_count": int(len(provenance_df)),
        "entity_node_count": int((nodes_df["node_type"] == "Entity").sum()) if len(nodes_df) else 0,
        "event_node_count": int((nodes_df["node_type"] == "Event").sum()) if len(nodes_df) else 0,
        "claim_node_count": int((nodes_df["node_type"] == "Claim").sum()) if len(nodes_df) else 0,
        "segment_node_count": int((nodes_df["node_type"] == "Segment").sum()) if len(nodes_df) else 0,
        "document_node_count": int((nodes_df["node_type"] == "Document").sum()) if len(nodes_df) else 0,
    }
    (graph_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B graph materialization.")
    parser.add_argument("--project-dir", required=True, help="Project directory created by Stage A.")
    parser.add_argument("--extraction-dir", default=None, help="Optional extraction directory override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    extraction_dir = choose_extraction_dir(project_dir, args.extraction_dir)
    build_graph(project_dir, extraction_dir)


if __name__ == "__main__":
    main()
