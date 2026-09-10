#!/usr/bin/env python3
"""
Stage A: paragraph-first KG extraction from prepared source segments.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ._common import (
    OpenAI,
    append_jsonl,
    call_openai_json,
    iter_progress,
    load_jsonl,
)


ALLOWED_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Government",
    "Location",
    "Country",
    "Military",
    "ReligiousGroup",
    "PoliticalGroup",
    "Document",
    "Other",
]

ALLOWED_EVENT_TYPES = [
    "Meeting",
    "Communication",
    "Demonstration",
    "Violence",
    "Arrest",
    "Travel",
    "Speech",
    "Negotiation",
    "PolicyAction",
    "MilitaryAction",
    "EconomicAction",
    "Other",
]

ALLOWED_RELATIONS = [
    "EVENT_PARTICIPANT",
    "EVENT_LOCATION",
    "EVENT_TIME",
    "AFFILIATION",
    "SUPPORT",
    "OPPOSITION",
    "NEGOTIATION",
    "REQUEST",
    "AGREEMENT",
    "COMMUNICATION",
    "TEMPORAL",
    "CAUSAL",
    "CLAIM_SPEAKER",
    "CLAIM_TARGET",
    "MENTIONS",
    "OTHER_RELATED",
]

ALLOWED_STANCES = {"support", "oppose", "neutral", "unclear"}
MAX_CHARS_PER_CHUNK = 3000
CHUNK_OVERLAP_SENTENCES = 1
CONTENTS_HEADING_RE = re.compile(r"^CONTENTS$", re.IGNORECASE)
NONEXTRACTIVE_HEADING_PATTERNS = [
    re.compile(r"^\(Charts\)$", re.IGNORECASE),
    re.compile(r"^Road Trail\b.*$", re.IGNORECASE),
]
NONEXTRACTIVE_TEXT_PATTERNS = [
    re.compile(r"Government Antigovernment concentrations", re.IGNORECASE),
    re.compile(r"0 25 50 75 100 Miles 0 25 50 75 100 Kilometers", re.IGNORECASE),
    re.compile(r"Boundary representation is not necessarily authoritative", re.IGNORECASE),
]

ENTITY_TYPE_LIST = "\n".join(f"- {value}" for value in ALLOWED_ENTITY_TYPES)
EVENT_TYPE_LIST = "\n".join(f"- {value}" for value in ALLOWED_EVENT_TYPES)
RELATION_LIST = "\n".join(f"- {value}" for value in ALLOWED_RELATIONS)


def safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def norm_space(value: object) -> str:
    return re.sub(r"\s+", " ", safe_str(value)).strip()


def is_effectively_empty(text: object) -> bool:
    raw = norm_space(text)
    if not raw:
        return True
    raw = raw.replace("[REDACTED]", " ").replace("[REDACTION]", " ")
    core = re.sub(r"[\W_]+", "", raw)
    return len(core) < 10


def sentence_split(text: str) -> List[str]:
    text = safe_str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])|\n{2,}", text)
    out = [norm_space(piece) for piece in pieces if norm_space(piece)]
    return out if out else [norm_space(text)]


def split_long_paragraph(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    text = norm_space(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = sentence_split(text)
    if len(sentences) <= 1:
        return [
            piece.strip()
            for piece in (text[i:i + max_chars] for i in range(0, len(text), max_chars))
            if piece.strip()
        ]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        extra = len(sentence) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append(" ".join(current).strip())
            overlap = current[-CHUNK_OVERLAP_SENTENCES:] if CHUNK_OVERLAP_SENTENCES > 0 else []
            current = overlap[:] if overlap else []
            current_len = len(" ".join(current).strip())
        current.append(sentence)
        current_len = len(" ".join(current).strip())
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def is_nonextractive_segment(heading: object, text: object) -> bool:
    heading_value = norm_space(heading)
    text_value = norm_space(text)
    if not text_value:
        return True
    if CONTENTS_HEADING_RE.fullmatch(heading_value):
        return True
    if text_value.upper().startswith("CONTENTS "):
        return True
    if any(pattern.match(heading_value) for pattern in NONEXTRACTIVE_HEADING_PATTERNS):
        return True
    if any(pattern.search(text_value) for pattern in NONEXTRACTIVE_TEXT_PATTERNS):
        sentence_like = bool(re.search(r"[.!?]", text_value))
        token_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9&'/.-]*", text_value))
        if token_count <= 40 or not sentence_like:
            return True
    return False


def chunk_rows_from_segments(segments: pd.DataFrame, variant: str) -> pd.DataFrame:
    text_col = f"{variant}_text"
    if text_col not in segments.columns:
        raise KeyError(f"Missing text column in source segments: {text_col}")

    rows: List[Dict[str, object]] = []
    for _, row in segments.iterrows():
        paragraph_text = norm_space(row.get(text_col, ""))
        if is_effectively_empty(paragraph_text):
            continue
        if is_nonextractive_segment(row.get("heading", ""), paragraph_text):
            continue
        chunks = split_long_paragraph(paragraph_text)
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{safe_str(row['paragraph_id'])}_c{idx:02d}"
            rows.append(
                {
                    "doc_id": safe_str(row["doc_id"]),
                    "date": safe_str(row.get("date", "")),
                    "subject": safe_str(row.get("subject", "")),
                    "source_path": safe_str(row.get("source_path", "")),
                    "heading": safe_str(row.get("heading", "")),
                    "paragraph_id": safe_str(row["paragraph_id"]),
                    "redaction_count": int(row.get("redaction_count", 0) or 0),
                    "segment_char_count": int(row.get("char_count", 0) or 0),
                    "segment_word_count": int(row.get("word_count", 0) or 0),
                    "source_variant": variant,
                    "paragraph_text": paragraph_text,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "chunk_char_len": len(chunk_text),
                }
            )
    return pd.DataFrame(rows)


def extraction_prompt(row: pd.Series) -> str:
    return f"""
You are extracting a controlled knowledge graph from a historical intelligence-document paragraph chunk.

Be conservative.
Use only information explicitly supported by the visible text.
Do not invent entities, events, claims, relations, dates, or locations.
If the chunk contains no useful extractable information, return empty arrays.

If the text contains placeholders such as [REDACTED], treat them as missing text.
Do not extract [REDACTED] or placeholder tokens as entities, events, claims, or evidence.

Allowed entity types:
{ENTITY_TYPE_LIST}

Allowed event types:
{EVENT_TYPE_LIST}

Allowed relation types:
{RELATION_LIST}

Important distinctions:
- Entity = person, organization, place, country, group, institution, or document.
- Event = something that happened or was reported to happen.
- Claim = a statement, allegation, interpretation, warning, or assertion.
- Relation = a connection between entities, events, or claims.
- Claims are not necessarily verified truth. Preserve them as claims.

Metadata:
doc_id={safe_str(row.get("doc_id", ""))}
paragraph_id={safe_str(row.get("paragraph_id", ""))}
chunk_id={safe_str(row.get("chunk_id", ""))}
subject={safe_str(row.get("subject", ""))}
heading={safe_str(row.get("heading", ""))}
date={safe_str(row.get("date", ""))}
source_variant={safe_str(row.get("source_variant", ""))}

Return ONLY valid JSON with this exact structure:
{{
  "entities": [
    {{
      "local_id": "ent_0",
      "label": "entity name",
      "type": "Person|Organization|Government|Location|Country|Military|ReligiousGroup|PoliticalGroup|Document|Other",
      "description": "brief description if supported",
      "confidence": 0.7
    }}
  ],
  "events": [
    {{
      "local_id": "event_0",
      "label": "short event label",
      "type": "Meeting|Communication|Demonstration|Violence|Arrest|Travel|Speech|Negotiation|PolicyAction|MilitaryAction|EconomicAction|Other",
      "date": "date if explicitly available, otherwise empty",
      "location": "location if explicitly available, otherwise empty",
      "description": "brief event description",
      "confidence": 0.7
    }}
  ],
  "claims": [
    {{
      "local_id": "claim_0",
      "claim_text": "the claim or assertion",
      "speaker": "who made the claim if known",
      "target": "target of the claim if known",
      "stance": "support|oppose|neutral|unclear",
      "confidence": 0.7
    }}
  ],
  "relations": [
    {{
      "source": "local_id or label",
      "target": "local_id or label",
      "relation": "one of the allowed relation types",
      "evidence": "short supporting phrase from the text",
      "confidence": 0.7
    }}
  ]
}}

Text:
{safe_str(row.get("chunk_text", ""))}
""".strip()


def safe_float(value: object, default: float = 0.7) -> float:
    try:
        out = float(value)
        if pd.isna(out):
            return default
        return max(0.0, min(1.0, out))
    except Exception:
        return default


def normalize_extraction(payload: object) -> Dict[str, List[dict]]:
    if not isinstance(payload, dict):
        payload = {}
    out = {
        "entities": payload.get("entities", []) or [],
        "events": payload.get("events", []) or [],
        "claims": payload.get("claims", []) or [],
        "relations": payload.get("relations", []) or [],
    }

    entities: List[dict] = []
    for idx, ent in enumerate(out["entities"]):
        if not isinstance(ent, dict):
            continue
        ent_type = safe_str(ent.get("type", "Other")).strip() or "Other"
        if ent_type not in ALLOWED_ENTITY_TYPES:
            ent_type = "Other"
        label = norm_space(ent.get("label", ""))
        if not label or label.upper() == "REDACTED":
            continue
        entities.append(
            {
                "local_id": safe_str(ent.get("local_id", f"ent_{idx}")),
                "label": label,
                "type": ent_type,
                "description": norm_space(ent.get("description", "")),
                "confidence": safe_float(ent.get("confidence", 0.7)),
            }
        )

    events: List[dict] = []
    for idx, event in enumerate(out["events"]):
        if not isinstance(event, dict):
            continue
        event_type = safe_str(event.get("type", "Other")).strip() or "Other"
        if event_type not in ALLOWED_EVENT_TYPES:
            event_type = "Other"
        label = norm_space(event.get("label", ""))
        if not label:
            continue
        events.append(
            {
                "local_id": safe_str(event.get("local_id", f"event_{idx}")),
                "label": label,
                "type": event_type,
                "date": norm_space(event.get("date", "")),
                "location": norm_space(event.get("location", "")),
                "description": norm_space(event.get("description", "")),
                "confidence": safe_float(event.get("confidence", 0.7)),
            }
        )

    claims: List[dict] = []
    for idx, claim in enumerate(out["claims"]):
        if not isinstance(claim, dict):
            continue
        stance = safe_str(claim.get("stance", "unclear")).strip().lower() or "unclear"
        if stance not in ALLOWED_STANCES:
            stance = "unclear"
        claim_text = norm_space(claim.get("claim_text", ""))
        if not claim_text:
            continue
        claims.append(
            {
                "local_id": safe_str(claim.get("local_id", f"claim_{idx}")),
                "claim_text": claim_text,
                "speaker": norm_space(claim.get("speaker", "")),
                "target": norm_space(claim.get("target", "")),
                "stance": stance,
                "confidence": safe_float(claim.get("confidence", 0.7)),
            }
        )

    relations: List[dict] = []
    for relation in out["relations"]:
        if not isinstance(relation, dict):
            continue
        rel_type = safe_str(relation.get("relation", "")).strip()
        if rel_type not in ALLOWED_RELATIONS:
            rel_type = "OTHER_RELATED"
        source = norm_space(relation.get("source", ""))
        target = norm_space(relation.get("target", ""))
        if not source or not target or source == target:
            continue
        relations.append(
            {
                "source": source,
                "target": target,
                "relation": rel_type,
                "evidence": norm_space(relation.get("evidence", "")),
                "confidence": safe_float(relation.get("confidence", 0.7)),
            }
        )

    return {
        "entities": entities,
        "events": events,
        "claims": claims,
        "relations": relations,
    }


def chunk_key(row: pd.Series) -> Tuple[str, str, str]:
    return (
        safe_str(row.get("doc_id", "")),
        safe_str(row.get("paragraph_id", "")),
        safe_str(row.get("chunk_id", "")),
    )


def checkpoint_keys(path: Path) -> set[Tuple[str, str, str]]:
    keys: set[Tuple[str, str, str]] = set()
    for row in load_jsonl(path):
        provenance = row.get("provenance", {}) if isinstance(row, dict) else {}
        keys.add(
            (
                safe_str(provenance.get("doc_id", "")),
                safe_str(provenance.get("paragraph_id", "")),
                safe_str(provenance.get("chunk_id", "")),
            )
        )
    return keys


def process_chunk(client: OpenAI, row: pd.Series, model: str) -> Dict[str, object]:
    prompt = extraction_prompt(row)
    parsed = call_openai_json(client, prompt=prompt, model=model)
    extraction = normalize_extraction(parsed)
    return {
        "provenance": {
            "doc_id": safe_str(row.get("doc_id", "")),
            "date": safe_str(row.get("date", "")),
            "subject": safe_str(row.get("subject", "")),
            "source_path": safe_str(row.get("source_path", "")),
            "heading": safe_str(row.get("heading", "")),
            "paragraph_id": safe_str(row.get("paragraph_id", "")),
            "chunk_id": safe_str(row.get("chunk_id", "")),
            "chunk_index": int(row.get("chunk_index", 0) or 0),
            "chunk_char_len": int(row.get("chunk_char_len", 0) or 0),
            "source_variant": safe_str(row.get("source_variant", "")),
            "redaction_count": int(row.get("redaction_count", 0) or 0),
            "chunk_text": safe_str(row.get("chunk_text", "")),
            "paragraph_text": safe_str(row.get("paragraph_text", "")),
            "model": model,
        },
        "extraction": extraction,
    }


def flatten_results(results: Sequence[dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chunk_rows: List[dict] = []
    entity_rows: List[dict] = []
    event_rows: List[dict] = []
    claim_rows: List[dict] = []
    relation_rows: List[dict] = []

    for row in results:
        provenance = row.get("provenance", {}) if isinstance(row, dict) else {}
        extraction = row.get("extraction", {}) if isinstance(row, dict) else {}
        base = {
            "doc_id": safe_str(provenance.get("doc_id", "")),
            "date": safe_str(provenance.get("date", "")),
            "subject": safe_str(provenance.get("subject", "")),
            "source_path": safe_str(provenance.get("source_path", "")),
            "heading": safe_str(provenance.get("heading", "")),
            "paragraph_id": safe_str(provenance.get("paragraph_id", "")),
            "chunk_id": safe_str(provenance.get("chunk_id", "")),
            "chunk_index": int(provenance.get("chunk_index", 0) or 0),
            "chunk_char_len": int(provenance.get("chunk_char_len", 0) or 0),
            "source_variant": safe_str(provenance.get("source_variant", "")),
            "redaction_count": int(provenance.get("redaction_count", 0) or 0),
            "model": safe_str(provenance.get("model", "")),
        }

        chunk_rows.append(
            {
                **base,
                "chunk_text": safe_str(provenance.get("chunk_text", "")),
                "paragraph_text": safe_str(provenance.get("paragraph_text", "")),
            }
        )

        for entity in extraction.get("entities", []):
            local_id = safe_str(entity.get("local_id", ""))
            mention_key = f"{base['doc_id']}::{base['paragraph_id']}::{base['chunk_id']}::{local_id}"
            entity_rows.append(
                {
                    **base,
                    **entity,
                    "mention_key": mention_key,
                    "node_key": mention_key,
                }
            )
        for event in extraction.get("events", []):
            local_id = safe_str(event.get("local_id", ""))
            node_key = f"{base['doc_id']}::{base['paragraph_id']}::{base['chunk_id']}::{local_id}"
            event_rows.append({**base, **event, "node_key": node_key})
        for claim in extraction.get("claims", []):
            local_id = safe_str(claim.get("local_id", ""))
            node_key = f"{base['doc_id']}::{base['paragraph_id']}::{base['chunk_id']}::{local_id}"
            claim_rows.append({**base, **claim, "node_key": node_key})
        for relation in extraction.get("relations", []):
            relation_rows.append({**base, **relation})

    return (
        pd.DataFrame(chunk_rows),
        pd.DataFrame(entity_rows),
        pd.DataFrame(event_rows),
        pd.DataFrame(claim_rows),
        pd.DataFrame(relation_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A extraction from prepared source segments.")
    parser.add_argument("PROJECT_DIR", help="Prepared project directory.")
    parser.add_argument(
        "--transcription-variant",
        choices=["unredacted", "redacted"],
        default=None,
        help="Optional override for which prepared variant to extract.",
    )
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for Stage A extraction.")
    parser.add_argument("--max-new-chunks", type=int, default=None, help="Optional cap on newly processed chunks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.PROJECT_DIR).expanduser().resolve()
    prepared_dir = project_dir / "prepared"
    source_segments_path = prepared_dir / "source_segments.csv"
    if not source_segments_path.exists():
        raise FileNotFoundError(f"Missing prepared source segments: {source_segments_path}")

    build_summary = json.loads((project_dir / "input_build_summary.json").read_text(encoding="utf-8"))
    variant = args.transcription_variant or safe_str(build_summary.get("transcription_variant", "unredacted")) or "unredacted"
    segments = pd.read_csv(source_segments_path, low_memory=False)
    chunk_df = chunk_rows_from_segments(segments, variant=variant)
    if len(chunk_df) == 0:
        raise ValueError("No non-empty chunk rows were generated from source segments.")

    extraction_dir = project_dir / "extraction"
    run_dir = project_dir / "extraction_run"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "chunk_results.jsonl"
    error_path = run_dir / "chunk_errors.jsonl"

    completed = checkpoint_keys(checkpoint_path)
    pending_mask = [chunk_key(row) not in completed for _, row in chunk_df.iterrows()]
    pending_df = chunk_df[pending_mask].copy()
    if args.max_new_chunks is not None:
        pending_df = pending_df.head(args.max_new_chunks).copy()

    total_for_progress = len(chunk_df)
    print(
        json.dumps(
            {
                "variant": variant,
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
                record = process_chunk(client, row, model=args.openai_model)
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
    chunks_df, entities_df, events_df, claims_df, relations_df = flatten_results(results)

    chunks_df.to_csv(extraction_dir / "chunks.csv", index=False)
    entities_df.to_csv(extraction_dir / "entities.csv", index=False)
    events_df.to_csv(extraction_dir / "events.csv", index=False)
    claims_df.to_csv(extraction_dir / "claims.csv", index=False)
    relations_df.to_csv(extraction_dir / "relations.csv", index=False)

    summary = {
        "variant": variant,
        "chunks_csv": str(extraction_dir / "chunks.csv"),
        "entities_rows": int(len(entities_df)),
        "events_rows": int(len(events_df)),
        "claims_rows": int(len(claims_df)),
        "relations_rows": int(len(relations_df)),
        "chunk_rows": int(len(chunks_df)),
        "checkpoint_rows": int(len(results)),
    }
    (extraction_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
