"""Authoritative query-package reader; no reparsing occurs in Matcher v0.4."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pandas as pd

from .contracts import safe_csv, safe_json
from .parser import (
    build_query_task_preflight,
    build_representation_reconciliation,
)
from .semantics import build_event_signature, clean_text, stable_hash


@dataclass(frozen=True)
class QueryPackage:
    root: Path
    manifest: Mapping[str, Any]
    fragments: pd.DataFrame
    entities: pd.DataFrame
    propositions: pd.DataFrame
    value_slots: pd.DataFrame
    sentence_slots: pd.DataFrame
    parse_sha256: str
    preflight: pd.DataFrame
    blocked_primary_target_ids: Tuple[str, ...]
    review_target_ids: Tuple[str, ...]

    @staticmethod
    def _is_primary(value: Any) -> bool:
        """Interpret persisted Boolean values without bool("False") leakage."""
        if isinstance(value, bool):
            return value
        return clean_text(value).lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _placeholder_strength(row: pd.Series) -> int:
        generic_predicates = {"claim", "claims", "statement", "assertion", "relation", "fact", "event"}
        strength = 0
        for field in (
            "canonical_subject",
            "canonical_object",
            "canonical_location",
            "event_time",
            "report_time",
            "date_value",
        ):
            if clean_text(row.get(field, "")):
                strength += 1
        predicate = clean_text(row.get("canonical_predicate", "")).lower()
        if predicate and predicate not in generic_predicates:
            strength += 1
        return strength

    @classmethod
    def _filter_placeholder_duplicates(
        cls,
        propositions: pd.DataFrame,
    ) -> pd.DataFrame:
        if propositions.empty:
            return propositions
        keys: list[str] = []
        strengths: list[int] = []
        for row in propositions.to_dict("records"):
            normalized = clean_text(
                row.get("normalized_proposition")
                or row.get("proposition_text")
            ).lower()
            fragment = clean_text(row.get("fragment_id", ""))
            keys.append(fragment + "||" + normalized)
            strengths.append(cls._placeholder_strength(pd.Series(row)))
        frame = propositions.copy()
        frame["_placeholder_key"] = keys
        frame["_placeholder_strength"] = strengths
        richest_by_key = frame.groupby("_placeholder_key")["_placeholder_strength"].transform("max")
        keep_mask = ~(
            (frame["_placeholder_strength"] == 0)
            & (richest_by_key >= 2)
        )
        filtered = frame.loc[keep_mask].drop(
            columns=["_placeholder_key", "_placeholder_strength"],
            errors="ignore",
        )
        return filtered.reset_index(drop=True)

    @classmethod
    def load(cls, root: Path, task: str = "") -> "QueryPackage":
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError("Query parse directory not found: " + str(resolved))
        manifest = safe_json(resolved / "query_parse_manifest.json")
        schema = clean_text(manifest.get("schema_version", ""))
        if schema not in {
            "query_parse_schema_v0_4_6",
            "query_parse_schema_v0_4_7",
        }:
            raise ValueError(
                "Matcher v0.4.20 requires a native query_parse_schema_v0_4_6 "
                "or query_parse_schema_v0_4_7 package; "
                f"received {schema or 'missing schema'}."
            )
        fragments = safe_csv(resolved / "query_fragments.csv")
        entities = safe_csv(resolved / "query_entities.csv")
        claims = safe_csv(resolved / "query_claims.csv")
        events = safe_csv(resolved / "query_events.csv")
        relations = safe_csv(resolved / "query_relations.csv")
        all_propositions = safe_csv(resolved / "query_propositions.csv")
        value_slots = safe_csv(resolved / "query_redaction_slots.csv")
        sentence_slots = safe_csv(resolved / "query_sentence_redaction_slots.csv")
        parsed_tables = {
            "fragments": fragments,
            "claims": claims,
            "events": events,
            "relations": relations,
            "propositions": all_propositions,
            "redaction_slots": value_slots,
            "sentence_redaction_slots": sentence_slots,
        }
        parsed_tables["representation_reconciliation"] = (
            build_representation_reconciliation(parsed_tables)
        )
        preflight = build_query_task_preflight(parsed_tables)
        scoped_preflight = preflight
        task_key = clean_text(task).lower()
        if task_key == "auto":
            scoped_preflight = preflight.iloc[0:0]
        if (
            task_key
            and task_key != "auto"
            and not preflight.empty
            and "task" in preflight.columns
        ):
            preflight_task = (
                "claims_or_contradictions"
                if task_key in {"claims", "contradictions"}
                else task_key
            )
            scoped_preflight = preflight[
                preflight["task"].fillna("").astype(str)
                .str.lower().eq(preflight_task)
            ]
        if "primary_for_matching" in all_propositions.columns:
            primary_mask = all_propositions["primary_for_matching"].map(
                cls._is_primary
            )
        else:
            # Native pre-v0.4.13 query packages did not always persist the
            # Boolean column. Their rows remain primary for backward
            # compatibility; new packages must provide the explicit field.
            primary_mask = pd.Series(
                [True] * len(all_propositions),
                index=all_propositions.index,
                dtype=bool,
            )
        primary_propositions = cls._filter_placeholder_duplicates(
            all_propositions[primary_mask].copy()
        )
        nonprimary_propositions = all_propositions[~primary_mask].copy()
        proposition_id_column = (
            "query_proposition_id"
            if "query_proposition_id" in all_propositions.columns
            else ""
        )
        primary_ids = {
            clean_text(value)
            for value in primary_propositions.get(
                proposition_id_column, pd.Series(dtype=str)
            ).tolist()
            if clean_text(value)
        }
        review_target_ids = tuple(sorted({
            clean_text(value)
            for value in nonprimary_propositions.get(
                proposition_id_column, pd.Series(dtype=str)
            ).tolist()
            if clean_text(value)
        }))
        blocked_primary_ids: set[str] = set()
        if (
            not scoped_preflight.empty
            and {"target_id", "status"}.issubset(scoped_preflight.columns)
        ):
            for row in scoped_preflight.itertuples(index=False):
                target_id = clean_text(getattr(row, "target_id", ""))
                status = clean_text(getattr(row, "status", ""))
                if target_id in primary_ids and status.startswith("BLOCKED_"):
                    blocked_primary_ids.add(target_id)
        # Claims can still provide useful bounded retrieval when a parser could
        # not establish a complete event frame. Preserve those proposition
        # surfaces for the claims engine, which labels them retrieval-only and
        # never promotes them to direct support without sufficient grounding.
        if (
            proposition_id_column
            and blocked_primary_ids
            and task_key != "claims"
        ):
            propositions = primary_propositions[
                ~primary_propositions[proposition_id_column]
                .fillna("").astype(str).map(clean_text)
                .isin(blocked_primary_ids)
            ].copy()
        else:
            propositions = primary_propositions
        if (
            task_key in {"claims", "contradictions"}
            and propositions.empty
        ):
            targets = ",".join(sorted(blocked_primary_ids))
            raise ValueError(
                "Query package has no executable primary target for "
                f"{task_key}; blocked primary target(s): "
                f"{targets or 'none'}."
            )
        parse_sha = clean_text(manifest.get("query_parse_sha256", "")) or stable_hash({
            "fragments": fragments.to_dict("records"),
            "propositions": all_propositions.to_dict("records"),
            "value_slots": value_slots.to_dict("records"),
            "sentence_slots": sentence_slots.to_dict("records"),
        })
        return cls(
            resolved, manifest, fragments, entities, propositions,
            value_slots, sentence_slots, parse_sha, preflight,
            tuple(sorted(blocked_primary_ids)), review_target_ids,
        )

    def proposition_records(self) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for index, row in self.propositions.iterrows():
            if (
                "primary_for_matching" in self.propositions.columns
                and not self._is_primary(row.get("primary_for_matching", False))
            ):
                continue
            text = clean_text(
                row.get("proposition_text", row.get("normalized_proposition", ""))
            )
            try:
                signature = json.loads(clean_text(row.get("event_signature_json", "")))
            except (TypeError, json.JSONDecodeError):
                signature = {}
            if not signature:
                signature = build_event_signature(
                    text,
                    actor=row.get("canonical_subject", ""),
                    predicate=row.get("canonical_predicate", ""),
                    obj=row.get("canonical_object", ""),
                    location=row.get("canonical_location", ""),
                    event_time=row.get("event_time", row.get("date_value", "")),
                    report_time=row.get("report_time", ""),
                    stance=row.get("stance", ""),
                    modality=row.get("modality", ""),
                )
            output.append({
                "id": clean_text(row.get("query_proposition_id", f"QP{index + 1}")),
                "fragment_id": clean_text(row.get("fragment_id", "U1")),
                "text": text,
                "signature": signature,
                "hypothesis_group_id": clean_text(
                    row.get("hypothesis_group_id", "")
                ),
                "proposition_hypothesis_id": clean_text(
                    row.get("proposition_hypothesis_id", "")
                ),
                "hypothesis_type": clean_text(
                    row.get(
                        "hypothesis_type",
                        "DIRECT_TYPED_INTERPRETATION",
                    )
                ),
                "subject_semantic_role": clean_text(
                    row.get("subject_semantic_role", "UNKNOWN")
                ),
                "hypothesis_confidence": row.get(
                    "hypothesis_confidence",
                    row.get("confidence", 1.0),
                ),
                "hypothesis_provenance_json": clean_text(
                    row.get("hypothesis_provenance_json", "")
                ),
            })
        return output
