"""Validated decision profiles for Matcher v0.4.20.

Profiles externalize thresholds and display policy only.  Score construction
weights remain frozen in task code.  Corpus terms, document identifiers,
expected answers, and score overrides are not valid profile fields.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional


PROFILE_SCHEMA = "matryoshka_decision_profile_v1"
PROFILE_MODES = frozenset({"score_only", "three_zone", "binary_threshold"})
DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent
    / "decision_profiles"
    / "default_decision_profile_v1.json"
)

DEFAULT_ENGINE_THRESHOLDS: Dict[str, float] = {
    "minimum_score": 0.42,
    "minimum_margin": 0.03,
    "reconstruction_critical_floor": 0.66,
    "reconstruction_identity_floor": 0.40,
    "sentence_context_floor": 0.20,
    "sentence_event_floor": 0.22,
    "sentence_one_sided_context_floor": 0.45,
    "sentence_one_sided_margin": 0.10,
    "claim_direct_identity": 0.62,
    "claim_conflict_identity": 0.72,
    "related_identity": 0.40,
    "contradiction_identity_floor": 0.48,
    "contradiction_anchor_floor": 0.40,
    "value_redaction_title_containment_floor": 0.55,
    "value_redaction_event_floor": 0.35,
    "sentence_local_anchor_floor": 0.34,
    "claim_background_lexical_floor": 0.12,
    "claim_raw_validation_floor": 0.40,
    "contradiction_participant_containment_floor": 0.66,
    "contradiction_object_containment_floor": 0.66,
    "contradiction_route_relevance_floor": 0.34,
    "contradiction_background_retrieval_floor": 0.35,
    "contradiction_raw_validation_floor": 0.40,
    "reconstruction_literal_containment_floor": 0.95,
}

DEFAULT_DISPLAY_POLICY: Dict[str, float] = {
    "lower_threshold": 0.42,
    "upper_threshold": 0.62,
    "minimum_margin": 0.03,
}

_ROOT_KEYS = frozenset({
    "schema_version", "profile_id", "profile_version", "description",
    "decision_mode", "thresholds", "display_policy",
})
_DISPLAY_KEYS = frozenset(DEFAULT_DISPLAY_POLICY)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(field + " must be a finite number between 0 and 1.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            field + " must be a finite number between 0 and 1."
        ) from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(field + " must be between 0 and 1 inclusive.")
    return result


def _read_profile(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Decision profile not found: " + str(resolved))
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Decision profile is invalid JSON: " + str(resolved)) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Decision profile root must be a JSON object.")
    unknown_root = sorted(set(payload) - _ROOT_KEYS)
    if unknown_root:
        raise ValueError(
            "Decision profile contains unsupported root field(s): "
            + ", ".join(unknown_root)
        )
    if payload.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("Decision profile schema must be " + PROFILE_SCHEMA)
    for field in ("profile_id", "profile_version"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError("Decision profile requires nonblank " + field)
    mode = payload.get("decision_mode", "score_only")
    if mode not in PROFILE_MODES:
        raise ValueError(
            "decision_mode must be one of: " + ", ".join(sorted(PROFILE_MODES))
        )
    thresholds = payload.get("thresholds", {})
    if not isinstance(thresholds, Mapping):
        raise ValueError("Decision profile thresholds must be an object.")
    unknown_thresholds = sorted(set(thresholds) - set(DEFAULT_ENGINE_THRESHOLDS))
    if unknown_thresholds:
        raise ValueError(
            "Decision profile contains unsupported threshold(s): "
            + ", ".join(unknown_thresholds)
        )
    display = payload.get("display_policy", {})
    if not isinstance(display, Mapping):
        raise ValueError("Decision profile display_policy must be an object.")
    unknown_display = sorted(set(display) - _DISPLAY_KEYS)
    if unknown_display:
        raise ValueError(
            "Decision profile contains unsupported display field(s): "
            + ", ".join(unknown_display)
        )
    return payload


@dataclass(frozen=True)
class DecisionProfile:
    profile_id: str
    profile_version: str
    decision_mode: str
    thresholds: Mapping[str, float]
    threshold_sources: Mapping[str, str]
    display_policy: Mapping[str, float]
    configured_path: Path
    configured_sha256: str
    packaged_default_path: Path
    packaged_default_sha256: str

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "decision_mode": self.decision_mode,
            "configured_path": str(self.configured_path),
            "configured_sha256": self.configured_sha256,
            "packaged_default_path": str(self.packaged_default_path),
            "packaged_default_sha256": self.packaged_default_sha256,
            "score_weights_externalized": False,
            "thresholds": dict(self.thresholds),
            "threshold_sources": dict(self.threshold_sources),
            "display_policy": dict(self.display_policy),
        }


def load_decision_profile(
    path: Optional[Path] = None,
    *,
    minimum_score_override: Optional[float] = None,
    minimum_margin_override: Optional[float] = None,
) -> DecisionProfile:
    """Load a partial custom profile over the packaged v0.4.16 defaults."""
    packaged = DEFAULT_PROFILE_PATH.resolve()
    default_payload = _read_profile(packaged)
    configured = path.expanduser().resolve() if path else packaged
    payload = default_payload if configured == packaged else _read_profile(configured)

    thresholds = dict(DEFAULT_ENGINE_THRESHOLDS)
    sources = {name: "packaged_default" for name in thresholds}
    for name, raw in payload.get("thresholds", {}).items():
        thresholds[name] = _number(raw, field="thresholds." + name)
        sources[name] = (
            "packaged_default" if configured == packaged else "decision_profile"
        )
    if minimum_score_override is not None:
        thresholds["minimum_score"] = _number(
            minimum_score_override, field="minimum_score_override"
        )
        sources["minimum_score"] = "cli_override"
    if minimum_margin_override is not None:
        thresholds["minimum_margin"] = _number(
            minimum_margin_override, field="minimum_margin_override"
        )
        sources["minimum_margin"] = "cli_override"

    if thresholds["related_identity"] > thresholds["claim_direct_identity"]:
        raise ValueError("related_identity may not exceed claim_direct_identity.")
    if thresholds["related_identity"] > thresholds["claim_conflict_identity"]:
        raise ValueError("related_identity may not exceed claim_conflict_identity.")

    display = dict(DEFAULT_DISPLAY_POLICY)
    for name, raw in payload.get("display_policy", {}).items():
        display[name] = _number(raw, field="display_policy." + name)
    if display["lower_threshold"] > display["upper_threshold"]:
        raise ValueError(
            "display_policy.lower_threshold may not exceed upper_threshold."
        )

    return DecisionProfile(
        profile_id=str(payload["profile_id"]).strip(),
        profile_version=str(payload["profile_version"]).strip(),
        decision_mode=str(payload.get("decision_mode", "score_only")),
        thresholds=MappingProxyType(thresholds),
        threshold_sources=MappingProxyType(sources),
        display_policy=MappingProxyType(display),
        configured_path=configured,
        configured_sha256=_sha256(configured),
        packaged_default_path=packaged,
        packaged_default_sha256=_sha256(packaged),
    )


def operational_band(
    *,
    eligible: bool,
    score: Any,
    margin: Any,
    runner_present: bool,
    mode: str,
    display_policy: Mapping[str, float],
) -> str:
    """Derive a display-only band without changing engine eligibility."""
    if not eligible:
        return "INELIGIBLE_SEMANTIC_MISMATCH"
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return "ELIGIBLE_SCORE_UNAVAILABLE"
    try:
        numeric_margin = float(margin)
    except (TypeError, ValueError):
        numeric_margin = 0.0
    if mode == "score_only":
        return "ELIGIBLE_SCORE_ONLY"
    margin_pass = (
        not runner_present
        or numeric_margin >= float(display_policy["minimum_margin"])
    )
    if mode == "binary_threshold":
        return (
            "ELIGIBLE_ABOVE_THRESHOLD"
            if numeric_score >= float(display_policy["upper_threshold"])
            and margin_pass
            else "ELIGIBLE_BELOW_THRESHOLD"
        )
    if numeric_score >= float(display_policy["upper_threshold"]) and margin_pass:
        return "ELIGIBLE_ABOVE_THRESHOLD"
    if numeric_score < float(display_policy["lower_threshold"]):
        return "ELIGIBLE_BELOW_THRESHOLD"
    return "ELIGIBLE_REVIEW_ZONE"
