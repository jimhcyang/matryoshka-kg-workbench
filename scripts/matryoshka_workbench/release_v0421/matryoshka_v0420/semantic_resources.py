"""Versioned generic and corpus-scoped resources for Matcher v0.4.20.

Generic semantic resources contain reusable language concepts.  Corpus entity
resources contain only versioned, provenance-bearing aliases selected through
configuration.  Neither resource may encode an expected task answer, preferred
source, score adjustment, or decision override.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .semantics import clean_text, semantic_tokens, sha256_file, stable_hash


RESOURCE_SCHEMA = "matryoshka_semantic_resources_v1"
CORPUS_ENTITY_RESOURCE_SCHEMA = (
    "matryoshka_corpus_entity_alias_resource_v1"
)
CORE_RESOURCE_PATH = (
    Path(__file__).resolve().parent.parent
    / "semantic_resources"
    / "core_semantic_resources_v2.json"
)

_ANSWER_KEY_FIELDS = frozenset({
    "answer", "answers", "answer_key", "expected_answer",
    "expected_value", "expected_outcome", "target_label", "case_id",
    "preferred_source", "preferred_source_doc", "preferred_source_document",
    "preferred_source_document_id", "score_bonus", "scoring_bonus",
    "decision_override", "forced_decision",
})
_ENTITY_TYPES = frozenset({
    "PERSON", "ORGANIZATION", "LOCATION", "FACILITY", "EVENT",
    "VEHICLE", "PRODUCT", "OTHER",
})
_REVIEW_STATES = frozenset({
    "CANDIDATE", "ACCEPTED", "REJECTED", "DEPRECATED",
})
_INDEX_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "with",
})


def _find_answer_key_fields(value: Any, prefix: str = "") -> List[str]:
    found: List[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = clean_text(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.casefold() in _ANSWER_KEY_FIELDS:
                found.append(path)
            found.extend(_find_answer_key_fields(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            found.extend(
                _find_answer_key_fields(child, f"{prefix}[{index}]")
            )
    return found


@dataclass(frozen=True)
class CorpusEntityResource:
    """Immutable, ambiguity-preserving corpus entity resource."""

    corpus_resource_id: str
    corpus_id: str
    resource_version: str
    entities: Tuple[Mapping[str, Any], ...]
    path: Path
    size_bytes: int
    sha256: str
    fingerprint: str

    def resolve(self, surface_form: Any) -> List[Dict[str, Any]]:
        normalized = _normalized_phrase(surface_form)
        if not normalized:
            return []
        matches: List[Dict[str, Any]] = []
        for entity in self.entities:
            for form in entity["surface_forms"]:
                if form["normalized_form"] != normalized:
                    continue
                if form["review_state"] in {"REJECTED", "DEPRECATED"}:
                    continue
                matches.append({
                    "canonical_entity_id":
                        entity["canonical_entity_id"],
                    "entity_type": entity["entity_type"],
                    "canonical_display_form":
                        entity["canonical_display_form"],
                    "matched_surface_form": form["surface_form"],
                    "alias_type": form["alias_type"],
                    "language": form.get("language", ""),
                    "script": form.get("script", ""),
                    "confidence": form["confidence"],
                    "review_state": form["review_state"],
                    "scope": dict(form.get("scope", {})),
                    "provenance": [
                        dict(item) for item in form["provenance"]
                    ],
                    "resolution_state": (
                        "AMBIGUOUS_ALIAS"
                        if sum(
                            1 for other in self.entities
                            for other_form in other["surface_forms"]
                            if other_form["normalized_form"] == normalized
                            and other_form["review_state"]
                            not in {"REJECTED", "DEPRECATED"}
                        ) > 1
                        else "RESOLVED_ALIAS"
                    ),
                })
        return sorted(
            matches,
            key=lambda item: (
                -float(item["confidence"]),
                item["canonical_entity_id"],
            ),
        )

    def resolve_open_world(
        self, surface_form: Any, *, entity_type: str = "OTHER"
    ) -> List[Dict[str, Any]]:
        matches = self.resolve(surface_form)
        if matches:
            return matches
        surface = clean_text(surface_form)
        if not surface:
            return []
        normalized_type = clean_text(entity_type).upper() or "OTHER"
        if normalized_type not in _ENTITY_TYPES:
            normalized_type = "OTHER"
        return [{
            "canonical_entity_id": "",
            "entity_type": normalized_type,
            "canonical_display_form": surface,
            "matched_surface_form": surface,
            "alias_type": "OPEN_WORLD",
            "language": "",
            "script": "",
            "confidence": 0.50,
            "review_state": "UNREVIEWED",
            "scope": {},
            "provenance": [],
            "resolution_state": "OPEN_WORLD_UNKNOWN",
        }]

    def alias_groups(self) -> Dict[str, Tuple[str, ...]]:
        groups: Dict[str, Tuple[str, ...]] = {}
        for entity in self.entities:
            accepted = {
                form["normalized_form"]
                for form in entity["surface_forms"]
                if form["review_state"] not in {"REJECTED", "DEPRECATED"}
            }
            canonical = _normalized_phrase(
                entity["canonical_display_form"]
            )
            if canonical:
                accepted.add(canonical)
            if accepted:
                groups[
                    "corpus_entity:" + entity["canonical_entity_id"]
                ] = tuple(sorted(accepted))
        return groups

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": CORPUS_ENTITY_RESOURCE_SCHEMA,
            "corpus_resource_id": self.corpus_resource_id,
            "corpus_id": self.corpus_id,
            "resource_version": self.resource_version,
            "configured_path": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "fingerprint": self.fingerprint,
            "entity_count": len(self.entities),
            "load_status": "LOADED",
        }


def load_corpus_entity_resource(path: Path) -> CorpusEntityResource:
    """Load and validate the normative corpus entity-resource v1 contract."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            "Corpus entity resource not found: " + str(resolved)
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Corpus entity resource JSON is invalid: " + str(resolved)
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Corpus entity resource root must be an object.")
    prohibited = _find_answer_key_fields(payload)
    if prohibited:
        raise ValueError(
            "Corpus entity resource contains prohibited answer-key "
            "field(s): " + ", ".join(prohibited)
        )
    if payload.get("schema_version") != CORPUS_ENTITY_RESOURCE_SCHEMA:
        raise ValueError(
            "Corpus entity resource schema must be "
            + CORPUS_ENTITY_RESOURCE_SCHEMA
        )
    required_root = (
        "corpus_resource_id", "corpus_id", "resource_version", "entities",
    )
    missing_root = [
        field for field in required_root
        if not payload.get(field) and field != "entities"
    ]
    if missing_root:
        raise ValueError(
            "Corpus entity resource missing field(s): "
            + ", ".join(missing_root)
        )
    raw_entities = payload.get("entities", [])
    if not isinstance(raw_entities, list):
        raise ValueError("Corpus entity resource entities must be an array.")
    entities: List[Mapping[str, Any]] = []
    seen_ids = set()
    for index, raw_entity in enumerate(raw_entities):
        if not isinstance(raw_entity, Mapping):
            raise ValueError(f"Entity record {index} must be an object.")
        entity_id = clean_text(raw_entity.get("canonical_entity_id", ""))
        entity_type = clean_text(
            raw_entity.get("entity_type", "")
        ).upper()
        display = clean_text(
            raw_entity.get("canonical_display_form", "")
        )
        if not entity_id or not display or entity_type not in _ENTITY_TYPES:
            raise ValueError(
                f"Entity record {index} has an invalid ID, type, or display."
            )
        if entity_id in seen_ids:
            raise ValueError(
                "Duplicate corpus-scoped canonical entity ID: " + entity_id
            )
        seen_ids.add(entity_id)
        raw_forms = raw_entity.get("surface_forms", [])
        if not isinstance(raw_forms, list) or not raw_forms:
            raise ValueError(
                f"Entity record {entity_id} requires surface forms."
            )
        forms: List[Mapping[str, Any]] = []
        for form_index, raw_form in enumerate(raw_forms):
            if not isinstance(raw_form, Mapping):
                raise ValueError(
                    f"Surface form {entity_id}[{form_index}] must be an object."
                )
            surface = clean_text(raw_form.get("surface_form", ""))
            review_state = clean_text(
                raw_form.get("review_state", "")
            ).upper()
            provenance = raw_form.get("provenance", [])
            try:
                confidence = float(raw_form.get("confidence"))
            except (TypeError, ValueError):
                confidence = -1.0
            if (
                not surface
                or not clean_text(raw_form.get("alias_type", ""))
                or review_state not in _REVIEW_STATES
                or not 0.0 <= confidence <= 1.0
                or not isinstance(provenance, list)
                or not provenance
            ):
                raise ValueError(
                    f"Surface form {entity_id}[{form_index}] violates "
                    "alias, confidence, review, or provenance requirements."
                )
            for provenance_index, item in enumerate(provenance):
                if not isinstance(item, Mapping) or not all(
                    clean_text(item.get(field, ""))
                    for field in (
                        "derivation_method", "source_table",
                        "source_row_id",
                    )
                ):
                    raise ValueError(
                        f"Surface form {entity_id}[{form_index}] provenance "
                        f"{provenance_index} is incomplete."
                    )
            forms.append(MappingProxyType({
                **dict(raw_form),
                "surface_form": surface,
                "normalized_form": (
                    _normalized_phrase(
                        raw_form.get("normalized_form", "")
                    )
                    or _normalized_phrase(surface)
                ),
                "confidence": confidence,
                "review_state": review_state,
                "provenance": tuple(
                    MappingProxyType(dict(item)) for item in provenance
                ),
            }))
        entities.append(MappingProxyType({
            **dict(raw_entity),
            "canonical_entity_id": entity_id,
            "entity_type": entity_type,
            "canonical_display_form": display,
            "surface_forms": tuple(forms),
        }))
    size = resolved.stat().st_size
    digest = sha256_file(resolved)
    identity = {
        "schema_version": CORPUS_ENTITY_RESOURCE_SCHEMA,
        "corpus_resource_id":
            clean_text(payload["corpus_resource_id"]),
        "resource_version": clean_text(payload["resource_version"]),
        "sha256": digest,
    }
    return CorpusEntityResource(
        corpus_resource_id=identity["corpus_resource_id"],
        corpus_id=clean_text(payload["corpus_id"]),
        resource_version=identity["resource_version"],
        entities=tuple(entities),
        path=resolved,
        size_bytes=size,
        sha256=digest,
        fingerprint=stable_hash(identity),
    )


def _normalized_phrase(value: Any) -> str:
    # Phrase matching must preserve order; ``semantic_tokens`` is a sorted
    # retrieval set and is therefore used only for expanded bag-of-token
    # scoring below.
    return " ".join(
        re.findall(r"[a-z0-9]+", clean_text(value).casefold())
    )


def _normalized_groups(
    payload: Mapping[str, Any], field: str
) -> Dict[str, Tuple[str, ...]]:
    groups: Dict[str, Tuple[str, ...]] = {}
    raw_groups = payload.get(field, {})
    if not isinstance(raw_groups, Mapping):
        raise ValueError(f"Semantic resource field {field!r} must be an object.")
    for raw_name, raw_values in raw_groups.items():
        name = clean_text(raw_name).casefold().replace(" ", "_")
        if not name:
            raise ValueError(f"Semantic resource field {field!r} has an empty key.")
        if not isinstance(raw_values, Sequence) or isinstance(
            raw_values, (str, bytes)
        ):
            raise ValueError(
                f"Semantic resource group {field}.{raw_name} must be an array."
            )
        values = sorted({
            phrase
            for phrase in (_normalized_phrase(value) for value in raw_values)
            if phrase
        })
        if not values:
            raise ValueError(
                f"Semantic resource group {field}.{raw_name} is empty."
            )
        groups[name] = tuple(values)
    return groups


def _read_resource(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Semantic resource file not found: " + str(resolved))
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Semantic resource JSON is invalid: " + str(resolved)
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Semantic resource root must be an object: " + str(resolved))
    if payload.get("schema") != RESOURCE_SCHEMA:
        raise ValueError(
            f"Semantic resource schema must be {RESOURCE_SCHEMA!r}: {resolved}"
        )
    normalized = {
        "schema": RESOURCE_SCHEMA,
        "resource_id": clean_text(payload.get("resource_id", resolved.stem)),
        "resource_version": clean_text(payload.get("resource_version", "")),
        "alias_groups": _normalized_groups(payload, "alias_groups"),
        "event_families": _normalized_groups(payload, "event_families"),
    }
    manifest = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "resource_id": normalized["resource_id"],
        "resource_version": normalized["resource_version"],
    }
    return normalized, manifest


def _phrase_present(normalized_text: str, phrase: str) -> bool:
    haystack = " " + normalized_text + " "
    needle = " " + phrase + " "
    return needle in haystack


def _anchor_tokens(normalized_phrase: str) -> Tuple[str, ...]:
    tokens = [
        token for token in normalized_phrase.split()
        if token and token not in _INDEX_STOPWORDS and len(token) >= 3
    ]
    if not tokens:
        tokens = [token for token in normalized_phrase.split() if token]
    if not tokens:
        return ()
    ranked = sorted(set(tokens), key=lambda token: (-len(token), token))
    return tuple(ranked[:2])


def _concept_lookup_tokens(normalized_text: str) -> Tuple[str, ...]:
    """Return phrase-index probes without losing aliases embedded in prose.

    Alias groups remain indexed by a small number of discriminative phrase
    tokens.  Querying that index with only the two longest tokens from a full
    sentence, however, misses aliases whose words are not globally longest.
    Probe every informative token and adjacent initialism run; the subsequent
    exact phrase check remains the precision gate.
    """
    raw_tokens = [token for token in normalized_text.split() if token]
    probes = {
        token
        for token in raw_tokens
        if token not in _INDEX_STOPWORDS and len(token) >= 2
    }
    run: List[str] = []
    for token in raw_tokens + [""]:
        if len(token) == 1 and token.isalnum():
            run.append(token)
            continue
        if len(run) >= 2:
            probes.update(run)
        run = []
    if not probes:
        probes.update(raw_tokens)
    return tuple(sorted(probes))


@dataclass(frozen=True)
class SemanticResources:
    """Immutable merged alias and event-family maps."""

    alias_groups: Mapping[str, Tuple[str, ...]]
    event_families: Mapping[str, Tuple[str, ...]]
    sources: Tuple[Mapping[str, Any], ...]
    fingerprint: str
    _concept_cache: Dict[Tuple[str, str], frozenset[str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _expanded_token_cache: Dict[str, frozenset[str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _alias_phrase_index: Dict[str, Tuple[Tuple[str, str], ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _event_phrase_index: Dict[str, Tuple[Tuple[str, str], ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _alias_phrase_trie: Dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _event_phrase_trie: Dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def _phrase_index(
        self,
        family: str,
    ) -> Dict[str, Tuple[Tuple[str, str], ...]]:
        target = (
            self._alias_phrase_index
            if family == "alias"
            else self._event_phrase_index
        )
        if target:
            return target
        groups = (
            self.alias_groups if family == "alias" else self.event_families
        )
        staged: Dict[str, List[Tuple[str, str]]] = {}
        for name, phrases in groups.items():
            for phrase in phrases:
                tokens = _anchor_tokens(phrase)
                for token in tokens:
                    staged.setdefault(token, []).append((name, phrase))
        for token, values in staged.items():
            target[token] = tuple(values)
        return target

    def _phrase_trie(self, family: str) -> Dict[str, Any]:
        """Build an exact token-boundary matcher for semantic phrases."""
        target = (
            self._alias_phrase_trie
            if family == "alias"
            else self._event_phrase_trie
        )
        if target:
            return target
        groups = (
            self.alias_groups if family == "alias" else self.event_families
        )
        for name, phrases in groups.items():
            for phrase in phrases:
                phrase_tokens = phrase.split()
                informative = any(
                    token not in _INDEX_STOPWORDS and len(token) >= 2
                    for token in phrase_tokens
                )
                initialism = (
                    len(phrase_tokens) >= 2
                    and all(
                        len(token) == 1 and token.isalnum()
                        for token in phrase_tokens
                    )
                )
                if not informative and not initialism:
                    # A corpus extractor can occasionally emit a stopword as
                    # an entity surface.  The legacy indexed matcher did not
                    # activate such aliases inside prose; retaining that
                    # precision guard also prevents groups such as "to" from
                    # matching nearly every document.
                    continue
                node = target
                for token in phrase_tokens:
                    node = node.setdefault(token, {})
                node.setdefault("", set()).add(name)
        return target

    def concepts(self, value: Any, family: str) -> frozenset[str]:
        normalized = _normalized_phrase(value)
        if not normalized:
            return frozenset()
        cache_key = (family, normalized)
        cached = self._concept_cache.get(cache_key)
        if cached is not None:
            return cached
        matched_names = set()
        alias_spans: List[Tuple[int, int, frozenset[str]]] = []
        trie = self._phrase_trie(family)
        tokens = normalized.split()
        for start, token in enumerate(tokens):
            node = trie.get(token)
            if node is None:
                continue
            end = start + 1
            while node:
                names = frozenset(node.get("", ()))
                if names:
                    if family == "alias":
                        alias_spans.append((start, end, names))
                    else:
                        matched_names.update(names)
                if end >= len(tokens):
                    break
                node = node.get(tokens[end])
                if node is None:
                    break
                end += 1
        if family == "alias" and alias_spans:
            # Entity surfaces can overlap while naming different referents,
            # for example "Zia" and "Zia Nassry". Prefer the longest
            # non-overlapping spans, while retaining every group attached to
            # an identical span so genuine ambiguity remains visible.
            occupied: set[int] = set()
            by_span: Dict[Tuple[int, int], set[str]] = {}
            for start, end, names in alias_spans:
                by_span.setdefault((start, end), set()).update(names)
            ordered = sorted(
                by_span.items(),
                key=lambda item: (
                    -(item[0][1] - item[0][0]),
                    item[0][0],
                    item[0][1],
                ),
            )
            for (start, end), names in ordered:
                span_tokens = set(range(start, end))
                if occupied & span_tokens:
                    continue
                matched_names.update(names)
                occupied.update(span_tokens)
        resolved = frozenset(matched_names)
        self._concept_cache[cache_key] = resolved
        return resolved

    def expanded_tokens(self, value: Any) -> frozenset[str]:
        normalized = _normalized_phrase(value)
        if not normalized:
            return frozenset()
        cached = self._expanded_token_cache.get(normalized)
        if cached is not None:
            return cached
        tokens = set(semantic_tokens(value))
        for concept in sorted(self.concepts(value, "alias")):
            tokens.add("alias:" + concept)
        for concept in sorted(self.concepts(value, "event")):
            tokens.add("event:" + concept)
        resolved = frozenset(tokens)
        self._expanded_token_cache[normalized] = resolved
        return resolved

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema": RESOURCE_SCHEMA,
            "fingerprint": self.fingerprint,
            "sources": [dict(item) for item in self.sources],
            "alias_group_count": len(self.alias_groups),
            "event_family_count": len(self.event_families),
        }


def load_semantic_resources(
    additional_path: Optional[Path] = None,
    *,
    include_core: bool = True,
    corpus_entity_resource: Optional[CorpusEntityResource] = None,
) -> SemanticResources:
    """Load core resources and optionally merge an explicitly supplied file."""
    paths = []
    if include_core:
        paths.append(CORE_RESOURCE_PATH)
    if additional_path is not None:
        paths.append(additional_path)
    if not paths:
        empty = {"alias_groups": {}, "event_families": {}, "sources": []}
        return SemanticResources(
            alias_groups=MappingProxyType({}),
            event_families=MappingProxyType({}),
            sources=(),
            fingerprint=stable_hash(empty),
        )

    aliases: Dict[str, Tuple[str, ...]] = {}
    events: Dict[str, Tuple[str, ...]] = {}
    manifests = []
    for path in paths:
        payload, manifest = _read_resource(path)
        manifests.append(manifest)
        for field, target in (
            ("alias_groups", aliases),
            ("event_families", events),
        ):
            for name, values in payload[field].items():
                target[name] = tuple(sorted(set(target.get(name, ())) | set(values)))
    if corpus_entity_resource is not None:
        for name, values in corpus_entity_resource.alias_groups().items():
            aliases[name] = tuple(
                sorted(set(aliases.get(name, ())) | set(values))
            )
        manifests.append({
            "path": str(corpus_entity_resource.path),
            "sha256": corpus_entity_resource.sha256,
            "resource_id": corpus_entity_resource.corpus_resource_id,
            "resource_version": corpus_entity_resource.resource_version,
            "resource_kind": "corpus_entity_alias_resource",
        })
    fingerprint = stable_hash({
        "schema": RESOURCE_SCHEMA,
        "alias_groups": aliases,
        "event_families": events,
        # Resource identity is content/version based and therefore remains
        # stable when an unchanged package is moved to another directory.
        "sources": [
            {
                "sha256": item["sha256"],
                "resource_id": item["resource_id"],
                "resource_version": item["resource_version"],
            }
            for item in manifests
        ],
    })
    return SemanticResources(
        alias_groups=MappingProxyType(aliases),
        event_families=MappingProxyType(events),
        sources=tuple(MappingProxyType(item) for item in manifests),
        fingerprint=fingerprint,
    )


def shared_concepts(
    resources: SemanticResources,
    left: Any,
    right: Any,
    family: str,
) -> frozenset[str]:
    """Return concepts explicitly supported by both texts."""
    return resources.concepts(left, family) & resources.concepts(right, family)
