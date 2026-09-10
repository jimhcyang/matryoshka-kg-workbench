#!/usr/bin/env python3
"""
Alias resolution over Stage A base-entity profiles.

This stage does not split same-spelling mentions into senses.
Instead, it asks the narrower question:

    when do two different spellings appear to refer to the same entity?

Outputs:
- alias/alias_candidates.csv
- alias/alias_edges.csv
- alias/canonical_entities.csv
- alias/alias_map.csv
- extraction_canonicalized/{entities,events,claims,relations}.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import pandas as pd
from pandas.errors import EmptyDataError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_scripts._common import (
    LOCATION_MARKER_TOKENS,
    MILITARY_MARKER_TOKENS,
    OpenAI,
    ORGANIZATION_MARKER_TOKENS,
    PERSON_TITLE_TOKENS,
    POLITICAL_MARKER_TOKENS,
    ROLE_MARKER_TOKENS,
    call_openai_json,
    cooccurrence_jaccard,
    infer_structural_family,
    iter_progress,
    jaccard,
    keyify,
    looks_like_acronym,
    norm_space,
    normalize_label_basic,
    safe_str,
    stable_base_entity_key,
    token_set,
    token_list,
    truncate_prompt_text,
    type_compatibility,
    type_set_from_text,
)


ALIAS_FIELD_POLICY = {
    "entities.csv": ["label", "normalized_label", "base_entity_label"],
    "claims.csv": ["speaker", "target", "source_actor", "claimant", "object", "about"],
    "events.csv": ["location", "place", "event_location"],
    "relations.csv": [
        "source",
        "target",
        "subject",
        "object",
        "source_label",
        "target_label",
        "head",
        "tail",
        "entity_a",
        "entity_b",
        "from",
        "to",
    ],
}

ALIAS_STOPWORDS = {
    "A",
    "AN",
    "AND",
    "AS",
    "AT",
    "BY",
    "FOR",
    "FROM",
    "IN",
    "INTO",
    "OF",
    "ON",
    "OR",
    "THE",
    "TO",
    "WITH",
}

ALIAS_HONORIFIC_TOKENS = PERSON_TITLE_TOKENS | {
    "AGHA",
    "AYATOLLAH",
    "ENG",
    "HAJ",
    "HAJI",
    "HOJATOLLAH",
    "HOJATOLESLAM",
    "IMAM",
    "MOLLAH",
    "MULLAH",
    "SAYYED",
    "SEYED",
    "SEYYED",
    "SHAH",
    "SHEIKH",
    "SYED",
}

ALIAS_PERSON_SUFFIX_TOKENS = {
    "JR",
    "SR",
    "II",
    "III",
    "IV",
    "V",
}

ALIAS_ROLEISH_TOKENS = (
    ROLE_MARKER_TOKENS
    | LOCATION_MARKER_TOKENS
    | ORGANIZATION_MARKER_TOKENS
    | POLITICAL_MARKER_TOKENS
    | MILITARY_MARKER_TOKENS
    | {
        "COLLEGE",
        "CONSUL",
        "COURT",
        "COMMAND",
        "CHARGE",
        "CENTRAL",
        "DIPLOMATIC",
        "EMBASSY",
        "FOREIGN",
        "FORCES",
        "GOVERNOR",
        "GUARD",
        "GUARDS",
        "LEADER",
        "PERSONAL",
        "PERSONNEL",
        "POLICE",
        "PRIME",
        "SECURITY",
        "STAFF",
        "TRIBAL",
        "UNIT",
        "UNITS",
        "US",
        "BANK",
        "LIAISON",
        "OFFICER",
        "OFFICERS",
        "SON",
        "LAW",
        "WIFE",
        "HUSBAND",
        "BROTHER",
        "SISTER",
        "FATHER",
        "MOTHER",
    }
)

GENERIC_GROUP_HEAD_TOKENS = {
    "COLLEGE",
    "EMBASSY",
    "FORCE",
    "FORCES",
    "GUARD",
    "GUARDS",
    "OFFICER",
    "OFFICERS",
    "STAFF",
    "UNIT",
    "UNITS",
}

ROMAN_NUMERAL_VALUES = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16,
    "XVII": 17,
    "XVIII": 18,
    "XIX": 19,
    "XX": 20,
}

COUNTRY_VARIANT_TRAILING_TOKENS = {
    "AMERICA",
}

COUNTRY_GOVERNMENT_SUBENTITY_MARKERS = {
    "AGENCY",
    "ASSEMBLY",
    "AUTHORITIES",
    "AUTHORITY",
    "BUREAU",
    "CABINET",
    "COMMISSION",
    "COMMITTEE",
    "CONGRESS",
    "COURT",
    "DEPARTMENT",
    "EMBASSY",
    "IMMIGRATION",
    "MINISTRY",
    "OFFICE",
    "OFFICIAL",
    "OFFICIALS",
    "PARLIAMENT",
    "SENATE",
    "SECRETARY",
    "SECRETARIAT",
    "STAFF",
    "TRIBUNAL",
}

ORGANIZATION_SUBENTITY_MARKERS = {
    "AGENCY",
    "ASSEMBLY",
    "BUREAU",
    "COMMISSIONER",
    "COUNCIL",
    "DIRECTORATE",
    "MISSION",
    "OFFICE",
    "SECRETARIAT",
    "SECRETARY",
    "SERVICE",
}

GEOPOLITICAL_DEMONYM_TOKENS = {
    "AMERICANS",
    "BRITISH",
    "CAMBODIANS",
    "CHINESE",
    "FILIPINOS",
    "GERMANS",
    "INDONESIANS",
    "IRANIANS",
    "IRAQIS",
    "JAPANESE",
    "KOREANS",
    "LAOTIANS",
    "RUSSIANS",
    "THAIS",
    "VIETNAMESE",
}

ROLEISH_FRAGMENT_MARKERS = {
    "EMBASSY",
    "CONSULATE",
    "MISSION",
    "SECRETARIAT",
    "DEPARTMENT",
    "MINISTRY",
    "OFFICE",
    "AUTHORITIES",
    "AUTHORITY",
    "BUREAU",
    "SENATE",
    "PARLIAMENT",
    "CABINET",
}


def content_tokens(label: str) -> Set[str]:
    return {
        token
        for token in token_set(label)
        if token and len(token) >= 3 and token not in ALIAS_STOPWORDS
    }


def head_token(label: str) -> str:
    tokens = [token for token in normalize_label_basic(label).split() if token and token not in ALIAS_STOPWORDS]
    if not tokens:
        return ""
    return tokens[-1]


def phrase_initials(label: str) -> str:
    tokens = [token for token in normalize_label_basic(label).split() if token and token not in ALIAS_STOPWORDS]
    if len(tokens) <= 1:
        return ""
    return "".join(token[0] for token in tokens if token)


def embedded_acronym_token_count(label: str) -> int:
    tokens = [
        token
        for token in normalize_label_basic(label).split()
        if token and token not in ALIAS_STOPWORDS
    ]
    if len(tokens) <= 1:
        return 0
    return sum(1 for token in tokens if looks_like_acronym(token))


def is_subsequence(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    pos = 0
    for char in haystack:
        if pos < len(needle) and needle[pos] == char:
            pos += 1
    return pos == len(needle)


def acronym_phrase_support(acronym_label: str, phrase_label: str) -> bool:
    acronym = normalize_label_basic(acronym_label).replace(" ", "")
    initials = phrase_initials(phrase_label)
    if len(acronym) < 2 or len(initials) < 2:
        return False
    phrase_tokens = set(token_list(phrase_label))
    structural_tokens = {
        "FEDERAL",
        "FEDERATION",
        "GOVERNMENT",
        "KINGDOM",
        "MINISTRY",
        "REPUBLIC",
        "STATE",
        "STATES",
        "UNION",
    }
    if initials == acronym or initials.startswith(acronym) or acronym.startswith(initials):
        return True
    if phrase_tokens & structural_tokens:
        return set(initials) <= set(acronym)
    return False


def contains_embedded_acronym_token(label: str) -> bool:
    """
    Return True when a would-be "full-form" phrase still contains acronym-like
    tokens of its own, e.g. ``USDAO SGN`` or ``UN SECRETARY GENERAL``.

    Stage-1 acronym merging should stay conservative: it is meant to resolve
    acronym -> fully expanded phrase pairs such as ``FBI`` ->
    ``FEDERAL BUREAU OF INVESTIGATION``, not acronym -> other short-form phrase
    collisions where the initials happen to line up.
    """
    tokens = [
        token
        for token in normalize_label_basic(label).split()
        if token and token not in ALIAS_STOPWORDS
    ]
    if len(tokens) <= 1:
        return False
    return any(looks_like_acronym(token) for token in tokens)


def numeric_token_set(label: str) -> Set[str]:
    values = set(re.findall(r"\d+", normalize_label_basic(label)))
    for token in token_list(label):
        roman = ROMAN_NUMERAL_VALUES.get(token)
        if roman is not None:
            values.add(str(roman))
    return values


def surface_has_government_token(label: str) -> bool:
    tokens = set(token_list(label))
    return bool(tokens & {"GOVERNMENT", "REPUBLIC", "KINGDOM", "STATE", "STATES", "UNION"})


def contains_country_government_subentity_marker(label: str) -> bool:
    tokens = set(token_list(label))
    return bool(tokens & COUNTRY_GOVERNMENT_SUBENTITY_MARKERS)


def contains_organization_subentity_marker(label: str) -> bool:
    tokens = set(token_list(label))
    return bool(tokens & ORGANIZATION_SUBENTITY_MARKERS)


def token_has_marker_fragment(token: str, markers: Set[str]) -> bool:
    token = normalize_label_basic(token)
    if not token:
        return False
    return any(
        token == marker
        or token.startswith(marker)
        or token.endswith(marker)
        for marker in markers
    )


def looks_like_geopolitical_demonym_token(token: str) -> bool:
    token = normalize_label_basic(token)
    return token in GEOPOLITICAL_DEMONYM_TOKENS


def official_longform_surface_ok(label: str, family: str) -> bool:
    tokens = [token for token in token_list(label) if token and token not in ALIAS_STOPWORDS]
    if not tokens:
        return False
    if len(tokens) <= 2 and any(token_has_marker_fragment(token, ROLEISH_FRAGMENT_MARKERS) for token in tokens):
        return False
    if family in {"country", "government"}:
        if contains_country_government_subentity_marker(label):
            return False
        if looks_like_geopolitical_demonym_token(tokens[-1]):
            return False
        if any(looks_like_geopolitical_demonym_token(token) for token in tokens):
            if not (
                surface_has_government_token(label)
                or any(token in {"REPUBLIC", "KINGDOM", "FEDERATION", "UNION"} for token in tokens)
            ):
                return False
        return True
    if family in {"organization", "political_group"}:
        if contains_organization_subentity_marker(label):
            return False
        return True
    return True


def acronym_longform_surface_ok(
    left_label: str,
    right_label: str,
    left_family: str,
    right_family: str,
    *,
    left_is_acronym: bool,
    right_is_acronym: bool,
) -> bool:
    if left_is_acronym == right_is_acronym:
        return True
    longer_label = right_label if left_is_acronym else left_label
    longer_family = right_family if left_is_acronym else left_family
    return official_longform_surface_ok(longer_label, longer_family)


def alias_person_name_tokens(label: str) -> List[str]:
    tokens = [token for token in token_list(label) if token]
    while tokens and tokens[0] in ALIAS_HONORIFIC_TOKENS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in ALIAS_PERSON_SUFFIX_TOKENS:
        tokens = tokens[:-1]
    cleaned = [
        token
        for token in tokens
        if len(token) > 1
        and token not in ALIAS_HONORIFIC_TOKENS
        and token not in ALIAS_PERSON_SUFFIX_TOKENS
        and token not in ALIAS_ROLEISH_TOKENS
    ]
    return cleaned


def has_nonleading_person_title_token(label: str) -> bool:
    tokens = [token for token in token_list(label) if token]
    while tokens and tokens[0] in ALIAS_HONORIFIC_TOKENS:
        tokens = tokens[1:]
    return any(token in ALIAS_HONORIFIC_TOKENS for token in tokens)


def alias_person_name_like(label: str, source_types: Set[str]) -> bool:
    cleaned = alias_person_name_tokens(label)
    if len(cleaned) < 2 or len(cleaned) > 5:
        return False
    if has_nonleading_person_title_token(label):
        return False
    original_tokens = token_list(label)
    if original_tokens and original_tokens[-1] in GENERIC_GROUP_HEAD_TOKENS:
        return False
    normalized_types = {normalize_label_basic(value) for value in source_types if safe_str(value)}
    if "PERSON" not in normalized_types:
        return False
    if any(token in ALIAS_ROLEISH_TOKENS for token in original_tokens) and len(cleaned) < 3:
        return False
    if any(token.isdigit() for token in cleaned):
        return False
    return True


def conflicting_inner_person_tokens(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> bool:
    if len(left_tokens) <= 2 or len(right_tokens) <= 2:
        return False
    left_inner = [token for token in left_tokens[1:-1] if len(token) > 1]
    right_inner = [token for token in right_tokens[1:-1] if len(token) > 1]
    if not left_inner or not right_inner:
        return False
    left_set = set(left_inner)
    right_set = set(right_inner)
    if left_set <= right_set or right_set <= left_set:
        return False
    return True


def person_terminal_name_conflict(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> bool:
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    shared = set(left_tokens) & set(right_tokens)
    if len(shared) < 2:
        return False
    if left_tokens[0] != right_tokens[0]:
        return False
    return left_tokens[-1] != right_tokens[-1]


def modifier_conflict_reason(left_label: str, right_label: str) -> str:
    left_tokens = [token for token in token_list(left_label) if token and token not in ALIAS_STOPWORDS]
    right_tokens = [token for token in token_list(right_label) if token and token not in ALIAS_STOPWORDS]
    if not left_tokens or not right_tokens:
        return ""
    if left_tokens[-1] != right_tokens[-1]:
        return ""
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    left_unique = left_set - right_set
    right_unique = right_set - left_set
    if not left_unique or not right_unique:
        return ""
    if (left_unique | right_unique) & (ALIAS_ROLEISH_TOKENS | ALIAS_HONORIFIC_TOKENS):
        return "modifier_conflict"
    return ""


def canonical_key(value: str) -> str:
    return keyify(value)


def repair_base_entity_key_collisions(
    extraction_dir: Path,
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    repaired = profiles.copy()
    if "normalized_label" not in repaired.columns:
        raise KeyError("base_entity_profiles.csv must contain `normalized_label`.")

    repaired["normalized_label"] = repaired["normalized_label"].map(norm_space)
    if repaired["normalized_label"].duplicated().any():
        duplicates = (
            repaired.loc[repaired["normalized_label"].duplicated(keep=False), "normalized_label"]
            .value_counts()
            .head(10)
            .to_dict()
        )
        raise ValueError(
            "base_entity_profiles.csv contains duplicate normalized_label rows, so alias resolution "
            f"cannot build one base profile per spelling. Sample duplicates: {duplicates}"
        )

    old_keys = repaired.get("base_entity_key", pd.Series([""] * len(repaired), index=repaired.index)).map(safe_str)
    new_keys = repaired["normalized_label"].map(stable_base_entity_key)
    if new_keys.duplicated().any():
        duplicates = new_keys[new_keys.duplicated(keep=False)].value_counts().head(10).to_dict()
        raise ValueError(
            "Stable base entity key generation still produced duplicates, which should not happen. "
            f"Sample duplicates: {duplicates}"
        )

    changed_rows = int((old_keys != new_keys).sum())
    duplicate_rows = int(old_keys.duplicated(keep=False).sum()) if len(old_keys) else 0
    duplicate_keys = int(old_keys[old_keys.duplicated(keep=False)].nunique()) if duplicate_rows else 0

    if changed_rows:
        repaired["base_entity_key"] = new_keys
        repaired.to_csv(extraction_dir / "base_entity_profiles.csv", index=False)

        entities_path = extraction_dir / "entities.csv"
        if entities_path.exists():
            entities = pd.read_csv(entities_path, low_memory=False)
            if "normalized_label" in entities.columns:
                label_to_key = dict(zip(repaired["normalized_label"], repaired["base_entity_key"]))
                entities["normalized_label"] = entities["normalized_label"].map(norm_space)
                entities["base_entity_key"] = entities["normalized_label"].map(
                    lambda value: label_to_key.get(value, stable_base_entity_key(value) if safe_str(value) else "")
                )
                if "base_entity_label" in entities.columns:
                    entities["base_entity_label"] = entities["normalized_label"]
                entities.to_csv(entities_path, index=False)

    repair_manifest = {
        "changed_rows": changed_rows,
        "duplicate_rows_detected": duplicate_rows,
        "duplicate_keys_detected": duplicate_keys,
        "profiles_path": str(extraction_dir / "base_entity_profiles.csv"),
        "entities_path": str(extraction_dir / "entities.csv"),
    }
    return repaired, repair_manifest


def parse_multivalue_field(text: object) -> List[str]:
    value = safe_str(text)
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def choose_canonical_label(rows: pd.DataFrame) -> str:
    ranked: List[Tuple[Tuple[int, int, int, int, int, float], str]] = []
    for _, row in rows.iterrows():
        label = safe_str(row.get("normalized_label", ""))
        mention_count = int(row.get("mention_count", 0) or 0)
        doc_count = int(row.get("doc_count", 0) or 0)
        family = infer_structural_family(
            label,
            type_set_from_text(row.get("type_distribution", "")),
        )
        label_tokens = [token for token in token_list(label) if token]
        if family in {"person", "person_name"}:
            cleaned_tokens = alias_person_name_tokens(label)
            title_penalty = -1 if set(label_tokens) & (ALIAS_HONORIFIC_TOKENS | ALIAS_ROLEISH_TOKENS) else 0
            specificity = len(cleaned_tokens) * 10 + len(" ".join(cleaned_tokens))
            embedded_acronym_penalty = 0
        else:
            title_penalty = -1 if set(label_tokens) & ALIAS_HONORIFIC_TOKENS else 0
            specificity = len(content_tokens(label)) * 10 + len(label)
            embedded_acronym_penalty = -embedded_acronym_token_count(label)
        acronym_penalty = 0 if not looks_like_acronym(label) else -1
        ranked.append(
            (
                (
                    acronym_penalty,
                    embedded_acronym_penalty,
                    title_penalty,
                    mention_count,
                    doc_count,
                    specificity,
                ),
                label,
            )
        )
    ranked.sort(reverse=True)
    return ranked[0][1] if ranked else ""


def build_token_idf(profiles: pd.DataFrame) -> Dict[str, float]:
    token_df: Counter[str] = Counter()
    total = max(1, len(profiles))
    for label in profiles["normalized_label"].fillna("").astype(str):
        for token in content_tokens(label):
            token_df[token] += 1
    out: Dict[str, float] = {}
    for token, df in token_df.items():
        out[token] = math.log(1.0 + ((total - df + 0.5) / (df + 0.5)))
    return out


def build_acronym_expansion_map(profiles: pd.DataFrame) -> Dict[str, Set[str]]:
    expansions: Dict[str, Set[str]] = defaultdict(set)
    for _, row in profiles.iterrows():
        label = safe_str(row.get("normalized_label", ""))
        if not label or looks_like_acronym(label) or contains_embedded_acronym_token(label):
            continue
        acronym = phrase_initials(label)
        if len(acronym) >= 2:
            expansions[acronym].add(label)
    return expansions


def parse_surface_forms(value: object) -> List[str]:
    return [part.strip() for part in safe_str(value).split(";") if part.strip()]


def runtime_normalized_phrase(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", safe_str(value).casefold()))


def runtime_title_case_surface(text: str) -> str:
    tokens = norm_space(text).split()
    if not tokens:
        return ""
    rendered: List[str] = []
    for token in tokens:
        if re.fullmatch(r"[A-Z0-9#]+", token):
            if len(token) <= 4 and token.isupper():
                rendered.append(token)
            else:
                rendered.append(token.capitalize())
        else:
            rendered.append(token)
    return " ".join(rendered)


def runtime_entity_type(value: object) -> str:
    label = norm_space(value).upper()
    if label in {"PERSON", "PERSON_NAME"}:
        return "PERSON"
    if label in {
        "COUNTRY",
        "LOCATION",
        "CITY",
        "PROVINCE",
        "REGION",
        "STATE",
        "PLACE",
        "TERRITORY",
        "AREA",
        "GPE",
        "GEOGRAPHIC_FEATURE",
    }:
        return "LOCATION"
    if label in {"GOVERNMENT", "ORGANIZATION", "ORG", "MILITARY", "POLITICAL_PARTY", "PARTY", "AGENCY"}:
        return "ORGANIZATION"
    if label == "FACILITY":
        return "FACILITY"
    if label == "EVENT":
        return "EVENT"
    if label == "VEHICLE":
        return "VEHICLE"
    if label == "PRODUCT":
        return "PRODUCT"
    return "OTHER"


def choose_runtime_entity_type(member_rows: pd.DataFrame) -> str:
    counts: Counter[str] = Counter()
    for _, row in member_rows.iterrows():
        primary = runtime_entity_type(row.get("primary_type", ""))
        counts[primary] += max(1, int(row.get("mention_count", 0) or 0))
        for raw_type, raw_count in parse_type_distribution_counts(row.get("type_distribution", "")).items():
            counts[runtime_entity_type(raw_type)] += int(raw_count or 0)
    if not counts:
        return "OTHER"
    return counts.most_common(1)[0][0]


def surface_display_rank(surface: str, canonical_norm: str) -> Tuple[int, int, int, int]:
    normalized = runtime_normalized_phrase(surface)
    same_as_canonical = 1 if canonical_norm and normalized == canonical_norm else 0
    acronym_penalty = 0 if not looks_like_acronym(surface) else -1
    uppercase_penalty = 0 if not re.fullmatch(r"[A-Z0-9 .,'()/:-]+", surface) else -1
    return (
        same_as_canonical,
        acronym_penalty,
        uppercase_penalty,
        len(surface),
    )


def choose_runtime_display_form(canonical_label: str, member_rows: pd.DataFrame) -> str:
    canonical_norm = runtime_normalized_phrase(canonical_label)
    candidates: List[str] = []
    seen: Set[str] = set()
    for _, row in member_rows.iterrows():
        for surface in parse_surface_forms(row.get("surface_forms", "")) + [safe_str(row.get("normalized_label", ""))]:
            cleaned = norm_space(surface)
            if not cleaned:
                continue
            dedupe_key = cleaned.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(cleaned)
    matching = [value for value in candidates if runtime_normalized_phrase(value) == canonical_norm]
    ranked_pool = matching or candidates
    if ranked_pool:
        ranked_pool = sorted(
            ranked_pool,
            key=lambda value: surface_display_rank(value, canonical_norm),
            reverse=True,
        )
        best = norm_space(ranked_pool[0])
        if best:
            return best
    return runtime_title_case_surface(canonical_label) or norm_space(canonical_label)


def infer_corpus_resource_id(project_dir: Path) -> str:
    stem = norm_space(project_dir.name).casefold()
    if stem.startswith("viet_"):
        return "vietnam"
    if stem.startswith("canb_"):
        return "canberra"
    if stem.startswith("fisc_"):
        return "fisc"
    if stem.startswith("iran_"):
        return "iran"
    if stem.startswith("cib_"):
        return "cib"
    return stem.replace("_full_unredacted", "").replace("_full_redacted", "")


def export_corpus_entity_resource(
    *,
    project_dir: Path,
    profiles: pd.DataFrame,
    canonical_rows: Sequence[Mapping[str, object]],
) -> Path:
    corpus_id = infer_corpus_resource_id(project_dir)
    entities_payload: List[Dict[str, object]] = []

    for row in canonical_rows:
        canonical_entity_id = safe_str(row.get("canonical_entity_id", ""))
        canonical_label = safe_str(row.get("canonical_label", ""))
        member_keys = parse_multivalue_field(row.get("member_base_entity_keys", ""))
        if not canonical_entity_id or not canonical_label or not member_keys:
            continue
        member_rows = profiles[profiles["base_entity_key"].isin(member_keys)].copy()
        if member_rows.empty:
            continue

        canonical_display = choose_runtime_display_form(canonical_label, member_rows)
        canonical_norm = runtime_normalized_phrase(canonical_display) or runtime_normalized_phrase(canonical_label)
        entity_type = choose_runtime_entity_type(member_rows)

        forms_by_normalized: Dict[str, Dict[str, object]] = {}

        def register_form(
            *,
            surface: str,
            alias_type: str,
            confidence: float,
            base_entity_key: str,
            doc_count: int,
            mention_count: int,
        ) -> None:
            cleaned = norm_space(surface)
            normalized = runtime_normalized_phrase(cleaned)
            if not cleaned or not normalized:
                return
            provenance_item = {
                "derivation_method": "ALIAS_CLUSTER_EXPORT",
                "source_table": "base_entity_profiles.csv",
                "source_row_id": base_entity_key,
                "corpus_doc_id": corpus_id,
                "cluster_entity_id": canonical_entity_id,
                "mention_count": int(mention_count),
                "doc_count": int(doc_count),
            }
            entry = forms_by_normalized.get(normalized)
            rank = surface_display_rank(cleaned, canonical_norm)
            if entry is None:
                forms_by_normalized[normalized] = {
                    "surface_form": cleaned,
                    "normalized_form": normalized,
                    "alias_type": alias_type,
                    "confidence": float(confidence),
                    "review_state": "ACCEPTED",
                    "provenance": [provenance_item],
                    "_rank": rank,
                }
                return
            if rank > entry["_rank"]:
                entry["surface_form"] = cleaned
                entry["_rank"] = rank
            if alias_type == "CANONICAL":
                entry["alias_type"] = "CANONICAL"
            entry["confidence"] = max(float(entry["confidence"]), float(confidence))
            existing = {
                (
                    safe_str(item.get("source_table", "")),
                    safe_str(item.get("source_row_id", "")),
                    safe_str(item.get("corpus_doc_id", "")),
                )
                for item in entry["provenance"]
                if isinstance(item, Mapping)
            }
            key = (
                provenance_item["source_table"],
                provenance_item["source_row_id"],
                provenance_item["corpus_doc_id"],
            )
            if key not in existing:
                entry["provenance"].append(provenance_item)

        register_form(
            surface=canonical_display,
            alias_type="CANONICAL",
            confidence=0.99,
            base_entity_key=canonical_entity_id,
            doc_count=int(row.get("doc_count", 0) or 0),
            mention_count=int(row.get("mention_count", 0) or 0),
        )
        register_form(
            surface=canonical_label,
            alias_type="CANONICAL",
            confidence=0.96,
            base_entity_key=canonical_entity_id,
            doc_count=int(row.get("doc_count", 0) or 0),
            mention_count=int(row.get("mention_count", 0) or 0),
        )

        for _, member in member_rows.iterrows():
            base_entity_key = safe_str(member.get("base_entity_key", ""))
            mention_count = int(member.get("mention_count", 0) or 0)
            doc_count = int(member.get("doc_count", 0) or 0)
            normalized_label = safe_str(member.get("normalized_label", ""))
            if normalized_label:
                register_form(
                    surface=normalized_label,
                    alias_type="ACCEPTED_ALIAS",
                    confidence=0.94,
                    base_entity_key=base_entity_key,
                    doc_count=doc_count,
                    mention_count=mention_count,
                )
            for surface in parse_surface_forms(member.get("surface_forms", "")):
                register_form(
                    surface=surface,
                    alias_type="OBSERVED_SURFACE",
                    confidence=0.90,
                    base_entity_key=base_entity_key,
                    doc_count=doc_count,
                    mention_count=mention_count,
                )

        surface_forms = []
        for payload in sorted(
            forms_by_normalized.values(),
            key=lambda item: (
                0 if item.get("alias_type") == "CANONICAL" else 1,
                -float(item.get("confidence", 0.0) or 0.0),
                safe_str(item.get("surface_form", "")).casefold(),
            ),
        ):
            materialized = dict(payload)
            materialized.pop("_rank", None)
            surface_forms.append(materialized)

        if not surface_forms:
            continue
        entities_payload.append(
            {
                "canonical_entity_id": canonical_entity_id,
                "entity_type": entity_type,
                "canonical_display_form": canonical_display,
                "surface_forms": surface_forms,
            }
        )

    payload = {
        "schema_version": "matryoshka_corpus_entity_alias_resource_v1",
        "corpus_resource_id": f"{corpus_id}_entity_alias_runtime_v1",
        "corpus_id": corpus_id,
        "resource_version": "alias_runtime_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entities": entities_payload,
    }
    output_path = project_dir / "alias" / "corpus_entity_resource_v1.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def parse_neighbor_counts(value: object) -> Counter[str]:
    counts: Counter[str] = Counter()
    for part in parse_surface_forms(value):
        if ":" not in part:
            counts[part] += 1
            continue
        label, raw_count = part.rsplit(":", 1)
        try:
            counts[label.strip()] += int(float(raw_count.strip()))
        except Exception:
            counts[part.strip()] += 1
    return counts


def parse_type_distribution_counts(value: object) -> Counter[str]:
    counts: Counter[str] = Counter()
    for part in parse_surface_forms(value):
        if ":" in part:
            label, raw_count = part.split(":", 1)
            try:
                counts[label.strip()] += int(float(raw_count.strip()))
            except Exception:
                counts[label.strip()] += 1
        elif part:
            counts[part] += 1
    return counts


def top_counter_text(counter: Counter[str], limit: int = 12) -> str:
    return "; ".join(f"{label}:{count}" for label, count in counter.most_common(limit))


def summarize_contexts(values: List[str], limit: int = 3) -> str:
    seen: List[str] = []
    for raw in values:
        for part in safe_str(raw).split("||"):
            cleaned = norm_space(part)
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
            if len(seen) >= limit:
                return " || ".join(seen)
    return " || ".join(seen)


def weighted_overlap(left: Set[str], right: Set[str], token_idf: Mapping[str, float]) -> Tuple[float, float, float]:
    if not left or not right:
        return 0.0, 0.0, 0.0
    inter = left & right
    union = left | right

    def score(tokens: Iterable[str]) -> float:
        return sum(token_idf.get(token, 1.0) for token in tokens)

    inter_w = score(inter)
    union_w = score(union)
    left_w = score(left)
    right_w = score(right)
    return (
        inter_w / union_w if union_w else 0.0,
        inter_w / left_w if left_w else 0.0,
        inter_w / right_w if right_w else 0.0,
    )


def candidate_blocks(profiles: pd.DataFrame) -> Dict[Tuple[str, str], Set[Tuple[str, str]]]:
    blocks: Dict[Tuple[str, str], Set[Tuple[str, str]]] = defaultdict(set)
    for _, row in profiles.iterrows():
        label = safe_str(row.get("normalized_label", ""))
        key = safe_str(row.get("base_entity_key", ""))
        signature = (key, label)
        tokens = content_tokens(label)
        head = head_token(label)
        initials = phrase_initials(label)
        acronym = normalize_label_basic(label).replace(" ", "") if looks_like_acronym(label) else ""
        if head:
            blocks[("head", head)].add(signature)
        if initials and len(initials) >= 2:
            blocks[("initials", initials)].add(signature)
            blocks[("acronym", initials)].add(signature)
        if acronym and len(acronym) >= 2:
            blocks[("acronym", acronym)].add(signature)
        for token in tokens:
            blocks[("token", token)].add(signature)
    return blocks


def generate_candidate_pairs(profiles: pd.DataFrame) -> Set[Tuple[str, str]]:
    blocks = candidate_blocks(profiles)
    pairs: Set[Tuple[str, str]] = set()
    for _, members in blocks.items():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                left = ordered[i][0]
                right = ordered[j][0]
                if left != right:
                    pairs.add(tuple(sorted((left, right))))
    return pairs


def family_from_row(row: Mapping[str, object]) -> str:
    return infer_structural_family(
        safe_str(row.get("normalized_label", "")),
        type_set_from_text(row.get("type_distribution", "")),
    )


def acronym_fullform_match(left: str, right: str) -> bool:
    left_clean = normalize_label_basic(left)
    right_clean = normalize_label_basic(right)
    if not left_clean or not right_clean:
        return False
    if (
        looks_like_acronym(left_clean)
        and not contains_embedded_acronym_token(right_clean)
        and phrase_initials(right_clean) == left_clean.replace(" ", "")
    ):
        return True
    if (
        looks_like_acronym(right_clean)
        and not contains_embedded_acronym_token(left_clean)
        and phrase_initials(left_clean) == right_clean.replace(" ", "")
    ):
        return True
    return False


def official_country_variant_match(
    left_label: str,
    right_label: str,
    left_family: str,
    right_family: str,
) -> bool:
    if left_family != "country" or right_family != "country":
        return False
    left_tokens = [token for token in token_list(left_label) if token and token not in ALIAS_STOPWORDS]
    right_tokens = [token for token in token_list(right_label) if token and token not in ALIAS_STOPWORDS]
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    shorter_tokens, longer_tokens = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    if shorter_tokens != longer_tokens[: len(shorter_tokens)]:
        return False
    extra_tokens = longer_tokens[len(shorter_tokens) :]
    if not extra_tokens or len(extra_tokens) > 2:
        return False
    return all(token in COUNTRY_VARIANT_TRAILING_TOKENS for token in extra_tokens)


def global_head_review_rescue_eligible(
    left_row: Mapping[str, object],
    right_row: Mapping[str, object],
    *,
    country_variant: bool,
    explicit_acronym_variant: bool,
) -> bool:
    left_docs = int(left_row.get("doc_count", 0) or 0)
    right_docs = int(right_row.get("doc_count", 0) or 0)
    left_mentions = int(left_row.get("mention_count", 0) or 0)
    right_mentions = int(right_row.get("mention_count", 0) or 0)
    max_docs = max(left_docs, right_docs)
    min_docs = min(left_docs, right_docs)
    max_mentions = max(left_mentions, right_mentions)
    min_mentions = min(left_mentions, right_mentions)
    combined_docs = left_docs + right_docs
    combined_mentions = left_mentions + right_mentions
    materially_relevant = (
        (max_docs >= 10 and combined_docs >= 15 and combined_mentions >= 25)
        or (max_mentions >= 60 and combined_mentions >= 80)
    )
    if explicit_acronym_variant:
        return (
            (max_docs >= 12 and min_docs >= 2)
            or (max_mentions >= 40 and min_mentions >= 3)
            or (materially_relevant and max_docs >= 10 and (min_docs >= 1 or min_mentions >= 2))
        )
    if country_variant:
        return (
            (max_docs >= 20 and min_docs >= 10)
            or (materially_relevant and max_docs >= 10 and min_mentions >= 3)
        )
    return (
        (max_docs >= 40 and min_docs >= 10)
        or (max_mentions >= 80 and min_mentions >= 20)
        or (materially_relevant and max_docs >= 12 and (min_docs >= 2 or min_mentions >= 3))
    )


def explicit_acronym_fullform_review_eligible(
    left_row: Mapping[str, object],
    right_row: Mapping[str, object],
    local: "PairScore",
    *,
    left_is_acronym: bool,
    right_is_acronym: bool,
) -> bool:
    if not (left_is_acronym ^ right_is_acronym):
        return False
    left_label = safe_str(left_row.get("normalized_label", ""))
    right_label = safe_str(right_row.get("normalized_label", ""))
    if not acronym_fullform_match(left_label, right_label):
        return False
    if not local.exact_initials_match:
        return False
    left_family = family_from_row(left_row)
    right_family = family_from_row(right_row)
    if not review_compatible_family(left_family, right_family, acronym_candidate=True):
        return False
    if not acronym_longform_surface_ok(
        left_label,
        right_label,
        left_family,
        right_family,
        left_is_acronym=left_is_acronym,
        right_is_acronym=right_is_acronym,
    ):
        return False
    semantic_support = max(
        local.cooccurrence_similarity,
        local.relation_neighbor_similarity,
        local.context_similarity,
    )
    min_docs = min(local.doc_count_left, local.doc_count_right)
    min_mentions = min(local.mention_count_left, local.mention_count_right)
    max_docs = max(local.doc_count_left, local.doc_count_right)
    if max_docs < 5:
        return False
    if min_docs < 2 and min_mentions < 3:
        return False
    if (
        semantic_support < 0.03
        and max(local.left_containment, local.right_containment) < 0.95
        and local.sequence_ratio < 0.18
    ):
        return False
    return True


def projected_head_review_relevance(
    left_row: Mapping[str, object],
    right_row: Mapping[str, object],
) -> tuple[int, int, int, bool]:
    left_docs = int(left_row.get("doc_count", 0) or 0)
    right_docs = int(right_row.get("doc_count", 0) or 0)
    left_mentions = int(left_row.get("mention_count", 0) or 0)
    right_mentions = int(right_row.get("mention_count", 0) or 0)
    combined_docs = left_docs + right_docs
    combined_mentions = left_mentions + right_mentions
    max_docs = max(left_docs, right_docs)
    max_mentions = max(left_mentions, right_mentions)
    materially_relevant = (
        (max_docs >= 10 and combined_docs >= 15 and combined_mentions >= 25)
        or (max_mentions >= 60 and combined_mentions >= 80)
    )
    return combined_docs, combined_mentions, max(max_docs, max_mentions), materially_relevant


def resolve_head_review_top_k(
    requested_top_k: int,
    profile_count: int,
) -> int:
    if requested_top_k > 0:
        return min(requested_top_k, max(profile_count, 0))
    if profile_count <= 0:
        return 0
    if profile_count <= 400:
        return profile_count
    sqrt_frontier = math.ceil(math.sqrt(profile_count) * 1.75)
    proportional_frontier = math.ceil(profile_count * 0.005)
    return min(max(50, sqrt_frontier, proportional_frontier), 400, profile_count)


HEAD_REVIEW_EXHAUSTIVE_PROFILE_LIMIT = 128
HEAD_REVIEW_MIN_PER_ANCHOR_CAP = 12
HEAD_REVIEW_MAX_PER_ANCHOR_CAP = 24


def resolve_head_review_per_anchor_cap(
    requested_cap: int,
    selected_profile_count: int,
) -> int:
    if requested_cap > 0:
        return requested_cap
    if selected_profile_count <= HEAD_REVIEW_EXHAUSTIVE_PROFILE_LIMIT:
        return 0
    return max(
        HEAD_REVIEW_MIN_PER_ANCHOR_CAP,
        min(
            HEAD_REVIEW_MAX_PER_ANCHOR_CAP,
            math.ceil(math.log2(selected_profile_count + 1) * 2.4),
        ),
    )


def resolve_head_review_total_cap(
    requested_cap: int,
    selected_profile_count: int,
) -> int:
    if requested_cap > 0:
        return requested_cap
    if selected_profile_count <= 0:
        return 0
    if selected_profile_count <= HEAD_REVIEW_EXHAUSTIVE_PROFILE_LIMIT:
        return max(40, min(300, selected_profile_count * 4))
    return min(500, max(120, math.ceil(selected_profile_count * 0.5)))


def head_review_support_signal_count(
    local: "PairScore",
    *,
    acronym_candidate: bool,
    country_variant_candidate: bool,
    location_candidate: bool,
) -> int:
    lexical_support = max(local.token_jaccard, local.weighted_jaccard, local.sequence_ratio)
    semantic_support = max(
        local.cooccurrence_similarity,
        local.relation_neighbor_similarity,
        local.context_similarity,
    )
    signals = 0
    if local.exact_initials_match:
        signals += 1
    if local.person_name_core_match:
        signals += 1
    if local.head_match:
        signals += 1
    if lexical_support >= 0.18:
        signals += 1
    if semantic_support >= 0.10:
        signals += 1
    if local.cooccurrence_similarity >= 0.14:
        signals += 1
    if local.relation_neighbor_similarity >= 0.12:
        signals += 1
    if local.context_similarity >= 0.08:
        signals += 1
    if max(local.left_containment, local.right_containment) >= 0.95:
        signals += 1
    if acronym_candidate and semantic_support >= 0.08:
        signals += 1
    if country_variant_candidate and local.type_score >= 0.95:
        signals += 1
    if location_candidate and semantic_support >= 0.08:
        signals += 1
    return signals


def same_head_noun_match(left: str, right: str) -> bool:
    head_left = head_token(left)
    head_right = head_token(right)
    return bool(head_left and head_right and head_left == head_right)


def person_name_signature(label: str) -> List[str]:
    tokens = [token for token in normalize_label_basic(label).split() if token and token not in ALIAS_STOPWORDS]
    if not tokens:
        return []
    return [token for token in tokens if len(token) > 1]


def person_name_core_match(left_row: Mapping[str, object], right_row: Mapping[str, object]) -> bool:
    left_types = type_set_from_text(left_row.get("type_distribution", "")) or {
        normalize_label_basic(left_row.get("primary_type", ""))
    }
    right_types = type_set_from_text(right_row.get("type_distribution", "")) or {
        normalize_label_basic(right_row.get("primary_type", ""))
    }
    left_label = safe_str(left_row.get("normalized_label", ""))
    right_label = safe_str(right_row.get("normalized_label", ""))
    if not alias_person_name_like(left_label, left_types):
        return False
    if not alias_person_name_like(right_label, right_types):
        return False
    left_tokens = alias_person_name_tokens(left_label)
    right_tokens = alias_person_name_tokens(right_label)
    shared = set(left_tokens) & set(right_tokens)
    if len(shared) < 2:
        return False
    if left_tokens[0] != right_tokens[0]:
        return False
    if left_tokens[-1] != right_tokens[-1]:
        return False
    if conflicting_inner_person_tokens(left_tokens, right_tokens):
        return False
    return True


def nonperson_modifier_conflict_reason(
    left_label: str,
    right_label: str,
    left_family: str,
    right_family: str,
) -> str:
    person_families = {"person", "person_name"}
    if left_family in person_families or right_family in person_families:
        return ""
    left_tokens = [token for token in token_list(left_label) if token and token not in ALIAS_STOPWORDS]
    right_tokens = [token for token in token_list(right_label) if token and token not in ALIAS_STOPWORDS]
    if not left_tokens or not right_tokens:
        return ""
    if left_tokens[-1] != right_tokens[-1]:
        return ""
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    left_unique = left_set - right_set
    right_unique = right_set - left_set
    if left_tokens[-1] in GENERIC_GROUP_HEAD_TOKENS and (left_unique or right_unique):
        if (left_unique | right_unique) & ALIAS_ROLEISH_TOKENS:
            return "nonperson_modifier_conflict"
    if left_unique and right_unique:
        return "nonperson_modifier_conflict"
    return ""


def hard_conflict_reason(left_row: Mapping[str, object], right_row: Mapping[str, object]) -> str:
    left_label = safe_str(left_row.get("normalized_label", ""))
    right_label = safe_str(right_row.get("normalized_label", ""))
    left_types = type_set_from_text(left_row.get("type_distribution", "")) or {
        normalize_label_basic(left_row.get("primary_type", ""))
    }
    right_types = type_set_from_text(right_row.get("type_distribution", "")) or {
        normalize_label_basic(right_row.get("primary_type", ""))
    }
    left_numbers = numeric_token_set(left_label)
    right_numbers = numeric_token_set(right_label)
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return "numeric_conflict"
    modifier_conflict = modifier_conflict_reason(left_label, right_label)
    if modifier_conflict:
        return modifier_conflict
    left_family = family_from_row(left_row)
    right_family = family_from_row(right_row)
    nonperson_conflict = nonperson_modifier_conflict_reason(
        left_label,
        right_label,
        left_family,
        right_family,
    )
    if nonperson_conflict:
        return nonperson_conflict
    if left_family == "person" and right_family not in {"person", "person_name"}:
        return "person_vs_nonperson"
    if right_family == "person" and left_family not in {"person", "person_name"}:
        return "person_vs_nonperson"
    if left_family in {"person", "person_name"} and right_family in {"person", "person_name"}:
        left_person_like = alias_person_name_like(left_label, left_types)
        right_person_like = alias_person_name_like(right_label, right_types)
        if left_person_like and right_person_like:
            left_person_tokens = alias_person_name_tokens(left_label)
            right_person_tokens = alias_person_name_tokens(right_label)
            if person_terminal_name_conflict(left_person_tokens, right_person_tokens):
                return "person_terminal_name_conflict"
            if (
                left_person_tokens
                and right_person_tokens
                and left_person_tokens[0] == right_person_tokens[0]
                and left_person_tokens[-1] == right_person_tokens[-1]
                and conflicting_inner_person_tokens(left_person_tokens, right_person_tokens)
            ):
                return "person_inner_name_conflict"
        if not left_person_like or not right_person_like:
            left_roleish = bool(set(token_list(left_label)) & ALIAS_ROLEISH_TOKENS)
            right_roleish = bool(set(token_list(right_label)) & ALIAS_ROLEISH_TOKENS)
            left_nonleading_title = has_nonleading_person_title_token(left_label)
            right_nonleading_title = has_nonleading_person_title_token(right_label)
            if left_roleish or right_roleish or left_nonleading_title or right_nonleading_title:
                return "role_label_conflict"
    if acronym_fullform_match(left_label, right_label):
        return ""
    if looks_like_acronym(left_label) and looks_like_acronym(right_label) and left_label != right_label:
        return ""
    left_head = head_token(left_label)
    right_head = head_token(right_label)
    if left_head and right_head and left_head != right_head:
        if len(content_tokens(left_label) & content_tokens(right_label)) == 0:
            return "different_head_noun"
    return ""


@dataclass
class PairScore:
    base_entity_key_left: str
    base_entity_key_right: str
    normalized_label_left: str
    normalized_label_right: str
    primary_type_left: str
    primary_type_right: str
    mention_count_left: int
    mention_count_right: int
    doc_count_left: int
    doc_count_right: int
    type_score: float
    type_reason: str
    exact_initials_match: bool
    person_name_core_match: bool
    head_match: bool
    token_jaccard: float
    weighted_jaccard: float
    left_containment: float
    right_containment: float
    sequence_ratio: float
    context_similarity: float
    cooccurrence_similarity: float
    relation_neighbor_similarity: float
    acronym_expansion_count: int
    score: float
    decision: str
    decision_source: str
    hard_conflict: str


def local_pair_score(
    left_row: Mapping[str, object],
    right_row: Mapping[str, object],
    token_idf: Mapping[str, float],
    acronym_expansion_map: Mapping[str, Set[str]] | None = None,
) -> PairScore:
    left_label = safe_str(left_row.get("normalized_label", ""))
    right_label = safe_str(right_row.get("normalized_label", ""))
    type_left = type_set_from_text(left_row.get("type_distribution", ""))
    type_right = type_set_from_text(right_row.get("type_distribution", ""))
    left_person_like = alias_person_name_like(left_label, type_left)
    right_person_like = alias_person_name_like(right_label, type_right)
    if left_person_like and right_person_like:
        left_token_list = alias_person_name_tokens(left_label)
        right_token_list = alias_person_name_tokens(right_label)
        left_tokens = set(left_token_list)
        right_tokens = set(right_token_list)
        head_match = bool(
            left_token_list
            and right_token_list
            and left_token_list[-1] == right_token_list[-1]
        )
    else:
        left_tokens = content_tokens(left_label)
        right_tokens = content_tokens(right_label)
        head_match = same_head_noun_match(left_label, right_label)
    type_score, type_reason = type_compatibility(type_left, type_right)
    exact_initials = acronym_fullform_match(left_label, right_label)
    person_core = person_name_core_match(left_row, right_row)
    token_j = jaccard(left_tokens, right_tokens)
    weighted_j, left_containment, right_containment = weighted_overlap(left_tokens, right_tokens, token_idf)
    seq = sequence_ratio(left_label, right_label)
    context_sim = jaccard(
        content_tokens(safe_str(left_row.get("representative_contexts", ""))),
        content_tokens(safe_str(right_row.get("representative_contexts", ""))),
    )
    cooccurrence_sim = cooccurrence_jaccard(
        left_row.get("cooccurring_entities", ""),
        right_row.get("cooccurring_entities", ""),
    )
    relation_neighbor_sim = cooccurrence_jaccard(
        left_row.get("relation_neighbor_entities", ""),
        right_row.get("relation_neighbor_entities", ""),
    )
    hard_conflict = hard_conflict_reason(left_row, right_row)
    left_mention_count = int(float(left_row.get("mention_count", 0) or 0))
    right_mention_count = int(float(right_row.get("mention_count", 0) or 0))
    left_doc_count = int(float(left_row.get("doc_count", 0) or 0))
    right_doc_count = int(float(right_row.get("doc_count", 0) or 0))

    score = 0.0
    score += 0.25 * type_score
    score += 0.20 * weighted_j
    score += 0.10 * token_j
    score += 0.10 * max(left_containment, right_containment)
    score += 0.08 * seq
    score += 0.10 * context_sim
    score += 0.07 * cooccurrence_sim
    score += 0.05 * relation_neighbor_sim
    if exact_initials:
        score += 0.15
    if person_core:
        score += 0.18
    if head_match:
        score += 0.08
    if hard_conflict:
        score -= 0.35
    score = max(0.0, min(1.0, score))

    acronym_label = ""
    acronym_is_left = False
    if looks_like_acronym(left_label):
        acronym_label = normalize_label_basic(left_label).replace(" ", "")
        acronym_is_left = True
    elif looks_like_acronym(right_label):
        acronym_label = normalize_label_basic(right_label).replace(" ", "")
    acronym_len = len(acronym_label)
    acronym_expansion_count = 0
    if acronym_label and acronym_expansion_map is not None:
        acronym_expansion_count = len(acronym_expansion_map.get(acronym_label, set()))
    acronym_supported = False
    if exact_initials and type_reason == "shared_type":
        if acronym_len <= 2:
            acronym_supported = (
                score >= 0.43
                and (
                    cooccurrence_sim >= 0.18
                    or context_sim >= 0.10
                    or relation_neighbor_sim >= 0.10
                )
            )
        else:
            acronym_supported = (
                score >= 0.42
                and (
                    cooccurrence_sim >= 0.08
                    or context_sim >= 0.08
                    or relation_neighbor_sim >= 0.05
                )
            )
            # Context-only acronym matches are often one-off branch/local names
            # rather than durable corpus-wide aliases. Keep them only when the
            # pair has enough corpus frequency to justify the softer evidence.
            if (
                acronym_supported
                and cooccurrence_sim < 0.08
                and relation_neighbor_sim < 0.05
                and context_sim >= 0.08
                and max(left_mention_count, right_mention_count) < 10
                and max(left_doc_count, right_doc_count) < 10
            ):
                acronym_supported = False
            non_acronym_doc_count = right_doc_count if acronym_is_left else left_doc_count
            non_acronym_mention_count = right_mention_count if acronym_is_left else left_mention_count
            if (
                acronym_supported
                and non_acronym_doc_count <= 1
                and non_acronym_mention_count <= 1
                and (
                    cooccurrence_sim < 0.20
                    or score < 0.44
                )
                and relation_neighbor_sim < 0.10
            ):
                acronym_supported = False
        if acronym_supported and acronym_expansion_count > 1:
            acronym_supported = False

    if hard_conflict and score < 0.70:
        decision = "separate"
        source = "local_hard_conflict"
    elif acronym_supported and score >= 0.35:
        decision = "same"
        source = "local_acronym_fullform"
    elif score >= 0.86:
        decision = "same"
        source = "local_strong"
    elif person_core and score >= 0.85:
        decision = "same"
        source = "local_person_name_core"
    elif score <= 0.48:
        decision = "separate"
        source = "local_weak"
    else:
        decision = "uncertain"
        source = "needs_review"

    return PairScore(
        base_entity_key_left=safe_str(left_row.get("base_entity_key", "")),
        base_entity_key_right=safe_str(right_row.get("base_entity_key", "")),
        normalized_label_left=left_label,
        normalized_label_right=right_label,
        primary_type_left=safe_str(left_row.get("primary_type", "")),
        primary_type_right=safe_str(right_row.get("primary_type", "")),
        mention_count_left=int(left_row.get("mention_count", 0) or 0),
        mention_count_right=int(right_row.get("mention_count", 0) or 0),
        doc_count_left=int(left_row.get("doc_count", 0) or 0),
        doc_count_right=int(right_row.get("doc_count", 0) or 0),
        type_score=round(type_score, 6),
        type_reason=type_reason,
        exact_initials_match=bool(exact_initials),
        person_name_core_match=bool(person_core),
        head_match=bool(head_match),
        token_jaccard=round(token_j, 6),
        weighted_jaccard=round(weighted_j, 6),
        left_containment=round(left_containment, 6),
        right_containment=round(right_containment, 6),
        sequence_ratio=round(seq, 6),
        context_similarity=round(context_sim, 6),
        cooccurrence_similarity=round(cooccurrence_sim, 6),
        relation_neighbor_similarity=round(relation_neighbor_sim, 6),
        acronym_expansion_count=int(acronym_expansion_count),
        score=round(score, 6),
        decision=decision,
        decision_source=source,
        hard_conflict=hard_conflict,
    )


def cluster_members_from_union_find(
    uf: "UnionFind",
    keys: Iterable[str],
) -> Dict[str, List[str]]:
    cluster_members: Dict[str, List[str]] = defaultdict(list)
    for key in keys:
        cluster_members[uf.find(key)].append(key)
    return cluster_members


def aggregate_cluster_profiles(
    cluster_members: Mapping[str, Sequence[str]],
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    profile_lookup = {
        safe_str(row["base_entity_key"]): row
        for row in profiles.to_dict(orient="records")
    }
    aggregate_rows: List[Dict[str, object]] = []
    for root_key, members in cluster_members.items():
        member_rows = [
            profile_lookup[key]
            for key in members
            if key in profile_lookup
        ]
        if not member_rows:
            continue
        frame = pd.DataFrame(member_rows)
        canonical_label = choose_canonical_label(frame)
        surface_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        cooccur_counter: Counter[str] = Counter()
        relation_counter: Counter[str] = Counter()
        contexts: List[str] = []
        generic_votes = 0
        acronym_votes = 0
        for member in member_rows:
            mention_count = int(float(member.get("mention_count", 0) or 0))
            for form in parse_surface_forms(member.get("surface_forms", "")):
                surface_counter[form] += max(1, mention_count)
            type_counter.update(parse_type_distribution_counts(member.get("type_distribution", "")))
            cooccur_counter.update(parse_neighbor_counts(member.get("cooccurring_entities", "")))
            relation_counter.update(parse_neighbor_counts(member.get("relation_neighbor_entities", "")))
            contexts.append(safe_str(member.get("representative_contexts", "")))
            if bool(member.get("is_generic", False)):
                generic_votes += 1
            if bool(member.get("looks_like_acronym", False)) or looks_like_acronym(
                safe_str(member.get("normalized_label", ""))
            ):
                acronym_votes += 1
        primary_type = type_counter.most_common(1)[0][0] if type_counter else ""
        aggregate_rows.append(
            {
                "base_entity_key": safe_str(root_key),
                "cluster_root_key": safe_str(root_key),
                "normalized_label": canonical_label,
                "mention_count": int(frame["mention_count"].fillna(0).astype(int).sum()),
                "doc_count": int(frame["doc_count"].fillna(0).astype(int).sum()),
                "paragraph_count": int(frame.get("paragraph_count", pd.Series(dtype=float)).fillna(0).astype(int).sum())
                if "paragraph_count" in frame.columns
                else 0,
                "primary_type": primary_type,
                "type_distribution": top_counter_text(type_counter, limit=12),
                "surface_forms": "; ".join(label for label, _ in surface_counter.most_common(30)),
                "representative_contexts": summarize_contexts(contexts, limit=5),
                "cooccurring_entities": top_counter_text(cooccur_counter, limit=24),
                "relation_neighbor_entities": top_counter_text(relation_counter, limit=24),
                "profile_summary": canonical_label,
                "is_generic": generic_votes >= max(1, len(member_rows) // 2),
                "looks_like_acronym": acronym_votes >= 1,
                "member_count": int(len(member_rows)),
                "member_labels": "; ".join(frame["normalized_label"].astype(str).tolist()),
                "member_base_entity_keys": "; ".join(frame["base_entity_key"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(aggregate_rows)


def select_head_review_profiles(
    canonical_profiles: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    effective_top_k = resolve_head_review_top_k(int(top_k), int(len(canonical_profiles)))
    ordered = canonical_profiles.sort_values(
        ["doc_count", "mention_count", "member_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    if effective_top_k <= 0 or len(ordered) <= effective_top_k:
        return ordered

    selected = ordered.head(effective_top_k).copy()
    selected_keys = set(selected["base_entity_key"].astype(str).tolist())
    ordered_rows = [row for _, row in ordered.iterrows()]
    row_lookup = {
        safe_str(row.get("base_entity_key", "")): row
        for row in ordered_rows
    }
    acronym_rows: Dict[str, Set[str]] = defaultdict(set)
    initials_rows: Dict[str, Set[str]] = defaultdict(set)
    prefix_variant_rows: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)

    for row in ordered_rows:
        key = safe_str(row.get("base_entity_key", ""))
        label = safe_str(row.get("normalized_label", ""))
        if not key or not label:
            continue
        if looks_like_acronym(label):
            acronym_rows[normalize_label_basic(label).replace(" ", "")].add(key)
        else:
            initials = phrase_initials(label)
            if len(initials) >= 2:
                initials_rows[initials].add(key)
        row_family = family_from_row(row)
        if row_family == "country":
            row_tokens = [token for token in token_list(label) if token and token not in ALIAS_STOPWORDS]
            if len(row_tokens) >= 3:
                for prefix_len in range(2, len(row_tokens)):
                    extra_tokens = row_tokens[prefix_len:]
                    if not extra_tokens or len(extra_tokens) > 2:
                        continue
                    if all(token in COUNTRY_VARIANT_TRAILING_TOKENS for token in extra_tokens):
                        prefix_variant_rows[tuple(row_tokens[:prefix_len])].add(key)

    extra_keys: Set[str] = set()
    for _, anchor in selected.iterrows():
        anchor_key = safe_str(anchor.get("base_entity_key", ""))
        anchor_label = safe_str(anchor.get("normalized_label", ""))
        anchor_family = family_from_row(anchor)
        if not anchor_key or not anchor_label:
            continue

        if looks_like_acronym(anchor_label):
            anchor_acronym = normalize_label_basic(anchor_label).replace(" ", "")
            candidate_keys = {
                key
                for initials, keys in initials_rows.items()
                if initials == anchor_acronym
                or initials.startswith(anchor_acronym)
                or anchor_acronym.startswith(initials)
                for key in keys
            }
        else:
            anchor_initials = phrase_initials(anchor_label)
            candidate_keys = {
                key
                for acronym, keys in acronym_rows.items()
                if acronym == anchor_initials
                or acronym.startswith(anchor_initials)
                or anchor_initials.startswith(acronym)
                for key in keys
            } if len(anchor_initials) >= 2 else set()

        for candidate_key in candidate_keys:
            if candidate_key in selected_keys or candidate_key == anchor_key:
                continue
            candidate_row = row_lookup.get(candidate_key)
            if candidate_row is None:
                continue
            candidate_label = safe_str(candidate_row.get("normalized_label", ""))
            if not candidate_label:
                continue
            candidate_family = family_from_row(candidate_row)
            if not review_compatible_family(
                anchor_family,
                candidate_family,
                acronym_candidate=True,
            ):
                continue
            if acronym_fullform_match(anchor_label, candidate_label):
                if acronym_longform_surface_ok(
                    anchor_label,
                    candidate_label,
                    anchor_family,
                    candidate_family,
                    left_is_acronym=looks_like_acronym(anchor_label),
                    right_is_acronym=looks_like_acronym(candidate_label),
                ):
                    extra_keys.add(candidate_key)
                continue
            if looks_like_acronym(anchor_label) ^ looks_like_acronym(candidate_label):
                if looks_like_acronym(anchor_label):
                    include = acronym_phrase_support(anchor_label, candidate_label)
                else:
                    include = acronym_phrase_support(candidate_label, anchor_label)
                if include and acronym_longform_surface_ok(
                    anchor_label,
                    candidate_label,
                    anchor_family,
                    candidate_family,
                    left_is_acronym=looks_like_acronym(anchor_label),
                    right_is_acronym=looks_like_acronym(candidate_label),
                ):
                    extra_keys.add(candidate_key)

    for row in ordered_rows:
        row_key = safe_str(row.get("base_entity_key", ""))
        row_label = safe_str(row.get("normalized_label", ""))
        row_family = family_from_row(row)
        if not row_key or not row_label:
            continue
        row_tokens = [token for token in token_list(row_label) if token and token not in ALIAS_STOPWORDS]
        rescue_keys: Set[str] = set()
        if looks_like_acronym(row_label):
            row_acronym = normalize_label_basic(row_label).replace(" ", "")
            rescue_keys.update(
                {
                    key
                    for initials, keys in initials_rows.items()
                    if initials == row_acronym
                    or initials.startswith(row_acronym)
                    or row_acronym.startswith(initials)
                    for key in keys
                }
            )
        elif len(row_tokens) >= 2:
            row_initials = phrase_initials(row_label)
            if len(row_initials) >= 2:
                rescue_keys.update(
                    {
                        key
                        for acronym, keys in acronym_rows.items()
                        if acronym == row_initials
                        or acronym.startswith(row_initials)
                        or row_initials.startswith(acronym)
                        for key in keys
                    }
                )
            rescue_keys.update(prefix_variant_rows.get(tuple(row_tokens), set()))
        for rescue_key in rescue_keys:
            if rescue_key == row_key:
                continue
            rescue_row = row_lookup.get(rescue_key)
            if rescue_row is None:
                continue
            rescue_label = safe_str(rescue_row.get("normalized_label", ""))
            rescue_family = family_from_row(rescue_row)
            if not rescue_label:
                continue
            if not review_compatible_family(
                row_family,
                rescue_family,
                acronym_candidate=True,
            ):
                continue
            country_variant = official_country_variant_match(
                row_label,
                rescue_label,
                row_family,
                rescue_family,
            )
            acronym_variant = (
                acronym_fullform_match(row_label, rescue_label)
                and acronym_longform_surface_ok(
                    row_label,
                    rescue_label,
                    row_family,
                    rescue_family,
                    left_is_acronym=looks_like_acronym(row_label),
                    right_is_acronym=looks_like_acronym(rescue_label),
                )
            )
            if (
                acronym_variant
                or country_variant
            ) and global_head_review_rescue_eligible(
                row,
                rescue_row,
                country_variant=country_variant,
                explicit_acronym_variant=acronym_variant,
            ):
                extra_keys.add(row_key)
                extra_keys.add(rescue_key)

    if extra_keys:
        extra_rows = ordered[ordered["base_entity_key"].astype(str).isin(extra_keys)].copy()
        selected = pd.concat([selected, extra_rows], ignore_index=True)
    selected = selected.drop_duplicates(subset=["base_entity_key"], keep="first").reset_index(drop=True)
    return selected


def same_or_compatible_family(left_family: str, right_family: str) -> bool:
    if left_family == right_family:
        return True
    if {left_family, right_family} <= {"country", "government"}:
        return True
    return False


def review_compatible_family(
    left_family: str,
    right_family: str,
    *,
    acronym_candidate: bool,
) -> bool:
    if same_or_compatible_family(left_family, right_family):
        return True
    family_pair = frozenset({left_family, right_family})
    if acronym_candidate and family_pair in {
        frozenset({"organization", "government"}),
        frozenset({"organization", "political_group"}),
        frozenset({"government", "political_group"}),
    }:
        return True
    return False


def review_acronym_like(row: Mapping[str, object], label: str) -> bool:
    normalized = normalize_label_basic(label)
    if not looks_like_acronym(normalized):
        return False
    family = family_from_row(row)
    primary_type = normalize_label_basic(row.get("primary_type", ""))
    country_like = family in {"country", "government"} or primary_type in {"COUNTRY", "GOVERNMENT"}
    if family in {"person", "person_name"} or primary_type == "PERSON":
        return False
    if normalized in ALIAS_HONORIFIC_TOKENS:
        return False
    if normalized in ALIAS_ROLEISH_TOKENS or normalized in GENERIC_GROUP_HEAD_TOKENS:
        if not country_like:
            return False
    if bool(row.get("is_generic", False)) and len(normalized) >= 4:
        return False
    return True


def geographic_specialization_candidate(
    left_row: Mapping[str, object],
    right_row: Mapping[str, object],
    local: PairScore,
) -> bool:
    left_primary = normalize_label_basic(left_row.get("primary_type", ""))
    right_primary = normalize_label_basic(right_row.get("primary_type", ""))
    if left_primary != "LOCATION" or right_primary != "LOCATION":
        return False
    if max(local.doc_count_left, local.doc_count_right) < 20:
        return False
    left_tokens = [token for token in token_list(safe_str(left_row.get("normalized_label", ""))) if token]
    right_tokens = [token for token in token_list(safe_str(right_row.get("normalized_label", ""))) if token]
    if not left_tokens or not right_tokens:
        return False
    shorter_tokens, longer_tokens = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    # Keep only appositional/location-expansion cases such as
    # ``WASHINGTON`` -> ``WASHINGTON D C``. Directional or geopolitical
    # specializations like ``WEST BERLIN`` / ``BERLIN`` should not be treated
    # as aliases.
    if longer_tokens[: len(shorter_tokens)] != shorter_tokens:
        return False
    if len(longer_tokens) == len(shorter_tokens):
        return False
    if max(local.left_containment, local.right_containment) < 0.95:
        return False
    if local.sequence_ratio < 0.55:
        return False
    semantic_support = max(
        local.cooccurrence_similarity,
        local.relation_neighbor_similarity,
        local.context_similarity,
    )
    if semantic_support < 0.08:
        return False
    return True


def classify_head_review_candidate(
    left_row: Mapping[str, object],
    right_row: Mapping[str, object],
    local: PairScore,
) -> tuple[str, float, str]:
    left_label = safe_str(left_row.get("normalized_label", ""))
    right_label = safe_str(right_row.get("normalized_label", ""))
    left_family = family_from_row(left_row)
    right_family = family_from_row(right_row)
    left_is_acronym = review_acronym_like(left_row, left_label)
    right_is_acronym = review_acronym_like(right_row, right_label)
    country_variant_candidate = official_country_variant_match(
        left_label,
        right_label,
        left_family,
        right_family,
    )
    left_tokens = [token for token in token_list(left_label) if token]
    right_tokens = [token for token in token_list(right_label) if token]
    left_token_count = len(left_tokens)
    right_token_count = len(right_tokens)
    if left_is_acronym and not right_is_acronym:
        acronym_semantic_support = acronym_phrase_support(left_label, right_label)
    elif right_is_acronym and not left_is_acronym:
        acronym_semantic_support = acronym_phrase_support(right_label, left_label)
    else:
        acronym_semantic_support = False
    official_longform_candidate = acronym_longform_surface_ok(
        left_label,
        right_label,
        left_family,
        right_family,
        left_is_acronym=left_is_acronym,
        right_is_acronym=right_is_acronym,
    )
    lexical_support = max(local.token_jaccard, local.weighted_jaccard, local.sequence_ratio)
    semantic_support = max(
        local.cooccurrence_similarity,
        local.relation_neighbor_similarity,
        local.context_similarity,
    )
    combined_docs, combined_mentions, _, materially_relevant = projected_head_review_relevance(
        left_row,
        right_row,
    )
    generic_candidate = bool(left_row.get("is_generic", False)) or bool(right_row.get("is_generic", False))
    acronym_candidate = bool(
        (left_is_acronym ^ right_is_acronym)
        and (local.exact_initials_match or acronym_semantic_support)
    )
    location_candidate = geographic_specialization_candidate(left_row, right_row, local)
    support_signal_count = head_review_support_signal_count(
        local,
        acronym_candidate=acronym_candidate,
        country_variant_candidate=country_variant_candidate,
        location_candidate=location_candidate,
    )
    acronym_label = ""
    if left_is_acronym and not right_is_acronym:
        acronym_label = normalize_label_basic(left_label).replace(" ", "")
    elif right_is_acronym and not left_is_acronym:
        acronym_label = normalize_label_basic(right_label).replace(" ", "")
    short_acronym = bool(acronym_label and len(acronym_label) <= 3)
    head_score = (
        0.18 * float(local.type_score)
        + 0.12 * lexical_support
        + 0.46 * float(local.cooccurrence_similarity)
        + 0.14 * float(local.relation_neighbor_similarity)
        + 0.10 * float(local.context_similarity)
    )
    if (
        not local.hard_conflict
        and local.type_score >= 0.95
        and local.token_jaccard >= 0.95
        and local.sequence_ratio >= 0.80
        and not generic_candidate
    ):
        return "deterministic_merge", head_score, "near_identical_surface"
    if (
        acronym_candidate
        and {left_family, right_family} <= {"country", "government"}
        and local.exact_initials_match
        and official_longform_candidate
        and (
            local.cooccurrence_similarity >= 0.10
            or local.context_similarity >= 0.08
            or local.relation_neighbor_similarity >= 0.05
        )
        and support_signal_count >= 2
    ):
        return "deterministic_merge", max(head_score, 0.30), "country_government_abbreviation"
    if acronym_candidate and short_acronym and local.exact_initials_match and not materially_relevant:
        if min(local.doc_count_left, local.doc_count_right) < 3 and min(local.mention_count_left, local.mention_count_right) < 5:
            return "reject", 0.0, "short_acronym_low_support"
    if acronym_candidate and short_acronym and local.exact_initials_match:
        semantic_support = max(
            local.cooccurrence_similarity,
            local.relation_neighbor_similarity,
            local.context_similarity,
        )
        if min(local.doc_count_left, local.doc_count_right) < 2 and min(local.mention_count_left, local.mention_count_right) < 2:
            if semantic_support < 0.12 and local.relation_neighbor_similarity < 0.12:
                return "reject", 0.0, "short_acronym_singleton_tail"
    if explicit_acronym_fullform_review_eligible(
        left_row,
        right_row,
        local,
        left_is_acronym=left_is_acronym,
        right_is_acronym=right_is_acronym,
    ):
        return "llm_review", max(head_score, 0.24), "explicit_acronym_fullform_candidate"
    if location_candidate and support_signal_count >= 2:
        return "llm_review", max(head_score, 0.30), "location_specialization_candidate"
    if not review_compatible_family(
        left_family,
        right_family,
        acronym_candidate=acronym_candidate,
    ):
        return "reject", 0.0, "family_mismatch"
    if local.hard_conflict and local.hard_conflict not in {"different_head_noun", ""}:
        return "reject", 0.0, local.hard_conflict
    if (
        left_is_acronym
        and right_is_acronym
        and not local.exact_initials_match
    ):
        return "reject", 0.0, "dual_acronym_collision"
    if (
        (left_is_acronym ^ right_is_acronym)
        and (
            (left_is_acronym and right_token_count <= 1)
            or (right_is_acronym and left_token_count <= 1)
        )
        and not local.exact_initials_match
    ):
        return "reject", 0.0, "underspecified_single_token_alias"
    if (
        generic_candidate
        and not local.exact_initials_match
        and not location_candidate
        and not acronym_candidate
    ):
        return "reject", 0.0, "generic_label"
    if acronym_candidate and short_acronym and not local.exact_initials_match:
        longer_label = right_label if left_is_acronym else left_label
        if len(acronym_label) <= 2:
            return "reject", 0.0, "short_acronym_noninitial_collision"
        if (
            contains_country_government_subentity_marker(longer_label)
            or contains_organization_subentity_marker(longer_label)
        ):
            return "reject", 0.0, "acronym_subentity_scope"
    if acronym_candidate and not official_longform_candidate:
        return "reject", 0.0, "non_official_longform_surface"
    if {left_family, right_family} == {"country", "government"}:
        government_label = left_label if left_family == "government" else right_label
        if contains_country_government_subentity_marker(government_label):
            return "reject", 0.0, "country_government_subentity_scope"
        if not surface_has_government_token(government_label):
            return "reject", 0.0, "country_government_scope_mismatch"
    if support_signal_count < 2 and not local.person_name_core_match:
        return "reject", 0.0, "insufficient_support_signals"
    if (
        acronym_candidate
        and max(local.doc_count_left, local.doc_count_right) >= 2
        and max(left_token_count, right_token_count) >= 2
        and (
            semantic_support >= 0.08
            or max(local.left_containment, local.right_containment) >= 0.95
        )
        and (
            local.acronym_expansion_count <= 1
            or semantic_support >= 0.14
            or max(local.doc_count_left, local.doc_count_right) >= 12
            or (
                local.exact_initials_match
                and max(local.doc_count_left, local.doc_count_right) >= 5
                and semantic_support >= 0.08
                and local.type_score >= 0.80
            )
        )
        and (
            support_signal_count >= 3
            or (
                support_signal_count >= 2
                and not short_acronym
                and semantic_support >= 0.10
            )
        )
        and (
            not short_acronym
            or local.acronym_expansion_count <= 1
            or semantic_support >= 0.14
            or materially_relevant
        )
    ):
        return "llm_review", max(head_score, 0.28), "acronym_semantic_candidate"
    if (
        acronym_candidate
        and {left_family, right_family} <= {"country", "government"}
        and max(left_token_count, right_token_count) >= 2
        and max(local.doc_count_left, local.doc_count_right) >= 5
        and official_longform_candidate
        and local.type_score >= 0.95
        and (
            semantic_support >= 0.03
            or max(local.doc_count_left, local.doc_count_right) >= 25
            or materially_relevant
        )
        and (
            semantic_support >= 0.08
            or local.sequence_ratio >= 0.35
            or max(local.left_containment, local.right_containment) >= 0.95
        )
        and support_signal_count >= 2
    ):
        return "llm_review", max(head_score, 0.26), "country_government_variant_candidate"
    if (
        country_variant_candidate
        and local.type_score >= 0.95
        and max(local.doc_count_left, local.doc_count_right) >= 5
        and (
            semantic_support >= 0.02
            or max(local.doc_count_left, local.doc_count_right) >= 20
            or materially_relevant
        )
        and support_signal_count >= 2
    ):
        return "llm_review", max(head_score, 0.27), "country_official_name_variant_candidate"
    if (
        left_family in {"country", "government", "location", "political_group"}
        and
        max(local.doc_count_left, local.doc_count_right) >= 10
        and acronym_candidate
        and max(left_token_count, right_token_count) >= 2
        and (
            local.cooccurrence_similarity >= 0.20
            or local.relation_neighbor_similarity >= 0.18
            or local.context_similarity >= 0.05
        )
        and semantic_support >= 0.18
        and support_signal_count >= 3
    ):
        return "llm_review", head_score, "high_impact_semantic_candidate"
    return "reject", head_score, "insufficient_semantic_support"


def build_head_review_candidates(
    candidate_profiles: pd.DataFrame,
    *,
    acronym_expansion_map: Mapping[str, Set[str]] | None = None,
    max_candidates_per_anchor: int,
    max_total_candidates: int,
) -> pd.DataFrame:
    if candidate_profiles.empty:
        return pd.DataFrame()
    effective_per_anchor_cap = resolve_head_review_per_anchor_cap(
        int(max_candidates_per_anchor),
        int(len(candidate_profiles)),
    )
    effective_total_cap = resolve_head_review_total_cap(
        int(max_total_candidates),
        int(len(candidate_profiles)),
    )
    profile_lookup = candidate_profiles.set_index("base_entity_key").to_dict(orient="index")
    token_idf = build_token_idf(candidate_profiles)
    ordered_keys = candidate_profiles["base_entity_key"].astype(str).tolist()
    rank_lookup = {key: idx for idx, key in enumerate(ordered_keys)}
    candidate_rows: List[Dict[str, object]] = []
    for idx, left_key in enumerate(iter_progress(ordered_keys, total=len(ordered_keys), desc="Alias head review shortlist")):
        left_row = profile_lookup[left_key]
        anchor_rows: List[Dict[str, object]] = []
        for right_key in ordered_keys[idx + 1 :]:
            right_row = profile_lookup[right_key]
            local = local_pair_score(left_row, right_row, token_idf, acronym_expansion_map)
            action, head_score, review_reason = classify_head_review_candidate(left_row, right_row, local)
            if action == "reject":
                continue
            rescue_priority = int(
                acronym_fullform_match(
                    safe_str(left_row.get("normalized_label", "")),
                    safe_str(right_row.get("normalized_label", "")),
                )
                or official_country_variant_match(
                    safe_str(left_row.get("normalized_label", "")),
                    safe_str(right_row.get("normalized_label", "")),
                    family_from_row(left_row),
                    family_from_row(right_row),
                )
            )
            row = local.__dict__.copy()
            row.update(
                {
                    "head_review_action": action,
                    "head_review_score": round(head_score, 6),
                    "head_review_reason": review_reason,
                    "support_signal_count": head_review_support_signal_count(
                        local,
                        acronym_candidate=bool(
                            (
                                review_acronym_like(left_row, safe_str(left_row.get("normalized_label", "")))
                                ^ review_acronym_like(right_row, safe_str(right_row.get("normalized_label", "")))
                            )
                            and (
                                local.exact_initials_match
                                or acronym_phrase_support(
                                    safe_str(left_row.get("normalized_label", "")),
                                    safe_str(right_row.get("normalized_label", "")),
                                )
                                or acronym_phrase_support(
                                    safe_str(right_row.get("normalized_label", "")),
                                    safe_str(left_row.get("normalized_label", "")),
                                )
                            )
                        ),
                        country_variant_candidate=official_country_variant_match(
                            safe_str(left_row.get("normalized_label", "")),
                            safe_str(right_row.get("normalized_label", "")),
                            family_from_row(left_row),
                            family_from_row(right_row),
                        ),
                        location_candidate=geographic_specialization_candidate(left_row, right_row, local),
                    ),
                    "rescue_priority": rescue_priority,
                    "left_member_labels": safe_str(left_row.get("member_labels", "")),
                    "right_member_labels": safe_str(right_row.get("member_labels", "")),
                    "left_member_count": int(left_row.get("member_count", 0) or 0),
                    "right_member_count": int(right_row.get("member_count", 0) or 0),
                    "left_surface_forms": safe_str(left_row.get("surface_forms", "")),
                    "right_surface_forms": safe_str(right_row.get("surface_forms", "")),
                    "left_cluster_root_key": left_key,
                    "right_cluster_root_key": right_key,
                    "left_rank": rank_lookup[left_key],
                    "right_rank": rank_lookup[right_key],
                }
            )
            anchor_rows.append(row)
        if effective_per_anchor_cap > 0 and len(anchor_rows) > effective_per_anchor_cap:
            anchor_rows.sort(
                key=lambda row: (
                    -int(row.get("rescue_priority", 0) or 0),
                    0 if row["head_review_action"] == "deterministic_merge" else 1,
                    -int(row.get("support_signal_count", 0) or 0),
                    -float(row["head_review_score"]),
                    -int(row["doc_count_right"]),
                    -int(row["mention_count_right"]),
                )
            )
            anchor_rows = anchor_rows[:effective_per_anchor_cap]
        candidate_rows.extend(anchor_rows)
    out = pd.DataFrame(candidate_rows)
    if len(out):
        llm_mask = out["head_review_action"] == "llm_review"
        if llm_mask.any():
            llm_pairs = out.loc[llm_mask, ["left_cluster_root_key", "right_cluster_root_key", "head_review_score"]].copy()
            entity_scores = pd.concat(
                [
                    llm_pairs.rename(columns={"left_cluster_root_key": "cluster_root_key"})[
                        ["cluster_root_key", "head_review_score"]
                    ],
                    llm_pairs.rename(columns={"right_cluster_root_key": "cluster_root_key"})[
                        ["cluster_root_key", "head_review_score"]
                    ],
                ],
                ignore_index=True,
            )
            entity_best_map = (
                entity_scores.groupby("cluster_root_key")["head_review_score"].max().to_dict()
            )
            left_best = out["left_cluster_root_key"].map(entity_best_map).fillna(float("-inf"))
            right_best = out["right_cluster_root_key"].map(entity_best_map).fillna(float("-inf"))
            left_threshold = pd.concat(
                [
                    left_best - 0.05,
                    left_best * 0.90,
                ],
                axis=1,
            ).max(axis=1)
            right_threshold = pd.concat(
                [
                    right_best - 0.05,
                    right_best * 0.90,
                ],
                axis=1,
            ).max(axis=1)
            review_keep = (
                (out.get("rescue_priority", 0) > 0)
                |
                (out["head_review_action"] != "llm_review")
                | (
                    (out.get("support_signal_count", 0).fillna(0).astype(int) >= 2)
                    &
                    (out["head_review_score"] >= left_threshold)
                    & (out["head_review_score"] >= right_threshold)
                )
            )
            out = out.loc[review_keep].copy()
        out = out.sort_values(
            ["rescue_priority", "head_review_action", "support_signal_count", "head_review_score", "doc_count_left", "doc_count_right", "mention_count_left", "mention_count_right"],
            ascending=[False, True, False, False, False, False, False, False],
        ).reset_index(drop=True)
        if effective_total_cap > 0:
            out = out.head(effective_total_cap).copy()
    return out


def format_head_review_entity(row: Mapping[str, object]) -> str:
    return "\n".join(
        [
            f"label: {safe_str(row.get('normalized_label', ''))}",
            f"type_distribution: {truncate_prompt_text(row.get('type_distribution', ''), 240)}",
            f"mention_count: {int(float(row.get('mention_count', 0) or 0))}",
            f"doc_count: {int(float(row.get('doc_count', 0) or 0))}",
            f"member_labels: {truncate_prompt_text(row.get('member_labels', ''), 400)}",
            f"surface_forms: {truncate_prompt_text(row.get('surface_forms', ''), 700)}",
            f"cooccurring_entities: {truncate_prompt_text(row.get('cooccurring_entities', ''), 700)}",
            f"relation_neighbor_entities: {truncate_prompt_text(row.get('relation_neighbor_entities', ''), 700)}",
            f"representative_contexts: {truncate_prompt_text(row.get('representative_contexts', ''), 1500)}",
        ]
    )


def head_review_edge_from_candidate_row(
    row: Mapping[str, object],
    *,
    decision_source: str,
    stage: str,
) -> Dict[str, object]:
    return {
        "left": safe_str(row.get("left_cluster_root_key", "")),
        "right": safe_str(row.get("right_cluster_root_key", "")),
        "score": float(row.get("head_review_score", 0.0) or 0.0),
        "decision_source": decision_source,
        "normalized_label_left": safe_str(row.get("normalized_label_left", "")),
        "normalized_label_right": safe_str(row.get("normalized_label_right", "")),
        "stage": stage,
    }


def create_openai_client(api_key_env: str) -> OpenAI:
    api_key = safe_str(os.environ.get(api_key_env, ""))
    if not api_key:
        raise RuntimeError(f"Missing {api_key_env} for head-review alias adjudication.")
    if OpenAI is None:
        raise RuntimeError("OpenAI package is not available in this environment.")
    return OpenAI(api_key=api_key, timeout=90.0, max_retries=2)


def adjudicate_head_review_with_openai(
    *,
    candidate_profiles: pd.DataFrame,
    review_candidates: pd.DataFrame,
    model: str,
    api_key_env: str,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if review_candidates.empty:
        return [], []
    client = create_openai_client(api_key_env)
    profile_lookup = candidate_profiles.set_index("base_entity_key").to_dict(orient="index")
    review_rows: List[Dict[str, object]] = []
    accepted_edges: List[Dict[str, object]] = []
    grouped = review_candidates.groupby("left_cluster_root_key", sort=False)
    for left_key, group in iter_progress(grouped, total=len(grouped), desc="Alias head review adjudication"):
        left_row = profile_lookup[str(left_key)]
        candidate_lines: List[str] = []
        for idx, (_, row) in enumerate(group.iterrows(), start=1):
            right_key = safe_str(row.get("right_cluster_root_key", ""))
            right_row = profile_lookup[right_key]
            candidate_lines.append(
                "\n".join(
                    [
                        f"CANDIDATE_{idx}",
                        format_head_review_entity(right_row),
                        f"precomputed_signal: head_review_score={row.get('head_review_score', 0.0)}; "
                        f"cooccurrence={row.get('cooccurrence_similarity', 0.0)}; "
                        f"context={row.get('context_similarity', 0.0)}; "
                        f"relation_neighbor={row.get('relation_neighbor_similarity', 0.0)}; "
                        f"hard_conflict={safe_str(row.get('hard_conflict', '')) or '[none]'}",
                    ]
                )
            )
        prompt = "\n\n".join(
            [
                "You are reviewing high-impact alias candidates for a historical intelligence corpus knowledge graph.",
                "Task: decide which candidates refer to exactly the same real-world entity as the anchor.",
                "Be strict. Reject broader/narrower institutions, subordinate offices versus parent states or ministries, adjectives/demonyms versus states, titles versus persons, and umbrella abbreviations that only happen to share initials.",
                "Use the corpus-local contexts heavily. If two labels are used interchangeably in this corpus, accept them even when one is an acronym, shortened state name, or longer official expansion.",
                "Accept canonical abbreviation/long-form pairs when they denote the same country, government, organization, or political group in this corpus, such as country abbreviations and common institutional acronyms whose contexts line up.",
                "For location names, only accept city/region expansions when the contexts indicate the same referent in this corpus; otherwise reject broader-versus-narrower place distinctions.",
                "Accept only when the anchor and candidate should share one canonical entity node.",
                "Return JSON with keys `accepted` and `rejected`. `accepted` is a list of candidate ids. `rejected` is an object mapping candidate ids to short reasons.",
                "ANCHOR",
                format_head_review_entity(left_row),
                "CANDIDATES",
                "\n\n".join(candidate_lines),
            ]
        )
        payload = call_openai_json(client, prompt, model)
        accepted = {
            safe_str(value)
            for value in payload.get("accepted", [])
            if safe_str(value)
        }
        rejected = payload.get("rejected", {}) if isinstance(payload.get("rejected", {}), dict) else {}
        for idx, (_, row) in enumerate(group.iterrows(), start=1):
            candidate_id = f"CANDIDATE_{idx}"
            right_key = safe_str(row.get("right_cluster_root_key", ""))
            decision = "reject"
            reason = safe_str(rejected.get(candidate_id, "not_accepted"))
            if candidate_id in accepted:
                decision = "accept"
                reason = "llm_alias_accept"
                accepted_edges.append(
                    {
                        "left": safe_str(row.get("left_cluster_root_key", "")),
                        "right": right_key,
                        "score": float(row.get("head_review_score", 0.0) or 0.0),
                        "decision_source": "head_review_llm",
                        "normalized_label_left": safe_str(row.get("normalized_label_left", "")),
                        "normalized_label_right": safe_str(row.get("normalized_label_right", "")),
                        "stage": "stage2_head_llm",
                    }
                )
            review_row = row.to_dict()
            review_row["head_review_llm_decision"] = decision
            review_row["head_review_llm_reason"] = reason
            review_rows.append(review_row)
    return accepted_edges, review_rows


def materialize_cluster_exports(
    profiles: pd.DataFrame,
    cluster_members: Mapping[str, Sequence[str]],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    canonical_rows: List[Dict[str, object]] = []
    alias_rows: List[Dict[str, object]] = []
    for cluster_idx, members in enumerate(sorted(cluster_members.values(), key=len, reverse=True), start=1):
        member_rows = profiles[profiles["base_entity_key"].isin(members)].copy()
        canonical_label = choose_canonical_label(member_rows)
        cluster_id = f"CANONICAL_{cluster_idx:05d}"
        mention_total = int(member_rows["mention_count"].fillna(0).astype(int).sum())
        doc_total = int(member_rows["doc_count"].fillna(0).astype(int).sum())
        type_counts = Counter()
        for value in member_rows["type_distribution"].fillna("").astype(str):
            for item in parse_multivalue_field(value):
                type_counts[item] += 1
        canonical_rows.append(
            {
                "canonical_entity_id": cluster_id,
                "canonical_label": canonical_label,
                "member_count": int(len(member_rows)),
                "mention_count": mention_total,
                "doc_count": doc_total,
                "member_labels": "; ".join(member_rows["normalized_label"].astype(str).tolist()),
                "member_base_entity_keys": "; ".join(member_rows["base_entity_key"].astype(str).tolist()),
                "type_distribution": "; ".join(f"{label}:{count}" for label, count in type_counts.most_common()),
            }
        )
        for _, member in member_rows.iterrows():
            alias_rows.append(
                {
                    "alias": safe_str(member.get("normalized_label", "")),
                    "canonical": canonical_label,
                    "alias_key": canonical_key(member.get("normalized_label", "")),
                    "canonical_key": canonical_key(canonical_label),
                    "base_entity_key": safe_str(member.get("base_entity_key", "")),
                    "canonical_entity_id": cluster_id,
                    "mention_count": int(member.get("mention_count", 0) or 0),
                    "doc_count": int(member.get("doc_count", 0) or 0),
                    "primary_type": safe_str(member.get("primary_type", "")),
                }
            )
    return canonical_rows, alias_rows


def _edge_rows_to_dicts(df: pd.DataFrame) -> List[Dict[str, object]]:
    if df.empty:
        return []
    out: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        out.append(
            {
                "left": safe_str(row.get("left", "")),
                "right": safe_str(row.get("right", "")),
                "score": float(row.get("score", 0.0) or 0.0),
                "decision_source": safe_str(row.get("decision_source", "")),
                "normalized_label_left": safe_str(row.get("normalized_label_left", "")),
                "normalized_label_right": safe_str(row.get("normalized_label_right", "")),
                "stage": safe_str(row.get("stage", "")),
            }
        )
    return out


def run_head_review_only(
    *,
    project_dir: str | Path,
    use_openai_head_review: bool,
    head_review_model: str = "gpt-4o",
    openai_api_key_env: str = "OPENAI_API_KEY",
) -> Dict[str, object]:
    project_dir = Path(project_dir)
    extraction_dir = project_dir / "extraction"
    alias_dir = project_dir / "alias"

    profiles_path = extraction_dir / "base_entity_profiles.csv"
    edges_path = alias_dir / "alias_edges.csv"
    head_profiles_path = alias_dir / "head_review_profiles.csv"
    head_candidates_path = alias_dir / "head_review_candidates.csv"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Missing base entity profiles: {profiles_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing alias edges: {edges_path}")
    if not head_profiles_path.exists():
        raise FileNotFoundError(f"Missing head review profiles: {head_profiles_path}")
    if not head_candidates_path.exists():
        raise FileNotFoundError(f"Missing head review candidates: {head_candidates_path}")

    profiles = pd.read_csv(profiles_path, low_memory=False)
    profiles, repair_manifest = repair_base_entity_key_collisions(extraction_dir, profiles)

    try:
        edges_df = pd.read_csv(edges_path, low_memory=False)
    except EmptyDataError:
        edges_df = pd.DataFrame(
            columns=[
                "left",
                "right",
                "score",
                "decision_source",
                "normalized_label_left",
                "normalized_label_right",
                "stage",
            ]
        )
    if "stage" in edges_df.columns:
        stage_series = edges_df["stage"].fillna("").astype(str)
        base_edges_df = edges_df[~stage_series.str.startswith("stage2_head_")].copy()
    else:
        base_edges_df = edges_df.copy()
        base_edges_df["stage"] = "stage1_local"
    base_edges = _edge_rows_to_dicts(base_edges_df)

    try:
        head_profiles = pd.read_csv(head_profiles_path, low_memory=False)
    except EmptyDataError:
        head_profiles = pd.DataFrame()
    try:
        head_candidates = pd.read_csv(head_candidates_path, low_memory=False)
    except EmptyDataError:
        head_candidates = pd.DataFrame()

    head_review_decision_rows: List[Dict[str, object]] = []
    head_review_edges: List[Dict[str, object]] = []
    if len(head_candidates):
        deterministic_rows = head_candidates[
            head_candidates["head_review_action"] == "deterministic_merge"
        ].copy()
        for _, row in deterministic_rows.iterrows():
            head_review_edges.append(
                head_review_edge_from_candidate_row(
                    row,
                    decision_source="head_review_deterministic",
                    stage="stage2_head_deterministic",
                )
            )
            decision_row = row.to_dict()
            decision_row["head_review_llm_decision"] = "accept"
            decision_row["head_review_llm_reason"] = "deterministic_merge"
            head_review_decision_rows.append(decision_row)

        llm_review_rows = head_candidates[
            head_candidates["head_review_action"] == "llm_review"
        ].copy()
        if use_openai_head_review and len(llm_review_rows):
            llm_edges, llm_decisions = adjudicate_head_review_with_openai(
                candidate_profiles=head_profiles,
                review_candidates=llm_review_rows,
                model=head_review_model,
                api_key_env=openai_api_key_env,
            )
            head_review_edges.extend(llm_edges)
            head_review_decision_rows.extend(llm_decisions)
        elif len(llm_review_rows):
            for _, row in llm_review_rows.iterrows():
                decision_row = row.to_dict()
                decision_row["head_review_llm_decision"] = "pending"
                decision_row["head_review_llm_reason"] = "openai_head_review_disabled"
                head_review_decision_rows.append(decision_row)

    if head_review_decision_rows:
        pd.DataFrame(head_review_decision_rows).to_csv(
            alias_dir / "head_review_decisions.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(alias_dir / "head_review_decisions.csv", index=False)

    accepted_edges = list(base_edges)
    accepted_edges.extend(head_review_edges)

    uf = UnionFind(profiles["base_entity_key"].astype(str).tolist())
    for edge in accepted_edges:
        uf.union(safe_str(edge.get("left", "")), safe_str(edge.get("right", "")))

    final_cluster_members = cluster_members_from_union_find(
        uf,
        profiles["base_entity_key"].astype(str).tolist(),
    )
    canonical_rows, alias_rows = materialize_cluster_exports(
        profiles,
        final_cluster_members,
    )

    final_edges_df = pd.DataFrame(accepted_edges)
    final_edges_df.to_csv(alias_dir / "alias_edges.csv", index=False)
    canonical_df = pd.DataFrame(canonical_rows).sort_values(
        ["mention_count", "member_count"],
        ascending=[False, False],
    )
    canonical_df.to_csv(alias_dir / "canonical_entities.csv", index=False)
    alias_map_df = pd.DataFrame(alias_rows).sort_values(
        ["canonical", "mention_count"],
        ascending=[True, False],
    )
    alias_map_df.to_csv(alias_dir / "alias_map.csv", index=False)
    runtime_resource_path = export_corpus_entity_resource(
        project_dir=project_dir,
        profiles=profiles,
        canonical_rows=canonical_rows,
    )

    manifest = apply_alias_map_to_extraction(project_dir, alias_map_df)
    summary = {
        "project_dir": str(project_dir),
        "profile_count": int(len(profiles)),
        "candidate_pair_count": None,
        "accepted_edge_count": int(len(accepted_edges)),
        "stage1_local_edge_count": int(
            sum(1 for row in accepted_edges if safe_str(row.get("stage", "")) == "stage1_local")
        ),
        "head_review_candidate_count": int(len(head_candidates)),
        "head_review_edge_count": int(
            sum(
                1
                for row in accepted_edges
                if safe_str(row.get("stage", "")).startswith("stage2_head_")
            )
        ),
        "head_review_openai_enabled": bool(use_openai_head_review),
        "canonical_entity_count": int(len(canonical_df)),
        "alias_rows": int(len(alias_map_df)),
        "runtime_corpus_entity_resource": str(runtime_resource_path),
        "canonicalized_output_dir": manifest["output_dir"],
        "base_entity_key_repair": repair_manifest,
        "head_review_only": True,
    }
    (alias_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def rebuild_head_review_shortlist_only(
    *,
    project_dir: str | Path,
    head_review_top_k: int = 0,
    head_review_max_candidates_per_anchor: int = 0,
    head_review_max_total_candidates: int = 0,
) -> Dict[str, object]:
    project_dir = Path(project_dir)
    extraction_dir = project_dir / "extraction"
    alias_dir = project_dir / "alias"

    profiles_path = extraction_dir / "base_entity_profiles.csv"
    head_profiles_path = alias_dir / "head_review_profiles.csv"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Missing base entity profiles: {profiles_path}")
    if not head_profiles_path.exists():
        raise FileNotFoundError(f"Missing head review profiles: {head_profiles_path}")

    profiles = pd.read_csv(profiles_path, low_memory=False)
    profiles, repair_manifest = repair_base_entity_key_collisions(extraction_dir, profiles)
    try:
        head_profiles = pd.read_csv(head_profiles_path, low_memory=False)
    except EmptyDataError:
        head_profiles = pd.DataFrame()

    acronym_expansion_map = build_acronym_expansion_map(profiles)
    effective_top_k = resolve_head_review_top_k(int(head_review_top_k), int(len(head_profiles)))
    selected_head_profiles = select_head_review_profiles(
        head_profiles,
        top_k=effective_top_k,
    )
    effective_per_anchor_cap = resolve_head_review_per_anchor_cap(
        int(head_review_max_candidates_per_anchor),
        int(len(selected_head_profiles)),
    )
    effective_total_cap = resolve_head_review_total_cap(
        int(head_review_max_total_candidates),
        int(len(selected_head_profiles)),
    )
    head_candidates = build_head_review_candidates(
        selected_head_profiles,
        acronym_expansion_map=acronym_expansion_map,
        max_candidates_per_anchor=effective_per_anchor_cap,
        max_total_candidates=effective_total_cap,
    )
    head_candidates.to_csv(alias_dir / "head_review_candidates.csv", index=False)
    pd.DataFrame().to_csv(alias_dir / "head_review_decisions.csv", index=False)

    summary = {
        "project_dir": str(project_dir),
        "profile_count": int(len(profiles)),
        "head_review_profile_count": int(len(head_profiles)),
        "head_review_selected_profile_count": int(len(selected_head_profiles)),
        "head_review_candidate_count": int(len(head_candidates)),
        "head_review_top_k": int(max(int(head_review_top_k), 0)),
        "head_review_top_k_effective": int(effective_top_k),
        "head_review_max_candidates_per_anchor": int(max(int(head_review_max_candidates_per_anchor), 0)),
        "head_review_max_candidates_per_anchor_effective": int(effective_per_anchor_cap),
        "head_review_max_total_candidates": int(max(int(head_review_max_total_candidates), 0)),
        "head_review_max_total_candidates_effective": int(effective_total_cap),
        "base_entity_key_repair": repair_manifest,
        "rebuild_head_review_shortlist_only": True,
    }
    (alias_dir / "head_review_shortlist_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def canonicalize_series(series: pd.Series, alias_map: Mapping[str, str]) -> Tuple[pd.Series, pd.Series]:
    original = series.fillna("").map(norm_space)
    canonical = original.map(lambda value: alias_map.get(canonical_key(value), value))
    changed = (original.map(canonical_key) != canonical.map(canonical_key)) & original.ne("")
    return canonical, changed


def apply_alias_map_to_extraction(project_dir: Path, alias_map_df: pd.DataFrame) -> Dict[str, object]:
    extraction_dir = Path(project_dir) / "extraction"
    output_dir = Path(project_dir) / "extraction_canonicalized"
    output_dir.mkdir(parents=True, exist_ok=True)

    alias_map = {
        canonical_key(alias): safe_str(canonical)
        for alias, canonical in zip(alias_map_df["alias"], alias_map_df["canonical"])
        if safe_str(alias) and safe_str(canonical)
    }

    manifest: Dict[str, object] = {"output_dir": str(output_dir), "tables": {}}
    for filename, fields in ALIAS_FIELD_POLICY.items():
        path = extraction_dir / filename
        if not path.exists():
            manifest["tables"][filename] = {"exists": False, "rows": 0, "changed_cells": 0}
            continue
        df = pd.read_csv(path, low_memory=False)
        out = df.copy()
        changed_total = 0
        for field in fields:
            if field not in out.columns:
                continue
            original_col = f"{field}_original"
            if original_col not in out.columns:
                out[original_col] = out[field]
            canonical, changed = canonicalize_series(out[field], alias_map)
            out[field] = canonical
            out[f"{field}_alias_applied"] = changed.map(lambda value: "true" if bool(value) else "false")
            changed_total += int(changed.sum())
        out.to_csv(output_dir / filename, index=False)
        manifest["tables"][filename] = {
            "exists": True,
            "rows": int(len(out)),
            "changed_cells": int(changed_total),
            "output_path": str(output_dir / filename),
        }
    (output_dir / "alias_application_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_alias_resolution(
    *,
    project_dir: Path,
    head_review_top_k: int = 0,
    head_review_max_candidates_per_anchor: int = 0,
    head_review_max_total_candidates: int = 0,
    use_openai_head_review: bool = False,
    head_review_model: str = "gpt-4o",
    openai_api_key_env: str = "OPENAI_API_KEY",
) -> Dict[str, object]:
    project_dir = Path(project_dir)
    extraction_dir = project_dir / "extraction"
    alias_dir = project_dir / "alias"
    alias_dir.mkdir(parents=True, exist_ok=True)

    profiles_path = extraction_dir / "base_entity_profiles.csv"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Missing base entity profiles: {profiles_path}")
    profiles = pd.read_csv(profiles_path, low_memory=False)
    if len(profiles) == 0:
        raise ValueError("No base entity profiles found.")

    profiles, repair_manifest = repair_base_entity_key_collisions(extraction_dir, profiles)
    profile_lookup = profiles.set_index("base_entity_key").to_dict(orient="index")
    token_idf = build_token_idf(profiles)
    acronym_expansion_map = build_acronym_expansion_map(profiles)
    pair_keys = sorted(generate_candidate_pairs(profiles))

    candidate_rows: List[Dict[str, object]] = []
    accepted_edges: List[Dict[str, object]] = []

    for left_key, right_key in iter_progress(pair_keys, total=len(pair_keys), desc="Alias candidate scoring"):
        left_row = profile_lookup[left_key]
        right_row = profile_lookup[right_key]
        local = local_pair_score(left_row, right_row, token_idf, acronym_expansion_map)
        row = local.__dict__.copy()

        if row["decision"] == "same":
            accepted_edges.append(
                {
                    "left": left_key,
                    "right": right_key,
                    "score": row["score"],
                    "decision_source": row["decision_source"],
                    "normalized_label_left": row["normalized_label_left"],
                    "normalized_label_right": row["normalized_label_right"],
                    "stage": "stage1_local",
                }
            )
        candidate_rows.append(row)

    candidate_columns = [
        "base_entity_key_left",
        "base_entity_key_right",
        "normalized_label_left",
        "normalized_label_right",
        "primary_type_left",
        "primary_type_right",
        "mention_count_left",
        "mention_count_right",
        "doc_count_left",
        "doc_count_right",
        "type_score",
        "type_reason",
        "exact_initials_match",
        "person_name_core_match",
        "head_match",
        "token_jaccard",
        "weighted_jaccard",
        "left_containment",
        "right_containment",
        "sequence_ratio",
        "context_similarity",
        "cooccurrence_similarity",
        "relation_neighbor_similarity",
        "acronym_expansion_count",
        "score",
        "decision",
        "decision_source",
        "hard_conflict",
    ]
    candidates_df = pd.DataFrame(candidate_rows, columns=candidate_columns)
    if len(candidates_df):
        candidates_df = candidates_df.sort_values(
            ["decision", "score", "mention_count_left", "mention_count_right"],
            ascending=[True, False, False, False],
        )
    candidates_df.to_csv(alias_dir / "alias_candidates.csv", index=False)

    uf = UnionFind(profiles["base_entity_key"].astype(str).tolist())
    for edge in accepted_edges:
        uf.union(edge["left"], edge["right"])

    stage1_cluster_members = cluster_members_from_union_find(
        uf,
        profiles["base_entity_key"].astype(str).tolist(),
    )
    head_profiles = aggregate_cluster_profiles(stage1_cluster_members, profiles)
    effective_top_k = resolve_head_review_top_k(int(head_review_top_k), int(len(head_profiles)))
    selected_head_profiles = select_head_review_profiles(
        head_profiles,
        top_k=effective_top_k,
    )
    effective_per_anchor_cap = resolve_head_review_per_anchor_cap(
        int(head_review_max_candidates_per_anchor),
        int(len(selected_head_profiles)),
    )
    effective_total_cap = resolve_head_review_total_cap(
        int(head_review_max_total_candidates),
        int(len(selected_head_profiles)),
    )
    head_candidates = build_head_review_candidates(
        selected_head_profiles,
        acronym_expansion_map=acronym_expansion_map,
        max_candidates_per_anchor=effective_per_anchor_cap,
        max_total_candidates=effective_total_cap,
    )
    head_profiles.to_csv(alias_dir / "head_review_profiles.csv", index=False)
    head_candidates.to_csv(alias_dir / "head_review_candidates.csv", index=False)

    head_review_decision_rows: List[Dict[str, object]] = []
    head_review_edges: List[Dict[str, object]] = []
    if len(head_candidates):
        deterministic_rows = head_candidates[
            head_candidates["head_review_action"] == "deterministic_merge"
        ].copy()
        for _, row in deterministic_rows.iterrows():
            edge = {
                "left": safe_str(row.get("left_cluster_root_key", "")),
                "right": safe_str(row.get("right_cluster_root_key", "")),
                "score": float(row.get("head_review_score", 0.0) or 0.0),
                "decision_source": "head_review_deterministic",
                "normalized_label_left": safe_str(row.get("normalized_label_left", "")),
                "normalized_label_right": safe_str(row.get("normalized_label_right", "")),
                "stage": "stage2_head_deterministic",
            }
            head_review_edges.append(edge)
            decision_row = row.to_dict()
            decision_row["head_review_llm_decision"] = "accept"
            decision_row["head_review_llm_reason"] = "deterministic_merge"
            head_review_decision_rows.append(decision_row)

        llm_review_rows = head_candidates[
            head_candidates["head_review_action"] == "llm_review"
        ].copy()
        if use_openai_head_review and len(llm_review_rows):
            llm_edges, llm_decisions = adjudicate_head_review_with_openai(
                candidate_profiles=selected_head_profiles,
                review_candidates=llm_review_rows,
                model=head_review_model,
                api_key_env=openai_api_key_env,
            )
            head_review_edges.extend(llm_edges)
            head_review_decision_rows.extend(llm_decisions)
        elif len(llm_review_rows):
            for _, row in llm_review_rows.iterrows():
                decision_row = row.to_dict()
                decision_row["head_review_llm_decision"] = "pending"
                decision_row["head_review_llm_reason"] = "openai_head_review_disabled"
                head_review_decision_rows.append(decision_row)

    if head_review_decision_rows:
        pd.DataFrame(head_review_decision_rows).to_csv(
            alias_dir / "head_review_decisions.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(alias_dir / "head_review_decisions.csv", index=False)

    for edge in head_review_edges:
        uf.union(edge["left"], edge["right"])
        accepted_edges.append(edge)

    final_cluster_members = cluster_members_from_union_find(
        uf,
        profiles["base_entity_key"].astype(str).tolist(),
    )
    canonical_rows, alias_rows = materialize_cluster_exports(
        profiles,
        final_cluster_members,
    )

    edges_df = pd.DataFrame(accepted_edges)
    edges_df.to_csv(alias_dir / "alias_edges.csv", index=False)
    canonical_df = pd.DataFrame(canonical_rows).sort_values(["mention_count", "member_count"], ascending=[False, False])
    canonical_df.to_csv(alias_dir / "canonical_entities.csv", index=False)
    alias_map_df = pd.DataFrame(alias_rows).sort_values(["canonical", "mention_count"], ascending=[True, False])
    alias_map_df.to_csv(alias_dir / "alias_map.csv", index=False)
    runtime_resource_path = export_corpus_entity_resource(
        project_dir=project_dir,
        profiles=profiles,
        canonical_rows=canonical_rows,
    )

    manifest = apply_alias_map_to_extraction(project_dir, alias_map_df)
    summary = {
        "project_dir": str(project_dir),
        "profile_count": int(len(profiles)),
        "candidate_pair_count": int(len(candidates_df)),
        "accepted_edge_count": int(len(accepted_edges)),
        "stage1_local_edge_count": int(sum(1 for row in accepted_edges if safe_str(row.get("stage", "")) == "stage1_local")),
        "head_review_candidate_count": int(len(head_candidates)),
        "head_review_edge_count": int(len(head_review_edges)),
        "head_review_top_k": int(max(int(head_review_top_k), 0)),
        "head_review_top_k_effective": int(effective_top_k),
        "head_review_max_candidates_per_anchor": int(max(int(head_review_max_candidates_per_anchor), 0)),
        "head_review_max_candidates_per_anchor_effective": int(effective_per_anchor_cap),
        "head_review_max_total_candidates": int(max(int(head_review_max_total_candidates), 0)),
        "head_review_max_total_candidates_effective": int(effective_total_cap),
        "head_review_openai_enabled": bool(use_openai_head_review),
        "canonical_entity_count": int(len(canonical_df)),
        "alias_rows": int(len(alias_map_df)),
        "runtime_corpus_entity_resource": str(runtime_resource_path),
        "canonicalized_output_dir": manifest["output_dir"],
        "base_entity_key_repair": repair_manifest,
    }
    (alias_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve cross-spelling aliases from Stage A base-entity profiles.")
    parser.add_argument("--project-dir", required=True, help="Project directory containing extraction outputs.")
    parser.add_argument("--head-review-top-k", type=int, default=0)
    parser.add_argument("--head-review-max-candidates-per-anchor", type=int, default=0)
    parser.add_argument("--head-review-max-total-candidates", type=int, default=0)
    parser.add_argument(
        "--rebuild-head-review-shortlist-only",
        action="store_true",
        help="Reuse existing stage-1 head-review profiles and rebuild only the shortlist candidates.",
    )
    parser.add_argument(
        "--head-review-only",
        action="store_true",
        help="Reuse existing stage-1 alias outputs and rerun only the stage-2 head review/materialization path.",
    )
    parser.add_argument("--use-openai-head-review", action="store_true")
    parser.add_argument("--head-review-model", default="gpt-4o")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    if args.rebuild_head_review_shortlist_only:
        rebuild_head_review_shortlist_only(
            project_dir=project_dir,
            head_review_top_k=args.head_review_top_k,
            head_review_max_candidates_per_anchor=args.head_review_max_candidates_per_anchor,
            head_review_max_total_candidates=args.head_review_max_total_candidates,
        )
    elif args.head_review_only:
        run_head_review_only(
            project_dir=project_dir,
            use_openai_head_review=bool(args.use_openai_head_review),
            head_review_model=args.head_review_model,
            openai_api_key_env=args.openai_api_key_env,
        )
    else:
        run_alias_resolution(
            project_dir=project_dir,
            head_review_top_k=args.head_review_top_k,
            head_review_max_candidates_per_anchor=args.head_review_max_candidates_per_anchor,
            head_review_max_total_candidates=args.head_review_max_total_candidates,
            use_openai_head_review=bool(args.use_openai_head_review),
            head_review_model=args.head_review_model,
            openai_api_key_env=args.openai_api_key_env,
        )


if __name__ == "__main__":
    main()
