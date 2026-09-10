"""Shared utilities for the standalone kg_scripts pipeline."""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def iter_progress(iterable, total: Optional[int] = None, desc: str = "", initial: Optional[int] = None):
    if tqdm is None:
        return iterable
    kwargs = {
        "total": total,
        "desc": desc,
        "unit": "item",
    }
    if initial is not None:
        kwargs["initial"] = initial
    return tqdm(iterable, **kwargs)


DEFAULT_ALIAS_INPUT_DIRNAME = "entity_directory"
DEFAULT_ALIAS_SEED_FILENAME = "entity_aliases.csv"
DEFAULT_CONFIG_DIR_NAME = "config"
DEFAULT_ALIAS_OVERRIDE_FILENAME = "alias_overrides.csv"
DEFAULT_PROFILE_OUTPUT_DIRNAME = "profiles"
DEFAULT_RAW_EXTRACTION_DIRNAME = "extraction"
MENTION_PROFILE_FILENAME = "entity_mentions.csv"
BASE_ENTITY_INVENTORY_FILENAME = "base_entity_inventory.csv"
PROFILE_SUMMARY_FILENAME = "profile_summary.json"

TOKEN_STOPWORDS = {
    "A",
    "AN",
    "AND",
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

PERSON_TITLE_TOKENS = {
    "ADMIRAL",
    "AMBASSADOR",
    "BRIGADIER",
    "CAPTAIN",
    "CHAIRMAN",
    "COL",
    "COLONEL",
    "COMMANDER",
    "DR",
    "DOCTOR",
    "GENERAL",
    "GENERALISSIMO",
    "GEN",
    "KING",
    "MAJOR",
    "MISTER",
    "MR",
    "MRS",
    "MS",
    "PRESIDENT",
    "PREMIER",
    "PRIME",
    "PRINCE",
    "PRINCESS",
    "PROFESSOR",
    "QUEEN",
    "SECRETARY",
    "SIR",
}

DIRECTIONAL_TOKEN_MAP = {
    "NORTH": "NORTH",
    "NORTHERN": "NORTH",
    "SOUTH": "SOUTH",
    "SOUTHERN": "SOUTH",
    "EAST": "EAST",
    "EASTERN": "EAST",
    "WEST": "WEST",
    "WESTERN": "WEST",
}

BRANCH_TOKEN_MAP = {
    "AIR": "AIR",
    "AIRFORCE": "AIR",
    "ARMY": "ARMY",
    "MARINE": "MARINE",
    "NAVAL": "NAVY",
    "NAVY": "NAVY",
}

LOCATION_MARKER_TOKENS = {
    "CAPITAL",
    "CITY",
    "ISLAND",
    "PREFECTURE",
    "PROVINCE",
    "REGION",
    "STATE",
    "TERRITORY",
}

GOVERNMENT_MARKER_TOKENS = {
    "GOVERNMENT",
    "KINGDOM",
    "REGIME",
    "REPUBLIC",
}

ORGANIZATION_MARKER_TOKENS = {
    "AGENCY",
    "BUREAU",
    "COMMISSION",
    "COMMITTEE",
    "CONFERENCE",
    "CONGRESS",
    "COUNCIL",
    "DELEGATION",
    "DEPARTMENT",
    "MINISTRY",
    "MISSION",
    "PRESIDIUM",
    "SECRETARIAT",
}

POLITICAL_MARKER_TOKENS = {
    "BLOC",
    "FRONT",
    "MOVEMENT",
    "NATIONALISTS",
    "PARTY",
    "UNION",
}

MILITARY_MARKER_TOKENS = {
    "AIRBORNE",
    "ARMY",
    "BATTALION",
    "DIVISION",
    "FORCE",
    "FORCES",
    "GUERRILLAS",
    "IRREGULARS",
    "MILITARY",
    "NAVY",
    "REGIMENT",
    "TROOPS",
}

ROLE_MARKER_TOKENS = {
    "ASSISTANT",
    "CHAIRMAN",
    "CHIEF",
    "COUNSELOR",
    "DELEGATE",
    "DEPUTY",
    "DIRECTOR",
    "ENVOY",
    "MINISTER",
    "REPRESENTATIVE",
    "SECRETARY",
}

GENERATED_ALIAS_FILENAME = "entity_aliases_generated.csv"

TYPE_COMPATIBILITY_GROUPS = [
    {"COUNTRY", "GOVERNMENT", "LOCATION"},
    {"ORGANIZATION", "POLITICALGROUP", "MILITARY"},
    {"PERSON"},
    {"DOCUMENT"},
    {"RELIGIOUSGROUP"},
    {"OTHER"},
]

SAFE_RULE_ACTIONS = {
    "leading_article",
    "parenthetical_base_match",
    "acronym_match",
    "person_title_variant",
}

OPENAI_PROMPT_VERSION = "kg_scripts_core_v1"
OPENAI_ACCEPT_CONFIDENCE_THRESHOLD = 0.80

NEGATIVE_DISTINGUISHER_TOKENS = {
    "ACTING",
    "AIR",
    "ARMY",
    "DEPUTY",
    "EAST",
    "FORCE",
    "INTERIM",
    "JOINT",
    "MARINE",
    "NAVAL",
    "NAVY",
    "NORTH",
    "SOUTH",
    "UNDER",
    "VICE",
    "WEST",
}

GENERIC_ENTITY_TOKENS = {
    "ADMINISTRATION",
    "AGENCY",
    "ARMY",
    "AUTHORITIES",
    "BLOC",
    "CABINET",
    "COMMISSION",
    "COMMITTEE",
    "CONFERENCE",
    "COUNCIL",
    "DEPARTMENT",
    "EMBASSY",
    "FORCE",
    "FORCES",
    "FRONT",
    "GOVERNMENT",
    "GROUP",
    "LEADERS",
    "LEADERSHIP",
    "MILITARY",
    "MINISTER",
    "MINISTRY",
    "MISSION",
    "MOVEMENT",
    "OFFICIAL",
    "OFFICIALS",
    "PARTY",
    "POLICE",
    "PRESIDENT",
    "REGIME",
    "SECRETARY",
    "SECURITY",
    "STATE",
    "UNION",
}


def safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def norm_space(value: object) -> str:
    return re.sub(r"\s+", " ", safe_str(value)).strip()


def normalize_label_basic(text: object) -> str:
    normalized = norm_space(text)
    normalized = normalized.replace("&amp;", "&")
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = normalized.upper()
    normalized = re.sub(r"[\"'`]+", "", normalized)
    normalized = re.sub(r"[.,;:()\[\]{}]", " ", normalized)
    normalized = re.sub(r"[-_/]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"^(THE|A|AN)\s+", "", normalized)
    normalized = normalized.replace("U S A", "USA")
    normalized = normalized.replace("U S", "US")
    normalized = normalized.replace("U N", "UN")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_label_for_base_key(text: object) -> str:
    """
    Normalize a label for internal base-entity keys without collapsing
    surface-distinct spellings such as `A ZHDANOV` into `ZHDANOV`.

    Alias and canonical matching can still strip leading articles later; the
    base key should preserve the Stage A spelling bucket.
    """

    normalized = norm_space(text)
    normalized = normalized.replace("&amp;", "&")
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = normalized.upper()
    normalized = re.sub(r"[\"'`]+", "", normalized)
    normalized = re.sub(r"[.,;:()\[\]{}]", " ", normalized)
    normalized = re.sub(r"[-_/]", " ", normalized)
    normalized = normalized.replace("U S A", "USA")
    normalized = normalized.replace("U S", "US")
    normalized = normalized.replace("U N", "UN")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def keyify(text: object) -> str:
    normalized = normalize_label_basic(text)
    normalized = re.sub(r"[^A-Z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "UNKNOWN"


def stable_base_entity_key(text: object) -> str:
    normalized = normalize_label_for_base_key(text)
    if not normalized:
        return "BASE_ENTITY:UNKNOWN"
    stem = re.sub(r"[^A-Z0-9]+", "_", normalized)
    stem = re.sub(r"_+", "_", stem).strip("_") or "UNKNOWN"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10].upper()
    return f"BASE_ENTITY:{stem}__{digest}"


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def resolve_entity_directory(project_dir: Path, entity_dir_arg: Optional[Path] = None) -> Path:
    if entity_dir_arg is not None:
        return Path(entity_dir_arg).expanduser().resolve()
    return project_dir / DEFAULT_ALIAS_INPUT_DIRNAME


def _coerce_alias_table(alias_df: pd.DataFrame) -> pd.DataFrame:
    if len(alias_df) == 0:
        return pd.DataFrame(columns=["alias", "canonical_label", "notes"])

    alias_col = first_existing_col(alias_df, ["alias", "variant", "label", "original_label"])
    canonical_col = first_existing_col(alias_df, ["canonical_label", "canonical", "canonical_name"])
    notes_col = first_existing_col(alias_df, ["notes", "note", "comment"])
    if alias_col is None or canonical_col is None:
        raise KeyError(
            f"Alias table must contain alias and canonical columns. Columns are: {alias_df.columns.tolist()}"
        )

    out = pd.DataFrame(
        {
            "alias": alias_df[alias_col].map(norm_space),
            "canonical_label": alias_df[canonical_col].map(norm_space),
            "notes": alias_df[notes_col].map(norm_space) if notes_col else "",
        }
    )
    out = out[(out["alias"] != "") & (out["canonical_label"] != "")].drop_duplicates()
    return out.reset_index(drop=True)


def load_entity_alias_table(entity_dir: Path) -> pd.DataFrame:
    alias_path = entity_dir / DEFAULT_ALIAS_SEED_FILENAME
    if not alias_path.exists():
        return pd.DataFrame(columns=["alias", "canonical_label", "notes"])
    return _coerce_alias_table(pd.read_csv(alias_path))


def load_manual_alias_overrides(project_dir: Path) -> pd.DataFrame:
    alias_path = project_dir / DEFAULT_CONFIG_DIR_NAME / DEFAULT_ALIAS_OVERRIDE_FILENAME
    if not alias_path.exists():
        return pd.DataFrame(columns=["alias", "canonical_label", "notes"])
    return _coerce_alias_table(pd.read_csv(alias_path))


def load_seed_alias_table(project_dir: Path, entity_dir: Path) -> pd.DataFrame:
    _ = entity_dir  # retained for backward-compatible call signatures
    manual = load_manual_alias_overrides(project_dir)
    return manual.reset_index(drop=True)


def build_entity_alias_map(alias_df: pd.DataFrame) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for _, row in alias_df.iterrows():
        alias_map[normalize_label_basic(row["alias"])] = normalize_label_basic(row["canonical_label"])
    return alias_map


def resolve_alias_chain(alias: str, alias_map: Dict[str, str], max_hops: int = 20) -> str:
    current = normalize_label_basic(alias)
    seen: Set[str] = set()
    for _ in range(max_hops):
        target = normalize_label_basic(alias_map.get(current, current))
        if not target or target == current:
            return current
        if target in seen:
            return target
        seen.add(current)
        current = target
    return current


def canonicalize_entity_label(label: object, alias_map: Dict[str, str]) -> str:
    raw = normalize_label_basic(label)
    if not raw:
        return ""
    return alias_map.get(raw, raw)


def strip_leading_article(label: str) -> str:
    stripped = re.sub(r"^(THE|A|AN)\s+", "", label).strip()
    return stripped or label


def strip_trailing_parenthetical(label: str) -> str:
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    return stripped or label


def acronym_from_label(label: str) -> str:
    tokens = [
        token
        for token in re.split(r"[\s\-]+", label)
        if token and token not in TOKEN_STOPWORDS
    ]
    if len(tokens) < 2:
        return ""
    return "".join(token[0] for token in tokens if token and token[0].isalpha())


def looks_like_acronym(label: str) -> bool:
    normalized = normalize_label_basic(label)
    compact = re.sub(r"[^A-Z]", "", normalized)
    if compact != normalized or not (2 <= len(normalized) <= 5):
        return False
    if len(normalized) <= 3:
        return True

    vowel_count = sum(char in "AEIOU" for char in normalized)
    if len(normalized) == 4:
        return vowel_count <= 1
    if len(normalized) == 5:
        return vowel_count == 0
    return False


def token_set(label: str) -> Set[str]:
    return {
        token
        for token in re.split(r"[^A-Z0-9]+", normalize_label_basic(label))
        if token and token not in TOKEN_STOPWORDS and len(token) > 1
    }


def token_list(label: str) -> List[str]:
    return [
        token
        for token in re.split(r"[^A-Z0-9]+", normalize_label_basic(label))
        if token
    ]


def strip_person_titles(label: str) -> str:
    tokens = token_list(label)
    while tokens and tokens[0] in PERSON_TITLE_TOKENS:
        tokens = tokens[1:]
    return " ".join(tokens).strip()


def is_person_name_like(label: str, source_types: Set[str]) -> bool:
    stripped = strip_person_titles(label)
    tokens = token_list(stripped)
    if not tokens or len(tokens) > 5:
        return False
    if "PERSON" not in source_types and not (token_list(label) and token_list(label)[0] in PERSON_TITLE_TOKENS):
        return False
    if "OF" in tokens:
        return False
    if token_set(stripped) & (
        LOCATION_MARKER_TOKENS
        | GOVERNMENT_MARKER_TOKENS
        | ORGANIZATION_MARKER_TOKENS
        | POLITICAL_MARKER_TOKENS
        | MILITARY_MARKER_TOKENS
        | ROLE_MARKER_TOKENS
    ):
        return False
    return True


def person_core_key(label: str, source_types: Set[str]) -> str:
    if not is_person_name_like(label, source_types):
        return ""
    return " ".join(token_list(strip_person_titles(label)))


def directional_signature(label: str) -> Set[str]:
    return {
        DIRECTIONAL_TOKEN_MAP[token]
        for token in token_list(label)
        if token in DIRECTIONAL_TOKEN_MAP
    }


def branch_signature(label: str) -> Set[str]:
    tokens = token_list(label)
    signature = {
        BRANCH_TOKEN_MAP[token]
        for token in tokens
        if token in BRANCH_TOKEN_MAP
    }
    if "AIR" in tokens and "FORCE" in tokens:
        signature.add("AIR")
    return signature


def infer_structural_family(label: str, source_types: Set[str]) -> str:
    tokens = token_set(label)
    normalized_types = {normalize_label_basic(value) for value in source_types if safe_str(value)}

    if "DOCUMENT" in normalized_types:
        return "document"
    if is_person_name_like(label, source_types):
        return "person_name"
    if "COUNTRY" in normalized_types:
        return "country"
    if "GOVERNMENT" in normalized_types:
        return "government"
    if "LOCATION" in normalized_types or tokens & LOCATION_MARKER_TOKENS:
        return "location"
    if "MILITARY" in normalized_types:
        return "military"
    if "POLITICALGROUP" in normalized_types:
        return "political_group"
    if "ORGANIZATION" in normalized_types:
        return "organization"
    if "RELIGIOUSGROUP" in normalized_types:
        return "religious_group"
    if "PERSON" in normalized_types:
        return "person"
    if tokens & GOVERNMENT_MARKER_TOKENS:
        return "government"
    if tokens & MILITARY_MARKER_TOKENS:
        return "military"
    if tokens & POLITICAL_MARKER_TOKENS:
        return "political_group"
    if tokens & ORGANIZATION_MARKER_TOKENS:
        return "organization"
    if tokens & ROLE_MARKER_TOKENS:
        return "role"
    return "other"


def structural_conflict_reason(
    alias_label: str,
    candidate_label: str,
    alias_types: Set[str],
    candidate_types: Set[str],
) -> str:
    alias_family = infer_structural_family(alias_label, alias_types)
    candidate_family = infer_structural_family(candidate_label, candidate_types)

    person_families = {"person", "person_name"}
    entity_families = {"country", "document", "government", "location", "military", "organization", "political_group", "religious_group", "role"}

    if alias_family in person_families and candidate_family in entity_families:
        return "person_vs_nonperson"
    if candidate_family in person_families and alias_family in entity_families:
        return "person_vs_nonperson"

    if {alias_family, candidate_family} == {"country", "government"}:
        return "country_vs_government"
    if {alias_family, candidate_family} == {"location", "government"}:
        return "location_vs_government"
    if {alias_family, candidate_family} == {"location", "country"}:
        return "location_vs_country"
    if {alias_family, candidate_family} == {"organization", "military"}:
        return "organization_vs_military"
    if {alias_family, candidate_family} == {"political_group", "military"}:
        return "political_group_vs_military"
    if {alias_family, candidate_family} == {"location", "military"}:
        return "location_vs_military"

    alias_dirs = directional_signature(alias_label)
    candidate_dirs = directional_signature(candidate_label)
    if alias_dirs and candidate_dirs and alias_dirs != candidate_dirs:
        return "directional_conflict"

    alias_branches = branch_signature(alias_label)
    candidate_branches = branch_signature(candidate_label)
    if alias_branches and candidate_branches and alias_branches != candidate_branches:
        return "branch_conflict"

    return ""


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def build_token_idf_map(catalog: pd.DataFrame) -> Dict[str, float]:
    if len(catalog) == 0:
        return {}

    token_df: Counter[str] = Counter()
    total = int(len(catalog))
    for label in catalog["normalized_label"].fillna("").astype(str):
        seen = token_set(label)
        for token in seen:
            token_df[token] += 1

    idf_map: Dict[str, float] = {}
    for token, df in token_df.items():
        # BM25-style IDF is more stable than raw df counts for short labels.
        idf_map[token] = math.log(1.0 + ((total - df + 0.5) / (df + 0.5)))
    return idf_map


def weighted_token_scores(
    left_tokens: Set[str],
    right_tokens: Set[str],
    token_idf: Dict[str, float],
) -> Tuple[float, float, float]:
    if not left_tokens or not right_tokens:
        return 0.0, 0.0, 0.0

    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens

    def score(tokens: Set[str]) -> float:
        return sum(token_idf.get(token, 1.0) for token in tokens)

    overlap_weight = score(intersection)
    union_weight = score(union)
    left_weight = score(left_tokens)
    right_weight = score(right_tokens)

    weighted_jaccard = overlap_weight / union_weight if union_weight else 0.0
    left_containment = overlap_weight / left_weight if left_weight else 0.0
    right_containment = overlap_weight / right_weight if right_weight else 0.0
    return weighted_jaccard, left_containment, right_containment


def informative_token_gap_penalty(
    left_tokens: Set[str],
    right_tokens: Set[str],
    token_idf: Dict[str, float],
) -> float:
    unmatched = (left_tokens ^ right_tokens) - GENERIC_ENTITY_TOKENS
    if not unmatched:
        return 0.0

    high_signal = [token for token in unmatched if token_idf.get(token, 0.0) >= 1.2]
    if len(high_signal) >= 2:
        return 0.10
    if len(high_signal) == 1:
        return 0.05
    return 0.0


def mean_pairwise_jaccard(token_sets: List[Set[str]]) -> float:
    if len(token_sets) < 2:
        return 1.0

    scores: List[float] = []
    for idx in range(len(token_sets)):
        for jdx in range(idx + 1, len(token_sets)):
            scores.append(jaccard(token_sets[idx], token_sets[jdx]))
    if not scores:
        return 1.0
    return float(sum(scores) / len(scores))


def label_ambiguity_score(
    normalized_label: str,
    source_types: List[str],
    context_token_sets: List[Set[str]],
    top_co_mentions: List[str],
) -> float:
    normalized_types = {
        normalize_label_basic(value)
        for value in source_types
        if safe_str(value)
    }
    family = infer_structural_family(normalized_label, normalized_types)
    coherence = mean_pairwise_jaccard(context_token_sets[:12])
    token_count = len(token_set(normalized_label))
    acronym_like = looks_like_acronym(normalized_label)
    generic_family = family in {"government", "organization", "role", "other", "political_group"}
    country_like = family in {"country", "location"}
    person_like = family in {"person", "person_name"}
    co_count = len([label for label in top_co_mentions if label])
    type_variety = len(normalized_types)

    score = 0.0
    if acronym_like:
        score += 0.30
    if generic_family:
        score += 0.22
    if token_count == 1 and generic_family:
        score += 0.10
    if type_variety >= 2 and not country_like and not person_like:
        score += 0.12

    if acronym_like or generic_family:
        if coherence < 0.12:
            score += 0.18
        elif coherence < 0.20:
            score += 0.10
        if co_count <= 2:
            score += 0.08

    if country_like and type_variety <= 2 and not acronym_like:
        score -= 0.15
    if person_like and not acronym_like:
        score -= 0.10

    return max(0.0, min(1.0, score))


def cooccurrence_jaccard(text_a: str, text_b: str) -> float:
    return jaccard(
        [item.strip() for item in safe_str(text_a).split(";") if item.strip()],
        [item.strip() for item in safe_str(text_b).split(";") if item.strip()],
    )


def type_set_from_text(text: object) -> Set[str]:
    raw = safe_str(text)
    if not raw:
        return set()
    out: Set[str] = set()
    for part in raw.split(";"):
        label = safe_str(part)
        if not label:
            continue
        if ":" in label:
            label = label.split(":", 1)[0]
        normalized = normalize_label_basic(label)
        if normalized:
            out.add(normalized)
    return out


def type_group(type_name: str) -> Optional[int]:
    normalized = normalize_label_basic(type_name)
    for idx, group in enumerate(TYPE_COMPATIBILITY_GROUPS):
        if normalized in group:
            return idx
    return None


def type_compatibility(source_types: Set[str], target_types: Set[str]) -> Tuple[float, str]:
    if not source_types or not target_types:
        return 0.45, "missing_type_info"

    if source_types & target_types:
        return 1.0, "shared_type"

    source_groups = {type_group(value) for value in source_types}
    target_groups = {type_group(value) for value in target_types}
    source_groups.discard(None)
    target_groups.discard(None)

    if source_groups and target_groups and source_groups & target_groups:
        return 0.82, "same_type_group"

    return 0.0, "type_mismatch"


def raw_extraction_dir(project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    legacy_dir = project_dir / DEFAULT_RAW_EXTRACTION_DIRNAME
    if (legacy_dir / "entities.csv").exists():
        return legacy_dir
    return project_dir / "extraction"


def load_entities(project_dir: Path) -> pd.DataFrame:
    path = raw_extraction_dir(project_dir) / "entities.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage A entity output: {path}")

    entities = pd.read_csv(path)
    label_col = first_existing_col(entities, ["label", "name", "entity", "text"])
    if label_col is None:
        raise KeyError(f"Could not find entity label column in {path}. Columns: {entities.columns.tolist()}")

    print("Loaded entities:", entities.shape)
    return entities


def _sample_text_list(values: Iterable[object], limit: int = 5) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = norm_space(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _flatten_semicolon_values(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = norm_space(value)
        if not text:
            continue
        parts = [part.strip() for part in text.split(";")]
        out.extend(part for part in parts if part)
    return out


def profile_output_dir(project_dir: Path) -> Path:
    return Path(project_dir) / DEFAULT_PROFILE_OUTPUT_DIRNAME


def require_stage_csv(project_dir: Path, filename: str) -> pd.DataFrame:
    path = profile_output_dir(project_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Required profile artifact not found: {path}")
    return pd.read_csv(path, low_memory=False)


def write_stage_summary(project_dir: Path, summary: Dict[str, object]) -> None:
    out_dir = profile_output_dir(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / PROFILE_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def truncate_prompt_text(value: object, max_chars: int = 400) -> str:
    text = norm_space(value)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 24:
        return text[:max_chars].rstrip()
    ellipsis = " ...[truncated]... "
    keep = max_chars - len(ellipsis)
    if keep <= 8:
        return text[:max_chars].rstrip()
    head = keep // 2
    tail = keep - head
    return f"{text[:head].rstrip()}{ellipsis}{text[-tail:].lstrip()}"


def _is_context_length_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "context_length_exceeded" in text or "maximum context length" in text


def call_openai_json(client: OpenAI, prompt: str, model: str, max_retries: int = 4) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "developer",
                        "content": "Return only valid JSON matching the requested schema.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as exc:
            last_error = exc
            if _is_context_length_error(exc):
                print(f"OpenAI JSON context-length error: {type(exc).__name__}: {exc}")
                raise
            sleep_seconds = min(60, (2 ** attempt) + random.random())
            print(f"OpenAI JSON error attempt {attempt + 1}/{max_retries}: {type(exc).__name__}: {exc}")
            print(f"Sleeping {sleep_seconds:.1f} seconds...")
            time.sleep(sleep_seconds)
    if last_error is None:
        raise RuntimeError("OpenAI JSON call failed without exception details.")
    raise last_error
