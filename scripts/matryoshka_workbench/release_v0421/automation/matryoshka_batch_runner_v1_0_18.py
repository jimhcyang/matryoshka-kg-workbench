#!/usr/bin/env python3
"""Unattended, resumable batch runner for Matryoshka coordinated test cases.

The runner intentionally contains no corpus-specific scoring logic.  Expected
sources and labels live in the external plan and act only as regression oracles.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CASE_HEARTBEAT_SECONDS = 300.0


def shared_workbench_cache_root(base: Path) -> Path:
    """Return a shared cache root for all workbench runs.

    Cache payloads are query-independent for a given extraction/task pair, so
    they should live in one reusable pool rather than inside each experiment.
    """
    override = os.environ.get("MATRYOSHKA_CACHE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    resolved = base.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "matryoshka_workbench":
            return candidate / ".matryoshka_cache"
    if resolved.name == "runs":
        return resolved.parent / ".matryoshka_cache"
    return resolved.parent / ".matryoshka_cache"


def _set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_set_csv_field_size_limit()


TERMINAL = {
    "COMPLETED",
    "COMPLETED_WITH_WARNINGS",
    "FAILED",
    "PARSER_FAILED",
    "PREFLIGHT_BLOCKED",
    "PAIR_RECONCILIATION_FAILED",
}

SOURCE_COLUMNS = (
    "selected_source_doc_id", "top_candidate_doc_id",
    "top_related_evidence_doc_id", "top_background_doc_id",
    "doc_id", "document_id", "candidate_doc_id", "source_doc_id",
    "corpus_doc_id",
)
AUTHORITATIVE_SOURCE_COLUMNS = ("selected_source_doc_id",)
CONTRADICTION_SOURCE_COLUMNS = (
    "selected_source_doc_id", "top_related_evidence_doc_id",
    "top_candidate_doc_id", "top_background_doc_id",
    "primary_finding_source_doc_id",
    "side_a_support_source_doc_id",
    "contradiction_or_update_source_doc_id",
    "uncertainty_source_doc_id",
    "qualification_or_scope_source_doc_id",
    "related_different_event_source_doc_id",
    "background_only_source_doc_id",
)
LABEL_COLUMNS = (
    "semantic_status", "primary_class", "classification", "decision",
    "status", "support_class", "primary_finding", "evidence_pattern",
)
CONTRADICTION_ROLE_SOURCE_COLUMNS = {
    "side_a_support": "side_a_support_source_doc_id",
    "contradiction_or_update":
        "contradiction_or_update_source_doc_id",
    "uncertainty": "uncertainty_source_doc_id",
    "qualification_or_scope":
        "qualification_or_scope_source_doc_id",
    "related_different_event":
        "related_different_event_source_doc_id",
    "background_only": "background_only_source_doc_id",
}
PROHIBITED_CORPUS_RESOURCE_FIELDS = {
    "answer", "answers", "answer_key", "expected_answer",
    "expected_value", "expected_outcome", "target_label", "case_id",
    "preferred_source", "preferred_source_doc",
    "preferred_source_document", "score_bonus", "scoring_bonus",
    "decision_override", "forced_decision",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON token: " + value)

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )


def strict_json_issues(root: Path) -> list[str]:
    issues: list[str] = []
    for path in sorted(root.glob("**/*.json")):
        try:
            strict_json_load(path)
        except (
            OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError
        ) as exc:
            issues.append(
                f"strict JSON validation failed: {path.name}: {exc}"
            )
    return issues


def score_contract_issues(
    rows: list[dict[str, str]],
) -> list[str]:
    issues: list[str] = []
    for row in rows:
        target = row.get("target_id", "<unknown>")
        best = str(row.get("best_score", "")).strip()
        competitor = str(
            row.get("decision_competitor_score", "")
        ).strip()
        margin = str(row.get("winner_margin", "")).strip()
        status = str(row.get("margin_status", "")).strip()
        if not competitor:
            if margin:
                issues.append(
                    f"{target}: margin present without comparable competitor"
                )
            if status and status != (
                "NOT_APPLICABLE_NO_COMPARABLE_COMPETITOR"
            ):
                issues.append(
                    f"{target}: invalid no-competitor margin status {status}"
                )
            continue
        try:
            expected = float(best) - float(competitor)
            observed = float(margin)
        except (TypeError, ValueError):
            issues.append(
                f"{target}: nonnumeric applicable score contract"
            )
            continue
        if not math.isclose(
            expected, observed, rel_tol=1e-9, abs_tol=1e-9
        ):
            issues.append(
                f"{target}: winner margin does not equal "
                "best minus competitor"
            )
    return issues


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def slug(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(filter(None, safe.split("_")))[:80] or "case"


def normalize_plan_query(value: Any) -> tuple[str, int]:
    """Convert escaped layout markers in plan JSON into real line breaks."""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    repairs = text.count("\\r\\n") + text.count("\\n")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return text.strip(), repairs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_manifest_issues(root: Path) -> list[str]:
    manifest = read_json(
        root / "final_output_manifest_v0420.json", {}
    )
    issues: list[str] = []
    if (
        manifest.get("schema_version")
        != "matryoshka_final_output_manifest_v0_4_20"
    ):
        return ["final output manifest missing or has wrong schema"]
    declared = set()
    for item in manifest.get("files", []):
        relative = str(item.get("path", "")).strip()
        declared.add(relative)
        path = root / relative
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            issues.append("unsafe final manifest path: " + relative)
        elif not path.is_file():
            issues.append("final manifest file missing: " + relative)
        else:
            try:
                expected_size = int(item.get("size_bytes", -1))
            except (TypeError, ValueError):
                expected_size = -1
            if (
                path.stat().st_size != expected_size
                or sha256(path) != str(item.get("sha256", ""))
            ):
                issues.append(
                    "final manifest byte mismatch: " + relative
                )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in {
            "final_output_manifest_v0420.json",
            "final_output_manifest_validation_v0420.json",
        }
        and not any(part.startswith(".") for part in path.parts)
    }
    for relative in sorted(actual - declared):
        issues.append("unsealed output file: " + relative)
    return issues


def corpus_resource_issues(path: Path) -> list[str]:
    payload = read_json(path, None)
    if not isinstance(payload, dict):
        return ["corpus entity resource is not valid JSON object"]
    issues: list[str] = []

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                location = f"{prefix}.{key}" if prefix else str(key)
                if str(key).casefold() in (
                    PROHIBITED_CORPUS_RESOURCE_FIELDS
                ):
                    issues.append(
                        "prohibited answer-key field: " + location
                    )
                visit(child, location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    visit(payload)
    if (
        payload.get("schema_version")
        != "matryoshka_corpus_entity_alias_resource_v1"
    ):
        issues.append("corpus entity resource schema is not v1")
    return issues


def selected_cases(plan: dict[str, Any], tiers: set[int], ids: set[str]) -> list[dict[str, Any]]:
    cases = []
    for case in plan.get("cases", []):
        if not case.get("enabled", True):
            continue
        if tiers and int(case.get("tier", 0)) not in tiers:
            continue
        if ids and str(case.get("id")) not in ids:
            continue
        cases.append(case)
    return sorted(cases, key=lambda x: (int(x.get("tier", 0)), int(x.get("order", 0))))


def parse_extraction_profiles(values: list[str]) -> dict[str, Path]:
    """Parse repeatable ``NAME=/absolute/path`` extraction mappings."""
    profiles: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "Extraction profiles must use NAME=/path syntax: " + value
            )
        name, raw_path = value.split("=", 1)
        name = name.strip().lower()
        if not name or not all(
            character.isalnum() or character in {"_", "-"}
            for character in name
        ):
            raise ValueError("Invalid extraction profile name: " + name)
        if name == "default":
            raise ValueError(
                "The default extraction must be supplied with "
                "--extraction-dir, not --extraction-profile."
            )
        if not raw_path.strip():
            raise ValueError("Extraction profile path is empty: " + name)
        profiles[name] = Path(raw_path).expanduser().resolve()
    return profiles


def parse_resource_profiles(
    values: list[str],
    *,
    label: str,
) -> dict[str, Path]:
    """Parse repeatable ``NAME=/absolute/path`` resource mappings."""
    profiles: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"{label} profiles must use NAME=/path syntax: " + value
            )
        name, raw_path = value.split("=", 1)
        name = name.strip().lower()
        if not name or not all(
            character.isalnum() or character in {"_", "-"}
            for character in name
        ):
            raise ValueError(f"Invalid {label} profile name: " + name)
        if name == "default":
            raise ValueError(
                f"The default {label} must be supplied with "
                f"--{label}, not --{label}-profile."
            )
        if not raw_path.strip():
            raise ValueError(f"{label} profile path is empty: " + name)
        profiles[name] = Path(raw_path).expanduser().resolve()
    return profiles


def resolve_case_extraction(
    case: dict[str, Any],
    default_extraction: Path,
    profiles: dict[str, Path],
) -> tuple[str, Path]:
    """Resolve a case's external corpus profile without corpus logic."""
    profile = str(case.get("extraction_profile", "default")).strip().lower()
    if not profile or profile == "default":
        return "default", default_extraction
    if profile not in profiles:
        raise ValueError(
            f"Case {case.get('id', '<unknown>')} requires extraction "
            f"profile '{profile}'. Supply "
            f"--extraction-profile \"{profile}=/absolute/path\"."
        )
    return profile, profiles[profile]


def resolve_case_resource(
    case: dict[str, Any],
    default_resource: Path | None,
    profiles: dict[str, Path],
    *,
    label: str,
) -> tuple[str, Path | None]:
    """Resolve a case's named resource profile using extraction_profile."""
    profile = str(case.get("extraction_profile", "default")).strip().lower()
    if not profile or profile == "default":
        return "default", default_resource
    if profile in profiles:
        return profile, profiles[profile]
    return profile, default_resource


def extraction_fingerprint(directory: Path) -> str:
    """Create a content-sensitive identity for one extraction directory."""
    digest = hashlib.sha256(str(directory).encode("utf-8"))
    if directory.is_dir():
        for path in sorted(directory.glob("*")):
            if not path.is_file():
                continue
            digest.update(path.name.encode("utf-8"))
            digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def extraction_contains_doc_ids(
    directory: Path,
    required_doc_ids: list[str],
) -> tuple[set[str], list[str]]:
    """Find required IDs in text extraction artifacts before long runs."""
    wanted = {
        str(value).strip().upper()
        for value in required_doc_ids
        if str(value).strip()
    }
    if not wanted or not directory.is_dir():
        return set(), []
    needles = {value: value.encode("utf-8") for value in wanted}
    found: set[str] = set()
    scanned: list[str] = []
    patterns = ("*.csv", "*.jsonl", "*.json", "*.txt")
    candidates = sorted({
        path
        for pattern in patterns
        for path in directory.rglob(pattern)
        if path.is_file()
    })
    for path in candidates:
        scanned.append(str(path))
        try:
            with path.open("rb") as stream:
                tail = b""
                while True:
                    block = stream.read(4 * 1024 * 1024)
                    if not block:
                        break
                    haystack = (tail + block).upper()
                    for value, needle in needles.items():
                        if value not in found and needle in haystack:
                            found.add(value)
                    if found == wanted:
                        return found, scanned
                    tail = haystack[-256:]
        except OSError:
            continue
    return found, scanned


def csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def find_eval_files(run_root: Path) -> Iterable[Path]:
    yield from run_root.glob("**/task_evaluation_v048.csv")


def values(rows: list[dict[str, str]], names: tuple[str, ...]) -> list[str]:
    lowered = {name.lower() for name in names}
    result: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key.lower() in lowered and value:
                result.append(str(value))
    return result


def contradiction_role_payloads(
    rows: list[dict[str, str]],
    role: str,
) -> list[dict[str, Any]]:
    """Read one role from the matcher-owned structured evidence contract."""
    payloads: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("evidence_roles_json", "")
        if not raw:
            continue
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            continue
        payload = decoded.get(role, {}) if isinstance(decoded, dict) else {}
        if isinstance(payload, dict) and payload:
            payloads.append(payload)
    return payloads


def has_nonblank_pseudo_evidence(rows: list[dict[str, str]]) -> bool:
    direct = values(
        rows,
        (
            "text", "component_text", "pseudo_text", "raw_text",
            "pseudo_view_text", "pseudo_chunk_text",
        ),
    )
    if any(value.strip() for value in direct):
        return True
    for payload in values(rows, ("pseudo_component_cards_json",)):
        try:
            cards = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if not isinstance(cards, list):
            continue
        for card in cards:
            if not isinstance(card, dict):
                continue
            if any(
                str(card.get(field, "")).strip()
                for field in (
                    "component_text", "text", "evidence",
                    "pseudo_text", "raw_text",
                )
            ):
                return True
    return False


def environment_preflight(
    python_executable: str,
    code_dir: Path,
    case_extractions: dict[str, tuple[str, Path]],
    cases: list[dict[str, Any]],
    *,
    require_openai_key: bool,
    openai_api_key_env: str,
) -> dict[str, Any]:
    """Check the interpreter and release before any test run is created."""
    required_files = [
        code_dir / "run_coordinated_v0_4_21.py",
        code_dir / "query_semantic_parser_v0_4_15.py",
        code_dir / "external_query_semantic_matcher_v0_4_20.py",
        code_dir / "requirements.txt",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    imports = "import json,sys,pandas,numpy"
    if require_openai_key:
        imports += ",openai"
    issues = []
    versions: dict[str, str] = {}
    try:
        probe = subprocess.run(
            [
                python_executable, "-c",
                (
                    imports + ";"
                    "print(json.dumps({'python':sys.version.split()[0],"
                    "'pandas':pandas.__version__,'numpy':numpy.__version__}))"
                ),
            ],
            text=True,
            capture_output=True,
        )
    except OSError as error:
        probe = None
        issues.append(
            "Selected Python interpreter could not be executed: "
            f"{python_executable}: {error}"
        )
    if probe is not None:
        if probe.returncode:
            issues.append(
                "Dependency import failed for the selected Python "
                "interpreter: "
                + (probe.stderr.strip() or probe.stdout.strip())
            )
        else:
            versions = read_json_text(probe.stdout, {})
            try:
                python_tuple = tuple(
                    int(part)
                    for part in versions.get(
                        "python", "0.0"
                    ).split(".")[:2]
                )
            except ValueError:
                python_tuple = (0, 0)
            if python_tuple < (3, 10):
                issues.append("Python 3.10 or newer is required.")
    if missing:
        issues.append("Missing release files: " + "; ".join(missing))
    extraction_checks: dict[str, Any] = {}
    by_case = {str(case.get("id")): case for case in cases}
    for case_id, (profile, extraction_dir) in case_extractions.items():
        check: dict[str, Any] = {
            "profile": profile,
            "path": str(extraction_dir),
            "present": extraction_dir.is_dir(),
            "fingerprint": extraction_fingerprint(extraction_dir),
            "required_doc_ids": [],
            "found_required_doc_ids": [],
        }
        if not extraction_dir.is_dir():
            issues.append(
                f"Extraction directory not found for case {case_id} "
                f"(profile {profile}): {extraction_dir}"
            )
        else:
            required = [
                str(value).strip().upper()
                for value in by_case.get(case_id, {}).get(
                    "required_doc_ids", []
                )
                if str(value).strip()
            ]
            check["required_doc_ids"] = required
            if required:
                found, scanned = extraction_contains_doc_ids(
                    extraction_dir, required
                )
                check["found_required_doc_ids"] = sorted(found)
                check["files_scanned_for_required_docs"] = len(scanned)
                missing_docs = sorted(set(required) - found)
                if missing_docs:
                    issues.append(
                        f"Case {case_id} extraction profile {profile} is "
                        "missing required document IDs: "
                        + ", ".join(missing_docs)
                    )
        extraction_checks[case_id] = check
    key_name = openai_api_key_env.strip() or "OPENAI_API_KEY"
    key_available = bool(os.getenv(key_name, "").strip())
    if require_openai_key and not key_available:
        issues.append(
            f"OpenAI was requested but environment variable {key_name} is empty."
        )
    return {
        "status": "PASS" if not issues else "FAIL",
        "python_executable": python_executable,
        "versions": versions,
        "required_files_present": not missing,
        "case_extractions": extraction_checks,
        "openai_requested": require_openai_key,
        "openai_key_environment": key_name,
        "openai_key_available": key_available,
        "issues": issues,
    }


def read_json_text(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default


def _is_failed_semantic_status(value: str) -> bool:
    status = str(value).strip().upper()
    return status.startswith((
        "UNRESOLVED", "AMBIGUOUS", "NOT_RECONSTRUCTABLE",
        "FAILED", "ERROR", "NO_CANDIDATE",
    ))


def _oracle_text_key(value: Any) -> str:
    """Normalize layout-only punctuation spacing for strict answer oracles."""
    text = " ".join(str(value or "").strip().split()).upper()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text


def _authoritative_eval_rows(
    case: dict[str, Any],
    eval_files: list[Path],
) -> list[dict[str, str]]:
    """Return rows that determine acceptance.

    A controlled pair qualifies its baseline result while the holdout is
    expected to lose excluded evidence.  Standalone cases qualify every row.
    """
    selected: list[dict[str, str]] = []
    pair = str(case.get("mode", "")).lower() == "pair"
    for path in eval_files:
        if pair and "holdout" in {part.lower() for part in path.parts}:
            continue
        selected.extend(csv_rows(path))
    return selected


def _split_roles(value: str) -> set[str]:
    return {
        item.strip().lower()
        for item in str(value).replace(",", ";").split(";")
        if item.strip()
    }


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_pair_transition(
    expected: dict[str, Any],
    run_root: Path,
) -> list[str]:
    """Validate the reconciled pair contract, not only its baseline arm."""
    if not expected:
        return []
    pair_files = list(run_root.glob("**/pair_comparison_v048.csv"))
    if not pair_files:
        return ["pair_comparison_v048.csv missing"]
    rows = [row for path in pair_files for row in csv_rows(path)]
    if not rows:
        return ["pair comparison contains no rows"]
    unique = {
        (
            row.get("task", ""),
            row.get("target_id", ""),
            row.get("change", ""),
            row.get("baseline_primary_finding_source_doc_id", ""),
            row.get("holdout_primary_finding_source_doc_id", ""),
        ): row
        for row in rows
    }
    rows = list(unique.values())
    issues: list[str] = []
    scalar_fields = {
        "change": "change",
        "baseline_primary_finding": "baseline_primary_finding",
        "baseline_primary_source_id":
            "baseline_primary_finding_source_doc_id",
        "holdout_primary_finding": "holdout_primary_finding",
        "holdout_primary_source_id":
            "holdout_primary_finding_source_doc_id",
        "holdout_relevance_status": "holdout_relevance_status",
    }
    for oracle_name, column in scalar_fields.items():
        oracle = str(expected.get(oracle_name, "")).strip().upper()
        if not oracle:
            continue
        observed = {
            str(row.get(column, "")).strip().upper() for row in rows
        }
        if oracle not in observed:
            issues.append(
                f"pair transition {oracle_name} expected {oracle}; "
                f"observed {', '.join(sorted(observed)) or '<blank>'}"
            )
    if "primary_finding_source_targeted_by_holdout" in expected:
        oracle_bool = bool(
            expected["primary_finding_source_targeted_by_holdout"]
        )
        observed_bools = {
            _as_bool(
                row.get(
                    "primary_finding_source_targeted_by_holdout", ""
                )
            )
            for row in rows
        }
        if oracle_bool not in observed_bools:
            issues.append(
                "pair transition primary-source targeting did not match "
                f"expected {oracle_bool}"
            )
    if "authoritative_source_targeted_by_holdout" in expected:
        oracle_bool = bool(
            expected["authoritative_source_targeted_by_holdout"]
        )
        observed_bools = {
            _as_bool(
                row.get(
                    "authoritative_source_targeted_by_holdout", ""
                )
            )
            for row in rows
        }
        if oracle_bool not in observed_bools:
            issues.append(
                "pair transition authoritative-source targeting did not "
                f"match expected {oracle_bool}"
            )
    for oracle_name, column in (
        ("removed_evidence_roles", "removed_evidence_roles"),
        ("retained_evidence_roles", "retained_evidence_roles"),
        ("added_evidence_roles", "added_evidence_roles"),
    ):
        oracle_roles = {
            str(role).strip().lower()
            for role in expected.get(oracle_name, [])
            if str(role).strip()
        }
        if not oracle_roles:
            continue
        observed_roles: set[str] = set()
        for row in rows:
            observed_roles.update(_split_roles(row.get(column, "")))
        missing = sorted(oracle_roles - observed_roles)
        if missing:
            issues.append(
                f"pair transition missing {oracle_name}: "
                + ", ".join(missing)
            )
    return issues


def validate_case(
    case: dict[str, Any],
    run_root: Path,
    return_code: int,
    *,
    require_llm: bool = False,
) -> tuple[str, list[str]]:
    issues: list[str] = strict_json_issues(run_root)
    status = read_json(run_root / "coordinated_status_v048.json", {})
    final = str(status.get("status", "MISSING_STATUS"))
    if return_code != 0 or final not in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}:
        issues.append(f"terminal status={final}, return_code={return_code}")

    expected = case.get("expected", {})
    if case.get("mode") == "pair" and status.get("pair_status") != "CONTROLLED_PAIR":
        issues.append(f"pair_status={status.get('pair_status', 'missing')}")
    if case.get("mode") == "pair":
        issues.extend(validate_pair_transition(
            case.get("expected_pair_transition", {}), run_root
        ))
    eval_files = list(find_eval_files(run_root))
    if not eval_files:
        issues.append("task_evaluation_v048.csv missing")
    rows = [row for path in eval_files for row in csv_rows(path)]
    issues.extend(score_contract_issues(rows))
    qualifying_rows = _authoritative_eval_rows(case, eval_files)
    if eval_files and not qualifying_rows:
        issues.append("no authoritative evaluation rows found")

    expected_sources = [str(x).upper() for x in expected.get("source_ids", [])]
    if expected_sources:
        source_columns = (
            CONTRADICTION_SOURCE_COLUMNS
            if str(case.get("task", "")).lower() == "contradictions"
            else AUTHORITATIVE_SOURCE_COLUMNS
        )
        observed = [x.upper() for x in values(qualifying_rows, source_columns)]
        for source in expected_sources:
            if not any(source == item for item in observed):
                issues.append(
                    f"expected authoritative source not selected: {source}"
                )
    acceptable_sources_any = [
        str(x).upper()
        for x in expected.get("acceptable_source_ids_any", [])
        if str(x).strip()
    ]
    if acceptable_sources_any:
        source_columns = (
            CONTRADICTION_SOURCE_COLUMNS
            if str(case.get("task", "")).lower() == "contradictions"
            else AUTHORITATIVE_SOURCE_COLUMNS
        )
        observed = [x.upper() for x in values(qualifying_rows, source_columns)]
        if not any(source in observed for source in acceptable_sources_any):
            issues.append(
                "none of the acceptable authoritative sources were selected: "
                + ", ".join(acceptable_sources_any)
            )

    expected_evidence = expected.get("expected_evidence", {})
    if expected_evidence:
        if str(case.get("task", "")).lower() != "contradictions":
            issues.append(
                "expected_evidence is only valid for contradiction cases"
            )
        elif not isinstance(expected_evidence, dict):
            issues.append("expected_evidence must be an object")
        else:
            for role, oracle_value in expected_evidence.items():
                normalized_role = str(role).strip().lower()
                source_column = CONTRADICTION_ROLE_SOURCE_COLUMNS.get(
                    normalized_role
                )
                if not source_column:
                    issues.append(
                        f"unknown contradiction evidence role: {role}"
                    )
                    continue
                oracle = (
                    oracle_value
                    if isinstance(oracle_value, dict)
                    else {"source_id": oracle_value}
                )
                expected_role_sources = [
                    str(item).upper()
                    for item in (
                        oracle.get("source_ids", [])
                        or [oracle.get("source_id", "")]
                    )
                    if str(item).strip()
                ]
                observed_role_sources = [
                    item.upper()
                    for item in values(
                        qualifying_rows, (source_column,)
                    )
                ]
                for source in expected_role_sources:
                    if source not in observed_role_sources:
                        issues.append(
                            "expected contradiction evidence role "
                            f"{normalized_role} did not select source: "
                            f"{source}"
                        )
                role_labels = [
                    str(item).upper()
                    for item in oracle.get("allowed_labels", [])
                ]
                if role_labels:
                    observed_role_labels = [
                        str(payload.get("class", "")).upper()
                        for payload in contradiction_role_payloads(
                            qualifying_rows, normalized_role
                        )
                    ]
                    if not any(
                        allowed in observed
                        for observed in observed_role_labels
                        for allowed in role_labels
                    ):
                        issues.append(
                            "contradiction evidence role "
                            f"{normalized_role} did not contain an allowed "
                            "label: " + ", ".join(role_labels)
                        )

    prohibited_roles = [
        str(role).strip().lower()
        for role in expected.get("prohibited_evidence_roles", [])
        if str(role).strip()
    ]
    if prohibited_roles:
        if str(case.get("task", "")).lower() != "contradictions":
            issues.append(
                "prohibited_evidence_roles is only valid for "
                "contradiction cases"
            )
        for role in prohibited_roles:
            if role not in CONTRADICTION_ROLE_SOURCE_COLUMNS:
                issues.append(
                    f"unknown prohibited contradiction evidence role: {role}"
                )
                continue
            observed_payloads = contradiction_role_payloads(
                qualifying_rows, role
            )
            observed_classes = sorted({
                str(payload.get("class", "")).upper()
                for payload in observed_payloads
                if str(payload.get("class", "")).strip()
            })
            if observed_classes:
                issues.append(
                    "prohibited contradiction evidence role observed "
                    f"{role}: " + ", ".join(observed_classes)
                )

    allowed_labels = [str(x).upper() for x in expected.get("allowed_labels", [])]
    if allowed_labels:
        observed_labels = [x.upper() for x in values(qualifying_rows, LABEL_COLUMNS)]
        if not observed_labels or not any(
            allowed in observed
            for observed in observed_labels
            for allowed in allowed_labels
        ):
            issues.append(
                "none of allowed labels observed: " + ", ".join(allowed_labels)
            )

    prohibited_labels = [str(x).upper() for x in expected.get("prohibited_labels", [])]
    if prohibited_labels:
        observed_labels = [x.upper() for x in values(qualifying_rows, LABEL_COLUMNS)]
        found = sorted({
            prohibited
            for observed in observed_labels
            for prohibited in prohibited_labels
            if prohibited in observed
        })
        if found:
            issues.append("prohibited labels observed: " + ", ".join(found))

    if not expected.get("allow_unresolved", False):
        failed_statuses = sorted({
            value.upper()
            for value in values(qualifying_rows, ("semantic_status",))
            if _is_failed_semantic_status(value)
        })
        if failed_statuses:
            issues.append(
                "unresolved authoritative result: "
                + ", ".join(failed_statuses)
            )

    for target_id, oracle in expected.get("targets", {}).items():
        target_rows = [
            row for row in qualifying_rows
            if str(row.get("target_id", "")).upper() == str(target_id).upper()
        ]
        if not target_rows:
            issues.append(f"expected target missing from evaluation: {target_id}")
            continue
        source = str(oracle.get("source_id", "")).upper()
        if source and source not in [
            item.upper()
            for item in values(target_rows, AUTHORITATIVE_SOURCE_COLUMNS)
        ]:
            issues.append(
                f"{target_id} did not select expected source: {source}"
            )
        expected_value = _oracle_text_key(oracle.get("selected_value", ""))
        if expected_value:
            observed_values = [
                _oracle_text_key(item)
                for item in values(
                    target_rows,
                    ("selected_value_or_text",),
                )
            ]
            if expected_value not in observed_values:
                issues.append(
                    f"{target_id} did not select expected value/text"
                )

    if expected.get("require_visualization", True):
        for output in run_root.glob("**/semantic_match_output"):
            if not (output / "task_visualization_v048.html").exists():
                issues.append(f"visualization missing: {output}")
            visualization = read_json(
                output / "task_visualization_data_v048.json", {}
            )
            if not visualization:
                issues.append(
                    f"visualization data missing or unreadable: {output}"
                )
            else:
                if str(visualization.get("task", "")) != str(
                    case.get("task", "")
                ):
                    issues.append(
                        f"visualization task does not match case: {output}"
                    )
                if not isinstance(visualization.get("evaluation"), list) or not (
                    visualization.get("evaluation")
                ):
                    issues.append(
                        f"visualization has no evaluation series: {output}"
                    )
                if not isinstance(
                    visualization.get("target_balanced_candidates"), list
                ):
                    issues.append(
                        f"visualization candidate series is absent: {output}"
                    )

    clause_ledger_path = (
        run_root / "query_clause_execution_ledger_v0415.csv"
    )
    clause_rows = csv_rows(clause_ledger_path)
    if not clause_rows:
        if case.get("mode") != "pair":
            issues.append("query clause execution ledger missing")
    else:
        nonexecuted = [
            row for row in clause_rows
            if str(row.get("is_substantive", "")).strip().lower()
            in {"1", "true", "yes", "y"}
            and str(row.get("execution_state", "")) not in {
                "EXECUTED", "EXECUTED_TARGET", "CONTEXT_CONSUMED",
            }
        ]
        if nonexecuted:
            issues.append(
                "substantive query clauses were not executed: "
                + ", ".join(
                    str(row.get("source_clause_id", ""))
                    for row in nonexecuted
                )
            )

    if expected.get("require_nonblank_pseudo_text"):
        pseudo_files = list(run_root.glob("**/*pseudo*.csv"))
        pseudo_files.extend(
            run_root.glob("**/fragment_source_passages_v048.csv")
        )
        if not pseudo_files:
            issues.append("pseudo-component table missing")
        else:
            pseudo_rows = [r for p in pseudo_files for r in csv_rows(p)]
            if not has_nonblank_pseudo_evidence(pseudo_rows):
                issues.append("pseudo-component text is blank")

    for output in run_root.glob("**/semantic_match_output"):
        matcher_manifest = read_json(
            output / "semantic_match_run_manifest_v048.json", {}
        )
        is_v0420_output = (
            matcher_manifest.get("matcher_implementation_version")
            == "v0.4.20"
        )
        if is_v0420_output and not (
            output / "semantic_resource_manifest_v0420.json"
        ).is_file():
            issues.append(f"semantic resource manifest missing: {output}")
        if is_v0420_output and not (
            output / "corpus_entity_resource_manifest_v0420.json"
        ).is_file():
            issues.append(
                f"corpus entity resource manifest missing: {output}"
            )
        if is_v0420_output:
            issues.extend(
                f"{output}: {issue}"
                for issue in final_manifest_issues(output)
            )
        if is_v0420_output and not (
            output / "candidate_accounting_v0420.csv"
        ).is_file():
            issues.append(f"candidate accounting missing: {output}")
        if is_v0420_output:
            accounting_rows = csv_rows(
                output / "candidate_accounting_v0420.csv"
            )
            for row in accounting_rows:
                for field in (
                    "candidate_row_count", "eligible_candidate_count",
                    "exported_audit_row_count",
                ):
                    value = str(row.get(field, "")).strip()
                    if not value or not re.fullmatch(r"\d+", value):
                        issues.append(
                            f"candidate accounting {field} is not a "
                            f"nonnegative integer: {output}"
                        )
                try:
                    if int(row["exported_audit_row_count"]) > int(
                        row["candidate_row_count"]
                    ):
                        issues.append(
                            f"exported audit exceeds candidate population: "
                            f"{output}"
                        )
                except (KeyError, TypeError, ValueError):
                    pass
            summary_path = output / "audit_export_summary_v0420.csv"
            if not summary_path.is_file():
                issues.append(f"audit export summary missing: {output}")
            summary_rows = csv_rows(summary_path)
            target_ids = {
                str(row.get("target_id", ""))
                for row in csv_rows(
                    output / "task_evaluation_v048.csv"
                )
                if str(row.get("target_id", ""))
            }
            summary_targets = {
                str(row.get("target_id", ""))
                for row in summary_rows
                if str(row.get("target_id", ""))
            }
            if target_ids != summary_targets:
                issues.append(
                    "audit export summary must contain exactly one "
                    f"target row per task target: {output}"
                )
            for row in summary_rows:
                try:
                    original = int(row["original_row_count"])
                    exported = int(row["exported_row_count"])
                    omitted = int(row["omitted_row_count"])
                    limit = int(row["row_limit_per_target"])
                    if exported > limit or original != exported + omitted:
                        issues.append(
                            f"audit export accounting is inconsistent: "
                            f"{output}"
                        )
                except (KeyError, TypeError, ValueError):
                    issues.append(
                        f"audit export accounting is unreadable: {output}"
                    )
        if (
            is_v0420_output
            and str(case.get("task", "")) == "contradictions"
        ):
            for filename in (
                "contradiction_document_class_trace_v0420.csv",
                "contradiction_collapse_trace_v0420.csv",
                "candidate_budget_audit_v0420.csv",
            ):
                if not (output / filename).is_file():
                    issues.append(
                        f"contradiction diagnostic missing: {filename}"
                    )
        evidence_path = output / "task_evidence_units_v0413.json"
        evidence = read_json(evidence_path, {})
        if not evidence:
            issues.append(f"v0.4.13 evidence units missing: {output}")
            continue
        if evidence.get("target_coverage_state") != "PASS":
            issues.append(
                "evidence target coverage failed: "
                + ", ".join(evidence.get("missing_target_ids", []))
            )
        query_clause_state = str(
            evidence.get("query_clause_coverage_state", "missing")
        )
        if query_clause_state != "PASS":
            if not (
                case.get("mode") == "pair"
                and query_clause_state in {"NOT_ASSESSED", "NOT_APPLICABLE"}
            ):
                issues.append(
                    "evidence query-clause coverage is not PASS: "
                    + query_clause_state
                )
        requested = [
            str(value) for value in evidence.get("requested_target_ids", [])
        ]
        emitted = [
            str(value) for value in evidence.get("emitted_target_ids", [])
        ]
        if requested != emitted:
            issues.append(
                "evidence target order/set differs from frozen parse"
            )
        unit_ids = [
            str(unit.get("target_id", ""))
            for unit in evidence.get("units", [])
        ]
        if unit_ids != requested:
            issues.append("one evidence unit per requested target was not met")
        for unit in evidence.get("units", []):
            seen_excerpts: set[str] = set()
            for fragment in unit.get("evidence_fragments", []):
                excerpt = str(fragment.get("excerpt", ""))
                limit = int(
                    fragment.get("excerpt_character_limit", 0) or 0
                )
                if limit and len(excerpt) > limit:
                    issues.append(
                        "evidence excerpt exceeds declared limit: "
                        + str(unit.get("target_id", ""))
                    )
                normalized = " ".join(excerpt.casefold().split())
                # Repeated excerpts in adjacent evidence fragments are noisy but
                # not a case-level correctness failure. Keep validating excerpt
                # length and target coverage, but do not fail the run solely
                # because the same normalized snippet appears twice.
                seen_excerpts.add(normalized)
        digest_path = output / "semantic_match_task_digest_v0413.md"
        digest = (
            digest_path.read_text(encoding="utf-8", errors="ignore")
            if digest_path.is_file() else ""
        )
        if not digest:
            issues.append(f"v0.4.13 deterministic digest missing: {output}")
        for target_id in requested:
            if f"## TARGET {target_id}" not in digest:
                issues.append(
                    f"deterministic digest missing target: {target_id}"
                )
        if str(case.get("task", "")) == "reconstruction":
            passage_path = (
                output / "reconstruction_candidate_passages_v0413.csv"
            )
            if not passage_path.is_file():
                issues.append(
                    "reconstruction candidate passage export missing"
                )
    if case.get("mode") == "pair":
        route_delta_path = (
            run_root / "pair_candidate_route_delta_v0420.csv"
        )
        route_delta_rows = csv_rows(route_delta_path)
        if not route_delta_rows:
            issues.append(
                "pair candidate route-delta accounting missing"
            )
        for row in route_delta_rows:
            if row.get("accounting_state") != "RECONCILED":
                issues.append(
                    "pair candidate route-delta accounting failed: "
                    + str(row.get("target_id", ""))
                )
        pair_rows = [
            row
            for path in run_root.glob("**/pair_comparison_v048.csv")
            for row in csv_rows(path)
        ]
        for row in pair_rows:
            if (
                row.get("holdout_relevance_status")
                == "BASELINE_SOURCE_TARGETED"
                and not _as_bool(
                    row.get(
                        "authoritative_source_targeted_by_holdout",
                        False,
                    )
                )
            ):
                issues.append(
                    "pair authority targeting boolean disagrees with "
                    "holdout relevance status"
                )
    if require_llm:
        if str(case.get("mode", "")) == "pair":
            llm_states = [
                read_json(
                    run_root / "pair_llm_digest_state_v0413.json", {}
                )
            ]
        else:
            llm_states = [
                read_json(path, {})
                for path in run_root.glob(
                    "**/llm_digest_state_v0413.json"
                )
            ]
        if not llm_states or any(
            state.get("state") != "VALIDATED_EXPLANATORY"
            for state in llm_states
        ):
            observed = ", ".join(
                str(state.get("state", "MISSING"))
                for state in llm_states
            ) or "MISSING"
            issues.append(
                "Optional LLM digest failed explanatory acceptance: "
                + observed
            )
    require_final_manifest = any(
        read_json(
            output / "semantic_match_run_manifest_v048.json", {}
        ).get("matcher_implementation_version") == "v0.4.20"
        for output in run_root.glob("**/semantic_match_output")
    )
    if require_final_manifest:
        issues.extend(
            "coordinated root: " + issue
            for issue in final_manifest_issues(run_root)
        )

    return ("PASS" if not issues else "FAIL"), issues


def write_summary(output_root: Path, records: list[dict[str, Any]]) -> None:
    atomic_json(output_root / "batch_summary.json", {"generated_at": now_iso(), "cases": records})
    fields = [
        "id", "tier", "task", "mode", "status", "return_code",
        "elapsed_seconds", "extraction_profile", "extraction_dir",
        "extraction_fingerprint", "run_root", "issues",
    ]
    with (output_root / "batch_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})
    lines = [
        "# Matryoshka batch test summary",
        "",
        f"Generated: {now_iso()}",
        "",
        "| Case | Tier | Task | Mode | Result | Minutes |",
        "|---|---:|---|---|---|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['id']} | {record['tier']} | {record['task']} | "
            f"{record['mode']} | {record['status']} | "
            f"{float(record.get('elapsed_seconds', 0)) / 60:.1f} |"
        )
    lines.extend(["", "## Issues", ""])
    for record in records:
        if record.get("issues"):
            lines.append(f"- **{record['id']}** — {record['issues']}")
    (output_root / "batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip(output_root: Path) -> Path:
    archive = output_root / "matryoshka_batch_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path != archive:
                bundle.write(path, path.relative_to(output_root))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--extraction-dir", required=True)
    parser.add_argument(
        "--extraction-profile",
        action="append",
        default=[],
        metavar="NAME=/ABSOLUTE/PATH",
        help=(
            "Repeatable named extraction mapping used by cases that declare "
            "extraction_profile in the test plan."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--tiers", default="", help="Comma-separated tiers; blank means all.")
    parser.add_argument("--cases", default="", help="Comma-separated case IDs; blank means all.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate interpreter, dependencies, extraction profiles, and "
            "required document IDs without executing a case."
        ),
    )
    parser.add_argument("--use-openai-parser", action="store_true")
    parser.add_argument("--use-llm-digest", action="store_true")
    parser.add_argument(
        "--parser-model", default="gpt-4o",
        help=(
            "OpenAI Parser model passed unchanged to the coordinator. "
            "The value is ignored unless --use-openai-parser is selected."
        ),
    )
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-elapsed-seconds", type=float, default=0.0)
    parser.add_argument(
        "--semantic-resources",
        default="",
        help="Optional semantic-resource JSON passed unchanged to every case.",
    )
    parser.add_argument(
        "--corpus-entity-resource",
        default="",
        help=(
            "Optional versioned corpus entity/alias resource v1 passed "
            "unchanged to every case."
        ),
    )
    parser.add_argument(
        "--corpus-entity-resource-profile",
        action="append",
        default=[],
        metavar="NAME=/ABSOLUTE/PATH",
        help=(
            "Repeatable named corpus entity/alias resource mapping used by "
            "cases that declare extraction_profile in the test plan."
        ),
    )
    parser.add_argument(
        "--decision-profile",
        default="",
        help="Optional decision-profile JSON passed unchanged to every case.",
    )
    parser.add_argument("--minimum-score", type=float, default=None)
    parser.add_argument("--minimum-margin", type=float, default=None)
    parser.add_argument(
        "--contradiction-propositions-per-class-per-document",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--contradiction-raw-sentences-per-class-per-document",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--shared-source-cache", default="",
        help="Persistent source-table cache; default is beside output roots.",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan).expanduser().resolve()
    if args.semantic_resources and not Path(
        args.semantic_resources
    ).expanduser().is_file():
        raise SystemExit(
            "Semantic resource file not found: " + args.semantic_resources
        )
    if args.decision_profile and not Path(
        args.decision_profile
    ).expanduser().is_file():
        raise SystemExit(
            "Decision profile file not found: " + args.decision_profile
        )
    code_dir = Path(args.code_dir).expanduser().resolve()
    extraction_dir = Path(args.extraction_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = read_json(plan_path, {})
    if str(plan.get("schema_version", "")) != "1.0.18":
        raise SystemExit(
            "Automation v1.0.18 requires test-plan schema 1.0.18."
        )
    tiers = {int(x) for x in args.tiers.split(",") if x.strip()}
    ids = {x.strip() for x in args.cases.split(",") if x.strip()}
    cases = selected_cases(plan, tiers, ids)
    if not cases:
        raise SystemExit("No enabled cases matched the requested selection.")
    try:
        extraction_profiles = parse_extraction_profiles(
            args.extraction_profile
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        corpus_entity_resource_profiles = parse_resource_profiles(
            args.corpus_entity_resource_profile,
            label="corpus-entity-resource",
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    plan_snapshot = output_root / "test_plan_snapshot.json"
    if not plan_snapshot.exists():
        shutil.copy2(plan_path, plan_snapshot)
    selection_counts: dict[tuple[str, str], int] = {}
    for case in cases:
        profile = str(
            case.get("extraction_profile", "default")
        ).strip().lower() or "default"
        key = (profile, str(case.get("task", "unknown")))
        selection_counts[key] = selection_counts.get(key, 0) + 1
    selection_summary = ", ".join(
        f"{profile}/{task}={case_count}"
        for (profile, task), case_count in sorted(selection_counts.items())
    )
    print(f"Selected {len(cases)} cases: {selection_summary}")
    if args.plan_only:
        print("PLAN ONLY: no parser, matcher, or corpus test was executed.")
        print("Remove --plan-only to run the selected cases.")
        return 0
    try:
        case_extractions = {
            str(case["id"]): resolve_case_extraction(
                case, extraction_dir, extraction_profiles
            )
            for case in cases
        }
    except ValueError as error:
        print("PRECHECK FAILED:", error, file=sys.stderr)
        return 4

    coordinator = code_dir / "run_coordinated_v0_4_21.py"
    default_corpus_entity_resource = (
        Path(args.corpus_entity_resource).expanduser().resolve()
        if args.corpus_entity_resource else None
    )
    resource_paths_to_check: dict[str, Path] = {}
    if default_corpus_entity_resource is not None:
        resource_paths_to_check["default"] = default_corpus_entity_resource
    resource_paths_to_check.update(corpus_entity_resource_profiles)
    for profile_name, resource_path in resource_paths_to_check.items():
        resource_issues = (
            corpus_resource_issues(resource_path)
            if resource_path.is_file()
            else ["corpus entity resource file not found"]
        )
        if resource_issues:
            for issue in resource_issues:
                print(
                    "PRECHECK FAILED: "
                    + issue
                    + f" [profile={profile_name}]",
                    file=sys.stderr,
                )
            return 4
    preflight = environment_preflight(
        args.python_executable,
        code_dir,
        case_extractions,
        cases,
        require_openai_key=args.use_openai_parser or args.use_llm_digest,
        openai_api_key_env=args.openai_api_key_env,
    )
    atomic_json(output_root / "batch_environment_preflight.json", preflight)
    if preflight["status"] != "PASS":
        for issue in preflight["issues"]:
            print("PRECHECK FAILED:", issue, file=sys.stderr)
        print(
            "Install with the same interpreter: "
            f"{args.python_executable} -m pip install -r "
            f"{code_dir / 'requirements.txt'}",
            file=sys.stderr,
        )
        return 4
    if args.preflight_only:
        print(
            "PREFLIGHT ONLY: environment, extraction profiles, and required "
            "document checks passed; no case was executed."
        )
        return 0

    explicit_shared_cache = (
        Path(args.shared_source_cache).expanduser().resolve()
        if args.shared_source_cache else None
    )

    # Cache files are query-independent for a given extraction+task pair.
    # Keep them in one shared workbench cache so future experiments,
    # including original/rephrased A/B runs, can reuse the same payloads
    # without bloating individual experiment folders.
    cache_root = shared_workbench_cache_root(output_root)

    def extraction_cache_identity(case_extraction: Path, task: str) -> str:
        """Return a relocation-safe cache identity.

        Content signatures are validated before reuse, so the package-local KG
        project name is sufficient for lookup and remains stable when the
        handoff directory is moved or renamed.
        """
        project_name = case_extraction.parent.name
        extraction_name = case_extraction.name
        return hashlib.sha256(
            (project_name + "/" + extraction_name + "|" + task).encode("utf-8")
        ).hexdigest()[:16]

    def shared_cache_for(case_extraction: Path, task: str) -> Path:
        identity = extraction_cache_identity(case_extraction, task)
        if explicit_shared_cache:
            return explicit_shared_cache.with_name(
                explicit_shared_cache.stem + "_" + task + "_" + identity
                + explicit_shared_cache.suffix
            )
        return cache_root / (identity + f"_{task}_source_tables_v0420.pkl")

    def feature_cache_for(case_extraction: Path, task: str) -> Path:
        identity = extraction_cache_identity(case_extraction, task)
        return cache_root / (identity + f"_{task}_features_v0420.pkl")

    def query_parse_cache_for(
        query_text: str,
        parser_model: str,
        use_openai_parser: bool,
    ) -> Path:
        query_sha = hashlib.sha256(
            query_text.encode("utf-8")
        ).hexdigest()[:16]
        parser_mode = parser_model if use_openai_parser else "heuristic"
        identity = hashlib.sha256(
            (query_sha + "|" + parser_mode + "|" + str(int(use_openai_parser))).encode("utf-8")
        ).hexdigest()[:16]
        return cache_root / "query_parse" / f"{identity}_{query_sha}"

    records: list[dict[str, Any]] = []
    state_path = output_root / "batch_state.json"
    previous = read_json(state_path, {}).get("cases", {}) if args.resume else {}
    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        extraction_profile, case_extraction_dir = case_extractions[case_id]
        _, case_corpus_entity_resource = resolve_case_resource(
            case,
            default_corpus_entity_resource,
            corpus_entity_resource_profiles,
            label="corpus-entity-resource",
        )
        case_extraction_fingerprint = extraction_fingerprint(
            case_extraction_dir
        )
        case_task = str(case["task"])
        shared_cache = shared_cache_for(case_extraction_dir, case_task)
        feature_cache = feature_cache_for(case_extraction_dir, case_task)
        case_root = output_root / f"{int(case.get('order', index)):03d}_{slug(case_id)}"
        prior = previous.get(case_id, {})
        resume_contract_matches = (
            prior.get("extraction_profile") == extraction_profile
            and prior.get("extraction_dir") == str(case_extraction_dir)
            and prior.get("extraction_fingerprint")
            == case_extraction_fingerprint
            and prior.get("case_contract_sha256")
            == hashlib.sha256(
                json.dumps(
                    case, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
        )
        if (
            args.resume
            and prior.get("status") == "PASS"
            and case_root.exists()
            and resume_contract_matches
        ):
            records.append(prior)
            print(f"[{index}/{len(cases)}] {case_id}: already PASS; skipped")
            continue
        if (
            args.resume
            and prior.get("status") == "PASS"
            and not resume_contract_matches
        ):
            print(
                f"[{index}/{len(cases)}] {case_id}: prior PASS does not "
                "match the current case/extraction contract; rerunning"
            )

        query_dir = output_root / "_queries"
        query_dir.mkdir(exist_ok=True)
        query_path = query_dir / f"{slug(case_id)}.txt"
        query_text, query_layout_repairs = normalize_plan_query(case["query"])
        query_path.write_text(query_text + "\n", encoding="utf-8")
        query_parse_cache = query_parse_cache_for(
            query_text,
            args.parser_model,
            args.use_openai_parser,
        )
        command = [
            args.python_executable, str(coordinator), str(case["mode"]),
            "--query-file", str(query_path),
            "--extraction-dir", str(case_extraction_dir),
            "--task", str(case["task"]),
            "--run-root", str(case_root),
            "--code-dir", str(code_dir),
            "--python-executable", args.python_executable,
            "--staged-top-docs", str(case.get("staged_top_docs", 150)),
            "--retrieval-mode", str(case.get("retrieval_mode", "staged")),
            "--raw-use-mode", str(case.get("raw_use_mode", "validate")),
            "--max-elapsed-seconds", str(
                case.get("max_elapsed_seconds", args.max_elapsed_seconds)
            ),
            "--llm-model", args.llm_model,
            "--parser-model", args.parser_model,
            "--openai-api-key-env", args.openai_api_key_env,
            "--shared-source-cache", str(shared_cache),
            "--task-feature-cache", str(feature_cache),
            "--query-parse-cache", str(query_parse_cache),
            "--contradiction-propositions-per-class-per-document",
            str(args.contradiction_propositions_per_class_per_document),
            "--contradiction-raw-sentences-per-class-per-document",
            str(args.contradiction_raw_sentences_per_class_per_document),
        ]
        if args.semantic_resources:
            command += [
                "--semantic-resources",
                str(Path(args.semantic_resources).expanduser().resolve()),
            ]
        if case_corpus_entity_resource is not None:
            command += [
                "--corpus-entity-resource",
                str(case_corpus_entity_resource),
            ]
        if args.decision_profile:
            command += [
                "--decision-profile",
                str(Path(args.decision_profile).expanduser().resolve()),
            ]
        if args.minimum_score is not None:
            command += ["--minimum-score", str(args.minimum_score)]
        if args.minimum_margin is not None:
            command += ["--minimum-margin", str(args.minimum_margin)]
        targets = case.get("target_doc_ids", [])
        if targets:
            command += ["--target-doc-ids", ",".join(map(str, targets))]
        exclusions = case.get("exclude_doc_ids", [])
        if exclusions:
            command += ["--exclude-doc-ids", ",".join(map(str, exclusions))]
        if args.use_openai_parser:
            command.append("--use-openai-parser")
        if args.use_llm_digest:
            command.append("--use-llm-digest")

        if case_root.exists():
            # A failed/incomplete coordinated root cannot be reused safely.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            case_root = case_root.with_name(case_root.name + "_retry_" + timestamp)
            command[command.index("--run-root") + 1] = str(case_root)
        print(f"[{index}/{len(cases)}] {case_id}: starting")
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=code_dir,
            env=os.environ.copy(),
        )
        while True:
            try:
                return_code = process.wait(timeout=CASE_HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                elapsed_now = time.monotonic() - started
                print(
                    f"[{index}/{len(cases)}] {case_id}: still running after {elapsed_now:.1f}s",
                    flush=True,
                )
        elapsed = round(time.monotonic() - started, 3)
        status, issues = validate_case(
            case,
            case_root,
            return_code,
            require_llm=args.use_llm_digest,
        )
        record = {
            "id": case_id,
            "tier": int(case.get("tier", 0)),
            "task": case["task"],
            "mode": case["mode"],
            "status": status,
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "extraction_profile": extraction_profile,
            "extraction_dir": str(case_extraction_dir),
            "extraction_fingerprint": case_extraction_fingerprint,
            "run_root": str(case_root),
            "issues": "; ".join(issues),
            "query_sha256": sha256(query_path),
            "query_layout_repairs": query_layout_repairs,
            "case_contract_sha256": hashlib.sha256(
                json.dumps(
                    case, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "completed_at": now_iso(),
        }
        records.append(record)
        previous[case_id] = record
        atomic_json(state_path, {"updated_at": now_iso(), "cases": previous})
        write_summary(output_root, records)
        print(f"[{index}/{len(cases)}] {case_id}: {status} in {elapsed / 60:.1f} min")
        if status == "FAIL" and args.stop_on_failure:
            print("Stopping at first failed gate.")
            break

    write_summary(output_root, records)
    archive = make_zip(output_root)
    print(f"Consolidated archive: {archive}")
    return 0 if records and all(r["status"] == "PASS" for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
