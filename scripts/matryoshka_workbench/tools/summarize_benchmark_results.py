#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKBENCH_ROOT / "release_v0421"))

from matryoshka_v0420.semantics import (  # noqa: E402
    build_event_signature,
    canonical_key,
    compare_argument,
)


SUCCESS_PREFIXES = {
    "reconstruction": ("reconstructable",),
    "claims": ("direct_support",),
    "value_redaction": ("resolved",),
    "sentence_redaction": ("resolved",),
    "contradictions": (
        "direct_opposition",
        "direct_numerical_contradiction",
        "attribute_contradiction",
        "location_contradiction",
        "side_a_support",
        "side_b_support",
    ),
}

TASK_ORDER = [
    "reconstruction",
    "claims",
    "value_redaction",
    "sentence_redaction",
    "contradictions",
]

CORPUS_ORDER = [
    "viet",
    "cib",
    "iran",
    "fisc",
    "canb",
]

CORPUS_NORMALIZATION = {
    "canberra": "canb",
    "canb": "canb",
    "fisc": "fisc",
    "iran": "iran",
    "cib": "cib",
    "vietnam": "viet",
    "viet": "viet",
}
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = " ".join(str(value).strip().split())
    return "" if text.casefold() in {"nan", "none", "<na>"} else text


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_space(value).lower()
    return text in {"1", "true", "yes", "y"}


def strip_parenthetical_suffix(text: Any) -> str:
    value = normalize_space(text)
    return re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()


def benchmark_truth_variants(value: Any) -> list[str]:
    raw = strip_parenthetical_suffix(value)
    if not raw:
        return []
    variants: list[str] = [raw]
    parts = [
        normalize_space(part)
        for part in re.split(r"\s*[|;]\s*", raw)
        if normalize_space(part)
    ]
    for part in parts:
        match = re.match(r"^[A-Z]\d+=(.+?)(?:@[^@]+)?$", part)
        if match:
            extracted = strip_parenthetical_suffix(match.group(1))
            if extracted:
                variants.append(extracted)
        elif "=" in part:
            _, rhs = part.split("=", 1)
            rhs = strip_parenthetical_suffix(rhs)
            if "@" in rhs:
                rhs = strip_parenthetical_suffix(rhs.split("@", 1)[0])
            if rhs:
                variants.append(rhs)
        else:
            variants.append(part)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        normalized = strip_parenthetical_suffix(item)
        key = normalized.casefold()
        if normalized and key not in seen:
            deduped.append(normalized)
            seen.add(key)
    return deduped


def relaxed_truth_match(truth: Any, predicted: Any) -> bool:
    pred_variants = benchmark_truth_variants(predicted)
    if not pred_variants:
        return False
    for gold in benchmark_truth_variants(truth):
        gold_cf = gold.casefold()
        if not gold_cf:
            continue
        for pred in pred_variants:
            pred_cf = pred.casefold()
            if not pred_cf:
                continue
            if gold_cf == pred_cf:
                return True
            if gold_cf in pred_cf or pred_cf in gold_cf:
                return True
            gold_tokens = {token.casefold() for token in WORD_RE.findall(gold_cf)}
            pred_tokens = {token.casefold() for token in WORD_RE.findall(pred_cf)}
            if not gold_tokens or not pred_tokens:
                continue
            overlap = len(gold_tokens & pred_tokens) / max(1, len(gold_tokens))
            if overlap >= 0.8:
                return True
            if semantic_proposition_truth_match(gold, pred):
                return True
    return False


def semantic_proposition_truth_match(truth: Any, predicted: Any) -> bool:
    """Match grounded proposition paraphrases without accepting topic overlap."""
    truth_signature = build_event_signature(normalize_space(truth))
    predicted_signature = build_event_signature(normalize_space(predicted))
    truth_predicate = canonical_key(truth_signature.get("predicate", ""))
    predicted_predicate = canonical_key(predicted_signature.get("predicate", ""))
    generic_predicates = {"", "be", "do", "have", "say"}
    if (
        truth_predicate in generic_predicates
        or predicted_predicate in generic_predicates
        or compare_argument(truth_predicate, predicted_predicate) != "MATCH"
    ):
        return False
    if (
        canonical_key(truth_signature.get("assertion_polarity", ""))
        != canonical_key(predicted_signature.get("assertion_polarity", ""))
    ):
        return False
    for dimension in ("actor", "object"):
        truth_value = truth_signature.get(dimension, "")
        predicted_value = predicted_signature.get(dimension, "")
        if not normalize_space(truth_value) or not normalize_space(predicted_value):
            return False
        if compare_argument(truth_value, predicted_value) != "MATCH":
            return False
    return True


def first_runtime_eval_path(run_root: Path, run_layout: str, arm: str) -> Path | None:
    if run_layout == "pair":
        base = run_root / arm / "semantic_match_output"
    else:
        if arm != "baseline":
            return None
        base = run_root / "semantic_match_output"
    if not base.exists():
        return None
    for pattern in (
        "*_evaluation_v048.csv",
        "*_evaluation_v043.csv",
        "task_evaluation_v048.csv",
        "task_evaluation_v043.csv",
    ):
        matches = sorted(base.glob(pattern))
        if matches:
            if len(matches) > 1:
                non_generic = [match for match in matches if not match.name.startswith("task_")]
                if non_generic:
                    return non_generic[0]
            return matches[0]
    return None


def runtime_output_dir(run_root: Path, run_layout: str, arm: str) -> Path | None:
    if run_layout == "pair":
        output = run_root / arm / "semantic_match_output"
    elif arm == "baseline":
        output = run_root / "semantic_match_output"
    else:
        return None
    return output if output.exists() else None


def selected_evidence_truth_match(output: Path | None, truth: Any) -> bool:
    """Return a conservative textual ground-truth/evidence equivalence proxy."""
    if output is None:
        return False
    path = output / "task_evidence_units_v0413.json"
    if not path.exists() or not benchmark_truth_variants(truth):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    decision_roles = {
        "selected", "supporting", "refuting", "support", "opposition",
    }
    for unit in payload.get("units", []):
        for fragment in unit.get("evidence_fragments", []):
            role = normalize_space(fragment.get("evidence_role", "")).casefold()
            if role not in decision_roles:
                continue
            for candidate in (
                fragment.get("candidate_value_or_text", ""),
                fragment.get("excerpt", ""),
            ):
                if relaxed_truth_match(truth, candidate):
                    return True
    return False


def load_runtime_eval_row(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return {}
    if frame.empty:
        return {}
    rows = frame.to_dict("records")
    task_names = {
        normalize_space(row.get("task", "")).casefold()
        for row in rows
        if normalize_space(row.get("task", ""))
    }
    if task_names == {"redaction"} and any(
        normalize_space(row.get("target_id", "")) for row in rows
    ):
        resolved = [
            normalize_space(row.get("semantic_status", "")).casefold().startswith("resolved")
            and bool(normalize_space(row.get("selected_value_or_text", "")))
            for row in rows
        ]
        all_resolved = all(resolved)
        row = next(
            (item for item, is_resolved in zip(rows, resolved) if not is_resolved),
            rows[0],
        ).copy()
        row["all_targets_resolved"] = all_resolved
        row["target_count"] = len(rows)
        row["resolved_target_count"] = sum(resolved)
        row["selected_value_or_text"] = " | ".join(
            f"{normalize_space(item.get('target_id', ''))}="
            f"{normalize_space(item.get('selected_value_or_text', ''))}"
            for item, is_resolved in zip(rows, resolved)
            if is_resolved
        )
        if len(rows) > 1:
            row["semantic_status"] = (
                "RESOLVED_ALL_TARGETS" if all_resolved else "UNRESOLVED_PARTIAL_TARGETS"
            )
        elif not all_resolved and normalize_space(
            row.get("semantic_status", "")
        ).casefold().startswith("resolved"):
            row["semantic_status"] = "UNRESOLVED_MISSING_SELECTED_VALUE"
        row["all_decision_source_doc_ids"] = " | ".join(dict.fromkeys(
            normalize_space(item.get("selected_source_doc_id", ""))
            for item in rows
            if normalize_space(item.get("selected_source_doc_id", ""))
        ))
        return {str(key): row.get(key, "") for key in row.keys()}

    def row_score(row: dict[str, Any]) -> tuple[float, float, float]:
        task = normalize_space(row.get("task", "") or row.get("task_family", ""))
        success = 1.0 if semantic_success(task, row.get("semantic_status", "")) else 0.0
        localized = 1.0 if normalize_space(
            row.get("localization_status", "")
        ) == "AUTHORITATIVE_SOURCE_SELECTED" else 0.0
        try:
            score = float(row.get("best_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return (success, localized, score)

    row = max(rows, key=row_score).copy()
    row["all_decision_source_doc_ids"] = " | ".join(dict.fromkeys(
        source
        for item in rows
        for field in (
            "primary_finding_source_doc_id", "authoritative_source_doc_id",
            "selected_source_doc_id", "top_candidate_doc_id",
        )
        if (source := normalize_space(item.get(field, "")))
    ))
    return {str(key): row.get(key, "") for key in row.keys()}


def augment_streamlit_with_runtime_truth(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty or "run_root_abs" not in base.columns:
        return base
    rows = base.to_dict("records")
    for row in rows:
        task = normalize_space(row.get("task_family", "")).casefold()
        run_root_value = normalize_space(row.get("run_root_abs", ""))
        if not run_root_value:
            continue
        run_root = Path(run_root_value)
        if not run_root.exists():
            continue
        run_layout = normalize_space(row.get("run_layout", "")).casefold() or "pair"
        ground_truth = row.get("ground_truth", "")
        for arm in ("baseline", "holdout"):
            output_dir = runtime_output_dir(run_root, run_layout, arm)
            eval_row = load_runtime_eval_row(
                first_runtime_eval_path(run_root, run_layout, arm)
            )
            if not eval_row:
                continue
            selected_value = normalize_space(eval_row.get("selected_value_or_text", ""))
            if selected_value and not normalize_space(row.get(f"{arm}_selected_value_or_text", "")):
                row[f"{arm}_selected_value_or_text"] = selected_value
            selected_doc_id = normalize_space(eval_row.get("selected_source_doc_id", ""))
            if selected_doc_id and not normalize_space(row.get(f"{arm}_selected_source_doc_id", "")):
                row[f"{arm}_selected_source_doc_id"] = selected_doc_id
            semantic_status = normalize_space(eval_row.get("semantic_status", ""))
            if semantic_status and not normalize_space(row.get(f"{arm}_semantic_status", "")):
                row[f"{arm}_semantic_status"] = semantic_status
            all_sources = normalize_space(
                eval_row.get("all_decision_source_doc_ids", "")
            )
            if all_sources:
                row[f"{arm}_all_decision_source_doc_ids"] = all_sources
            all_targets_resolved = truthy(eval_row.get("all_targets_resolved", True))
            value_match = relaxed_truth_match(ground_truth, selected_value)
            evidence_match = selected_evidence_truth_match(output_dir, ground_truth)
            truth_match = all_targets_resolved and (value_match or evidence_match)
            row[f"{arm}_value_truth_match"] = value_match
            row[f"{arm}_evidence_truth_match"] = evidence_match
            row[f"{arm}_effective_truth_match"] = truth_match
            row[f"{arm}_truth_match"] = truth_match
    return pd.DataFrame(rows)


def semantic_success(task_family: str, status: Any) -> bool:
    prefixes = SUCCESS_PREFIXES.get(normalize_space(task_family).casefold(), ())
    lowered = normalize_space(status).casefold()
    if any(lowered.startswith(prefix) for prefix in prefixes):
        return True
    fragments = [
        fragment.strip()
        for fragment in lowered.replace("=>", "->").split("->")
        if fragment.strip()
    ]
    return any(
        any(fragment.startswith(prefix) for prefix in prefixes)
        for fragment in fragments
    )


def load_results(experiment_root: Path) -> pd.DataFrame:
    summary_root = experiment_root / "summaries"
    case_results = summary_root / "case_results.csv"
    streamlit_index = summary_root / "streamlit_run_index.csv"
    retrieval_diagnostics = summary_root / "retrieval_diagnostics.csv"
    if case_results.exists():
        return pd.read_csv(case_results, low_memory=False)
    if streamlit_index.exists():
        base = pd.read_csv(streamlit_index, low_memory=False)
        if retrieval_diagnostics.exists():
            diag = pd.read_csv(retrieval_diagnostics, low_memory=False)
            for column in [
                "intended_doc_hit_selected",
                "intended_doc_hit_top_candidate",
                "intended_doc_hit_any_fragment",
                "intended_doc_hit_any",
                "intended_paragraph_hit_selected",
                "intended_paragraph_hit_any_fragment",
            ]:
                if column in diag.columns:
                    diag[column] = diag[column].map(truthy)
            key_cols = ["corpus", "task_family", "case_id"]
            diag["relevant_retrieval_any"] = False
            for column in [
                "intended_doc_hit_selected",
                "intended_doc_hit_top_candidate",
                "intended_doc_hit_any_fragment",
                "intended_doc_hit_any",
                "intended_paragraph_hit_selected",
                "intended_paragraph_hit_any_fragment",
            ]:
                if column in diag.columns:
                    diag["relevant_retrieval_any"] = diag["relevant_retrieval_any"] | diag[column].fillna(False)
            agg = (
                diag.groupby(key_cols, dropna=False)
                .agg(
                    relevant_retrieval_any=("relevant_retrieval_any", "max"),
                )
                .reset_index()
            )
            if "arm" in diag.columns and "semantic_status" in diag.columns:
                arm_status = (
                    diag.pivot_table(
                        index=key_cols,
                        columns="arm",
                        values="semantic_status",
                        aggfunc="first",
                    )
                    .reset_index()
                )
                rename_map = {}
                if "baseline" in arm_status.columns:
                    rename_map["baseline"] = "baseline_semantic_status"
                if "holdout" in arm_status.columns:
                    rename_map["holdout"] = "holdout_semantic_status"
                if "primary" in arm_status.columns:
                    rename_map["primary"] = "primary_semantic_status"
                arm_status = arm_status.rename(columns=rename_map)
                agg = agg.merge(arm_status, on=key_cols, how="left")
            base = base.merge(agg, on=key_cols, how="left")
        base = augment_streamlit_with_runtime_truth(base)
        return base
    raise FileNotFoundError(
        f"Could not find `case_results.csv` or `streamlit_run_index.csv` under {summary_root}"
    )


def enrich_results(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["task_family"] = data["task_family"].map(normalize_space)
    data["corpus"] = data["corpus"].map(normalize_space).map(
        lambda value: CORPUS_NORMALIZATION.get(value.casefold(), value.casefold()) if value else ""
    )
    run_layout = (
        data["run_layout"].fillna("").map(normalize_space).str.casefold()
        if "run_layout" in data.columns
        else pd.Series([""] * len(data), index=data.index)
    )
    batch_status = (
        data["batch_status"].fillna("").map(normalize_space).str.upper()
        if "batch_status" in data.columns
        else pd.Series([""] * len(data), index=data.index)
    )

    def series_or_default(column: str, default: Any) -> pd.Series:
        if column in data.columns:
            return data[column]
        return pd.Series([default] * len(data), index=data.index)

    def normalized_key_series(column: str) -> pd.Series:
        return series_or_default(column, "").map(lambda value: normalize_space(value).casefold())

    def delimited_source_contains(column: str, targets: pd.Series) -> pd.Series:
        values = series_or_default(column, "")
        return pd.Series(
            [
                normalize_space(target).casefold()
                in {
                    normalize_space(part).casefold()
                    for part in str(value or "").split("|")
                    if normalize_space(part)
                }
                for value, target in zip(values, targets)
            ],
            index=data.index,
        )

    baseline_status_series = pd.Series([""] * len(data), index=data.index, dtype="object")
    if "baseline_semantic_status" in data.columns:
        baseline_status_series = data["baseline_semantic_status"].fillna("")
    if "primary_semantic_status" in data.columns:
        baseline_status_series = baseline_status_series.mask(
            baseline_status_series.map(normalize_space).eq(""),
            data["primary_semantic_status"].fillna(""),
        )
    if "semantic_status" in data.columns:
        baseline_status_series = baseline_status_series.mask(
            baseline_status_series.map(normalize_space).eq(""),
            data["semantic_status"].fillna(""),
        )
    data["baseline_success"] = [
        semantic_success(task, status)
        for task, status in zip(data["task_family"], baseline_status_series)
    ]
    single_run_mask = run_layout.ne("pair") & batch_status.ne("")
    data.loc[single_run_mask, "baseline_success"] = batch_status[single_run_mask].eq("PASS")

    if "holdout_semantic_status" in data.columns:
        data["holdout_success"] = [
            semantic_success(task, status)
            for task, status in zip(data["task_family"], data["holdout_semantic_status"])
        ]
    else:
        data["holdout_success"] = False

    data["pass_any"] = data["baseline_success"] | data["holdout_success"]
    data["pass_both"] = data["baseline_success"] & data["holdout_success"]
    data["baseline_truth_match"] = series_or_default("baseline_effective_truth_match", False).map(truthy)
    data["holdout_truth_match"] = series_or_default("holdout_effective_truth_match", False).map(truthy)
    data["baseline_grounded"] = (
        data["baseline_success"] & data["baseline_truth_match"]
    )
    data["holdout_grounded"] = (
        data["holdout_success"] & data["holdout_truth_match"]
    )
    doc_key = normalized_key_series("doc_id")
    data["baseline_doc_match"] = (
        (normalized_key_series("baseline_selected_source_doc_id") == doc_key)
        | (normalized_key_series("baseline_top_candidate_doc_id") == doc_key)
        | (normalized_key_series("baseline_authoritative_source_doc_id") == doc_key)
        | (normalized_key_series("baseline_primary_finding_source_doc_id") == doc_key)
        | delimited_source_contains("baseline_all_decision_source_doc_ids", doc_key)
    )
    data["holdout_doc_match"] = (
        (normalized_key_series("holdout_selected_source_doc_id") == doc_key)
        | (normalized_key_series("holdout_top_candidate_doc_id") == doc_key)
        | (normalized_key_series("holdout_authoritative_source_doc_id") == doc_key)
        | (normalized_key_series("holdout_primary_finding_source_doc_id") == doc_key)
        | delimited_source_contains("holdout_all_decision_source_doc_ids", doc_key)
    )
    if "relevant_retrieval_any" in data.columns:
        data["relevant_retrieval_any"] = data["relevant_retrieval_any"].map(truthy)
    else:
        data["relevant_retrieval_any"] = (
            data["baseline_truth_match"]
            | data["holdout_truth_match"]
            | data["baseline_doc_match"]
            | data["holdout_doc_match"]
        )

    def classify(row: pd.Series) -> str:
        if row["baseline_grounded"] and row["holdout_grounded"]:
            return "grounded_both"
        if row["baseline_grounded"]:
            return "grounded_baseline"
        if row["baseline_success"]:
            return "ungrounded_baseline_pass"
        if row["pass_both"]:
            return "ungrounded_pass_both"
        if row["holdout_success"] and not row["baseline_success"]:
            return "pass_holdout_only"
        if row["relevant_retrieval_any"]:
            return "fail_relevant_retrieval"
        if normalize_space(row.get("outcome_reason", "")).casefold().startswith("preflight"):
            return "fail_preflight"
        return "fail_no_relevant_retrieval"

    data["outcome_bucket"] = data.apply(classify, axis=1)
    data["outcome_reason_short"] = series_or_default("outcome_reason", "").map(normalize_space)
    return data


def make_analyst_paths_portable(data: pd.DataFrame) -> pd.DataFrame:
    """Replace internal absolute paths with package-relative handoff paths."""
    result = data.copy()
    path_columns = {
        "query_file_abs": "query_file",
        "run_root_abs": "run_path",
        "streamlit_open_path_abs": "streamlit_open_path",
        "streamlit_holdout_path_abs": "streamlit_holdout_path",
        "primary_digest_artifact_abs": "primary_digest_artifact",
        "holdout_digest_artifact_abs": "holdout_digest_artifact",
    }

    def portable(value: Any) -> str:
        text = normalize_space(value)
        if not text:
            return ""
        path = Path(text).expanduser()
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(PACKAGE_ROOT).as_posix()
        except ValueError:
            # External paths are not portable and should not leak into a
            # handoff report. Retain only the artifact name for diagnosis.
            return path.name

    for source, destination in path_columns.items():
        if source not in result.columns:
            continue
        result[destination] = result[source].map(portable)
        result = result.drop(columns=[source])
    return result


def build_metric_table(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        data.groupby(["corpus", "task_family"], dropna=False)
        .agg(total=("case_id", "count"), hits=(metric, "sum"))
        .reset_index()
    )
    grouped["rate"] = grouped["hits"] / grouped["total"].clip(lower=1)
    return grouped


def count_frame(metric_table: pd.DataFrame, value_column: str) -> pd.DataFrame:
    frame = metric_table.pivot(index="corpus", columns="task_family", values=value_column)
    frame = frame.reindex(index=[c for c in CORPUS_ORDER if c in frame.index])
    frame = frame.reindex(columns=[t for t in TASK_ORDER if t in frame.columns])
    return frame.fillna(0).astype(int)


def plot_useful_outcome_heatmap(
    success_df: pd.DataFrame,
    relevant_fail_df: pd.DataFrame,
    total_df: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    cmap: str = "RdYlBu",
) -> None:
    if success_df.empty:
        return
    denom = total_df.replace(0, np.nan).astype(float)
    useful_rate = (success_df + relevant_fail_df) / denom
    useful_rate = useful_rate.fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(useful_rate.values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(useful_rate.columns)))
    ax.set_xticklabels(useful_rate.columns, fontsize=11, fontweight="bold")
    ax.set_yticks(np.arange(len(useful_rate.index)))
    ax.set_yticklabels(useful_rate.index, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
    for i, corpus in enumerate(useful_rate.index):
        for j, task in enumerate(useful_rate.columns):
            success = int(success_df.loc[corpus, task])
            relevant_fail = int(relevant_fail_df.loc[corpus, task])
            total = int(total_df.loc[corpus, task])
            rate = float(useful_rate.loc[corpus, task])
            color = "white" if rate >= 0.55 else "black"
            ax.text(
                j,
                i,
                f"{success} (+{relevant_fail}) / {total}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Useful Outcome Rate", rotation=270, labelpad=22, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_grounded_outcome_heatmap(
    grounded_df: pd.DataFrame,
    operational_df: pd.DataFrame,
    total_df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    """Render truth-grounded results as the primary benchmark matrix."""
    if grounded_df.empty:
        return
    denom = total_df.replace(0, np.nan).astype(float)
    grounded_rate = (grounded_df / denom).fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(
        grounded_rate.values,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(grounded_rate.columns)))
    ax.set_xticklabels(
        grounded_rate.columns, fontsize=11, fontweight="bold"
    )
    ax.set_yticks(np.arange(len(grounded_rate.index)))
    ax.set_yticklabels(
        grounded_rate.index, fontsize=11, fontweight="bold"
    )
    ax.set_title(
        "Grounded Baseline Answers (operational decisions shown second)",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    for i, corpus in enumerate(grounded_rate.index):
        for j, task in enumerate(grounded_rate.columns):
            grounded = int(grounded_df.loc[corpus, task])
            operational = int(operational_df.loc[corpus, task])
            total = int(total_df.loc[corpus, task])
            rate = float(grounded_rate.loc[corpus, task])
            ax.text(
                j,
                i,
                f"G {grounded}/{total}\nOp {operational}/{total}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if rate <= 0.35 or rate >= 0.80 else "black",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(
        "Grounded Rate", rotation=270, labelpad=20, fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_reason_summary(data: pd.DataFrame, output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for (corpus, task), group in data.groupby(["corpus", "task_family"], dropna=False):
        top = (
            group["outcome_reason_short"]
            .replace("", np.nan)
            .dropna()
            .value_counts()
            .head(5)
            .to_dict()
        )
        rows.append(
            {
                "corpus": corpus,
                "task_family": task,
                "top_outcome_reasons": "; ".join(f"{k} ({v})" for k, v in top.items()),
            }
        )
    pd.DataFrame(rows).sort_values(["corpus", "task_family"]).to_csv(output_path, index=False)


def summarize_experiment(experiment_root: Path) -> Path:
    experiment_root = experiment_root.resolve()
    summary_root = experiment_root / "summaries"
    output_root = summary_root / "analyst_report"
    output_root.mkdir(parents=True, exist_ok=True)

    data = make_analyst_paths_portable(
        enrich_results(load_results(experiment_root))
    )
    data.to_csv(output_root / "case_results.csv", index=False)

    for stale_name in [
        "success_any_matrix.png",
        "success_both_matrix.png",
        "relevant_retrieval_matrix.png",
        "success_any_table.csv",
        "success_both_table.csv",
        "relevant_retrieval_table.csv",
        "success_table.csv",
        "relevant_fail_table.csv",
        "case_review_table.csv",
        "outcome_bucket_summary.csv",
        "useful_outcomes.csv",
        "useful_outcomes_heatmap.png",
    ]:
        stale_path = output_root / stale_name
        if stale_path.exists():
            stale_path.unlink()

    grounded_table = build_metric_table(data, "baseline_grounded")
    operational_table = build_metric_table(data, "baseline_success")
    grounded_hits_df = count_frame(grounded_table, "hits")
    operational_hits_df = count_frame(operational_table, "hits")
    total_df = count_frame(grounded_table, "total")
    combined_table = grounded_table.rename(
        columns={"hits": "grounded_hits", "rate": "grounded_rate"}
    ).merge(
        operational_table[["corpus", "task_family", "hits", "rate"]].rename(
            columns={
                "hits": "operational_hits",
                "rate": "operational_rate",
            }
        ),
        on=["corpus", "task_family"],
        how="left",
    ).fillna({"operational_hits": 0, "operational_rate": 0.0})
    combined_table["label"] = combined_table.apply(
        lambda row: (
            f"G {int(row['grounded_hits'])}/{int(row['total'])}; "
            f"Op {int(row['operational_hits'])}/{int(row['total'])}"
        ),
        axis=1,
    )
    combined_table.to_csv(output_root / "grounded_outcomes.csv", index=False)
    plot_grounded_outcome_heatmap(
        grounded_hits_df,
        operational_hits_df,
        total_df,
        output_path=output_root / "grounded_outcomes_heatmap.png",
    )

    write_reason_summary(data, output_root / "failure_reasons.csv")

    readme = output_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# Analyst Report for `{experiment_root.name}`",
                "",
                "- `case_results.csv`: comprehensive case, query, truth, retrieval, and outcome table.",
                "- `grounded_outcomes_heatmap.png`: primary matrix; `G` is a truth-grounded operational answer and `Op` is any operational decision.",
                "- `grounded_outcomes.csv`: tabular grounded and operational counts and rates.",
                "- `failure_reasons.csv`: top outcome reasons per corpus and task.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the compact analyst report for a benchmark experiment."
    )
    parser.add_argument("--experiment-root", required=True, help="Benchmark experiment root directory.")
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).resolve()
    output_root = summarize_experiment(experiment_root)
    print(f"Analyst report written to: {output_root}")
    outcomes = pd.read_csv(output_root / "grounded_outcomes.csv")
    total = int(outcomes["total"].sum())
    grounded = int(outcomes["grounded_hits"].sum())
    operational = int(outcomes["operational_hits"].sum())
    grounded_rate = grounded / total if total else 0.0
    operational_rate = operational / total if total else 0.0
    print(
        "Primary result: "
        f"Grounded {grounded}/{total} ({grounded_rate:.1%}); "
        f"Operational {operational}/{total} ({operational_rate:.1%})"
    )
    for corpus in CORPUS_ORDER:
        rows = outcomes[outcomes["corpus"] == corpus].copy()
        if rows.empty:
            continue
        by_task = {row.task_family: row for row in rows.itertuples()}
        cells = []
        for task in TASK_ORDER:
            row = by_task.get(task)
            if row is None:
                continue
            cells.append(
                f"{task}=G{int(row.grounded_hits)}/{int(row.total)}"
                f",Op{int(row.operational_hits)}/{int(row.total)}"
            )
        print(f"  {corpus}: " + " | ".join(cells))
    cases = pd.read_csv(output_root / "case_results.csv", low_memory=False)
    preflight_count = int(
        cases.get("outcome_reason_short", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.contains("preflight", case=False)
        .sum()
    )
    ungrounded_passes = int(
        (
            cases.get("baseline_success", False).map(truthy)
            & ~cases.get("baseline_grounded", False).map(truthy)
        ).sum()
    )
    print(
        f"Review flags: preflight={preflight_count}; "
        f"operational-but-not-grounded={ungrounded_passes}"
    )
    if experiment_root.name.endswith("__rephrased"):
        original_root = experiment_root.with_name(
            experiment_root.name.removesuffix("__rephrased") + "__original"
        )
        original_report = original_root / "summaries" / "analyst_report" / "grounded_outcomes.csv"
        if original_report.exists():
            original_outcomes = pd.read_csv(original_report)
            original_total = int(original_outcomes["total"].sum())
            original_grounded = int(original_outcomes["grounded_hits"].sum())
            original_rate = original_grounded / original_total if original_total else 0.0
            retention = grounded / original_grounded if original_grounded else 0.0
            print(
                "A/B grounded: "
                f"original {original_grounded}/{original_total} ({original_rate:.1%}) -> "
                f"semantic {grounded}/{total} ({grounded_rate:.1%}); "
                f"retention {retention:.1%}; delta {grounded - original_grounded:+d}"
            )


if __name__ == "__main__":
    main()
