#!/usr/bin/env python3
"""Run an original vs rephrased Matryoshka benchmark A/B experiment.

The public interface intentionally exposes one OpenAI model. That model is
used by the runtime query parser, optional digest, query rewriter, and rewrite
judge so a handoff user does not have to coordinate four model flags.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = PACKAGE_ROOT / "output" / "matryoshka_workbench" / "benchmarks"
MULTICORPUS_BENCHMARK = PACKAGE_ROOT / "scripts" / "matryoshka_workbench" / "tools" / "multicorpus_benchmark.py"
REPHRASE_TOOL = PACKAGE_ROOT / "scripts" / "matryoshka_workbench" / "tools" / "rephrase_benchmark_queries.py"
SUMMARIZE_TOOL = PACKAGE_ROOT / "scripts" / "matryoshka_workbench" / "tools" / "summarize_benchmark_results.py"
DEFAULT_CONFIG = str(
    PACKAGE_ROOT
    / "scripts"
    / "matryoshka_workbench"
    / "config"
    / "benchmark.default.json"
)


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(argv: list[str]) -> None:
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=PACKAGE_ROOT, check=True)


def build_multicorpus_args(args: argparse.Namespace, *, command: str, experiment_name: str) -> list[str]:
    argv = [
        sys.executable,
        str(MULTICORPUS_BENCHMARK),
        command,
        "--corpora",
        args.corpora,
        "--config",
        args.config,
        "--experiment-name",
        experiment_name,
    ]
    if args.tasks:
        argv.extend(["--tasks", args.tasks])
    if args.per_task is not None:
        argv.extend(["--per-task", str(args.per_task)])
    if args.seed is not None:
        argv.extend(["--seed", str(args.seed)])
    if args.runner_mode:
        argv.extend(["--runner-mode", args.runner_mode])
    if args.use_openai_parser:
        argv.append("--use-openai-parser")
    if args.use_llm_digest:
        argv.append("--use-llm-digest")
    if args.openai_model:
        argv.extend(["--parser-model", args.openai_model])
        argv.extend(["--llm-model", args.openai_model])
    if args.overwrite:
        argv.append("--overwrite")
    return argv


def truthy(value: Any) -> bool:
    return normalize_space(value).lower() in {"1", "true", "yes"}


def arm_truth_match(row: dict[str, str], arm: str) -> bool:
    for key in (
        f"{arm}_effective_truth_match",
        f"{arm}_truth_match",
    ):
        if normalize_space(row.get(key, "")).lower() in {"1", "true", "yes"}:
            return True
    return False


def arm_success(row: dict[str, str], arm: str) -> bool:
    return truthy(row.get(f"{arm}_success", ""))


def arm_relevant_retrieval(row: dict[str, str], arm: str) -> bool:
    """Use the arm's target-document match; never mix baseline and holdout."""
    return truthy(row.get(f"{arm}_doc_match", ""))


def build_comparison(
    *,
    source_root: Path,
    dest_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = read_csv_rows(
        source_root / "summaries" / "analyst_report" / "case_results.csv"
    )
    dest_rows = read_csv_rows(
        dest_root / "summaries" / "analyst_report" / "case_results.csv"
    )
    source_answer = {
        normalize_space(row.get("case_id", "")): row
        for row in read_csv_rows(source_root / "query_manifests" / "query_answer_key.csv")
    }
    dest_answer = {
        normalize_space(row.get("case_id", "")): row
        for row in read_csv_rows(dest_root / "query_manifests" / "query_answer_key.csv")
    }
    dest_rephrase_audit = {
        normalize_space(row.get("case_id", "")): row
        for row in read_csv_rows(dest_root / "summaries" / "query_rephrase_audit.csv")
    }
    source_by_case = {
        normalize_space(row.get("case_id", "")): row
        for row in source_rows
    }
    dest_by_case = {
        normalize_space(row.get("case_id", "")): row
        for row in dest_rows
    }

    comparison_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], dict[str, int]] = {}
    all_case_ids = sorted(set(source_by_case) | set(dest_by_case))
    for case_id in all_case_ids:
        src = source_by_case.get(case_id, {})
        dst = dest_by_case.get(case_id, {})
        src_answer = source_answer.get(case_id, {})
        dst_answer = dest_answer.get(case_id, {})
        audit = dest_rephrase_audit.get(case_id, {})
        corpus = normalize_space(src.get("corpus") or dst.get("corpus"))
        task = normalize_space(src.get("task_family") or dst.get("task_family"))
        src_pass = arm_success(src, "baseline")
        dst_pass = arm_success(dst, "baseline")
        src_holdout_pass = arm_success(src, "holdout")
        dst_holdout_pass = arm_success(dst, "holdout")
        src_retrieval = arm_relevant_retrieval(src, "baseline")
        dst_retrieval = arm_relevant_retrieval(dst, "baseline")
        src_truth = arm_truth_match(src, "baseline")
        dst_truth = arm_truth_match(dst, "baseline")
        src_holdout_truth = arm_truth_match(src, "holdout")
        dst_holdout_truth = arm_truth_match(dst, "holdout")
        row = {
            "case_id": case_id,
            "corpus": corpus,
            "task_family": task,
            "original_query": src_answer.get("query_text", ""),
            "rephrased_query": dst_answer.get("query_text", ""),
            "rewrite_changed": audit.get("changed", ""),
            "rewrite_mode": audit.get("rewrite_mode", ""),
            "rewrite_attempt_count": audit.get("attempt_count", ""),
            "rewrite_policy": audit.get("rewrite_policy", ""),
            "rewrite_runtime_ready": audit.get("runtime_ready", ""),
            "rewrite_validation_reason": audit.get("validation_reason", ""),
            "rewrite_final_reason": audit.get("final_reason", ""),
            "rewrite_parser_validation_reason": audit.get("parser_validation_reason", ""),
            "rewrite_judge_equivalent": audit.get("judge_equivalent", ""),
            "rewrite_same_target": audit.get("judge_same_target", ""),
            "rewrite_same_task_intent": audit.get("judge_same_task_intent", ""),
            "rewrite_preserved_critical_anchors": audit.get("judge_preserved_critical_anchors", ""),
            "rewrite_added_false_constraints": audit.get("judge_added_false_constraints", ""),
            "rewrite_omitted_only_noncritical_detail": audit.get("judge_omitted_only_noncritical_detail", ""),
            "rewrite_judge_confidence": audit.get("judge_confidence", ""),
            "rewrite_lexical_reuse_ratio": audit.get("lexical_reuse_ratio", ""),
            "rewrite_lexical_jaccard": audit.get("lexical_jaccard", ""),
            "rewrite_minimum_clause_anchor_recall": audit.get("minimum_clause_anchor_recall", ""),
            "rewrite_surface_similarity_score": audit.get("surface_similarity_score", ""),
            "rewrite_surface_distance_score": audit.get("surface_distance_score", ""),
            "original_pass_any": src.get("pass_any", ""),
            "rephrased_pass_any": dst.get("pass_any", ""),
            "original_baseline_pass": src_pass,
            "rephrased_baseline_pass": dst_pass,
            "original_holdout_pass": src_holdout_pass,
            "rephrased_holdout_pass": dst_holdout_pass,
            "original_outcome_bucket": src.get("outcome_bucket", ""),
            "rephrased_outcome_bucket": dst.get("outcome_bucket", ""),
            "original_relevant_retrieval_any": src.get("relevant_retrieval_any", ""),
            "rephrased_relevant_retrieval_any": dst.get("relevant_retrieval_any", ""),
            "original_effective_truth_match": src_truth,
            "rephrased_effective_truth_match": dst_truth,
            "original_holdout_truth_match": src_holdout_truth,
            "rephrased_holdout_truth_match": dst_holdout_truth,
            "original_pass_and_truth": src_pass and src_truth,
            "rephrased_pass_and_truth": dst_pass and dst_truth,
            "original_grounded": src_pass and src_truth,
            "rephrased_grounded": dst_pass and dst_truth,
            "original_holdout_pass_and_truth": src_holdout_pass and src_holdout_truth,
            "rephrased_holdout_pass_and_truth": dst_holdout_pass and dst_holdout_truth,
            "original_batch_status": src.get("batch_status", ""),
            "rephrased_batch_status": dst.get("batch_status", ""),
        }
        comparison_rows.append(row)
        key = (corpus, task)
        bucket = counts.setdefault(
            key,
            {
                "case_count": 0,
                "original_pass": 0,
                "rephrased_pass": 0,
                "original_useful": 0,
                "rephrased_useful": 0,
                "original_truth": 0,
                "rephrased_truth": 0,
                "original_pass_and_truth": 0,
                "rephrased_pass_and_truth": 0,
                "original_grounded": 0,
                "rephrased_grounded": 0,
                "original_holdout_pass": 0,
                "rephrased_holdout_pass": 0,
                "original_holdout_truth": 0,
                "rephrased_holdout_truth": 0,
                "original_holdout_pass_and_truth": 0,
                "rephrased_holdout_pass_and_truth": 0,
                "improved": 0,
                "regressed": 0,
                "unchanged": 0,
            },
        )
        bucket["case_count"] += 1
        bucket["original_pass"] += int(src_pass)
        bucket["rephrased_pass"] += int(dst_pass)
        bucket["original_useful"] += int(src_pass or src_retrieval)
        bucket["rephrased_useful"] += int(dst_pass or dst_retrieval)
        bucket["original_truth"] += int(src_truth)
        bucket["rephrased_truth"] += int(dst_truth)
        bucket["original_pass_and_truth"] += int(src_pass and src_truth)
        bucket["rephrased_pass_and_truth"] += int(dst_pass and dst_truth)
        bucket["original_grounded"] += int(src_pass and src_truth)
        bucket["rephrased_grounded"] += int(dst_pass and dst_truth)
        bucket["original_holdout_pass"] += int(src_holdout_pass)
        bucket["rephrased_holdout_pass"] += int(dst_holdout_pass)
        bucket["original_holdout_truth"] += int(src_holdout_truth)
        bucket["rephrased_holdout_truth"] += int(dst_holdout_truth)
        bucket["original_holdout_pass_and_truth"] += int(src_holdout_pass and src_holdout_truth)
        bucket["rephrased_holdout_pass_and_truth"] += int(dst_holdout_pass and dst_holdout_truth)
        if (dst_pass and dst_truth) and not (src_pass and src_truth):
            bucket["improved"] += 1
        elif (src_pass and src_truth) and not (dst_pass and dst_truth):
            bucket["regressed"] += 1
        else:
            bucket["unchanged"] += 1

    for (corpus, task), stats in sorted(counts.items()):
        summary_rows.append(
            {
                "corpus": corpus,
                "task_family": task,
                **stats,
            }
        )
    return comparison_rows, summary_rows


def append_comparison_readme(path: Path, *, original_name: str, rephrased_name: str) -> None:
    existing = path.read_text(encoding="utf-8").rstrip() if path.exists() else "# Analyst Report"
    marker = "\n## Original vs Semantic Rewrite\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip()
    comparison_section = "\n".join(
        [
            "",
            "## Original vs Semantic Rewrite",
            "",
            f"Original branch: `{original_name}`",
            f"Semantic branch: `{rephrased_name}`",
            "",
            "- `original_vs_rephrased_cases.csv`: case-level queries, rewrite diagnostics, retrieval, truth, and outcomes.",
            "- `original_vs_rephrased_summary.csv`: baseline A/B metrics plus separately reported holdout metrics.",
            "",
            "Headline A/B metrics are grounded rates: an operational decision must also match benchmark truth. Holdout outcomes are reported separately and never mixed into baseline rates.",
            "Rewrites are one-shot analyst-style semantic compressions. Diagnostic judging records drift but never rejects a rewrite or substitutes the original query.",
        ]
    )
    path.write_text(existing + "\n" + comparison_section + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["build", "run", "summarize", "all"],
        default="all",
        help=(
            "build: create original benchmark manifests and the rephrased clone only; "
            "run: execute already-built original + rephrased benchmarks and summarize; "
            "summarize: regenerate the A/B comparison from existing run artifacts only; "
            "all: build, then run, then summarize."
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--corpora",
        default="vietnam,cib,iran,fisc,canberra",
        help="Comma-separated corpus profiles. Defaults to all five shipped corpora.",
    )
    parser.add_argument(
        "--tasks",
        default="reconstruction,claims,value_redaction,sentence_redaction,contradictions",
        help="Comma-separated task families. Defaults to all five shipped tasks.",
    )
    parser.add_argument(
        "--per-task",
        type=int,
        default=10,
        help="Maximum selected cases per corpus/task (default: 10).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Selection seed (default: 42).")
    parser.add_argument("--runner-mode", choices=["single", "pair"], default="pair")
    parser.add_argument("--use-openai-parser", action="store_true")
    parser.add_argument("--use-llm-digest", action="store_true")
    parser.add_argument(
        "--openai-model",
        default="gpt-4o",
        help="One model for runtime parsing/digests and rewrite generation/judging.",
    )
    parser.add_argument(
        "--rephrase-mode",
        choices=["semantic"],
        default="semantic",
        help="The shipped A/B workflow supports one-shot semantic analyst-note rewrites.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_experiment = f"{args.experiment_name}__original"
    rephrased_experiment = f"{args.experiment_name}__rephrased"

    def build_phase() -> None:
        run_command(
            build_multicorpus_args(
                args,
                command="build",
                experiment_name=original_experiment,
            )
        )

        rewrite_cmd = [
            sys.executable,
            str(REPHRASE_TOOL),
            "--source-experiment-name",
            original_experiment,
            "--dest-experiment-name",
            rephrased_experiment,
            "--model",
            args.openai_model,
            "--judge-model",
            args.openai_model,
            "--mode",
            args.rephrase_mode,
            "--overwrite",
        ]
        if args.tasks:
            rewrite_cmd.extend(["--tasks", args.tasks])
        if args.corpora:
            rewrite_cmd.extend(["--corpora", args.corpora])
        run_command(rewrite_cmd)

    def comparison_phase() -> None:
        original_root = BENCHMARK_ROOT / original_experiment
        rephrased_root = BENCHMARK_ROOT / rephrased_experiment
        required = (
            original_root / "summaries" / "analyst_report" / "case_results.csv",
            rephrased_root / "summaries" / "analyst_report" / "case_results.csv",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit(
                "Missing analyst reports required for A/B comparison:\n- "
                + "\n- ".join(missing)
            )
        comparison_rows, summary_rows = build_comparison(
            source_root=original_root,
            dest_root=rephrased_root,
        )
        analyst_root = rephrased_root / "summaries" / "analyst_report"
        write_csv_rows(analyst_root / "original_vs_rephrased_cases.csv", comparison_rows)
        write_csv_rows(analyst_root / "original_vs_rephrased_summary.csv", summary_rows)
        append_comparison_readme(
            analyst_root / "README.md",
            original_name=original_experiment,
            rephrased_name=rephrased_experiment,
        )
        print(f"Original experiment: {original_root}")
        print(f"Rephrased experiment: {rephrased_root}")
        print(f"Comparison summary: {analyst_root / 'original_vs_rephrased_summary.csv'}")

    def run_phase() -> None:
        original_root = BENCHMARK_ROOT / original_experiment
        rephrased_root = BENCHMARK_ROOT / rephrased_experiment
        original_plan = original_root / "query_manifests" / "benchmark_plan.json"
        rephrased_plan = rephrased_root / "query_manifests" / "benchmark_plan.json"
        if not original_plan.exists():
            raise SystemExit(
                f"Missing original build plan: {original_plan}. "
                "Run with --stage build or --stage all first."
            )
        if not rephrased_plan.exists():
            raise SystemExit(
                f"Missing rephrased build plan: {rephrased_plan}. "
                "Run with --stage build or --stage all first."
            )

        run_command(
            build_multicorpus_args(
                args,
                command="run",
                experiment_name=original_experiment,
            )
        )
        run_command(
            [
                sys.executable,
                str(SUMMARIZE_TOOL),
                "--experiment-root",
                str(original_root),
            ]
        )

        run_command(
            build_multicorpus_args(
                args,
                command="run",
                experiment_name=rephrased_experiment,
            )
        )
        run_command(
            [
                sys.executable,
                str(SUMMARIZE_TOOL),
                "--experiment-root",
                str(rephrased_root),
            ]
        )
        comparison_phase()

    if args.stage in {"build", "all"}:
        build_phase()
    if args.stage in {"run", "all"}:
        run_phase()
    if args.stage == "summarize":
        comparison_phase()


if __name__ == "__main__":
    main()
