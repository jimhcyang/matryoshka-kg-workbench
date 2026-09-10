#!/usr/bin/env python3
"""End-to-end kg_scripts build runner with step-level progress."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_scripts._common import iter_progress


SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(cmd: List[str]) -> None:
    print("\n== Running ==")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def expected_stage_outputs(project_dir: Path) -> dict[str, Path]:
    return {
        "A": project_dir / "extraction" / "summary.json",
        "ALIAS": project_dir / "alias" / "summary.json",
        "B": project_dir / "graph" / "summary.json",
        "C": project_dir / "tasks" / "summary.json",
    }


def build_steps(args: argparse.Namespace) -> List[Tuple[str, List[str]]]:
    project_dir = Path(args.project_dir).expanduser().resolve()
    steps: List[Tuple[str, List[str]]] = [
        (
            "A",
            [
                sys.executable,
                str(SCRIPT_DIR / "stage_a.py"),
                "--corpus-root",
                str(Path(args.corpus_root).expanduser().resolve()),
                "--project-dir",
                str(project_dir),
                "--dataset-mode",
                args.dataset_mode,
                "--transcription-variant",
                args.transcription_variant,
                "--gap-marker",
                args.gap_marker,
                "--openai-model",
                args.openai_model,
            ],
        ),
        (
            "ALIAS",
            [
                sys.executable,
                str(SCRIPT_DIR / "alias_resolution.py"),
                "--project-dir",
                str(project_dir),
            ],
        ),
        (
            "B",
            [
                sys.executable,
                str(SCRIPT_DIR / "stage_b.py"),
                "--project-dir",
                str(project_dir),
            ],
        ),
        (
            "C",
            [
                sys.executable,
                str(SCRIPT_DIR / "stage_c.py"),
                "--project-dir",
                str(project_dir),
                "--min-redaction-words",
                str(args.min_redaction_words),
                "--max-redaction-words",
                str(args.max_redaction_words),
            ],
        ),
    ]
    if args.limit is not None:
        steps[0][1].extend(["--limit", str(args.limit)])
    if args.max_new_chunks is not None:
        steps[0][1].extend(["--max-new-chunks", str(args.max_new_chunks)])
    if args.use_openai_head_review:
        steps[1][1].append("--use-openai-head-review")
        steps[1][1].extend(["--head-review-model", args.head_review_model])
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the kg_scripts A→alias→B→C pipeline.")
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--dataset-mode", choices=["auto", "generic", "cib", "fisc"], default="auto")
    parser.add_argument("--transcription-variant", choices=["unredacted", "redacted"], default="unredacted")
    parser.add_argument("--gap-marker", default="[REDACTED]")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-chunks", type=int, default=None)
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--use-openai-head-review", action="store_true")
    parser.add_argument("--head-review-model", default="gpt-4o")
    parser.add_argument("--min-redaction-words", type=int, default=1)
    parser.add_argument("--max-redaction-words", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--start-from",
        choices=["A", "ALIAS", "B", "C"],
        default="A",
        help="Start the pipeline from this stage. Useful when Stage A extraction already exists.",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    outputs = expected_stage_outputs(project_dir)
    steps = build_steps(args)
    stage_order = ["A", "ALIAS", "B", "C"]
    start_index = stage_order.index(args.start_from)
    steps = [step for step in steps if stage_order.index(step[0]) >= start_index]
    runnable: List[Tuple[str, List[str]]] = []
    for name, cmd in steps:
        if args.skip_existing and outputs[name].exists():
            print(f"Skipping {name}: {outputs[name]} already exists")
            continue
        runnable.append((name, cmd))

    for step_name, command in iter_progress(
        runnable,
        total=len(runnable),
        desc="kg_scripts pipeline",
    ):
        print(f"\n[{step_name}]")
        run_command(command)


if __name__ == "__main__":
    main()
