#!/usr/bin/env python3
"""Dependency, release-file, extraction, and optional OpenAI preflight."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


REQUIRED_MODULES = ("pandas", "numpy", "streamlit")
REQUIRED_FILES = (
    "query_semantic_parser_v0_4_15.py",
    "external_query_semantic_matcher_v0_4_20.py",
    "run_coordinated_v0_4_21.py",
    "retry_llm_digest_v0_4_14.py",
    "streamlit_semantic_workbench_v3_19.py",
    "automation/matryoshka_batch_runner_v1_0_18.py",
    "semantic_resources/core_semantic_resources_v2.json",
    "requirements.txt",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", default="")
    parser.add_argument("--require-openai", action="store_true")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    issues = []
    versions = {}
    if sys.version_info < (3, 11):
        issues.append("Python 3.11 or newer is required.")
    for module_name in REQUIRED_MODULES + (
        ("openai",) if args.require_openai else ()
    ):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = str(
                getattr(module, "__version__", "installed")
            )
        except Exception as exc:
            issues.append(f"{module_name} import failed: {type(exc).__name__}: {exc}")
    missing_files = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing_files:
        issues.append("Missing release files: " + ", ".join(missing_files))
    extraction = (
        Path(args.extraction_dir).expanduser().resolve()
        if args.extraction_dir else None
    )
    if extraction and not extraction.is_dir():
        issues.append("Extraction directory not found: " + str(extraction))
    key_name = args.openai_api_key_env.strip() or "OPENAI_API_KEY"
    key_available = bool(os.getenv(key_name, "").strip())
    if args.require_openai and not key_available:
        issues.append(f"Environment variable {key_name} is empty.")
    result = {
        "status": "PASS" if not issues else "FAIL",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "versions": versions,
        "release": {
            "query_parser": "v0.4.15",
            "semantic_matcher": "v0.4.20",
            "digest_builder": "v0.4.14",
            "streamlit_workbench": "v3.19",
            "automation": "v1.0.18",
        },
        "release_files_present": not missing_files,
        "extraction_directory_present": (
            extraction.is_dir() if extraction else "NOT_CHECKED"
        ),
        "openai_requested": args.require_openai,
        "openai_key_environment": key_name,
        "openai_key_available": key_available,
        "issues": issues,
        "install_command": (
            f"{sys.executable} -m pip install -r {root / 'requirements.txt'}"
        ),
    }
    print(json.dumps(result, indent=2))
    return 0 if not issues else 4


if __name__ == "__main__":
    raise SystemExit(main())
