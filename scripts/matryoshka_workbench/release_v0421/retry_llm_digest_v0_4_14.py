#!/usr/bin/env python3
"""Retry only the optional explanatory digest for a completed v0.4 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from matryoshka_v0420.contracts import output_inventory, safe_json
from matryoshka_v0420.matcher import _generate_llm_digest
from matryoshka_v0420.semantics import atomic_write_json, utc_now_iso


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retry a v0.4 Optional LLM digest without rerunning parsing or matching."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-max-output-tokens", type=int, default=1600)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    pair_level = (
        output_dir.joinpath("semantic_match_pair_digest_v0413.md").is_file()
        and output_dir.joinpath("task_evidence_units_v0413.json").is_file()
    )
    evidence_path = output_dir / "task_evidence_units_v0413.json"
    digest_path = output_dir / (
        "semantic_match_pair_digest_v0413.md"
        if pair_level else "semantic_match_task_digest_v0413.md"
    )
    manifest_path = output_dir / "semantic_match_run_manifest_v048.json"
    if not evidence_path.is_file() or not digest_path.is_file():
        raise FileNotFoundError(
            "The directory has no completed v0.4.13-schema evidence/digest "
            "bundle for Digest Builder v0.4.14."
        )
    evidence_payload = json.loads(
        evidence_path.read_text(encoding="utf-8")
    )
    options = SimpleNamespace(
        use_llm_digest=True,
        llm_model=args.llm_model,
        openai_api_key_env=args.openai_api_key_env,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_max_retries=args.llm_max_retries,
        llm_max_output_tokens=args.llm_max_output_tokens,
    )
    state = _generate_llm_digest(
        options,
        evidence_payload,
        digest_path.read_text(encoding="utf-8"),
        output_dir,
        pair_level=pair_level,
    )
    state["digest_only_retry"] = True
    state["retry_completed_at"] = utc_now_iso()
    atomic_write_json(
        output_dir / (
            "pair_llm_digest_state_v0413.json"
            if pair_level else "llm_digest_state_v0413.json"
        ),
        state,
    )
    if manifest_path.is_file():
        manifest = safe_json(manifest_path)
        manifest["llm_digest_state"] = state
        manifest["output_inventory"] = output_inventory(output_dir)
        atomic_write_json(manifest_path, manifest)
    print(state["state"])
    return 0 if state["state"] == "VALIDATED_EXPLANATORY" else 4


if __name__ == "__main__":
    raise SystemExit(main())
