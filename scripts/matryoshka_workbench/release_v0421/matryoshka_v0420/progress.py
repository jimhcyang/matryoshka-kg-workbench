"""Append-only progress, cancellation, and bounded-work controls."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .semantics import utc_now_iso


class CancelledError(RuntimeError):
    pass


class WorkLimitError(RuntimeError):
    pass


@dataclass
class StageTimingRecorder:
    """Reconcile owned stage time with coordinated wall time.

    Callers record measured stages.  The recorder does not infer ownership for
    gaps; it exposes them explicitly as unexplained overhead.
    """

    total_wall_seconds: float = 0.0
    stages: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        stage: str,
        elapsed_seconds: float,
        *,
        owner: str,
        state: str = "COMPLETED",
        attempt: int = 1,
        cache_state: str = "",
    ) -> None:
        elapsed = max(0.0, float(elapsed_seconds))
        self.stages.append({
            "stage": str(stage),
            "owner": str(owner),
            "elapsed_seconds": elapsed,
            "state": str(state),
            "attempt": int(attempt),
            "cache_state": str(cache_state),
        })

    def report(self, *, tolerance_seconds: float = 0.25) -> Dict[str, Any]:
        owned = sum(
            float(item["elapsed_seconds"]) for item in self.stages
        )
        total = max(0.0, float(self.total_wall_seconds))
        overhead = max(0.0, total - owned)
        overcount = max(0.0, owned - total)
        within_tolerance = overcount <= max(
            0.0, float(tolerance_seconds)
        )
        return {
            "schema_version":
                "matryoshka_coordinated_performance_v0_4_17",
            "total_wall_seconds": total,
            "owned_stage_seconds": owned,
            "unexplained_overhead_seconds": overhead,
            "overcount_seconds": overcount,
            "tolerance_seconds": max(
                0.0, float(tolerance_seconds)
            ),
            "reconciliation_state": (
                "PASS_WITH_OVERHEAD"
                if within_tolerance and overhead > tolerance_seconds
                else "PASS"
                if within_tolerance
                else "FAIL_STAGE_OVERCOUNT"
            ),
            "stages": [dict(item) for item in self.stages],
        }


@dataclass
class RunControl:
    output_dir: Path
    max_candidates_per_target: int = 100_000
    max_elapsed_seconds: float = 0.0
    started: float = field(default_factory=time.perf_counter)

    @property
    def progress_path(self) -> Path:
        return self.output_dir / "progress_v048.jsonl"

    @property
    def cancel_path(self) -> Path:
        return self.output_dir / "CANCEL"

    def check(self, candidate_count: int = 0) -> None:
        if self.cancel_path.exists():
            raise CancelledError("Cancellation requested through output/CANCEL.")
        if candidate_count > self.max_candidates_per_target:
            raise WorkLimitError(
                f"Candidate safety limit exceeded: {candidate_count} > "
                f"{self.max_candidates_per_target}"
            )
        if self.max_elapsed_seconds > 0:
            elapsed = time.perf_counter() - self.started
            if elapsed > self.max_elapsed_seconds:
                raise WorkLimitError(
                    f"Run time limit exceeded: {elapsed:.1f}s > {self.max_elapsed_seconds:.1f}s"
                )

    def emit(self, stage: str, **values: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record: Dict[str, Any] = {
            "stage": stage,
            "elapsed_seconds": time.perf_counter() - self.started,
            "timestamp": utc_now_iso(),
            **values,
        }
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
