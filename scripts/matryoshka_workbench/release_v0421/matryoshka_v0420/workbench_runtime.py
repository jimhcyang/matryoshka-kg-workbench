"""Pure Workbench v3.14 process/session isolation helpers."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from typing import Any, Mapping, Optional


def resolve_credential_source(
    source: str,
    environment_name: str,
    session_key: str = "",
    *,
    environment: Optional[Mapping[str, str]] = None,
    streamlit_secrets: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Resolve one explicit source without mutating process state."""
    name = str(environment_name or "OPENAI_API_KEY").strip()
    process_environment = environment if environment is not None else os.environ
    secrets = streamlit_secrets or {}
    if source == "Session-only key":
        value = str(session_key or "").strip()
        source_name = "session-only browser field"
    elif source == "Streamlit secret":
        value = str(secrets.get(name, "") or "").strip()
        source_name = "streamlit secret:" + name
    else:
        value = str(process_environment.get(name, "") or "").strip()
        source_name = "process environment:" + name
    return {
        "source": source,
        "source_name": source_name,
        "environment_name": name,
        "populated": bool(value),
        "value": value,
    }


def collision_safe_run_name(
    label: str = "",
    *,
    timestamp: Optional[datetime] = None,
    nonce: str = "",
) -> str:
    """Return a process-safe run name with timestamp, PID, and random nonce."""
    moment = timestamp or datetime.now()
    safe_label = re.sub(
        r"[^A-Za-z0-9_-]+", "_", str(label or "")
    ).strip("_")[:60]
    suffix = re.sub(
        r"[^A-Za-z0-9]+", "", nonce
    )[:12] or uuid.uuid4().hex[:12]
    name = (
        moment.strftime("run_%Y%m%d_%H%M%S_%f")
        + f"_p{os.getpid()}_{suffix}"
    )
    return name + ("_" + safe_label if safe_label else "")


def validate_instance_port(port: int) -> int:
    value = int(port)
    if not 1 <= value <= 65535:
        raise ValueError("Streamlit port must be between 1 and 65535.")
    return value


def terminal_status_for_exited_process(
    returncode: Optional[int],
    current_status: Mapping[str, Any],
    stderr_tail: str = "",
) -> Optional[dict[str, Any]]:
    """Translate a dead child with no terminal status into an explicit state."""
    if returncode is None:
        return None
    terminal = {
        "COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED", "PARSER_FAILED",
        "PREFLIGHT_BLOCKED", "PAIR_RECONCILIATION_FAILED", "CANCELLED",
        "PROCESS_EXITED_NONZERO", "PROCESS_EXITED_WITHOUT_TERMINAL_STATUS",
    }
    prior = str(current_status.get("status", "") or "").strip()
    if prior in terminal:
        return None
    state = (
        "PROCESS_EXITED_NONZERO"
        if int(returncode) != 0
        else "PROCESS_EXITED_WITHOUT_TERMINAL_STATUS"
    )
    return {
        **dict(current_status),
        "status": state,
        "prior_status": prior or "STATUS_NOT_WRITTEN",
        "return_code": int(returncode),
        "error_detail": str(stderr_tail or "").strip()[-4000:],
    }


def summarize_llm_validation_failures(
    state: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Group digest validation reasons by target without masking fallback."""
    requested = [
        str(value).strip()
        for value in evidence_payload.get("requested_target_ids", [])
        if str(value).strip()
    ]
    if not requested:
        requested = [
            str(item.get("target_id", "")).strip()
            for item in (
                evidence_payload.get("targets", [])
                or evidence_payload.get("units", [])
            )
            if str(item.get("target_id", "")).strip()
        ]
    requested = list(dict.fromkeys(requested))
    reasons = [
        str(value).strip()
        for value in state.get("reasons", [])
        if str(value).strip()
    ]
    by_target = {target_id: [] for target_id in requested}
    global_reasons = []
    for reason in reasons:
        matched = False
        fields = reason.split(":")
        for target_id in requested:
            if target_id in fields:
                by_target[target_id].append(reason)
                matched = True
        if not matched:
            global_reasons.append(reason)
    failed = [
        target_id for target_id in requested
        if by_target[target_id]
    ]
    return {
        "requested_target_count": len(requested),
        "failed_target_count": len(failed),
        "failed_target_ids": failed,
        "per_target_reasons": {
            target_id: by_target[target_id]
            for target_id in failed
        },
        "global_reasons": global_reasons,
        "deterministic_fallback_authoritative": (
            state.get("authoritative_fallback")
            == "deterministic_digest"
        ),
    }
