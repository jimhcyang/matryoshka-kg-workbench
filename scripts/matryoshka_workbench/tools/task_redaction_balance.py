#!/usr/bin/env python3
"""Value-redaction slot counting and balancing helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Callable

LOCATION_PAREN_SHELL_RE = re.compile(
    r"^\[redact:[^\]]+\]\s*\(\s*\[redact:[^\]]+\]\)",
    flags=re.IGNORECASE,
)
LOCATION_ADMIN_LEAD_RE = re.compile(
    r"^\[redact:[^\]]+\]\s+(?:command|district|province|provinces|village|villages)\b",
    flags=re.IGNORECASE,
)
LOCATION_PROVINCE_LIST_LEAD_RE = re.compile(
    r"^\[redact:[^\]]+\](?:\s+\[redact:[^\]]+\])?(?:\s+and\s+[A-Za-z][A-Za-z-]+)?\s+provinces?\b",
    flags=re.IGNORECASE,
)
GENERIC_MR_LEAD_RE = re.compile(
    r"^(?:general in|in)\s+mr\s*\d+\b",
    flags=re.IGNORECASE,
)
VALUE_REDACTION_PAIR_SENTENCE_WEAK_LEAD_RE = re.compile(
    r"^(?:"
    r"there are reports that|"
    r"during a meeting in or about|"
    r"personnel evacuated from|"
    r"their action is in protest against|"
    r"on the political side,\s*there are reports that"
    r")\b",
    flags=re.IGNORECASE,
)
VALUE_REDACTION_APPOSITIVE_ROLE_RE = re.compile(
    r"(?:and|,)\s+\[redact:[^\]]+\]\s*,?\s+a\s+[a-z-]+(?:\s+[a-z-]+){0,2},",
    flags=re.IGNORECASE,
)
VALUE_REDACTION_SENTENCE_HOSTILE_RE = re.compile(
    r"\b(?:the article opined|an unnamed source|an unidentified source)\b",
    flags=re.IGNORECASE,
)
VALUE_REDACTION_CONTEXT_DEPENDENT_LEAD_RE = re.compile(
    r"^(?:"
    r"the departure of|"
    r"it is unclear whether|"
    r"while the government appears to be|"
    r"when [a-z][a-z-]+ had briefed|"
    r"ambassador [a-z][a-z-]+ comments that"
    r")\b",
    flags=re.IGNORECASE,
)
VALUE_REDACTION_EXECUTION_APPOSITIVE_RE = re.compile(
    r"^in\s+\[redact:[^\]]+\]\s+[a-z][^,]+,\s+a\s+[a-z-]+,\s+(?:was|were)\b",
    flags=re.IGNORECASE,
)


def _slot_bearing_selection_key(case: dict[str, Any]) -> tuple[str, str, str]:
    """Collapse a query and a context-extended copy into one recovery case."""
    query = " ".join(str(case.get("query", "") or "").split())
    marker = re.search(r"\[redact:[^\]]+\]", query, flags=re.IGNORECASE)
    if marker:
        boundaries = list(re.finditer(r"(?<=[.!?])\s+(?=[A-Z\[])", query))
        start = max(
            (boundary.end() for boundary in boundaries if boundary.end() <= marker.start()),
            default=0,
        )
        end = min(
            (boundary.start() for boundary in boundaries if boundary.start() >= marker.end()),
            default=len(query),
        )
        query = query[start:end]
    query = (
        query.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    targets = case.get("answer_targets") or case.get("expected_targets") or {}
    target_parts: list[str] = []
    if isinstance(targets, dict):
        for slot_id, payload in sorted(targets.items()):
            if isinstance(payload, dict):
                value = payload.get("selected_value", "")
                source = payload.get("source_id", "")
            else:
                value = payload
                source = ""
            target_parts.append(f"{slot_id}:{value}@{source}".casefold())
    return (
        str(case.get("doc_id", "")).casefold(),
        "|".join(target_parts),
        query.rstrip(".!? ").casefold(),
    )


def value_redaction_slot_count(
    case: dict[str, Any],
    *,
    format_expected_targets: Callable[[dict[str, Any]], dict[str, Any]],
    count_redaction_slots: Callable[[str], int],
) -> int:
    answer_targets = case.get("answer_targets")
    if isinstance(answer_targets, dict) and answer_targets:
        return len(answer_targets)
    expected_targets = format_expected_targets(case)
    if expected_targets:
        return len(expected_targets)
    return max(1, count_redaction_slots(case.get("query", "")))


def geometric_slot_targets(
    total_count: int,
    *,
    max_slot_count: int,
    decay: float,
) -> dict[int, int]:
    weights = [decay ** (slot - 1) for slot in range(1, max_slot_count + 1)]
    normalizer = sum(weights) or 1.0
    raw_counts = {
        slot: total_count * (weights[slot - 1] / normalizer)
        for slot in range(1, max_slot_count + 1)
    }
    targets = {slot: int(raw_counts[slot]) for slot in raw_counts}
    remaining = max(0, total_count - sum(targets.values()))
    remainders = sorted(
        raw_counts,
        key=lambda slot: (raw_counts[slot] - targets[slot], -slot),
        reverse=True,
    )
    for slot in remainders[:remaining]:
        targets[slot] += 1
    return targets


def _selection_priority(case: dict[str, Any]) -> tuple[float, int]:
    query = " ".join(str(case.get("query", "") or "").split())
    lowered = query.casefold()
    query_kind = str(case.get("query_kind", "") or "").strip().casefold()
    entity_type = str(case.get("entity_type", "") or "").strip().casefold()
    selector_score = float(case.get("selector_score", 0.0) or 0.0)
    word_count = int(case.get("word_count", 0) or 0)
    slot_count = int(case.get("slot_count", 1) or 1)
    visible_anchor_count = int(case.get("visible_anchor_count", 0) or 0)
    local_support_hits = int(case.get("local_support_hits", 0) or 0)
    same_type_hits = int(case.get("same_type_hits", 0) or 0)
    has_date_or_number = bool(case.get("has_date_or_number", False))
    clean_direct_slot = bool(case.get("clean_direct_slot", False))
    label_word_count = int(case.get("label_word_count", 0) or 0)
    label_acronym_like = bool(case.get("label_acronym_like", False))
    score = selector_score

    if query_kind == "pair_sentence_slot":
        score += 4.0
        if slot_count >= 2:
            score += 6.0
        elif clean_direct_slot or visible_anchor_count >= 1 or local_support_hits >= 2:
            score += 2.0
        if slot_count == 1 and entity_type in {"location", "country"} and not has_date_or_number:
            score -= 16.0
        if slot_count == 1 and visible_anchor_count <= 0 and local_support_hits < 2:
            score -= 12.0
    elif query_kind == "entity_slot":
        if clean_direct_slot or visible_anchor_count >= 1 or local_support_hits >= 2:
            score += 5.0
    score += min(slot_count, 3) * 4.0
    score += min(visible_anchor_count, 2) * 4.0
    score += min(local_support_hits, 2) * 3.0
    if has_date_or_number:
        score += 2.0
    if clean_direct_slot:
        score += 4.0

    if slot_count == 1 and entity_type == "person":
        if query_kind == "entity_slot":
            score -= 6.0
        if re.match(
            r"^(?:source claims|there are,?\s+however,?\s+reports?\s+that|the\s+(?:visit|arrest)\s+of)\b",
            lowered,
        ):
            score -= 24.0
        if re.match(
            r"^(?:the\s+assertions\s+made\s+by|have discussed with)\s+\[redact:[^\]]+\]",
            lowered,
        ):
            score -= 18.0
        if re.match(r"^the\s+\[redact:[^\]]+\]'s\s+action\b", lowered):
            score -= 24.0
        if re.match(r"^(?:the\s+)?\[redact:[^\]]+\]\s+(?:was|is|did|does|has|had)\b", lowered):
            score -= 14.0
        if word_count > 40:
            score -= 10.0
        if visible_anchor_count <= 0 and local_support_hits <= 1:
            score -= 8.0
    if query_kind == "pair_sentence_slot" and slot_count == 1:
        if re.match(r"^(?:there are reports that|the departure of|it is unclear whether)\b", lowered):
            score -= 22.0
        if VALUE_REDACTION_PAIR_SENTENCE_WEAK_LEAD_RE.search(lowered):
            score -= 26.0
        if VALUE_REDACTION_CONTEXT_DEPENDENT_LEAD_RE.search(lowered):
            score -= 24.0
        if VALUE_REDACTION_APPOSITIVE_ROLE_RE.search(lowered):
            score -= 22.0
        if VALUE_REDACTION_EXECUTION_APPOSITIVE_RE.search(lowered):
            score -= 24.0
        if VALUE_REDACTION_SENTENCE_HOSTILE_RE.search(lowered):
            score -= 18.0
        if re.match(r"^in\s+\[redact:[^\]]+\].*?,\s+a\s+\w+,", lowered):
            score -= 18.0
        if entity_type == "political_party":
            score -= 10.0
        if (
            entity_type in {"location", "country", "organization", "government", "military", "political_party"}
            and visible_anchor_count <= 0
            and local_support_hits < 2
            and not has_date_or_number
            and not clean_direct_slot
        ):
            score -= 14.0
    if (
        slot_count == 1
        and entity_type in {"location", "country"}
        and same_type_hits > 2
        and visible_anchor_count <= 0
        and not has_date_or_number
    ):
        score -= 8.0
    if (
        slot_count == 1
        and entity_type in {"location", "country"}
        and same_type_hits >= 4
        and local_support_hits < 2
        and not has_date_or_number
    ):
        score -= 12.0
    if slot_count == 1 and entity_type in {"location", "country"}:
        if label_acronym_like and visible_anchor_count < 2:
            score -= 12.0
        if label_word_count == 1 and visible_anchor_count < 2:
            score -= 8.0
        if (
            label_acronym_like
            and visible_anchor_count <= 0
            and local_support_hits <= 2
            and not has_date_or_number
        ):
            score -= 16.0
        if re.match(r"^(?:says|heavy fighting reported)\b", lowered):
            score -= 16.0
        if re.search(r"\bmoving into\s+\[redact:[^\]]+\]\s+in the direction of\b", lowered):
            score -= 18.0
        if re.search(r"\bnorth of\s+\[redact:[^\]]+\]\b", lowered):
            score -= 14.0
        if LOCATION_PAREN_SHELL_RE.search(lowered):
            score -= 22.0
        if LOCATION_ADMIN_LEAD_RE.search(lowered):
            score -= 20.0
        if LOCATION_PROVINCE_LIST_LEAD_RE.search(lowered):
            score -= 18.0
        if GENERIC_MR_LEAD_RE.search(lowered):
            score -= 14.0
    return score, word_count


def rebalance_value_redaction_cases(
    candidates: list[dict[str, Any]],
    target_count: int,
    *,
    max_cases_per_doc: int,
    max_slot_count: int,
    slot_count_decay: float,
    format_expected_targets: Callable[[dict[str, Any]], dict[str, Any]],
    count_redaction_slots: Callable[[str], int],
) -> list[dict[str, Any]]:
    if not candidates or target_count <= 0:
        return []
    multi_slot_quality_slack = 18.0
    by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for case in candidates:
        slot_count = min(
            max_slot_count,
            max(
                1,
                value_redaction_slot_count(
                    case,
                    format_expected_targets=format_expected_targets,
                    count_redaction_slots=count_redaction_slots,
                ),
            ),
        )
        by_slot[slot_count].append(case)
    for slot in by_slot:
        by_slot[slot].sort(
            key=_selection_priority,
            reverse=True,
        )
    targets = geometric_slot_targets(
        target_count,
        max_slot_count=max_slot_count,
        decay=slot_count_decay,
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_semantic_keys: set[tuple[str, str, str]] = set()
    per_doc_counts: Counter[str] = Counter()
    slot_counts: Counter[int] = Counter()

    def peek_best(slot: int, *, ignore_doc_cap: bool = False) -> tuple[dict[str, Any] | None, float]:
        bucket = by_slot.get(slot, [])
        for case in bucket:
            case_id = str(case.get("case_id", ""))
            doc_id = str(case.get("doc_id", ""))
            if not case_id or case_id in selected_ids:
                continue
            if _slot_bearing_selection_key(case) in selected_semantic_keys:
                continue
            if not ignore_doc_cap and per_doc_counts[doc_id] >= max_cases_per_doc:
                continue
            return case, _selection_priority(case)[0]
        return None, float("-inf")

    def try_take(slot: int, *, ignore_doc_cap: bool = False) -> bool:
        case, _score = peek_best(slot, ignore_doc_cap=ignore_doc_cap)
        if case is not None:
            case_id = str(case.get("case_id", ""))
            doc_id = str(case.get("doc_id", ""))
            selected.append(case)
            selected_ids.add(case_id)
            selected_semantic_keys.add(_slot_bearing_selection_key(case))
            per_doc_counts[doc_id] += 1
            slot_counts[slot] += 1
            return True
        return False

    def try_take_best_any(*, ignore_doc_cap: bool = False) -> bool:
        best_slot: int | None = None
        best_score = float("-inf")
        for probe_slot in range(1, max_slot_count + 1):
            _probe_case, probe_score = peek_best(probe_slot, ignore_doc_cap=ignore_doc_cap)
            if probe_score > best_score:
                best_score = probe_score
                best_slot = probe_slot
        if best_slot is None or best_score == float("-inf"):
            return False
        return try_take(best_slot, ignore_doc_cap=ignore_doc_cap)

    progress = True
    while len(selected) < target_count and progress:
        progress = False
        best_any_score = float("-inf")
        for probe_slot in range(1, max_slot_count + 1):
            _probe_case, probe_score = peek_best(probe_slot)
            if probe_score > best_any_score:
                best_any_score = probe_score
        for slot in range(1, max_slot_count + 1):
            if slot_counts[slot] >= targets.get(slot, 0):
                continue
            _slot_case, slot_score = peek_best(slot)
            if _slot_case is None:
                continue
            if slot > 1 and best_any_score > float("-inf") and slot_score + multi_slot_quality_slack < best_any_score:
                continue
            if try_take(slot):
                progress = True

    while len(selected) < target_count:
        if try_take_best_any():
            continue
        if try_take_best_any(ignore_doc_cap=True):
            continue
        break
    return selected[:target_count]
