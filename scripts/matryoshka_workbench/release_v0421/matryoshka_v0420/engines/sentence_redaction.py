"""Independent raw-sentence redaction engine."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..contracts import task_row
from ..decision import nearest_alternative
from ..retrieval import canonical_tokens, containment, rank_documents
from ..semantics import canonical_key, clean_text, has_semantic_tokens, semantic_tokens
from .base import EngineContext, EngineResult, TaskEngine


_INITIALISM_FRAGMENT_RE = re.compile(r"(?:\b[A-Za-z]\.){2,}$")
_ABBREVIATED_BOUNDARY_RE = re.compile(
    r"\b(?:adm|amb|assn|col|corp|dept|dr|gen|govt|inc|min|mr|mrs|ms|"
    r"no|pres|prof|sec|st)\.$",
    flags=re.IGNORECASE,
)
_REDACTED_SENTENCE_SURFACE_RE = re.compile(
    r"\[REDACT(?:_SENT)?:[^\]]+\]|_{3,}|\bX{4,}\b|\bREDACTED\b|\u2588{2,}",
    flags=re.IGNORECASE,
)
_TELEGRAM_ROUTING_HEADER_RE = re.compile(
    r"\bFM\s+(?:AM)?EMBASSY\b.{0,100}\bTO\s+"
    r"(?:SECSTATE|STATE|AMEMBASSY)\b",
    flags=re.IGNORECASE,
)


def _expanded_canonical_tokens(context: EngineContext, value: Any) -> set[str]:
    """Use the same lexical families for sentence and document retrieval."""
    return canonical_tokens(context.semantic_resources.expanded_tokens(value))


def _sentence_views(records: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    """Expose complete source sentences after chained boundary repairs."""
    def contiguous(left: Any, right: Any) -> bool:
        return bool(
            left
            and right
            and left.chunk_id == right.chunk_id
            and left.sentence_index + 1 == right.sentence_index
        )

    def repaired_at(index: int) -> Tuple[str, str, int]:
        sentence = clean_text(records[index].sentence)
        end_index = index
        repair_modes: List[str] = []
        while end_index + 1 < len(records):
            current = records[end_index]
            following = records[end_index + 1]
            if not contiguous(current, following):
                break
            current_surface = clean_text(current.sentence)
            join_initialism = bool(
                _INITIALISM_FRAGMENT_RE.search(current_surface)
            )
            join_abbreviation = bool(
                _ABBREVIATED_BOUNDARY_RE.search(current_surface)
                and 0 < len(semantic_tokens(following.sentence)) <= 5
            )
            if not join_initialism and not join_abbreviation:
                break
            sentence = clean_text(sentence + " " + following.sentence)
            repair_modes.append(
                "initialism" if join_initialism else "abbreviation"
            )
            end_index += 1
        if "initialism" in repair_modes:
            version = "chained_initialism_boundary_repair_v0_4_21"
        elif repair_modes:
            version = "abbreviation_continuation_repair_v0_4_21"
        else:
            version = "enumeration_bridge_v0_4_9"
        return sentence, version, end_index

    views: List[Dict[str, Any]] = []
    index = 0
    while index < len(records):
        record = records[index]
        sentence, segmentation_version, end_index = repaired_at(index)
        views.append({
            "chunk_id": record.chunk_id,
            "sentence_index": record.sentence_index,
            "left_neighbor": clean_text(record.left_neighbor),
            "sentence": sentence,
            "right_neighbor": clean_text(records[end_index].right_neighbor),
            "segmentation_version": segmentation_version,
            "_start_index": index,
            "_end_index": end_index,
        })
        index = end_index + 1

    # Continuation fragments are no longer independent candidates. Rebuild
    # neighboring context from the same repaired views so adjacency scores
    # compare complete sentences on both sides.
    for index, view in enumerate(views):
        start_index = int(view.pop("_start_index"))
        end_index = int(view.pop("_end_index"))
        if index > 0:
            previous = views[index - 1]
            previous_record = records[start_index - 1]
            if contiguous(previous_record, records[start_index]):
                view["left_neighbor"] = clean_text(previous["sentence"])
        if index + 1 < len(views):
            following = views[index + 1]
            next_record = records[end_index + 1]
            if contiguous(records[end_index], next_record):
                view["right_neighbor"] = clean_text(following["sentence"])
    return views


def _sentence_span_views(
    records: Tuple[Any, ...],
    approximate_tokens: int,
) -> List[Dict[str, Any]]:
    """Add bounded contiguous spans when one sentence cannot fill the blank."""
    views = _sentence_views(records)
    if approximate_tokens < 12 or len(views) < 2:
        return views
    spans = list(views)
    lower = max(6, int(approximate_tokens * 0.55))
    upper = max(lower, int(approximate_tokens * 1.45) + 2)
    for start in range(len(views) - 1):
        combined: List[str] = [clean_text(views[start]["sentence"])]
        for end in range(start + 1, min(len(views), start + 3)):
            if views[end]["chunk_id"] != views[start]["chunk_id"]:
                break
            combined.append(clean_text(views[end]["sentence"]))
            combined_text = clean_text(" ".join(combined))
            token_count = len(semantic_tokens(combined_text))
            if token_count > upper:
                break
            if token_count < lower:
                continue
            spans.append({
                "chunk_id": views[start]["chunk_id"],
                "sentence_index": (
                    str(views[start]["sentence_index"])
                    + "-"
                    + str(views[end]["sentence_index"])
                ),
                "left_neighbor": views[start]["left_neighbor"],
                "sentence": combined_text,
                "right_neighbor": views[end]["right_neighbor"],
                "segmentation_version": "contiguous_sentence_span_v0_4_21",
            })
    return spans


class SentenceRedactionEngine(TaskEngine):
    task_name = "sentence_redaction"

    def run(self, context: EngineContext) -> EngineResult:
        slots = context.query.sentence_slots
        audit_rows: List[Dict[str, Any]] = []
        evaluation_rows: List[Dict[str, Any]] = []
        context.control.emit("sentence_redaction_started", target_count=len(slots))
        for index, slot in slots.iterrows():
            slot_id = clean_text(slot.get("sentence_slot_label", "")) or f"S{index + 1}"
            left = clean_text(
                slot.get("normalized_sentence_before", slot.get("sentence_before", ""))
            )
            right = clean_text(
                slot.get("normalized_sentence_after", slot.get("sentence_after", ""))
            )
            left_ok = has_semantic_tokens(left)
            right_ok = has_semantic_tokens(right)
            mode = "two_sided" if left_ok and right_ok else "one_sided"
            context_text = clean_text(left + " " + right)
            try:
                approximate_tokens = int(float(slot.get("approx_tokens", 0) or 0))
            except (TypeError, ValueError):
                approximate_tokens = 0
            document_limit = (
                len(context.corpus.doc_ids)
                if context.config.raw_use_mode == "discover"
                or context.config.retrieval_mode == "exhaustive"
                else context.config.staged_top_docs
            )
            ranked_docs = rank_documents(
                context_text,
                context.corpus,
                document_limit,
                context.semantic_resources,
            )
            left_tokens = context.semantic_resources.expanded_tokens(left)
            right_tokens = context.semantic_resources.expanded_tokens(right)
            family_left_tokens = _expanded_canonical_tokens(context, left)
            family_right_tokens = _expanded_canonical_tokens(context, right)
            generic_context = {
                "been", "from", "into", "over", "that", "the", "their",
                "there", "they", "this", "those", "turned", "vehicle",
            }
            context_tokens = {
                token for token in context.semantic_resources.expanded_tokens(
                    context_text
                )
                if len(token) >= 4 and token not in generic_context
            }
            generic_event_terms = {
                "event", "reported", "informed", "occurred", "happened",
                "person", "organization", "location", "authorities",
            }
            local_identity_anchors = {
                token for token in context_tokens
                if len(token) >= 5 and token not in generic_event_terms
            }
            family_context_tokens = {
                token for token in _expanded_canonical_tokens(context, context_text)
                if len(token) >= 4 and token not in generic_context
            }
            family_identity_anchors = {
                token for token in family_context_tokens
                if len(token) >= 5 and token not in generic_event_terms
            }
            candidates: List[Dict[str, Any]] = []
            for doc_id, retrieval_score in ranked_docs:
                records = context.corpus.sentences_by_doc.get(doc_id, ())
                for record in _sentence_span_views(records, approximate_tokens):
                    candidate_window = clean_text(
                        record["left_neighbor"]
                        + " "
                        + record["sentence"]
                        + " "
                        + record["right_neighbor"]
                    )
                    # Event identity is candidate-local.  A matching name or
                    # place elsewhere in a long document cannot certify an
                    # unrelated sentence boundary.
                    event_score = containment(
                        context_tokens,
                        context.semantic_resources.expanded_tokens(
                            candidate_window
                        ),
                    )
                    local_anchor_score = (
                        containment(
                            local_identity_anchors,
                            context.semantic_resources.expanded_tokens(
                                candidate_window
                            ),
                        )
                        if local_identity_anchors else 1.0
                    )
                    candidate_sentence_anchor_score = (
                        containment(
                            local_identity_anchors,
                            context.semantic_resources.expanded_tokens(
                                record["sentence"]
                            ),
                        )
                        if local_identity_anchors else 1.0
                    )
                    left_score = (
                        containment(
                            left_tokens,
                            context.semantic_resources.expanded_tokens(
                                record["left_neighbor"]
                            ),
                        )
                        if left_ok else 0.0
                    )
                    right_score = (
                        containment(
                            right_tokens,
                            context.semantic_resources.expanded_tokens(
                                record["right_neighbor"]
                            ),
                        )
                        if right_ok else 0.0
                    )
                    bridge = (
                        min(left_score, right_score)
                        if mode == "two_sided"
                        else max(left_score, right_score)
                    )
                    family_candidate_window_tokens = _expanded_canonical_tokens(
                        context, candidate_window
                    )
                    family_candidate_sentence_tokens = _expanded_canonical_tokens(
                        context, record["sentence"]
                    )
                    family_left_score = (
                        containment(
                            family_left_tokens,
                            _expanded_canonical_tokens(
                                context, record["left_neighbor"]
                            ),
                        )
                        if left_ok else 0.0
                    )
                    family_right_score = (
                        containment(
                            family_right_tokens,
                            _expanded_canonical_tokens(
                                context, record["right_neighbor"]
                            ),
                        )
                        if right_ok else 0.0
                    )
                    family_event_score = containment(
                        family_context_tokens,
                        family_candidate_window_tokens,
                    )
                    family_local_anchor_score = (
                        containment(
                            family_identity_anchors,
                            family_candidate_window_tokens,
                        )
                        if family_identity_anchors else 1.0
                    )
                    family_candidate_sentence_anchor_score = (
                        containment(
                            family_identity_anchors,
                            family_candidate_sentence_tokens,
                        )
                        if family_identity_anchors else 1.0
                    )
                    sentence_token_count = len(semantic_tokens(record["sentence"]))
                    length_fit_score = (
                        min(sentence_token_count, approximate_tokens)
                        / max(sentence_token_count, approximate_tokens)
                        if sentence_token_count and approximate_tokens else 1.0
                    )
                    if bridge <= 0.0 and event_score < context.config.sentence_event_floor:
                        continue
                    sentence_type_pass = has_semantic_tokens(
                        record["sentence"], minimum=3
                    ) and not _REDACTED_SENTENCE_SURFACE_RE.search(
                        clean_text(record["sentence"])
                    ) and not _TELEGRAM_ROUTING_HEADER_RE.search(
                        clean_text(record["sentence"])
                    )
                    adjacency_pass = (
                        left_score >= context.config.sentence_context_floor
                        and right_score >= context.config.sentence_context_floor
                        if mode == "two_sided"
                        else max(left_score, right_score)
                        >= context.config.sentence_one_sided_context_floor
                    )
                    event_pass = event_score >= context.config.sentence_event_floor
                    asymmetric_semantic_bridge = bool(
                        mode == "two_sided"
                        and max(left_score, right_score) >= 0.70
                        and min(left_score, right_score) >= 0.10
                        and event_score >= 0.45
                        and local_anchor_score >= 0.45
                        and candidate_sentence_anchor_score >= 0.10
                    )
                    if asymmetric_semantic_bridge:
                        adjacency_pass = True
                    boundary_confirmed_event = bool(
                        mode == "two_sided"
                        and min(left_score, right_score) >= 0.35
                        and event_score >= 0.30
                        and candidate_sentence_anchor_score >= 0.10
                    )
                    semantic_family_bridge = bool(
                        mode == "two_sided"
                        and max(family_left_score, family_right_score) >= 0.55
                        and min(family_left_score, family_right_score) >= 0.20
                        and family_event_score >= 0.40
                        and family_local_anchor_score >= 0.30
                        and family_candidate_sentence_anchor_score >= 0.10
                    )
                    if semantic_family_bridge:
                        adjacency_pass = True
                    local_anchor_pass = (
                        local_anchor_score
                        >= context.config.sentence_local_anchor_floor
                        or boundary_confirmed_event
                        or semantic_family_bridge
                        if local_identity_anchors else True
                    )
                    candidate_tokens = context.semantic_resources.expanded_tokens(
                        record["sentence"]
                    )
                    boundary_copy_score = max(
                        containment(candidate_tokens, left_tokens),
                        containment(candidate_tokens, right_tokens),
                    )
                    # A candidate that largely restates either visible boundary
                    # is usually that boundary itself, not the missing middle
                    # sentence.  Start the penalty above majority semantic-token
                    # containment and scale it continuously rather than applying
                    # a brittle pass/fail gate.
                    boundary_copy_penalty = 0.20 * max(
                        0.0,
                        min(1.0, (boundary_copy_score - 0.40) / 0.60),
                    )
                    eligible = (
                        sentence_type_pass
                        and adjacency_pass
                        and event_pass
                        and local_anchor_pass
                    )
                    candidates.append({
                        "slot_id": slot_id,
                        "doc_id": doc_id,
                        "raw_chunk_id": record["chunk_id"],
                        "sentence_index": record["sentence_index"],
                        "selected_sentence": record["sentence"],
                        "left_neighbor": record["left_neighbor"],
                        "right_neighbor": record["right_neighbor"],
                        "candidate_local_context": candidate_window,
                        "context_mode": mode,
                        "sentence_segmentation_version":
                            record["segmentation_version"],
                        "retrieval_score": retrieval_score,
                        "left_context_score": left_score,
                        "right_context_score": right_score,
                        "semantic_bridge_score": bridge,
                        "event_identity_score": event_score,
                        "candidate_local_anchor_score": local_anchor_score,
                        "candidate_sentence_anchor_score":
                            candidate_sentence_anchor_score,
                        "family_left_context_score": family_left_score,
                        "family_right_context_score": family_right_score,
                        "family_event_identity_score": family_event_score,
                        "family_local_anchor_score": family_local_anchor_score,
                        "family_candidate_sentence_anchor_score":
                            family_candidate_sentence_anchor_score,
                        "semantic_family_bridge_pass": semantic_family_bridge,
                        "length_fit_score": length_fit_score,
                        "boundary_copy_score": boundary_copy_score,
                        "boundary_copy_penalty": boundary_copy_penalty,
                        "sentence_type_gate_pass": sentence_type_pass,
                        "adjacency_gate_pass": adjacency_pass,
                        "event_identity_gate_pass": event_pass,
                        "candidate_local_anchor_gate_pass": local_anchor_pass,
                        "asymmetric_semantic_bridge_pass":
                            asymmetric_semantic_bridge,
                        "boundary_confirmed_event_pass":
                            boundary_confirmed_event,
                        "raw_certification_status": (
                            "RAW_VALIDATED_SUPPORT" if eligible else "RAW_CONTEXT_ONLY"
                        ),
                        "eligible": eligible,
                        "final_score": (
                            0.42 * bridge
                            + 0.08 * max(left_score, right_score)
                            + 0.20 * event_score
                            + 0.10 * local_anchor_score
                            + 0.10 * retrieval_score
                            + 0.10 * float(sentence_type_pass)
                            + (
                                0.08 * length_fit_score
                                + 0.08 * family_candidate_sentence_anchor_score
                                if semantic_family_bridge else 0.0
                            )
                            - boundary_copy_penalty
                        ),
                        "failed_gates": ";".join(
                            name for name, passed in (
                                ("SENTENCE_TYPE", sentence_type_pass),
                                ("ADJACENCY", adjacency_pass),
                                ("EVENT_IDENTITY", event_pass),
                                ("LOCAL_EVENT_ANCHORS", local_anchor_pass),
                            )
                            if not passed
                        ),
                    })
            context.control.check(len(candidates))
            distinct: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for candidate in candidates:
                identity = (
                    candidate["doc_id"],
                    clean_text(candidate["selected_sentence"]).lower(),
                )
                previous = distinct.get(identity)
                if previous is None or (
                    bool(candidate["eligible"]), float(candidate["final_score"])
                ) > (
                    bool(previous["eligible"]), float(previous["final_score"])
                ):
                    distinct[identity] = candidate
            ranked = sorted(
                distinct.values(),
                key=lambda row: (
                    -float(row["eligible"]),
                    -float(row["final_score"]),
                    row["doc_id"],
                    row["selected_sentence"],
                ),
            )
            # A bounded multi-sentence candidate is useful only when no
            # comparable single sentence explains the same text. Otherwise
            # the synthetic super-span can tie its own constituent and force
            # a false margin abstention.
            eligible_singles = [
                row for row in ranked
                if row["eligible"] and "-" not in str(row["sentence_index"])
            ]
            nonredundant_ranked: List[Dict[str, Any]] = []
            for candidate in ranked:
                if "-" not in str(candidate["sentence_index"]):
                    nonredundant_ranked.append(candidate)
                    continue
                candidate_key = canonical_key(candidate["selected_sentence"])
                redundant = any(
                    single["doc_id"] == candidate["doc_id"]
                    and canonical_key(single["selected_sentence"])
                    and canonical_key(single["selected_sentence"])
                    in candidate_key
                    and (
                        float(candidate["final_score"])
                        <= float(single["final_score"])
                        + context.config.minimum_margin
                        or float(single.get("length_fit_score", 0.0))
                        >= float(candidate.get("length_fit_score", 0.0)) - 0.02
                    )
                    for single in eligible_singles
                )
                if not redundant:
                    nonredundant_ranked.append(candidate)
            ranked = nonredundant_ranked
            answer_groups: Dict[str, List[Dict[str, Any]]] = {}
            for candidate in ranked:
                answer_key = clean_text(
                    candidate["selected_sentence"]
                ).casefold()
                answer_groups.setdefault(answer_key, []).append(candidate)
            decision_ranked: List[Dict[str, Any]] = []
            for group in answer_groups.values():
                representative = max(
                    group,
                    key=lambda row: (
                        bool(row["eligible"]),
                        float(row["final_score"]),
                        row["doc_id"],
                    ),
                )
                supporting_docs = sorted({row["doc_id"] for row in group})
                for row in group:
                    row["equivalent_answer_support_count"] = len(group)
                    row["equivalent_answer_support_doc_ids"] = "|".join(
                        supporting_docs
                    )
                decision_ranked.append(representative)
            decision_ranked.sort(
                key=lambda row: (
                    -float(row["eligible"]),
                    -float(row["final_score"]),
                    row["doc_id"],
                    row["selected_sentence"],
                )
            )
            audit_rows.extend(ranked)
            eligible = [row for row in decision_ranked if row["eligible"]]
            winner = eligible[0] if eligible else {}
            runner = eligible[1] if len(eligible) > 1 else {}
            margin = (
                float(winner.get("final_score", 0.0))
                - float(runner.get("final_score", 0.0))
                if winner and runner else None
            )
            minimum_margin = (
                context.config.minimum_margin
                if mode == "two_sided"
                else context.config.sentence_one_sided_margin
            )
            boundary_dominance = bool(
                mode == "two_sided"
                and winner
                and runner
                and float(winner.get("left_context_score", 0.0))
                >= float(runner.get("left_context_score", 0.0))
                and float(winner.get("right_context_score", 0.0))
                >= float(runner.get("right_context_score", 0.0))
                and (
                    float(winner.get("left_context_score", 0.0))
                    >= float(runner.get("left_context_score", 0.0))
                    + minimum_margin
                    or float(winner.get("right_context_score", 0.0))
                    >= float(runner.get("right_context_score", 0.0))
                    + minimum_margin
                )
            )
            boundary_novelty_dominance = bool(
                winner
                and runner
                and float(winner.get("final_score", 0.0))
                > float(runner.get("final_score", 0.0))
                and float(winner.get("boundary_copy_score", 1.0)) + 0.15
                <= float(runner.get("boundary_copy_score", 0.0))
            )
            asymmetric_boundary_dominance = bool(
                mode == "two_sided"
                and winner
                and runner
                and float(winner.get("final_score", 0.0))
                > float(runner.get("final_score", 0.0))
                and max(
                    float(winner.get("left_context_score", 0.0)),
                    float(winner.get("right_context_score", 0.0)),
                ) >= 0.75
                and max(
                    float(winner.get("left_context_score", 0.0)),
                    float(winner.get("right_context_score", 0.0)),
                ) >= max(
                    float(runner.get("left_context_score", 0.0)),
                    float(runner.get("right_context_score", 0.0)),
                ) + 0.15
                and min(
                    float(winner.get("left_context_score", 0.0)),
                    float(winner.get("right_context_score", 0.0)),
                ) >= context.config.sentence_context_floor
                and float(winner.get("event_identity_score", 0.0)) >= 0.45
                and float(winner.get("candidate_local_anchor_score", 0.0)) >= 0.45
            )
            resolved = bool(
                winner
                and (
                    not runner
                    or float(margin or 0.0) >= minimum_margin
                    or boundary_dominance
                    or boundary_novelty_dominance
                    or asymmetric_boundary_dominance
                )
            )
            nearest = nearest_alternative(
                decision_ranked,
                winner,
                identity_fields=("selected_sentence",),
            )
            status = (
                "RESOLVED_RAW_CERTIFIED" if resolved
                else "AMBIGUOUS_INSUFFICIENT_MARGIN" if winner
                else "UNRESOLVED_NO_SUPPORTED_SENTENCE"
            )
            failed_gates = ""
            if not resolved:
                failed_gates = (
                    "WINNER_MARGIN"
                    if winner and not clean_text(winner.get("failed_gates", ""))
                    else clean_text(
                        winner.get(
                            "failed_gates", "NO_ELIGIBLE_CANDIDATE"
                        )
                    )
                )
            evaluation_rows.append(task_row(
                "sentence_redaction", slot_id, status,
                localization_status=(
                    "AUTHORITATIVE_SOURCE_SELECTED"
                    if resolved
                    else "RELATED_EVIDENCE_FOUND"
                    if winner
                    else "NO_SOURCE_FOUND"
                ),
                raw_status=(
                    "RAW_VALIDATED_SUPPORT"
                    if resolved
                    else "RAW_VALIDATED_CANDIDATE_NOT_SELECTED"
                    if winner
                    else "RAW_NOT_FOUND_FOR_SELECTED_EVIDENCE"
                ),
                selected_text=(
                    clean_text(winner.get("selected_sentence", "")) if resolved else ""
                ),
                selected_source=(
                    clean_text(winner.get("doc_id", "")) if resolved else ""
                ),
                top_candidate=clean_text(ranked[0].get("doc_id", "")) if ranked else "",
                related_source=(
                    clean_text(winner.get("doc_id", ""))
                    if winner and not resolved else ""
                ),
                best_score=winner.get("final_score", ""),
                runner_up=runner.get("final_score", ""),
                margin=margin,
                decision_competitor_doc_id=clean_text(
                    runner.get("doc_id", "")
                ),
                decision_competitor_passage_id=(
                    clean_text(runner.get("raw_chunk_id", ""))
                    + "::sentence_"
                    + clean_text(runner.get("sentence_index", ""))
                    if runner else ""
                ),
                nearest_alternative_score=nearest.get("final_score"),
                nearest_alternative_doc_id=clean_text(
                    nearest.get("doc_id", "")
                ),
                nearest_alternative_passage_id=(
                    clean_text(nearest.get("raw_chunk_id", ""))
                    + "::sentence_"
                    + clean_text(nearest.get("sentence_index", ""))
                    if nearest else ""
                ),
                nearest_alternative_eligible=(
                    bool(nearest.get("eligible"))
                    if nearest else None
                ),
                nearest_alternative_failed_gates=clean_text(
                    nearest.get("failed_gates", "")
                ),
                reason=(
                    "raw_sentence_boundary_pareto_dominance"
                    if boundary_dominance
                    and runner
                    and float(margin or 0.0) < minimum_margin
                    else "raw_sentence_boundary_novelty_dominance"
                    if boundary_novelty_dominance
                    and runner
                    and float(margin or 0.0) < minimum_margin
                    else "raw_sentence_asymmetric_boundary_dominance"
                    if asymmetric_boundary_dominance
                    and runner
                    and float(margin or 0.0) < minimum_margin
                    else "raw_sentence_adjacency_bridge_event_and_margin_gates"
                ),
                unresolved="" if resolved else status,
                failed=failed_gates,
            ))
            context.control.emit(
                "sentence_redaction_target_complete",
                target_id=slot_id,
                raw_use_mode=context.config.raw_use_mode,
                shortlisted_document_count=len(ranked_docs),
                candidate_sentence_count=len(ranked),
                status=status,
            )
        return EngineResult(
            task=self.task_name,
            evaluation=pd.DataFrame(evaluation_rows),
            artifacts={
                "sentence_redaction_boundary_audit_v048.csv": pd.DataFrame(audit_rows)
            },
        )
