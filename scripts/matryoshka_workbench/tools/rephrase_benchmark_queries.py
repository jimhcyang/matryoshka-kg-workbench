#!/usr/bin/env python3
"""Clone a built benchmark and rewrite its queries for semantic robustness tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = PACKAGE_ROOT / "output" / "matryoshka_workbench" / "benchmarks"
RELEASE_ROOT = PACKAGE_ROOT / "scripts" / "matryoshka_workbench" / "release_v0421"
REDACTION_MARKER_PREFIXES = ("[REDACT:", "[REDACT_SENT:")
REDACTION_MARKER_RE = re.compile(r"\[(?:REDACT:[^\]]+|REDACT_SENT:[^\]]+)\]")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
YEAR_TOKEN_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
CRITICAL_NUMBER_TOKEN_RE = re.compile(r"\$?\d+(?:,\d{3})*(?:\.\d+)?\b")
CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:;|,\s*but\s+|,\s*while\s+|\s+but\s+|\s+while\s+|\s+yet\s+|\s+although\s+|\s+though\s+)\s*",
    flags=re.IGNORECASE,
)
ANCHOR_TAIL_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|did|does|do|has|have|had|will|would|can|could|should|may|might|must|wants|want|wanted|seeks|seek|sought|plans|plan|planned|expects|expect|expected|urges|urge|urged|continues|continue|continued|remains|remain|remained)\b(?P<tail>.+)$",
    flags=re.IGNORECASE,
)
ATTRIBUTION_FRAME_RE = re.compile(
    r"\b(?:according\s+to|say|says|said|stat(?:e|es|ed|ing)|"
    r"report(?:s|ed|ing)?|assert(?:s|ed|ing)?|alleg(?:e|es|ed|ing)|"
    r"claim(?:s|ed|ing)?|maintain(?:s|ed|ing)?|charg(?:e|es|ed|ing)|"
    r"argu(?:e|es|ed|ing)|indicat(?:e|es|ed|ing)|suggest(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)
LIGHT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "even",
    "was",
    "were",
    "with",
}

LOCAL_ANCHOR_EQUIVALENTS = {
    "appreciation": "appreciate",
    "appreciations": "appreciate",
    "aim": "seek",
    "aimed": "seek",
    "aims": "seek",
    "assassinated": "assassinate",
    "assassination": "assassinate",
    "amicable": "friendly",
    "deliberation": "deliberation",
    "deliberations": "deliberation",
    "discussion": "deliberation",
    "discussions": "deliberation",
    "departed": "leave",
    "distant": "long_range",
    "expertise": "talent",
    "friendly": "friendly",
    "goal": "goal",
    "goals": "goal",
    "journalistic": "journalistic",
    "negotiation": "negotiation",
    "negotiations": "negotiation",
    "objective": "goal",
    "objectives": "goal",
    "replace": "replace",
    "replaced": "replace",
    "replacing": "replace",
    "receive": "receive",
    "received": "receive",
    "receives": "receive",
    "rocket": "rocket",
    "rockets": "rocket",
    "pursue": "seek",
    "pursues": "seek",
    "pursuing": "seek",
    "search": "seek",
    "searches": "seek",
    "searching": "seek",
    "resolution": "solution",
    "resolutions": "solution",
    "support": "support",
    "supports": "support",
    "supported": "support",
    "supporting": "support",
    "study": "study",
    "studied": "study",
    "studies": "study",
    "studying": "study",
    "teach": "teach",
    "teaches": "teach",
    "teaching": "teach",
    "taught": "teach",
    "thank": "appreciate",
    "thanks": "appreciate",
    "thanked": "appreciate",
    "thanking": "appreciate",
    "using": "use",
    "uses": "use",
    "cites": "use",
    "citing": "use",
    "value": "value",
    "values": "value",
    "prioritize": "prioritize",
    "prioritizes": "prioritize",
    "call": "call",
    "calls": "call",
    "called": "call",
    "calling": "call",
    "persist": "unresolved",
    "persists": "unresolved",
    "persisted": "unresolved",
    "seeks": "seek",
    "seek": "seek",
    "seeking": "seek",
    "solution": "solution",
    "solutions": "solution",
    "talent": "talent",
    "talents": "talent",
    "talk": "discussion",
    "talks": "discussion",
    "want": "seek",
    "wanted": "seek",
    "wants": "seek",
    "long": "long_range",
    "range": "long_range",
    "lab": "laboratory",
    "labs": "laboratory",
    "worked": "employ",
    "working": "employ",
    "employed": "employ",
    "employment": "employ",
}
def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def normalize_query_layout(value: Any) -> str:
    lines = [
        " ".join(part.split()).strip()
        for part in str(value or "").replace("\r\n", "\n").split("\n")
    ]
    return "\n".join(line for line in lines if line).strip()


def has_attribution_frame(value: Any) -> bool:
    return bool(ATTRIBUTION_FRAME_RE.search(normalize_space(value)))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def parse_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def extract_redaction_markers(text: str) -> list[str]:
    markers: list[str] = []
    start = 0
    while True:
        index = text.find("[", start)
        if index < 0:
            break
        end = text.find("]", index)
        if end < 0:
            break
        token = text[index : end + 1]
        if token.startswith(REDACTION_MARKER_PREFIXES):
            markers.append(token)
        start = end + 1
    return markers


def redact_markers_for_similarity(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"\[REDACT:[^\]]+\]", " ", value)
    value = re.sub(r"\[REDACT_SENT:[^\]]+\]", " ", value)
    return value


def content_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in WORD_RE.findall(redact_markers_for_similarity(text))
        if token.casefold() not in LIGHT_STOPWORDS
    ]


def lexical_reuse_ratio(original: str, rewritten: str) -> float:
    original_tokens = content_tokens(original)
    rewritten_tokens = set(content_tokens(rewritten))
    if not original_tokens:
        return 1.0
    reused = sum(1 for token in original_tokens if token in rewritten_tokens)
    return reused / max(1, len(original_tokens))


def lexical_jaccard(original: str, rewritten: str) -> float:
    left = set(content_tokens(original))
    right = set(content_tokens(rewritten))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def rewrite_surface_similarity_score(
    *,
    task: str,
    original: str,
    rewritten: str,
) -> float:
    reuse = lexical_reuse_ratio(original, rewritten)
    jaccard = lexical_jaccard(original, rewritten)
    anchor_recall = minimum_clause_anchor_recall(original, rewritten)
    if task in {"value_redaction", "sentence_redaction"}:
        score = (0.40 * reuse) + (0.20 * jaccard) + (0.40 * anchor_recall)
    else:
        score = (0.35 * reuse) + (0.20 * jaccard) + (0.45 * anchor_recall)
    return max(0.0, min(1.0, score))


def normalized_anchor_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in WORD_RE.findall(redact_markers_for_similarity(text)):
        key = token.casefold()
        if key.endswith("'s") or key.endswith("’s"):
            key = key[:-2]
        if key in LIGHT_STOPWORDS:
            continue
        normalized = LOCAL_ANCHOR_EQUIVALENTS.get(key, key)
        if len(normalized) < 2:
            continue
        tokens.append(normalized)
    return tokens


def critical_year_tokens(text: str) -> list[str]:
    return YEAR_TOKEN_RE.findall(normalize_space(text))


def critical_number_tokens(text: str) -> list[str]:
    years = set(critical_year_tokens(text))
    numbers: list[str] = []
    for token in CRITICAL_NUMBER_TOKEN_RE.findall(normalize_space(text)):
        digits = re.sub(r"\D", "", token)
        if token in years:
            continue
        if len(digits) >= 2 or "$" in token or "," in token:
            numbers.append(token)
    return numbers


def critical_numeric_date_anchors_preserved(original: str, rewritten: str) -> bool:
    original_years = critical_year_tokens(original)
    rewritten_years = critical_year_tokens(rewritten)
    if original_years and original_years != rewritten_years:
        return False
    original_numbers = critical_number_tokens(original)
    rewritten_numbers = critical_number_tokens(rewritten)
    if original_numbers and original_numbers != rewritten_numbers:
        return False
    return True


def redaction_marker_windows(text: str, radius: int = 10) -> list[tuple[list[str], list[str]]]:
    windows: list[tuple[list[str], list[str]]] = []
    source = str(text or "")
    for match in REDACTION_MARKER_RE.finditer(source):
        before_tokens = normalized_anchor_tokens(source[: match.start()])[-radius:]
        after_tokens = normalized_anchor_tokens(source[match.end() :])[:radius]
        windows.append((before_tokens, after_tokens))
    return windows


def redaction_attachment_hints(text: str, radius: int = 12) -> list[str]:
    """Return visible, answer-free local frames around redaction markers."""
    source = str(text or "")
    hints: list[str] = []
    for match in REDACTION_MARKER_RE.finditer(source):
        before_matches = list(WORD_RE.finditer(source[: match.start()]))
        after_matches = list(WORD_RE.finditer(source[match.end() :]))
        left = (
            before_matches[-radius].start()
            if len(before_matches) >= radius
            else 0
        )
        right = len(source)
        if len(after_matches) > radius:
            right = match.end() + after_matches[radius - 1].end()
        frame = normalize_space(
            source[left : match.start()]
            + match.group(0)
            + source[match.end() : right]
        )
        if frame:
            hints.append(frame)
    return hints


def local_anchor_side_preserved(
    original_tokens: list[str],
    rewritten_tokens: list[str],
) -> bool:
    if not original_tokens:
        return True
    if not rewritten_tokens:
        return False
    original_set = set(original_tokens)
    rewritten_set = set(rewritten_tokens)
    shared = len(original_set & rewritten_set)
    if len(original_set) <= 3:
        required = 1
    elif len(original_set) <= 7:
        required = 2
    elif len(original_set) <= 11:
        required = 3
    else:
        required = 4
    return shared >= required


def redaction_local_anchor_preserved(original: str, rewritten: str) -> bool:
    original_windows = redaction_marker_windows(original)
    rewritten_windows = redaction_marker_windows(rewritten)
    if len(original_windows) != len(rewritten_windows):
        return False
    for (original_before, original_after), (rewritten_before, rewritten_after) in zip(
        original_windows,
        rewritten_windows,
    ):
        original_before_set = set(original_before)
        rewritten_before_set = set(rewritten_before)
        before_shared = len(original_before_set & rewritten_before_set)
        if (
            not original_after
            and not rewritten_after
            and len(original_before_set) >= 6
            and before_shared < 4
        ):
            return False
        before_ok = local_anchor_side_preserved(
            original_before,
            rewritten_before,
        )
        after_ok = local_anchor_side_preserved(
            original_after,
            rewritten_after,
        )
        if not before_ok or not after_ok:
            original_all = set(original_before + original_after)
            rewritten_all = set(rewritten_before + rewritten_after)
            total_shared = len(original_all & rewritten_all)
            if len(original_all) <= 5:
                total_required = 2
            elif len(original_all) <= 10:
                total_required = 4
            else:
                total_required = 5
            if total_shared < total_required:
                return False
    return True


def restore_exact_redaction_markers(original: str, rewritten: str) -> str:
    original_markers = extract_redaction_markers(original)
    if not original_markers:
        return rewritten
    rewritten_matches = list(REDACTION_MARKER_RE.finditer(rewritten))
    if len(rewritten_matches) != len(original_markers):
        return rewritten
    pieces: list[str] = []
    cursor = 0
    for marker, match in zip(original_markers, rewritten_matches):
        pieces.append(rewritten[cursor : match.start()])
        pieces.append(marker)
        cursor = match.end()
    pieces.append(rewritten[cursor:])
    return "".join(pieces)


def semantic_rewrite_reuse_ceiling(
    *,
    task: str,
    original: str,
) -> float:
    """Allow short executable queries to stay somewhat closer lexically."""
    token_count = len(content_tokens(original))
    ceiling = 0.82
    if token_count <= 10:
        ceiling = 0.94
    elif token_count <= 16:
        ceiling = 0.90
    elif token_count <= 24:
        ceiling = 0.86
    if task in {"value_redaction", "sentence_redaction"}:
        ceiling = min(0.96, ceiling + 0.03)
    return ceiling


def logical_clauses(text: str) -> list[str]:
    clauses = [
        normalize_space(part)
        for part in CLAUSE_SPLIT_RE.split(redact_markers_for_similarity(text))
        if normalize_space(part)
    ]
    return clauses or [normalize_space(text)]


def minimum_clause_anchor_recall(original: str, rewritten: str) -> float:
    original_clauses = [
        set(normalized_anchor_tokens(clause)) for clause in logical_clauses(original)
    ]
    rewritten_clauses = [
        set(normalized_anchor_tokens(clause)) for clause in logical_clauses(rewritten)
    ]
    original_clauses = [clause for clause in original_clauses if clause]
    rewritten_clauses = [clause for clause in rewritten_clauses if clause]
    if not original_clauses or not rewritten_clauses:
        return 1.0
    minimum = 1.0
    for original_clause in original_clauses:
        best = 0.0
        for rewritten_clause in rewritten_clauses:
            overlap = len(original_clause & rewritten_clause) / max(1, len(original_clause))
            if overlap > best:
                best = overlap
        minimum = min(minimum, best)
    return minimum


def minimum_rewrite_token_floor(
    *,
    task: str,
    mode: str,
    original: str,
) -> int:
    original_tokens = content_tokens(original)
    token_count = len(original_tokens)
    if task in {"value_redaction", "sentence_redaction"}:
        return 5 if token_count <= 12 else 6
    if mode == "semantic":
        if token_count <= 8:
            return 4
        if token_count <= 18:
            return 5
        return 6
    if token_count <= 8:
        return 4
    if token_count <= 18:
        return 5
    return 7


def clause_order_preserved(original: str, rewritten: str) -> bool:
    original_clauses = [
        set(content_tokens(clause))
        for clause in logical_clauses(original)
        if set(content_tokens(clause))
    ]
    rewritten_clauses = [
        set(content_tokens(clause))
        for clause in logical_clauses(rewritten)
        if set(content_tokens(clause))
    ]
    if len(original_clauses) <= 1 or len(rewritten_clauses) <= 1:
        return True
    matched_indices: list[int] = []
    for original_clause in original_clauses:
        best_index = -1
        best_overlap = 0.0
        for index, rewritten_clause in enumerate(rewritten_clauses):
            overlap = len(original_clause & rewritten_clause) / max(1, len(original_clause))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index < 0 or best_overlap < 0.50:
            return False
        matched_indices.append(best_index)
    return matched_indices == sorted(matched_indices)


def semantic_anchor_phrases(text: str) -> list[str]:
    anchors: list[str] = []
    for clause in logical_clauses(text):
        match = ANCHOR_TAIL_RE.search(clause)
        anchor = normalize_space(match.group("tail") if match else clause)
        anchor = re.sub(r"^(?:to\s+)", "", anchor, flags=re.IGNORECASE)
        if not anchor:
            continue
        tokens = content_tokens(anchor)
        if len(tokens) < 2:
            continue
        if len(anchor) > 90:
            words = anchor.split()
            anchor = " ".join(words[:10]).strip()
        anchors.append(anchor)
    deduped: list[str] = []
    seen: set[str] = set()
    for anchor in anchors:
        key = anchor.casefold()
        if key not in seen:
            deduped.append(anchor)
            seen.add(key)
    return deduped


def task_family(row: dict[str, str]) -> str:
    return normalize_space(
        row.get("task_family")
        or row.get("task")
        or row.get("runner_task")
    ).casefold()


def corpus_name(row: dict[str, str]) -> str:
    return normalize_space(
        row.get("corpus_full") or row.get("corpus")
    ).casefold()


def is_short_copular_claim(query: str) -> bool:
    text = normalize_space(query)
    if len(content_tokens(text)) > 10:
        return False
    return bool(
        re.search(
            r"\b(?:is|are|was|were)\b\s+(?:not\s+)?(?:an?\s+)?(?:agent(?:\s+of)?(?:\s+a)?\s+foreign power|foreign power|affiliate|ally|member|official|participant|supporter|opponent)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def is_interrogative_surface(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    if value.endswith("?"):
        return True
    if re.match(
        r"^(?:who|what|when|where|why|how|has|have|had|is|are|was|were|did|does|do|can|could|would|should|may)(?:['’]s)\b",
        value,
        flags=re.IGNORECASE,
    ):
        return False
    match = re.match(
        r"^(?:who|what|when|where|why|how|has|have|had|is|are|was|were|did|does|do|can|could|would|should|may)\b",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    if len(value) > match.end() and value[match.end()] in {"'", "’"}:
        return False
    return True


def repair_rewrite_candidate(
    *,
    task: str,
    mode: str,
    original: str,
    rewritten: str,
) -> str:
    words = normalize_query_layout(rewritten).split()
    # Remove accidental immediately repeated multiword spans emitted by a
    # one-shot rewrite (for example, "were reported were reported"). This is
    # formatting cleanup only: no source facts or benchmark answers are used.
    index = 0
    while index < len(words):
        removed = False
        for width in range(min(6, (len(words) - index) // 2), 1, -1):
            left = [
                re.sub(r"^\W+|\W+$", "", token).casefold()
                for token in words[index : index + width]
            ]
            right = [
                re.sub(r"^\W+|\W+$", "", token).casefold()
                for token in words[index + width : index + 2 * width]
            ]
            if left == right and all(left):
                del words[index + width : index + 2 * width]
                removed = True
                break
        if not removed:
            index += 1
    repaired = normalize_query_layout(" ".join(words))

    # Modal force is part of the proposition, not style. A one-shot model can
    # still substitute should/must or may/will despite the prompt. When both
    # surfaces contain exactly one unambiguous modal, restore only that modal;
    # leave multi-modal clauses untouched rather than guessing attachment.
    source_modals = re.findall(
        r"\b(?:should|must|may|might|could|would|will|can)\b",
        original,
        flags=re.IGNORECASE,
    )
    rewritten_modals = list(
        re.finditer(
            r"\b(?:should|must|may|might|could|would|will|can)\b",
            repaired,
            flags=re.IGNORECASE,
        )
    )
    if len(source_modals) == 1 and len(rewritten_modals) == 1:
        source_modal = source_modals[0].casefold()
        rewritten_modal = rewritten_modals[0]
        if rewritten_modal.group(0).casefold() != source_modal:
            repaired = normalize_query_layout(
                repaired[: rewritten_modal.start()]
                + source_modal
                + repaired[rewritten_modal.end() :]
            )

    # A one-shot rewrite is never allowed to silently mutate an explicit year,
    # quantity, or money amount. When source and rewrite contain the same
    # number of ordered numeric anchors, restore the source values in place.
    # Ambiguous insertions/deletions remain untouched and are surfaced by the
    # diagnostic validator rather than guessed.
    numeric_pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:\$\s*)?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z0-9])"
    )
    source_numeric = list(numeric_pattern.finditer(original))
    rewritten_numeric = list(numeric_pattern.finditer(repaired))
    if source_numeric and len(source_numeric) == len(rewritten_numeric):
        for source_match, rewritten_match in reversed(
            list(zip(source_numeric, rewritten_numeric))
        ):
            if source_match.group(0) != rewritten_match.group(0):
                repaired = (
                    repaired[: rewritten_match.start()]
                    + source_match.group(0)
                    + repaired[rewritten_match.end() :]
                )

    del task, mode
    return repaired


def build_prompt(
    *,
    task: str,
    query: str,
    mode: str,
    prior_feedback: str = "",
) -> str:
    style_examples: list[str] = []
    base_rules = [
        "Rewrite the analyst query as one concise intelligence-note or historian-note sentence.",
        "Preserve the same core factual claim, target, and evidence neighborhood.",
        "Keep only the main actor(s), central action, key object, polarity, and any essential time or location constraint.",
        "You may omit secondary detail, supporting clauses, and conversational wording.",
        "Do not introduce new facts, new entities, broader geography, new dates, new capabilities, or a stronger claim than the source supports.",
        "The rewrite may be slightly narrower than the original, but it must still point to the same underlying claim, contradiction, reconstruction target, or redaction answer.",
        "Use materially different wording and sentence structure from the original. Reuse fewer of the original content words when possible.",
        "Preserve retrieval-critical anchor language when it carries technical or geopolitical meaning: keep the core action head and object phrase close enough for retrieval instead of swapping in looser abstractions.",
        "Preserve explicit years, numeric quantities, and money amounts exactly.",
        "If the source contains an explicit negative marker such as not, no, never, without, denied, refused, or lacked, retain an explicit negative or denial marker in the same clause; do not hide it inside an antonym such as unconcerned, unable, infeasible, or unsatisfactory.",
        "Preserve the force of modal verbs. In particular, do not exchange should/must with expected/likely, or may/might/could with will/would.",
        "Modal strength is exact benchmark meaning: if the original says should, keep should; if it says must, keep must; if it says may, might, could, would, or will, retain that same modal rather than substituting another.",
        "Alias substitutions are allowed only when they are unambiguous in context.",
        "When a core entity has a common unambiguous alternate name or acronym/full-form in context, you may swap to that alternate surface to better resemble a natural analyst query.",
        "Do not force every entity to change; only change entity wording when the alternate name is common and still clearly points to the same real-world referent.",
        "Use a neutral, terse, factual tone.",
        "Write a single declarative sentence, not a question.",
        "Never ask a question, request information, or begin with who, what, when, where, why, how, did, does, do, is, are, was, were, can, could, should, or may.",
        "Prefer about 5 to 15 words when safe; use up to about 22 words when needed to preserve the target faithfully.",
        "Do not add metadata scaffolds such as Claim:, Stance:, Context:, Source:, Document:, or bullet headers.",
        (
            "Return JSON with keys rewritten_before, rewritten_after, and notes."
            if task == "sentence_redaction"
            else "Return JSON with keys rewritten_query and notes."
        ),
    ]
    if task == "claims":
        base_rules.extend(
            [
                "For claims, preserve one explicit predicate frame, the named actor, and one object anchor; do not collapse the claim into a generic thematic summary.",
                "If the source expresses desire, willingness, search, concern, support, recognition, or similar stance, keep that stance-bearing predicate family explicit.",
                "Keep a finite-verb claim frame such as supports, seeks, wants, denies, admits, remains, or led to; do not rewrite the claim as a bare noun phrase or topical label.",
                "When the actor is a government, party, movement, position, ambassador, or named official, keep that institutional actor explicit instead of broadening it to the country alone.",
                "If the source uses a policy, position, or attribution frame, preserve that frame explicitly instead of flattening it into a simpler thematic statement.",
            ]
        )
    if task == "contradictions":
        base_rules.extend(
            [
                "For contradictions, preserve negation, denial, uncertainty, and commitment markers strictly enough that contradiction scope does not change.",
                "Do not rewrite a negated statement into a positive summary unless the same negation or uncertainty remains explicit.",
                "Keep the negative or uncertainty word visibly attached to the same predicate; lexical antonyms are not sufficient because contradiction scope is audited explicitly.",
            ]
        )
    if task == "reconstruction":
        base_rules.extend(
            [
                "For reconstruction, retain at least one rare or technical anchor phrase from the source so the originating evidence neighborhood remains retrievable.",
                "Do not replace a distinctive institution, program, or technical noun phrase with a looser abstraction.",
                "If the source uses an attribution frame such as charges, claims, reports, or said, preserve that attribution frame explicitly instead of flattening it away.",
                "A direct assertion must remain direct. For example, rewrite 'The movement is morally entitled to the presidency' as 'The movement has a moral right to the presidency,' never as 'The movement claims entitlement.'",
                "Preserve normative versus predictive force: should/must is not equivalent to expected/likely, and the rewrite must not cross that boundary.",
            ]
        )
    base_rules.extend(
        [
            "Compress to the core proposition while keeping the same target.",
            "It is acceptable to omit minor supporting detail, but not any fact that changes what evidence should count as relevant.",
            "Prefer semantic compression over clause-by-clause paraphrase: keep the core allegation or fact pattern, change framing, and drop expendable scaffolding.",
            "If the original is already terse, still make the rewrite visibly distinct by changing sentence shape or swapping a non-critical evaluation term for a close equivalent.",
            "Prefer an analyst-style compression over a clause-by-clause paraphrase: keep the core allegation or fact pattern, but allow concise wording and lighter detail.",
        ]
    )
    style_examples = [
        "Example style: 'The NDF wants the referendum limited to whether the monarch should be replaced by a republic.'",
        "Example style: 'Razak viewed neutrality as a distant goal.'",
    ]
    if prior_feedback:
        base_rules.append(
            "Fix the previously rejected rewrite issues exactly as described below."
        )
    if task == "value_redaction":
        base_rules.extend(
            [
                "Keep every [REDACT:Rn ...] marker exactly unchanged and in the same order.",
                "Do not reveal, guess, or paraphrase the hidden redacted values.",
                "Only rewrite the visible surrounding prose.",
                "Preserve the same left-context and right-context cues around each redaction marker, even if phrased differently.",
                "The output must still contain each literal [REDACT:...] marker exactly once.",
                "Keep the same number of redaction slots and preserve which visible facts constrain each slot.",
                "Preserve each marker's semantic argument role and attachment: a subject, object, location, source, destination, possessor, organization, or person slot must remain that same role after rewriting.",
                "Keep the marker attached to the same governing noun or verb and preserve its local preposition or an unambiguously equivalent preposition.",
                "Never move a marker from one nearby relation to another; for example, a location in 'plant at [REDACT:...]' must not become the object in 'subcontractor for [REDACT:...]'.",
                "A marker coordinated with an actor must remain a participant in that action. For example, 'France and [REDACT:R1 type=LOCATION] signed an agreement to build a plant' may become 'France agreed with [REDACT:R1 type=LOCATION] to build a plant', but never 'France built a plant in [REDACT:R1 type=LOCATION]'.",
                "Equivalent prepositional wording may change without changing the role; for example, 'sought the cooperation of [REDACT:R1 type=LOCATION]' may become 'sought cooperation from [REDACT:R1 type=LOCATION]'.",
                "Keep the rewritten query as an analyst note or excerpt-style statement, not as a direct question.",
                "Treat the result as a terse archival sentence or note, not an information request.",
                "Keep the clause nearest each redaction marker especially stable.",
                "You may omit a secondary clause only if the surviving clause still constrains the same answer.",
                "Do not move a date, quantity, list neighbor, or prepositional phrase from another clause onto a redaction marker.",
                "For coordinated lists, preserve the immediate visible item before and after the marker and preserve their order.",
            ]
        )
        attachment_hints = redaction_attachment_hints(query)
        if attachment_hints:
            base_rules.append(
                "Preserve the semantic attachment shown by each original local slot frame below; these are attachment constraints, not instructions to copy the wording:"
            )
            base_rules.extend(
                f"- Slot {index}: {hint}"
                for index, hint in enumerate(attachment_hints, start=1)
            )
    elif task == "sentence_redaction":
        base_rules.extend(
            [
                "Do not include the [REDACT_SENT:Sn ...] marker in either returned side field; the caller reinserts the original marker between them.",
                "Do not reveal or paraphrase the hidden redacted sentence content.",
                "Only rewrite the visible surrounding prose.",
                "The caller will place the literal [REDACT_SENT:...] marker exactly once between rewritten_before and rewritten_after.",
                "Preserve the same visible contextual anchors before and after the sentence-redaction marker.",
                "Rewrite only the original text before the marker into rewritten_before and only the original text after the marker into rewritten_after.",
                "Never move a fact, entity, date, or relation from one side of the marker to the other.",
                "Treat the marker as an immutable sentence boundary. A clause that originally ends before the marker must remain in rewritten_before, even when you compress it; a clause that begins after the marker must remain in rewritten_after.",
                "If one original side is empty, return an empty string for that side; do not move the marker to manufacture two-sided context.",
                "Keep the rewritten query as an analyst note or excerpt-style statement, not as a direct question.",
                "Treat the result as a terse archival sentence or note, not an information request.",
                "You may compress side detail, but keep the surviving visible context sufficient to constrain the same hidden sentence.",
            ]
        )
        style_examples.append(
            "Sentence-redaction style: 'The FBI assessed Page likely met the same individual in Singapore. [REDACT_SENT:S1 approx_tokens=19] Gazprombank's authorized capital is divided into ordinary, Type A preferred, and Type B preferred shares.'"
        )
    else:
        base_rules.append(
            "Keep the query executable for semantic retrieval and preserve the original factual ask."
        )
    if task in {"claims", "contradictions", "reconstruction"}:
        base_rules.extend(
            [
                "For claim-like tasks, compress toward the core proposition rather than paraphrasing every clause.",
                "If the query contains multiple linked clauses, keep only the clauses necessary to preserve the same target and evidence neighborhood.",
                "If the original uses linked or contrastive clauses and both are necessary, preserve their original order.",
                "Preserve the original domain terms for the key capability, event, or allegation whenever possible, and change the framing around them instead.",
                "Preserve one explicit predicate frame and one concrete object anchor from the original; do not collapse the rewrite into a generic topical summary.",
                "Keep modality, intent, polarity, and stance markers explicit when they drive the claim, including willing, wants, seeks, denied, admitted, not, likely, probably, may, and might.",
                "If one actor is citing, referring to, or reaffirming another actor's position, preserve both the reporting actor and the reported proposition rather than collapsing to only the inner claim.",
                "Preserve attribution status exactly: never add said, reported, claimed, alleged, asserted, or a similar attribution when the original states the proposition directly, and never remove an attribution that the original contains.",
                "Do not rewrite a short negated copular claim into a lack-style sentence or a topical summary.",
                "Keep claim-like rewrites as short declarative analyst statements, not questions.",
            ]
        )
        if task == "contradictions":
            base_rules.append(
                "For contradictions, preserve negation and uncertainty markers strictly and keep their scope attached to the same actor and predicate."
            )
            base_rules.append(
                "For dated event contradictions, preserve the original event verb when possible instead of swapping it for a near-synonym that may shift event identity."
            )
        if task == "reconstruction":
            base_rules.append(
                "For reconstruction, retain at least one rare or technical anchor phrase from the original so the same source-document neighborhood stays salient."
            )
        anchors = semantic_anchor_phrases(query)
        if anchors:
            base_rules.append(
                "Keep the following retrieval-critical clause anchors close to the original wording:"
            )
            base_rules.extend([f"- {anchor}" for anchor in anchors])
        if is_short_copular_claim(query):
            base_rules.append(
                "If the original is a short copular classification claim, preserve that same classification frame and keep the object head close to the original instead of inventing a new predicate frame."
            )
    if mode == "semantic" and task == "value_redaction":
        base_rules.extend(
            [
                "For value_redaction, keep the immediate verb phrase and nearest noun phrase on both sides of each [REDACT:...] marker close to the original.",
                "Change broader sentence framing before changing the local slot neighborhood.",
                "Do not replace slot-local anchor terms with looser abstractions if that would change which evidence rows should score highest.",
                "Preserve the slot-local relation frame: retain the governing predicate or noun, the marker's grammatical role, and the preposition that expresses that role.",
                "Prefer an explicit local verb frame near the slot, such as moved from, sent to, denied, admitted, or building, instead of nominalized abstractions like movement, transfer, construction, or procurement.",
                "If the slot neighborhood includes a technical facility, organization, or parenthetical full-form, keep that local technical phrase close to the original.",
            ]
        )
    if style_examples:
        base_rules.append("Target style examples:")
        base_rules.extend([f"- {example}" for example in style_examples])
    query_block = "\n\nQuery:\n" + query.strip()
    if task == "sentence_redaction":
        markers = list(REDACTION_MARKER_RE.finditer(query))
        if len(markers) == 1 and markers[0].group(0).startswith("[REDACT_SENT:"):
            marker = markers[0]
            query_block = (
                "\n\nOriginal text BEFORE the immutable marker:\n"
                + query[: marker.start()].strip()
                + "\n\nImmutable marker (do not return it in either field):\n"
                + marker.group(0)
                + "\n\nOriginal text AFTER the immutable marker:\n"
                + query[marker.end() :].strip()
            )
    return (
        "\n".join(base_rules)
        + query_block
        + (
            "\n\nPrevious rejection feedback:\n" + prior_feedback.strip() + "\n"
            if prior_feedback
            else "\n"
        )
        + "\n"
    )


def build_judge_prompt(*, task: str, mode: str, original: str, rewritten: str) -> str:
    if mode == "semantic":
        return (
            "You are validating whether a rewritten analyst query stays on the same target "
            "and evidence neighborhood as the original query for a controlled retrieval benchmark.\n"
            "Be strict about target drift, but allow mild compression of secondary detail.\n"
            "Return JSON with keys:\n"
            "- same_target: boolean\n"
            "- same_task_intent: boolean\n"
            "- preserved_markers: boolean\n"
            "- preserved_critical_anchors: boolean\n"
            "- added_false_constraints: boolean\n"
            "- omitted_only_noncritical_detail: boolean\n"
            "- issues: short string\n"
            "- confidence: number between 0 and 1\n"
            "\n"
            "Rules:\n"
            "- same_target means the rewritten query should still retrieve essentially the same documents/evidence set.\n"
            "- Keep the same core entities or unambiguous aliases, timeframe, polarity, central action, and task intent.\n"
            "- preserved_critical_anchors means the rewrite keeps the retrieval-driving technical or geopolitical action/object phrases close enough that a conservative parser and matcher should still land on the same evidence.\n"
            "- Mild narrowing or omission of supporting detail is allowed only if it does not redirect retrieval to a different event or answer.\n"
            "- Introducing new facts, new entities, new geography, or new dates is NOT allowed.\n"
            "- For redaction tasks, all [REDACT:...] or [REDACT_SENT:...] markers must remain exactly preserved and in the same order.\n"
            "- For value_redaction and sentence_redaction, the visible context on both sides of each marker must still constrain the same hidden answer.\n"
            f"- Task family: {task}\n"
            "\n"
            "Original query:\n"
            f"{original.strip()}\n\n"
            "Rewritten query:\n"
            f"{rewritten.strip()}\n"
        )
    return (
        "You are validating whether a rewritten analyst query is semantically identical "
        "to the original query for a controlled retrieval benchmark.\n"
        "Be strict.\n"
        "Return JSON with keys:\n"
        "- equivalent: boolean\n"
        "- issues: short string\n"
        "- preserved_markers: boolean\n"
        "- confidence: number between 0 and 1\n"
        "\n"
        "Rules:\n"
        "- Equivalent means every factual requirement, entity, date, quantity, polarity, "
        "causal relation, stance, and task intent is preserved.\n"
        "- Preserve the same actors, acted-upon objects, and actions. A different wording is allowed; a different fact pattern is not.\n"
        "- Stylistic changes are allowed.\n"
        "- Any loss, weakening, or addition of meaning is NOT equivalent.\n"
        "- For redaction tasks, all [REDACT:...] or [REDACT_SENT:...] markers must remain "
        "exactly preserved and in the same order.\n"
        "- For value_redaction and sentence_redaction, the visible context on both sides of each marker must still constrain the same hidden answer.\n"
        f"- Task family: {task}\n"
        "\n"
        "Original query:\n"
        f"{original.strip()}\n\n"
        "Rewritten query:\n"
        f"{rewritten.strip()}\n"
    )


def sentence_redaction_rewrite_from_payload(
    original: str,
    payload: dict[str, Any],
) -> str:
    """Rebuild a sentence-gap query without allowing marker migration."""
    markers = list(REDACTION_MARKER_RE.finditer(original))
    if len(markers) != 1 or not markers[0].group(0).startswith("[REDACT_SENT:"):
        return normalize_query_layout(payload.get("rewritten_query", ""))
    marker_match = markers[0]
    original_before = normalize_query_layout(original[: marker_match.start()])
    original_after = normalize_query_layout(original[marker_match.end() :])
    before = (
        normalize_query_layout(
            REDACTION_MARKER_RE.sub(" ", str(payload.get("rewritten_before", "")))
        )
        if original_before
        else ""
    )
    after = (
        normalize_query_layout(
            REDACTION_MARKER_RE.sub(" ", str(payload.get("rewritten_after", "")))
        )
        if original_after
        else ""
    )

    def side_overlap(source: str, candidate: str) -> float:
        source_tokens = set(normalized_anchor_tokens(source))
        candidate_tokens = set(normalized_anchor_tokens(candidate))
        if not source_tokens:
            return 1.0 if not candidate_tokens else 0.0
        if not candidate_tokens:
            return 0.0
        # An analyst-style compression may retain only a small core from a
        # long boundary.  The overlap coefficient recognizes that concise
        # subset without demanding full source-side recall.
        return len(source_tokens & candidate_tokens) / min(
            len(source_tokens), len(candidate_tokens)
        )

    # Sentence recovery is an adjacency task: text from one visible boundary
    # cannot migrate across the hidden-sentence marker. Models occasionally
    # compress both source boundaries into one returned field despite the
    # prompt. Reassign unmistakably crossed model output without copying source
    # text or retrying the rewrite.
    before_own = side_overlap(original_before, before)
    before_cross = side_overlap(original_after, before)
    after_own = side_overlap(original_after, after)
    after_cross = side_overlap(original_before, after)
    migrated_before_to_after = bool(
        before
        and original_after
        and before_cross >= 0.25
        and before_cross > before_own + 0.10
    )
    migrated_after_to_before = bool(
        after
        and original_before
        and after_cross >= 0.25
        and after_cross > after_own + 0.10
    )
    if migrated_before_to_after and migrated_after_to_before:
        before, after = after, before
    elif migrated_before_to_after:
        after = normalize_query_layout(" ".join(part for part in (before, after) if part))
        before = ""
    elif migrated_after_to_before:
        before = normalize_query_layout(" ".join(part for part in (before, after) if part))
        after = ""

    # Repair partial boundary migration at sentence granularity. A model can
    # correctly rewrite both sides yet prepend one before-marker clause to the
    # after field. Move only the returned sentence whose anchors clearly belong
    # to the opposite source side; do not copy source text or call the model
    # again.
    def split_sentences(value: str) -> list[str]:
        return [
            normalize_query_layout(part)
            for part in re.split(r"(?<=[.!?])\s+", value)
            if normalize_query_layout(part)
        ]

    before_parts = split_sentences(before)
    after_parts = split_sentences(after)
    retained_before: list[str] = []
    moved_to_after: list[str] = []
    for part in before_parts:
        own = side_overlap(original_before, part)
        cross = side_overlap(original_after, part)
        if original_after and cross >= 0.40 and cross > own + 0.20:
            moved_to_after.append(part)
        else:
            retained_before.append(part)
    retained_after: list[str] = []
    moved_to_before: list[str] = []
    for part in after_parts:
        own = side_overlap(original_after, part)
        cross = side_overlap(original_before, part)
        if original_before and cross >= 0.40 and cross > own + 0.20:
            moved_to_before.append(part)
        else:
            retained_after.append(part)
    before = normalize_query_layout(" ".join((*retained_before, *moved_to_before)))
    after = normalize_query_layout(" ".join((*moved_to_after, *retained_after)))

    return normalize_query_layout(
        " ".join(
            part
            for part in (before, marker_match.group(0), after)
            if part
        )
    )


def openai_client(api_key: str):
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key, timeout=90.0, max_retries=2)


def call_openai_json(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_retries: int = 4,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "developer",
                        "content": "Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            if not model.startswith("gpt-5"):
                request_kwargs["temperature"] = 0
            response = client.chat.completions.create(
                **request_kwargs,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as exc:  # pragma: no cover - runtime API path
            last_error = exc
            sleep_seconds = min(30.0, (2**attempt) + 0.5)
            time.sleep(sleep_seconds)
    if last_error is None:
        raise RuntimeError("OpenAI rewrite failed without exception details.")
    raise last_error


def validate_rewrite(
    *,
    task: str,
    mode: str,
    original: str,
    rewritten: str,
) -> tuple[bool, str]:
    if not normalize_space(rewritten):
        return False, "empty_rewrite"
    if is_interrogative_surface(rewritten):
        return False, "interrogative_rewrite"
    if task in {"value_redaction", "sentence_redaction"}:
        original_markers = extract_redaction_markers(original)
        rewritten_markers = extract_redaction_markers(rewritten)
        if original_markers != rewritten_markers:
            return False, "marker_mismatch"
    if task == "sentence_redaction" and "[REDACT_SENT:" not in rewritten:
        return False, "missing_sentence_marker"
    if task == "value_redaction" and "[REDACT:" not in rewritten:
        return False, "missing_value_marker"
    if len(content_tokens(rewritten)) < minimum_rewrite_token_floor(
        task=task,
        mode=mode,
        original=original,
    ):
        return False, "rewrite_too_short"
    if mode == "semantic":
        reuse = lexical_reuse_ratio(original, rewritten)
        if (
            reuse > semantic_rewrite_reuse_ceiling(
                task=task,
                original=original,
            )
            and not (
                task in {"claims", "contradictions", "reconstruction"}
                and is_short_copular_claim(original)
                and normalize_space(original) == normalize_space(rewritten)
            )
        ):
            return False, "semantic_rewrite_too_close"
        if task in {"claims", "contradictions", "reconstruction"}:
            if is_interrogative_surface(rewritten):
                return False, "semantic_interrogative_rewrite"
            if not critical_numeric_date_anchors_preserved(original, rewritten):
                return False, "semantic_numeric_anchor_drift"
            clause_recall = minimum_clause_anchor_recall(original, rewritten)
            if len(logical_clauses(original)) > 1 and clause_recall < 0.60:
                return False, "semantic_clause_anchor_loss"
            if len(logical_clauses(original)) > 1 and not clause_order_preserved(original, rewritten):
                return False, "semantic_clause_order_changed"
            if len(logical_clauses(original)) <= 1 and len(content_tokens(original)) <= 16:
                if clause_recall < 0.35:
                    return False, "semantic_short_anchor_loss"
            if task in {"claims", "contradictions"} and is_short_copular_claim(original):
                if not re.search(r"\b(?:is|are|was|were)\b", rewritten, flags=re.IGNORECASE):
                    return False, "semantic_short_copular_frame_drift"
            if has_attribution_frame(original) != has_attribution_frame(rewritten):
                return False, "semantic_attribution_frame_drift"
        if task == "value_redaction" and not redaction_local_anchor_preserved(
            original,
            rewritten,
        ):
            return False, "semantic_redaction_local_anchor_loss"
    return True, "ok"


def judge_rewrite(
    *,
    client: Any,
    model: str,
    task: str,
    mode: str,
    original: str,
    rewritten: str,
) -> dict[str, Any]:
    payload = call_openai_json(
        client=client,
        model=model,
        prompt=build_judge_prompt(
            task=task,
            mode=mode,
            original=original,
            rewritten=rewritten,
        ),
    )
    issues = normalize_space(payload.get("issues", ""))
    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    if mode == "semantic":
        return {
            "equivalent": bool(payload.get("same_target", False))
            and bool(payload.get("same_task_intent", False))
            and bool(payload.get("preserved_critical_anchors", False))
            and not bool(payload.get("added_false_constraints", False)),
            "same_target": bool(payload.get("same_target", False)),
            "same_task_intent": bool(payload.get("same_task_intent", False)),
            "preserved_markers": bool(payload.get("preserved_markers", False)),
            "preserved_critical_anchors": bool(payload.get("preserved_critical_anchors", False)),
            "added_false_constraints": bool(payload.get("added_false_constraints", False)),
            "omitted_only_noncritical_detail": bool(payload.get("omitted_only_noncritical_detail", False)),
            "issues": issues,
            "confidence": max(0.0, min(1.0, confidence)),
        }
    equivalent = bool(payload.get("equivalent", False))
    return {
        "equivalent": equivalent,
        "preserved_markers": bool(payload.get("preserved_markers", False)),
        "issues": issues,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def parser_accepts_rewrite(
    *,
    task: str,
    rewritten: str,
    parser_model: str,
    api_key_env: str,
) -> tuple[bool, str]:
    task = normalize_space(task).casefold()
    if task not in {"claims", "contradictions", "reconstruction"}:
        return True, "not_checked"
    if not normalize_space(parser_model):
        return True, "not_requested"
    release_root = str(RELEASE_ROOT)
    if release_root not in sys.path:
        sys.path.insert(0, release_root)
    from matryoshka_v0420.parser import build_query_task_preflight, parse_query_text  # type: ignore

    validation_runs = 2 if task in {"claims", "contradictions"} else 1
    observed_signatures: set[str] = set()
    blocked_details: list[str] = []
    for _ in range(validation_runs):
        parsed, _parser_mode, _warnings = parse_query_text(
            rewritten,
            use_openai=True,
            openai_model=parser_model,
            api_key_env=api_key_env,
        )
        preflight = build_query_task_preflight(parsed)
        if not isinstance(preflight, pd.DataFrame) or preflight.empty:
            return False, "parser_preflight_missing"
        relevant = preflight[
            preflight["task"].fillna("").astype(str).eq("claims_or_contradictions")
        ]
        if relevant.empty:
            return False, "parser_preflight_missing"
        signature_rows: list[str] = []
        for _, row in relevant.iterrows():
            signature_rows.append(
                "|".join(
                    [
                        normalize_space(row.get("status", "")),
                        normalize_space(row.get("reason", "")),
                    ]
                )
            )
        observed_signatures.add(" || ".join(sorted(signature_rows)))
        blocked = relevant[
            ~relevant["status"].fillna("").astype(str).str.startswith("EXECUTABLE")
        ]
        if not blocked.empty:
            for _, row in blocked.iterrows():
                status = normalize_space(row.get("status", ""))
                reason = normalize_space(row.get("reason", ""))
                blocked_details.append(f"{status}:{reason}")
    if blocked_details:
        return False, "parser_preflight_blocked:" + " | ".join(sorted(set(blocked_details)))[:240]
    if len(observed_signatures) > 1:
        return True, "ok_unstable"
    return True, "ok"


def update_plan_queries(
    plan_path: Path,
    rewritten_by_case: dict[str, str],
) -> None:
    if not plan_path.exists():
        return
    payload = read_json(plan_path)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return
    changed = False
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = normalize_space(case.get("id", ""))
        if case_id and case_id in rewritten_by_case:
            case["query"] = rewritten_by_case[case_id]
            changed = True
    if changed:
        write_json(plan_path, payload)


def clone_experiment(
    *,
    source_root: Path,
    dest_root: Path,
    overwrite: bool,
) -> None:
    if dest_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination experiment already exists: {dest_root}"
            )
        shutil.rmtree(dest_root)
    shutil.copytree(
        source_root,
        dest_root,
        ignore=shutil.ignore_patterns("runs", ".matryoshka_cache"),
    )
    (dest_root / "runs").mkdir(parents=True, exist_ok=True)
    # Build diagnostics describe the source selection pass and are identical in
    # both branches. Keep one copy in the original experiment; the rephrased
    # branch retains only its rewritten catalog and rewrite-specific reports.
    for inherited_name in ("build_diagnostics.json", "build_stage_counts.csv"):
        inherited_path = dest_root / "summaries" / inherited_name
        if inherited_path.exists():
            inherited_path.unlink()


def refresh_summary_catalogs(
    summary_root: Path,
    rewritten_by_case: dict[str, str],
) -> None:
    for filename in ("final_query_catalog.csv",):
        path = summary_root / filename
        rows = parse_csv(path)
        if not rows:
            continue
        changed = False
        for row in rows:
            case_id = normalize_space(
                row.get("case_id") or row.get("id")
            )
            if not case_id or case_id not in rewritten_by_case:
                continue
            if "query" in row:
                row["query"] = rewritten_by_case[case_id]
                changed = True
            if "query_text" in row:
                row["query_text"] = rewritten_by_case[case_id]
                changed = True
        if changed:
            write_csv(path, rows)


def resolve_source_root(args: argparse.Namespace) -> Path:
    if args.source_experiment_root:
        return Path(args.source_experiment_root).expanduser().resolve()
    if not args.source_experiment_name:
        raise ValueError(
            "Provide --source-experiment-root or --source-experiment-name."
        )
    return (BENCHMARK_ROOT / args.source_experiment_name).resolve()


def parse_name_filter(raw: str) -> set[str]:
    return {
        normalize_space(part).casefold()
        for part in raw.split(",")
        if normalize_space(part)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--source-experiment-root", default="")
    parser.add_argument("--source-experiment-name", default="")
    parser.add_argument("--dest-experiment-name", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--parser-validation-model", default="")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--corpora", default="")
    parser.add_argument(
        "--mode",
        choices=["semantic"],
        default="semantic",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = normalize_space(os.environ.get(args.openai_api_key_env, ""))
    if not api_key:
        raise EnvironmentError(
            f"Missing {args.openai_api_key_env} for query rephrasing."
        )

    source_root = resolve_source_root(args)
    if not source_root.exists():
        raise FileNotFoundError(f"Source experiment not found: {source_root}")
    dest_root = (BENCHMARK_ROOT / args.dest_experiment_name).resolve()
    clone_experiment(
        source_root=source_root,
        dest_root=dest_root,
        overwrite=args.overwrite,
    )

    task_filter = parse_name_filter(args.tasks)
    corpus_filter = parse_name_filter(args.corpora)
    manifest_path = dest_root / "query_manifests" / "case_manifest.csv"
    answer_key_path = dest_root / "query_manifests" / "query_answer_key.csv"
    manifest_rows = parse_csv(manifest_path)
    answer_rows = parse_csv(answer_key_path)
    answer_by_case = {
        normalize_space(row.get("case_id", "")): row for row in answer_rows
    }

    client = openai_client(api_key)
    rewritten_by_case: dict[str, str] = {}
    audit_rows: list[dict[str, Any]] = []
    judge_model = normalize_space(args.judge_model) or args.model

    for row in manifest_rows:
        case_id = normalize_space(row.get("case_id", ""))
        task = task_family(row)
        corpus = corpus_name(row)
        if task_filter and task not in task_filter:
            continue
        if corpus_filter and corpus not in corpus_filter:
            continue
        query_rel = normalize_space(row.get("query_file", ""))
        if not case_id or not query_rel:
            continue
        query_path = dest_root / query_rel
        original_query = query_path.read_text(encoding="utf-8").strip()
        rewritten_query = original_query
        valid = False
        validation_reason = "not_attempted"
        final_reason = "not_attempted"
        parser_validation_reason = "not_checked"
        parser_ok = True
        judge_result: dict[str, Any] = {
            "equivalent": False,
            "preserved_markers": False,
            "issues": "",
            "confidence": 0.0,
        }
        prompt = build_prompt(
            task=task,
            query=original_query,
            mode=args.mode,
            prior_feedback="",
        )
        payload = call_openai_json(
            client=client,
            model=args.model,
            prompt=prompt,
        )
        candidate_query = (
            sentence_redaction_rewrite_from_payload(original_query, payload)
            if task == "sentence_redaction"
            else normalize_query_layout(payload.get("rewritten_query", ""))
        )
        if task in {"value_redaction", "sentence_redaction"}:
            candidate_query = normalize_query_layout(
                restore_exact_redaction_markers(original_query, candidate_query)
            )
        if not normalize_space(candidate_query):
            raise ValueError(f"Rewrite model returned an empty query for {case_id}.")
        raw_candidate_query = candidate_query
        candidate_query = repair_rewrite_candidate(
            task=task,
            mode=args.mode,
            original=original_query,
            rewritten=candidate_query,
        )
        repair_applied = sha1_text(raw_candidate_query) != sha1_text(candidate_query)
        valid, validation_reason = validate_rewrite(
            task=task,
            mode=args.mode,
            original=original_query,
            rewritten=candidate_query,
        )
        judge_result = judge_rewrite(
            client=client,
            model=judge_model,
            task=task,
            mode=args.mode,
            original=original_query,
            rewritten=candidate_query,
        )
        if normalize_space(args.parser_validation_model):
            parser_ok, parser_validation_reason = parser_accepts_rewrite(
                task=task,
                rewritten=candidate_query,
                parser_model=args.parser_validation_model,
                api_key_env=args.openai_api_key_env,
            )
        rewritten_query = candidate_query
        surface_similarity_score = rewrite_surface_similarity_score(
            task=task,
            original=original_query,
            rewritten=rewritten_query,
        )
        surface_distance_score = 1.0 - surface_similarity_score
        strict_runtime_ready = bool(
            valid
            and judge_result.get("equivalent", False)
            and (
                task not in {"value_redaction", "sentence_redaction"}
                or judge_result.get("preserved_markers", False)
            )
            and parser_ok
        )
        diagnostic_flags: list[str] = []
        if not valid:
            diagnostic_flags.append(validation_reason)
        if not judge_result.get("equivalent", False):
            diagnostic_flags.append("judge_non_equivalent")
        if (
            task in {"value_redaction", "sentence_redaction"}
            and not judge_result.get("preserved_markers", False)
        ):
            diagnostic_flags.append("judge_marker_warning")
        if (
            not parser_ok
            and normalize_space(parser_validation_reason)
            and parser_validation_reason not in {"not_checked", "not_requested"}
        ):
            diagnostic_flags.append(parser_validation_reason)
        final_reason = (
            "diagnostic_ok"
            if not diagnostic_flags
            else "diagnostic:" + " | ".join(diagnostic_flags)
        )
        query_path.write_text(rewritten_query + "\n", encoding="utf-8")
        rewritten_by_case[case_id] = rewritten_query
        answer_row = answer_by_case.get(case_id, {})
        if answer_row:
            answer_row["query_text"] = rewritten_query
            answer_row["query_word_count"] = str(
                len(rewritten_query.split())
            )
        audit_rows.append(
            {
                "case_id": case_id,
                "corpus": row.get("corpus_full", row.get("corpus", "")),
                "task_family": task,
                "rewrite_mode": args.mode,
                "query_file": query_rel,
                "rewrite_valid": valid,
                "validation_reason": validation_reason,
                "final_reason": final_reason,
                "judge_equivalent": judge_result.get("equivalent", False),
                "judge_preserved_markers": judge_result.get("preserved_markers", False),
                "judge_same_target": judge_result.get("same_target", ""),
                "judge_same_task_intent": judge_result.get("same_task_intent", ""),
                "judge_preserved_critical_anchors": judge_result.get("preserved_critical_anchors", ""),
                "judge_added_false_constraints": judge_result.get("added_false_constraints", ""),
                "judge_omitted_only_noncritical_detail": judge_result.get("omitted_only_noncritical_detail", ""),
                "judge_confidence": judge_result.get("confidence", 0.0),
                "judge_issues": judge_result.get("issues", ""),
                "parser_validation_reason": parser_validation_reason,
                "parser_validation_ok": parser_ok,
                "attempt_count": 1,
                "repair_applied": repair_applied,
                "lexical_reuse_ratio": f"{lexical_reuse_ratio(original_query, rewritten_query):.4f}",
                "lexical_jaccard": f"{lexical_jaccard(original_query, rewritten_query):.4f}",
                "minimum_clause_anchor_recall": f"{minimum_clause_anchor_recall(original_query, rewritten_query):.4f}",
                "surface_similarity_score": f"{surface_similarity_score:.4f}",
                "surface_distance_score": f"{surface_distance_score:.4f}",
                "runtime_ready": True,
                "strict_runtime_ready": strict_runtime_ready,
                "rewrite_policy": "first_shot_local_repair_no_retry_no_fallback_always_execute",
                "changed": (
                    sha1_text(original_query) != sha1_text(rewritten_query)
                ),
                "original_query": original_query,
                "raw_rewritten_query": raw_candidate_query,
                "rewritten_query": rewritten_query,
                "notes": payload.get("notes", ""),
            }
        )
        print(
            f"[rewrite] {case_id}: "
            f"{'changed' if audit_rows[-1]['changed'] else 'unchanged'} "
            f"({final_reason}; distance={surface_distance_score:.3f}; strict_ready={strict_runtime_ready})",
            flush=True,
        )

    if answer_rows:
        write_csv(answer_key_path, answer_rows)

    for plan_path in sorted(
        (dest_root / "query_manifests").glob("benchmark_plan*.json")
    ):
        update_plan_queries(plan_path, rewritten_by_case)

    refresh_summary_catalogs(
        dest_root / "summaries",
        rewritten_by_case,
    )

    manifest = {
        "source_experiment": source_root.name,
        "dest_experiment": dest_root.name,
        "rewrite_model": args.model,
        "judge_model": judge_model,
        "rewrite_mode": args.mode,
        "rewrite_policy": "first_shot_local_repair_no_retry_no_fallback_always_execute",
        "max_attempts": 1,
        "allow_original_fallback": False,
        "tasks_filter": sorted(task_filter),
        "corpora_filter": sorted(corpus_filter),
        "rewritten_case_count": len(rewritten_by_case),
        "openai_api_key_env": args.openai_api_key_env,
    }
    write_json(
        dest_root / "summaries" / "query_rephrase_manifest.json",
        manifest,
    )
    write_csv(
        dest_root / "summaries" / "query_rephrase_audit.csv",
        audit_rows,
    )
    print(
        "Rephrased benchmark written to: "
        + str(dest_root),
        flush=True,
    )


if __name__ == "__main__":
    main()
