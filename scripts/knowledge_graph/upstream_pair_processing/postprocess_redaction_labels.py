from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai_pdf_pipeline import WORD_RE, align_monotonic, clean_ocr_text


MONTH_TOKENS = {
    "jan",
    "january",
    "feb",
    "february",
    "mar",
    "march",
    "apr",
    "april",
    "may",
    "jun",
    "june",
    "jul",
    "july",
    "aug",
    "august",
    "sep",
    "sept",
    "september",
    "oct",
    "october",
    "nov",
    "november",
    "dec",
    "december",
}

HARD_NONCONTENT_PHRASES = [
    "approved for release",
    "state dept review completed",
    "state department review completed",
    "central intelligence bulletin",
    "current intelligence bulletin",
    "central intelligence agency",
    "office of current intelligence",
    "daily brief",
    "security information",
    "security info",
    "secur information",
    "copy no",
    "document no",
    "no change in class",
    "class changed to",
    "next review date",
    "state review completed",
    "reviewer",
    "top secret",
    "secret sabre",
    "eider noforn",
    "noforn",
    "ts s",
    "on reverse page",
    "reverse page",
    "backup page",
    "page denied",
    "document exempt",
    "document denied",
    "in document exempt",
    "in document denied",
    "classification determination information",
    "this redacted version only",
    "tracking no",
    "oi tracking no",
    "of tracking no",
    "ol tracking no",
    "signed eastern time",
    "judge united states foreign intelligence surveillance court",
    "fisa acquired information subject to sequestration",
    "sensitive information",
    "sensitive method",
    "source description",
    "foreign government third party equity",
    "foreign government third party equivalent",
]

SHORT_NONCONTENT_PHRASES = [
    "contents",
    "summary",
    "comment on",
    "continued control",
    "map",
    "legend",
    "figure",
    "photo",
    "chart",
    "distribution",
    "unclassified",
    "confidential",
    "secret",
    "top secret",
    "general",
    "far east",
    "eastern europe",
    "western europe",
    "south asia",
    "southeast asia",
    "near east",
    "near east africa",
    "asia africa",
    "latin america",
    "the west",
    "the president",
    "the vice president",
    "federal bureau of investigation",
    "national security agency",
    "national indications center",
    "atomic energy commission",
]

SHORT_SECTION_HEADING_PHRASES = {
    "eastern europe",
    "far east",
    "near east",
    "near east africa",
    "south asia",
    "southeast asia",
    "soviet union",
    "the arab israeli situation",
    "the formosa straits",
    "the west",
    "western europe",
}

MAP_LEGEND_PHRASES = [
    "statute miles",
    "nautical miles",
    "kilometers",
    "scale",
    "selected roads",
    "selected road",
    "selected railroad",
    "selected railroads",
    "international boundary",
    "national capital",
    "railroad road trail",
    "railroad road",
    "roads railroads",
    "road trail",
    "main route",
    "route number",
    "available airfield",
    "airfield site",
    "government forces",
    "anti government",
    "antigovernment",
]

MAP_LEGEND_HINTS = {
    "airfield",
    "airfields",
    "antigovernment",
    "boundary",
    "capital",
    "kilometer",
    "kilometers",
    "legend",
    "map",
    "mile",
    "miles",
    "nautical",
    "rail",
    "railroad",
    "railroads",
    "railway",
    "road",
    "roads",
    "route",
    "scale",
    "selected",
    "statute",
    "trail",
    "trails",
    "unclassified",
}

ADMIN_STUB_PATTERNS = [
    re.compile(r"^next\s+\d+\s+page\(s\)\s+in\s+document\s+(?:exempt|denied)$", re.I),
    re.compile(r"^page\s+denied(?:\s+next\s+\d+\s+page\(s\)\s+in\s+document\s+denied)?$", re.I),
    re.compile(r"^(?:top\s+secret|secret|confidential|unclassified)(?:\s+(?:top\s+secret|secret|confidential|unclassified))*$", re.I),
    re.compile(r"^approved\s+for\s+release\b.*(?:document\s+(?:exempt|denied)|top\s+secret)?$", re.I),
]

ADMIN_TOKEN_HINTS = {
    "approved",
    "release",
    "cia",
    "rdp",
    "state",
    "dept",
    "department",
    "review",
    "completed",
    "document",
    "exempt",
    "denied",
    "declassified",
    "class",
    "changed",
    "auth",
    "reviewer",
    "copy",
    "page",
    "top",
    "secret",
    "confidential",
    "unclassified",
    "bulletin",
    "intelligence",
    "tracking",
    "signed",
    "fisa",
    "surveillance",
    "court",
    "sensitive",
    "source",
    "description",
    "sequestration",
}

FISC_ADMIN_LINE_PATTERNS = [
    re.compile(r"^classified by:.*$", re.I),
    re.compile(r"^classification determination information$", re.I),
    re.compile(r"^derived from:.*$", re.I),
    re.compile(r"^declassify on:.*$", re.I),
    re.compile(r"^declassified by .*$", re.I),
    re.compile(r"^declasified by .*$", re.I),
    re.compile(r"^(?:this )?redacted version only$", re.I),
    re.compile(r"^(?:oi|of|ol)\s+tracking\s+no\.?.*$", re.I),
    re.compile(r"^docket number:.*$", re.I),
    re.compile(r"^signed .* eastern time$", re.I),
    re.compile(r"^judge,\s+united states foreign intelligence surveillance court$", re.I),
    re.compile(r"^i,\s*lee ?ann flynn hall,\s*clerk,\s*fisc,.*true and correct copy of the original\.?$", re.I),
    re.compile(r"^fisc,?\s*(?:i\s+hereby\s+)?certify.*true and correct copy of the orig(?:inal|in)\.?$", re.I),
    re.compile(r"^classify by:.*$", re.I),
    re.compile(r"^reason:\s*.*$", re.I),
    re.compile(r"^foreign intelligence surveillance court$", re.I),
    re.compile(r"^washington,\s*d\.?\s*c\.?$", re.I),
    re.compile(r"^united states$", re.I),
    re.compile(r"^date\s+time$", re.I),
    re.compile(r"^this authorization regarding .* expires at eastern time.*$", re.I),
    re.compile(r"^this authorization regarding .* expires .*day of .*?$", re.I),
    re.compile(r"^all information is considered unclassified except where otherwise shown$", re.I),
    re.compile(r"^(?:xx\s+secret|secretm|secref|senket|secret)$", re.I),
    re.compile(r"^[\-]{2,}\s*the remainder of this page is intentionally left blank\.?\s*[\-]{2,}$", re.I),
    re.compile(r"^\(?fbi nsicg, dated .*\)?$", re.I),
    re.compile(r"^(?:b\d+[a-z]?-\d+[\s,]*){2,}$", re.I),
    re.compile(r"^[\-\u2013\u2014]?\d+[\-\u2013\u2014]?$"),
    re.compile(r"^(?:_{2,}\s*)+$"),
    re.compile(r"^x{3,}$", re.I),
]

FISC_PLACEHOLDER_LINE_PATTERNS = [
    re.compile(
        r"^(?:\d+\.\s*)?(?:\([A-Z0-9/$.-]+\)\s*)*(?:_{2,}\s*)*"
        r"(?:Sensitive Information|Sensitive Method|Source Description|"
        r"FISA-?Acquired Information Subject to Sequestration|"
        r"Foreign Government Third Party Equit(?:y|ies|ent))"
        r"(?:\s+(?:Sensitive Information|Sensitive Method|Source Description|"
        r"FISA-?Acquired Information Subject to Sequestration|"
        r"Foreign Government Third Party Equit(?:y|ies|ent)))*"
        r"\.?\s*$",
        re.I,
    ),
    re.compile(
        r"^(?:\d+\.\s*)?(?:\([A-Z0-9/$.-]+\)\s*)*(?:_{2,}\s*)*"
        r"(?:Source Description|Sensitive Information)\s*$",
        re.I,
    ),
    re.compile(r"^(?:\d+\.\s*)?(?:\([A-Z0-9/$.-]+\)\s*)*(?:X{3,}\s*)+$", re.I),
]

FISC_CONTROL_MARKER_HEADS = {
    "u",
    "c",
    "s",
    "ts",
    "x",
    "si",
    "nf",
    "tk",
    "hcs",
    "fgi",
    "sci",
}

FISC_CONTROL_MARKER_FOLLOWERS = {
    "u",
    "c",
    "s",
    "ts",
    "x",
    "a",
    "b",
    "d",
    "e",
    "g",
    "si",
    "nf",
    "noforn",
    "orcon",
    "rel",
    "tk",
    "hcs",
    "fgi",
    "sci",
}

FISC_LEADING_ENUM_RE = re.compile(r"^(?P<prefix>(?:(?:\d+|[a-z])\.\s*)*)(?P<body>.*)$", re.I)
FISC_LEADING_MARKER_RE = re.compile(r"^\((?P<inner>[A-Z][A-Z0-9/.-]{0,20})\)\s*")
FISC_TRAILING_PAGE_REF_RE = re.compile(r"\s*[\[(](?:Page|pg)\.?\s+\d+[)\]]\s*$", re.I)
FISC_INLINE_PAGE_NUMBER_RE = re.compile(r"(?:^|\s)[\-\u2013\u2014](\d{1,4})[\-\u2013\u2014](?:\s|$)")

ORIGINAL_DIFFERENCE_FILES = [
    "redacted.aligned.txt",
    "unredacted.aligned.txt",
    "unredacted_bracketed.aligned.txt",
    "redaction_chunks.aligned.txt",
    "page_alignment.json",
    "difference.summary.json",
]


@dataclass
class PageContext:
    pair_key: str
    aligned_page_index: int
    page_tag: str
    row: dict[str, Any]
    red_text: str
    unred_text: str
    red_tokens: list[str]
    unred_tokens: list[str]
    unred_matches: list[re.Match[str]]
    mapping: list[int]
    full_page_failed: bool
    full_page_details: dict[str, Any]


@dataclass
class Candidate:
    pair_key: str
    original_redaction_id: int
    segment_id: int
    aligned_page: int
    page_tag: str
    red_source_page: int | None
    unred_source_page: int | None
    row_kind: str
    alignment_score: float | None
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    text: str
    token_text: str
    token_count: int
    red_gap_text: str
    red_gap_token_count: int
    recut_from_original: bool
    recut_action: str
    recut_reason: str | None
    removed_by_recut: bool


def normalize_ws(text: str) -> str:
    return " ".join(str(text).replace("\r\n", "\n").replace("\r", "\n").split())


def is_probable_fisc_pair(pair_key: str) -> bool:
    return str(pair_key).startswith("Pair_")


def fisc_placeholder_hits(text: str) -> int:
    norm = normalize_ws(text)
    if not norm:
        return 0
    hits = 0
    for pattern in FISC_PLACEHOLDER_LINE_PATTERNS:
        if pattern.match(norm):
            hits += 1
    return hits


def is_fisc_admin_line(text: str) -> bool:
    norm = normalize_ws(text)
    if not norm:
        return False
    return any(pattern.match(norm) for pattern in FISC_ADMIN_LINE_PATTERNS)


def is_fisc_placeholder_line(text: str) -> bool:
    norm = normalize_ws(text)
    if not norm:
        return False
    return any(pattern.match(norm) for pattern in FISC_PLACEHOLDER_LINE_PATTERNS)


def is_fisc_control_marker(inner: str) -> bool:
    norm = normalize_ws(inner).upper()
    if not norm or len(norm) > 21:
        return False
    parts = [part for part in re.split(r"/{1,2}|[.-]+", norm) if part]
    if not parts:
        return False
    if parts[0].lower() not in FISC_CONTROL_MARKER_HEADS:
        return False
    return all(part.isalnum() and part.lower() in FISC_CONTROL_MARKER_FOLLOWERS for part in parts)


def strip_fisc_inline_noise(text: str) -> str:
    norm = normalize_ws(text)
    if not norm:
        return ""
    norm = FISC_TRAILING_PAGE_REF_RE.sub("", norm).strip()
    norm = FISC_INLINE_PAGE_NUMBER_RE.sub(" ", norm)
    norm = normalize_ws(norm)
    while True:
        match = FISC_LEADING_ENUM_RE.match(norm)
        prefix = match.group("prefix") if match else ""
        body = match.group("body") if match else norm
        marker_match = FISC_LEADING_MARKER_RE.match(body)
        if not marker_match:
            break
        if not is_fisc_control_marker(marker_match.group("inner")):
            break
        body = body[marker_match.end() :].lstrip()
        norm = normalize_ws(f"{prefix}{body}")
    if re.fullmatch(r"[\-\u2013\u2014]?\d+[\-\u2013\u2014]?", norm):
        return ""
    return normalize_ws(norm)


def looks_like_heading_line(text: str) -> bool:
    norm = normalize_ws(text)
    if not norm:
        return False
    if re.match(r"^(?:[IVXLCM]+\.)\s+", norm, re.I):
        return True
    if re.match(r"^(?:\d+\.|\([A-Z0-9/$.-]+\)|[A-Z]\.)\s+", norm):
        return True
    tokens = WORD_RE.findall(norm)
    alpha_count = sum(ch.isalpha() for ch in norm)
    if 0 < len(tokens) <= 12 and alpha_count >= 6 and norm.upper() == norm:
        return True
    if norm.endswith(":") and len(tokens) <= 12:
        return True
    return False


def line_join_without_space(prev: str, curr: str) -> bool:
    prev = prev.rstrip()
    curr = curr.lstrip()
    if not prev or not curr:
        return False
    return prev.endswith("-") and bool(re.match(r"^[a-z]", curr))


def line_join_with_space(prev: str, curr: str) -> bool:
    prev = prev.rstrip()
    curr = curr.lstrip()
    if not prev or not curr:
        return False
    if looks_like_heading_line(prev) or looks_like_heading_line(curr):
        return False
    if re.match(r"^(?:[-*•]|\d+\.|\([A-Z0-9/$.-]+\))\s*", curr):
        return False
    if re.search(r"[.!?:;)\]\"']\s*$", prev):
        return False
    return bool(re.match(r"^(?:[a-z(\[]|\d{1,2}\b)", curr))


def clean_fisc_page_text(text: str) -> str:
    src = str(text).replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [line.rstrip() for line in src.split("\n")]
    filtered: list[str] = []
    for line in raw_lines:
        norm = normalize_ws(line)
        if not norm:
            filtered.append("")
            continue
        if is_fisc_admin_line(norm):
            continue
        if is_fisc_placeholder_line(norm):
            continue
        cleaned = strip_fisc_inline_noise(line.strip())
        if not cleaned:
            continue
        filtered.append(cleaned)

    while filtered and not normalize_ws(filtered[0]):
        filtered.pop(0)
    while filtered and not normalize_ws(filtered[-1]):
        filtered.pop()

    merged: list[str] = []
    for line in filtered:
        norm = normalize_ws(line)
        if not norm:
            if merged and merged[-1] != "":
                merged.append("")
            continue
        if not merged:
            merged.append(norm)
            continue
        prev = merged[-1]
        if prev == "":
            merged.append(norm)
            continue
        if line_join_without_space(prev, norm):
            merged[-1] = prev[:-1] + norm
        elif line_join_with_space(prev, norm):
            merged[-1] = prev + " " + norm
        else:
            merged.append(norm)

    out = "\n".join(merged)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def normalize_token(token: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", str(token)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def normalize_for_ocr_compare(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def phrase_tokens(phrase: str) -> tuple[str, ...]:
    return tuple(t for t in (normalize_token(part) for part in phrase.split()) if t)


HARD_PHRASE_TOKENS = [phrase_tokens(p) for p in HARD_NONCONTENT_PHRASES]
SHORT_PHRASE_TOKENS = [phrase_tokens(p) for p in SHORT_NONCONTENT_PHRASES]
MAP_LEGEND_PHRASE_TOKENS = [phrase_tokens(p) for p in MAP_LEGEND_PHRASES]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def iter_pair_dirs(in_root: Path, pair_key: str | None, max_pairs: int | None) -> list[Path]:
    if pair_key is not None:
        selected = [in_root / pair_key]
    else:
        selected = sorted(p for p in in_root.iterdir() if p.is_dir())
    pairs = [
        p
        for p in selected
        if (p / "difference" / "page_alignment.json").exists()
        and (p / "pages" / "redacted").exists()
        and (p / "pages" / "unredacted").exists()
    ]
    if max_pairs is not None:
        pairs = pairs[: max(0, int(max_pairs))]
    return pairs


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        out = dst / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)


def copy_pair_inputs(pair_dir: Path, out_pair_dir: Path, *, copy_source_pdfs: bool) -> None:
    for subdir in ("document", "pages"):
        copy_tree_contents(pair_dir / subdir, out_pair_dir / subdir)
    if copy_source_pdfs:
        copy_tree_contents(pair_dir / "source_pdfs", out_pair_dir / "source_pdfs")

    out_diff = out_pair_dir / "difference"
    out_diff.mkdir(parents=True, exist_ok=True)
    for name in ORIGINAL_DIFFERENCE_FILES:
        src = pair_dir / "difference" / name
        if src.exists():
            shutil.copy2(src, out_diff / f"original.{name}")
            if name in {"page_alignment.json", "difference.summary.json", "redacted.aligned.txt", "unredacted.aligned.txt"}:
                shutil.copy2(src, out_diff / name)


def page_body_path(pair_dir: Path, side: str, page_index_0based: int | None) -> Path | None:
    if page_index_0based is None:
        return None
    return pair_dir / "pages" / side / f"page_{page_index_0based + 1:04d}.txt"


def load_side_page_texts(pair_dir: Path, side: str, *, apply_fisc_cleanup: bool) -> dict[int, str]:
    page_dir = pair_dir / "pages" / side
    out: dict[int, str] = {}
    if not page_dir.exists():
        return out
    for path in sorted(page_dir.glob("page_*.txt")):
        match = re.search(r"page_(\d+)\.txt$", path.name)
        if not match:
            continue
        idx = int(match.group(1))
        text = read_text(path).rstrip("\n")
        if apply_fisc_cleanup:
            text = clean_fisc_page_text(text)
        out[idx] = text
    return out


def render_pages_to_document_text(page_texts: dict[int, str]) -> str:
    if not page_texts:
        return ""
    width = max(2, len(str(max(page_texts))))
    lines: list[str] = []
    for idx in sorted(page_texts):
        lines.append(f"[PAGE {idx:0{width}d}]")
        text = page_texts[idx].rstrip()
        if text:
            lines.append(text)
    return "\n".join(lines).rstrip() + "\n"


def write_cleaned_page_outputs(out_pair_dir: Path, side: str, page_texts: dict[int, str]) -> None:
    page_dir = out_pair_dir / "pages" / side
    page_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx in sorted(page_texts):
        text = page_texts[idx]
        out_path = page_dir / f"page_{idx:04d}.txt"
        out_path.write_text(text.rstrip() + "\n" if text else "", encoding="utf-8")
        rows.append(
            {
                "page_index_1based": idx,
                "source_page_no": idx,
                "source_label": f"{idx:02d}",
                "text_file": str(out_path),
                "char_len": len(text),
            }
        )
    (out_pair_dir / "pages" / f"{side}.pages.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def page_tokens_and_matches(text: str) -> tuple[list[str], list[re.Match[str]]]:
    clean = clean_ocr_text(text, strip_noise_lines=False).text
    matches = list(WORD_RE.finditer(clean))
    return [m.group(0).lower() for m in matches], matches


def build_runs(mask: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if mask[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1] == 1:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def nearest_left(mapping: list[int], idx: int) -> int | None:
    j = idx
    while j >= 0:
        if mapping[j] >= 0:
            return int(mapping[j])
        j -= 1
    return None


def nearest_right(mapping: list[int], idx: int) -> int | None:
    j = idx
    while j < len(mapping):
        if mapping[j] >= 0:
            return int(mapping[j])
        j += 1
    return None


def red_gap_for_span(ctx: PageContext, start: int, end: int) -> tuple[str, int]:
    left = nearest_left(ctx.mapping, start - 1)
    right = nearest_right(ctx.mapping, end + 1)
    if left is None or right is None or right <= left + 1:
        return "", 0
    tokens = ctx.red_tokens[left + 1 : right]
    return " ".join(tokens), len(tokens)


def token_text(tokens: list[str], start: int, end: int) -> str:
    if end < start:
        return ""
    return " ".join(tokens[start : end + 1])


def make_candidate(
    ctx: PageContext,
    *,
    original_redaction_id: int,
    segment_id: int,
    token_start: int,
    token_end: int,
    recut_from_original: bool,
    recut_action: str,
    recut_reason: str | None,
    removed_by_recut: bool,
) -> Candidate:
    start_char = ctx.unred_matches[token_start].start()
    end_char = ctx.unred_matches[token_end].end()
    red_gap, red_gap_count = red_gap_for_span(ctx, token_start, token_end)
    score = ctx.row.get("score")
    return Candidate(
        pair_key=ctx.pair_key,
        original_redaction_id=original_redaction_id,
        segment_id=segment_id,
        aligned_page=ctx.aligned_page_index,
        page_tag=ctx.page_tag,
        red_source_page=ctx.row.get("red_source_page_no"),
        unred_source_page=ctx.row.get("unred_source_page_no"),
        row_kind=str(ctx.row.get("kind")),
        alignment_score=float(score) if score is not None else None,
        token_start=token_start,
        token_end=token_end,
        char_start=start_char,
        char_end=end_char,
        text=ctx.unred_text[start_char:end_char],
        token_text=token_text(ctx.unred_tokens, token_start, token_end),
        token_count=token_end - token_start + 1,
        red_gap_text=red_gap,
        red_gap_token_count=red_gap_count,
        recut_from_original=recut_from_original,
        recut_action=recut_action,
        recut_reason=recut_reason,
        removed_by_recut=removed_by_recut,
    )


def matches_phrase(tokens: list[str], pos: int, phrase: tuple[str, ...]) -> bool:
    if not phrase or pos + len(phrase) > len(tokens):
        return False
    return tuple(tokens[pos : pos + len(phrase)]) == phrase


def near_boundary(start: int, end: int, cand_start: int, cand_end: int, page_token_count: int) -> bool:
    return (
        start <= cand_start + 4
        or cand_end - end <= 4
        or start <= 12
        or page_token_count - end <= 12
    )


def add_span(spans: list[tuple[int, int, str]], start: int, end: int, reason: str) -> None:
    if end >= start:
        spans.append((start, end, reason))


def detect_recut_spans(tokens: list[str], cand_start: int, cand_end: int) -> list[tuple[int, int, str]]:
    norm_tokens = [normalize_token(t) for t in tokens]
    page_count = len(norm_tokens)
    cand_len = cand_end - cand_start + 1
    spans: list[tuple[int, int, str]] = []

    for i in range(cand_start, cand_end + 1):
        if i + 3 <= cand_end and norm_tokens[i].isdigit() and norm_tokens[i + 1] in MONTH_TOKENS:
            if norm_tokens[i + 2].isdigit():
                for phrase in HARD_PHRASE_TOKENS:
                    if phrase and matches_phrase(norm_tokens, i + 3, phrase):
                        j = i + 2 + len(phrase)
                        if j + 2 <= cand_end and norm_tokens[j + 1] == "page" and norm_tokens[j + 2].isdigit():
                            j += 2
                        add_span(spans, i, min(j, cand_end), "noncontent_footer_date_phrase")

        for phrase in HARD_PHRASE_TOKENS:
            if not matches_phrase(norm_tokens, i, phrase):
                continue
            end = i + len(phrase) - 1
            if phrase in {phrase_tokens("central intelligence bulletin"), phrase_tokens("current intelligence bulletin")}:
                j = end
                if j + 2 <= cand_end and norm_tokens[j + 1] == "page" and norm_tokens[j + 2].isdigit():
                    j += 2
                add_span(spans, i, j, "noncontent_document_furniture")
            elif near_boundary(i, end, cand_start, cand_end, page_count) or cand_len <= 14:
                j = end
                if phrase == phrase_tokens("copy no"):
                    j = min(cand_end, i + 4)
                add_span(spans, i, j, "noncontent_document_furniture")

        if norm_tokens[i] == "page" and i + 1 <= cand_end and norm_tokens[i + 1].isdigit():
            if cand_len <= 10 or near_boundary(i, i + 1, cand_start, cand_end, page_count):
                add_span(spans, i, i + 1, "noncontent_page_reference")
        if (
            norm_tokens[i] == "backup"
            and i + 2 <= cand_end
            and norm_tokens[i + 1] == "page"
            and norm_tokens[i + 2].isdigit()
        ):
            if cand_len <= 12 or near_boundary(i, i + 2, cand_start, cand_end, page_count):
                add_span(spans, i, i + 2, "noncontent_page_reference")
        if norm_tokens[i] in {"map", "legend"}:
            if cand_len <= 8 or near_boundary(i, i, cand_start, cand_end, page_count):
                add_span(spans, i, i, "noncontent_map_legend_reference")

    return merge_spans(spans, cand_start, cand_end)


def merge_spans(
    spans: list[tuple[int, int, str]], cand_start: int, cand_end: int
) -> list[tuple[int, int, str]]:
    clipped = [
        (max(cand_start, s), min(cand_end, e), reason)
        for s, e, reason in spans
        if min(cand_end, e) >= max(cand_start, s)
    ]
    if not clipped:
        return []
    clipped.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[int, int, set[str]]] = []
    for s, e, reason in clipped:
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e, {reason}))
        else:
            old_s, old_e, reasons = merged[-1]
            reasons.add(reason)
            merged[-1] = (old_s, max(old_e, e), reasons)
    return [(s, e, ",".join(sorted(reasons))) for s, e, reasons in merged]


def split_candidate_by_recut(
    ctx: PageContext, original_id: int, start: int, end: int
) -> list[Candidate]:
    recut_spans = detect_recut_spans(ctx.unred_tokens, start, end)
    if not recut_spans:
        return [
            make_candidate(
                ctx,
                original_redaction_id=original_id,
                segment_id=1,
                token_start=start,
                token_end=end,
                recut_from_original=False,
                recut_action="none",
                recut_reason=None,
                removed_by_recut=False,
            )
        ]

    out: list[Candidate] = []
    segment_id = 1
    cursor = start
    for recut_start, recut_end, reason in recut_spans:
        if cursor <= recut_start - 1:
            out.append(
                make_candidate(
                    ctx,
                    original_redaction_id=original_id,
                    segment_id=segment_id,
                    token_start=cursor,
                    token_end=recut_start - 1,
                    recut_from_original=True,
                    recut_action="kept_after_recut",
                    recut_reason=reason,
                    removed_by_recut=False,
                )
            )
            segment_id += 1
        out.append(
            make_candidate(
                ctx,
                original_redaction_id=original_id,
                segment_id=segment_id,
                token_start=recut_start,
                token_end=recut_end,
                recut_from_original=True,
                recut_action="removed_noncontent_segment",
                recut_reason=reason,
                removed_by_recut=True,
            )
        )
        segment_id += 1
        cursor = recut_end + 1
    if cursor <= end:
        out.append(
            make_candidate(
                ctx,
                original_redaction_id=original_id,
                segment_id=segment_id,
                token_start=cursor,
                token_end=end,
                recut_from_original=True,
                recut_action="kept_after_recut",
                recut_reason="noncontent_segment_removed",
                removed_by_recut=False,
            )
        )
    return out


def is_admin_stub_page(text: str, token_count: int) -> bool:
    compact = normalize_ws(text)
    if not compact:
        return True
    for pattern in ADMIN_STUB_PATTERNS:
        if pattern.search(compact):
            return True
    norm_tokens = [normalize_token(t) for t in WORD_RE.findall(compact)]
    if token_count <= 12 and norm_tokens:
        admin_hits = sum(1 for token in norm_tokens if token in ADMIN_TOKEN_HINTS or token.isdigit())
        if admin_hits / len(norm_tokens) >= 0.75:
            return True
    return False


def page_full_redaction_details(
    row: dict[str, Any],
    *,
    red_text: str,
    red_token_count: int,
    unred_token_count: int,
    candidate_token_total: int,
    full_page_coverage: float,
    low_alignment_score: float,
    min_full_page_tokens: int,
) -> tuple[bool, dict[str, Any]]:
    score = row.get("score")
    score_float = float(score) if score is not None else None
    coverage = candidate_token_total / unred_token_count if unred_token_count else 0.0
    red_admin_stub = is_admin_stub_page(red_text, red_token_count)
    failed = (
        unred_token_count >= min_full_page_tokens
        and coverage >= full_page_coverage
        and (
            row.get("kind") == "unred_only"
            or red_admin_stub
            or red_token_count <= 10
            or (score_float is not None and score_float < low_alignment_score)
        )
    )
    return failed, {
        "coverage": round(coverage, 6),
        "candidate_token_total": candidate_token_total,
        "unred_token_count": unred_token_count,
        "red_token_count": red_token_count,
        "red_admin_stub": red_admin_stub,
        "row_kind": row.get("kind"),
        "alignment_score": score_float,
        "thresholds": {
            "full_page_coverage": full_page_coverage,
            "low_alignment_score": low_alignment_score,
            "min_full_page_tokens": min_full_page_tokens,
        },
    }


def contains_phrase(norm_tokens: list[str], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(phrase) > len(norm_tokens):
        return False
    for i in range(0, len(norm_tokens) - len(phrase) + 1):
        if tuple(norm_tokens[i : i + len(phrase)]) == phrase:
            return True
    return False


def text_matches_page_reference(norm_tokens: list[str]) -> bool:
    if len(norm_tokens) == 2 and norm_tokens[0] == "page" and norm_tokens[1].isdigit():
        return True
    if len(norm_tokens) == 2 and norm_tokens[0] == "page" and norm_tokens[1][:1].isdigit():
        return True
    if len(norm_tokens) == 2 and norm_tokens == ["backup", "page"]:
        return True
    if len(norm_tokens) == 3 and norm_tokens[:2] == ["backup", "page"] and norm_tokens[2].isdigit():
        return True
    if len(norm_tokens) <= 4 and "page" in norm_tokens and any(t.isdigit() for t in norm_tokens):
        return True
    return False


def text_is_short_numeric_or_garbled(raw_text: str, tokens: list[str], norm_tokens: list[str]) -> bool:
    if len(tokens) > 3:
        return False
    if norm_tokens and all(token.isdigit() for token in norm_tokens):
        return True
    if len(norm_tokens) <= 2 and any(token.isdigit() for token in norm_tokens):
        if any(token.isalpha() and len(token) == 1 for token in norm_tokens):
            return True
    compact_text = str(raw_text).strip()
    alpha_count = sum(1 for ch in compact_text if ch.isalpha())
    raw_count = sum(1 for ch in compact_text if not ch.isspace())
    if raw_count == 0:
        return False
    alpha_ratio = alpha_count / raw_count
    has_punct_noise = any(ch in "!,:;`'\"" for ch in compact_text)
    return raw_count <= 12 and has_punct_noise and alpha_ratio < 0.8


def is_year_token(token: str) -> bool:
    if not token.isdigit():
        return False
    if len(token) == 2:
        return True
    if len(token) == 4:
        year = int(token)
        return 1900 <= year <= 2099
    return False


def is_day_token(token: str) -> bool:
    return token.isdigit() and 1 <= int(token) <= 31


def text_matches_date_only(norm_tokens: list[str]) -> bool:
    if len(norm_tokens) == 2:
        return (
            (norm_tokens[0] in MONTH_TOKENS and is_year_token(norm_tokens[1]))
            or (is_day_token(norm_tokens[0]) and norm_tokens[1] in MONTH_TOKENS)
        )
    if len(norm_tokens) != 3:
        return False
    return (
        is_day_token(norm_tokens[0])
        and norm_tokens[1] in MONTH_TOKENS
        and is_year_token(norm_tokens[2])
    ) or (
        norm_tokens[0] in MONTH_TOKENS
        and is_day_token(norm_tokens[1])
        and is_year_token(norm_tokens[2])
    )


def text_matches_short_section_heading(norm_tokens: list[str], near_page_edge: bool) -> bool:
    joined = " ".join(norm_tokens)
    if joined in SHORT_SECTION_HEADING_PHRASES:
        return True
    if 2 <= len(norm_tokens) <= 3 and norm_tokens[0].isdigit():
        return any(not token.isdigit() for token in norm_tokens[1:])
    return False


def text_looks_like_doc_stamp(raw_text: str, norm_tokens: list[str]) -> bool:
    compact = normalize_ws(raw_text)
    if not compact:
        return False
    if len(norm_tokens) > 3:
        return False
    if (
        compact.upper() == compact
        and "-" in compact
        and any(ch.isdigit() for ch in compact)
        and sum(ch.isalpha() for ch in compact) >= 4
        and len(compact) >= 10
    ):
        return True
    return False


def text_looks_like_mask_placeholder(raw_text: str, norm_tokens: list[str]) -> bool:
    compact = normalize_ws(raw_text)
    if not compact:
        return False
    if len(norm_tokens) <= 2 and re.fullmatch(r"X{2,}", compact.upper()):
        return True
    if "redacted" in compact.lower() and len(norm_tokens) <= 3:
        return True
    return False


def text_looks_like_fisc_control_marker(raw_text: str, norm_tokens: list[str]) -> bool:
    compact = normalize_ws(raw_text).strip("()")
    if not compact:
        return False
    if len(norm_tokens) > 2:
        return False
    return is_fisc_control_marker(compact)


def map_legend_failure_details(norm_tokens: list[str], token_count: int) -> tuple[bool, dict[str, Any]]:
    matched_phrases: list[str] = []
    for phrase_text, phrase in zip(MAP_LEGEND_PHRASES, MAP_LEGEND_PHRASE_TOKENS):
        if contains_phrase(norm_tokens, phrase):
            matched_phrases.append(phrase_text)

    hint_count = sum(1 for token in norm_tokens if token in MAP_LEGEND_HINTS)
    has_measurement = any(
        token in {"kilometer", "kilometers", "mile", "miles", "nautical", "scale", "statute"}
        for token in norm_tokens
    )
    has_route_pair = (
        any(token in {"road", "roads", "route"} for token in norm_tokens)
        and any(token in {"rail", "railroad", "railroads", "railway", "trail", "trails"} for token in norm_tokens)
    )
    has_weird_map_line_symbol = any(len(token) >= 3 and set(token) == {"o"} for token in norm_tokens)

    short_map_phrase = token_count <= 35 and bool(matched_phrases)
    dense_map_hints = token_count <= 35 and hint_count >= 3 and (has_measurement or has_route_pair)
    compact_map_pair = token_count <= 8 and has_route_pair
    map_line_symbol = token_count <= 10 and has_weird_map_line_symbol and has_route_pair
    failed = short_map_phrase or dense_map_hints or compact_map_pair or map_line_symbol
    return failed, {
        "matched_map_phrases": matched_phrases,
        "map_hint_count": hint_count,
        "has_measurement": has_measurement,
        "has_route_pair": has_route_pair,
        "has_weird_map_line_symbol": has_weird_map_line_symbol,
    }


def noncontent_failure(candidate: Candidate, page_token_count: int) -> tuple[bool, dict[str, Any]]:
    raw_tokens = candidate.token_text.split()
    norm_tokens = [normalize_token(t) for t in raw_tokens]
    norm_tokens = [t for t in norm_tokens if t]
    joined = " ".join(norm_tokens)
    near_page_edge = candidate.token_start <= 12 or page_token_count - candidate.token_end <= 12

    matched_phrases: list[str] = []
    for phrase_text, phrase in zip(HARD_NONCONTENT_PHRASES, HARD_PHRASE_TOKENS):
        if contains_phrase(norm_tokens, phrase):
            matched_phrases.append(phrase_text)
    for phrase_text, phrase in zip(SHORT_NONCONTENT_PHRASES, SHORT_PHRASE_TOKENS):
        if contains_phrase(norm_tokens, phrase):
            matched_phrases.append(phrase_text)

    exact_short = candidate.token_count <= 8 and (
        joined in {normalize_ws(p).lower() for p in SHORT_NONCONTENT_PHRASES + HARD_NONCONTENT_PHRASES}
        or text_matches_page_reference(norm_tokens)
    )
    short_with_furniture = candidate.token_count <= 8 and bool(matched_phrases)
    edge_with_hard_furniture = near_page_edge and any(p in HARD_NONCONTENT_PHRASES for p in matched_phrases)
    date_only = text_matches_date_only(norm_tokens)
    map_legend_like, map_details = map_legend_failure_details(norm_tokens, candidate.token_count)
    short_numeric_or_garbled = text_is_short_numeric_or_garbled(candidate.text, raw_tokens, norm_tokens)
    short_section_heading = text_matches_short_section_heading(norm_tokens, near_page_edge)
    control_marker_like = text_looks_like_fisc_control_marker(candidate.text, norm_tokens)
    doc_stamp_like = text_looks_like_doc_stamp(candidate.text, norm_tokens)
    mask_placeholder_like = text_looks_like_mask_placeholder(candidate.text, norm_tokens)

    admin_hits = sum(1 for token in norm_tokens if token in ADMIN_TOKEN_HINTS or token.isdigit())
    admin_ratio = admin_hits / len(norm_tokens) if norm_tokens else 0.0
    admin_short = candidate.token_count <= 12 and admin_ratio >= 0.75 and bool(matched_phrases)

    failed = (
        candidate.removed_by_recut
        or exact_short
        or short_with_furniture
        or edge_with_hard_furniture
        or admin_short
        or date_only
        or map_legend_like
        or short_numeric_or_garbled
        or short_section_heading
        or control_marker_like
        or doc_stamp_like
        or mask_placeholder_like
    )
    return failed, {
        "matched_phrases": matched_phrases,
        "near_page_edge": near_page_edge,
        "admin_token_ratio": round(admin_ratio, 6),
        "date_only": date_only,
        "map_legend_like": map_legend_like,
        "short_numeric_or_garbled": short_numeric_or_garbled,
        "short_section_heading": short_section_heading,
        "control_marker_like": control_marker_like,
        "doc_stamp_like": doc_stamp_like,
        "mask_placeholder_like": mask_placeholder_like,
        **map_details,
        "removed_by_recut": candidate.removed_by_recut,
        "recut_reason": candidate.recut_reason,
    }


def ocr_equivalence_failure(
    candidate: Candidate,
    *,
    threshold: float,
    short_threshold: float,
) -> tuple[bool, dict[str, Any]]:
    a = normalize_for_ocr_compare(candidate.token_text)
    b = normalize_for_ocr_compare(candidate.red_gap_text)
    if not a or not b:
        return False, {
            "similarity": None,
            "length_ratio": None,
            "red_gap_text": candidate.red_gap_text,
            "reason": "missing_red_gap",
        }
    length_ratio = min(len(a), len(b)) / max(len(a), len(b))
    similarity = difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()
    contains = (a in b or b in a) and length_ratio >= 0.72
    active_threshold = short_threshold if max(candidate.token_count, candidate.red_gap_token_count) <= 4 else threshold
    short_alternative_red_gap = (
        candidate.token_count <= 2
        and 0 < candidate.red_gap_token_count <= 3
    )
    failed = length_ratio >= 0.45 and (similarity >= active_threshold or contains)
    return failed, {
        "similarity": round(similarity, 6),
        "length_ratio": round(length_ratio, 6),
        "threshold": active_threshold,
        "red_gap_text": candidate.red_gap_text,
        "contains_after_normalization": contains,
        "short_alternative_red_gap": short_alternative_red_gap,
    }


def nonempty_redacted_alternate_conflict_failure(
    candidate: Candidate,
    ctx: PageContext,
    *,
    near_full_page_coverage: float,
) -> tuple[bool, dict[str, Any]]:
    coverage = float(ctx.full_page_details.get("coverage") or 0.0)
    reasons: list[str] = []

    if candidate.red_gap_token_count > 0:
        if candidate.token_count == 2:
            reasons.append("two_token_nonempty_alternate")
        if 3 <= candidate.token_count <= 5 and candidate.red_gap_token_count > 1:
            reasons.append("short_span_nontrivial_alternate")
        if candidate.red_gap_token_count >= max(4, candidate.token_count * 3):
            reasons.append("alternate_3x_longer_than_candidate")
        if candidate.token_count <= 10 and candidate.red_gap_token_count >= 20:
            reasons.append("short_candidate_long_alternate")
        if coverage >= near_full_page_coverage:
            reasons.append("near_full_page_boundary")

    return bool(reasons), {
        "conflict_reasons": reasons,
        "token_count": candidate.token_count,
        "red_gap_token_count": candidate.red_gap_token_count,
        "red_gap_text": candidate.red_gap_text,
        "page_candidate_coverage": round(coverage, 6),
        "thresholds": {
            "near_full_page_coverage": near_full_page_coverage,
            "short_span_max_tokens": 5,
            "short_span_nontrivial_alternate_min_red_gap_tokens": 2,
            "alternate_length_multiplier": 3,
            "short_candidate_max_tokens": 10,
            "long_alternate_min_red_gap_tokens": 20,
        },
    }


def evaluate_candidate(
    candidate: Candidate,
    ctx: PageContext,
    *,
    min_tokens: int,
    ocr_similarity_threshold: float,
    short_ocr_similarity_threshold: float,
    alternate_near_full_page_coverage: float,
) -> tuple[bool, list[str], dict[str, Any]]:
    tests: dict[str, Any] = {}

    full_failed = ctx.full_page_failed
    tests["full_page_or_admin"] = {
        "passed": not full_failed,
        "details": ctx.full_page_details,
    }

    noncontent_failed, noncontent_details = noncontent_failure(candidate, len(ctx.unred_tokens))
    tests["noncontent_keyword"] = {
        "passed": not noncontent_failed,
        "details": noncontent_details,
    }

    short_failed = candidate.token_count < min_tokens
    tests["short_span_min_tokens"] = {
        "passed": not short_failed,
        "details": {
            "token_count": candidate.token_count,
            "min_tokens": min_tokens,
        },
    }

    ocr_failed, ocr_details = ocr_equivalence_failure(
        candidate,
        threshold=ocr_similarity_threshold,
        short_threshold=short_ocr_similarity_threshold,
    )
    tests["ocr_equivalent_to_redacted_text"] = {
        "passed": not ocr_failed,
        "details": ocr_details,
    }

    alternate_failed, alternate_details = nonempty_redacted_alternate_conflict_failure(
        candidate,
        ctx,
        near_full_page_coverage=alternate_near_full_page_coverage,
    )
    tests["nonempty_redacted_alternate_conflict"] = {
        "passed": not alternate_failed,
        "details": alternate_details,
    }

    failed_tests = [name for name, result in tests.items() if not result["passed"]]
    return not failed_tests, failed_tests, tests


def candidate_to_record(
    candidate: Candidate,
    *,
    decision: str,
    clean_redaction_id: int | None,
    failed_tests: list[str],
    tests: dict[str, Any],
) -> dict[str, Any]:
    return {
        "decision": decision,
        "clean_redaction_id": clean_redaction_id,
        "pair_key": candidate.pair_key,
        "original_redaction_id": candidate.original_redaction_id,
        "segment_id": candidate.segment_id,
        "aligned_page": candidate.aligned_page,
        "page_tag": candidate.page_tag,
        "red_source_page": candidate.red_source_page,
        "unred_source_page": candidate.unred_source_page,
        "row_kind": candidate.row_kind,
        "alignment_score": candidate.alignment_score,
        "token_start_0based": candidate.token_start,
        "token_end_0based": candidate.token_end,
        "char_start_0based": candidate.char_start,
        "char_end_0based": candidate.char_end,
        "token_count": candidate.token_count,
        "text": candidate.text,
        "token_text": candidate.token_text,
        "red_gap_text": candidate.red_gap_text,
        "red_gap_token_count": candidate.red_gap_token_count,
        "recut_from_original": candidate.recut_from_original,
        "recut_action": candidate.recut_action,
        "recut_reason": candidate.recut_reason,
        "failed_tests": failed_tests,
        "tests": tests,
    }


def annotate_filtered_text(raw_text: str, kept: list[dict[str, Any]]) -> str:
    if not kept:
        return raw_text
    ordered = sorted(kept, key=lambda row: (row["char_start_0based"], row["char_end_0based"]))
    out: list[str] = []
    cursor = 0
    for row in ordered:
        start = int(row["char_start_0based"])
        end = int(row["char_end_0based"])
        if start < cursor:
            continue
        rid = int(row["clean_redaction_id"])
        out.append(raw_text[cursor:start])
        out.append(f"[[CLEAN_REDACTION_{rid}]]")
        out.append(raw_text[start:end])
        out.append(f"[[/CLEAN_REDACTION_{rid}]]")
        cursor = end
    out.append(raw_text[cursor:])
    return "".join(out)


def format_filtered_chunks(
    page_sections: list[dict[str, Any]],
) -> str:
    sections: list[str] = []
    for section in page_sections:
        row = section["row"]
        lines = [
            "=" * 72,
            section["page_tag"],
            f"kind={row.get('kind')} red_source_page={row.get('red_source_page_no')} unred_source_page={row.get('unred_source_page_no')}",
        ]
        if row.get("score") is not None:
            lines.append(f"alignment_score={float(row['score']):.6f}")
        lines.append(f"kept_candidate_count={len(section['kept'])}")
        lines.append(f"excluded_candidate_count={len(section['excluded'])}")
        if not section["kept"]:
            lines.append("No kept high-precision redaction chunks on this aligned page.")
        for record in section["kept"]:
            lines.extend(
                [
                    "",
                    f"Clean Chunk #{record['clean_redaction_id']} (original #{record['original_redaction_id']}, segment {record['segment_id']})",
                    f"Text: {normalize_ws(record['text'])}",
                    f"Token count: {record['token_count']}",
                    f"Recut: {record['recut_action']}",
                ]
            )
        if section["excluded"]:
            lines.append("")
            lines.append("Excluded candidates:")
        for record in section["excluded"]:
            lines.extend(
                [
                    f"- original #{record['original_redaction_id']} segment {record['segment_id']}: {', '.join(record['failed_tests'])}",
                    f"  Text: {normalize_ws(record['text'])}",
                ]
            )
        sections.append("\n".join(lines).strip())
    return "\n\n".join(sections).rstrip() + "\n"


def process_pair(
    pair_dir: Path,
    out_root: Path,
    *,
    copy_source_pdfs: bool,
    min_tokens: int,
    full_page_coverage: float,
    low_alignment_score: float,
    min_full_page_tokens: int,
    ocr_similarity_threshold: float,
    short_ocr_similarity_threshold: float,
    alternate_near_full_page_coverage: float,
) -> dict[str, Any]:
    pair_key = pair_dir.name
    out_pair_dir = out_root / pair_key
    copy_pair_inputs(pair_dir, out_pair_dir, copy_source_pdfs=copy_source_pdfs)
    apply_fisc_cleanup = is_probable_fisc_pair(pair_key)

    red_page_texts = load_side_page_texts(pair_dir, "redacted", apply_fisc_cleanup=apply_fisc_cleanup)
    unred_page_texts = load_side_page_texts(pair_dir, "unredacted", apply_fisc_cleanup=apply_fisc_cleanup)
    if apply_fisc_cleanup:
        write_cleaned_page_outputs(out_pair_dir, "redacted", red_page_texts)
        write_cleaned_page_outputs(out_pair_dir, "unredacted", unred_page_texts)
        doc_dir = out_pair_dir / "document"
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "redacted.transcription.txt").write_text(
            render_pages_to_document_text(red_page_texts),
            encoding="utf-8",
        )
        (doc_dir / "unredacted.transcription.txt").write_text(
            render_pages_to_document_text(unred_page_texts),
            encoding="utf-8",
        )

    alignment_rows = json.loads(read_text(pair_dir / "difference" / "page_alignment.json"))
    total_aligned = len(alignment_rows)
    width = max(2, len(str(total_aligned)))

    page_sections: list[dict[str, Any]] = []
    bracketed_pages: list[str] = []
    kept_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    failure_counter: Counter[str] = Counter()
    recut_removed_count = 0
    original_candidate_count = 0
    segment_candidate_count = 0
    clean_id = 1
    original_id = 1

    for row in alignment_rows:
        aligned_page = int(row["aligned_page_index_1based"])
        page_tag = f"[PAGE {aligned_page:0{width}d}]"
        red_idx = row.get("red_page_index_0based")
        unred_idx = row.get("unred_page_index_0based")
        red_text = red_page_texts.get((red_idx or 0) + 1, "") if red_idx is not None else ""
        unred_text = unred_page_texts.get((unred_idx or 0) + 1, "") if unred_idx is not None else ""

        red_tokens, _ = page_tokens_and_matches(red_text)
        unred_tokens, unred_matches = page_tokens_and_matches(unred_text)
        if red_idx is None and unred_idx is not None:
            mapping = [-1] * len(unred_tokens)
        else:
            mapping = align_monotonic(unred_tokens, red_tokens)
        mask = [1 if value < 0 else 0 for value in mapping]
        runs = build_runs(mask)
        candidate_token_total = sum(end - start + 1 for start, end in runs)
        full_failed, full_details = page_full_redaction_details(
            row,
            red_text=red_text,
            red_token_count=len(red_tokens),
            unred_token_count=len(unred_tokens),
            candidate_token_total=candidate_token_total,
            full_page_coverage=full_page_coverage,
            low_alignment_score=low_alignment_score,
            min_full_page_tokens=min_full_page_tokens,
        )
        ctx = PageContext(
            pair_key=pair_key,
            aligned_page_index=aligned_page,
            page_tag=page_tag,
            row=row,
            red_text=red_text,
            unred_text=unred_text,
            red_tokens=red_tokens,
            unred_tokens=unred_tokens,
            unred_matches=unred_matches,
            mapping=mapping,
            full_page_failed=full_failed,
            full_page_details=full_details,
        )

        page_kept: list[dict[str, Any]] = []
        page_excluded: list[dict[str, Any]] = []
        for start, end in runs:
            if not unred_matches:
                continue
            original_candidate_count += 1
            segments = split_candidate_by_recut(ctx, original_id, start, end)
            for candidate in segments:
                segment_candidate_count += 1
                passed, failed_tests, tests = evaluate_candidate(
                    candidate,
                    ctx,
                    min_tokens=min_tokens,
                    ocr_similarity_threshold=ocr_similarity_threshold,
                    short_ocr_similarity_threshold=short_ocr_similarity_threshold,
                    alternate_near_full_page_coverage=alternate_near_full_page_coverage,
                )
                if passed:
                    record = candidate_to_record(
                        candidate,
                        decision="kept",
                        clean_redaction_id=clean_id,
                        failed_tests=[],
                        tests=tests,
                    )
                    clean_id += 1
                    page_kept.append(record)
                    kept_records.append(record)
                else:
                    for failed in failed_tests:
                        failure_counter[failed] += 1
                    if candidate.removed_by_recut:
                        recut_removed_count += 1
                    record = candidate_to_record(
                        candidate,
                        decision="excluded",
                        clean_redaction_id=None,
                        failed_tests=failed_tests,
                        tests=tests,
                    )
                    page_excluded.append(record)
                    excluded_records.append(record)
            original_id += 1

        bracketed_pages.append(page_tag)
        bracketed_pages.append(annotate_filtered_text(unred_text, page_kept))
        page_sections.append({"page_tag": page_tag, "row": row, "kept": page_kept, "excluded": page_excluded})

    out_diff = out_pair_dir / "difference"
    out_diff.mkdir(parents=True, exist_ok=True)
    (out_diff / "unredacted_bracketed.filtered.aligned.txt").write_text(
        "\n".join(bracketed_pages).rstrip() + "\n",
        encoding="utf-8",
    )
    (out_diff / "redaction_chunks.filtered.txt").write_text(format_filtered_chunks(page_sections), encoding="utf-8")
    append_jsonl(out_diff / "kept_redactions.jsonl", kept_records)
    append_jsonl(out_diff / "excluded_redactions.jsonl", excluded_records)

    summary = {
        "pair_key": pair_key,
        "aligned_page_count": total_aligned,
        "original_candidate_count": original_candidate_count,
        "segment_candidate_count_after_recut": segment_candidate_count,
        "kept_candidate_count": len(kept_records),
        "excluded_candidate_count": len(excluded_records),
        "recut_removed_segment_count": recut_removed_count,
        "failure_counts": dict(sorted(failure_counter.items())),
        "config": {
            "min_tokens": min_tokens,
            "full_page_coverage": full_page_coverage,
            "low_alignment_score": low_alignment_score,
            "min_full_page_tokens": min_full_page_tokens,
            "ocr_similarity_threshold": ocr_similarity_threshold,
            "short_ocr_similarity_threshold": short_ocr_similarity_threshold,
            "alternate_near_full_page_coverage": alternate_near_full_page_coverage,
            "copy_source_pdfs": copy_source_pdfs,
        },
    }
    write_json(out_diff / "postprocess_summary.json", summary)
    return summary


def run_postprocess(
    *,
    in_root: Path,
    out_root: Path,
    pair_key: str | None,
    max_pairs: int | None,
    copy_source_pdfs: bool,
    min_tokens: int,
    full_page_coverage: float,
    low_alignment_score: float,
    min_full_page_tokens: int,
    ocr_similarity_threshold: float,
    short_ocr_similarity_threshold: float,
    alternate_near_full_page_coverage: float,
    progress_every: int,
) -> None:
    pair_dirs = iter_pair_dirs(in_root, pair_key=pair_key, max_pairs=max_pairs)
    if not pair_dirs:
        raise SystemExit(f"No processed transcription pairs found under {in_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    total_failures: Counter[str] = Counter()
    total_original = 0
    total_segments = 0
    total_kept = 0
    total_excluded = 0

    for idx, pair_dir in enumerate(pair_dirs, start=1):
        if progress_every > 0 and (idx == 1 or idx == len(pair_dirs) or idx % progress_every == 0):
            print(f"[{idx}/{len(pair_dirs)}] postprocessing {pair_dir.name}", flush=True)
        summary = process_pair(
            pair_dir,
            out_root,
            copy_source_pdfs=copy_source_pdfs,
            min_tokens=min_tokens,
            full_page_coverage=full_page_coverage,
            low_alignment_score=low_alignment_score,
            min_full_page_tokens=min_full_page_tokens,
            ocr_similarity_threshold=ocr_similarity_threshold,
            short_ocr_similarity_threshold=short_ocr_similarity_threshold,
            alternate_near_full_page_coverage=alternate_near_full_page_coverage,
        )
        manifest_rows.append(summary)
        total_original += int(summary["original_candidate_count"])
        total_segments += int(summary["segment_candidate_count_after_recut"])
        total_kept += int(summary["kept_candidate_count"])
        total_excluded += int(summary["excluded_candidate_count"])
        total_failures.update(summary["failure_counts"])

    append_jsonl(out_root / "postprocess_manifest.jsonl", manifest_rows)
    write_json(out_root / "postprocess_manifest.json", manifest_rows)
    write_json(
        out_root / "postprocess_summary.json",
        {
            "input_root": str(in_root),
            "output_root": str(out_root),
            "pair_count": len(pair_dirs),
            "original_candidate_count": total_original,
            "segment_candidate_count_after_recut": total_segments,
            "kept_candidate_count": total_kept,
            "excluded_candidate_count": total_excluded,
            "failure_counts": dict(sorted(total_failures.items())),
            "config": {
                "min_tokens": min_tokens,
                "full_page_coverage": full_page_coverage,
                "low_alignment_score": low_alignment_score,
                "min_full_page_tokens": min_full_page_tokens,
                "ocr_similarity_threshold": ocr_similarity_threshold,
                "short_ocr_similarity_threshold": short_ocr_similarity_threshold,
                "alternate_near_full_page_coverage": alternate_near_full_page_coverage,
                "copy_source_pdfs": copy_source_pdfs,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create high-precision, auditable redaction labels from existing transcription diffs."
    )
    parser.add_argument("--in_root", default="transcriptions", help="Existing transcription output root")
    parser.add_argument("--out_root", default="postprocessed", help="Postprocessed output root")
    parser.add_argument("--pair_key", default=None, help="Run exactly one pair key")
    parser.add_argument("--max_pairs", type=int, default=None, help="Optional cap for testing")
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=2,
        help="Reject candidate spans with fewer tokens than this. Default 2 rejects only one-token spans as a short-span proxy.",
    )
    parser.add_argument("--full_page_coverage", type=float, default=0.95)
    parser.add_argument("--low_alignment_score", type=float, default=0.25)
    parser.add_argument("--min_full_page_tokens", type=int, default=50)
    parser.add_argument("--ocr_similarity_threshold", type=float, default=0.50)
    parser.add_argument("--short_ocr_similarity_threshold", type=float, default=0.50)
    parser.add_argument(
        "--alternate_near_full_page_coverage",
        type=float,
        default=0.90,
        help="Reject non-empty redacted-side alternate conflicts on pages where candidate coverage is at least this high.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Print one progress line every N pairs. Use 1 for every pair or 0 for no progress lines.",
    )
    parser.add_argument(
        "--no_copy_source_pdfs",
        action="store_true",
        help="Do not copy source_pdfs into the postprocessed tree.",
    )
    args = parser.parse_args()

    run_postprocess(
        in_root=Path(args.in_root),
        out_root=Path(args.out_root),
        pair_key=args.pair_key,
        max_pairs=args.max_pairs,
        copy_source_pdfs=not bool(args.no_copy_source_pdfs),
        min_tokens=max(1, int(args.min_tokens)),
        full_page_coverage=float(args.full_page_coverage),
        low_alignment_score=float(args.low_alignment_score),
        min_full_page_tokens=max(1, int(args.min_full_page_tokens)),
        ocr_similarity_threshold=float(args.ocr_similarity_threshold),
        short_ocr_similarity_threshold=float(args.short_ocr_similarity_threshold),
        alternate_near_full_page_coverage=float(args.alternate_near_full_page_coverage),
        progress_every=max(0, int(args.progress_every)),
    )


if __name__ == "__main__":
    main()
