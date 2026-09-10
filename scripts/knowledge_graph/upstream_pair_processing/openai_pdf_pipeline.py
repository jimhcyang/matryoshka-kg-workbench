from __future__ import annotations

import csv
import difflib
import json
import os
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRANSCRIPTION_PROMPT = """You are a careful archival transcription assistant. Your task is to transcribe a PDF page by page into plain text as faithfully as possible.

Please follow these rules:

1. Transcribe only meaningful document text.
2. Ignore scanner noise, specks, punch holes, skew lines, dark borders, and other meaningless visual artifacts.
3. Ignore repeated release stamps, archive routing numbers, side identifiers, repeated classification markings, and other page furniture unless they are clearly part of the document's intended readable content.
4. Preserve the document's structure in simple plain text:

   * Start each page with [PAGE XX], using two digits.
   * Then transcribe the page's actual content, including headings, dates, To/From/Subject lines, paragraphs, section headings, and numbered or bulleted lists in the order they appear.
5. Use best-effort transcription throughout the entire document.
6. Do not insert placeholders such as [illegible] or guess missing text.
7. If text has been redacted, blacked out, or clearly removed, skip it silently.
8. Do not invent or reconstruct text that is not readable.
9. If a page contains no meaningful document text, output only the page label for that page.
10. Return the transcription page by page in pure plain text only, with no commentary, no summaries, and no extra formatting beyond basic line breaks.

Transcribe the provided PDF now."""

PAGE_LABEL_RE = re.compile(r"(?m)^\[PAGE\s+(\d+)\]\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
INSTRUCTION_LINE_PATTERNS = [
    re.compile(r"^\s*(preserve|ignore)\s+line\s*breaks?\b", flags=re.IGNORECASE),
    re.compile(r"^\s*do\s+not\s+.*line\s*breaks?\b", flags=re.IGNORECASE),
    re.compile(r"^\s*press\s+the\b", flags=re.IGNORECASE),
    re.compile(r"^\s*line\s*breaks?\s+(are|in)\b", flags=re.IGNORECASE),
    re.compile(r"^\s*approved\s+for\s+release\b", flags=re.IGNORECASE),
]
NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*[#=\-]{3,}\s*$"),
]


@dataclass
class PDFPair:
    pair_key: str
    unredacted_numeric_id: str
    redacted_doc_id: str
    row: list[str]
    row_index_1based: int
    redacted_pdf: Path
    unredacted_pdf: Path


@dataclass
class PairCollectionStats:
    rows_total: int = 0
    rows_missing_required_fields: int = 0
    rows_invalid_unredacted_id_format: int = 0
    rows_missing_redacted_pdf: int = 0
    rows_missing_unredacted_pdf: int = 0
    rows_duplicate_unredacted_id: int = 0
    rows_duplicate_redacted_id: int = 0
    rows_valid: int = 0


@dataclass
class CleanTextResult:
    text: str
    dropped_lines: list[str]
    dropped_line_count: int


@dataclass
class PageBlock:
    source_page_no: int
    source_label: str
    text: str


def _progress(iterable: Any, *, total: int | None, desc: str) -> Any:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True, mininterval=0.2)


def _normalize_text(text: str) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _normalize_csv_row(row: list[str], width: int = 11) -> list[str]:
    out = list(row)
    if len(out) < width:
        out.extend([""] * (width - len(out)))
    return out


def _parse_unredacted_id(raw: str) -> int | None:
    s = str(raw).strip()
    if not s or not s.isdigit():
        return None
    if len(s) > 8:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _canonical_unredacted_pdf_path(unredacted_id_int: int, unredacted_dir: Path) -> Path:
    return unredacted_dir / f"cib_{unredacted_id_int:08d}.pdf"


def _clean_filename_component(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "x"


def collect_pdf_pairs_with_stats(
    csv_path: Path,
    redacted_dir: Path,
    unredacted_dir: Path,
    max_pairs: int | None = None,
) -> tuple[list[PDFPair], PairCollectionStats]:
    pairs: list[PDFPair] = []
    stats = PairCollectionStats()
    seen_unred_ids: set[str] = set()
    seen_red_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, raw_row in enumerate(reader, start=1):
            stats.rows_total += 1
            row = _normalize_csv_row(raw_row)
            unred_id_raw = str(row[1]).strip()
            red_doc_id = str(row[5]).strip()

            if not unred_id_raw or not red_doc_id:
                stats.rows_missing_required_fields += 1
                continue

            unred_id_int = _parse_unredacted_id(unred_id_raw)
            if unred_id_int is None:
                stats.rows_invalid_unredacted_id_format += 1
                continue

            unred_id_norm = str(unred_id_int)
            if unred_id_norm in seen_unred_ids:
                stats.rows_duplicate_unredacted_id += 1
                continue
            if red_doc_id in seen_red_ids:
                stats.rows_duplicate_redacted_id += 1
                continue

            red_pdf = redacted_dir / f"{red_doc_id}.pdf"
            if not red_pdf.exists():
                stats.rows_missing_redacted_pdf += 1
                continue

            unred_pdf = _canonical_unredacted_pdf_path(unred_id_int, unredacted_dir)
            if not unred_pdf.exists():
                stats.rows_missing_unredacted_pdf += 1
                continue

            seen_unred_ids.add(unred_id_norm)
            seen_red_ids.add(red_doc_id)

            pair_key = _clean_filename_component(f"{unred_id_norm}_{red_doc_id}")
            pairs.append(
                PDFPair(
                    pair_key=pair_key,
                    unredacted_numeric_id=unred_id_norm,
                    redacted_doc_id=red_doc_id,
                    row=row,
                    row_index_1based=i,
                    redacted_pdf=red_pdf,
                    unredacted_pdf=unred_pdf,
                )
            )
            stats.rows_valid += 1

            if max_pairs is not None and len(pairs) >= max_pairs:
                break

    return pairs, stats


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for rx in INSTRUCTION_LINE_PATTERNS:
        if rx.search(s):
            return True
    for rx in NOISE_LINE_PATTERNS:
        if rx.search(s):
            return True
    return False


def clean_ocr_text(raw_text: str, strip_noise_lines: bool = True) -> CleanTextResult:
    src = _normalize_text(raw_text)
    dropped: list[str] = []
    kept: list[str] = []
    for line in src.split("\n"):
        if strip_noise_lines and _is_noise_line(line):
            dropped.append(line)
            continue
        kept.append(line.rstrip())

    while kept and kept[0] == "":
        kept.pop(0)
    while kept and kept[-1] == "":
        kept.pop()

    return CleanTextResult(text="\n".join(kept), dropped_lines=dropped, dropped_line_count=len(dropped))


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def align_monotonic(a_tokens: list[str], b_tokens: list[str]) -> list[int]:
    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    mapping = [-1] * len(a_tokens)
    for block in sm.get_matching_blocks():
        if block.size <= 0:
            continue
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    return mapping


def build_chunks_from_mask(tokens: list[str], mask: list[int]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    i = 0
    cid = 1
    n = len(tokens)
    while i < n:
        if mask[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1] == 1:
            j += 1
        chunk_tokens = tokens[i : j + 1]
        chunks.append(
            {
                "chunk_id": cid,
                "start_token_idx_0based": i,
                "end_token_idx_0based": j,
                "token_count": len(chunk_tokens),
                "text": " ".join(chunk_tokens),
                "tokens": chunk_tokens,
            }
        )
        cid += 1
        i = j + 1
    return chunks


def annotate_text_with_redaction_mask(
    raw_text: str,
    token_mask: list[int],
    label_prefix: str = "REDACTION",
    start_index: int = 1,
) -> str:
    token_spans = list(WORD_RE.finditer(raw_text))
    if not token_spans or not token_mask:
        return raw_text

    n = min(len(token_spans), len(token_mask))
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if token_mask[i] != 1:
            i += 1
            continue
        j = i
        while j + 1 < n and token_mask[j + 1] == 1:
            j += 1
        runs.append((i, j))
        i = j + 1

    if not runs:
        return raw_text

    out: list[str] = []
    cursor = 0
    for rid, (s, e) in enumerate(runs, start=start_index):
        start_char = token_spans[s].start()
        end_char = token_spans[e].end()
        out.append(raw_text[cursor:start_char])
        out.append(f"[[{label_prefix}_{rid}]]")
        out.append(raw_text[start_char:end_char])
        out.append(f"[[/{label_prefix}_{rid}]]")
        cursor = end_char
    out.append(raw_text[cursor:])
    return "".join(out)


def load_valid_pairs(
    docs_root: Path,
    *,
    csv_name: str = "cibcia.csv",
    redacted_dir_name: str = "redacted_pdfs",
    unredacted_dir_name: str = "unredacted_pdfs",
    pair_key: str | None = None,
    max_pairs: int | None = None,
) -> tuple[list[PDFPair], PairCollectionStats]:
    csv_path = docs_root / csv_name
    redacted_dir = docs_root / redacted_dir_name
    unredacted_dir = docs_root / unredacted_dir_name
    pairs, stats = collect_pdf_pairs_with_stats(
        csv_path=csv_path,
        redacted_dir=redacted_dir,
        unredacted_dir=unredacted_dir,
        max_pairs=None,
    )
    if pair_key is not None:
        pairs = [pair for pair in pairs if pair.pair_key == pair_key]
    if max_pairs is not None:
        pairs = pairs[: max(0, int(max_pairs))]
    return pairs, stats


def ensure_openai_client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit("OpenAI Python client is not installed. Install with: pip install openai") from exc
    return OpenAI()


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                parts.append(str(getattr(content, "text", "")))
    return "\n".join(part for part in parts if part).strip()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def transcribe_pdf_with_openai(
    client: Any,
    *,
    pdf_path: Path,
    model: str,
    prompt: str,
    max_retries: int = 5,
    delete_uploaded_file: bool = True,
    sleep_seconds: float = 5.0,
) -> tuple[str, dict[str, Any]]:
    uploaded = None
    try:
        with pdf_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="user_data")

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = client.responses.create(
                    model=model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_file", "file_id": uploaded.id},
                            ],
                        }
                    ],
                )
                text = _extract_response_text(response)
                meta = {
                    "model": model,
                    "response_id": getattr(response, "id", None),
                    "uploaded_file_id": getattr(uploaded, "id", None),
                    "usage": _json_safe(getattr(response, "usage", None)),
                    "attempts": attempt,
                }
                return text, meta
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                time.sleep(sleep_seconds * attempt)
        assert last_error is not None
        raise last_error
    finally:
        if delete_uploaded_file and uploaded is not None:
            try:
                client.files.delete(uploaded.id)
            except Exception:
                pass


def pair_root(out_root: Path, pair: PDFPair) -> Path:
    return out_root / pair.pair_key


def document_dir(out_root: Path, pair: PDFPair) -> Path:
    return pair_root(out_root, pair) / "document"


def pages_dir(out_root: Path, pair: PDFPair) -> Path:
    return pair_root(out_root, pair) / "pages"


def difference_dir(out_root: Path, pair: PDFPair) -> Path:
    return pair_root(out_root, pair) / "difference"


def document_paths(out_root: Path, pair: PDFPair) -> dict[str, Path]:
    base = document_dir(out_root, pair)
    source_dir = pair_root(out_root, pair) / "source_pdfs"
    return {
        "dir": base,
        "source_dir": source_dir,
        "redacted_pdf_copy": source_dir / pair.redacted_pdf.name,
        "unredacted_pdf_copy": source_dir / pair.unredacted_pdf.name,
        "redacted_txt": base / "redacted.transcription.txt",
        "unredacted_txt": base / "unredacted.transcription.txt",
        "metadata_json": base / "pair_metadata.json",
        "prompt_txt": base / "prompt.txt",
    }


def pages_paths(out_root: Path, pair: PDFPair) -> dict[str, Path]:
    base = pages_dir(out_root, pair)
    return {
        "dir": base,
        "redacted_dir": base / "redacted",
        "unredacted_dir": base / "unredacted",
        "redacted_json": base / "redacted.pages.json",
        "unredacted_json": base / "unredacted.pages.json",
    }


def difference_paths(out_root: Path, pair: PDFPair) -> dict[str, Path]:
    base = difference_dir(out_root, pair)
    return {
        "dir": base,
        "alignment_json": base / "page_alignment.json",
        "summary_json": base / "difference.summary.json",
        "redacted_aligned_txt": base / "redacted.aligned.txt",
        "unredacted_aligned_txt": base / "unredacted.aligned.txt",
        "unredacted_bracketed_txt": base / "unredacted_bracketed.aligned.txt",
        "redaction_chunks_txt": base / "redaction_chunks.aligned.txt",
    }


def split_transcription_into_pages(text: str) -> list[PageBlock]:
    src = _normalize_text(text)
    matches = list(PAGE_LABEL_RE.finditer(src))
    if not matches:
        body = src.strip()
        return [PageBlock(source_page_no=1, source_label="01", text=body)] if body else []

    pages: list[PageBlock] = []
    for idx, match in enumerate(matches):
        label = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(src)
        body = src[start:end].strip("\n")
        try:
            page_no = int(label)
        except Exception:
            page_no = idx + 1
        pages.append(PageBlock(source_page_no=page_no, source_label=str(label), text=body.strip()))
    return pages


def save_split_pages(base_dir: Path, pages: list[PageBlock]) -> list[dict[str, Any]]:
    base_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, page in enumerate(pages, start=1):
        out_path = base_dir / f"page_{idx:04d}.txt"
        out_path.write_text(page.text.rstrip() + "\n" if page.text else "", encoding="utf-8")
        rows.append(
            {
                "page_index_1based": idx,
                "source_page_no": page.source_page_no,
                "source_label": page.source_label,
                "text_file": str(out_path),
                "char_len": len(page.text),
            }
        )
    return rows


def _page_similarity(red_text: str, unred_text: str) -> float:
    red_clean = clean_ocr_text(red_text, strip_noise_lines=False).text
    unred_clean = clean_ocr_text(unred_text, strip_noise_lines=False).text
    red_tokens = tokenize_words(red_clean)
    unred_tokens = tokenize_words(unred_clean)
    if not red_tokens and not unred_tokens:
        return 1.0
    if not red_tokens:
        return 0.0
    mapping = align_monotonic(red_tokens, unred_tokens)
    matched = sum(1 for value in mapping if value >= 0)
    recall = matched / len(red_tokens)
    precision = matched / len(unred_tokens) if unred_tokens else 0.0
    return 0.85 * recall + 0.15 * precision


def global_align_pages(
    red_pages: list[PageBlock],
    unred_pages: list[PageBlock],
    *,
    gap_penalty: float = -0.18,
    match_bias: float = 0.35,
) -> list[dict[str, Any]]:
    n = len(red_pages)
    m = len(unred_pages)
    sim = [[_page_similarity(red_pages[i].text, unred_pages[j].text) for j in range(m)] for i in range(n)]

    dp = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    backtrace: list[list[tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        backtrace[i][0] = ("red_only", i - 1, -1)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
        backtrace[0][j] = ("unred_only", -1, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = dp[i - 1][j - 1] + (sim[i - 1][j - 1] - match_bias)
            red_only_score = dp[i - 1][j] + gap_penalty
            unred_only_score = dp[i][j - 1] + gap_penalty
            best = match_score
            move: tuple[str, int, int] = ("match", i - 1, j - 1)
            if red_only_score > best:
                best = red_only_score
                move = ("red_only", i - 1, -1)
            if unred_only_score > best:
                best = unred_only_score
                move = ("unred_only", -1, j - 1)
            dp[i][j] = best
            backtrace[i][j] = move

    rows: list[dict[str, Any]] = []
    i = n
    j = m
    while i > 0 or j > 0:
        move = backtrace[i][j]
        if move is None:
            break
        kind, ri, ui = move
        row: dict[str, Any] = {"kind": kind}
        if kind == "match":
            row["red_page_index_0based"] = ri
            row["unred_page_index_0based"] = ui
            row["score"] = sim[ri][ui]
            i -= 1
            j -= 1
        elif kind == "red_only":
            row["red_page_index_0based"] = ri
            row["unred_page_index_0based"] = None
            row["score"] = None
            i -= 1
        else:
            row["red_page_index_0based"] = None
            row["unred_page_index_0based"] = ui
            row["score"] = None
            j -= 1
        rows.append(row)

    rows.reverse()
    for idx, row in enumerate(rows, start=1):
        row["aligned_page_index_1based"] = idx
        red_idx = row.get("red_page_index_0based")
        unred_idx = row.get("unred_page_index_0based")
        row["red_source_page_no"] = red_pages[red_idx].source_page_no if red_idx is not None else None
        row["unred_source_page_no"] = unred_pages[unred_idx].source_page_no if unred_idx is not None else None
    return rows


def _nearest_left(mapping: list[int], idx: int) -> tuple[int | None, int | None]:
    j = idx
    while j >= 0:
        if mapping[j] >= 0:
            return j, int(mapping[j])
        j -= 1
    return None, None


def _nearest_right(mapping: list[int], idx: int) -> tuple[int | None, int | None]:
    j = idx
    while j < len(mapping):
        if mapping[j] >= 0:
            return j, int(mapping[j])
        j += 1
    return None, None


def _slice_tokens(tokens: list[str], start: int, end: int) -> str:
    if not tokens:
        return ""
    s = max(0, start)
    e = min(len(tokens) - 1, end)
    if e < s:
        return ""
    return " ".join(tokens[s : e + 1])


def _aligned_page_label(index_1based: int, total: int) -> str:
    width = max(2, len(str(total)))
    return f"[PAGE {index_1based:0{width}d}]"


def build_difference_artifacts(
    *,
    pair: PDFPair,
    red_pages: list[PageBlock],
    unred_pages: list[PageBlock],
    alignment_rows: list[dict[str, Any]],
    out_root: Path,
) -> None:
    diff_paths = difference_paths(out_root, pair)
    diff_paths["dir"].mkdir(parents=True, exist_ok=True)

    total_aligned = len(alignment_rows)
    red_out: list[str] = []
    unred_out: list[str] = []
    bracketed_out: list[str] = []
    chunk_sections: list[str] = []
    chunk_total = 0
    next_redaction_id = 1

    for row in alignment_rows:
        aligned_idx = int(row["aligned_page_index_1based"])
        page_tag = _aligned_page_label(aligned_idx, total_aligned)
        red_idx = row.get("red_page_index_0based")
        unred_idx = row.get("unred_page_index_0based")
        red_text = red_pages[red_idx].text if red_idx is not None else ""
        unred_text = unred_pages[unred_idx].text if unred_idx is not None else ""

        red_out.append(page_tag)
        red_out.append(red_text)
        unred_out.append(page_tag)
        unred_out.append(unred_text)

        unred_tokens = tokenize_words(clean_ocr_text(unred_text, strip_noise_lines=False).text)
        if red_idx is None and unred_idx is not None:
            mask = [1] * len(unred_tokens)
            mapping: list[int] = [-1] * len(unred_tokens)
            red_tokens: list[str] = []
        else:
            red_tokens = tokenize_words(clean_ocr_text(red_text, strip_noise_lines=False).text)
            mapping = align_monotonic(unred_tokens, red_tokens)
            mask = [1 if value < 0 else 0 for value in mapping]

        chunks = build_chunks_from_mask(unred_tokens, mask)
        bracketed_text = annotate_text_with_redaction_mask(
            unred_text,
            token_mask=mask,
            label_prefix="PRED_REDACTION",
            start_index=next_redaction_id,
        )
        bracketed_out.append(page_tag)
        bracketed_out.append(bracketed_text)

        chunk_total += len(chunks)
        section: list[str] = [
            "=" * 72,
            page_tag,
            f"kind={row['kind']} red_source_page={row.get('red_source_page_no')} unred_source_page={row.get('unred_source_page_no')}",
        ]
        if row.get("score") is not None:
            section.append(f"alignment_score={float(row['score']):.6f}")
        if not chunks:
            section.append("No candidate difference chunks on this aligned page.")
        for redaction_id, chunk in enumerate(chunks, start=next_redaction_id):
            start = int(chunk["start_token_idx_0based"])
            end = int(chunk["end_token_idx_0based"])
            _, left_anchor = _nearest_left(mapping, start - 1)
            _, right_anchor = _nearest_right(mapping, end + 1)
            unred_context = " ".join(
                part
                for part in [
                    _slice_tokens(unred_tokens, start - 3, start - 1),
                    f"[UNRED_ONLY: {chunk['text']}]",
                    _slice_tokens(unred_tokens, end + 1, end + 3),
                ]
                if part
            ).strip()
            if left_anchor is None and right_anchor is None:
                red_context = "(no redacted anchors available)"
            elif left_anchor is None:
                red_context = f"[R@{right_anchor}] {_slice_tokens(red_tokens, right_anchor - 3, right_anchor + 3)}"
            elif right_anchor is None:
                red_context = f"[L@{left_anchor}] {_slice_tokens(red_tokens, left_anchor - 3, left_anchor + 3)}"
            else:
                gap = _slice_tokens(red_tokens, left_anchor + 1, right_anchor - 1)
                red_context = " ".join(
                    part
                    for part in [
                        _slice_tokens(red_tokens, left_anchor - 3, left_anchor - 1),
                        f"[L@{left_anchor}:{red_tokens[left_anchor]}]",
                        f"[RED_GAP:{gap or '<empty>'}]",
                        f"[R@{right_anchor}:{red_tokens[right_anchor]}]",
                        _slice_tokens(red_tokens, right_anchor + 1, right_anchor + 3),
                    ]
                    if part
                ).strip()
            section.extend(
                [
                    "",
                    f"Chunk #{redaction_id}",
                    f"Predicted text: {chunk['text']}",
                    f"Unredacted context: {unred_context}",
                    f"Redacted context: {red_context}",
                ]
            )
        next_redaction_id += len(chunks)
        chunk_sections.append("\n".join(section).strip())

    diff_paths["redacted_aligned_txt"].write_text("\n".join(red_out).rstrip() + "\n", encoding="utf-8")
    diff_paths["unredacted_aligned_txt"].write_text("\n".join(unred_out).rstrip() + "\n", encoding="utf-8")
    diff_paths["unredacted_bracketed_txt"].write_text("\n".join(bracketed_out).rstrip() + "\n", encoding="utf-8")
    diff_paths["redaction_chunks_txt"].write_text("\n\n".join(chunk_sections).rstrip() + "\n", encoding="utf-8")
    diff_paths["alignment_json"].write_text(json.dumps(alignment_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    diff_paths["summary_json"].write_text(
        json.dumps(
            {
                "pair_key": pair.pair_key,
                "aligned_page_count": len(alignment_rows),
                "redacted_page_count": len(red_pages),
                "unredacted_page_count": len(unred_pages),
                "candidate_chunk_count": chunk_total,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def transcribe_pairs(
    *,
    docs_root: Path,
    out_root: Path,
    model: str,
    prompt: str = TRANSCRIPTION_PROMPT,
    csv_name: str = "cibcia.csv",
    redacted_dir_name: str = "redacted_pdfs",
    unredacted_dir_name: str = "unredacted_pdfs",
    pair_key: str | None = None,
    max_pairs: int | None = None,
    skip_existing: bool = True,
    delete_uploaded_files: bool = True,
    max_retries: int = 5,
) -> None:
    pairs, stats = load_valid_pairs(
        docs_root,
        csv_name=csv_name,
        redacted_dir_name=redacted_dir_name,
        unredacted_dir_name=unredacted_dir_name,
        pair_key=pair_key,
        max_pairs=max_pairs,
    )
    print(f"[PAIRS] rows_total={stats.rows_total} valid_pairs={stats.rows_valid} selected={len(pairs)}")
    if not pairs:
        raise SystemExit("No valid PDF pairs selected.")

    client = ensure_openai_client()
    out_root.mkdir(parents=True, exist_ok=True)

    for pair in _progress(pairs, total=len(pairs), desc="OpenAI PDF transcription"):
        paths = document_paths(out_root, pair)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        paths["source_dir"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(pair.redacted_pdf, paths["redacted_pdf_copy"])
        shutil.copy2(pair.unredacted_pdf, paths["unredacted_pdf_copy"])
        paths["prompt_txt"].write_text(prompt + "\n", encoding="utf-8")
        metadata = {
            "pair_key": pair.pair_key,
            "row_index_1based": pair.row_index_1based,
            "unredacted_pdf": str(pair.unredacted_pdf),
            "redacted_pdf": str(pair.redacted_pdf),
            "redacted_pdf_copy": str(paths["redacted_pdf_copy"]),
            "unredacted_pdf_copy": str(paths["unredacted_pdf_copy"]),
            "csv_row": pair.row,
            "model": model,
        }

        plan = [
            ("redacted", pair.redacted_pdf, paths["redacted_txt"]),
            ("unredacted", pair.unredacted_pdf, paths["unredacted_txt"]),
        ]
        pair_failed = False
        for label, pdf_path, out_path in plan:
            if skip_existing and out_path.exists() and out_path.read_text(encoding="utf-8").strip():
                metadata[f"{label}_status"] = "skipped_existing"
                continue
            try:
                text, meta = transcribe_pdf_with_openai(
                    client,
                    pdf_path=pdf_path,
                    model=model,
                    prompt=prompt,
                    max_retries=max_retries,
                    delete_uploaded_file=delete_uploaded_files,
                )
            except Exception as exc:
                err = str(exc)
                metadata[f"{label}_status"] = "skipped_error"
                metadata[f"{label}_error"] = err
                metadata["pair_status"] = "skipped_error"
                metadata["pair_error_on"] = label
                metadata["pair_error_pdf"] = str(pdf_path)
                metadata["pair_error"] = err
                print(f"[WARN] skipping pair {pair.pair_key}: {label} PDF failed: {err}")
                pair_failed = True
                break
            out_path.write_text(_normalize_text(text).rstrip() + "\n", encoding="utf-8")
            metadata[f"{label}_status"] = "completed"
            metadata[f"{label}_response"] = meta

        if not pair_failed and "pair_status" not in metadata:
            metadata["pair_status"] = "completed"
        paths["metadata_json"].write_text(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8")


def postprocess_pairs(
    *,
    docs_root: Path,
    out_root: Path,
    csv_name: str = "cibcia.csv",
    redacted_dir_name: str = "redacted_pdfs",
    unredacted_dir_name: str = "unredacted_pdfs",
    pair_key: str | None = None,
    max_pairs: int | None = None,
    skip_existing: bool = True,
) -> None:
    pairs, stats = load_valid_pairs(
        docs_root,
        csv_name=csv_name,
        redacted_dir_name=redacted_dir_name,
        unredacted_dir_name=unredacted_dir_name,
        pair_key=pair_key,
        max_pairs=max_pairs,
    )
    print(f"[PAIRS] rows_total={stats.rows_total} valid_pairs={stats.rows_valid} selected={len(pairs)}")
    if not pairs:
        raise SystemExit("No valid PDF pairs selected.")

    for pair in _progress(pairs, total=len(pairs), desc="Postprocess transcriptions"):
        doc_paths = document_paths(out_root, pair)
        if not doc_paths["redacted_txt"].exists() or not doc_paths["unredacted_txt"].exists():
            print(f"[WARN] skipping {pair.pair_key}: missing document transcription files")
            continue

        diff_paths = difference_paths(out_root, pair)
        if skip_existing and diff_paths["summary_json"].exists():
            continue

        red_text = doc_paths["redacted_txt"].read_text(encoding="utf-8")
        unred_text = doc_paths["unredacted_txt"].read_text(encoding="utf-8")
        red_pages = split_transcription_into_pages(red_text)
        unred_pages = split_transcription_into_pages(unred_text)

        page_paths = pages_paths(out_root, pair)
        page_paths["redacted_dir"].mkdir(parents=True, exist_ok=True)
        page_paths["unredacted_dir"].mkdir(parents=True, exist_ok=True)
        red_rows = save_split_pages(page_paths["redacted_dir"], red_pages)
        unred_rows = save_split_pages(page_paths["unredacted_dir"], unred_pages)
        page_paths["redacted_json"].write_text(json.dumps(red_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        page_paths["unredacted_json"].write_text(json.dumps(unred_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        alignment_rows = global_align_pages(red_pages, unred_pages)
        build_difference_artifacts(
            pair=pair,
            red_pages=red_pages,
            unred_pages=unred_pages,
            alignment_rows=alignment_rows,
            out_root=out_root,
        )
