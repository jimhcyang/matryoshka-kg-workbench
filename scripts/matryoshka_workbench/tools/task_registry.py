#!/usr/bin/env python3
"""Shared corpus/task registry constants for the benchmark toolchain."""

from __future__ import annotations

from typing import Any


TASK_NAME_MAP = {
    "reconstruction": "reconstruction",
    "claims": "claims",
    "value_redaction": "redaction",
    "sentence_redaction": "sentence_redaction",
    "contradictions": "contradictions",
}

TASK_PREFIX_MAP = {
    "reconstruction": "rc",
    "claims": "cl",
    "value_redaction": "vr",
    "sentence_redaction": "sr",
    "contradictions": "cd",
}

TASK_TIER_ORDER = {
    "reconstruction": 10,
    "claims": 20,
    "value_redaction": 30,
    "sentence_redaction": 40,
    "contradictions": 50,
}

CORPUS_ALIASES = {
    "vietnam": "vietnam",
    "viet": "vietnam",
    "iran": "iran",
    "fisc": "fisc",
    "cib": "cib",
    "canberra": "canberra",
    "canb": "canberra",
}

CORPUS_SHORT_CODE_MAP = {
    "vietnam": "viet",
    "iran": "iran",
    "fisc": "fisc",
    "cib": "cib",
    "canberra": "canb",
}

DEFAULT_CORPORA = ("vietnam", "iran", "fisc", "cib", "canberra")


def short_corpus_code(corpus: str) -> str:
    return CORPUS_SHORT_CODE_MAP.get(corpus, corpus)


def corpus_code_from_config(value: Any) -> str:
    return short_corpus_code(str(value or "").casefold())

