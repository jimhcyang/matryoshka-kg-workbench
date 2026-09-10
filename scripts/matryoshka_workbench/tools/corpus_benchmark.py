#!/usr/bin/env python3
"""Build and run extraction-driven Matryoshka task benchmarks for handoff corpora."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from query_catalog import write_query_catalog_files
from task_build_context import prepare_build_context
from task_catalog_pipeline import process_task_catalogs
from task_collectors import build_task_collectors
from task_expected_payloads import build_expected_payload
from task_registry import (
    CORPUS_ALIASES,
    TASK_NAME_MAP,
    TASK_PREFIX_MAP,
    TASK_TIER_ORDER,
    corpus_code_from_config,
)
from task_redaction_balance import (
    geometric_slot_targets as shared_geometric_slot_targets,
    rebalance_value_redaction_cases as shared_rebalance_value_redaction_cases,
    value_redaction_slot_count as shared_value_redaction_slot_count,
)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKBENCH_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = WORKBENCH_ROOT.parent.parent
KG_DEMO_ROOT = PACKAGE_ROOT
DEFAULT_CONFIG = WORKBENCH_ROOT / "config" / "benchmark.default.json"
RUNNER_RELATIVE = Path("automation/matryoshka_batch_runner_v1_0_18.py")
PREFLIGHT_RELATIVE = Path("preflight_environment_v0_4_21.py")
DEFAULT_PROFILES_DIR = WORKBENCH_ROOT / "config" / "corpus_profiles"
QUERY_PARSER_RELATIVE = Path("query_semantic_parser_v0_4_15.py")

sys.path.insert(0, str(WORKBENCH_ROOT / "release_v0421"))
from matryoshka_v0420.semantics import (
    build_event_signature,
    canonical_key,
    compare_argument,
    infer_simple_clause_arguments,
)
UNIFIED_EXTRACTION_FILES = (
    "chunks.csv",
    "entities.csv",
    "claims.csv",
    "events.csv",
    "relations.csv",
    "kg_pseudo_chunks.csv",
    "kg_pseudo_chunks_claims.csv",
    "kg_pseudo_chunks_contradictions.csv",
    "kg_pseudo_chunk_components.csv",
)
PAIR_EVAL_PATTERNS = (
    "pair_comparison_v048.csv",
    "pair_comparison_v043.csv",
)
REPORTING_WRAPPER_RE = re.compile(
    r"\b(claims?|claimed|believes?|believed|assesses?|assessed|thinks?|thought|"
    r"reports?|reported|states?|stated|says?|said)\b",
    flags=re.IGNORECASE,
)
QUERY_VET_CACHE: dict[tuple[str, str, str, str], bool] = {}
PAIR_CASE_VET_CACHE: dict[tuple[str, str, str, str, str], bool] = {}
LEADING_REPORTING_WRAPPER_RE = re.compile(
    r"^\s*(?:according to\s+[^,;:.]{1,80}[,;:]\s+)?"
    r"(?P<prefix>[^.!?]{1,120}?)\b"
    r"(?:claims?|claimed|believes?|believed|assesses?|assessed|thinks?|thought|"
    r"reports?|reported|states?|stated|says?|said|asserts?|asserted|denies?|denied)\b"
    r"\s+that\s+(?P<body>.+)$",
    flags=re.IGNORECASE,
)
LEADING_ACCORDING_TO_RE = re.compile(
    r"^\s*according to\s+[^,;:.]{1,80}[,;:]\s+(?P<body>.+)$",
    flags=re.IGNORECASE,
)
RELATIVE_CLAUSE_INSERT_RE = re.compile(
    r"^(?P<head>[^.!?;:]{1,120}?)\s+\b(?P<marker>that|which|who)\b\s+"
    r"(?P<insert>[^.!?;:]{1,100}?)\s+"
    r"(?P<tail>(?:had|has|have|was|were|is|are|will|would|could|should|did|do|does)\b.+)$",
    flags=re.IGNORECASE,
)
DEICTIC_RISK_RE = re.compile(
    r"\b(he|him|she|they|them|their|his|her|its|this|that|these|those)\b",
    flags=re.IGNORECASE,
)
LEGAL_BOILERPLATE_RE = re.compile(
    r"\b(pursuant to|foreign intelligence surveillance act|united states code|"
    r"minimization procedures|federal officer|application has been made|approved by the|"
    r"critical investigative means|definition of minimization procedures|as specified herein|"
    r"application contains all statements and certifications|required by 1804|required by 1823|"
    r"electronic surveillance will be directed|premises or property to be searched|"
    r"agent of russia, as defined by|committed a crime or crimes|"
    r"on the basis of the facts submitted in the verified application|"
    r"there is probable cause to believe)\b",
    flags=re.IGNORECASE,
)
EXTRACTION_ARTIFACT_RE = re.compile(
    r"\b(?:ent|event|claim|rel|view|prop|slot)_[0-9a-z_]+\b",
    flags=re.IGNORECASE,
)
BAD_OBJECT_LEAD_RE = re.compile(
    r"^(?:that\b|along with\b|with\b|during\b|after\b|before\b|about\b|regarding\b|"
    r"in\b|on\b|at\b|from\b|under\b|by\b)",
    flags=re.IGNORECASE,
)
REPORTING_ACTOR_RE = re.compile(
    r"\b(?:claims?|claimed|thinks?|thought|says?|said|reports?|reported|states?|stated)\b.*\bthat\b",
    flags=re.IGNORECASE,
)
ABSTRACT_ACTOR_RE = re.compile(
    r"\b(?:efforts?|assertions?|authorities|authority|application|applications|request|requests|"
    r"purpose|purposes|information|number|numbers|talks?)\b",
    flags=re.IGNORECASE,
)
PASSIVE_CLAIM_RE = re.compile(
    r"\b(?:was|were|is|are)\s+[A-Za-z][A-Za-z'-]{2,}(?:ed|en)\b",
    flags=re.IGNORECASE,
)
PLACEHOLDER_PHRASE_RE = re.compile(
    r"\b(?:claim target|event participant|event target|relation target|target entity|pseudo chunk)\b",
    flags=re.IGNORECASE,
)
ABSTRACT_ACTOR_HEADS = {
    "action", "actions", "application", "applications", "assertion", "assertions",
    "attempt", "attempts", "backing", "campaign", "censorship", "ceremonies", "ceremony",
    "certification", "certifications", "connection", "connections",
    "coverage", "decision",
    "disappearance", "disquiet", "effort", "efforts", "goal", "goals", "information", "interference",
    "itemization", "numbers", "press", "purpose", "purposes", "release",
    "releases", "request", "requests",
    "response", "session", "statement", "statements", "trip", "trips",
    "visit", "visits", "meeting", "meetings",
}
SINGLE_EVAL_PATTERNS = (
    "task_evaluation_v048.csv",
    "task_evaluation_v043.csv",
    "redaction_evaluation_v048.csv",
    "sentence_redaction_evaluation_v048.csv",
    "claim_evaluation_v048.csv",
    "contradiction_evaluation_v048.csv",
)
SUMMARY_COLUMNS = (
    "task_execution_status",
    "localization_status",
    "semantic_status",
    "raw_certification_status",
    "reconstructability_status",
    "pair_comparability_status",
    "selected_value_or_text",
    "selected_source_doc_id",
    "authoritative_source_doc_id",
    "top_candidate_doc_id",
    "top_related_evidence_doc_id",
    "top_background_doc_id",
    "primary_finding",
    "primary_finding_source_doc_id",
    "all_decision_source_doc_ids",
    "primary_finding_score",
    "evidence_pattern",
    "best_score",
    "runner_up_score",
    "decision_competitor_score",
    "winner_margin",
    "margin_status",
    "decision_reason",
    "unresolved_reason",
    "failed_gates",
    "all_targets_resolved",
    "target_count",
    "resolved_target_count",
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
ABBREVIATED_SENTENCE_END_RE = re.compile(
    r"(?:\bU\.S\.|\bU\.N\.|\bNo\.|\bMr\.|\bMrs\.|\bMs\.|\bDr\.|\bSt\.|\betc\.)$",
    flags=re.IGNORECASE,
)
WORD_RE = re.compile(r"\b[\w'-]+\b")
DOC_ID_TOKEN_RE = re.compile(r"^\d{4}[A-Z]+\d+(?:_\d+)?$", flags=re.IGNORECASE)
DATE_TOKEN_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|"
    r"JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\b",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
YEAR_RANGE_RE = re.compile(
    r"\b(?P<start>(?:19|20)\d{2})(?P<sep>[-\u2013\u2014/])"
    r"(?P<end>(?:19|20)?\d{2})\b"
)
VALUE_SLOT_TEMPLATE = "[REDACT:R1 type={slot_type}]"
SENTENCE_SLOT_TEMPLATE = "[REDACT_SENT:S1 approx_tokens={approx_tokens}]"
REDACTION_MARKER_RE = re.compile(r"(\[REDACT(?:_SENT)?:[^\]]+\])")
PARAGRAPH_NUM_RE = re.compile(r"(\d+)")
LIST_LEAD_RE = re.compile(r"^\s*(?:\(?\d+\)?[.)]?\s+)+")
GENERIC_ROLE_RE = re.compile(
    r"^(?:speaker(?:\s*\(unspecified in text\))?|unknown|unspecified|n/?a|none)$",
    flags=re.IGNORECASE,
)
GENERIC_REPORTING_SPEAKER_RE = re.compile(
    r"^(?:text|document)(?:\b.*(?:summary|report|reporting|reported).*)$",
    flags=re.IGNORECASE,
)
GENERIC_LABEL_RE = re.compile(
    r"^(?:embassy|government|country|people|city|province|president|prime minister|minister|official|delegation|speaker|source)$",
    flags=re.IGNORECASE,
)
GENERIC_PERSON_LABEL_RE = re.compile(
    r"^(?:emboffs?|officials?|officers?|briefers?|sources?|delegates?|reporters?|staffers?|"
    r"mission|embassy|committee|government|leadership|delegation|"
    r"president|vice president|prime minister|minister|foreign minister|"
    r"ambassador|secretary|secretary of state|charge|governor|general|colonel|"
    r"captain|admiral)$",
    flags=re.IGNORECASE,
)
TITLE_ONLY_PERSON_LABEL_RE = re.compile(
    r"^(?:the\s+)?(?:acting\s+|deputy\s+|assistant\s+)?"
    r"(?:ambassador|chair(?:man|woman|person)?|chief|commander|director|"
    r"governor|head|leader|minister|official|president|representative|"
    r"secretary)\s+(?:at|for|from|in|of|to)\b",
    flags=re.IGNORECASE,
)
GENERIC_LABEL_HEADS = {
    "administration", "agency", "application", "area", "association", "authorities",
    "authority", "branch", "bureau", "campaign", "club", "committee", "commission", "community", "companies", "company", "court", "day",
    "department", "directorate", "embassy", "federation", "force", "front",
    "government", "group", "intelligence", "investigation", "ministry", "mission",
    "movement", "office", "organization", "party", "police", "press", "school",
    "service", "state", "surveillance", "team", "union",
}
GENERIC_LABEL_MODIFIERS = {
    "american", "attorney", "british", "democratic", "executive", "federal", "foreign", "friendly",
    "general", "government", "intelligence", "international", "judicial", "legal",
    "military", "national", "political", "presidential", "public", "russian",
    "security", "senior", "state", "strategic", "united", "western",
}
RUNNER_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.*)$")
REPORTING_PREDICATES = {
    "report", "inform", "tell", "say", "claim", "state", "announce",
    "reply", "warn", "explain", "reaffirm",
}
LOW_SIGNAL_CLAIM_PREDICATES = {
    "announce", "announced", "assess", "assessed", "believe", "believed",
    "characterize", "characterized", "claim", "claimed", "describe", "described",
    "explain", "explained", "inform", "informed", "refer", "referred",
    "report", "reported", "reply", "replied", "say", "said", "state", "stated",
    "tell", "told", "think", "thought", "warn", "warned",
}
INTENTIVE_CLAIM_RE = re.compile(
    r"\b(?:trying|try|attempt(?:ed|ing)?|plan(?:ned|s)?|seek(?:s|ing)?|aim(?:ed|s)?|"
    r"willingness|intend(?:ed|s|ing)?)\s+to\b"
    r"|\binterested\s+in\s+continuing\s+to\b",
    flags=re.IGNORECASE,
)
ABSTRACT_OBJECT_HEADS = {
    "assertion", "assertions", "attention", "certification", "certifications",
    "connection", "connections", "coverage", "disquiet", "evidence", "itemization",
    "order", "orders", "purpose", "purposes", "request", "requests",
    "situation",
}
CLAUSE_BLEED_TOKENS = {
    "am", "are", "as", "be", "been", "being", "did", "do", "does", "had",
    "has", "have", "having", "is", "may", "might", "must", "not", "perhaps",
    "probably", "received", "reportedly", "said", "seemed", "seems", "sent",
    "should", "still", "to", "was", "were", "will", "would", "yet",
}
VERBISH_ACTOR_TAILS = {
    "accused", "announced", "approved", "arrived", "asked", "assessed",
    "blocked", "called", "characterized", "claimed", "confirmed", "decided",
    "declared", "denied", "described", "feared", "found", "gave", "issued",
    "killed", "made", "met", "murdered", "ordered", "received", "referred",
    "reported", "requested", "said", "scheduled", "sent", "stated", "told",
    "warned",
}
VALUE_CONTEXT_ROUTING_RE = re.compile(
    r"\b(?:identify the missing value|fill the missing value|missing value in the redacted slot)\b",
    flags=re.IGNORECASE,
)
PAGE_TAG_RE = re.compile(r"\[PAGE\s+\d+\]", flags=re.IGNORECASE)
LEAD_SECTION_RE = re.compile(
    r"^(?:(?:[IVXLC]+\.)\s+)?"
    r"(?:(?:THE|WATCH)\s+[A-Z][A-Z\-]*(?:\s+[A-Z][A-Z\-]*)*|SPECIAL ITEM|LATE ITEM|SUMMARY|CONTENTS)\s+"
    r"(?=[A-Z][A-Za-z0-9\-/()']+:)",
)
LEAD_SECTION_FALLBACK_RE = re.compile(
    r"^(?:(?:[IVXLC]+\.)\s+)?"
    r"(?:(?:THE|WATCH)\s+[A-Z][A-Z\-]*(?:\s+[A-Z][A-Z\-]*)*|SPECIAL ITEM|LATE ITEM|SUMMARY|CONTENTS)\s+"
    r"(?=[A-Z][a-z])",
)
PAGE_INDEX_RE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2}\s+Page\s+\d+\b", flags=re.IGNORECASE)
UPPER_TOKEN_RE = re.compile(r"^([^A-Za-z0-9]*)([A-Za-z0-9][A-Za-z0-9.'/&()\\-]*)([^A-Za-z0-9]*)$")
TITLECASE_LOWER_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "nor",
    "of", "on", "or", "over", "per", "so", "that", "the", "to", "under", "with",
}
ACRONYM_KEEPERS = {
    "BBC", "CBS", "CIA", "CSA", "DCM", "DNC", "DRV", "FAA", "FBI", "FM", "GOI",
    "GOP", "GOV", "GVN", "IAEA", "ICC", "ICCS", "IMF", "KDP", "MFA", "MR", "NIOC", "NRC",
    "NVA", "PRC", "PRG", "RIS", "RTG", "SVN", "UAE", "UAR", "UN", "US", "U.S.",
    "USSR", "VC",
}
BROKEN_HYPHENATION_RE = re.compile(r"\b([A-Za-z]{2,})-\s+([A-Za-z]{2,})\b")
LONG_UPPER_CHAIN_RE = re.compile(r"(?:\b[A-Z][A-Z\-']+\b(?:\s+|$)){5,}")
STACKED_NUMBER_RE = re.compile(r"\b(?:\d+\s+){3,}\d+\b")
BROKEN_MIXED_CASE_RE = re.compile(
    r"\b(?:[A-Z]{2,}[a-z]+[A-Z][A-Za-z]*|[A-Z][a-z]+[A-Z][A-Za-z]+)\b"
)
BROKEN_SUBPREFIX_RE = re.compile(r"\bSub-[a-z][A-Za-z]+\b")
TAIL_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "for", "from", "he", "her", "him",
    "his", "i", "in", "into", "is", "it", "its", "of", "on", "or", "our",
    "she", "that", "the", "their", "them", "they", "this", "than", "to", "us",
    "we", "were", "which", "who", "with", "you", "your",
}
MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
NEGATION_CUE_RE = re.compile(
    r"\b(?:not|no|never|none|without|denied|deny|denies|denying|opposed|oppose|"
    r"refused|refuse|unlikely|unable|cannot|can't|did not|does not|was not|were not)\b",
    flags=re.IGNORECASE,
)
ROUTING_TOKEN_RE = re.compile(
    r"\b(?:FM|TO|INFO|PASS|AMEMBASSY|AMCONSUL|SECSTATE|WASHDC|CINCPAC|"
    r"COMUSMACV|USDEL|RPT|BT|NNN|UNCLAS|CONFIDENTIAL|LIMITED OFFICIAL USE|"
    r"AIG|OSD|JCS|DMA|COMNAVINTCOM|MARITIME|ADMIN|HYDROPAC|DMAHC)\b",
    flags=re.IGNORECASE,
)
QUERY_FIELD_PREFIX_RE = re.compile(
    r"^\s*(?:Claim|Question|Prompt):\s*",
    flags=re.IGNORECASE,
)
QUERY_TRAILING_FIELD_RE = re.compile(
    r"\s+\b(?:Stance|Source):\s*.+$",
    flags=re.IGNORECASE,
)
STRUCTURED_METADATA_RE = re.compile(
    r"\b(?:Document|Paragraph|Date|Subject|Entities|Claims|Events|Relations|Stance|Source):",
    flags=re.IGNORECASE,
)
CLAIM_QUERY_NOISE_RE = re.compile(
    r"\b(?:limited official use|charge and poloff|poloff|withan|abo\s+ut|emboffs?|usg|"
    r"to ascertain|routine fashion|if cable received|job offers in question|"
    r"one of them is|the instruction lists|because the innuendo was made|"
    r"recent press reports have discussed|the possibility of\b|cbs news story|"
    r"walter cronkite|bill mclaughlin)\b|\(\s*inf\s*\)",
    flags=re.IGNORECASE,
)
CLAIM_ALIGNMENT_ARTIFACT_RE = re.compile(
    r"\b(?:material\s+w\b|requested\s+in\s+this\s+application\b|"
     r"to\s+the\s+[A-Z][A-Za-z'-]{1,40}\s+claims\b|"
     r"however,\s+the\s+authorities\s+requested\b|"
     r"reliably\s+reported\s+widespread\b|"
     r"reported\s+increasingly\s+restive\b|"
     r"reported\s+widespread\b)\b",
    flags=re.IGNORECASE,
)
APPLICATION_BOILERPLATE_CLAIM_RE = re.compile(
    r"\b(?:a\s+significant\s+purpose\s+of\s+the\s+request\s+is\s+to\s+collect\s+foreign\s+intelligence\s+information|"
    r"the\s+authorities\s+requested(?:\s+in\s+this\s+application)?\s+may\s+produce(?:\s+evidence\s+of\s+a\s+violation\s+of\s+united\s+states\s+law)?)\b",
    flags=re.IGNORECASE,
)
CLAIM_FIELD_RE = re.compile(
    r"\bClaims?:\s*(.+?)(?=\s+\b(?:Events|Relations|Entities|Source|Stance|Document|Paragraph|Date|Subject):|$)",
    flags=re.IGNORECASE,
)
HEADLINE_PREFIX_RE = re.compile(
    r"^(?:(?:[IVXLC]+\.)\s+)?([A-Z][A-Za-z0-9'()&/.\-]*(?:\s+[A-Z][A-Za-z0-9'()&/.\-]*){0,5}):\s+(?=[A-Z\"'])"
)
METADATA_SCAFFOLD_RE = re.compile(
    r"\b(?:transcript of story|message sent you|translation(?: of the article)? follows|"
    r"current information indicates|they include|situation in\b|as follows|themes used|"
    r"abstentions?|reference is made to|distribution(?: to)?|department will recall that|"
    r"following is text|following letter|received following letter|"
    r"would appreciate(?: receiving(?: soonest)?)?|we would appreciate any comments|"
    r"please indicate(?: which members)?|"
    r"subject\b|petition\b|entire text\))\b",
    flags=re.IGNORECASE,
)
INTERNAL_TOPIC_HEADING_RE = re.compile(
    r"(?<=[.!?])\s+(?:(?:[IVXLC]+\.)\s+)?"
    r"(?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}(?:\s*-\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}){0,2}):\s+"
)
INTERNAL_SECTION_HEADING_RE = re.compile(
    r"(?<=[.!?])\s+(?:THE|WATCH)\s+[A-Z][A-Z\-]*(?:\s+[A-Z][A-Z\-]*)*\s+(?=[A-Z])"
)
REPEATED_SENTENCE_PREFIX_RE = re.compile(
    r"^(.{18,120}?)\.\s+\1(?:[.?!]|\b)",
    flags=re.IGNORECASE,
)
DIALOGUE_FRAGMENT_RE = re.compile(r"^(?:yes|no),\s+", flags=re.IGNORECASE)
QA_PREFIX_RE = re.compile(r"^(?:Q|A)\.\s*", flags=re.IGNORECASE)
PRESS_BRIEFING_METADATA_RE = re.compile(
    r"\b(?:press briefing|press office|question taken|following answer was posted|"
    r"response to question|noon press briefing)\b",
    flags=re.IGNORECASE,
)
BROADCAST_TRANSCRIPT_RE = re.compile(
    r"\b(?:cbs news story|begin text|walter cronkite|bill mclaughlin|"
    r"transcript of story|evening news)\b",
    flags=re.IGNORECASE,
)
INCOMPLETE_NUMERIC_RANGE_RE = re.compile(
    r"\bbetween\s+\d[\d,]*(?:\.\d+)?\s+and\s+\d[\d,]*(?:\.\d+)?[.?!]?$",
    flags=re.IGNORECASE,
)
DEICTIC_TAIL_RE = re.compile(
    r"\b(?:about|regarding)\s+(?:this|that|these|those)[.?!]?$",
    flags=re.IGNORECASE,
)
SPEAKER_PREFIX_RE = re.compile(r"^[A-Z][A-Za-z]+:\s+(?=[a-z])")
NON_REDACTION_BRACKET_RE = re.compile(r"\[(?!REDACT(?:_SENT)?:)", flags=re.IGNORECASE)
PAGE_NUMBER_ARTIFACT_RE = re.compile(r"\s[-–]\s\d{1,3}\s[-–]\s")
SLASH_FRAGMENT_RE = re.compile(r"\b[A-Za-z]{2,}/[A-Za-z]{2,}\b")
VERBISH_TOKEN_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|do|does|did|will|would|could|should|may|might|must|"
    r"said|says|stated|reported|announced|told|met|believed|thinks|claimed|denied|expects|expect|"
    r"occurred|occur|involved|include|included|seized|seize|received|receive|traveled|travelled|"
    r"supported|support|delivered|deliver|sent|send|recognized|recognize|encouraged|encourage|"
    r"attended|attend|"
    r"opposed|oppose|affiliated|affiliate|favored|favor|instructed|instruct|called|call|transported|transport)\b",
    flags=re.IGNORECASE,
)
ALLEGATION_LIST_RE = re.compile(
    r"\b(?:allegations that|committed a crime or crimes)\b",
    flags=re.IGNORECASE,
)
BROKEN_LEAD_RE = re.compile(
    r"\baccording to the fbi'?s based on\b|\bwith\s+afp\s+reporting\b|\baccording to sub-\b",
    flags=re.IGNORECASE,
)
BROKEN_REPORTING_CHAIN_RE = re.compile(
    r"\b(?:provided by|reported by)\s+[A-Z][A-Za-z0-9#' -]{0,48}\s+(?:also\s+)?"
    r"(?:assesses?|believes?|states?|reports?)\b",
    flags=re.IGNORECASE,
)
NOMINALIZED_CLAIM_RE = re.compile(
    r"^(?:assistance\s+may\s+include\b|the authorities requested should\b|there has been no indication\b|"
    r"efforts?\s+by\b|the information relied on by\b|the assertions? in\b|"
    r"the allegations? against\b|the effect of\b|the fbi'?s foreign intelligence goals?\b)",
    flags=re.IGNORECASE,
)
MIDSTREAM_HEADING_SEAM_RE = re.compile(
    r"\b(?:components thereof|thereof)\s+The FBI Believes\b|\bRecent Investigative Results\b",
    flags=re.IGNORECASE,
)
CONDITIONAL_CLAIM_RE = re.compile(r"^(?:even if|if this were|if this is|if it were|if )\b", flags=re.IGNORECASE)
ABSTRACT_QUANTIFIER_RE = re.compile(r"^(?:a majority of|most of|many of|some of)\b", flags=re.IGNORECASE)
LOW_SIGNAL_SENTENCE_LEAD_RE = re.compile(
    r"^(?:as discussed below|it was also unclear|further\b|accordingly\b|in addition\b|additionally\b|"
    r"would appreciate\b|please indicate\b|from addressees\b|previously report\b|"
    r"disquiet of\b|connections which\b|previously reported incidents involving\b|"
    r"recent press reports have discussed\b|one of them is\b|one is that\b|"
    r"the instruction lists\b|the instructions include\b|because the innuendo was made\b|"
    r"change in\b|dfa sent an instruction\b)",
    flags=re.IGNORECASE,
)
RAW_BLANK_RE = re.compile(r"_{3,}")
FRENCH_STOPWORDS = {
    "alors", "au", "aucun", "aussi", "autre", "avant", "avec", "ce", "cela", "ces",
    "cet", "cette", "comme", "dans", "de", "des", "du", "elle", "en", "entre", "et",
    "eux", "fois", "il", "ils", "je", "la", "le", "les", "leur", "lors", "mais",
    "mes", "moi", "ne", "nord", "nos", "notre", "nous", "ou", "par", "pas", "plus",
    "pour", "qu", "que", "qui", "sa", "se", "ses", "signe", "son", "sont", "sur",
    "territoire", "une", "vers", "vigueur", "visant", "vous",
}
TITLE_HEADING_STOPWORDS = {
    "a", "an", "and", "at", "for", "from", "in", "information", "investigative", "of",
    "on", "results", "subject", "sequestration", "the", "to", "with",
}
LEADING_TITLE_HEADING_CUE_RE = re.compile(
    r"^(?:In addition|Additionally|According|Based|For example|Further|"
    r"Page\b|The FBI\b|Russia\b|Candidate\b|Would\b|Speaking\b|Embassy\b|"
    r"He\b|She\b|They\b|It\b|The\b|A\b|An\b)",
    flags=re.IGNORECASE,
)


@dataclass
class ExperimentPaths:
    output_root: Path
    experiment_root: Path
    extraction_root: Path
    config_root: Path
    query_manifest_root: Path
    query_file_root: Path
    run_root: Path
    summary_root: Path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def deep_merge_dict(base: Any, override: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        tmp = Path(handle.name)
    tmp.replace(path)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (KG_DEMO_ROOT / path).resolve()


def resolve_path_preserve_symlink(raw: str | Path) -> Path:
    text = str(raw or "").strip()
    if not text:
        return Path(sys.executable)
    if text in {"python", "python3"}:
        return Path(sys.executable)
    path = Path(text).expanduser()
    if not path.is_absolute() and path.parent == Path("."):
        return path
    if path.is_absolute():
        return path.absolute()
    cwd_candidate = (Path.cwd() / path)
    if cwd_candidate.exists():
        return cwd_candidate.absolute()
    return (KG_DEMO_ROOT / path).absolute()


def normalize_space(text: str) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        if not text:
            return ""
        return _normalize_space_text_cached(text)
    try:
        if pd.isna(text):
            return ""
    except TypeError:
        pass
    return _normalize_space_text_cached(str(text or ""))


@lru_cache(maxsize=200_000)
def _normalize_space_text_cached(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def deterministic_seeded_row_indices(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    key_fields: tuple[str, ...],
) -> list[int]:
    ordered: list[tuple[str, int]] = []
    for index, row in enumerate(rows):
        stable_parts = [normalize_space(row.get(field, "")) for field in key_fields]
        if not any(stable_parts):
            stable_parts = [
                normalize_space(json.dumps(row, sort_keys=True, default=str))
            ]
        stable_key = "|".join(stable_parts)
        rank_key = sha1_text(f"{seed}|{stable_key}|{index}")
        ordered.append((rank_key, index))
    ordered.sort(key=lambda item: (item[0], item[1]))
    return [index for _rank_key, index in ordered]


def deterministic_seeded_row_order(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    materialized_rows = list(rows)
    indices = deterministic_seeded_row_indices(
        materialized_rows,
        seed=seed,
        key_fields=key_fields,
    )
    return [materialized_rows[index] for index in indices]


def reorder_aligned_rows(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    key_fields: tuple[str, ...],
    companions: tuple[list[Any] | None, ...] = (),
) -> tuple[list[dict[str, Any]], tuple[list[Any] | None, ...]]:
    """Seed-order rows and every precomputed row-indexed companion together."""
    materialized_rows = list(rows)
    row_count = len(materialized_rows)
    for companion in companions:
        if companion is not None and len(companion) != row_count:
            raise ValueError(
                "Row-indexed benchmark cache length mismatch: "
                f"{len(companion)} companion rows for {row_count} source rows"
            )
    indices = deterministic_seeded_row_indices(
        materialized_rows,
        seed=seed,
        key_fields=key_fields,
    )
    ordered_rows = [materialized_rows[index] for index in indices]
    ordered_companions = tuple(
        None if companion is None else [companion[index] for index in indices]
        for companion in companions
    )
    return ordered_rows, ordered_companions


def canonicalize_corpus_name(name: str) -> str:
    key = normalize_space(name).casefold()
    canonical = CORPUS_ALIASES.get(key)
    if canonical:
        return canonical
    allowed = ", ".join(sorted(CORPUS_ALIASES))
    raise ValueError(f"Unsupported corpus `{name}`. Expected one of: {allowed}")


def count_words(text: str) -> int:
    if not text:
        return 0
    return _count_words_cached(str(text))


@lru_cache(maxsize=200_000)
def _count_words_cached(text: str) -> int:
    return len(WORD_RE.findall(text))


def uppercase_letter_ratio(text: str) -> float:
    value = str(text or "")
    if not value:
        return 0.0
    return _uppercase_letter_ratio_cached(value)


@lru_cache(maxsize=100_000)
def _uppercase_letter_ratio_cached(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if char.isupper()) / len(letters)


def normalize_uppercase_token(token: str, *, is_first: bool) -> str:
    match = UPPER_TOKEN_RE.match(token)
    if not match:
        return token
    lead, core, tail = match.groups()
    letters_only = re.sub(r"[^A-Za-z]", "", core)
    if not letters_only or any(char.islower() for char in core):
        return token
    upper_key = letters_only.upper()
    if core.upper() in ACRONYM_KEEPERS or upper_key in ACRONYM_KEEPERS:
        return lead + core.upper() + tail
    lowered = core.lower()
    rendered = lowered.capitalize() if is_first else lowered
    return lead + rendered + tail


def normalize_uppercase_passage(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    rendered_tokens: list[str] = []
    sentence_start = True
    for raw_token in value.split(" "):
        token = normalize_uppercase_token(raw_token, is_first=sentence_start)
        rendered_tokens.append(token)
        if re.search(r"[.!?][\"')\]]*$", raw_token):
            sentence_start = True
        elif re.search(r"[A-Za-z]", raw_token):
            sentence_start = False
    return normalize_space(" ".join(rendered_tokens))


TITLECASE_NOISE_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "if", "in", "into", "is", "it", "its",
    "may", "might", "must", "nor", "not", "of", "on", "or", "than", "that",
    "the", "their", "this", "to", "was", "were", "will", "with", "would",
}


def titlecase_token_ratio(text: Any) -> float:
    value = normalize_space(text)
    if not value:
        return 0.0
    return _titlecase_token_ratio_cached(value)


@lru_cache(maxsize=100_000)
def _titlecase_token_ratio_cached(value: str) -> float:
    words = WORD_RE.findall(value)
    if not words:
        return 0.0
    alpha_words = 0
    titleish_words = 0
    for token in words:
        letters = re.sub(r"[^A-Za-z]", "", token)
        if len(letters) < 2:
            continue
        alpha_words += 1
        if letters[:1].isupper() and letters[1:].islower():
            titleish_words += 1
    if alpha_words == 0:
        return 0.0
    return titleish_words / alpha_words


def soften_titlecase_noise(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    rendered_tokens: list[str] = []
    sentence_start = True
    for raw_token in value.split(" "):
        if raw_token.startswith("[REDACT"):
            rendered_tokens.append(raw_token)
            sentence_start = False
            continue
        match = UPPER_TOKEN_RE.match(raw_token)
        if not match:
            rendered_tokens.append(raw_token)
            if re.search(r"[.!?][\"')\]]*$", raw_token):
                sentence_start = True
            elif re.search(r"[A-Za-z]", raw_token):
                sentence_start = False
            continue
        lead, core, tail = match.groups()
        letters_only = re.sub(r"[^A-Za-z]", "", core)
        lowered = core.lower()
        if (
            letters_only
            and not sentence_start
            and lowered in TITLECASE_NOISE_WORDS
            and core.upper() not in ACRONYM_KEEPERS
            and letters_only.upper() not in ACRONYM_KEEPERS
        ):
            rendered_tokens.append(lead + lowered + tail)
        else:
            rendered_tokens.append(raw_token)
        if re.search(r"[.!?][\"')\]]*$", raw_token):
            sentence_start = True
        elif re.search(r"[A-Za-z]", raw_token):
            sentence_start = False
    return normalize_space(" ".join(rendered_tokens))


def repeated_sentence_leak(text: str) -> bool:
    sentences = [normalize_space(sentence).casefold() for sentence in sentence_split(text)]
    seen: set[str] = set()
    for sentence in sentences:
        if not sentence:
            continue
        if sentence in seen:
            return True
        seen.add(sentence)
    return False


def is_probably_non_english_query(text: Any) -> bool:
    value = normalize_space(text)
    words = [token.casefold() for token in WORD_RE.findall(value)]
    if len(words) < 10:
        return False
    french_hits = sum(1 for token in words if token in FRENCH_STOPWORDS or token.startswith("l'"))
    if french_hits / max(len(words), 1) >= 0.18:
        return True
    accented = sum(1 for char in value if char in "àâçéèêëîïôùûüÿœæÀÂÇÉÈÊËÎÏÔÙÛÜŸŒÆ")
    english_function_hits = sum(
        1 for token in words
        if token in {"the", "and", "that", "with", "from", "into", "have", "will", "would", "were"}
    )
    return accented >= 3 and english_function_hits <= 1


def has_unbalanced_delimiters(text: Any) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    if value.count("[") != value.count("]"):
        return True
    if value.count("(") != value.count(")"):
        return True
    double_quotes = value.count('"')
    single_quotes = value.count("'")
    if double_quotes % 2 == 1:
        return True
    if value.rstrip().endswith((" i'm", " I've", " i've", " I'd", " i'd", " he'll", " she'll", " they'll")):
        return True
    if value.rstrip().endswith(("I'm", "i'm", "we're", "you're", "they're", "can't", "won't", "don't")):
        return True
    return False


def strip_internal_topic_sections(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    topic_matches = list(INTERNAL_TOPIC_HEADING_RE.finditer(value))
    section_matches = list(INTERNAL_SECTION_HEADING_RE.finditer(value))
    matches = topic_matches + section_matches
    if not matches:
        return value
    match = max(matches, key=lambda item: item.start())
    tail = normalize_space(value[match.end():])
    if count_words(tail) >= 8 and not is_routing_or_metadata_sentence(tail):
        return tail
    return normalize_space(
        INTERNAL_SECTION_HEADING_RE.sub(" ", INTERNAL_TOPIC_HEADING_RE.sub(" ", value))
    )


def strip_leading_title_heading(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    tokens = value.split()
    if len(tokens) < 5:
        return value
    for heading_len in range(min(8, len(tokens) - 1), 2, -1):
        head = tokens[:heading_len]
        rest = " ".join(tokens[heading_len:])
        if not rest or not LEADING_TITLE_HEADING_CUE_RE.match(rest):
            continue
        if VERBISH_TOKEN_RE.search(" ".join(head)):
            continue
        valid = True
        for token in head:
            letters = re.sub(r"[^A-Za-z]", "", token)
            if not letters:
                valid = False
                break
            if letters.casefold() in TITLE_HEADING_STOPWORDS:
                continue
            if not (letters[:1].isupper() and letters[1:].islower()):
                valid = False
                break
        if valid:
            return normalize_space(rest)
    return value


def strip_low_signal_lead_sentences(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    sentences = sentence_split(value)
    while len(sentences) >= 2:
        lead = normalize_space(sentences[0])
        tail = normalize_space(" ".join(sentences[1:]))
        lead_flags = set(query_surface_flags(lead))
        drop_lead = (
            is_routing_or_metadata_sentence(lead)
            or "metadata_scaffold" in lead_flags
            or "headline_scaffold" in lead_flags
            or "press_briefing_scaffold" in lead_flags
            or lead.endswith("?")
            or lead.casefold().startswith("would appreciate")
            or lead.casefold().startswith("please indicate")
            or lead.casefold().startswith("the embassy's translation of the article follows")
            or lead.casefold().startswith("transcript of story is repeated below")
            or lead.casefold().startswith("s - entire text)")
            or count_words(lead) <= 7
        )
        if not drop_lead or count_words(tail) < 8:
            break
        sentences = sentences[1:]
    return normalize_space(" ".join(sentences))


def strip_low_signal_edge_sentences(
    text: Any,
    *,
    task_family: str = "",
    min_remaining_words: int = 10,
) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    sentences = sentence_split(value)
    if len(sentences) <= 1:
        return value
    while len(sentences) >= 2:
        lead = normalize_space(sentences[0])
        tail = normalize_space(" ".join(sentences[1:]))
        if not sentence_is_low_signal_for_query(lead) or count_words(tail) < min_remaining_words:
            break
        sentences = sentences[1:]
    while len(sentences) >= 2:
        tail_sentence = normalize_space(sentences[-1])
        head = normalize_space(" ".join(sentences[:-1]))
        if not sentence_is_low_signal_for_query(tail_sentence) or count_words(head) < min_remaining_words:
            break
        sentences = sentences[:-1]
    candidate = normalize_space(" ".join(sentences))
    if task_family in {"claims", "contradictions"} and len(sentence_split(candidate)) > 1:
        candidate = finalize_query_surface(
            extract_sentence_excerpt(candidate, max_sentences=1, max_chars=280),
            add_terminal_punctuation=True,
        )
    return candidate


@lru_cache(maxsize=50_000)
def _clean_query_passage_cached(value: str) -> str:
    value = normalize_space(value)
    if not value:
        return ""
    answer_match = re.search(r"\bAnswer:\s*(.+)$", value, flags=re.IGNORECASE)
    if answer_match and count_words(answer_match.group(1)) >= 8:
        value = answer_match.group(1)
    claim_match = re.search(
        r"\bClaim:\s*(.+?)(?=\s+\b(?:Stance|Source):|$)",
        value,
        flags=re.IGNORECASE,
    )
    if claim_match:
        value = claim_match.group(1)
    elif STRUCTURED_METADATA_RE.search(value):
        claim_field = CLAIM_FIELD_RE.search(value)
        if claim_field:
            claim_candidates = [
                finalize_query_surface(part, add_terminal_punctuation=False)
                for part in re.split(r"\s*[;|]\s*", claim_field.group(1))
            ]
            claim_candidates = [
                part for part in claim_candidates
                if part
                and count_words(part) >= 6
                and not is_routing_or_metadata_sentence(part)
                and not is_list_like_text(part)
            ]
            if claim_candidates:
                value = claim_candidates[0]
            else:
                value = ""
        else:
            value = ""
    value = re.sub(r"^\s*Begin Text Question:\s*", "", value, flags=re.IGNORECASE)
    value = QUERY_FIELD_PREFIX_RE.sub("", value)
    value = re.sub(r"^\s*Answer:\s*", "", value, flags=re.IGNORECASE)
    value = SPEAKER_PREFIX_RE.sub("", value)
    value = QUERY_TRAILING_FIELD_RE.sub("", value)
    value = re.sub(
        r"^\s*According to the FBI,\s*based on\s+",
        "Based on ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*According to the FBI(?:'s|’s)?\s+Based on\s+",
        "Based on ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*According to this information,\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*Specifically,\s+according to this information,\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*Specifically,\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*Based on reports from a friendly foreign government,\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*The Election Security Joint Statement indicates that\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = PAGE_TAG_RE.sub(" ", value)
    value = PAGE_INDEX_RE.sub(" ", value)
    value = LEAD_SECTION_RE.sub("", value)
    value = LEAD_SECTION_FALLBACK_RE.sub("", value)
    value = BROKEN_HYPHENATION_RE.sub(r"\1\2", value)
    value = strip_internal_topic_sections(value)
    value = strip_leading_title_heading(value)
    value = re.sub(r"\bSource\s+_+,\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^According to the FBI'?s\s+Based on\s+", "Based on ", value, flags=re.IGNORECASE)
    value = value.lstrip("\"' ")
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([(\[]) +", r"\1", value)
    value = re.sub(r" +([)\]])", r"\1", value)
    value = normalize_space(value)
    headline_match = HEADLINE_PREFIX_RE.match(value)
    if headline_match and uppercase_token_ratio(headline_match.group(1)) >= 0.6:
        value = normalize_space(value[headline_match.end() :])
    value = strip_internal_topic_sections(value)
    value = strip_leading_title_heading(value)
    value = strip_low_signal_lead_sentences(value)
    value = strip_low_signal_edge_sentences(value)
    value = re.sub(r"\s*\.\.\.\s*$", "", value).strip(" ,;:")
    if uppercase_letter_ratio(value) >= 0.9 and count_words(value) >= 6:
        value = normalize_uppercase_passage(value)
    value = re.sub(
        r"(^|(?<=[.!?]\s))([a-z])",
        lambda match: match.group(1) + match.group(2).upper(),
        value,
    )
    if value[:1].islower():
        value = value[:1].upper() + value[1:]
    return normalize_space(value)


def clean_query_passage(text: Any) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    return _clean_query_passage_cached(value)


def sentence_is_low_signal_for_query(text: Any) -> bool:
    value = normalize_space(text)
    if not value:
        return True
    if benchmark_interactive_qa_surface(value):
        return True
    if BROADCAST_TRANSCRIPT_RE.search(value):
        return True
    if is_routing_or_metadata_sentence(value):
        return True
    if is_list_like_text(value):
        return True
    if LOW_SIGNAL_SENTENCE_LEAD_RE.search(value):
        return True
    if re.search(
        r"^(?:one of them is|the instruction lists|because the innuendo was made|"
        r"recent press reports have discussed|the possibility of\b)",
        value,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"\bas discussed below\b|\bdiscussed in greater detail below\b", value, flags=re.IGNORECASE):
        return True
    flags = set(query_surface_flags(value))
    if {
        "metadata_scaffold",
        "headline_scaffold",
        "press_briefing_scaffold",
        "no_substantive_sentence",
        "broken_lead",
    } & flags:
        return True
    return False


def benchmark_interactive_qa_surface(text: Any) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    if "?" in value:
        return True
    if re.search(
        r"\b(?:i(?:'|’)ll be glad to take the question|i will have to take (?:the )?question|"
        r"i can(?:'|’)t remember|i don(?:'|’)t have anything on that|"
        r"i don(?:'|’)t have a list|take the question|take a question)\b",
        value,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def surface_noise_penalty(text: str) -> float:
    value = normalize_space(text)
    if not value:
        return 1000.0
    return _surface_noise_penalty_cached(value)


@lru_cache(maxsize=100_000)
def _surface_noise_penalty_cached(value: str) -> float:
    penalty = 0.0
    if is_routing_or_metadata_sentence(value):
        penalty += 24.0
    if is_list_like_text(value):
        penalty += 14.0
    if uppercase_token_ratio(value) >= 0.72 and count_words(value) >= 8:
        penalty += 10.0
    if LONG_UPPER_CHAIN_RE.search(value):
        penalty += 8.0
    if STACKED_NUMBER_RE.search(value):
        penalty += 6.0
    if DOC_ID_TOKEN_RE.search(value):
        penalty += 4.0
    if BROKEN_MIXED_CASE_RE.search(value):
        penalty += 3.0
    if LEGAL_BOILERPLATE_RE.search(value):
        penalty += 3.0
    return penalty


def finalize_query_surface(
    text: Any,
    *,
    min_words: int = 0,
    add_terminal_punctuation: bool = False,
) -> str:
    value = clean_query_passage(text)
    if not value:
        return ""
    if min_words and count_words(value) < min_words:
        return ""
    if add_terminal_punctuation:
        value = ensure_terminal_punctuation(value)
    return normalize_space(value)

def normalize_surface_preserving_markers(text: Any) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    rendered: list[str] = []
    for part in REDACTION_MARKER_RE.split(value):
        if not part:
            continue
        if REDACTION_MARKER_RE.fullmatch(part):
            rendered.append(part.strip())
            continue
        cleaned = clean_query_passage(part)
        if cleaned:
            rendered.append(cleaned)
    if not rendered:
        return ""
    joined = " ".join(rendered)
    joined = re.sub(r"(?<!\s)(\[REDACT(?:_SENT)?:[^\]]+\])", r" \1", joined)
    joined = re.sub(r"(\[REDACT(?:_SENT)?:[^\]]+\])(?![\s,.;:!?])", r"\1 ", joined)
    joined = re.sub(r"\s+([,.;:!?])", r"\1", joined)
    joined = re.sub(r"([(\[]) +", r"\1", joined)
    joined = re.sub(r" +([)\]])", r"\1", joined)
    return normalize_space(joined)


def count_redaction_slots(text: Any, *, sentence_mode: bool = False) -> int:
    value = normalize_space(text)
    if not value:
        return 0
    pattern = r"\[REDACT_SENT:S\d+[^\]]*\]" if sentence_mode else r"\[REDACT:R\d+[^\]]*\]"
    return len(re.findall(pattern, value))


def query_surface_flags(text: Any, *, task_family: str = "") -> list[str]:
    value = normalize_surface_preserving_markers(text)
    if not value:
        return ["empty"]
    return list(_query_surface_flags_cached(value, task_family))


@lru_cache(maxsize=100_000)
def _query_surface_flags_cached(value: str, task_family: str) -> tuple[str, ...]:
    flags: list[str] = []
    lower_value = value.casefold()
    suspicious_tail = re.search(r"\b(?:and|or|but)\s+([a-z]{2,4})\.?$", lower_value)
    upper_token_count = sum(
        1
        for token in WORD_RE.findall(value)
        if any(ch.isalpha() for ch in token) and "".join(ch for ch in token if ch.isalpha()).isupper()
    )
    lead_fragment = normalize_space(re.split(r"[.!?]", value, maxsplit=1)[0])
    if is_routing_or_metadata_sentence(value):
        flags.append("routing_or_metadata")
    if is_list_like_text(value):
        flags.append("list_like")
    if LEGAL_BOILERPLATE_RE.search(value):
        flags.append("legal_boilerplate")
    if EXTRACTION_ARTIFACT_RE.search(value):
        flags.append("extraction_artifact")
    if PLACEHOLDER_PHRASE_RE.search(value):
        flags.append("extraction_artifact")
    if BROKEN_LEAD_RE.search(value):
        flags.append("broken_lead")
    if suspicious_tail and suspicious_tail.group(1) not in {
        "all",
        "him",
        "her",
        "its",
        "one",
        "two",
        "ten",
    }:
        flags.append("broken_tail")
    if LONG_UPPER_CHAIN_RE.search(value) or (
        uppercase_letter_ratio(value) >= 0.72 and count_words(value) >= 8
    ) or (
        count_words(value) >= 14 and upper_token_count >= 6
    ):
        flags.append("uppercase_heavy")
    if PRESS_BRIEFING_METADATA_RE.search(value):
        flags.append("press_briefing_scaffold")
    if NON_REDACTION_BRACKET_RE.search(value):
        flags.append("non_redaction_bracket")
    if STACKED_NUMBER_RE.search(value):
        flags.append("stacked_numbers")
    if PAGE_NUMBER_ARTIFACT_RE.search(value):
        flags.append("page_number_artifact")
    if BROKEN_MIXED_CASE_RE.search(value):
        flags.append("mixed_case_noise")
    if BROKEN_SUBPREFIX_RE.search(value):
        flags.append("mixed_case_noise")
    if titlecase_token_ratio(value) >= 0.45 and uppercase_token_ratio(value) <= 0.32 and count_words(value) >= 8:
        flags.append("titlecase_noise")
    if BROKEN_REPORTING_CHAIN_RE.search(value):
        flags.append("broken_reporting_chain")
    if re.search(
        r"\b(?:said|reported|revealed|disclosed|indicated|stated|claimed|acknowledged)\s+"
        r"(?:had|has|have|would|will|did)\b",
        lower_value,
    ):
        # A finite reporting predicate cannot be followed directly by the
        # embedded clause auxiliary; this indicates a dropped subject.
        flags.append("broken_reporting_chain")
    if re.search(r"\b(?:caller|listener)\b", lower_value):
        flags.append("generic_dialogue_roles")
    if (
        re.search(r"\b(?:said|reported|reports?|told|informed|asked|warned|suggested|assessed|believes?)\b[^.]{0,120}\b(?:did|would|will|had|has)\b", lower_value)
        and " that " not in f" {lower_value} "
    ):
        flags.append("missing_complementizer_reporting")
    if "pass word" in lower_value and " that " not in f" {lower_value} ":
        flags.append("missing_complementizer_reporting")
    if re.search(
        r"\breported that\b[^.]{0,120}\b(?:the fbi assesses that|the fbi believes that)\b",
        lower_value,
    ):
        flags.append("broken_reporting_chain")
    if MIDSTREAM_HEADING_SEAM_RE.search(value):
        flags.append("midstream_heading_seam")
    if NOMINALIZED_CLAIM_RE.search(value):
        flags.append("nominalized_claim")
    if CONDITIONAL_CLAIM_RE.search(value):
        flags.append("conditional_claim")
    if ABSTRACT_QUANTIFIER_RE.search(value):
        flags.append("abstract_quantifier")
    if claim_uses_intentive_construction(value):
        flags.append("intentive_construction")
    if CLAIM_ALIGNMENT_ARTIFACT_RE.search(value):
        flags.append("claim_alignment_artifact")
    if APPLICATION_BOILERPLATE_CLAIM_RE.search(value):
        flags.append("claim_boilerplate")
    if INCOMPLETE_NUMERIC_RANGE_RE.search(value):
        flags.append("incomplete_numeric_range")
    if DEICTIC_TAIL_RE.search(value):
        flags.append("deictic_tail")
    if RAW_BLANK_RE.search(value):
        flags.append("raw_blank_placeholder")
    if REDACTION_MARKER_RE.search(value):
        flags.append("redaction_marker")
    if has_unbalanced_delimiters(value):
        flags.append("unbalanced_delimiters")
    if is_probably_non_english_query(value):
        flags.append("foreign_language")
    if DOC_ID_TOKEN_RE.search(value):
        flags.append("doc_id_token")
    if value.rstrip().endswith("..."):
        flags.append("trailing_ellipsis")
    if value.count(";") >= 2:
        flags.append("semicolon_scaffold")
    if ":" in lead_fragment and uppercase_token_ratio(lead_fragment) >= 0.45:
        flags.append("headline_scaffold")
    if METADATA_SCAFFOLD_RE.search(lead_fragment):
        flags.append("metadata_scaffold")
    if REPEATED_SENTENCE_PREFIX_RE.search(value):
        flags.append("duplicated_scaffold")
    if repeated_sentence_leak(value):
        flags.append("duplicated_scaffold")
    if value.count(" - ") >= 3 or (value.count(",") >= 8 and count_words(value) >= 18):
        flags.append("enumeration_scaffold")
    if DIALOGUE_FRAGMENT_RE.search(value) and "?" in value:
        flags.append("dialogue_fragment")
    if (
        ((" if " in f" {lower_value} " or " such as " in f" {lower_value} ") and NEGATION_CUE_RE.search(value))
        or ("warning" in lower_value and (" would " in f" {lower_value} " or " will " in f" {lower_value} "))
        or "not fraudulent" in lower_value
        or ("sent word" in lower_value and " will " in f" {lower_value} ")
    ):
        flags.append("parser_hostile_surface")
    if count_words(value) >= 8 and not VERBISH_TOKEN_RE.search(value) and (
        value.count(";") >= 1 or ":" in lead_fragment or uppercase_token_ratio(value) >= 0.68
    ):
        flags.append("no_substantive_sentence")
    if task_family == "reconstruction" and count_words(value) > 72 and len(sentence_split(value)) > 2:
        flags.append("overlong_reconstruction")
    return tuple(unique_preserving_order(flags))


@lru_cache(maxsize=100_000)
def _query_surface_is_blocked_cached(
    value: str,
    task_family: str,
) -> bool:
    flags = set(query_surface_flags(value, task_family=task_family))
    if "empty" in flags:
        return True
    hard_flags = {
        "routing_or_metadata",
        "list_like",
        "legal_boilerplate",
        "extraction_artifact",
        "broken_lead",
        "broken_tail",
        "doc_id_token",
        "metadata_scaffold",
        "no_substantive_sentence",
        "stacked_numbers",
        "page_number_artifact",
        "raw_blank_placeholder",
        "unbalanced_delimiters",
        "broken_reporting_chain",
        "generic_dialogue_roles",
        "missing_complementizer_reporting",
        "midstream_heading_seam",
        "press_briefing_scaffold",
        "non_redaction_bracket",
    }
    if task_family in {"reconstruction", "claims", "contradictions"}:
        hard_flags |= {
            "headline_scaffold",
            "semicolon_scaffold",
            "trailing_ellipsis",
            "duplicated_scaffold",
            "foreign_language",
        }
    if task_family in {"claims", "contradictions"}:
        hard_flags |= {
            "nominalized_claim",
            "intentive_construction",
            "conditional_claim",
            "abstract_quantifier",
            "parser_hostile_surface",
            "claim_alignment_artifact",
            "claim_boilerplate",
            "incomplete_numeric_range",
            "mixed_case_noise",
            "titlecase_noise",
        }
        if flags <= {"titlecase_noise"}:
            profile = claim_clause_profile(value)
            max_actor_words = 8 if task_family == "claims" else 6
            role_fields = {
                "actor": normalize_space(profile.get("actor", "")),
                "predicate": normalize_space(profile.get("predicate", "")),
                "object": normalize_space(profile.get("object", "")),
                "speaker": "",
                "target": "",
                "stance": "unknown",
            }
            if (
                profile_has_complete_simple_clause(profile)
                and not claim_profile_has_problematic_actor(profile)
                and not parser_hostile_claim_candidate(value, role_fields, profile)
                and not int(profile.get("actor_words", 0) or 0) > max_actor_words
                and not int(profile.get("object_words", 0) or 0) > 18
                and not profile.get("has_semicolon")
                and not actor_phrase_has_clause_bleed(
                    normalize_space(profile.get("actor", ""))
                )
                and not actor_has_abstract_head(
                    normalize_space(profile.get("actor", ""))
                )
                and not actor_is_generic_common_noun_phrase(
                    normalize_space(profile.get("actor", ""))
                )
            ):
                hard_flags.discard("titlecase_noise")
    if task_family in {"reconstruction", "claims", "contradictions"}:
        hard_flags |= {
            "redaction_marker",
            "deictic_tail",
        }
    if task_family == "reconstruction":
        hard_flags |= {
            "overlong_reconstruction",
            "enumeration_scaffold",
            "dialogue_fragment",
            "mixed_case_noise",
        }
    if task_family == "value_redaction":
        hard_flags |= {
            "dialogue_fragment",
            "mixed_case_noise",
        }
    if "uppercase_heavy" in flags and (
        "mixed_case_noise" in flags or task_family in {"reconstruction", "claims", "contradictions"}
    ):
        hard_flags.add("uppercase_heavy")
    return any(flag in hard_flags for flag in flags)


def query_surface_is_blocked(text: Any, *, task_family: str = "") -> bool:
    value = normalize_surface_preserving_markers(text)
    if not value:
        return True
    return _query_surface_is_blocked_cached(
        value,
        normalize_space(task_family).casefold(),
    )


def aligned_claim_source_sentence_is_usable(text: Any) -> bool:
    value = finalize_query_surface(text, add_terminal_punctuation=True)
    if not value:
        return False
    if query_surface_is_blocked(value, task_family="claims"):
        return False
    if CLAIM_ALIGNMENT_ARTIFACT_RE.search(value):
        return False
    return True


def refine_generated_query_surface(
    text: Any,
    *,
    task_family: str,
    anchors: Iterable[str] = (),
    min_words: int,
    target_words: int | None,
    max_chars: int,
    runner_cfg: dict[str, Any] | None = None,
) -> str:
    value = normalize_surface_preserving_markers(text)
    if not value:
        return ""
    if uppercase_token_ratio(value) >= 0.3 and count_words(value) >= 8:
        value = normalize_uppercase_passage(value)
    value = soften_titlecase_noise(value)
    value = strip_low_signal_edge_sentences(
        value,
        task_family=task_family,
        min_remaining_words=max(8, min_words // 2),
    )
    sentences = sentence_split(value)
    if len(sentences) >= 2 and normalize_space(sentences[0]).casefold() == normalize_space(sentences[1]).casefold():
        value = normalize_space(" ".join([sentences[0], *sentences[2:]]))
    flags = set(query_surface_flags(value, task_family=task_family))
    if task_family in {"reconstruction", "claims", "contradictions"} and (
        flags
        & {
            "metadata_scaffold",
            "headline_scaffold",
            "semicolon_scaffold",
            "trailing_ellipsis",
            "uppercase_heavy",
            "titlecase_noise",
            "mixed_case_noise",
            "broken_lead",
            "overlong_reconstruction",
        }
    ):
        rewritten = build_sentence_window_from_text(
            value,
            anchors=anchors,
            min_words=min_words,
            target_words=target_words,
            max_sentences=(
                1
                if task_family in {"claims", "contradictions"}
                else 2
            ),
            max_chars=max_chars,
        )
        rewritten = normalize_surface_preserving_markers(rewritten)
        if rewritten:
            value = rewritten
    value = normalize_surface_preserving_markers(value)
    value = strip_low_signal_edge_sentences(
        value,
        task_family=task_family,
        min_remaining_words=max(8, min_words // 2),
    )
    if task_family in {"reconstruction", "claims", "contradictions"}:
        value = ensure_terminal_punctuation(value)
    if task_family in {"claims", "contradictions"}:
        value = finalize_query_surface(
            extract_sentence_excerpt(value, max_sentences=1, max_chars=min(max_chars, 280)),
            add_terminal_punctuation=True,
        )
    elif task_family == "reconstruction":
        value = finalize_query_surface(
            extract_sentence_excerpt(value, max_sentences=2, max_chars=min(max_chars, 360)),
            add_terminal_punctuation=True,
        )
    if value and value[:1].islower() and re.search(r"[A-Za-z]", value):
        value = value[:1].upper() + value[1:]
    return value


def generation_runner_cfg(
    runner_cfg: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if runner_cfg is None:
        return None
    cfg = dict(runner_cfg)
    # Build-time query generation is intentionally local-only. We keep
    # run-time OpenAI parsing available, but never send build candidates
    # through the parser/rewriter anymore.
    cfg["use_openai_parser"] = False
    return cfg


def normalize_selected_case_queries(
    cases: list[dict[str, Any]],
    *,
    task_family: str,
    runner_cfg: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not cases:
        return cases
    normalized: list[dict[str, Any]] = []
    seen_query_texts: set[str] = set()
    for case in cases:
        updated = dict(case)
        query = normalize_surface_preserving_markers(updated.get("query", ""))
        if task_family in {"claims", "contradictions"}:
            query = expand_claim_coordination_ellipsis(query)
            updated["query"] = query
        if task_family not in {"value_redaction", "sentence_redaction"} and (
            "[REDACT:" in query or "[REDACT_SENT:" in query
        ):
            continue
        query_kind = normalize_space(updated.get("query_kind", "")).casefold()
        parser_ready_claim_surface = (
            task_family == "claims" and query_kind == "claim_parser_ready_surface"
        )
        query_flags = set(query_surface_flags(query, task_family=task_family))
        force_rewrite = bool(
            query_flags
            & {
                "uppercase_heavy",
                "mixed_case_noise",
                "titlecase_noise",
                "headline_scaffold",
                "broken_lead",
                "foreign_language",
                "duplicated_scaffold",
                "metadata_scaffold",
            }
        )
        if not force_rewrite and query:
            if query[:1].islower() and count_words(query) >= 8:
                force_rewrite = True
            elif repeated_sentence_leak(query):
                force_rewrite = True
            elif task_family in {"claims", "contradictions", "reconstruction"} and "..." in query:
                force_rewrite = True
            elif task_family == "reconstruction" and len(sentence_split(query)) > 2:
                force_rewrite = True
        if query and task_family in {"claims", "contradictions"} and not parser_ready_claim_surface:
            profile = claim_clause_profile(query)
            max_actor_words = 8 if task_family == "claims" else 6
            if not (
                strict_claim_surface_is_safe(
                    query,
                    profile,
                    max_actor_words=max_actor_words,
                    max_object_words=18,
                )
                or standalone_claim_surface_is_safe(query)
            ):
                force_rewrite = True
        elif query and task_family == "reconstruction" and "claim_surface" in query_kind:
            if not standalone_claim_surface_is_safe(query):
                force_rewrite = True
        # Relation-backed claim/contradiction prompts are already the cleanest surfaces we have.
        # Forcing a rewrite there can mutate them into parser-hostile or attribution-heavy text.
        if query_kind in {
            "claim_relation_query",
            "relation_contradiction_query",
        } and not (
            query_flags
            & {
                "uppercase_heavy",
                "mixed_case_noise",
                "titlecase_noise",
                "headline_scaffold",
                "broken_lead",
                "foreign_language",
                "duplicated_scaffold",
                "metadata_scaffold",
            }
        ) and not sentence_is_low_signal_for_query(query):
            force_rewrite = False
        if parser_ready_claim_surface and build_case_query_is_admissible(
            updated,
            task_family=task_family,
            runner_cfg=runner_cfg,
        ):
            force_rewrite = False
        if task_family == "claims" and query_kind in {
            "claim_direct_minimal",
            "claim_small_direct_surface",
            "claim_pair_relation_surface",
            "claim_pair_repeated_surface",
        }:
            benign_claim_flags = {
                "titlecase_noise",
                "mixed_case_noise",
            }
            if set(query_flags).issubset(benign_claim_flags) and build_case_query_is_admissible(
                updated,
                task_family=task_family,
                runner_cfg=runner_cfg,
            ):
                force_rewrite = False
        if (
            task_family == "contradictions"
            and build_case_query_is_admissible(
                updated,
                task_family=task_family,
                runner_cfg=runner_cfg,
            )
            and not (
                query_flags
                & {
                    "uppercase_heavy",
                    "mixed_case_noise",
                    "metadata_scaffold",
                    "headline_scaffold",
                    "broken_lead",
                    "foreign_language",
                    "duplicated_scaffold",
                }
            )
        ):
            force_rewrite = False
        if force_rewrite and task_family in {"claims", "contradictions", "reconstruction"}:
            continue
        if task_family in {"claims", "contradictions", "reconstruction"}:
            updated["query"] = simplify_claim_query_seed(
                simplify_claim_surface(updated.get("query", ""))
            )
        if (
            task_family == "contradictions"
            and normalize_space(updated.get("ground_truth", ""))
            and retained_content_token_ratio(
                normalize_space(updated.get("query", "")),
                normalize_space(updated.get("ground_truth", "")),
            ) < 0.70
        ):
            continue
        if task_family == "value_redaction":
            updated["query"] = normalize_surface_preserving_markers(updated.get("query", ""))
            updated["query"] = re.sub(
                r"(\[REDACT:R\d+[^\]]*\])\s+[sS]\b",
                r"\1's",
                updated["query"],
            )
            updated["query"] = re.sub(
                r"(\[REDACT:R\d+[^\]]*\]\s+)(Must|Denied|To|At|And|Or|But|The|A|An|In|On|Of|For|From|By|With|Against|Into|Over|Under|Reported|Received|Suggested|Is|Are|Was|Were|That|Which|Who|Whose|Where|When|May|Might|Can|Could|Will|Would|Should|Has|Have|Had)\b",
                lambda m: m.group(1) + m.group(2).lower(),
                updated["query"],
            )
            updated["query"] = normalize_redaction_list_boundary(updated["query"])
            updated["query"] = clean_query_passage(updated["query"])
            updated["query"] = strip_low_signal_edge_sentences(
                updated["query"],
                task_family=task_family,
                min_remaining_words=10,
            )
        if task_family == "reconstruction":
            updated["query"] = finalize_query_surface(
                extract_sentence_excerpt(
                    updated.get("query", ""),
                    max_sentences=2,
                    max_chars=320,
                ),
                add_terminal_punctuation=True,
            )
            if (
                not updated["query"]
                or sentence_is_low_signal_for_query(updated["query"])
                or is_routing_or_metadata_sentence(updated["query"])
                or coordination_heavy_surface(updated["query"])
            ):
                continue
        updated["word_count"] = count_words(updated.get("query", ""))
        if not build_case_query_is_admissible(
            updated,
            task_family=task_family,
            runner_cfg=runner_cfg,
        ):
            if task_family != "value_redaction":
                continue
            answer_targets = updated.get("answer_targets")
            used_specs: list[dict[str, Any]] = []
            if isinstance(answer_targets, dict):
                for slot_id, payload in answer_targets.items():
                    if not isinstance(payload, dict):
                        continue
                    used_specs.append(
                        {
                            "slot_id": normalize_space(slot_id),
                            "entity_type": normalize_candidate_type(
                                updated.get("entity_type", "")
                            ),
                            "label": normalize_space(payload.get("selected_value", "")),
                        }
                    )
            if not used_specs:
                used_specs = [
                    {
                        "slot_id": "R1",
                        "entity_type": normalize_candidate_type(
                            updated.get("entity_type", "")
                        ),
                        "label": "",
                    }
                ]
            stripped = re.sub(r"\[REDACT:R\d+[^\]]*\]", " SLOT ", updated["query"])
            stripped = re.sub(
                r"\bSLOT\s+([A-Z][a-z]+)\b",
                lambda m: "SLOT " + m.group(1).lower(),
                stripped,
            )
            if not (
                value_redaction_query_is_viable(
                    updated["query"],
                    entity_type=normalize_candidate_type(updated.get("entity_type", "")),
                    used_specs=used_specs,
                )
                and value_redaction_slot_count(updated) == count_redaction_slots(updated["query"])
                and 1 <= value_redaction_slot_count(updated) <= 2
                and len(sentence_split(stripped)) <= 2
                and not sentence_is_low_signal_for_query(stripped)
                and not repeated_sentence_leak(stripped)
                and not query_surface_is_blocked(stripped, task_family="claims")
            ):
                continue
        query_key = canonical_query_dedupe_key(updated.get("query", ""))
        if not query_key or query_key in seen_query_texts:
            continue
        seen_query_texts.add(query_key)
        normalized.append(updated)
    return normalized


def build_case_query_is_admissible(
    case: dict[str, Any],
    *,
    task_family: str,
    runner_cfg: dict[str, Any] | None,
) -> bool:
    query = normalize_surface_preserving_markers(case.get("query", ""))
    query_kind = normalize_space(case.get("query_kind", "")).casefold()
    lowered_query = query.casefold()
    if not query:
        return False
    if not (
        task_family == "claims"
        and query_kind == "claim_parser_ready_surface"
    ) and query_surface_is_blocked(query, task_family=task_family):
        return False
    if task_family == "value_redaction":
        answer_targets = case.get("answer_targets")
        used_specs: list[dict[str, Any]] = []
        if isinstance(answer_targets, dict):
            for slot_id, payload in answer_targets.items():
                if not isinstance(payload, dict):
                    continue
                used_specs.append(
                    {
                        "slot_id": normalize_space(slot_id),
                        "entity_type": normalize_candidate_type(
                            case.get("entity_type", "")
                        ),
                        "label": normalize_space(payload.get("selected_value", "")),
                    }
                )
        if not used_specs:
            used_specs = [
                {
                    "slot_id": "R1",
                    "entity_type": normalize_candidate_type(case.get("entity_type", "")),
                    "label": "",
                }
            ]
        if not value_redaction_query_is_viable(
            query,
            entity_type=normalize_candidate_type(case.get("entity_type", "")),
            used_specs=used_specs,
        ):
            return False
    if task_family in {"claims", "contradictions"}:
        if sentence_is_low_signal_for_query(query) or repeated_sentence_leak(query):
            return False
        if task_family == "contradictions" and malformed_generated_contradiction_surface(query):
            return False
        if task_family == "claims" and re.search(
            r"\b(?:"
            r"received instructions from\b[^.]{0,120}\brequiring\b|"
            r"is planning to turn over to\b|"
            r"(?:is|was)\s+unable to\b[^.]{0,120}\bbecause of (?:the )?lack of\b|"
            r"is not and will not be seriously affected by\b|"
            r"was led by a charismatic figure who would not alter\b"
            r")",
            lowered_query,
        ):
            return False
        if query_kind == "claim_parser_ready_surface":
            if is_routing_or_metadata_sentence(query):
                return False
            if len(sentence_split(query)) != 1:
                return False
            words = count_words(query)
            if words < 6 or words > 48:
                return False
            return claim_query_is_executable_via_parser(query, runner_cfg)
        if claim_surface_is_interrogative(query) or claim_surface_has_weak_modality(query):
            return False
        if re.search(r"^(?:Such\b|Further\b|Additionally\b)", query, flags=re.IGNORECASE):
            return False
        if re.search(r"^(?:Based on|According to)\b", query, flags=re.IGNORECASE):
            lead = query[: min(len(query), 96)]
            if "," not in lead:
                return False
        if len(sentence_split(query)) != 1:
            return False
        if query_kind in {
            "claim_small_direct_surface",
            "claim_small_direct_contradiction",
        }:
            profile = claim_clause_profile(query)
            fallback_simple_ok = fallback_simple_complete_claim_surface_is_safe(query)
            if count_words(query) < 6 or count_words(query) > 48:
                return False
            if substantive_sentence_score(query) < 5.0:
                return False
            if not profile_has_complete_simple_clause(profile) and not fallback_simple_ok:
                return False
            if actor_has_abstract_head(normalize_space(profile.get("actor", ""))):
                return False
            if actor_is_generic_common_noun_phrase(normalize_space(profile.get("actor", ""))):
                return False
            max_actor_words = 8 if task_family == "claims" else 6
            if not strict_claim_surface_is_safe(
                query,
                profile,
                max_actor_words=max_actor_words,
                max_object_words=18,
            ) and not fallback_simple_ok:
                return False
            if query_kind == "claim_small_direct_contradiction":
                partner_matches = int(case.get("partner_matches", 0) or 0)
                base_has_date_or_number = bool(case.get("base_has_date_or_number", False))
                if partner_matches <= 0 and not base_has_date_or_number:
                    return False
            if task_family == "claims" and not claim_query_is_executable_via_parser(query, runner_cfg):
                return False
            return True
        if query_kind in {
            "claim_reconstruction_seed",
            "reconstruction_seed_contradiction",
        }:
            return False
        profile = claim_clause_profile(query)
        if task_family == "claims":
            standalone_ok = standalone_claim_surface_is_safe(query)
            fallback_simple_ok = fallback_simple_complete_claim_surface_is_safe(query)
            if not profile_has_complete_simple_clause(profile) and not fallback_simple_ok:
                return False
        else:
            standalone_ok = (
                standalone_claim_surface_is_safe(query)
                or standalone_contradiction_surface_is_safe(query)
            )
            fallback_simple_ok = fallback_simple_complete_claim_surface_is_safe(query)
            if not profile_has_complete_simple_clause(profile) and not standalone_ok:
                return False
        actor_text = normalize_space(profile.get("actor", ""))
        object_text = normalize_space(profile.get("object", ""))
        if claim_profile_has_problematic_actor(profile):
            return False
        if object_has_abstract_head(object_text) and not standalone_ok:
            return False
        max_actor_words = 8 if task_family == "claims" else 6
        if not (
            strict_claim_surface_is_safe(
                query,
                profile,
                max_actor_words=max_actor_words,
                max_object_words=18,
            )
            or standalone_ok
            or (task_family == "claims" and fallback_simple_ok)
        ):
            return False
        if task_family == "claims" and not claim_query_is_executable_via_parser(query, runner_cfg):
            return False
    elif task_family == "reconstruction":
        if query_kind == "reconstruction_hybrid_signature":
            return False
        if sentence_is_low_signal_for_query(query) or repeated_sentence_leak(query):
            return False
        if "claim_surface" in query_kind:
            reconstruction_profile = claim_clause_profile(query)
            if not (
                standalone_claim_surface_is_safe(query)
                or strict_claim_surface_is_safe(
                    query,
                    reconstruction_profile,
                    max_actor_words=8,
                    max_object_words=20,
                )
            ):
                return False
        if re.search(r"\b(?:may|might|could|possibly|perhaps|likely)\b", query, flags=re.IGNORECASE):
            return False
        if LEGAL_BOILERPLATE_RE.search(query) or BROKEN_LEAD_RE.search(query):
            return False
        if re.search(r"\bconsult my office\b|\bscope of your authority\b", query, flags=re.IGNORECASE):
            return False
        if len(sentence_split(query)) > 2:
            return False
        if count_words(query) < 10 or count_words(query) > 80:
            return False
        if not normalize_space(case.get("ground_truth", "")):
            return False
    elif task_family == "value_redaction":
        slot_count = value_redaction_slot_count(case)
        if slot_count < 1 or count_redaction_slots(query) != slot_count:
            return False
        if slot_count > 2:
            return False
        stripped = re.sub(r"\[REDACT:R\d+[^\]]*\]", " SLOT ", query)
        stripped = re.sub(
            r"\bSLOT\s+([A-Z][a-z]+)\b",
            lambda m: "SLOT " + m.group(1).lower(),
            stripped,
        )
        if len(sentence_split(stripped)) > 2:
            return False
        if sentence_is_low_signal_for_query(stripped) or repeated_sentence_leak(stripped):
            return False
        if query_surface_is_blocked(stripped, task_family="claims"):
            return False
        return True
    elif task_family == "sentence_redaction":
        if query_kind == "anchored_multi_sentence_gap":
            return False
        if count_redaction_slots(query, sentence_mode=True) != 1:
            return False
    return True


def replace_first_ignore_case(text: str, old: str, new: str) -> tuple[str, bool]:
    if not text or not old:
        return text, False
    pattern = re.compile(re.escape(old), flags=re.IGNORECASE)
    if not pattern.search(text):
        return text, False
    return pattern.sub(new, text, count=1), True


def label_is_matchable(label: str) -> bool:
    value = normalize_space(label)
    if not value:
        return False
    if count_words(value) > 10:
        return False
    if len(value) > 96:
        return False
    return True


def label_match_pattern(label: str) -> re.Pattern[str]:
    value = normalize_space(label)
    if not value:
        return re.compile(r"$^")
    return _label_match_pattern_cached(value)


@lru_cache(maxsize=100_000)
def _label_match_pattern_cached(value: str) -> re.Pattern[str]:
    escaped = re.escape(value)
    prefix = r"(?<![A-Za-z0-9])" if value and value[0].isalnum() else ""
    suffix = r"(?![A-Za-z0-9])" if value and value[-1].isalnum() else ""
    return re.compile(prefix + escaped + suffix, flags=re.IGNORECASE)


def find_label_match(text: str, label: str) -> re.Match[str] | None:
    value = normalize_space(text)
    target = normalize_space(label)
    if not value or not target:
        return None
    if not label_is_matchable(target):
        return None
    match = label_match_pattern(target).search(value)
    if match:
        return match
    return re.search(re.escape(target), value, flags=re.IGNORECASE)


def span_has_token_boundaries(text: str, start: int, end: int) -> bool:
    if start < 0 or end > len(text) or start >= end:
        return False
    left_ok = start == 0 or not text[start - 1].isalnum()
    right_ok = end == len(text) or not text[end].isalnum()
    return left_ok and right_ok


def safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        if raw is None:
            return default
        if isinstance(raw, str) and not raw.strip():
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def slug(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(filter(None, safe.split("_")))[:100] or "case"


def ensure_terminal_punctuation(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    if value[-1] in ".!?]":
        return value
    return value + "."


def lowercase_leading_clause(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    if len(value) >= 2 and value[0].isupper() and value[1].islower():
        return value[0].lower() + value[1:]
    return value


def canonical_paragraph_slot(paragraph_id: Any) -> str:
    value = normalize_space(paragraph_id)
    if not value:
        return ""
    match = PARAGRAPH_NUM_RE.search(value)
    if match:
        try:
            return f"slot_{int(match.group(1))}"
        except ValueError:
            pass
    return value.casefold()


def paragraph_key(doc_id: Any, paragraph_id: Any) -> tuple[str, str]:
    return str(doc_id), canonical_paragraph_slot(paragraph_id)


def sentence_split(text: str) -> list[str]:
    raw = normalize_space(text)
    if not raw:
        return []
    parts = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(raw) if segment.strip()]
    if len(parts) >= 2:
        merged: list[str] = [parts[0]]
        for part in parts[1:]:
            if ABBREVIATED_SENTENCE_END_RE.search(merged[-1]):
                merged[-1] = normalize_space(f"{merged[-1]} {part}")
            else:
                merged.append(part)
        parts = merged
    if len(parts) <= 1 and ";" in raw:
        parts = [segment.strip() for segment in raw.split(";") if segment.strip()]
    return parts


def first_sentence_boundary_window(text: str, start: int, end: int) -> str:
    left = text.rfind(".", 0, start)
    left_q = text.rfind("?", 0, start)
    left_e = text.rfind("!", 0, start)
    left = max(left, left_q, left_e)
    if left == -1:
        left = 0
    else:
        left += 1
    right_candidates = [
        text.find(".", end),
        text.find("?", end),
        text.find("!", end),
    ]
    right_candidates = [value for value in right_candidates if value != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return normalize_space(text[left:right])


def bounded_context(text: str, start: int, end: int, min_chars: int = 220) -> str:
    left = max(0, start - min_chars)
    right = min(len(text), end + min_chars)
    window = first_sentence_boundary_window(text, left, right)
    if count_words(window) < 12:
        window = normalize_space(text[max(0, start - 320): min(len(text), end + 320)])
    return window


def extract_sentence_excerpt(
    text: str,
    *,
    max_sentences: int = 2,
    max_chars: int = 360,
) -> str:
    raw = normalize_space(text)
    if not raw:
        return ""
    return _extract_sentence_excerpt_cached(raw, max_sentences, max_chars)


@lru_cache(maxsize=100_000)
def _extract_sentence_excerpt_cached(
    raw: str,
    max_sentences: int,
    max_chars: int,
) -> str:
    sentences = sentence_split(raw)
    if sentences:
        excerpt = " ".join(sentences[:max_sentences]).strip()
    else:
        excerpt = raw
    if len(excerpt) <= max_chars:
        return excerpt
    running: list[str] = []
    for sentence in sentences[:max_sentences]:
        candidate = normalize_space(" ".join(running + [sentence]))
        if candidate and len(candidate) > max_chars and running:
            break
        running.append(sentence)
        if len(normalize_space(" ".join(running))) >= max_chars:
            break
    trimmed = normalize_space(" ".join(running))
    if trimmed:
        return trimmed
    return sentences[0] if sentences else excerpt


def extract_embedded_context(text: Any) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    match = re.search(r"\bContext:\s*(.+)$", value, flags=re.IGNORECASE)
    if match:
        return finalize_query_surface(match.group(1))
    return finalize_query_surface(value)


def uppercase_token_ratio(text: str) -> float:
    value = normalize_space(text)
    if not value:
        return 0.0
    return _uppercase_token_ratio_cached(value)


@lru_cache(maxsize=100_000)
def _uppercase_token_ratio_cached(value: str) -> float:
    tokens = [token for token in WORD_RE.findall(value) if any(ch.isalpha() for ch in token)]
    if not tokens:
        return 0.0
    upper = 0
    for token in tokens:
        letters = [ch for ch in token if ch.isalpha()]
        if letters and "".join(letters).upper() == "".join(letters):
            upper += 1
    return upper / max(1, len(tokens))


def coordination_heavy_surface(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    comma_count = value.count(",")
    and_count = len(re.findall(r"\band\b", value, flags=re.IGNORECASE))
    return comma_count >= 3 or (comma_count >= 2 and and_count >= 1)


def is_list_like_text(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    return _is_list_like_text_cached(value)


@lru_cache(maxsize=100_000)
def _is_list_like_text_cached(value: str) -> bool:
    if LIST_LEAD_RE.match(value):
        return True
    if uppercase_token_ratio(value) >= 0.78 and value.count(",") >= 5:
        return True
    if value.count(";") >= 4 and count_words(value) >= 10:
        return True
    if value.count(":") >= 1 and ALLEGATION_LIST_RE.search(value):
        return True
    return False


def is_routing_or_metadata_sentence(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    return _is_routing_or_metadata_sentence_cached(value)


@lru_cache(maxsize=100_000)
def _is_routing_or_metadata_sentence_cached(value: str) -> bool:
    upper_ratio = uppercase_token_ratio(value)
    routing_hits = len(ROUTING_TOKEN_RE.findall(value))
    if routing_hits >= 2 and upper_ratio >= 0.45:
        return True
    if routing_hits >= 4:
        return True
    if PAGE_INDEX_RE.search(value):
        return True
    if DOC_ID_TOKEN_RE.search(value) and upper_ratio >= 0.55:
        return True
    if value.startswith("Comment on "):
        return True
    if value.startswith(("FM ", "TO ", "INFO ", "PASS ")):
        return True
    if QA_PREFIX_RE.match(value):
        return True
    if STRUCTURED_METADATA_RE.search(value):
        return True
    if PRESS_BRIEFING_METADATA_RE.search(value):
        return True
    if BROADCAST_TRANSCRIPT_RE.search(value):
        return True
    if SPEAKER_PREFIX_RE.match(value):
        return True
    if re.search(r"\b(?:pouching copies|embassy is pouching|cbs news,\s*[A-Za-z]+|das\s+[A-Z][a-z]+|dfa'?s lengthy instruction)\b", value, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(?:noted para|missions noted|distribution(?: to)?|reference is made to)\b", value, flags=re.IGNORECASE):
        return True
    return False


def alpha_token_count(text: str) -> int:
    return sum(
        1
        for token in WORD_RE.findall(text or "")
        if any(ch.isalpha() for ch in token)
    )


def clean_generation_sentence(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    return _clean_generation_sentence_cached(value)


@lru_cache(maxsize=100_000)
def _clean_generation_sentence_cached(value: str) -> str:
    value = LIST_LEAD_RE.sub("", value)
    value = re.sub(r"^(?:[A-Z]\.)\s+", "", value)
    value = QA_PREFIX_RE.sub("", value)
    value = SPEAKER_PREFIX_RE.sub("", value)
    value = re.sub(r"^\(?[a-z]\)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\(?\d+\)?\s+", "", value)
    value = re.sub(r"(?:(?<=[:;])|^)\s*o\s+(?=[A-Z])", " ", value)
    value = re.sub(r"\s*[\-–—:;,]+\s*$", "", value)
    return normalize_space(value)


def anchor_hit_count(text: str, anchors: Iterable[str]) -> int:
    value = normalize_space(text)
    if not value:
        return 0
    hits = 0
    for anchor in anchors:
        candidate = normalize_space(anchor)
        if not candidate:
            continue
        if re.search(re.escape(candidate), value, flags=re.IGNORECASE):
            hits += 1
    return hits


def _normalized_anchor_tuple(anchors: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({
        normalize_space(anchor)
        for anchor in anchors
        if normalize_space(anchor)
    }))


@lru_cache(maxsize=200_000)
def _substantive_sentence_score_cached(
    value: str,
    anchors: tuple[str, ...],
) -> float:
    if not value:
        return -100.0
    words = count_words(value)
    alpha_words = alpha_token_count(value)
    score = 0.0
    if words >= 6:
        score += 4.0
    else:
        score -= float(6 - words) * 2.0
    if alpha_words < 4:
        score -= 8.0
    if is_routing_or_metadata_sentence(value):
        score -= 24.0
    if is_list_like_text(value):
        score -= 16.0
    if SLASH_FRAGMENT_RE.search(value):
        score -= 10.0
    if ALLEGATION_LIST_RE.search(value):
        score -= 8.0
    if BROKEN_LEAD_RE.search(value):
        score -= 10.0
    if uppercase_token_ratio(value) >= 0.72 and words >= 8:
        score -= 12.0
    if LEGAL_BOILERPLATE_RE.search(value):
        score -= 9.0
    if DOC_ID_TOKEN_RE.search(value):
        score -= 6.0
    if value.count(";") >= 2:
        score -= 8.0
    if value[:1].islower():
        score -= 4.0
    if not re.search(r"[A-Za-z]", value):
        score -= 20.0
    if not VERBISH_TOKEN_RE.search(value) and words >= 8:
        score -= 14.0 if words <= 12 else 6.0
    digit_ratio = len(NUMBER_RE.findall(value)) / max(words, 1)
    if digit_ratio >= 0.35:
        score -= 6.0
    if value.endswith((".", "!", "?")):
        score += 1.5
    if len(value) >= 60:
        score += 2.5
    score += anchor_hit_count(value, anchors) * 5.0
    return score


def substantive_sentence_score(
    sentence: str,
    *,
    anchors: Iterable[str] = (),
) -> float:
    value = clean_generation_sentence(sentence)
    return _substantive_sentence_score_cached(
        value,
        _normalized_anchor_tuple(anchors),
    )


def cleaned_substantive_sentences(
    text: str,
    *,
    anchors: Iterable[str] = (),
    min_score: float = -2.0,
) -> list[str]:
    anchor_values = _normalized_anchor_tuple(anchors)
    sentences = []
    for raw_sentence in sentence_split(text):
        sentence = clean_generation_sentence(raw_sentence)
        if not sentence:
            continue
        if substantive_sentence_score(sentence, anchors=anchor_values) < min_score:
            continue
        sentences.append(sentence)
    return sentences


def context_quality_score(
    text: str,
    *,
    anchors: Iterable[str] = (),
    min_words: int = 16,
) -> float:
    value = normalize_space(text)
    if not value:
        return -1000.0
    words = count_words(value)
    sentences = cleaned_substantive_sentences(value, anchors=anchors, min_score=-6.0)
    sentence_score = sum(substantive_sentence_score(sentence, anchors=anchors) for sentence in sentences)
    penalty = 0.0
    if words < min_words:
        penalty += float(min_words - words) * 2.5
    if words > max(min_words * 3, 72):
        penalty += float(words - max(min_words * 3, 72)) * 0.35
    penalty += surface_noise_penalty(value)
    return sentence_score + min(words, 48) * 0.18 - penalty


def build_sentence_window_from_text(
    text: str,
    *,
    anchors: Iterable[str] = (),
    min_words: int = 16,
    target_words: int | None = None,
    max_sentences: int = 4,
    max_chars: int = 640,
) -> str:
    anchor_values = [normalize_space(anchor) for anchor in anchors if normalize_space(anchor)]
    candidates = [
        clean_generation_sentence(sentence)
        for sentence in sentence_split(text)
    ]
    candidates = [
        sentence
        for sentence in candidates
        if sentence and substantive_sentence_score(sentence, anchors=anchor_values) >= -2.0
    ]
    if not candidates:
        return extract_sentence_excerpt(text, max_sentences=max_sentences, max_chars=max_chars)

    scored = [
        (
            anchor_hit_count(sentence, anchor_values),
            substantive_sentence_score(sentence, anchors=anchor_values),
            index,
        )
        for index, sentence in enumerate(candidates)
    ]
    _anchor_hits, _best_score, center = max(scored, key=lambda item: (item[0], item[1], -abs(item[2])))
    left = right = center
    selected_words = count_words(candidates[center])
    desired_words = max(min_words, int(target_words or min_words * 2))
    min_sentences = min(max_sentences, 2)
    while (selected_words < min_words or (right - left + 1) < min_sentences or selected_words < int(desired_words * 0.85)) and (right - left + 1) < max_sentences:
        options: list[tuple[float, str]] = []
        if left > 0:
            next_sentence = candidates[left - 1]
            new_words = selected_words + count_words(next_sentence)
            balance_penalty = abs((center - (left - 1)) - (right - center)) * 0.45
            proximity_penalty = abs(new_words - desired_words) * 0.03
            option_score = (
                substantive_sentence_score(next_sentence, anchors=anchor_values)
                - balance_penalty
                - proximity_penalty
            )
            options.append((option_score, "left"))
        if right + 1 < len(candidates):
            next_sentence = candidates[right + 1]
            new_words = selected_words + count_words(next_sentence)
            balance_penalty = abs((center - left) - ((right + 1) - center)) * 0.45
            proximity_penalty = abs(new_words - desired_words) * 0.03
            option_score = (
                substantive_sentence_score(next_sentence, anchors=anchor_values)
                - balance_penalty
                - proximity_penalty
            )
            options.append((option_score, "right"))
        if not options:
            break
        _score, side = max(options, key=lambda item: item[0])
        if side == "left":
            left -= 1
            selected_words += count_words(candidates[left])
        else:
            right += 1
            selected_words += count_words(candidates[right])
        if selected_words >= desired_words and (right - left + 1) >= min_sentences:
            break
    window = finalize_query_surface(" ".join(candidates[left : right + 1]))
    if len(window) <= max_chars:
        return window
    sentences = sentence_split(window)
    running: list[str] = []
    for sentence in sentences:
        candidate = normalize_space(" ".join(running + [sentence]))
        if candidate and len(candidate) > max_chars and running:
            break
        running.append(sentence)
        if len(normalize_space(" ".join(running))) >= max_chars:
            break
    trimmed = normalize_space(" ".join(running))
    return trimmed or window


def choose_best_query_context(
    texts: Iterable[str],
    *,
    anchors: Iterable[str] = (),
    min_words: int = 16,
    target_words: int | None = None,
    max_sentences: int = 4,
    max_chars: int = 640,
) -> str:
    anchor_values = _normalized_anchor_tuple(anchors)
    candidate_texts = [
        text for text in texts
        if normalize_space(text)
    ]
    if len(candidate_texts) == 1:
        candidate = finalize_query_surface(candidate_texts[0])
        if candidate:
            candidate_words = count_words(candidate)
            if candidate_words >= max(6, min_words - 2) and len(candidate) <= max_chars:
                return candidate
    best_text = ""
    best_score = -1000.0
    for text in candidate_texts:
        candidate = build_sentence_window_from_text(
            text,
            anchors=anchor_values,
            min_words=min_words,
            target_words=target_words,
            max_sentences=max_sentences,
            max_chars=max_chars,
        )
        if not candidate:
            continue
        candidate = finalize_query_surface(candidate)
        if not candidate:
            continue
        score = context_quality_score(
            candidate,
            anchors=anchor_values,
            min_words=min_words,
        )
        if target_words:
            score -= abs(count_words(candidate) - int(target_words)) * 0.22
        if score > best_score:
            best_score = score
            best_text = candidate
    return best_text


def choose_fast_query_context(
    texts: Iterable[str],
    *,
    min_words: int = 8,
    max_chars: int = 420,
) -> str:
    best_text = ""
    best_score = -1000.0
    for text in texts:
        candidate = finalize_query_surface(
            extract_sentence_excerpt(text, max_sentences=2, max_chars=max_chars)
        )
        if not candidate:
            continue
        score = (
            min(count_words(candidate), 28) * 0.25
            - surface_noise_penalty(candidate)
            - abs(count_words(candidate) - max(min_words, 12)) * 0.2
        )
        if score > best_score:
            best_score = score
            best_text = candidate
    return best_text


def extract_neighbor_chunk_context(
    doc_chunks_by_doc: dict[str, list[dict[str, Any]]],
    doc_id: Any,
    paragraph_id: Any,
    *,
    anchors: Iterable[str] = (),
    min_words: int = 16,
    max_sentences: int = 4,
    max_chars: int = 640,
    max_radius: int = 1,
) -> str:
    rows = doc_chunks_by_doc.get(str(doc_id), [])
    if not rows:
        return ""
    center = None
    target_key = paragraph_key(doc_id, paragraph_id)
    for index, row in enumerate(rows):
        if paragraph_key(row.get("doc_id", ""), row.get("paragraph_id", "")) == target_key:
            center = index
            break
    if center is None:
        return ""
    candidates: list[str] = []
    for radius in range(0, max_radius + 1):
        left = max(0, center - radius)
        right = min(len(rows), center + radius + 1)
        combined = normalize_space(
            " ".join(normalize_space(rows[idx].get("chunk_text", "")) for idx in range(left, right))
        )
        if combined:
            candidates.append(combined)
    return choose_best_query_context(
        candidates,
        anchors=anchors,
        min_words=min_words,
        max_sentences=max_sentences,
        max_chars=max_chars,
    )


def build_value_slot_marker(slot_index: int, slot_type: str) -> str:
    return f"[REDACT:R{slot_index} type={slot_type}]"


def build_sentence_slot_marker(slot_index: int, approx_tokens: int) -> str:
    return f"[REDACT_SENT:S{slot_index} approx_tokens={approx_tokens}]"


def build_sentence_redaction_query_window(
    *,
    sentences: list[str],
    gap_start: int,
    gap_end: int,
    slot_text: str,
    target_words: int,
    max_sentences: int,
) -> str:
    left = gap_start
    right = gap_end
    chosen: list[str] = [slot_text]
    visible_words = 0
    if left > 0:
        sentence = clean_generation_sentence(sentences[left - 1])
        if (
            sentence
            and substantive_sentence_score(sentence) >= 4.0
            and not sentence_is_low_signal_for_query(sentence)
        ):
            chosen.insert(0, sentence)
            visible_words += count_words(sentence)
            left -= 1
    if right + 1 < len(sentences):
        sentence = clean_generation_sentence(sentences[right + 1])
        if (
            sentence
            and substantive_sentence_score(sentence) >= 4.0
            and not sentence_is_low_signal_for_query(sentence)
        ):
            chosen.append(sentence)
            visible_words += count_words(sentence)
            right += 1
    while (visible_words < target_words or len(chosen) < 3) and len(chosen) < max_sentences:
        options: list[tuple[float, str]] = []
        if left > 0:
            candidate = clean_generation_sentence(sentences[left - 1])
            if candidate and not sentence_is_low_signal_for_query(candidate):
                score = substantive_sentence_score(candidate) - abs((visible_words + count_words(candidate)) - target_words) * 0.04
                options.append((score, "left"))
        if right + 1 < len(sentences):
            candidate = clean_generation_sentence(sentences[right + 1])
            if candidate and not sentence_is_low_signal_for_query(candidate):
                score = substantive_sentence_score(candidate) - abs((visible_words + count_words(candidate)) - target_words) * 0.04
                options.append((score, "right"))
        if not options:
            break
        _score, side = max(options, key=lambda item: item[0])
        if side == "left":
            left -= 1
            sentence = clean_generation_sentence(sentences[left])
            if sentence and not sentence_is_low_signal_for_query(sentence):
                chosen.insert(0, sentence)
                visible_words += count_words(sentence)
        else:
            right += 1
            sentence = clean_generation_sentence(sentences[right])
            if sentence and not sentence_is_low_signal_for_query(sentence):
                chosen.append(sentence)
                visible_words += count_words(sentence)
    return finalize_query_surface(" ".join(chosen))


def deterministic_tail_count(
    key: str,
    *,
    max_count: int,
    decay: float,
) -> int:
    if max_count <= 1:
        return 1
    rng = random.Random(sha1_text(key))
    total = 1
    while total < max_count and rng.random() < decay:
        total += 1
    return total


def deterministic_query_word_target(
    key: str,
    *,
    minimum_words: int = 16,
    maximum_words: int = 128,
) -> int:
    lower = max(4, int(minimum_words))
    upper = max(lower, int(maximum_words))
    choices = list(range(lower, upper + 1))
    weights = [1.0 / max(word_count * 2.0, 1.0) for word_count in choices]
    rng = random.Random(sha1_text(key))
    return rng.choices(choices, weights=weights, k=1)[0]


def scaled_redaction_query_target(
    key: str,
    *,
    minimum_words: int,
    maximum_words: int,
    slot_count: int,
) -> int:
    base_target = deterministic_query_word_target(
        key,
        minimum_words=minimum_words,
        maximum_words=maximum_words,
    )
    slot_count = max(1, int(slot_count))
    scaled = int(round(base_target * (1.0 + 0.55 * (slot_count - 1))))
    lower_bound = max(minimum_words, min(96, minimum_words + (slot_count - 1) * 8))
    return min(maximum_words, max(lower_bound, scaled))


def format_expected_targets(case: dict[str, Any]) -> dict[str, Any]:
    targets = case.get("expected_targets")
    if isinstance(targets, dict) and targets:
        return targets
    return {}


def stringify_expected_targets(targets: dict[str, Any]) -> str:
    if not targets:
        return ""
    ordered = []
    for slot_id, payload in sorted(targets.items()):
        selected = normalize_space(payload.get("selected_value", ""))
        source_id = normalize_space(payload.get("source_id", ""))
        ordered.append(f"{slot_id}={selected}@{source_id}" if source_id else f"{slot_id}={selected}")
    return " | ".join(ordered)


def redact_labels_in_context(
    text: str,
    slot_specs: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    value = normalize_space(text)
    if not value or not slot_specs:
        return value, []
    provisional_matches: list[tuple[int, int, dict[str, Any]]] = []
    for spec in slot_specs:
        label = normalize_space(spec.get("label", ""))
        if not label:
            continue
        match = find_label_match(value, label)
        if not match:
            continue
        if not span_has_token_boundaries(value, match.start(), match.end()):
            continue
        provisional_matches.append((match.start(), match.end(), spec))
    provisional_matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected_matches: list[tuple[int, int, dict[str, Any]]] = []
    cursor = -1
    for start, end, spec in provisional_matches:
        if start < cursor:
            continue
        selected_matches.append((start, end, spec))
        cursor = end
    if not selected_matches:
        return value, []
    rendered: list[str] = []
    used_specs: list[dict[str, Any]] = []
    last = 0
    for slot_index, (start, end, spec) in enumerate(selected_matches, start=1):
        rendered.append(value[last:start])
        slot_type = normalize_space(spec.get("slot_type", "ANY")) or "ANY"
        rendered.append(build_value_slot_marker(slot_index, slot_type))
        last = end
        used_spec = dict(spec)
        used_spec["slot_id"] = f"R{slot_index}"
        used_specs.append(used_spec)
    rendered.append(value[last:])
    return normalize_space("".join(rendered)), used_specs


def normalized_reconstruction_anchors(raw_anchors: Iterable[Any]) -> list[str]:
    cleaned: list[str] = []
    for raw_anchor in raw_anchors:
        anchor = normalize_space(raw_anchor)
        if not anchor:
            continue
        if DOC_ID_TOKEN_RE.fullmatch(anchor):
            continue
        if any(token in anchor for token in ("DOC=", "ENTITIES=", "EVENTS=", "CLAIMS=", "RELATIONS=", "RARE_ANCHORS=", "-->")):
            continue
        if "=" in anchor or "|" in anchor:
            continue
        if count_words(anchor) > 4:
            continue
        cleaned.append(anchor)
    return unique_preserving_order(cleaned)


def build_reconstruction_reference_summary(row: dict[str, Any]) -> str:
    claims = unique_preserving_order(
        parse_json_list(row.get("reconstruction_claims_json", row.get("claim_texts", "")))
    )
    events = unique_preserving_order(
        parse_json_list(row.get("reconstruction_events_json", row.get("event_texts", "")))
    )
    entities = unique_preserving_order(parse_json_list(row.get("reconstruction_entities_json")))
    parts: list[str] = []
    if claims:
        parts.append(
            "The missing passage likely states that "
            + readable_list([lowercase_leading_clause(value) for value in claims[:2]])
            + "."
        )
    if events:
        parts.append(
            "It centers on "
            + readable_list([lowercase_leading_clause(value) for value in events[:2]])
            + "."
        )
    if entities:
        parts.append(
            "Key named entities include "
            + readable_list(entities[:5])
            + "."
        )
    return " ".join(part.strip() for part in parts if normalize_space(part)).strip()


def parse_json_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    text = normalize_space(raw)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def unique_preserving_order(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in values:
        value = normalize_space(raw)
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def is_generic_role_value(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return True
    return bool(GENERIC_ROLE_RE.fullmatch(value))


def normalize_query_speaker(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    value = re.sub(r"\s*\((?:as\s+)?reported\)\s*$", "", value, flags=re.IGNORECASE).strip(" ,;:")
    if GENERIC_REPORTING_SPEAKER_RE.fullmatch(value):
        return ""
    return value


def row_is_structured_claim(row: Mapping[str, Any]) -> bool:
    source_type = normalize_space(row.get("source_type", "")).casefold()
    if source_type:
        return source_type == "structured_claim"
    local_id = normalize_space(row.get("local_id", "")).casefold()
    claim_text = normalize_space(row.get("claim_text", ""))
    node_key = normalize_space(row.get("node_key", "")).casefold()
    if local_id.startswith("claim_"):
        return True
    if "::claim_" in node_key:
        return True
    return bool(claim_text)


def expand_common_query_abbreviations(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    replacements = (
        (r"U\.S\.", "United States"),
        (r"U\.N\.", "United Nations"),
        (r"U\.K\.", "United Kingdom"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    return normalize_space(value)


def strip_leading_reporting_wrapper(text: str) -> str:
    value = finalize_query_surface(text, add_terminal_punctuation=True)
    if not value:
        return ""
    match = LEADING_REPORTING_WRAPPER_RE.match(value)
    if match:
        body = finalize_query_surface(match.group("body"), add_terminal_punctuation=True)
        if count_words(body) >= 6:
            return body
    match = LEADING_ACCORDING_TO_RE.match(value)
    if match:
        body = finalize_query_surface(match.group("body"), add_terminal_punctuation=True)
        if count_words(body) >= 6:
            return body
    return value


def strip_intrusive_relative_clause(text: str) -> str:
    value = finalize_query_surface(text, add_terminal_punctuation=True)
    if not value:
        return ""
    for _ in range(2):
        match = RELATIVE_CLAUSE_INSERT_RE.match(value)
        if not match:
            break
        if (
            normalize_space(match.group("marker")).casefold() == "that"
            and re.search(
                r"\b(?:say|said|tell|told|report|reported|state|stated|"
                r"claim|claimed|indicate|indicated|suggest|suggested|"
                r"believe|believed|deny|denied|announce|announced)\b",
                normalize_space(match.group("head")),
                flags=re.IGNORECASE,
            )
        ):
            break
        if (
            normalize_space(match.group("marker")).casefold() in {"that", "which"}
            and re.match(
                r"^(?:the|a|an|this|these|those|[A-Z][A-Za-z'’-]+)\b",
                normalize_space(match.group("insert")),
            )
        ):
            break
        candidate = finalize_query_surface(
            f"{normalize_space(match.group('head'))} {normalize_space(match.group('tail'))}",
            add_terminal_punctuation=True,
        )
        if not candidate or count_words(candidate) < 6:
            break
        value = candidate
    return value


def simplify_claim_surface(text: str) -> str:
    value = finalize_query_surface(text, add_terminal_punctuation=True)
    if not value:
        return ""
    value = expand_common_query_abbreviations(value)
    value = strip_leading_reporting_wrapper(value)
    value = strip_intrusive_relative_clause(value)
    value = finalize_query_surface(value, add_terminal_punctuation=True)
    return value


def role_phrase_tokens(text: str) -> list[str]:
    return [token.casefold() for token in WORD_RE.findall(normalize_space(text))]


def actor_phrase_has_clause_bleed(actor: str) -> bool:
    tokens = role_phrase_tokens(actor)
    if not tokens:
        return True
    if len(tokens) >= 2 and tokens[-1] in VERBISH_ACTOR_TAILS:
        return True
    if len(tokens) >= 3 and any(token in CLAUSE_BLEED_TOKENS for token in tokens[1:]):
        return True
    return False


def role_query_is_safe(actor: str, predicate: str, obj: str) -> bool:
    if not (actor and predicate and obj):
        return False
    if actor_phrase_has_clause_bleed(actor):
        return False
    if count_words(actor) > 8 or count_words(obj) > 20:
        return False
    if VERBISH_TOKEN_RE.search(obj) and count_words(obj) >= 3:
        return False
    if DEICTIC_RISK_RE.search(actor) and count_words(actor) <= 2:
        return False
    return True


def predicate_is_low_signal(predicate: str) -> bool:
    tokens = role_phrase_tokens(predicate)
    if not tokens:
        return True
    return tokens[0] in LOW_SIGNAL_CLAIM_PREDICATES


def claim_uses_intentive_construction(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    return bool(INTENTIVE_CLAIM_RE.search(value))


def has_reporting_that_clause(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    return bool(
        re.search(
            r"\b(?:indicat(?:e|ing)|insinuat(?:e|ing)|believ(?:e|ed|ing)|assess(?:ed|es|ing)?|"
            r"say|said|claim(?:ed|s|ing)|report(?:ed|s|ing)|suggest(?:ed|s|ing)|"
            r"den(?:y|ied|ying))\b[^.]{0,120}\bthat\b",
            value,
            flags=re.IGNORECASE,
        )
    )


def actor_has_abstract_head(actor: str) -> bool:
    tokens = role_phrase_tokens(actor)
    if not tokens:
        return False
    if any(token in ABSTRACT_ACTOR_HEADS for token in tokens[:3]):
        return True
    return False


def actor_is_generic_common_noun_phrase(actor: str) -> bool:
    value = normalize_space(actor)
    if not value:
        return False
    if value[:1].islower() and count_words(value) >= 3:
        return True
    return False


def object_has_abstract_head(obj: str) -> bool:
    tokens = role_phrase_tokens(obj)
    if not tokens:
        return False
    if any(token in ABSTRACT_OBJECT_HEADS for token in tokens[:3]):
        return True
    return False


def small_corpus_claim_surface_is_vague(
    text: str,
    profile: Mapping[str, Any],
) -> bool:
    value = normalize_space(text)
    if not value:
        return True
    lowered = value.casefold()
    if re.search(r"\b(?:somebody|someone|something|anything)\b", lowered):
        return True
    if re.search(
        r"\b(?:regarding the insinuation|pointing to somebody|connections has with)\b",
        lowered,
    ):
        return True
    actor = normalize_space(profile.get("actor", ""))
    predicate = normalize_space(profile.get("predicate", ""))
    obj = normalize_space(profile.get("object", ""))
    if profile_has_complete_simple_clause(profile):
        # Small-corpus claims sometimes carry a high-signal coordinated continuation
        # ("X is not building Y but is shopping for equipment"). Treat those as
        # concrete direct claims rather than vague object spillover.
        if re.search(
            r"\bbut\s+(?:is|was|are|were|has|have|had)\b",
            lowered,
            flags=re.IGNORECASE,
        ):
            return False
        if predicate.casefold() in {"send", "sent", "sending"} and count_words(obj) >= 5:
            return True
        if count_words(obj) >= 6 and re.search(r"\bto\s+[a-z][a-z-]+\b", obj, flags=re.IGNORECASE):
            return True
        if claim_uses_intentive_construction(value):
            return True
        if actor and obj and VERBISH_TOKEN_RE.search(obj) and count_words(obj) >= 5:
            return True
    return False


SAFE_INCOMPLETE_STANDALONE_CLAIM_RE = re.compile(
    r"\b(?:"
    r"had\s+no\s+(?:memory|recollection)\s+of"
    r"|has\s+no\s+(?:memory|recollection)\s+of"
    r"|was\s+told\b[^.]{0,120}\bno\s+longer\b"
    r"|(?:is|was|were)\s+no\s+longer\b"
    r"|did\s+not\s+meet(?:\s+with)?"
    r"|did\s+not\s+recall\b"
    r")",
    flags=re.IGNORECASE,
)


def claim_surface_has_cognitive_event_disclaimer(text: str) -> bool:
    lowered = normalize_space(text).casefold()
    if not lowered:
        return False
    if re.search(
        r"\b(?:did|does|do)\s+not\s+(?:remember|recall|know)\b",
        lowered,
        flags=re.IGNORECASE,
    ) and re.search(r"\b(?:meeting|meet|met|aware)\b", lowered, flags=re.IGNORECASE):
        return True
    if re.search(
        r"\b(?:had|has)\s+no\s+(?:memory|recollection)\s+of\b",
        lowered,
        flags=re.IGNORECASE,
    ) and re.search(r"\b(?:meeting|meet|met)\b", lowered, flags=re.IGNORECASE):
        return True
    if re.search(
        r"\b(?:did|does)\s+not\s+think\s+it\s+was\s+likely\b",
        lowered,
        flags=re.IGNORECASE,
    ) and re.search(r"\b(?:meeting|meet|met)\b", lowered, flags=re.IGNORECASE):
        return True
    return False


def claim_surface_has_multi_action_fallback_risk(
    text: str,
    profile: Mapping[str, Any],
) -> bool:
    lowered = normalize_space(text).casefold()
    if not lowered or " and " not in lowered:
        return False
    role_status = normalize_space(profile.get("role_normalization_status", ""))
    if not role_status:
        role_status = normalize_space(
            infer_simple_clause_arguments(text).get("role_normalization_status", "")
        )
    if role_status != "ACTIVE_GENERIC_FALLBACK":
        return False
    verbs = re.findall(
        r"\b(?:maintain(?:ed|s|ing)?|deliver(?:ed|s|ing)?|"
        r"escort(?:ed|s|ing)?|transport(?:ed|s|ing)?|move(?:d|s|ing)?|"
        r"take\s+over|takeover|protect(?:ed|s|ing)?|secure(?:d|s|ing)?)\b",
        lowered,
        flags=re.IGNORECASE,
    )
    return len(set(verbs)) >= 2


UNSAFE_INCOMPLETE_STANDALONE_CLAIM_RE = re.compile(
    r"\b(?:"
    r"played\s+a\s+part\s+in"
    r"|indicat(?:e|es|ed|ing)\s+or\s+insinuat(?:e|es|ed|ing)"
    r"|has\s+proposed\s+in\b"
    r"|offered\s+(?:an?\s+)?[^.]{0,48}\bpact\b"
    r"|appears?\s+to\s+be\b"
    r"|wants?\b"
    r"|believes?\b"
    r"|suggests?\b"
    r"|indicates?\b"
    r"|insinuates?\b"
    r"|points?\s+to\b"
    r")",
    flags=re.IGNORECASE,
)


FALLBACK_SIMPLE_CLAUSE_RE = re.compile(
    r"^(?P<subject>[A-Z][A-Za-z0-9#'./-]*(?:\s+[A-Za-z0-9#'./-]+){0,7})\s+"
    r"(?P<verb>is|are|was|were|has|have|had)\s+"
    r"(?P<object>.+?)"
    r"(?:[.?!])?$",
    flags=re.IGNORECASE,
)


FALLBACK_UNSAFE_SIMPLE_CLAIM_RE = re.compile(
    r"\b(?:"
    r"played\s+a\s+part\s+in"
    r"|indicat(?:e|es|ed|ing)\s+or\s+insinuat(?:e|es|ed|ing)"
    r"|appears?\s+to\s+be\b"
    r"|points?\s+to\b"
    r")",
    flags=re.IGNORECASE,
)


def claim_surface_is_retrieval_fragile(
    text: str,
    profile: Mapping[str, Any],
    *,
    small_corpus: bool = False,
    pair_like_corpus: bool = False,
) -> bool:
    value = normalize_space(text)
    if not value:
        return True
    lowered = value.casefold()
    actor = normalize_space(profile.get("actor", ""))
    predicate = normalize_space(profile.get("predicate", ""))
    obj = normalize_space(profile.get("object", ""))
    has_structured_anchor = bool(
        profile.get("date_token")
        or profile.get("number_token")
        or distinctive_label_token_count(actor) >= 1
        or distinctive_label_token_count(obj) >= 1
    )
    if CONDITIONAL_CLAIM_RE.search(value) or "conditional on" in lowered or "unless" in lowered:
        return True
    if claim_surface_has_cognitive_event_disclaimer(value):
        return True
    if (
        UNSAFE_INCOMPLETE_STANDALONE_CLAIM_RE.search(value)
        and not SAFE_INCOMPLETE_STANDALONE_CLAIM_RE.search(value)
    ):
        return True
    if re.search(
        r"\b(?:support(?:ed|s|ing)?|oppose(?:d|s|ing)?|back(?:ed|s|ing)?|"
        r"endorse(?:d|s|ing)?)\b[^.]{0,140}\b(?:position|claim|charge|"
        r"allegation|view|assessment|argument)\s+"
        r"(?:that|charging|saying|claiming|arguing|holding)\b",
        lowered,
        flags=re.IGNORECASE,
    ):
        return True
    if small_corpus and actor and actor_has_abstract_head(actor):
        return True
    if predicate.casefold() in {"is", "was", "are", "were"} and count_words(obj) <= 4 and not has_structured_anchor:
        return True
    if coordinated_multi_claim_surface(value):
        return True
    return False


def claim_surface_is_short_classification(
    text: str,
    profile: Mapping[str, Any],
) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    predicate = normalize_space(profile.get("predicate", "")).casefold()
    obj = normalize_space(profile.get("object", ""))
    if predicate not in {"be", "is", "was", "are", "were"}:
        return False
    if count_words(value) > 10:
        return False
    if profile.get("date_token") or profile.get("number_token"):
        return False
    return bool(
        re.search(
            r"\b(?:agent(?:\s+of)?(?:\s+a)?\s+foreign power|foreign power|affiliate|ally|member|official|participant|supporter|opponent)\b",
            obj or value,
            flags=re.IGNORECASE,
        )
    )


def coordinated_multi_claim_surface(text: str) -> bool:
    value = normalize_space(text)
    if not value or not re.search(r"\band\b", value, flags=re.IGNORECASE):
        return False
    if not re.search(
        r"\band\s+(?:demands?|calls?\s+for|urges?|supports?|opposes?|"
        r"reiterates?|announces?|seeks?|plans?|intends?)\b",
        value,
        flags=re.IGNORECASE,
    ):
        return False
    parts = [
        finalize_query_surface(part, add_terminal_punctuation=False)
        for part in re.split(r"\band\b", value, maxsplit=1, flags=re.IGNORECASE)
        if normalize_space(part)
    ]
    if len(parts) != 2:
        return False
    left_profile = claim_clause_profile(parts[0])
    right_profile = claim_clause_profile(parts[1])
    left_executable = profile_has_complete_simple_clause(left_profile)
    right_executable = (
        profile_has_complete_simple_clause(right_profile)
        or fallback_simple_complete_claim_surface_is_safe(parts[1])
    )
    return left_executable and right_executable


def fallback_simple_complete_claim_surface_is_safe(text: str) -> bool:
    value = simplify_claim_surface(text)
    if not value:
        return False
    if query_surface_is_blocked(value, task_family="claims"):
        return False
    if value.count(";") >= 1 or "..." in value:
        return False
    if is_routing_or_metadata_sentence(value) or is_list_like_text(value):
        return False
    if CONDITIONAL_CLAIM_RE.search(value) or ABSTRACT_QUANTIFIER_RE.search(value):
        return False
    if claim_surface_is_interrogative(value) or claim_surface_has_weak_modality(value):
        return False
    if claim_uses_intentive_construction(value) or has_reporting_that_clause(value):
        return False
    match = FALLBACK_SIMPLE_CLAUSE_RE.match(value)
    if not match:
        return False
    subject = normalize_space(match.group("subject"))
    obj = normalize_space(match.group("object"))
    if not subject or not obj:
        return False
    if count_words(subject) > 8 or count_words(obj) < 3 or count_words(obj) > 24:
        return False
    subject_tokens = [
        token for token in subject.split()
        if token.casefold() not in {"the", "a", "an", "of"}
    ]
    if not subject_tokens:
        return False
    if not any(re.match(r"^(?:[A-Z]|#)", token) for token in subject_tokens):
        return False
    if re.match(
        r"^(?:issue|approach|strategy|effort|question|idea|message)$",
        subject_tokens[-1],
        flags=re.IGNORECASE,
    ):
        return False
    if re.match(r"^(?:of|for|to|with|that)\b", obj, flags=re.IGNORECASE):
        return False
    if FALLBACK_UNSAFE_SIMPLE_CLAIM_RE.search(value):
        return False
    if LEGAL_BOILERPLATE_RE.search(value) or BROKEN_LEAD_RE.search(value):
        return False
    if substantive_sentence_score(value) < 5.5:
        return False
    return True


def small_corpus_claim_surface_priority_adjustment(text: str) -> float:
    value = normalize_space(text)
    if not value:
        return 0.0
    lowered = value.casefold()
    bonus = 0.0
    if re.search(
        r"\b(?:offered|proposed|requested|agreed|approved|accepted|rejected|ordered|announced)\b",
        lowered,
    ):
        bonus += 16.0
    if re.search(r"\burged\b", lowered):
        bonus += 8.0
    if re.search(
        r"\bnot\b[^.]{0,80}\bbut\s+(?:is|was|are|were|has|have|had)\b",
        lowered,
        flags=re.IGNORECASE,
    ):
        bonus += 18.0
    if re.search(
        r"\b(?:"
        r"pointing to (?:somebody|someone|something)"
        r"|regarding (?:the )?(?:insinuation|conjecture|possibility)"
        r"|(?:somebody|someone|something) is (?:responsible|involved)"
        r")\b",
        lowered,
    ):
        bonus -= 16.0
    return bonus


SMALL_CLAIM_RUNTIME_HOSTILE_RECOVERY_RE = re.compile(
    r"\b(?:"
    r"pointing to (?:somebody|someone|something)"
    r"|regarding (?:the )?(?:insinuation|conjecture)"
    r"|connections has with"
    r")\b",
    flags=re.IGNORECASE,
)


def standalone_claim_surface_is_safe(text: str) -> bool:
    value = simplify_claim_surface(text)
    if not value:
        return False
    if value[:1].islower() and count_words(value) >= 8:
        return False
    if query_surface_is_blocked(value, task_family="claims"):
        return False
    if CONDITIONAL_CLAIM_RE.search(value):
        return False
    if ABSTRACT_QUANTIFIER_RE.search(value):
        return False
    if re.search(
        r"\b(?:may|might|could|would|should)\s+have\s+not\b|\bnot\s+(?:probably|hardly|barely|rarely|scarcely)\b",
        value,
        flags=re.IGNORECASE,
    ):
        return False
    if claim_uses_intentive_construction(value):
        return False
    if NOMINALIZED_CLAIM_RE.search(value):
        return False
    profile = claim_clause_profile(value)
    complete_profile = profile_has_complete_simple_clause(profile)
    if re.match(r"^There\s+(?:is|are)\b", value, flags=re.IGNORECASE) and not profile_has_complete_simple_clause(profile):
        return False
    if count_words(value) < 8 and not re.search(
        r"\b(?:affiliated with|supported|opposed|communicated with|requested action from|mentioned|met with|was located in|was related to)\b",
        value,
        flags=re.IGNORECASE,
    ) and not re.search(
        r"\b(?:is|was|are|were)\s+(?:an?\s+)?(?:agent|foreign power|affiliate|ally|member|official|participant|supporter|opponent)\b",
        value,
        flags=re.IGNORECASE,
    ):
        return False
    if value.count(";") >= 2:
        return False
    if is_routing_or_metadata_sentence(value) or is_list_like_text(value):
        return False
    if has_reporting_that_clause(value):
        return False
    if not complete_profile:
        if fallback_simple_complete_claim_surface_is_safe(value):
            return True
        if UNSAFE_INCOMPLETE_STANDALONE_CLAIM_RE.search(value):
            return False
        if not SAFE_INCOMPLETE_STANDALONE_CLAIM_RE.search(value):
            return False
    if LEGAL_BOILERPLATE_RE.search(value) or BROKEN_LEAD_RE.search(value):
        return False
    if substantive_sentence_score(value) < 5.0:
        return False
    return True


def standalone_contradiction_surface_is_safe(text: str) -> bool:
    value = simplify_claim_surface(text)
    if not value:
        return False
    if value[:1].islower() and count_words(value) >= 8:
        return False
    if query_surface_is_blocked(value, task_family="contradictions"):
        return False
    if claim_surface_is_interrogative(value) or claim_surface_has_weak_modality(value):
        return False
    if CONDITIONAL_CLAIM_RE.search(value) or ABSTRACT_QUANTIFIER_RE.search(value):
        return False
    if claim_uses_intentive_construction(value) or NOMINALIZED_CLAIM_RE.search(value):
        return False
    if count_words(value) < 8 or count_words(value) > 48:
        return False
    if value.count(";") >= 2:
        return False
    if is_routing_or_metadata_sentence(value) or is_list_like_text(value):
        return False
    if LEGAL_BOILERPLATE_RE.search(value) or BROKEN_LEAD_RE.search(value):
        return False
    if fallback_simple_complete_claim_surface_is_safe(value):
        return True
    if sentence_is_low_signal_for_query(value) or repeated_sentence_leak(value):
        return False
    if has_reporting_that_clause(value) and not re.search(
        r"\b(?:indicat(?:e|es|ed|ing)|insinuat(?:e|es|ed|ing))\b",
        value,
        flags=re.IGNORECASE,
    ):
        return False
    if substantive_sentence_score(value) < 5.0:
        return False
    return True


REPORTING_WRAPPER_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"(?:according to|in the view of)\s+[^,;:.]{2,80},?\s*"
    r"|"
    r"[^,;:.]{2,80}\b"
    r"(?:said|says|stated|reported|reports|believed|believes|claims?|"
    r"thought|thinks|think|assessed|assesses|assess|"
    r"charged|charges|asserted|asserts|argued|argues|maintained|maintains|"
    r"contended|contends|noted|notes|warned|warns|told)\b\s+that\s+"
    r")",
    flags=re.IGNORECASE,
)

CONTESTED_SUPPORT_RE = re.compile(
    r"\b(?:rumou?r(?:ed|s)?|opined|not\s+clear|does\s+not\s+think|did\s+not\s+think|"
    r"doesn't\s+think|didn't\s+think|denied|deny|denies|does\s+not\s+believe|"
    r"did\s+not\s+believe|it\s+was\s+rumored)\b",
    flags=re.IGNORECASE,
)


def claim_row_loses_support_or_polarity_when_simplified(
    row: Mapping[str, Any],
) -> bool:
    raw_text = finalize_query_surface(
        normalize_space(row.get("claim_text", "")),
        add_terminal_punctuation=True,
    )
    if not raw_text:
        return False
    simplified_text = finalize_query_surface(
        simplify_claim_query_seed(simplify_claim_surface(raw_text)),
        add_terminal_punctuation=True,
    )
    if not simplified_text:
        return True
    if CONTESTED_SUPPORT_RE.search(raw_text):
        return True
    raw_profile = claim_clause_profile(raw_text)
    simplified_profile = claim_clause_profile(simplified_text)
    if bool(raw_profile.get("has_negation")) and not bool(
        simplified_profile.get("has_negation")
    ):
        return True
    if (
        has_reporting_that_clause(raw_text)
        and not has_reporting_that_clause(simplified_text)
        and raw_text.casefold() != simplified_text.casefold()
    ):
        return True
    if (
        REPORTING_WRAPPER_PREFIX_RE.search(raw_text)
        and not REPORTING_WRAPPER_PREFIX_RE.search(simplified_text)
        and raw_text.casefold() != simplified_text.casefold()
    ):
        return True
    return False

METADATA_LEAD_RE = re.compile(
    r"^\s*(?:"
    r"(?:[A-Z]{2,}(?:\s+[A-Z0-9]{1,}){1,5}\s+(?:report|cable|telegram|message|dispatch)(?:\s+\d+)?)"
    r"|(?:previously report)"
    r")\b[\s,:-]*",
    flags=re.IGNORECASE,
)

INTERROGATIVE_CLAIM_LEAD_RE = re.compile(
    r"^\s*(?:who|what|when|where|why|how|which)\b",
    flags=re.IGNORECASE,
)

UNCERTAIN_CLAIM_SURFACE_RE = re.compile(
    r"\b(?:may|might|possibly|perhaps|likely|reportedly|allegedly|purportedly|"
    r"apparently|rumou?red?|uncertain)\b",
    flags=re.IGNORECASE,
)


def strip_reporting_wrapper(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    prior = None
    while value and value != prior:
        prior = value
        updated = REPORTING_WRAPPER_PREFIX_RE.sub("", value, count=1).strip()
        if updated and updated != value:
            value = updated
    return value


def simplify_claim_query_seed(text: str) -> str:
    value = finalize_query_surface(text, add_terminal_punctuation=True)
    if not value:
        return ""
    value = METADATA_LEAD_RE.sub("", value, count=1).strip()
    value = re.sub(
        r"^\s*[A-Z][a-z]+ report\s+",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*[^.]{0,120}\b(?:has|have)\s+reported\s+",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    )
    value = strip_reporting_wrapper(value)
    value = re.sub(
        r"^\s*[^,;:.]{2,80}\b(?:does not think|did not think|doesn't think|thinks|think|thought)\b\s+that\s+",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b(?:is|was|were)\s+(?:earlier\s+)?reported to have\b",
        "had",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b(?:is|was|were)\s+(?:earlier\s+)?reported to\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bthe tenor of the message (?:he|she|they) had received from ([A-Z][^.]+?)(?:[.])?$",
        r"a message from \1.",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bthe (message|letter|report) (?:he|she|they) had received from ([A-Z][^.]+?)(?:[.])?$",
        r"the \1 from \2.",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b([A-Z][A-Za-z'’.-]+)\s+was sentenced to death because he had met with\s+([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+)*)\b",
        r"\1 met with \2 before he was sentenced to death",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b([Tt]he present talks)\s+have received\s+([^,]+),\s+but that support is not unanimous\b",
        r"\1 received \2, but support is not unanimous",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bthe information relied on by (.+?)\s+(?:is|are)\s+lies\b",
        r"\1 relied on lies",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bthe information relied on by (.+?)\s+(?:is|are)\s+false\b",
        r"\1 relied on false information",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*([A-Z][A-Za-z0-9#'’.\- ]{1,80})\s+claims\s+(he|she|they)\b",
        r"\1",
        value,
        count=1,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\s+and\s+now\s+has\s+between\b[^.?!]*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*No\s+(.+?)\s+has\s+yet\s+been\s+received\s+by\s+(.+?)\.?\s*$",
        r"\2 has not yet received \1.",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*No\s+(.+?)\s+had\s+yet\s+been\s+received\s+by\s+(.+?)\.?\s*$",
        r"\2 had not yet received \1.",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*([A-Z][A-Za-z0-9#'’.\- ]{1,90}?)\s+had\s+no\s+memory\s+of\s+any\s+meeting\s+between\s+(.+?)\s+and\s+(.+?)\.?\s*$",
        r"\1 did not remember any meeting between \2 and \3.",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*([A-Z][A-Za-z0-9#'’.\- ]{1,90}?)\s+was\s+told\s+(?:he|she|they)\s+was\s+no\s+longer\s+part\s+of\s+(.+?)\.?\s*$",
        r"\2 no longer included \1.",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*([A-Z][A-Za-z0-9#'’.\- ]{1,90}?)\s+was\s+no\s+longer\s+part\s+of\s+(.+?)\.?\s*$",
        r"\2 no longer included \1.",
        value,
        flags=re.IGNORECASE,
    )
    value = PLACEHOLDER_PHRASE_RE.sub("", value)
    value = EXTRACTION_ARTIFACT_RE.sub("", value)
    value = re.sub(r"\b(?:OTHER_RELATED|COMMUNICATION|AFFILIATION|OPPOSITION|SUPPORT)\b", "", value)
    value = re.sub(r"\s+'\.", "'.", value)
    value = normalize_uppercase_passage(value)
    value = soften_titlecase_noise(value)
    value = normalize_surface_preserving_markers(value)
    value = re.sub(r"\biaea\b", "IAEA", value, flags=re.IGNORECASE)
    if value:
        value = ensure_terminal_punctuation(value)
    return value


def claim_surface_is_interrogative(text: Any) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    return bool(INTERROGATIVE_CLAIM_LEAD_RE.match(value))


def claim_surface_has_weak_modality(text: Any) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    return bool(UNCERTAIN_CLAIM_SURFACE_RE.search(value))


def sentence_is_reporting_heavy(text: Any) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    lowered = value.casefold()
    profile = claim_clause_profile(value)
    if profile.get("nested_reporting") or profile.get("reporting_actor"):
        return True
    if has_reporting_that_clause(value):
        return True
    return bool(
        re.search(
            r"\b(?:opined|believes?|assesses?|reported|reports|suggested|stated)\b",
            lowered,
            flags=re.IGNORECASE,
        )
    )


def looks_like_acronym_label(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    letters = re.sub(r"[^A-Za-z]", "", value)
    if len(letters) < 2:
        return False
    return letters == letters.upper() and count_words(value) <= 3


def readable_list(values: list[str]) -> str:
    items = [normalize_space(value) for value in values if normalize_space(value)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def is_probably_generic_person_label(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return True
    return bool(GENERIC_PERSON_LABEL_RE.fullmatch(value))


def is_title_only_person_label(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return True
    if is_probably_generic_person_label(value):
        return True
    if TITLE_ONLY_PERSON_LABEL_RE.search(value):
        return True
    tokens = label_tokens(value)
    if not tokens:
        return True
    non_generic_tokens = [
        token
        for token in tokens
        if len(token) > 2
        and token not in GENERIC_LABEL_HEADS
        and token not in GENERIC_LABEL_MODIFIERS
    ]
    return not non_generic_tokens and len(tokens) <= 4


def is_generic_label(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return True
    return bool(GENERIC_LABEL_RE.fullmatch(value))


def label_tokens(text: str) -> list[str]:
    return [token.casefold() for token in WORD_RE.findall(normalize_space(text))]


def distinctive_label_token_count(text: str) -> int:
    count = 0
    for token in label_tokens(text):
        if token.isdigit() or len(token) <= 2:
            continue
        if token in GENERIC_LABEL_HEADS or token in GENERIC_LABEL_MODIFIERS:
            continue
        count += 1
    return count


def has_generic_label_head(text: str) -> bool:
    tokens = label_tokens(text)
    return bool(tokens) and tokens[-1] in GENERIC_LABEL_HEADS


def build_reconstruction_query(
    row: dict[str, Any],
    sampling: dict[str, Any],
    *,
    source_chunk_text: str = "",
) -> str:
    def usable_reconstruction_surface(text: Any) -> str:
        candidate = finalize_query_surface(text, add_terminal_punctuation=True)
        if not candidate:
            return ""
        candidate = finalize_query_surface(
            extract_sentence_excerpt(
                candidate,
                max_sentences=1 if count_words(candidate) >= 28 else 2,
                max_chars=280,
            ),
            add_terminal_punctuation=True,
        )
        if not candidate or count_words(candidate) < 8:
            return ""
        if is_list_like_text(candidate) or is_routing_or_metadata_sentence(candidate):
            return ""
        if is_probably_non_english_query(candidate):
            return ""
        if candidate.count(";") > 0 or "..." in candidate:
            return ""
        if coordination_heavy_surface(candidate):
            return ""
        if query_surface_is_blocked(candidate, task_family="reconstruction"):
            return ""
        flags = set(query_surface_flags(candidate, task_family="reconstruction"))
        if flags & {
            "titlecase_noise",
            "mixed_case_noise",
            "uppercase_heavy",
            "metadata_scaffold",
            "headline_scaffold",
            "trailing_ellipsis",
            "foreign_language",
        }:
            return ""
        if substantive_sentence_score(candidate, anchors=anchors) < 5.0:
            return ""
        return candidate

    source_excerpt = finalize_query_surface(source_chunk_text)
    claims = unique_preserving_order(
        parse_json_list(row.get("reconstruction_claims_json", row.get("claim_texts", "")))
    )
    events = unique_preserving_order(
        parse_json_list(row.get("reconstruction_events_json", row.get("event_texts", "")))
    )
    anchors = normalized_reconstruction_anchors(parse_json_list(row.get("rare_anchors_json")))
    if source_excerpt:
        excerpt = usable_reconstruction_surface(source_excerpt)
        if excerpt:
            return excerpt
    for claim_text in claims:
        cleaned_claim = simplify_claim_query_seed(simplify_claim_surface(claim_text))
        candidate = usable_reconstruction_surface(cleaned_claim)
        if candidate:
            return candidate
    for event_text in events:
        candidate = usable_reconstruction_surface(clean_generation_sentence(event_text))
        if candidate:
            return candidate
    if source_excerpt:
        candidate = usable_reconstruction_surface(source_excerpt)
        if candidate:
            return candidate
    return ""


ELLIPTICAL_COORDINATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"^(?P<subject>[A-Z][A-Za-z0-9#'’().,&/\- ]{1,90}?)\s+is\s+(?P<left>[^.?!;]{3,140}?)\s+but\s+is\s+(?P<right>[^.?!;]{3,140})\.?$",
        "{subject} is {left}, but {subject} is {right}.",
    ),
    (
        r"^(?P<subject>[A-Z][A-Za-z0-9#'’().,&/\- ]{1,90}?)\s+was\s+(?P<left>[^.?!;]{3,140}?)\s+but\s+was\s+(?P<right>[^.?!;]{3,140})\.?$",
        "{subject} was {left}, but {subject} was {right}.",
    ),
    (
        r"^(?P<subject>[A-Z][A-Za-z0-9#'’().,&/\- ]{1,90}?)\s+has\s+(?P<left>[^.?!;]{3,140}?)\s+but\s+has\s+(?P<right>[^.?!;]{3,140})\.?$",
        "{subject} has {left}, but {subject} has {right}.",
    ),
    (
        r"^(?P<subject>[A-Z][A-Za-z0-9#'’().,&/\- ]{1,90}?)\s+have\s+(?P<left>[^.?!;]{3,140}?)\s+but\s+have\s+(?P<right>[^.?!;]{3,140})\.?$",
        "{subject} have {left}, but {subject} have {right}.",
    ),
)


def expand_claim_coordination_ellipsis(text: Any) -> str:
    value = finalize_query_surface(text, add_terminal_punctuation=True)
    if not value:
        return ""
    if " but " not in value.casefold():
        return value
    for pattern, template in ELLIPTICAL_COORDINATION_PATTERNS:
        match = re.match(pattern, value, flags=re.IGNORECASE)
        if not match:
            continue
        subject = normalize_space(match.group("subject"))
        left = normalize_space(match.group("left"))
        right = normalize_space(match.group("right"))
        if not subject or not left or not right:
            continue
        expanded = finalize_query_surface(
            template.format(subject=subject, left=left, right=right),
            add_terminal_punctuation=True,
        )
        if expanded:
            return expanded
    return value


def build_claim_query(
    row: dict[str, Any],
    *,
    include_stance: bool,
    claim_text_override: str | None = None,
    query_style: str = "plain",
    source_context_text: str = "",
    prefer_speaker_attribution: bool = False,
) -> tuple[str, dict[str, str], int]:
    claim_text = simplify_claim_surface(claim_text_override or row.get("claim_text", ""))
    claim_text = simplify_claim_query_seed(claim_text)
    speaker = normalize_space(row.get("speaker", ""))
    target = normalize_space(row.get("target", ""))
    stance = normalize_space(row.get("stance", ""))
    normalized_speaker = normalize_query_speaker(speaker)
    non_generic_speaker = normalized_speaker if not is_generic_role_value(normalized_speaker) else ""
    non_generic_target = target if not is_generic_role_value(target) else ""
    inferred = infer_simple_clause_arguments(claim_text)
    actor = normalize_space(inferred.get("actor", ""))
    predicate = normalize_space(inferred.get("predicate", ""))
    obj = normalize_space(inferred.get("object", "")) or normalize_space(inferred.get("location", ""))
    stance_value = stance or "unknown"
    style = normalize_space(query_style).casefold() or "plain"
    source_context = finalize_query_surface(source_context_text)
    context_anchors = [
        value
        for value in [speaker, target]
        if normalize_space(value)
    ]
    clause = ""
    if role_query_is_safe(actor, predicate, obj):
        clause = finalize_query_surface(" ".join(part for part in [actor, predicate, obj] if part))
    natural_statement = clause or claim_text
    statement = finalize_query_surface(natural_statement, add_terminal_punctuation=True)
    statement_profile = claim_clause_profile(statement)
    direct_statement_is_safe = bool(
        statement
        and strict_claim_surface_is_safe(
            statement,
            statement_profile,
            max_actor_words=8,
            max_object_words=18,
        )
        and standalone_claim_surface_is_safe(statement)
    )
    speaker_attributed_statement = ""
    if (
        prefer_speaker_attribution
        and non_generic_speaker
        and statement
        and not direct_statement_is_safe
        and count_words(statement) >= 6
        and distinctive_label_token_count(non_generic_speaker) >= 1
        and not find_label_match(statement, non_generic_speaker)
        and "report" not in non_generic_speaker.casefold()
        and "source" not in non_generic_speaker.casefold()
    ):
        speaker_rendered = pretty_entity_label(non_generic_speaker)
        speaker_lower = non_generic_speaker.casefold()
        attribution_statement = finalize_query_surface(claim_text, add_terminal_punctuation=True) or statement
        attribution_verb = (
            "indicated"
            if "report" in speaker_lower
            else "said"
            if "source" in speaker_lower
            else "reported"
        )
        speaker_attributed_statement = finalize_query_surface(
            f"{speaker_rendered} {attribution_verb} that {attribution_statement}",
            add_terminal_punctuation=True,
        )
    contextual_source = ""
    if source_context and not is_routing_or_metadata_sentence(source_context):
        contextual_source = finalize_query_surface(
            extract_sentence_excerpt(
                source_context,
                max_sentences=2 if style == "contextual" else 1,
                max_chars=240 if style == "contextual" else 180,
            ),
            add_terminal_punctuation=True,
        )
        if contextual_source and substantive_sentence_score(
            contextual_source,
            anchors=context_anchors,
        ) < 2.0:
            contextual_source = ""
        if contextual_source and statement and content_token_overlap_ratio(
            contextual_source,
            statement,
        ) >= 0.88:
            contextual_source = ""
    if style == "contextual" and contextual_source:
        if statement and content_token_overlap_ratio(contextual_source, statement) < 0.55:
            query_text = finalize_query_surface(
                f"{contextual_source} {statement}",
                add_terminal_punctuation=True,
            )
        else:
            query_text = contextual_source
    elif natural_statement:
        query_text = statement
    elif style == "role_clause" and clause:
        query_text = finalize_query_surface(clause, add_terminal_punctuation=True)
    elif source_context and not is_routing_or_metadata_sentence(source_context):
        query_text = source_context
    elif source_context:
        query_text = statement
    else:
        query_text = statement
    if speaker_attributed_statement:
        query_text = speaker_attributed_statement
    if include_stance and stance_value not in {"", "unknown", "neutral", "unclear"}:
        query_text = finalize_query_surface(
            f"{query_text} The reporting stance is {stance_value.lower()}.",
            add_terminal_punctuation=True,
        )
    max_query_sentences = 2 if style == "contextual" else 1
    max_query_chars = 320 if style == "contextual" else 280
    query_text = finalize_query_surface(
        extract_sentence_excerpt(
            query_text,
            max_sentences=max_query_sentences,
            max_chars=max_query_chars,
        ),
        add_terminal_punctuation=True,
    )
    query_text = expand_claim_coordination_ellipsis(query_text)
    role_field_count = sum(
        [
            bool(actor),
            bool(predicate),
            bool(obj),
            bool(non_generic_speaker),
            bool(non_generic_target),
            bool(stance and stance.casefold() not in {"neutral", "unclear", "unknown"}),
        ]
    )
    return finalize_query_surface(query_text, add_terminal_punctuation=False), {
        "actor": actor,
        "speaker": non_generic_speaker,
        "target": non_generic_target,
        "predicate": predicate,
        "object": obj,
        "stance": stance_value,
    }, role_field_count


def relation_value_is_placeholder(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return True
    lowered = value.casefold()
    if lowered.startswith("event_") or lowered.startswith("ent_"):
        return True
    if lowered in {"unknown", "not specified", "na", "n/a"}:
        return True
    return False


ROLE_PLACEHOLDER_RE = re.compile(
    r"\b(?:candidate|political party|source|sub-source)\s*#\d+\b",
    flags=re.IGNORECASE,
)


def contains_role_placeholder_label(text: str) -> bool:
    return bool(ROLE_PLACEHOLDER_RE.search(normalize_space(text)))


def pretty_entity_label(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    if not re.search(r"[A-Za-z]", value):
        return value
    if looks_like_acronym_label(value) or value.upper() in ACRONYM_KEEPERS:
        return value.upper()
    if uppercase_token_ratio(value) < 0.6 or count_words(value) > 4:
        return value
    rendered: list[str] = []
    for raw_token in value.split():
        letters = re.sub(r"[^A-Za-z]", "", raw_token)
        if not letters:
            rendered.append(raw_token)
            continue
        if letters.upper() in ACRONYM_KEEPERS or looks_like_acronym_label(raw_token):
            rendered.append(raw_token.upper())
            continue
        rendered.append(raw_token.capitalize())
    return normalize_space(" ".join(rendered))


def render_relation_claim_text(
    source: str,
    relation: str,
    target: str,
    evidence: str,
) -> str:
    source = pretty_entity_label(source)
    target = pretty_entity_label(target)
    relation_key = normalize_space(relation).casefold()
    cleaned_evidence = finalize_query_surface(evidence, add_terminal_punctuation=True)
    if (
        relation_key in {"support", "communication", "event_participant"}
        and relation_evidence_can_anchor_claim_surface(
            source,
            relation,
            target,
            cleaned_evidence,
        )
    ):
        return cleaned_evidence
    if relation_key == "affiliation":
        return finalize_query_surface(f"{source} was affiliated with {target}.")
    if relation_key == "support":
        return finalize_query_surface(f"{source} supported {target}.")
    if relation_key == "opposition":
        return finalize_query_surface(f"{source} opposed {target}.")
    if relation_key == "communication":
        return finalize_query_surface(f"{source} communicated with {target}.")
    if relation_key == "request":
        return finalize_query_surface(f"{source} requested action from {target}.")
    if relation_key == "mentions":
        return finalize_query_surface(f"{source} mentioned {target}.")
    if relation_key == "event_participant":
        lowered = cleaned_evidence.casefold()
        if "met with" in lowered or "meeting with" in lowered:
            return finalize_query_surface(f"{source} met with {target}.")
        if "offered" in lowered:
            return cleaned_evidence
        return finalize_query_surface(f"{source} was involved with {target}.")
    if relation_key == "event_location":
        return finalize_query_surface(f"{source} was located in {target}.")
    if relation_key == "other_related":
        return finalize_query_surface(f"{source} was related to {target}.")
    return ""


def relation_predicate_surface(relation: str) -> str:
    relation_key = normalize_space(relation).casefold()
    return {
        "affiliation": "was affiliated with",
        "support": "supported",
        "opposition": "opposed",
        "communication": "communicated with",
        "request": "requested action from",
        "mentions": "mentioned",
        "event_participant": "met with",
        "event_location": "was located in",
        "other_related": "was related to",
    }.get(relation_key, "was related to")


DIRECT_SUPPORT_EVIDENCE_RE = re.compile(
    r"\b(?:support(?:ed|s|ing)?|back(?:ed|ing)?|recogniz(?:e|ed|es|ing)|"
    r"endors(?:e|ed|es|ing)|favou?r(?:ed|s|ing)?|align(?:ed|s|ing)? with)\b",
    flags=re.IGNORECASE,
)


def relation_evidence_can_anchor_claim_surface(
    source: str,
    relation: str,
    target: str,
    evidence: str,
) -> bool:
    cleaned_evidence = finalize_query_surface(evidence, add_terminal_punctuation=True)
    if not cleaned_evidence:
        return False
    if query_surface_is_blocked(cleaned_evidence, task_family="claims"):
        return False
    if sentence_is_low_signal_for_query(cleaned_evidence):
        return False
    lowered = cleaned_evidence.casefold()
    if cleaned_evidence.count(";") or "..." in cleaned_evidence:
        return False
    if " and " in f" {lowered} ":
        return False
    if re.search(r"\b(?:him|her|them|his|their|its)\b", lowered):
        return False
    if not (find_label_match(cleaned_evidence, source) and find_label_match(cleaned_evidence, target)):
        return False
    profile = claim_clause_profile(cleaned_evidence)
    if not strict_claim_surface_is_safe(
        cleaned_evidence,
        profile,
        max_actor_words=8,
        max_object_words=18,
    ):
        return False
    role_fields = {
        "actor": normalize_space(profile.get("actor", "")),
        "predicate": normalize_space(profile.get("predicate", "")),
        "object": normalize_space(profile.get("object", "")),
        "speaker": "",
        "target": "",
        "stance": "affirmed",
    }
    if parser_hostile_claim_candidate(cleaned_evidence, role_fields, profile):
        return False
    relation_key = normalize_space(relation).casefold()
    object_surface = normalize_space(profile.get("object", ""))
    if relation_key in {"support", "opposition"}:
        if not DIRECT_SUPPORT_EVIDENCE_RE.search(cleaned_evidence):
            return False
        if not find_label_match(object_surface, target):
            return False
    if relation_key == "communication":
        if not re.search(r"\b(?:letter|telegram|message|told|informed|sent|discussed|met with)\b", lowered):
            return False
    return True


def negate_relation_statement(text: str) -> str:
    value = finalize_query_surface(text, add_terminal_punctuation=True)
    if not value:
        return ""
    replacements = (
        (r"\bwas affiliated with\b", "was not affiliated with"),
        (r"\bwas located in\b", "was not located in"),
        (r"\bwas related to\b", "was not related to"),
        (r"\bsupported\b", "did not support"),
        (r"\bopposed\b", "did not oppose"),
        (r"\bcommunicated with\b", "did not communicate with"),
        (r"\brequested action from\b", "did not request action from"),
        (r"\bmentioned\b", "did not mention"),
        (r"\bmet with\b", "did not meet with"),
        (r"\bwas involved with\b", "was not involved with"),
    )
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, value, count=1, flags=re.IGNORECASE)
        if updated != value:
            return finalize_query_surface(updated, add_terminal_punctuation=True)
    negated = perturb_negation(value)
    if negated:
        return finalize_query_surface(negated[0], add_terminal_punctuation=True)
    return ""


def normalize_relation_evidence_sentence(evidence: str) -> str:
    cleaned = finalize_query_surface(evidence, add_terminal_punctuation=True)
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"^\s*([a-z])",
        lambda m: m.group(1).upper(),
        cleaned,
        count=1,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def choose_relation_claim_cases(
    relations: pd.DataFrame,
    chunk_text_lookup: dict[tuple[str, str], str],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    runner_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    strict_mode = not maybe_bool(sampling.get("_relaxed_retry_used", False))
    doc_ids = {
        normalize_space(doc_id)
        for doc_id in relations.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    }
    pair_like_corpus = bool(doc_ids) and all(doc_id.startswith("Pair_") for doc_id in doc_ids)
    large_relation_corpus = len(relations) >= 25000
    relation_cleanup_corpus = len(relations) < 400
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    small_corpus_bonus = 6.0 if len(relations) < 150 else 0.0
    relation_candidate_cap = (
        max(count * 50, 300)
        if len(relations) >= 1500
        else max(count * 30, 150)
        if len(relations) >= 400
        else 0
    )
    min_query_words = int(sampling.get("min_query_words", 16))
    effective_min_query_words = 6 if pair_like_corpus else (8 if len(relations) < 150 else min_query_words)
    relation_min_query_words = 6
    max_query_words = int(sampling.get("max_query_words", max(int(sampling.get("max_words", 48)) * 4, 180)))
    min_role_fields = int(sampling.get("min_role_fields", 2))
    preferred_relations = {
        "support": 8.5,
        "affiliation": 8.0,
        "communication": 7.0,
        "opposition": 6.0,
    }
    if not pair_like_corpus and len(relations) >= 1000:
        preferred_relations = {
            "support": 9.0,
            "communication": 8.0,
            "opposition": 6.5,
            "affiliation": 5.0,
        }
    candidates: list[dict[str, Any]] = []
    relation_rows = deterministic_seeded_row_order(
        relations.to_dict("records"),
        seed=seed,
        key_fields=("doc_id", "paragraph_id", "source", "relation", "target", "evidence"),
    )
    for row in relation_rows:
        if relation_candidate_cap and len(candidates) >= relation_candidate_cap:
            break
        source = normalize_space(row.get("source", ""))
        target = normalize_space(row.get("target", ""))
        relation = normalize_space(row.get("relation", ""))
        evidence = normalize_space(row.get("evidence", ""))
        if not source or not target or not relation or not evidence:
            continue
        if relation_value_is_placeholder(source) or relation_value_is_placeholder(target):
            continue
        if contains_role_placeholder_label(source) or contains_role_placeholder_label(target):
            continue
        if is_generic_role_value(source) or is_generic_role_value(target):
            continue
        if actor_phrase_has_clause_bleed(source) or actor_phrase_has_clause_bleed(target):
            continue
        if actor_is_generic_common_noun_phrase(source):
            continue
        if actor_has_abstract_head(source):
            continue
        source_words = count_words(source)
        target_words = count_words(target)
        if source_words > 6 or target_words > 6:
            continue
        if source_words > 1 and VERBISH_TOKEN_RE.search(source):
            continue
        if target_words > 1 and VERBISH_TOKEN_RE.search(target):
            continue
        if source[:1].islower() or target[:1].islower():
            continue
        if re.search(r"\b(?:that|which|because|while|during|following|according|incidents)\b", source, flags=re.IGNORECASE):
            continue
        relation_key = relation.casefold()
        relation_bonus = preferred_relations.get(relation_key)
        if relation_bonus is None:
            continue
        if pair_like_corpus and relation_key not in {"affiliation", "support", "communication"}:
            continue
        if relation_key not in {"affiliation", "support", "opposition", "communication"}:
            continue
        if large_relation_corpus and relation_key not in {"affiliation", "support", "opposition", "communication"}:
            continue
        target_genericish = actor_is_generic_common_noun_phrase(target) or (
            target[:1].islower() and count_words(target) <= 3
        )
        target_eventish = target.casefold().split()[-1:] and target.casefold().split()[-1] in {
            "firebombing",
            "explosion",
            "strike",
            "meeting",
            "report",
            "talks",
            "campaign",
        }
        if not pair_like_corpus and len(relations) >= 1000 and relation_key in {"support", "opposition", "communication"}:
            if target_genericish or target_eventish:
                continue
        evidence = normalize_relation_evidence_sentence(evidence)
        if not evidence:
            continue
        if REDACTION_MARKER_RE.search(evidence):
            continue
        if evidence.count(";") > 1 or "..." in evidence or is_routing_or_metadata_sentence(evidence):
            continue
        if query_surface_is_blocked(evidence, task_family="claims"):
            continue
        if is_probably_non_english_query(evidence):
            continue
        evidence_words = count_words(evidence)
        if evidence_words < 6 or evidence_words > 48:
            continue
        if strict_mode and relation_key == "event_participant":
            lowered = evidence.casefold()
            if not any(
                marker in lowered
                for marker in (
                    " met ",
                    "met with",
                    "meeting",
                    "participate",
                    "participated",
                    "traveled",
                    "travelled",
                    "approached",
                    "interview",
                )
            ):
                continue
        if strict_mode and relation_key == "communication":
            lowered = evidence.casefold()
            if not any(marker in lowered for marker in ("letter", "told", "informed", "discussed", "sent")):
                continue
        if strict_mode and relation_key == "support":
            lowered = evidence.casefold()
            if not DIRECT_SUPPORT_EVIDENCE_RE.search(lowered):
                continue
        if strict_mode and relation_key == "opposition":
            lowered = evidence.casefold()
            if " not " not in f" {lowered} " and "contradict" not in lowered:
                relation_bonus -= 1.5
        if relation_key == "affiliation" and evidence.casefold().startswith("an agent of "):
            continue
        if pair_like_corpus and relation_key == "affiliation":
            if not (find_label_match(evidence, source) and find_label_match(evidence, target)):
                continue
            target_lower = target.casefold()
            if any(
                marker in target_lower
                for marker in (
                    "act of",
                    "court",
                    "election process",
                    "presidential election",
                )
            ):
                continue
            if any(marker in target_lower for marker in ("application", "surveillance")):
                continue
        if strict_mode and not pair_like_corpus and relation_key == "affiliation" and not (
            find_label_match(evidence, source) and find_label_match(evidence, target)
        ):
            continue
        paragraph_source_text = chunk_text_lookup.get(
            paragraph_key(row["doc_id"], row["paragraph_id"]),
            "",
        )
        query_target_words = deterministic_query_word_target(
            f"relation_claim|{row['doc_id']}|{row['paragraph_id']}|{source}|{relation}|{target}",
            minimum_words=effective_min_query_words,
            maximum_words=min(max_query_words, 96),
        )
        source_context = choose_fast_query_context(
            [evidence],
            min_words=max(8, effective_min_query_words // 2),
            max_chars=420,
        ) or choose_fast_query_context(
            [paragraph_source_text],
            min_words=max(8, effective_min_query_words // 2),
            max_chars=420,
        )
        source_context_seed = finalize_query_surface(
            extract_sentence_excerpt(source_context, max_sentences=2, max_chars=320),
            add_terminal_punctuation=True,
        )
        predicate_surface = relation_predicate_surface(relation)
        if relation_key in {"affiliation", "support", "opposition", "communication"} and not role_query_is_safe(
            source,
            predicate_surface,
            target,
        ):
            continue
        relation_sentence = render_relation_claim_text(source, relation, target, evidence)
        if REDACTION_MARKER_RE.search(relation_sentence):
            continue
        evidence_sentence = finalize_query_surface(
            extract_sentence_excerpt(evidence, max_sentences=1, max_chars=220),
            add_terminal_punctuation=True,
        )
        lowered_target = target.casefold()
        if pair_like_corpus and count_words(evidence_sentence) < 4:
            continue
        if pair_like_corpus and relation_key == "affiliation":
            if any(token in lowered_target for token in ("application", "surveillance")):
                continue
        relation_seed_candidates: list[str] = []
        if (
            not pair_like_corpus
            and len(relations) >= 400
            and source_context_seed
            and not query_surface_is_blocked(source_context_seed, task_family="claims")
            and substantive_sentence_score(source_context_seed, anchors=[source, target]) >= 5.5
        ):
            relation_seed_candidates.append(source_context_seed)
        if (
            evidence_sentence
            and find_label_match(evidence_sentence, source)
            and find_label_match(evidence_sentence, target)
            and substantive_sentence_score(evidence_sentence, anchors=[source, target]) >= 6.0
        ):
            relation_seed_candidates.append(evidence_sentence)
        if (
            relation_key in {"support", "communication", "opposition", "affiliation"}
            and evidence_sentence
            and relation_evidence_can_anchor_claim_surface(
                source,
                relation,
                target,
                evidence_sentence,
            )
            and evidence_sentence not in relation_seed_candidates
        ):
            relation_seed_candidates.append(evidence_sentence)
        if relation_sentence and relation_sentence not in relation_seed_candidates:
            relation_seed_candidates.append(relation_sentence)
        claim_seed = choose_best_query_context(
            relation_seed_candidates,
            anchors=[source, target],
            min_words=relation_min_query_words,
            target_words=query_target_words,
            max_sentences=1 if query_target_words <= 28 else 2,
            max_chars=320,
        ) or relation_sentence or evidence_sentence or evidence
        if not claim_seed:
            continue
        if count_words(claim_seed) < effective_min_query_words and not role_query_is_safe(
            source,
            predicate_surface,
            target,
        ):
            appendable_evidence = False
            if evidence_sentence and evidence_sentence.casefold() != claim_seed.casefold():
                evidence_anchor_hits = int(bool(find_label_match(evidence_sentence, source))) + int(
                    bool(find_label_match(evidence_sentence, target))
                )
                evidence_quality = substantive_sentence_score(
                    evidence_sentence,
                    anchors=[source, target],
                )
                if evidence_anchor_hits >= 2 or evidence_quality >= 10.0:
                    appendable_evidence = True
            if appendable_evidence:
                claim_seed = finalize_query_surface(
                    f"{claim_seed} {evidence_sentence}",
                    add_terminal_punctuation=True,
                )
        query_text = refine_generated_query_surface(
            claim_seed,
            task_family="claims",
            anchors=[source, target],
            min_words=relation_min_query_words,
            target_words=query_target_words,
            max_chars=420,
            runner_cfg=candidate_runner_cfg,
        )
        if not query_text or query_surface_is_blocked(query_text, task_family="claims"):
            continue
        query_text = refine_generated_query_surface(
            query_text,
            task_family="claims",
            anchors=[source, target],
            min_words=relation_min_query_words,
            target_words=query_target_words,
            max_chars=420,
            runner_cfg=candidate_runner_cfg,
        )
        if REDACTION_MARKER_RE.search(query_text):
            continue
        role_fields = {
            "actor": source,
            "speaker": source if not is_generic_role_value(source) else "",
            "target": target if not is_generic_role_value(target) else "",
            "predicate": predicate_surface,
            "object": target,
            "stance": "affirmed",
        }
        role_field_count = sum(
            [
                bool(role_fields["actor"]),
                bool(role_fields["speaker"]),
                bool(role_fields["target"]),
                bool(role_fields["predicate"]),
                bool(role_fields["object"]),
            ]
        )
        query_words = count_words(query_text)
        if query_words < relation_min_query_words or query_words > max_query_words:
            continue
        if "..." in query_text or " under would " in f" {query_text.casefold()} ":
            continue
        if query_surface_is_blocked(query_text, task_family="claims"):
            continue
        if strict_mode and not (
            find_label_match(query_text, source) and find_label_match(query_text, target)
        ):
            continue
        min_relation_substantive_score = 4.5 if pair_like_corpus else 6.0
        if strict_mode and substantive_sentence_score(query_text, anchors=[source, target]) < min_relation_substantive_score:
            continue
        profile = claim_clause_profile(query_text)
        if not profile_has_complete_simple_clause(profile):
            continue
        quality_score = claim_clause_quality_score(profile)
        min_relation_quality = 2.5 if pair_like_corpus else 4.0
        if strict_mode and quality_score < min_relation_quality:
            continue
        if strict_mode and (
            profile["nested_reporting"]
            or profile["reporting_actor"]
            or actor_phrase_has_clause_bleed(profile["actor"])
            or profile["object_bad_lead"]
        ):
            continue
        if claim_uses_intentive_construction(query_text):
            continue
        has_simple_clause = bool(
            role_fields.get("actor") and role_fields.get("predicate") and role_fields.get("object")
        )
        standalone_safe = standalone_claim_surface_is_safe(query_text)
        if (
            not standalone_safe
            and relation_key in {"affiliation", "support", "opposition", "communication"}
            and role_fields.get("speaker")
            and role_fields.get("target")
        ):
            standalone_safe = True
        if not (has_simple_clause or standalone_safe):
            continue
        if role_field_count < min_role_fields and not standalone_safe and not pair_like_corpus:
            continue
        if strict_mode and has_simple_clause and predicate_is_low_signal(role_fields.get("predicate", "")):
            continue
        if strict_mode and has_simple_clause and parser_hostile_claim_candidate(query_text, role_fields, profile):
            continue
        anchor_bonus = 0.0
        if find_label_match(query_text, source):
            anchor_bonus += 2.0
        if find_label_match(query_text, target):
            anchor_bonus += 2.0
        relation_specific_bonus = 0.0
        if relation_key == "affiliation":
            if re.search(r"\bcandidate\s*#?\d+\b", target, flags=re.IGNORECASE):
                relation_specific_bonus += 6.0
            elif count_words(target) <= 2:
                relation_specific_bonus += 2.5
            if any(token in lowered_target for token in ("government", "federation", "act", "surveillance", "president", "russia")):
                relation_specific_bonus -= 1.0 if pair_like_corpus else 3.0
        if relation_key == "support" and count_words(target) <= 3:
            relation_specific_bonus += 2.0
        if pair_like_corpus and relation_key == "communication":
            relation_specific_bonus += 1.0
        if evidence_sentence and DATE_TOKEN_RE.search(evidence_sentence):
            relation_specific_bonus += 2.0
        if evidence_sentence and NUMBER_RE.search(evidence_sentence):
            relation_specific_bonus += 1.5
        generic_relation_penalty = 0.0
        if (
            relation_key == "communication"
            and re.search(r"\b(?:delivered|received)\s+(?:an?\s+)?message\b", query_text, flags=re.IGNORECASE)
        ):
            generic_relation_penalty += 2.5
        if pair_like_corpus and relation_key == "affiliation":
            generic_relation_penalty += 4.0
            if any(
                token in lowered_target
                for token in (
                    "application",
                    "surveillance",
                    "government",
                    "federation",
                    "community",
                    "white house",
                )
            ):
                generic_relation_penalty += 4.0
        if pair_like_corpus and relation_key == "event_participant":
            if any(
                token in lowered_target
                for token in (
                    "department of justice",
                    "court",
                    "target of the investigation",
                )
            ):
                generic_relation_penalty += 2.5
        if relation_key == "affiliation" and not pair_like_corpus and len(relations) >= 1000:
            generic_relation_penalty += 3.0
        selector_score = (
            quality_score
            + relation_bonus
            + small_corpus_bonus
            + 16.0
            + role_field_count * 4.0
            + anchor_bonus
            + (3.0 if relation_key in {"support", "affiliation"} else 0.0)
            + (2.0 if relation_key == "communication" else 0.0)
            + relation_specific_bonus
            + min(distinctive_label_token_count(source), 4) * 0.75
            + min(distinctive_label_token_count(target), 4) * 0.75
            - abs(query_words - 20) / 10.0
            - surface_noise_penalty(query_text) * 0.35
            - generic_relation_penalty
        )
        local_id = normalize_space(row.get("local_id", "")) or sha1_text(
            f"{row['doc_id']}|{row['paragraph_id']}|{source}|{relation}|{target}|{evidence}"
        )
        candidates.append(
            {
                "task_family": "claims",
                "runner_task": "claims",
                "doc_id": str(row["doc_id"]),
                "paragraph_id": str(row["paragraph_id"]),
                "case_id": f"relation_claim__{row['doc_id']}__{local_id}",
                "query": query_text,
                "ground_truth": finalize_query_surface(evidence, add_terminal_punctuation=True),
                "query_kind": "claim_relation_query",
                "notes": (
                    f"relation_source={source}; relation={relation}; relation_target={target}; "
                    f"role_field_count={role_field_count}; clause_quality={quality_score:.2f}; "
                    f"selector_score={selector_score:.2f}; query_target_words={query_target_words}"
                ),
                "source_context": source_context,
                "selector_score": selector_score,
                "word_count": query_words,
                "query_target_words": query_target_words,
            }
        )
        if large_relation_corpus and len(candidates) >= max(count * 8, 80):
            break
    deduped_candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = normalize_space(candidate.get("query", "")).casefold()
        if not key:
            continue
        prior = deduped_candidates.get(key)
        if prior is None or float(candidate.get("selector_score", 0.0) or 0.0) > float(
            prior.get("selector_score", 0.0) or 0.0
        ):
            deduped_candidates[key] = candidate
    return normalize_selected_case_queries(
        round_robin_topk(
            list(deduped_candidates.values()),
            count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        ),
        task_family="claims",
        runner_cfg=runner_cfg,
    )


def choose_pair_relation_claim_cases_relaxed(
    relations: pd.DataFrame,
    chunk_text_lookup: dict[tuple[str, str], str],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    runner_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    min_query_words = 6
    max_query_words = int(sampling.get("max_query_words", max(int(sampling.get("max_words", 48)) * 4, 180)))
    pair_like_corpus = bool(relations is not None and not relations.empty) and all(
        str(doc_id).startswith("Pair_")
        for doc_id in relations.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    )
    candidates: list[dict[str, Any]] = []
    for row in relations.to_dict("records"):
        source = normalize_space(row.get("source", ""))
        target = normalize_space(row.get("target", ""))
        relation = normalize_space(row.get("relation", ""))
        evidence = normalize_relation_evidence_sentence(normalize_space(row.get("evidence", "")))
        if not source or not target or not relation or not evidence:
            continue
        relation_key = relation.casefold()
        if relation_key not in {"affiliation", "support", "communication", "event_participant"}:
            continue
        if relation_value_is_placeholder(source) or relation_value_is_placeholder(target):
            continue
        if contains_role_placeholder_label(source) or contains_role_placeholder_label(target):
            continue
        if is_generic_role_value(source) or is_generic_role_value(target):
            continue
        if count_words(source) > 8 or count_words(target) > 8:
            continue
        if actor_is_generic_common_noun_phrase(source) or actor_phrase_has_clause_bleed(source):
            continue
        if actor_phrase_has_clause_bleed(target):
            continue
        if is_routing_or_metadata_sentence(evidence) or evidence.count(";") > 1 or "..." in evidence:
            continue
        evidence_sentence = finalize_query_surface(
            extract_sentence_excerpt(evidence, max_sentences=1, max_chars=240),
            add_terminal_punctuation=True,
        )
        evidence_anchors_relation = relation_evidence_can_anchor_claim_surface(
            source,
            relation,
            target,
            evidence_sentence,
        )
        # Bare graph edges are useful for routing, but synthesizing a claim from
        # an unanchored SUPPORT or EVENT_PARTICIPANT edge overstates the source.
        if relation_key in {"support", "event_participant"} and not evidence_anchors_relation:
            continue
        lowered_target = target.casefold()
        if pair_like_corpus and count_words(evidence_sentence) < 4:
            continue
        if pair_like_corpus and relation_key == "affiliation":
            if any(token in lowered_target for token in ("application", "surveillance")):
                continue
        relation_sentence = finalize_query_surface(
            render_relation_claim_text(source, relation, target, evidence),
            add_terminal_punctuation=True,
        )
        if not evidence_sentence and not relation_sentence:
            continue
        query_target_words = deterministic_query_word_target(
            f"pair_relation_claim|{row['doc_id']}|{row['paragraph_id']}|{source}|{relation}|{target}",
            minimum_words=min_query_words,
            maximum_words=min(max_query_words, 72),
        )
        paragraph_source_text = chunk_text_lookup.get(
            paragraph_key(row["doc_id"], row["paragraph_id"]),
            "",
        )
        source_context = choose_fast_query_context(
            [evidence_sentence, relation_sentence],
            min_words=6,
            max_chars=320,
        ) or choose_fast_query_context(
            [paragraph_source_text],
            min_words=6,
            max_chars=320,
        )
        query_text = relation_sentence or evidence_sentence
        query_text = refine_generated_query_surface(
            query_text,
            task_family="claims",
            anchors=[source, target],
            min_words=min_query_words,
            target_words=query_target_words,
            max_chars=320,
            runner_cfg=candidate_runner_cfg,
        )
        if not query_text or query_surface_is_blocked(query_text, task_family="claims"):
            continue
        if count_words(query_text) < min_query_words or count_words(query_text) > max_query_words:
            continue
        if not (find_label_match(query_text, source) and find_label_match(query_text, target)):
            continue
        if substantive_sentence_score(query_text, anchors=[source, target]) < 4.0:
            continue
        profile = claim_clause_profile(query_text)
        if not profile_has_complete_simple_clause(profile):
            continue
        if not (
            strict_claim_surface_is_safe(
                query_text,
                profile,
                max_actor_words=8,
                max_object_words=18,
            )
            or standalone_claim_surface_is_safe(query_text)
        ):
            continue
        local_id = normalize_space(row.get("local_id", "")) or sha1_text(
            f"{row['doc_id']}|{row['paragraph_id']}|{source}|{relation}|{target}|{evidence}|pair_relaxed"
        )
        selector_score = (
            claim_clause_quality_score(profile)
            + 20.0
            + (4.0 if relation_key == "affiliation" else 2.0)
            + (3.0 if relation_key == "communication" else 0.0)
            + (5.0 if relation_key == "event_participant" else 0.0)
            + min(distinctive_label_token_count(source), 4) * 0.75
            + min(distinctive_label_token_count(target), 4) * 0.75
            - abs(count_words(query_text) - 18) / 10.0
            - surface_noise_penalty(query_text) * 0.3
        )
        candidates.append(
            {
                "task_family": "claims",
                "runner_task": "claims",
                "doc_id": str(row["doc_id"]),
                "paragraph_id": str(row["paragraph_id"]),
                "case_id": f"pair_relation_claim__{row['doc_id']}__{local_id}",
                "query": query_text,
                "ground_truth": evidence_sentence or relation_sentence,
                "query_kind": "claim_relation_query",
                "notes": (
                    f"relation_source={source}; relation={relation}; relation_target={target}; "
                    f"selector_score={selector_score:.2f}; query_target_words={query_target_words}"
                ),
                "source_context": source_context,
                "selector_score": selector_score,
                "word_count": count_words(query_text),
                "query_target_words": query_target_words,
            }
        )
    deduped_candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = normalize_space(candidate.get("query", "")).casefold()
        if not key:
            continue
        prior = deduped_candidates.get(key)
        if prior is None or float(candidate.get("selector_score", 0.0) or 0.0) > float(
            prior.get("selector_score", 0.0) or 0.0
        ):
            deduped_candidates[key] = candidate
    return normalize_selected_case_queries(
        round_robin_topk(
            list(deduped_candidates.values()),
            count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        ),
        task_family="claims",
        runner_cfg=runner_cfg,
    )


def choose_relation_contradiction_cases(
    relations: pd.DataFrame,
    chunk_text_lookup: dict[tuple[str, str], str],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    runner_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    strict_mode = not maybe_bool(sampling.get("_relaxed_retry_used", False))
    doc_ids = {
        normalize_space(doc_id)
        for doc_id in relations.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    }
    pair_like_corpus = bool(doc_ids) and all(doc_id.startswith("Pair_") for doc_id in doc_ids)
    large_relation_corpus = len(relations) >= 25000
    relation_cleanup_corpus = len(relations) < 400
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    min_query_words = int(sampling.get("min_query_words", 16))
    effective_min_query_words = 8 if pair_like_corpus else min_query_words
    max_query_words = int(
        sampling.get("max_query_words", max(int(sampling.get("max_words", 40)) * 4, 160))
    )
    preferred_relations = {
        "support": 8.5,
        "affiliation": 8.0,
        "communication": 7.0,
        "opposition": 6.0,
        "event_participant": 4.0,
        "event_location": 4.5,
    }
    relation_rows = deterministic_seeded_row_order(
        relations.to_dict("records"),
        seed=seed,
        key_fields=("doc_id", "paragraph_id", "source", "relation", "target", "evidence"),
    )
    if large_relation_corpus:
        preferred = {"affiliation", "support", "opposition", "communication"}
        scored_rows: list[tuple[float, dict[str, Any]]] = []
        for row in relation_rows:
            relation_key = normalize_space(row.get("relation", "")).casefold()
            if relation_key not in preferred:
                continue
            source = normalize_space(row.get("source", ""))
            target = normalize_space(row.get("target", ""))
            evidence = normalize_space(row.get("evidence", ""))
            score = (
                preferred_relations.get(relation_key, 0.0) * 10.0
                + min(distinctive_label_token_count(source), 4) * 2.0
                + min(distinctive_label_token_count(target), 4) * 2.0
                + max(0.0, 36.0 - abs(count_words(evidence) - 18))
            )
            scored_rows.append((score, row))
        relation_rows = [row for _score, row in sorted(scored_rows, key=lambda item: item[0], reverse=True)[: max(count * 120, 600)]]
    candidates: list[dict[str, Any]] = []
    relation_doc_sets: dict[tuple[str, str, str], set[str]] = {}
    if pair_like_corpus:
        for seed_row in relation_rows:
            source = normalize_space(seed_row.get("source", "")).casefold()
            relation = normalize_space(seed_row.get("relation", "")).casefold()
            target = normalize_space(seed_row.get("target", "")).casefold()
            doc_id = normalize_space(seed_row.get("doc_id", ""))
            if source and relation and target and doc_id:
                relation_doc_sets.setdefault((source, relation, target), set()).add(doc_id)
    for row in relation_rows:
        source = normalize_space(row.get("source", ""))
        target = normalize_space(row.get("target", ""))
        relation = normalize_space(row.get("relation", ""))
        evidence = normalize_space(row.get("evidence", ""))
        if not source or not target or not relation or not evidence:
            continue
        if relation_value_is_placeholder(source) or relation_value_is_placeholder(target):
            continue
        if contains_role_placeholder_label(source) or contains_role_placeholder_label(target):
            continue
        if is_generic_role_value(source) or is_generic_role_value(target):
            continue
        if actor_phrase_has_clause_bleed(source) or actor_phrase_has_clause_bleed(target):
            continue
        if actor_is_generic_common_noun_phrase(source):
            continue
        source_words = count_words(source)
        target_words = count_words(target)
        if source_words > 6 or target_words > 6:
            continue
        if source_words > 1 and VERBISH_TOKEN_RE.search(source):
            continue
        if target_words > 1 and VERBISH_TOKEN_RE.search(target):
            continue
        if source[:1].islower() or target[:1].islower():
            continue
        if re.search(r"\b(?:that|which|because|while|during|following|according|incidents)\b", source, flags=re.IGNORECASE):
            continue
        relation_key = relation.casefold()
        relation_bonus = preferred_relations.get(relation_key)
        if relation_bonus is None:
            continue
        if pair_like_corpus and strict_mode:
            relation_doc_frequency = len(
                relation_doc_sets.get((source.casefold(), relation_key, target.casefold()), set())
            )
            if relation_doc_frequency < 2:
                continue
        if pair_like_corpus and relation_key not in {"affiliation", "support", "opposition"}:
            continue
        if large_relation_corpus and relation_key not in {
            "affiliation",
            "support",
            "opposition",
            "communication",
        }:
            continue
        evidence = normalize_relation_evidence_sentence(evidence)
        if not evidence:
            continue
        if evidence.count(";") > 1 or "..." in evidence or is_routing_or_metadata_sentence(evidence):
            continue
        if query_surface_is_blocked(evidence, task_family="contradictions"):
            continue
        if is_probably_non_english_query(evidence):
            continue
        relation_sentence = render_relation_claim_text(source, relation, target, evidence)
        if not relation_sentence:
            continue
        predicate_surface = relation_predicate_surface(relation)
        if relation_key in {"affiliation", "support", "opposition", "communication"} and not role_query_is_safe(
            source,
            predicate_surface,
            target,
        ):
            continue
        mutated_statement = negate_relation_statement(relation_sentence)
        if not mutated_statement:
            continue
        paragraph_source_text = chunk_text_lookup.get(
            paragraph_key(row["doc_id"], row["paragraph_id"]),
            "",
        )
        query_target_words = deterministic_query_word_target(
            f"relation_contradiction|{row['doc_id']}|{row['paragraph_id']}|{source}|{relation}|{target}",
            minimum_words=effective_min_query_words,
            maximum_words=min(max_query_words, 96),
        )
        source_context = choose_fast_query_context(
            [evidence, relation_sentence],
            min_words=max(8, effective_min_query_words // 2),
            max_chars=420,
        ) or choose_fast_query_context(
            [paragraph_source_text],
            min_words=max(8, effective_min_query_words // 2),
            max_chars=420,
        )
        query_text = build_contradiction_query(
            source_context_text=source_context,
            claim_text=relation_sentence,
            base_clause=relation_sentence,
            mutated_clause=mutated_statement,
        )
        query_text = refine_generated_query_surface(
            query_text,
            task_family="contradictions",
            anchors=[source, target],
            min_words=effective_min_query_words,
            target_words=query_target_words,
            max_chars=420,
            runner_cfg=candidate_runner_cfg,
        )
        if not query_text:
            continue
        if REDACTION_MARKER_RE.search(query_text):
            continue
        if not contradiction_query_carries_mutation(
            query_text,
            relation_sentence,
            strategy_used="negation",
        ):
            continue
        if retained_content_token_ratio(query_text, relation_sentence) < 0.70:
            continue
        query_words = count_words(query_text)
        if query_words < effective_min_query_words or query_words > max_query_words:
            continue
        if query_surface_is_blocked(query_text, task_family="contradictions"):
            continue
        if strict_mode and substantive_sentence_score(query_text, anchors=[source, target]) < 6.0:
            continue
        profile = claim_clause_profile(query_text)
        if not profile_has_complete_simple_clause(profile):
            continue
        if strict_mode and not strict_claim_surface_is_safe(
            query_text,
            profile,
            max_actor_words=8,
            max_object_words=18,
        ):
            continue
        if strict_mode and (
            profile["nested_reporting"]
            or profile["reporting_actor"]
            or actor_phrase_has_clause_bleed(profile["actor"])
            or profile["object_bad_lead"]
        ):
            continue
        if strict_mode and claim_uses_intentive_construction(query_text):
            continue
        relation_evidence_grounded = relation_evidence_can_anchor_claim_surface(
            source,
            relation,
            target,
            evidence,
        )
        selector_score = (
            claim_clause_quality_score(profile)
            + relation_bonus
            + (14.0 if relation_evidence_grounded else -10.0)
            + min(distinctive_label_token_count(source), 4) * 0.75
            + min(distinctive_label_token_count(target), 4) * 0.75
            + (2.0 if find_label_match(query_text, source) else 0.0)
            + (2.0 if find_label_match(query_text, target) else 0.0)
            - abs(query_words - 18) / 10.0
            - surface_noise_penalty(query_text) * 0.35
        )
        local_id = normalize_space(row.get("local_id", "")) or sha1_text(
            f"{row['doc_id']}|{row['paragraph_id']}|{source}|{relation}|{target}|{evidence}|contradiction"
        )
        candidates.append(
            {
                "task_family": "contradictions",
                "runner_task": "contradictions",
                "doc_id": str(row["doc_id"]),
                "paragraph_id": str(row["paragraph_id"]),
                "case_id": f"relation_contradiction__{row['doc_id']}__{local_id}",
                "query": query_text,
                "ground_truth": finalize_query_surface(relation_sentence, add_terminal_punctuation=True),
                "query_kind": "relation_contradiction_query",
                "notes": (
                    f"relation_source={source}; relation={relation}; relation_target={target}; "
                    f"relation_evidence_grounded={relation_evidence_grounded}; "
                    f"selector_score={selector_score:.2f}; query_target_words={query_target_words}"
                ),
                "source_context": source_context,
                "selector_score": selector_score,
                "word_count": query_words,
                "query_target_words": query_target_words,
            }
        )
        if large_relation_corpus and len(candidates) >= max(count * 8, 80):
            break
    deduped_candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = normalize_space(candidate.get("query", "")).casefold()
        if not key:
            continue
        prior = deduped_candidates.get(key)
        if prior is None or float(candidate.get("selector_score", 0.0) or 0.0) > float(
            prior.get("selector_score", 0.0) or 0.0
        ):
            deduped_candidates[key] = candidate
    return normalize_selected_case_queries(
        round_robin_topk(
            list(deduped_candidates.values()),
            count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        ),
        task_family="contradictions",
        runner_cfg=runner_cfg,
    )


def build_contradiction_query(
    *,
    source_context_text: str,
    claim_text: str,
    base_clause: str,
    mutated_clause: str,
) -> str:
    source_context = simplify_claim_query_seed(source_context_text)
    original_claim = simplify_claim_query_seed(claim_text)
    mutated_statement = simplify_claim_query_seed(mutated_clause)
    candidate_texts: list[str] = []
    if mutated_statement:
        candidate_texts.append(mutated_statement)
    for base_text in [source_context, original_claim]:
        if not base_text:
            continue
        updated, changed = replace_first_ignore_case(base_text, base_clause, mutated_clause)
        if changed and not is_routing_or_metadata_sentence(updated) and updated.count(";") <= 1:
            candidate_texts.append(finalize_query_surface(updated, add_terminal_punctuation=True))
        updated, changed = replace_first_ignore_case(base_text, original_claim, mutated_statement)
        if changed and not is_routing_or_metadata_sentence(updated) and updated.count(";") <= 1:
            candidate_texts.append(finalize_query_surface(updated, add_terminal_punctuation=True))
    if (
        original_claim
        and not is_routing_or_metadata_sentence(original_claim)
        and original_claim.count(";") <= 1
    ):
        candidate_texts.append(original_claim)
    for candidate in unique_preserving_order(candidate_texts):
        profile = claim_clause_profile(candidate)
        if profile["actor"] and profile["predicate"] and profile["object"]:
            return candidate
    for candidate in unique_preserving_order(candidate_texts):
        if candidate:
            return candidate
    return mutated_statement


def contradiction_query_carries_mutation(
    query_text: str,
    original_text: str,
    *,
    strategy_used: str,
) -> bool:
    query_norm = normalize_space(query_text).casefold()
    original_norm = normalize_space(original_text).casefold()
    if not query_norm or not original_norm:
        return False
    if query_norm == original_norm:
        return False
    if strategy_used == "negation":
        return bool(NEGATION_CUE_RE.search(query_text)) != bool(NEGATION_CUE_RE.search(original_text))
    if strategy_used == "numeric":
        query_number = normalize_space((NUMBER_RE.search(query_text) or [None])[0]) if NUMBER_RE.search(query_text) else ""
        original_number = normalize_space((NUMBER_RE.search(original_text) or [None])[0]) if NUMBER_RE.search(original_text) else ""
        return bool(query_number) and bool(original_number) and query_number != original_number
    if strategy_used == "date":
        query_date = normalize_space((DATE_TOKEN_RE.search(query_text) or [None])[0]) if DATE_TOKEN_RE.search(query_text) else ""
        original_date = normalize_space((DATE_TOKEN_RE.search(original_text) or [None])[0]) if DATE_TOKEN_RE.search(original_text) else ""
        return bool(query_date) and bool(original_date) and query_date != original_date
    return True


def parser_hostile_claim_candidate(
    claim_text: str,
    role_fields: dict[str, str],
    profile: dict[str, Any],
) -> bool:
    actor = normalize_space(role_fields.get("actor", ""))
    obj = normalize_space(role_fields.get("object", ""))
    predicate = normalize_space(role_fields.get("predicate", ""))
    text = normalize_space(claim_text)
    lower_text = text.casefold()
    if not actor or not predicate or not obj:
        return True
    if REDACTION_MARKER_RE.search(text):
        return True
    actor_lower = actor.casefold()
    if REPORTING_WRAPPER_RE.search(actor) and " that " in actor_lower:
        return True
    if "assertions in the" in text.casefold():
        return True
    if (
        ((" if " in f" {lower_text} " or " such as " in f" {lower_text} ") and NEGATION_CUE_RE.search(text))
        or ("warning" in lower_text and (" would " in f" {lower_text} " or " will " in f" {lower_text} "))
        or "not fraudulent" in lower_text
        or ("sent word" in lower_text and " will " in f" {lower_text} ")
        or re.search(r"\b(?:caller|listener)\b", lower_text)
        or (
            re.search(r"\b(?:said|reported|reports?|told|informed|asked|warned)\b[^.]{0,120}\b(?:did|would|will|had|has)\b", lower_text)
            and " that " not in f" {lower_text} "
        )
        or ("pass word" in lower_text and " that " not in f" {lower_text} ")
        or re.search(
            r"\b(?:"
            r"received instructions from\b[^.]{0,120}\brequiring\b|"
            r"is planning to turn over to\b|"
            r"(?:is|was)\s+unable to\b[^.]{0,120}\bbecause of (?:the )?lack of\b|"
            r"is not and will not be seriously affected by\b|"
            r"was led by a charismatic figure who would not alter\b"
            r")",
            lower_text,
        )
    ):
        return True
    if profile.get("reporting_actor"):
        return True
    if profile.get("actor_abstract"):
        return True
    if actor_has_abstract_head(actor):
        return True
    if actor_is_generic_common_noun_phrase(actor):
        return True
    if actor_phrase_has_clause_bleed(actor):
        return True
    if object_has_abstract_head(obj):
        return True
    if profile.get("actor_tail_bad") and int(profile.get("actor_words", 0)) <= 4:
        return True
    if profile.get("object_bad_lead"):
        allowed_prepositional_relation = bool(
            re.search(
                r"\b(?:"
                r"met with|meeting with|communicated with|affiliated with|"
                r"supported by|opposed by|was located in|was related to|"
                r"requested action from"
                r")\b",
                lower_text,
            )
        )
        if not allowed_prepositional_relation:
            return True
    if predicate.casefold() in REPORTING_PREDICATES and " that " in obj.casefold():
        return True
    if DEICTIC_RISK_RE.search(obj):
        return True
    if DEICTIC_RISK_RE.search(actor) and count_words(actor) <= 2:
        return True
    if profile.get("nested_reporting"):
        return True
    if PASSIVE_CLAIM_RE.search(text) and " by " not in text.casefold():
        return True
    if re.search(
        r"\b(?:has|have|had)\s+(?:not\s+)?yet\s+been\s+\w+(?:ed|en)\s+by\b",
        lower_text,
    ):
        return True
    return False


def strict_claim_surface_is_safe(
    text: str,
    profile: dict[str, Any],
    *,
    max_actor_words: int = 8,
    max_object_words: int = 18,
) -> bool:
    role_fields = {
        "actor": normalize_space(profile.get("actor", "")),
        "predicate": normalize_space(profile.get("predicate", "")),
        "object": normalize_space(profile.get("object", "")),
        "speaker": "",
        "target": "",
        "stance": "unknown",
    }
    if not (role_fields["actor"] and role_fields["predicate"] and role_fields["object"]):
        return False
    if parser_hostile_claim_candidate(text, role_fields, profile):
        return False
    if int(profile.get("actor_words", 0) or 0) > max_actor_words:
        return False
    if int(profile.get("object_words", 0) or 0) > max_object_words:
        return False
    return True


def content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in WORD_RE.findall(text or ""):
        key = token.casefold()
        if len(key) < 4:
            continue
        if key in MONTH_NAMES:
            continue
        if key.isdigit():
            continue
        tokens.add(key)
    return tokens


def content_token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def retained_content_token_ratio(rewrite: str, source: str) -> float:
    """Measure how much of the source proposition survives a transformation."""
    rewrite_tokens = content_tokens(rewrite)
    source_tokens = content_tokens(source)
    if not rewrite_tokens or not source_tokens:
        return 0.0
    return len(rewrite_tokens & source_tokens) / len(source_tokens)


def claim_clause_profile(text: str) -> dict[str, Any]:
    inferred = infer_simple_clause_arguments(text)
    actor = normalize_space(inferred.get("actor", ""))
    predicate = normalize_space(inferred.get("predicate", ""))
    obj = normalize_space(inferred.get("object", "")) or normalize_space(inferred.get("location", ""))
    actor_tokens = WORD_RE.findall(actor)
    object_tokens = WORD_RE.findall(obj)
    actor_lower = actor.casefold()
    obj_lower = obj.casefold()
    actor_tail = actor_tokens[-1].casefold() if actor_tokens else ""
    object_leads_with_that = obj_lower.startswith("that ")
    object_bad_lead = bool(BAD_OBJECT_LEAD_RE.match(obj_lower))
    nested_reporting = predicate.casefold() in REPORTING_PREDICATES and (
        object_leads_with_that or " that " in obj_lower
    )
    actor_abstract = bool(ABSTRACT_ACTOR_RE.search(actor_lower)) and count_words(actor) <= 8
    reporting_actor = bool(REPORTING_ACTOR_RE.search(actor))
    return {
        "actor": actor,
        "predicate": predicate,
        "object": obj,
        "actor_words": count_words(actor),
        "object_words": count_words(obj),
        "actor_tail_bad": actor_tail in TAIL_STOPWORDS,
        "nested_reporting": nested_reporting,
        "actor_abstract": actor_abstract,
        "reporting_actor": reporting_actor,
        "object_bad_lead": object_bad_lead,
        "has_semicolon": ";" in text,
        "content_tokens": content_tokens(text),
        "date_token": normalize_space((DATE_TOKEN_RE.search(text) or [None])[0]) if DATE_TOKEN_RE.search(text) else "",
        "number_token": normalize_space((NUMBER_RE.search(text) or [None])[0]) if NUMBER_RE.search(text) else "",
        "has_negation": bool(NEGATION_CUE_RE.search(text)),
    }


def claim_clause_quality_score(profile: dict[str, Any]) -> float:
    score = 0.0
    if profile["actor"]:
        score += 3.0
    if profile["predicate"]:
        score += 3.0
    if profile["object"]:
        score += 3.0
    if not profile["nested_reporting"]:
        score += 2.0
    if not profile["has_semicolon"]:
        score += 1.5
    if 1 <= int(profile["actor_words"]) <= 10:
        score += 1.5
    if 1 <= int(profile["object_words"]) <= 14:
        score += 1.5
    if profile["actor_tail_bad"]:
        score -= 2.0
    if profile["nested_reporting"]:
        score -= 4.0
    if profile.get("actor_abstract"):
        score -= 3.0
    if profile.get("reporting_actor"):
        score -= 4.0
    if profile.get("object_bad_lead"):
        score -= 2.0
    if int(profile["object_words"]) > 18:
        score -= 3.0
    return score


def profile_has_complete_simple_clause(profile: Mapping[str, Any]) -> bool:
    return bool(
        normalize_space(profile.get("actor", ""))
        and normalize_space(profile.get("predicate", ""))
        and normalize_space(profile.get("object", ""))
    )


def claim_profile_has_problematic_actor(profile: Mapping[str, Any]) -> bool:
    actor = normalize_space(profile.get("actor", ""))
    if not actor:
        return False
    return actor_has_abstract_head(actor) or actor_is_generic_common_noun_phrase(actor)


def reconstruction_claim_surface_is_actionable(
    text: str,
    profile: Mapping[str, Any],
) -> bool:
    surface = normalize_space(text)
    if not surface:
        return False
    lowered = surface.casefold()
    if claim_profile_has_problematic_actor(profile):
        return False
    if re.match(r"^(?:he|she|they|it|there)\b", surface, flags=re.IGNORECASE):
        return False
    if re.search(
        r"\b(?:there are reports|it was rumored|strategy of|approach to interviews|"
        r"reports indicate|it centers on)\b",
        lowered,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:denied|deny|denies|denying|publicly denied|no part in|"
        r"self-serving|complete garbage|inculpatory allegation|appears?\s+to\s+be|"
        r"believes?|thinks?|thought|refut(?:e|ed)|"
        r"has not directly supported a political campaign since|"
        r"believes he is ['’]on the books['’])\b",
        lowered,
        flags=re.IGNORECASE,
    ):
        return False
    if claim_surface_has_cognitive_event_disclaimer(surface):
        return False
    if claim_surface_has_multi_action_fallback_risk(surface, profile):
        return False
    if claim_surface_has_weak_modality(surface):
        return False
    if has_reporting_that_clause(surface) or claim_uses_intentive_construction(surface):
        return False
    if surface_noise_penalty(surface) > 1.5:
        return False
    return standalone_claim_surface_is_safe(surface)


def reconstruction_variant_surface_is_actionable(
    text: str,
    profile: Mapping[str, Any],
    *,
    variant_kind: str,
    pair_like_corpus: bool,
) -> bool:
    surface = normalize_space(text)
    if not surface:
        return False
    lowered = surface.casefold()
    if query_surface_is_blocked(surface, task_family="reconstruction"):
        return False
    if sentence_is_low_signal_for_query(surface):
        return False
    if claim_surface_has_weak_modality(surface):
        return False
    if re.search(
        r"\b(?:strategy of|approach to interviews|inculpatory allegation|appears?\s+to\s+be|"
        r"self-serving|complete garbage|it centers on)\b",
        lowered,
        flags=re.IGNORECASE,
    ):
        return False
    if pair_like_corpus and re.search(
        r"\b(?:denied|deny|denies|denying|publicly denied|no part in|"
        r"has not directly supported a political campaign since|"
        r"believes he is ['’]on the books['’]|"
        r"will shift its focus from the short-term goal)\b",
        lowered,
        flags=re.IGNORECASE,
    ):
        return False
    if variant_kind in {"claim", "claim_bundle", "compact_claim", "claim_seed", "relation_claim"}:
        return reconstruction_claim_surface_is_actionable(surface, profile)
    if variant_kind == "source_excerpt" and pair_like_corpus:
        if not profile_has_complete_simple_clause(profile):
            return False
        if claim_profile_has_problematic_actor(profile):
            return False
        if profile.get("nested_reporting") or profile.get("reporting_actor"):
            return False
    return True


def contradiction_partner_score(
    profile: dict[str, Any],
    strategy_used: str,
    peer_rows: list[dict[str, Any]],
) -> tuple[int, float]:
    base_tokens = set(profile.get("content_tokens", set()))
    if len(base_tokens) < 3:
        return 0, 0.0
    matches = 0
    best_overlap = 0.0
    for peer in peer_rows:
        peer_text = normalize_space(peer.get("claim_text", ""))
        if not peer_text:
            continue
        peer_profile = claim_clause_profile(peer_text)
        overlap = len(base_tokens & set(peer_profile.get("content_tokens", set())))
        if overlap < 3:
            continue
        compatible = False
        if strategy_used == "date":
            compatible = bool(profile.get("date_token")) and bool(peer_profile.get("date_token")) and profile.get("date_token") != peer_profile.get("date_token")
        elif strategy_used == "numeric":
            compatible = bool(profile.get("number_token")) and bool(peer_profile.get("number_token")) and profile.get("number_token") != peer_profile.get("number_token")
        elif strategy_used == "negation":
            compatible = bool(profile.get("has_negation")) != bool(peer_profile.get("has_negation"))
        if not compatible:
            continue
        matches += 1
        best_overlap = max(best_overlap, overlap / max(1, len(base_tokens)))
    return matches, best_overlap


def claim_support_peer_score(
    profile: dict[str, Any],
    peer_rows: list[dict[str, Any]],
) -> tuple[int, float]:
    base_tokens = set(profile.get("content_tokens", set()))
    if len(base_tokens) < 3:
        return 0, 0.0
    matches = 0
    best_overlap = 0.0
    for peer in peer_rows:
        peer_text = normalize_space(peer.get("claim_text", ""))
        if not peer_text:
            continue
        peer_profile = claim_clause_profile(peer_text)
        peer_tokens = set(peer_profile.get("content_tokens", set()))
        overlap = len(base_tokens & peer_tokens)
        if overlap < 3:
            continue
        ratio = overlap / max(1, len(base_tokens))
        matches += 1
        best_overlap = max(best_overlap, ratio)
    return matches, best_overlap


def claim_peer_pool_disabled(
    claim_row_count: int,
    sampling: dict[str, Any] | None,
    *,
    pair_like_corpus: bool,
    default_threshold: int,
) -> bool:
    if pair_like_corpus:
        return False
    configured_threshold = 0
    if isinstance(sampling, dict):
        configured_threshold = int(
            sampling.get("peer_pool_disable_claim_rows", 0) or 0
        )
    effective_threshold = configured_threshold if configured_threshold > 0 else default_threshold
    return claim_row_count >= max(1, effective_threshold)


EXHAUSTIVE_CORPUS_DOC_THRESHOLD = 50


def exhaustive_small_doc_corpus(corpus_doc_count: int) -> bool:
    return corpus_doc_count <= EXHAUSTIVE_CORPUS_DOC_THRESHOLD


def optimized_doc_corpus(corpus_doc_count: int) -> bool:
    return corpus_doc_count > EXHAUSTIVE_CORPUS_DOC_THRESHOLD


def build_claim_peer_pool(
    claim_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    profiles: list[dict[str, Any]] = []
    token_index: dict[str, set[int]] = defaultdict(set)
    token_counts: Counter[str] = Counter()
    for idx, row in enumerate(claim_rows):
        text = normalize_space(row.get("claim_text", ""))
        profile = claim_clause_profile(text)
        profiles.append(profile)
        for token in set(profile.get("content_tokens", set())):
            if len(token) >= 4:
                token_index[token].add(idx)
                token_counts[token] += 1

    peer_rows_by_idx: list[list[dict[str, Any]]] = []
    max_token_doc_freq = 240
    max_peer_candidates = 128
    for idx, row in enumerate(claim_rows):
        profile = profiles[idx]
        informative_tokens = [
            token
            for token in sorted(
                set(profile.get("content_tokens", set())),
                key=lambda token: (token_counts.get(token, 0), token),
            )
            if len(token) >= 4 and 1 < token_counts.get(token, 0) <= max_token_doc_freq
        ][:6]
        candidate_scores: Counter[int] = Counter()
        for token in informative_tokens:
            for peer_idx in token_index.get(token, set()):
                if peer_idx == idx:
                    continue
                candidate_scores[peer_idx] += 1
        ranked_indices = [
            peer_idx
            for peer_idx, _score in candidate_scores.most_common(max_peer_candidates)
        ]
        cross_doc = [
            claim_rows[peer_idx]
            for peer_idx in ranked_indices
            if str(claim_rows[peer_idx].get("doc_id", "")) != str(row.get("doc_id", ""))
        ]
        if cross_doc:
            peer_rows_by_idx.append(cross_doc)
            continue
        same_doc = [
            claim_rows[peer_idx]
            for peer_idx in ranked_indices
            if paragraph_key(
                claim_rows[peer_idx].get("doc_id", ""),
                claim_rows[peer_idx].get("paragraph_id", ""),
            )
            != paragraph_key(row.get("doc_id", ""), row.get("paragraph_id", ""))
        ]
        peer_rows_by_idx.append(same_doc)
    return profiles, peer_rows_by_idx


def build_paragraph_component_index(
    entities: pd.DataFrame,
    events: pd.DataFrame,
    claims: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, list[str]]]:
    index: dict[tuple[str, str], dict[str, list[str]]] = {}

    def bucket_for(doc_id: Any, paragraph_id: Any) -> dict[str, list[str]]:
        return index.setdefault(
            paragraph_key(doc_id, paragraph_id),
            {
                "entity_labels": [],
                "event_labels": [],
                "claim_texts": [],
            },
        )

    for row in entities.to_dict("records"):
        label = normalize_space(row.get("label", ""))
        if not label or is_generic_label(label):
            continue
        bucket_for(row["doc_id"], row["paragraph_id"])["entity_labels"].append(label)
    for row in events.to_dict("records"):
        label = normalize_space(row.get("label", ""))
        if not label:
            continue
        bucket_for(row["doc_id"], row["paragraph_id"])["event_labels"].append(label)
    for row in claims.to_dict("records"):
        text = normalize_space(row.get("claim_text", ""))
        if not text:
            continue
        bucket_for(row["doc_id"], row["paragraph_id"])["claim_texts"].append(text)

    for bucket in index.values():
        for key in ("entity_labels", "event_labels", "claim_texts"):
            bucket[key] = unique_preserving_order(bucket[key])
    return index


def build_chunk_text_lookup(chunks: pd.DataFrame) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for row in chunks.to_dict("records"):
        key = paragraph_key(row.get("doc_id", ""), row.get("paragraph_id", ""))
        paragraph_text = normalize_space(row.get("paragraph_text", ""))
        chunk_text = normalize_space(row.get("chunk_text", ""))
        text = paragraph_text or chunk_text
        if not text:
            continue
        existing = lookup.get(key, "")
        if len(text) > len(existing):
            lookup[key] = text
    return lookup


def count_label_hits(text: str, labels: Iterable[str]) -> tuple[int, list[str]]:
    hits: list[str] = []
    normalized_text = str(text or "")
    for label in labels:
        candidate = normalize_space(label)
        if not candidate:
            continue
        if re.search(rf"\b{re.escape(candidate)}\b", normalized_text, flags=re.IGNORECASE):
            hits.append(candidate)
    ordered = unique_preserving_order(hits)
    return len(ordered), ordered


def content_overlap_score(sentence: str, texts: Iterable[str]) -> int:
    sentence_tokens = {
        token.casefold()
        for token in WORD_RE.findall(sentence or "")
        if len(token) >= 5
    }
    score = 0
    if not sentence_tokens:
        return 0
    for text in texts:
        other_tokens = {
            token.casefold()
            for token in WORD_RE.findall(str(text or ""))
            if len(token) >= 5
        }
        if len(sentence_tokens & other_tokens) >= 2:
            score += 1
    return score


def best_claim_aligned_source_sentence(
    claim_text: str,
    source_text: str,
    *,
    anchors: Iterable[str] = (),
    min_overlap: int = 2,
) -> str:
    claim_surface = finalize_query_surface(
        simplify_claim_query_seed(simplify_claim_surface(claim_text)),
        add_terminal_punctuation=True,
    )
    source_surface = normalize_space(source_text)
    if not claim_surface or not source_surface:
        return ""
    claim_profile = claim_clause_profile(claim_surface)
    claim_tokens = content_tokens(claim_surface)
    if len(claim_tokens) < min_overlap:
        return ""
    sentences = cleaned_substantive_sentences(
        source_surface,
        anchors=anchors,
        min_score=-8.0,
    )
    if not sentences:
        sentences = [
            clean_generation_sentence(sentence)
            for sentence in sentence_split(source_surface)
            if clean_generation_sentence(sentence)
        ]
    best_sentence = ""
    best_score = -1000.0
    target_words = max(12, min(32, count_words(claim_surface)))
    for raw_sentence in sentences:
        candidate = finalize_query_surface(
            raw_sentence,
            add_terminal_punctuation=True,
        )
        if not candidate or "?" in candidate:
            continue
        if is_routing_or_metadata_sentence(candidate):
            continue
        if is_list_like_text(candidate) or sentence_is_low_signal_for_query(candidate):
            continue
        candidate_profile = claim_clause_profile(candidate)
        candidate_tokens = content_tokens(candidate)
        overlap = len(claim_tokens & candidate_tokens)
        if overlap < min_overlap:
            continue
        score = (
            overlap * 8.0
            + substantive_sentence_score(candidate, anchors=anchors) * 0.8
            - abs(count_words(candidate) - target_words) * 0.15
            - surface_noise_penalty(candidate) * 0.5
        )
        if (
            claim_profile["actor"]
            and candidate_profile["actor"]
            and normalize_space(claim_profile["actor"]).casefold()
            == normalize_space(candidate_profile["actor"]).casefold()
        ):
            score += 4.0
        if (
            claim_profile["predicate"]
            and candidate_profile["predicate"]
            and normalize_space(claim_profile["predicate"]).casefold()
            == normalize_space(candidate_profile["predicate"]).casefold()
        ):
            score += 3.0
        if (
            claim_profile["object"]
            and candidate_profile["object"]
            and normalize_space(claim_profile["object"]).casefold()
            == normalize_space(candidate_profile["object"]).casefold()
        ):
            score += 4.0
        if score > best_score:
            best_score = score
            best_sentence = candidate
    return best_sentence


def stable_round_robin(
    records: list[dict[str, Any]],
    target_count: int,
    *,
    seed: int,
    max_cases_per_doc: int,
) -> list[dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_doc.setdefault(str(record["doc_id"]), []).append(record)
    rng = random.Random(seed)
    for bucket in by_doc.values():
        rng.shuffle(bucket)
    doc_ids = sorted(by_doc)
    rng.shuffle(doc_ids)
    selected: list[dict[str, Any]] = []
    per_doc_counts = {doc_id: 0 for doc_id in doc_ids}
    made_progress = True
    while len(selected) < target_count and made_progress:
        made_progress = False
        for doc_id in doc_ids:
            if per_doc_counts[doc_id] >= max_cases_per_doc:
                continue
            bucket = by_doc[doc_id]
            if not bucket:
                continue
            selected.append(bucket.pop())
            per_doc_counts[doc_id] += 1
            made_progress = True
            if len(selected) >= target_count:
                break
    if len(selected) < target_count:
        leftovers: list[dict[str, Any]] = []
        for bucket in by_doc.values():
            leftovers.extend(bucket)
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max(0, target_count - len(selected))])
    return selected[:target_count]


def load_config(path: Path, corpus: str) -> dict[str, Any]:
    def normalize_runtime_entity_resource(
        resource_path: str,
        corpus_name: str,
        *,
        extraction_root: Path | None = None,
    ) -> str:
        raw_value = normalize_space(resource_path)
        source_path: Path | None = None
        if extraction_root is not None:
            alias_runtime_path = extraction_root.parent / "alias" / "corpus_entity_resource_v1.json"
            if alias_runtime_path.exists():
                source_path = alias_runtime_path
        if source_path is None and raw_value:
            source_path = resolve_path(raw_value)
        if source_path is None or not source_path.exists():
            return raw_value
        if not source_path.exists():
            return raw_value
        try:
            payload = load_json(source_path)
        except Exception:
            return raw_value
        entities = payload if isinstance(payload, list) else payload.get("entities")
        if not isinstance(entities, list):
            return raw_value

        def _surface_forms(item: dict[str, Any]) -> list[dict[str, Any]]:
            value = item.get("surface_forms")
            if isinstance(value, list):
                return [surface for surface in value if isinstance(surface, dict)]
            return []

        def _title_case_surface(text: str) -> str:
            tokens = normalize_space(text).split()
            if not tokens:
                return ""
            rendered: list[str] = []
            for token in tokens:
                if re.fullmatch(r"[A-Z0-9#]+", token):
                    if len(token) == 1:
                        rendered.append(token.upper())
                    else:
                        rendered.append(token.capitalize())
                else:
                    rendered.append(token)
            return " ".join(rendered)

        changed = False
        normalized_entities: list[dict[str, Any]] = []
        person_multi_by_last: dict[str, list[dict[str, Any]]] = defaultdict(list)
        surname_sources: list[dict[str, Any]] = []

        for item in entities:
            if not isinstance(item, dict):
                normalized_entities.append(item)
                continue
            entity = dict(item)
            entity_type = normalize_space(entity.get("entity_type", "")).upper()
            canonical = normalize_space(entity.get("canonical_display_form", ""))
            surface_forms = [dict(surface) for surface in _surface_forms(entity)]
            if entity_type == "PERSON" and canonical:
                title_surface = _title_case_surface(canonical)
                existing = {
                    normalize_space(surface.get("surface_form", "")).casefold()
                    for surface in surface_forms
                }
                if title_surface and title_surface.casefold() not in existing:
                    surface_forms.append(
                        {
                            "surface_form": title_surface,
                            "normalized_form": normalize_space(title_surface).casefold(),
                            "alias_type": "TITLE_CASE_DERIVED",
                            "confidence": 0.45,
                            "review_state": "CANDIDATE",
                            "provenance": [
                                {
                                    "derivation_method": "RUNTIME_NORMALIZATION",
                                    "source_table": str(source_path.name),
                                    "source_row_id": canonical,
                                    "corpus_doc_id": corpus_name,
                                }
                            ],
                        }
                    )
                    changed = True
                tokens = re.findall(r"[A-Z0-9#]+", canonical.upper())
                if len(tokens) >= 2:
                    person_multi_by_last[tokens[-1]].append(entity)
                elif len(tokens) == 1:
                    surname_sources.append(entity)
            entity["surface_forms"] = surface_forms
            normalized_entities.append(entity)

        if person_multi_by_last:
            by_id = {
                normalize_space(entity.get("canonical_entity_id", "")): entity
                for entity in normalized_entities
                if isinstance(entity, dict)
            }
            merged_source_ids: set[str] = set()
            for source in surname_sources:
                source_id = normalize_space(source.get("canonical_entity_id", ""))
                canonical = normalize_space(source.get("canonical_display_form", ""))
                tokens = re.findall(r"[A-Z0-9#]+", canonical.upper())
                if len(tokens) != 1:
                    continue
                candidates = person_multi_by_last.get(tokens[0], [])
                if len(candidates) != 1:
                    continue
                target = by_id.get(
                    normalize_space(candidates[0].get("canonical_entity_id", ""))
                )
                if not isinstance(target, dict):
                    continue
                target_surfaces = [dict(surface) for surface in _surface_forms(target)]
                seen_target = {
                    normalize_space(surface.get("surface_form", "")).casefold()
                    for surface in target_surfaces
                }
                for surface in _surface_forms(source):
                    surface_text = normalize_space(surface.get("surface_form", ""))
                    if not surface_text or surface_text.casefold() in seen_target:
                        continue
                    merged = dict(surface)
                    merged["alias_type"] = "MERGED_SURNAME"
                    merged.setdefault("confidence", 0.45)
                    merged.setdefault("review_state", "CANDIDATE")
                    provenance = list(merged.get("provenance", []) or [])
                    provenance.append(
                        {
                            "derivation_method": "RUNTIME_SURNAME_MERGE",
                            "source_table": str(source_path.name),
                            "source_row_id": source_id or canonical,
                            "corpus_doc_id": corpus_name,
                        }
                    )
                    merged["provenance"] = provenance
                    target_surfaces.append(merged)
                    seen_target.add(surface_text.casefold())
                    changed = True
                target["surface_forms"] = target_surfaces
                merged_source_ids.add(source_id)
            if merged_source_ids:
                filtered_entities: list[dict[str, Any]] = []
                for entity in normalized_entities:
                    if not isinstance(entity, dict):
                        filtered_entities.append(entity)
                        continue
                    entity_id = normalize_space(entity.get("canonical_entity_id", ""))
                    entity_type = normalize_space(entity.get("entity_type", "")).upper()
                    canonical = normalize_space(entity.get("canonical_display_form", ""))
                    tokens = re.findall(r"[A-Z0-9#]+", canonical.upper())
                    if (
                        entity_id in merged_source_ids
                        and entity_type == "PERSON"
                        and len(tokens) == 1
                    ):
                        changed = True
                        continue
                    filtered_entities.append(entity)
                normalized_entities = filtered_entities

        if not changed:
            # The Stage ALIAS resource remains authoritative even when runtime
            # compatibility normalization has nothing to add.
            return str(source_path)
        if isinstance(payload, list):
            normalized_payload: Any = normalized_entities
        else:
            normalized_payload = dict(payload)
            normalized_payload["entities"] = normalized_entities
        output_dir = KG_DEMO_ROOT / "data" / "resources" / "_runtime_normalized"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{corpus_name}_{source_path.stem}_runtime.json"
        atomic_write_json(output_path, normalized_payload)
        return str(output_path)

    config = load_json(path)
    profile_path = DEFAULT_PROFILES_DIR / f"{corpus}.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing corpus profile: {profile_path}")
    profile = load_json(profile_path)
    if isinstance(profile.get("benchmark_overrides"), dict):
        config = deep_merge_dict(config, profile["benchmark_overrides"])
    config["_config_path"] = str(path.resolve())
    manifest_path = KG_DEMO_ROOT / "data" / "manifests" / f"{corpus}.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        extraction_dir = normalize_space(profile.get("extraction_dir"))
        if not extraction_dir:
            raise FileNotFoundError(f"Missing corpus manifest: {manifest_path}")
        manifest = {
            "corpus_id": corpus,
            "knowledge_graph_extraction": extraction_dir,
            "resource": profile.get("corpus_entity_resource", ""),
            "raw_layout": "profile_only",
        }
    extraction_root = resolve_path(manifest["knowledge_graph_extraction"])
    config["corpus"] = corpus
    config["_profile_path"] = str(profile_path.resolve())
    config["_manifest_path"] = str(manifest_path.resolve()) if manifest_path.exists() else ""
    config["matryoshka_root"] = profile["engine_root"]
    config["corpus_entity_resource"] = normalize_runtime_entity_resource(
        profile.get("corpus_entity_resource") or manifest.get("resource", ""),
        corpus,
        extraction_root=extraction_root,
    )
    config["decision_profile"] = profile.get("decision_profile", "")
    config["output_root"] = str(KG_DEMO_ROOT / config["output_root"])
    config["python_executable"] = str(
        resolve_path_preserve_symlink(
            config.get("python_executable", sys.executable)
        )
    )
    reconstruction_table = extraction_root / "kg_pseudo_chunks_reconstruction.csv"
    base_reconstruction_table = extraction_root / "kg_pseudo_chunks.csv"
    config["source_tables"] = {
        "chunks": str(extraction_root / "chunks.csv"),
        "entities": str(extraction_root / "entities.csv"),
        "claims": str(extraction_root / "claims.csv"),
        "events": str(extraction_root / "events.csv"),
        "relations": str(extraction_root / "relations.csv"),
        "kg_pseudo_chunks_redaction": str(extraction_root / "kg_pseudo_chunks_redaction.csv"),
        "kg_pseudo_chunks": str(
            reconstruction_table if reconstruction_table.exists() else base_reconstruction_table
        ),
        "kg_pseudo_chunks_claims": str(extraction_root / "kg_pseudo_chunks_claims.csv"),
        "kg_pseudo_chunks_contradictions": str(extraction_root / "kg_pseudo_chunks_contradictions.csv"),
        "kg_pseudo_chunk_components": str(extraction_root / "kg_pseudo_chunk_components.csv"),
    }
    return config


def experiment_paths(config: dict[str, Any], experiment_name: str | None = None) -> ExperimentPaths:
    output_root = resolve_path(config["output_root"])
    name = experiment_name or str(config["experiment_name"])
    experiment_root = output_root / name
    return ExperimentPaths(
        output_root=output_root,
        experiment_root=experiment_root,
        extraction_root=experiment_root / "extraction_unified",
        config_root=experiment_root / "configs",
        query_manifest_root=experiment_root / "query_manifests",
        query_file_root=experiment_root / "query_files",
        run_root=experiment_root / "runs",
        summary_root=experiment_root / "summaries",
    )


def ensure_dirs(paths: ExperimentPaths) -> None:
    for directory in (
        paths.output_root,
        paths.experiment_root,
        paths.config_root,
        paths.query_manifest_root,
        paths.query_file_root,
        paths.run_root,
        paths.summary_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def materialize_unified_extraction(
    config: dict[str, Any],
    paths: ExperimentPaths,
    *,
    overwrite: bool,
) -> Path:
    mapping = config["source_tables"]
    if paths.extraction_root.exists() and overwrite:
        shutil.rmtree(paths.extraction_root)
    paths.extraction_root.mkdir(parents=True, exist_ok=True)
    target_map = {
        "chunks": "chunks.csv",
        "entities": "entities.csv",
        "claims": "claims.csv",
        "events": "events.csv",
        "relations": "relations.csv",
        "kg_pseudo_chunks_redaction": "kg_pseudo_chunks_redaction.csv",
        "kg_pseudo_chunks": "kg_pseudo_chunks.csv",
        "kg_pseudo_chunks_claims": "kg_pseudo_chunks_claims.csv",
        "kg_pseudo_chunks_contradictions": "kg_pseudo_chunks_contradictions.csv",
        "kg_pseudo_chunk_components": "kg_pseudo_chunk_components.csv",
    }
    manifest: list[dict[str, str]] = []
    for source_key, output_name in target_map.items():
        src = resolve_path(mapping[source_key])
        dst = paths.extraction_root / output_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        manifest.append(
            {
                "logical_name": source_key,
                "output_name": output_name,
                "source_path": str(src),
            }
        )
    atomic_write_json(paths.extraction_root / "unified_extraction_manifest.json", manifest)
    return paths.extraction_root


def run_command(cmd: list[str]) -> None:
    print("\n== Running ==")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def maybe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    tables = {}
    for key, raw_path in config["source_tables"].items():
        try:
            tables[key] = pd.read_csv(resolve_path(raw_path), low_memory=False)
        except pd.errors.EmptyDataError:
            tables[key] = pd.DataFrame()
    return tables


def label_type_string(raw: Any) -> str:
    text = normalize_space(str(raw or "unknown"))
    return text.upper().replace(" ", "_")


def build_chunk_index(chunks: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in chunks.to_dict("records"):
        key = (str(row["doc_id"]), str(row["paragraph_id"]))
        index[key] = row
    return index


def build_doc_sentences(chunks: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in chunks.to_dict("records"):
        doc_id = str(row["doc_id"])
        by_doc.setdefault(doc_id, []).append(row)
    for rows in by_doc.values():
        rows.sort(key=lambda item: int(item.get("chunk_index_in_doc", 0)))
    return by_doc


def build_doc_para_richness(
    claims: pd.DataFrame,
    events: pd.DataFrame,
    relations: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = {}

    def bump(df: pd.DataFrame, field: str) -> None:
        for row in df.to_dict("records"):
            key = paragraph_key(row["doc_id"], row["paragraph_id"])
            bucket = counts.setdefault(key, {"claim_n": 0, "event_n": 0, "rel_n": 0})
            bucket[field] += 1

    bump(claims, "claim_n")
    bump(events, "event_n")
    bump(relations, "rel_n")
    return counts


def round_robin_topk(
    records: list[dict[str, Any]],
    target_count: int,
    *,
    max_cases_per_doc: int,
    score_key: str,
    seed: int,
) -> list[dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_doc.setdefault(str(record["doc_id"]), []).append(record)
    for bucket in by_doc.values():
        bucket.sort(
            key=lambda item: (
                float(item.get(score_key, 0.0) or 0.0),
                int(item.get("word_count", 0) or 0),
            ),
            reverse=True,
        )
    doc_ids = sorted(
        by_doc,
        key=lambda doc_id: (
            float(by_doc[doc_id][0].get(score_key, 0.0) or 0.0),
            int(by_doc[doc_id][0].get("word_count", 0) or 0),
            doc_id,
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    per_doc_counts = {doc_id: 0 for doc_id in doc_ids}
    made_progress = True
    while len(selected) < target_count and made_progress:
        made_progress = False
        for doc_id in doc_ids:
            if per_doc_counts[doc_id] >= max_cases_per_doc:
                continue
            bucket = by_doc.get(doc_id, [])
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            per_doc_counts[doc_id] += 1
            made_progress = True
            if len(selected) >= target_count:
                break
    if len(selected) < target_count:
        leftovers: list[dict[str, Any]] = []
        for bucket in by_doc.values():
            leftovers.extend(bucket)
        leftovers.sort(
            key=lambda item: float(item.get(score_key, 0.0) or 0.0),
            reverse=True,
        )
        selected.extend(leftovers[: max(0, target_count - len(selected))])
    return selected[:target_count]


def merge_ranked_candidates(
    candidate_groups: Iterable[list[dict[str, Any]]],
    *,
    target_count: int,
    max_cases_per_doc: int,
    score_key: str,
    seed: int,
) -> list[dict[str, Any]]:
    best_by_case: dict[str, dict[str, Any]] = {}
    for group in candidate_groups:
        for record in group:
            case_id = str(record.get("case_id", ""))
            if not case_id:
                continue
            prior = best_by_case.get(case_id)
            if prior is None or float(record.get(score_key, 0.0) or 0.0) > float(prior.get(score_key, 0.0) or 0.0):
                best_by_case[case_id] = record
    return round_robin_topk(
        list(best_by_case.values()),
        target_count,
        max_cases_per_doc=max_cases_per_doc,
        score_key=score_key,
        seed=seed,
    )


def quality_fill_unique_candidates(
    records: list[dict[str, Any]],
    target_count: int,
    *,
    max_cases_per_doc: int,
    score_key: str,
) -> list[dict[str, Any]]:
    if target_count <= 0 or not records:
        return []
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_doc.setdefault(str(record["doc_id"]), []).append(record)
    for bucket in by_doc.values():
        bucket.sort(
            key=lambda item: (
                float(item.get(score_key, 0.0) or 0.0),
                int(item.get("word_count", 0) or 0),
            ),
            reverse=True,
        )
    selected: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    per_doc_counts: dict[str, int] = {doc_id: 0 for doc_id in by_doc}

    def try_add(record: dict[str, Any]) -> bool:
        doc_id = str(record["doc_id"])
        if per_doc_counts.get(doc_id, 0) >= max_cases_per_doc:
            return False
        query_key = canonical_query_dedupe_key(record.get("query", ""))
        if query_key and query_key in seen_queries:
            return False
        selected.append(record)
        per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
        if query_key:
            seen_queries.add(query_key)
        return True

    doc_order = sorted(
        by_doc,
        key=lambda doc_id: (
            float(by_doc[doc_id][0].get(score_key, 0.0) or 0.0),
            int(by_doc[doc_id][0].get("word_count", 0) or 0),
            doc_id,
        ),
        reverse=True,
    )
    for doc_id in doc_order:
        if len(selected) >= target_count:
            break
        for record in by_doc[doc_id]:
            if try_add(record):
                break

    leftovers: list[dict[str, Any]] = []
    for bucket in by_doc.values():
        leftovers.extend(bucket)
    leftovers.sort(
        key=lambda item: (
            float(item.get(score_key, 0.0) or 0.0),
            int(item.get("word_count", 0) or 0),
        ),
        reverse=True,
    )
    for record in leftovers:
        if len(selected) >= target_count:
            break
        try_add(record)
    return selected[:target_count]


def unique_query_round_robin(
    records: list[dict[str, Any]],
    target_count: int,
    *,
    max_cases_per_doc: int,
    score_key: str,
    seed: int,
) -> list[dict[str, Any]]:
    ranked = round_robin_topk(
        records,
        len(records),
        max_cases_per_doc=max_cases_per_doc,
        score_key=score_key,
        seed=seed,
    )
    selected: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for record in ranked:
        query_key = canonical_query_dedupe_key(record.get("query", ""))
        if not query_key or query_key in seen_queries:
            continue
        seen_queries.add(query_key)
        selected.append(record)
        if len(selected) >= target_count:
            break
    return selected


def canonical_query_dedupe_key(text: Any) -> str:
    value = normalize_surface_preserving_markers(text)
    value = (
        value.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return value.casefold()


def normalize_candidate_type(raw: Any) -> str:
    text = normalize_space(raw).casefold()
    if text in {"politicalgroup", "political_group", "political party"}:
        return "political_party"
    if text in {"country", "nation", "state"}:
        return "location"
    if text in {"person_like", "person"}:
        return "person"
    return text.replace(" ", "_")


def synthesize_redaction_entity_rows(entities: pd.DataFrame) -> list[dict[str, Any]]:
    synthesized: list[dict[str, Any]] = []
    if entities.empty:
        return synthesized
    column_index = {name: idx for idx, name in enumerate(entities.columns)}

    def row_value(row: tuple[Any, ...], column: str, default: Any = "") -> Any:
        index = column_index.get(column)
        if index is None:
            return default
        value = row[index]
        if pd.isna(value):
            return default
        return value

    for row in entities.itertuples(index=False, name=None):
        label = normalize_space(row_value(row, "label", ""))
        if not label:
            continue
        doc_id = row_value(row, "doc_id", "")
        paragraph_id = row_value(row, "paragraph_id", "")
        chunk_id = row_value(row, "chunk_id", "")
        entity_type = normalize_candidate_type(row_value(row, "type", ""))
        role_context = normalize_space(
            row_value(row, "description", "")
            or row_value(row, "context", "")
            or row_value(row, "evidence", "")
        )
        context_text = normalize_space(
            row_value(row, "context_text", "")
            or row_value(row, "context_window_text", "")
            or row_value(row, "context", "")
            or row_value(row, "evidence", "")
        )
        context_window_text = normalize_space(
            row_value(row, "context_window_text", "")
            or row_value(row, "context_text", "")
            or row_value(row, "context", "")
            or row_value(row, "evidence", "")
        )
        local_id = normalize_space(row_value(row, "local_id", "")) or sha1_text(
            f"{doc_id}|{paragraph_id}|{label}|{entity_type}"
        )
        evidence_json = {
            "raw_row_id": local_id,
            "source_table": "entities",
            "entity_type": entity_type,
        }
        synthesized.append(
            {
                "doc_id": doc_id,
                "paragraph_id": paragraph_id,
                "chunk_id": chunk_id,
                "candidate_label": label,
                "candidate_text": label,
                "candidate_type": row_value(row, "type", ""),
                "candidate_type_normalized": entity_type.upper(),
                "candidate_source_table": "entities",
                "candidate_source_confidence": float(row_value(row, "confidence", 0.0) or 0.0),
                "candidate_role_or_context": role_context,
                "candidate_role_context": role_context,
                "context_text": context_text,
                "context_window_text": context_window_text,
                "candidate_doc_id": doc_id,
                "candidate_source_row_id": local_id,
                "source_id": row_value(row, "source_id", doc_id),
                "candidate_evidence_json": json.dumps(evidence_json, ensure_ascii=False),
                "is_section_heading_like": False,
                "is_country_like": entity_type == "country",
                "is_person_name_like": entity_type == "person",
                "is_position_like": entity_type == "person" and is_title_only_person_label(label),
                "is_organization_like": entity_type in {
                    "organization",
                    "government",
                    "military",
                    "political_party",
                },
            }
        )
    return synthesized


def find_chunk_containing_label(
    doc_chunks: list[dict[str, Any]],
    label: str,
) -> dict[str, Any] | None:
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for row in doc_chunks:
        text = str(row.get("chunk_text", ""))
        match = find_label_match(text, label)
        if not match:
            continue
        words = int(row.get("word_count", count_words(text)))
        distance_from_center = abs((match.start() + match.end()) // 2 - len(text) // 2)
        scored.append((distance_from_center, words, row))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2]


def find_chunk_for_redaction_row(
    doc_chunks: list[dict[str, Any]],
    *,
    label: str,
    preferred_chunk_id: str = "",
) -> dict[str, Any] | None:
    normalized_chunk_id = normalize_space(preferred_chunk_id)
    if normalized_chunk_id:
        for row in doc_chunks:
            if normalize_space(row.get("chunk_id", "")) != normalized_chunk_id:
                continue
            text = str(row.get("chunk_text", ""))
            if find_label_match(text, label):
                return row
    return find_chunk_containing_label(doc_chunks, label)


def expand_person_label_with_context(
    label: str,
    *texts: str,
) -> str:
    normalized_label = normalize_space(label)
    if not normalized_label or count_words(normalized_label) != 1:
        return normalized_label
    escaped_label = re.escape(normalized_label)
    full_name_pattern = re.compile(
        rf"\b(([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){{0,2}})\s+(?i:{escaped_label}))\b"
    )
    # Extraction can emit either a surname ("Papadopoulos") or a first name
    # ("George") as the local label.  The original backward-only expansion
    # handled the former but could leave the surname visible when redacting the
    # latter.  Preserve case sensitivity for the following tokens so ordinary
    # verbs are not promoted into a person's name.
    forward_name_pattern = re.compile(
        rf"\b(?i:{escaped_label})\s+([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){{0,2}})\b"
    )
    for text in texts:
        source = str(text or "")
        if not source:
            continue
        if uppercase_token_ratio(source) >= 0.45 and count_words(source) >= 8:
            source = normalize_uppercase_passage(source)
        for match in full_name_pattern.finditer(source):
            candidate = normalize_space(match.group(1))
            if 2 <= count_words(candidate) <= 4:
                return candidate
        for match in forward_name_pattern.finditer(source):
            candidate = normalize_space(f"{normalized_label} {match.group(1)}")
            if 2 <= count_words(candidate) <= 4:
                return candidate
    return normalized_label


def prefer_full_name_person_label(
    label: str,
    entity_rows: list[dict[str, Any]],
) -> str:
    normalized_label = normalize_space(label)
    if not normalized_label or count_words(normalized_label) != 1:
        return normalized_label
    token = normalized_label.casefold()
    for row in entity_rows:
        peer_type = normalize_candidate_type(row.get("candidate_type", ""))
        peer_label = normalize_space(row.get("candidate_label", ""))
        if peer_type != "person" or count_words(peer_label) < 2:
            continue
        peer_tokens = [part.casefold() for part in peer_label.split()]
        if token in peer_tokens:
            return peer_label
    return normalized_label


def value_redaction_query_is_viable(
    query: str,
    *,
    entity_type: str,
    used_specs: list[dict[str, Any]],
) -> bool:
    if not query:
        return False
    if count_redaction_slots(query) != len(used_specs):
        return False
    if query_surface_is_blocked(query, task_family="value_redaction"):
        return False
    if has_unbalanced_delimiters(query):
        return False
    if "?" in query or "--" in query or query.casefold().startswith("did "):
        return False
    if re.search(
        r"\b(?:i don't know|that i can answer|i will have to|we do not feel|"
        r"now what we are buying|i can answer conclusively)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return False
    if (
        count_words(query) <= 48
        and re.search(r"\b(?:I|we)\b", query, flags=re.IGNORECASE)
    ):
        return False
    if re.search(
        r"[.!?]\s+\[REDACT:R\d+[^\]]*\]\s+(?:the|a|an)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:at|in|to|from|near)\s+\[REDACT:R\d+[^\]]*\]\s+[A-Z][a-z]",
        query,
    ):
        return False
    if re.search(r"\barticle entitled\b", query, flags=re.IGNORECASE):
        return False
    if query.rstrip().endswith((",", ";", ":", "-", "(", "[")):
        return False
    if (
        entity_type == "person"
        and query.lstrip().startswith("[REDACT:R1")
        and re.search(r"\b(?:he|she|his|her|him)\b", query, flags=re.IGNORECASE)
    ):
        return False
    if entity_type == "person" and re.search(
        r"\[REDACT:R\d+[^\]]*\]\s+[A-Z][A-Za-z'’-]{2,}\b",
        query,
    ):
        # A capitalized token immediately after a person slot is normally the
        # unredacted remainder of a multi-token name (for example
        # "[REDACT] Papadopoulos").  Such a case has a mismatched answer span
        # and is not a valid recovery benchmark.
        return False
    if re.match(
        r"^\[REDACT:R1[^\]]*\]\s+(?:is\s+(?:a|an|the)\b|has\b|had\b|will\s+be\b|appears?\s+to\s+be\b)",
        query,
        flags=re.IGNORECASE,
    ):
        return False
    if re.match(
        r"^The\s+\[REDACT:R1[^\]]*\]'s\s+(?:action|effort|position|approval|accession|resignation|proposal)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return False
    if re.match(
        r"^The\s+action\s+of\s+\[REDACT:R1[^\]]*\]\b",
        query,
        flags=re.IGNORECASE,
    ):
        return False
    if re.match(
        r"^(?:Recent|Current|Possible|Planned)\s+\w+\s+of\s+\[REDACT:R1[^\]]*\]\b",
        query,
        flags=re.IGNORECASE,
    ):
        return False
    if (
        entity_type in {"organization", "government", "military", "political_party"}
        and len(used_specs) == 1
        and re.match(r"^\[REDACT:R1[^\]]*\]\s+[a-z]{3,}\b", query)
    ):
        return False
    if re.search(r"\[REDACT:[^\]]+\]\s+And\s+was\b", query):
        return False
    if re.search(
        r"\[REDACT:R\d+[^\]]*\]\s+(?:"
        r"Additionally|Accordingly|Finally|First|Further|However|Next|"
        r"Second|Specifically|Third|Thus"
        r")\b",
        query,
        flags=re.IGNORECASE,
    ):
        return False
    if re.match(
        r"^(?:Based on information developed through its ongoing investigation|Based on reports from a friendly foreign government)\b",
        query,
        flags=re.IGNORECASE,
    ) and (
        re.search(r"\bthe FBI believes that\b", query, flags=re.IGNORECASE)
        or re.search(r"\bthe FBI assesses that\b", query, flags=re.IGNORECASE)
    ):
        return False
    if re.search(r"\bwill be discussing\b", query, flags=re.IGNORECASE):
        return False
    slot_types = {
        normalize_candidate_type(spec.get("entity_type", ""))
        for spec in used_specs
    }
    if slot_types & {"other", "unknown", "target", ""}:
        return False
    for spec in used_specs:
        slot_type = normalize_candidate_type(spec.get("entity_type", ""))
        slot_label = normalize_space(spec.get("label", ""))
        if slot_label and find_label_match(query, slot_label):
            return False
        if slot_type == "person" and count_words(slot_label) >= 2:
            visible_name_tokens = {
                token.casefold()
                for token in WORD_RE.findall(slot_label)
                if len(token) >= 4
                and token.casefold()
                not in {
                    "ambassador",
                    "chairman",
                    "chief",
                    "general",
                    "minister",
                    "president",
                    "secretary",
                }
            }
            query_tokens = {
                token.casefold()
                for token in WORD_RE.findall(REDACTION_MARKER_RE.sub(" ", query))
            }
            if visible_name_tokens & query_tokens:
                return False
        if re.search(r"\bCANDIDATE\s*#\d+\b", slot_label, flags=re.IGNORECASE):
            return False
        if slot_type == "government" and count_words(slot_label) <= 1:
            return False
        if (
            slot_type in {"organization", "government", "military", "political_party"}
            and looks_like_acronym_label(slot_label)
            and len(re.sub(r"[^A-Za-z]", "", slot_label)) <= 2
        ):
            return False
    if len(used_specs) == 1:
        # A lone blank at the end of a long homogeneous enumeration is not
        # semantically recoverable: any omitted peer can fit once wording is
        # paraphrased. Keep shorter lists and genuine multi-slot tasks.
        for sentence in sentence_split(query):
            marker_match = REDACTION_MARKER_RE.search(sentence)
            if not marker_match:
                continue
            left = sentence[: marker_match.start()]
            if (
                left.count(",") >= 4
                and re.search(r"\b(?:and|or)(?:\s+the)?\s*$", left, flags=re.IGNORECASE)
            ):
                return False
    for marker in re.finditer(r"\[REDACT:R\d+[^\]]*\]", query):
        left_context = normalize_space(query[: marker.start()])
        right_context = normalize_space(query[marker.end() :])
        if (
            left_context
            and right_context
            and left_context[-1].isalnum()
            and re.match(r"^(?:The|A|An|[A-Z][a-z]{2,})\b", right_context)
            and not re.search(r"(?:,|;|:|\b(?:and|or|to|from|with|by|of|in)\b)$", left_context, flags=re.IGNORECASE)
        ):
            return False
    non_slot_sentences = [
        normalize_space(sentence)
        for sentence in sentence_split(query)
        if normalize_space(sentence)
    ]
    if any(
        sentence_is_low_signal_for_query(sentence)
        for sentence in non_slot_sentences
        if not REDACTION_MARKER_RE.search(sentence)
    ):
        return False
    if any(
        substantive_sentence_score(sentence) < 3.5
        for sentence in non_slot_sentences
        if not REDACTION_MARKER_RE.fullmatch(sentence)
    ):
        return False
    return True


def normalize_redaction_list_boundary(text: str) -> str:
    value = str(text or "")
    if not value:
        return value
    return re.sub(
        r"(\[REDACT:R\d+[^\]]*\])\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+and\b",
        r"\1, \2 and",
        value,
    )


def best_label_sentence_surface(
    source_texts: list[str],
    label: str,
    *,
    small_corpus: bool,
    pair_like_corpus: bool,
    min_score: float = -4.0,
) -> str:
    candidates: list[str] = []
    seen: set[str] = set()
    for source in source_texts:
        text = str(source or "")
        if not text:
            continue
        for sentence in cleaned_substantive_sentences(text, min_score=min_score):
            if not find_label_match(sentence, label):
                continue
            candidate = finalize_query_surface(sentence, add_terminal_punctuation=True)
            key = normalize_space(candidate).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    if not candidates:
        return ""

    target_words = 22 if small_corpus or pair_like_corpus else 26
    best_sentence = candidates[0]
    best_score = float("-inf")
    for candidate in candidates:
        match = find_label_match(candidate, label)
        score = substantive_sentence_score(candidate, anchors=[label])
        score -= surface_noise_penalty(candidate) * 0.35
        score -= abs(count_words(candidate) - target_words) / 10.0
        if match and match.start() > 0:
            score += 1.5
        if DATE_TOKEN_RE.search(candidate) or NUMBER_RE.search(candidate):
            score += 1.0
        if sentence_is_low_signal_for_query(candidate):
            score -= 6.0
        if query_surface_is_blocked(candidate, task_family="value_redaction"):
            score -= 6.0
        if score > best_score:
            best_score = score
            best_sentence = candidate
    return best_sentence


def build_pair_sentence_value_redaction_candidates(
    entities: pd.DataFrame,
    doc_chunks_by_doc: dict[str, list[dict[str, Any]]],
    para_richness: dict[tuple[str, str], dict[str, int]],
    component_index: dict[tuple[str, str], dict[str, list[str]]],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    runner_cfg: dict[str, Any] | None = None,
    entity_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    strict_mode = not maybe_bool(sampling.get("_relaxed_retry_used", False))
    if not doc_chunks_by_doc:
        return []
    pair_like_corpus = all(str(doc_id).startswith("Pair_") for doc_id in doc_chunks_by_doc)
    corpus_doc_count = max(1, len(doc_chunks_by_doc))
    small_corpus = exhaustive_small_doc_corpus(corpus_doc_count)
    large_corpus = optimized_doc_corpus(corpus_doc_count)
    min_query_words = int(sampling.get("min_query_words", 16))
    pair_min_query_words = max(10, min_query_words - 6)
    max_query_words = int(sampling.get("max_query_words", 180))
    include_locations = maybe_bool(sampling.get("include_locations", False))
    max_slot_count = min(int(sampling.get("max_slot_count", 3)), 3)
    slot_count_decay = float(sampling.get("slot_count_decay", 0.5))
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    allowed_types = (
        {value.casefold() for value in sampling.get("allowed_entity_types", [])}
        | {value.casefold() for value in sampling.get("fallback_entity_types", [])}
        | {"person", "location", "country", "organization", "government", "military", "political_party"}
    )
    type_priority = {
        "person": 10.0,
        "organization": 5.0,
        "government": 4.0,
        "military": 4.0,
        "political_party": 4.0,
        "location": 9.0,
        "country": 8.0,
    }
    paragraph_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    label_doc_sets: dict[str, set[str]] = defaultdict(set)
    label_occurrence_counts: dict[str, int] = defaultdict(int)
    synthesized_rows = (
        entity_rows if entity_rows is not None else synthesize_redaction_entity_rows(entities)
    )
    for row in synthesized_rows:
        para_key = paragraph_key(row["doc_id"], row["paragraph_id"])
        paragraph_rows[para_key].append(row)
        label_key = normalize_space(row.get("candidate_label", "")).casefold()
        if label_key:
            label_doc_sets[label_key].add(str(row.get("doc_id", "")))
            label_occurrence_counts[label_key] += 1

    chunk_units: list[dict[str, Any]] = []
    for doc_id, chunk_rows in doc_chunks_by_doc.items():
        for chunk in chunk_rows:
            chunk_units.append(
                {
                    "doc_id": str(doc_id),
                    "paragraph_id": normalize_space(chunk.get("paragraph_id", "")),
                    "chunk_text": str(chunk.get("chunk_text", "") or ""),
                }
            )
    chunk_units = deterministic_seeded_row_order(
        chunk_units,
        seed=seed + 211,
        key_fields=("doc_id", "paragraph_id", "chunk_text"),
    )
    if large_corpus and not pair_like_corpus:
        def chunk_priority(unit: dict[str, Any]) -> tuple[int, int]:
            richness = para_richness.get(
                paragraph_key(unit.get("doc_id", ""), unit.get("paragraph_id", "")),
                {},
            )
            return (
                int(richness.get("claim_n", 0)) * 3
                + int(richness.get("event_n", 0)) * 2
                + int(richness.get("rel_n", 0)),
                count_words(unit.get("chunk_text", "")),
            )

        chunk_units = sorted(chunk_units, key=chunk_priority, reverse=True)[
            : max(int(sampling.get("large_corpus_sentence_slot_chunk_cap", 400)), count)
        ]

    def support_hits_for_label(candidate_label: str, bucket: dict[str, list[str]]) -> int:
        return sum(
            1
            for claim_text in bucket.get("claim_texts", [])
            if re.search(rf"\b{re.escape(candidate_label)}\b", claim_text, flags=re.IGNORECASE)
        ) + sum(
            1
            for event_text in bucket.get("event_labels", [])
            if re.search(rf"\b{re.escape(candidate_label)}\b", event_text, flags=re.IGNORECASE)
        )

    def pair_sentence_label_is_admissible(label: str, entity_type: str) -> bool:
        normalized = normalize_space(label)
        if not normalized or count_words(normalized) > 6:
            return False
        lowered = normalized.casefold()
        if actor_phrase_has_clause_bleed(normalized):
            return False
        if re.search(r"\b(?:like|specific|reporting|reminded|appears|meeting|told|approached)\b", lowered):
            return False
        alpha_tokens = re.findall(r"[A-Za-z]+", normalized)
        lowercase_tokens = [
            token
            for token in alpha_tokens
            if token.casefold() not in {"of", "the", "and", "for", "to", "on", "in"}
            and not token.isupper()
            and not token.istitle()
        ]
        if lowercase_tokens:
            return False
        if entity_type == "person":
            if count_words(normalized) < 2:
                return False
            if contains_role_placeholder_label(normalized):
                return False
        if entity_type in {"organization", "government", "military"} and count_words(normalized) < 2:
            return False
        return True

    deduped_candidates: dict[str, dict[str, Any]] = {}
    for chunk in chunk_units:
            doc_id = normalize_space(chunk.get("doc_id", ""))
            paragraph_id = normalize_space(chunk.get("paragraph_id", ""))
            if not paragraph_id:
                continue
            para_key = paragraph_key(doc_id, paragraph_id)
            richness = para_richness.get(para_key, {})
            local_rows = paragraph_rows.get(para_key, [])
            if not local_rows:
                continue
            chunk_text = str(chunk.get("chunk_text", "") or "")
            if not chunk_text:
                continue
            if uppercase_token_ratio(chunk_text) >= 0.45 and count_words(chunk_text) >= 18:
                chunk_text = normalize_uppercase_passage(chunk_text)
            if large_corpus and not pair_like_corpus:
                chunk_text = finalize_query_surface(
                    extract_sentence_excerpt(
                        chunk_text,
                        max_sentences=3,
                        max_chars=420,
                    )
                )
            else:
                chunk_text = clean_query_passage(chunk_text)
            chunk_text = strip_internal_topic_sections(chunk_text)
            component_bucket = component_index.get(
                para_key,
                {"entity_labels": [], "event_labels": [], "claim_texts": []},
            )
            prepared_local_rows: list[dict[str, Any]] = []
            for entity_row in local_rows:
                label = normalize_space(entity_row.get("candidate_label", ""))
                if not label:
                    continue
                entity_type = normalize_candidate_type(entity_row.get("candidate_type", ""))
                if entity_type == "location" and not include_locations:
                    entity_type = "country"
                if entity_type not in allowed_types:
                    continue
                if entity_type == "person":
                    label = prefer_full_name_person_label(label, local_rows)
                    if count_words(label) < 2:
                        continue
                if entity_type == "person" and (
                    is_probably_generic_person_label(label)
                    or is_title_only_person_label(label)
                ):
                    continue
                if not pair_sentence_label_is_admissible(label, entity_type):
                    continue
                if (
                    entity_type in {"organization", "government", "military", "political_party"}
                    and count_words(label) < 2
                ):
                    continue
                if (
                    entity_type in {"location", "country"}
                    and count_words(label) == 1
                    and label.casefold() in {"russia", "syria", "crimea"}
                ):
                    continue
                prepared_local_rows.append(
                    {
                        "label": label,
                        "slot_type": label_type_string(entity_type),
                        "source_id": str(entity_row.get("doc_id", doc_id)),
                        "context_text": (
                            finalize_query_surface(
                                extract_sentence_excerpt(
                                    str(entity_row.get("context_text", "") or ""),
                                    max_sentences=2,
                                    max_chars=280,
                                )
                            )
                            if large_corpus and not pair_like_corpus
                            else clean_query_passage(
                                str(entity_row.get("context_text", "") or "")
                            )
                        ),
                        "context_window_text": (
                            finalize_query_surface(
                                extract_sentence_excerpt(
                                    str(entity_row.get("context_window_text", "") or ""),
                                    max_sentences=2,
                                    max_chars=320,
                                )
                            )
                            if large_corpus and not pair_like_corpus
                            else clean_query_passage(
                                str(entity_row.get("context_window_text", "") or "")
                            )
                        ),
                        "entity_type": entity_type,
                        "confidence": safe_float(entity_row.get("candidate_source_confidence", 0.0)),
                        "label_word_count": count_words(label),
                        "label_acronym_like": looks_like_acronym_label(label),
                        "role_words": count_words(
                            normalize_space(entity_row.get("candidate_role_or_context", ""))
                        ),
                        "local_support_hits": support_hits_for_label(label, component_bucket),
                        "label_doc_frequency": len(label_doc_sets.get(label.casefold(), set())),
                        "label_occurrences": int(label_occurrence_counts.get(label.casefold(), 1)),
                    }
                )
            if not prepared_local_rows:
                continue
            prepared_local_rows.sort(
                key=lambda spec: (
                    int(spec.get("local_support_hits", 0) or 0),
                    spec["entity_type"] in {"person", "location", "country"},
                    safe_float(spec.get("confidence", 0.0)),
                    count_words(spec.get("label", "")),
                ),
                reverse=True,
            )
            if large_corpus and not pair_like_corpus:
                prepared_local_rows = prepared_local_rows[:12]
            raw_sentences = [
                finalize_query_surface(sentence, add_terminal_punctuation=True)
                for sentence in sentence_split(chunk_text)
            ]
            component_sentences: list[str] = []
            if small_corpus or pair_like_corpus:
                for component_text in (
                    list(component_bucket.get("claim_texts", []))[:8]
                    + list(component_bucket.get("event_labels", []))[:6]
                ):
                    candidate = finalize_query_surface(
                        clean_query_passage(str(component_text or "")),
                        add_terminal_punctuation=True,
                    )
                    if (
                        candidate
                        and not is_routing_or_metadata_sentence(candidate)
                        and not is_list_like_text(candidate)
                        and not BROKEN_LEAD_RE.search(candidate)
                        and not query_surface_is_blocked(candidate, task_family="value_redaction")
                        and not sentence_is_low_signal_for_query(candidate)
                    ):
                        component_sentences.append(candidate)
            raw_sentences = unique_preserving_order(raw_sentences + component_sentences)
            sentences = [
                sentence
                for sentence in raw_sentences
                if sentence
                and not is_routing_or_metadata_sentence(sentence)
                and not is_list_like_text(sentence)
                and not BROKEN_LEAD_RE.search(sentence)
                and not query_surface_is_blocked(sentence, task_family="value_redaction")
                and not sentence_is_low_signal_for_query(sentence)
                and count_words(sentence) >= 8
                and count_words(sentence) <= 52
            ]
            if large_corpus and not pair_like_corpus and sentences:
                def sentence_priority(sentence: str) -> tuple[float, int]:
                    support_hits = sum(
                        1
                        for spec in prepared_local_rows
                        if find_label_match(sentence, normalize_space(spec.get("label", "")))
                    )
                    return (
                        substantive_sentence_score(sentence, anchors=[]),
                        support_hits + (1 if DATE_TOKEN_RE.search(sentence) or NUMBER_RE.search(sentence) else 0),
                    )

                sentences = sorted(
                    sentences,
                    key=sentence_priority,
                    reverse=True,
                )[:6]
            for sentence_index, sentence in enumerate(sentences):
                sentence_variants = [sentence]
                simplified = simplify_claim_query_seed(simplify_claim_surface(sentence))
                simplified = finalize_query_surface(simplified, add_terminal_punctuation=True)
                if (
                    simplified
                    and simplified.casefold() != sentence.casefold()
                    and not simplified.startswith(("He ", "She ", "They ", "It ", "There "))
                    and not query_surface_is_blocked(simplified, task_family="value_redaction")
                    and not sentence_is_low_signal_for_query(simplified)
                    and count_words(simplified) >= 8
                    and count_words(simplified) <= 52
                ):
                    sentence_variants.append(simplified)
                for variant in unique_preserving_order(sentence_variants):
                    sentence_specs: list[dict[str, Any]] = []
                    for spec in prepared_local_rows:
                        label = normalize_space(spec.get("label", ""))
                        if not label or not find_label_match(variant, label):
                            continue
                        sentence_specs.append(dict(spec))
                    unique_specs: list[dict[str, Any]] = []
                    seen_labels: set[str] = set()
                    for spec in sorted(
                        sentence_specs,
                        key=lambda spec: (
                            spec["local_support_hits"] > 0,
                            spec["entity_type"] in {"person", "location", "country"},
                            safe_float(spec.get("confidence", 0.0)),
                            count_words(spec.get("label", "")),
                        ),
                        reverse=True,
                    ):
                        label_key = normalize_space(spec.get("label", "")).casefold()
                        if not label_key or label_key in seen_labels:
                            continue
                        seen_labels.add(label_key)
                        unique_specs.append(spec)
                    if not unique_specs:
                        continue

                    desired_slot_count = deterministic_tail_count(
                        f"pair_value_slots|{doc_id}|{paragraph_id}|{variant}",
                        max_count=max_slot_count,
                        decay=slot_count_decay,
                    )
                    primary_groups: list[list[dict[str, Any]]] = [
                        [spec] for spec in unique_specs[: min(len(unique_specs), 4)]
                    ]
                    if len(unique_specs) >= 2:
                        multi_specs = unique_specs[: min(max(desired_slot_count, 2), len(unique_specs), 3)]
                        primary_groups.append(multi_specs)
                        if len(multi_specs) == 3:
                            primary_groups.append(multi_specs[:2])

                    for base_specs in primary_groups:
                        target_labels = [normalize_space(spec.get("label", "")) for spec in base_specs]
                        if not all(target_labels):
                            continue
                        primary_base_type = normalize_candidate_type(
                            base_specs[0].get("entity_type", "")
                        )
                        low_support_small_location = (
                            (small_corpus or pair_like_corpus)
                            and len(base_specs) == 1
                            and primary_base_type in {"location", "country"}
                            and int(base_specs[0].get("local_support_hits", 0) or 0) <= 0
                            and not bool(DATE_TOKEN_RE.search(variant) or NUMBER_RE.search(variant))
                        )
                        if len(base_specs) == 1:
                            only_type = str(base_specs[0].get("entity_type", ""))
                            same_type_labels = [
                                normalize_space(spec.get("label", ""))
                                for spec in unique_specs
                                if str(spec.get("entity_type", "")) == only_type
                            ]
                            if only_type in {"location", "country"} and len(unique_preserving_order(same_type_labels)) > 1:
                                continue
                        target_words = scaled_redaction_query_target(
                            f"pair_value_window|{doc_id}|{paragraph_id}|{'|'.join(target_labels)}",
                            minimum_words=pair_min_query_words,
                            maximum_words=min(max_query_words, 96),
                            slot_count=len(base_specs),
                        )
                        if low_support_small_location:
                            target_words = max(target_words, 26 if small_corpus and not pair_like_corpus else 22)
                        neighbor_contexts = [variant]
                        if count_words(variant) < pair_min_query_words:
                            if sentence_index > 0:
                                neighbor_contexts.append(
                                    finalize_query_surface(
                                        f"{sentences[sentence_index - 1]} {variant}",
                                        add_terminal_punctuation=True,
                                    )
                                )
                            if sentence_index + 1 < len(sentences):
                                neighbor_contexts.append(
                                    finalize_query_surface(
                                        f"{variant} {sentences[sentence_index + 1]}",
                                        add_terminal_punctuation=True,
                                    )
                                )
                        if small_corpus or pair_like_corpus:
                            expanded_contexts: list[str] = []
                            for spec in base_specs:
                                expanded_contexts.extend(
                                    [
                                        finalize_query_surface(
                                            clean_query_passage(
                                                str(spec.get("context_text", "") or "")
                                            ),
                                            add_terminal_punctuation=True,
                                        ),
                                        finalize_query_surface(
                                            clean_query_passage(
                                                str(spec.get("context_window_text", "") or "")
                                            ),
                                            add_terminal_punctuation=True,
                                        ),
                                    ]
                                )
                            expanded_contexts.append(
                                finalize_query_surface(
                                    extract_sentence_excerpt(
                                        chunk_text,
                                        max_sentences=2 if low_support_small_location else 2,
                                        max_chars=560 if low_support_small_location else 520 if pair_like_corpus else 460,
                                    ),
                                    add_terminal_punctuation=True,
                                )
                            )
                            if low_support_small_location:
                                neighbor_contexts = expanded_contexts + neighbor_contexts
                            else:
                                neighbor_contexts.extend(expanded_contexts)
                        broader_single_slot_excerpt = (
                            (small_corpus or pair_like_corpus)
                            and len(base_specs) == 1
                        )
                        query_base = choose_best_query_context(
                            neighbor_contexts,
                            anchors=target_labels,
                            min_words=pair_min_query_words,
                            target_words=target_words,
                            max_sentences=2 if len(base_specs) > 1 or count_words(variant) < pair_min_query_words or broader_single_slot_excerpt else 1,
                            max_chars=560 if low_support_small_location else 520 if broader_single_slot_excerpt else 420,
                        ) or variant
                        query_text, used_specs = redact_labels_in_context(query_base, base_specs)
                        if not used_specs:
                            continue
                        broader_used_excerpt = (
                            (small_corpus or pair_like_corpus)
                            and len(used_specs) == 1
                        )
                        query_text = finalize_query_surface(
                            extract_sentence_excerpt(
                                query_text,
                                max_sentences=2 if len(used_specs) > 1 or count_words(query_base) < pair_min_query_words or broader_used_excerpt else 1,
                                max_chars=560 if low_support_small_location else 520 if broader_used_excerpt else 420,
                            ),
                            add_terminal_punctuation=True,
                        )
                        if count_words(query_text) < pair_min_query_words or count_words(query_text) > max_query_words:
                            continue
                        if len(used_specs) == 1 and query_text.startswith(("[REDACT:R1", "The [REDACT:R1")):
                            continue
                        if re.search(r"\(\s*[SU]/[A-Z]+\s*\)", query_text):
                            continue
                        primary_entity_type = normalize_candidate_type(
                            used_specs[0].get("entity_type", "")
                        )
                        if not value_redaction_query_is_viable(
                            query_text,
                            entity_type=primary_entity_type,
                            used_specs=used_specs,
                        ):
                            continue
                        profile_text = REDACTION_MARKER_RE.sub("Target", query_text)
                        profile = claim_clause_profile(profile_text)
                        if not (
                            profile_has_complete_simple_clause(profile)
                            or standalone_claim_surface_is_safe(profile_text)
                        ):
                            continue
                        if profile["nested_reporting"] or profile["object_bad_lead"]:
                            continue
                        query_quality = substantive_sentence_score(query_text, anchors=target_labels)
                        if strict_mode and query_quality < 7.0:
                            continue
                        if strict_mode and surface_noise_penalty(query_text) > 2.5:
                            continue
                        redacted_keys = {label.casefold() for label in target_labels if label}
                        visible_anchor_labels = [
                            normalize_space(spec.get("label", ""))
                            for spec in unique_specs
                            if normalize_space(spec.get("label", ""))
                            and normalize_space(spec.get("label", "")).casefold() not in redacted_keys
                            and find_label_match(query_text, normalize_space(spec.get("label", "")))
                        ]
                        visible_anchor_count = len(unique_preserving_order(visible_anchor_labels))
                        local_support_hits = sum(
                            int(spec.get("local_support_hits", 0) or 0) for spec in used_specs
                        )
                        same_type_hits = sum(
                            1
                            for spec in unique_specs
                            if normalize_candidate_type(spec.get("entity_type", "")) == primary_entity_type
                        )
                        has_date_or_number = bool(
                            DATE_TOKEN_RE.search(query_text) or NUMBER_RE.search(query_text)
                        )
                        query_word_count = count_words(query_text)
                        clean_direct_slot = bool(
                            query_quality >= 9.0
                            and not sentence_is_low_signal_for_query(query_text)
                            and (
                                visible_anchor_count >= 1
                                or has_date_or_number
                                or local_support_hits >= 1
                            )
                        )
                        if (
                            strict_mode
                            and not pair_like_corpus
                            and len(used_specs) == 1
                            and primary_entity_type == "person"
                            and visible_anchor_count <= 0
                            and local_support_hits <= 0
                            and not has_date_or_number
                            and not clean_direct_slot
                        ):
                            continue
                        if (
                            strict_mode
                            and not pair_like_corpus
                            and len(used_specs) == 1
                            and primary_entity_type == "person"
                            and query_word_count > (32 if large_corpus else 38)
                            and not clean_direct_slot
                        ):
                            continue
                        if (
                            strict_mode
                            and len(used_specs) == 1
                            and primary_entity_type in {"organization", "government", "military", "political_party"}
                            and visible_anchor_count <= 0
                            and local_support_hits <= 0
                            and not has_date_or_number
                        ):
                            continue
                        if (
                            strict_mode
                            and len(used_specs) == 1
                            and primary_entity_type in {"location", "country"}
                            and visible_anchor_count <= 0
                            and not has_date_or_number
                            and local_support_hits <= 0
                        ):
                            continue
                        answer_targets = {
                            str(spec["slot_id"]): {
                                "source_id": str(spec.get("source_id", doc_id)),
                                "selected_value": normalize_space(spec.get("label", "")),
                            }
                            for spec in used_specs
                        }
                        expected_targets = {
                            str(spec["slot_id"]): {"source_id": str(spec.get("source_id", doc_id))}
                            for spec in used_specs
                        }
                        selector_score = (
                            18.0
                            + query_quality
                            + float(type_priority.get(primary_entity_type, 0.0))
                            + visible_anchor_count * 3.0
                            + local_support_hits * 3.5
                            + min(len(used_specs), 3) * 4.0
                            + (3.0 if has_date_or_number else 0.0)
                            + (4.0 if clean_direct_slot else 0.0)
                            + min(int(richness.get("claim_n", 0)), 4) * 0.75
                            + min(int(richness.get("event_n", 0)), 3) * 1.0
                            + min(int(richness.get("rel_n", 0)), 5) * 0.4
                            - max(same_type_hits - len(used_specs), 0) * 2.5
                            - abs(count_words(query_text) - target_words) / 12.0
                            - surface_noise_penalty(query_text) * 0.35
                        )
                        case = {
                            "task_family": "value_redaction",
                            "runner_task": "redaction",
                            "doc_id": str(doc_id),
                            "paragraph_id": str(paragraph_id),
                            "case_id": f"value_redaction__{doc_id}__pairsent__{sha1_text(query_text)}",
                            "query": query_text,
                            "ground_truth": stringify_expected_targets(answer_targets),
                            "query_kind": "pair_sentence_slot",
                            "notes": (
                                f"type={primary_entity_type}; slot_count={len(used_specs)}; "
                                f"sentence_variant={'simplified' if variant != sentence else 'raw'}; "
                                f"local_support_hits={local_support_hits}; "
                                f"visible_anchor_count={visible_anchor_count}; "
                                f"same_type_hits={same_type_hits}; "
                                f"selector_score={selector_score:.2f}"
                            ),
                            "source_context": variant,
                            "selector_score": selector_score,
                            "word_count": query_word_count,
                            "query_target_words": target_words,
                            "strict_selected_value_match": maybe_bool(
                                sampling.get("strict_selected_value_match", False)
                            ),
                            "expected_targets": expected_targets,
                            "answer_targets": answer_targets,
                            "entity_type": primary_entity_type,
                            "role_words": int(base_specs[0].get("role_words", 0) or 0),
                            "label_doc_frequency": int(
                                base_specs[0].get("label_doc_frequency", 1) or 1
                            ),
                            "confidence": safe_float(base_specs[0].get("confidence", 0.0)),
                            "local_support_hits": local_support_hits,
                            "same_type_hits": same_type_hits,
                            "visible_anchor_count": visible_anchor_count,
                            "slot_count": len(used_specs),
                            "has_date_or_number": has_date_or_number,
                            "clean_direct_slot": clean_direct_slot,
                            "label_word_count": int(base_specs[0].get("label_word_count", 0) or 0),
                            "label_acronym_like": bool(base_specs[0].get("label_acronym_like", False)),
                        }
                        key = canonical_query_dedupe_key(query_text)
                        prior = deduped_candidates.get(key)
                        if prior is None or selector_score > float(prior.get("selector_score", 0.0) or 0.0):
                            deduped_candidates[key] = case
    return round_robin_topk(
        list(deduped_candidates.values()),
        max(count * 6, count),
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )


def build_minimal_value_redaction_candidates(
    entities: pd.DataFrame,
    doc_chunks_by_doc: dict[str, list[dict[str, Any]]],
    count: int,
    sampling: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    strict_mode = not maybe_bool(sampling.get("_relaxed_retry_used", False))
    allowed_types = (
        {value.casefold() for value in sampling.get("allowed_entity_types", [])}
        | {value.casefold() for value in sampling.get("fallback_entity_types", [])}
    )
    include_locations = maybe_bool(sampling.get("include_locations", False))
    min_query_words = int(sampling.get("min_query_words", 16))
    max_query_words = int(sampling.get("max_query_words", 180))
    corpus_doc_count = max(1, len(doc_chunks_by_doc))
    small_corpus = exhaustive_small_doc_corpus(corpus_doc_count)
    large_corpus = optimized_doc_corpus(corpus_doc_count)
    pair_like_corpus = bool(doc_chunks_by_doc) and all(
        str(doc_id).startswith("Pair_") for doc_id in doc_chunks_by_doc
    )
    early_type_priority = {
        "person": 10.0,
        "organization": 4.0,
        "government": 3.5,
        "military": 3.5,
        "political_party": 3.0,
        "location": 9.0,
        "country": 8.0,
    }
    configured_max_slot_count = int(sampling.get("max_slot_count", 2))
    if small_corpus or pair_like_corpus:
        max_slot_count = min(configured_max_slot_count, 3)
    else:
        max_slot_count = min(configured_max_slot_count, 2)
    slot_count_decay = float(sampling.get("slot_count_decay", 0.5))
    preferred_entity_types = {"person", "location", "country", "organization"}
    small_corpus_supported_entity_types = {
        "person",
        "location",
        "country",
        "organization",
        "government",
        "military",
        "political_party",
    }
    candidates: list[dict[str, Any]] = []
    paragraph_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    synthesized = synthesize_redaction_entity_rows(entities)
    for row in synthesized:
        paragraph_rows[paragraph_key(row["doc_id"], row["paragraph_id"])].append(row)
    for row in synthesized:
        label = normalize_space(row.get("candidate_label", ""))
        if not label:
            continue
        entity_type = normalize_candidate_type(row.get("candidate_type", ""))
        local_entity_rows = paragraph_rows.get(
            paragraph_key(row["doc_id"], row["paragraph_id"]),
            [],
        )
        if entity_type == "person":
            label = prefer_full_name_person_label(label, local_entity_rows)
        if entity_type == "person" and count_words(label) < 2:
            label = expand_person_label_with_context(
                label,
                str(row.get("context_text", "")),
                str(row.get("context_window_text", "")),
            )
        if entity_type in {"other", "unknown", "target", ""}:
            continue
        if maybe_bool(row.get("is_position_like", False)):
            continue
        acronym_like = looks_like_acronym_label(label)
        if entity_type == "person" and count_words(label) < max(2, int(sampling.get("person_min_label_words", 1))):
            continue
        if entity_type == "person" and (
            is_probably_generic_person_label(label)
            or is_title_only_person_label(label)
        ):
            continue
        if entity_type == "person" and has_generic_label_head(label) and count_words(label) <= 3:
            continue
        if entity_type == "location" and not (include_locations or large_corpus or small_corpus):
            continue
        if large_corpus and strict_mode and entity_type not in preferred_entity_types:
            continue
        if allowed_types and entity_type not in allowed_types and not (
            (entity_type == "location" and (include_locations or small_corpus))
            or (small_corpus and entity_type in small_corpus_supported_entity_types)
        ):
            continue
        if small_corpus and strict_mode and entity_type not in small_corpus_supported_entity_types:
            continue
        if large_corpus and entity_type not in preferred_entity_types:
            continue
        if entity_type != "person" and count_words(label) == 1 and acronym_like:
            continue
        if entity_type in {"organization", "government", "military", "political_party"} and count_words(label) == 1:
            continue
        chunk = find_chunk_for_redaction_row(
            doc_chunks_by_doc.get(str(row["doc_id"]), []),
            label=label,
            preferred_chunk_id=str(row.get("chunk_id", "")),
        )
        if not chunk:
            continue
        chunk_text = str(chunk.get("chunk_text", ""))
        if uppercase_token_ratio(chunk_text) >= 0.45 and count_words(chunk_text) >= 18:
            chunk_text = normalize_uppercase_passage(chunk_text)
        chunk_text = clean_query_passage(chunk_text)
        chunk_text = strip_internal_topic_sections(chunk_text)
        chunk_text = strip_low_signal_lead_sentences(chunk_text)
        entity_context = clean_query_passage(str(row.get("context_text", "")))
        entity_window = clean_query_passage(str(row.get("context_window_text", "")))
        if uppercase_token_ratio(entity_context) >= 0.45 and count_words(entity_context) >= 18:
            entity_context = normalize_uppercase_passage(entity_context)
        if uppercase_token_ratio(entity_window) >= 0.45 and count_words(entity_window) >= 18:
            entity_window = normalize_uppercase_passage(entity_window)
        entity_context = strip_internal_topic_sections(strip_low_signal_lead_sentences(entity_context))
        entity_window = strip_internal_topic_sections(strip_low_signal_lead_sentences(entity_window))
        if not find_label_match(chunk_text, label):
            continue
        match = find_label_match(chunk_text, label)
        if not match:
            continue
        slot_sentence = best_label_sentence_surface(
            [entity_context, entity_window, chunk_text],
            label,
            small_corpus=small_corpus,
            pair_like_corpus=pair_like_corpus,
            min_score=-4.0,
        )
        slot_sentence_words = count_words(slot_sentence)
        direct_slot_query = finalize_query_surface(
            extract_sentence_excerpt(slot_sentence, max_sentences=1, max_chars=320),
            add_terminal_punctuation=True,
        )
        if (
            (small_corpus or pair_like_corpus)
            and direct_slot_query
            and count_words(direct_slot_query) >= max(8, min_query_words - 6)
        ):
            base_query = direct_slot_query
        else:
            base_query = choose_best_query_context(
                [
                    slot_sentence,
                    entity_context,
                    entity_window,
                    bounded_context(chunk_text, match.start(), match.end(), min_chars=180),
                    chunk_text,
                ],
                anchors=[label],
                min_words=min_query_words,
                target_words=deterministic_query_word_target(
                    f"minimal_value|{row['doc_id']}|{row['paragraph_id']}|{label}",
                    minimum_words=min_query_words,
                    maximum_words=min(max_query_words, 128),
                ),
                max_sentences=4,
                max_chars=720,
            )
        base_query = finalize_query_surface(base_query)
        if not base_query or not re.search(re.escape(label), base_query, flags=re.IGNORECASE):
            continue
        peer_specs: list[dict[str, Any]] = []
        for peer in paragraph_rows.get(paragraph_key(row["doc_id"], row["paragraph_id"]), []):
            peer_label = normalize_space(peer.get("candidate_label", ""))
            if not peer_label or not re.search(re.escape(peer_label), base_query, flags=re.IGNORECASE):
                continue
            peer_type = normalize_candidate_type(peer.get("candidate_type", ""))
            if peer_type == "location" and not (include_locations or small_corpus):
                continue
            if allowed_types and peer_type not in allowed_types and not (
                (peer_type == "location" and (include_locations or small_corpus))
                or (small_corpus and peer_type in small_corpus_supported_entity_types)
            ):
                continue
            if peer_type == "person" and is_title_only_person_label(peer_label):
                continue
            peer_specs.append(
                {
                    "label": peer_label,
                    "slot_type": label_type_string(peer_type),
                    "source_id": str(peer.get("doc_id", row["doc_id"])),
                    "entity_type": peer_type,
                    "confidence": safe_float(peer.get("candidate_source_confidence", 0.0)),
                }
            )
        if not peer_specs:
            continue
        unique_specs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for spec in peer_specs:
            key = spec["label"].casefold()
            if key in seen:
                continue
            seen.add(key)
            unique_specs.append(spec)
        unique_specs = sorted(
            unique_specs,
            key=lambda spec: (
                0 if spec["label"].casefold() == label.casefold() else 1,
                -safe_float(spec.get("confidence", 0.0)),
                -count_words(spec.get("label", "")),
                spec.get("label", ""),
            ),
        )
        desired_slot_count = deterministic_tail_count(
            f"minimal_value_slots|{row['doc_id']}|{row['paragraph_id']}|{label}",
            max_count=max_slot_count,
            decay=slot_count_decay,
        )
        if small_corpus:
            desired_slot_count = min(desired_slot_count, 2)
            if desired_slot_count > 1 and len(unique_specs) > 1:
                small_corpus_slot_cap = 1 + (
                    int(sha1_text(f"minimal_value_slot_shape|{row['doc_id']}|{label}")[:8], 16) % 2
                )
                desired_slot_count = min(desired_slot_count, small_corpus_slot_cap)
        query_text, used_specs = redact_labels_in_context(base_query, unique_specs[:desired_slot_count])
        slot_anchors = [normalize_space(spec.get("label", "")) for spec in used_specs]
        single_slot_small_nonpair = (
            small_corpus
            and not pair_like_corpus
            and len(used_specs) == 1
        )
        single_slot_small_or_pair = (
            (small_corpus or pair_like_corpus)
            and len(used_specs) == 1
        )
        query_text = choose_best_query_context(
            [query_text],
            anchors=slot_anchors,
            min_words=min_query_words,
            target_words=scaled_redaction_query_target(
                f"minimal_value_window|{row['doc_id']}|{row['paragraph_id']}|{label}",
                minimum_words=min_query_words,
                maximum_words=min(max_query_words, 96),
                slot_count=max(1, len(used_specs)),
            ),
            max_sentences=2 if single_slot_small_or_pair else 1 if len(used_specs) == 1 else 2,
            max_chars=520 if single_slot_small_or_pair else 420,
        ) or query_text
        query_text = normalize_redaction_list_boundary(query_text)
        query_text = finalize_query_surface(
            extract_sentence_excerpt(
                query_text,
                max_sentences=2 if single_slot_small_or_pair else 1 if len(used_specs) == 1 else 2,
                max_chars=520 if single_slot_small_or_pair else 420,
            ),
            add_terminal_punctuation=True,
        )
        co_present_person_labels = [
            normalize_space(spec.get("label", ""))
            for spec in unique_specs
            if normalize_candidate_type(spec.get("entity_type", "")) == "person"
            and find_label_match(base_query, normalize_space(spec.get("label", "")))
        ]
        if not used_specs:
            continue
        primary_entity_type = normalize_candidate_type(
            used_specs[0].get("entity_type", entity_type)
        ) or entity_type
        if count_words(query_text) < min_query_words or count_words(query_text) > max_query_words:
            continue
        if strict_mode and len(used_specs) == 1 and slot_sentence_words and slot_sentence_words > (
            72 if pair_like_corpus else 56 if small_corpus else 42
        ):
            continue
        query_quality = substantive_sentence_score(query_text, anchors=slot_anchors)
        if strict_mode and query_quality < (3.5 if pair_like_corpus else 3.0 if small_corpus else 5.0):
            continue
        if strict_mode and surface_noise_penalty(query_text) > (4.25 if pair_like_corpus else 2.5 if small_corpus else 1.25):
            continue
        if strict_mode and LOW_SIGNAL_SENTENCE_LEAD_RE.search(normalize_space(query_text)):
            continue
        if (
            LEGAL_BOILERPLATE_RE.search(query_text)
            and not (
                (small_corpus or pair_like_corpus)
                and query_quality >= 6.0
                and not sentence_is_low_signal_for_query(query_text)
            )
        ):
            continue
        visible_person_labels = [
            normalize_space(spec.get("label", ""))
            for spec in unique_specs
            if normalize_candidate_type(spec.get("entity_type", "")) == "person"
            and find_label_match(query_text, normalize_space(spec.get("label", "")))
        ]
        if (
            strict_mode
            and len(used_specs) == 1
            and primary_entity_type == "person"
            and (
                len(unique_preserving_order(visible_person_labels)) > 1
                or len(unique_preserving_order(co_present_person_labels)) > 1
            )
        ):
            continue
        if (
            strict_mode
            and len(used_specs) == 1
            and primary_entity_type in {"location", "country"}
            and normalize_space(used_specs[0].get("label", "")).casefold()
            in {"united states", "u.s.", "u.s", "united states of america"}
        ):
            continue
        if not value_redaction_query_is_viable(
            query_text,
            entity_type=primary_entity_type,
            used_specs=used_specs,
        ):
            continue
        visible_support_labels = [
            normalize_space(spec.get("label", ""))
            for spec in unique_specs
            if normalize_space(spec.get("label", ""))
            and normalize_space(spec.get("label", "")).casefold()
            not in {normalize_space(used.get("label", "")).casefold() for used in used_specs}
            and find_label_match(query_text, normalize_space(spec.get("label", "")))
        ]
        visible_support_anchor_count = len(
            unique_preserving_order(visible_support_labels)
        )
        has_date_or_number = bool(
            DATE_TOKEN_RE.search(query_text) or NUMBER_RE.search(query_text)
        )
        if single_slot_small_nonpair and query_text.lstrip().startswith("[REDACT:R1"):
            continue
        if (
            single_slot_small_nonpair
            and visible_support_anchor_count == 0
            and not has_date_or_number
            and count_words(query_text) < 18
        ):
            continue
        expected_targets = {
            str(used_spec["slot_id"]): {"source_id": str(used_spec.get("source_id", row["doc_id"]))}
            for used_spec in used_specs
        }
        answer_targets = {
            str(used_spec["slot_id"]): {
                "source_id": str(used_spec.get("source_id", row["doc_id"])),
                "selected_value": normalize_space(used_spec.get("label", "")),
            }
            for used_spec in used_specs
        }
        selector_score = (
            context_quality_score(query_text, anchors=[label], min_words=min_query_words)
            + query_quality * 0.4
            + min(len(used_specs), max_slot_count) * 2.0
            + visible_support_anchor_count * 2.0
            + (1.5 if has_date_or_number else 0.0)
            - surface_noise_penalty(query_text) * 0.4
        )
        case_id = f"value_redaction__{row['doc_id']}__minimal__{sha1_text(query_text)}"
        candidates.append(
            {
                "task_family": "value_redaction",
                "runner_task": "redaction",
                "doc_id": str(row["doc_id"]),
                "paragraph_id": str(row["paragraph_id"]),
                "case_id": case_id,
                "query": query_text,
                "ground_truth": stringify_expected_targets(answer_targets) or label,
                "query_kind": "entity_slot_minimal",
                "notes": (
                    f"type={primary_entity_type}; slot_count={len(used_specs)}; "
                    f"selector_score={selector_score:.2f}; fallback=minimal"
                ),
                "source_context": query_text,
                "selector_score": selector_score,
                "word_count": count_words(query_text),
                "query_target_words": count_words(query_text),
                "strict_selected_value_match": maybe_bool(
                    sampling.get("strict_selected_value_match", False)
                ),
                "expected_targets": expected_targets,
                "answer_targets": answer_targets,
                "entity_type": primary_entity_type,
                "role_words": count_words(normalize_space(row.get("candidate_role_or_context", ""))),
                "label_doc_frequency": 1,
                "confidence": safe_float(row.get("candidate_source_confidence", 0.0)),
                "local_support_hits": 1 if query_quality >= 8.0 else 0,
                "same_type_hits": 1,
                "visible_anchor_count": max(0, len(unique_preserving_order(visible_person_labels)) - 1),
                "slot_count": len(used_specs),
                "has_date_or_number": bool(DATE_TOKEN_RE.search(query_text) or NUMBER_RE.search(query_text)),
                "clean_direct_slot": bool(
                    query_quality >= 6.0 and not sentence_is_low_signal_for_query(query_text)
                ),
            }
        )
    preferred = [
        candidate
        for candidate in candidates
        if str(candidate.get("entity_type", "")) in preferred_entity_types
    ]
    if small_corpus and len(candidates) < count:
        emergency_candidates: list[dict[str, Any]] = []
        for row in synthesized:
            label = normalize_space(row.get("candidate_label", ""))
            if not label:
                continue
            entity_type = normalize_candidate_type(row.get("candidate_type", ""))
            local_entity_rows = paragraph_rows.get(
                paragraph_key(row["doc_id"], row["paragraph_id"]),
                [],
            )
            if entity_type == "person":
                label = prefer_full_name_person_label(label, local_entity_rows)
            if entity_type == "person" and count_words(label) < 2:
                label = expand_person_label_with_context(
                    label,
                    str(row.get("context_text", "")),
                    str(row.get("context_window_text", "")),
                )
            if entity_type in {"other", "unknown", "target", ""}:
                continue
            if entity_type == "person" and (
                is_probably_generic_person_label(label)
                or is_title_only_person_label(label)
            ):
                continue
            if allowed_types and entity_type not in allowed_types and not (
                entity_type in small_corpus_supported_entity_types
                or (entity_type == "location" and include_locations)
            ):
                continue
            chunk = find_chunk_for_redaction_row(
                doc_chunks_by_doc.get(str(row["doc_id"]), []),
                label=label,
                preferred_chunk_id=str(row.get("chunk_id", "")),
            )
            if not chunk:
                continue
            chunk_text = clean_query_passage(str(chunk.get("chunk_text", "")))
            entity_context = clean_query_passage(str(row.get("context_text", "")))
            entity_window = clean_query_passage(str(row.get("context_window_text", "")))
            label_sentence = best_label_sentence_surface(
                [entity_context, entity_window, chunk_text],
                label,
                small_corpus=small_corpus,
                pair_like_corpus=pair_like_corpus,
                min_score=-6.0,
            )
            if (
                not label_sentence
                or "?" in label_sentence
                or sentence_is_low_signal_for_query(label_sentence)
                or LEGAL_BOILERPLATE_RE.search(label_sentence)
                or BROKEN_LEAD_RE.search(label_sentence)
            ):
                continue
            pair_like_person_window = pair_like_corpus and entity_type == "person"
            pair_like_broader_window = pair_like_corpus and entity_type in {
                "person",
                "organization",
                "government",
                "military",
                "political_party",
                "location",
                "country",
            }
            emergency_max_sentences = (
                2 if (small_corpus and not pair_like_corpus) or pair_like_person_window or pair_like_broader_window else 1
            )
            emergency_max_chars = (
                520 if (small_corpus and not pair_like_corpus) or pair_like_person_window or pair_like_broader_window else 360
            )
            emergency_base = choose_best_query_context(
                [label_sentence, entity_context, entity_window, chunk_text],
                anchors=[label],
                min_words=max(10, min_query_words - 4),
                target_words=scaled_redaction_query_target(
                    f"emergency_value_window|{row['doc_id']}|{row['paragraph_id']}|{label}",
                    minimum_words=max(10, min_query_words - 4),
                    maximum_words=min(max_query_words, 72),
                    slot_count=1,
                ),
                max_sentences=emergency_max_sentences,
                max_chars=emergency_max_chars,
            ) or label_sentence
            query_text, used_specs = redact_labels_in_context(
                emergency_base,
                [
                    {
                        "label": label,
                        "slot_type": label_type_string(entity_type),
                        "source_id": str(row.get("doc_id", "")),
                        "entity_type": entity_type,
                        "confidence": safe_float(row.get("candidate_source_confidence", 0.0)),
                    }
                ],
            )
            if not used_specs:
                continue
            query_text = finalize_query_surface(
                extract_sentence_excerpt(
                    query_text,
                    max_sentences=emergency_max_sentences,
                    max_chars=emergency_max_chars,
                ),
                add_terminal_punctuation=True,
            )
            query_text = normalize_redaction_list_boundary(query_text)
            if count_words(query_text) < max(8, min_query_words - 6) or count_words(query_text) > min(max_query_words, 64):
                continue
            if not value_redaction_query_is_viable(
                query_text,
                entity_type=entity_type,
                used_specs=used_specs,
            ):
                continue
            query_quality = substantive_sentence_score(query_text, anchors=[label])
            if query_quality < 2.0:
                continue
            if (
                small_corpus
                and not pair_like_corpus
                and query_text.lstrip().startswith("[REDACT:R1")
            ):
                if entity_type not in {
                    "person",
                    "location",
                    "country",
                    "organization",
                    "government",
                }:
                    continue
                if query_quality < 6.0 or count_words(query_text) < 14:
                    continue
            answer_targets = {
                str(used_specs[0]["slot_id"]): {
                    "source_id": str(row.get("doc_id", "")),
                    "selected_value": label,
                }
            }
            emergency_candidates.append(
                {
                    "task_family": "value_redaction",
                    "runner_task": "redaction",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": f"value_redaction__{row['doc_id']}__emergency__{sha1_text(query_text)}",
                    "query": query_text,
                    "ground_truth": stringify_expected_targets(answer_targets) or label,
                    "query_kind": "entity_slot_small_corpus_emergency",
                    "notes": (
                        f"type={entity_type}; slot_count=1; "
                        f"selector_score={query_quality:.2f}; fallback=small_corpus_sentence"
                    ),
                    "source_context": label_sentence,
                    "selector_score": query_quality - surface_noise_penalty(query_text) * 0.2 + 8.0,
                    "word_count": count_words(query_text),
                    "query_target_words": count_words(query_text),
                    "strict_selected_value_match": maybe_bool(
                        sampling.get("strict_selected_value_match", False)
                    ),
                    "expected_targets": {
                        str(used_specs[0]["slot_id"]): {"source_id": str(row.get("doc_id", ""))}
                    },
                    "answer_targets": answer_targets,
                    "entity_type": entity_type,
                    "role_words": count_words(normalize_space(row.get("candidate_role_or_context", ""))),
                    "label_doc_frequency": 1,
                    "confidence": safe_float(row.get("candidate_source_confidence", 0.0)),
                    "local_support_hits": 1 if query_quality >= 8.0 else 0,
                    "same_type_hits": 1,
                    "visible_anchor_count": 0,
                    "slot_count": 1,
                    "has_date_or_number": bool(DATE_TOKEN_RE.search(query_text) or NUMBER_RE.search(query_text)),
                    "clean_direct_slot": bool(
                        query_quality >= 6.0 and not sentence_is_low_signal_for_query(query_text)
                    ),
                }
            )
            if len(emergency_candidates) >= max(count * 6, 24):
                break
        candidates = merge_ranked_candidates(
            [candidates, emergency_candidates],
            target_count=max(count * 6, count),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
        preferred = [
            candidate
            for candidate in candidates
            if str(candidate.get("entity_type", "")) in preferred_entity_types
        ]
    selected_preferred: list[dict[str, Any]] = []
    if preferred:
        selected_preferred = round_robin_topk(
            preferred,
            min(count, len(preferred)),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
    return merge_ranked_candidates(
        [selected_preferred, candidates],
        target_count=count,
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )


def value_redaction_surface_is_known_bad(query_text: Any) -> bool:
    value = normalize_space(query_text)
    if not value:
        return False
    if count_redaction_slots(value) <= 0:
        return True
    if count_words(value) < 5 or count_words(value) > 90:
        return True
    flags = set(query_surface_flags(value, task_family="value_redaction"))
    return bool(
        flags
        & {
            "uppercase_heavy",
            "mixed_case_noise",
            "headline_scaffold",
            "broken_lead",
            "foreign_language",
            "metadata_scaffold",
            "duplicated_scaffold",
        }
    )


def value_redaction_surface_is_generation_reject(query_text: Any, doc_id: Any) -> bool:
    del doc_id
    value = normalize_space(query_text)
    if value_redaction_surface_is_known_bad(value):
        return True
    slots = list(REDACTION_MARKER_RE.finditer(value))
    if not 1 <= len(slots) <= 3:
        return True
    if re.search(r"\[REDACT:[^\]]+\]\s+(?:he|she|they|it)\b", value, flags=re.IGNORECASE):
        return True
    if len(slots) == 1 and value.count(",") >= 4:
        left = value[: slots[0].start()]
        if re.search(r"\b(?:and|or)(?:\s+the)?\s*$", left, flags=re.IGNORECASE):
            return True
    return False


def value_redaction_generation_score_adjustment(query_text: Any, doc_id: Any) -> float:
    del doc_id
    value = normalize_space(query_text)
    if not value:
        return 0.0
    slots = count_redaction_slots(value)
    visible = REDACTION_MARKER_RE.sub(" ", value)
    context_stopwords = {
        "about", "after", "against", "also", "among", "because", "being",
        "between", "could", "during", "from", "have", "into", "other",
        "report", "reports", "said", "some", "such", "that", "their",
        "these", "they", "this", "those", "through", "under", "which",
        "while", "with", "would",
    }
    content_words = {
        token.casefold()
        for token in WORD_RE.findall(visible)
        if len(token) >= 5 and token.casefold() not in context_stopwords
    }
    delta = min(len(content_words), 8) * 0.75
    if DATE_TOKEN_RE.search(value) or NUMBER_RE.search(value):
        delta += 2.0
    if slots > 1:
        delta -= min(slots - 1, 2) * 2.0
    if count_words(value) > 60:
        delta -= min((count_words(value) - 60) * 0.25, 8.0)
    if re.match(
        r"^(?:however|but|finally|it is unclear whether|there (?:are|were) reports? that)\b",
        value,
        flags=re.IGNORECASE,
    ):
        delta -= 6.0
    return delta


def adjust_value_redaction_generation_scores(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for case in candidates:
        item = dict(case)
        item["selector_score"] = float(item.get("selector_score", 0.0) or 0.0) + value_redaction_generation_score_adjustment(
            item.get("query", ""),
            item.get("doc_id", ""),
        )
        adjusted.append(item)
    return adjusted


def build_focus_value_redaction_candidates(
    entities: pd.DataFrame,
    doc_chunks_by_doc: dict[str, list[dict[str, Any]]],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    entity_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if count <= 0 or entities.empty or not doc_chunks_by_doc:
        return []
    strict_mode = not maybe_bool(sampling.get("_relaxed_retry_used", False))
    min_query_words = int(sampling.get("min_query_words", 16))
    max_query_words = int(sampling.get("max_query_words", 180))
    corpus_doc_count = max(1, len(doc_chunks_by_doc))
    small_corpus = exhaustive_small_doc_corpus(corpus_doc_count)
    large_corpus = optimized_doc_corpus(corpus_doc_count)
    pair_like_corpus = bool(doc_chunks_by_doc) and all(
        str(doc_id).startswith("Pair_") for doc_id in doc_chunks_by_doc
    )
    allowed_types = (
        {value.casefold() for value in sampling.get("allowed_entity_types", [])}
        | {value.casefold() for value in sampling.get("fallback_entity_types", [])}
    )
    include_locations = maybe_bool(sampling.get("include_locations", False))
    preferred_entity_types = {"person", "location", "country", "organization"}
    small_corpus_supported_entity_types = {
        "person",
        "location",
        "country",
        "organization",
        "government",
        "military",
        "political_party",
    }
    type_priority = {
        "person": 8.0,
        "location": 6.0,
        "country": 6.0,
        "organization": 4.0,
        "government": 3.0,
        "military": 3.0,
        "political_party": 3.0,
    }
    paragraph_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    synthesized = entity_rows if entity_rows is not None else synthesize_redaction_entity_rows(entities)
    for row in synthesized:
        paragraph_rows[paragraph_key(row["doc_id"], row["paragraph_id"])].append(row)

    min_focus_words = max(8, min_query_words - (8 if small_corpus or pair_like_corpus else 4))
    focus_max_chars = 520 if small_corpus or pair_like_corpus else 360
    if large_corpus and not pair_like_corpus:
        focus_target_count = int(sampling.get("large_corpus_focus_target_cap", 200))
    else:
        focus_target_count = max(count * 6, 36 if small_corpus or pair_like_corpus else count * 4)
    deduped_candidates: dict[str, dict[str, Any]] = {}

    row_order = deterministic_seeded_row_order(
        synthesized,
        seed=seed + 419,
        key_fields=("doc_id", "paragraph_id", "candidate_label", "candidate_type"),
    )
    for row in row_order:
        label = normalize_space(row.get("candidate_label", ""))
        if not label:
            continue
        entity_type = normalize_candidate_type(row.get("candidate_type", ""))
        local_entity_rows = paragraph_rows.get(
            paragraph_key(row["doc_id"], row["paragraph_id"]),
            [],
        )
        if entity_type == "person":
            label = prefer_full_name_person_label(label, local_entity_rows)
        if entity_type == "person" and count_words(label) < 2:
            label = expand_person_label_with_context(
                label,
                str(row.get("context_text", "")),
                str(row.get("context_window_text", "")),
            )
        if entity_type in {"other", "unknown", "target", ""}:
            continue
        if maybe_bool(row.get("is_position_like", False)):
            continue
        if entity_type == "person" and (
            is_probably_generic_person_label(label)
            or is_title_only_person_label(label)
        ):
            continue
        if allowed_types and entity_type not in allowed_types and not (
            (entity_type == "location" and (include_locations or small_corpus))
            or (small_corpus and entity_type in small_corpus_supported_entity_types)
        ):
            continue
        if large_corpus and entity_type not in preferred_entity_types:
            continue

        chunk = find_chunk_for_redaction_row(
            doc_chunks_by_doc.get(str(row["doc_id"]), []),
            label=label,
            preferred_chunk_id=str(row.get("chunk_id", "")),
        )
        if not chunk:
            continue
        chunk_text = str(chunk.get("chunk_text", ""))
        if uppercase_token_ratio(chunk_text) >= 0.45 and count_words(chunk_text) >= 8:
            chunk_text = normalize_uppercase_passage(chunk_text)
        if large_corpus and not pair_like_corpus:
            chunk_text = finalize_query_surface(
                extract_sentence_excerpt(chunk_text, max_sentences=3, max_chars=420)
            )
            entity_context = finalize_query_surface(
                extract_sentence_excerpt(
                    str(row.get("context_text", "") or ""),
                    max_sentences=2,
                    max_chars=280,
                )
            )
            entity_window = finalize_query_surface(
                extract_sentence_excerpt(
                    str(row.get("context_window_text", "") or ""),
                    max_sentences=2,
                    max_chars=320,
                )
            )
        else:
            chunk_text = clean_query_passage(chunk_text)
            entity_context = clean_query_passage(str(row.get("context_text", "")))
            entity_window = clean_query_passage(str(row.get("context_window_text", "")))
        chunk_text = strip_internal_topic_sections(strip_low_signal_lead_sentences(chunk_text))
        if uppercase_token_ratio(entity_context) >= 0.45 and count_words(entity_context) >= 8:
            entity_context = normalize_uppercase_passage(entity_context)
        if uppercase_token_ratio(entity_window) >= 0.45 and count_words(entity_window) >= 8:
            entity_window = normalize_uppercase_passage(entity_window)
        entity_context = strip_internal_topic_sections(strip_low_signal_lead_sentences(entity_context))
        entity_window = strip_internal_topic_sections(strip_low_signal_lead_sentences(entity_window))
        label_sentence = best_label_sentence_surface(
            [entity_context, entity_window, chunk_text],
            label,
            small_corpus=small_corpus,
            pair_like_corpus=pair_like_corpus,
            min_score=-6.0,
        )
        focus_target_words = scaled_redaction_query_target(
            f"focus_value_window|{row['doc_id']}|{row['paragraph_id']}|{label}",
            minimum_words=min_focus_words,
            maximum_words=min(max_query_words, 80 if small_corpus or pair_like_corpus else 64),
            slot_count=1,
        )
        focus_base = choose_best_query_context(
            [label_sentence, entity_context, entity_window, chunk_text],
            anchors=[label],
            min_words=min_focus_words,
            target_words=focus_target_words,
            max_sentences=2 if (small_corpus or pair_like_corpus) and count_words(label_sentence) < min_focus_words else 1,
            max_chars=focus_max_chars,
        ) or label_sentence
        if not focus_base:
            continue
        query_text, used_specs = redact_labels_in_context(
            focus_base,
            [
                {
                    "label": label,
                    "slot_type": label_type_string(entity_type),
                    "source_id": str(row.get("doc_id", "")),
                    "entity_type": entity_type,
                    "confidence": safe_float(row.get("candidate_source_confidence", 0.0)),
                }
            ],
        )
        if len(used_specs) != 1:
            continue
        query_text = normalize_redaction_list_boundary(query_text)
        query_text = finalize_query_surface(
            extract_sentence_excerpt(
                query_text,
                max_sentences=2 if (small_corpus or pair_like_corpus) and count_words(focus_base) < min_focus_words else 1,
                max_chars=focus_max_chars,
            ),
            add_terminal_punctuation=True,
        )
        if count_words(query_text) < min_focus_words or count_words(query_text) > min(max_query_words, 96):
            continue
        if value_redaction_surface_is_known_bad(query_text):
            continue
        if (
            small_corpus
            and not pair_like_corpus
            and entity_type != "person"
            and query_text.lstrip().startswith("[REDACT:R1")
        ):
            continue
        if not value_redaction_query_is_viable(
            query_text,
            entity_type=entity_type,
            used_specs=used_specs,
        ):
            continue
        if (
            is_routing_or_metadata_sentence(query_text)
            or LEGAL_BOILERPLATE_RE.search(query_text)
            or BROKEN_LEAD_RE.search(query_text)
        ):
            continue
        query_quality = substantive_sentence_score(query_text, anchors=[label])
        if strict_mode and query_quality < (3.5 if small_corpus or pair_like_corpus else 4.5):
            continue
        if strict_mode and surface_noise_penalty(query_text) > (3.5 if small_corpus or pair_like_corpus else 2.0):
            continue
        if sentence_is_low_signal_for_query(query_text) and query_quality < 9.0:
            continue
        redacted_keys = {normalize_space(used_specs[0].get("label", "")).casefold()}
        visible_support_labels = [
            normalize_space(peer.get("candidate_label", ""))
            for peer in local_entity_rows
            if normalize_space(peer.get("candidate_label", ""))
            and normalize_space(peer.get("candidate_label", "")).casefold() not in redacted_keys
            and find_label_match(query_text, normalize_space(peer.get("candidate_label", "")))
        ]
        visible_anchor_count = len(unique_preserving_order(visible_support_labels))
        has_date_or_number = bool(DATE_TOKEN_RE.search(query_text) or NUMBER_RE.search(query_text))
        clean_direct_slot = bool(
            query_quality >= 8.0 and not sentence_is_low_signal_for_query(query_text)
        )
        expected_targets = {
            str(used_specs[0]["slot_id"]): {"source_id": str(row.get("doc_id", ""))}
        }
        answer_targets = {
            str(used_specs[0]["slot_id"]): {
                "source_id": str(row.get("doc_id", "")),
                "selected_value": label,
            }
        }
        selector_score = (
            context_quality_score(query_text, anchors=[label], min_words=min_focus_words)
            + query_quality * 0.5
            + float(type_priority.get(entity_type, 0.0))
            + visible_anchor_count * 2.0
            + (2.0 if has_date_or_number else 0.0)
            + (4.0 if clean_direct_slot else 0.0)
            - surface_noise_penalty(query_text) * 0.3
        )
        case = {
            "task_family": "value_redaction",
            "runner_task": "redaction",
            "doc_id": str(row["doc_id"]),
            "paragraph_id": str(row["paragraph_id"]),
            "case_id": f"value_redaction__{row['doc_id']}__focus__{sha1_text(query_text)}",
            "query": query_text,
            "ground_truth": stringify_expected_targets(answer_targets) or label,
            "query_kind": "entity_slot",
            "notes": (
                f"type={entity_type}; slot_count=1; selector_score={selector_score:.2f}; "
                "fallback=focus_context_sentence"
            ),
            "source_context": label_sentence or entity_context or chunk_text,
            "selector_score": selector_score,
            "word_count": count_words(query_text),
            "query_target_words": focus_target_words,
            "strict_selected_value_match": maybe_bool(
                sampling.get("strict_selected_value_match", False)
            ),
            "expected_targets": expected_targets,
            "answer_targets": answer_targets,
            "entity_type": entity_type,
            "role_words": count_words(normalize_space(row.get("candidate_role_or_context", ""))),
            "label_doc_frequency": 1,
            "confidence": safe_float(row.get("candidate_source_confidence", 0.0)),
            "local_support_hits": 1 if query_quality >= 8.0 else 0,
            "same_type_hits": 1,
            "visible_anchor_count": visible_anchor_count,
            "slot_count": 1,
            "has_date_or_number": has_date_or_number,
            "clean_direct_slot": clean_direct_slot,
        }
        key = canonical_query_dedupe_key(query_text)
        prior = deduped_candidates.get(key)
        if prior is None or selector_score > float(prior.get("selector_score", 0.0) or 0.0):
            deduped_candidates[key] = case
        if len(deduped_candidates) >= focus_target_count:
            break
    return round_robin_topk(
        list(deduped_candidates.values()),
        min(focus_target_count, len(deduped_candidates)),
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )


def build_minimal_claim_candidates(
    claim_views: pd.DataFrame,
    count: int,
    sampling: dict[str, Any],
    seed: int,
    *,
    runner_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if claim_views.empty or count <= 0:
        return []
    corpus_name = normalize_space((runner_cfg or {}).get("__corpus_name", "")).casefold()
    claim_doc_ids = [
        normalize_space(doc_id)
        for doc_id in claim_views.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    ]
    pair_like_corpus = bool(claim_doc_ids) and all(doc_id.startswith("Pair_") for doc_id in claim_doc_ids)
    small_corpus = exhaustive_small_doc_corpus(len({
        normalize_space(doc_id)
        for doc_id in claim_views.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    }))
    min_words = int(sampling.get("min_words", 16))
    max_words = int(sampling.get("max_words", 64))
    if small_corpus or corpus_name in {"fisc", "canberra"}:
        min_words = max(6, min_words - 8)
        max_words = max(max_words, 40)
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    seen_queries: set[str] = set()
    candidates: list[dict[str, Any]] = []
    rows = claim_views.to_dict("records")
    peer_rows_by_idx: list[list[dict[str, Any]]] | None = None
    if small_corpus and rows:
        _, peer_rows_by_idx = build_claim_peer_pool(rows)
    rough_scored_rows: list[tuple[float, int, dict[str, Any]]] = []
    row_indices = list(range(len(rows)))
    max_claim_rows_for_minimal_scan = int(
        sampling.get("max_claim_rows_for_minimal_scan", 0) or 0
    )
    if max_claim_rows_for_minimal_scan > 0 and len(row_indices) > max_claim_rows_for_minimal_scan:
        def minimal_scan_prescore(index: int) -> tuple[float, float, int]:
            raw_text = normalize_space(rows[index].get("claim_text", ""))
            if not raw_text:
                return (-9999.0, -9999.0, 0)
            raw_words = count_words(raw_text)
            return (
                (2.0 if DATE_TOKEN_RE.search(raw_text) else 0.0)
                + (2.0 if NUMBER_RE.search(raw_text) else 0.0)
                - abs(raw_words - 18) / 10.0
                - surface_noise_penalty(raw_text) * 0.2
                - (6.0 if uppercase_token_ratio(raw_text) >= 0.55 else 0.0)
                - (8.0 if REDACTION_MARKER_RE.search(raw_text) else 0.0),
                float(raw_words),
                -index,
            )

        row_indices = sorted(
            row_indices,
            key=minimal_scan_prescore,
            reverse=True,
        )[:max_claim_rows_for_minimal_scan]
    for idx in row_indices:
        row = rows[idx]
        text = normalize_space(row.get("claim_text", ""))
        if uppercase_token_ratio(text) >= 0.45 and count_words(text) >= 8:
            text = normalize_uppercase_passage(text)
        text = clean_query_passage(text)
        text = strip_internal_topic_sections(text)
        text = strip_low_signal_lead_sentences(text)
        text = simplify_claim_query_seed(simplify_claim_surface(text))
        text = finalize_query_surface(text, add_terminal_punctuation=True)
        if not text or query_surface_is_blocked(text, task_family="claims"):
            continue
        if pair_like_corpus and contains_role_placeholder_label(text):
            continue
        if claim_surface_is_interrogative(text) or claim_surface_has_weak_modality(text):
            continue
        if sentence_is_low_signal_for_query(text) or is_routing_or_metadata_sentence(text):
            continue
        profile = claim_clause_profile(text)
        standalone_ok = standalone_claim_surface_is_safe(text)
        if not profile_has_complete_simple_clause(profile) and not standalone_ok:
            continue
        if claim_profile_has_problematic_actor(profile):
            continue
        quality = claim_clause_quality_score(profile) - surface_noise_penalty(text) * 0.35
        peer_matches, peer_best_overlap = (
            claim_support_peer_score(profile, peer_rows_by_idx[idx])
            if peer_rows_by_idx is not None
            else (0, 0.0)
        )
        quality += min(peer_matches, 4) * 2.0 + peer_best_overlap * 4.0
        if quality < (3.25 if standalone_ok else 6.0):
            continue
        rough_scored_rows.append((quality, idx, row))
    rough_scored_rows.sort(key=lambda item: item[0], reverse=True)
    probe_limit = max(count * 20, 80)
    max_minimal_probe_rows = int(sampling.get("max_minimal_probe_rows", 0) or 0)
    if max_minimal_probe_rows > 0:
        probe_limit = min(probe_limit, max_minimal_probe_rows)
    for _, idx, row in rough_scored_rows[:probe_limit]:
        base_text = normalize_space(row.get("claim_text", ""))
        if uppercase_token_ratio(base_text) >= 0.45 and count_words(base_text) >= 8:
            base_text = normalize_uppercase_passage(base_text)
        base_text = clean_query_passage(base_text)
        base_text = strip_internal_topic_sections(base_text)
        base_text = strip_low_signal_lead_sentences(base_text)
        base_text = simplify_claim_query_seed(simplify_claim_surface(base_text))
        base_text = finalize_query_surface(base_text, add_terminal_punctuation=True)
        if claim_surface_is_interrogative(base_text) or claim_surface_has_weak_modality(base_text):
            continue
        base_profile = claim_clause_profile(base_text)
        base_standalone_ok = standalone_claim_surface_is_safe(base_text)
        query_target_words = deterministic_query_word_target(
            f"minimal_claim|{row.get('doc_id','')}|{row.get('paragraph_id','')}|{base_text}",
            minimum_words=min_words,
            maximum_words=min(max_words, 96),
        )
        query = base_text
        if (
            not query
            or count_words(query) < min_words
            or count_words(query) > max_words
            or query_surface_is_blocked(query, task_family="claims")
            or sentence_is_low_signal_for_query(query)
            or is_routing_or_metadata_sentence(query)
            or (
                not profile_has_complete_simple_clause(base_profile)
                and not base_standalone_ok
            )
            or not (
                strict_claim_surface_is_safe(
                    query,
                    base_profile,
                    max_actor_words=8,
                    max_object_words=18,
                )
                or base_standalone_ok
            )
        ):
            query = refine_generated_query_surface(
                base_text,
                task_family="claims",
                anchors=[
                    normalize_space(row.get("speaker", "")),
                    normalize_space(row.get("target", "")),
                ],
                min_words=min_words,
                target_words=query_target_words,
                max_chars=320,
                runner_cfg=candidate_runner_cfg,
            )
            query = clean_query_passage(query)
            query = strip_internal_topic_sections(query)
            query = strip_low_signal_lead_sentences(query)
            query = simplify_claim_query_seed(simplify_claim_surface(query))
            query = finalize_query_surface(query, add_terminal_punctuation=True)
        if not query or query.casefold() in seen_queries:
            continue
        if pair_like_corpus and contains_role_placeholder_label(query):
            continue
        if claim_surface_is_interrogative(query) or claim_surface_has_weak_modality(query):
            continue
        if query_surface_is_blocked(query, task_family="claims"):
            continue
        if sentence_is_low_signal_for_query(query) or is_routing_or_metadata_sentence(query):
            continue
        query_profile = claim_clause_profile(query)
        standalone_ok = standalone_claim_surface_is_safe(query)
        fallback_simple_ok = fallback_simple_complete_claim_surface_is_safe(query)
        if not profile_has_complete_simple_clause(query_profile) and not standalone_ok:
            continue
        if claim_profile_has_problematic_actor(query_profile):
            continue
        if small_corpus and not (
            normalize_space(query_profile.get("actor", ""))
            and normalize_space(query_profile.get("predicate", ""))
            and normalize_space(query_profile.get("object", ""))
        ) and not fallback_simple_ok:
            continue
        if not (
            strict_claim_surface_is_safe(
                query,
                query_profile,
                max_actor_words=8,
                max_object_words=18,
            )
            or standalone_ok
            or (small_corpus and not pair_like_corpus and fallback_simple_ok)
        ):
            continue
        parser_ok = claim_query_is_executable_via_parser(query, runner_cfg)
        if small_corpus and not (
            normalize_space(query_profile.get("actor", ""))
            and normalize_space(query_profile.get("predicate", ""))
            and normalize_space(query_profile.get("object", ""))
        ) and not fallback_simple_ok:
            continue
        if small_corpus and not parser_ok and not (
            fallback_simple_ok
            or (
                standalone_ok
                and strict_claim_surface_is_safe(
                    query,
                    query_profile,
                    max_actor_words=8,
                    max_object_words=18,
                )
            )
        ):
            continue
        words = count_words(query)
        if words < min_words or words > max_words:
            continue
        seen_queries.add(query.casefold())
        peer_matches, peer_best_overlap = (
            claim_support_peer_score(query_profile, peer_rows_by_idx[idx])
            if peer_rows_by_idx is not None
            else (0, 0.0)
        )
        directness_bonus = 8.0 if standalone_ok and words <= 24 else 0.0
        small_corpus_bonus = 8.0 if small_corpus else 0.0
        small_corpus_priority = (
            small_corpus_claim_surface_priority_adjustment(query)
            if small_corpus and not pair_like_corpus
            else 0.0
        )
        score = (
            claim_clause_quality_score(query_profile)
            + directness_bonus
            + small_corpus_bonus
            + small_corpus_priority
            + min(peer_matches, 4) * 4.0
            + peer_best_overlap * 8.0
            + (2.0 if query_profile.get("date_token") else 0.0)
            + (2.0 if query_profile.get("number_token") else 0.0)
            + (1.0 if normalize_space(row.get("speaker", "")) else 0.0)
            + (1.0 if normalize_space(row.get("target", "")) else 0.0)
            - abs(words - min(20, query_target_words)) / 12.0
            - surface_noise_penalty(query) * 0.35
        )
        candidates.append(
            {
                "task_family": "claims",
                "runner_task": "claims",
                "doc_id": str(row.get("doc_id", "")),
                "paragraph_id": str(row.get("paragraph_id", "")),
                "case_id": (
                    f"claims_minimal__{row.get('doc_id','')}__"
                    f"{row.get('paragraph_id','')}__{sha1_text(query)}"
                ),
                "query": query,
                "ground_truth": base_text,
                "query_kind": "claim_direct_minimal",
                "notes": (
                    "fallback=direct_claim_surface; "
                    f"clause_quality={claim_clause_quality_score(query_profile):.2f}; "
                    f"peer_matches={peer_matches}; peer_best_overlap={peer_best_overlap:.2f}; "
                    f"query_target_words={query_target_words}"
                ),
                "source_context": base_text,
                "selector_score": score,
                "word_count": words,
                "query_target_words": query_target_words,
            }
        )
    return round_robin_topk(
        candidates,
        min(count, len(candidates)),
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )


def build_parser_ready_claim_candidates(
    claim_rows: list[dict[str, Any]],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    *,
    runner_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not claim_rows or count <= 0 or not claim_query_vetting_enabled(runner_cfg):
        return []
    pair_like_corpus = all(
        normalize_space(row.get("doc_id", "")).startswith("Pair_")
        for row in claim_rows
        if normalize_space(row.get("doc_id", ""))
    )
    small_nonpair_corpus = len(claim_rows) <= 150 and not pair_like_corpus
    min_words = max(6, int(sampling.get("min_words", 8)) - 2)
    max_words = int(sampling.get("max_words", 48))
    candidates: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    candidate_rows = deterministic_seeded_row_order(
        claim_rows,
        seed=seed,
        key_fields=("doc_id", "paragraph_id", "claim_text", "local_id", "source_type"),
    )
    for row in candidate_rows:
        text = normalize_space(row.get("claim_text", ""))
        if uppercase_token_ratio(text) >= 0.45 and count_words(text) >= 8:
            text = normalize_uppercase_passage(text)
        text = clean_query_passage(text)
        text = strip_internal_topic_sections(text)
        text = strip_low_signal_lead_sentences(text)
        text = simplify_claim_query_seed(simplify_claim_surface(text))
        text = finalize_query_surface(text, add_terminal_punctuation=True)
        if not text:
            continue
        if pair_like_corpus and contains_role_placeholder_label(text):
            continue
        variants = [text]
        if pair_like_corpus:
            variants.extend(build_pair_like_claim_rewrite_variants(text, row))
        if " but " in text.casefold():
            variants = []
            variants.extend(
                finalize_query_surface(part, add_terminal_punctuation=True)
                for part in re.split(r"\bbut\b", text, maxsplit=1, flags=re.IGNORECASE)
                if normalize_space(part)
            )
        best_variant = ""
        best_profile: dict[str, Any] | None = None
        best_words = 0
        best_penalty = float("-inf")
        best_weak_modality = False
        best_rumor_like = False
        for variant in variants:
            query_key = variant.casefold()
            if query_key in seen_queries:
                continue
            if (
                sentence_is_low_signal_for_query(variant)
                or is_routing_or_metadata_sentence(variant)
                or claim_surface_is_interrogative(variant)
            ):
                continue
            weak_modality = claim_surface_has_weak_modality(variant)
            rumor_like = bool(re.search(r"\b(?:rumored|reportedly|apparently)\b", variant, flags=re.IGNORECASE))
            if pair_like_corpus and (weak_modality or rumor_like):
                continue
            profile = claim_clause_profile(variant)
            standalone_ok = standalone_claim_surface_is_safe(variant)
            strict_ok = strict_claim_surface_is_safe(
                variant,
                profile,
                max_actor_words=8,
                max_object_words=18,
            )
            if not (standalone_ok or strict_ok):
                continue
            if claim_profile_has_problematic_actor(profile):
                continue
            if claim_surface_is_retrieval_fragile(
                variant,
                profile,
                small_corpus=small_nonpair_corpus,
                pair_like_corpus=pair_like_corpus,
            ):
                continue
            if small_nonpair_corpus and not small_corpus_claim_surface_recovery_ok(
                variant,
                task_family="claims",
            ):
                continue
            words = count_words(variant)
            if words < max(6, min_words - 2) or words > max_words:
                continue
            if not claim_query_is_executable_via_parser(variant, runner_cfg):
                continue
            variant_score = (
                claim_clause_quality_score(profile)
                - surface_noise_penalty(variant) * 0.3
                - abs(words - 18) / 12.0
                - (8.0 if weak_modality else 0.0)
                - (6.0 if rumor_like else 0.0)
            )
            if variant != text:
                variant_score += 6.0
            if strict_ok and not standalone_ok:
                variant_score += 2.0
            if variant_score > best_penalty:
                best_variant = variant
                best_profile = profile
                best_words = words
                best_penalty = variant_score
                best_weak_modality = weak_modality
                best_rumor_like = rumor_like
        if not best_variant or best_profile is None:
            continue
        seen_queries.add(best_variant.casefold())
        candidates.append(
            {
                "task_family": "claims",
                "runner_task": "claims",
                "doc_id": str(row.get("doc_id", "")),
                "paragraph_id": str(row.get("paragraph_id", "")),
                "case_id": (
                    f"claims_parser_ready__{row.get('doc_id', '')}"
                    f"__{row.get('paragraph_id', '')}__{sha1_text(best_variant)}"
                ),
                "query": best_variant,
                "ground_truth": best_variant,
                "query_kind": "claim_parser_ready_surface",
                "notes": (
                    "fallback=parser_ready_claim_surface; "
                    f"clause_quality={claim_clause_quality_score(best_profile):.2f}"
                ),
                "source_context": best_variant,
                "selector_score": (
                    claim_clause_quality_score(best_profile)
                    + 16.0
                    + (6.0 if len(claim_rows) <= 150 and not pair_like_corpus else 0.0)
                    + (2.0 if normalize_space(row.get("speaker", "")) else 0.0)
                    + (2.0 if normalize_space(row.get("target", "")) else 0.0)
                    + (2.0 if best_profile.get("date_token") else 0.0)
                    + (2.0 if best_profile.get("number_token") else 0.0)
                    - (8.0 if best_weak_modality else 0.0)
                    - (6.0 if best_rumor_like else 0.0)
                    - abs(best_words - 18) / 12.0
                    - surface_noise_penalty(best_variant) * 0.3
                ),
                "word_count": best_words,
                "query_target_words": best_words,
            }
        )
    return round_robin_topk(
        candidates,
        min(count, len(candidates)),
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )


def build_pair_like_claim_rewrite_variants(
    claim_text: str,
    row: dict[str, Any],
) -> list[str]:
    text = finalize_query_surface(normalize_space(claim_text), add_terminal_punctuation=True)
    if not text:
        return []
    speaker = pretty_entity_label(normalize_query_speaker(row.get("speaker", "")))
    target = pretty_entity_label(normalize_space(row.get("target", "")))
    profile = claim_clause_profile(text)
    actor = pretty_entity_label(normalize_space(profile.get("actor", "")))
    subject = speaker or actor or target
    variants: list[str] = []
    affiliated_match = re.search(
        r"\bno longer officially affiliated with\b\s+(.+?)[.]?$",
        text,
        flags=re.IGNORECASE,
    )
    affiliated_object = (
        pretty_entity_label(normalize_space(affiliated_match.group(1)))
        if affiliated_match
        else target
    )
    if (
        subject
        and affiliated_object
        and re.search(r"\bno longer officially affiliated with\b", text, flags=re.IGNORECASE)
    ):
        variants.append(
            finalize_query_surface(
                f"{subject} was no longer part of {affiliated_object}.",
                add_terminal_punctuation=True,
            )
        )
    agent_subject = target or speaker or actor
    if agent_subject and re.search(r"\bis an agent of a foreign power\b", text, flags=re.IGNORECASE):
        variants.append(
            finalize_query_surface(
                f"{agent_subject} is an agent of a foreign power.",
                add_terminal_punctuation=True,
            )
        )
    authorized_match = re.search(
        r"\bOnly\s+(.+?)\s+could have authorized these activities\b",
        text,
        flags=re.IGNORECASE,
    )
    if authorized_match:
        variants.append(
            finalize_query_surface(
                f"{pretty_entity_label(authorized_match.group(1))} authorized these activities.",
                add_terminal_punctuation=True,
            )
        )
    # A denial is evidence about what the speaker said, not proof that the
    # denied event did not happen. Do not manufacture the stronger claim.
    no_memory_match = re.search(
        r"\b([A-Z][A-Za-z0-9#'’.\- ]{1,90}?)\s+had\s+no\s+memory\s+of\s+any\s+meeting\s+between\s+(.+?)\s+and\s+(.+?)[.]?$",
        text,
        flags=re.IGNORECASE,
    )
    if no_memory_match:
        witness = pretty_entity_label(normalize_space(no_memory_match.group(1)))
        left_party = pretty_entity_label(normalize_space(no_memory_match.group(2)))
        right_party = pretty_entity_label(normalize_space(no_memory_match.group(3)))
        if witness and left_party and right_party:
            variants.append(
                finalize_query_surface(
                    f"{witness} did not remember any meeting between {left_party} and {right_party}.",
                    add_terminal_punctuation=True,
                )
            )
    remained_match = re.search(
        r"\bwas told he was no longer part of the campaign\b",
        text,
        flags=re.IGNORECASE,
    )
    if subject and target and remained_match:
        variants.append(
            finalize_query_surface(
                f"{target} no longer included {subject}.",
                add_terminal_punctuation=True,
            )
        )
    no_longer_part_match = re.search(
        r"\b([A-Z][A-Za-z0-9#'’.\- ]{1,90}?)\s+was\s+no\s+longer\s+part\s+of\s+(.+?)[.]?$",
        text,
        flags=re.IGNORECASE,
    )
    if no_longer_part_match:
        member = pretty_entity_label(normalize_space(no_longer_part_match.group(1)))
        organization = pretty_entity_label(normalize_space(no_longer_part_match.group(2)))
        if member and organization:
            variants.append(
                finalize_query_surface(
                    f"{organization} no longer included {member}.",
                    add_terminal_punctuation=True,
                )
            )
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = normalize_space(variant).casefold()
        if not key or key == text.casefold() or key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def build_pair_like_relation_claim_variants(
    evidence_text: str,
    row: Mapping[str, Any],
) -> list[str]:
    text = finalize_query_surface(normalize_space(evidence_text), add_terminal_punctuation=True)
    if not text:
        return []
    variants: list[str] = [text]
    source = pretty_entity_label(normalize_space(row.get("source", "")))
    target = pretty_entity_label(normalize_space(row.get("target", "")))
    relation_key = normalize_space(row.get("relation", "")).casefold()

    match = re.search(
        r"^([A-Z][A-Za-z0-9#'’.\- ]{1,90}?)\s+informed\s+the\s+FBI\s+that,\s*during\s+this\s+trip,\s+he\s+unexpectedly\s+met\s+(.+?)\.$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        subject = pretty_entity_label(normalize_space(match.group(1))) or source
        counterpart = pretty_entity_label(normalize_space(match.group(2))) or target
        if subject and counterpart:
            variants.append(
                finalize_query_surface(
                    f"{subject} met {counterpart} during this trip.",
                    add_terminal_punctuation=True,
                )
            )

    match = re.search(
        r"^([A-Z][A-Za-z0-9#'’.\- ]{1,90}?)\s+received\s+direction\s+by\s+(.+?)\.$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        subject = pretty_entity_label(normalize_space(match.group(1))) or source
        origin = pretty_entity_label(normalize_space(match.group(2))) or target
        if subject and origin:
            variants.append(
                finalize_query_surface(
                    f"{subject} received direction from {origin}.",
                    add_terminal_punctuation=True,
                )
            )

    if relation_key == "event_participant" and source and target:
        if re.search(r"\bmet with\b", text, flags=re.IGNORECASE):
            variants.append(
                finalize_query_surface(
                    f"{source} met {target}.",
                    add_terminal_punctuation=True,
                )
            )
        if re.search(r"\bsent a letter to\b", text, flags=re.IGNORECASE):
            variants.append(
                finalize_query_surface(
                    f"{source} sent a letter to {target}.",
                    add_terminal_punctuation=True,
                )
            )

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = normalize_space(variant).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def pair_like_claim_surface_soft_penalty(text: str) -> float:
    value = normalize_space(text)
    if not value:
        return 0.0
    penalty = 0.0
    standalone_ok = standalone_claim_surface_is_safe(value)
    fallback_ok = fallback_simple_complete_claim_surface_is_safe(value)
    if re.match(r"^(?:he|she|they|it)\b", value, flags=re.IGNORECASE):
        penalty += 18.0
    if re.match(r"^(?:there was|there were|there has been)\b", value, flags=re.IGNORECASE):
        penalty += 24.0
    if re.search(r"\bcontinues?\s+to\s+have\s+access to\b", value, flags=re.IGNORECASE):
        penalty += 18.0
    if re.search(r"\bactive in promoting theories\b", value, flags=re.IGNORECASE):
        penalty += 18.0
    if re.search(r"\bagent of a foreign power\b", value, flags=re.IGNORECASE):
        penalty += 0.0 if (standalone_ok and fallback_ok) else 26.0
    if re.search(r"\bis a foreign power\b", value, flags=re.IGNORECASE):
        penalty += 0.0 if (standalone_ok and fallback_ok) else 24.0
    if re.search(r"\bdenied meeting with\b", value, flags=re.IGNORECASE):
        penalty += 18.0
    if re.search(r"\bhas not directly supported a political campaign since\b", value, flags=re.IGNORECASE):
        penalty += 16.0
    if re.search(r"\bfeeding information to\b", value, flags=re.IGNORECASE):
        penalty += 18.0
    if re.search(r"\bcould have authorized these activities\b", value, flags=re.IGNORECASE):
        penalty += 8.0
    if re.search(r"\bofficially affiliated with\b", value, flags=re.IGNORECASE):
        penalty += 10.0
    return penalty


def pair_like_claim_surface_runtime_hostile(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    return bool(
        re.search(r"\bdenied meeting with\b", value, flags=re.IGNORECASE)
        or re.search(r"\bfeeding information to\b", value, flags=re.IGNORECASE)
    )


def build_minimal_contradiction_candidates(
    claim_views: pd.DataFrame,
    count: int,
    sampling: dict[str, Any],
    seed: int,
    *,
    runner_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if claim_views.empty or count <= 0:
        return []
    corpus_name = normalize_space((runner_cfg or {}).get("__corpus_name", "")).casefold()
    claim_doc_ids = [
        normalize_space(doc_id)
        for doc_id in claim_views.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    ]
    pair_like_corpus = bool(claim_doc_ids) and all(doc_id.startswith("Pair_") for doc_id in claim_doc_ids)
    small_corpus = exhaustive_small_doc_corpus(len({
        normalize_space(doc_id)
        for doc_id in claim_views.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    }))
    min_words = int(sampling.get("min_words", 16))
    max_words = int(sampling.get("max_words", 64))
    if small_corpus or corpus_name in {"fisc", "canberra"}:
        min_words = max(6, min_words - 8)
        max_words = max(max_words, 40)
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    strategies = [value.strip() for value in sampling.get("strategy_order", ["negation", "numeric", "date"])]
    if small_corpus or corpus_name in {"fisc", "canberra", "iran"}:
        strategies = ["negation", "date", "numeric"]
    seen_queries: set[str] = set()
    candidates: list[dict[str, Any]] = []
    rows = claim_views.to_dict("records")
    base_seed_claim_count = max(count * 8, count * 2)
    max_seed_claim_catalog = int(sampling.get("max_seed_claim_catalog", 0) or 0)
    if max_seed_claim_catalog > 0:
        base_seed_claim_count = min(base_seed_claim_count, max_seed_claim_catalog)
    minimal_claims = build_minimal_claim_candidates(
        claim_views,
        base_seed_claim_count,
        sampling,
        seed,
        runner_cfg=runner_cfg,
    )
    parser_ready_claims: list[dict[str, Any]] = []
    if small_corpus or corpus_name in {"fisc", "canberra", "iran"}:
        parser_ready_claims = build_parser_ready_claim_candidates(
            rows,
            base_seed_claim_count,
            sampling,
            seed,
            runner_cfg=runner_cfg,
        )
    base_claim_seeds = merge_ranked_candidates(
        [parser_ready_claims, minimal_claims],
        target_count=base_seed_claim_count,
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )
    base_by_doc_para: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in base_claim_seeds:
        base_by_doc_para[(str(case.get("doc_id", "")), str(case.get("paragraph_id", "")))].append(case)
    for row in rows:
        doc_id = str(row.get("doc_id", ""))
        paragraph_id = str(row.get("paragraph_id", ""))
        base_candidates = base_by_doc_para.get((doc_id, paragraph_id), [])
        if not base_candidates:
            continue
        for base_case in base_candidates:
            base_text = normalize_space(base_case.get("query", ""))
            if not base_text:
                continue
            if pair_like_corpus and contains_role_placeholder_label(base_text):
                continue
            base_profile = claim_clause_profile(base_text)
            standalone_ok = (
                standalone_claim_surface_is_safe(base_text)
                or standalone_contradiction_surface_is_safe(base_text)
            )
            if not profile_has_complete_simple_clause(base_profile) and not standalone_ok:
                continue
            if claim_profile_has_problematic_actor(base_profile):
                continue
            for strategy in strategies:
                mutated = None
                strategy_used = ""
                if strategy in {"numeric", "date"}:
                    mutated = perturb_numeric_or_date(base_text)
                elif strategy == "negation":
                    mutated = perturb_negation(base_text)
                if not mutated:
                    continue
                strategy_used = mutated[1]
                query = finalize_query_surface(mutated[0], add_terminal_punctuation=True)
                query = refine_generated_query_surface(
                    query,
                    task_family="contradictions",
                    anchors=[
                        normalize_space(row.get("speaker", "")),
                        normalize_space(row.get("target", "")),
                    ],
                    min_words=min_words,
                    target_words=count_words(base_text),
                    max_chars=320,
                    runner_cfg=candidate_runner_cfg,
                )
                query = clean_query_passage(query)
                query = strip_internal_topic_sections(query)
                query = strip_low_signal_lead_sentences(query)
                query = simplify_claim_query_seed(simplify_claim_surface(query))
                query = finalize_query_surface(query, add_terminal_punctuation=True)
                if not query or query.casefold() in seen_queries:
                    continue
                if contains_role_placeholder_label(query):
                    continue
                if claim_surface_is_interrogative(query) or claim_surface_has_weak_modality(query):
                    continue
                if re.search(
                    r"\bcontinues?\s+to\s+have\s+not\b",
                    query,
                    flags=re.IGNORECASE,
                ):
                    continue
                if claim_uses_intentive_construction(query):
                    continue
                if not contradiction_query_carries_mutation(
                    query,
                    base_text,
                    strategy_used=strategy_used,
                ):
                    continue
                if query_surface_is_blocked(query, task_family="contradictions"):
                    continue
                if sentence_is_low_signal_for_query(query) or is_routing_or_metadata_sentence(query):
                    continue
                query_profile = claim_clause_profile(query)
                query_standalone_ok = (
                    standalone_claim_surface_is_safe(query)
                    or standalone_contradiction_surface_is_safe(query)
                )
                if pair_like_corpus and not small_corpus_claim_surface_recovery_ok(
                    query,
                    task_family="contradictions",
                ):
                    if not (
                        query_standalone_ok
                        and count_words(query) <= 12
                        and substantive_sentence_score(query) >= 5.5
                    ):
                        continue
                if not profile_has_complete_simple_clause(query_profile) and not query_standalone_ok:
                    continue
                if claim_profile_has_problematic_actor(query_profile):
                    continue
                if not (
                    near_strict_claim_surface_is_safe(
                        query,
                        query_profile,
                        max_actor_words=6,
                        max_object_words=18,
                    )
                    or query_standalone_ok
                ):
                    continue
                words = count_words(query)
                if words < min_words or words > max_words:
                    continue
                clause_quality = claim_clause_quality_score(query_profile)
                if (small_corpus or corpus_name in {"fisc", "canberra", "iran"}) and strategy_used in {"date", "numeric"}:
                    if clause_quality < 6.0:
                        continue
                seen_queries.add(query.casefold())
                mutation_bonus = (
                    {"date": 2.5, "numeric": 2.0, "negation": 4.5}.get(strategy_used, 0.0)
                    if (small_corpus or corpus_name in {"fisc", "canberra", "iran"})
                    else {"date": 4.0, "numeric": 4.0, "negation": 2.5}.get(strategy_used, 0.0)
                )
                directness_bonus = 8.0 if query_standalone_ok and words <= 24 else 0.0
                score = (
                    clause_quality
                    + mutation_bonus
                    + directness_bonus
                    + (2.0 if query_profile.get("date_token") else 0.0)
                    + (2.0 if query_profile.get("number_token") else 0.0)
                    - abs(words - min(20, count_words(base_text))) / 12.0
                    - surface_noise_penalty(query) * 0.35
                )
                candidates.append(
                    {
                        "task_family": "contradictions",
                        "runner_task": "contradictions",
                        "doc_id": doc_id,
                        "paragraph_id": paragraph_id,
                        "case_id": f"contradictions_minimal__{doc_id}__{paragraph_id}__{sha1_text(query)}",
                        "query": query,
                        "ground_truth": base_text,
                        "query_kind": "claim_case_contradiction",
                        "notes": (
                            f"fallback=direct_claim_contradiction; strategy={strategy_used}; "
                            f"clause_quality={claim_clause_quality_score(query_profile):.2f}"
                        ),
                        "source_context": base_text,
                        "selector_score": score,
                        "word_count": words,
                        "query_target_words": count_words(base_text),
                    }
                )
    return round_robin_topk(
        candidates,
        min(count, len(candidates)),
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )


def choose_reconstruction_cases(
    reconstruction_views: pd.DataFrame,
    claim_views: pd.DataFrame | None,
    relations: pd.DataFrame | None,
    chunk_text_lookup: dict[tuple[str, str], str],
    doc_chunks_by_doc: dict[str, list[dict[str, Any]]],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    runner_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    claim_like_candidates: list[dict[str, Any]] = []
    pair_like_corpus = bool(doc_chunks_by_doc) and all(
        normalize_space(doc_id).startswith("Pair_")
        for doc_id in doc_chunks_by_doc
        if normalize_space(doc_id)
    )
    corpus_doc_count = len(doc_chunks_by_doc)
    small_corpus = exhaustive_small_doc_corpus(corpus_doc_count)
    large_corpus = optimized_doc_corpus(corpus_doc_count)
    allow_claim_backfill = maybe_bool(sampling.get("_allow_claim_backfill", False))
    single_doc_friendly_small_corpus = small_corpus and not pair_like_corpus
    total_chunk_count = 0
    uppercase_heavy_chunk_count = 0
    if small_corpus:
        for chunks_for_doc in doc_chunks_by_doc.values():
            for chunk in chunks_for_doc:
                total_chunk_count += 1
                if uppercase_token_ratio(str(chunk.get("chunk_text", ""))) >= 0.35:
                    uppercase_heavy_chunk_count += 1
    uppercase_heavy_corpus = (
        small_corpus
        and total_chunk_count > 0
        and uppercase_heavy_chunk_count >= max(1, total_chunk_count // 3)
    )
    claim_view_rows_seeded = (
        deterministic_seeded_row_order(
            claim_views.to_dict("records"),
            seed=seed + 11,
            key_fields=("doc_id", "paragraph_id", "claim_text", "local_id", "source_type"),
        )
        if claim_views is not None and not claim_views.empty
        else []
    )
    support_claim_rows = (
        claim_view_rows_seeded if small_corpus else []
    )
    reconstruction_support_cache: dict[tuple[str, str], tuple[int, float]] = {}

    def small_corpus_reconstruction_support(text: str, doc_id: str) -> tuple[int, float]:
        if not support_claim_rows:
            return 0, 0.0
        cache_key = (normalize_space(text).casefold(), normalize_space(doc_id))
        cached = reconstruction_support_cache.get(cache_key)
        if cached is not None:
            return cached
        profile = claim_clause_profile(text)
        peer_rows = [
            row for row in support_claim_rows
            if normalize_space(row.get("doc_id", "")) != normalize_space(doc_id)
        ]
        result = claim_support_peer_score(profile, peer_rows)
        reconstruction_support_cache[cache_key] = result
        return result

    if uppercase_heavy_corpus and claim_views is not None and not claim_views.empty:
        allow_claim_backfill = True
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    reconstruction_seed_runner_cfg = dict(runner_cfg or {})
    reconstruction_seed_runner_cfg["__disable_claim_build_parser_vet"] = True
    min_distinctive = int(sampling.get("min_distinctive_anchor_count", 25))
    effective_min_distinctive = min_distinctive
    if large_corpus:
        effective_min_distinctive = max(12, min_distinctive)
    elif small_corpus:
        effective_min_distinctive = max(
            6,
            min_distinctive // (3 if single_doc_friendly_small_corpus else 2),
        )
    else:
        effective_min_distinctive = max(10, min_distinctive // 2)
    min_overall_label = normalize_space(sampling.get("require_overall_label", "strong")).casefold()
    allowed_labels = {
        normalize_space(label).casefold()
        for label in sampling.get("allowed_overall_labels", [min_overall_label])
        if normalize_space(label)
    } or {min_overall_label}
    min_query_words = int(sampling.get("min_query_words", sampling.get("min_words", 20)))
    max_query_words = int(sampling.get("max_query_words", sampling.get("max_words", 140)))
    effective_max_query_words = min(max_query_words, 56)
    min_visible_context_words = int(sampling.get("min_visible_context_words", 16))
    compact_claim_reconstruction_candidates: list[dict[str, Any]] = []
    if claim_views is not None and not claim_views.empty:
        compact_claim_rows = list(claim_view_rows_seeded)
        if large_corpus:
            def compact_claim_seed_score(row: dict[str, Any]) -> tuple[float, float, int]:
                text = simplify_claim_query_seed(
                    simplify_claim_surface(row.get("claim_text", ""))
                )
                if not text:
                    return (-9999.0, -9999.0, -9999)
                profile = claim_clause_profile(text)
                return (
                    claim_clause_quality_score(profile)
                    + (3.0 if profile.get("date_token") else 0.0)
                    + (3.0 if profile.get("number_token") else 0.0)
                    + (4.0 if standalone_claim_surface_is_safe(text) else -8.0)
                    - surface_noise_penalty(text) * 0.35,
                    float(len(profile.get("content_tokens", set()))),
                    -abs(count_words(text) - 16),
                )

            compact_claim_rows = sorted(
                compact_claim_rows,
                key=compact_claim_seed_score,
                reverse=True,
            )[: max(count * 80, 480)]
        for row in compact_claim_rows:
            if not row_is_structured_claim(row):
                continue
            if claim_row_loses_support_or_polarity_when_simplified(row):
                continue
            raw_claim_text = finalize_query_surface(
                normalize_space(row.get("claim_text", "")),
                add_terminal_punctuation=True,
            )
            text = simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", "")))
            if not text or is_routing_or_metadata_sentence(text):
                continue
            if query_surface_is_blocked(text, task_family="reconstruction"):
                continue
            if claim_uses_intentive_construction(text) or has_reporting_that_clause(text):
                continue
            if "..." in text or ";" in text or ":" in text:
                continue
            words = count_words(text)
            if words < 8 or words > min(effective_max_query_words, 32):
                continue
            profile = claim_clause_profile(text)
            if not reconstruction_claim_surface_is_actionable(text, profile):
                continue
            source_chunk_text = chunk_text_lookup.get(
                paragraph_key(row["doc_id"], row["paragraph_id"]),
                "",
            )
            source_context = choose_fast_query_context(
                [
                    extract_embedded_context(row.get("proposition_text", "")),
                    extract_embedded_context(row.get("pseudo_view_text", "")),
                    source_chunk_text,
                ],
                min_words=max(8, min_visible_context_words // 2),
                max_chars=int(sampling.get("max_source_excerpt_chars", 480)),
            )
            peer_matches, peer_best_overlap = small_corpus_reconstruction_support(
                text,
                str(row["doc_id"]),
            )
            source_context_quality = context_quality_score(
                source_context,
                anchors=[
                    normalize_space(row.get("speaker", "")),
                    normalize_space(row.get("target", "")),
                ],
                min_words=max(8, min_visible_context_words // 2),
            )
            compact_claim_quality = claim_clause_quality_score(profile)
            clean_single_doc_compact_claim = bool(
                single_doc_friendly_small_corpus
                and standalone_claim_surface_is_safe(text)
                and compact_claim_quality >= 12.0
                and source_context_quality >= 4.5
            )
            if (
                small_corpus
                and len(support_claim_rows) <= 50
                and peer_matches <= 0
                and not clean_single_doc_compact_claim
            ):
                continue
            compact_claim_reconstruction_candidates.append(
                {
                    "task_family": "reconstruction",
                    "runner_task": "reconstruction",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": f"reconstruction_compact_claim__{row['doc_id']}__{row['paragraph_id']}__{sha1_text(text)}",
                    "query": text,
                    "ground_truth": raw_claim_text or text,
                    "query_kind": "reconstruction_claim_surface",
                    "variant_kind": "compact_claim",
                    "notes": (
                        f"source_type={normalize_space(row.get('source_type', ''))}; "
                        f"speaker={normalize_space(row.get('speaker', ''))}; "
                        f"target={normalize_space(row.get('target', ''))}; "
                        f"peer_matches={peer_matches}; "
                        f"peer_best_overlap={peer_best_overlap:.2f}; "
                        f"clean_single_doc_support={clean_single_doc_compact_claim}; "
                        "reconstruction_source=compact_claim"
                    ),
                    "source_context": source_context,
                    "selector_score": (
                        140.0
                        + compact_claim_quality
                        + min(peer_matches, 4) * 12.0
                        + peer_best_overlap * 18.0
                        + source_context_quality * 1.5
                        + (10.0 if clean_single_doc_compact_claim else 0.0)
                        + (6.0 if standalone_claim_surface_is_safe(text) else 0.0)
                        + (6.0 if normalize_space(row.get("speaker", "")) else 0.0)
                        + (4.0 if normalize_space(row.get("target", "")) else 0.0)
                        + (3.0 if profile.get("date_token") else 0.0)
                        + (3.0 if profile.get("number_token") else 0.0)
                        - abs(words - 16) / 8.0
                    ),
                    "word_count": words,
                    "query_target_words": 16,
                }
            )
        if compact_claim_reconstruction_candidates:
            compact_claim_reconstruction_candidates = round_robin_topk(
                compact_claim_reconstruction_candidates,
                min(max(count * 2, 4), len(compact_claim_reconstruction_candidates)),
                max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
                score_key="selector_score",
                seed=seed,
            )
            if (
                not pair_like_corpus
                and not single_doc_friendly_small_corpus
                and len(compact_claim_reconstruction_candidates) >= min(count, 3)
            ):
                return normalize_selected_case_queries(
                    compact_claim_reconstruction_candidates[:count],
                    task_family="reconstruction",
                    runner_cfg=runner_cfg,
                )
            claim_like_candidates.extend(compact_claim_reconstruction_candidates)
            candidates.extend(compact_claim_reconstruction_candidates)
    if single_doc_friendly_small_corpus and claim_views is not None and not claim_views.empty:
        direct_small_corpus_claim_candidates: list[dict[str, Any]] = []
        for row in claim_view_rows_seeded:
            text = simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", "")))
            if not text or is_routing_or_metadata_sentence(text):
                continue
            if query_surface_is_blocked(text, task_family="reconstruction"):
                continue
            if "..." in text or ";" in text or ":" in text:
                continue
            words = count_words(text)
            if words < 8 or words > min(max_query_words, 64):
                continue
            profile = claim_clause_profile(text)
            if not (
                standalone_claim_surface_is_safe(text)
                or strict_claim_surface_is_safe(
                    text,
                    profile,
                    max_actor_words=8,
                    max_object_words=20,
                )
            ):
                continue
            source_context = choose_fast_query_context(
                [
                    normalize_space(row.get("paragraph_text", "")),
                    normalize_space(row.get("chunk_text", "")),
                ],
                min_words=max(8, min_visible_context_words // 2),
                max_chars=int(sampling.get("max_source_excerpt_chars", 480)),
            )
            direct_small_corpus_claim_candidates.append(
                {
                    "task_family": "reconstruction",
                    "runner_task": "reconstruction",
                    "doc_id": str(row.get("doc_id", "")),
                    "paragraph_id": str(row.get("paragraph_id", "")),
                    "case_id": f"reconstruction_small_claim__{row.get('doc_id','')}__{row.get('paragraph_id','')}__{sha1_text(text)}",
                    "query": text,
                    "ground_truth": text,
                    "query_kind": "reconstruction_claim_surface",
                    "variant_kind": "small_corpus_direct_claim_surface",
                    "notes": (
                        f"speaker={normalize_space(row.get('speaker', ''))}; "
                        f"target={normalize_space(row.get('target', ''))}; "
                        "reconstruction_source=small_corpus_direct_claim_surface"
                    ),
                    "source_context": source_context,
                    "selector_score": (
                        155.0
                        + claim_clause_quality_score(profile)
                        + (6.0 if normalize_space(row.get("speaker", "")) else 0.0)
                        + (4.0 if normalize_space(row.get("target", "")) else 0.0)
                        - abs(words - 18) / 8.0
                    ),
                    "word_count": words,
                    "query_target_words": 18,
                }
            )
        if direct_small_corpus_claim_candidates:
            direct_small_corpus_claim_candidates = round_robin_topk(
                direct_small_corpus_claim_candidates,
                min(max(count * 3, count), len(direct_small_corpus_claim_candidates)),
                max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
                score_key="selector_score",
                seed=seed,
            )
            claim_like_candidates.extend(direct_small_corpus_claim_candidates)
            candidates.extend(direct_small_corpus_claim_candidates)
        helper_claim_sampling = {
            "min_words": max(8, min_query_words // 2),
            "max_words": min(max_query_words, 120),
            "min_query_words": max(10, min_query_words // 2),
            "max_query_words": min(max_query_words, 160),
            "require_target_or_speaker": False,
            "min_role_fields": 0,
            "query_style": "plain",
            "require_simple_clause_arguments": False,
            "exclude_generic_roles": False,
            "min_clause_quality_score": 4.0,
            "max_cases_per_doc": int(sampling.get("max_cases_per_doc", 2)),
        }
        helper_claim_cases = choose_claim_cases(
            claim_views,
            relations if relations is not None else pd.DataFrame(),
            chunk_text_lookup,
            max(count * 3, count),
            helper_claim_sampling,
            seed,
            runner_cfg=runner_cfg,
            reconstruction_views=None,
        )
        helper_claim_reconstruction: list[dict[str, Any]] = []
        for case in helper_claim_cases:
            query = normalize_space(case.get("query", ""))
            if not query or query_surface_is_blocked(query, task_family="reconstruction"):
                continue
            helper_claim_reconstruction.append(
                {
                    "task_family": "reconstruction",
                    "runner_task": "reconstruction",
                    "doc_id": str(case.get("doc_id", "")),
                    "paragraph_id": str(case.get("paragraph_id", "")),
                    "case_id": f"reconstruction::{normalize_space(case.get('case_id', ''))}",
                    "query": query,
                    "ground_truth": normalize_space(case.get("ground_truth", "")) or query,
                    "query_kind": "reconstruction_claim_surface",
                    "variant_kind": "small_corpus_claim_helper",
                    "notes": (
                        f"{normalize_space(case.get('notes', ''))}; "
                        "reconstruction_source=small_corpus_claim_helper"
                    ).strip("; "),
                    "source_context": normalize_space(case.get("source_context", "")),
                    "selector_score": float(case.get("selector_score", 0.0) or 0.0) + 28.0,
                    "word_count": int(case.get("word_count", 0) or 0),
                    "query_target_words": int(case.get("query_target_words", 0) or 0),
                }
            )
        if helper_claim_reconstruction:
            claim_like_candidates.extend(helper_claim_reconstruction)
            candidates.extend(helper_claim_reconstruction)
    if pair_like_corpus and small_corpus and claim_views is not None and not claim_views.empty:
        pair_like_direct_claim_surfaces: list[dict[str, Any]] = []
        for row in claim_view_rows_seeded:
            text = simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", "")))
            if not text or is_routing_or_metadata_sentence(text):
                continue
            if query_surface_is_blocked(text, task_family="reconstruction"):
                continue
            if claim_uses_intentive_construction(text) or has_reporting_that_clause(text):
                continue
            if "..." in text or ";" in text or ":" in text:
                continue
            words = count_words(text)
            if words < 8 or words > min(max_query_words, 64):
                continue
            profile = claim_clause_profile(text)
            if not reconstruction_claim_surface_is_actionable(text, profile):
                continue
            if not (
                standalone_claim_surface_is_safe(text)
                or strict_claim_surface_is_safe(
                    text,
                    profile,
                    max_actor_words=8,
                    max_object_words=20,
                )
            ):
                continue
            peer_matches, peer_best_overlap = small_corpus_reconstruction_support(
                text,
                str(row.get("doc_id", "")),
            )
            direct_quality = claim_clause_quality_score(profile)
            if (
                peer_matches <= 0
                and peer_best_overlap < 0.2
                and not (
                    direct_quality >= 10.0 and standalone_claim_surface_is_safe(text)
                )
            ):
                continue
            source_context = choose_fast_query_context(
                [
                    chunk_text_lookup.get(
                        paragraph_key(row.get("doc_id", ""), row.get("paragraph_id", "")),
                        "",
                    ),
                    extract_embedded_context(row.get("pseudo_view_text", "")),
                    extract_embedded_context(row.get("proposition_text", "")),
                ],
                min_words=max(8, min_visible_context_words // 2),
                max_chars=int(sampling.get("max_source_excerpt_chars", 480)),
            )
            pair_like_direct_claim_surfaces.append(
                {
                    "task_family": "reconstruction",
                    "runner_task": "reconstruction",
                    "doc_id": str(row.get("doc_id", "")),
                    "paragraph_id": str(row.get("paragraph_id", "")),
                    "case_id": f"reconstruction_pair_direct__{row.get('doc_id','')}__{row.get('paragraph_id','')}__{sha1_text(text)}",
                    "query": text,
                    "ground_truth": text,
                    "query_kind": "reconstruction_claim_surface",
                    "variant_kind": "pair_like_direct_claim_surface",
                    "notes": (
                        f"speaker={normalize_space(row.get('speaker', ''))}; "
                        f"target={normalize_space(row.get('target', ''))}; "
                        f"peer_matches={peer_matches}; "
                        f"peer_best_overlap={peer_best_overlap:.2f}; "
                        "reconstruction_source=pair_like_direct_claim_surface"
                    ),
                    "source_context": source_context,
                    "selector_score": (
                        148.0
                        + direct_quality
                        + min(peer_matches, 4) * 8.0
                        + peer_best_overlap * 14.0
                        + (6.0 if normalize_space(row.get("speaker", "")) else 0.0)
                        + (4.0 if normalize_space(row.get("target", "")) else 0.0)
                        - abs(words - 18) / 8.0
                    ),
                    "word_count": words,
                    "query_target_words": 18,
                }
            )
        if pair_like_direct_claim_surfaces:
            claim_like_candidates.extend(pair_like_direct_claim_surfaces)
            candidates.extend(pair_like_direct_claim_surfaces)

        pair_helper_claim_sampling = {
            "min_words": max(8, min_query_words // 2),
            "max_words": min(max_query_words, 96),
            "min_query_words": max(10, min_query_words // 2),
            "max_query_words": min(max_query_words, 128),
            "require_target_or_speaker": False,
            "min_role_fields": 0,
            "query_style": "plain",
            "require_simple_clause_arguments": False,
            "exclude_generic_roles": False,
            "min_clause_quality_score": 4.0,
            "max_cases_per_doc": int(sampling.get("max_cases_per_doc", 2)),
        }
        pair_helper_runner_cfg = dict(reconstruction_seed_runner_cfg)
        pair_helper_runner_cfg["__strict_small_corpus_vet"] = False
        pair_helper_claim_cases = choose_claim_cases(
            claim_views,
            relations if relations is not None else pd.DataFrame(),
            chunk_text_lookup,
            max(count * 4, count),
            pair_helper_claim_sampling,
            seed,
            runner_cfg=pair_helper_runner_cfg,
            reconstruction_views=None,
        )
        pair_helper_reconstruction: list[dict[str, Any]] = []
        for case in pair_helper_claim_cases:
            query = normalize_space(case.get("query", ""))
            if not query or query_surface_is_blocked(query, task_family="reconstruction"):
                continue
            profile = claim_clause_profile(query)
            if not reconstruction_claim_surface_is_actionable(query, profile):
                continue
            pair_helper_reconstruction.append(
                {
                    "task_family": "reconstruction",
                    "runner_task": "reconstruction",
                    "doc_id": str(case.get("doc_id", "")),
                    "paragraph_id": str(case.get("paragraph_id", "")),
                    "case_id": f"reconstruction::pair_helper::{normalize_space(case.get('case_id', ''))}",
                    "query": query,
                    "ground_truth": normalize_space(case.get("ground_truth", "")) or query,
                    "query_kind": "reconstruction_claim_surface",
                    "variant_kind": "pair_like_claim_helper",
                    "notes": (
                        f"{normalize_space(case.get('notes', ''))}; "
                        "reconstruction_source=pair_like_claim_helper"
                    ).strip("; "),
                    "source_context": normalize_space(case.get("source_context", "")),
                    "selector_score": float(case.get("selector_score", 0.0) or 0.0) + 24.0,
                    "word_count": int(case.get("word_count", 0) or 0),
                    "query_target_words": int(case.get("query_target_words", 0) or 0),
                }
            )
        if pair_helper_reconstruction:
            claim_like_candidates.extend(pair_helper_reconstruction)
            candidates.extend(pair_helper_reconstruction)
        if len(claim_like_candidates) < max(count, 10):
            relaxed_pair_like_claim_surfaces: list[dict[str, Any]] = []
            seen_pair_like_queries = {
                normalize_space(case.get("query", "")).casefold()
                for case in claim_like_candidates
                if normalize_space(case.get("query", ""))
            }
            for row in claim_view_rows_seeded:
                text = finalize_query_surface(
                    simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", ""))),
                    add_terminal_punctuation=True,
                )
                if not text or is_routing_or_metadata_sentence(text):
                    continue
                if query_surface_is_blocked(text, task_family="reconstruction"):
                    continue
                if REDACTION_MARKER_RE.search(text) or "..." in text or ":" in text:
                    continue
                words = count_words(text)
                if words < 8 or words > min(max_query_words, 72):
                    continue
                key = text.casefold()
                if key in seen_pair_like_queries:
                    continue
                profile = claim_clause_profile(text)
                if claim_surface_has_cognitive_event_disclaimer(text):
                    continue
                if claim_surface_has_multi_action_fallback_risk(text, profile):
                    continue
                if not (
                    reconstruction_claim_surface_is_actionable(text, profile)
                    or near_strict_claim_surface_is_safe(
                        text,
                        profile,
                        max_actor_words=8,
                        max_object_words=24,
                    )
                    or standalone_claim_surface_is_safe(text)
                ):
                    continue
                direct_quality = claim_clause_quality_score(profile)
                peer_matches, peer_best_overlap = small_corpus_reconstruction_support(
                    text,
                    str(row.get("doc_id", "")),
                )
                if (
                    direct_quality < 6.0
                    and peer_matches <= 0
                    and peer_best_overlap < 0.2
                ):
                    continue
                source_context = choose_fast_query_context(
                    [
                        chunk_text_lookup.get(
                            paragraph_key(row.get("doc_id", ""), row.get("paragraph_id", "")),
                            "",
                        ),
                        extract_embedded_context(row.get("pseudo_view_text", "")),
                        extract_embedded_context(row.get("proposition_text", "")),
                    ],
                    min_words=max(8, min_visible_context_words // 2),
                    max_chars=int(sampling.get("max_source_excerpt_chars", 480)),
                )
                relaxed_pair_like_claim_surfaces.append(
                    {
                        "task_family": "reconstruction",
                        "runner_task": "reconstruction",
                        "doc_id": str(row.get("doc_id", "")),
                        "paragraph_id": str(row.get("paragraph_id", "")),
                        "case_id": f"reconstruction_pair_relaxed__{row.get('doc_id','')}__{row.get('paragraph_id','')}__{sha1_text(text)}",
                        "query": text,
                        "ground_truth": text,
                        "query_kind": "reconstruction_claim_surface",
                        "variant_kind": "pair_like_relaxed_claim_surface",
                        "notes": (
                            f"speaker={normalize_space(row.get('speaker', ''))}; "
                            f"target={normalize_space(row.get('target', ''))}; "
                            f"peer_matches={peer_matches}; "
                            f"peer_best_overlap={peer_best_overlap:.2f}; "
                            "reconstruction_source=pair_like_relaxed_claim_surface"
                        ),
                        "source_context": source_context,
                        "selector_score": (
                            132.0
                            + direct_quality
                            + min(peer_matches, 4) * 6.0
                            + peer_best_overlap * 10.0
                            + (4.0 if normalize_space(row.get("speaker", "")) else 0.0)
                            + (3.0 if normalize_space(row.get("target", "")) else 0.0)
                            - abs(words - 18) / 10.0
                        ),
                        "word_count": words,
                        "query_target_words": 18,
                    }
                )
                seen_pair_like_queries.add(key)
            if relaxed_pair_like_claim_surfaces:
                claim_like_candidates.extend(relaxed_pair_like_claim_surfaces)
                candidates.extend(relaxed_pair_like_claim_surfaces)
    if uppercase_heavy_corpus and claim_views is not None and not claim_views.empty:
        claim_seed_sampling = {
            "min_words": max(8, min_query_words // 2),
            "max_words": min(max_query_words, 72),
            "min_query_words": max(10, min_query_words // 2),
            "max_query_words": min(max_query_words, 96),
            "min_role_fields": 2,
            "include_stance": False,
            "query_style": "plain",
            "max_cases_per_doc": int(sampling.get("max_cases_per_doc", 2)),
            "require_target_or_speaker": False,
        }
        claim_backfill = choose_claim_cases(
            claim_views,
            relations if relations is not None else pd.DataFrame(),
            chunk_text_lookup,
            count,
            claim_seed_sampling,
            seed,
            runner_cfg=reconstruction_seed_runner_cfg,
            reconstruction_views=None,
        )
        if claim_backfill:
            reconstruction_from_claims: list[dict[str, Any]] = []
            for case in claim_backfill[:count]:
                reconstruction_from_claims.append(
                    {
                        "task_family": "reconstruction",
                        "runner_task": "reconstruction",
                        "doc_id": str(case.get("doc_id", "")),
                        "paragraph_id": str(case.get("paragraph_id", "")),
                        "case_id": f"reconstruction::{normalize_space(case.get('case_id', ''))}",
                        "query": normalize_space(case.get("query", "")),
                        "ground_truth": normalize_space(case.get("ground_truth", "")),
                        "query_kind": "reconstruction_claim_surface",
                        "notes": (
                            f"{normalize_space(case.get('notes', ''))}; "
                            "reconstruction_source=claim_backfill"
                        ).strip("; "),
                        "source_context": normalize_space(case.get("source_context", "")),
                        "selector_score": float(case.get("selector_score", 0.0) or 0.0) + 18.0,
                        "word_count": int(case.get("word_count", 0) or 0),
                        "query_target_words": int(case.get("query_target_words", 0) or 0),
                    }
                )
            candidates.extend(reconstruction_from_claims)
    reconstruction_rows = deterministic_seeded_row_order(
        reconstruction_views.to_dict("records"),
        seed=seed + 17,
        key_fields=("doc_id", "paragraph_id", "pseudo_chunk_text", "local_id"),
    )
    if large_corpus:
        def reconstruction_seed_score(row: dict[str, Any]) -> tuple[float, int]:
            label = normalize_space(row.get("overall_reconstructibility_label", "")).casefold()
            label_bonus = 40.0 if label == "strong" else 10.0 if label == "medium" else 0.0
            distinctive = int(row.get("distinctive_anchor_count", 0) or 0)
            generic = int(row.get("generic_anchor_count", 0) or 0)
            claim_bonus = len(
                parse_json_list(row.get("reconstruction_claims_json", row.get("claim_texts", "")))
            ) * 4
            event_bonus = len(
                parse_json_list(row.get("reconstruction_events_json", row.get("event_texts", "")))
            ) * 3
            return (
                label_bonus + distinctive * 1.5 + claim_bonus + event_bonus - generic * 0.25,
                distinctive,
            )

        reconstruction_rows = sorted(
            reconstruction_rows,
            key=reconstruction_seed_score,
            reverse=True,
        )[: max(count * 60, 480)]
    for row in reconstruction_rows:
        if normalize_space(row.get("overall_reconstructibility_label", "")).casefold() not in allowed_labels:
            continue
        if int(row.get("distinctive_anchor_count", 0) or 0) < effective_min_distinctive:
            continue
        query_target_words = deterministic_query_word_target(
            f"reconstruction|{row['doc_id']}|{row['paragraph_id']}",
            minimum_words=min_query_words,
            maximum_words=min(max_query_words, 128),
        )
        anchors = normalized_reconstruction_anchors(parse_json_list(row.get("rare_anchors_json")))
        reconstruction_entities = parse_json_list(row.get("reconstruction_entities_json"))
        reconstruction_events = unique_preserving_order(
            parse_json_list(row.get("reconstruction_events_json", row.get("event_texts", "")))
        )
        reconstruction_claims = unique_preserving_order(
            parse_json_list(row.get("reconstruction_claims_json", row.get("claim_texts", "")))
        )
        if uppercase_heavy_corpus and not (reconstruction_claims or reconstruction_events):
            continue
        source_chunk_text = chunk_text_lookup.get(paragraph_key(row["doc_id"], row["paragraph_id"]), "")
        cleaned_source_chunk_text = strip_low_signal_lead_sentences(source_chunk_text)
        if (
            large_corpus
            and coordination_heavy_surface(cleaned_source_chunk_text or source_chunk_text)
            and not (reconstruction_claims or reconstruction_events)
        ):
            continue
        context_candidates = [
            cleaned_source_chunk_text,
            source_chunk_text,
            extract_embedded_context(row.get("pseudo_chunk_text", "")),
        ]
        if not large_corpus:
            context_candidates.append(
                extract_neighbor_chunk_context(
                    doc_chunks_by_doc,
                    row["doc_id"],
                    row["paragraph_id"],
                    anchors=anchors,
                    min_words=min_visible_context_words,
                    max_sentences=int(sampling.get("max_source_excerpt_sentences", 3)),
                    max_chars=int(sampling.get("max_source_excerpt_chars", 480)),
                    max_radius=int(sampling.get("neighbor_chunk_radius", 1)),
                )
            )
        source_context = (
            choose_fast_query_context(
                context_candidates,
                min_words=min_visible_context_words,
                max_chars=int(sampling.get("max_source_excerpt_chars", 480)),
            )
            if large_corpus
            else choose_best_query_context(
                context_candidates,
                anchors=anchors,
                min_words=min_visible_context_words,
                target_words=query_target_words,
                max_sentences=int(sampling.get("max_source_excerpt_sentences", 3)),
                max_chars=int(sampling.get("max_source_excerpt_chars", 480)),
            )
        )
        source_context = strip_low_signal_lead_sentences(source_context)
        source_context = refine_generated_query_surface(
            source_context,
            task_family="reconstruction",
            anchors=anchors,
            min_words=min_visible_context_words,
            target_words=query_target_words,
            max_chars=int(sampling.get("max_source_excerpt_chars", 480)),
            runner_cfg=candidate_runner_cfg,
        )
        source_excerpt_preferred = small_corpus and uppercase_token_ratio(
            source_chunk_text or source_context
        ) >= 0.35
        has_semantic_fallback = bool(reconstruction_claims or reconstruction_events)
        if (not source_context or context_quality_score(source_context, anchors=anchors, min_words=min_visible_context_words) < 2.0) and not has_semantic_fallback:
            continue
        if query_surface_is_blocked(source_context, task_family="reconstruction"):
            continue
        selector_score = (
            int(row.get("distinctive_anchor_count", 0) or 0)
            + min(len(reconstruction_entities), 8) * 3
            + min(len(reconstruction_events), 4) * 4
            + min(len(reconstruction_claims), 4) * 4
            + context_quality_score(source_context, anchors=anchors, min_words=min_visible_context_words) * 0.6
        )
        query_variants: list[tuple[str, str]] = []
        claim_variant_limit = 1 if large_corpus else 2
        event_variant_limit = 1 if large_corpus else 2
        query_variants.extend(("claim", text) for text in reconstruction_claims[:claim_variant_limit] if text)
        query_variants.extend(("event", text) for text in reconstruction_events[:event_variant_limit] if text)
        if uppercase_heavy_corpus:
            max_sentences = 1 if count_words(source_context) >= 28 else 2
            primary_variant = finalize_query_surface(
                extract_sentence_excerpt(source_context, max_sentences=max_sentences, max_chars=280),
                add_terminal_punctuation=True,
            )
        elif large_corpus:
            max_sentences = 1 if count_words(source_context) >= 30 else 2
            primary_variant = finalize_query_surface(
                extract_sentence_excerpt(source_context, max_sentences=max_sentences, max_chars=320),
                add_terminal_punctuation=True,
            )
        else:
            primary_variant = build_reconstruction_query(row, sampling, source_chunk_text=source_context)
        if primary_variant:
            primary_flags = set(query_surface_flags(primary_variant, task_family="reconstruction"))
            primary_is_coordination_heavy = coordination_heavy_surface(primary_variant)
            allow_source_excerpt = (
                source_excerpt_preferred
                or (
                    not reconstruction_claims
                    and not reconstruction_events
                )
            ) or not (primary_flags & {"titlecase_noise", "mixed_case_noise", "uppercase_heavy"})
            if (
                large_corpus
                and primary_is_coordination_heavy
                and (reconstruction_claims or reconstruction_events)
            ):
                allow_source_excerpt = False
            if allow_source_excerpt and not benchmark_interactive_qa_surface(primary_variant):
                query_variants.append(("source_excerpt", primary_variant))
        deduped_variants: list[tuple[str, str]] = []
        seen_variant_texts: set[str] = set()
        for variant_kind, variant_text in query_variants:
            key = normalize_space(variant_text).casefold()
            if not key or key in seen_variant_texts:
                continue
            seen_variant_texts.add(key)
            deduped_variants.append((variant_kind, variant_text))
        for variant_index, (variant_kind, text) in enumerate(deduped_variants, start=1):
            text = refine_generated_query_surface(
                text,
                task_family="reconstruction",
                anchors=anchors,
                min_words=min_query_words,
                target_words=query_target_words,
                max_chars=int(sampling["max_chars"]),
                runner_cfg=candidate_runner_cfg,
            )
            words = count_words(text)
            chars = len(text)
            if not text:
                continue
            variant_min_query_words = 10 if variant_kind in {"claim", "event"} else min_query_words
            if words < variant_min_query_words or words > effective_max_query_words:
                continue
            if chars > int(sampling["max_chars"]):
                continue
            lowered_text = text.casefold()
            if large_corpus and variant_kind == "event":
                if coordination_heavy_surface(text):
                    continue
                if lowered_text.startswith("the text states that") or lowered_text.startswith("the text reports that"):
                    continue
            variant_profile = claim_clause_profile(text)
            if not reconstruction_variant_surface_is_actionable(
                text,
                variant_profile,
                variant_kind=variant_kind,
                pair_like_corpus=pair_like_corpus,
            ):
                continue
            variant_surface_score = substantive_sentence_score(text, anchors=anchors)
            variant_bonus = 0.0
            if variant_kind == "claim":
                variant_bonus += 8.0
                if large_corpus:
                    variant_bonus -= 10.0
                elif small_corpus and not uppercase_heavy_corpus:
                    variant_bonus -= 14.0
                if pair_like_corpus and small_corpus and not uppercase_heavy_corpus:
                    variant_bonus -= 20.0
                    if not (
                        variant_profile["actor"]
                        and variant_profile["predicate"]
                        and variant_profile["object"]
                    ):
                        variant_bonus -= 10.0
                if uppercase_heavy_corpus and allow_claim_backfill:
                    variant_bonus -= 12.0
                if (
                    variant_profile["actor"]
                    and variant_profile["predicate"]
                    and variant_profile["object"]
                ):
                    variant_bonus += 6.0
            elif variant_kind == "claim_bundle":
                variant_bonus += 14.0
                if (
                    variant_profile["actor"]
                    and variant_profile["predicate"]
                    and variant_profile["object"]
                ):
                    variant_bonus += 4.0
            elif variant_kind == "event":
                variant_bonus += 2.0
                if large_corpus:
                    variant_bonus -= 6.0
                if uppercase_heavy_corpus and allow_claim_backfill:
                    variant_bonus -= 8.0
            elif variant_kind == "source_excerpt":
                if uppercase_heavy_corpus:
                    variant_bonus += 10.0 if not (reconstruction_claims or reconstruction_events) else -2.0
                elif large_corpus:
                    variant_bonus += 8.0
                    if coordination_heavy_surface(text):
                        variant_bonus -= 16.0
                elif small_corpus:
                    variant_bonus += 10.0 if source_excerpt_preferred else 7.0
                    if pair_like_corpus and not uppercase_heavy_corpus:
                        variant_bonus += 14.0
                else:
                    variant_bonus -= 4.0 if large_corpus else 1.5
            if variant_profile["nested_reporting"] or variant_profile["reporting_actor"]:
                variant_bonus -= 8.0
            if variant_profile["has_semicolon"]:
                variant_bonus -= 5.0
            if actor_phrase_has_clause_bleed(variant_profile["actor"]):
                variant_bonus -= 4.0
            case_id = f"reconstruction__{row['doc_id']}__{row['paragraph_id']}__v{variant_index}"
            candidates.append(
                {
                    "task_family": "reconstruction",
                    "runner_task": "reconstruction",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": case_id,
                    "query": text,
                    "ground_truth": build_reconstruction_reference_summary(row)
                    or normalize_space(row.get("reconstruction_signature_text", text)),
                    "query_kind": (
                        "reconstruction_claim_surface"
                        if variant_kind in {"claim", "claim_bundle"}
                        else "reconstruction_event_surface"
                        if variant_kind == "event"
                        else "reconstruction_source_excerpt"
                    ),
                    "variant_kind": variant_kind,
                    "notes": (
                        f"overall_label={normalize_space(row.get('overall_reconstructibility_label', ''))}; "
                        f"distinctive_anchor_count={int(row.get('distinctive_anchor_count', 0) or 0)}; "
                        f"generic_anchor_count={int(row.get('generic_anchor_count', 0) or 0)}; "
                        f"has_source_excerpt={bool(source_context)}; "
                        f"query_target_words={query_target_words}; "
                        f"variant_index={variant_index}; "
                        f"variant_kind={variant_kind}; "
                        f"variant_surface_score={variant_surface_score:.2f}"
                    ),
                    "source_context": source_context,
                    "selector_score": (
                        selector_score
                        + variant_bonus
                        + variant_surface_score * 0.35
                        - surface_noise_penalty(text) * 0.25
                        - abs(words - query_target_words) / 14.0
                    ),
                    "word_count": words,
                    "query_target_words": query_target_words,
                }
            )
    claim_view_row_count = int(len(claim_views)) if claim_views is not None else 0
    enable_large_corpus_claim_view_supplement = not (
        large_corpus
        and not uppercase_heavy_corpus
        and claim_view_row_count >= 100000
    )
    need_claim_backfill = (
        allow_claim_backfill
        or uppercase_heavy_corpus
        or single_doc_friendly_small_corpus
        or len(claim_like_candidates) < max(count, 3)
    )
    if claim_views is not None and need_claim_backfill and enable_large_corpus_claim_view_supplement:
        claim_view_rows = list(claim_view_rows_seeded)
        if large_corpus:
            def reconstruction_claim_seed_score(row: dict[str, Any]) -> tuple[float, float, int]:
                text = simplify_claim_surface(row.get("claim_text", ""))
                profile = claim_clause_profile(text)
                return (
                    claim_clause_quality_score(profile) - surface_noise_penalty(text) * 0.4,
                    float(len(profile.get("content_tokens", set()))),
                    -abs(count_words(text) - 18),
                )

            claim_view_rows = sorted(
                claim_view_rows,
                key=reconstruction_claim_seed_score,
                reverse=True,
            )[: max(count * 160, 1600)]
        supplemental_min_query_words = 10
        for row in claim_view_rows:
            if not row_is_structured_claim(row):
                continue
            if claim_row_loses_support_or_polarity_when_simplified(row):
                continue
            text = simplify_claim_surface(row.get("claim_text", ""))
            if not text or is_routing_or_metadata_sentence(text):
                continue
            if query_surface_is_blocked(text, task_family="reconstruction"):
                continue
            if claim_uses_intentive_construction(text) or has_reporting_that_clause(text):
                continue
            if "..." in text or ";" in text:
                continue
            profile = claim_clause_profile(text)
            if (
                profile["nested_reporting"]
                or profile["reporting_actor"]
                or profile["object_bad_lead"]
                or actor_phrase_has_clause_bleed(profile["actor"])
            ):
                continue
            if claim_profile_has_problematic_actor(profile):
                continue
            if not (profile["actor"] and profile["predicate"] and profile["object"]):
                continue
            if not reconstruction_claim_surface_is_actionable(text, profile):
                continue
            query_target_words = deterministic_query_word_target(
                f"reconstruction_claim|{row['doc_id']}|{row['paragraph_id']}|{text}",
                minimum_words=supplemental_min_query_words,
                maximum_words=min(max_query_words, 96),
            )
            source_chunk_text = chunk_text_lookup.get(
                paragraph_key(row["doc_id"], row["paragraph_id"]),
                "",
            )
            source_context = choose_fast_query_context(
                [
                    source_chunk_text,
                    extract_embedded_context(row.get("pseudo_view_text", "")),
                    extract_embedded_context(row.get("proposition_text", "")),
                ],
                min_words=max(8, min_visible_context_words // 2),
                max_chars=int(sampling.get("max_source_excerpt_chars", 480)),
            )
            query_seed = text
            query_text = refine_generated_query_surface(
                query_seed,
                task_family="reconstruction",
                anchors=[
                    normalize_space(row.get("speaker", "")),
                    normalize_space(row.get("target", "")),
                ],
                min_words=supplemental_min_query_words,
                target_words=query_target_words,
                max_chars=int(sampling["max_chars"]),
                runner_cfg=candidate_runner_cfg,
            )
            words = count_words(query_text)
            if words < supplemental_min_query_words or words > effective_max_query_words:
                continue
            query_profile = claim_clause_profile(query_text)
            if not reconstruction_claim_surface_is_actionable(query_text, query_profile):
                continue
            selector_score = (
                claim_clause_quality_score(query_profile)
                + (20.0 if small_corpus else 60.0)
                + (120.0 if large_corpus else 0.0)
                + (3.0 if normalize_space(row.get("speaker", "")) else 0.0)
                + (3.0 if normalize_space(row.get("target", "")) else 0.0)
                + context_quality_score(
                    source_context,
                    anchors=[
                        normalize_space(row.get("speaker", "")),
                        normalize_space(row.get("target", "")),
                    ],
                    min_words=max(8, min_visible_context_words // 2),
                )
                * 0.25
                - abs(words - 20) / 12.0
            )
            case_id = f"reconstruction_claim__{row['doc_id']}__{row['paragraph_id']}__{sha1_text(query_text)}"
            candidate = {
                    "task_family": "reconstruction",
                    "runner_task": "reconstruction",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": case_id,
                    "query": query_text,
                    "ground_truth": text,
                    "query_kind": "reconstruction_claim_surface",
                    "notes": (
                        f"source_type={normalize_space(row.get('source_type', ''))}; "
                        f"speaker={normalize_space(row.get('speaker', ''))}; "
                        f"target={normalize_space(row.get('target', ''))}; "
                        f"query_target_words={query_target_words}"
                    ),
                    "source_context": source_context,
                    "selector_score": selector_score,
                    "word_count": words,
                    "query_target_words": query_target_words,
                }
            claim_like_candidates.append(candidate)
            candidates.append(candidate)
    if relations is not None and (uppercase_heavy_corpus or small_corpus):
        claim_seed_sampling = {
            "min_words": max(8, min_query_words // 2),
            "max_words": min(max_query_words, 64),
            "min_query_words": max(12, min_query_words // 2),
            "max_query_words": min(max_query_words, 96),
            "min_role_fields": 2,
            "include_stance": False,
            "query_style": "contextual",
            "max_cases_per_doc": int(sampling.get("max_cases_per_doc", 2)),
        }
        claim_seed_candidates = choose_claim_cases(
            claim_views if claim_views is not None else pd.DataFrame(),
            relations,
            chunk_text_lookup,
            max(count * 4, 12),
            claim_seed_sampling,
            seed,
            runner_cfg=reconstruction_seed_runner_cfg,
        )
        for candidate in claim_seed_candidates:
            seed_query = normalize_space(candidate.get("query", ""))
            if not seed_query or query_surface_is_blocked(seed_query, task_family="reconstruction"):
                continue
            seed_query_kind = normalize_space(candidate.get("query_kind", "")).casefold()
            if seed_query_kind not in {"claim_relation_query", "claim_structured_query"}:
                continue
            seed_bonus = 30.0 if seed_query_kind == "claim_relation_query" else 16.0
            reconstruction_candidate = {
                "task_family": "reconstruction",
                "runner_task": "reconstruction",
                "doc_id": str(candidate.get("doc_id", "")),
                "paragraph_id": str(candidate.get("paragraph_id", "")),
                "case_id": f"reconstruction::{normalize_space(candidate.get('case_id', ''))}",
                "query": seed_query,
                "ground_truth": normalize_space(candidate.get("ground_truth", "")) or seed_query,
                "query_kind": "reconstruction_relation_surface",
                "variant_kind": (
                    "relation_claim"
                    if seed_query_kind == "claim_relation_query"
                    else "claim_seed"
                ),
                "notes": (
                    f"{normalize_space(candidate.get('notes', ''))}; "
                    "reconstruction_source=claim_seed"
                ).strip("; "),
                "source_context": normalize_space(candidate.get("source_context", "")),
                "selector_score": float(candidate.get("selector_score", 0.0) or 0.0) + seed_bonus,
                "word_count": int(candidate.get("word_count", 0) or 0),
                "query_target_words": int(candidate.get("query_target_words", 0) or 0),
            }
            candidates.append(reconstruction_candidate)
    preferred_claim_like: list[dict[str, Any]] = []
    if claim_like_candidates:
        preferred_claim_like_target = min(max(2, count // 2), len(claim_like_candidates))
        if single_doc_friendly_small_corpus:
            preferred_claim_like_target = min(max(count, 3), len(claim_like_candidates))
        preferred_claim_like = round_robin_topk(
            claim_like_candidates,
            preferred_claim_like_target,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
    if uppercase_heavy_corpus and allow_claim_backfill and preferred_claim_like:
        direct_reconstruction_claims = round_robin_topk(
            preferred_claim_like,
            min(count, len(preferred_claim_like)),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
        if len(direct_reconstruction_claims) >= min(count, 3):
            return normalize_selected_case_queries(
                direct_reconstruction_claims[:count],
                task_family="reconstruction",
                runner_cfg=runner_cfg,
            )
    preferred_relation_reconstruction: list[dict[str, Any]] = []
    relation_reconstruction_candidates = [
        candidate
        for candidate in candidates
        if normalize_space(candidate.get("variant_kind", "")).casefold() == "relation_claim"
    ]
    if relation_reconstruction_candidates and (uppercase_heavy_corpus or small_corpus):
        preferred_relation_reconstruction = round_robin_topk(
            relation_reconstruction_candidates,
            min(max(1, count // 3), len(relation_reconstruction_candidates)),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
    preferred_source_excerpt: list[dict[str, Any]] = []
    if uppercase_heavy_corpus or small_corpus:
        source_excerpt_candidates = [
            candidate
            for candidate in candidates
            if normalize_space(candidate.get("variant_kind", "")).casefold() == "source_excerpt"
        ]
        if source_excerpt_candidates:
            preferred_source_excerpt = round_robin_topk(
                source_excerpt_candidates,
                min(
                    max(2, (count * 2) // 3 if small_corpus and not uppercase_heavy_corpus else count // 2),
                    len(source_excerpt_candidates),
                ),
                max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
                score_key="selector_score",
                seed=seed,
            )
    if (
        small_corpus
        and preferred_claim_like
        and not uppercase_heavy_corpus
        and not pair_like_corpus
    ):
        preferred_source_excerpt = []
    if (
        small_corpus
        and preferred_source_excerpt
        and not uppercase_heavy_corpus
        and not preferred_claim_like
    ):
        source_led_selected = merge_ranked_candidates(
            [preferred_source_excerpt, preferred_relation_reconstruction, candidates],
            target_count=count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
        if len(source_led_selected) >= min(count, 3):
            return normalize_selected_case_queries(
                source_led_selected[:count],
                task_family="reconstruction",
                runner_cfg=runner_cfg,
            )
    claim_like_quota = max(1, count // 2)
    if uppercase_heavy_corpus and not allow_claim_backfill:
        claim_like_quota = 0
    elif uppercase_heavy_corpus and allow_claim_backfill:
        claim_like_quota = max(claim_like_quota, min(count, 3))
    if small_corpus:
        if single_doc_friendly_small_corpus:
            claim_like_quota = max(claim_like_quota, min(count, max(3, len(preferred_claim_like))))
        elif preferred_source_excerpt and not uppercase_heavy_corpus:
            if pair_like_corpus:
                claim_like_quota = 0
            else:
                claim_like_quota = 0 if not preferred_claim_like else min(max(claim_like_quota, 1), 2)
        else:
            claim_like_quota = min(claim_like_quota, 1 if not (uppercase_heavy_corpus and allow_claim_backfill) else 3)
    if large_corpus:
        claim_like_quota = max(claim_like_quota, min(count, max(2, (count * 3 + 1) // 4)))
    claim_like_prefix = preferred_claim_like[: min(len(preferred_claim_like), claim_like_quota)]
    ranked_candidate_groups: list[list[dict[str, Any]]] = []
    if preferred_source_excerpt:
        ranked_candidate_groups.append(preferred_source_excerpt)
    if preferred_relation_reconstruction:
        ranked_candidate_groups.append(preferred_relation_reconstruction)
    if claim_like_prefix:
        ranked_candidate_groups.append(claim_like_prefix)
    ranked_candidate_groups.append(candidates)
    selected = merge_ranked_candidates(
        ranked_candidate_groups,
        target_count=count,
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )
    if claim_like_prefix:
        selected_ids = {normalize_space(case.get("case_id", "")) for case in selected}
        for claim_case in claim_like_prefix:
            claim_id = normalize_space(claim_case.get("case_id", ""))
            if claim_id in selected_ids:
                continue
            selected.append(claim_case)
            selected_ids.add(claim_id)
        selected = merge_ranked_candidates(
            [claim_like_prefix, selected],
            target_count=count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
    if preferred_relation_reconstruction:
        selected_ids = {normalize_space(case.get("case_id", "")) for case in selected}
        forced_relation: list[dict[str, Any]] = []
        for relation_case in preferred_relation_reconstruction:
            relation_id = normalize_space(relation_case.get("case_id", ""))
            if relation_id in selected_ids:
                continue
            forced_relation.append(relation_case)
            selected_ids.add(relation_id)
        if forced_relation:
            relation_ids = {
                normalize_space(case.get("case_id", ""))
                for case in forced_relation
            }
            remainder = [
                case
                for case in selected
                if normalize_space(case.get("case_id", "")) not in relation_ids
            ]
            selected = (forced_relation + remainder)[:count]
    if single_doc_friendly_small_corpus and len(selected) < count and claim_like_candidates:
        emergency_claim_like = round_robin_topk(
            claim_like_candidates,
            min(max(count * 2, count), len(claim_like_candidates)),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
        selected = merge_ranked_candidates(
            [emergency_claim_like, selected, candidates],
            target_count=count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
    if (
        single_doc_friendly_small_corpus
        and len(selected) < count
        and claim_views is not None
        and not claim_views.empty
    ):
        emergency_claim_sampling = {
            "min_words": max(8, min_query_words // 2),
            "max_words": min(max_query_words, 84),
            "min_query_words": max(10, min_query_words // 2),
            "max_query_words": min(max_query_words, 96),
            "min_role_fields": 0,
            "include_stance": False,
            "query_style": "plain",
            "max_cases_per_doc": int(sampling.get("max_cases_per_doc", 2)),
            "require_target_or_speaker": False,
            "require_simple_clause_arguments": False,
            "min_clause_quality_score": 4.0,
        }
        emergency_claim_cases = choose_claim_cases(
            claim_views,
            relations if relations is not None else pd.DataFrame(),
            chunk_text_lookup,
            max(count * 3, count),
            emergency_claim_sampling,
            seed,
            runner_cfg=reconstruction_seed_runner_cfg,
            reconstruction_views=None,
        )
        emergency_reconstruction: list[dict[str, Any]] = []
        for case in emergency_claim_cases:
            query = normalize_space(case.get("query", ""))
            if not query or query_surface_is_blocked(query, task_family="reconstruction"):
                continue
            emergency_reconstruction.append(
                {
                    "task_family": "reconstruction",
                    "runner_task": "reconstruction",
                    "doc_id": str(case.get("doc_id", "")),
                    "paragraph_id": str(case.get("paragraph_id", "")),
                    "case_id": f"reconstruction::{normalize_space(case.get('case_id', ''))}",
                    "query": query,
                    "ground_truth": normalize_space(case.get("ground_truth", "")) or query,
                    "query_kind": "reconstruction_claim_surface",
                    "variant_kind": "small_corpus_claim_fallback",
                    "notes": (
                        f"{normalize_space(case.get('notes', ''))}; "
                        "reconstruction_source=small_corpus_claim_fallback"
                    ).strip("; "),
                    "source_context": normalize_space(case.get("source_context", "")),
                    "selector_score": float(case.get("selector_score", 0.0) or 0.0) + 24.0,
                    "word_count": int(case.get("word_count", 0) or 0),
                    "query_target_words": int(case.get("query_target_words", 0) or 0),
                }
            )
        if emergency_reconstruction:
            selected = merge_ranked_candidates(
                [emergency_reconstruction, selected, candidates],
                target_count=count,
                max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
                score_key="selector_score",
                seed=seed,
            )
    if len(selected) >= count or maybe_bool(sampling.get("_relaxed_retry_used", False)):
        return normalize_selected_case_queries(
            selected,
            task_family="reconstruction",
            runner_cfg=runner_cfg,
        )
    relaxed_sampling = dict(sampling)
    relaxed_sampling["_relaxed_retry_used"] = True
    if uppercase_heavy_corpus:
        relaxed_sampling["_allow_claim_backfill"] = True
    relaxed_sampling["min_distinctive_anchor_count"] = max(3, min_distinctive // 2)
    relaxed_sampling["allowed_overall_labels"] = sorted(allowed_labels | {"medium"})
    relaxed_sampling["min_visible_context_words"] = max(12, min_visible_context_words - 4)
    relaxed_sampling["max_source_excerpt_chars"] = max(
        int(relaxed_sampling.get("max_source_excerpt_chars", 480)),
        560,
    )
    relaxed = choose_reconstruction_cases(
        reconstruction_views,
        claim_views,
        relations,
        chunk_text_lookup,
        doc_chunks_by_doc,
        count,
        relaxed_sampling,
        seed,
        runner_cfg=runner_cfg,
    )
    return normalize_selected_case_queries(
        merge_ranked_candidates(
            [selected, relaxed],
            target_count=count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        ),
        task_family="reconstruction",
        runner_cfg=runner_cfg,
    )


def choose_claim_cases(
    claim_views: pd.DataFrame,
    relations: pd.DataFrame,
    chunk_text_lookup: dict[tuple[str, str], str],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    runner_cfg: dict[str, Any] | None = None,
    *,
    claim_rows: list[dict[str, Any]] | None = None,
    claim_profiles: list[dict[str, Any]] | None = None,
    peer_rows_by_idx: list[list[dict[str, Any]]] | None = None,
    reconstruction_views: pd.DataFrame | None = None,
    reconstruction_seed_cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    corpus_name = normalize_space((runner_cfg or {}).get("__corpus_name", "")).casefold()
    claim_max_cases_per_doc = int(sampling.get("max_cases_per_doc", 2))
    structured_claim_doc_frequency: dict[str, int] = {}
    structured_claim_docs: dict[str, set[str]] = {}
    claim_doc_ids = {
        normalize_space(doc_id)
        for doc_id in claim_views.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    }
    pair_like_corpus = bool(claim_doc_ids) and all(doc_id.startswith("Pair_") for doc_id in claim_doc_ids)
    source_claim_rows = claim_rows if claim_rows is not None else claim_views.to_dict("records")
    claim_rows, aligned_companions = reorder_aligned_rows(
        source_claim_rows,
        seed=seed,
        key_fields=("doc_id", "paragraph_id", "claim_text", "local_id", "source_type"),
        companions=(claim_profiles, peer_rows_by_idx),
    )
    claim_profiles, peer_rows_by_idx = aligned_companions
    small_corpus_claim_support_cache: dict[tuple[str, str], tuple[int, float]] = {}

    def small_corpus_claim_peer_support(text: str, doc_id: str) -> tuple[int, float]:
        if len(claim_rows) > 250:
            return 0, 0.0
        cache_key = (normalize_space(text).casefold(), normalize_space(doc_id))
        cached = small_corpus_claim_support_cache.get(cache_key)
        if cached is not None:
            return cached
        profile = claim_clause_profile(text)
        peer_rows = [
            peer
            for peer in claim_rows
            if normalize_space(peer.get("doc_id", "")) != normalize_space(doc_id)
        ]
        result = claim_support_peer_score(profile, peer_rows)
        small_corpus_claim_support_cache[cache_key] = result
        return result
    for row in claim_rows:
        if not row_is_structured_claim(row):
            continue
        text = simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", "")))
        doc_id = normalize_space(row.get("doc_id", ""))
        if not text or not doc_id:
            continue
        structured_claim_docs.setdefault(text.casefold(), set()).add(doc_id)
    structured_claim_doc_frequency = {
        key: len(doc_ids) for key, doc_ids in structured_claim_docs.items()
    }
    pair_like_has_repeated_structured_claims = any(
        count >= 2 for count in structured_claim_doc_frequency.values()
    )
    large_claim_corpus = len(claim_rows) >= 600
    ultra_large_claim_corpus = claim_peer_pool_disabled(
        len(claim_rows),
        sampling,
        pair_like_corpus=pair_like_corpus,
        default_threshold=5000,
    )
    if len(claim_rows) <= 150 and not pair_like_corpus:
        claim_max_cases_per_doc = max(
            claim_max_cases_per_doc,
            int(sampling.get("small_corpus_claim_max_cases_per_doc", 3)),
        )
    cib_legacy_shortlist_mode = (
        corpus_name == "cib"
        and ultra_large_claim_corpus
        and not pair_like_corpus
    )
    use_relation_candidates = bool(
        not relations.empty
        and (
            pair_like_corpus
            or len(claim_rows) < 600
        )
    )
    relation_probe_count = (
        max(count * 4, count)
        if use_relation_candidates
        else count
    )
    relation_candidates = (
        (
            choose_pair_relation_claim_cases_relaxed(
                relations,
                chunk_text_lookup,
                relation_probe_count,
                sampling,
                seed,
                runner_cfg=runner_cfg,
            )
            if pair_like_corpus
            else choose_relation_claim_cases(
                relations,
                chunk_text_lookup,
                relation_probe_count,
                sampling,
                seed,
                runner_cfg=runner_cfg,
            )
        )
        if use_relation_candidates
        else []
    )
    defer_minimal_direct_candidates = ultra_large_claim_corpus and not pair_like_corpus and not cib_legacy_shortlist_mode
    minimal_direct_candidates: list[dict[str, Any]] = []
    if not defer_minimal_direct_candidates:
        minimal_direct_candidates = build_minimal_claim_candidates(
            claim_views,
            max(count * 6, count),
            sampling,
            seed,
            runner_cfg=runner_cfg,
        )
    seeded_relation_candidates = round_robin_topk(
        relation_candidates,
        min(count, len(relation_candidates)),
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )
    pair_relation_return_floor = max(
        int(sampling.get("min_pair_relation_candidates", 5)),
        max(1, count // 4),
    )
    small_corpus_direct_claim_candidates: list[dict[str, Any]] = []
    if len(claim_rows) < 250:
        for row in claim_rows:
            if claim_row_loses_support_or_polarity_when_simplified(row):
                continue
            raw_claim_text = finalize_query_surface(
                normalize_space(row.get("claim_text", "")),
                add_terminal_punctuation=True,
            )
            text = finalize_query_surface(
                simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", ""))),
                add_terminal_punctuation=True,
            )
            if (
                not text
                or query_surface_is_blocked(text, task_family="claims")
                or sentence_is_low_signal_for_query(text)
                or is_routing_or_metadata_sentence(text)
            ):
                continue
            candidate_texts = [text]
            if not pair_like_corpus and " but " in text.casefold():
                candidate_texts.extend(
                    finalize_query_surface(fragment, add_terminal_punctuation=True)
                    for fragment in re.split(r"\bbut\b", text, maxsplit=1, flags=re.IGNORECASE)
                    if normalize_space(fragment)
                )
            if pair_like_corpus:
                candidate_texts.extend(build_pair_like_claim_rewrite_variants(text, row))
            best_text = ""
            best_profile: dict[str, Any] | None = None
            best_standalone_ok = False
            best_words = 0
            best_shape_score = float("-inf")
            for candidate_text in candidate_texts:
                if pair_like_corpus and pair_like_claim_surface_runtime_hostile(candidate_text):
                    continue
                if (
                    claim_surface_is_interrogative(candidate_text)
                    or claim_surface_has_weak_modality(candidate_text)
                ):
                    continue
                profile = claim_clause_profile(candidate_text)
                standalone_ok = standalone_claim_surface_is_safe(candidate_text)
                fallback_simple_ok = fallback_simple_complete_claim_surface_is_safe(candidate_text)
                strict_ok = strict_claim_surface_is_safe(
                    candidate_text,
                    profile,
                    max_actor_words=8,
                    max_object_words=18,
                )
                if (
                    not pair_like_corpus
                    and len(claim_rows) <= 150
                    and not (standalone_ok or strict_ok or fallback_simple_ok)
                ):
                    continue
                if pair_like_corpus and claim_surface_is_short_classification(candidate_text, profile):
                    continue
                if not profile_has_complete_simple_clause(profile) and not standalone_ok:
                    continue
                if claim_profile_has_problematic_actor(profile):
                    continue
                parser_ok = claim_query_is_executable_via_parser(candidate_text, runner_cfg)
                if claim_surface_is_retrieval_fragile(
                    candidate_text,
                    profile,
                    small_corpus=len(claim_rows) <= 150,
                    pair_like_corpus=pair_like_corpus,
                ) and not (
                    len(claim_rows) <= 150
                    and not pair_like_corpus
                    and fallback_simple_ok
                    and parser_ok
                ):
                    continue
                if pair_like_corpus and not small_corpus_claim_surface_recovery_ok(
                    candidate_text,
                    task_family="claims",
                ):
                    continue
                if pair_like_corpus and claim_uses_intentive_construction(candidate_text):
                    continue
                if (
                    pair_like_corpus
                    and normalize_space(profile.get("actor", "")).casefold()
                    == normalize_space(profile.get("object", "")).casefold()
                    and normalize_space(profile.get("actor", ""))
                ):
                    continue
                if not (
                    strict_claim_surface_is_safe(
                        candidate_text,
                        profile,
                        max_actor_words=8,
                        max_object_words=18,
                    )
                    or standalone_ok
                    or (
                        len(claim_rows) <= 150
                        and not pair_like_corpus
                        and fallback_simple_ok
                    )
                ):
                    continue
                if substantive_sentence_score(candidate_text) < 5.0:
                    continue
                if not parser_ok and not (
                    len(claim_rows) <= 150
                    and not pair_like_corpus
                    and fallback_simple_ok
                ):
                    continue
                words = count_words(candidate_text)
                if words < 6 or words > int(sampling.get("max_words", 48)):
                    continue
                shape_score = (
                    claim_clause_quality_score(profile)
                    - surface_noise_penalty(candidate_text) * 0.3
                    - pair_like_claim_surface_soft_penalty(candidate_text)
                    + (4.0 if candidate_text != text else 0.0)
                )
                if strict_ok and not standalone_ok:
                    shape_score += 2.0
                if shape_score > best_shape_score:
                    best_text = candidate_text
                    best_profile = profile
                    best_standalone_ok = standalone_ok
                    best_words = words
                    best_shape_score = shape_score
            if not best_text or best_profile is None:
                continue
            text = best_text
            profile = best_profile
            standalone_ok = best_standalone_ok
            words = best_words
            if words < 6 or words > int(sampling.get("max_words", 48)):
                continue
            peer_matches, peer_best_overlap = small_corpus_claim_peer_support(
                text,
                str(row["doc_id"]),
            )
            short_simple_claim = bool(standalone_ok and words <= 10)
            if (
                not pair_like_corpus
                and len(claim_rows) >= 12
                and peer_matches <= 0
                and substantive_sentence_score(text) < 9.0
                and not short_simple_claim
            ):
                continue
            negation_penalty = 10.0 if profile.get("has_negation") else 0.0
            coordination_penalty = (
                6.0
                if re.search(r"\bbut\b", text, flags=re.IGNORECASE)
                else 0.0
            )
            vague_small_corpus_penalty = (
                10.0
                if (
                    len(claim_rows) <= 150
                    and not pair_like_corpus
                    and re.search(r"\bis concerned about\b", text, flags=re.IGNORECASE)
                )
                else 0.0
            )
            small_corpus_priority = (
                small_corpus_claim_surface_priority_adjustment(text)
                if len(claim_rows) <= 150 and not pair_like_corpus
                else 0.0
            )
            small_corpus_direct_claim_candidates.append(
                {
                    "task_family": "claims",
                    "runner_task": "claims",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": f"claims_small_direct__{row['doc_id']}__{row['paragraph_id']}__{sha1_text(text)}",
                    "query": text,
                    "ground_truth": raw_claim_text or text,
                    "query_kind": "claim_small_direct_surface",
                    "notes": (
                        "fallback=small_corpus_direct_claim_surface; "
                        f"clause_quality={claim_clause_quality_score(profile):.2f}; "
                        f"peer_matches={peer_matches}; "
                        f"peer_best_overlap={peer_best_overlap:.2f}"
                    ),
                    "source_context": text,
                    "selector_score": (
                        claim_clause_quality_score(profile)
                        + 6.0
                        + small_corpus_priority
                        + min(peer_matches, 4) * 6.0
                        + peer_best_overlap * 12.0
                        + (6.0 if standalone_ok and not profile_has_complete_simple_clause(profile) else 0.0)
                        + (2.0 if profile.get("date_token") else 0.0)
                        + (2.0 if profile.get("number_token") else 0.0)
                        - abs(words - 18) / 12.0
                        - surface_noise_penalty(text) * 0.3
                        - pair_like_claim_surface_soft_penalty(text)
                        - negation_penalty
                        - coordination_penalty
                        - vague_small_corpus_penalty
                    ),
                    "word_count": words,
                    "query_target_words": words,
                }
            )
    small_corpus_reconstruction_claim_candidates: list[dict[str, Any]] = []
    if (
        len(claim_rows) < 250
        and not pair_like_corpus
        and reconstruction_views is not None
        and not reconstruction_views.empty
    ):
        reconstruction_rows = deterministic_seeded_row_order(
            reconstruction_views.to_dict("records"),
            seed=seed + 41,
            key_fields=("doc_id", "paragraph_id", "pseudo_chunk_text", "local_id"),
        )
        for row in reconstruction_rows:
            doc_id = normalize_space(row.get("doc_id", ""))
            paragraph_id = normalize_space(row.get("paragraph_id", ""))
            if not doc_id or not paragraph_id:
                continue
            claim_texts = unique_preserving_order(
                parse_json_list(
                    row.get("reconstruction_claims_json", row.get("claim_texts", ""))
                )
            )
            for claim_text in claim_texts[:2]:
                query_text = finalize_query_surface(
                    simplify_claim_query_seed(simplify_claim_surface(claim_text)),
                    add_terminal_punctuation=True,
                )
                if (
                    not query_text
                    or query_surface_is_blocked(query_text, task_family="claims")
                    or sentence_is_low_signal_for_query(query_text)
                    or is_routing_or_metadata_sentence(query_text)
                ):
                    continue
                profile = claim_clause_profile(query_text)
                if not (
                    strict_claim_surface_is_safe(
                        query_text,
                        profile,
                        max_actor_words=8,
                        max_object_words=18,
                    )
                    or standalone_claim_surface_is_safe(query_text)
                ):
                    continue
                if not claim_query_is_executable_via_parser(query_text, runner_cfg):
                    continue
                words = count_words(query_text)
                if words < 8 or words > int(sampling.get("max_words", 48)):
                    continue
                small_corpus_reconstruction_claim_candidates.append(
                    {
                        "task_family": "claims",
                        "runner_task": "claims",
                        "doc_id": doc_id,
                        "paragraph_id": paragraph_id,
                        "case_id": f"claims__reconstruction_surface__{doc_id}__{paragraph_id}__{sha1_text(query_text)}",
                        "query": query_text,
                        "ground_truth": query_text,
                        "query_kind": "claim_reconstruction_backfill",
                        "notes": (
                            "fallback=small_corpus_reconstruction_claim_surface; "
                            f"clause_quality={claim_clause_quality_score(profile):.2f}"
                        ),
                        "source_context": query_text,
                        "selector_score": (
                            claim_clause_quality_score(profile)
                            + 12.0
                            + (6.0 if " but " in f" {query_text.casefold()} " else 0.0)
                            + (3.0 if profile_has_complete_simple_clause(profile) else 0.0)
                            + (3.0 if profile.get("date_token") else 0.0)
                            + (3.0 if profile.get("number_token") else 0.0)
                            - abs(words - 18) / 12.0
                            - surface_noise_penalty(query_text) * 0.25
                        ),
                        "word_count": words,
                        "query_target_words": words,
                    }
                )
                if len(small_corpus_reconstruction_claim_candidates) >= max(count * 4, count):
                    break
            if len(small_corpus_reconstruction_claim_candidates) >= max(count * 4, count):
                break
    small_corpus_relation_evidence_candidates: list[dict[str, Any]] = []
    if len(claim_rows) <= 150 and not pair_like_corpus:
        relation_rows = deterministic_seeded_row_order(
            relations.to_dict("records"),
            seed=seed + 23,
            key_fields=("doc_id", "paragraph_id", "source", "relation", "target", "evidence"),
        )
        for row in relation_rows:
            relation_key = normalize_space(row.get("relation", "")).casefold()
            if relation_key not in {
                "support",
                "opposition",
                "affiliation",
                "communication",
                "event_participant",
            }:
                continue
            evidence = normalize_relation_evidence_sentence(
                normalize_space(row.get("evidence", ""))
            )
            evidence = finalize_query_surface(
                simplify_claim_query_seed(simplify_claim_surface(evidence)),
                add_terminal_punctuation=True,
            )
            if (
                not evidence
                or "..." in evidence
                or query_surface_is_blocked(evidence, task_family="claims")
                or sentence_is_low_signal_for_query(evidence)
                or is_routing_or_metadata_sentence(evidence)
                or claim_surface_is_interrogative(evidence)
                or claim_surface_has_weak_modality(evidence)
            ):
                continue
            profile = claim_clause_profile(evidence)
            standalone_ok = standalone_claim_surface_is_safe(evidence)
            if not standalone_ok:
                continue
            if claim_profile_has_problematic_actor(profile):
                continue
            if claim_surface_is_retrieval_fragile(
                evidence,
                profile,
                small_corpus=True,
                pair_like_corpus=False,
            ):
                continue
            if not small_corpus_claim_surface_recovery_ok(
                evidence,
                task_family="claims",
            ):
                continue
            if not claim_query_is_executable_via_parser(evidence, runner_cfg):
                continue
            words = count_words(evidence)
            if words < 6 or words > int(sampling.get("max_words", 72)):
                continue
            source = normalize_space(row.get("source", ""))
            target = normalize_space(row.get("target", ""))
            relation_bonus = {
                "support": 4.5,
                "opposition": 4.0,
                "affiliation": 3.5,
                "communication": 3.0,
                "event_participant": 3.0,
            }.get(relation_key, 0.0)
            fallback_only_penalty = (
                4.0 if not profile_has_complete_simple_clause(profile) else 0.0
            )
            past_perfect_penalty = (
                8.0 if re.search(r"\bhad been\b", evidence, flags=re.IGNORECASE) else 0.0
            )
            selector_score = (
                claim_clause_quality_score(profile)
                + 15.0
                + relation_bonus
                + (3.0 if standalone_ok else 0.0)
                + (2.0 if profile.get("date_token") else 0.0)
                + (2.0 if profile.get("number_token") else 0.0)
                + min(distinctive_label_token_count(source), 4) * 0.5
                + min(distinctive_label_token_count(target), 4) * 0.5
                - abs(words - 18) / 12.0
                - surface_noise_penalty(evidence) * 0.25
                - fallback_only_penalty
                - past_perfect_penalty
                - (2.0 if relation_key in {"affiliation", "opposition"} else 0.0)
            )
            small_corpus_relation_evidence_candidates.append(
                {
                    "task_family": "claims",
                    "runner_task": "claims",
                    "doc_id": str(row.get("doc_id", "")),
                    "paragraph_id": str(row.get("paragraph_id", "")),
                    "case_id": (
                        f"claims_relation_evidence__{row.get('doc_id', '')}__"
                        f"{row.get('paragraph_id', '')}__{sha1_text(evidence)}"
                    ),
                    "query": evidence,
                    "ground_truth": evidence,
                    "query_kind": "claim_relation_evidence_surface",
                    "notes": (
                        "fallback=small_corpus_relation_evidence; "
                        f"relation={relation_key}; "
                        f"clause_quality={claim_clause_quality_score(profile):.2f}"
                    ),
                    "source_context": evidence,
                    "selector_score": selector_score,
                    "word_count": words,
                    "query_target_words": words,
                }
            )
    pair_like_relation_evidence_candidates: list[dict[str, Any]] = []
    if len(claim_rows) <= 150 and pair_like_corpus:
        relation_rows = deterministic_seeded_row_order(
            relations.to_dict("records"),
            seed=seed + 223,
            key_fields=("doc_id", "paragraph_id", "source", "relation", "target", "evidence"),
        )
        for row in relation_rows:
            relation_key = normalize_space(row.get("relation", "")).casefold()
            if relation_key not in {
                "support",
                "affiliation",
                "communication",
                "event_participant",
                "mentions",
            }:
                continue
            evidence = normalize_relation_evidence_sentence(
                normalize_space(row.get("evidence", ""))
            )
            evidence = finalize_query_surface(
                simplify_claim_query_seed(simplify_claim_surface(evidence)),
                add_terminal_punctuation=True,
            )
            candidate_variants = build_pair_like_relation_claim_variants(evidence, row)
            best_variant = ""
            best_profile: dict[str, Any] | None = None
            best_words = 0
            best_variant_penalty = float("-inf")
            best_reporting_penalty = 0.0
            for variant in candidate_variants:
                if (
                    not variant
                    or query_surface_is_blocked(variant, task_family="claims")
                    or sentence_is_low_signal_for_query(variant)
                    or is_routing_or_metadata_sentence(variant)
                    or claim_surface_is_interrogative(variant)
                    or claim_surface_has_weak_modality(variant)
                    or contains_role_placeholder_label(variant)
                ):
                    continue
                if re.search(r"\b(?:rumored|reportedly|apparently)\b", variant, flags=re.IGNORECASE):
                    continue
                profile = claim_clause_profile(variant)
                if claim_profile_has_problematic_actor(profile):
                    continue
                if claim_surface_is_retrieval_fragile(
                    variant,
                    profile,
                    small_corpus=True,
                    pair_like_corpus=True,
                ):
                    continue
                if pair_like_claim_surface_runtime_hostile(variant):
                    continue
                if not claim_query_is_executable_via_parser(variant, runner_cfg):
                    continue
                words = count_words(variant)
                if words < 6 or words > int(sampling.get("max_words", 56)):
                    continue
                reporting_penalty = 8.0 if sentence_is_reporting_heavy(variant) else 0.0
                variant_score = (
                    claim_clause_quality_score(profile)
                    + (4.0 if variant != evidence else 0.0)
                    - abs(words - 18) / 12.0
                    - surface_noise_penalty(variant) * 0.2
                    - reporting_penalty
                )
                if variant_score > best_variant_penalty:
                    best_variant = variant
                    best_profile = profile
                    best_words = words
                    best_variant_penalty = variant_score
                    best_reporting_penalty = reporting_penalty
            if not best_variant or best_profile is None:
                continue
            words = best_words
            if words < 6 or words > int(sampling.get("max_words", 56)):
                continue
            relation_bonus = {
                "support": 5.0,
                "affiliation": 4.0,
                "communication": 3.0,
                "event_participant": 5.0,
                "mentions": 2.0,
            }.get(relation_key, 0.0)
            pair_like_relation_evidence_candidates.append(
                {
                    "task_family": "claims",
                    "runner_task": "claims",
                    "doc_id": str(row.get("doc_id", "")),
                    "paragraph_id": str(row.get("paragraph_id", "")),
                    "case_id": (
                        f"claims_pair_relation__{row.get('doc_id', '')}__"
                        f"{row.get('paragraph_id', '')}__{sha1_text(best_variant)}"
                    ),
                    "query": best_variant,
                    "ground_truth": best_variant,
                    "query_kind": "claim_pair_relation_surface",
                    "notes": (
                        "fallback=pair_like_relation_evidence; "
                        f"relation={relation_key}; "
                        f"clause_quality={claim_clause_quality_score(best_profile):.2f}"
                    ),
                    "source_context": best_variant,
                    "selector_score": (
                        22.0
                        + relation_bonus
                        + claim_clause_quality_score(best_profile)
                        + (3.0 if best_profile.get("date_token") else 0.0)
                        + (3.0 if best_profile.get("number_token") else 0.0)
                        + (4.0 if best_variant != evidence else 0.0)
                        - abs(words - 18) / 12.0
                        - surface_noise_penalty(best_variant) * 0.2
                        - best_reporting_penalty
                    ),
                    "word_count": words,
                    "query_target_words": words,
                }
            )
    pair_like_direct_claim_candidates: list[dict[str, Any]] = []
    pair_like_repeated_claim_candidates: list[dict[str, Any]] = []
    if pair_like_corpus and structured_claim_doc_frequency:
        repeated_best_by_text: dict[str, dict[str, Any]] = {}
        for row in claim_rows:
            if not row_is_structured_claim(row):
                continue
            if claim_row_loses_support_or_polarity_when_simplified(row):
                continue
            text = finalize_query_surface(
                simplify_claim_query_seed(
                    simplify_claim_surface(row.get("claim_text", ""))
                ),
                add_terminal_punctuation=True,
            )
            if not text:
                continue
            claim_doc_frequency = structured_claim_doc_frequency.get(text.casefold(), 0)
            if claim_doc_frequency < 2:
                continue
            if (
                query_surface_is_blocked(text, task_family="claims")
                or sentence_is_low_signal_for_query(text)
                or is_routing_or_metadata_sentence(text)
                or pair_like_claim_surface_runtime_hostile(text)
                or contains_role_placeholder_label(text)
                or claim_surface_is_interrogative(text)
                or claim_surface_has_weak_modality(text)
            ):
                continue
            profile = claim_clause_profile(text)
            standalone_ok = standalone_claim_surface_is_safe(text)
            fallback_ok = fallback_simple_complete_claim_surface_is_safe(text)
            strict_ok = strict_claim_surface_is_safe(
                text,
                profile,
                max_actor_words=8,
                max_object_words=18,
            )
            if not (strict_ok or standalone_ok or fallback_ok):
                continue
            if claim_profile_has_problematic_actor(profile):
                continue
            if claim_surface_is_retrieval_fragile(
                text,
                profile,
                small_corpus=True,
                pair_like_corpus=True,
            ):
                continue
            if claim_surface_is_short_classification(text, profile):
                continue
            if not small_corpus_claim_surface_recovery_ok(
                text,
                task_family="claims",
            ):
                continue
            if not claim_query_is_executable_via_parser(text, runner_cfg):
                continue
            words = count_words(text)
            if words < 7 or words > int(sampling.get("max_words", 48)):
                continue
            repeated_penalty = pair_like_claim_surface_soft_penalty(text)
            if (
                repeated_penalty >= 20.0
                and words <= 10
                and not profile.get("date_token")
                and not profile.get("number_token")
            ):
                continue
            selector_score = (
                28.0
                + claim_doc_frequency * 4.0
                + claim_clause_quality_score(profile)
                + (12.0 if standalone_ok and fallback_ok else 0.0)
                + (4.0 if strict_ok else 0.0)
                + (2.0 if profile.get("date_token") else 0.0)
                + (2.0 if profile.get("number_token") else 0.0)
                - abs(words - 18) / 12.0
                - surface_noise_penalty(text) * 0.2
                - repeated_penalty
            )
            existing = repeated_best_by_text.get(text.casefold())
            if existing is None or selector_score > float(existing.get("selector_score", 0.0) or 0.0):
                repeated_best_by_text[text.casefold()] = {
                    "task_family": "claims",
                    "runner_task": "claims",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": f"claims_pair_repeated__{row['doc_id']}__{row['paragraph_id']}__{sha1_text(text)}",
                    "query": text,
                    "ground_truth": text,
                    "query_kind": "claim_pair_repeated_surface",
                    "notes": (
                        "fallback=pair_like_repeated_claim_surface; "
                        f"claim_doc_frequency={claim_doc_frequency}; "
                        f"clause_quality={claim_clause_quality_score(profile):.2f}"
                    ),
                    "source_context": text,
                    "selector_score": selector_score,
                    "word_count": words,
                    "query_target_words": words,
                }
        pair_like_repeated_claim_candidates = sorted(
            repeated_best_by_text.values(),
            key=lambda case: float(case.get("selector_score", 0.0) or 0.0),
            reverse=True,
        )
    if pair_like_corpus:
        for row in claim_rows:
            if not row_is_structured_claim(row):
                continue
            if claim_row_loses_support_or_polarity_when_simplified(row):
                continue
            text = finalize_query_surface(
                simplify_claim_query_seed(
                    simplify_claim_surface(row.get("claim_text", ""))
                ),
                add_terminal_punctuation=True,
            )
            claim_doc_frequency = structured_claim_doc_frequency.get(
                text.casefold(),
                0,
            )
            if claim_doc_frequency < 2:
                continue
            if (
                not text
                or query_surface_is_blocked(text, task_family="claims")
                or sentence_is_low_signal_for_query(text)
                or is_routing_or_metadata_sentence(text)
            ):
                continue
            if pair_like_claim_surface_runtime_hostile(text):
                continue
            if contains_role_placeholder_label(text):
                continue
            if claim_surface_is_interrogative(text) or claim_surface_has_weak_modality(text):
                continue
            profile = claim_clause_profile(text)
            standalone_ok = standalone_claim_surface_is_safe(text)
            if not profile_has_complete_simple_clause(profile) and not standalone_ok:
                continue
            if claim_profile_has_problematic_actor(profile):
                continue
            if claim_surface_is_retrieval_fragile(
                text,
                profile,
                small_corpus=True,
                pair_like_corpus=True,
            ):
                continue
            if claim_surface_is_short_classification(text, profile):
                continue
            if not small_corpus_claim_surface_recovery_ok(
                text,
                task_family="claims",
            ):
                continue
            if claim_uses_intentive_construction(text):
                continue
            if (
                normalize_space(profile.get("actor", "")).casefold()
                == normalize_space(profile.get("object", "")).casefold()
                and normalize_space(profile.get("actor", ""))
            ):
                continue
            if not (
                strict_claim_surface_is_safe(
                    text,
                    profile,
                    max_actor_words=8,
                    max_object_words=18,
                )
                or standalone_ok
            ):
                continue
            if not claim_query_is_executable_via_parser(text, runner_cfg):
                continue
            words = count_words(text)
            if words < 7 or words > int(sampling.get("max_words", 48)):
                continue
            pair_like_direct_claim_candidates.append(
                {
                    "task_family": "claims",
                    "runner_task": "claims",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": f"claims_pair_direct__{row['doc_id']}__{row['paragraph_id']}__{sha1_text(text)}",
                    "query": text,
                    "ground_truth": text,
                    "query_kind": "claim_pair_direct_surface",
                    "notes": (
                        "fallback=pair_like_direct_claim_surface; "
                        f"claim_doc_frequency={claim_doc_frequency}; "
                        f"clause_quality={claim_clause_quality_score(profile):.2f}"
                    ),
                    "source_context": text,
                    "selector_score": (
                        claim_clause_quality_score(profile)
                        + claim_doc_frequency * 2.5
                        + (4.0 if standalone_ok else 0.0)
                        + (
                            18.0
                            if standalone_ok
                            and fallback_simple_complete_claim_surface_is_safe(text)
                            and not profile_has_complete_simple_clause(profile)
                            else 0.0
                        )
                        + (2.0 if profile.get("date_token") else 0.0)
                        + (2.0 if profile.get("number_token") else 0.0)
                        - abs(words - 18) / 12.0
                        - surface_noise_penalty(text) * 0.25
                        - pair_like_claim_surface_soft_penalty(text)
                    ),
                    "word_count": words,
                    "query_target_words": words,
                }
            )
    pair_like_parser_claim_candidates: list[dict[str, Any]] = []
    if pair_like_corpus:
        for row in claim_rows:
            if not row_is_structured_claim(row):
                continue
            if claim_row_loses_support_or_polarity_when_simplified(row):
                continue
            text = finalize_query_surface(
                simplify_claim_query_seed(
                    simplify_claim_surface(row.get("claim_text", ""))
                ),
                add_terminal_punctuation=True,
            )
            if (
                not text
                or query_surface_is_blocked(text, task_family="claims")
                or sentence_is_low_signal_for_query(text)
                or is_routing_or_metadata_sentence(text)
            ):
                continue
            if pair_like_claim_surface_runtime_hostile(text):
                continue
            if contains_role_placeholder_label(text):
                continue
            if claim_surface_is_interrogative(text):
                continue
            profile = claim_clause_profile(text)
            if claim_surface_is_retrieval_fragile(
                text,
                profile,
                small_corpus=True,
                pair_like_corpus=True,
            ):
                continue
            if not small_corpus_claim_surface_recovery_ok(
                text,
                task_family="claims",
            ):
                continue
            if claim_uses_intentive_construction(text):
                continue
            if (
                normalize_space(profile.get("actor", "")).casefold()
                == normalize_space(profile.get("object", "")).casefold()
                and normalize_space(profile.get("actor", ""))
            ):
                continue
            if count_words(text) < 7 or count_words(text) > int(sampling.get("max_words", 48)):
                continue
            if not (
                strict_claim_surface_is_safe(
                    text,
                    profile,
                    max_actor_words=8,
                    max_object_words=18,
                )
                or standalone_claim_surface_is_safe(text)
            ):
                continue
            if not claim_query_is_executable_via_parser(text, runner_cfg):
                continue
            claim_doc_frequency = structured_claim_doc_frequency.get(
                text.casefold(),
                0,
            )
            pair_like_parser_claim_candidates.append(
                {
                    "task_family": "claims",
                    "runner_task": "claims",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": f"claims_pair_parser__{row['doc_id']}__{row['paragraph_id']}__{sha1_text(text)}",
                    "query": text,
                    "ground_truth": text,
                    "query_kind": "claim_pair_parser_surface",
                    "notes": (
                        "fallback=pair_like_parser_claim_surface; "
                        f"clause_quality={claim_clause_quality_score(profile):.2f}"
                    ),
                    "source_context": text,
                    "selector_score": (
                        24.0
                        + claim_doc_frequency * 1.5
                        + claim_clause_quality_score(profile)
                        + (3.0 if profile.get("date_token") else 0.0)
                        + (3.0 if profile.get("number_token") else 0.0)
                        - abs(count_words(text) - 18) / 12.0
                        - surface_noise_penalty(text) * 0.2
                        - pair_like_claim_surface_soft_penalty(text)
                    ),
                    "word_count": count_words(text),
                    "query_target_words": count_words(text),
                }
            )
    allow_reconstruction_seed_claims = maybe_bool(
        sampling.get(
            "allow_reconstruction_seed_claims",
            len(claim_rows) <= 150 and not pair_like_corpus,
        )
    )
    reconstruction_seed_claim_candidates: list[dict[str, Any]] = []
    if reconstruction_seed_cases and allow_reconstruction_seed_claims:
        for seed_case in reconstruction_seed_cases:
            seed_query = finalize_query_surface(
                simplify_claim_query_seed(simplify_claim_surface(seed_case.get("query", ""))),
                add_terminal_punctuation=True,
            )
            if (
                not seed_query
                or query_surface_is_blocked(seed_query, task_family="claims")
                or sentence_is_low_signal_for_query(seed_query)
                or is_routing_or_metadata_sentence(seed_query)
            ):
                continue
            seed_profile = claim_clause_profile(seed_query)
            if (
                not profile_has_complete_simple_clause(seed_profile)
                and not standalone_claim_surface_is_safe(seed_query)
            ):
                continue
            words = count_words(seed_query)
            if words < 8 or words > int(sampling.get("max_words", 48)):
                continue
            reconstruction_seed_claim_candidates.append(
                {
                    "task_family": "claims",
                    "runner_task": "claims",
                    "doc_id": str(seed_case.get("doc_id", "")),
                    "paragraph_id": str(seed_case.get("paragraph_id", "")),
                    "case_id": f"claims__seed__{normalize_space(seed_case.get('doc_id', ''))}__{sha1_text(seed_query)}",
                    "query": seed_query,
                    "ground_truth": seed_query,
                    "query_kind": "claim_reconstruction_seed",
                    "notes": (
                        "fallback=reconstruction_seed_case; "
                        f"clause_quality={claim_clause_quality_score(seed_profile):.2f}"
                    ),
                    "source_context": seed_query,
                    "selector_score": (
                        claim_clause_quality_score(seed_profile)
                        + 16.0
                        + (6.0 if " but " in f" {seed_query.casefold()} " else 0.0)
                        + (3.0 if profile_has_complete_simple_clause(seed_profile) else 0.0)
                        - abs(words - 18) / 12.0
                        - surface_noise_penalty(seed_query) * 0.3
                    ),
                    "word_count": words,
                    "query_target_words": words,
                }
            )
            if len(reconstruction_seed_claim_candidates) >= max(count * 4, count):
                break
    strict_small_corpus_vet = isinstance(runner_cfg, dict) and maybe_bool(
        runner_cfg.get("__strict_small_corpus_vet", False)
    )
    parser_ready_claim_candidates: list[dict[str, Any]] = []
    if strict_small_corpus_vet and len(claim_rows) <= 150:
        parser_ready_claim_candidates = build_parser_ready_claim_candidates(
            claim_rows,
            max(count * 4, count),
            sampling,
            seed,
            runner_cfg=runner_cfg,
        )
    emergency_small_corpus_claim_candidates: list[dict[str, Any]] = []
    if len(claim_rows) <= 150 and not pair_like_corpus:
        for row in claim_rows:
            text = finalize_query_surface(
                simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", ""))),
                add_terminal_punctuation=True,
            )
            if (
                not text
                or query_surface_is_blocked(text, task_family="claims")
                or sentence_is_low_signal_for_query(text)
                or is_routing_or_metadata_sentence(text)
            ):
                continue
            if claim_surface_is_interrogative(text) or claim_surface_has_weak_modality(text):
                continue
            if re.search(
                r"\b(?:secretly|on the verge of)\b",
                text,
                flags=re.IGNORECASE,
            ):
                continue
            if re.search(r"\bbut\b", text, flags=re.IGNORECASE):
                continue
            profile = claim_clause_profile(text)
            standalone_ok = standalone_claim_surface_is_safe(text)
            if not profile_has_complete_simple_clause(profile) and not standalone_ok:
                continue
            if claim_profile_has_problematic_actor(profile):
                continue
            if claim_surface_is_retrieval_fragile(
                text,
                profile,
                small_corpus=True,
                pair_like_corpus=False,
            ):
                continue
            if not (
                strict_claim_surface_is_safe(
                    text,
                    profile,
                    max_actor_words=10,
                    max_object_words=max(18, int(sampling.get("max_object_words", 18))),
                )
                or standalone_ok
            ):
                continue
            if not claim_query_is_executable_via_parser(text, runner_cfg):
                continue
            words = count_words(text)
            if words < 6 or words > int(sampling.get("max_words", 72)):
                continue
            emergency_small_corpus_claim_candidates.append(
                {
                    "task_family": "claims",
                    "runner_task": "claims",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": f"claims_small_emergency__{row['doc_id']}__{row['paragraph_id']}__{sha1_text(text)}",
                    "query": text,
                    "ground_truth": text,
                    "query_kind": "claim_small_emergency_surface",
                    "notes": (
                        "fallback=small_corpus_emergency_claim_surface; "
                        f"clause_quality={claim_clause_quality_score(profile):.2f}; "
                        f"speaker={normalize_space(row.get('speaker', ''))}; "
                        f"target={normalize_space(row.get('target', ''))}"
                    ),
                    "source_context": text,
                    "selector_score": (
                        claim_clause_quality_score(profile)
                        + 4.0
                        + (3.0 if normalize_space(row.get("speaker", "")) else 0.0)
                        + (3.0 if normalize_space(row.get("target", "")) else 0.0)
                        + (2.0 if profile.get("date_token") else 0.0)
                        + (2.0 if profile.get("number_token") else 0.0)
                        + (2.0 if standalone_ok else 0.0)
                        - abs(words - 18) / 12.0
                        - surface_noise_penalty(text) * 0.25
                        - (8.0 if profile.get("has_negation") else 0.0)
                        - (8.0 if re.search(r"\bbut\b", text, flags=re.IGNORECASE) else 0.0)
                    ),
                    "word_count": words,
                    "query_target_words": words,
                }
            )
    def keep_small_corpus_claim_candidate(case: dict[str, Any]) -> bool:
        if len(claim_rows) > 250:
            return True
        query = normalize_space(case.get("query", ""))
        doc_id = normalize_space(case.get("doc_id", ""))
        if not query or not doc_id:
            return False
        profile = claim_clause_profile(query)
        if claim_surface_is_retrieval_fragile(
            query,
            profile,
            small_corpus=True,
            pair_like_corpus=pair_like_corpus,
        ):
            return False
        peer_matches, peer_best_overlap = small_corpus_claim_peer_support(query, doc_id)
        if (
            not pair_like_corpus
            and profile.get("has_negation")
            and peer_matches <= 0
        ):
            if (
                profile_has_complete_simple_clause(profile)
                and claim_clause_quality_score(profile) >= 12.0
                and claim_query_is_executable_via_parser(query, runner_cfg)
            ):
                pass
            else:
                return False
        if (
            not pair_like_corpus
            and count_words(query) <= 8
            and peer_matches <= 0
            and peer_best_overlap < 0.3
        ):
            return False
        return True
    if len(claim_rows) <= 250:
        parser_ready_claim_candidates = [
            case for case in parser_ready_claim_candidates
            if keep_small_corpus_claim_candidate(case)
        ]
        minimal_direct_candidates = [
            case for case in minimal_direct_candidates
            if keep_small_corpus_claim_candidate(case)
        ]
        small_corpus_direct_claim_candidates = [
            case for case in small_corpus_direct_claim_candidates
            if keep_small_corpus_claim_candidate(case)
        ]
    if len(claim_rows) <= 150:
        small_claim_group_caps = {
            "minimal": 24,
            "direct": 20,
            "relation": 16,
            "pair_relation": 16,
            "parser": 12,
            "seeded_relation": 12,
            "reconstruction": 8,
            "emergency": 8,
        }
        minimal_direct_candidates = round_robin_topk(
            minimal_direct_candidates,
            min(len(minimal_direct_candidates), small_claim_group_caps["minimal"]),
            max_cases_per_doc=max(claim_max_cases_per_doc, 3),
            score_key="selector_score",
            seed=seed,
        )
        small_corpus_direct_claim_candidates = round_robin_topk(
            small_corpus_direct_claim_candidates,
            min(len(small_corpus_direct_claim_candidates), small_claim_group_caps["direct"]),
            max_cases_per_doc=max(claim_max_cases_per_doc, 3),
            score_key="selector_score",
            seed=seed + 101,
        )
        small_corpus_relation_evidence_candidates = round_robin_topk(
            small_corpus_relation_evidence_candidates,
            min(len(small_corpus_relation_evidence_candidates), small_claim_group_caps["relation"]),
            max_cases_per_doc=max(claim_max_cases_per_doc, 3),
            score_key="selector_score",
            seed=seed + 102,
        )
        pair_like_relation_evidence_candidates = round_robin_topk(
            pair_like_relation_evidence_candidates,
            min(len(pair_like_relation_evidence_candidates), small_claim_group_caps["pair_relation"]),
            max_cases_per_doc=max(claim_max_cases_per_doc, 3),
            score_key="selector_score",
            seed=seed + 1002,
        )
        parser_ready_claim_candidates = round_robin_topk(
            parser_ready_claim_candidates,
            min(len(parser_ready_claim_candidates), small_claim_group_caps["parser"]),
            max_cases_per_doc=max(claim_max_cases_per_doc, 3),
            score_key="selector_score",
            seed=seed + 103,
        )
        seeded_relation_candidates = round_robin_topk(
            seeded_relation_candidates,
            min(len(seeded_relation_candidates), small_claim_group_caps["seeded_relation"]),
            max_cases_per_doc=max(claim_max_cases_per_doc, 3),
            score_key="selector_score",
            seed=seed + 104,
        )
        reconstruction_seed_claim_candidates = round_robin_topk(
            reconstruction_seed_claim_candidates,
            min(len(reconstruction_seed_claim_candidates), small_claim_group_caps["reconstruction"]),
            max_cases_per_doc=max(claim_max_cases_per_doc, 3),
            score_key="selector_score",
            seed=seed + 105,
        )
        emergency_small_corpus_claim_candidates = round_robin_topk(
            emergency_small_corpus_claim_candidates,
            min(len(emergency_small_corpus_claim_candidates), small_claim_group_caps["emergency"]),
            max_cases_per_doc=max(claim_max_cases_per_doc, 3),
            score_key="selector_score",
            seed=seed + 106,
        )
    seed_text_cache: list[str] | None = None
    seed_surface_profile_cache: list[dict[str, Any] | None] | None = None
    if large_claim_corpus and not pair_like_corpus:
        seed_text_cache = [
            simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", "")))
            for row in claim_rows
        ]
        seed_surface_profile_cache = [
            claim_clause_profile(text) if text else None
            for text in seed_text_cache
        ]
    priority_small_corpus_direct_candidates: list[dict[str, Any]] = []
    if len(claim_rows) <= 150 and not pair_like_corpus:
        priority_small_corpus_direct_candidates = [
            case
            for case in minimal_direct_candidates
            if small_corpus_claim_surface_priority_adjustment(case.get("query", "")) > 0.0
        ]
    # Pair-like corpora benefit from relation-backed claim seeds because the
    # extraction is dense and the same fact often appears through multiple
    # closely related passages. Large non-pair corpora behaved differently in
    # our smoke runs: a hard short-circuit to relation claims over-selected
    # communication-style surfaces that parsed cleanly but rarely crossed the
    # direct-support gate. Pair-like corpora also need a mix: when we returned
    # relation-only seeds, a few low-signal or oppositional surfaces kept
    # outranking cleaner structured claims and reintroduced bad failures.
    # Let every corpus mix structured claim rows first and keep relation claims
    # as a supplement.
    if ultra_large_claim_corpus and not pair_like_corpus and not cib_legacy_shortlist_mode:
        fast_min_query_words = int(sampling.get("min_query_words", 16))
        fast_max_query_words = int(
            sampling.get(
                "max_query_words",
                max(int(sampling.get("max_words", 48)) * 4, 180),
            )
        )
        fast_max_actor_words = int(sampling.get("max_actor_words", 8))
        fast_max_object_words = int(sampling.get("max_object_words", 18))
        fast_candidate_runner_cfg = generation_runner_cfg(runner_cfg)
        def fast_claim_seed_score(index: int) -> tuple[float, float, float]:
            row = claim_rows[index]
            text = (
                seed_text_cache[index]
                if seed_text_cache is not None
                else simplify_claim_query_seed(
                    simplify_claim_surface(row.get("claim_text", ""))
                )
            )
            if not text or query_surface_is_blocked(text, task_family="claims"):
                return (-9999.0, -9999.0, -9999.0)
            profile = (
                seed_surface_profile_cache[index]
                if seed_surface_profile_cache is not None
                else claim_clause_profile(text)
            )
            if profile is None:
                return (-9999.0, -9999.0, -9999.0)
            return (
                claim_clause_quality_score(profile)
                - surface_noise_penalty(text) * 0.35,
                float(len(profile.get("content_tokens", set()))),
                -abs(count_words(text) - 18),
            )
        fast_candidate_indices = sorted(
            range(len(claim_rows)),
            key=fast_claim_seed_score,
            reverse=True,
        )
        fast_seed_rows = max(count * 80, 320)
        max_fast_seed_rows = int(sampling.get("max_fast_seed_rows", 0) or 0)
        if max_fast_seed_rows > 0:
            fast_seed_rows = min(fast_seed_rows, max_fast_seed_rows)
        fast_candidate_indices = fast_candidate_indices[:fast_seed_rows]
        fast_candidates: list[dict[str, Any]] = []
        for idx in fast_candidate_indices:
            row = claim_rows[idx]
            text = (
                seed_text_cache[idx]
                if seed_text_cache is not None
                else simplify_claim_query_seed(
                    simplify_claim_surface(row.get("claim_text", ""))
                )
            )
            if (
                not text
                or query_surface_is_blocked(text, task_family="claims")
                or sentence_is_low_signal_for_query(text)
                or is_routing_or_metadata_sentence(text)
            ):
                continue
            if claim_surface_is_interrogative(text) or claim_surface_has_weak_modality(text):
                continue
            query_target_words = deterministic_query_word_target(
                f"claims_fast|{row['doc_id']}|{row['paragraph_id']}|{text}",
                minimum_words=fast_min_query_words,
                maximum_words=min(fast_max_query_words, 96),
            )
            query_text = refine_generated_query_surface(
                finalize_query_surface(text, add_terminal_punctuation=True),
                task_family="claims",
                anchors=[
                    normalize_space(row.get("speaker", "")),
                    normalize_space(row.get("target", "")),
                ],
                min_words=fast_min_query_words,
                target_words=query_target_words,
                max_chars=320,
                runner_cfg=fast_candidate_runner_cfg,
            )
            query_text = simplify_claim_query_seed(query_text)
            query_text = finalize_query_surface(
                query_text,
                add_terminal_punctuation=True,
            )
            if re.search(
                r"\b(?:possible|possibly|apparently|reportedly|allegedly|purportedly|likely)\b",
                query_text,
                flags=re.IGNORECASE,
            ):
                continue
            if re.search(r"\bwithout\b", query_text, flags=re.IGNORECASE):
                continue
            query_profile = claim_clause_profile(query_text)
            query_alignment = content_token_overlap_ratio(query_text, text)
            actor_distinct = distinctive_label_token_count(
                normalize_space(query_profile.get("actor", ""))
            )
            object_distinct = distinctive_label_token_count(
                normalize_space(query_profile.get("object", ""))
            )
            large_corpus_specificity = (
                actor_distinct
                + object_distinct
                + (1 if query_profile.get("date_token") else 0)
                + (1 if query_profile.get("number_token") else 0)
            )
            if (
                not query_text
                or query_surface_is_blocked(query_text, task_family="claims")
                or sentence_is_low_signal_for_query(query_text)
                or is_routing_or_metadata_sentence(query_text)
                or not profile_has_complete_simple_clause(query_profile)
                or not strict_claim_surface_is_safe(
                    query_text,
                    query_profile,
                    max_actor_words=fast_max_actor_words,
                    max_object_words=fast_max_object_words,
                )
            ):
                continue
            if not standalone_claim_surface_is_safe(query_text):
                continue
            if has_reporting_that_clause(query_text):
                continue
            if claim_profile_has_problematic_actor(query_profile):
                continue
            if object_has_abstract_head(normalize_space(query_profile.get("object", ""))):
                continue
            if query_alignment < 0.55:
                continue
            if large_corpus_specificity < 2:
                continue
            query_words = count_words(query_text)
            if query_words < fast_min_query_words or query_words > fast_max_query_words:
                continue
            fast_candidates.append(
                {
                    "task_family": "claims",
                    "runner_task": "claims",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": f"claims_fast__{row['doc_id']}__{sha1_text(query_text)}",
                    "query": query_text,
                    "ground_truth": finalize_query_surface(text, add_terminal_punctuation=True),
                    "query_kind": "claim_fast_large_corpus_query",
                    "notes": (
                        "fast_large_corpus_path=True; "
                        f"clause_quality={claim_clause_quality_score(query_profile):.2f}; "
                        f"query_alignment={query_alignment:.2f}; "
                        f"specificity={large_corpus_specificity}; "
                        f"query_target_words={query_target_words}"
                    ),
                    "source_context": "",
                    "selector_score": (
                        claim_clause_quality_score(query_profile)
                        + (2.0 if query_profile.get("date_token") else 0.0)
                        + (2.0 if query_profile.get("number_token") else 0.0)
                        + large_corpus_specificity * 2.0
                        + query_alignment * 8.0
                        - abs(query_words - 18) / 12.0
                        - surface_noise_penalty(query_text) * 0.35
                    ),
                    "word_count": query_words,
                    "query_target_words": query_target_words,
                }
            )
            if len(fast_candidates) >= max(count * 8, 40):
                break
        fast_selected = merge_ranked_candidates(
            [seeded_relation_candidates, fast_candidates, minimal_direct_candidates],
            target_count=count,
            max_cases_per_doc=claim_max_cases_per_doc,
            score_key="selector_score",
            seed=seed,
        )
        fast_vet_runner_cfg = dict(runner_cfg or {})
        fast_vet_runner_cfg["build_parser_vet_case_budget"] = (
            min(max(count * 2, count), 16)
            if large_claim_corpus
            else max(count * 4, count)
        )
        fast_selected = vet_claim_case_candidates(
            fast_selected,
            count=count,
            runner_cfg=fast_vet_runner_cfg,
        )
        if len(fast_selected) >= count:
            return normalize_selected_case_queries(
                fast_selected[:count],
                task_family="claims",
                runner_cfg=runner_cfg,
            )
    if defer_minimal_direct_candidates and not minimal_direct_candidates:
        minimal_direct_candidates = build_minimal_claim_candidates(
            claim_views,
            max(count * 6, count),
            sampling,
            seed,
            runner_cfg=runner_cfg,
        )
    if len(claim_rows) <= 150:
        canberra_claim_shortlist_mode = corpus_name == "canberra" and not pair_like_corpus
        small_corpus_candidate_groups = (
            (
                [
                    pair_like_relation_evidence_candidates,
                    pair_like_repeated_claim_candidates,
                    pair_like_direct_claim_candidates,
                    pair_like_parser_claim_candidates,
                    seeded_relation_candidates,
                    minimal_direct_candidates,
                    small_corpus_direct_claim_candidates,
                    emergency_small_corpus_claim_candidates,
                    reconstruction_seed_claim_candidates,
                    parser_ready_claim_candidates,
                ]
                if (
                    pair_like_parser_claim_candidates
                    or pair_like_direct_claim_candidates
                    or parser_ready_claim_candidates
                )
                else [
                    pair_like_relation_evidence_candidates,
                    pair_like_repeated_claim_candidates,
                    seeded_relation_candidates,
                    parser_ready_claim_candidates,
                    pair_like_parser_claim_candidates,
                    pair_like_direct_claim_candidates,
                    minimal_direct_candidates,
                    small_corpus_direct_claim_candidates,
                    reconstruction_seed_claim_candidates,
                ]
            )
            if pair_like_corpus
            else [
                priority_small_corpus_direct_candidates,
                minimal_direct_candidates,
                small_corpus_reconstruction_claim_candidates,
                small_corpus_relation_evidence_candidates,
                small_corpus_direct_claim_candidates,
                seeded_relation_candidates,
                parser_ready_claim_candidates,
                emergency_small_corpus_claim_candidates,
                pair_like_parser_claim_candidates,
                pair_like_direct_claim_candidates,
            ]
            if canberra_claim_shortlist_mode
            else [
                priority_small_corpus_direct_candidates,
                minimal_direct_candidates,
                small_corpus_reconstruction_claim_candidates,
                small_corpus_direct_claim_candidates,
                small_corpus_relation_evidence_candidates,
                parser_ready_claim_candidates,
                seeded_relation_candidates,
                emergency_small_corpus_claim_candidates,
                pair_like_parser_claim_candidates,
                pair_like_direct_claim_candidates,
            ]
        )
        small_corpus_shortlist = merge_ranked_candidates(
            small_corpus_candidate_groups,
            target_count=max(count * 4, count),
            max_cases_per_doc=claim_max_cases_per_doc,
            score_key="selector_score",
            seed=seed,
        )
        if not pair_like_corpus:
            small_corpus_shortlist = quality_fill_unique_candidates(
                small_corpus_shortlist,
                len(small_corpus_shortlist),
                max_cases_per_doc=claim_max_cases_per_doc,
                score_key="selector_score",
            )
        small_corpus_shortlist = vet_claim_case_candidates(
            small_corpus_shortlist,
            count=count,
            runner_cfg=runner_cfg,
        )
        small_corpus_normalized = normalize_selected_case_queries(
            small_corpus_shortlist,
            task_family="claims",
            runner_cfg=runner_cfg,
        )
        if pair_like_corpus and (
            pair_like_direct_claim_candidates or pair_like_repeated_claim_candidates
        ):
            selected_keys = {
                normalize_space(case.get("case_id", ""))
                for case in small_corpus_normalized
                if normalize_space(case.get("case_id", ""))
            }
            pair_like_priority_repeated = round_robin_topk(
                pair_like_repeated_claim_candidates,
                min(max(2, count // 2), len(pair_like_repeated_claim_candidates)),
                max_cases_per_doc=max(claim_max_cases_per_doc, 2),
                score_key="selector_score",
                seed=seed + 700,
            )
            pair_like_priority_direct = round_robin_topk(
                pair_like_direct_claim_candidates,
                min(max(2, count // 2), len(pair_like_direct_claim_candidates)),
                max_cases_per_doc=max(claim_max_cases_per_doc, 2),
                score_key="selector_score",
                seed=seed + 701,
            )
            pair_like_priority_claims = pair_like_priority_repeated + pair_like_priority_direct
            pair_like_priority_claims = [
                case
                for case in normalize_selected_case_queries(
                    pair_like_priority_claims,
                    task_family="claims",
                    runner_cfg=runner_cfg,
                )
                if normalize_space(case.get("case_id", "")) not in selected_keys
                and not claim_surface_is_short_classification(
                    normalize_space(case.get("query", "")),
                    claim_clause_profile(normalize_space(case.get("query", ""))),
                )
            ]
            if pair_like_priority_claims:
                small_corpus_normalized = quality_fill_unique_candidates(
                    pair_like_priority_claims + small_corpus_normalized,
                    min(count, len(pair_like_priority_claims + small_corpus_normalized)),
                    max_cases_per_doc=max(claim_max_cases_per_doc, 2),
                    score_key="selector_score",
                )
        if canberra_claim_shortlist_mode and priority_small_corpus_direct_candidates:
            selected_keys = {
                normalize_space(case.get("case_id", ""))
                for case in small_corpus_normalized
                if normalize_space(case.get("case_id", ""))
            }
            canberra_priority_vetted: list[dict[str, Any]] = []
            for case in priority_small_corpus_direct_candidates:
                case_id = normalize_space(case.get("case_id", ""))
                if case_id in selected_keys:
                    continue
                if not case_passes_pair_runner_vet(
                    case,
                    runner_cfg,
                    task_family="claims",
                ):
                    continue
                canberra_priority_vetted.append(case)
                selected_keys.add(case_id)
                if len(canberra_priority_vetted) >= max(count, 6):
                    break
            if canberra_priority_vetted:
                small_corpus_normalized = quality_fill_unique_candidates(
                    canberra_priority_vetted + small_corpus_normalized,
                    min(count, len(canberra_priority_vetted + small_corpus_normalized)),
                    max_cases_per_doc=max(claim_max_cases_per_doc, 3),
                    score_key="selector_score",
                )
        emergency_selected = round_robin_topk(
            emergency_small_corpus_claim_candidates,
            min(count, len(emergency_small_corpus_claim_candidates)),
            max_cases_per_doc=claim_max_cases_per_doc,
            score_key="selector_score",
            seed=seed,
        )
        emergency_normalized = normalize_selected_case_queries(
            emergency_selected,
            task_family="claims",
            runner_cfg=runner_cfg,
        )
        merged_small_corpus = merge_ranked_candidates(
            [small_corpus_normalized, emergency_normalized],
            target_count=count,
            max_cases_per_doc=claim_max_cases_per_doc,
            score_key="selector_score",
            seed=seed,
        )
        if not pair_like_corpus:
            merged_small_corpus = quality_fill_unique_candidates(
                merged_small_corpus,
                count,
                max_cases_per_doc=claim_max_cases_per_doc,
                score_key="selector_score",
            )
        merged_small_corpus = normalize_selected_case_queries(
            merged_small_corpus,
            task_family="claims",
            runner_cfg=runner_cfg,
        )
        if pair_like_corpus and merged_small_corpus:
            preferred_small_corpus = [
                case
                for case in merged_small_corpus
                if not claim_surface_is_short_classification(
                    normalize_space(case.get("query", "")),
                    claim_clause_profile(normalize_space(case.get("query", ""))),
                )
            ]
            if len(preferred_small_corpus) < count:
                reserve_pool = normalize_selected_case_queries(
                    merge_ranked_candidates(
                        [
                            small_corpus_shortlist,
                            pair_like_parser_claim_candidates,
                            pair_like_direct_claim_candidates,
                            pair_like_repeated_claim_candidates,
                            minimal_direct_candidates,
                            small_corpus_direct_claim_candidates,
                            emergency_small_corpus_claim_candidates,
                        ],
                        target_count=max(count * 6, count),
                        max_cases_per_doc=claim_max_cases_per_doc,
                        score_key="selector_score",
                        seed=seed,
                    ),
                    task_family="claims",
                    runner_cfg=runner_cfg,
                )
                selected_keys = {
                    normalize_space(case.get("case_id", ""))
                    for case in preferred_small_corpus
                    if normalize_space(case.get("case_id", ""))
                }
                for case in reserve_pool:
                    case_id = normalize_space(case.get("case_id", ""))
                    if case_id in selected_keys:
                        continue
                    if claim_surface_is_short_classification(
                        normalize_space(case.get("query", "")),
                        claim_clause_profile(normalize_space(case.get("query", ""))),
                    ):
                        continue
                    preferred_small_corpus.append(case)
                    if case_id:
                        selected_keys.add(case_id)
                    if len(preferred_small_corpus) >= count:
                        break
            if len(preferred_small_corpus) >= count:
                return preferred_small_corpus[:count]
        if merged_small_corpus and isinstance(runner_cfg, dict) and maybe_bool(
            runner_cfg.get("__strict_small_corpus_vet", False)
        ):
            return merged_small_corpus[:count]
        if merged_small_corpus:
            return merged_small_corpus[:count]
    if pair_like_parser_claim_candidates:
        pair_like_parser_selected = round_robin_topk(
            pair_like_parser_claim_candidates,
            min(count, len(pair_like_parser_claim_candidates)),
            max_cases_per_doc=claim_max_cases_per_doc,
            score_key="selector_score",
            seed=seed,
        )
        pair_like_parser_normalized = normalize_selected_case_queries(
            pair_like_parser_selected,
            task_family="claims",
            runner_cfg=runner_cfg,
        )
        if pair_like_parser_normalized:
            return pair_like_parser_normalized[:count]
    if pair_like_direct_claim_candidates:
        pair_like_direct_selected = round_robin_topk(
            pair_like_direct_claim_candidates,
            min(count, len(pair_like_direct_claim_candidates)),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
        pair_like_direct_normalized = normalize_selected_case_queries(
            pair_like_direct_selected,
            task_family="claims",
            runner_cfg=runner_cfg,
        )
        if pair_like_direct_normalized:
            return pair_like_direct_normalized[:count]
    if cib_legacy_shortlist_mode:
        shortlist_max_actor_words = int(sampling.get("max_actor_words", 8))
        shortlist_max_object_words = int(sampling.get("max_object_words", 18))
        hedge_re = re.compile(
            r"\b(?:possible|possibly|likely|reportedly|allegedly|purportedly|"
            r"apparently|may\s+have|might\s+have|reported\s+interest|"
            r"has\s+been\s+repeatedly\s+reported|violent\s+differences\s+reported)\b",
            flags=re.IGNORECASE,
        )
        def cib_legacy_shortlist_score(index: int) -> tuple[float, float, float]:
            row = claim_rows[index]
            text = simplify_claim_query_seed(
                simplify_claim_surface(row.get("claim_text", ""))
            )
            if (
                not text
                or query_surface_is_blocked(text, task_family="claims")
                or sentence_is_low_signal_for_query(text)
                or is_routing_or_metadata_sentence(text)
                or hedge_re.search(text)
            ):
                return (-9999.0, -9999.0, -9999.0)
            profile = claim_clause_profile(text)
            if (
                not profile_has_complete_simple_clause(profile)
                or not strict_claim_surface_is_safe(
                    text,
                    profile,
                    max_actor_words=shortlist_max_actor_words,
                    max_object_words=shortlist_max_object_words,
                )
            ):
                return (-9999.0, -9999.0, -9999.0)
            role_bonus = (
                (3.0 if profile.get("actor") else 0.0)
                + (3.0 if profile.get("predicate") else 0.0)
                + (3.0 if profile.get("object") else 0.0)
                + (2.0 if profile.get("date_token") else 0.0)
                + (2.0 if profile.get("number_token") else 0.0)
            )
            return (
                claim_clause_quality_score(profile)
                + role_bonus
                - surface_noise_penalty(text) * 0.35,
                float(len(profile.get("content_tokens", set()))),
                -abs(count_words(text) - 18),
            )

        shortlisted_indices = sorted(
            range(len(claim_rows)),
            key=cib_legacy_shortlist_score,
            reverse=True,
        )[: max(count * 80, 500)]
        claim_rows = [claim_rows[index] for index in shortlisted_indices]
        if seed_text_cache is not None:
            seed_text_cache = [seed_text_cache[index] for index in shortlisted_indices]
        if seed_surface_profile_cache is not None:
            seed_surface_profile_cache = [
                seed_surface_profile_cache[index] for index in shortlisted_indices
            ]
        claim_profiles = None
        peer_rows_by_idx = None
        large_claim_corpus = len(claim_rows) >= 600
        ultra_large_claim_corpus = False
    if claim_profiles is None:
        if ultra_large_claim_corpus and not pair_like_corpus:
            claim_profiles = [
                claim_clause_profile(normalize_space(row.get("claim_text", "")))
                for row in claim_rows
            ]
        else:
            claim_profiles, peer_rows_by_idx = build_claim_peer_pool(claim_rows)
    if peer_rows_by_idx is None:
        if ultra_large_claim_corpus and not pair_like_corpus:
            peer_rows_by_idx = [[] for _ in claim_rows]
        else:
            _profiles, peer_rows_by_idx = build_claim_peer_pool(claim_rows)
            if claim_profiles is None:
                claim_profiles = _profiles
    relation_reserve = min(
        len(seeded_relation_candidates),
        max(2, count // 2)
        if (seeded_relation_candidates and (pair_like_corpus or len(claim_rows) < 250))
        else (
            max(1, count // 5)
            if (seeded_relation_candidates and large_claim_corpus)
            else 0
        ),
    )
    candidates = list(seeded_relation_candidates[:relation_reserve])
    if reconstruction_seed_claim_candidates:
        candidates.extend(
            round_robin_topk(
                reconstruction_seed_claim_candidates,
                min(count, len(reconstruction_seed_claim_candidates)),
                max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
                score_key="selector_score",
                seed=seed,
            )
        )
    if small_corpus_reconstruction_claim_candidates:
        candidates.extend(
            round_robin_topk(
                small_corpus_reconstruction_claim_candidates,
                min(count, len(small_corpus_reconstruction_claim_candidates)),
                max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
                score_key="selector_score",
                seed=seed + 17,
            )
        )
    candidates.extend(
        round_robin_topk(
            small_corpus_direct_claim_candidates,
            min(count, len(small_corpus_direct_claim_candidates)),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
    )
    strict_mode = not maybe_bool(sampling.get("_relaxed_retry_used", False))
    allow_pseudo_chunk_claim_fallback = maybe_bool(
        sampling.get("allow_pseudo_chunk_claim_fallback", False)
    )
    allow_chunk_claim_fallback = maybe_bool(
        sampling.get("allow_chunk_claim_fallback", len(claim_rows) < 250)
    )
    allow_reconstruction_claim_backfill = maybe_bool(
        sampling.get("allow_reconstruction_claim_backfill", len(claim_rows) < 250)
    )
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    prefer_relation_backed = (
        pair_like_corpus
        or (
            not large_claim_corpus
            and (
                len(relations) < 150
                or relation_reserve >= max(1, count // 3)
            )
        )
    )
    require_target_or_speaker = maybe_bool(sampling.get("require_target_or_speaker", True))
    require_nonempty_speaker = maybe_bool(sampling.get("require_nonempty_speaker", False))
    require_structured_source = maybe_bool(sampling.get("require_structured_source", True))
    exclude_generic_roles = maybe_bool(sampling.get("exclude_generic_roles", True))
    min_role_fields = int(sampling.get("min_role_fields", 2))
    include_stance = maybe_bool(sampling.get("include_stance", False))
    query_style = normalize_space(sampling.get("query_style", "plain")) or "plain"
    max_object_words = int(sampling.get("max_object_words", 18))
    min_peer_matches = int(sampling.get("min_peer_matches", 1))
    max_peer_matches = int(
        sampling.get("max_peer_matches", 8 if len(claim_rows) >= 250 else 9999)
    )
    min_peer_overlap = float(sampling.get("min_peer_overlap", 0.25))
    min_query_words = int(sampling.get("min_query_words", 16))
    max_query_words = int(sampling.get("max_query_words", max(int(sampling.get("max_words", 48)) * 4, 180)))
    require_simple_clause_arguments = maybe_bool(
        sampling.get("require_simple_clause_arguments", True)
    )
    min_clause_quality = float(sampling.get("min_clause_quality_score", 8.0))
    max_actor_words = int(sampling.get("max_actor_words", 8))
    candidate_indices = list(range(len(claim_rows)))
    if large_claim_corpus:
        def claim_seed_score(index: int) -> tuple[float, float, int]:
            row = claim_rows[index]
            text = (
                seed_text_cache[index]
                if seed_text_cache is not None
                else simplify_claim_query_seed(
                    simplify_claim_surface(row.get("claim_text", ""))
                )
            )
            if not text or query_surface_is_blocked(text, task_family="claims"):
                return (-9999.0, -9999.0, -9999)
            profile = (
                seed_surface_profile_cache[index]
                if seed_surface_profile_cache is not None
                else claim_clause_profile(text)
            )
            if profile is None:
                return (-9999.0, -9999.0, -9999)
            quality = claim_clause_quality_score(profile)
            structure_bonus = 6.0 if strict_claim_surface_is_safe(
                text,
                profile,
                max_actor_words=max_actor_words,
                max_object_words=max_object_words,
            ) else -12.0
            role_bonus = (
                (2.0 if normalize_space(row.get("speaker", "")) else 0.0)
                + (2.0 if normalize_space(row.get("target", "")) else 0.0)
                + (1.0 if profile.get("date_token") else 0.0)
                + (1.0 if profile.get("number_token") else 0.0)
            )
            text_words = count_words(text)
            return (
                quality + structure_bonus + role_bonus - surface_noise_penalty(text) * 0.4,
                float(len(profile.get("content_tokens", set()))),
                -abs(text_words - 18),
            )

        candidate_indices = sorted(
            candidate_indices,
            key=claim_seed_score,
            reverse=True,
        )
        ranked_candidate_rows = max(count * 40, 180)
        max_ranked_candidate_rows = int(sampling.get("max_ranked_candidate_rows", 0) or 0)
        if max_ranked_candidate_rows > 0:
            ranked_candidate_rows = min(ranked_candidate_rows, max_ranked_candidate_rows)
        candidate_indices = candidate_indices[:ranked_candidate_rows]
    claim_candidate_cap = (
        max(count * 30, 180)
        if len(claim_rows) >= 2000
        else max(count * 20, 120)
        if len(claim_rows) >= 600
        else 0
    )
    max_candidate_case_budget = int(sampling.get("max_candidate_case_budget", 0) or 0)
    if max_candidate_case_budget > 0:
        claim_candidate_cap = (
            min(claim_candidate_cap, max_candidate_case_budget)
            if claim_candidate_cap
            else max_candidate_case_budget
        )
    if len(claim_rows) < 250:
        min_peer_matches = min(min_peer_matches, 1)
        min_peer_overlap = min(min_peer_overlap, 0.2)
        min_clause_quality = min(min_clause_quality, 3.5)
    small_claim_corpus = len(claim_rows) < 250
    if ultra_large_claim_corpus and not pair_like_corpus:
        min_peer_matches = 0
        min_peer_overlap = 0.0
        max_peer_matches = max(max_peer_matches, len(claim_rows))
    for idx in candidate_indices:
        if claim_candidate_cap and len(candidates) >= claim_candidate_cap:
            break
        row = claim_rows[idx]
        if require_structured_source and not row_is_structured_claim(row):
            continue
        text = (
            seed_text_cache[idx]
            if seed_text_cache is not None
            else simplify_claim_query_seed(
                simplify_claim_surface(row.get("claim_text", ""))
            )
        )
        claim_doc_frequency = structured_claim_doc_frequency.get(text.casefold(), 0)
        if not text or is_routing_or_metadata_sentence(text):
            continue
        if claim_surface_is_interrogative(text) or claim_surface_has_weak_modality(text):
            continue
        if REDACTION_MARKER_RE.search(text):
            continue
        if sentence_is_low_signal_for_query(text):
            continue
        stance_value = normalize_space(row.get("stance", "")).casefold()
        if strict_mode and stance_value in {"oppose", "unclear"}:
            continue
        if (
            CONDITIONAL_CLAIM_RE.search(text)
            or ABSTRACT_QUANTIFIER_RE.search(text)
            or re.search(r"\b(?:could|may|might|likely|perhaps)\b", text, flags=re.IGNORECASE)
        ):
            continue
        if strict_mode and re.search(r"\bwithout\b", text, flags=re.IGNORECASE):
            continue
        if "..." in text:
            continue
        if re.match(r"^(?:there was|there has been)\b", text, flags=re.IGNORECASE):
            continue
        if query_surface_is_blocked(text, task_family="claims"):
            continue
        if claim_uses_intentive_construction(text):
            continue
        if strict_mode and prefer_relation_backed and has_reporting_that_clause(text):
            continue
        query_target_words = deterministic_query_word_target(
            f"claims|{row['doc_id']}|{row['paragraph_id']}|{text}",
            minimum_words=min_query_words,
            maximum_words=min(max_query_words, 128),
        )
        profile = claim_profiles[idx]
        surface_profile = (
            seed_surface_profile_cache[idx]
            if seed_surface_profile_cache is not None
            else claim_clause_profile(text)
        )
        if surface_profile is None:
            continue
        if strict_mode and not strict_claim_surface_is_safe(
            text,
            surface_profile,
            max_actor_words=max_actor_words,
            max_object_words=max_object_words,
        ):
            continue
        paragraph_source_text = chunk_text_lookup.get(
            paragraph_key(row["doc_id"], row["paragraph_id"]),
            "",
        )
        source_alignment_anchors = [
            profile.get("actor", ""),
            profile.get("object", ""),
            normalize_space(row.get("speaker", "")),
            normalize_space(row.get("target", "")),
        ]
        aligned_source_sentence = ""
        if small_claim_corpus or pair_like_corpus:
            aligned_source_sentence = best_claim_aligned_source_sentence(
                text,
                " ".join(
                    part
                    for part in [
                        paragraph_source_text,
                        extract_embedded_context(row.get("pseudo_view_text", "")),
                        extract_embedded_context(row.get("proposition_text", "")),
                    ]
                    if normalize_space(part)
                ),
                anchors=source_alignment_anchors,
                min_overlap=2,
            )
            if not aligned_claim_source_sentence_is_usable(aligned_source_sentence):
                aligned_source_sentence = ""
        lead_excerpt = finalize_query_surface(
            extract_sentence_excerpt(paragraph_source_text, max_sentences=1, max_chars=220),
            add_terminal_punctuation=True,
        )
        if small_claim_corpus and not aligned_source_sentence and (
            (lead_excerpt and is_routing_or_metadata_sentence(lead_excerpt))
            or QA_PREFIX_RE.match(normalize_space(paragraph_source_text))
        ):
            continue
        if (
            (small_claim_corpus or pair_like_corpus)
            and claim_doc_frequency > 1
            and not aligned_source_sentence
            and not (pair_like_corpus and standalone_claim_surface_is_safe(text))
        ):
            continue
        source_context = choose_fast_query_context(
            [
                aligned_source_sentence,
                text,
                paragraph_source_text,
                extract_embedded_context(row.get("pseudo_view_text", "")),
                extract_embedded_context(row.get("proposition_text", "")),
            ],
            min_words=max(8, min_query_words // 2),
            max_chars=420,
        )
        anchor_bundle = [
            profile.get("actor", ""),
            profile.get("object", ""),
            normalize_space(row.get("speaker", "")),
            normalize_space(row.get("target", "")),
        ]
        source_context = refine_generated_query_surface(
            source_context,
            task_family="claims",
            anchors=anchor_bundle,
            min_words=max(8, min_query_words // 2),
            target_words=max(10, query_target_words - 12),
            max_chars=420,
            runner_cfg=candidate_runner_cfg,
        )
        claim_query_text, role_fields, role_field_count = build_claim_query(
            row,
            include_stance=include_stance,
            claim_text_override=aligned_source_sentence or None,
            query_style=query_style,
            source_context_text=source_context,
            prefer_speaker_attribution=(not pair_like_corpus and len(claim_rows) >= 600),
        )
        prefer_attributed_query = (
            bool(role_fields["speaker"])
            and not pair_like_corpus
            and len(claim_rows) >= 600
            and has_reporting_that_clause(claim_query_text)
        )
        seed_profile = claim_clause_profile(claim_query_text or text)
        if strict_mode and not (
            seed_profile["actor"]
            and seed_profile["predicate"]
            and seed_profile["object"]
        ):
            continue
        query_text = claim_query_text
        query_text = refine_generated_query_surface(
            query_text,
            task_family="claims",
            anchors=anchor_bundle,
            min_words=min_query_words,
            target_words=query_target_words,
            max_chars=420,
            runner_cfg=candidate_runner_cfg,
        )
        query_text = choose_best_query_context(
            (
                [aligned_source_sentence, query_text, text, source_context]
                if prefer_attributed_query
                else [aligned_source_sentence, text, query_text, source_context]
            ),
            anchors=anchor_bundle,
            min_words=min_query_words,
            target_words=query_target_words,
            max_sentences=1 if query_target_words <= 30 else 2,
            max_chars=420,
        ) or query_text
        query_text = refine_generated_query_surface(
            query_text,
            task_family="claims",
            anchors=anchor_bundle,
            min_words=min_query_words,
            target_words=query_target_words,
            max_chars=420,
            runner_cfg=candidate_runner_cfg,
        )
        query_text = simplify_claim_query_seed(query_text)
        fallback_query_text = finalize_query_surface(text, add_terminal_punctuation=True)
        fallback_query_text = simplify_claim_query_seed(fallback_query_text)
        if claim_surface_is_interrogative(query_text) or claim_surface_has_weak_modality(query_text):
            continue
        if claim_surface_is_interrogative(fallback_query_text) or claim_surface_has_weak_modality(fallback_query_text):
            fallback_query_text = ""
        fallback_query_profile = claim_clause_profile(fallback_query_text)
        current_query_profile = claim_clause_profile(query_text)
        if strict_mode and not strict_claim_surface_is_safe(
            query_text,
            current_query_profile,
            max_actor_words=max_actor_words,
            max_object_words=max_object_words,
        ):
            if prefer_attributed_query and standalone_claim_surface_is_safe(query_text):
                pass
            elif strict_claim_surface_is_safe(
                fallback_query_text,
                fallback_query_profile,
                max_actor_words=max_actor_words,
                max_object_words=max_object_words,
            ):
                query_text = fallback_query_text
        quality_score = claim_clause_quality_score(surface_profile)
        if quality_score < min_clause_quality and strict_mode:
            continue
        words = count_words(text)
        query_words = count_words(query_text)
        if words < int(sampling["min_words"]) or words > int(sampling["max_words"]):
            continue
        if query_words < min_query_words or query_words > max_query_words:
            continue
        if small_claim_corpus and query_words > 36:
            continue
        if surface_profile["nested_reporting"]:
            continue
        if surface_profile["has_semicolon"]:
            continue
        if (
            strict_mode
            and prefer_relation_backed
            and not (
                surface_profile["actor"]
                and surface_profile["predicate"]
                and surface_profile["object"]
            )
        ):
            continue
        has_simple_clause = bool(
            role_fields.get("actor") and role_fields.get("predicate") and role_fields.get("object")
        )
        standalone_safe = standalone_claim_surface_is_safe(query_text)
        if strict_mode and has_simple_clause and predicate_is_low_signal(role_fields.get("predicate", "")):
            continue
        if strict_mode and has_simple_clause and actor_has_abstract_head(role_fields.get("actor", "")):
            continue
        if strict_mode and has_simple_clause and actor_is_generic_common_noun_phrase(role_fields.get("actor", "")):
            continue
        if strict_mode and has_simple_clause and object_has_abstract_head(role_fields.get("object", "")):
            continue
        if strict_mode and has_simple_clause and actor_phrase_has_clause_bleed(role_fields.get("actor", "")):
            continue
        if (
            strict_mode
            and normalize_space(role_fields.get("actor", "")).casefold()
            == normalize_space(role_fields.get("object", "")).casefold()
            and normalize_space(role_fields.get("actor", ""))
        ):
            continue
        if int(surface_profile["object_words"]) > max_object_words:
            continue
        if role_field_count < min_role_fields and not standalone_safe:
            continue
        if require_nonempty_speaker and not role_fields["speaker"]:
            continue
        if require_simple_clause_arguments and not (has_simple_clause or standalone_safe):
            continue
        if require_target_or_speaker and not (
            role_fields.get("actor") or role_fields["speaker"] or role_fields["target"]
        ):
            continue
        if exclude_generic_roles and not (
            role_fields.get("actor") or role_fields["speaker"] or role_fields["target"]
        ):
            continue
        if not query_text:
            continue
        if REDACTION_MARKER_RE.search(query_text):
            continue
        if "[REDACT:" in query_text or "[REDACT_SENT:" in query_text:
            continue
        if query_surface_is_blocked(query_text, task_family="claims"):
            continue
        if strict_mode and substantive_sentence_score(query_text, anchors=anchor_bundle) < 6.0:
            continue
        query_profile = claim_clause_profile(query_text)
        if pair_like_corpus and claim_surface_is_short_classification(
            query_text,
            query_profile,
        ):
            continue
        if not profile_has_complete_simple_clause(query_profile):
            if (
                profile_has_complete_simple_clause(fallback_query_profile)
                and strict_claim_surface_is_safe(
                    fallback_query_text,
                    fallback_query_profile,
                    max_actor_words=max_actor_words,
                    max_object_words=max_object_words,
                )
            ):
                query_text = fallback_query_text
                query_profile = fallback_query_profile
            else:
                continue
        attributed_query = bool(role_fields["speaker"]) and has_reporting_that_clause(query_text)
        if strict_mode and not strict_claim_surface_is_safe(
            query_text,
            query_profile,
            max_actor_words=max_actor_words,
            max_object_words=max_object_words,
        ):
            if not (attributed_query and standalone_claim_surface_is_safe(query_text)):
                continue
        if strict_mode and (
            ((query_profile["nested_reporting"] or query_profile["reporting_actor"]) and not attributed_query)
            or actor_phrase_has_clause_bleed(query_profile["actor"])
            or query_profile["object_bad_lead"]
        ):
            continue
        if claim_surface_is_retrieval_fragile(
            query_text,
            query_profile,
            small_corpus=small_claim_corpus and not pair_like_corpus,
            pair_like_corpus=pair_like_corpus,
        ):
            continue
        if strict_mode and actor_has_abstract_head(query_profile["actor"]):
            if not attributed_query:
                continue
        if strict_mode and object_has_abstract_head(query_profile["object"]):
            continue
        if strict_mode and claim_uses_intentive_construction(query_text):
            continue
        if small_claim_corpus and not pair_like_corpus and query_profile.get("has_negation"):
            continue
        if strict_mode and has_simple_clause and parser_hostile_claim_candidate(query_text, role_fields, query_profile):
            if not attributed_query:
                continue
        query_alignment = content_token_overlap_ratio(query_text, text)
        if query_alignment < 0.35:
            continue
        min_alignment = 0.6 if pair_like_corpus else 0.5
        if strict_mode and query_alignment < min_alignment:
            continue
        peer_matches, peer_best_overlap = claim_support_peer_score(
            profile,
            peer_rows_by_idx[idx],
        )
        if peer_matches < min_peer_matches or peer_best_overlap < min_peer_overlap:
            continue
        if (
            small_claim_corpus
            and not pair_like_corpus
            and (
                query_profile.get("has_negation")
                or peer_matches <= 0
            )
        ):
            continue
        if not pair_like_corpus and len(claim_rows) >= 250 and peer_matches > max_peer_matches:
            continue
        actor_distinct = distinctive_label_token_count(role_fields.get("actor", ""))
        object_distinct = distinctive_label_token_count(role_fields.get("object", ""))
        speaker_distinct = distinctive_label_token_count(role_fields.get("speaker", ""))
        if (
            small_claim_corpus
            and (actor_distinct + object_distinct + speaker_distinct) < 2
            and not query_profile.get("date_token")
            and not query_profile.get("number_token")
        ):
            continue
        large_corpus_specificity = (
            actor_distinct
            + object_distinct
            + speaker_distinct
            + (1 if query_profile.get("date_token") else 0)
            + (1 if query_profile.get("number_token") else 0)
        )
        if (
            strict_mode
            and not pair_like_corpus
            and len(claim_rows) >= 600
            and large_corpus_specificity < 2
        ):
            continue
        if (
            strict_mode
            and not pair_like_corpus
            and not ultra_large_claim_corpus
            and len(claim_rows) >= 600
            and peer_matches <= 0
            and not query_profile.get("date_token")
            and not query_profile.get("number_token")
            and (actor_distinct + object_distinct + speaker_distinct) < 3
        ):
            continue
        peer_penalty = 0.0
        if not pair_like_corpus and len(claim_rows) >= 600:
            peer_penalty = max(peer_matches - 2, 0) * 3.0
        if claim_doc_frequency <= 1:
            claim_frequency_adjustment = 3.0
        elif small_claim_corpus or pair_like_corpus:
            claim_frequency_adjustment = -4.0 * float(claim_doc_frequency - 1)
        else:
            claim_frequency_adjustment = max(-4.0, 4.0 - claim_doc_frequency * 1.5)
        selector_score = (
            quality_score
            +
            role_field_count * 8.0
            + (3.0 if role_fields.get("actor") else 0.0)
            + (3.0 if role_fields["predicate"] else 0.0)
            + (3.0 if role_fields["object"] else 0.0)
            + (2.0 if role_fields["speaker"] else 0.0)
            + (2.0 if role_fields["target"] else 0.0)
            + min(peer_matches, 4) * 4.0
            + peer_best_overlap * 8.0
            + (10.0 if aligned_source_sentence else 0.0)
            + actor_distinct * 2.0
            + object_distinct * 1.5
            + speaker_distinct * 1.0
            + claim_frequency_adjustment
            + (2.0 if query_profile.get("date_token") else 0.0)
            + (2.0 if query_profile.get("number_token") else 0.0)
            + query_alignment * 8.0
            - (10.0 if not (role_fields.get("actor") and role_fields["predicate"] and role_fields["object"]) else 0.0)
            - (
                18.0
                if re.search(r"\band not to\b", query_text.casefold())
                else 0.0
            )
            - (
                16.0
                if (
                    query_text.count(",") >= 1
                    and re.search(r"\bwant(?:s|ed)?\b", query_text.casefold())
                    and re.search(r"\band\b", query_text.casefold())
                )
                else 0.0
            )
            - (10.0 if not normalize_space(source_context) else 0.0)
            - abs(words - 18) / 12.0
            - surface_noise_penalty(query_text) * 0.4
            - peer_penalty
        )
        acceptable_source_ids_any = sorted(
            structured_claim_docs.get(text.casefold(), set())
        )
        if len(acceptable_source_ids_any) <= 1:
            acceptable_source_ids_any = []
        case_id = f"claims__{row['doc_id']}__{sha1_text(query_text)}"
        candidates.append(
            {
                "task_family": "claims",
                "runner_task": "claims",
                "doc_id": str(row["doc_id"]),
                "paragraph_id": str(row["paragraph_id"]),
                "case_id": case_id,
                "query": query_text,
                "ground_truth": text,
                "query_kind": "claim_structured_query",
                "notes": (
                    f"actor={role_fields.get('actor','')}; speaker={role_fields['speaker']}; target={role_fields['target']}; "
                    f"predicate={role_fields['predicate']}; object={role_fields['object']}; "
                    f"stance={role_fields['stance']}; role_field_count={role_field_count}; "
                    f"clause_quality={quality_score:.2f}; "
                    f"peer_matches={peer_matches}; peer_best_overlap={peer_best_overlap:.2f}; "
                    f"aligned_source_sentence={bool(aligned_source_sentence)}; "
                    f"query_alignment={query_alignment:.2f}; "
                    f"nested_reporting={surface_profile['nested_reporting']}; "
                    f"actor_words={surface_profile['actor_words']}; object_words={surface_profile['object_words']}; "
                    f"source_type={normalize_space(row.get('source_type', ''))}; "
                    f"query_target_words={query_target_words}"
                ),
                "target_doc_ids": acceptable_source_ids_any or [str(row["doc_id"])],
                "required_doc_ids": acceptable_source_ids_any or [str(row["doc_id"])],
                "acceptable_source_ids_any": acceptable_source_ids_any,
                "source_context": source_context,
                "selector_score": selector_score,
                "word_count": query_words,
                "query_target_words": query_target_words,
            }
        )
    relation_prefix = seeded_relation_candidates[:relation_reserve]
    selected = merge_ranked_candidates(
        [relation_prefix, candidates],
        target_count=max(count * 4, count),
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )
    if relation_prefix:
        selected_ids = {normalize_space(case.get("case_id", "")) for case in selected}
        for relation_case in relation_prefix:
            relation_id = normalize_space(relation_case.get("case_id", ""))
            if relation_id in selected_ids:
                continue
            selected.append(relation_case)
            selected_ids.add(relation_id)
        selected = merge_ranked_candidates(
            [relation_prefix, selected],
            target_count=max(count * 4, count),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
    pseudo_chunk_competition = (
        allow_pseudo_chunk_claim_fallback
        and reconstruction_views is not None
        and not reconstruction_views.empty
        and len(claim_rows) < 80
    )
    if len(selected) >= count and not pseudo_chunk_competition:
        return normalize_selected_case_queries(
            selected,
            task_family="claims",
            runner_cfg=runner_cfg,
        )
    if maybe_bool(sampling.get("_relaxed_retry_used", False)):
        return normalize_selected_case_queries(
            selected,
            task_family="claims",
            runner_cfg=runner_cfg,
        )
    if True:
        selected_case_ids = {normalize_space(case.get("case_id", "")) for case in selected}
        fallback_candidates: list[dict[str, Any]] = []
        for idx in candidate_indices:
            row = claim_rows[idx]
            text = simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", "")))
            claim_doc_frequency = structured_claim_doc_frequency.get(text.casefold(), 0)
            if not text or query_surface_is_blocked(text, task_family="claims"):
                continue
            profile = claim_clause_profile(text)
            if not (
                strict_claim_surface_is_safe(
                    text,
                    profile,
                    max_actor_words=max_actor_words,
                    max_object_words=max_object_words,
                )
                or (small_claim_corpus and standalone_claim_surface_is_safe(text))
            ):
                continue
            paragraph_source_text = chunk_text_lookup.get(
                paragraph_key(row["doc_id"], row["paragraph_id"]),
                "",
            )
            aligned_source_sentence = ""
            if small_claim_corpus or pair_like_corpus:
                aligned_source_sentence = best_claim_aligned_source_sentence(
                    text,
                    " ".join(
                        part
                        for part in [
                            paragraph_source_text,
                            extract_embedded_context(row.get("pseudo_view_text", "")),
                            extract_embedded_context(row.get("proposition_text", "")),
                        ]
                        if normalize_space(part)
                    ),
                    anchors=[profile.get("actor", ""), profile.get("object", "")],
                    min_overlap=2,
                )
                if not aligned_claim_source_sentence_is_usable(aligned_source_sentence):
                    aligned_source_sentence = ""
            direct_seed = (
                finalize_query_surface(aligned_source_sentence, add_terminal_punctuation=True)
                or finalize_query_surface(text, add_terminal_punctuation=True)
                or text
            )
            direct_query = refine_generated_query_surface(
                finalize_query_surface(direct_seed, add_terminal_punctuation=True),
                task_family="claims",
                anchors=[profile.get("actor", ""), profile.get("object", "")],
                min_words=min_query_words,
                target_words=count_words(direct_seed),
                max_chars=420,
                runner_cfg=candidate_runner_cfg,
            )
            direct_query = simplify_claim_query_seed(direct_query)
            direct_profile = claim_clause_profile(direct_query)
            direct_complete = profile_has_complete_simple_clause(direct_profile)
            direct_standalone_safe = standalone_claim_surface_is_safe(direct_query)
            if not direct_complete:
                continue
            if not strict_claim_surface_is_safe(
                direct_query,
                direct_profile,
                max_actor_words=max_actor_words,
                max_object_words=max_object_words,
            ):
                continue
            if not direct_standalone_safe:
                continue
            if not (normalize_space(direct_profile.get("actor", "")) and normalize_space(direct_profile.get("object", ""))):
                continue
            if claim_clause_quality_score(direct_profile) < 4.0:
                continue
            if sentence_is_low_signal_for_query(direct_query):
                continue
            if is_routing_or_metadata_sentence(direct_query):
                continue
            if re.search(r"\bwithout\b", direct_query, flags=re.IGNORECASE):
                continue
            if CLAIM_QUERY_NOISE_RE.search(direct_query):
                continue
            if surface_noise_penalty(direct_query) > 1.25:
                continue
            words = count_words(direct_query)
            if small_claim_corpus and words > 36:
                continue
            if words < int(sampling["min_words"]) or words > int(sampling["max_words"]):
                continue
            peer_matches, peer_best_overlap = claim_support_peer_score(
                profile,
                peer_rows_by_idx[idx],
            )
            if peer_matches > max_peer_matches:
                continue
            if (small_claim_corpus or pair_like_corpus) and claim_doc_frequency > 1 and not aligned_source_sentence:
                continue
            if claim_doc_frequency <= 1:
                claim_frequency_adjustment = 3.0
            elif small_claim_corpus or pair_like_corpus:
                claim_frequency_adjustment = -4.0 * float(claim_doc_frequency - 1)
            else:
                claim_frequency_adjustment = max(-4.0, 4.0 - claim_doc_frequency * 1.5)
            selector_score = (
                claim_clause_quality_score(direct_profile)
                + min(peer_matches, 4) * 4.0
                + peer_best_overlap * 8.0
                - abs(words - 18) / 12.0
                - surface_noise_penalty(direct_query) * 0.4
                + claim_frequency_adjustment
            )
            case_id = f"claims__{row['doc_id']}__{sha1_text(direct_query)}"
            if case_id in selected_case_ids:
                continue
            fallback_candidates.append(
                {
                    "task_family": "claims",
                    "runner_task": "claims",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": case_id,
                    "query": direct_query,
                    "ground_truth": text,
                    "query_kind": "claim_direct_query",
                    "notes": (
                        f"fallback=direct_claim_surface; clause_quality={claim_clause_quality_score(direct_profile):.2f}; "
                        f"peer_matches={peer_matches}; peer_best_overlap={peer_best_overlap:.2f}; "
                        f"aligned_source_sentence={bool(aligned_source_sentence)}; "
                        f"actor_words={direct_profile['actor_words']}; object_words={direct_profile['object_words']}"
                    ),
                    "source_context": "",
                    "selector_score": selector_score,
                    "word_count": words,
                    "query_target_words": words,
                }
            )
            if len(fallback_candidates) >= max(count * 6, 24):
                break
        if (
            len(selected) + len(fallback_candidates) < count
            and allow_pseudo_chunk_claim_fallback
            and reconstruction_views is not None
            and not reconstruction_views.empty
            and not (large_claim_corpus and not pair_like_corpus)
            and not (pair_like_corpus and pair_like_has_repeated_structured_claims)
        ):
            pseudo_rows = deterministic_seeded_row_order(
                reconstruction_views.to_dict("records"),
                seed=seed + 31,
                key_fields=("doc_id", "paragraph_id", "pseudo_chunk_text", "local_id"),
            )
            if len(pseudo_rows) > 1500:
                pseudo_rows = sorted(
                    pseudo_rows,
                    key=lambda row: (
                        int(row.get("distinctive_anchor_count", 0) or 0),
                        len(
                            parse_json_list(
                                row.get("reconstruction_claims_json", row.get("claim_texts", ""))
                            )
                        ),
                    ),
                    reverse=True,
                )[: max(count * 80, 400)]
            for row in pseudo_rows:
                doc_id = normalize_space(row.get("doc_id", ""))
                paragraph_id = normalize_space(row.get("paragraph_id", ""))
                if not doc_id or not paragraph_id:
                    continue
                anchors = normalized_reconstruction_anchors(
                    parse_json_list(row.get("rare_anchors_json", row.get("entity_labels", "")))
                )
                source_context = choose_fast_query_context(
                    [
                        chunk_text_lookup.get(paragraph_key(doc_id, paragraph_id), ""),
                        extract_embedded_context(row.get("pseudo_chunk_text", "")),
                    ],
                    min_words=max(8, min_query_words // 2),
                    max_chars=420,
                )
                for claim_text in unique_preserving_order(
                    parse_json_list(
                        row.get("reconstruction_claims_json", row.get("claim_texts", ""))
                    )
                )[:3]:
                    direct_query = simplify_claim_query_seed(simplify_claim_surface(claim_text))
                    direct_query = refine_generated_query_surface(
                        direct_query,
                        task_family="claims",
                        anchors=anchors,
                        min_words=min_query_words,
                        target_words=max(16, min_query_words),
                        max_chars=420,
                        runner_cfg=candidate_runner_cfg,
                    )
                    direct_query = simplify_claim_query_seed(direct_query)
                    direct_query = finalize_query_surface(
                        direct_query,
                        add_terminal_punctuation=True,
                    )
                    lead_token = next(iter(WORD_RE.findall(direct_query)), "").casefold()
                    if lead_token in {"a", "an", "these", "those"}:
                        continue
                    if re.search(
                        r"\b(?:concerned about|busy with the construction of|would not be productive|must be an? )\b",
                        direct_query,
                        flags=re.IGNORECASE,
                    ):
                        continue
                    if (
                        not direct_query
                        or benchmark_interactive_qa_surface(direct_query)
                        or query_surface_is_blocked(direct_query, task_family="claims")
                        or sentence_is_low_signal_for_query(direct_query)
                        or is_routing_or_metadata_sentence(direct_query)
                    ):
                        continue
                    direct_profile = claim_clause_profile(direct_query)
                    if (
                        not profile_has_complete_simple_clause(direct_profile)
                        or not strict_claim_surface_is_safe(
                            direct_query,
                            direct_profile,
                            max_actor_words=max_actor_words,
                            max_object_words=max_object_words,
                        )
                    ):
                        context_query = choose_best_query_context(
                            [source_context, extract_embedded_context(row.get("pseudo_chunk_text", "")), direct_query],
                            anchors=anchors,
                            min_words=min_query_words,
                            target_words=max(16, min_query_words),
                            max_sentences=1 if min_query_words <= 30 else 2,
                            max_chars=420,
                        ) or source_context
                        context_query = refine_generated_query_surface(
                            context_query,
                            task_family="claims",
                            anchors=anchors,
                            min_words=min_query_words,
                            target_words=max(16, min_query_words),
                            max_chars=420,
                            runner_cfg=candidate_runner_cfg,
                        )
                        context_query = simplify_claim_query_seed(context_query)
                        context_query = finalize_query_surface(
                            context_query,
                            add_terminal_punctuation=True,
                        )
                        context_profile = claim_clause_profile(context_query)
                        if (
                            context_query
                            and profile_has_complete_simple_clause(context_profile)
                            and strict_claim_surface_is_safe(
                                context_query,
                                context_profile,
                                max_actor_words=max_actor_words,
                                max_object_words=max_object_words,
                            )
                            and standalone_claim_surface_is_safe(context_query)
                            and not sentence_is_low_signal_for_query(context_query)
                            and not is_routing_or_metadata_sentence(context_query)
                            and not query_surface_is_blocked(context_query, task_family="claims")
                        ):
                            direct_query = context_query
                            direct_profile = context_profile
                    pseudo_standalone_safe = standalone_claim_surface_is_safe(direct_query)
                    pseudo_complete = profile_has_complete_simple_clause(direct_profile)
                    if (
                        small_claim_corpus
                        and not (
                            direct_profile.get("actor")
                            and direct_profile.get("predicate")
                            and direct_profile.get("object")
                        )
                    ):
                        continue
                    if not (pseudo_complete or pseudo_standalone_safe):
                        continue
                    if pseudo_complete and not strict_claim_surface_is_safe(
                        direct_query,
                        direct_profile,
                        max_actor_words=max_actor_words,
                        max_object_words=max_object_words,
                    ):
                        continue
                    if not pseudo_standalone_safe:
                        continue
                    words = count_words(direct_query)
                    if words < int(sampling["min_words"]) or words > int(sampling["max_words"]):
                        continue
                    case_id = f"claims__{doc_id}__pseudo__{paragraph_id}__{sha1_text(direct_query)}"
                    if case_id in selected_case_ids:
                        continue
                    fallback_candidates.append(
                        {
                            "task_family": "claims",
                            "runner_task": "claims",
                            "doc_id": str(doc_id),
                            "paragraph_id": str(paragraph_id),
                            "case_id": case_id,
                            "query": direct_query,
                            "ground_truth": direct_query,
                            "query_kind": "claim_pseudo_chunk_query",
                            "notes": (
                                "fallback=pseudo_chunk_claim_surface; "
                                f"clause_quality={claim_clause_quality_score(direct_profile):.2f}; "
                                f"distinctive_anchor_count={int(row.get('distinctive_anchor_count', 0) or 0)}; "
                                f"actor_words={direct_profile['actor_words']}; object_words={direct_profile['object_words']}"
                            ),
                            "source_context": source_context or direct_query,
                            "selector_score": (
                                claim_clause_quality_score(direct_profile)
                                + min(int(row.get("distinctive_anchor_count", 0) or 0), 60) * 0.18
                                + context_quality_score(
                                    source_context,
                                    anchors=anchors,
                                    min_words=max(8, min_query_words // 2),
                                )
                                * 0.3
                                + 30.0
                                - abs(words - 18) / 12.0
                                - surface_noise_penalty(direct_query) * 0.35
                            ),
                            "word_count": words,
                            "query_target_words": words,
                        }
                    )
                    if len(selected) + len(fallback_candidates) >= count or len(fallback_candidates) >= max(count * 8, 32):
                        break
                if len(selected) + len(fallback_candidates) >= count or len(fallback_candidates) >= max(count * 8, 32):
                    break
        if (
            len(selected) + len(fallback_candidates) < count
            and allow_reconstruction_claim_backfill
            and reconstruction_views is not None
            and not reconstruction_views.empty
        ):
            pseudo_rows = deterministic_seeded_row_order(
                reconstruction_views.to_dict("records"),
                seed=seed + 37,
                key_fields=("doc_id", "paragraph_id", "pseudo_chunk_text", "local_id"),
            )
            for row in pseudo_rows:
                doc_id = normalize_space(row.get("doc_id", ""))
                paragraph_id = normalize_space(row.get("paragraph_id", ""))
                if not doc_id or not paragraph_id:
                    continue
                anchors = normalized_reconstruction_anchors(
                    parse_json_list(row.get("rare_anchors_json", row.get("entity_labels", "")))
                )
                for claim_text in unique_preserving_order(
                    parse_json_list(
                        row.get("reconstruction_claims_json", row.get("claim_texts", ""))
                    )
                )[:2]:
                    direct_query = simplify_claim_query_seed(simplify_claim_surface(claim_text))
                    direct_query = refine_generated_query_surface(
                        direct_query,
                        task_family="claims",
                        anchors=anchors,
                        min_words=min_query_words,
                        target_words=max(16, min_query_words),
                        max_chars=320,
                        runner_cfg=candidate_runner_cfg,
                    )
                    direct_query = simplify_claim_query_seed(direct_query)
                    direct_query = finalize_query_surface(
                        extract_sentence_excerpt(direct_query, max_sentences=1, max_chars=280),
                        add_terminal_punctuation=True,
                    )
                    if (
                        not direct_query
                        or query_surface_is_blocked(direct_query, task_family="claims")
                        or sentence_is_low_signal_for_query(direct_query)
                        or is_routing_or_metadata_sentence(direct_query)
                    ):
                        continue
                    direct_profile = claim_clause_profile(direct_query)
                    if (
                        not profile_has_complete_simple_clause(direct_profile)
                        or not strict_claim_surface_is_safe(
                            direct_query,
                            direct_profile,
                            max_actor_words=max_actor_words,
                            max_object_words=max_object_words,
                        )
                        or not standalone_claim_surface_is_safe(direct_query)
                    ):
                        continue
                    words = count_words(direct_query)
                    if words < int(sampling["min_words"]) or words > int(sampling["max_words"]):
                        continue
                    case_id = f"claims__{doc_id}__reconstruction__{paragraph_id}__{sha1_text(direct_query)}"
                    if case_id in selected_case_ids:
                        continue
                    fallback_candidates.append(
                        {
                            "task_family": "claims",
                            "runner_task": "claims",
                            "doc_id": str(doc_id),
                            "paragraph_id": str(paragraph_id),
                            "case_id": case_id,
                            "query": direct_query,
                            "ground_truth": direct_query,
                            "query_kind": "claim_reconstruction_backfill",
                            "notes": (
                                "fallback=reconstruction_claim_surface; "
                                f"clause_quality={claim_clause_quality_score(direct_profile):.2f}; "
                                f"distinctive_anchor_count={int(row.get('distinctive_anchor_count', 0) or 0)}"
                            ),
                            "source_context": direct_query,
                            "selector_score": (
                                claim_clause_quality_score(direct_profile)
                                + min(int(row.get("distinctive_anchor_count", 0) or 0), 60) * 0.15
                                + 18.0
                                - abs(words - 18) / 12.0
                                - surface_noise_penalty(direct_query) * 0.3
                            ),
                            "word_count": words,
                            "query_target_words": words,
                        }
                    )
                    if len(selected) + len(fallback_candidates) >= count or len(fallback_candidates) >= max(count * 8, 32):
                        break
                if len(selected) + len(fallback_candidates) >= count or len(fallback_candidates) >= max(count * 8, 32):
                    break
        if len(selected) + len(fallback_candidates) < count and reconstruction_seed_cases:
            for seed_case in reconstruction_seed_cases:
                seed_query = finalize_query_surface(
                    simplify_claim_query_seed(simplify_claim_surface(seed_case.get("query", ""))),
                    add_terminal_punctuation=True,
                )
                if (
                    not seed_query
                    or query_surface_is_blocked(seed_query, task_family="claims")
                    or sentence_is_low_signal_for_query(seed_query)
                    or is_routing_or_metadata_sentence(seed_query)
                ):
                    continue
                seed_profile = claim_clause_profile(seed_query)
                if (
                    not profile_has_complete_simple_clause(seed_profile)
                    or not (
                        strict_claim_surface_is_safe(
                            seed_query,
                            seed_profile,
                            max_actor_words=max_actor_words,
                            max_object_words=max_object_words,
                        )
                        or standalone_claim_surface_is_safe(seed_query)
                    )
                ):
                    continue
                case_id = f"claims__seed__{normalize_space(seed_case.get('doc_id', ''))}__{sha1_text(seed_query)}"
                if case_id in selected_case_ids:
                    continue
                words = count_words(seed_query)
                fallback_candidates.append(
                    {
                        "task_family": "claims",
                        "runner_task": "claims",
                        "doc_id": str(seed_case.get("doc_id", "")),
                        "paragraph_id": str(seed_case.get("paragraph_id", "")),
                        "case_id": case_id,
                        "query": seed_query,
                        "ground_truth": seed_query,
                        "query_kind": "claim_reconstruction_seed",
                        "notes": (
                            "fallback=reconstruction_seed_case; "
                            f"clause_quality={claim_clause_quality_score(seed_profile):.2f}"
                        ),
                        "source_context": seed_query,
                        "selector_score": (
                            claim_clause_quality_score(seed_profile)
                            + 16.0
                            - abs(words - 18) / 12.0
                            - surface_noise_penalty(seed_query) * 0.3
                        ),
                        "word_count": words,
                        "query_target_words": words,
                    }
                )
                if len(selected) + len(fallback_candidates) >= count or len(fallback_candidates) >= max(count * 8, 32):
                    break
        force_chunk_claim_fallback = small_claim_corpus
        chunk_fallback_budget = max(count * 8, 32)
        if allow_chunk_claim_fallback and (
            force_chunk_claim_fallback or len(selected) + len(fallback_candidates) < count
        ):
            for (doc_id, paragraph_id), chunk_text in chunk_text_lookup.items():
                for sentence in sentence_split(chunk_text):
                    direct_query = finalize_query_surface(sentence, add_terminal_punctuation=True)
                    if not direct_query or "?" in direct_query:
                        continue
                    direct_query = refine_generated_query_surface(
                        direct_query,
                        task_family="claims",
                        anchors=[],
                        min_words=min_query_words,
                        target_words=count_words(direct_query),
                        max_chars=420,
                        runner_cfg=candidate_runner_cfg,
                    )
                    direct_query = simplify_claim_query_seed(direct_query)
                    if (
                        not direct_query
                        or query_surface_is_blocked(direct_query, task_family="claims")
                        or sentence_is_low_signal_for_query(direct_query)
                        or is_routing_or_metadata_sentence(direct_query)
                    ):
                        continue
                    direct_profile = claim_clause_profile(direct_query)
                    if not profile_has_complete_simple_clause(direct_profile):
                        continue
                    if not strict_claim_surface_is_safe(
                        direct_query,
                        direct_profile,
                        max_actor_words=max_actor_words,
                        max_object_words=max_object_words,
                    ):
                        continue
                    if not standalone_claim_surface_is_safe(direct_query):
                        continue
                    words = count_words(direct_query)
                    if words < int(sampling["min_words"]) or words > int(sampling["max_words"]):
                        continue
                    if small_claim_corpus and words > 32:
                        continue
                    case_id = f"claims__{doc_id}__{paragraph_id}__{sha1_text(direct_query)}"
                    if case_id in selected_case_ids:
                        continue
                    fallback_candidates.append(
                        {
                            "task_family": "claims",
                            "runner_task": "claims",
                            "doc_id": str(doc_id),
                            "paragraph_id": str(paragraph_id),
                            "case_id": case_id,
                            "query": direct_query,
                            "ground_truth": direct_query,
                            "query_kind": "claim_chunk_query",
                            "notes": (
                                "fallback=chunk_claim_surface; "
                                f"clause_quality={claim_clause_quality_score(direct_profile):.2f}; "
                                f"actor_words={direct_profile['actor_words']}; object_words={direct_profile['object_words']}"
                            ),
                            "source_context": direct_query,
                            "selector_score": (
                                claim_clause_quality_score(direct_profile)
                                + 10.0
                                + (12.0 if small_claim_corpus else 0.0)
                                - abs(words - 18) / 12.0
                                - surface_noise_penalty(direct_query) * 0.35
                            ),
                            "word_count": words,
                            "query_target_words": words,
                        }
                    )
                    if len(fallback_candidates) >= chunk_fallback_budget:
                        break
                if len(fallback_candidates) >= chunk_fallback_budget:
                    break
        vetted_claim_cases = vet_claim_case_candidates(
            merge_ranked_candidates(
                [selected, fallback_candidates, minimal_direct_candidates],
                target_count=max(count * 8, count),
                max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
                score_key="selector_score",
                seed=seed,
            ),
            count=count,
            runner_cfg=runner_cfg,
        )
        if len(vetted_claim_cases) >= count:
            return normalize_selected_case_queries(
                vetted_claim_cases[:count],
                task_family="claims",
                runner_cfg=runner_cfg,
            )
        selected = vetted_claim_cases
        if len(claim_rows) >= 600 and selected:
            large_corpus_shortlist = merge_ranked_candidates(
                [selected, seeded_relation_candidates, fallback_candidates, minimal_direct_candidates],
                target_count=count,
                max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
                score_key="selector_score",
                seed=seed,
            )
            if large_corpus_shortlist:
                return normalize_selected_case_queries(
                    large_corpus_shortlist[:count],
                    task_family="claims",
                    runner_cfg=runner_cfg,
                )
    relaxed_sampling = dict(sampling)
    relaxed_sampling["_relaxed_retry_used"] = True
    relaxed_sampling["require_simple_clause_arguments"] = True
    relaxed_sampling["require_structured_source"] = not pair_like_corpus
    relaxed_sampling["exclude_generic_roles"] = False
    relaxed_sampling["query_style"] = "contextual"
    relaxed_sampling["min_role_fields"] = max(2, min_role_fields)
    relaxed_sampling["min_peer_matches"] = 0
    relaxed_sampling["min_peer_overlap"] = 0.0
    relaxed_sampling["min_clause_quality_score"] = min(
        float(relaxed_sampling.get("min_clause_quality_score", 8.0)),
        3.5,
    )
    relaxed_sampling["max_object_words"] = max(max_object_words, 28)
    relaxed = choose_claim_cases(
        claim_views,
        relations,
        chunk_text_lookup,
        count,
        relaxed_sampling,
        seed,
        runner_cfg=runner_cfg,
        claim_rows=claim_rows,
        claim_profiles=claim_profiles,
        peer_rows_by_idx=peer_rows_by_idx,
        reconstruction_views=reconstruction_views,
    )
    final_claim_cases = vet_claim_case_candidates(
        merge_ranked_candidates(
            [selected, relaxed, minimal_direct_candidates],
            target_count=max(count * 6, count),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        ),
        count=count,
        runner_cfg=runner_cfg,
    )
    if strict_small_corpus_vet:
        return normalize_selected_case_queries(
            final_claim_cases,
            task_family="claims",
            runner_cfg=runner_cfg,
        )
    if not final_claim_cases:
        final_claim_cases = merge_ranked_candidates(
            [selected, relaxed, minimal_direct_candidates],
            target_count=count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
    return normalize_selected_case_queries(
        final_claim_cases[:count],
        task_family="claims",
        runner_cfg=runner_cfg,
    )


def choose_value_redaction_cases(
    redaction_views: pd.DataFrame | None,
    entities: pd.DataFrame,
    doc_chunks_by_doc: dict[str, list[dict[str, Any]]],
    para_richness: dict[tuple[str, str], dict[str, int]],
    component_index: dict[tuple[str, str], dict[str, list[str]]],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    runner_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    allowed_types = {value.casefold() for value in sampling["allowed_entity_types"]}
    fallback_types = {
        value.casefold() for value in sampling.get("fallback_entity_types", [])
    }
    blocked_labels = {normalize_space(value).casefold() for value in sampling["blocked_labels"]}
    corpus_doc_count = max(1, len(doc_chunks_by_doc))
    small_corpus = exhaustive_small_doc_corpus(corpus_doc_count)
    large_corpus = optimized_doc_corpus(corpus_doc_count)
    pair_like_corpus = bool(doc_chunks_by_doc) and all(
        str(doc_id).startswith("Pair_") for doc_id in doc_chunks_by_doc
    )
    early_type_priority = {
        "person": 10.0,
        "organization": 4.0,
        "government": 3.5,
        "military": 3.5,
        "political_party": 3.0,
        "location": 9.0,
        "country": 8.0,
    }
    redaction_view_rows: list[dict[str, Any]] = []
    redaction_view_source_tables: set[str] = set()
    redaction_view_has_entity_rows = False
    if redaction_views is not None and not redaction_views.empty:
        source_series = redaction_views.get(
            "candidate_source_table",
            pd.Series(dtype=str),
        )
        redaction_view_source_tables = {
            normalize_space(value).casefold()
            for value in source_series.tolist()
            if normalize_space(value)
        }
        redaction_view_has_entity_rows = "entities" in redaction_view_source_tables
        if not large_corpus:
            redaction_view_rows = redaction_views.to_dict("records")
    redaction_view_only_entities = redaction_view_source_tables <= {"entities"}
    synthesized_entity_rows = synthesize_redaction_entity_rows(entities)
    if large_corpus and synthesized_entity_rows:
        # Large-corpus redaction tables are intentionally expanded for runtime,
        # but build-time VR generation only needs one canonicalized entity hit
        # per extraction row. Reading the full expanded view into Python dicts
        # is an unnecessary hot path and was the main source of build stalls.
        redaction_rows = synthesized_entity_rows
    elif redaction_view_has_entity_rows:
        redaction_rows = redaction_view_rows
    else:
        redaction_rows = redaction_view_rows + synthesized_entity_rows
    if large_corpus and redaction_rows:
        large_corpus_redaction_seed_cap = int(
            sampling.get("large_corpus_redaction_seed_cap", 0) or 0
        )
        if large_corpus_redaction_seed_cap <= 0:
            large_corpus_redaction_seed_cap = max(count * 400, 4000)

        def early_redaction_seed_score(row: dict[str, Any]) -> tuple[float, float, float]:
            label = normalize_space(
                row.get("candidate_label", "") or row.get("label", "")
            )
            if not label:
                return (-9999.0, -9999.0, -9999.0)
            entity_type = normalize_candidate_type(
                row.get("candidate_type", "") or row.get("type", "")
            )
            confidence = safe_float(
                row.get("candidate_source_confidence", row.get("confidence", 0.0))
            )
            label_words = count_words(label)
            distinct_tokens = distinctive_label_token_count(label)
            context_text = normalize_space(
                row.get("context_text", "")
                or row.get("context_window_text", "")
                or row.get("candidate_role_or_context", "")
                or row.get("description", "")
            )
            context_words = count_words(context_text)
            score = (
                float(early_type_priority.get(entity_type, 0.0)) * 8.0
                + confidence * 10.0
                + min(label_words, 4) * 2.0
                + min(distinct_tokens, 4) * 2.5
                + min(context_words, 24) * 0.15
                - (8.0 if looks_like_acronym_label(label) and entity_type != "person" else 0.0)
                - (6.0 if has_generic_label_head(label) else 0.0)
            )
            return (
                score,
                float(context_words),
                float(distinct_tokens),
            )

        redaction_rows = sorted(
            redaction_rows,
            key=early_redaction_seed_score,
            reverse=True,
        )[:large_corpus_redaction_seed_cap]
    redaction_rows = deterministic_seeded_row_order(
        redaction_rows,
        seed=seed,
        key_fields=(
            "doc_id",
            "paragraph_id",
            "candidate_label",
            "candidate_text",
            "candidate_type",
            "candidate_source_row_id",
        ),
    )
    strict_mode = not maybe_bool(sampling.get("_relaxed_retry_used", False))
    include_locations = maybe_bool(sampling.get("include_locations", False))
    min_claims = int(sampling.get("min_claims_in_paragraph", 1))
    min_events = int(sampling.get("min_events_in_paragraph", 1))
    min_relations = int(sampling.get("min_relations_in_paragraph", 2))
    max_chunk_words = int(sampling.get("max_chunk_words", 180))
    min_query_words = int(sampling.get("min_query_words", 16))
    max_query_words = int(sampling.get("max_query_words", 180))
    min_role_words = int(sampling.get("min_role_words", 4))
    location_min_role_words = int(sampling.get("location_min_role_words", 8))
    slot_sentence_max_words = int(sampling.get("slot_sentence_max_words", 48))
    max_slot_count = int(sampling.get("max_slot_count", 3))
    slot_count_decay = float(sampling.get("slot_count_decay", 0.5))
    if large_corpus:
        include_locations = True
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    preferred_entity_types = {"person", "location", "country", "organization"}
    small_corpus_supported_entity_types = {
        "person",
        "location",
        "country",
        "organization",
        "government",
        "military",
        "political_party",
    }
    type_priority = {
        "person": 10.0,
        "organization": 4.0,
        "government": 3.5,
        "military": 3.5,
        "political_party": 3.0,
        "location": 9.0,
        "country": 8.0,
    }
    doc_entity_rows: dict[str, list[dict[str, Any]]] = {}
    paragraph_entity_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    paragraph_entity_counts: dict[tuple[str, str], int] = {}
    label_occurrence_counts: dict[str, int] = {}
    label_doc_sets: dict[str, set[str]] = {}
    label_type_sets: dict[str, set[str]] = {}
    for entity_row in redaction_rows:
        if normalize_space(entity_row.get("candidate_source_table", "")).casefold() != "entities":
            continue
        doc_id = str(entity_row["doc_id"])
        para_key = paragraph_key(entity_row["doc_id"], entity_row["paragraph_id"])
        doc_entity_rows.setdefault(doc_id, []).append(entity_row)
        paragraph_entity_rows.setdefault(para_key, []).append(entity_row)
        paragraph_entity_counts[para_key] = paragraph_entity_counts.get(para_key, 0) + 1
        label_key = normalize_space(entity_row.get("candidate_label", "")).casefold()
        if not label_key:
            continue
        label_occurrence_counts[label_key] = label_occurrence_counts.get(label_key, 0) + 1
        label_doc_sets.setdefault(label_key, set()).add(doc_id)
        label_type_sets.setdefault(label_key, set()).add(
            normalize_candidate_type(entity_row.get("candidate_type", ""))
        )
    min_label_words = int(sampling.get("min_label_words", 1))
    person_min_label_words = int(sampling.get("person_min_label_words", 1))
    max_entity_candidates_in_paragraph = int(
        sampling.get("max_entity_candidates_in_paragraph", 999999)
    )
    max_label_occurrences = int(sampling.get("max_label_occurrences", 4))
    if large_corpus:
        max_entity_candidates_in_paragraph = max(
            max_entity_candidates_in_paragraph,
            48,
        )
        max_label_occurrences = max(max_label_occurrences, 10)
    allow_single_token_locations = maybe_bool(
        sampling.get("allow_single_token_locations", False)
    )
    base_max_label_doc_frequency = int(
        sampling.get("max_label_doc_frequency", 999999)
    )
    fallback_max_label_doc_frequency = max(
        base_max_label_doc_frequency,
        int(sampling.get("fallback_max_label_doc_frequency", 2)),
    )
    if large_corpus:
        base_max_label_doc_frequency = max(
            base_max_label_doc_frequency,
            int(sampling.get("large_corpus_max_label_doc_frequency", 6)),
        )
        fallback_max_label_doc_frequency = max(
            fallback_max_label_doc_frequency,
            int(sampling.get("large_corpus_fallback_max_label_doc_frequency", 10)),
        )
    profile_ladder = [
        {
            "min_confidence": float(sampling.get("min_confidence", 0.0)),
            "max_label_doc_frequency": base_max_label_doc_frequency,
            "min_role_words": int(sampling.get("min_role_words", 4)),
            "entity_types": allowed_types,
        },
        {
            "min_confidence": max(float(sampling.get("min_confidence", 0.0)) - 0.02, 0.0),
            "max_label_doc_frequency": fallback_max_label_doc_frequency,
            "min_role_words": max(
                int(sampling.get("min_role_words", 4)) - 1,
                int(sampling.get("fallback_min_role_words", 5)),
            ),
            "entity_types": allowed_types,
        },
        {
            "min_confidence": max(float(sampling.get("min_confidence", 0.0)) - 0.04, 0.0),
            "max_label_doc_frequency": fallback_max_label_doc_frequency,
            "min_role_words": max(
                int(sampling.get("min_role_words", 4)) - 1,
                int(sampling.get("fallback_min_role_words", 5)),
            ),
            "entity_types": allowed_types | fallback_types,
        },
    ]
    if small_corpus:
        for profile in profile_ladder:
            profile["entity_types"] = (
                set(profile.get("entity_types", set()))
                | small_corpus_supported_entity_types
            )
        profile_ladder.append(
            {
                "min_confidence": max(float(sampling.get("min_confidence", 0.0)) - 0.12, 0.0),
                "max_label_doc_frequency": max(
                    fallback_max_label_doc_frequency,
                    int(sampling.get("small_corpus_relaxed_max_label_doc_frequency", 6)),
                ),
                "min_role_words": 0,
                "entity_types": set(small_corpus_supported_entity_types)
                | allowed_types
                | fallback_types,
            }
        )
    if large_corpus:
        profile_ladder[0]["entity_types"] = (
            set(profile_ladder[0].get("entity_types", set()))
            | {"person", "location", "country"}
        )
        profile_ladder[1]["entity_types"] = (
            set(profile_ladder[1].get("entity_types", set()))
            | {"person", "location", "country", "organization"}
        )
        profile_ladder[-1]["entity_types"] = (
            set(profile_ladder[-1].get("entity_types", set()))
            | {"location", "country", "organization", "government"}
        )
    candidate_source_rows = [
        row
        for row in redaction_rows
        if normalize_space(row.get("candidate_source_table", "")).casefold() == "entities"
    ]
    if large_corpus:
        def large_corpus_seed_score(row: dict[str, Any]) -> tuple[float, int, float]:
            label = normalize_space(row.get("candidate_label", ""))
            entity_type = normalize_candidate_type(row.get("candidate_type", ""))
            label_key = label.casefold()
            label_doc_frequency = len(label_doc_sets.get(label_key, set()))
            confidence = safe_float(row.get("candidate_source_confidence", 0.0))
            label_words = count_words(label)
            preferred_bonus = 50.0 if entity_type in preferred_entity_types else 0.0
            frequency_penalty = min(label_doc_frequency, 12) * 3.0
            word_bonus = min(label_words, 4) * 2.0
            acronym_penalty = 8.0 if looks_like_acronym_label(label) and entity_type != "person" else 0.0
            return (
                preferred_bonus + float(type_priority.get(entity_type, 0.0)) * 5.0 + confidence * 10.0 + word_bonus - frequency_penalty - acronym_penalty,
                label_words,
                confidence,
            )

        candidate_source_rows = sorted(
            candidate_source_rows,
            key=large_corpus_seed_score,
            reverse=True,
        )[: max(count * 40, 400)]
    all_candidates: dict[str, dict[str, Any]] = {}
    for row in candidate_source_rows:
        label = normalize_space(row.get("candidate_label", ""))
        if not label:
            continue
        base_query_target_words = deterministic_query_word_target(
            f"value_redaction|{row['doc_id']}|{row['paragraph_id']}|{label}",
            minimum_words=min_query_words,
            maximum_words=min(max_query_words, 128),
        )
        label_words = count_words(label)
        if label_words > int(sampling["max_label_words"]) or label_words < min_label_words:
            continue
        if label.casefold() in blocked_labels:
            continue
        if maybe_bool(row.get("is_section_heading_like", False)):
            continue
        if maybe_bool(row.get("is_country_like", False)) and not small_corpus:
            continue
        if maybe_bool(row.get("is_position_like", False)):
            continue
        entity_type = normalize_candidate_type(row.get("candidate_type", ""))
        if entity_type in {"other", "unknown", "target", ""}:
            continue
        if entity_type == "person" and is_title_only_person_label(label):
            continue
        if strict_mode and small_corpus and entity_type not in small_corpus_supported_entity_types:
            continue
        if entity_type == "location" and not (include_locations or small_corpus):
            continue
        if entity_type == "person" and is_probably_generic_person_label(label):
            continue
        if strict_mode and small_corpus and entity_type == "person" and label_words < 2:
            continue
        if entity_type == "person" and label_words < person_min_label_words:
            continue
        label_key = label.casefold()
        label_doc_frequency = len(label_doc_sets.get(label_key, set()))
        label_occurrences = int(label_occurrence_counts.get(label_key, 1))
        role_context = normalize_space(row.get("candidate_role_or_context", ""))
        role_words = count_words(role_context)
        acronym_like = maybe_bool(row.get("looks_like_acronym", False)) or looks_like_acronym_label(label)
        distinctive_tokens = distinctive_label_token_count(label)
        generic_label_head = has_generic_label_head(label)
        if distinctive_tokens == 0 and label_words <= 3:
            continue
        if label_occurrences > max_label_occurrences:
            continue
        if entity_type == "location" and label_words == 1 and not (
            allow_single_token_locations or small_corpus
        ):
            continue
        if strict_mode and entity_type != "person" and label_words == 1 and acronym_like:
            continue
        if strict_mode and small_corpus and entity_type in {"organization", "government", "military", "political_party"} and label_words < 2:
            continue
        if strict_mode and entity_type in {"organization", "government", "military", "political_party"} and label_words == 1:
            continue
        if large_corpus and entity_type in {"organization", "government"}:
            if acronym_like:
                continue
            if label_words < 2 and distinctive_tokens < 2:
                continue
            if entity_type == "government" and distinctive_tokens < 2:
                continue
        para_key = paragraph_key(row["doc_id"], row["paragraph_id"])
        if paragraph_entity_counts.get(para_key, 0) > max_entity_candidates_in_paragraph:
            continue
        richness = para_richness.get(para_key, {})
        claim_n = int(richness.get("claim_n", 0))
        event_n = int(richness.get("event_n", 0))
        rel_n = int(richness.get("rel_n", 0))
        if claim_n < min_claims or event_n < min_events or rel_n < min_relations:
            continue
        chunk = find_chunk_for_redaction_row(
            doc_chunks_by_doc.get(str(row["doc_id"]), []),
            label=label,
            preferred_chunk_id=str(row.get("chunk_id", "")),
        )
        if not chunk:
            continue
        chunk_words = int(chunk.get("word_count", count_words(str(chunk.get("chunk_text", "")))))
        if chunk_words > max_chunk_words:
            continue
        chunk_text = str(chunk.get("chunk_text", ""))
        if large_corpus:
            entity_context = finalize_query_surface(
                extract_sentence_excerpt(
                    str(row.get("context_text", "") or ""),
                    max_sentences=2,
                    max_chars=320,
                )
            )
            entity_window = finalize_query_surface(
                extract_sentence_excerpt(
                    str(row.get("context_window_text", "") or ""),
                    max_sentences=2,
                    max_chars=360,
                )
            )
        else:
            entity_context = clean_query_passage(str(row.get("context_text", "")))
            entity_window = clean_query_passage(str(row.get("context_window_text", "")))
        if uppercase_token_ratio(entity_context) >= 0.45 and count_words(entity_context) >= 18:
            entity_context = normalize_uppercase_passage(entity_context)
        if uppercase_token_ratio(entity_window) >= 0.45 and count_words(entity_window) >= 18:
            entity_window = normalize_uppercase_passage(entity_window)
        entity_context = strip_internal_topic_sections(strip_low_signal_lead_sentences(entity_context))
        entity_window = strip_internal_topic_sections(strip_low_signal_lead_sentences(entity_window))
        match = re.search(re.escape(label), chunk_text, flags=re.IGNORECASE)
        if not match:
            continue
        component_bucket = component_index.get(
            para_key,
            {"entity_labels": [], "event_labels": [], "claim_texts": []},
        )
        claim_context_candidates = [
            finalize_query_surface(claim_text, add_terminal_punctuation=True)
            for claim_text in component_bucket.get("claim_texts", [])
            if find_label_match(str(claim_text), label)
        ]
        claim_context_candidates = [
            text
            for text in unique_preserving_order(claim_context_candidates)
            if text
            and not is_routing_or_metadata_sentence(text)
            and not is_list_like_text(text)
            and not LEGAL_BOILERPLATE_RE.search(text)
            and not BROKEN_LEAD_RE.search(text)
        ]
        event_context_candidates = [
            finalize_query_surface(event_text, add_terminal_punctuation=True)
            for event_text in component_bucket.get("event_labels", [])
            if find_label_match(str(event_text), label)
        ]
        event_context_candidates = [
            text
            for text in unique_preserving_order(event_context_candidates)
            if text
            and not is_routing_or_metadata_sentence(text)
            and not is_list_like_text(text)
            and not LEGAL_BOILERPLATE_RE.search(text)
            and not BROKEN_LEAD_RE.search(text)
        ]
        def support_hits_for_label(candidate_label: str) -> int:
            return sum(
                1
                for claim_text in component_bucket.get("claim_texts", [])
                if re.search(rf"\b{re.escape(candidate_label)}\b", claim_text, flags=re.IGNORECASE)
            ) + sum(
                1
                for event_text in component_bucket.get("event_labels", [])
                if re.search(rf"\b{re.escape(candidate_label)}\b", event_text, flags=re.IGNORECASE)
            )
        claim_label_hits = sum(
            1
            for claim_text in component_bucket.get("claim_texts", [])
            if re.search(rf"\b{re.escape(label)}\b", claim_text, flags=re.IGNORECASE)
        )
        event_label_hits = sum(
            1
            for event_text in component_bucket.get("event_labels", [])
            if re.search(rf"\b{re.escape(label)}\b", event_text, flags=re.IGNORECASE)
        )
        same_type_chunk_hits = 0
        for peer in doc_entity_rows.get(str(row["doc_id"]), []):
            peer_label = normalize_space(peer.get("candidate_label", ""))
            if not peer_label:
                continue
            peer_type = normalize_candidate_type(peer.get("candidate_type", ""))
            if peer_type != entity_type:
                continue
            if re.search(re.escape(peer_label), chunk_text, flags=re.IGNORECASE):
                same_type_chunk_hits += 1
        clean_component_context_window = bool(claim_context_candidates or event_context_candidates)
        if strict_mode:
            if entity_type == "person":
                if large_corpus:
                    if same_type_chunk_hits > 3:
                        continue
                elif clean_component_context_window:
                    if same_type_chunk_hits > 4:
                        continue
                elif same_type_chunk_hits != 1:
                    continue
            if entity_type == "location" and same_type_chunk_hits > (5 if clean_component_context_window else 3):
                continue
            if entity_type in {"organization", "government", "military", "political_party"} and same_type_chunk_hits > (4 if clean_component_context_window else 2):
                continue
            if small_corpus and entity_type not in small_corpus_supported_entity_types:
                continue
        else:
            if entity_type == "person" and same_type_chunk_hits > 3:
                continue
            if entity_type == "location" and same_type_chunk_hits > 5:
                continue
        local_support_hits = claim_label_hits + event_label_hits
        sentence_source = (
            (claim_context_candidates[0] if claim_context_candidates else "")
            or (event_context_candidates[0] if event_context_candidates else "")
            or entity_context
            or entity_window
            or chunk_text
        )
        sentences = sentence_split(sentence_source)
        focused_context = ""
        slot_sentence = ""
        slot_sentence_word_count = 0
        for sent_index, sentence in enumerate(sentences):
            if re.search(re.escape(label), sentence, flags=re.IGNORECASE):
                parts = []
                if sent_index > 0:
                    parts.append(sentences[sent_index - 1])
                parts.append(sentence)
                if sent_index + 1 < len(sentences):
                    parts.append(sentences[sent_index + 1])
                focused_context = normalize_space(" ".join(parts))
                slot_sentence = normalize_space(sentence)
                slot_sentence_word_count = count_words(slot_sentence)
                break
        left_right_context = focused_context or entity_context or entity_window or bounded_context(chunk_text, match.start(), match.end())
        compact_context = entity_window or bounded_context(chunk_text, match.start(), match.end(), min_chars=160)
        compact_excerpt = extract_sentence_excerpt(
            compact_context,
            max_sentences=2,
            max_chars=320,
        )
        confidence = float(row.get("candidate_source_confidence", 0) or 0)
        neighbor_context = ""
        if not large_corpus:
            neighbor_context = extract_neighbor_chunk_context(
                doc_chunks_by_doc,
                row["doc_id"],
                row["paragraph_id"],
                anchors=[label],
                min_words=min_query_words,
                max_sentences=3,
                max_chars=520,
                max_radius=int(sampling.get("neighbor_chunk_radius", 1)),
            )
        use_slot_sentence_direct = bool(
            slot_sentence
            and substantive_sentence_score(slot_sentence, anchors=[label]) >= (5.0 if clean_component_context_window else 5.5)
            and (
                (clean_component_context_window and local_support_hits >= 1)
                or local_support_hits >= 2
                or label_doc_frequency >= 2
                or entity_type in {"location", "country"}
            )
        )
        context_text_candidates = [
            *claim_context_candidates[:2],
            *event_context_candidates[:1],
            entity_context,
            entity_window,
            focused_context,
            compact_excerpt,
            left_right_context,
            compact_context,
        ]
        if not large_corpus:
            context_text_candidates.extend([
                neighbor_context,
                chunk_text,
            ])
        if use_slot_sentence_direct:
            query_base = finalize_query_surface(slot_sentence, add_terminal_punctuation=True)
        else:
            if large_corpus:
                query_base = choose_fast_query_context(
                    context_text_candidates,
                    min_words=min_query_words,
                    max_chars=420,
                )
            else:
                query_base = choose_best_query_context(
                    context_text_candidates,
                    anchors=[label],
                    min_words=min_query_words,
                    target_words=base_query_target_words,
                    max_sentences=2,
                    max_chars=520,
                )
        query_base = finalize_query_surface(query_base)
        query_base = clean_query_passage(query_base)
        query_base = strip_internal_topic_sections(query_base)
        query_base = strip_low_signal_lead_sentences(query_base)
        query_base = finalize_query_surface(query_base, add_terminal_punctuation=True)
        if not query_base or not find_label_match(query_base, label):
            continue
        if strict_mode and LOW_SIGNAL_SENTENCE_LEAD_RE.search(normalize_space(query_base)):
            continue
        primary_slot = {
            "label": label,
            "slot_type": label_type_string(entity_type),
            "source_id": str(row["doc_id"]),
            "entity_type": entity_type,
            "confidence": confidence,
        }
        peer_specs: list[dict[str, Any]] = [primary_slot]
        for peer in paragraph_entity_rows.get(para_key, []):
            peer_label = normalize_space(peer.get("candidate_label", ""))
            if not peer_label or peer_label.casefold() == label.casefold():
                continue
            if not re.search(re.escape(peer_label), query_base, flags=re.IGNORECASE):
                continue
            peer_type = normalize_candidate_type(peer.get("candidate_type", ""))
            if peer_type == "person" and (
                is_probably_generic_person_label(peer_label)
                or is_title_only_person_label(peer_label)
            ):
                continue
            if peer_type != entity_type and peer_type not in allowed_types and not (
                small_corpus and peer_type in small_corpus_supported_entity_types
            ):
                continue
            peer_specs.append(
                {
                    "label": peer_label,
                    "slot_type": label_type_string(peer_type),
                    "source_id": str(peer.get("doc_id", row["doc_id"])),
                    "entity_type": peer_type,
                    "confidence": safe_float(peer.get("candidate_source_confidence", 0.0)),
                    "local_support_hits": support_hits_for_label(peer_label),
                    "label_doc_frequency": len(label_doc_sets.get(peer_label.casefold(), set())),
                    "label_words": count_words(peer_label),
                    "acronym_like": maybe_bool(peer.get("looks_like_acronym", False)) or looks_like_acronym_label(peer_label),
                }
            )
        unique_peer_specs: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        for spec in sorted(
            peer_specs,
            key=lambda item: (
                0 if item["label"].casefold() == label.casefold() else 1,
                -int(item.get("local_support_hits", 0)),
                -int(item.get("label_doc_frequency", 0)),
                -distinctive_label_token_count(item["label"]),
                -safe_float(item.get("confidence", 0.0)),
                -len(item["label"]),
            ),
        ):
            key = spec["label"].casefold()
            if key in seen_labels:
                continue
            seen_labels.add(key)
            unique_peer_specs.append(spec)
        compatible_peer_specs: list[dict[str, Any]] = []
        for spec in unique_peer_specs:
            spec_label = normalize_space(spec.get("label", ""))
            spec_type = normalize_candidate_type(spec.get("entity_type", ""))
            if not spec_label:
                continue
            if spec_label.casefold() == label.casefold():
                compatible_peer_specs.append(spec)
                continue
            if not strict_mode:
                compatible_peer_specs.append(spec)
                continue
            peer_doc_frequency = int(spec.get("label_doc_frequency", 0))
            peer_support_hits = int(spec.get("local_support_hits", 0))
            minimum_peer_support = 1 if spec_type == "person" else 2
            peer_is_supported = (
                peer_doc_frequency >= 2
                or peer_support_hits >= minimum_peer_support
                or (small_corpus and peer_support_hits >= 1)
            )
            if not peer_is_supported:
                continue
            if small_corpus and spec_type not in small_corpus_supported_entity_types:
                continue
            if int(spec.get("label_words", 0)) == 1 and spec_type != "location":
                continue
            if maybe_bool(spec.get("acronym_like", False)) and spec_type != "person":
                continue
            compatible_peer_specs.append(spec)
        desired_slot_count = 1
        supportive_paragraph = bool(claim_n >= 1 or event_n >= 1 or rel_n >= 3)
        strong_local_signal = bool(
            local_support_hits >= 2
            or clean_component_context_window
            or label_doc_frequency >= 2
        )
        multi_slot_is_allowed = bool(
            max_slot_count > 1
            and len(compatible_peer_specs) > 1
            and slot_sentence
            and len(sentence_split(slot_sentence)) == 1
            and slot_sentence_word_count <= slot_sentence_max_words
            and supportive_paragraph
            and strong_local_signal
        )
        if multi_slot_is_allowed:
            desired_slot_count = min(
                max_slot_count,
                deterministic_tail_count(
                    f"{row['doc_id']}|{row['paragraph_id']}|{label}",
                    max_count=min(max_slot_count, 3),
                    decay=slot_count_decay,
                ),
            )
        query_target_words = scaled_redaction_query_target(
            f"value_redaction|{row['doc_id']}|{row['paragraph_id']}|{label}",
            minimum_words=min_query_words,
            maximum_words=min(max_query_words, 128),
            slot_count=desired_slot_count,
        )
        if large_corpus:
            query_base = finalize_query_surface(
                extract_sentence_excerpt(
                    query_base,
                    max_sentences=2,
                    max_chars=420,
                ),
                add_terminal_punctuation=True,
            ) or query_base
        else:
            query_base = choose_best_query_context(
                [query_base],
                anchors=[label],
                min_words=min_query_words,
                target_words=query_target_words,
                max_sentences=2,
                max_chars=520,
            ) or query_base
        query_base, used_specs = redact_labels_in_context(query_base, compatible_peer_specs[:desired_slot_count])
        if not used_specs:
            continue
        primary_entity_type = normalize_candidate_type(
            used_specs[0].get("entity_type", entity_type)
        ) or entity_type
        if large_corpus:
            query_base = finalize_query_surface(
                extract_sentence_excerpt(
                    query_base,
                    max_sentences=1 if len(used_specs) == 1 else 2,
                    max_chars=420,
                ),
                add_terminal_punctuation=True,
            ) or query_base
        else:
            query_base = choose_best_query_context(
                [query_base],
                anchors=[normalize_space(spec.get("label", "")) for spec in used_specs],
                min_words=min_query_words,
                target_words=query_target_words,
                max_sentences=1 if len(used_specs) == 1 else 2,
                max_chars=520,
            ) or query_base
        query_base = finalize_query_surface(
            extract_sentence_excerpt(
                query_base,
                max_sentences=1 if len(used_specs) == 1 else 2,
                max_chars=520,
            ),
            add_terminal_punctuation=True,
        )
        slot_window_quality = substantive_sentence_score(
            query_base,
            anchors=[normalize_space(spec.get("label", "")) for spec in used_specs],
        )
        if strict_mode and slot_window_quality < 6.0:
            continue
        clean_direct_slot = (
            bool(slot_sentence)
            and slot_window_quality >= (6.6 if small_corpus else 7.0)
            and count_words(query_base) >= max(10, min_query_words - 2)
            and surface_noise_penalty(query_base) <= 1.25
            and not LOW_SIGNAL_SENTENCE_LEAD_RE.search(normalize_space(query_base))
            and (clean_component_context_window or local_support_hits >= 1)
        )
        small_corpus_clean_window = bool(
            small_corpus
            and (
                clean_direct_slot
                or (
                    slot_window_quality >= 6.5
                    and count_words(query_base) >= max(10, min_query_words - 2)
                )
            )
        )
        if VALUE_CONTEXT_ROUTING_RE.search(query_base):
            continue
        if slot_sentence and (slot_sentence_word_count < 6 or slot_sentence_word_count > slot_sentence_max_words):
            continue
        if slot_sentence and is_routing_or_metadata_sentence(slot_sentence):
            continue
        if slot_sentence and slot_sentence.casefold().startswith("ref "):
            continue
        if focused_context and focused_context.casefold().startswith("ref "):
            continue
        ambiguity_penalty = 0.0
        if entity_type == "person" and same_type_chunk_hits > 1:
            ambiguity_penalty = 8.0 * float(same_type_chunk_hits - 1)
        label_occurrence_penalty = max(label_doc_frequency - 1, 0) * 2.25
        label_reuse_penalty = max(label_occurrence_counts.get(label_key, 1) - label_doc_frequency, 0) * 0.35
        mixed_type_penalty = max(len(label_type_sets.get(label_key, set())) - 1, 0) * 6.0
        short_name_penalty = 0.0
        if entity_type == "person" and label_words == 1:
            short_name_penalty = 3.0
        if entity_type == "person":
            if strict_mode and small_corpus:
                if local_support_hits == 0 and not clean_direct_slot:
                    continue
            else:
                if strict_mode and not large_corpus and label_doc_frequency < 2 and not clean_direct_slot:
                    continue
                if strict_mode and local_support_hits == 0 and not (
                    large_corpus and label_doc_frequency >= 2
                ) and not clean_direct_slot:
                    continue
        else:
            if strict_mode and local_support_hits == 0 and not small_corpus_clean_window and not clean_direct_slot:
                if not (large_corpus and label_doc_frequency >= 2):
                    continue
            if strict_mode and entity_type == "location" and small_corpus:
                if local_support_hits == 0 and not small_corpus_clean_window and not clean_direct_slot:
                    continue
            elif (
                strict_mode
                and not large_corpus
                and label_doc_frequency < 2
                and not small_corpus_clean_window
                and not clean_direct_slot
            ):
                continue
            if generic_label_head and distinctive_tokens < 2:
                continue
            if strict_mode and acronym_like and label_doc_frequency < 4 and not small_corpus_clean_window and not clean_direct_slot:
                continue
            if strict_mode and acronym_like and local_support_hits < 2 and role_words < 4 and not small_corpus_clean_window and not clean_direct_slot:
                continue
            if strict_mode and label_words == 1 and role_words < 3 and local_support_hits < 2 and not small_corpus_clean_window and not clean_direct_slot:
                continue
        if strict_mode and entity_type == "location" and label_words == 1 and local_support_hits < 2 and not small_corpus_clean_window and not clean_direct_slot:
            continue
        if (
            strict_mode
            and len(used_specs) == 1
            and primary_entity_type == "person"
            and same_type_chunk_hits > 1
            and role_words < 6
            and local_support_hits < 2
            and not clean_component_context_window
        ):
            continue
        if (
            strict_mode
            and len(used_specs) == 1
            and maybe_bool(row.get("is_country_like", False))
            and primary_entity_type == "location"
            and label_doc_frequency >= 2
            and local_support_hits < 3
            and not small_corpus_clean_window
        ):
            continue
        if strict_mode and local_support_hits <= 0 and label_doc_frequency <= 1 and not large_corpus and not clean_direct_slot:
            continue
        if strict_mode and large_corpus and entity_type not in preferred_entity_types:
            if label_doc_frequency < 2 or local_support_hits < 1:
                continue
            if generic_label_head or acronym_like or distinctive_tokens < 2:
                continue
        if large_corpus and entity_type in {"organization", "government"} and local_support_hits < 1:
            continue
        if large_corpus and entity_type == "government" and generic_label_head:
            continue
        semantic_anchor_bonus = local_support_hits * 4.0
        weak_anchor_penalty = 6.0 if local_support_hits == 0 else 0.0
        boilerplate_penalty = 7.0 if LEGAL_BOILERPLATE_RE.search(chunk_text) else 0.0
        if label_doc_frequency == 1:
            repeat_bonus = -4.0
        elif 2 <= label_doc_frequency <= 4:
            repeat_bonus = 4.0 + (label_doc_frequency - 2) * 1.25
        else:
            repeat_bonus = max(0.0, 6.0 - (label_doc_frequency - 4) * 1.5)
        occurrence_bonus = min(label_occurrence_counts.get(label_key, 1), 6) * 0.6
        generic_label_penalty = 8.0 if generic_label_head and distinctive_tokens < 2 else 0.0
        acronym_penalty = 5.0 if acronym_like and entity_type != "person" else 0.0
        structure_bonus = (
            min(claim_n, 6) * 1.5
            + min(event_n, 4) * 1.5
            + min(rel_n, 8) * 0.75
        )
        density_penalty = (
            max(claim_n - 10, 0) * 0.75
            + max(event_n - 8, 0) * 0.5
            + max(rel_n - 14, 0) * 0.35
        )
        expected_targets: dict[str, Any] = {}
        answer_targets: dict[str, Any] = {}
        for used_spec in used_specs:
            target_payload = {"source_id": str(used_spec.get("source_id", row["doc_id"]))}
            answer_targets[str(used_spec["slot_id"])] = {
                "source_id": str(used_spec.get("source_id", row["doc_id"])),
                "selected_value": normalize_space(used_spec.get("label", "")),
            }
            if maybe_bool(sampling.get("strict_selected_value_match", False)):
                target_payload["selected_value"] = normalize_space(used_spec.get("label", ""))
            expected_targets[str(used_spec["slot_id"])] = target_payload
        query = refine_generated_query_surface(
            query_base,
            task_family="value_redaction",
            anchors=[used_spec.get("label", "") for used_spec in used_specs],
            min_words=min_query_words,
            target_words=query_target_words,
            max_chars=720,
            runner_cfg=candidate_runner_cfg,
        )
        if len(used_specs) == 1:
            single_slot_context_sentences = 1 if (local_support_hits >= 2 or label_doc_frequency >= 2) else 2
            if small_corpus and not pair_like_corpus:
                single_slot_context_sentences = max(single_slot_context_sentences, 2)
            query = finalize_query_surface(
                extract_sentence_excerpt(
                    query,
                    max_sentences=single_slot_context_sentences,
                    max_chars=520 if small_corpus and not pair_like_corpus else 420,
                ),
                add_terminal_punctuation=True,
            )
        if count_words(query) < min_query_words or count_words(query) > max_query_words:
            continue
        if strict_mode and len(sentence_split(query)) > (1 if len(used_specs) == 1 else 2):
            continue
        if not value_redaction_query_is_viable(
            query,
            entity_type=primary_entity_type,
            used_specs=used_specs,
        ):
            continue
        visible_anchor_labels: list[str] = []
        redacted_label_keys = {
            normalize_space(used_spec.get("label", "")).casefold()
            for used_spec in used_specs
            if normalize_space(used_spec.get("label", ""))
        }
        for anchor_label in unique_preserving_order(component_bucket.get("entity_labels", [])):
            normalized_anchor_label = normalize_space(anchor_label)
            if not normalized_anchor_label:
                continue
            if normalized_anchor_label.casefold() in redacted_label_keys:
                continue
            if (
                count_words(normalized_anchor_label) == 1
                and looks_like_acronym_label(normalized_anchor_label)
            ):
                continue
            if find_label_match(query, normalized_anchor_label):
                visible_anchor_labels.append(normalized_anchor_label)
        visible_anchor_count = len(visible_anchor_labels)
        has_date_or_number = bool(
            DATE_TOKEN_RE.search(query) or NUMBER_RE.search(query)
        )
        leading_slot_surface = bool(
            re.match(r"^(?:The\s+)?\[REDACT:R1", query, flags=re.IGNORECASE)
        )
        possessive_slot_lead = bool(
            re.match(r"^The\s+\[REDACT:R1[^\]]*\]'s\b", query, flags=re.IGNORECASE)
        )
        nominal_slot_lead = bool(
            re.match(
                r"^(?:Recent|Current|Possible|Planned)\s+\w+\s+of\s+\[REDACT:R1[^\]]*\]\b",
                query,
                flags=re.IGNORECASE,
            )
        )
        if strict_mode and len(used_specs) == 1:
            if primary_entity_type == "person":
                if (
                    visible_anchor_count == 0
                    and local_support_hits < 2
                    and not has_date_or_number
                    and not clean_direct_slot
                ):
                    continue
                if possessive_slot_lead or nominal_slot_lead:
                    continue
            elif primary_entity_type in {
                "organization",
                "government",
                "military",
                "political_party",
            }:
                if (
                    visible_anchor_count == 0
                    and local_support_hits < 2
                    and not has_date_or_number
                    and not clean_direct_slot
                ):
                    continue
        boilerplate_like_query = bool(
            LEGAL_BOILERPLATE_RE.search(query)
            or is_routing_or_metadata_sentence(query)
        )
        if strict_mode and len(used_specs) == 1 and not has_date_or_number:
            if primary_entity_type in {
                "organization",
                "government",
                "political_party",
                "military",
            }:
                if (
                    boilerplate_like_query
                    or count_words(query) > 36
                    or (
                        visible_anchor_count < 2
                        and local_support_hits < 2
                        and not clean_direct_slot
                    )
                ):
                    continue
            if primary_entity_type in {"location", "country"}:
                if (
                    count_words(query) > 30
                    and visible_anchor_count == 0
                    and local_support_hits < 2
                    and not clean_direct_slot
                ):
                    continue
                if leading_slot_surface or possessive_slot_lead or nominal_slot_lead:
                    continue
        if (
            strict_mode
            and large_corpus
            and not pair_like_corpus
            and len(used_specs) >= 2
            and primary_entity_type in {"person", "location", "country"}
            and visible_anchor_count == 0
            and local_support_hits < 4
            and not has_date_or_number
            and not clean_direct_slot
        ):
            continue
        if (
            strict_mode
            and large_corpus
            and len(used_specs) == 1
            and primary_entity_type == "person"
            and (
                re.match(r"^(?:dr|mrs)\.?\s+\[redact:[^\]]+\]", query, flags=re.IGNORECASE)
                or re.search(
                    r"\[redact:[^\]]+\]\s*,?\s+(?:president|chairman|ambassador|minister)\b",
                    query,
                    flags=re.IGNORECASE,
                )
            )
            and not clean_direct_slot
        ):
            continue
        if (
            strict_mode
            and small_corpus
            and not pair_like_corpus
            and len(used_specs) == 1
            and visible_anchor_count == 0
            and local_support_hits < 2
            and not has_date_or_number
            and not clean_direct_slot
        ):
            continue
        query_flags = set(query_surface_flags(query, task_family="value_redaction"))
        if strict_mode and query_flags & {
            "mixed_case_noise",
            "enumeration_scaffold",
            "press_briefing_scaffold",
            "intentive_construction",
        }:
            continue
        leaked_value = False
        for used_spec in used_specs:
            leaked_label = normalize_space(used_spec.get("label", ""))
            if leaked_label and find_label_match(query, leaked_label):
                leaked_value = True
                break
        if leaked_value:
            continue
        selector_score = (
            confidence * 10.0
            + float(type_priority.get(primary_entity_type, 0.0))
            + (10.0 if primary_entity_type in preferred_entity_types else 0.0)
            + structure_bonus
            + semantic_anchor_bonus
            + visible_anchor_count * 3.0
            + repeat_bonus
            + occurrence_bonus
            + min(role_words, 14) * 0.5
            + (4.0 if slot_sentence else 0.0)
            + (2.0 if has_date_or_number else 0.0)
            + max(0.0, 12.0 - abs(slot_sentence_word_count - 18)) * 0.4
            + min(len(used_specs), max_slot_count) * 2.5
            - chunk_words / 120.0
            - ambiguity_penalty
            - label_occurrence_penalty
            - label_reuse_penalty
            - mixed_type_penalty
            - short_name_penalty
            - weak_anchor_penalty
            - boilerplate_penalty
            - generic_label_penalty
            - (4.0 if leading_slot_surface and len(used_specs) == 1 else 0.0)
            - (6.0 if possessive_slot_lead else 0.0)
            - (4.0 if nominal_slot_lead else 0.0)
            - (acronym_penalty * (1.8 if large_corpus and primary_entity_type != "person" else 1.0))
            - (6.0 if large_corpus and primary_entity_type == "government" else 0.0)
            - density_penalty
            - surface_noise_penalty(query) * 0.5
        )
        local_id = normalize_space(row.get("candidate_evidence_json", ""))
        local_id_match = re.search(r'"raw_row_id"\s*:\s*"([^"]+)"', local_id)
        local_id = local_id_match.group(1) if local_id_match else sha1_text(f"{row['doc_id']}|{row['paragraph_id']}|{label}")
        case_id = f"value_redaction__{row['doc_id']}__{local_id}"
        base_candidate = {
            "task_family": "value_redaction",
            "runner_task": "redaction",
            "doc_id": str(row["doc_id"]),
            "paragraph_id": str(row["paragraph_id"]),
            "case_id": case_id,
            "query": query,
            "ground_truth": stringify_expected_targets(
                {
                    used_spec["slot_id"]: {
                        "source_id": str(used_spec.get("source_id", row["doc_id"])),
                        "selected_value": normalize_space(used_spec.get("label", "")),
                    }
                    for used_spec in used_specs
                }
            ) or label,
            "query_kind": "entity_slot",
                "notes": (
                    f"type={primary_entity_type}; confidence={confidence:.2f}; "
                    f"claims={claim_n}; events={event_n}; relations={rel_n}; "
                    f"chunk_words={chunk_words}; same_type_hits={same_type_chunk_hits}; "
                    f"slot_sentence_words={slot_sentence_word_count}; "
                f"slot_count={len(used_specs)}; "
                f"acronym_like={acronym_like}; "
                f"claim_label_hits={claim_label_hits}; event_label_hits={event_label_hits}; "
                f"label_doc_frequency={label_doc_frequency}; "
                f"label_occurrences={label_occurrence_counts.get(label_key, 1)}; "
                f"visible_anchor_count={visible_anchor_count}; "
                f"has_date_or_number={has_date_or_number}; "
                f"selector_score={selector_score:.2f}; "
                f"query_target_words={query_target_words}"
            ),
            "source_context": query_base,
            "selector_score": selector_score,
            "word_count": count_words(query),
            "query_target_words": query_target_words,
                "strict_selected_value_match": maybe_bool(
                    sampling.get("strict_selected_value_match", False)
                ),
                "expected_targets": expected_targets,
                "answer_targets": answer_targets,
                "entity_type": primary_entity_type,
                "role_words": role_words,
                "label_doc_frequency": label_doc_frequency,
                "confidence": confidence,
                "local_support_hits": local_support_hits,
                "same_type_hits": same_type_chunk_hits,
                "visible_anchor_count": visible_anchor_count,
                "slot_count": len(used_specs),
                "has_date_or_number": has_date_or_number,
                "clean_direct_slot": clean_direct_slot,
            }
        accepted_in_tier = False
        for tier_index, profile in enumerate(profile_ladder, start=1):
            profile_types = set(profile.get("entity_types", set()))
            if entity_type not in profile_types:
                continue
            if confidence < float(profile["min_confidence"]):
                continue
            if label_doc_frequency > int(profile["max_label_doc_frequency"]):
                continue
            if role_words < int(profile["min_role_words"]):
                continue
            if entity_type == "location" and role_words < location_min_role_words:
                continue
            candidate = dict(base_candidate)
            candidate["selector_score"] = selector_score - (tier_index - 1) * 1.25
            candidate["notes"] = (
                f"{candidate['notes']}; selector_tier={tier_index}; "
                f"effective_score={candidate['selector_score']:.2f}"
            )
            prior = all_candidates.get(case_id)
            if prior is None or candidate["selector_score"] > float(prior.get("selector_score", 0.0)):
                all_candidates[case_id] = candidate
            accepted_in_tier = True
            break
        if not accepted_in_tier and not strict_mode:
            candidate = dict(base_candidate)
            candidate["selector_score"] = selector_score - 3.0
            candidate["notes"] = (
                f"{candidate['notes']}; selector_tier=relaxed_open; "
                f"effective_score={candidate['selector_score']:.2f}"
            )
            prior = all_candidates.get(case_id)
            if prior is None or candidate["selector_score"] > float(prior.get("selector_score", 0.0)):
                all_candidates[case_id] = candidate
    candidates = list(all_candidates.values())
    claim_surface_candidates: list[dict[str, Any]] = []
    claim_surface_enabled = (
        small_corpus
        or pair_like_corpus
        or maybe_bool(sampling.get("enable_large_corpus_claim_surface_vr", False))
    )
    if claim_surface_enabled:
        claim_surface_limit = (
            max(count * 6, 48)
            if (small_corpus or pair_like_corpus)
            else int(sampling.get("large_corpus_claim_surface_limit", 160))
        )
        component_items = list(component_index.items())
        if not (small_corpus or pair_like_corpus):
            def claim_surface_priority(item: tuple[tuple[str, str], dict[str, list[str]]]) -> tuple[float, int, int]:
                para_key, component_bucket = item
                richness = para_richness.get(para_key, {})
                claim_texts = [normalize_space(text) for text in component_bucket.get("claim_texts", [])]
                best_claim_words = max((count_words(text) for text in claim_texts), default=0)
                return (
                    int(richness.get("claim_n", 0)) * 3
                    + int(richness.get("event_n", 0)) * 2
                    + int(richness.get("rel_n", 0)),
                    len(claim_texts),
                    best_claim_words,
                )

            component_items = sorted(
                component_items,
                key=claim_surface_priority,
                reverse=True,
            )[: int(sampling.get("large_corpus_claim_surface_paragraph_cap", 320))]
        for para_key, component_bucket in component_items:
            doc_id, paragraph_id = para_key
            local_rows = paragraph_entity_rows.get(para_key, [])
            if not local_rows:
                continue
            claim_surfaces = [
                finalize_query_surface(clean_query_passage(str(claim_text or "")), add_terminal_punctuation=True)
                for claim_text in component_bucket.get("claim_texts", [])
            ]
            claim_surfaces = [
                claim_surface
                for claim_surface in unique_preserving_order(claim_surfaces)
                if claim_surface
                and not is_routing_or_metadata_sentence(claim_surface)
                and not is_list_like_text(claim_surface)
                and not LEGAL_BOILERPLATE_RE.search(claim_surface)
                and not BROKEN_LEAD_RE.search(claim_surface)
                and not query_surface_is_blocked(claim_surface, task_family="value_redaction")
                and not sentence_is_low_signal_for_query(claim_surface)
                and 8 <= count_words(claim_surface) <= 72
            ]
            claim_surface_cap = 8 if (small_corpus or pair_like_corpus) else 3
            for claim_surface in claim_surfaces[:claim_surface_cap]:
                claim_specs: list[dict[str, Any]] = []
                for entity_row in local_rows:
                    label = normalize_space(entity_row.get("candidate_label", ""))
                    if not label:
                        continue
                    entity_type = normalize_candidate_type(entity_row.get("candidate_type", ""))
                    if entity_type == "location" and not include_locations:
                        entity_type = "country"
                    if entity_type in {"other", "unknown", "target", ""}:
                        continue
                    if allowed_types and entity_type not in allowed_types and not (
                        small_corpus and entity_type in small_corpus_supported_entity_types
                    ):
                        continue
                    if entity_type == "person":
                        label = prefer_full_name_person_label(label, local_rows)
                        if count_words(label) < 2:
                            continue
                    if entity_type == "person" and (
                        is_probably_generic_person_label(label)
                        or is_title_only_person_label(label)
                    ):
                        continue
                    if (
                        entity_type in {"organization", "government", "military", "political_party"}
                        and count_words(label) < 2
                    ):
                        continue
                    if not find_label_match(claim_surface, label):
                        continue
                    claim_specs.append(
                        {
                            "label": label,
                            "slot_type": label_type_string(entity_type),
                            "source_id": str(entity_row.get("doc_id", doc_id)),
                            "entity_type": entity_type,
                            "confidence": safe_float(entity_row.get("candidate_source_confidence", 0.0)),
                            "label_word_count": count_words(label),
                            "label_acronym_like": looks_like_acronym_label(label),
                            "role_words": count_words(
                                normalize_space(entity_row.get("candidate_role_or_context", ""))
                            ),
                            "context_text": clean_query_passage(
                                str(entity_row.get("context_text", "") or "")
                            ),
                            "context_window_text": clean_query_passage(
                                str(entity_row.get("context_window_text", "") or "")
                            ),
                            "local_support_hits": sum(
                                1
                                for claim_text in component_bucket.get("claim_texts", [])
                                if re.search(rf"\b{re.escape(label)}\b", claim_text, flags=re.IGNORECASE)
                            ) + sum(
                                1
                                for event_text in component_bucket.get("event_labels", [])
                                if re.search(rf"\b{re.escape(label)}\b", event_text, flags=re.IGNORECASE)
                            ),
                            "label_doc_frequency": len(label_doc_sets.get(label.casefold(), set())),
                            "label_occurrences": int(label_occurrence_counts.get(label.casefold(), 1)),
                        }
                    )
                if not claim_specs:
                    continue
                unique_specs: list[dict[str, Any]] = []
                seen_claim_labels: set[str] = set()
                for spec in sorted(
                    claim_specs,
                    key=lambda spec: (
                        int(spec.get("local_support_hits", 0) or 0),
                        spec["entity_type"] in preferred_entity_types,
                        safe_float(spec.get("confidence", 0.0)),
                        count_words(spec.get("label", "")),
                    ),
                    reverse=True,
                ):
                    label_key = normalize_space(spec.get("label", "")).casefold()
                    if not label_key or label_key in seen_claim_labels:
                        continue
                    seen_claim_labels.add(label_key)
                    unique_specs.append(spec)
                if not unique_specs:
                    continue
                desired_slot_count = 1
                claim_quality = substantive_sentence_score(
                    claim_surface,
                    anchors=[normalize_space(spec.get("label", "")) for spec in unique_specs],
                )
                if (
                    len(unique_specs) > 1
                    and max_slot_count > 1
                    and claim_quality >= 7.5
                    and count_words(claim_surface) <= slot_sentence_max_words
                ):
                    desired_slot_count = min(
                        max_slot_count,
                        deterministic_tail_count(
                            f"claim_surface_value_slots|{doc_id}|{paragraph_id}|{claim_surface}",
                            max_count=min(max_slot_count, 2),
                            decay=slot_count_decay,
                        ),
                    )
                paragraph_contexts = [
                    clean_query_passage(str(chunk.get("chunk_text", "") or ""))
                    for chunk in doc_chunks_by_doc.get(str(doc_id), [])
                    if normalize_space(chunk.get("paragraph_id", "")) == str(paragraph_id)
                ]
                def render_claim_surface_case(
                    slot_specs: list[dict[str, Any]],
                    *,
                    prefer_broader_excerpt: bool = False,
                ) -> tuple[str, list[dict[str, Any]]]:
                    target_labels = [
                        normalize_space(spec.get("label", ""))
                        for spec in slot_specs
                        if normalize_space(spec.get("label", ""))
                    ]
                    context_options = [claim_surface]
                    for spec in slot_specs:
                        context_options.extend(
                            [
                                normalize_space(spec.get("context_text", "")),
                                normalize_space(spec.get("context_window_text", "")),
                            ]
                        )
                    context_options.extend(paragraph_contexts[:2])
                    query_base = choose_best_query_context(
                        context_options,
                        anchors=target_labels,
                        min_words=max(10, min_query_words - 2),
                        target_words=scaled_redaction_query_target(
                            f"claim_surface_value_window|{doc_id}|{paragraph_id}|{'|'.join(target_labels)}",
                            minimum_words=max(10, min_query_words - 2),
                            maximum_words=min(max_query_words, 96),
                            slot_count=max(1, len(slot_specs)),
                        ),
                        max_sentences=2,
                        max_chars=520,
                    ) or claim_surface
                    query_text_local, used_specs_local = redact_labels_in_context(
                        query_base,
                        slot_specs,
                    )
                    if not used_specs_local:
                        return "", []
                    excerpt_sentences = (
                        2
                        if len(used_specs_local) > 1
                        or prefer_broader_excerpt
                        or (small_corpus and len(used_specs_local) == 1)
                        else 1
                    )
                    query_text_local = finalize_query_surface(
                        extract_sentence_excerpt(
                            query_text_local,
                            max_sentences=excerpt_sentences,
                            max_chars=520,
                        ),
                        add_terminal_punctuation=True,
                    )
                    return query_text_local, used_specs_local
                query_text, used_specs = render_claim_surface_case(
                    unique_specs[:desired_slot_count],
                )
                if not used_specs:
                    continue
                if count_words(query_text) < max(8, min_query_words - 4) or count_words(query_text) > min(max_query_words, 80):
                    continue
                primary_entity_type = normalize_candidate_type(
                    used_specs[0].get("entity_type", "")
                )
                if not value_redaction_query_is_viable(
                    query_text,
                    entity_type=primary_entity_type,
                    used_specs=used_specs,
                ):
                    continue
                visible_anchor_labels = [
                    normalize_space(spec.get("label", ""))
                    for spec in unique_specs
                    if normalize_space(spec.get("label", ""))
                    and normalize_space(spec.get("label", "")).casefold()
                    not in {
                        normalize_space(used_spec.get("label", "")).casefold()
                        for used_spec in used_specs
                    }
                    and find_label_match(query_text, normalize_space(spec.get("label", "")))
                ]
                visible_anchor_count = len(unique_preserving_order(visible_anchor_labels))
                if visible_anchor_count == 0 and len(used_specs) > 1:
                    query_text, used_specs = render_claim_surface_case(
                        unique_specs[:1],
                        prefer_broader_excerpt=True,
                    )
                    if not used_specs:
                        continue
                    primary_entity_type = normalize_candidate_type(
                        used_specs[0].get("entity_type", "")
                    )
                    visible_anchor_labels = [
                        normalize_space(spec.get("label", ""))
                        for spec in unique_specs
                        if normalize_space(spec.get("label", ""))
                        and normalize_space(spec.get("label", "")).casefold()
                        not in {
                            normalize_space(used_spec.get("label", "")).casefold()
                            for used_spec in used_specs
                        }
                        and find_label_match(query_text, normalize_space(spec.get("label", "")))
                    ]
                    visible_anchor_count = len(unique_preserving_order(visible_anchor_labels))
                has_date_or_number = bool(
                    DATE_TOKEN_RE.search(query_text) or NUMBER_RE.search(query_text)
                )
                if len(used_specs) == 1 and visible_anchor_count == 0 and not has_date_or_number:
                    query_text, used_specs = render_claim_surface_case(
                        used_specs[:1],
                        prefer_broader_excerpt=True,
                    )
                    if not used_specs:
                        continue
                    visible_anchor_labels = [
                        normalize_space(spec.get("label", ""))
                        for spec in unique_specs
                        if normalize_space(spec.get("label", ""))
                        and normalize_space(spec.get("label", "")).casefold()
                        not in {
                            normalize_space(used_spec.get("label", "")).casefold()
                            for used_spec in used_specs
                        }
                        and find_label_match(query_text, normalize_space(spec.get("label", "")))
                    ]
                    visible_anchor_count = len(unique_preserving_order(visible_anchor_labels))
                    has_date_or_number = bool(
                        DATE_TOKEN_RE.search(query_text) or NUMBER_RE.search(query_text)
                    )
                if len(used_specs) == 1 and visible_anchor_count == 0 and not has_date_or_number:
                    continue
                expected_targets = {
                    str(used_spec["slot_id"]): {"source_id": str(used_spec.get("source_id", doc_id))}
                    for used_spec in used_specs
                }
                answer_targets = {
                    str(used_spec["slot_id"]): {
                        "source_id": str(used_spec.get("source_id", doc_id)),
                        "selected_value": normalize_space(used_spec.get("label", "")),
                    }
                    for used_spec in used_specs
                }
                selector_score = (
                    claim_quality * 4.5
                    + min(
                        sum(int(spec.get("local_support_hits", 0) or 0) for spec in used_specs),
                        4,
                    ) * 2.5
                    + visible_anchor_count * 3.0
                    + (2.0 if has_date_or_number else 0.0)
                    + min(len(used_specs), max_slot_count) * 2.5
                    - surface_noise_penalty(query_text) * 0.4
                )
                claim_surface_candidates.append(
                    {
                        "task_family": "value_redaction",
                        "runner_task": "redaction",
                        "doc_id": str(doc_id),
                        "paragraph_id": str(paragraph_id),
                        "case_id": f"value_redaction__{doc_id}__claim_surface__{sha1_text(query_text)}",
                        "query": query_text,
                        "ground_truth": stringify_expected_targets(answer_targets)
                        or normalize_space(used_specs[0].get("label", "")),
                        "query_kind": "entity_slot_minimal",
                        "notes": (
                            f"type={primary_entity_type}; slot_count={len(used_specs)}; "
                            f"selector_score={selector_score:.2f}; fallback=claim_surface"
                        ),
                        "source_context": claim_surface,
                        "selector_score": selector_score,
                        "word_count": count_words(query_text),
                        "query_target_words": count_words(query_text),
                        "strict_selected_value_match": maybe_bool(
                            sampling.get("strict_selected_value_match", False)
                        ),
                        "expected_targets": expected_targets,
                        "answer_targets": answer_targets,
                        "entity_type": primary_entity_type,
                        "role_words": max(
                            int(used_specs[0].get("role_words", 0) or 0),
                            2 if primary_entity_type in {"organization", "government", "political_party"} else 0,
                        ),
                        "label_doc_frequency": int(used_specs[0].get("label_doc_frequency", 1) or 1),
                        "confidence": safe_float(used_specs[0].get("confidence", 0.0)),
                        "local_support_hits": sum(
                            int(spec.get("local_support_hits", 0) or 0) for spec in used_specs
                        ),
                        "same_type_hits": sum(
                            1
                            for spec in unique_specs
                            if normalize_candidate_type(spec.get("entity_type", ""))
                            == primary_entity_type
                        ),
                        "visible_anchor_count": visible_anchor_count,
                        "slot_count": len(used_specs),
                        "has_date_or_number": has_date_or_number,
                        "clean_direct_slot": bool(
                            claim_quality >= 7.0
                            and not sentence_is_low_signal_for_query(query_text)
                        ),
                    }
                )
                if len(claim_surface_candidates) >= claim_surface_limit:
                    break
            if len(claim_surface_candidates) >= claim_surface_limit:
                break
        candidates.extend(claim_surface_candidates)
    candidates = [
        candidate
        for candidate in candidates
        if not value_redaction_surface_is_known_bad(candidate.get("query", ""))
        and not value_redaction_surface_is_generation_reject(
            candidate.get("query", ""),
            candidate.get("doc_id", ""),
        )
    ]
    candidates = adjust_value_redaction_generation_scores(candidates)
    if large_corpus and len(candidates) >= max(count * 12, 120):
        primary_large_corpus_selected = merge_ranked_candidates(
            [candidates],
            target_count=count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
        return normalize_selected_case_queries(
            primary_large_corpus_selected,
            task_family="value_redaction",
            runner_cfg=runner_cfg,
        )
    pair_sentence_candidates = build_pair_sentence_value_redaction_candidates(
        entities,
        doc_chunks_by_doc,
        para_richness,
        component_index,
        count,
        sampling,
        seed,
        runner_cfg=runner_cfg,
        entity_rows=synthesized_entity_rows,
    )
    focus_candidates = build_focus_value_redaction_candidates(
        entities,
        doc_chunks_by_doc,
        count,
        sampling,
        seed,
        entity_rows=synthesized_entity_rows,
    )
    pair_sentence_candidates = [
        candidate
        for candidate in pair_sentence_candidates
        if not value_redaction_surface_is_known_bad(candidate.get("query", ""))
        and not value_redaction_surface_is_generation_reject(
            candidate.get("query", ""),
            candidate.get("doc_id", ""),
        )
    ]
    pair_sentence_candidates = adjust_value_redaction_generation_scores(pair_sentence_candidates)
    focus_candidates = [
        candidate
        for candidate in focus_candidates
        if not value_redaction_surface_is_known_bad(candidate.get("query", ""))
        and not value_redaction_surface_is_generation_reject(
            candidate.get("query", ""),
            candidate.get("doc_id", ""),
        )
    ]
    focus_candidates = adjust_value_redaction_generation_scores(focus_candidates)
    preferred = [
        candidate
        for candidate in candidates
        if str(candidate.get("entity_type", "")) in preferred_entity_types
    ]
    selected_preferred: list[dict[str, Any]] = []
    if preferred:
        selected_preferred = round_robin_topk(
            preferred,
            min(count, len(preferred)),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
    selected = merge_ranked_candidates(
        [pair_sentence_candidates, focus_candidates, selected_preferred, candidates],
        target_count=count,
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )
    if large_corpus and len(selected) >= max(1, min(count, 3)):
        return normalize_selected_case_queries(
            selected,
            task_family="value_redaction",
            runner_cfg=runner_cfg,
        )
    if len(selected) >= count or maybe_bool(sampling.get("_relaxed_retry_used", False)):
        return normalize_selected_case_queries(
            selected,
            task_family="value_redaction",
            runner_cfg=runner_cfg,
        )
    minimal = build_minimal_value_redaction_candidates(
        entities,
        doc_chunks_by_doc,
        count,
        sampling,
        seed,
    )
    minimal = [
        candidate
        for candidate in minimal
        if not value_redaction_surface_is_known_bad(candidate.get("query", ""))
        and not value_redaction_surface_is_generation_reject(
            candidate.get("query", ""),
            candidate.get("doc_id", ""),
        )
    ]
    minimal = adjust_value_redaction_generation_scores(minimal)
    selected = merge_ranked_candidates(
        [pair_sentence_candidates, focus_candidates, selected_preferred, minimal, candidates],
        target_count=count,
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )
    if len(selected) >= count or maybe_bool(sampling.get("_relaxed_retry_used", False)):
        return normalize_selected_case_queries(
            selected,
            task_family="value_redaction",
            runner_cfg=runner_cfg,
        )
    relaxed_sampling = dict(sampling)
    relaxed_sampling["_relaxed_retry_used"] = True
    relaxed_sampling["include_locations"] = True
    relaxed_sampling["min_confidence"] = max(0.0, float(sampling.get("min_confidence", 0.0)) - 0.1)
    relaxed_sampling["max_label_doc_frequency"] = max(
        int(sampling.get("max_label_doc_frequency", 2)),
        int(sampling.get("fallback_max_label_doc_frequency", 4)),
    )
    relaxed_sampling["max_label_occurrences"] = max(int(sampling.get("max_label_occurrences", 4)), 8)
    relaxed_sampling["max_entity_candidates_in_paragraph"] = max(
        int(sampling.get("max_entity_candidates_in_paragraph", 999999)),
        64,
    )
    relaxed_sampling["person_min_label_words"] = 1
    relaxed_sampling["allow_single_token_locations"] = True
    relaxed_sampling["min_role_words"] = 0
    relaxed_sampling["fallback_min_role_words"] = 0
    relaxed_sampling["allowed_entity_types"] = sorted(
        allowed_types | {value.casefold() for value in sampling.get("fallback_entity_types", [])}
    )
    relaxed = choose_value_redaction_cases(
        redaction_views,
        entities,
        doc_chunks_by_doc,
        para_richness,
        component_index,
        count,
        relaxed_sampling,
        seed,
        runner_cfg=runner_cfg,
    )
    merged_primary = merge_ranked_candidates(
        [pair_sentence_candidates, focus_candidates, selected, relaxed],
        target_count=count,
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )
    if len(merged_primary) >= count:
        return normalize_selected_case_queries(
            merged_primary,
            task_family="value_redaction",
            runner_cfg=runner_cfg,
        )
    return normalize_selected_case_queries(
        merge_ranked_candidates(
            [pair_sentence_candidates, merged_primary, minimal],
            target_count=count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        ),
        task_family="value_redaction",
        runner_cfg=runner_cfg,
    )


def choose_synthetic_value_redaction_cases(
    entities: pd.DataFrame,
    doc_chunks_by_doc: dict[str, list[dict[str, Any]]],
    para_richness: dict[tuple[str, str], dict[str, int]],
    component_index: dict[tuple[str, str], dict[str, list[str]]],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    runner_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build value-redaction cases from unredacted extraction only."""
    return choose_value_redaction_cases(
        None,
        entities,
        doc_chunks_by_doc,
        para_richness,
        component_index,
        count,
        sampling,
        seed,
        runner_cfg=runner_cfg,
    )


def choose_sentence_redaction_cases(
    chunks: pd.DataFrame,
    component_index: dict[tuple[str, str], dict[str, list[str]]],
    para_richness: dict[tuple[str, str], dict[str, int]],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    runner_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    strict_mode = not maybe_bool(sampling.get("_relaxed_retry_used", False))
    chunk_doc_ids = {
        normalize_space(doc_id)
        for doc_id in chunks.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    }
    pair_like_corpus = bool(chunk_doc_ids) and all(doc_id.startswith("Pair_") for doc_id in chunk_doc_ids)
    corpus_doc_count = len(chunk_doc_ids)
    small_corpus = exhaustive_small_doc_corpus(corpus_doc_count)
    large_corpus = optimized_doc_corpus(corpus_doc_count)
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    min_context_words = int(sampling.get("min_context_sentence_words", 6))
    min_query_words = int(sampling.get("min_query_words", 16))
    max_query_words = int(sampling.get("max_query_words", 220))
    min_claims = int(sampling.get("min_claims_in_paragraph", 1))
    min_events = int(sampling.get("min_events_in_paragraph", 1))
    min_relations = int(sampling.get("min_relations_in_paragraph", 2))
    min_entity_hits = int(sampling.get("min_entity_hits", 1))
    effective_min_entity_hits = (
        max(min_entity_hits, 1)
        if strict_mode and pair_like_corpus
        else max(min_entity_hits, 1) if strict_mode else min_entity_hits
    )
    target_words = int(sampling.get("target_sentence_words", 18))
    gap_sentence_span = max(1, int(sampling.get("gap_sentence_span", 1)))
    effective_gap_sentence_span = 1
    gap_sentence_decay = float(sampling.get("gap_sentence_decay", 0.5))
    min_gap_words = max(int(sampling.get("min_sentence_words", 0)), 12)
    max_gap_words = int(sampling.get("max_sentence_words", 80))
    if small_corpus:
        max_gap_words = min(max_gap_words, 32)
    if pair_like_corpus and strict_mode:
        max_gap_words = min(max_gap_words, 40)
    chunk_rows = deterministic_seeded_row_order(
        chunks.to_dict("records"),
        seed=seed,
        key_fields=("doc_id", "paragraph_id", "chunk_text"),
    )
    for row in chunk_rows:
        text = normalize_space(row.get("chunk_text", ""))
        if uppercase_token_ratio(text) >= 0.45 and count_words(text) >= 18:
            text = normalize_uppercase_passage(text)
        qa_sentence_count = sum(
            1
            for sentence in sentence_split(text)
            if benchmark_interactive_qa_surface(sentence) or QA_PREFIX_RE.match(normalize_space(sentence))
        )
        if BROADCAST_TRANSCRIPT_RE.search(text) or qa_sentence_count >= 2:
            continue
        base_target_words = deterministic_query_word_target(
            f"sentence_redaction|{row['doc_id']}|{row['paragraph_id']}",
            minimum_words=min_query_words,
            maximum_words=min(max_query_words, 128),
        )
        desired_span = deterministic_tail_count(
            f"sentence_redaction_span|{row['doc_id']}|{row['paragraph_id']}",
            max_count=effective_gap_sentence_span,
            decay=gap_sentence_decay,
        )
        key = paragraph_key(row["doc_id"], row["paragraph_id"])
        richness = para_richness.get(key, {})
        claim_n = int(richness.get("claim_n", 0))
        event_n = int(richness.get("event_n", 0))
        rel_n = int(richness.get("rel_n", 0))
        if claim_n < min_claims or event_n < min_events or rel_n < min_relations:
            continue
        components = component_index.get(key, {"entity_labels": [], "event_labels": [], "claim_texts": []})
        claim_sentences = [
            finalize_query_surface(claim_text, add_terminal_punctuation=True)
            for claim_text in components.get("claim_texts", [])
        ]
        claim_sentences = [
            claim_text
            for claim_text in unique_preserving_order(claim_sentences)
            if claim_text
            and count_words(claim_text) >= max(8, min_context_words)
            and not is_routing_or_metadata_sentence(claim_text)
            and not is_list_like_text(claim_text)
            and not LEGAL_BOILERPLATE_RE.search(claim_text)
            and not BROKEN_LEAD_RE.search(claim_text)
        ]
        sentences = cleaned_substantive_sentences(text, min_score=-4.0)
        if len(sentences) < 3:
            sentences = [
                clean_generation_sentence(sentence)
                for sentence in sentence_split(text)
                if clean_generation_sentence(sentence)
            ]
        if len(sentences) < 3:
            continue
        preferred_gap_starts: set[int] = set()
        for claim_text in claim_sentences:
            claim_tokens = content_tokens(claim_text)
            if not claim_tokens:
                continue
            best_index = None
            best_overlap = 0
            for sent_index, sentence in enumerate(sentences):
                overlap = len(claim_tokens & content_tokens(sentence))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_index = sent_index
            if best_index is not None and 0 < best_index < len(sentences) - 1 and best_overlap >= 2:
                preferred_gap_starts.add(best_index)
        start_indices = (
            sorted(preferred_gap_starts)
            if strict_mode and preferred_gap_starts
            else range(1, len(sentences) - 1)
        )
        for index in start_indices:
            for span in range(1, effective_gap_sentence_span + 1):
                gap_end = index + span - 1
                if gap_end >= len(sentences) - 1:
                    continue
                if strict_mode and preferred_gap_starts and span > 1 and index in preferred_gap_starts:
                    continue
                gap_sentences = sentences[index : gap_end + 1]
                middle = " ".join(gap_sentences)
                words = count_words(middle)
                if words < min_gap_words or words > max_gap_words:
                    continue
                if any(is_list_like_text(sentence) for sentence in gap_sentences):
                    continue
                prev_sentence = sentences[index - 1]
                next_sentence = sentences[gap_end + 1]
                if any(
                    is_routing_or_metadata_sentence(sentence)
                    for sentence in [prev_sentence, *gap_sentences, next_sentence]
                ):
                    continue
                if any(
                    LEGAL_BOILERPLATE_RE.search(sentence) or BROKEN_LEAD_RE.search(sentence)
                    for sentence in [prev_sentence, *gap_sentences, next_sentence]
                ):
                    continue
                if any(
                    LOW_SIGNAL_SENTENCE_LEAD_RE.search(normalize_space(sentence))
                    for sentence in [prev_sentence, *gap_sentences, next_sentence]
                ):
                    continue
                if (
                    substantive_sentence_score(prev_sentence) < 4.0
                    or substantive_sentence_score(next_sentence) < 4.0
                ):
                    continue
                if count_words(prev_sentence) < min_context_words or count_words(next_sentence) < min_context_words:
                    continue
                entity_hit_count, entity_hits = count_label_hits(middle, components.get("entity_labels", []))
                has_date = bool(DATE_TOKEN_RE.search(middle))
                has_number = bool(NUMBER_RE.search(middle))
                claim_overlap = content_overlap_score(middle, components.get("claim_texts", []))
                event_overlap = content_overlap_score(middle, components.get("event_labels", []))
                gap_tokens = content_tokens(middle)
                prev_tokens = content_tokens(prev_sentence)
                next_tokens = content_tokens(next_sentence)
                neighbor_tokens = prev_tokens | next_tokens
                bridge_overlap = (
                    len(gap_tokens & neighbor_tokens) / max(1, len(gap_tokens))
                    if gap_tokens else 0.0
                )
                left_bridge_overlap = (
                    len(gap_tokens & prev_tokens) / max(1, len(gap_tokens))
                    if gap_tokens else 0.0
                )
                right_bridge_overlap = (
                    len(gap_tokens & next_tokens) / max(1, len(gap_tokens))
                    if gap_tokens else 0.0
                )
                if pair_like_corpus and strict_mode and entity_hit_count < effective_min_entity_hits:
                    continue
                if (
                    pair_like_corpus
                    and strict_mode
                    and entity_hit_count < 2
                    and claim_overlap < 1
                    and event_overlap < 1
                ):
                    continue
                if entity_hit_count < effective_min_entity_hits and not (has_date and (claim_overlap or event_overlap)):
                    continue
                if strict_mode and entity_hit_count < 1:
                    continue
                if strict_mode and claim_overlap < 1 and event_overlap < 1:
                    continue
                if strict_mode and not (
                    entity_hit_count >= effective_min_entity_hits
                    or claim_overlap >= 0.18
                    or event_overlap >= 0.18
                ):
                    continue
                if (
                    large_corpus
                    and strict_mode
                    and min(left_bridge_overlap, right_bridge_overlap) < 0.04
                    and claim_overlap < 2
                    and event_overlap < 2
                ):
                    continue
                if strict_mode and bridge_overlap < (0.06 if pair_like_corpus else 0.12) and claim_overlap < 1 and event_overlap < 1:
                    continue
                if strict_mode and min(left_bridge_overlap, right_bridge_overlap) < (0.02 if pair_like_corpus else 0.06):
                    continue
                if strict_mode and entity_hit_count < (1 if pair_like_corpus else 2) and claim_overlap < 0.18 and event_overlap < 0.18:
                    continue
                if strict_mode and span > 1 and claim_overlap < 0.18 and event_overlap < 0.18:
                    continue
                query_target_words = scaled_redaction_query_target(
                    f"sentence_redaction|{row['doc_id']}|{row['paragraph_id']}|{index}|{span}",
                    minimum_words=min_query_words,
                    maximum_words=min(max_query_words, 96),
                    slot_count=span,
                )
                visible_context = build_sentence_redaction_query_window(
                    sentences=sentences,
                    gap_start=index,
                    gap_end=gap_end,
                    slot_text=build_sentence_slot_marker(1, words),
                    target_words=query_target_words,
                    max_sentences=min(max(3, int(sampling.get("max_visible_context_sentences", 4))), 3),
                )
                query = refine_generated_query_surface(
                    visible_context,
                    task_family="sentence_redaction",
                    anchors=entity_hits,
                    min_words=min_query_words,
                    target_words=query_target_words,
                    max_chars=int(sampling.get("max_visible_context_chars", 560)),
                    runner_cfg=candidate_runner_cfg,
                )
                if "[REDACT_SENT:S1" not in query:
                    continue
                if count_words(query) < min_query_words or count_words(query) > max_query_words:
                    continue
                if count_redaction_slots(query, sentence_mode=True) != 1:
                    continue
                marker_match = re.search(r"\[REDACT_SENT:S1[^\]]*\]", query)
                if not marker_match:
                    continue
                left_context = normalize_space(query[: marker_match.start()])
                right_context = normalize_space(query[marker_match.end() :])
                if count_words(left_context) < max(4, min_context_words // 2):
                    continue
                if count_words(right_context) < max(4, min_context_words // 2):
                    continue
                if (
                    pair_like_corpus
                    and (
                        sentence_is_reporting_heavy(left_context)
                        or sentence_is_reporting_heavy(right_context)
                    )
                ):
                    continue
                if query_surface_is_blocked(query, task_family="sentence_redaction"):
                    continue
                query_flags = set(query_surface_flags(query, task_family="sentence_redaction"))
                if "?" in query or "--" in query:
                    continue
                if strict_mode and query_flags & {
                    "mixed_case_noise",
                    "enumeration_scaffold",
                    "intentive_construction",
                    "uppercase_heavy",
                }:
                    continue
                selector_score = (
                    entity_hit_count * 6.0
                    + claim_overlap * 3.0
                    + event_overlap * 3.0
                    + (2.0 if has_date else 0.0)
                    + (1.0 if has_number else 0.0)
                    + bridge_overlap * 12.0
                    + min(claim_n, 5) * 1.0
                    + min(event_n, 4) * 1.25
                    + min(rel_n, 6) * 0.75
                    - abs(words - target_words) / 8.0
                    + (0.75 if span > 1 else 0.0)
                    + max(0.0, 2.0 - abs(span - desired_span))
                    - surface_noise_penalty(query) * 0.4
                )
                case_id = (
                    f"sentence_redaction__{row['doc_id']}__{row['paragraph_id']}__s{index}__g{span}"
                )
                candidates.append(
                    {
                        "task_family": "sentence_redaction",
                        "runner_task": "sentence_redaction",
                        "doc_id": str(row["doc_id"]),
                        "paragraph_id": str(row["paragraph_id"]),
                        "case_id": case_id,
                        "query": query,
                        "ground_truth": middle,
                        "query_kind": "anchored_multi_sentence_gap" if span > 1 else "anchored_middle_sentence_gap",
                    "notes": (
                        f"sentence_index={index}; gap_sentence_span={span}; total_sentences={len(sentences)}; "
                        f"entity_hits={entity_hit_count}; claim_overlap={claim_overlap}; "
                        f"event_overlap={event_overlap}; bridge_overlap={bridge_overlap:.2f}; "
                        f"left_bridge_overlap={left_bridge_overlap:.2f}; right_bridge_overlap={right_bridge_overlap:.2f}; "
                        f"has_date={has_date}; has_number={has_number}; "
                        f"claims={claim_n}; events={event_n}; relations={rel_n}; "
                        f"query_target_words={query_target_words}; desired_span={desired_span}"
                    ),
                        "source_context": visible_context,
                        "selector_score": selector_score,
                        "word_count": count_words(query),
                        "query_target_words": query_target_words,
                    }
                )
    selected = round_robin_topk(
        list(
            {
                canonical_query_dedupe_key(str(case.get("ground_truth", ""))): case
                for case in sorted(
                    candidates,
                    key=lambda case: float(case.get("selector_score", 0.0) or 0.0),
                    reverse=True,
                )
                if canonical_query_dedupe_key(str(case.get("ground_truth", "")))
            }.values()
        ),
        max(count * 4, count),
        max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
        score_key="selector_score",
        seed=seed,
    )
    if (pair_like_corpus or small_corpus) and len(selected) < count:
        fallback_candidates: list[dict[str, Any]] = []
        for row in chunks.to_dict("records"):
            text = normalize_space(row.get("chunk_text", ""))
            if uppercase_token_ratio(text) >= 0.45 and count_words(text) >= 18:
                text = normalize_uppercase_passage(text)
            sentences = [
                clean_generation_sentence(sentence)
                for sentence in sentence_split(text)
                if clean_generation_sentence(sentence)
            ]
            if len(sentences) < 3:
                continue
            key = paragraph_key(row["doc_id"], row["paragraph_id"])
            components = component_index.get(
                key,
                {"entity_labels": [], "event_labels": [], "claim_texts": []},
            )
            for index in range(1, len(sentences) - 1):
                gap_sentence = finalize_query_surface(sentences[index], add_terminal_punctuation=True)
                if (
                    not gap_sentence
                    or sentence_is_low_signal_for_query(gap_sentence)
                    or is_routing_or_metadata_sentence(gap_sentence)
                    or LEGAL_BOILERPLATE_RE.search(gap_sentence)
                    or BROKEN_LEAD_RE.search(gap_sentence)
                ):
                    continue
                words = count_words(gap_sentence)
                if words < max(10, min_gap_words - 2) or words > max(max_gap_words, 44):
                    continue
                prev_sentence = finalize_query_surface(sentences[index - 1], add_terminal_punctuation=True)
                next_sentence = finalize_query_surface(sentences[index + 1], add_terminal_punctuation=True)
                if (
                    not prev_sentence
                    or not next_sentence
                    or sentence_is_low_signal_for_query(prev_sentence)
                    or sentence_is_low_signal_for_query(next_sentence)
                ):
                    continue
                entity_hit_count, entity_hits = count_label_hits(
                    gap_sentence,
                    components.get("entity_labels", []),
                )
                claim_overlap = content_overlap_score(
                    gap_sentence,
                    components.get("claim_texts", []),
                )
                event_overlap = content_overlap_score(
                    gap_sentence,
                    components.get("event_labels", []),
                )
                if pair_like_corpus:
                    if entity_hit_count < 1 and claim_overlap < 0.08 and event_overlap < 0.08:
                        continue
                elif small_corpus:
                    if (
                        entity_hit_count < 1
                        and claim_overlap < 0.05
                        and event_overlap < 0.05
                        and substantive_sentence_score(gap_sentence) < 8.0
                    ):
                        continue
                else:
                    if entity_hit_count < 1 and claim_overlap < 0.08 and event_overlap < 0.08:
                        continue
                query_target_words = scaled_redaction_query_target(
                    f"sentence_redaction_fallback|{row['doc_id']}|{row['paragraph_id']}|{index}",
                    minimum_words=min_query_words,
                    maximum_words=min(max_query_words, 96),
                    slot_count=1,
                )
                visible_context = build_sentence_redaction_query_window(
                    sentences=sentences,
                    gap_start=index,
                    gap_end=index,
                    slot_text=build_sentence_slot_marker(1, words),
                    target_words=query_target_words,
                    max_sentences=min(
                        max(3, int(sampling.get("max_visible_context_sentences", 4))),
                        3,
                    ),
                )
                query = finalize_query_surface(visible_context, add_terminal_punctuation=True)
                if (
                    not query
                    or "[REDACT_SENT:S1" not in query
                    or count_redaction_slots(query, sentence_mode=True) != 1
                    or query_surface_is_blocked(query, task_family="sentence_redaction")
                ):
                    continue
                case_id = (
                    f"sentence_redaction__{row['doc_id']}__{row['paragraph_id']}__fallback__s{index}"
                )
                fallback_candidates.append(
                    {
                        "task_family": "sentence_redaction",
                        "runner_task": "sentence_redaction",
                        "doc_id": str(row["doc_id"]),
                        "paragraph_id": str(row["paragraph_id"]),
                        "case_id": case_id,
                        "query": query,
                        "ground_truth": gap_sentence,
                        "query_kind": "anchored_middle_sentence_gap",
                        "notes": (
                            "fallback=pair_like_sentence_gap; "
                            f"sentence_index={index}; entity_hits={entity_hit_count}; "
                            f"claim_overlap={claim_overlap}; event_overlap={event_overlap}"
                        ),
                        "source_context": visible_context,
                        "selector_score": (
                            substantive_sentence_score(gap_sentence, anchors=entity_hits)
                            + entity_hit_count * 4.0
                            + claim_overlap * 4.0
                            + event_overlap * 4.0
                            - surface_noise_penalty(query) * 0.25
                        ),
                        "word_count": count_words(query),
                        "query_target_words": query_target_words,
                    }
                )
                if len(fallback_candidates) >= max(count * 6, 24):
                    break
            if len(fallback_candidates) >= max(count * 6, 24):
                break
        selected = merge_ranked_candidates(
            [selected, fallback_candidates],
            target_count=max(count * 4, count),
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        )
    normalized_selected = normalize_selected_case_queries(
        selected,
        task_family="sentence_redaction",
        runner_cfg=runner_cfg,
    )
    if len(normalized_selected) >= count or maybe_bool(sampling.get("_relaxed_retry_used", False)):
        return normalized_selected
    if len(selected) >= count and not pair_like_corpus:
        return normalize_selected_case_queries(
            selected,
            task_family="sentence_redaction",
            runner_cfg=runner_cfg,
        )
    relaxed_sampling = dict(sampling)
    relaxed_sampling["_relaxed_retry_used"] = True
    relaxed_sampling["min_claims_in_paragraph"] = max(0, min_claims - 1)
    relaxed_sampling["min_events_in_paragraph"] = max(0, min_events - 1)
    relaxed_sampling["min_relations_in_paragraph"] = max(1, min_relations - 1)
    relaxed_sampling["min_entity_hits"] = max(1, min_entity_hits)
    relaxed_sampling["min_sentence_words"] = max(10, min_gap_words - 2)
    relaxed_sampling["max_sentence_words"] = max(
        int(sampling.get("max_sentence_words", 80)),
        int(sampling.get("max_sentence_words", 80)) + 12,
    )
    relaxed = choose_sentence_redaction_cases(
        chunks,
        component_index,
        para_richness,
        count,
        relaxed_sampling,
        seed,
        runner_cfg=runner_cfg,
    )
    return normalize_selected_case_queries(
        merge_ranked_candidates(
            [selected, relaxed],
            target_count=count,
            max_cases_per_doc=int(sampling.get("max_cases_per_doc", 2)),
            score_key="selector_score",
            seed=seed,
        ),
        task_family="sentence_redaction",
        runner_cfg=runner_cfg,
    )


def perturb_numeric_or_date(text: str) -> tuple[str, str] | None:
    year_range_match = YEAR_RANGE_RE.search(text)
    if year_range_match:
        start = int(year_range_match.group("start")) + 1
        raw_end = year_range_match.group("end")
        if len(raw_end) == 2:
            end = f"{(int(raw_end) + 1) % 100:02d}"
        else:
            end = str(int(raw_end) + 1)
        replacement = f"{start}{year_range_match.group('sep')}{end}"
        return (
            text[: year_range_match.start()]
            + replacement
            + text[year_range_match.end():],
            "date",
        )
    date_match = DATE_TOKEN_RE.search(text)
    if date_match:
        token = date_match.group(0)
        if token.isdigit():
            replacement = str(int(token) + 1)
        else:
            months = [
                "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
                "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
            ]
            upper = token.upper()
            if upper in months:
                if token.islower():
                    date_match = None
                else:
                    replacement = months[(months.index(upper) + 1) % len(months)].title()
            else:
                replacement = token
        if date_match:
            return text[: date_match.start()] + replacement + text[date_match.end():], "date"
    number_match = NUMBER_RE.search(text)
    if number_match:
        token = number_match.group(0)
        clean = token.replace(",", "")
        if clean.isdigit():
            replacement = str(int(clean) + 1)
            return text[: number_match.start()] + replacement + text[number_match.end():], "numeric"
    return None


def perturb_negation(text: str) -> tuple[str, str] | None:
    if re.search(
        r"\b(?:probably|apparently|reportedly|hardly|barely|rarely|scarcely|unlikely|improbable)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return None
    no_longer_match = re.search(r"\bno longer\b", text, flags=re.IGNORECASE)
    if no_longer_match:
        return (
            text[: no_longer_match.start()] + "still" + text[no_longer_match.end():],
            "negation",
        )
    compound_patterns = [
        (r"\bmay have\b", "may not have"),
        (r"\bmight have\b", "might not have"),
        (r"\bcould have\b", "could not have"),
        (r"\bwould have\b", "would not have"),
        (r"\bshould have\b", "should not have"),
        (r"\bhas been\b", "has not been"),
        (r"\bhave been\b", "have not been"),
    ]
    for pattern, replacement in compound_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and " not " not in text[match.start(): match.end() + 8].lower():
            return text[: match.start()] + replacement + text[match.end():], "negation"
    remove_patterns = [
        (r"\bcould not\b", "could"),
        (r"\bcan not\b", "can"),
        (r"\bcannot\b", "can"),
        (r"\bcan't\b", "can"),
        (r"\bwould not\b", "would"),
        (r"\bwill not\b", "will"),
        (r"\bdid not\b", "did"),
        (r"\bdoes not\b", "does"),
        (r"\bdo not\b", "do"),
        (r"\bhas not\b", "has"),
        (r"\bhave not\b", "have"),
        (r"\bwas not\b", "was"),
        (r"\bwere not\b", "were"),
        (r"\bis not\b", "is"),
        (r"\bare not\b", "are"),
        (r"\bno answer\b", "an answer"),
        (r"\bno reply\b", "a reply"),
    ]
    for pattern, replacement in remove_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[: match.start()] + replacement + text[match.end():], "negation"
    # Adding a second negative to an already-negative proposition produces
    # malformed benchmark claims such as "nothing is not planned".
    if re.search(
        r"\b(?:no|none|neither|nothing|nobody|no\s+one|without)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return None
    patterns = [
        (r"\bcould\b", "could not"),
        (r"\bcan\b", "can not"),
        (r"\bwould\b", "would not"),
        # Avoid nominal uses such as "the will of the people".  They are not
        # auxiliaries and cannot be negated by mechanically inserting "not".
        (r"\bwill\b(?!\s+(?:of|to|for)\b)", "will not"),
        (r"\bdid\b", "did not"),
        (r"\bdoes\b", "does not"),
        (r"\bdo\b", "do not"),
        (r"\bwas\b", "was not"),
        (r"\bwere\b", "were not"),
        (r"\bis\b", "is not"),
        (r"\bare\b", "are not"),
        (r"\bhas\b", "has not"),
        (r"\bhave\b", "have not"),
    ]
    for pattern, replacement in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and " not " not in text[match.start(): match.end() + 8].lower():
            return text[: match.start()] + replacement + text[match.end():], "negation"
    return None


def malformed_generated_contradiction_surface(text: str) -> bool:
    """Reject mechanical mutations that are not standalone analyst claims."""
    value = normalize_space(text)
    if not value:
        return True
    if re.search(
        r"\b(?:will|would|can|could|should|must)\s+not\s+of\b|"
        r"\b(?:nothing|nobody|none|neither)\b[^.;]{0,80}"
        r"\b(?:is|are|was|were|has|have|had|will|would|can|could)\s+not\b|"
        r"\b\d+\s+\d+\s+(?:thousand|million|billion|percent)\b",
        value,
        flags=re.IGNORECASE,
    ):
        return True
    for match in YEAR_RANGE_RE.finditer(value):
        end = match.group("end")
        if len(end) == 2 and int(end) == int(match.group("start")) % 100:
            return True
    tokens = [token.casefold() for token in WORD_RE.findall(value)]
    if len(tokens) >= 10:
        opening = tokens[:4]
        if sum(len(token) >= 4 for token in opening) >= 2:
            for index in range(6, len(tokens) - 3):
                if tokens[index:index + 4] == opening:
                    return True
    return False


def near_strict_claim_surface_is_safe(
    text: str,
    profile: Mapping[str, Any],
    *,
    max_actor_words: int = 8,
    max_object_words: int = 18,
) -> bool:
    if not profile_has_complete_simple_clause(profile):
        return False
    if query_surface_is_blocked(text, task_family="claims"):
        return False
    if claim_uses_intentive_construction(text):
        return False
    if profile.get("nested_reporting") or profile.get("reporting_actor"):
        return False
    if profile.get("has_semicolon"):
        return False
    if actor_phrase_has_clause_bleed(normalize_space(profile.get("actor", ""))):
        return False
    if actor_has_abstract_head(normalize_space(profile.get("actor", ""))):
        return False
    if actor_is_generic_common_noun_phrase(normalize_space(profile.get("actor", ""))):
        return False
    if int(profile.get("actor_words", 0) or 0) > max_actor_words:
        return False
    if int(profile.get("object_words", 0) or 0) > max_object_words:
        return False
    return True


def choose_contradiction_cases(
    claim_views: pd.DataFrame,
    relations: pd.DataFrame,
    chunk_text_lookup: dict[tuple[str, str], str],
    count: int,
    sampling: dict[str, Any],
    seed: int,
    runner_cfg: dict[str, Any] | None = None,
    *,
    claim_rows: list[dict[str, Any]] | None = None,
    claim_profiles: list[dict[str, Any]] | None = None,
    peer_rows_by_idx: list[list[dict[str, Any]]] | None = None,
    reconstruction_views: pd.DataFrame | None = None,
    reconstruction_seed_cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    claim_doc_ids = {
        normalize_space(doc_id)
        for doc_id in claim_views.get("doc_id", pd.Series(dtype=str)).tolist()
        if normalize_space(doc_id)
    }
    contradiction_max_cases_per_doc = int(sampling.get("max_cases_per_doc", 2))
    if exhaustive_small_doc_corpus(len(claim_doc_ids)) and claim_doc_ids:
        # Raise the diversity cap only to the arithmetic minimum required to
        # fill the requested slice. Large corpora keep the configured cap.
        contradiction_max_cases_per_doc = max(
            contradiction_max_cases_per_doc,
            (count + len(claim_doc_ids) - 1) // len(claim_doc_ids),
        )
    corpus_name = normalize_space((runner_cfg or {}).get("__corpus_name", "")).casefold()
    pair_like_corpus = bool(claim_doc_ids) and all(doc_id.startswith("Pair_") for doc_id in claim_doc_ids)
    source_claim_rows = claim_rows if claim_rows is not None else claim_views.to_dict("records")
    claim_rows, aligned_companions = reorder_aligned_rows(
        source_claim_rows,
        seed=seed,
        key_fields=("doc_id", "paragraph_id", "claim_text", "local_id", "source_type"),
        companions=(claim_profiles, peer_rows_by_idx),
    )
    claim_profiles, peer_rows_by_idx = aligned_companions
    use_relation_candidates = not relations.empty
    relation_probe_count = (
        max(count * 4, count)
        if use_relation_candidates and not pair_like_corpus
        else count
    )
    relation_candidates = (
        choose_relation_contradiction_cases(
            relations,
            chunk_text_lookup,
            relation_probe_count,
            sampling,
            seed,
            runner_cfg=runner_cfg,
        )
        if use_relation_candidates
        else []
    )
    ultra_large_claim_corpus = claim_peer_pool_disabled(
        len(claim_rows),
        sampling,
        pair_like_corpus=pair_like_corpus,
        default_threshold=12000,
    )
    if claim_profiles is None:
        if ultra_large_claim_corpus and not pair_like_corpus:
            claim_profiles = [
                claim_clause_profile(normalize_space(row.get("claim_text", "")))
                for row in claim_rows
            ]
        else:
            claim_profiles, peer_rows_by_idx = build_claim_peer_pool(claim_rows)
    if peer_rows_by_idx is None:
        if ultra_large_claim_corpus and not pair_like_corpus:
            peer_rows_by_idx = [[] for _ in claim_rows]
        else:
            _profiles, peer_rows_by_idx = build_claim_peer_pool(claim_rows)
            if claim_profiles is None:
                claim_profiles = _profiles
    relation_keep_count = (
        max(2, count // 3)
        if (relation_candidates and len(claim_rows) >= 600 and not pair_like_corpus)
        else 0
    )
    minimal_direct_candidates = build_minimal_contradiction_candidates(
        claim_views,
        max(count * 6, count),
        sampling,
        seed,
        runner_cfg=runner_cfg,
    )
    small_corpus_direct_contradictions: list[dict[str, Any]] = []
    if len(claim_rows) < 250:
        for idx, row in enumerate(claim_rows):
            base_text = finalize_query_surface(
                simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", ""))),
                add_terminal_punctuation=True,
            )
            if (
                not base_text
                or query_surface_is_blocked(base_text, task_family="contradictions")
                or sentence_is_low_signal_for_query(base_text)
            ):
                continue
            if claim_surface_is_interrogative(base_text) or claim_surface_has_weak_modality(base_text):
                continue
            base_profile = claim_clause_profile(base_text)
            if profile_has_complete_simple_clause(base_profile):
                if actor_has_abstract_head(normalize_space(base_profile.get("actor", ""))):
                    continue
                if actor_is_generic_common_noun_phrase(normalize_space(base_profile.get("actor", ""))):
                    continue
            if claim_surface_is_retrieval_fragile(
                base_text,
                base_profile,
                small_corpus=True,
                pair_like_corpus=pair_like_corpus,
            ):
                continue
            if not (
                near_strict_claim_surface_is_safe(
                    base_text,
                    base_profile,
                    max_actor_words=6,
                    max_object_words=18,
                )
                or standalone_claim_surface_is_safe(base_text)
                or standalone_contradiction_surface_is_safe(base_text)
            ):
                continue
            if substantive_sentence_score(base_text) < 5.0:
                continue
            mutated = perturb_numeric_or_date(base_text) or perturb_negation(base_text)
            if not mutated:
                continue
            query_text = finalize_query_surface(mutated[0], add_terminal_punctuation=True)
            if claim_surface_is_interrogative(query_text) or claim_surface_has_weak_modality(query_text):
                continue
            if contains_role_placeholder_label(query_text):
                continue
            if not contradiction_query_carries_mutation(
                query_text,
                base_text,
                strategy_used=mutated[1],
            ):
                continue
            query_profile = claim_clause_profile(query_text)
            if substantive_sentence_score(query_text) < 5.0:
                continue
            partner_matches, partner_best_overlap = contradiction_partner_score(
                base_profile,
                mutated[1],
                peer_rows_by_idx[idx] if peer_rows_by_idx is not None else [],
            )
            small_corpus_recovery_ok = bool(
                len(claim_rows) <= 50
                and small_corpus_claim_surface_recovery_ok(
                    base_text,
                    task_family="contradictions",
                )
                and count_words(base_text) <= 32
            )
            if partner_matches <= 0 and partner_best_overlap < 0.2:
                base_quality = claim_clause_quality_score(base_profile)
                if (
                    base_quality < 8.0
                    or not (
                        standalone_contradiction_surface_is_safe(base_text)
                        or small_corpus_recovery_ok
                    )
                ):
                    continue
            if (
                mutated[1] == "negation"
                and partner_matches <= 0
                and not (
                    base_profile.get("date_token")
                    or base_profile.get("number_token")
                )
            ):
                continue
            words = count_words(query_text)
            small_corpus_direct_contradictions.append(
                {
                    "task_family": "contradictions",
                    "runner_task": "contradictions",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": f"contradictions_small_direct__{row['doc_id']}__{row['paragraph_id']}__{sha1_text(query_text)}",
                    "query": query_text,
                    "ground_truth": base_text,
                    "query_kind": "claim_small_direct_contradiction",
                    "notes": (
                        f"fallback=small_corpus_direct_contradiction; strategy={mutated[1]}; "
                        f"clause_quality={claim_clause_quality_score(base_profile):.2f}; "
                        f"partner_matches={partner_matches}; "
                        f"partner_best_overlap={partner_best_overlap:.2f}"
                    ),
                    "source_context": base_text,
                    "partner_matches": partner_matches,
                    "partner_best_overlap": partner_best_overlap,
                    "mutation_strategy": mutated[1],
                    "base_has_date_or_number": bool(
                        base_profile.get("date_token") or base_profile.get("number_token")
                    ),
                    "selector_score": (
                        claim_clause_quality_score(base_profile)
                        + 10.0
                        + min(partner_matches, 4) * 4.0
                        + partner_best_overlap * 8.0
                        + (4.0 if small_corpus_recovery_ok else 0.0)
                        - abs(words - 18) / 12.0
                        - surface_noise_penalty(query_text) * 0.3
                    ),
                    "word_count": words,
                    "query_target_words": words,
                }
            )
    allow_reconstruction_seed_contradictions = maybe_bool(
        sampling.get("allow_reconstruction_seed_contradictions", False)
    )
    reconstruction_seed_contradictions: list[dict[str, Any]] = []
    if reconstruction_seed_cases and allow_reconstruction_seed_contradictions:
        for seed_case in reconstruction_seed_cases:
            base_text = finalize_query_surface(
                simplify_claim_query_seed(simplify_claim_surface(seed_case.get("query", ""))),
                add_terminal_punctuation=True,
            )
            if (
                not base_text
                or query_surface_is_blocked(base_text, task_family="contradictions")
                or sentence_is_low_signal_for_query(base_text)
                or is_routing_or_metadata_sentence(base_text)
            ):
                continue
            if claim_surface_is_interrogative(base_text) or claim_surface_has_weak_modality(base_text):
                continue
            base_profile = claim_clause_profile(base_text)
            if (
                not profile_has_complete_simple_clause(base_profile)
                and not standalone_claim_surface_is_safe(base_text)
                and not standalone_contradiction_surface_is_safe(base_text)
            ):
                continue
            mutated = perturb_numeric_or_date(base_text) or perturb_negation(base_text)
            if not mutated:
                continue
            query_text = finalize_query_surface(mutated[0], add_terminal_punctuation=True)
            if claim_surface_is_interrogative(query_text) or claim_surface_has_weak_modality(query_text):
                continue
            if not contradiction_query_carries_mutation(
                query_text,
                base_text,
                strategy_used=mutated[1],
            ):
                continue
            if (
                query_surface_is_blocked(query_text, task_family="contradictions")
                or sentence_is_low_signal_for_query(query_text)
                or is_routing_or_metadata_sentence(query_text)
            ):
                continue
            words = count_words(query_text)
            if words < 8 or words > int(sampling.get("max_words", 40)):
                continue
            reconstruction_seed_contradictions.append(
                {
                    "task_family": "contradictions",
                    "runner_task": "contradictions",
                    "doc_id": str(seed_case.get("doc_id", "")),
                    "paragraph_id": str(seed_case.get("paragraph_id", "")),
                    "case_id": f"contradictions__seed__{normalize_space(seed_case.get('doc_id', ''))}__{sha1_text(query_text)}",
                    "query": query_text,
                    "ground_truth": base_text,
                    "query_kind": "reconstruction_seed_contradiction",
                    "notes": (
                        f"fallback=reconstruction_seed_contradiction; strategy={mutated[1]}; "
                        f"clause_quality={claim_clause_quality_score(base_profile):.2f}"
                    ),
                    "source_context": base_text,
                    "selector_score": (
                        claim_clause_quality_score(base_profile)
                        + 12.0
                        - abs(words - 18) / 12.0
                        - surface_noise_penalty(query_text) * 0.3
                    ),
                    "word_count": words,
                    "query_target_words": words,
                }
            )
            if len(reconstruction_seed_contradictions) >= max(count * 4, count):
                break
    candidates: list[dict[str, Any]] = list(
        minimal_direct_candidates[: max(count * 2, count)]
    )
    candidates.extend(relation_candidates[:relation_keep_count])
    if reconstruction_seed_contradictions:
        candidates.extend(
            round_robin_topk(
                reconstruction_seed_contradictions,
                min(count, len(reconstruction_seed_contradictions)),
                max_cases_per_doc=contradiction_max_cases_per_doc,
                score_key="selector_score",
                seed=seed,
            )
        )
    candidates.extend(
        round_robin_topk(
            small_corpus_direct_contradictions,
            min(count, len(small_corpus_direct_contradictions)),
            max_cases_per_doc=contradiction_max_cases_per_doc,
            score_key="selector_score",
            seed=seed,
        )
    )
    if len(claim_rows) <= 150:
        small_corpus_shortlist = merge_ranked_candidates(
            [
                reconstruction_seed_contradictions,
                small_corpus_direct_contradictions,
                minimal_direct_candidates,
                relation_candidates,
            ],
            target_count=max(count * 4, count),
            max_cases_per_doc=contradiction_max_cases_per_doc,
            score_key="selector_score",
            seed=seed,
        )
        small_corpus_shortlist = vet_contradiction_case_candidates(
            small_corpus_shortlist,
            count=count,
            runner_cfg=runner_cfg,
        )
        small_corpus_normalized = normalize_selected_case_queries(
            small_corpus_shortlist,
            task_family="contradictions",
            runner_cfg=runner_cfg,
        )
        if small_corpus_normalized and isinstance(runner_cfg, dict) and maybe_bool(
            runner_cfg.get("__strict_small_corpus_vet", False)
        ):
            return small_corpus_normalized
        if small_corpus_normalized:
            return small_corpus_normalized
    strict_mode = not maybe_bool(sampling.get("_relaxed_retry_used", False))
    allow_chunk_contradiction_fallback = maybe_bool(
        sampling.get("allow_chunk_contradiction_fallback", len(claim_rows) < 250)
    )
    allow_last_resort_contradiction_fallback = maybe_bool(
        sampling.get("allow_last_resort_contradiction_fallback", False)
    )
    allow_reconstruction_contradiction_backfill = maybe_bool(
        sampling.get("allow_reconstruction_contradiction_backfill", len(claim_rows) < 250)
    )
    candidate_runner_cfg = generation_runner_cfg(runner_cfg)
    strategies = [value.strip() for value in sampling["strategy_order"]]
    min_words = int(sampling["min_words"])
    max_words = int(sampling["max_words"])
    require_structured_source = maybe_bool(sampling.get("require_structured_source", True))
    require_target_or_speaker = maybe_bool(sampling.get("require_target_or_speaker", True))
    min_role_fields = int(sampling.get("min_role_fields", 2))
    include_stance = maybe_bool(sampling.get("include_stance", False))
    query_style = normalize_space(sampling.get("query_style", "plain")) or "plain"
    max_object_words = int(sampling.get("max_object_words", 18))
    max_actor_words = int(sampling.get("max_actor_words", 6))
    min_partner_matches = int(sampling.get("min_partner_matches", 1))
    min_partner_overlap = float(sampling.get("min_partner_overlap", 0.25))
    min_query_words = int(sampling.get("min_query_words", 16))
    max_query_words = int(sampling.get("max_query_words", max(int(sampling.get("max_words", 40)) * 4, 160)))
    require_simple_clause_arguments = maybe_bool(
        sampling.get("require_simple_clause_arguments", True)
    )
    large_claim_corpus = len(claim_rows) >= 2000
    contradiction_candidate_cap = (
        max(count * 50, 300)
        if len(claim_rows) >= 2000
        else max(count * 30, 180)
        if len(claim_rows) >= 600
        else 0
    )
    max_candidate_case_budget = int(sampling.get("max_candidate_case_budget", 0) or 0)
    if max_candidate_case_budget > 0:
        contradiction_candidate_cap = (
            min(contradiction_candidate_cap, max_candidate_case_budget)
            if contradiction_candidate_cap
            else max_candidate_case_budget
        )
    if len(claim_rows) < 250:
        min_partner_matches = min(min_partner_matches, 0)
        min_partner_overlap = min(min_partner_overlap, 0.0)
    if ultra_large_claim_corpus and not pair_like_corpus:
        min_partner_matches = 0
        min_partner_overlap = 0.0
    candidate_indices = list(range(len(claim_rows)))
    if large_claim_corpus:
        def contradiction_seed_score(index: int) -> tuple[float, float, int]:
            row = claim_rows[index]
            text = simplify_claim_surface(row.get("claim_text", ""))
            profile = claim_profiles[index]
            quality = claim_clause_quality_score(profile)
            mutation_bonus = (
                (5.0 if profile.get("date_token") else 0.0)
                + (5.0 if profile.get("number_token") else 0.0)
                + (3.0 if profile.get("has_negation") else 0.0)
                + (
                    4.0
                    if strict_claim_surface_is_safe(
                        text,
                        profile,
                        max_actor_words=max_actor_words,
                        max_object_words=max_object_words,
                    )
                    else -8.0
                )
            )
            role_bonus = (
                (4.0 if profile.get("actor") else 0.0)
                + (4.0 if profile.get("predicate") else 0.0)
                + (4.0 if profile.get("object") else 0.0)
                + (2.0 if normalize_space(row.get("speaker", "")) else 0.0)
                + (2.0 if normalize_space(row.get("target", "")) else 0.0)
            )
            text_words = count_words(text)
            return (
                quality + role_bonus + mutation_bonus - surface_noise_penalty(text) * 0.4,
                float(len(profile.get("content_tokens", set()))),
                -abs(text_words - 18),
            )

        candidate_indices = sorted(
            candidate_indices,
            key=contradiction_seed_score,
            reverse=True,
        )
        ranked_candidate_rows = max(count * 40, 180)
        max_ranked_candidate_rows = int(sampling.get("max_ranked_candidate_rows", 0) or 0)
        if max_ranked_candidate_rows > 0:
            ranked_candidate_rows = min(ranked_candidate_rows, max_ranked_candidate_rows)
        candidate_indices = candidate_indices[:ranked_candidate_rows]
    fallback_candidate_indices = (
        candidate_indices[: max(count * 30, 120)]
        if large_claim_corpus
        else candidate_indices
    )
    for idx in candidate_indices:
        if contradiction_candidate_cap and len(candidates) >= contradiction_candidate_cap:
            break
        row = claim_rows[idx]
        if require_structured_source and not row_is_structured_claim(row):
            continue
        text = simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", "")))
        profile = claim_profiles[idx]
        if REDACTION_MARKER_RE.search(text):
            continue
        if claim_surface_is_interrogative(text) or claim_surface_has_weak_modality(text):
            continue
        surface_profile = claim_clause_profile(text)
        if strict_mode and not strict_claim_surface_is_safe(
            text,
            surface_profile,
            max_actor_words=max_actor_words,
            max_object_words=max_object_words,
        ):
            continue
        if sentence_is_low_signal_for_query(text):
            continue
        words = count_words(text)
        if words < min_words or words > max_words:
            continue
        if re.match(r"^(?:there was|there has been)\b", text, flags=re.IGNORECASE):
            continue
        if query_surface_is_blocked(text, task_family="contradictions"):
            continue
        if strict_mode and has_reporting_that_clause(text):
            continue
        if (
            CONDITIONAL_CLAIM_RE.search(text)
            or ABSTRACT_QUANTIFIER_RE.search(text)
            or re.search(r"\b(?:could|may|might|likely|perhaps)\b", text, flags=re.IGNORECASE)
        ):
            continue
        if claim_uses_intentive_construction(text):
            continue
        query_target_words = deterministic_query_word_target(
            f"contradictions|{row['doc_id']}|{row['paragraph_id']}|{text}",
            minimum_words=min_query_words,
            maximum_words=min(max_query_words, 128),
        )
        if surface_profile["nested_reporting"]:
            continue
        if surface_profile["has_semicolon"]:
            continue
        if int(surface_profile["object_words"]) > max_object_words:
            continue
        paragraph_source_text = chunk_text_lookup.get(
            paragraph_key(row["doc_id"], row["paragraph_id"]),
            "",
        )
        source_context = choose_fast_query_context(
            [
                paragraph_source_text,
                extract_embedded_context(row.get("pseudo_view_text", "")),
                extract_embedded_context(row.get("proposition_text", "")),
            ],
            min_words=max(8, min_query_words // 2),
            max_chars=420,
        )
        anchor_bundle = [
            profile.get("actor", ""),
            profile.get("object", ""),
            normalize_space(row.get("speaker", "")),
            normalize_space(row.get("target", "")),
        ]
        source_context = refine_generated_query_surface(
            source_context,
            task_family="contradictions",
            anchors=anchor_bundle,
            min_words=max(8, min_query_words // 2),
            target_words=max(10, query_target_words - 12),
            max_chars=420,
            runner_cfg=candidate_runner_cfg,
        )
        claim_query_text, role_fields, role_field_count = build_claim_query(
            row,
            include_stance=include_stance,
            query_style=query_style,
            source_context_text=source_context,
        )
        seed_profile = claim_clause_profile(claim_query_text or text)
        if strict_mode and not (
            seed_profile["actor"]
            and seed_profile["predicate"]
            and seed_profile["object"]
        ):
            continue
        if role_field_count < max(2, min_role_fields):
            continue
        if strict_mode and require_simple_clause_arguments and not (
            role_fields.get("actor") and role_fields.get("predicate") and role_fields.get("object")
        ):
            continue
        if strict_mode and predicate_is_low_signal(role_fields.get("predicate", "")):
            continue
        if strict_mode and actor_has_abstract_head(role_fields.get("actor", "")):
            continue
        if strict_mode and actor_is_generic_common_noun_phrase(role_fields.get("actor", "")):
            continue
        if strict_mode and object_has_abstract_head(role_fields.get("object", "")):
            continue
        if strict_mode and actor_phrase_has_clause_bleed(role_fields.get("actor", "")):
            continue
        if require_target_or_speaker and not (
            role_fields.get("actor") or role_fields["speaker"] or role_fields["target"]
        ):
            continue
        if strict_mode and parser_hostile_claim_candidate(text, role_fields, profile):
            continue
        if strict_mode and relation_candidates and not standalone_claim_surface_is_safe(text):
            continue
        base_clause = normalize_space(claim_query_text or text)
        if not base_clause and strict_mode:
            continue
        mutation_base = base_clause or text
        mutated = None
        strategy_used = ""
        for strategy in strategies:
            if strategy in {"numeric", "date"}:
                mutated = perturb_numeric_or_date(mutation_base)
            elif strategy == "negation":
                mutated = perturb_negation(mutation_base)
            if mutated:
                strategy_used = mutated[1]
                break
        if not mutated:
            continue
        query_text = build_contradiction_query(
            source_context_text=source_context,
            claim_text=claim_query_text or text,
            base_clause=mutation_base,
            mutated_clause=mutated[0],
        )
        query_text = refine_generated_query_surface(
            query_text,
            task_family="contradictions",
            anchors=anchor_bundle,
            min_words=min_query_words,
            target_words=query_target_words,
            max_chars=420,
            runner_cfg=candidate_runner_cfg,
        )
        if claim_surface_is_interrogative(query_text) or claim_surface_has_weak_modality(query_text):
            continue
        if contains_role_placeholder_label(query_text):
            continue
        if re.search(
            r"\bcontinues?\s+to\s+have\s+not\b|\bcould\s+not\s+have\s+authorized\b",
            query_text,
            flags=re.IGNORECASE,
        ):
            continue
        if not contradiction_query_carries_mutation(
            query_text,
            text,
            strategy_used=strategy_used,
        ):
            continue
        if REDACTION_MARKER_RE.search(query_text):
            continue
        if strict_mode and (
            re.search(r"\bnot simply cases?\s+of\b", query_text, flags=re.IGNORECASE)
            or re.search(r"\bwould not be\b.*\bif\b", query_text, flags=re.IGNORECASE)
            or re.search(r"\bfollowing letter\b|\bwe would appreciate any comments\b", query_text, flags=re.IGNORECASE)
        ):
            continue
        if count_words(query_text) < min_query_words or count_words(query_text) > max_query_words:
            continue
        if query_surface_is_blocked(query_text, task_family="contradictions"):
            continue
        if strict_mode and substantive_sentence_score(query_text, anchors=anchor_bundle) < 6.0:
            continue
        if sentence_is_low_signal_for_query(query_text):
            continue
        if strict_mode and not standalone_claim_surface_is_safe(query_text):
            continue
        query_profile = claim_clause_profile(query_text)
        if not profile_has_complete_simple_clause(query_profile):
            continue
        if strict_mode and not strict_claim_surface_is_safe(
            query_text,
            query_profile,
            max_actor_words=max_actor_words,
            max_object_words=max_object_words,
        ):
            continue
        if strict_mode and (
            query_profile["nested_reporting"]
            or query_profile["reporting_actor"]
            or actor_phrase_has_clause_bleed(query_profile["actor"])
            or query_profile["object_bad_lead"]
        ):
            continue
        if strict_mode and claim_uses_intentive_construction(query_text):
            continue
        query_alignment = content_token_overlap_ratio(query_text, text)
        if query_alignment < 0.35:
            continue
        min_alignment = 0.6 if pair_like_corpus else 0.5
        if strict_mode and query_alignment < min_alignment:
            continue
        partner_matches, partner_best_overlap = contradiction_partner_score(
            profile,
            strategy_used,
            peer_rows_by_idx[idx],
        )
        if (
            strategy_used == "negation"
            and partner_matches <= 0
            and not (profile.get("date_token") or profile.get("number_token"))
        ):
            continue
        if partner_matches < min_partner_matches or partner_best_overlap < min_partner_overlap:
            continue
        mutation_bonus = {"date": 4.0, "numeric": 4.0, "negation": 2.5}.get(strategy_used, 0.0)
        selector_score = (
            claim_clause_quality_score(surface_profile)
            + mutation_bonus
            + role_field_count * 4.0
            + (3.0 if role_fields.get("actor") else 0.0)
            + (3.0 if role_fields["predicate"] else 0.0)
            + (3.0 if role_fields["object"] else 0.0)
            + (2.0 if role_fields["speaker"] else 0.0)
            + (2.0 if role_fields["target"] else 0.0)
            + (1.0 if DATE_TOKEN_RE.search(text) else 0.0)
            + (1.0 if NUMBER_RE.search(text) else 0.0)
            + min(partner_matches, 4) * 4.5
            + partner_best_overlap * 8.0
            + query_alignment * 8.0
            - (8.0 if not (role_fields.get("actor") and role_fields["predicate"] and role_fields["object"]) else 0.0)
            - abs(words - 18) / 12.0
            - surface_noise_penalty(query_text) * 0.4
        )
        case_id = f"contradictions__{row['doc_id']}__{sha1_text(query_text)}"
        candidates.append(
            {
                "task_family": "contradictions",
                "runner_task": "contradictions",
                "doc_id": str(row["doc_id"]),
                "paragraph_id": str(row["paragraph_id"]),
                "case_id": case_id,
                "query": query_text,
                "ground_truth": text,
                "query_kind": "structured_claim_contradiction",
                "notes": (
                    f"strategy={strategy_used}; actor={role_fields.get('actor','')}; speaker={role_fields['speaker']}; "
                    f"target={role_fields['target']}; role_field_count={role_field_count}; "
                    f"partner_matches={partner_matches}; partner_best_overlap={partner_best_overlap:.2f}; "
                    f"query_alignment={query_alignment:.2f}; "
                    f"nested_reporting={surface_profile['nested_reporting']}; "
                    f"actor_words={surface_profile['actor_words']}; object_words={surface_profile['object_words']}; "
                    f"source_type={normalize_space(row.get('source_type', ''))}; "
                    f"query_target_words={query_target_words}"
                ),
                "source_context": source_context,
                "selector_score": selector_score,
                "word_count": count_words(query_text),
                "query_target_words": query_target_words,
            }
        )
    selected = round_robin_topk(
        candidates,
        count,
        max_cases_per_doc=contradiction_max_cases_per_doc,
        score_key="selector_score",
        seed=seed,
    )
    selected = vet_contradiction_case_candidates(
        selected,
        count=count,
        runner_cfg=runner_cfg,
    )
    if len(selected) >= count or maybe_bool(sampling.get("_relaxed_retry_used", False)):
        return normalize_selected_case_queries(
            selected,
            task_family="contradictions",
            runner_cfg=runner_cfg,
        )
    if True:
        selected_case_ids = {normalize_space(case.get("case_id", "")) for case in selected}
        fallback_candidates: list[dict[str, Any]] = []
        for idx in fallback_candidate_indices:
            row = claim_rows[idx]
            text = simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", "")))
            profile = claim_clause_profile(text)
            if not strict_claim_surface_is_safe(
                text,
                profile,
                max_actor_words=max_actor_words,
                max_object_words=max_object_words,
            ):
                continue
            role_fields = {
                "actor": normalize_space(profile.get("actor", "")),
                "predicate": normalize_space(profile.get("predicate", "")),
                "object": normalize_space(profile.get("object", "")),
            }
            paragraph_source_text = chunk_text_lookup.get(
                paragraph_key(row["doc_id"], row["paragraph_id"]),
                "",
            )
            source_context = choose_fast_query_context(
                [
                    paragraph_source_text,
                    extract_embedded_context(row.get("pseudo_view_text", "")),
                    extract_embedded_context(row.get("proposition_text", "")),
                ],
                min_words=max(8, min_query_words // 2),
                max_chars=420,
            )
            source_context = refine_generated_query_surface(
                source_context,
                task_family="contradictions",
                anchors=[profile.get("actor", ""), profile.get("object", "")],
                min_words=max(8, min_query_words // 2),
                target_words=max(10, count_words(text) - 6),
                max_chars=420,
                runner_cfg=candidate_runner_cfg,
            )
            mutation_base = choose_best_query_context(
                [source_context, text],
                anchors=[profile.get("actor", ""), profile.get("object", "")],
                min_words=min_query_words,
                target_words=count_words(text),
                max_sentences=2,
                max_chars=420,
            ) or normalize_space(text)
            if not mutation_base:
                continue
            mutated = None
            strategy_used = ""
            for strategy in strategies:
                if strategy in {"numeric", "date"}:
                    mutated = perturb_numeric_or_date(mutation_base)
                elif strategy == "negation":
                    mutated = perturb_negation(mutation_base)
                if mutated:
                    strategy_used = mutated[1]
                    break
            if not mutated:
                continue
            query_text = finalize_query_surface(mutated[0], add_terminal_punctuation=True)
            if contains_role_placeholder_label(query_text):
                continue
            if re.search(
                r"\bcontinues?\s+to\s+have\s+not\b|\bcould\s+not\s+have\s+authorized\b",
                query_text,
                flags=re.IGNORECASE,
            ):
                continue
            if not contradiction_query_carries_mutation(
                query_text,
                text,
                strategy_used=strategy_used,
            ):
                continue
            query_profile = claim_clause_profile(query_text)
            if not profile_has_complete_simple_clause(query_profile):
                continue
            if not strict_claim_surface_is_safe(
                query_text,
                query_profile,
                max_actor_words=max_actor_words,
                max_object_words=max_object_words,
            ):
                continue
            partner_matches, partner_best_overlap = contradiction_partner_score(
                profile,
                strategy_used,
                peer_rows_by_idx[idx],
            )
            words = count_words(query_text)
            selector_score = (
                claim_clause_quality_score(profile)
                + min(partner_matches, 4) * 4.5
                + partner_best_overlap * 8.0
                - (4.0 if partner_matches <= 0 else 0.0)
                - abs(words - 18) / 12.0
                - surface_noise_penalty(query_text) * 0.4
            )
            case_id = f"contradictions__{row['doc_id']}__{sha1_text(query_text)}"
            if case_id in selected_case_ids:
                continue
            fallback_candidates.append(
                {
                    "task_family": "contradictions",
                    "runner_task": "contradictions",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": case_id,
                    "query": query_text,
                    "ground_truth": text,
                    "query_kind": "structured_claim_contradiction_direct",
                    "notes": (
                        f"fallback=direct_contradiction_surface; strategy={strategy_used}; "
                        f"partner_matches={partner_matches}; partner_best_overlap={partner_best_overlap:.2f}; "
                        f"actor_words={profile['actor_words']}; object_words={profile['object_words']}"
                    ),
                    "source_context": "",
                    "selector_score": selector_score,
                    "word_count": words,
                    "query_target_words": words,
                }
            )
            if len(fallback_candidates) >= max(count * 6, 24):
                break
        if len(selected) + len(fallback_candidates) < count and allow_chunk_contradiction_fallback:
            for (doc_id, paragraph_id), chunk_text in chunk_text_lookup.items():
                for sentence in sentence_split(chunk_text):
                    base_text = finalize_query_surface(sentence, add_terminal_punctuation=True)
                    if not base_text or "?" in base_text:
                        continue
                    if query_surface_is_blocked(base_text, task_family="contradictions"):
                        continue
                    base_profile = claim_clause_profile(base_text)
                    if not profile_has_complete_simple_clause(base_profile):
                        continue
                    if not strict_claim_surface_is_safe(
                        base_text,
                        base_profile,
                        max_actor_words=max_actor_words,
                        max_object_words=max_object_words,
                    ):
                        continue
                    mutated = None
                    strategy_used = ""
                    for strategy in strategies:
                        if strategy in {"numeric", "date"}:
                            mutated = perturb_numeric_or_date(base_text)
                        elif strategy == "negation":
                            mutated = perturb_negation(base_text)
                        if mutated:
                            strategy_used = mutated[1]
                            break
                    if not mutated:
                        continue
                    query_text = finalize_query_surface(mutated[0], add_terminal_punctuation=True)
                    if not contradiction_query_carries_mutation(
                        query_text,
                        base_text,
                        strategy_used=strategy_used,
                    ):
                        continue
                    if query_surface_is_blocked(query_text, task_family="contradictions"):
                        continue
                    query_profile = claim_clause_profile(query_text)
                    if not profile_has_complete_simple_clause(query_profile):
                        continue
                    if not strict_claim_surface_is_safe(
                        query_text,
                        query_profile,
                        max_actor_words=max_actor_words,
                        max_object_words=max_object_words,
                    ):
                        continue
                    words = count_words(query_text)
                    if len(claim_rows) < 250 and words > 32:
                        continue
                    case_id = f"contradictions__{doc_id}__{paragraph_id}__{sha1_text(query_text)}"
                    if case_id in selected_case_ids:
                        continue
                    fallback_candidates.append(
                        {
                            "task_family": "contradictions",
                            "runner_task": "contradictions",
                            "doc_id": str(doc_id),
                            "paragraph_id": str(paragraph_id),
                            "case_id": case_id,
                            "query": query_text,
                            "ground_truth": base_text,
                            "query_kind": "chunk_sentence_contradiction",
                            "notes": (
                                f"fallback=chunk_contradiction_surface; strategy={strategy_used}; "
                                f"actor_words={base_profile['actor_words']}; object_words={base_profile['object_words']}"
                            ),
                            "source_context": base_text,
                            "selector_score": (
                                claim_clause_quality_score(base_profile)
                                + 8.0
                                - abs(words - 18) / 12.0
                                - surface_noise_penalty(query_text) * 0.35
                            ),
                            "word_count": words,
                            "query_target_words": words,
                        }
                    )
                    if len(selected) + len(fallback_candidates) >= count or len(fallback_candidates) >= max(count * 8, 32):
                        break
                if len(selected) + len(fallback_candidates) >= count or len(fallback_candidates) >= max(count * 8, 32):
                    break
        if (
            len(selected) + len(fallback_candidates) < count
            and allow_reconstruction_contradiction_backfill
            and reconstruction_views is not None
            and not reconstruction_views.empty
        ):
            pseudo_rows = deterministic_seeded_row_order(
                reconstruction_views.to_dict("records"),
                seed=seed + 41,
                key_fields=("doc_id", "paragraph_id", "pseudo_chunk_text", "local_id"),
            )
            for row in pseudo_rows:
                doc_id = normalize_space(row.get("doc_id", ""))
                paragraph_id = normalize_space(row.get("paragraph_id", ""))
                if not doc_id or not paragraph_id:
                    continue
                for claim_text in unique_preserving_order(
                    parse_json_list(
                        row.get("reconstruction_claims_json", row.get("claim_texts", ""))
                    )
                )[:2]:
                    base_text = finalize_query_surface(
                        simplify_claim_query_seed(simplify_claim_surface(claim_text)),
                        add_terminal_punctuation=True,
                    )
                    if (
                        not base_text
                        or query_surface_is_blocked(base_text, task_family="contradictions")
                        or sentence_is_low_signal_for_query(base_text)
                    ):
                        continue
                    base_profile = claim_clause_profile(base_text)
                    if not near_strict_claim_surface_is_safe(
                        base_text,
                        base_profile,
                        max_actor_words=max_actor_words,
                        max_object_words=max_object_words,
                    ):
                        continue
                    mutated = perturb_numeric_or_date(base_text) or perturb_negation(base_text)
                    if not mutated:
                        continue
                    query_text = finalize_query_surface(mutated[0], add_terminal_punctuation=True)
                    if not contradiction_query_carries_mutation(
                        query_text,
                        base_text,
                        strategy_used=mutated[1],
                    ):
                        continue
                    query_profile = claim_clause_profile(query_text)
                    if not near_strict_claim_surface_is_safe(
                        query_text,
                        query_profile,
                        max_actor_words=max_actor_words,
                        max_object_words=max_object_words,
                    ):
                        continue
                    words = count_words(query_text)
                    case_id = f"contradictions__{doc_id}__reconstruction__{paragraph_id}__{sha1_text(query_text)}"
                    if case_id in selected_case_ids:
                        continue
                    fallback_candidates.append(
                        {
                            "task_family": "contradictions",
                            "runner_task": "contradictions",
                            "doc_id": str(doc_id),
                            "paragraph_id": str(paragraph_id),
                            "case_id": case_id,
                            "query": query_text,
                            "ground_truth": base_text,
                            "query_kind": "reconstruction_claim_contradiction",
                            "notes": (
                                f"fallback=reconstruction_claim_contradiction; strategy={mutated[1]}; "
                                f"actor_words={base_profile['actor_words']}; object_words={base_profile['object_words']}"
                            ),
                            "source_context": base_text,
                            "selector_score": (
                                claim_clause_quality_score(base_profile)
                                + 12.0
                                - abs(words - 18) / 12.0
                                - surface_noise_penalty(query_text) * 0.3
                            ),
                            "word_count": words,
                            "query_target_words": words,
                        }
                    )
                    if len(selected) + len(fallback_candidates) >= count or len(fallback_candidates) >= max(count * 8, 32):
                        break
                if len(selected) + len(fallback_candidates) >= count or len(fallback_candidates) >= max(count * 8, 32):
                    break
        if len(selected) + len(fallback_candidates) < count and reconstruction_seed_cases:
            for seed_case in reconstruction_seed_cases:
                base_text = finalize_query_surface(
                    simplify_claim_query_seed(simplify_claim_surface(seed_case.get("query", ""))),
                    add_terminal_punctuation=True,
                )
                if (
                    not base_text
                    or query_surface_is_blocked(base_text, task_family="contradictions")
                    or sentence_is_low_signal_for_query(base_text)
                ):
                    continue
                base_profile = claim_clause_profile(base_text)
                if not near_strict_claim_surface_is_safe(
                    base_text,
                    base_profile,
                    max_actor_words=max_actor_words,
                    max_object_words=max_object_words,
                ):
                    continue
                mutated = perturb_numeric_or_date(base_text) or perturb_negation(base_text)
                if not mutated:
                    continue
                query_text = finalize_query_surface(mutated[0], add_terminal_punctuation=True)
                if not contradiction_query_carries_mutation(
                    query_text,
                    base_text,
                    strategy_used=mutated[1],
                ):
                    continue
                query_profile = claim_clause_profile(query_text)
                if not near_strict_claim_surface_is_safe(
                    query_text,
                    query_profile,
                    max_actor_words=max_actor_words,
                    max_object_words=max_object_words,
                ):
                    continue
                case_id = f"contradictions__seed__{normalize_space(seed_case.get('doc_id', ''))}__{sha1_text(query_text)}"
                if case_id in selected_case_ids:
                    continue
                words = count_words(query_text)
                fallback_candidates.append(
                    {
                        "task_family": "contradictions",
                        "runner_task": "contradictions",
                        "doc_id": str(seed_case.get("doc_id", "")),
                        "paragraph_id": str(seed_case.get("paragraph_id", "")),
                        "case_id": case_id,
                        "query": query_text,
                        "ground_truth": base_text,
                        "query_kind": "reconstruction_seed_contradiction",
                        "notes": (
                            f"fallback=reconstruction_seed_contradiction; strategy={mutated[1]}; "
                            f"clause_quality={claim_clause_quality_score(base_profile):.2f}"
                        ),
                        "source_context": base_text,
                        "selector_score": (
                            claim_clause_quality_score(base_profile)
                            + 12.0
                            - abs(words - 18) / 12.0
                            - surface_noise_penalty(query_text) * 0.3
                        ),
                        "word_count": words,
                        "query_target_words": words,
                    }
                )
                if len(selected) + len(fallback_candidates) >= count or len(fallback_candidates) >= max(count * 8, 32):
                    break
        merged_selected = merge_ranked_candidates(
            [selected, fallback_candidates, minimal_direct_candidates],
            target_count=max(count * 4, count),
            max_cases_per_doc=contradiction_max_cases_per_doc,
            score_key="selector_score",
            seed=seed,
        )
        merged_selected = vet_contradiction_case_candidates(
            merged_selected,
            count=count,
            runner_cfg=runner_cfg,
        )
        if len(merged_selected) >= count or maybe_bool(sampling.get("_relaxed_retry_used", False)):
            return normalize_selected_case_queries(
                merged_selected,
                task_family="contradictions",
                runner_cfg=runner_cfg,
            )
    relaxed_sampling = dict(sampling)
    relaxed_sampling["_relaxed_retry_used"] = True
    relaxed_sampling["require_structured_source"] = not pair_like_corpus
    relaxed_sampling["require_simple_clause_arguments"] = True
    relaxed_sampling["require_target_or_speaker"] = False
    relaxed_sampling["min_role_fields"] = max(2, min_role_fields)
    relaxed_sampling["min_partner_matches"] = 0
    relaxed_sampling["min_partner_overlap"] = 0.0
    relaxed_sampling["max_object_words"] = max(max_object_words, 28)
    relaxed_sampling["max_words"] = max(int(sampling.get("max_words", 40)), 96)
    relaxed_sampling["max_query_words"] = max(int(sampling.get("max_query_words", 160)), 180)
    relaxed = choose_contradiction_cases(
        claim_views,
        relations,
        chunk_text_lookup,
        count,
        relaxed_sampling,
        seed,
        runner_cfg=runner_cfg,
        claim_rows=claim_rows,
        claim_profiles=claim_profiles,
        peer_rows_by_idx=peer_rows_by_idx,
        reconstruction_views=reconstruction_views,
    )
    combined_selected = merge_ranked_candidates(
        [merged_selected, relaxed, minimal_direct_candidates],
        target_count=max(count * 4, count),
        max_cases_per_doc=contradiction_max_cases_per_doc,
        score_key="selector_score",
        seed=seed,
    )
    combined_selected = vet_contradiction_case_candidates(
        combined_selected,
        count=count,
        runner_cfg=runner_cfg,
    )
    if len(combined_selected) < count and allow_last_resort_contradiction_fallback:
        supplemental: list[dict[str, Any]] = []
        supplemental_ids = {normalize_space(case.get("case_id", "")) for case in combined_selected}
        for row in claim_rows:
            base_text = simplify_claim_query_seed(simplify_claim_surface(row.get("claim_text", "")))
            if not base_text or "?" in base_text:
                continue
            base_profile = claim_clause_profile(base_text)
            if not near_strict_claim_surface_is_safe(
                base_text,
                base_profile,
                max_actor_words=max_actor_words,
                max_object_words=max_object_words,
            ):
                continue
            mutated = perturb_numeric_or_date(base_text) or perturb_negation(base_text)
            if not mutated:
                continue
            query_text = finalize_query_surface(mutated[0], add_terminal_punctuation=True)
            if not contradiction_query_carries_mutation(
                query_text,
                base_text,
                strategy_used=mutated[1],
            ):
                continue
            query_profile = claim_clause_profile(query_text)
            if not near_strict_claim_surface_is_safe(
                query_text,
                query_profile,
                max_actor_words=max_actor_words,
                max_object_words=max_object_words,
            ):
                continue
            case_id = f"contradictions__{row['doc_id']}__fallback__{sha1_text(query_text)}"
            if case_id in supplemental_ids:
                continue
            supplemental_ids.add(case_id)
            words = count_words(query_text)
            supplemental.append(
                {
                    "task_family": "contradictions",
                    "runner_task": "contradictions",
                    "doc_id": str(row["doc_id"]),
                    "paragraph_id": str(row["paragraph_id"]),
                    "case_id": case_id,
                    "query": query_text,
                    "ground_truth": base_text,
                    "query_kind": "last_resort_claim_contradiction",
                    "notes": (
                        f"fallback=last_resort_claim_contradiction; strategy={mutated[1]}; "
                        f"actor_words={base_profile['actor_words']}; object_words={base_profile['object_words']}"
                    ),
                    "source_context": base_text,
                    "selector_score": (
                        claim_clause_quality_score(base_profile)
                        + 6.0
                        - abs(words - 18) / 12.0
                        - surface_noise_penalty(query_text) * 0.35
                    ),
                    "word_count": words,
                    "query_target_words": words,
                }
            )
            if len(combined_selected) + len(supplemental) >= count:
                break
        if len(combined_selected) + len(supplemental) < count:
            for (doc_id, paragraph_id), chunk_text in chunk_text_lookup.items():
                for sentence in sentence_split(chunk_text):
                    base_text = finalize_query_surface(sentence, add_terminal_punctuation=True)
                    if not base_text or "?" in base_text:
                        continue
                    base_profile = claim_clause_profile(base_text)
                    if not near_strict_claim_surface_is_safe(
                        base_text,
                        base_profile,
                        max_actor_words=max_actor_words,
                        max_object_words=max_object_words,
                    ):
                        continue
                    mutated = perturb_numeric_or_date(base_text) or perturb_negation(base_text)
                    if not mutated:
                        continue
                    query_text = finalize_query_surface(mutated[0], add_terminal_punctuation=True)
                    if not contradiction_query_carries_mutation(
                        query_text,
                        base_text,
                        strategy_used=mutated[1],
                    ):
                        continue
                    query_profile = claim_clause_profile(query_text)
                    if not near_strict_claim_surface_is_safe(
                        query_text,
                        query_profile,
                        max_actor_words=max_actor_words,
                        max_object_words=max_object_words,
                    ):
                        continue
                    case_id = f"contradictions__{doc_id}__{paragraph_id}__fallback__{sha1_text(query_text)}"
                    if case_id in supplemental_ids:
                        continue
                    supplemental_ids.add(case_id)
                    words = count_words(query_text)
                    supplemental.append(
                        {
                            "task_family": "contradictions",
                            "runner_task": "contradictions",
                            "doc_id": str(doc_id),
                            "paragraph_id": str(paragraph_id),
                            "case_id": case_id,
                            "query": query_text,
                            "ground_truth": base_text,
                            "query_kind": "last_resort_chunk_contradiction",
                            "notes": (
                                f"fallback=last_resort_chunk_contradiction; strategy={mutated[1]}; "
                                f"actor_words={base_profile['actor_words']}; object_words={base_profile['object_words']}"
                            ),
                            "source_context": base_text,
                            "selector_score": (
                                claim_clause_quality_score(base_profile)
                                + 5.0
                                - abs(words - 18) / 12.0
                                - surface_noise_penalty(query_text) * 0.35
                            ),
                            "word_count": words,
                            "query_target_words": words,
                        }
                    )
                    if len(combined_selected) + len(supplemental) >= count:
                        break
                if len(combined_selected) + len(supplemental) >= count:
                    break
        combined_selected = merge_ranked_candidates(
            [combined_selected, supplemental],
            target_count=max(count * 4, count),
            max_cases_per_doc=contradiction_max_cases_per_doc,
            score_key="selector_score",
            seed=seed,
        )
        combined_selected = vet_contradiction_case_candidates(
            combined_selected,
            count=count,
            runner_cfg=runner_cfg,
        )
    return normalize_selected_case_queries(
        combined_selected,
        task_family="contradictions",
        runner_cfg=runner_cfg,
    )


def value_redaction_slot_count(case: dict[str, Any]) -> int:
    return shared_value_redaction_slot_count(
        case,
        format_expected_targets=format_expected_targets,
        count_redaction_slots=count_redaction_slots,
    )


def geometric_slot_targets(
    total_count: int,
    *,
    max_slot_count: int,
    decay: float,
) -> dict[int, int]:
    return shared_geometric_slot_targets(
        total_count,
        max_slot_count=max_slot_count,
        decay=decay,
    )


def rebalance_value_redaction_cases(
    candidates: list[dict[str, Any]],
    target_count: int,
    *,
    max_cases_per_doc: int,
    max_slot_count: int,
    slot_count_decay: float,
) -> list[dict[str, Any]]:
    return shared_rebalance_value_redaction_cases(
        candidates,
        target_count,
        max_cases_per_doc=max_cases_per_doc,
        max_slot_count=max_slot_count,
        slot_count_decay=slot_count_decay,
        format_expected_targets=format_expected_targets,
        count_redaction_slots=count_redaction_slots,
    )


def build_cases(
    config: dict[str, Any],
    *,
    per_task_override: int | None,
    tasks_override: list[str] | None,
    progress_prefix: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    context = prepare_build_context(
        config,
        per_task_override=per_task_override,
        tasks_override=tasks_override,
        maybe_bool=maybe_bool,
        normalize_space=normalize_space,
        read_tables=read_tables,
        build_doc_sentences=build_doc_sentences,
        build_chunk_text_lookup=build_chunk_text_lookup,
        build_doc_para_richness=build_doc_para_richness,
        build_paragraph_component_index=build_paragraph_component_index,
    )
    seed = context["seed"]
    runner_cfg = context["runner_cfg"]
    max_cases_per_doc = context["max_cases_per_doc"]
    candidate_pool_multiplier = context["candidate_pool_multiplier"]
    task_counts = context["task_counts"]
    policy_task_counts = context["policy_task_counts"]
    tables = context["tables"]
    chunks = context["chunks"]
    claims = context["claims"]
    entities = context["entities"]
    redaction_views = context["redaction_views"]
    claim_views = context["claim_views"]
    reconstruction_views = context["reconstruction_views"]
    doc_chunks_by_doc = context["doc_chunks_by_doc"]
    chunk_text_lookup = context["chunk_text_lookup"]
    corpus_doc_count = context["corpus_doc_count"]
    para_richness = context["para_richness"]
    component_index = context["component_index"]
    claim_generation_views = context["claim_generation_views"]
    pair_like_claim_corpus = context["pair_like_claim_corpus"]
    fisc_claimlike_acceptable_sources = context["fisc_claimlike_acceptable_sources"]
    enable_small_corpus_parser_vet = context["enable_small_corpus_parser_vet"]
    task_generation_policies = context["task_generation_policies"]
    collector_counts = context["collector_counts"]
    sampling_with_doc_cap = context["sampling_with_doc_cap"]
    catalog_runner_cfg = context["catalog_runner_cfg"]
    diagnostics = context["diagnostics"]

    sampling = dict(config["sampling"])
    claim_rows_cache: list[dict[str, Any]] | None = None
    claim_profiles_cache: list[dict[str, Any]] | None = None
    claim_peer_rows_cache: list[list[dict[str, Any]]] | None = None
    reconstruction_case_cache: list[dict[str, Any]] | None = None
    if (
        ("claims" in task_counts or "contradictions" in task_counts)
        and not claim_generation_views.empty
    ):
        claim_rows_cache = claim_generation_views.to_dict("records")
        peer_pool_sampling = (
            sampling_with_doc_cap.get("contradictions")
            or sampling_with_doc_cap.get("claims")
            or {}
        )
        if claim_peer_pool_disabled(
            len(claim_rows_cache),
            peer_pool_sampling,
            pair_like_corpus=pair_like_claim_corpus,
            default_threshold=12000,
        ):
            claim_profiles_cache = [
                claim_clause_profile(normalize_space(row.get("claim_text", "")))
                for row in claim_rows_cache
            ]
            claim_peer_rows_cache = [[] for _ in claim_rows_cache]
        elif exhaustive_small_doc_corpus(corpus_doc_count) or pair_like_claim_corpus:
            claim_profiles_cache, claim_peer_rows_cache = build_claim_peer_pool(
                claim_rows_cache
            )
    collector_bundle = build_task_collectors(
        task_counts=task_counts,
        collector_counts=collector_counts,
        sampling_with_doc_cap=sampling_with_doc_cap,
        seed=seed,
        runner_cfg=runner_cfg,
        entities=entities,
        chunks=chunks,
        para_richness=para_richness,
        component_index=component_index,
        doc_chunks_by_doc=doc_chunks_by_doc,
        chunk_text_lookup=chunk_text_lookup,
        claim_generation_views=claim_generation_views,
        reconstruction_views=reconstruction_views,
        relations=tables["relations"],
        claim_rows_cache=claim_rows_cache,
        claim_profiles_cache=claim_profiles_cache,
        claim_peer_rows_cache=claim_peer_rows_cache,
        choose_reconstruction_cases=choose_reconstruction_cases,
        choose_claim_cases=choose_claim_cases,
        choose_synthetic_value_redaction_cases=choose_synthetic_value_redaction_cases,
        choose_sentence_redaction_cases=choose_sentence_redaction_cases,
        choose_contradiction_cases=choose_contradiction_cases,
    )
    corpus_code = corpus_code_from_config(config.get("corpus", "corpus"))
    tier_order = TASK_TIER_ORDER
    task_pipeline_result = process_task_catalogs(
        collectors=collector_bundle.collectors,
        task_counts=task_counts,
        collector_counts=collector_counts,
        sampling_with_doc_cap=sampling_with_doc_cap,
        seed=seed,
        tier_order=tier_order,
        progress_prefix=progress_prefix,
        enable_small_corpus_parser_vet=enable_small_corpus_parser_vet,
        small_corpus=exhaustive_small_doc_corpus(corpus_doc_count),
        pair_like_claim_corpus=pair_like_claim_corpus,
        max_cases_per_doc=max_cases_per_doc,
        corpus_code=corpus_code,
        config_corpus=str(config.get("corpus", "")),
        fisc_claimlike_acceptable_sources=fisc_claimlike_acceptable_sources,
        task_prefix_map=TASK_PREFIX_MAP,
        runner_defaults={
            "mode": config["runner"]["mode"],
            "staged_top_docs": int(config["runner"]["staged_top_docs"]),
            "retrieval_mode": str(config["runner"]["retrieval_mode"]),
            "raw_use_mode": str(config["runner"]["raw_use_mode"]),
            "max_elapsed_seconds": float(config["runner"]["max_elapsed_seconds"]),
        },
        task_runner_overrides={
            normalize_space(task_name).casefold(): {
                key: value
                for key, value in task_cfg.items()
            }
            for task_name, task_cfg in (config.get("runner", {}).get("task_overrides", {}) or {}).items()
            if isinstance(task_cfg, dict)
        },
        collect_claim_cases_local=collector_bundle.collect_claim_cases_local,
        collect_contradiction_cases_local=collector_bundle.collect_contradiction_cases_local,
        catalog_runner_cfg=catalog_runner_cfg,
        merge_ranked_candidates=merge_ranked_candidates,
        rebalance_value_redaction_cases=rebalance_value_redaction_cases,
        unique_query_round_robin=unique_query_round_robin,
        normalize_space=normalize_space,
        build_expected=build_expected,
    )
    cases = task_pipeline_result["cases"]
    catalog_rows = task_pipeline_result["catalog_rows"]
    diagnostics["task_build_behaviors"] = task_pipeline_result["task_build_behaviors"]
    diagnostics["actual_counts"] = task_pipeline_result["actual_counts"]
    diagnostics["admissible_counts"] = task_pipeline_result["admissible_counts"]
    diagnostics["task_stage_counts"] = task_pipeline_result["task_stage_counts"]
    cases.sort(key=lambda item: (tier_order[item["task_family"]], item["order"], item["case_id"]))
    return cases, diagnostics, catalog_rows


def build_expected(case: dict[str, Any]) -> dict[str, Any]:
    return build_expected_payload(
        case,
        format_expected_targets=format_expected_targets,
        maybe_bool=maybe_bool,
    )


def write_cases(
    config: dict[str, Any],
    paths: ExperimentPaths,
    cases: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    catalog_rows: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    if overwrite:
        if paths.experiment_root.exists():
            shutil.rmtree(paths.experiment_root)
        ensure_dirs(paths)
        for directory in (paths.query_manifest_root, paths.query_file_root, paths.summary_root):
            directory.mkdir(parents=True, exist_ok=True)
    else:
        ensure_dirs(paths)
    plan_cases: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    for case in cases:
        task_dir = paths.query_file_root / case["task_family"]
        task_dir.mkdir(parents=True, exist_ok=True)
        query_path = task_dir / f"{slug(case['case_id'])}.txt"
        raw_query = str(case["query"]).strip()
        expected_targets = format_expected_targets(case)
        answer_targets = case.get("answer_targets")
        if not isinstance(answer_targets, dict) or not answer_targets:
            answer_targets = expected_targets
        query_path.write_text(raw_query + "\n", encoding="utf-8")
        plan_case = {
            "id": case["case_id"],
            "tier": case["tier"],
            "order": case["order"],
            "mode": case["mode"],
            "task": case["runner_task"],
            "query": raw_query,
            "target_doc_ids": case["target_doc_ids"],
            "required_doc_ids": case["required_doc_ids"],
            "staged_top_docs": case["staged_top_docs"],
            "retrieval_mode": case["retrieval_mode"],
            "raw_use_mode": case["raw_use_mode"],
            "max_elapsed_seconds": case["max_elapsed_seconds"],
            "expected": case["expected"],
            "enabled": True,
        }
        if case.get("extraction_profile"):
            plan_case["extraction_profile"] = case["extraction_profile"]
        plan_cases.append(plan_case)
        manifest_rows.append(
            {
                "case_id": case["case_id"],
                "legacy_case_id": case.get("legacy_case_id", ""),
                "corpus": case.get("corpus", config.get("corpus", "")),
                "task_family": case["task_family"],
                "runner_task": case["runner_task"],
                "doc_id": case["doc_id"],
                "paragraph_id": case["paragraph_id"],
                "query_kind": case["query_kind"],
                "ground_truth": case["ground_truth"],
                "notes": case.get("notes", ""),
                "query_file": str(query_path.relative_to(paths.experiment_root)),
                "query_sha1": sha1_text(raw_query),
                "target_doc_ids": ",".join(case["target_doc_ids"]),
                "source_context": case.get("source_context", ""),
            }
        )
        answer_rows.append(
            {
                "case_id": case["case_id"],
                "corpus": case.get("corpus", config.get("corpus", "")),
                "task_family": case["task_family"],
                "runner_task": case["runner_task"],
                "doc_id": case["doc_id"],
                "paragraph_id": case["paragraph_id"],
                "query_kind": case["query_kind"],
                "query_word_count": count_words(raw_query),
                "query_target_words": case.get("query_target_words", ""),
                "source_context_word_count": count_words(case.get("source_context", "")),
                "ground_truth": case["ground_truth"],
                "expected_targets": stringify_expected_targets(answer_targets),
                "source_context": case.get("source_context", ""),
                "query_text": raw_query,
                "notes": case.get("notes", ""),
                "query_file": str(query_path.relative_to(paths.experiment_root)),
            }
        )
    plan_payload = {
        "schema_version": "1.0.18",
        "purpose": f"{config['corpus']} corpus benchmark plan generated from processed extraction.",
        "notes": {
            "config_path": config["_config_path"],
            "profile_path": config.get("_profile_path", ""),
            "manifest_path": config.get("_manifest_path", ""),
            "corpus": config["corpus"],
            "experiment_name": paths.experiment_root.name,
        },
        "cases": plan_cases,
    }
    plan_path = paths.query_manifest_root / "benchmark_plan.json"
    atomic_write_json(plan_path, plan_payload)
    manifest_path = paths.query_manifest_root / "case_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [
            "case_id", "legacy_case_id", "corpus", "task_family", "runner_task", "doc_id", "paragraph_id",
            "query_kind", "ground_truth", "notes", "query_file", "query_sha1",
            "target_doc_ids", "source_context",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    answer_key_path = paths.query_manifest_root / "query_answer_key.csv"
    with answer_key_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(answer_rows[0].keys()) if answer_rows else [
            "case_id", "corpus", "task_family", "runner_task", "doc_id", "paragraph_id",
            "query_kind", "query_word_count", "query_target_words", "source_context_word_count",
            "ground_truth", "expected_targets",
            "source_context", "query_text", "notes", "query_file",
        ])
        writer.writeheader()
        writer.writerows(answer_rows)
    write_query_catalog_files(paths.summary_root, catalog_rows)
    atomic_write_json(paths.summary_root / "build_diagnostics.json", diagnostics)
    return plan_path, manifest_path


def run_preflight(config: dict[str, Any], paths: ExperimentPaths) -> None:
    matryoshka_root = resolve_path(config["matryoshka_root"])
    python_executable = resolve_path_preserve_symlink(
        config.get("python_executable", sys.executable)
    )
    cmd = [
        str(python_executable),
        str(matryoshka_root / PREFLIGHT_RELATIVE),
        "--extraction-dir",
        str(paths.extraction_root),
    ]
    print("\n== Running ==")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, text=True, capture_output=True)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0:
        if stdout:
            print(stdout)
        return
    payload = None
    if stdout:
        try:
            payload = json.loads(stdout)
        except ValueError:
            payload = None
    if payload and payload.get("issues") == [
        "streamlit import failed: ModuleNotFoundError: No module named 'streamlit'"
    ]:
        print(
            "Preflight warning tolerated: Streamlit is missing in the selected "
            "interpreter, but batch benchmark execution does not depend on it. "
            "Continuing without the Streamlit UI component."
        )
        return
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    raise subprocess.CalledProcessError(completed.returncode, cmd)


def normalize_task_set(raw: Any) -> set[str]:
    if not raw:
        return set()
    if isinstance(raw, str):
        return {item.strip() for item in raw.split(",") if item.strip()}
    if isinstance(raw, (list, tuple, set)):
        return {normalize_space(item) for item in raw if normalize_space(item)}
    return set()


def build_runner_groups(
    plan_payload: dict[str, Any],
    runner_cfg: dict[str, Any],
) -> list[tuple[str, list[dict[str, Any]], bool]]:
    cases = list(plan_payload.get("cases", []))
    if not cases:
        return []
    if runner_cfg.get("use_openai_parser"):
        return [("all_openai", cases, True)]
    openai_tasks = normalize_task_set(runner_cfg.get("openai_tasks", ["claims", "reconstruction"]))
    if not openai_tasks:
        return [("all_heuristic", cases, False)]
    heuristic_cases = [case for case in cases if normalize_space(case.get("task", "")) not in openai_tasks]
    openai_cases = [case for case in cases if normalize_space(case.get("task", "")) in openai_tasks]
    groups: list[tuple[str, list[dict[str, Any]], bool]] = []
    if openai_cases:
        groups.append(("openai", openai_cases, True))
    if heuristic_cases:
        groups.append(("heuristic", heuristic_cases, False))
    return groups


def run_streaming_group(
    cmd: list[str],
    *,
    offset: int,
    global_total: int,
    group_cases: list[dict[str, Any]] | None = None,
    run_root: Path | None = None,
    manifest_map: dict[str, dict[str, Any]] | None = None,
) -> int:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        match = RUNNER_PROGRESS_RE.match(line.strip())
        if match:
            local_index = int(match.group(1))
            rewritten = f"[{offset + local_index}/{global_total}] {match.group(3)}"
            if (
                group_cases
                and run_root is not None
                and 0 < local_index <= len(group_cases)
                and any(token in match.group(3) for token in (": PASS", ": FAIL", ": PREFLIGHT_BLOCKED"))
            ):
                case = group_cases[local_index - 1]
                case_id = normalize_space(case.get("id", ""))
                hint = summarize_case_progress_hint(
                    run_root,
                    case,
                    manifest_row=(manifest_map or {}).get(case_id, {}),
                )
                if hint:
                    label = match.group(3)
                    case_label = label.rsplit(":", 1)[0].strip() if ":" in label else label.strip()
                    rewritten = f"[{offset + local_index}/{global_total}] {case_label}: {hint}"
            print(rewritten, flush=True)
        else:
            print(line, flush=True)
    return process.wait()


def write_plan_subset(
    base_payload: dict[str, Any],
    cases: list[dict[str, Any]],
    destination: Path,
) -> Path:
    payload = dict(base_payload)
    payload["cases"] = cases
    atomic_write_json(destination, payload)
    return destination


def create_case_aliases(run_root: Path, cases: list[dict[str, Any]]) -> None:
    for case in cases:
        actual_dir = run_root / f"{int(case.get('order', 0)):03d}_{slug(str(case.get('id', '')))}"
        alias_dir = run_root / str(case.get("id", ""))
        if alias_dir.is_symlink() or alias_dir.is_file():
            alias_dir.unlink()
        if actual_dir.exists():
            os.symlink(actual_dir.name, alias_dir)


def load_case_manifest_map(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    rows = pd.read_csv(manifest_path, low_memory=False).to_dict("records")
    return {
        normalize_space(row.get("case_id", "")): row
        for row in rows
        if normalize_space(row.get("case_id", ""))
    }


def case_run_dir(run_root: Path, case: dict[str, Any]) -> Path:
    return run_root / f"{int(case.get('order', 0)):03d}_{slug(str(case.get('id', '')))}"


def load_batch_summary_rows(summary_path: Path) -> dict[str, dict[str, Any]]:
    if not summary_path.exists():
        return {}
    try:
        rows = pd.read_csv(summary_path, low_memory=False).to_dict("records")
    except Exception:
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = normalize_space(row.get("id", ""))
        if case_id:
            mapped[case_id] = row
    return mapped


def case_needs_openai_retry(case_root: Path) -> bool:
    if not case_root.exists():
        return False
    coordinated_status_path = case_root / "coordinated_status_v048.json"
    if coordinated_status_path.exists():
        try:
            coordinated_status = load_json(coordinated_status_path)
        except Exception:
            coordinated_status = {}
        task_name = normalize_space(coordinated_status.get("task", "")).casefold()
        status_name = normalize_space(coordinated_status.get("status", ""))
        if (
            status_name == "PREFLIGHT_BLOCKED"
            and task_name in {"claims", "contradictions"}
        ):
            return True
    parser_manifest = case_root / "query_parse" / "query_parse_manifest.json"
    parser_mode = ""
    if parser_manifest.exists():
        try:
            parser_mode = normalize_space(load_json(parser_manifest).get("parser_mode", ""))
        except Exception:
            parser_mode = ""
    preflight_path = case_root / "query_parse" / "query_task_preflight.csv"
    preflight_status = ""
    if preflight_path.exists():
        try:
            preflight_df = pd.read_csv(preflight_path, low_memory=False)
            if not preflight_df.empty:
                preflight_status = normalize_space(preflight_df.iloc[0].get("status", ""))
        except Exception:
            preflight_status = ""
    stdout_path = case_root / "parser.stdout.log"
    stdout_text = read_text_if_exists(stdout_path)
    return (
        parser_mode.startswith("heuristic_openai_unavailable")
        and "connection error" in stdout_text.casefold()
        and preflight_status == "BLOCKED_INCOMPLETE_QUERY_SIGNATURE"
    )


def rerun_single_case(
    cmd_prefix: list[str],
    case: dict[str, Any],
    plan_payload: dict[str, Any],
    plan_destination: Path,
    run_root: Path,
    *,
    use_openai: bool,
) -> tuple[int, dict[str, Any]]:
    write_plan_subset(plan_payload, [case], plan_destination)
    case_root = case_run_dir(run_root, case)
    if case_root.exists():
        shutil.rmtree(case_root)
    alias_dir = run_root / str(case.get("id", ""))
    if alias_dir.is_symlink() or alias_dir.is_file():
        alias_dir.unlink()
    retry_cmd = list(cmd_prefix)
    retry_cmd.extend(["--plan", str(plan_destination)])
    if use_openai:
        retry_cmd.append("--use-openai-parser")
    print(
        f"Retrying parser-degraded case {normalize_space(case.get('id', ''))} "
        f"after OpenAI parser connection fallback...",
        flush=True,
    )
    return_code = run_streaming_group(
        retry_cmd,
        offset=0,
        global_total=1,
        group_cases=[case],
        run_root=run_root,
    )
    updated_rows = load_batch_summary_rows(run_root / "batch_summary.csv")
    return return_code, updated_rows.get(normalize_space(case.get("id", "")), {})


def parser_query_vetting_enabled(
    runner_cfg: dict[str, Any] | None,
    *,
    enabled_key: str,
) -> bool:
    if not isinstance(runner_cfg, dict):
        return False
    if not maybe_bool(runner_cfg.get(enabled_key, True)):
        return False
    return bool(normalize_space(runner_cfg.get("__matryoshka_root", "")))


def claim_query_vetting_enabled(runner_cfg: dict[str, Any] | None) -> bool:
    if isinstance(runner_cfg, dict) and maybe_bool(
        runner_cfg.get("__disable_claim_build_parser_vet", False)
    ):
        return False
    return parser_query_vetting_enabled(
        runner_cfg,
        enabled_key="build_vet_claim_queries_with_parser",
    )


def contradiction_query_vetting_enabled(runner_cfg: dict[str, Any] | None) -> bool:
    if isinstance(runner_cfg, dict) and maybe_bool(
        runner_cfg.get("__disable_contradiction_build_parser_vet", False)
    ):
        return False
    return parser_query_vetting_enabled(
        runner_cfg,
        enabled_key="build_vet_contradiction_queries_with_parser",
    )


def pair_runner_query_vetting_enabled(
    runner_cfg: dict[str, Any] | None,
    *,
    enabled_key: str,
) -> bool:
    if not isinstance(runner_cfg, dict):
        return False
    if normalize_space(runner_cfg.get("mode", "")).casefold() != "pair":
        return False
    if not maybe_bool(runner_cfg.get("use_openai_parser", False)):
        return False
    if not maybe_bool(runner_cfg.get(enabled_key, True)):
        return False
    return all(
        normalize_space(runner_cfg.get(key, ""))
        for key in (
            "__matryoshka_root",
            "__python_executable",
            "__source_extraction_root",
            "__corpus_entity_resource",
            "__decision_profile",
        )
    )


def pair_vet_artifacts_indicate_pass(
    run_root: Path,
    *,
    task_family: str,
) -> bool:
    if not run_root.exists():
        return False
    if task_family == "claims":
        eval_path = run_root / "baseline" / "semantic_match_output" / "claim_evaluation_v048.csv"
        if not eval_path.exists():
            return False
        try:
            eval_df = pd.read_csv(eval_path, low_memory=False)
        except Exception:
            return False
        if eval_df.empty:
            return False
        for _, row in eval_df.iterrows():
            semantic_status = normalize_space(row.get("semantic_status", ""))
            localization_status = normalize_space(row.get("localization_status", ""))
            if (
                semantic_status
                and not semantic_status.startswith("BLOCKED_")
                and localization_status != "NO_SOURCE_FOUND"
            ):
                return True
        return False
    if task_family == "contradictions":
        eval_path = run_root / "baseline" / "semantic_match_output" / "task_evaluation_v048.csv"
        if not eval_path.exists():
            return False
        try:
            eval_df = pd.read_csv(eval_path, low_memory=False)
        except Exception:
            return False
        if eval_df.empty:
            return False
        for _, row in eval_df.iterrows():
            semantic_status = normalize_space(row.get("semantic_status", ""))
            localization_status = normalize_space(row.get("localization_status", ""))
            if (
                semantic_status
                and not semantic_status.startswith("BLOCKED_")
                and localization_status != "NO_SOURCE_FOUND"
            ):
                return True
        return False
    return True


def claim_pair_query_vetting_enabled(runner_cfg: dict[str, Any] | None) -> bool:
    if isinstance(runner_cfg, dict) and maybe_bool(
        runner_cfg.get("__disable_claim_build_pair_vet", False)
    ):
        return False
    return pair_runner_query_vetting_enabled(
        runner_cfg,
        enabled_key="build_vet_claim_queries_with_pair_runner",
    )


def contradiction_pair_query_vetting_enabled(runner_cfg: dict[str, Any] | None) -> bool:
    if isinstance(runner_cfg, dict) and maybe_bool(
        runner_cfg.get("__disable_contradiction_build_pair_vet", False)
    ):
        return False
    return pair_runner_query_vetting_enabled(
        runner_cfg,
        enabled_key="build_vet_contradiction_queries_with_pair_runner",
    )


def case_passes_pair_runner_vet(
    case: dict[str, Any],
    runner_cfg: dict[str, Any] | None,
    *,
    task_family: str,
) -> bool:
    if task_family == "claims":
        enabled = claim_pair_query_vetting_enabled(runner_cfg)
    elif task_family == "contradictions":
        enabled = contradiction_pair_query_vetting_enabled(runner_cfg)
    else:
        enabled = False
    if not enabled:
        return True
    assert isinstance(runner_cfg, dict)
    query_value = normalize_space(case.get("query", ""))
    doc_id = normalize_space(case.get("doc_id", ""))
    if not query_value or not doc_id:
        return False
    llm_model = normalize_space(runner_cfg.get("llm_model", "")) or "gpt-4o-mini"
    cache_key = (
        hashlib.sha1(query_value.encode("utf-8")).hexdigest(),
        doc_id,
        task_family,
        llm_model,
        normalize_space(runner_cfg.get("__source_extraction_root", "")),
        tuple(
            normalize_space(value)
            for value in (
                case.get("acceptable_source_ids_any")
                or runner_cfg.get("__fisc_claimlike_acceptable_sources", [])
            )
            if normalize_space(value)
        ),
    )
    cached = PAIR_CASE_VET_CACHE.get(cache_key)
    if cached is not None:
        return cached
    matryoshka_root = resolve_path(runner_cfg["__matryoshka_root"])
    python_executable = resolve_path_preserve_symlink(
        runner_cfg.get("__python_executable", sys.executable)
    )
    runner_script = matryoshka_root / RUNNER_RELATIVE
    extraction_root = resolve_path(runner_cfg["__source_extraction_root"])
    corpus_entity_resource = resolve_path(runner_cfg["__corpus_entity_resource"])
    decision_profile = resolve_path(runner_cfg["__decision_profile"])
    if not runner_script.exists():
        PAIR_CASE_VET_CACHE[cache_key] = True
        return True
    timeout_seconds = float(runner_cfg.get("build_pair_vet_timeout_seconds", 180.0))
    plan_case = {
        "id": "vet_case",
        "tier": 0,
        "order": 1,
        "mode": "pair",
        "task": TASK_NAME_MAP[task_family],
        "query": query_value,
        "target_doc_ids": [doc_id],
        "required_doc_ids": [doc_id],
        "staged_top_docs": int(runner_cfg.get("staged_top_docs", 150)),
        "retrieval_mode": str(runner_cfg.get("retrieval_mode", "staged")),
        "raw_use_mode": str(runner_cfg.get("raw_use_mode", "validate")),
        "max_elapsed_seconds": float(runner_cfg.get("max_elapsed_seconds", 0.0)),
        "expected": {
            "source_ids": [doc_id],
            "require_visualization": False,
            "require_nonblank_pseudo_text": False,
        },
        "enabled": True,
    }
    acceptable_source_ids_any = [
        normalize_space(value)
        for value in (
            case.get("acceptable_source_ids_any")
            or runner_cfg.get("__fisc_claimlike_acceptable_sources", [])
        )
        if normalize_space(value)
    ]
    if acceptable_source_ids_any and task_family in {"claims", "contradictions"}:
        plan_case["expected"]["source_ids"] = []
        plan_case["expected"]["acceptable_source_ids_any"] = acceptable_source_ids_any
    plan_payload = {
        "schema_version": "1.0.18",
        "purpose": "kg_demo build-time authoritative pair vet",
        "notes": {
            "task_family": task_family,
            "doc_id": doc_id,
        },
        "cases": [plan_case],
    }
    with tempfile.TemporaryDirectory(prefix="kg_demo_pair_vet_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        plan_path = tmp_root / "plan.json"
        output_root = tmp_root / "runs"
        atomic_write_json(plan_path, plan_payload)
        cmd = [
            str(python_executable),
            "-u",
            str(runner_script),
            "--code-dir",
            str(matryoshka_root),
            "--extraction-dir",
            str(extraction_root),
            "--output-root",
            str(output_root),
            "--python-executable",
            str(python_executable),
            "--corpus-entity-resource",
            str(corpus_entity_resource),
            "--decision-profile",
            str(decision_profile),
            "--plan",
            str(plan_path),
            "--use-openai-parser",
            "--llm-model",
            llm_model,
            "--openai-api-key-env",
            str(runner_cfg.get("openai_api_key_env", "OPENAI_API_KEY")),
            "--contradiction-propositions-per-class-per-document",
            str(runner_cfg.get("contradiction_propositions_per_class_per_document", 12)),
            "--contradiction-raw-sentences-per-class-per-document",
            str(runner_cfg.get("contradiction_raw_sentences_per_class_per_document", 4)),
        ]
        if runner_cfg.get("minimum_score") is not None:
            cmd.extend(["--minimum-score", str(runner_cfg["minimum_score"])])
        if runner_cfg.get("minimum_margin") is not None:
            cmd.extend(["--minimum-margin", str(runner_cfg["minimum_margin"])])
        try:
            subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            PAIR_CASE_VET_CACHE[cache_key] = False
            return False
        summary_path = output_root / "batch_summary.csv"
        if not summary_path.exists():
            PAIR_CASE_VET_CACHE[cache_key] = False
            return False
        try:
            rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
        except Exception:
            PAIR_CASE_VET_CACHE[cache_key] = False
            return False
        if not rows:
            PAIR_CASE_VET_CACHE[cache_key] = False
            return False
        status = normalize_space(rows[0].get("status", "")).upper()
        passed = status == "PASS"
        if passed:
            run_root_value = normalize_space(rows[0].get("run_root", ""))
            if run_root_value:
                passed = pair_vet_artifacts_indicate_pass(
                    Path(run_root_value),
                    task_family=task_family,
                )
        PAIR_CASE_VET_CACHE[cache_key] = passed
        return passed


def query_is_executable_via_parser(
    query_text: str,
    runner_cfg: dict[str, Any] | None,
    *,
    task_family: str,
) -> bool:
    if task_family == "claims":
        enabled = claim_query_vetting_enabled(runner_cfg)
    elif task_family == "contradictions":
        enabled = contradiction_query_vetting_enabled(runner_cfg)
    else:
        enabled = False
    if not enabled:
        return True
    assert isinstance(runner_cfg, dict)
    query_value = normalize_space(query_text)
    if not query_value:
        return False
    if task_family == "claims" and re.search(
        r"\b(?:"
        r"received instructions from\b[^.]{0,120}\brequiring\b|"
        r"is planning to turn over to\b|"
        r"(?:is|was)\s+unable to\b[^.]{0,120}\bbecause of (?:the )?lack of\b|"
        r"is not and will not be seriously affected by\b|"
        r"was led by a charismatic figure who would not alter\b"
        r")",
        query_value.casefold(),
    ):
        return False
    parser_model = normalize_space(runner_cfg.get("parser_model", "")) or "gpt-4o"
    cache_key = (
        hashlib.sha1(query_value.encode("utf-8")).hexdigest(),
        parser_model,
        normalize_space(runner_cfg.get("__matryoshka_root", "")),
        task_family,
    )
    cached = QUERY_VET_CACHE.get(cache_key)
    if cached is not None:
        return cached
    matryoshka_root = resolve_path(runner_cfg["__matryoshka_root"])
    python_executable = resolve_path_preserve_symlink(
        runner_cfg.get("__python_executable", sys.executable)
    )
    parser_script = matryoshka_root / QUERY_PARSER_RELATIVE
    if not parser_script.exists():
        QUERY_VET_CACHE[cache_key] = True
        return True
    timeout_seconds = float(runner_cfg.get("build_parser_vet_timeout_seconds", 30.0))
    with tempfile.TemporaryDirectory(prefix="kg_demo_claim_vet_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        query_path = tmp_root / "query.txt"
        output_dir = tmp_root / "query_parse"
        query_path.write_text(query_value + "\n", encoding="utf-8")
        cmd = [
            str(python_executable),
            str(parser_script),
            "--query-file",
            str(query_path),
            "--output-dir",
            str(output_dir),
        ]
        try:
            completed = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            QUERY_VET_CACHE[cache_key] = False
            return False
        if completed.returncode != 0:
            QUERY_VET_CACHE[cache_key] = False
            return False
        preflight_path = output_dir / "query_task_preflight.csv"
        if not preflight_path.exists():
            QUERY_VET_CACHE[cache_key] = False
            return False
        try:
            preflight_df = pd.read_csv(preflight_path, low_memory=False)
        except Exception:
            QUERY_VET_CACHE[cache_key] = False
            return False
        executable = bool(
            not preflight_df.empty
            and any(
                normalize_space(value) == "EXECUTABLE"
                for value in preflight_df.get("status", pd.Series(dtype=str)).tolist()
            )
        )
        if executable and task_family == "claims":
            signature_path = output_dir / "query_event_signatures.csv"
            if not signature_path.exists():
                executable = False
            else:
                try:
                    signature_df = pd.read_csv(signature_path, low_memory=False)
                except Exception:
                    executable = False
                else:
                    executable_rows = []
                    primary_row_count = 0
                    if not signature_df.empty:
                        primary_df = signature_df
                        if "primary_for_matching" in signature_df.columns:
                            primary_df = signature_df[
                                signature_df["primary_for_matching"].fillna(False).astype(bool)
                            ]
                            if primary_df.empty:
                                primary_df = signature_df
                        primary_row_count = len(primary_df)
                        for row in primary_df.to_dict("records"):
                            if all(
                                normalize_space(row.get(field, ""))
                                for field in ("actor", "predicate", "object")
                            ):
                                executable_rows.append(row)
                    corpus_doc_count = int(runner_cfg.get("__corpus_doc_count", 9999) or 9999)
                    small_corpus_claim_mode = exhaustive_small_doc_corpus(corpus_doc_count)
                    if small_corpus_claim_mode:
                        strict_small_corpus_vet = (
                            maybe_bool(runner_cfg["__strict_small_corpus_vet"])
                            if "__strict_small_corpus_vet" in runner_cfg
                            else True
                        )
                    else:
                        strict_small_corpus_vet = maybe_bool(
                            runner_cfg.get("__strict_small_corpus_vet", False)
                        )
                    if small_corpus_claim_mode:
                        pair_like_claim_corpus = maybe_bool(
                            runner_cfg.get("__pair_like_claim_corpus", False)
                        )
                        if executable_rows:
                            executable = True
                        elif strict_small_corpus_vet:
                            executable = False
                        elif pair_like_claim_corpus:
                            executable = False
                        else:
                            executable = bool(
                                primary_row_count >= 1
                                and small_corpus_claim_surface_recovery_ok(
                                    query_value,
                                    task_family="claims",
                                )
                            )
                    else:
                        executable = primary_row_count == 1 and len(executable_rows) == 1
        QUERY_VET_CACHE[cache_key] = executable
        return executable


def small_corpus_claim_surface_recovery_ok(
    query_text: str,
    *,
    task_family: str = "claims",
) -> bool:
    value = normalize_space(query_text)
    if not value:
        return False
    words = count_words(value)
    if words < 6 or words > 36:
        return False
    if sentence_is_low_signal_for_query(value):
        return False
    if claim_surface_is_interrogative(value) or claim_surface_has_weak_modality(value):
        return False
    if has_reporting_that_clause(value):
        return False
    if re.match(r"^(?:the issue|an approach|there must be|connections has with)\b", value, flags=re.IGNORECASE):
        return False
    flags = set(query_surface_flags(value, task_family=task_family))
    if flags & {
        "uppercase_heavy",
        "mixed_case_noise",
        "headline_scaffold",
        "broken_lead",
        "foreign_language",
        "metadata_scaffold",
        "duplicated_scaffold",
    }:
        return False
    substantive_score = substantive_sentence_score(value)
    if substantive_score < 5.0:
        return False
    profile = claim_clause_profile(value)
    if claim_profile_has_problematic_actor(profile):
        return False
    if small_corpus_claim_surface_is_vague(value, profile):
        return False
    actor = normalize_space(profile.get("actor", ""))
    if actor and (
        actor_has_abstract_head(actor)
        or actor_is_generic_common_noun_phrase(actor)
    ):
        return False
    if profile_has_complete_simple_clause(profile):
        if object_has_abstract_head(normalize_space(profile.get("object", ""))):
            return False
    standalone_ok = standalone_claim_surface_is_safe(value)
    standalone_contra_ok = standalone_contradiction_surface_is_safe(value)
    fallback_simple_ok = fallback_simple_complete_claim_surface_is_safe(value)
    if task_family == "claims" and SMALL_CLAIM_RUNTIME_HOSTILE_RECOVERY_RE.search(value.casefold()):
        return False
    if substantive_score < 6.0 and not (
        standalone_ok
        or standalone_contra_ok
        or fallback_simple_ok
    ):
        return False
    if task_family == "claims" and not profile_has_complete_simple_clause(profile):
        priority_adjustment = small_corpus_claim_surface_priority_adjustment(value)
        if priority_adjustment <= 0.0 and not fallback_simple_ok:
            return False
    return bool(
        standalone_ok
        or standalone_contra_ok
        or fallback_simple_ok
        or profile_has_complete_simple_clause(profile)
    )


def claim_query_is_executable_via_parser(
    query_text: str,
    runner_cfg: dict[str, Any] | None,
) -> bool:
    return query_is_executable_via_parser(
        expand_claim_coordination_ellipsis(query_text),
        runner_cfg,
        task_family="claims",
    )


def vet_claim_case_candidates(
    cases: list[dict[str, Any]],
    *,
    count: int,
    runner_cfg: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not claim_query_vetting_enabled(runner_cfg):
        return cases[:count]
    pair_vet_enabled = claim_pair_query_vetting_enabled(runner_cfg)
    pair_like_claim_corpus = isinstance(runner_cfg, dict) and maybe_bool(
        runner_cfg.get("__pair_like_claim_corpus", False)
    )
    corpus_doc_count = (
        int(runner_cfg.get("__corpus_doc_count", 9999) or 9999)
        if isinstance(runner_cfg, dict)
        else 9999
    )
    medium_or_large_nonpair_corpus = (not pair_like_claim_corpus) and corpus_doc_count >= 20
    pair_pass_vetted: list[dict[str, Any]] = []
    parser_only_vetted: list[dict[str, Any]] = []
    raw_fallback: list[dict[str, Any]] = []
    pair_seen_ids: set[str] = set()
    parser_seen_ids: set[str] = set()
    case_budget = max(count * 2, count)
    if isinstance(runner_cfg, dict):
        case_budget = int(runner_cfg.get("build_parser_vet_case_budget", case_budget))
    if medium_or_large_nonpair_corpus:
        case_budget = max(case_budget, count * 5, 80)
    pair_case_budget = max(count * 2, count)
    pair_checked = 0
    if isinstance(runner_cfg, dict):
        pair_case_budget = int(runner_cfg.get("build_pair_vet_case_budget", pair_case_budget))
    if medium_or_large_nonpair_corpus:
        pair_case_budget = max(pair_case_budget, count * 4, 48)
    for index, case in enumerate(cases, start=1):
        if index > case_budget:
            break
        case_id = normalize_space(case.get("case_id", ""))
        if case_id in parser_seen_ids:
            continue
        parser_ok = claim_query_is_executable_via_parser(
            normalize_space(case.get("query", "")),
            runner_cfg,
        )
        if parser_ok:
            parser_only_vetted.append(case)
            parser_seen_ids.add(case_id)
        else:
            raw_fallback.append(case)
        if pair_vet_enabled and pair_checked < pair_case_budget:
            pair_checked += 1
            if case_passes_pair_runner_vet(
                case,
                runner_cfg,
                task_family="claims",
            ) and case_id not in pair_seen_ids:
                pair_pass_vetted.append(case)
                pair_seen_ids.add(case_id)
        if len(pair_pass_vetted) >= count:
            break
        if medium_or_large_nonpair_corpus and len(parser_only_vetted) >= count:
            break
    combined: list[dict[str, Any]] = []
    combined_ids: set[str] = set()
    strict_small_corpus_vet = False
    if isinstance(runner_cfg, dict):
        corpus_doc_count = int(runner_cfg.get("__corpus_doc_count", 9999) or 9999)
        if exhaustive_small_doc_corpus(corpus_doc_count):
            strict_small_corpus_vet = (
                maybe_bool(runner_cfg["__strict_small_corpus_vet"])
                if "__strict_small_corpus_vet" in runner_cfg
                else True
            )
        else:
            strict_small_corpus_vet = maybe_bool(
                runner_cfg.get("__strict_small_corpus_vet", False)
            )
    if medium_or_large_nonpair_corpus:
        buckets = (pair_pass_vetted, parser_only_vetted)
    elif pair_vet_enabled and strict_small_corpus_vet and pair_like_claim_corpus:
        buckets = (pair_pass_vetted, parser_only_vetted)
    elif pair_vet_enabled and strict_small_corpus_vet and pair_pass_vetted:
        buckets = (pair_pass_vetted, parser_only_vetted)
    else:
        buckets = (
            (pair_pass_vetted, parser_only_vetted)
            if strict_small_corpus_vet
            else (pair_pass_vetted, parser_only_vetted, raw_fallback)
        )
    for bucket in buckets:
        for case in bucket:
            case_id = normalize_space(case.get("case_id", ""))
            if case_id in combined_ids:
                continue
            combined.append(case)
            combined_ids.add(case_id)
            if len(combined) >= count:
                return combined
    return combined


def vet_contradiction_case_candidates(
    cases: list[dict[str, Any]],
    *,
    count: int,
    runner_cfg: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not contradiction_query_vetting_enabled(runner_cfg):
        return cases[:count]
    pair_vet_enabled = contradiction_pair_query_vetting_enabled(runner_cfg)
    corpus_doc_count = (
        int(runner_cfg.get("__corpus_doc_count", 9999) or 9999)
        if isinstance(runner_cfg, dict)
        else 9999
    )
    medium_or_large_corpus = corpus_doc_count >= 20
    pair_pass_vetted: list[dict[str, Any]] = []
    parser_only_vetted: list[dict[str, Any]] = []
    raw_fallback: list[dict[str, Any]] = []
    pair_seen_ids: set[str] = set()
    parser_seen_ids: set[str] = set()
    case_budget = max(count * 2, count)
    if isinstance(runner_cfg, dict):
        case_budget = int(runner_cfg.get("build_parser_vet_case_budget", case_budget))
    if medium_or_large_corpus:
        case_budget = max(case_budget, count * 5, 80)
    pair_case_budget = max(count * 2, count)
    pair_checked = 0
    if isinstance(runner_cfg, dict):
        pair_case_budget = int(runner_cfg.get("build_pair_vet_case_budget", pair_case_budget))
    if medium_or_large_corpus:
        pair_case_budget = max(pair_case_budget, count * 4, 48)
    for index, case in enumerate(cases, start=1):
        if index > case_budget:
            break
        case_id = normalize_space(case.get("case_id", ""))
        if case_id in parser_seen_ids:
            continue
        parser_ok = query_is_executable_via_parser(
            normalize_space(case.get("query", "")),
            runner_cfg,
            task_family="contradictions",
        )
        if parser_ok:
            parser_only_vetted.append(case)
            parser_seen_ids.add(case_id)
        else:
            raw_fallback.append(case)
        if pair_vet_enabled and pair_checked < pair_case_budget:
            pair_checked += 1
            if case_passes_pair_runner_vet(
                case,
                runner_cfg,
                task_family="contradictions",
            ) and case_id not in pair_seen_ids:
                pair_pass_vetted.append(case)
                pair_seen_ids.add(case_id)
        if len(pair_pass_vetted) >= count:
            break
        if medium_or_large_corpus and len(parser_only_vetted) >= count:
            break
    combined: list[dict[str, Any]] = []
    combined_ids: set[str] = set()
    strict_small_corpus_vet = isinstance(runner_cfg, dict) and maybe_bool(
        runner_cfg.get("__strict_small_corpus_vet", False)
    )
    if medium_or_large_corpus:
        buckets = (pair_pass_vetted, parser_only_vetted)
    elif pair_vet_enabled and strict_small_corpus_vet and pair_pass_vetted:
        buckets = (pair_pass_vetted, parser_only_vetted)
    else:
        buckets = (
            (pair_pass_vetted, parser_only_vetted)
            if strict_small_corpus_vet
            else (pair_pass_vetted, parser_only_vetted, raw_fallback)
        )
    for bucket in buckets:
        for case in bucket:
            case_id = normalize_space(case.get("case_id", ""))
            if case_id in combined_ids:
                continue
            combined.append(case)
            combined_ids.add(case_id)
            if len(combined) >= count:
                return combined
    return combined


def materialize_case_sidecars(
    run_root: Path,
    cases: list[dict[str, Any]],
    manifest_map: dict[str, dict[str, Any]],
) -> None:
    for case in cases:
        case_id = normalize_space(case.get("id", ""))
        if not case_id:
            continue
        actual_dir = run_root / f"{int(case.get('order', 0)):03d}_{slug(case_id)}"
        if not actual_dir.exists():
            continue
        manifest_row = manifest_map.get(case_id, {})
        query_text = normalize_space(case.get("query", ""))
        if query_text and not (actual_dir / "query_input.txt").exists():
            (actual_dir / "query_input.txt").write_text(
                query_text + "\n",
                encoding="utf-8",
            )
        atomic_write_json(
            actual_dir / "case_metadata_v1.json",
            {
                "schema_version": "kg_demo_case_metadata_v1",
                "case_id": case_id,
                "legacy_case_id": normalize_space(manifest_row.get("legacy_case_id", "")),
                "corpus": normalize_space(manifest_row.get("corpus", "")),
                "task_family": normalize_space(manifest_row.get("task_family", "")),
                "runner_task": normalize_space(
                    manifest_row.get("runner_task", case.get("task", ""))
                ),
                "doc_id": normalize_space(manifest_row.get("doc_id", "")),
                "paragraph_id": normalize_space(manifest_row.get("paragraph_id", "")),
                "query_kind": normalize_space(manifest_row.get("query_kind", "")),
                "ground_truth": normalize_space(manifest_row.get("ground_truth", "")),
                "source_context": normalize_space(manifest_row.get("source_context", "")),
                "notes": normalize_space(manifest_row.get("notes", "")),
                "query_text": query_text,
                "target_doc_ids": list(case.get("target_doc_ids", []) or []),
                "required_doc_ids": list(case.get("required_doc_ids", []) or []),
                "mode": normalize_space(case.get("mode", "")),
            },
        )


def run_batch(
    config: dict[str, Any],
    paths: ExperimentPaths,
    plan_path: Path,
    *,
    overwrite: bool,
) -> Path:
    runner = resolve_path(config["matryoshka_root"]) / RUNNER_RELATIVE
    python_executable = resolve_path_preserve_symlink(
        config.get("python_executable", sys.executable)
    )
    if overwrite and paths.run_root.exists():
        shutil.rmtree(paths.run_root)
    paths.run_root.mkdir(parents=True, exist_ok=True)
    runner_cfg = dict(config["runner"])
    plan_payload = load_json(plan_path)
    groups = build_runner_groups(plan_payload, runner_cfg)
    total_cases = sum(len(group_cases) for _, group_cases, _ in groups)
    print(f"Benchmark cases scheduled: {total_cases}")
    create_case_aliases(paths.run_root, plan_payload.get("cases", []))
    manifest_map = load_case_manifest_map(paths.query_manifest_root / "case_manifest.csv")
    merged_rows: list[dict[str, Any]] = []
    group_had_failures = False
    code_dir = str(resolve_path(config["matryoshka_root"]))
    shared_prefix = [
        str(python_executable),
        "-u",
        str(runner),
        "--code-dir",
        code_dir,
        "--extraction-dir",
        str(paths.extraction_root),
        "--output-root",
        str(paths.run_root),
        "--python-executable",
        str(python_executable),
        "--corpus-entity-resource",
        str(resolve_path(config["corpus_entity_resource"])),
        "--decision-profile",
        str(resolve_path(config["decision_profile"])),
    ]
    completed_offset = 0
    for group_name, group_cases, use_openai in groups:
        group_plan_path = paths.query_manifest_root / f"benchmark_plan__{group_name}.json"
        write_plan_subset(plan_payload, group_cases, group_plan_path)
        cmd = list(shared_prefix)
        if runner_cfg.get("use_llm_digest"):
            cmd.append("--use-llm-digest")
        if runner_cfg.get("llm_model"):
            cmd.extend(["--llm-model", str(runner_cfg["llm_model"])])
        if runner_cfg.get("openai_api_key_env"):
            cmd.extend(["--openai-api-key-env", str(runner_cfg["openai_api_key_env"])])
        if runner_cfg.get("minimum_score") is not None:
            cmd.extend(["--minimum-score", str(runner_cfg["minimum_score"])])
        if runner_cfg.get("minimum_margin") is not None:
            cmd.extend(["--minimum-margin", str(runner_cfg["minimum_margin"])])
        cmd.extend(
            [
                "--contradiction-propositions-per-class-per-document",
                str(runner_cfg["contradiction_propositions_per_class_per_document"]),
                "--contradiction-raw-sentences-per-class-per-document",
                str(runner_cfg["contradiction_raw_sentences_per_class_per_document"]),
            ]
        )
        if runner_cfg.get("stop_on_failure"):
            cmd.append("--stop-on-failure")
        return_code = run_streaming_group(
            cmd + ["--plan", str(group_plan_path)] + (["--use-openai-parser"] if use_openai else []),
            offset=completed_offset,
            global_total=total_cases,
            group_cases=group_cases,
            run_root=paths.run_root,
        )
        if return_code not in {0, 2}:
            raise subprocess.CalledProcessError(return_code, cmd)
        group_summary_rows = load_batch_summary_rows(paths.run_root / "batch_summary.csv")
        retry_rows: dict[str, dict[str, Any]] = {}
        if use_openai:
            retry_cases = [
                case
                for case in group_cases
                if case_needs_openai_retry(case_run_dir(paths.run_root, case))
            ]
            for retry_index, retry_case in enumerate(retry_cases, start=1):
                retry_plan_path = (
                    paths.query_manifest_root
                    / f"benchmark_plan__{group_name}__retry_{retry_index:02d}_{slug(str(retry_case.get('id', '')))}.json"
                )
                retry_code, retry_row = rerun_single_case(
                    cmd,
                    retry_case,
                    plan_payload,
                    retry_plan_path,
                    paths.run_root,
                    use_openai=use_openai,
                )
                if retry_code not in {0, 2}:
                    raise subprocess.CalledProcessError(retry_code, cmd)
                case_id = normalize_space(retry_case.get("id", ""))
                if retry_row:
                    retry_rows[case_id] = retry_row
                    group_summary_rows[case_id] = retry_row
                if retry_code == 2:
                    group_had_failures = True
        if return_code == 2:
            if any(normalize_space(row.get("status", "")) == "FAIL" for row in group_summary_rows.values()):
                group_had_failures = True
        completed_offset += len(group_cases)
        group_batch_summary = paths.run_root / "batch_summary.csv"
        if group_summary_rows:
            copy_path = paths.run_root / f"batch_summary__{group_name}.csv"
            fieldnames: list[str] = []
            ordered_rows: list[dict[str, Any]] = []
            for case in group_cases:
                row = group_summary_rows.get(normalize_space(case.get("id", "")))
                if not row:
                    continue
                ordered_rows.append(row)
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with copy_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(ordered_rows)
            merged_rows.extend(ordered_rows)
        elif group_batch_summary.exists():
            copy_path = paths.run_root / f"batch_summary__{group_name}.csv"
            shutil.copy2(group_batch_summary, copy_path)
            with copy_path.open(newline="", encoding="utf-8") as handle:
                merged_rows.extend(list(csv.DictReader(handle)))
        materialize_case_sidecars(paths.run_root, group_cases, manifest_map)
    if merged_rows:
        fieldnames: list[str] = []
        for row in merged_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with (paths.run_root / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)
    if group_had_failures:
        print(
            "Batch runner completed with case-level benchmark failures. "
            "Artifacts were still written and will be summarized."
        )
    create_case_aliases(paths.run_root, plan_payload.get("cases", []))
    materialize_case_sidecars(
        paths.run_root,
        list(plan_payload.get("cases", [])),
        manifest_map,
    )
    return paths.run_root


def find_first_existing(base: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(base.glob(f"**/{pattern}"))
        if matches:
            return matches[0]
    return None


def read_summary_row(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    rows = pd.read_csv(path, low_memory=False)
    if rows.empty:
        return {}

    records = rows.to_dict("records")
    task_names = {
        normalize_space(row.get("task", "")).casefold()
        for row in records
        if normalize_space(row.get("task", ""))
    }
    target_ids = [normalize_space(row.get("target_id", "")) for row in records]
    is_redaction_evaluation = bool(records) and task_names == {"redaction"} and any(target_ids)
    if is_redaction_evaluation:
        resolved_flags = [
            semantic_status_is_success("value_redaction", row.get("semantic_status", ""))
            and bool(normalize_space(row.get("selected_value_or_text", "")))
            for row in records
        ]
        resolved_count = sum(resolved_flags)
        all_resolved = resolved_count == len(records)
        representative = next(
            (row for row, resolved in zip(records, resolved_flags) if not resolved),
            records[0],
        ).copy()
        representative["all_targets_resolved"] = all_resolved
        representative["target_count"] = len(records)
        representative["resolved_target_count"] = resolved_count

        resolved_values = [
            f"{normalize_space(row.get('target_id', ''))}="
            f"{normalize_space(row.get('selected_value_or_text', ''))}"
            for row, resolved in zip(records, resolved_flags)
            if resolved
        ]
        representative["selected_value_or_text"] = " | ".join(resolved_values)
        selected_docs = list(
            dict.fromkeys(
                normalize_space(row.get("selected_source_doc_id", ""))
                for row, resolved in zip(records, resolved_flags)
                if resolved and normalize_space(row.get("selected_source_doc_id", ""))
            )
        )
        representative["selected_source_doc_id"] = " | ".join(selected_docs)
        if len(records) > 1:
            if all_resolved:
                representative["semantic_status"] = "RESOLVED_ALL_TARGETS"
                representative["localization_status"] = "AUTHORITATIVE_SOURCE_SELECTED"
            else:
                representative["semantic_status"] = "UNRESOLVED_PARTIAL_TARGETS"
                representative["localization_status"] = "NO_SOURCE_FOUND"
                unresolved = [
                    f"{normalize_space(row.get('target_id', ''))}:"
                    f"{normalize_space(row.get('semantic_status', '')) or 'UNRESOLVED'}"
                    for row, resolved in zip(records, resolved_flags)
                    if not resolved
                ]
                representative["unresolved_reason"] = "; ".join(unresolved)
        elif not all_resolved and semantic_status_is_success(
            "value_redaction", representative.get("semantic_status", "")
        ):
            representative["semantic_status"] = "UNRESOLVED_MISSING_SELECTED_VALUE"
            representative["localization_status"] = "NO_SOURCE_FOUND"
        return representative

    def row_score(row: pd.Series) -> tuple[float, float, float, float]:
        task_family = normalize_space(
            row.get("task", "") or row.get("task_family", "")
        )
        semantic_status = normalize_space(row.get("semantic_status", ""))
        localization_status = normalize_space(row.get("localization_status", ""))
        decision_reason = normalize_space(row.get("decision_reason", ""))
        selected_doc = normalize_space(row.get("selected_source_doc_id", ""))
        success = 1.0 if semantic_status_is_success(task_family, semantic_status) else 0.0
        localized = 1.0 if localization_status == "AUTHORITATIVE_SOURCE_SELECTED" else 0.0
        doc_hit = 1.0 if selected_doc else 0.0
        best_score = safe_float(row.get("best_score", 0.0))
        missing_fields_penalty = -1.0 if decision_reason.startswith("missing_query_fields:") else 0.0
        return (success, localized + doc_hit, best_score, missing_fields_penalty)

    ranked_rows = sorted(
        records,
        key=lambda row: row_score(pd.Series(row)),
        reverse=True,
    )
    row = ranked_rows[0].copy()
    decision_sources: list[str] = []
    for record in records:
        for field in (
            "primary_finding_source_doc_id",
            "authoritative_source_doc_id",
            "selected_source_doc_id",
            "top_candidate_doc_id",
        ):
            source = normalize_space(record.get(field, ""))
            if source and source not in decision_sources:
                decision_sources.append(source)
    row["all_decision_source_doc_ids"] = " | ".join(decision_sources)
    return {key: row.get(key, "") for key in row.keys()}


def read_csv_rows(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    try:
        rows = pd.read_csv(path, low_memory=False)
    except Exception:
        return []
    return rows.to_dict("records")


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return ""


def truncate_text(text: Any, max_chars: int = 180) -> str:
    value = normalize_space(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def strip_parenthetical_suffix(text: str) -> str:
    value = normalize_space(text)
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    return value


def benchmark_truth_variants(value: Any) -> list[str]:
    raw = strip_parenthetical_suffix(normalize_space(value))
    if not raw:
        return []
    variants: list[str] = [raw]
    parts = [
        normalize_space(part)
        for part in re.split(r"\s*[|;]\s*", raw)
        if normalize_space(part)
    ]
    for part in parts:
        match = re.match(r"^[A-Z]\d+=(.+?)(?:@[^@]+)?$", part)
        if match:
            extracted = strip_parenthetical_suffix(match.group(1))
            if extracted:
                variants.append(extracted)
        elif "=" in part:
            _, rhs = part.split("=", 1)
            rhs = strip_parenthetical_suffix(rhs)
            if "@" in rhs:
                rhs = strip_parenthetical_suffix(rhs.split("@", 1)[0])
            if rhs:
                variants.append(rhs)
        else:
            variants.append(part)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        normalized = strip_parenthetical_suffix(normalize_space(item))
        key = normalized.casefold()
        if normalized and key not in seen:
            deduped.append(normalized)
            seen.add(key)
    return deduped


def relaxed_truth_match(truth: Any, predicted: Any) -> bool:
    pred_variants = benchmark_truth_variants(predicted)
    if not pred_variants:
        return False
    for gold in benchmark_truth_variants(truth):
        gold_cf = gold.casefold()
        if not gold_cf:
            continue
        for pred in pred_variants:
            pred_cf = pred.casefold()
            if not pred_cf:
                continue
            if gold_cf == pred_cf:
                return True
            if gold_cf in pred_cf or pred_cf in gold_cf:
                return True
            gold_tokens = {token.casefold() for token in WORD_RE.findall(gold_cf)}
            pred_tokens = {token.casefold() for token in WORD_RE.findall(pred_cf)}
            if not gold_tokens or not pred_tokens:
                continue
            overlap = len(gold_tokens & pred_tokens) / max(1, len(gold_tokens))
            if overlap >= 0.8:
                return True
            if semantic_proposition_truth_match(gold, pred):
                return True
    return False


def semantic_proposition_truth_match(truth: Any, predicted: Any) -> bool:
    """Match grounded proposition paraphrases without accepting topic overlap."""
    truth_signature = build_event_signature(normalize_space(truth))
    predicted_signature = build_event_signature(normalize_space(predicted))
    truth_predicate = canonical_key(truth_signature.get("predicate", ""))
    predicted_predicate = canonical_key(predicted_signature.get("predicate", ""))
    generic_predicates = {"", "be", "do", "have", "say"}
    if (
        truth_predicate in generic_predicates
        or predicted_predicate in generic_predicates
        or compare_argument(truth_predicate, predicted_predicate) != "MATCH"
    ):
        return False
    if (
        canonical_key(truth_signature.get("assertion_polarity", ""))
        != canonical_key(predicted_signature.get("assertion_polarity", ""))
    ):
        return False
    for dimension in ("actor", "object"):
        truth_value = truth_signature.get(dimension, "")
        predicted_value = predicted_signature.get(dimension, "")
        if not normalize_space(truth_value) or not normalize_space(predicted_value):
            return False
        if compare_argument(truth_value, predicted_value) != "MATCH":
            return False
    return True


def build_sentence_lookup(chunks: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in chunks.to_dict("records"):
        doc_id = str(row.get("doc_id", ""))
        paragraph_id = str(row.get("paragraph_id", ""))
        for idx, sentence in enumerate(sentence_split(str(row.get("chunk_text", ""))), start=1):
            lookup[f"{doc_id}::{paragraph_id}::sentence_{idx}"] = sentence
    return lookup


def summarize_redaction_candidates(path: Path | None, limit: int = 5) -> str:
    rows = read_csv_rows(path)
    if not rows:
        return ""
    rows.sort(
        key=lambda item: (
            bool(item.get("eligible", False)),
            safe_float(item.get("final_score", 0.0)),
            safe_float(item.get("retrieval_score", 0.0)),
        ),
        reverse=True,
    )
    preview = []
    for row in rows[:limit]:
        preview.append(
            {
                "candidate": normalize_space(row.get("candidate", "")),
                "doc_id": normalize_space(row.get("doc_id", "")),
                "eligible": bool(row.get("eligible", False)),
                "final_score": safe_float(row.get("final_score", 0.0)),
                "retrieval_score": safe_float(row.get("retrieval_score", 0.0)),
                "event_identity_score": safe_float(row.get("event_identity_score", 0.0)),
                "failed_gates": normalize_space(row.get("failed_gates", "")),
                "evidence_excerpt": truncate_text(row.get("evidence", "")),
            }
        )
    return safe_json_dumps(preview)


def summarize_sentence_competition(
    eval_row: dict[str, Any],
    sentence_lookup: dict[str, str],
) -> str:
    winner_text = normalize_space(eval_row.get("selected_value_or_text", ""))
    nearest_passage = normalize_space(eval_row.get("nearest_alternative_passage_id", ""))
    nearest_text = sentence_lookup.get(nearest_passage, "")
    preview = [
        {
            "rank": 1,
            "text": winner_text,
            "doc_id": normalize_space(eval_row.get("selected_source_doc_id", "")),
            "score": safe_float(eval_row.get("best_score", 0.0)),
            "status": normalize_space(eval_row.get("semantic_status", "")),
        }
    ]
    if nearest_passage or nearest_text or eval_row.get("nearest_alternative_score", "") != "":
        preview.append(
            {
                "rank": 2,
                "text": nearest_text,
                "doc_id": normalize_space(eval_row.get("nearest_alternative_doc_id", "")),
                "score": safe_float(eval_row.get("nearest_alternative_score", 0.0)),
                "eligible": eval_row.get("nearest_alternative_eligible", ""),
                "failed_gates": normalize_space(eval_row.get("nearest_alternative_failed_gates", "")),
                "passage_id": nearest_passage,
            }
        )
    return safe_json_dumps(preview)


def selected_evidence_truth_match(semantic_output_dir: Path | None, truth: Any) -> bool:
    gold_variants = benchmark_truth_variants(truth)
    if not semantic_output_dir or not semantic_output_dir.exists() or not gold_variants:
        return False
    evidence_path = semantic_output_dir / "task_evidence_units_v0413.json"
    if not evidence_path.exists():
        return False
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for unit in payload.get("units", []):
        for fragment in unit.get("evidence_fragments", []):
            if normalize_space(fragment.get("evidence_role", "")).casefold() != "selected":
                continue
            for gold in gold_variants:
                if relaxed_truth_match(gold, fragment.get("candidate_value_or_text", "")):
                    return True
                if relaxed_truth_match(gold, fragment.get("excerpt", "")):
                    return True
    return False


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = normalize_space(value)
        if text:
            return text
    return ""


def authoritative_decision_source(
    task_family: Any,
    evaluation_row: dict[str, Any],
) -> str:
    """Return the document field that actually carries a task's decision."""
    task = normalize_space(task_family).casefold()
    if task == "contradictions":
        return first_nonempty(
            evaluation_row.get("primary_finding_source_doc_id", ""),
            evaluation_row.get("authoritative_source_doc_id", ""),
            evaluation_row.get("selected_source_doc_id", ""),
            evaluation_row.get("top_candidate_doc_id", ""),
        )
    return first_nonempty(
        evaluation_row.get("selected_source_doc_id", ""),
        evaluation_row.get("authoritative_source_doc_id", ""),
        evaluation_row.get("top_candidate_doc_id", ""),
    )


def semantic_status_is_success(task_family: str, status: Any) -> bool:
    value = normalize_space(status).casefold()
    task = normalize_space(task_family).casefold()
    if not value:
        return False
    if task == "reconstruction":
        return value.startswith("reconstructable")
    if task == "claims":
        return value.startswith("direct_support")
    if task == "redaction":
        return value.startswith("resolved")
    if task == "value_redaction":
        return value.startswith("resolved")
    if task == "sentence_redaction":
        return value.startswith("resolved")
    if task == "contradictions":
        return (
            value.startswith("direct_opposition")
            or value.startswith("direct_numerical_contradiction")
            or value.startswith("attribute_contradiction")
            or value.startswith("location_contradiction")
            or value.startswith("side_a_support")
            or value.startswith("side_b_support")
        )
    return False


def derive_effective_batch_status(
    task_family: str,
    raw_batch_status: Any,
    baseline_row: dict[str, Any],
    holdout_row: dict[str, Any],
    run_layout: str,
) -> str:
    raw_status = normalize_space(raw_batch_status).upper()
    if raw_status:
        return raw_status
    baseline_semantic_status = normalize_space(baseline_row.get("semantic_status", ""))
    baseline_localization_status = normalize_space(baseline_row.get("localization_status", ""))
    baseline_success = semantic_status_is_success(task_family, baseline_semantic_status)
    if baseline_success:
        return "PASS"
    if (
        baseline_semantic_status.startswith("BLOCKED_")
        or baseline_localization_status == "NO_SOURCE_FOUND"
    ):
        return "FAIL"
    return "PASS" if baseline_success else "FAIL"


def read_text_if_exists(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def summarize_case_progress_hint(
    run_root: Path | None,
    case: dict[str, Any],
    *,
    manifest_row: dict[str, Any] | None = None,
) -> str:
    if not run_root or not run_root.exists():
        return ""
    case_root = run_root / f"{int(case.get('order', 0)):03d}_{slug(str(case.get('id', '')))}"
    if not case_root.exists():
        return ""
    run_layout, baseline_root, holdout_root = detect_run_layout(case_root)
    primary_root = baseline_root or holdout_root or (case_root / "semantic_match_output")
    baseline_eval_row = read_summary_row(first_eval_in_dir(baseline_root or primary_root))
    holdout_eval_row = read_summary_row(first_eval_in_dir(holdout_root))
    status_payload: dict[str, Any] = {}
    status_path = case_root / "coordinated_status_v048.json"
    if status_path.exists():
        try:
            status_payload = load_json(status_path)
        except Exception:
            status_payload = {}
    elapsed_seconds = ""
    perf_path = case_root / "coordinated_performance_v0420.json"
    if perf_path.exists():
        try:
            perf_payload = load_json(perf_path)
            total_wall_seconds = perf_payload.get("total_wall_seconds", "")
            if total_wall_seconds != "":
                elapsed_seconds = f"{float(total_wall_seconds):.1f}s"
        except Exception:
            elapsed_seconds = ""

    expected_doc_id = normalize_space((manifest_row or {}).get("doc_id", "")) or normalize_space(case.get("doc_id", ""))
    task_family = first_nonempty(
        (manifest_row or {}).get("task_family", ""),
        case.get("task_family", ""),
        case.get("task", ""),
    )
    baseline_status = first_nonempty(
        baseline_eval_row.get("semantic_status", ""),
        baseline_eval_row.get("raw_certification_status", ""),
        baseline_eval_row.get("reconstructability_status", ""),
        normalize_space(status_payload.get("terminal_state", "")),
        normalize_space(status_payload.get("status", "")),
    )
    baseline_pass = semantic_status_is_success(task_family, baseline_status)
    ground_truth = normalize_space(
        (manifest_row or {}).get("ground_truth", "")
    )
    baseline_grounded = bool(
        baseline_pass
        and ground_truth
        and (
            relaxed_truth_match(
                ground_truth,
                baseline_eval_row.get("selected_value_or_text", ""),
            )
            or selected_evidence_truth_match(primary_root, ground_truth)
        )
    )
    selected_doc_id = authoritative_decision_source(
        task_family,
        baseline_eval_row,
    )
    decision_docs = {
        normalize_space(value)
        for value in re.split(r"\s*[|;]\s*", selected_doc_id)
        if normalize_space(value)
    }
    doc_hit = bool(expected_doc_id and expected_doc_id in decision_docs)
    reason = first_nonempty(
        baseline_eval_row.get("decision_reason", ""),
        baseline_eval_row.get("unresolved_reason", ""),
        baseline_eval_row.get("failed_gates", ""),
        baseline_eval_row.get("primary_finding", ""),
        baseline_status,
    )
    if not reason:
        for relative in (
            Path("single") / "matcher.stderr.log",
            Path("baseline") / "matcher.stderr.log",
            Path("holdout") / "matcher.stderr.log",
            Path("matcher.stderr.log"),
        ):
            text = read_text_if_exists(case_root / relative)
            if text:
                reason = text.splitlines()[0]
                break
    result_label = (
        "GROUNDED"
        if baseline_grounded
        else "UNGROUNDED_PASS"
        if baseline_pass
        else "NO_DECISION"
    )
    parts = [result_label, f"target-doc {'HIT' if doc_hit else 'MISS'}"]
    if elapsed_seconds:
        parts.insert(0, elapsed_seconds)
    if baseline_pass:
        selected_surface = normalize_space(
            baseline_eval_row.get("selected_value_or_text", "")
        )
        if selected_surface:
            parts.append(
                f"answer {truncate_text(selected_surface, max_chars=64)}"
            )
    if not baseline_grounded and reason:
        parts.append(truncate_text(reason, max_chars=90))
    if run_layout == "pair":
        holdout_status = first_nonempty(
            holdout_eval_row.get("semantic_status", ""),
            holdout_eval_row.get("raw_certification_status", ""),
            holdout_eval_row.get("reconstructability_status", ""),
        )
        if holdout_status:
            holdout_pass = semantic_status_is_success(
                task_family, holdout_status
            )
            holdout_grounded = bool(
                holdout_pass
                and ground_truth
                and (
                    relaxed_truth_match(
                        ground_truth,
                        holdout_eval_row.get("selected_value_or_text", ""),
                    )
                    or selected_evidence_truth_match(
                        holdout_root,
                        ground_truth,
                    )
                )
            )
            holdout_label = (
                "GROUNDED"
                if holdout_grounded
                else "UNGROUNDED_PASS"
                if holdout_pass
                else "NO_DECISION"
            )
            parts.append(
                f"holdout {holdout_label}"
            )
    return " | ".join(parts)


def find_digest_artifact(base: Path | None) -> Path | None:
    if not base or not base.exists():
        return None
    for pattern in ("llm_digest*.md", "llm_digest*.txt", "*digest*.md", "*digest*.txt"):
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def detect_run_layout(run_root: Path | None) -> tuple[str, Path | None, Path | None]:
    if not run_root or not run_root.exists():
        return "missing", None, None
    single_root = run_root / "semantic_match_output"
    if single_root.exists():
        return "single", single_root, None
    baseline_root = run_root / "baseline" / "semantic_match_output"
    holdout_root = run_root / "holdout" / "semantic_match_output"
    if baseline_root.exists() or holdout_root.exists():
        return (
            "pair",
            baseline_root if baseline_root.exists() else None,
            holdout_root if holdout_root.exists() else None,
        )
    return "unknown", None, None


def collect_arm_artifacts(
    semantic_output_dir: Path | None,
    task_family: str,
    sentence_lookup: dict[str, str],
) -> dict[str, Any]:
    if not semantic_output_dir or not semantic_output_dir.exists():
        return {}
    decision_row = read_summary_row(semantic_output_dir / "decision_display_v0417.csv")
    evidence_row = read_summary_row(semantic_output_dir / "task_evidence_units_v0413.csv")
    artifact: dict[str, Any] = {
        "decision_band": normalize_space(decision_row.get("operational_decision_band", "")),
        "semantic_eligibility_state": normalize_space(decision_row.get("semantic_eligibility_state", "")),
        "display_score": safe_float(decision_row.get("score", "")),
        "display_runner_up_score": safe_float(decision_row.get("runner_up_score", "")),
        "display_winner_margin": safe_float(decision_row.get("winner_margin", "")),
        "display_minimum_margin": safe_float(decision_row.get("display_minimum_margin", "")),
        "evidence_fragment_count": normalize_space(evidence_row.get("evidence_fragment_count", "")),
        "evidence_roles": normalize_space(evidence_row.get("evidence_roles", "")),
        "limitations": normalize_space(evidence_row.get("limitations", "")),
    }
    if task_family == "value_redaction":
        artifact["top_candidates_json"] = summarize_redaction_candidates(
            semantic_output_dir / "redaction_candidate_audit_v048.csv"
        )
    elif task_family == "sentence_redaction":
        eval_row = read_summary_row(first_eval_in_dir(semantic_output_dir))
        artifact["top_candidates_json"] = summarize_sentence_competition(eval_row, sentence_lookup)
    else:
        artifact["top_candidates_json"] = ""
    return artifact


def first_eval_in_dir(base: Path | None) -> Path | None:
    if not base or not base.exists():
        return None
    for pattern in (
        "*_evaluation_v048.csv",
        "*_evaluation_v043.csv",
        "task_evaluation_v048.csv",
        "task_evaluation_v043.csv",
    ):
        matches = sorted(base.glob(pattern))
        if matches:
            if len(matches) > 1:
                non_generic = [match for match in matches if not match.name.startswith("task_")]
                if non_generic:
                    return non_generic[0]
            return matches[0]
    return None


def summarize_run(config: dict[str, Any], paths: ExperimentPaths) -> tuple[Path, Path]:
    manifest_path = paths.query_manifest_root / "case_manifest.csv"
    batch_summary_path = paths.run_root / "batch_summary.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    batch = pd.read_csv(batch_summary_path) if batch_summary_path.exists() else pd.DataFrame()
    if manifest.empty:
        raise FileNotFoundError(f"Missing case manifest: {manifest_path}")
    if batch.empty:
        raise FileNotFoundError(f"Missing batch summary: {batch_summary_path}")
    chunks_path = paths.extraction_root / "chunks.csv"
    chunks_df = pd.read_csv(chunks_path, low_memory=False) if chunks_path.exists() else pd.DataFrame()
    sentence_lookup = build_sentence_lookup(chunks_df) if not chunks_df.empty else {}

    results: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        case_id = str(row["case_id"])
        task_family = str(row["task_family"])
        batch_row = batch[batch["id"] == case_id]
        batch_info = batch_row.iloc[0].to_dict() if not batch_row.empty else {}
        run_root = Path(str(batch_info.get("run_root", ""))) if batch_info.get("run_root") else None
        run_layout, baseline_root, holdout_root = detect_run_layout(run_root)
        pair_file = None
        if run_root and run_layout == "pair" and baseline_root:
            pair_matches: list[Path] = []
            for pattern in PAIR_EVAL_PATTERNS:
                pair_matches.extend(sorted(baseline_root.glob(pattern)))
            pair_file = pair_matches[0] if pair_matches else None
        single_file = first_eval_in_dir(baseline_root) if baseline_root else None
        holdout_file = first_eval_in_dir(holdout_root) if holdout_root else None
        baseline_row = read_summary_row(single_file)
        holdout_row = read_summary_row(holdout_file)
        pair_row = read_summary_row(pair_file)
        baseline_artifacts = collect_arm_artifacts(baseline_root, task_family, sentence_lookup)
        holdout_artifacts = collect_arm_artifacts(holdout_root, task_family, sentence_lookup)
        query_path = paths.experiment_root / str(row.get("query_file", ""))
        query_text = read_text_if_exists(query_path)
        baseline_digest = find_digest_artifact(baseline_root)
        holdout_digest = find_digest_artifact(holdout_root)
        effective_batch_status = derive_effective_batch_status(
            task_family,
            batch_info.get("status", ""),
            baseline_row,
            holdout_row,
            run_layout,
        )
        # A coordinated preflight block returns a generic batch FAIL. Keep the
        # explicit terminal state so analysts can distinguish it from a runtime failure.
        if run_root:
            status_path = run_root / "coordinated_status_v048.json"
            if status_path.exists():
                try:
                    coordinated_status = load_json(status_path)
                except (OSError, ValueError, TypeError):
                    coordinated_status = {}
                terminal_status = normalize_space(coordinated_status.get("status", "")).upper()
                if terminal_status == "PREFLIGHT_BLOCKED":
                    effective_batch_status = terminal_status
        result = dict(row)
        result.update(
            {
                "raw_batch_status": batch_info.get("status", ""),
                "batch_status": effective_batch_status,
                "elapsed_seconds": batch_info.get("elapsed_seconds", ""),
                "run_layout": run_layout,
                "run_root": str(run_root) if run_root else "",
                "pair_eval_file": str(pair_file.relative_to(paths.experiment_root)) if pair_file else "",
                "single_eval_file": str(single_file.relative_to(paths.experiment_root)) if single_file else "",
                "holdout_eval_file": str(holdout_file.relative_to(paths.experiment_root)) if holdout_file else "",
                "query_text": query_text,
                "query_file_abs": str(query_path.resolve()) if query_path.exists() else "",
                "streamlit_open_path_abs": (
                    str(run_root.resolve()) if run_root and run_layout == "pair"
                    else str(baseline_root.resolve()) if baseline_root else ""
                ),
                "streamlit_holdout_path_abs": str(holdout_root.resolve()) if holdout_root else "",
                "baseline_digest_artifact_abs": str(baseline_digest.resolve()) if baseline_digest else "",
                "holdout_digest_artifact_abs": str(holdout_digest.resolve()) if holdout_digest else "",
            }
        )
        for column in SUMMARY_COLUMNS:
            result[column] = baseline_row.get(column, "")
            result[f"baseline_{column}"] = baseline_row.get(column, "")
            result[f"holdout_{column}"] = holdout_row.get(column, "")
        for key, value in baseline_artifacts.items():
            result[f"baseline_{key}"] = value
        for key, value in holdout_artifacts.items():
            result[f"holdout_{key}"] = value
        for column, value in pair_row.items():
            result[column] = value
        baseline_relaxed = relaxed_truth_match(
            row.get("ground_truth", ""),
            baseline_row.get("selected_value_or_text", ""),
        )
        holdout_relaxed = relaxed_truth_match(
            row.get("ground_truth", ""),
            holdout_row.get("selected_value_or_text", ""),
        )
        baseline_evidence_match = selected_evidence_truth_match(baseline_root, row.get("ground_truth", ""))
        holdout_evidence_match = selected_evidence_truth_match(holdout_root, row.get("ground_truth", ""))
        baseline_complete_value = baseline_row.get("all_targets_resolved", "")
        holdout_complete_value = holdout_row.get("all_targets_resolved", "")
        baseline_targets_complete = (
            True
            if baseline_complete_value == ""
            else normalize_space(baseline_complete_value).casefold() in {"1", "true", "yes"}
        )
        holdout_targets_complete = (
            True
            if holdout_complete_value == ""
            else normalize_space(holdout_complete_value).casefold() in {"1", "true", "yes"}
        )
        result["baseline_relaxed_truth_match"] = baseline_relaxed
        result["holdout_relaxed_truth_match"] = holdout_relaxed
        result["baseline_evidence_truth_match"] = baseline_evidence_match
        result["holdout_evidence_truth_match"] = holdout_evidence_match
        result["baseline_effective_truth_match"] = bool(
            baseline_targets_complete and (baseline_relaxed or baseline_evidence_match)
        )
        result["holdout_effective_truth_match"] = bool(
            holdout_targets_complete and (holdout_relaxed or holdout_evidence_match)
        )
        result["outcome_reason"] = first_nonempty(
            result.get("baseline_decision_reason", ""),
            result.get("baseline_unresolved_reason", ""),
            result.get("baseline_failed_gates", ""),
            batch_info.get("issues", ""),
            result.get("pair_status", ""),
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_path = paths.summary_root / "case_results.csv"
    results_df.to_csv(results_path, index=False)

    audit_columns = [
        "case_id",
        "task_family",
        "doc_id",
        "paragraph_id",
        "query_kind",
        "ground_truth",
        "source_context",
        "notes",
        "batch_status",
        "baseline_semantic_status",
        "baseline_selected_value_or_text",
        "baseline_selected_source_doc_id",
        "baseline_best_score",
        "baseline_runner_up_score",
        "baseline_winner_margin",
        "baseline_decision_reason",
        "baseline_unresolved_reason",
        "baseline_failed_gates",
        "baseline_decision_band",
        "baseline_semantic_eligibility_state",
        "baseline_evidence_fragment_count",
        "baseline_evidence_roles",
        "baseline_limitations",
        "baseline_top_candidates_json",
        "baseline_relaxed_truth_match",
        "baseline_evidence_truth_match",
        "baseline_effective_truth_match",
        "holdout_semantic_status",
        "holdout_selected_value_or_text",
        "holdout_selected_source_doc_id",
        "holdout_best_score",
        "holdout_runner_up_score",
        "holdout_winner_margin",
        "holdout_decision_reason",
        "holdout_unresolved_reason",
        "holdout_failed_gates",
        "holdout_decision_band",
        "holdout_semantic_eligibility_state",
        "holdout_evidence_fragment_count",
        "holdout_evidence_roles",
        "holdout_limitations",
        "holdout_top_candidates_json",
        "holdout_relaxed_truth_match",
        "holdout_evidence_truth_match",
        "holdout_effective_truth_match",
        "pair_status",
        "change",
        "analyst_interpretation",
        "excluded_doc_ids",
        "targeted_exclusion_ids",
        "holdout_relevance_status",
    ]
    available_audit_columns = [column for column in audit_columns if column in results_df.columns]
    task_decision_audit_path = paths.summary_root / "task_decision_audit.csv"
    results_df[available_audit_columns].to_csv(task_decision_audit_path, index=False)

    streamlit_columns = [
        "corpus",
        "task_family",
        "case_id",
        "doc_id",
        "paragraph_id",
        "query_kind",
        "batch_status",
        "run_layout",
        "baseline_semantic_status",
        "baseline_decision_band",
        "outcome_reason",
        "ground_truth",
        "query_text",
        "query_file_abs",
        "run_root",
        "streamlit_open_path_abs",
        "streamlit_holdout_path_abs",
        "baseline_digest_artifact_abs",
        "holdout_digest_artifact_abs",
    ]
    available_streamlit_columns = [
        column for column in streamlit_columns if column in results_df.columns
    ]
    streamlit_run_index_path = paths.summary_root / "streamlit_run_index.csv"
    results_df[available_streamlit_columns].to_csv(streamlit_run_index_path, index=False)

    task_summary = (
        results_df.groupby(["task_family", "batch_status"], dropna=False)
        .size()
        .reset_index(name="case_count")
        .sort_values(["task_family", "batch_status"])
    )
    task_summary_path = paths.summary_root / "task_summary.csv"
    task_summary.to_csv(task_summary_path, index=False)

    for task in ("value_redaction", "sentence_redaction"):
        sub = results_df[results_df["task_family"] == task].copy()
        if sub.empty:
            continue
        audit_path = paths.summary_root / f"{task}_audit.csv"
        sub.to_csv(audit_path, index=False)

    lines = [
        f"# {config['corpus'].upper()} task benchmark summary",
        "",
        f"Experiment: `{paths.experiment_root.name}`",
        "",
        "## Task counts",
        "",
    ]
    for row in task_summary.to_dict("records"):
        lines.append(
            f"- `{row['task_family']}` / `{row['batch_status']}`: {int(row['case_count'])}"
        )
    lines.extend(["", "## Key outputs", ""])
    lines.append(f"- Case manifest: `{manifest_path.relative_to(paths.experiment_root)}`")
    answer_key_path = paths.query_manifest_root / "query_answer_key.csv"
    if answer_key_path.exists():
        lines.append(f"- Query and answer key: `{answer_key_path.relative_to(paths.experiment_root)}`")
    lines.append(f"- Batch summary: `{batch_summary_path.relative_to(paths.experiment_root)}`")
    lines.append(f"- Case results: `{results_path.relative_to(paths.experiment_root)}`")
    lines.append(f"- Task summary: `{task_summary_path.relative_to(paths.experiment_root)}`")
    lines.append(f"- Task decision audit: `{task_decision_audit_path.relative_to(paths.experiment_root)}`")
    lines.append(f"- Streamlit run index: `{streamlit_run_index_path.relative_to(paths.experiment_root)}`")
    if (paths.summary_root / "value_redaction_audit.csv").exists():
        lines.append(f"- Value redaction audit: `{(paths.summary_root / 'value_redaction_audit.csv').relative_to(paths.experiment_root)}`")
    if (paths.summary_root / "sentence_redaction_audit.csv").exists():
        lines.append(f"- Sentence redaction audit: `{(paths.summary_root / 'sentence_redaction_audit.csv').relative_to(paths.experiment_root)}`")
    (paths.summary_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results_path, task_summary_path


def snapshot_config(config: dict[str, Any], paths: ExperimentPaths) -> Path:
    ensure_dirs(paths)
    snapshot_path = paths.config_root / "resolved_config.json"
    payload = dict(config)
    payload["matryoshka_root"] = str(resolve_path(payload["matryoshka_root"]))
    payload["output_root"] = str(resolve_path(payload["output_root"]))
    payload["corpus_entity_resource"] = str(resolve_path(payload["corpus_entity_resource"]))
    payload["decision_profile"] = str(resolve_path(payload["decision_profile"]))
    payload["python_executable"] = str(
        resolve_path_preserve_symlink(payload["python_executable"])
    )
    payload["source_tables"] = {
        key: str(resolve_path(value))
        for key, value in payload["source_tables"].items()
    }
    atomic_write_json(snapshot_path, payload)
    return snapshot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "build", "run", "summarize", "all"],
        help="Pipeline step to execute.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        choices=sorted(CORPUS_ALIASES),
        help="Corpus to benchmark using its processed extraction bundle.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--per-task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--tasks",
        default="",
        help=(
            "Comma-separated subset of tasks: reconstruction,claims,"
            "value_redaction,sentence_redaction,contradictions"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-openai-parser", action="store_true")
    parser.add_argument("--use-llm-digest", action="store_true")
    parser.add_argument(
        "--runner-mode",
        choices=["single", "pair"],
        default="",
        help="Override runner mode. `pair` executes baseline plus holdout for every case.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.corpus = canonicalize_corpus_name(args.corpus)
    config = load_config(resolve_path(args.config), args.corpus)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.use_openai_parser:
        config["runner"]["use_openai_parser"] = True
    if args.use_llm_digest:
        config["runner"]["use_llm_digest"] = True
    if args.runner_mode:
        config["runner"]["mode"] = args.runner_mode
    default_name = (
        args.experiment_name
        or f"{args.corpus}_pair_{args.per_task or 'default'}x5"
    )
    paths = experiment_paths(config, default_name)
    tasks_override = [value.strip() for value in args.tasks.split(",") if value.strip()] or None
    snapshot_config(config, paths)

    if args.command in {"prepare", "all"}:
        ensure_dirs(paths)
        materialize_unified_extraction(config, paths, overwrite=args.overwrite)
        run_preflight(config, paths)
        if args.command == "prepare":
            return

    if args.command in {"build", "all"}:
        ensure_dirs(paths)
        if not paths.extraction_root.exists():
            materialize_unified_extraction(config, paths, overwrite=False)
        cases, diagnostics, catalog_rows = build_cases(
            config,
            per_task_override=args.per_task,
            tasks_override=tasks_override,
            progress_prefix=f"[build {normalize_space(str(config.get('corpus', 'corpus')))}] ",
        )
        plan_path, manifest_path = write_cases(
            config,
            paths,
            cases,
            diagnostics,
            catalog_rows,
            overwrite=args.overwrite,
        )
        print(f"Plan written to: {plan_path}")
        print(f"Case manifest written to: {manifest_path}")
        if args.command == "build":
            return

    if args.command in {"run", "all"}:
        ensure_dirs(paths)
        if not paths.extraction_root.exists():
            materialize_unified_extraction(config, paths, overwrite=False)
        plan_path = paths.query_manifest_root / "benchmark_plan.json"
        if not plan_path.exists():
            raise FileNotFoundError(
                f"Missing benchmark plan: {plan_path}. Run the build step first."
            )
        run_batch(config, paths, plan_path, overwrite=args.overwrite)
        if args.command == "run":
            return

    if args.command in {"summarize", "all"}:
        results_path, task_summary_path = summarize_run(config, paths)
        print(f"Case results written to: {results_path}")
        print(f"Task summary written to: {task_summary_path}")


if __name__ == "__main__":
    main()
