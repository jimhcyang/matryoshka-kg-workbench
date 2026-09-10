#!/usr/bin/env python3
"""Shared semantic contracts for Matryoshka Requirements v1.10 Final.

This module is intentionally dependency-light. Parser v0.4.14, Matcher v0.4.20,
and Workbench v3.14 import the same normalization and schema helpers so the
components do not silently assign different meanings to the same query,
number, date, event, or empty output table.

The module implements shared Requirements v1.9 foundations rather than task
ranking policy.  In particular:

* punctuation-only sentence fragments are never treated as semantic context;
* number words are parsed inside named semantic fields (packets, kg, grams);
* missing and mismatching arguments are distinct states;
* event identity is represented explicitly and remains corpus-agnostic;
* CSV files always contain declared headers, even when they contain zero rows;
* cache writes are atomic and fingerprinted.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

COMMON_VERSION = "v1.10"
NORMALIZATION_VERSION = "matryoshka_v0410_semantics_v12"

PREDICATE_PHRASE_LEMMAS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:is|are|was|were|be|been|being)\s+directed\s+to\b|"
            r"\b(?:is|are|was|were|be|been|being)\s+under\s+orders?\s+to\b",
            re.IGNORECASE,
        ),
        "order",
    ),
    (
        re.compile(
            r"\b(?:hold|holds|held|holding)\s+(?:in|at|by)\b",
            re.IGNORECASE,
        ),
        "detain",
    ),
    (
        re.compile(
            r"\b(?:has|have|had)\s+no\s+memory\s+of\b|"
            r"\b(?:do|does|did)\s+not\s+remember\b",
            re.IGNORECASE,
        ),
        "remember",
    ),
    (
        re.compile(
            r"\b(?:is|are|was|were|be|been|being)\s+(?:an?\s+)?agent\s+of\b|"
            r"\b(?:is|are|was|were|be|been|being)\s+affiliated\s+with\b|"
            r"\b(?:affiliate|affiliates|affiliated|affiliating|associate|"
            r"associates|associated|associating)\s+with\b",
            re.IGNORECASE,
        ),
        "affiliation",
    ),
    (
        re.compile(
            r"\b(?:is|are|was|were|be|been|being|has|have|had)\s+"
            r"(?:(?:covertly|secretly|actively)\s+)?engaged\s+in\s+"
            r"(?:build|builds|building|built|construct|constructs|constructed|constructing|construction)\b",
            re.IGNORECASE,
        ),
        "build",
    ),
    (
        re.compile(
            r"\b(?:is|are|was|were|be|been|being)\s+"
            r"(?:(?:covertly|secretly|actively)\s+)?busy\s+with\s+"
            r"(?:the\s+)?construction\b",
            re.IGNORECASE,
        ),
        "build",
    ),
    (
        re.compile(
            r"\b(?:is|are|was|were|be|been|being)\s+(?:very\s+)?close\s+to\s+join(?:ing)?\b|"
            r"\b(?:is|are|was|were|be|been|being)\s+on\s+the\s+verge\s+of\s+enter(?:ing)?\b",
            re.IGNORECASE,
        ),
        "enter",
    ),
    (
        re.compile(
            r"\b(?:align|aligns|aligned|aligning)\s+with\b|"
            r"\b(?:is|are|was|were|be|been|being)\s+consistent\s+with\b|"
            r"\b(?:correspond|corresponds|corresponded|corresponding)\s+"
            r"(?:to|with)\b|"
            r"\b(?:agree|agrees|agreed|agreeing)\s+with\b|"
            r"\b(?:is|are|was|were|be|been|being)\s+"
            r"(?:(?:very|pretty|reasonably|closely)\s+){0,2}close\s+to\b(?!\s+\w+ing\b)",
            re.IGNORECASE,
        ),
        "correspond",
    ),
    (
        re.compile(
            r"\bcall(?:s|ed|ing)?\s+for\b",
            re.IGNORECASE,
        ),
        "demand",
    ),
    (
        re.compile(
            r"\bcall(?:s|ed|ing)?\s+on\b",
            re.IGNORECASE,
        ),
        "request",
    ),
    (
        re.compile(
            r"\bstep(?:s|ped|ping)?\s+up\b",
            re.IGNORECASE,
        ),
        "increase",
    ),
    (
        re.compile(
            r"\b(?:is|are|was|were|be|been|being)\s+"
            r"(?:an?\s+)?(?:example|instance)\s+of\b",
            re.IGNORECASE,
        ),
        "exemplify",
    ),
    (
        re.compile(
            r"\bget\s+(?:(?:the|a)\s+)?(?:country|nation|organization)?\s*"
            r"back\s+to\s+work\b",
            re.IGNORECASE,
        ),
        "resume",
    ),
)

PREDICATE_LEMMAS: Dict[str, str] = {
    "be": "be", "is": "be", "are": "be", "was": "be", "were": "be",
    "been": "be", "being": "be",
    "remain": "be", "remains": "be", "remained": "be",
    "remaining": "be",
    "disregard": "ignore", "disregards": "ignore",
    "disregarded": "ignore", "disregarding": "ignore",
    "forget": "ignore", "forgets": "ignore", "forgot": "ignore",
    "forgotten": "ignore", "forgetting": "ignore",
    "ignore": "ignore", "ignores": "ignore", "ignored": "ignore",
    "ignoring": "ignore",
    "seize": "seize", "seized": "seize", "seizes": "seize",
    "seizure": "seize", "confiscate": "seize", "confiscated": "seize",
    "confiscates": "seize", "confiscating": "seize",
    "confiscation": "seize",
    "offer": "offer", "offers": "offer", "offered": "offer", "offering": "offer",
    "propose": "propose", "proposes": "propose", "proposed": "propose", "proposing": "propose",
    "proposal": "propose", "proposals": "propose",
    "urge": "urge", "urges": "urge", "urged": "urge", "urging": "urge",
    "aim": "aim", "aims": "aim", "aimed": "aim", "aiming": "aim",
    "buy": "buy", "buys": "buy", "bought": "buy", "buying": "buy",
    "seek": "seek", "seeks": "seek", "sought": "seek", "seeking": "seek",
    "search": "seek", "searches": "seek", "searched": "seek", "searching": "seek",
    "explore": "seek", "explores": "seek", "explored": "seek", "exploring": "seek",
    "plan": "plan", "plans": "plan", "planned": "plan", "planning": "plan",
    "intend": "intend", "intends": "intend", "intended": "intend", "intending": "intend",
    "press": "press", "presses": "press", "pressed": "press", "pressing": "press",
    "demand": "demand", "demands": "demand", "demanded": "demand",
    "demanding": "demand",
    "receive": "receive", "receives": "receive", "received": "receive", "receiving": "receive",
    "support": "support", "supports": "support", "supported": "support", "supporting": "support",
    "back": "support", "backs": "support", "backed": "support",
    "backing": "support",
    "use": "use", "uses": "use", "used": "use", "using": "use",
    "cite": "use", "cites": "use", "cited": "use", "citing": "use",
    "value": "value", "values": "value", "valued": "value", "valuing": "value",
    "prioritize": "prioritize", "prioritizes": "prioritize", "prioritized": "prioritize", "prioritizing": "prioritize",
    "affirm": "affirm", "affirms": "affirm", "affirmed": "affirm", "affirming": "affirm",
    "reaffirm": "affirm", "reaffirms": "affirm", "reaffirmed": "affirm",
    "reaffirming": "affirm",
    "reiterate": "affirm", "reiterates": "affirm", "reiterated": "affirm",
    "reiterating": "affirm",
    "mislead": "mislead", "misleads": "mislead", "misled": "mislead", "misleading": "mislead",
    "concern": "concern", "concerns": "concern", "concerned": "concern", "concerning": "concern",
    "enter": "enter", "enters": "enter", "entered": "enter", "entering": "enter",
    "build": "build", "builds": "build", "building": "build", "built": "build",
    "construct": "build", "constructs": "build", "constructed": "build",
    "constructing": "build", "construction": "build",
    "develop": "develop", "develops": "develop", "developed": "develop", "developing": "develop",
    "allow": "allow", "allows": "allow", "allowed": "allow", "allowing": "allow",
    "permit": "allow", "permits": "allow", "permitted": "allow",
    "permitting": "allow",
    "require": "require", "requires": "require", "required": "require", "requiring": "require",
    "oblige": "require", "obliges": "require", "obliged": "require", "obliging": "require",
    "obligate": "require", "obligates": "require", "obligated": "require",
    "obligating": "require", "obligation": "require", "obligations": "require",
    "eat": "eat", "eats": "eat", "ate": "eat", "eaten": "eat", "eating": "eat",
    "dine": "eat", "dines": "eat", "dined": "eat", "dining": "eat",
    "maintain": "maintain", "maintains": "maintain", "maintained": "maintain", "maintaining": "maintain",
    "spend": "spend", "spends": "spend", "spent": "spend", "spending": "spend",
    "expend": "spend", "expends": "spend", "expended": "spend", "expending": "spend",
    "delay": "delay", "delays": "delay", "delayed": "delay", "delaying": "delay",
    "wait": "wait", "waits": "wait", "waited": "wait", "waiting": "wait",
    "occupy": "occupy", "occupies": "occupy", "occupied": "occupy", "occupying": "occupy",
    "resume": "resume", "resumes": "resume", "resumed": "resume", "resuming": "resume",
    "restart": "resume", "restarts": "resume", "restarted": "resume", "restarting": "resume",
    "apply": "apply", "applies": "apply", "applied": "apply", "applying": "apply",
    "exert": "apply", "exerts": "apply", "exerted": "apply", "exerting": "apply",
    "exemplify": "exemplify", "exemplifies": "exemplify",
    "exemplified": "exemplify", "exemplifying": "exemplify",
    "occur": "occur", "occurs": "occur", "occurred": "occur", "occurring": "occur",
    "happen": "occur", "happens": "occur", "happened": "occur", "happening": "occur",
    "want": "want", "wants": "want", "wanted": "want", "wanting": "want",
    "deny": "deny", "denies": "deny", "denied": "deny", "denying": "deny",
    "admit": "admit", "admits": "admit", "admitted": "admit", "admitting": "admit",
    "limit": "limit", "limits": "limit", "limited": "limit", "limiting": "limit",
    "replace": "replace", "replaces": "replace", "replaced": "replace", "replacing": "replace",
    "call": "call", "calls": "call", "called": "call", "calling": "call",
    "lead": "lead", "leads": "lead", "led": "lead", "leading": "lead",
    "lend": "support", "lends": "support", "lent": "support", "lending": "support",
    "conduct": "conduct", "conducts": "conduct", "conducted": "conduct", "conducting": "conduct",
    "engage": "conduct", "engages": "conduct", "engaged": "conduct", "engaging": "conduct",
    "attack": "attack", "attacks": "attack", "attacked": "attack", "attacking": "attack",
    "exhibit": "exhibit", "exhibits": "exhibit", "exhibited": "exhibit",
    "exhibiting": "exhibit",
    "project": "project", "projects": "project", "projected": "project",
    "projecting": "project", "projection": "project", "projections": "project",
    "forecast": "project", "forecasts": "project", "forecasted": "project",
    "forecasting": "project",
    "strengthen": "strengthen", "strengthens": "strengthen",
    "strengthened": "strengthen", "strengthening": "strengthen",
    "improve": "strengthen", "improves": "strengthen",
    "improved": "strengthen", "improving": "strengthen",
    "promote": "promote", "promotes": "promote", "promoted": "promote",
    "promoting": "promote", "advance": "promote", "advances": "promote",
    "advanced": "promote", "advancing": "promote",
    "rise": "promote", "rises": "promote", "rose": "promote",
    "risen": "promote", "rising": "promote",
    "comply": "comply", "complies": "comply", "complied": "comply",
    "complying": "comply", "satisfy": "satisfy", "satisfies": "satisfy",
    "satisfied": "satisfy", "satisfying": "satisfy",
    "fulfill": "satisfy", "fulfills": "satisfy", "fulfilled": "satisfy",
    "fulfilling": "satisfy", "fulfil": "satisfy", "fulfils": "satisfy",
    "meet": "meet", "meets": "meet", "met": "meet",
    "view": "view", "views": "view", "viewed": "view", "viewing": "view",
    "regard": "view", "regards": "view", "regarded": "view",
    "consider": "view", "considers": "view",
    "considered": "view", "considering": "view",
    "enjoy": "have", "enjoys": "have", "enjoyed": "have",
    "enjoying": "have", "possess": "have", "possesses": "have",
    "possessed": "have", "possessing": "have",
    "exploit": "exploit", "exploits": "exploit", "exploited": "exploit",
    "exploiting": "exploit", "exploitation": "exploit",
    "leave": "leave", "leaves": "leave", "left": "leave", "leaving": "leave",
    "depart": "leave", "departs": "leave", "departed": "leave",
    "departing": "leave",
    "withdraw": "withdraw", "withdraws": "withdraw", "withdrew": "withdraw",
    "withdrawing": "withdraw", "withdrawal": "withdraw",
    "departure": "leave", "departures": "leave",
    "appoint": "appoint", "appoints": "appoint", "appointed": "appoint",
    "appointing": "appoint", "appointment": "appoint",
    "appointments": "appoint", "nominate": "appoint",
    "nominates": "appoint", "nominated": "appoint",
    "nominating": "appoint", "nomination": "appoint",
    "nominations": "appoint",
    "prepare": "prepare", "prepares": "prepare", "prepared": "prepare",
    "preparing": "prepare", "ready": "prepare",
    "order": "order", "orders": "order", "ordered": "order",
    "ordering": "order", "direct": "order", "directs": "order",
    "directed": "order", "directing": "order",
    "refuse": "refuse", "refuses": "refuse", "refused": "refuse",
    "refusing": "refuse",
    "agree": "agree", "agrees": "agree", "agreed": "agree",
    "agreeing": "agree",
    "provide": "give", "provides": "give", "provided": "give",
    "providing": "give", "supply": "give", "supplies": "give",
    "supplied": "give", "supplying": "give",
    "speak": "discuss", "speaks": "discuss", "spoke": "discuss",
    "spoken": "discuss", "speaking": "discuss",
    "removal": "withdraw", "remove": "withdraw", "removes": "withdraw",
    "removed": "withdraw", "removing": "withdraw",
    "relocate": "relocate", "relocates": "relocate", "relocated": "relocate", "relocating": "relocate",
    "move": "relocate", "moves": "relocate", "moved": "relocate", "moving": "relocate",
    "claim": "claim", "claims": "claim", "claimed": "claim", "claiming": "claim",
    "assert": "assert", "asserts": "assert", "asserted": "assert", "asserting": "assert",
    "allege": "allege", "alleges": "allege", "alleged": "allege", "alleging": "allege",
    "argue": "argue", "argues": "argue", "argued": "argue", "arguing": "argue",
    "charge": "charge", "charges": "charge", "charged": "charge", "charging": "charge",
    "accuse": "accuse", "accuses": "accuse", "accused": "accuse", "accusing": "accuse",
    "accusation": "accuse", "accusations": "accuse",
    "blame": "accuse", "blames": "accuse", "blamed": "accuse", "blaming": "accuse",
    "cause": "cause", "causes": "cause", "caused": "cause", "causing": "cause",
    "give": "give", "gives": "give", "gave": "give", "giving": "give",
    "bear": "bear", "bears": "bear", "bearing": "bear", "bore": "bear",
    "resolve": "resolve", "resolves": "resolve", "resolved": "resolve", "resolving": "resolve",
    "discuss": "discuss", "discusses": "discuss", "discussed": "discuss", "discussing": "discuss",
    "sign": "sign", "signs": "sign", "signed": "sign", "signing": "sign",
    "wind": "wind", "winds": "wind", "winding": "wind",
    "injure": "injure", "injures": "injure", "injured": "injure",
    "injuring": "injure", "hurt": "injure", "hurts": "injure",
    "hurting": "injure", "wound": "injure", "wounds": "injure",
    "wounded": "injure", "wounding": "injure",
    "have": "have", "has": "have", "had": "have",
    "lack": "lack", "lacks": "lack", "lacking": "lack", "lacked": "lack",
    "attend": "attend", "attended": "attend", "attending": "attend",
    "attendance": "attend", "represented": "attend", "represent": "attend",
    "travel": "travel", "travels": "travel", "traveled": "travel",
    "travelled": "travel", "traveling": "travel", "travelling": "travel",
    "come": "travel", "comes": "travel", "came": "travel",
    "coming": "travel", "arrive": "travel", "arrives": "travel",
    "arrived": "travel", "arriving": "travel", "arrival": "travel",
    "transport": "transport", "transported": "transport",
    "transporting": "transport", "transportation": "transport",
    "deliver": "transport", "delivers": "transport",
    "delivered": "transport", "delivering": "transport",
    "delivery": "transport",
    "remember": "remember", "remembers": "remember",
    "remembered": "remember", "remembering": "remember",
    "memory": "remember", "recall": "remember",
    "recalls": "remember", "recalled": "remember",
    "recalling": "remember",
    "reported": "report", "reports": "report",
    "report": "report", "inform": "report", "informs": "report", "informed": "report",
    "say": "report", "says": "report", "said": "report", "saying": "report",
    "tell": "report", "tells": "report", "told": "report", "telling": "report",
    "state": "report", "states": "report", "stated": "report", "stating": "report",
    "coordinate": "coordinate", "coordinates": "coordinate",
    "coordinated": "coordinate", "coordinating": "coordinate",
    "direct": "direct", "directs": "direct", "directed": "direct",
    "directing": "direct",
    "contradict": "contradict", "contradicts": "contradict",
    "contradicted": "contradict", "contradicting": "contradict",
    "follow": "follow", "follows": "follow", "followed": "follow",
    "following": "follow",
    "participate": "involve", "participates": "involve", "participated": "involve",
    "participating": "involve",
    "join": "involve", "joins": "involve", "joined": "involve",
    "joining": "involve",
    "recognize": "recognize", "recognizes": "recognize",
    "recognized": "recognize", "recognising": "recognize",
    "recognizing": "recognize",
    "realize": "recognize", "realizes": "recognize",
    "realized": "recognize", "realizing": "recognize",
    "understand": "recognize", "understands": "recognize",
    "understood": "recognize", "understanding": "recognize",
    "request": "request", "requests": "request",
    "requested": "request", "requesting": "request",
    "ask": "ask", "asks": "ask", "asked": "ask", "asking": "ask",
    "inquiry": "ask", "inquiries": "ask",
    "inquire": "ask", "inquires": "ask", "inquired": "ask",
    "inquiring": "ask", "enquire": "ask", "enquires": "ask",
    "enquired": "ask", "enquiring": "ask",
    "deploy": "deploy", "deploys": "deploy", "deployed": "deploy",
    "deploying": "deploy", "deployment": "deploy",
    "station": "deploy", "stations": "deploy", "stationed": "deploy",
    "stationing": "deploy",
    "infiltrate": "infiltrate", "infiltrates": "infiltrate",
    "infiltrated": "infiltrate", "infiltrating": "infiltrate",
    "infiltration": "infiltrate",
    "confirm": "confirm", "confirms": "confirm",
    "confirmed": "confirm", "confirming": "confirm",
    "confirmation": "confirm",
    "hope": "hope", "hopes": "hope", "hoped": "hope", "hoping": "hope",
    "start": "start", "starts": "start", "started": "start",
    "starting": "start", "initiate": "start", "initiates": "start",
    "initiated": "start", "initiating": "start", "initiation": "start",
    "solve": "resolve", "solves": "resolve", "solved": "resolve",
    "solving": "resolve",
    "purchase": "buy", "purchases": "buy", "purchased": "buy",
    "purchasing": "buy",
    "increase": "increase", "increases": "increase",
    "increased": "increase", "increasing": "increase",
    "heighten": "increase", "heightens": "increase",
    "heightened": "increase", "heightening": "increase",
    "intensify": "increase", "intensifies": "increase",
    "intensified": "increase", "intensifying": "increase",
    "anticipate": "anticipate", "anticipates": "anticipate",
    "anticipated": "anticipate", "anticipating": "anticipate",
    "expect": "anticipate", "expects": "anticipate",
    "expected": "anticipate", "expecting": "anticipate",
    "condemn": "condemn", "condemns": "condemn",
    "condemned": "condemn", "condemning": "condemn",
    "condemnation": "condemn",
    "sabotage": "sabotage", "sabotages": "sabotage",
    "sabotaged": "sabotage", "sabotaging": "sabotage",
    "sapper": "sabotage", "sappered": "sabotage",
    "reopen": "reopen", "reopens": "reopen",
    "reopened": "reopen", "reopening": "reopen",
    "lift": "lift", "lifts": "lift", "lifted": "lift", "lifting": "lift",
    "arrest": "arrest", "arrested": "arrest", "arrests": "arrest",
    "respect": "respect", "respects": "respect", "respected": "respect",
    "respecting": "respect",
    "involve": "involve", "involves": "involve", "involved": "involve",
    "involving": "involve",
    "authorize": "authorize", "authorizes": "authorize",
    "authorized": "authorize", "authorizing": "authorize",
    "protect": "protect", "protects": "protect", "protected": "protect",
    "protecting": "protect", "shield": "protect", "shields": "protect",
    "shielded": "protect", "shielding": "protect",
    "criticize": "criticize", "criticizes": "criticize",
    "criticized": "criticize", "criticizing": "criticize",
    "criticise": "criticize", "criticises": "criticize",
    "criticised": "criticize", "criticising": "criticize",
    "contact": "contact", "contacts": "contact", "contacted": "contact",
    "contacting": "contact",
    "shell": "shell", "shells": "shell", "shelled": "shell",
    "shelling": "shell", "bombard": "shell", "bombards": "shell",
    "bombarded": "shell", "bombarding": "shell",
    "detain": "detain", "detains": "detain", "detained": "detain",
    "detaining": "detain", "imprison": "imprison",
    "imprisons": "imprison", "imprisoned": "imprison",
    "imprisoning": "imprison",
    "hold": "hold", "holds": "hold", "held": "hold", "holding": "hold",
    "retain": "retain", "retains": "retain", "retained": "retain",
    "retaining": "retain",
    "head": "travel", "heads": "travel", "headed": "travel",
    "heading": "travel",
    "mandate": "require", "mandates": "require", "mandated": "require",
    "mandating": "require",
    "prevent": "prevent", "prevents": "prevent", "prevented": "prevent",
    "preventing": "prevent", "avoid": "avoid", "avoids": "avoid",
    "avoided": "avoid", "avoiding": "avoid",
    "secure": "acquire", "secures": "acquire", "secured": "acquire",
    "securing": "acquire",
    "keep": "maintain", "keeps": "maintain", "kept": "maintain",
    "keeping": "maintain",
    "mention": "mention", "mentions": "mention", "mentioned": "mention",
    "mentioning": "mention",
    "point": "point", "points": "point", "pointed": "point",
    "pointing": "point",
    "implicate": "implicate", "implicates": "implicate",
    "implicated": "implicate", "implicating": "implicate",
    "influence": "influence", "influences": "influence",
    "influenced": "influence", "influencing": "influence",
    "oppose": "oppose", "opposes": "oppose", "opposed": "oppose",
    "opposing": "oppose",
    "extend": "extend", "extends": "extend", "extended": "extend",
    "extending": "extend", "extension": "extend",
    "align": "correspond", "aligns": "correspond",
    "aligned": "correspond", "aligning": "correspond",
    "correspond": "correspond", "corresponds": "correspond",
    "corresponded": "correspond", "corresponding": "correspond",
    "match": "correspond", "matches": "correspond",
    "matched": "correspond", "matching": "correspond",
}

# Only genuine event nominals belong here.  Deriving this inventory from every
# irregular predicate spelling admitted finite forms such as ``are``, ``left``,
# and ``said`` and let nominal parsing preempt ordinary clauses.
NOMINAL_EVENT_FORMS: frozenset[str] = frozenset({
    "attendance",
    "confiscation",
    "construction",
    "delivery",
    "departure",
    "memory",
    "proposal",
    "removal",
    "seizure",
    "transportation",
    "withdrawal",
})

_PREDICATE_NEAR_EQUIVALENCE_GROUPS: Tuple[frozenset[str], ...] = (
    frozenset({"offer", "propose"}),
    frozenset({"seek", "want", "aim", "plan", "intend", "press", "demand"}),
    frozenset({"seek", "search"}),
    frozenset({"support", "affirm"}),
    frozenset({"press", "urge", "demand"}),
    frozenset({"claim", "assert", "allege", "charge", "report"}),
    frozenset({"accuse", "allege", "blame", "charge"}),
    frozenset({"withdraw", "leave"}),
    frozenset({"remember", "recall"}),
    frozenset({"transport", "deliver"}),
    frozenset({"authorize", "direct"}),
    frozenset({"influence", "mislead"}),
    frozenset({"correspond", "agree"}),
    frozenset({"comply", "meet", "satisfy"}),
    frozenset({"view", "consider"}),
    frozenset({"strengthen", "improve"}),
    frozenset({"promote", "advance"}),
    frozenset({"have", "enjoy", "possess"}),
    frozenset({"have", "maintain"}),
    frozenset({"request", "confirm", "seek", "call"}),
    frozenset({"enter", "involve"}),
    frozenset({"be", "have"}),
    frozenset({"protect", "shield"}),
    frozenset({"detain", "imprison"}),
    frozenset({"retain", "hold"}),
    frozenset({"hold", "occupy"}),
    frozenset({"delay", "wait"}),
    frozenset({"resume", "restart"}),
    frozenset({"prevent", "avoid"}),
    frozenset({"ask", "request"}),
    frozenset({"deploy", "station"}),
)

_ARGUMENT_TOKEN_EQUIVALENTS: Dict[str, str] = {
    "above": "threshold",
    "agreement": "agreement",
    "agreements": "agreement",
    "agree": "commitment",
    "agreed": "commitment",
    "commit": "commitment",
    "commits": "commitment",
    "committed": "commitment",
    "committing": "commitment",
    "amicable": "friendly",
    "desirability": "desirable",
    "desirable": "desirable",
    "entitled": "right",
    "entitlement": "right",
    "morally": "moral",
    "discussion": "discussion",
    "discussions": "discussion",
    "deal": "agreement",
    "deals": "agreement",
    "egyptian": "egypt",
    "egyptians": "egypt",
    "russian": "russia",
    "france": "france",
    "french": "france",
    "friendly": "friendly",
    "beginning": "start",
    "beginnings": "start",
    "step": "start",
    "steps": "start",
    "basis": "basis",
    "beyond": "threshold",
    "foundation": "basis",
    "crucial": "important",
    "crucially": "important",
    "critical": "important",
    "critically": "important",
    "conjecture": "uncertainty",
    "global": "world",
    "indian": "india",
    "importance": "important",
    "important": "important",
    "vital": "important",
    "speculation": "uncertainty",
    "prerogative": "privilege",
    "prerogatives": "privilege",
    "privilege": "privilege",
    "privileges": "privilege",
    "vitally": "important",
    "iranian": "iran",
    "iranians": "iran",
    "officials": "official",
    "authorities": "official",
    "prompt": "prompt",
    "promptly": "prompt",
    "speedy": "prompt",
    "speedily": "prompt",
    "covert": "secret",
    "covertly": "secret",
    "secretly": "secret",
    "confirmation": "confirm",
    "confirmed": "confirm",
    "confirming": "confirm",
    "anticipated": "anticipate",
    "expected": "anticipate",
    "del": "delegation",
    "delegations": "delegation",
    "distinct": "separate",
    "own": "separate",
    "pact": "agreement",
    "pacts": "agreement",
    "proposal": "proposal",
    "proposals": "proposal",
    "relation": "relation",
    "relations": "relation",
    "relationship": "relation",
    "relationships": "relation",
    "regional": "region",
    "regions": "region",
    "russia": "russia",
    "russian": "russia",
    "russians": "russia",
    "senior": "senior",
    "top": "senior",
    "adequate": "enough",
    "enough": "enough",
    "sufficient": "enough",
    "monarch": "monarch",
    "monarchies": "monarch",
    "monarchy": "monarch",
    "equitable": "fair",
    "fair": "fair",
    "possibility": "way",
    "possibilities": "way",
    "pursue": "seek",
    "pursues": "seek",
    "pursuing": "seek",
    "produce": "develop",
    "produced": "develop",
    "produces": "develop",
    "producing": "develop",
    "production": "develop",
    "development": "develop",
    "resolution": "solution",
    "resolutions": "solution",
    "saudi": "saudi arabia",
    "saudis": "saudi arabia",
    "seek": "seek",
    "seeking": "seek",
    "settlement": "agreement",
    "settlements": "agreement",
    "solution": "solution",
    "solutions": "solution",
    "talk": "discussion",
    "talks": "discussion",
    "tie": "relation",
    "ties": "relation",
    "way": "way",
    "ways": "way",
    "vital": "important",
    "vitally": "important",
    "worldwide": "world",
    "companies": "company",
    "firms": "company",
    "issue": "problem",
    "issues": "problem",
    "japanese": "japan",
    "problems": "problem",
}

LOCATION_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:city|town|province|district|municipality)\s+of\s+",
    re.IGNORECASE,
)

# Negation is preserved because it is semantically load-bearing for claim and
# contradiction tasks.  These are token filters, not a linguistic ontology.
SEMANTIC_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
PUNCTUATION_ONLY_RE = re.compile(r"^[\s\W_]*$", re.UNICODE)

MONTHS: Dict[str, int] = {
    name.lower(): index
    for index, name in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
MONTH_ALIASES: Dict[str, str] = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "jun": "june", "jul": "july", "aug": "august", "sep": "september",
    "sept": "september", "oct": "october", "nov": "november", "dec": "december",
}

NUMBER_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
}
NUMBER_CONNECTORS = {"and"}

NEGATION_TERMS = {
    "no", "not", "never", "none", "neither", "nor", "without", "denied",
    "deny", "denies", "refused", "rejected", "absent", "absence",
    "lack", "lacks", "lacked", "lacking",
    "exempt", "exempted", "exemption",
}
UNCERTAINTY_PHRASES = {
    "no news", "not known", "unknown", "uncertain", "unclear", "unconfirmed",
    "could not confirm", "cannot confirm", "not confirmed", "reportedly",
    "allegedly", "possibly", "perhaps", "may", "might", "could",
    "conjecture", "speculation", "speculative", "unsubstantiated",
}
APPROXIMATION_MARKERS = {
    "approximately", "approximate", "about", "around", "roughly", "nearly",
    "some",
}

ARGUMENT_STATES = {
    "MATCH", "MISMATCH", "MISSING_IN_QUERY", "MISSING_IN_CANDIDATE",
    "UNPARSEABLE", "NOT_APPLICABLE",
}

# These clause frames are narrow grammatical normalizations, not guesses from
# isolated keywords.  Parser reconciliation may prefer them over a malformed
# model-produced event label because each frame recovers an explicit actor,
# event, and argument structure from the analyst's own sentence.
STRONG_CLAUSE_NORMALIZATION_STATUSES = frozenset({
    "ASSESSMENT_FRAME_NORMALIZED",
    "AWARENESS_NORMALIZED",
    "CONSTRUCTION_ENGAGEMENT_NORMALIZED",
    "COORDINATED_REOPENING_NORMALIZED",
    "EVIDENCE_ABSENCE_NORMALIZED",
    "NORMATIVE_CONDEMNATION_NORMALIZED",
    "NEGATED_EXHIBITION_NORMALIZED",
    "MODAL_STATE_ACTION_NORMALIZED",
    "PASSIVE_ACTION_NORMALIZED",
    "PASSIVE_OBLIGATION_NORMALIZED",
    "POSSESSION_FRAME_NORMALIZED",
    "PROMOTION_ACTIVITY_NORMALIZED",
    "QUANTITATIVE_PROJECTION_NORMALIZED",
    "QUESTION_MEANING_NORMALIZED",
    "REQUEST_CONFIRMATION_NORMALIZED",
    "SABOTAGE_EVENT_NORMALIZED",
    "SECURITY_INTENSIFICATION_NORMALIZED",
})

AUXILIARY_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had",
    "will", "would", "can", "could", "may", "might", "shall", "should", "must",
    "do", "does", "did",
}
GENERIC_PREDICATE_HEADS = {
    "be", "is", "are", "was", "were", "been", "being",
    "offer", "offers", "offered", "offering",
    "propose", "proposes", "proposed", "proposing",
    "urge", "urges", "urged", "urging",
    "buy", "buys", "bought", "buying",
    "seek", "seeks", "sought", "seeking",
    "search", "searches", "searched", "searching",
    "aim", "aims", "aimed", "aiming",
    "plan", "plans", "planned", "planning",
    "intend", "intends", "intended", "intending",
    "press", "presses", "pressed", "pressing",
    "receive", "receives", "received", "receiving",
    "support", "supports", "supported", "supporting",
    "use", "uses", "used", "using",
    "cite", "cites", "cited", "citing",
    "value", "values", "valued", "valuing",
    "prioritize", "prioritizes", "prioritized", "prioritizing",
    "affirm", "affirms", "affirmed", "affirming",
    "mislead", "misleads", "misled", "misleading",
    "concern", "concerns", "concerned", "concerning",
    "enter", "enters", "entered", "entering",
    "build", "builds", "building", "built",
    "develop", "develops", "developed", "developing",
    "allow", "allows", "allowed", "allowing",
    "require", "requires", "required", "requiring",
    "maintain", "maintains", "maintained", "maintaining",
    "strengthen", "strengthens", "strengthened", "strengthening",
    "improve", "improves", "improved", "improving",
    "promote", "promotes", "promoted", "promoting",
    "advance", "advances", "advanced", "advancing",
    "comply", "complies", "complied", "complying",
    "satisfy", "satisfies", "satisfied", "satisfying",
    "fulfill", "fulfills", "fulfilled", "fulfilling",
    "fulfil", "fulfils",
    "meet", "meets", "met",
    "view", "views", "viewed", "viewing",
    "regard", "regards", "regarded",
    "consider", "considers", "considered", "considering",
    "enjoy", "enjoys", "enjoyed", "enjoying",
    "possess", "possesses", "possessed", "possessing",
    "remain", "remains", "remained", "remaining",
    "want", "wants", "wanted", "wanting",
    "give", "gives", "gave", "giving",
    "bear", "bears", "bearing", "bore",
    "resolve", "resolves", "resolved", "resolving",
    "discuss", "discusses", "discussed", "discussing",
    "sign", "signs", "signed", "signing",
    "wind", "winds", "winding", "wound",
    "have", "has", "had",
    "lack", "lacks", "lacking", "lacked",
    "participate", "participates", "participated", "participating",
    "responsible",
}
GENERIC_PREDICATE_PARTICLES = {
    "to", "for", "with", "in", "on", "at", "from", "against", "under",
    "over", "down", "up", "out", "back", "off", "into", "through",
    "responsible", "able",
}
GENERIC_PREDICATE_ADVERBS = {
    "not", "never", "no", "longer", "partly", "probably", "already", "still", "now",
    "actively", "firmly", "openly", "publicly", "privately", "directly",
    "quietly", "repeatedly", "currently", "reportedly", "likely", "further",
}
PREDICATE_FRAME_ADVERB_RE = (
    r"not|never|still|already|currently|directly|strongly|openly|"
    r"publicly|privately|actively|firmly|quietly|repeatedly|"
    r"reportedly|probably|likely|further"
)
BARE_FRAGMENT_PREDICATE_RE = (
    r"den(?:y|ies|ied|ying)|admit(?:ted|s|ting)?|claim(?:ed|s|ing)?|"
    r"assert(?:ed|s|ing)?|alleg(?:ed|es|ing)?|argu(?:ed|es|ing)|"
    r"support(?:s|ed|ing)?|back(?:s|ed|ing)?|oppos(?:e|es|ed|ing)?|"
    r"seek|seeks|sought|seeking|want(?:s|ed|ing)?|intend(?:s|ed|ing)?|"
    r"plan(?:s|ned|ning)?|press(?:es|ed|ing)?|demand(?:s|ed|ing)?|"
    r"build(?:s|ing|t)?|construct(?:s|ed|ing|ion)?|"
    r"develop(?:s|ed|ing|ment)?|withdraw(?:s|ing|al|drew)?|"
    r"leave|leaves|left|leaving|move(?:d|s|ing)?|"
    r"relocat(?:e|ed|es|ing)|conduct(?:ed|s|ing)?|attack(?:ed|s|ing)?"
)
REPORTING_VERB_RE = (
    r"reported|informed|said|told|state(?:d|s|ing)?|"
    r"claim(?:ed|s|ing)?|assert(?:ed|s|ing)?|alleg(?:ed|es|ing)?|"
    r"argu(?:ed|es|ing)|maintain(?:ed|s|ing)|charg(?:ed|es|ing)|"
    r"indicat(?:e|ed|es|ing)|suggest(?:ed|s|ing)|show(?:ed|s|ing)?|"
    r"quot(?:e|ed|es|ing)|not(?:e|ed|es|ing)|warn(?:ed|s|ing)?|"
    r"advis(?:e|ed|es|ing)|estimat(?:e|ed|es|ing)"
)
CLAUSE_ARGUMENT_HEAD_RE = r"(?:\[[^\]]+\]|[A-Za-z][A-Za-z0-9'().&/,\- ]{1,120}?)"


def _reporting_clause_parts(value: Any) -> Optional[Dict[str, str]]:
    """Return a conservative reporter/verb/embedded-clause decomposition.

    ``state`` is both a reporting verb and part of common proper names.  A
    regex-only search previously parsed ``United States did not support X`` as
    reporter ``United`` plus an embedded clause beginning with ``did``.  A
    present-tense ``state(s)`` followed directly by an auxiliary is not a
    reporting construction; it is the tail of the grammatical subject.
    """
    source = clean_text(value)
    if not source:
        return None
    pattern = re.compile(
        rf"\b(?P<verb>{REPORTING_VERB_RE})\b",
        flags=re.I,
    )
    for match in pattern.finditer(source):
        reporter = clean_text(source[:match.start()]).strip(" ,;:")
        tail = clean_text(source[match.end():]).lstrip(" ,;:")
        if not reporter or not tail:
            continue
        verb_surface = clean_text(match.group("verb"))
        if (
            verb_surface.casefold() in {"state", "states"}
            and re.search(r"\bunited\s*$", reporter, flags=re.I)
        ):
            continue
        that_match = re.search(r"\bthat\b", tail, flags=re.I)
        if that_match:
            embedded = clean_text(tail[that_match.end():])
        else:
            embedded = tail
            if (
                verb_surface.casefold() in {"state", "states"}
                and re.match(
                    r"^(?:do|does|did|is|are|was|were|has|have|had|"
                    r"will|would|can|could|should|may|might|must|shall)\b",
                    embedded,
                    flags=re.I,
                )
            ):
                continue
        if embedded:
            return {
                "reporter": reporter,
                "reporting_verb": verb_surface,
                "embedded": embedded,
            }
    return None


def utc_now_iso() -> str:
    """Return a timezone-aware ISO timestamp suitable for manifests."""
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any, max_len: Optional[int] = None) -> str:
    """Normalize whitespace without changing semantic word content."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if max_len is not None and len(text) > max_len:
        return text[: max(0, max_len - 1)].rstrip() + "…"
    return text


def normalize_escaped_layout(value: Any) -> Tuple[str, int]:
    """Repair line breaks that were serialized as literal ``\\n`` text.

    Query plans, CSV cells, and shell-generated files sometimes contain a
    visible backslash followed by ``n`` where the author intended a line
    break.  Paragraph boundaries are semantic input to Matryoshka, so silently
    retaining those characters can merge unrelated events and destroy
    sentence-redaction adjacency.  Only escaped layout characters are
    repaired; ordinary backslashes and semantic text are preserved.

    Returns the repaired text and the number of replacements.
    """
    if value is None:
        return "", 0
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    replacements = text.count("\\r\\n") + text.count("\\n")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return text, replacements


def canonical_key(value: Any) -> str:
    """Return a stable, conservative comparison key."""
    return " ".join(re.findall(r"[a-z0-9]+", clean_text(value).lower()))


def canonical_predicate(value: Any) -> str:
    """Normalize morphology without turning unrelated verbs into synonyms."""
    raw_text = clean_text(value)
    if raw_text:
        for pattern, lemma in PREDICATE_PHRASE_LEMMAS:
            if pattern.search(raw_text):
                return lemma
    key = canonical_key(raw_text).replace("_", " ")
    if not key:
        return ""
    for token in key.split():
        if token in PREDICATE_LEMMAS:
            return PREDICATE_LEMMAS[token]
    return PREDICATE_LEMMAS.get(key, key)


def canonical_location(value: Any) -> str:
    """Normalize appositional administrative prefixes, preserving surface text elsewhere."""
    text = re.sub(r"^(?:at|in|near)\s+(?:the\s+)?", "", clean_text(value), flags=re.I)
    return canonical_key(LOCATION_PREFIX_RE.sub("", text))


def canonical_object_head(value: Any) -> str:
    """Return a conservative, open-vocabulary object phrase.

    The normalization is intentionally symmetric and grammatical.  It does
    not select from a bundled list of corpus objects.
    """
    source = clean_text(value)
    # Quantities and weights describe an open-vocabulary patient/theme; they
    # are not part of its identity. Extract the noun phrase after a generic
    # container-of construction before canonical token cleanup.
    measured = re.search(
        r"\b(?:packets?|packages?|crates?|containers?|units?|items?)"
        r"\s+of\s+(?P<object>[A-Za-z][A-Za-z0-9'’&/\-]*"
        r"(?:\s+[A-Za-z][A-Za-z0-9'’&/\-]*){0,5}?)"
        r"(?=\s+(?:weighing|weighed|weight|at|in|on|during|"
        r"before|after|that|which)\b|[.,;:]|$)",
        source,
        flags=re.I,
    )
    measured_weight = re.search(
        r"\b(?:kilograms?|kilogrammes?|kilos?|kg|grams?|g)\s+of\s+"
        r"(?P<object>[A-Za-z][A-Za-z0-9'’&/\-]*"
        r"(?:\s+[A-Za-z][A-Za-z0-9'’&/\-]*){0,5}?)"
        r"(?=\s+(?:in|on|at|near|during|before|after|that|which)\b"
        r"|[.,;:]|$)",
        source,
        flags=re.I,
    )
    if measured:
        source = measured.group("object")
    elif measured_weight:
        source = measured_weight.group("object")
    else:
        source = re.sub(
            r"\b(?:weighing|weighed|with\s+(?:a\s+)?weight(?:\s+of)?)"
            r"\b.*$",
            "",
            source,
            flags=re.I,
        )
    source = re.sub(
        r"\b(?:but|yet|while|though|although|however|whereas|and|or)\s+"
        r"(?:it|he|she|they|we|i|this|that|these|those|"
        r"is|are|was|were|has|have|had|will|would|could|should|"
        r"may|might|can|did|do|does)\b.*$",
        "",
        source,
        flags=re.I,
    )
    source = re.sub(
        r"^(?:offered|offering|offer|proposed|proposing|propose|proposal)\s+",
        "",
        source,
        flags=re.I,
    )
    text = canonical_key(source)
    text = re.sub(
        r"\b(?:a|an|the|units?|items?|containers?|kilograms?|grams?|kg|"
        r"of|weighing|approximately|about|some)\b",
        " ",
        text,
    )
    text = re.sub(
        r"\b(?:at|in|near|on|during|before|after)\b.*$",
        " ",
        text,
    )
    return clean_text(text)[:160]


_ARGUMENT_COMPARISON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "both",
    "but", "by", "for", "from", "had", "has", "have", "in", "into", "is",
    "it", "its", "of", "on", "or", "see", "that", "the", "their", "there",
    "existence", "these", "they", "this", "those", "to", "view", "was", "were", "while",
    "with", "point", "about", "concerning", "regarding",
    "anybody", "anyone", "somebody", "someone", "most", "only", "first",
    "government", "federation",
}


def _predicate_family_equivalent(query_predicate: str, candidate_predicate: str) -> bool:
    if not query_predicate or not candidate_predicate:
        return False
    if query_predicate == candidate_predicate:
        return True
    return any(
        query_predicate in group and candidate_predicate in group
        for group in _PREDICATE_NEAR_EQUIVALENCE_GROUPS
    )


def _comparison_content_tokens(value: Any) -> Set[str]:
    text = clean_text(value)
    if not text:
        return set()
    text = re.sub(r"\ball\s+possible\b", "maximum", text, flags=re.I)
    text = re.sub(
        r"\bUnited\s+States(?:\s+of\s+America)?\b",
        "unitedstates",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bU\.?\s*S\.?\s*A\.?\b", "unitedstates", text, flags=re.I)
    text = re.sub(r"\bU\.?\s*S\.?\b", "unitedstates", text, flags=re.I)
    text = re.sub(r"\bAmerican\b", "unitedstates", text, flags=re.I)
    text = re.sub(
        r"\b(?:a\s+)?(?:very\s+)?long[-\s]+range\b",
        "distant",
        text,
        flags=re.I,
    )
    objectish = canonical_object_head(text) or text
    tokens: Set[str] = set()
    # Preserve coordinated/list tails that canonical_object_head deliberately
    # trims. They often carry the very anchor that makes a paraphrase
    # equivalent (for example, "region and world").
    for surface in dict.fromkeys((objectish, text)):
        for token in semantic_tokens(surface, preserve_negation=False):
            normalized = PREDICATE_LEMMAS.get(token, token)
            normalized = re.sub(r"(?:'s|’s)$", "", normalized)
            normalized = _ARGUMENT_TOKEN_EQUIVALENTS.get(normalized, normalized)
            if len(normalized) < 2 or normalized in _ARGUMENT_COMPARISON_STOPWORDS:
                continue
            tokens.add(normalized)
    return tokens


def _near_equivalent_argument(query_value: Any, candidate_value: Any) -> bool:
    query_text = clean_text(query_value)
    candidate_text = clean_text(candidate_value)
    if not query_text or not candidate_text:
        return False
    query_key = canonical_key(query_text)
    candidate_key = canonical_key(candidate_text)
    if query_key and candidate_key:
        query_words = query_key.split()
        candidate_words = candidate_key.split()
        shorter = min(len(query_words), len(candidate_words))
        if shorter >= 2 and (
            query_key in candidate_key
            or candidate_key in query_key
        ):
            return True
    query_tokens = _comparison_content_tokens(query_text)
    candidate_tokens = _comparison_content_tokens(candidate_text)
    if query_tokens == candidate_tokens and len(query_tokens) == 1:
        return len(next(iter(query_tokens))) >= 4
    if min(len(query_tokens), len(candidate_tokens)) < 2:
        return False
    shared = query_tokens & candidate_tokens
    if len(shared) < 2:
        return False
    containment = len(shared) / float(min(len(query_tokens), len(candidate_tokens)))
    jaccard = len(shared) / float(len(query_tokens | candidate_tokens))
    if containment >= 1.0:
        return True
    return containment >= 0.75 and (len(shared) >= 3 or jaccard >= 0.55)


def _looks_acronym_surface(value: Any) -> bool:
    surface = clean_text(value)
    if not surface:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", surface)
    if not compact or len(compact) > 8:
        return False
    return bool(re.search(r"[A-Z]", compact) and compact == compact.upper())


def _short_alias_token_match(query_value: Any, candidate_value: Any) -> bool:
    query_text = clean_text(query_value)
    candidate_text = clean_text(candidate_value)
    if not query_text or not candidate_text:
        return False
    q_key = canonical_key(query_text)
    c_key = canonical_key(candidate_text)
    if not q_key or not c_key:
        return False
    for short_text, short_key, long_key in (
        (query_text, q_key, c_key),
        (candidate_text, c_key, q_key),
    ):
        tokens = short_key.split()
        if len(tokens) != 1:
            continue
        token = tokens[0]
        if len(token) < 2 or len(token) > 8:
            continue
        if not _looks_acronym_surface(short_text):
            continue
        long_tokens = [
            item for item in long_key.split()
            if item not in _ARGUMENT_COMPARISON_STOPWORDS
        ]
        if len(long_tokens) >= 2 and token == "".join(
            item[0] for item in long_tokens if item
        ):
            return True
        # An acronym followed only by a generic institutional qualifier still
        # names the same actor (for example "PLO" and "PLO leadership").
        # Do not treat an all-caps personal-name token as an acronym merely
        # because it is contained in a fuller name ("ZIA" != "Zia Nassry").
        generic_qualifiers = {
            "administration", "agency", "authorities", "delegation",
            "forces", "government", "leadership", "military", "mission",
            "movement", "official", "officials", "organization", "party",
            "regime", "representative", "representatives",
        }
        if token in long_tokens:
            remainder = [item for item in long_tokens if item != token]
            if remainder and all(
                item in generic_qualifiers for item in remainder
            ):
                return True
    return False


def semantic_tokens(value: Any, preserve_negation: bool = True) -> Set[str]:
    """Tokenize text for semantic-validity and light-weight overlap checks."""
    tokens = {m.group(0).lower() for m in SEMANTIC_TOKEN_RE.finditer(clean_text(value))}
    if preserve_negation:
        return tokens
    return tokens - NEGATION_TERMS


def has_semantic_tokens(value: Any, minimum: int = 1) -> bool:
    """True only when text contains meaningful alphanumeric tokens.

    A period, comma, em dash, or other punctuation-only fragment is therefore
    invalid sentence context even though it is a non-empty Python string.
    """
    return len(semantic_tokens(value)) >= max(1, int(minimum))


def stable_hash(payload: Any, length: Optional[int] = None) -> str:
    """Hash strings or JSON-serializable objects deterministically."""
    if isinstance(payload, (dict, list, tuple)):
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    else:
        text = str(payload)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def sha256_file(path: Path) -> str:
    """Return an empty string for a missing file and SHA-256 otherwise."""
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def meaningful_sentence_parts(text: Any) -> List[str]:
    """Split text into sentence-like parts and remove punctuation-only pieces.

    The regular expression intentionally does not attempt full linguistic
    sentence segmentation.  Its contract is narrower: obtain stable local
    context around an analyst marker and never return a bare punctuation token.
    """
    # Preserve real line boundaries until after splitting.  Calling
    # ``clean_text`` first would collapse a newline-only sentence boundary.
    source, _repairs = normalize_escaped_layout(text)
    source = source.strip()
    if not source:
        return []
    parts = re.split(r"(?<=[.!?])\s*|[\r\n]+", source)
    output: List[str] = []
    for part in parts:
        item = clean_text(part)
        if not item or PUNCTUATION_ONLY_RE.fullmatch(item):
            continue
        if has_semantic_tokens(item):
            output.append(item)
    return output


def nearest_meaningful_sentence(text: Any, side: str) -> str:
    """Return nearest meaningful sentence from a left or right text region."""
    parts = meaningful_sentence_parts(text)
    if parts:
        return parts[-1] if side == "before" else parts[0]
    source = clean_text(text)
    if not has_semantic_tokens(source):
        return ""
    return source[-500:] if side == "before" else source[:500]


def normalize_sentence_slot_context(full_text: str, start: int, end: int) -> Dict[str, Any]:
    """Canonicalize the visible text around a sentence/span-redaction marker.

    The returned hash is based on normalized meaningful neighboring sentences,
    so clean-line, space-adjacent, and punctuation-adjacent marker forms become
    comparable controlled fixtures.
    """
    before_region = str(full_text)[:start]
    after_region = str(full_text)[end:]
    original_before = clean_text(before_region[-500:])
    original_after = clean_text(after_region[:500])
    normalized_before = nearest_meaningful_sentence(before_region, "before")
    normalized_after = nearest_meaningful_sentence(after_region, "after")
    repair_reasons: List[str] = []
    if clean_text(original_before) != clean_text(normalized_before):
        repair_reasons.append("normalized_left_boundary")
    if clean_text(original_after) != clean_text(normalized_after):
        repair_reasons.append("normalized_right_boundary")
    if not has_semantic_tokens(clean_text(original_before).split(" ")[-1] if original_before else "") and normalized_before:
        repair_reasons.append("ignored_left_punctuation_only_fragment")
    if original_after and not has_semantic_tokens(original_after[:4]) and normalized_after:
        repair_reasons.append("ignored_right_punctuation_only_fragment")
    payload = {
        "normalized_sentence_before": normalized_before,
        "normalized_sentence_after": normalized_after,
    }
    return {
        "original_sentence_before": original_before,
        "original_sentence_after": original_after,
        "normalized_sentence_before": normalized_before,
        "normalized_sentence_after": normalized_after,
        "left_context_semantic_token_count": len(semantic_tokens(normalized_before)),
        "right_context_semantic_token_count": len(semantic_tokens(normalized_after)),
        "boundary_repair_applied": bool(repair_reasons),
        "boundary_repair_reason": ";".join(sorted(set(repair_reasons))),
        "context_mode_detected": (
            "two_sided" if normalized_before and normalized_after
            else "left_only" if normalized_before
            else "right_only" if normalized_after
            else "missing_both"
        ),
        "normalized_slot_context_sha256": stable_hash(payload),
    }


def parse_number_phrase(value: Any) -> Optional[float]:
    """Parse one number phrase without consuming neighboring semantic fields.

    Examples: ``sixteen`` -> 16, ``nineteen`` -> 19,
    ``four hundred`` -> 400, and ``1,250`` -> 1250.
    """
    text = clean_text(value).lower().replace(",", " ").replace("-", " ")
    if not text:
        return None
    numeric = re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", text)
    if numeric:
        return float(text)
    tokens = [t for t in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text) if t not in NUMBER_CONNECTORS]
    if not tokens:
        return None
    total = 0.0
    current = 0.0
    found = False
    for token in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            current += float(token)
            found = True
            continue
        if token not in NUMBER_WORDS:
            continue
        found = True
        amount = NUMBER_WORDS[token]
        if amount == 100:
            current = max(1.0, current) * 100.0
        elif amount in {1000, 1_000_000}:
            total += max(1.0, current) * float(amount)
            current = 0.0
        else:
            current += float(amount)
    return total + current if found else None


_NUMBER_PHRASE_PATTERN = (
    r"(?:\d+(?:[.,]\d+)?|"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|and)"
    r"(?:[\s-]+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|and))*)"
)


def normalize_date(value: Any) -> Dict[str, Any]:
    """Normalize common partial/full dates while preserving precision."""
    text = clean_text(value)
    if not text:
        return {"normalized": "", "precision": "missing", "surface": ""}
    month_names = "|".join(sorted(set(MONTHS) | set(MONTH_ALIASES), key=len, reverse=True))
    match = re.search(rf"\b({month_names})\.?\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b", text, flags=re.I)
    if match:
        month_key = match.group(1).lower()
        month_key = MONTH_ALIASES.get(month_key, month_key)
        month = MONTHS[month_key]
        day = int(match.group(2))
        year = match.group(3) or ""
        normalized = f"{year + '-' if year else ''}{month:02d}-{day:02d}"
        return {"normalized": normalized, "precision": "day", "surface": match.group(0)}
    match = re.search(
        rf"\b(\d{{1,2}})\s+({month_names})\.?(?:,?\s+(\d{{4}}))?\b",
        text,
        flags=re.I,
    )
    if match:
        day = int(match.group(1))
        month_key = match.group(2).lower()
        month_key = MONTH_ALIASES.get(month_key, month_key)
        month = MONTHS[month_key]
        year = match.group(3) or ""
        normalized = f"{year + '-' if year else ''}{month:02d}-{day:02d}"
        return {
            "normalized": normalized,
            "precision": "day",
            "surface": match.group(0),
        }
    match = re.search(r"\b(?:(\d{4})[-/])?(\d{1,2})[-/](\d{1,2})\b", text)
    if match:
        year = match.group(1) or ""
        month, day = int(match.group(2)), int(match.group(3))
        normalized = f"{year + '-' if year else ''}{month:02d}-{day:02d}"
        return {"normalized": normalized, "precision": "day", "surface": match.group(0)}
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if match:
        return {"normalized": match.group(1), "precision": "year", "surface": match.group(0)}
    return {"normalized": "", "precision": "unparseable", "surface": ""}


def normalize_numeric_arguments(text: Any) -> Dict[str, Any]:
    """Extract semantic numeric fields without cross-field arithmetic.

    A count such as ``sixteen packets`` is stored as ``packet_count`` and never
    enters weight arithmetic.  Kilograms and grams are combined only because
    they are two units of the *same* weight field.
    """
    source = clean_text(text).lower()
    fields: Dict[str, Any] = {
        "packet_count": None,
        "weight_kg": None,
        "weight_g": None,
        "total_weight_grams": None,
        "date_value": "",
        "date_precision": "missing",
        "date_surface": "",
        "numeric_field_audit_json": "[]",
    }
    audit: List[Dict[str, Any]] = []

    unit_specs = [
        ("packet_count", r"packets?|packages?"),
        ("weight_kg", r"kilograms?|kilos?|kg"),
        ("weight_g", r"grams?|g"),
    ]
    for field, unit_pattern in unit_specs:
        # The bounded phrase prevents the old bug in which all preceding number
        # words (including packet count) were accumulated into a weight value.
        pattern = re.compile(rf"\b(?P<number>{_NUMBER_PHRASE_PATTERN})\s+(?P<unit>{unit_pattern})\b", re.I)
        matches = list(pattern.finditer(source))
        if not matches:
            continue
        # Use the first explicitly typed value for each field.  Additional
        # occurrences remain in the audit for analyst review.
        for index, match in enumerate(matches):
            parsed = parse_number_phrase(match.group("number"))
            audit.append({
                "field": field,
                "surface": match.group(0),
                "number_surface": match.group("number"),
                "parsed_value": parsed,
                "selected": index == 0,
            })
            if index == 0 and parsed is not None:
                fields[field] = float(parsed)

    kg = fields["weight_kg"]
    grams = fields["weight_g"]
    if kg is not None or grams is not None:
        fields["total_weight_grams"] = float(kg or 0.0) * 1000.0 + float(grams or 0.0)

    date = normalize_date(source)
    fields["date_value"] = date["normalized"]
    fields["date_precision"] = date["precision"]
    fields["date_surface"] = date["surface"]
    fields["numeric_field_audit_json"] = json.dumps(audit, ensure_ascii=False, sort_keys=True)
    return fields


def infer_assertion_polarity(text: Any) -> str:
    """Return positive/negative/uncertain while preserving epistemic nuance."""
    source = clean_text(text).lower()
    if any(phrase in source for phrase in UNCERTAINTY_PHRASES):
        return "uncertain"
    # Comparative bounds are affirmative quantity constraints, not event
    # negation: "no fewer than seven" means "at least seven". Remove only
    # the scoped comparative marker before inspecting proposition polarity.
    polarity_source = re.sub(
        r"\b(?:no|not)\s+(?:fewer|less|more|greater)\s+than\b",
        " ",
        source,
    )
    tokens = semantic_tokens(polarity_source)
    return "negative" if tokens & NEGATION_TERMS else "positive"


def infer_epistemic_state(text: Any) -> str:
    source = clean_text(text).lower()
    return "uncertain_or_insufficient_information" if any(p in source for p in UNCERTAINTY_PHRASES) else "asserted"


def infer_numeric_approximation_fields(text: Any) -> List[str]:
    """Return typed numeric fields modified by a nearby approximation cue.

    Approximation is scoped to the numeric argument that it modifies. It is
    not a document-level epistemic state: ``approximately 19 kilograms`` is an
    asserted quantity with bounded precision, while ``officials could not
    confirm the seizure`` is genuine event-level uncertainty.
    """
    source = clean_text(text).lower()
    if not source:
        return []
    marker = r"(?:approximately|approximate|about|around|roughly|nearly|some)"
    number = (
        r"(?:\d+(?:[.,]\d+)?|zero|one|two|three|four|five|six|seven|"
        r"eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
        r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
        r"fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
    )
    pattern = re.compile(
        rf"\b{marker}\b(?:\s+[a-z-]+){{0,3}}\s+{number}"
        r"(?:\s+(?:and|-)\s+" + number + r")*\s+"
        r"(?P<unit>packets?|packages?|kilograms?|kilogrammes?|kg|grams?|g)\b",
        re.I,
    )
    fields: Set[str] = set()
    for match in pattern.finditer(source):
        unit = match.group("unit").lower()
        if unit.startswith(("packet", "package")):
            fields.add("packet_count")
        elif unit == "kg" or unit.startswith(("kilogram", "kilogramme")):
            fields.update({"weight_kg", "total_weight_grams"})
        else:
            fields.update({"weight_g", "total_weight_grams"})
    return sorted(fields)


def resolve_epistemic_scope(
    text: Any,
    declared_state: Any = "",
) -> Dict[str, Any]:
    """Resolve event uncertainty without promoting numeric approximation.

    Parser or extraction signatures may conservatively declare a proposition
    uncertain. When the only identifiable cue is attached to a typed numeric
    field, Matcher v0.4.20 records the approximation in that field and treats
    the event assertion as asserted. Genuine lexical uncertainty always
    remains event-level uncertainty.
    """
    lexical_state = infer_epistemic_state(text)
    declared = canonical_key(declared_state).replace(" ", "_")
    approximation_fields = infer_numeric_approximation_fields(text)
    uncertain_declared = declared in {
        "uncertain", "uncertainty",
        "uncertain_or_insufficient_information",
        "insufficient_information",
    }
    if lexical_state == "uncertain_or_insufficient_information":
        state = lexical_state
        reason = "lexical_event_uncertainty_cue"
    elif uncertain_declared and approximation_fields:
        state = "asserted"
        reason = "numeric_approximation_scoped_to_typed_fields"
    elif uncertain_declared:
        state = "uncertain_or_insufficient_information"
        reason = "declared_event_uncertainty"
    else:
        state = "asserted"
        reason = "asserted_no_event_uncertainty"
    return {
        "epistemic_state": state,
        "epistemic_scope_reason": reason,
        "numeric_approximation_fields": approximation_fields,
    }


def infer_vehicle_identity(text: Any) -> str:
    """Extract a conservative transported-vehicle attribute.

    This is intentionally a generic surface extractor, not a catalogue of
    corpus answers.  It recognizes a vehicle introduced by a transport/use
    relation and stops before licence, registration, punctuation, or a new
    clause.  An absent or ambiguous attribute remains empty.
    """
    source = clean_text(text)
    if not source:
        return ""
    patterns = (
        r"\b(?:transport(?:ed|ing)?|travel(?:led|ed|ing)?|drove|driven|used)"
        r"[^.;]{0,90}?\b(?:in|by|using)\s+(?:an?\s+|the\s+)?"
        r"(?P<vehicle>[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,4})",
        r"\b(?:vehicle|car|truck|automobile)\s+(?:was\s+|is\s+)?"
        r"(?P<vehicle>[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,4})",
    )
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.I)
        if not match:
            continue
        value = re.split(
            r"\b(?:bearing|licen[cs]e|registration|number|with|which|that)\b|[,;.]",
            clean_text(match.group("vehicle")),
            maxsplit=1,
            flags=re.I,
        )[0]
        value = re.sub(
            r"\s+(?:and|on|at|to|from|near|when|where)$", "", value,
            flags=re.I,
        )
        if value and canonical_key(value) not in {"vehicle", "car", "truck"}:
            return clean_text(value)
        if value:
            return clean_text(value)
    return ""


def infer_attendance_state(text: Any) -> str:
    """Normalize generic participation state without corpus vocabulary."""
    source = clean_text(text).lower()
    absent_patterns = [
        r"\b(?:would|will|did|does|do|is|are|was|were)\s+not\s+"
        r"(?:attend|participate|join|appear)\b",
        r"\b(?:is|are|was|were|be|been|being)\s+not\s+"
        r"(?:represented|involved\s+in)\b",
        r"\bnot\s+(?:attending|participating|joining|appearing)\b",
        r"\b(?:had|has|have|played|took)\s+no\s+part\s+in\b",
        r"\bden(?:y|ies|ied|ying)\s+(?:any\s+)?involvement\s+in\b",
        r"\bwithout\s+(?:attending|participating|joining|being\s+involved)\b",
        r"\b(?:no|without)\s+(?:[a-z0-9'’&/\-]+\s+){0,6}"
        r"(?:delegation|representative|participant|attendee)s?\b",
        r"\babsence\b",
    ]
    present_patterns = [
        r"\b(?:will|would|did|does|do|is|are|was|were)\s+"
        r"(?:attend|participate|appear)\b",
        r"\b(?:is|are|was|were|be|been|being)\s+(?:represented|involved\s+in)\b",
        r"\b(?:attending|participating|appearing)\b",
        r"\b(?:joined|participated|attended|took\s+part)\s+in\b",
        r"\bjoin(?:s|ed|ing)?\s+(?:the\s+)?(?:meeting|conference|session|talks|"
        r"negotiations|delegation|committee|summit|event)\b",
    ]
    if any(re.search(pattern, source) for pattern in absent_patterns):
        return "absent"
    if any(re.search(pattern, source) for pattern in present_patterns):
        return "present"
    return "unknown"


def infer_disclosure_state(text: Any) -> str:
    """Normalize explicit public-versus-secret status assertions."""
    source = clean_text(text).lower()
    public_patterns = (
        r"\b(?:is|was|became|become|now)\s+(?:widely\s+)?public(?:ly\s+known)?\b",
        r"\bpublic\s+knowledge\b",
        r"\bno\s+longer\s+(?:a\s+)?secret\b",
        r"\bnot\s+(?:a\s+)?secret\b",
    )
    secret_patterns = (
        r"\b(?:is|was|remains?|remained|kept)\s+(?:a\s+)?secret\b",
        r"\bnot\s+(?:public|publicly\s+known)\b",
    )
    if any(re.search(pattern, source) for pattern in public_patterns):
        return "public"
    if any(re.search(pattern, source) for pattern in secret_patterns):
        return "secret"
    return "unknown"


def infer_evaluative_state(text: Any) -> str:
    """Normalize common positive/negative quality formulations.

    Lexical negation can be internal to an adjective: ``not satisfactory``
    and ``unsatisfactory`` assert the same deficient state even though their
    surface polarity differs. Encoding that state explicitly prevents the
    assertion gate from treating a harmless paraphrase as a contradiction.
    """
    source = clean_text(text).lower()
    dissatisfied_patterns = (
        r"\b(?:dissatisfied|displeased)\b",
        r"\bnot\s+(?:satisfied|pleased|happy)\b",
    )
    positive_patterns = (
        r"\b(?:pleased|happy)\b",
    )
    deficient_patterns = (
        r"\b(?:unsatisfactory|inadequate|insufficient|unacceptable)\b",
        r"\bnot\s+(?:satisfactory|adequate|sufficient|acceptable)\b",
    )
    satisfactory_patterns = (
        r"\b(?:satisfactory|adequate|sufficient|acceptable)\b",
    )
    mistreated_patterns = (
        r"\b(?:ill[-\s]?treated|mistreated|badly\s+treated|"
        r"poorly\s+treated|unfairly\s+treated)\b",
        r"\b(?:offended|aggrieved)(?:\s+and\s+bitter)?\s+"
        r"(?:at|about|by|over)\s+(?:his|her|their|the)?\s*treatment\b",
    )
    if any(re.search(pattern, source) for pattern in dissatisfied_patterns):
        return "dissatisfied"
    if any(re.search(pattern, source) for pattern in deficient_patterns):
        return "deficient"
    if any(re.search(pattern, source) for pattern in mistreated_patterns):
        return "mistreated"
    if any(re.search(pattern, source) for pattern in satisfactory_patterns):
        return "satisfactory"
    if any(re.search(pattern, source) for pattern in positive_patterns):
        return "positive"
    return "unknown"


def _known_value_slot_predicates(
    text: str,
) -> List[Tuple[re.Match[str], str]]:
    """Return explicit action predicates without guessing from nearby nouns."""
    matches: List[Tuple[re.Match[str], str]] = []
    source = clean_text(text)
    for match in SEMANTIC_TOKEN_RE.finditer(source):
        surface = match.group(0).casefold()
        if surface not in PREDICATE_LEMMAS:
            continue
        predicate = PREDICATE_LEMMAS[surface]
        if predicate in {"be", "have"}:
            continue
        if surface == "held" and re.match(
            r"^\s+(?:in|at|by)\b", source[match.end():], flags=re.I
        ):
            predicate = "detain"
        matches.append((match, predicate))
    return matches


def infer_value_slot_frame(before: Any, after: Any, expected_type: Any = "") -> Dict[str, Any]:
    """Infer auditable clause/relation constraints for value redaction.

    The logic is deliberately generic.  It detects grammatical subject slots
    before a governing verb and parenthetical administrative-container slots.
    It does not contain corpus document IDs or expected answers.
    """
    left = clean_text(before)
    right = clean_text(after)
    combined = clean_text(left + " [SLOT] " + right)
    right_predicates = _known_value_slot_predicates(right[:220])
    left_predicates = _known_value_slot_predicates(left[-260:])
    passive_action_match = re.search(
        r"^\s*(?:\([A-Za-z]+\)\s*)?"
        r"(?:(?:has|have|had)\s+been|"
        r"(?:can|could|may|might|must|shall|should|will|would)\s+"
        r"(?:have\s+)?(?:be|been)|is|are|was|were|be|been|being)\s+"
        r"(?P<predicate>"
        r"seized|confiscated|arrested|transported|moved|sent|directed|"
        r"transferred|delivered|appointed|received|represented|attended|"
        r"attacked|exploited"
        r")\b",
        right[:220],
        flags=re.I,
    )
    transport_participle_match = re.search(
        r"\b(?P<predicate>"
        r"taken|moved|sent|transported|directed|transferred|delivered"
        r")\b\s+(?:to|from|into|toward|towards|across|via)\b",
        right[:220],
        flags=re.I,
    )
    # Search through short adjuncts for a known action. Unknown neighboring
    # words are deliberately left unknown rather than promoted to predicates.
    verb_match = right_predicates[0][0] if right_predicates else None
    right_predicate = right_predicates[0][1] if right_predicates else ""

    left_predicate_match: Optional[re.Match[str]] = None
    left_predicate = ""
    left_tail = ""
    if left_predicates:
        candidate_match, candidate_predicate = left_predicates[-1]
        left_window = left[-260:]
        candidate_tail = clean_text(left_window[candidate_match.end():])
        tail_tokens = SEMANTIC_TOKEN_RE.findall(candidate_tail)
        # Periods inside an initialism such as ``U.S. embassy in`` are not
        # clause boundaries. Treating them as sentence punctuation detached
        # location blanks from their governing event.
        boundary_tail = re.sub(
            r"\b(?:[A-Za-z]\.){2,}",
            lambda match: match.group(0).replace(".", ""),
            candidate_tail,
        )
        if (
            len(tail_tokens) <= 10
            and not re.search(r"[.;!?]", boundary_tail)
            and not (
                candidate_predicate == "report" and len(tail_tokens) > 1
            )
        ):
            left_predicate_match = candidate_match
            left_predicate = candidate_predicate
            left_tail = candidate_tail.casefold()

    possessive_slot = bool(re.match(r"^\s*(?:'s|’s)\b", right, flags=re.I))
    infinitive_subject_slot = bool(
        right_predicates
        and re.match(r"^\s*to\s+", right, flags=re.I)
    )
    if passive_action_match:
        governing_predicate = passive_action_match.group("predicate").lower()
    elif transport_participle_match:
        governing_predicate = transport_participle_match.group("predicate").lower()
    elif possessive_slot:
        governing_predicate = ""
    elif infinitive_subject_slot:
        # In ``may allow [ENTITY] to conduct ...``, the missing entity is the
        # actor of the following infinitive.  The earlier control verb describes
        # the enabling event and must not become the slot's hard predicate.
        governing_predicate = right_predicate
    elif left_predicate_match is not None:
        governing_predicate = left_predicate
    else:
        governing_predicate = right_predicate
    grammatical_role = "unknown"
    frame = "generic_value_slot"
    required_relation = ""
    modified_head = ""
    clause_level = "main_or_unknown"
    administrative_container_expected = False
    expected_upper = clean_text(expected_type).upper()

    if possessive_slot:
        grammatical_role = "possessive_modifier"
        frame = "possessive_entity_slot"
    elif left_predicate_match is not None:
        tail = left_tail
        if re.search(
            r"\b(?:to|into|toward|towards|from|in|at)(?:\s+the)?$",
            tail,
        ) or re.search(
            r"\b(?:to|into|toward|towards|from|in|at)\b"
            r"[^.!?]{0,140}(?:,\s*[^.!?]{1,50}){1,8}"
            r"\s*(?:and|or)(?:\s+the)?\s*$",
            tail,
            flags=re.I,
        ):
            grammatical_role = "prepositional_object"
            frame = (
                "travel_destination_slot"
                if canonical_predicate(governing_predicate)
                in {"leave", "relocate", "transport", "travel"}
                else "event_argument_slot"
            )
        elif re.search(r"\bwith(?:\s+the)?$", tail):
            grammatical_role = "prepositional_object"
            frame = "counterparty_slot"
        elif re.search(r"\bby(?:\s+the)?$", tail):
            grammatical_role = "prepositional_object"
            frame = "agentive_by_slot"
    if (
        frame == "generic_value_slot"
        and expected_upper in {"PERSON", "ORGANIZATION", "GOVERNMENT"}
        and re.search(r"\bwith(?:\s+the)?$", left, flags=re.I)
    ):
        grammatical_role = "prepositional_object"
        frame = "counterparty_slot"

    # A value slot before the first finite/action verb in the following clause
    # is a subject candidate even when a short prepositional adjunct intervenes.
    if (
        frame == "generic_value_slot"
        and governing_predicate
        and (verb_match is not None and verb_match.start() <= 180)
    ):
        grammatical_role = "grammatical_subject"
        frame = "organization_as_grammatical_subject" if "ORG" in expected_upper or "ORGAN" in expected_upper else "entity_as_grammatical_subject"

    # A person slot can complete a titled subject rather than replace the
    # entire grammatical subject:
    # ``DEPUTY DIRECTOR FOR ASIA (INDOCHINA) [PERSON] INFORMED ...``.
    # Treat the title/name relationship as an explicit, auditable frame.  This
    # prevents the person candidate from being rejected merely because the KG
    # stores the title and name in different representations.
    title_match = re.search(
        r"(?P<title>(?:(?:DEPUTY|ASSISTANT|ACTING|FOREIGN|PRIME|VICE|"
        r"UNDER|CHIEF|DIRECTOR|MINISTER|AMBASSADOR|SECRETARY|GENERAL|"
        r"COLONEL|GOVERNOR|PRESIDENT)\b[^.!?]{2,140}))$",
        left,
        flags=re.I,
    )
    if (
        "PERSON" in expected_upper
        and title_match
        and re.match(
            r"^\s*(?:informed|reported|said|told|stated|met|received|"
            r"appointed|attended|represented)\b",
            right,
            flags=re.I,
        )
    ):
        modified_head = clean_text(title_match.group("title"), 160)
        grammatical_role = "subject_identity"
        frame = "titled_person_subject_identity"
        required_relation = "identity_appositive_to_title"
    # Parenthetical slot directly following a location head.
    head_match = re.search(r"(?:city\s+of|town\s+of|province\s+of)?\s*([A-Z][A-Za-z0-9'\- ]{1,80})\s*\(\s*$", left)
    if head_match and re.match(r"^[\s)]", right or ")"):
        modified_head = re.sub(
            r"^(?:at|in|near)\s+(?:the\s+)?(?:city|town|province|district|municipality)\s+of\s+",
            "",
            clean_text(head_match.group(1)),
            flags=re.I,
        )
        grammatical_role = "administrative_container"
        frame = "parenthetical_administrative_container"
        required_relation = "administratively_contained_by"
        administrative_container_expected = True
    if re.search(r"\b(informed|reported|said|told)\b[^.]{0,150}\bthat\s*$", left, flags=re.I):
        clause_level = "embedded_clause"
    return {
        "slot_frame": frame,
        "grammatical_role": grammatical_role,
        "governing_predicate": governing_predicate,
        "governing_predicate_surface": governing_predicate,
        "governing_predicate_normalized":
            canonical_predicate(governing_predicate),
        "clause_level": clause_level,
        "required_relation": required_relation,
        "modified_head": modified_head,
        "administrative_container_expected": administrative_container_expected,
        "frame_trigger": clean_text(combined, 300),
    }



def infer_simple_clause_arguments(text: Any) -> Dict[str, str]:
    """Infer conservative subject/predicate/object/location fields.

    This is a deterministic fallback for declarative analyst queries and corpus
    rows.  It deliberately handles a small family of common clause shapes and
    leaves uncertain fields empty rather than inventing structure.  OpenAI or
    richer parser rows may supply more precise values upstream.
    """
    source = clean_text(text)
    result = {
        "actor": "",
        "predicate": "",
        "object": "",
        "location": "",
        "patient_or_theme": "",
        "recipient": "",
        "destination": "",
        "nominal_modifier": "",
        "voice": "unknown",
        "role_normalization_status": "NOT_APPLICABLE",
    }
    if not source:
        return result

    attributed_action = re.match(
        rf"^(?:the\s+)?(?P<actor>{CLAUSE_ARGUMENT_HEAD_RE})\s+"
        rf"(?:{REPORTING_VERB_RE})\s+(?:having\s+)?"
        r"(?P<action>[A-Za-z][A-Za-z'\-]*(?:ed|ing))\s+(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if attributed_action:
        action_surface = clean_text(attributed_action.group("action"))
        predicate = canonical_predicate(action_surface)
        obj = clean_text(attributed_action.group("object"))
        if predicate in set(PREDICATE_LEMMAS.values()) and obj:
            result.update({
                "actor": clean_text(attributed_action.group("actor")),
                "predicate": predicate,
                "object": canonical_object_head(obj),
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status":
                    "ATTRIBUTED_ACTION_NORMALIZED",
            })
            return result

    directed_action = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:is|are|was|were)\s+(?:directed|under\s+orders?)\s+to\s+"
        r"(?P<action>[A-Za-z][A-Za-z'\-]*)\s+(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if directed_action:
        obj = clean_text(directed_action.group("object"))
        result.update({
            "actor": clean_text(directed_action.group("actor")),
            "predicate": canonical_predicate(
                directed_action.group("action")
            ),
            "object": canonical_object_head(obj),
            "patient_or_theme": obj,
            "voice": "active",
            "role_normalization_status": "DIRECTED_ACTION_NORMALIZED",
        })
        return result

    nominal_objective = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,120}?"
        r"\b(?:proposal|plan|policy|strategy|initiative))\s+"
        r"(?:(?:is|was)\s+(?:an?\s+)?move\s+to|aims?\s+to|"
        r"seeks?\s+to|intends?\s+to)\s+"
        r"(?P<action>[A-Za-z][A-Za-z'\-]*)\s+(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if nominal_objective:
        obj = clean_text(nominal_objective.group("object"))
        action = canonical_predicate(nominal_objective.group("action"))
        if action in {"seize", "take"} and re.search(
            r"\binitiative\b", obj, flags=re.I
        ):
            action = "gain"
        result.update({
            "actor": clean_text(nominal_objective.group("actor")),
            "predicate": action,
            "object": canonical_object_head(obj),
            "patient_or_theme": obj,
            "voice": "active",
            "role_normalization_status": "NOMINAL_OBJECTIVE_NORMALIZED",
        })
        return result

    copular_assessment = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:is|are|was|were)\s+(?:regarded|considered|viewed)\s+"
        r"(?:as\s+)?(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if copular_assessment:
        obj = clean_text(copular_assessment.group("object"))
        result.update({
            "actor": clean_text(copular_assessment.group("actor")),
            "predicate": "view",
            "object": canonical_object_head(obj),
            "patient_or_theme": obj,
            "voice": "active",
            "role_normalization_status": "COPULAR_ASSESSMENT_NORMALIZED",
        })
        return result

    exemplification = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:exemplif(?:y|ies|ied|ying)|(?:is|are|was|were)\s+"
        r"(?:an?\s+)?(?:example|instance)\s+of)\s+(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if exemplification:
        obj = clean_text(exemplification.group("object"))
        result.update({
            "actor": clean_text(exemplification.group("actor")),
            "predicate": "exemplify",
            "object": canonical_object_head(obj),
            "patient_or_theme": obj,
            "voice": "active",
            "role_normalization_status": "EXEMPLIFICATION_NORMALIZED",
        })
        return result

    timed_modal_state = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)"
        r"(?:\s+by\s+(?:19|20)\d{2})?\s+"
        r"(?:should|would|will|could|may|might|must)\s+be\s+"
        r"(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if timed_modal_state:
        obj = re.sub(
            r"\s+by\s+(?:19|20)\d{2}\b",
            " ",
            clean_text(timed_modal_state.group("object")),
            flags=re.I,
        )
        result.update({
            "actor": clean_text(timed_modal_state.group("actor")),
            "predicate": "be",
            "object": canonical_object_head(obj),
            "patient_or_theme": obj,
            "voice": "active",
            "role_normalization_status": "TIMED_MODAL_STATE_NORMALIZED",
        })
        return result

    option_search = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:is|are|was|were)\s+(?:exploring|searching\s+for)\s+"
        r"(?:options?|possibilities)\s+to\s+"
        r"(?P<action>[A-Za-z][A-Za-z'\-]*)\s+(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if option_search:
        obj = clean_text(option_search.group("object"))
        result.update({
            "actor": clean_text(option_search.group("actor")),
            "predicate": canonical_predicate(option_search.group("action")),
            "object": canonical_object_head(obj),
            "patient_or_theme": obj,
            "voice": "active",
            "role_normalization_status": "OPTION_SEARCH_ACTION_NORMALIZED",
        })
        return result

    attitude_complement = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?P<predicate>hopes?|expects?|believes?)\s+(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if attitude_complement:
        obj = clean_text(attitude_complement.group("object"))
        result.update({
            "actor": clean_text(attitude_complement.group("actor")),
            "predicate": canonical_predicate(
                attitude_complement.group("predicate")
            ),
            "object": canonical_object_head(obj),
            "patient_or_theme": obj,
            "voice": "active",
            "role_normalization_status": "ATTITUDE_COMPLEMENT_NORMALIZED",
        })
        return result

    # Attribution fragments are common analyst inputs. Preserve the actual
    # reporter instead of treating an introductory phrase such as "private
    # talks" as the actor of a synthetic reporting event.
    passive_report = re.match(
        r"^(?P<object>.+?)\s+reported\s+to\s+(?P<recipient>.+?)\s+by\s+"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,100})$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if passive_report:
        actor = clean_text(passive_report.group("actor"))
        obj = clean_text(passive_report.group("object"))
        recipient = clean_text(passive_report.group("recipient"))
        if actor and obj:
            result.update({
                "actor": actor,
                "predicate": "report",
                "object": obj,
                "patient_or_theme": obj,
                "recipient": recipient,
                "voice": "active",
                "role_normalization_status": "PASSIVE_REPORT_NORMALIZED",
            })
            return result

    # Scope refusal/declination over the embedded action. This represents
    # "X did not agree to provide Y" as a negative provide event rather than
    # losing the action and assigning Y as the grammatical actor.
    refused_action = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,90}?)\s+"
        r"(?:(?:did\s+not|would\s+not)\s+agree|refus(?:e|es|ed|ing)|"
        r"declin(?:e|es|ed|ing))\s+to\s+"
        r"(?P<predicate>[A-Za-z][A-Za-z'\-]{1,30})\s+(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if refused_action:
        actor = clean_text(refused_action.group("actor"))
        predicate = canonical_predicate(refused_action.group("predicate"))
        obj = clean_text(refused_action.group("object"))
        if actor and predicate and obj:
            result.update({
                "actor": actor,
                "predicate": predicate,
                "object": canonical_object_head(obj[:160]),
                "patient_or_theme": obj[:160],
                "voice": "active",
                "role_normalization_status": "REFUSED_ACTION_NORMALIZED",
            })
            return result

    communication_about = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,90}?)\s+"
        r"(?:speak|speaks|spoke|spoken|speaking)\s+"
        r"(?:(?:instead|primarily|mainly)\s+)?(?:of|about|concerning)\s+"
        r"(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if communication_about:
        actor = clean_text(communication_about.group("actor"))
        obj = clean_text(communication_about.group("object"))
        if actor and obj:
            result.update({
                "actor": actor,
                "predicate": "discuss",
                "object": canonical_object_head(obj[:160]),
                "patient_or_theme": obj[:160],
                "voice": "active",
                "role_normalization_status": "COMMUNICATION_TOPIC_NORMALIZED",
            })
            return result

    question_meaning = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,90}?)\s+"
        r"(?:ask|asks|asked|asking)\s+what\s+(?:is|was)\s+meant\s+by\s+"
        r"(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if question_meaning:
        actor = clean_text(question_meaning.group("actor"))
        obj = clean_text(question_meaning.group("object"))
        if actor and obj:
            result.update({
                "actor": actor,
                "predicate": "ask",
                "object": canonical_object_head(obj[:160]),
                "patient_or_theme": obj[:160],
                "voice": "active",
                "role_normalization_status": "QUESTION_MEANING_NORMALIZED",
            })
            return result

    modal_state_action = re.match(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?P<modal>will|would|can|could|may|might|shall|should|must)\s+"
        r"(?:remain|stay|be)\s+"
        r"(?P<predicate>[A-Za-z][A-Za-z'\-]*(?:ed|en))\s+"
        r"(?P<object>.+)$",
        source.rstrip(" ."),
        flags=re.I,
    )
    if modal_state_action:
        actor = clean_text(modal_state_action.group("actor"))
        predicate = canonical_predicate(
            modal_state_action.group("predicate")
        )
        raw_object = clean_text(modal_state_action.group("object"))
        obj = re.sub(
            r"^(?:at\s+least\s+)?(?:in|at|on)\s+",
            "",
            raw_object,
            flags=re.I,
        )
        obj = re.sub(
            r"^(?:no|not)\s+(?:fewer|less|more|greater)\s+than\s+",
            "",
            obj,
            flags=re.I,
        )
        location = ""
        if re.search(r"\b(?:in|at|on)\b", raw_object, flags=re.I):
            location = re.sub(r"^\d+(?:\.\d+)?\s+", "", obj).strip()
        if actor and predicate and obj:
            result.update({
                "actor": actor,
                "predicate": predicate,
                "object": canonical_object_head(obj),
                "location": location,
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status":
                    "MODAL_STATE_ACTION_NORMALIZED",
            })
            return result

    # Normalize common aspect wrappers before identifying the actor and event
    # head. Otherwise "engaged in building" can be misread as a location
    # headed by "building" or as a generic project event.
    source = re.sub(
        r"\b(?:is|are|was|were)\s+"
        r"(?P<adverb>(?:(?:covertly|secretly|actively)\s+)?)"
        r"engaged\s+in\s+(?:the\s+)?"
        r"(?:building|construction)\s+(?:of\s+)?",
        "builds ",
        source,
        flags=re.I,
    )
    source = re.sub(
        r"\b(?:is|are|was|were)\s+(?:very\s+)?close\s+to\s+joining\b",
        "is entering",
        source,
        flags=re.I,
    )
    source = re.sub(
        r"\b(?:is|are|was|were)\s+on\s+the\s+verge\s+of\s+entering\b",
        "is entering",
        source,
        flags=re.I,
    )

    evidence_support = re.match(
        r"^(?P<modifier>.+?\bevidence)(?:\s+concerning\s+.+?)?\s+"
        r"(?:lend|lends|lent)\s+(?:credence|support)\s+to\s+"
        r"(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if evidence_support:
        actor = clean_text(evidence_support.group("modifier"))
        obj = clean_text(evidence_support.group("object")).rstrip(" .")
        if actor and obj:
            result.update({
                "actor": actor,
                "predicate": "support",
                "object": canonical_object_head(obj),
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status": "EVIDENCE_SUPPORT_NORMALIZED",
            })
            return result

    # Reporting syntax has priority over nominal-event detection in the
    # embedded assertion.  Otherwise ``X states that Y opposed the proposal``
    # can be misread as a proposal event whose actor is the whole preamble.
    embedded_reporting = _reporting_clause_parts(source)
    if embedded_reporting:
        nested = infer_simple_clause_arguments(embedded_reporting["embedded"])
        if nested.get("predicate"):
            return nested

    direction_receipt = re.search(
        r"^(?:the\s+)?"
        r"(?P<patient>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:(?:has|have|had)\s+)?receiv(?:e|es|ed|ing)\s+"
        r"(?:(?:a|the)\s+)?(?P<head>direction|instructions?|orders?)\s+"
        r"(?:from|by)\s+(?P<agent>.+)$",
        source,
        flags=re.I,
    )
    if direction_receipt:
        patient = clean_text(direction_receipt.group("patient")).rstrip(" .")
        agent = clean_text(direction_receipt.group("agent")).rstrip(" .")
        head = canonical_key(direction_receipt.group("head"))
        predicate = (
            "direct"
            if head == "direction"
            else "instruct"
            if head.startswith("instruction")
            else "order"
        )
        if patient and agent:
            result.update({
                "actor": agent,
                "predicate": predicate,
                "object": canonical_object_head(patient),
                "patient_or_theme": patient,
                "voice": "active",
                "role_normalization_status": "RECEIVED_DIRECTION_NORMALIZED",
            })
            return result

    passive_obligation = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:(?:has|have|had)\s+been|is|are|was|were)\s+"
        r"(?:(?:no\s+longer|not)\s+)?(?:required|obliged|obligated)\s+to\s+"
        r"(?P<action>.+)$",
        source,
        flags=re.I,
    )
    exemption = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:(?:has|have|had)\s+been|is|are|was|were)\s+"
        r"exempt(?:ed)?\s+from\s+(?P<action>.+)$",
        source,
        flags=re.I,
    )
    obligation = passive_obligation or exemption
    if obligation:
        actor = clean_text(obligation.group("actor"))
        action = clean_text(obligation.group("action")).rstrip(" .")
        action_parts = action.split(maxsplit=1)
        if action_parts:
            action_head = canonical_predicate(action_parts[0])
            action = clean_text(
                " ".join((action_head, action_parts[1]))
                if len(action_parts) > 1 else action_head
            )
        if actor and action:
            result.update({
                "actor": actor,
                "predicate": "require",
                "object": canonical_object_head(action),
                "patient_or_theme": action,
                "voice": "active",
                "role_normalization_status":
                    "PASSIVE_OBLIGATION_NORMALIZED",
            })
            return result

    negated_exhibition = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:(?:has|have|had)\s+)?"
        r"(?:exhibit(?:s|ed|ing)?|show(?:s|ed|ing)?|display(?:s|ed|ing)?)\s+"
        r"(?:(?:practically|virtually|essentially|almost)\s+)?"
        r"(?:no|little)\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if negated_exhibition:
        actor = clean_text(negated_exhibition.group("actor"))
        obj = clean_text(negated_exhibition.group("object")).rstrip(" .")
        if actor and obj:
            result.update({
                "actor": actor,
                "predicate": "exhibit",
                "object": canonical_object_head(obj),
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status":
                    "NEGATED_EXHIBITION_NORMALIZED",
            })
            return result

    direct_projection = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:(?:has|have|had)\s+)?"
        r"(?:project(?:s|ed|ing)?|forecast(?:s|ed|ing)?)\s+"
        r"(?P<object>.+)$",
        source,
        flags=re.I,
    )
    contemplated_projection = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:is|are|was|were)\s+"
        r"(?:(?:now|currently|reportedly|already)\s+){0,3}"
        r"(?:thinking|considering)\s+(?:of|about)\s+"
        r"(?P<object>.+\b(?:projection|forecast)\b.*)$",
        source,
        flags=re.I,
    )
    contemplated_object = (
        clean_text(contemplated_projection.group("object"))
        if contemplated_projection else ""
    )
    contemplated_is_quantitative = bool(re.search(
        r"\b(?:growth|inflation|rate|percent(?:age)?|output|production|"
        r"revenue|cost|price|budget|population|volume|quantity|economic)\b|%",
        contemplated_object,
        flags=re.I,
    ))
    projection = direct_projection or (
        contemplated_projection if contemplated_is_quantitative else None
    )
    if projection:
        actor = clean_text(projection.group("actor"))
        raw_object = clean_text(projection.group("object")).rstrip(" .")
        metrics = []
        if re.search(r"\bgrowth\s+rate\b", raw_object, flags=re.I):
            metrics.append("annual growth rate")
        if re.search(r"\binflation\b", raw_object, flags=re.I):
            metrics.append("annual inflation")
        normalized_object = " and ".join(metrics) or raw_object
        if actor and normalized_object:
            result.update({
                "actor": actor,
                "predicate": "project",
                "object": canonical_object_head(normalized_object),
                "patient_or_theme": normalized_object,
                "voice": "active",
                "role_normalization_status":
                    "QUANTITATIVE_PROJECTION_NORMALIZED",
            })
            return result

    construction_engagement = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:is|are|was|were|has|have|had)\s+"
        r"(?:(?:reportedly|allegedly|secretly|covertly|actively|already|"
        r"currently)\s+){0,4}(?:been\s+)?(?:engaged|busy)\s+"
        r"(?:in|with)\s+(?:the\s+)?(?:"
        r"(?P<nominal>construction|building|development)\s+of\s+"
        r"(?P<object_after>[A-Za-z0-9][A-Za-z0-9'().&/,\- ]{1,150})|"
        r"(?P<object_before>[A-Za-z0-9][A-Za-z0-9'().&/,\- ]{1,150}?)\s+"
        r"(?P<trailing_nominal>construction|building|development)"
        r")\s*[.!?]?$",
        source,
        flags=re.I,
    )
    if construction_engagement:
        actor = clean_text(construction_engagement.group("actor"))
        obj = clean_text(
            construction_engagement.group("object_after")
            or construction_engagement.group("object_before")
        ).rstrip(" .")
        if actor and obj:
            result.update({
                "actor": actor,
                "predicate": "build",
                "object": canonical_object_head(obj),
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status":
                    "CONSTRUCTION_ENGAGEMENT_NORMALIZED",
            })
            return result

    promotion_activity = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:is|are|was|were)\s+"
        r"(?:(?:currently|reportedly|publicly|actively)\s+){0,3}"
        r"active\s+in\s+promoting\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if promotion_activity:
        actor = clean_text(promotion_activity.group("actor"))
        obj = clean_text(promotion_activity.group("object")).rstrip(" .")
        if actor and obj:
            result.update({
                "actor": actor,
                "predicate": "promote",
                "object": canonical_object_head(obj),
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status":
                    "PROMOTION_ACTIVITY_NORMALIZED",
            })
            return result

    possession = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:(?:is|are|was|were)\s+)?"
        r"(?:(?:presently|currently|still|already|reportedly)\s+){0,3}"
        r"(?:(?:continues?|continued)\s+to\s+)?"
        r"(?:has|have|had|enjoy(?:s|ed|ing)?|possess(?:es|ed|ing)?)\s+"
        r"(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if possession:
        actor = clean_text(possession.group("actor"))
        obj = clean_text(possession.group("object")).rstrip(" .")
        invalid_possession_actor = bool(re.search(
            r"\b(?:to|claim(?:s|ed|ing)?|report(?:s|ed|ing)?|"
            r"say|says|said|saying|state(?:s|d|ing)?)$",
            actor,
            flags=re.I,
        ))
        possession_head = re.search(
            r"\b(?:access|prerogatives?|privileges?|rights?|authority|"
            r"control|custody|capacity|ability|relationship|ties?|contacts?)\b",
            obj,
            flags=re.I,
        )
        if actor and obj and possession_head and not invalid_possession_actor:
            result.update({
                "actor": actor,
                "predicate": "have",
                "object": canonical_object_head(obj),
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status":
                    "POSSESSION_FRAME_NORMALIZED",
            })
            return result

    direct_assessment = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:view(?:s|ed|ing)?|regard(?:s|ed)?)\s+"
        r"(?P<object>.+)$",
        source,
        flags=re.I,
    )
    realistic_assessment = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:is|are|was|were)\s+realistic\s+(?:about|concerning|regarding)\s+"
        r"(?P<theme>.+?)\s+as\s+(?P<assessment>.+)$",
        source,
        flags=re.I,
    )
    # The adjective frame must win before the generic ``regarding`` verb
    # spelling; otherwise ``Razak was realistic`` is swallowed as the actor.
    assessment = realistic_assessment or direct_assessment
    if assessment:
        actor = clean_text(assessment.group("actor"))
        if realistic_assessment:
            obj = clean_text(
                realistic_assessment.group("theme")
                + " as "
                + realistic_assessment.group("assessment")
            )
        else:
            obj = clean_text(assessment.group("object"))
        obj = re.sub(
            r"\b(?:a\s+)?(?:very\s+)?long[-\s]+range\b",
            "distant",
            obj,
            flags=re.I,
        )
        actor_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", actor)
        invalid_active_actor = bool(
            re.search(
                r"\b(?:is|are|was|were|be|been|being|not|in|on|about|"
                r"concerning)$",
                actor,
                flags=re.I,
            )
            or "," in actor
            or len(actor_words) > 10
        )
        if actor and obj and not invalid_active_actor:
            result.update({
                "actor": actor,
                "predicate": "view",
                "object": canonical_object_head(obj),
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status":
                    "ASSESSMENT_FRAME_NORMALIZED",
            })
            return result

    awareness = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:is|are|was|were|remains?|remained)\s+"
        r"(?:(?:keenly|fully|well|acutely|clearly|personally)\s+){0,3}"
        r"aware\s+of\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if awareness:
        actor = clean_text(awareness.group("actor"))
        obj = clean_text(awareness.group("object")).rstrip(" .")
        if actor and obj:
            result.update({
                "actor": actor,
                "predicate": "recognize",
                "object": canonical_object_head(obj),
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status": "AWARENESS_NORMALIZED",
            })
            return result

    evidence_absence = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"(?:has|have|had)\s+no\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if evidence_absence:
        actor = clean_text(evidence_absence.group("actor"))
        obj = re.sub(
            r"\b(?:regarding|concerning|about|on)\b",
            " ",
            clean_text(evidence_absence.group("object")),
            flags=re.I,
        )
        obj = clean_text(obj).rstrip(" .")
        if actor and obj:
            result.update({
                "actor": actor,
                "predicate": "lack",
                "object": canonical_object_head(obj),
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status": "EVIDENCE_ABSENCE_NORMALIZED",
            })
            return result

    lexical_absence = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,90}?)\s+"
        r"lack(?:s|ed|ing)?\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if lexical_absence:
        actor = clean_text(lexical_absence.group("actor"))
        obj = re.sub(
            r"\b(?:regarding|concerning|about|on)\b",
            " ",
            clean_text(lexical_absence.group("object")),
            flags=re.I,
        )
        obj = clean_text(obj).rstrip(" .")
        if actor and obj:
            result.update({
                "actor": actor,
                "predicate": "lack",
                "object": canonical_object_head(obj),
                "patient_or_theme": obj,
                "voice": "active",
                "role_normalization_status": "EVIDENCE_ABSENCE_NORMALIZED",
            })
            return result

    security_intensification = re.search(
        r"^(?:the\s+)?(?P<actor>security)"
        r"(?:\s+(?:at|around|in|near)\s+"
        r"(?P<location>[A-Za-z0-9][A-Za-z0-9'().&/\- ]{0,70}?))?\s+"
        r"(?:(?:has|have|had|is|are|was|were)\s+)?"
        r"(?P<predicate>step(?:s|ped|ping)?\s+up|increase(?:s|d|ing)?|"
        r"heighten(?:s|ed|ing)?|intensif(?:y|ies|ied|ying))\b"
        r"(?P<tail>.*)$",
        source,
        flags=re.I,
    )
    if security_intensification:
        actor = clean_text(security_intensification.group("actor"))
        location = clean_text(
            security_intensification.group("location")
        )
        if actor:
            result.update({
                "actor": actor,
                "predicate": "increase",
                "object": "security",
                "location": location,
                "patient_or_theme": "security",
                "voice": "active",
                "role_normalization_status":
                    "SECURITY_INTENSIFICATION_NORMALIZED",
            })
            return result

    direct_confirmation_request = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,80}?)\s+"
        r"request(?:s|ed|ing)?\s+"
        r"(?P<recipient>[A-Za-z0-9][A-Za-z0-9'().&/,\- ]{1,100}?)\s+"
        r"to\s+confirm\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    indirect_confirmation_request = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{0,80}?)\s+"
        r"(?:(?:is|are|was|were)\s+)?call(?:s|ed|ing)?\s+on\s+"
        r"(?P<recipient>[A-Za-z0-9][A-Za-z0-9'().&/,\- ]{1,100}?)\s+"
        r"(?:to\s+confirm|seek(?:s|ing)?\s+confirmation\s+of)\s+"
        r"(?P<object>.+)$",
        source,
        flags=re.I,
    )
    confirmation_request = (
        direct_confirmation_request or indirect_confirmation_request
    )
    if confirmation_request:
        actor = clean_text(confirmation_request.group("actor"))
        recipient = clean_text(confirmation_request.group("recipient"))
        obj = clean_text(confirmation_request.group("object")).rstrip(" .")
        normalized_object = clean_text(
            f"{recipient} confirm {obj}"
        )
        if actor and recipient and obj:
            result.update({
                "actor": actor,
                "predicate": "request",
                "object": canonical_object_head(normalized_object),
                "patient_or_theme": obj,
                "recipient": recipient,
                "voice": "active",
                "role_normalization_status":
                    "REQUEST_CONFIRMATION_NORMALIZED",
            })
            return result

    sabotage_placement = re.search(
        r"^(?P<instrument>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,100}?)\s+"
        r"(?:is|are|was|were)\s+placed\b.*?\bto\s+sabotag(?:e|ed|ing)\s+"
        r"(?P<target>[^.;]+)",
        source,
        flags=re.I,
    )
    sabotage_passive = re.search(
        r"^(?P<target>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,100}?)\s+"
        r"(?:is|are|was|were)\s+(?:sappered|sabotaged)\b"
        r"(?:\s+(?:with|by)\s+(?P<instrument>[^.;]+))?",
        source,
        flags=re.I,
    )
    sabotage_event = sabotage_placement or sabotage_passive
    if sabotage_event:
        target = clean_text(sabotage_event.group("target")).rstrip(" .")
        if target:
            result.update({
                "actor": "",
                "predicate": "sabotage",
                "object": canonical_object_head(target),
                "patient_or_theme": target,
                "voice": "passive",
                "role_normalization_status": "SABOTAGE_EVENT_NORMALIZED",
            })
            return result

    coordinated_reopening = re.search(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/\- ]{0,70}?)"
        r"(?:,\s*closed\b[^,]{0,120},|\s+(?:was|were)\s+closed\b"
        r"[^.;]{0,140}?\band)\s+reopen(?:s|ed|ing)?\b",
        source,
        flags=re.I,
    )
    if coordinated_reopening:
        actor = clean_text(coordinated_reopening.group("actor"))
        if actor:
            result.update({
                "actor": actor,
                "predicate": "reopen",
                "object": canonical_object_head(actor),
                "patient_or_theme": actor,
                "voice": "active",
                "role_normalization_status":
                    "COORDINATED_REOPENING_NORMALIZED",
            })
            return result

    normative_condemnation = re.search(
        r"^(?P<actor>(?:each|every)\s+"
        r"(?:delegation|del|delegate|representative))\s+"
        r"(?:held|had)\s+(?P<object>.+?\bviews?)\s+and\s+"
        r"should\s+not\s+(?:be\s+condemned|face\s+condemnation)\b",
        source,
        flags=re.I,
    )
    if normative_condemnation:
        actor = clean_text(normative_condemnation.group("actor"))
        obj = clean_text(normative_condemnation.group("object"))
        result.update({
            "actor": actor,
            "predicate": "condemn",
            "object": canonical_object_head(obj),
            "patient_or_theme": obj,
            "voice": "passive",
            "role_normalization_status":
                "NORMATIVE_CONDEMNATION_NORMALIZED",
        })
        return result

    coordinated_transport = re.search(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,90}?)\s+"
        r"(?:(?:is|are|was|were|be|been|being|has|have|had|do|does|did|"
        r"will|would|can|could|should|may|might|must|shall)\s+)?"
        r"(?P<predicate1>[A-Za-z][A-Za-z' -]{1,30})\s+.+?\s+and\s+"
        r"(?P<predicate2>deliver(?:ed|s|ing)?|transport(?:ed|s|ing)?|"
        r"send(?:s|ing)?|sent|escort(?:ed|s|ing)?|take(?:n|s|ing)?|"
        r"move(?:d|s|ing)?|transfer(?:red|s|ring)?)\s+"
        r"(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if coordinated_transport:
        actor = clean_text(coordinated_transport.group("actor"))
        actor = re.sub(
            r"\s+(?:is|are|was|were|be|been|being|has|have|had|do|does|did|"
            r"will|would|can|could|should|may|might|must|shall)"
            r"(?:\s+(?:not|never|yet|already|still|likely|probably)){0,4}$",
            "",
            actor,
            flags=re.I,
        )
        actor = re.sub(
            r"\s+(?:likely|probably|significantly|genuinely|still|already|"
            r"both|currently|further)$",
            "",
            actor,
            flags=re.I,
        )
        tail = clean_text(coordinated_transport.group("object"))
        location_match = re.search(
            r"\b(?:to|toward|towards|into|onto)\s+"
            r"(?P<location>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,100})"
            r"(?=\s+(?:where|when|while|after|before)\b|[.;,]|$)",
            tail,
            flags=re.I,
        )
        obj = tail
        destination = ""
        if location_match:
            destination = clean_text(location_match.group("location"))
            obj = clean_text(tail[:location_match.start()])
        if actor and obj:
            result["actor"] = actor
            result["predicate"] = canonical_predicate(
                coordinated_transport.group("predicate2")
            )
            result["object"] = canonical_object_head(obj[:160])
            result["patient_or_theme"] = clean_text(obj[:160])
            result["location"] = destination
            result["destination"] = destination
            result["voice"] = "active"
            result["role_normalization_status"] = (
                "COORDINATED_TRANSPORT_FALLBACK"
            )
            return result

    no_mention = re.search(
        r"^(?:there\s+(?:is|was|has\s+been)\s+)?no\s+mention\s+"
        r"(?:was\s+made\s+)?of\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    passive_unmentioned = re.search(
        r"^(?P<object>.+?)\s+"
        r"(?:(?:has|have|had)\s+been|is|are|was|were)\s+"
        r"(?:not|never)\s+mentioned\b(?:\s+.*)?$",
        source,
        flags=re.I,
    )
    mention_object = clean_text(
        no_mention.group("object") if no_mention else (
            passive_unmentioned.group("object")
            if passive_unmentioned else ""
        )
    )
    if mention_object:
        result.update({
            "predicate": "mention",
            "object": canonical_object_head(mention_object[:160]),
            "patient_or_theme": mention_object[:160],
            "voice": "passive",
            "role_normalization_status": "MENTION_ABSENCE_NORMALIZED",
        })
        return result

    # A denial scopes over the embedded event rather than making ``deny`` the
    # event being compared.  Only a recognized gerund is lifted, so ordinary
    # noun-object denials retain the established fallback behavior.
    denied_event = re.search(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,90}?)\s+"
        r"den(?:y|ies|ied|ying)\s+"
        r"(?P<predicate>[A-Za-z][A-Za-z'\-]{2,30}ing)\b"
        r"(?P<object>.*)$",
        source,
        flags=re.I,
    )
    if denied_event:
        predicate_surface = clean_text(denied_event.group("predicate")).lower()
        predicate = canonical_predicate(predicate_surface)
        obj = clean_text(denied_event.group("object"))
        if predicate_surface in PREDICATE_LEMMAS and predicate not in {
            "be", "have", "deny", "report",
        } and obj:
            if predicate == "implicate":
                obj = re.sub(
                    r"^(?:anybody|anyone|somebody|someone)\s+"
                    r"(?:in|about|concerning|regarding)\s+",
                    "",
                    obj,
                    flags=re.I,
                )
            result.update({
                "actor": clean_text(denied_event.group("actor")),
                "predicate": predicate,
                "object": canonical_object_head(obj[:160]),
                "patient_or_theme": obj[:160],
                "voice": "active",
                "role_normalization_status": "DENIED_EVENT_NORMALIZED",
            })
            return result

    # Nominalized events are interpreted from an explicit grammatical
    # inventory.  Finite and irregular verb forms must never enter this path.
    nominal_forms = sorted(NOMINAL_EVENT_FORMS, key=len, reverse=True)
    nominal_event = re.search(
        r"^(?:the\s+)?(?P<prefix>[A-Za-z][A-Za-z0-9'’&/\- ]{0,80}?)?"
        r"\b(?P<nominal>"
        + "|".join(re.escape(value) for value in nominal_forms)
        + r")\b(?P<rest>.*)$",
        source,
        flags=re.I,
    ) if nominal_forms else None
    nominal_prefix_tokens = (
        canonical_key(nominal_event.group("prefix")).split()
        if nominal_event else []
    )
    nominal_prefix_has_finite_predicate = any(
        token in PREDICATE_LEMMAS
        and token not in {"state", "states"}
        and (
            token.endswith(("ed", "ing", "s"))
            or token in {
                "bore", "bought", "built", "gave", "left", "met", "said",
                "sought", "told", "withdrew", "wound",
            }
        )
        for token in nominal_prefix_tokens
    )
    bare_fragment_like = re.match(
        rf"^(?:(?:{PREDICATE_FRAME_ADVERB_RE})\s+){{0,3}}"
        rf"(?:{BARE_FRAGMENT_PREDICATE_RE})\b",
        source,
        flags=re.I,
    )
    nominal_withdrawal_like = re.search(
        r"^(?:calls?\s+for|demands?|urges?)\s+"
        r"(?:withdrawal|departure|removal)\s+of\b",
        source,
        flags=re.I,
    )
    if (
        nominal_event
        and not nominal_prefix_has_finite_predicate
        and not nominal_withdrawal_like
        and not bare_fragment_like
    ):
        rest = clean_text(nominal_event.group("rest"))
        result["nominal_modifier"] = clean_text(
            nominal_event.group("prefix")
        )
        location_match = re.search(
            r"\b(?:occurred|took\s+place|was)\s+(?:at|in|near)\s+"
            r"(?P<location>[A-Za-z][A-Za-z0-9'().&/\- ]{1,70})",
            rest,
            flags=re.I,
        )
        measured_object_match = re.search(
            r"\b(?:packets?|packages?|crates?|containers?|units?|items?)"
            r"\s+of\s+(?P<object>[A-Za-z][A-Za-z0-9'’&/\-]*"
            r"(?:\s+[A-Za-z][A-Za-z0-9'’&/\-]*){0,5}?)"
            r"(?=\s+(?:weighing|weighed|weight|at|in|on|during|"
            r"before|after|that|which)\b|[.,;:]|$)",
            rest,
            flags=re.I,
        )
        result["predicate"] = canonical_predicate(
            nominal_event.group("nominal")
        )
        result["object"] = canonical_object_head(
            measured_object_match.group("object")
            if measured_object_match
            else nominal_event.group("prefix")
        )
        result["location"] = re.sub(r"[\s,.;:]+$", "", clean_text(
            location_match.group("location")
            if location_match else ""
        ))
        return result

    action_re = (
        r"seized|confiscated|arrested|transported|attended|represented|"
        r"reported|informed|met|received|transferred|delivered|sent|"
        r"referred|turned over|offered|propos(?:e|es|ed|ing)|"
        r"urge(?:s|d|ing)?|misled|bought|"
        r"buying|seek|seeks|sought|seeking|search|searches|searched|searching|aim|aims|aimed|aiming|"
        r"plan|plans|planned|planning|intend|intends|intended|intending|"
        r"press|presses|pressed|pressing|support|supports|supported|supporting|"
        r"receive|receives|received|receiving|participat(?:e|es|ed|ing)|"
        r"valu(?:e|es|ed|ing)|prioritiz(?:e|es|ed|ing)|affirm|affirms|affirmed|affirming|"
        r"us(?:e|es|ed|ing)|cit(?:e|es|ed|ing)|claimed|claims|claiming|"
        r"den(?:y|ies|ied|ying)|admit(?:ted|s|ting)?|demand(?:ed|s|ing)?|"
        r"want(?:s|ed|ing)?|limit(?:ed|s|ing)?|replace(?:d|s|ing)?|"
        r"call(?:ed|s|ing)?|reaffirm(?:ed|s|ing)?|back(?:ed|s|ing)?|"
        r"lead|leads|led|leading|lend|lends|lent|lending|conduct(?:ed|s|ing)?|engag(?:e|es|ed|ing)|leave|leaves|leaving|left|"
        r"relocat(?:e|ed|es|ing)|lacks|lack|lacking|entered|entering|concern(?:s|ed|ing)?|"
        r"remember(?:ed|s|ing)?|recall(?:ed|s|ing)?|"
        r"travel(?:led|ed|s|ing)?|come|comes|came|coming|"
        r"arriv(?:e|es|ed|ing)|head(?:s|ed|ing)?|"
        r"injur(?:e|es|ed|ing)|hurt|hurts|hurting|"
        r"wound(?:s|ed|ing)?|coordinat(?:e|es|ed|ing)|"
        r"direct(?:ed|s|ing)?|contradict(?:ed|s|ing)?|"
        r"follow(?:ed|s|ing)?|recogniz(?:e|es|ed|ing)|lift(?:ed|s|ing)|"
        r"interact(?:ed|s|ing)?|meet(?:ing|s)?|met|"
        r"authoriz(?:e|es|ed|ing)|mention(?:s|ed|ing)?|"
        r"point(?:s|ed|ing)?|implicat(?:e|es|ed|ing)|"
        r"influenc(?:e|es|ed|ing)|oppos(?:e|es|ed|ing)|"
        r"protect(?:s|ed|ing)?|shield(?:s|ed|ing)?|"
        r"criticiz(?:e|es|ed|ing)|criticis(?:e|es|ed|ing)|"
        r"contact(?:s|ed|ing)?|shell(?:s|ed|ing)?|bombard(?:s|ed|ing)?|"
        r"detain(?:s|ed|ing)?|imprison(?:s|ed|ing)?|hold|holds|held|holding|"
        r"retain(?:s|ed|ing)?|"
        r"build(?:s|ing)?|built|construct(?:s|ed|ing)?|"
        r"develop(?:s|ed|ing)?|allow(?:s|ed|ing)?|"
        r"requir(?:e|es|ed|ing)|oblig(?:e|es|ed|ing)|maintain(?:s|ed|ing)?|"
        r"strengthen(?:s|ed|ing)?|improv(?:e|es|ed|ing)|"
        r"promot(?:e|es|ed|ing)|advanc(?:e|es|ed|ing)|"
        r"compl(?:y|ies|ied|ying)|"
        r"fulfil(?:l|s|ls|led|ling)?|meet(?:s|ing)?|met|"
        r"view(?:s|ed|ing)?|regard(?:s|ed)?|"
        r"enjoy(?:s|ed|ing)?|possess(?:es|ed|ing)?|"
        r"remain(?:s|ed|ing)?|"
        r"attack(?:s|ed|ing)?|exhibit(?:s|ed|ing)?|"
        r"project(?:s|ed|ing)?|forecast(?:s|ed|ing)?|"
        r"exploit(?:s|ed|ing)?|eat(?:s|ing)?|ate|eaten"
        r"|extend(?:s|ed|ing)?"
    )
    passive_action_re = (
        r"seized|confiscated|arrested|transported|attended|represented|"
        r"reported|informed|met|received|transferred|delivered|sent|"
        r"referred|turned over|offered|proposed|urged|misled|bought|"
        r"claimed|lacked|entered|travel(?:led|ed)|coordinated|directed|"
        r"moved|relocated|"
        r"contradicted|followed|recognized|lifted|interacted|"
        r"authorized|mentioned|implicated|influenced|opposed|supported|"
        r"protected|shielded|criticized|criticised|contacted|shelled|bombarded|"
        r"detained|imprisoned|held|retained|"
        r"built|constructed|developed|allowed|required|obliged|maintained|"
        r"attacked|projected|forecasted|exploited|strengthened|improved|"
        r"promoted|advanced|fulfilled"
    )

    # Passive voice is evaluated before the active fallback.  The surface
    # subject is a patient/theme, and an unexpressed agent remains unknown.
    passive = re.search(
        rf"^(?:the\s+)?(?P<patient>{CLAUSE_ARGUMENT_HEAD_RE})"
        rf"\s+(?:(?:has|have|had)\s+"
        rf"(?:(?:{PREDICATE_FRAME_ADVERB_RE})\s+){{0,3}}been|"
        rf"(?:will|would|can|could|should|may|might|must|shall)\s+"
        rf"(?:(?:{PREDICATE_FRAME_ADVERB_RE})\s+){{0,3}}(?:have\s+)?been|"
        rf"(?:is|are|was|were|be|been|being)(?:\s+being)?)\s+"
        rf"(?:(?:{PREDICATE_FRAME_ADVERB_RE})\s+){{0,4}}"
        rf"(?P<predicate>{passive_action_re})\b(?P<tail>.*)$",
        source,
        flags=re.I,
    )
    if passive:
        tail = clean_text(passive.group("tail"))
        patient = re.sub(
            r"\s+(?:has|have|had|is|are|was|were|be|been|being|"
            r"can|could|may|might|must|shall|should|will|would)$",
            "",
            clean_text(passive.group("patient")),
            flags=re.I,
        )
        agent_match = re.search(
            r"\bby\s+[\"']*(?P<agent>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,100}?)"
            r"(?=\s+(?:to|for|at|in|on|during|after|before)\b|[.;,]|$)",
            tail,
            flags=re.I,
        )
        target_match = re.search(
            r"\bto\s+(?P<target>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,100})"
            r"(?=\s+(?:at|in|on|during|after|before)\b|[.;,]|$)",
            tail,
            flags=re.I,
        )
        target = (
            clean_text(target_match.group("target"))
            if target_match else ""
        )
        recipient = (
            target
            if re.search(
                r"\b(official|minister|director|tribunal|court|authority|"
                r"authorities|police|officer|committee|person)\b",
                target,
                flags=re.I,
            )
            else ""
        )
        result.update({
            "actor": (
                clean_text(agent_match.group("agent")).strip("\"' .")
                if agent_match else ""
            ),
            "predicate": canonical_predicate(passive.group("predicate")),
            "object": canonical_object_head(patient),
            "patient_or_theme": clean_text(patient),
            "recipient": recipient,
            "destination": target if target and not recipient else "",
            "voice": "passive",
            "role_normalization_status": (
                "PASSIVE_AGENT_EXPLICIT"
                if agent_match else "PASSIVE_AGENT_UNKNOWN"
            ),
        })
        return result

    willingness = re.search(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,90}?)\s+"
        r"(?:(?:'s)?\s*position\s+is\s+that\s+(?:it|they)\s+)?"
        r"(?:(?:is|are|was|were|be|been|being|remain|remains|remained)\s+)?"
        r"(?:still\s+|already\s+)?willing\s+to\s+"
        r"(?:(?:continue|resume)\s+)?"
        r"(?P<predicate>discuss|discussion|talk|talks)\s+"
        r"(?:of\s+)?(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if willingness:
        actor = re.sub(
            r"(?:'s|’s)$",
            "",
            clean_text(willingness.group("actor")),
            flags=re.I,
        )
        obj = clean_text(willingness.group("object"))
        actor_word_count = len(
            re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", actor)
        )
        if actor and obj and actor_word_count <= 10:
            result["actor"] = actor
            result["predicate"] = "discuss"
            result["object"] = canonical_object_head(obj[:160])
            result["patient_or_theme"] = clean_text(obj[:160])
            result["voice"] = "active"
            result["role_normalization_status"] = (
                "WILLINGNESS_DISCUSSION_FALLBACK"
            )
            return result

    involvement = re.search(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,90}?)\s+"
        r"(?:is|are|was|were|be|been|being)\s+not\s+refusing\s+to\s+"
        r"get\s+involved\s+in\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if involvement:
        actor = clean_text(involvement.group("actor"))
        obj = clean_text(involvement.group("object"))
        actor_word_count = len(
            re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", actor)
        )
        if actor and obj and actor_word_count <= 12:
            result["actor"] = actor
            result["predicate"] = "involve"
            result["object"] = canonical_object_head(obj[:160])
            result["patient_or_theme"] = clean_text(obj[:160])
            result["location"] = clean_text(obj[:160])
            result["voice"] = "active"
            result["role_normalization_status"] = (
                "NEGATED_INVOLVEMENT_FALLBACK"
            )
            return result

    nominal_withdrawal = re.search(
        r"^(?P<cue>calls?\s+for|demands?|urges?)\s+"
        r"(?P<head>withdrawal|departure|removal)\s+of\s+"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,120}?)\s+"
        r"from\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if nominal_withdrawal:
        actor = clean_text(nominal_withdrawal.group("actor"))
        obj = clean_text(nominal_withdrawal.group("object"))
        head = canonical_key(nominal_withdrawal.group("head"))
        predicate = (
            "leave"
            if head == "departure"
            else "withdraw"
        )
        if actor and obj:
            result["actor"] = actor
            result["predicate"] = predicate
            result["object"] = canonical_object_head(obj[:160])
            result["patient_or_theme"] = clean_text(obj[:160])
            result["location"] = clean_text(obj[:160])
            result["voice"] = "active"
            result["role_normalization_status"] = (
                "NOMINAL_WITHDRAWAL_FALLBACK"
            )
            return result

    nominal_claim = re.search(
        r"^(?:the\s+)?(?P<head>support|backing|endorsement|reaffirmation|"
        r"opposition|denial)\s+(?:for|of|to)\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if nominal_claim:
        predicate_map = {
            "support": "support",
            "backing": "support",
            "endorsement": "support",
            "reaffirmation": "affirm",
            "opposition": "oppose",
            "denial": "deny",
        }
        head = clean_text(nominal_claim.group("head")).lower()
        obj = clean_text(nominal_claim.group("object"))
        if obj:
            result["predicate"] = predicate_map.get(head, head)
            result["object"] = canonical_object_head(obj[:160])
            result["patient_or_theme"] = clean_text(obj[:160])
            result["voice"] = "active"
            result["role_normalization_status"] = "NOMINAL_CLAIM_FALLBACK"
            return result

    entitlement = re.search(
        r"^(?:(?P<modifier>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,60}?)\s+)?"
        r"entitlement\s+to\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if entitlement:
        obj = clean_text(entitlement.group("object"))
        if obj:
            result["predicate"] = "entitle"
            result["object"] = canonical_object_head(obj[:160])
            result["patient_or_theme"] = clean_text(obj[:160])
            result["nominal_modifier"] = clean_text(
                entitlement.group("modifier")
            )
            result["voice"] = "active"
            result["role_normalization_status"] = "ENTITLEMENT_FALLBACK"
            return result

    policy_causation = re.search(
        r"^(?:the\s+)?"
        r"(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,140}?\bpolicy\b"
        r"[A-Za-z0-9'().&/,\- ]{0,140}?)\s+"
        r"(?:(?:has|have|had)\s+)?"
        r"(?:(?:already|also|well|recently|therefore)\s+){0,3}"
        r"(?P<predicate>lead|leads|led|leading)\s+"
        r"(?:(?:it|them)\s+)?(?:to\s+)?(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if policy_causation:
        actor = clean_text(policy_causation.group("actor"))
        obj = clean_text(policy_causation.group("object"))
        if actor and obj:
            result["actor"] = actor
            result["predicate"] = canonical_predicate(
                policy_causation.group("predicate")
            )
            result["object"] = canonical_object_head(obj[:160])
            result["patient_or_theme"] = clean_text(obj[:160])
            result["voice"] = "active"
            result["role_normalization_status"] = (
                "POLICY_CAUSATION_FALLBACK"
            )
            return result

    # Normalize explicit passive action clauses before the generic active
    # frame can mistake the patient for the actor.  Only known predicate forms
    # enter this path, and an explicit ``by <agent>`` phrase is required.
    passive_action = re.search(
        rf"^(?:the\s+)?"
        rf"(?P<patient>[A-Za-z0-9][A-Za-z0-9'().&/,\- ]{{1,120}}?)\s+"
        rf"(?:(?:has|have|had)\s+been|is|are|was|were)\s+"
        rf"(?:(?:{PREDICATE_FRAME_ADVERB_RE})\s+){{0,3}}"
        rf"(?P<predicate>{action_re})\s+by\s+(?P<agent>.+)$",
        source,
        flags=re.I,
    )
    if passive_action:
        actor = clean_text(passive_action.group("agent")).rstrip(" .")
        # A following means phrase is evidence context, not part of the agent.
        actor = re.split(
            r"\s+(?:using|throwing|firing|employing|wielding)\b",
            actor,
            maxsplit=1,
            flags=re.I,
        )[0]
        patient = clean_text(passive_action.group("patient")).rstrip(" .")
        predicate = canonical_predicate(passive_action.group("predicate"))
        if actor and patient and predicate not in {"", "be", "have", "other"}:
            result.update({
                "actor": actor,
                "predicate": predicate,
                "object": canonical_object_head(patient[:160]),
                "patient_or_theme": patient[:160],
                "voice": "active",
                "role_normalization_status": "PASSIVE_ACTION_NORMALIZED",
            })
            return result

    # Actor + optional location adjunct + governing action.  The actor pattern
    # is bounded and non-greedy so a date/quantity phrase is not absorbed.
    match = re.search(
        rf"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/\- ]{{1,90}}?)"
        rf"(?:\s+(?:at|in|near)\s+(?P<location>[A-Za-z][A-Za-z0-9'().&/\- ]{{1,70}}?))?"
        rf"\s+(?:(?:"
        rf"(?:will|would|can|could|should|may|might|must|shall)\s+"
        rf"(?:(?:{PREDICATE_FRAME_ADVERB_RE})\s+){{0,4}}(?:have\s+)?|"
        rf"(?:has|have|had)\s+"
        rf"(?:(?:{PREDICATE_FRAME_ADVERB_RE})\s+){{0,4}}(?:been\s+)?|"
        rf"(?:is|are|was|were|be|been|being|do|does|did)\s+"
        rf"(?:(?:{PREDICATE_FRAME_ADVERB_RE})\s+){{0,4}}"
        rf")?)"
        rf"(?:(?:continue|continues|continued|continuing)\s+to\s+)?"
        rf"(?:(?:{PREDICATE_FRAME_ADVERB_RE})\s+){{0,4}}"
        rf"(?P<predicate>{action_re})\b(?P<object>.*)$",
        source,
        flags=re.I,
    )
    if match:
        actor = clean_text(match.group("actor"))
        # A passive auxiliary belongs to the predicate frame, never to the
        # actor.  Keeping it in the actor produced values such as
        # ``trafficker was`` for ``The trafficker was arrested``.
        actor = re.sub(
            r"\s+(?:is|are|was|were|be|been|being|has|have|had|do|does|did|"
            r"will|would|can|could|should|may|might|must|shall)"
            r"(?:\s+(?:not|never|yet|already|still|likely|probably)){0,4}$",
            "",
            actor,
            flags=re.I,
        )
        actor = re.sub(
            r"\s+(?:likely|probably|significantly|genuinely|still|already|"
            r"both|currently)$",
            "",
            actor,
            flags=re.I,
        )
        result["actor"] = clean_text(actor)
        result["predicate"] = canonical_predicate(match.group("predicate"))
        result["location"] = clean_text(match.group("location"))
        obj = clean_text(match.group("object"))
        if result["predicate"] in {
            "aim", "expect", "intend", "plan", "seek", "want",
        }:
            controlled_action = re.match(
                r"^to\s+(?P<predicate>[A-Za-z][A-Za-z-]*)\s*"
                r"(?P<object>.*)$",
                obj,
                flags=re.I,
            )
            if controlled_action:
                inner_predicate = canonical_predicate(
                    controlled_action.group("predicate")
                )
                inner_object = clean_text(controlled_action.group("object"))
                if inner_predicate and inner_predicate != "other":
                    result["predicate"] = inner_predicate
                    obj = inner_object
                    result["role_normalization_status"] = (
                        "CONTROLLED_ACTION_NORMALIZED"
                    )
        if result["predicate"] == "point":
            implication = re.match(
                r"^to\s+(?:an?\s+)?(?:anybody|anyone|somebody|someone|"
                r"individual|person)\s+"
                r"(?:about|concerning|regarding)\s+(?P<context>.+)$",
                obj,
                flags=re.I,
            )
            if implication:
                result["predicate"] = "implicate"
                obj = clean_text(implication.group("context"))
        if result["predicate"] == "lack":
            # ``on/regarding X`` is the topic of the missing evidence, not a
            # dispensable location adjunct.  Keep it in the proposition
            # object so ``lacked evidence on X`` can match ``had no evidence
            # regarding X`` without weakening ordinary location parsing.
            obj = re.sub(
                r"\b(?:regarding|concerning|about|on)\b",
                " ",
                obj,
                flags=re.I,
            )
            obj = clean_text(obj)
        if result["predicate"] in {"lead", "led", "conduct", "conducted"}:
            obj = re.sub(r"^(?:to|the)\s+", "", obj, flags=re.I)
        # Explicit means adjuncts do not change the event patient.  Keeping
        # them in the object made active and passive descriptions of the same
        # event appear structurally different.
        obj = re.split(
            r"\s+by\s+(?:using|throwing|firing|employing|wielding)\b",
            obj,
            maxsplit=1,
            flags=re.I,
        )[0]
        measured_object_match = re.search(
            r"\b(?:packets?|packages?|crates?|containers?|units?|items?)"
            r"\s+of\s+(?P<object>[A-Za-z][A-Za-z0-9'’&/\-]*"
            r"(?:\s+[A-Za-z][A-Za-z0-9'’&/\-]*){0,5}?)"
            r"(?=\s+(?:weighing|weighed|weight|at|in|on|during|"
            r"before|after|that|which)\b|[.,;:]|$)",
            obj,
            flags=re.I,
        )
        result["object"] = canonical_object_head(
            measured_object_match.group("object")
            if measured_object_match else obj[:160]
        )
        if (
            result["predicate"] in {"conduct", "conducted"}
            and result["object"] == "attack"
        ):
            result["predicate"] = "attack"
        result["voice"] = "active"
        if result.get("role_normalization_status") != (
            "CONTROLLED_ACTION_NORMALIZED"
        ):
            result["role_normalization_status"] = "ACTIVE_ROLES_NORMALIZED"
        return result

    bare_predicate_fragment = re.search(
        rf"^(?:(?:{PREDICATE_FRAME_ADVERB_RE})\s+){{0,3}}"
        rf"(?P<predicate>{BARE_FRAGMENT_PREDICATE_RE})\b"
        r"(?:\s+(?P<object>.+))?$",
        source,
        flags=re.I,
    )
    if bare_predicate_fragment:
        obj = clean_text(bare_predicate_fragment.group("object"))
        if obj:
            result["predicate"] = canonical_predicate(
                bare_predicate_fragment.group("predicate")
            )
            result["object"] = canonical_object_head(obj[:160])
            result["patient_or_theme"] = clean_text(obj[:160])
            result["voice"] = "active"
            result["role_normalization_status"] = (
                "BARE_PREDICATE_FRAGMENT_FALLBACK"
            )
            return result

    passive_agent = re.search(
        r"^(?:the\s+)?(?P<object>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,120}?)\s+"
        r"(?P<link>(?:is|are|was|were|remains?|remained|"
        r"has\s+been|have\s+been|had\s+been))\s+"
        r"(?P<verb>made|committed|conducted|carried\s+out|launched)\s+by\s+"
        r"(?P<actor>.+)$",
        source,
        flags=re.I,
    )
    if passive_agent:
        actor = clean_text(passive_agent.group("actor"))
        actor = re.sub(r"[\s,.;:]+$", "", actor)
        obj_surface = clean_text(passive_agent.group("object"))
        predicate = canonical_predicate(obj_surface)
        if actor and predicate and predicate != "be":
            result["actor"] = actor
            result["predicate"] = predicate
            result["object"] = canonical_object_head(obj_surface[:160])
            result["patient_or_theme"] = result["object"]
            result["voice"] = "passive"
            result["role_normalization_status"] = "PASSIVE_AGENT_FALLBACK"
            return result

    modal_copular = re.search(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,90}?)\s+"
        r"[\"']?(?:can|could|may|might|will|would|should|must)\s+"
        r"(?:possibly\s+|probably\s+|likely\s+)?be\s+(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if modal_copular:
        actor = clean_text(modal_copular.group("actor"))
        obj = clean_text(modal_copular.group("object"))
        actor_word_count = len(
            re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", actor)
        )
        if actor and obj and actor_word_count <= 10:
            result.update({
                "actor": actor,
                "predicate": "be",
                "object": canonical_object_head(obj[:160]),
                "patient_or_theme": obj[:160],
                "voice": "active",
                "role_normalization_status": "MODAL_COPULAR_NORMALIZED",
            })
            return result

    copular = re.search(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,90}?)\s+"
        r"(?P<link>(?:is|are|was|were|has\s+been|have\s+been|had\s+been))\s+"
        r"(?P<object>.+)$",
        source,
        flags=re.I,
    )
    if copular:
        actor = clean_text(copular.group("actor"))
        obj = clean_text(copular.group("object"))
        obj = re.sub(
            r"^(?:deemed|considered|regarded\s+as|viewed\s+as)\s+",
            "",
            obj,
            flags=re.I,
        )
        actor_word_count = len(
            re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", actor)
        )
        if actor and obj and actor_word_count <= 10:
            result["actor"] = actor
            result["predicate"] = "be"
            result["object"] = canonical_object_head(obj[:160])
            result["patient_or_theme"] = result["object"]
            result["voice"] = "active"
            result["role_normalization_status"] = "COPULAR_FALLBACK"
            return result

    generic = re.search(
        r"^(?:the\s+)?(?P<actor>[A-Za-z][A-Za-z0-9'().&/,\- ]{1,90}?)\s+(?P<rest>[A-Za-z][A-Za-z0-9'().&/,\- ].+)$",
        source,
        flags=re.I,
    )
    if generic:
        actor = clean_text(generic.group("actor"))
        rest = clean_text(generic.group("rest"))
        tokens = rest.split()
        index = 0
        while index < len(tokens) and clean_text(tokens[index]).lower() in AUXILIARY_VERBS and index < 2:
            index += 1
            while index < len(tokens) and clean_text(tokens[index]).lower() in GENERIC_PREDICATE_ADVERBS:
                index += 1
        if index < len(tokens):
            predicate_tokens: List[str] = []
            head = clean_text(tokens[index]).lower()
            if head in GENERIC_PREDICATE_HEADS or head in PREDICATE_LEMMAS:
                predicate_tokens.append(tokens[index])
                index += 1
                while index < len(tokens) and len(predicate_tokens) < 4:
                    token = clean_text(tokens[index]).lower()
                    if token not in GENERIC_PREDICATE_PARTICLES:
                        break
                    predicate_tokens.append(tokens[index])
                    index += 1
                    if token == "to" and index < len(tokens):
                        predicate_tokens.append(tokens[index])
                        index += 1
                obj = clean_text(" ".join(tokens[index:]))
                obj = re.split(
                    r"\s+by\s+(?:using|throwing|firing|employing|wielding)\b",
                    obj,
                    maxsplit=1,
                    flags=re.I,
                )[0]
                actor_word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", actor))
                if actor and obj and actor_word_count <= 10:
                    result["actor"] = actor
                    result["predicate"] = canonical_predicate(" ".join(predicate_tokens))
                    result["object"] = canonical_object_head(obj[:160])
                    result["patient_or_theme"] = result["object"]
                    result["voice"] = "active"
                    result["role_normalization_status"] = "ACTIVE_GENERIC_FALLBACK"
                    return result

    return result

def build_event_signature(
    text: Any,
    *,
    actor: Any = "",
    predicate: Any = "",
    obj: Any = "",
    location: Any = "",
    event_time: Any = "",
    report_time: Any = "",
    source_attribution: Any = "",
    stance: Any = "",
    modality: Any = "",
    inferred_arguments: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create one canonical cross-task event/proposition signature."""
    proposition = clean_text(text)
    explicit_actor = bool(clean_text(actor))
    explicit_predicate = bool(clean_text(predicate))
    explicit_object = bool(clean_text(obj))
    inferred = (
        dict(inferred_arguments)
        if inferred_arguments is not None
        else infer_simple_clause_arguments(proposition)
    )
    actor = (
        inferred.get("actor", "")
        if inferred.get("voice") == "passive"
        else clean_text(actor) or inferred.get("actor", "")
    )
    predicate_surface = clean_text(predicate) or inferred.get("predicate", "")
    predicate = canonical_predicate(predicate_surface)
    # A small number of relation predicates are expressed as multiword frames
    # whose head alone is semantically empty (for example, ``is close to``).
    # Prefer that explicit frame over a parser-supplied copula, while leaving
    # ordinary clauses untouched.
    for phrase_pattern, phrase_predicate in PREDICATE_PHRASE_LEMMAS:
        if phrase_pattern.search(proposition):
            predicate = phrase_predicate
            break
    obj_surface = (
        inferred.get("patient_or_theme", "")
        if inferred.get("voice") == "passive"
        else clean_text(obj) or inferred.get("object", "")
    )
    if predicate and not obj_surface:
        predicate_forms = [
            surface for surface, lemma in PREDICATE_LEMMAS.items()
            if lemma == predicate
        ]
        if predicate_forms:
            action = re.search(
                r"\b(?:" + "|".join(
                    re.escape(value) for value in sorted(
                        predicate_forms, key=len, reverse=True
                    )
                ) + r")\b(?P<object>.*)$",
                proposition,
                flags=re.I,
            )
            if action:
                obj_surface = clean_text(action.group("object"))
    obj = canonical_object_head(obj_surface)
    if (
        not obj
        and clean_text(inferred.get("role_normalization_status", ""))
        == "PASSIVE_REPORT_NORMALIZED"
    ):
        # Introductory report objects such as "in private talks" begin with a
        # preposition and are intentionally trimmed by canonical_object_head.
        # They are still essential for distinguishing one attributed report
        # from every other statement by the same reporter.
        obj = canonical_key(obj_surface)
    if predicate == "affiliation":
        affiliation = re.search(
            r"\b(?:agent\s+of|affiliated\s+with|associated\s+with)\s+"
            r"(?P<object>.+?)[.!?]?$",
            proposition,
            flags=re.I,
        )
        if affiliation:
            obj_surface = clean_text(affiliation.group("object"))
            obj = canonical_object_head(obj_surface)
    if predicate == "remember":
        memory = re.search(
            r"\b(?:no\s+memory\s+of|not\s+remember)\s+"
            r"(?P<object>.+?)[.!?]?$",
            proposition,
            flags=re.I,
        )
        if memory:
            obj_surface = clean_text(memory.group("object"))
            obj = canonical_object_head(obj_surface)
    location = clean_text(location) or inferred.get("location", "")
    # Numeric arguments are clause-local.  ``event_time`` is a typed temporal
    # field, not a container for surrounding prose; joining it here allowed
    # quantities from one sibling clause to leak into every proposition.
    numeric = normalize_numeric_arguments(proposition)
    date = normalize_date(event_time or proposition)
    attendance_state = infer_attendance_state(proposition)
    disclosure_state = infer_disclosure_state(proposition)
    evaluative_state = infer_evaluative_state(proposition)
    vehicle_identity = infer_vehicle_identity(proposition)
    if attendance_state != "unknown":
        # Keep the established field name for parser/cache compatibility; the
        # value has always represented generic participation as well as literal
        # meeting attendance.
        state_variable = "attendance_status"
        state_value = attendance_state
    elif disclosure_state != "unknown":
        state_variable = "disclosure_status"
        state_value = disclosure_state
    elif evaluative_state != "unknown":
        state_variable = "evaluative_status"
        state_value = evaluative_state
    else:
        state_variable = ""
        state_value = ""
    reporter = ""
    embedded_actor = actor
    reporting = (
        None
        if clean_text(inferred.get("role_normalization_status", ""))
        == "PASSIVE_REPORT_NORMALIZED"
        else _reporting_clause_parts(proposition)
    )
    if reporting:
        reporter = clean_text(reporting["reporter"])
        nested = infer_simple_clause_arguments(reporting["embedded"])
        embedded_actor = clean_text(nested.get("actor", "")) or embedded_actor
        if nested.get("predicate") and not explicit_predicate:
            predicate = canonical_predicate(nested["predicate"])
        if nested.get("object") and not explicit_object:
            obj = canonical_object_head(nested["object"])
    if attendance_state != "unknown":
        participation_scope = re.search(
            r"\b(?:"
            r"(?:had|has|have|played|took)\s+no\s+part\s+in|"
            r"den(?:y|ies|ied|ying)\s+(?:any\s+)?involvement\s+in|"
            r"(?:did|does|do|is|are|was|were)\s+not\s+involved\s+in"
            r")\s+(?P<event>[^.;]+)",
            proposition,
            flags=re.I,
        )
        if participation_scope:
            scope = clean_text(participation_scope.group("event"))
            if scope:
                obj_surface = scope
                scope_for_identity = re.sub(
                    r"\b(?:regarding|concerning|about)\b",
                    "on",
                    scope,
                    flags=re.I,
                )
                obj = canonical_object_head(scope_for_identity)
        if reporting and not explicit_actor:
            embedded_surface = clean_text(reporting["embedded"])
            if re.match(
                r"^(?:that\s+)?(?:he|she|they|it)\b",
                embedded_surface,
                flags=re.I,
            ):
                # In a direct "X stated that he/she ..." participation
                # assertion, the reporting subject is the only explicit
                # antecedent.  Preserve the reporter separately while using
                # it as the semantic participant for cross-surface matching.
                actor = reporter
                embedded_actor = reporter
        participant = re.search(
            r"\b(?:no|without|a|an|the)\s+"
            r"(?P<participant>[A-Za-z][A-Za-z0-9'’&/\-]+"
            r"(?:\s+[A-Za-z][A-Za-z0-9'’&/\-]+){0,4}?)"
            r"\s+(?:delegation|representative|participant|attendee)s?\b",
            proposition,
            flags=re.I,
        ) or re.search(
            r"\b(?P<participant>[A-Z][A-Za-z0-9'’&/\-]+"
            r"(?:\s+[A-Z][A-Za-z0-9'’&/\-]+){0,5})"
            r"\s+(?i:delegation|representative|participant|attendee)s?\b",
            proposition,
        )
        if participant:
            embedded_actor = clean_text(participant.group("participant"))
            # For an atomic attendance proposition the participant is the
            # semantic actor even when the surface clause uses an existential
            # form ("there would be no ... delegation").  Keeping actor blank
            # forces downstream engines to depend on a secondary field and
            # made equivalent support/update propositions compare unevenly.
            # This signature represents the embedded participation assertion;
            # the outer reporter remains separately available in ``reporter``.
            # Combining both into one actor creates an impossible conjunctive
            # identity and breaks equivalent nested/direct reports.
            actor = embedded_actor
        if not explicit_predicate or predicate in {"attend", "participate"}:
            predicate = "attend"
        month_names = "|".join(
            sorted(
                set(MONTHS) | set(MONTH_ALIASES),
                key=len,
                reverse=True,
            )
        )
        event_match = re.search(
            r"\b(?:at|for)\s+(?:the\s+)?"
            r"(?P<event>[A-Za-z][A-Za-z0-9'’&/\- ]{2,100}?)"
            r"(?=\s+(?:on\s+)?(?:" + month_names + r")\.?\s+\d"
            r"|\s+(?:because|due\s+to)\b|[.;,]|$)",
            proposition,
            flags=re.I,
        )
        if event_match or not obj:
            obj = canonical_object_head(
                event_match.group("event") if event_match else "event"
            )
    return {
        "actor": clean_text(actor),
        "agent": clean_text(actor),
        "patient_or_theme": clean_text(
            inferred.get("patient_or_theme", "")
        ),
        "recipient": clean_text(inferred.get("recipient", "")),
        "destination": clean_text(inferred.get("destination", "")),
        "voice": clean_text(inferred.get("voice", "unknown")),
        "role_normalization_status": clean_text(
            inferred.get("role_normalization_status", "NOT_APPLICABLE")
        ),
        "reporter": reporter,
        "embedded_actor": embedded_actor,
        "predicate": clean_text(predicate),
        "predicate_surface": predicate_surface,
        "object": clean_text(obj),
        "object_surface": obj_surface,
        "location": clean_text(location),
        "location_identity": canonical_location(location),
        "vehicle_identity": canonical_key(vehicle_identity),
        "vehicle_surface": vehicle_identity,
        # A typed temporal field may contain only a normalized date.  The
        # surrounding prose remains available in ``text`` and must not leak
        # into event_time when no date can be parsed.
        "event_time": date["normalized"],
        "event_time_precision": date["precision"] if date["normalized"] else "",
        "report_time": normalize_date(report_time)["normalized"] or clean_text(report_time),
        "source_attribution": clean_text(source_attribution),
        "stance": clean_text(stance),
        "modality": clean_text(modality),
        "assertion_polarity": infer_assertion_polarity(proposition),
        "epistemic_state": infer_epistemic_state(proposition),
        "semantic_state_variable": state_variable,
        "semantic_state_value": state_value,
        "semantic_tokens": sorted(semantic_tokens(proposition)),
        **numeric,
    }


def compare_argument(query_value: Any, candidate_value: Any, *, applicable: bool = True) -> str:
    """Return a Requirements v1.9 field-comparison state."""
    if not applicable:
        return "NOT_APPLICABLE"
    q = clean_text(query_value)
    c = clean_text(candidate_value)
    if not q and not c:
        return "NOT_APPLICABLE"
    if not q:
        return "MISSING_IN_QUERY"
    if not c:
        return "MISSING_IN_CANDIDATE"
    q_predicate, c_predicate = canonical_predicate(q), canonical_predicate(c)
    if q_predicate and _predicate_family_equivalent(q_predicate, c_predicate):
        return "MATCH"
    q_location, c_location = canonical_location(q), canonical_location(c)
    if q_location and q_location == c_location:
        return "MATCH"
    q_object, c_object = canonical_object_head(q), canonical_object_head(c)
    if q_object and q_object == c_object:
        return "MATCH"
    if _short_alias_token_match(q, c):
        return "MATCH"
    if _near_equivalent_argument(q, c):
        return "MATCH"
    return "MATCH" if canonical_key(q) == canonical_key(c) else "MISMATCH"


def compare_dates(query_value: Any, candidate_value: Any) -> str:
    """Compare partial dates while preserving missing/unparseable states."""
    q = normalize_date(query_value)
    c = normalize_date(candidate_value)
    if not clean_text(query_value) and not clean_text(candidate_value):
        return "NOT_APPLICABLE"
    if not clean_text(query_value):
        return "MISSING_IN_QUERY"
    if not clean_text(candidate_value):
        return "MISSING_IN_CANDIDATE"
    if not q["normalized"] or not c["normalized"]:
        return "UNPARSEABLE"
    q_tail = q["normalized"][-5:]
    c_tail = c["normalized"][-5:]
    return "MATCH" if q_tail == c_tail else "MISMATCH"


def compare_numeric(query_value: Any, candidate_value: Any, tolerance: float = 1e-6) -> str:
    if query_value is None and candidate_value is None:
        return "NOT_APPLICABLE"
    if query_value is None:
        return "MISSING_IN_QUERY"
    if candidate_value is None:
        return "MISSING_IN_CANDIDATE"
    try:
        return "MATCH" if abs(float(query_value) - float(candidate_value)) <= tolerance else "MISMATCH"
    except (TypeError, ValueError):
        return "UNPARSEABLE"


def schema_dataframe(rows: Any, columns: Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame with required columns first and all extras retained."""
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows or [])
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    extras = [column for column in frame.columns if column not in columns]
    return frame[list(columns) + extras]


def write_csv_with_schema(path: Path, rows: Any, columns: Sequence[str]) -> pd.DataFrame:
    """Write a schema-valid CSV, including a header for zero-row outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = schema_dataframe(rows, columns)
    frame.to_csv(path, index=False)
    return frame


def strict_json_value(value: Any) -> Any:
    """Return an interoperable JSON value with non-finite numbers as null.

    Python's default JSON encoder emits NaN and Infinity even though neither is
    valid JSON.  Matryoshka output is consumed outside Python, so every JSON
    boundary uses this recursive normalizer and the encoder's strict mode.
    """
    if isinstance(value, Mapping):
        return {
            str(key): strict_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # NumPy and pandas scalar values expose ``item``.  Convert them without
    # importing either package at this low-level serialization boundary.
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return strict_json_value(converted)
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict JSON via a temporary sibling and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False, prefix=path.name + ".tmp.") as handle:
        json.dump(
            strict_json_value(payload),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        temp_name = handle.name
    os.replace(temp_name, path)


@contextlib.contextmanager
def file_lock(lock_path: Path, timeout_seconds: float = 300.0, poll_seconds: float = 0.25) -> Iterator[None]:
    """Simple cross-process lock using exclusive lock-file creation.

    It is intentionally conservative and adequate for a local analyst
    workstation.  The lock file records PID and creation time for diagnostics.
    """
    deadline = time.monotonic() + timeout_seconds
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "created_at": utc_now_iso()}, handle)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def corpus_fingerprint(paths: Sequence[Path], extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build a cache fingerprint from source file hashes and schema versions."""
    files = {
        str(Path(path).name): {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(Path(path)),
            "size_bytes": Path(path).stat().st_size if Path(path).exists() else None,
        }
        for path in paths
    }
    payload = {"files": files, "extra": dict(extra or {})}
    return {**payload, "fingerprint_sha256": stable_hash(payload), "created_at": utc_now_iso()}


@dataclass
class StageResult:
    """Machine-readable result for staged release orchestration."""

    stage: str
    status: str
    started_at: str
    finished_at: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
