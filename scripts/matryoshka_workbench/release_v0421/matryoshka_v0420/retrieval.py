"""Bounded document retrieval over a pre-tokenized immutable index."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .corpus import CorpusIndex
from .semantics import PREDICATE_LEMMAS, semantic_tokens
from .semantic_resources import SemanticResources


ALIASES = {
    "seized": "seize", "seizure": "seize", "confiscated": "seize",
    "reported": "report", "informed": "report",
    "attended": "attend", "attendance": "attend", "represented": "attend",
    "constitute": "be", "constitutes": "be", "constituted": "be",
    "constituting": "be",
    "acquired": "acquire", "acquire": "acquire", "acquires": "acquire",
    "acquiring": "acquire", "obtain": "acquire", "obtained": "acquire",
    "obtains": "acquire", "obtaining": "acquire",
    "buy": "acquire", "buys": "acquire", "bought": "acquire",
    "buying": "acquire", "purchase": "acquire", "purchases": "acquire",
    "purchased": "acquire", "purchasing": "acquire",
    "shop": "acquire", "shops": "acquire", "shopping": "acquire",
    "procure": "acquire", "procures": "acquire", "procured": "acquire",
    "procurement": "acquire",
    "assistance": "support", "supports": "support",
    "finance": "fund", "financed": "fund", "finances": "fund",
    "financing": "fund", "financial": "fund", "funding": "fund",
    "funds": "fund", "money": "fund",
    "bound": "bound", "destined": "bound", "headed": "bound",
    "components": "part", "component": "part", "parts": "part",
    "firms": "company", "firm": "company", "companies": "company",
    "citizen": "national", "citizens": "national",
    "deliberation": "deliberation", "deliberations": "deliberation",
    "discussion": "deliberation", "discussions": "deliberation",
    "document": "paper", "documents": "paper",
    "equitable": "fair", "fair": "fair",
    "expertise": "talent", "skill": "talent", "skills": "talent",
    "lab": "laboratory", "labs": "laboratory",
    "possibility": "way", "possibilities": "way", "way": "way", "ways": "way",
    "agreement": "pact", "agreements": "pact",
    "produce": "develop", "produces": "develop", "produced": "develop",
    "producing": "develop", "production": "develop", "development": "develop",
    "holds": "hold", "holding": "hold",
    "prepare": "prepare", "prepared": "prepare",
    "requirement": "requirement", "requirements": "requirement",
    "resolution": "solution", "resolutions": "solution",
    "retain": "hold", "retained": "hold", "retains": "hold",
    "standard": "requirement", "standards": "requirement",
    "stringent": "strict",
    "toward": "bound", "towards": "bound",
    "uk": "britain", "british": "britain",
    "american": "unitedstates", "americans": "unitedstates",
    "chinese": "china", "cuban": "cuba", "french": "france",
    "indian": "india", "indians": "india",
    "iranian": "iran", "iranians": "iran",
    "pakistani": "pakistan", "pakistanis": "pakistan",
    "russian": "russia", "russians": "russia",
    "vietnamese": "vietnam",
    "vessel": "ship", "vessels": "ship",
    "countries": "country", "nations": "country",
    "acknowledge": "recognize", "acknowledges": "recognize",
    "acknowledged": "recognize", "acknowledging": "recognize",
}
ALIASES.update({
    token: lemma
    for token, lemma in PREDICATE_LEMMAS.items()
    if lemma and token != lemma
})
# Retrieval treats acquisition verbs as one broad recall family. Task engines
# still retain their stricter predicate, completion, modality, and polarity
# gates, so this cannot by itself certify support or contradiction.
ALIASES.update({
    "acquired": "acquire", "acquire": "acquire", "acquires": "acquire",
    "acquiring": "acquire", "obtain": "acquire", "obtained": "acquire",
    "obtains": "acquire", "obtaining": "acquire",
    "buy": "acquire", "buys": "acquire", "bought": "acquire",
    "buying": "acquire", "purchase": "acquire", "purchases": "acquire",
    "purchased": "acquire", "purchasing": "acquire",
    "shop": "acquire", "shops": "acquire", "shopping": "acquire",
    "procure": "acquire", "procures": "acquire", "procured": "acquire",
    "procurement": "acquire",
})
# Conservative cross-part-of-speech families used by document retrieval.  The
# matcher still applies task-specific event, role, polarity, and margin gates;
# these aliases only keep a semantically equivalent document in the shortlist.
ALIASES.update({
    "affect": "effect", "affected": "effect", "affecting": "effect",
    "cause": "effect", "caused": "effect", "causing": "effect",
    "impact": "effect", "impacted": "effect", "impacting": "effect",
    "trouble": "effect", "troubles": "effect",
    "discontent": "unrest", "dissatisfaction": "unrest",
    "divide": "divide", "divided": "divide", "divides": "divide",
    "dividing": "divide", "division": "divide", "divisions": "divide",
    "split": "divide", "splits": "divide", "splitting": "divide",
    "ineffective": "ineffective", "ineffectual": "ineffective",
    "useless": "ineffective", "worthless": "ineffective",
    "grow": "increase", "grew": "increase", "growing": "increase",
    "increased": "increase", "increases": "increase",
    "increasing": "increase",
    "note": "report", "noted": "report", "notes": "report",
    "observe": "report", "observed": "report", "observes": "report",
    "observing": "report", "observation": "report",
    "observations": "report",
    "conjecture": "speculation", "speculative": "speculation",
    "perspective": "view", "perspectives": "view",
    "outreach": "approach", "approaches": "approach",
    "postpone": "delay", "postponed": "delay", "postponing": "delay",
    "postponement": "delay", "delayed": "delay", "delaying": "delay",
    "delays": "delay",
})


def canonical_tokens(values: Iterable[str]) -> Set[str]:
    normalized: Set[str] = set()
    for value in values:
        token = str(value)
        # Runtime semantic-resource concepts such as ``alias:soviet_union``
        # are already canonical and must never be lemmatized away.
        if ":" in token:
            normalized.add(token)
        else:
            normalized.add(ALIASES.get(token, token))
    return normalized


def semantic_token_set(
    value: Any,
    resources: SemanticResources | None = None,
) -> Set[str]:
    if resources is None:
        return set(semantic_tokens(value))
    return set(resources.expanded_tokens(value))


def containment(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a | b))


def score_text(
    query: str,
    candidate: str,
    resources: SemanticResources | None = None,
) -> float:
    query_tokens = canonical_tokens(semantic_token_set(query, resources))
    candidate_tokens = canonical_tokens(
        semantic_token_set(candidate, resources)
    )
    return score_token_sets(query_tokens, candidate_tokens)


def score_token_sets(
    query_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
) -> float:
    normalized_query = canonical_tokens(query_tokens)
    normalized_candidate = canonical_tokens(candidate_tokens)
    return 0.62 * containment(
        normalized_query, normalized_candidate
    ) + 0.38 * jaccard(normalized_query, normalized_candidate)


def _corpus_canonical_document_tokens(
    corpus: CorpusIndex,
) -> Mapping[str, Set[str]]:
    cached = getattr(corpus, "_canonical_document_tokens_cache", None)
    cached_fingerprint = getattr(
        corpus,
        "_canonical_document_tokens_cache_fingerprint",
        "",
    )
    if cached is not None and cached_fingerprint == corpus.fingerprint:
        return cached
    canonicalized: Dict[str, Set[str]] = {
        doc_id: canonical_tokens(tokens)
        for doc_id, tokens in corpus.document_tokens.items()
    }
    object.__setattr__(
        corpus,
        "_canonical_document_tokens_cache",
        canonicalized,
    )
    object.__setattr__(
        corpus,
        "_canonical_document_tokens_cache_fingerprint",
        corpus.fingerprint,
    )
    return canonicalized


def _corpus_retrieval_views(
    corpus: CorpusIndex,
) -> tuple[
    Mapping[str, Set[str]],
    Mapping[str, Tuple[str, ...]],
    Mapping[str, int],
]:
    cached_tokens = getattr(corpus, "_canonical_document_tokens_cache", None)
    cached_postings = getattr(corpus, "_canonical_document_postings_cache", None)
    cached_sizes = getattr(corpus, "_canonical_document_token_count_cache", None)
    cached_fingerprint = getattr(
        corpus,
        "_canonical_document_tokens_cache_fingerprint",
        "",
    )
    if (
        cached_tokens is not None
        and cached_postings is not None
        and cached_sizes is not None
        and cached_fingerprint == corpus.fingerprint
    ):
        return cached_tokens, cached_postings, cached_sizes
    canonicalized = _corpus_canonical_document_tokens(corpus)
    postings: Dict[str, List[str]] = defaultdict(list)
    sizes: Dict[str, int] = {}
    for doc_id, tokens in canonicalized.items():
        sizes[doc_id] = len(tokens)
        for token in tokens:
            postings[token].append(doc_id)
    frozen_postings = {
        token: tuple(sorted(doc_ids))
        for token, doc_ids in postings.items()
    }
    object.__setattr__(
        corpus,
        "_canonical_document_postings_cache",
        frozen_postings,
    )
    object.__setattr__(
        corpus,
        "_canonical_document_token_count_cache",
        sizes,
    )
    return canonicalized, frozen_postings, sizes


def rank_documents(
    query: str,
    corpus: CorpusIndex,
    limit: int,
    resources: SemanticResources | None = None,
) -> List[Tuple[str, float]]:
    query_tokens = canonical_tokens(semantic_token_set(query, resources))
    return rank_documents_tokens(query_tokens, corpus, limit)


def rank_documents_tokens(
    query_tokens: Iterable[str],
    corpus: CorpusIndex,
    limit: int,
) -> List[Tuple[str, float]]:
    query_tokens = canonical_tokens(query_tokens)
    document_tokens, postings, document_sizes = _corpus_retrieval_views(corpus)
    overlap_counts: Dict[str, int] = defaultdict(int)
    for token in query_tokens:
        for doc_id in postings.get(token, ()):
            overlap_counts[doc_id] += 1
    query_size = max(1, len(query_tokens))
    scores: List[Tuple[str, float]] = []
    for doc_id, overlap in overlap_counts.items():
        candidate_size = max(1, document_sizes.get(doc_id, 0))
        containment_score = overlap / query_size
        union_size = max(1, query_size + candidate_size - overlap)
        jaccard_score = overlap / union_size
        score = 0.62 * containment_score + 0.38 * jaccard_score
        scores.append((doc_id, float(score)))
    limit = max(1, limit)
    if len(scores) < limit:
        for doc_id in sorted(document_tokens):
            if doc_id in overlap_counts:
                continue
            scores.append((doc_id, 0.0))
            if len(scores) >= limit:
                break
    return sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]
