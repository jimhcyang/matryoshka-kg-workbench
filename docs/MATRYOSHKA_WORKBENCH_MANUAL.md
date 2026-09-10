# Matryoshka Analyst Workbench Manual

**Canonical handoff manual**  
**Updated: September 1, 2026**

## 1. Purpose

The Matryoshka workbench turns a canonicalized corpus extraction into a
controlled retrieval-and-reasoning benchmark. It supports five task families:

1. reconstruction;
2. claims;
3. value redaction;
4. sentence redaction; and
5. contradictions.

The shipped final experiment compares two query surfaces:

- the original benchmark query; and
- a one-shot semantic rewrite in concise analyst-note language.

Both surfaces run in two corpus conditions:

- **baseline**: the complete configured corpus is available; and
- **holdout**: the query's known source document is excluded.

The resulting four views answer different questions. Baseline measures whether
the system can retrieve and reason over available evidence. Holdout measures
whether independent corpus evidence can sustain the same conclusion. Original
versus rewritten measures robustness to realistic lexical and structural drift.

This manual explains the complete workflow, including intermediary objects,
alias-aware retrieval, evidence tables, task gates, truth accounting, caching,
reports, Streamlit, and debugging.

## 2. Supported Workflow

The public final entry point is:

```text
scripts/matryoshka_workbench/tools/run_rephrase_ab_benchmark.py
```

It orchestrates:

```text
build original benchmark
  -> clone benchmark metadata
  -> rewrite selected queries once
  -> run original baseline + holdout
  -> run rewritten baseline + holdout
  -> summarize each branch
  -> create original-vs-rewritten comparison
```

The package intentionally does not ship alternative triplet, lexical-rewrite,
staged-query, or CALIBER bridge workflows. Lower-level supported tools are:

- `multicorpus_benchmark.py` for build/run/all without query rewriting;
- `summarize_benchmark_results.py` for rebuilding analyst reports; and
- the Streamlit application for interactive inspection.

All paths resolve from the package root, the directory containing the top-level
`README.md`. The package can be renamed or moved.

## 3. Environment And API Boundaries

Install from the package root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

OpenAI has three narrowly defined roles in the final A/B workflow:

1. **Runtime query parser**: enabled with `--use-openai-parser`; called once for
   each unique selected query surface, with a persistent query-parse cache.
2. **Semantic rewriter**: called once for each selected original query.
3. **Rewrite judge**: called once for each rewritten query to record whether
   target, intent, markers, and critical anchors appear preserved.

The rewrite judge is diagnostic. It never rejects a rewrite, retries it, or
replaces it with the original. The actual first-shot rewritten query always
runs. Local repair may restore malformed redaction markers or critical
structural anchors, but there is no model-generation retry loop.

API transport libraries may retry transient connection errors. Those network
retries are different from semantic rewrite retries; they attempt to obtain the
same one-shot model response after a failed transport.

**Build-stage catalog generation does not call OpenAI.** Even when
`--use-openai-parser` appears on the final command, candidate generation,
filtering, ranking, and selection remain local. Only the final selected queries
are parsed with OpenAI during runtime.

## 4. Corpora And Configuration

The default order is always:

```text
VIET, CIB, IRAN, FISC, CANB
```

The base configuration is:

```text
scripts/matryoshka_workbench/config/benchmark.default.json
```

Corpus profiles are:

```text
scripts/matryoshka_workbench/config/corpus_profiles/vietnam.json
scripts/matryoshka_workbench/config/corpus_profiles/cib.json
scripts/matryoshka_workbench/config/corpus_profiles/iran.json
scripts/matryoshka_workbench/config/corpus_profiles/fisc.json
scripts/matryoshka_workbench/config/corpus_profiles/canberra.json
```

Each profile identifies:

- the canonicalized extraction root;
- the runtime engine root;
- the decision profile;
- a fallback corpus entity resource;
- corpus-specific sampling overrides; and
- readiness status.

At runtime, the loader first looks for the current KG-generated resource next
to the extraction:

```text
output/knowledge_graph/<project>/alias/corpus_entity_resource_v1.json
```

If that file exists, it takes precedence over the fallback profile resource.
The loader normalizes the resource into `data/resources/_runtime_normalized/`
and fingerprints its content. This ensures the workbench uses the aliases
created by the current KG alias stage rather than an older static list.

## 5. Benchmark Build

### 5.1 Build inputs

Build reads `extraction_canonicalized/`, including:

- `chunks.csv`;
- `entities.csv`;
- `events.csv`;
- `claims.csv`;
- `relations.csv`; and
- the `kg_pseudo_chunks_*.csv` task views.

It also reads corpus profile overrides and the configured random seed.

### 5.2 Candidate catalogs

For each corpus and task, the builder attempts to form a ranked catalog of
admissible query cases. The default uniform catalog ceiling is 400. This means:

```text
generate and rank up to 400 admissible catalog cases, or retain all available
cases when fewer than 400 survive
```

It does **not** mean that 400 OpenAI calls occur. Build is local, and generators
often inspect more raw rows than the ceiling before stopping.

The ceiling is configurable through `catalog_target_uniform`. It is a workload
and audit frontier, not a claim that every corpus contains exactly 400 equally
good tasks.

### 5.3 Why target, admissible, and selected differ

A terminal line such as:

```text
value_redaction: admissible 16 from target 400; selected 6
```

has three distinct quantities:

- **target 400**: the catalog search ceiling;
- **admissible 16**: unique cases that survived task-specific structural,
  context, label, ambiguity, source, and preflight checks;
- **selected 6**: cases retained after ranking, deterministic seed ordering,
  document-diversity limits, duplicate suppression, and final selected-slice
  validation.

An admissible case can be omitted because it duplicates a stronger query,
exceeds `max_cases_per_doc`, conflicts with a selected answer/slot, has a weak
selected-slice parser representation, or ranks below the requested top-k after
diversity. Underfill is visible and is never silently padded with arbitrary bad
queries.

### 5.4 Seed behavior

The seed controls deterministic row ordering and tie/diversity selection. It
does not randomize model output. Given the same source tables, code, config,
seed, and OpenAI responses for selected query parsing/rewriting, the benchmark
plan is reproducible.

The final shipped default is seed 42. Changing the seed provides a useful
anti-overfitting audit because it changes which close-scoring candidates reach
the selected set.

### 5.5 Task-specific generation

Each task has an independent collector, policy, backfill, ranking, and selected
slice. A change to value-redaction selection should not alter claims or
contradictions selection.

#### Reconstruction generation

Reconstruction queries are bounded source excerpts enriched by structured
anchors. Quality checks require substantive context, distinctive anchors,
reasonable length, and source provenance. The generator favors evidence
neighborhoods that can be reconstructed from one or a coherent set of corpus
records rather than arbitrary paragraphs.

#### Claims generation

Claims are formed from structured `claims.csv` and claim-oriented pseudo views.
The generator requires an executable assertion surface, a defensible predicate
frame, useful actor/object roles, and enough context for retrieval. It removes
routing lines, relation-only labels, fragments, boilerplate, and generic role
surfaces.

#### Value-redaction generation

Value redaction chooses an extracted entity mention and replaces its exact
surface with a typed marker:

```text
[REDACT:R1 type=LOCATION]
```

The visible query is built from a real processed source context containing the
entity, not from a fabricated claim label. The generator prioritizes, in order:

1. a clean extracted claim containing the entity;
2. a clean event context containing the entity;
3. entity context/window text;
4. a focused sentence neighborhood; and
5. a bounded chunk excerpt as fallback.

Large-corpus generation avoids repeatedly cleaning full chunk text. It uses
entity-backed Stage C rows and compact claim/event/context excerpts first. The
selected query still has source provenance and the marker replaces a verbatim
entity mention in the chosen processed context.

Multi-slot queries are allowed when a single coherent sentence contains more
than one independently supported entity and ambiguity remains manageable.

#### Sentence-redaction generation

Sentence redaction removes one complete sentence and retains visible sentences
before and after it:

```text
Visible sentence before. [REDACT_SENT:S1 approx_tokens=19] Visible sentence after.
```

It requires substantive left and right context, a target sentence of acceptable
length, and source provenance. This differs from value redaction: the missing
object is a sentence/proposition, not a typed entity slot.

#### Contradiction generation

Contradiction queries are structured assertions selected to support meaningful
opposition analysis. Candidate strategy emphasizes numeric, date, and negation
changes with actor, event, scope, polarity, and uncertainty accounting. The
generator avoids manufacturing a contradiction from unrelated events that only
share vocabulary.

### 5.6 Build outputs

For experiment `<name>`, build writes:

```text
output/matryoshka_workbench/benchmarks/<name>/
|-- configs/resolved_multicorpus_config.json
|-- query_files/<task>/<corpus>/<case>.txt
|-- query_manifests/benchmark_plan.json
|-- query_manifests/case_manifest.csv
|-- query_manifests/query_answer_key.csv
`-- summaries/
    |-- final_query_catalog.csv
    |-- build_stage_counts.csv
    `-- build_diagnostics.json
```

`final_query_catalog.csv` is the single canonical build catalog. It includes
selected and non-selected retained catalog rows, ranks, source document,
query text, ground truth, source context, and notes. There is no duplicate
minimal catalog.

The semantic clone keeps its rewritten `final_query_catalog.csv` but does not
copy `build_stage_counts.csv` or `build_diagnostics.json`: selection happened
once, in the original branch, so a second identical copy would be misleading
clutter. The semantic branch instead adds `query_rephrase_audit.csv` and
`query_rephrase_manifest.json` under `summaries/`.

## 6. Semantic Rewrite Branch

The A/B wrapper clones the built original experiment without copying run
outputs. It then rewrites each selected query exactly once with the configured
OpenAI model.

### 6.1 Rewrite objective

The rewrite is not clause-by-clause synonym replacement. It compresses the
query into neutral intelligence/historical note style while preserving the
same target and evidence neighborhood.

Example:

```text
Original:
Razak was realistic regarding neutrality as a very long range goal.

Semantic analyst rewrite:
Razak viewed neutrality as a distant goal.
```

For claim-like tasks, the prompt preserves:

- named actor;
- an explicit predicate frame;
- one concrete object anchor;
- polarity and negation;
- modality, intent, denial, and uncertainty;
- exact dates and quantities; and
- one rare/technical reconstruction anchor when needed.

For redaction tasks, marker strings are restored exactly and the nearest
left/right slot context is protected.

### 6.2 Alias substitutions

The rewriter may use a common unambiguous alternate entity surface, such as
`Soviet Union` for `USSR`. It is encouraged but not forced. It must not replace
an entity with a related but different institution, geography, or historical
regime.

### 6.3 One-shot policy

The policy is:

```text
one model generation
  -> deterministic layout/marker/anchor repair
  -> local validation metrics
  -> one diagnostic judge call
  -> always execute the resulting rewrite
```

There is no semantic retry and no original-query fallback. An unchanged or
weak rewrite remains visible in the audit rather than being hidden.

### 6.4 Rewrite diagnostics

`query_rephrase_audit.csv` records:

- original, raw rewritten, and final rewritten query;
- whether local repair changed the raw response;
- lexical reuse ratio and Jaccard;
- surface similarity and distance;
- clause-anchor recall;
- marker validation;
- judge target/intent/anchor findings;
- judge confidence and issues; and
- whether the rewrite changed.

The judge does not contribute to benchmark PASS/FAIL. It contextualizes the
difficulty and validity of the A/B perturbation.

## 7. Runtime Query Parsing

Each selected query is parsed into a run-local `query_parse/` directory.
OpenAI parsing and deterministic parsing are reconciled rather than allowing a
single unverified narrative response to drive the matcher.

### 7.1 Principal parser files

| File | Contents |
| --- | --- |
| `query_fragments.csv` | executable query units |
| `query_entities.csv` | entity surfaces, types, roles, canonical keys |
| `query_claims.csv` | assertion text, speaker, target, stance |
| `query_events.csv` | event frames, participants, place, time |
| `query_relations.csv` | typed query-side relations |
| `query_propositions.csv` | normalized proposition hypotheses and signatures |
| `query_redaction_slots.csv` | value-redaction marker/type/context records |
| `query_sentence_redaction_slots.csv` | sentence-gap records |
| `query_task_preflight.csv` | task executability checks |
| `query_parse_manifest.json` | parser mode, fingerprints, and outputs |

The parser emits more files than the three most visible tables because
different tasks need different intermediary objects. Claims and contradictions
center on `query_propositions.csv`; value redaction centers on slot records plus
entity/event/proposition context; sentence redaction centers on sentence-gap
context; reconstruction can use entities, events, claims, relations, and
propositions together.

### 7.2 Concrete parser example

An actual Vietnam claims query is:

```text
The Republic of Vietnam is willing to seek a mutually acceptable solution to
the issue through friendly discussions and negotiations.
```

OpenAI parsing produced an entity row similar to:

```text
label: Republic of Vietnam
canonical_key: republic of vietnam
type: COUNTRY
role_or_context: Negotiating party
parser_source: openai
```

It produced a claim row:

```text
speaker: Republic of Vietnam
target: the issue
stance: affirmed
claim_text: the full query assertion
```

The proposition row retained:

```text
canonical_subject: Republic of Vietnam
canonical_predicate: be
canonical_object: willing to seek mutually acceptable solution to issue
                  through friendly discussions and negotiations
polarity: positive
modality: explicit
stance: affirmed
```

The proposition is not merely a bag of words. It carries actor, predicate,
object, polarity, modality, stance, event/date fields, and alternate
hypotheses where parsing is ambiguous.

## 8. Alias-Aware Matching

### 8.1 Query-side and corpus-side expansion

The runtime loads core semantic resources plus the corpus entity resource.
Every accepted alias group is assigned a concept name such as:

```text
corpus_entity:CANONICAL_00031
```

When a known phrase appears in text, semantic expansion adds a concept token:

```text
alias:corpus_entity:CANONICAL_00031
```

The same expansion function runs on query text and corpus evidence surfaces.
The system does not need to print every alias in every pseudo chunk. The query
and candidate independently emit the same concept token from the loaded alias
resource.

### 8.2 USSR example

Suppose the corpus alias map contains:

```text
USSR                                -> CANONICAL_00031 / SOVIET UNION
SOVIET UNION                        -> CANONICAL_00031 / SOVIET UNION
UNION OF SOVIET SOCIALIST REPUBLICS -> CANONICAL_00031 / SOVIET UNION
```

An analyst enters:

```text
The Union of Soviet Socialist Republics supported the proposal.
```

The parser records the long-form entity. Semantic expansion detects that phrase
and adds the canonical alias concept. A corpus claim containing `USSR` and a
pseudo chunk displaying `SOVIET UNION` independently add the same concept.
They can therefore overlap during document retrieval and evidence scoring even
though literal entity words differ.

Canonical fields in pseudo chunks improve auditability, but the decisive
mechanism is shared resource expansion on both sides.

### 8.3 What alias-off means

With corpus alias expansion disabled, the system still has normalization,
predicate lemmas, core semantic groups, and ordinary lexical overlap. Many
verbatim or near-verbatim benchmark queries can still pass. Alias-off therefore
does not mean the matcher is nonfunctional; it means cross-surface entity recall
is weaker and depends more on other shared words.

Alias expansion is used by all five tasks at document retrieval and semantic
scoring. Contradictions are not the only consumer.

## 9. CorpusIndex And Evidence Preparation

### 9.1 Single-load design

Task engines do not repeatedly open CSV files. `CorpusIndex` loads the
task-relevant tables, normalizes them, builds proposition and provenance
records, tokenizes document surfaces, and freezes indexes once per
corpus/task/fingerprint.

It creates query-independent intermediary structures such as:

- source table frames;
- document-to-row mappings;
- canonical document token sets;
- inverted token postings;
- proposition records;
- entity, event, raw-sentence, and provenance indexes;
- task feature caches; and
- source-table signatures.

### 9.2 Exact source tables by task

The runtime source contract is explicit.

#### Reconstruction

```text
chunks.csv
entities.csv
claims.csv
events.csv
relations.csv
kg_pseudo_chunks.csv
kg_pseudo_chunks_reconstruction.csv
kg_pseudo_chunk_components.csv
```

#### Value redaction

Runtime task key: `redaction`.

```text
entities.csv
claims.csv
events.csv
relations.csv
kg_pseudo_chunks.csv
kg_pseudo_chunks_redaction.csv
kg_pseudo_chunk_components.csv
```

`chunks.csv` is intentionally absent from first-stage value-redaction document
retrieval. Source-like text remains available later for raw validation and
provenance, but it does not dominate top-document selection.

#### Sentence redaction

```text
chunks.csv
entities.csv
claims.csv
events.csv
relations.csv
kg_pseudo_chunks.csv
kg_pseudo_chunk_components.csv
```

Sentence reconstruction needs broad visible neighboring context, so source-like
chunks remain a retrieval source.

#### Claims

```text
chunks.csv
entities.csv
claims.csv
events.csv
relations.csv
kg_pseudo_chunks.csv
kg_pseudo_chunks_claims.csv
kg_pseudo_chunk_components.csv
```

#### Contradictions

```text
chunks.csv
entities.csv
claims.csv
events.csv
relations.csv
kg_pseudo_chunks.csv
kg_pseudo_chunks_contradictions.csv
kg_pseudo_chunk_components.csv
```

Component rows provide provenance and structured membership. They are not
treated as independent documents during first-stage ranking.

### 9.3 What a PropositionRecord contains

A runtime proposition record can originate from a claim, event, relation, or
task pseudo surface. It carries, where available:

- document and passage ID;
- source table and row ID;
- readable proposition text;
- normalized subject, predicate, and object;
- reporter and embedded actor;
- agent, patient/theme, recipient, and destination;
- location and date/time;
- polarity, modality, stance, and epistemic state;
- event signature;
- canonical entity concepts;
- raw/source validation pointers; and
- extraction/component provenance.

The evidence table is therefore not just four concatenated text files. It is a
typed, provenance-bearing set of candidate records assembled from the task's
source-table contract.

## 10. Document Retrieval

### 10.1 Document vocabulary

For each document, the runtime aggregates semantic tokens from the task's
source tables. Tokens include normalized words, predicate lemmas, event-family
concepts, and corpus alias concepts. This is the document retrieval
representation.

The workbench does not re-tokenize every document for every query. It caches
document token sets and builds an inverted posting map from token to document
IDs.

### 10.2 Retrieval formula

Let `Q` be the expanded canonical query-token set and `D` the expanded
canonical document-token set.

```text
containment(Q, D) = |Q intersect D| / |Q|
jaccard(Q, D)     = |Q intersect D| / |Q union D|

document_score = 0.62 * containment(Q, D)
               + 0.38 * jaccard(Q, D)
```

Containment rewards a document that covers most of the query concepts. Jaccard
penalizes very broad documents whose vocabulary contains the query but much
unrelated material.

Only documents sharing at least one posting token need a nonzero score. The
runtime fills remaining slots deterministically with zero-score document IDs if
fewer than the configured top-doc count overlap.

### 10.3 Top documents

The default staged frontier is 150 documents. That is a maximum, not a claim
that every task always needs 150. Small corpora are effectively exhaustive.
Large corpora use the posting index and bounded top-doc frontier to prevent each
query from building evidence over every document.

Document retrieval is only Stage 1. A high document score says a document has
relevant concepts; it does not prove that a candidate passes the task-specific
event, role, polarity, date, or answer gates.

## 11. Evidence Table And Candidate Scoring

After document ranking:

1. retain rows whose document IDs are in the top-document set;
2. assemble task-relevant proposition/evidence records;
3. compare each query proposition or slot against those records;
4. compute lexical, identity, event, role, and task features;
5. apply noncompensatory gates; and
6. select a winner only when score and margin policy are satisfied.

A large lexical score cannot compensate for a wrong actor, reversed polarity,
wrong event date, or wrong redaction answer type. This is why the final score is
not merely cosine or token overlap.

Core semantic normalization includes reusable predicate families such as
`construct/build/develop`, stance/modal handling, and structured role
comparison. These rules are general semantic resources, not benchmark case-ID
exceptions.

## 12. Task Engines

### 12.1 Reconstruction

**Input:** a source-like or analyst-style proposition/excerpt.  
**Goal:** determine whether one eligible evidence unit or a coherent evidence
set reconstructs the target.  
**Output:** reconstructable/not reconstructable, selected source documents,
evidence units, score, margin, and failed gates.

Reconstruction permits a collective set when no single row contains the entire
target but compatible rows cover its critical elements. Event identity,
participant consistency, dates, polarity, and source eligibility constrain the
set. A top-ranked source document can still be ineligible if its proposition
has a polarity or event mismatch.

### 12.2 Claims

**Input:** one or more query propositions.  
**Goal:** find direct supporting evidence for the same assertion.  
**Output:** direct support, related evidence, or unresolved status with source
document and proposition audit.

Claims compares actor, predicate family, object, location/date, polarity,
modality, and stance. A proposition can be lexically close but classified as a
related different event. Direct support requires critical arguments and
assertion state to be compatible.

In the concrete Republic of Vietnam example, baseline selected document
`1974JAKART14207`. The winning `claims.csv` row repeated the willingness to
seek a mutually acceptable solution through discussions and negotiations. It
received direct support with actor, predicate, and object matches plus raw
sentence confirmation.

### 12.3 Contradictions

**Input:** a structured assertion with polarity, modality, participants, and
event scope.  
**Goal:** identify direct opposition, attribute contradiction, location
contradiction, or supported sides of a tension.  
**Output:** contradiction category, side-specific evidence, scope/uncertainty
status, and decision audit.

Negation must remain attached to the same actor and predicate. `X did not do Y`
does not contradict `Z did Y`; they are different actors. `X may do Y` is not
equivalent to `X did Y`; uncertainty is part of assertion state.

### 12.4 Value redaction

**Input:** visible text containing one or more typed `[REDACT:Rn ...]` slots.  
**Goal:** recover the hidden entity/value from semantically compatible corpus
evidence.  
**Output:** selected value(s), type/role/event gate audit, source document,
truth match, score, and margin.

The runtime parses each marker, left visible context, right visible context,
expected type, and nearby proposition frame. It retrieves documents using
entities, claims, events, relations, and pseudo surfaces rather than raw chunks.
Candidate values are grounded in corpus entity/event records. Source-like text
is then used to validate that the candidate is actually supported, not to win
solely by verbatim copying.

The engine applies bounded noncompensatory gates for:

- candidate type;
- role compatibility;
- relation compatibility;
- event identity;
- local left/right context;
- source eligibility;
- answer uniqueness; and
- winner margin.

### 12.5 Sentence redaction

**Input:** visible context around `[REDACT_SENT:Sn ...]`.  
**Goal:** identify a missing sentence/proposition that coherently bridges the
visible context.  
**Output:** recovered text/evidence, source, coherence gates, score, and margin.

Unlike value redaction, sentence redaction does not choose one typed entity. It
needs proposition-level continuity and therefore includes source-like chunks in
document retrieval.

## 13. Value Redaction In Detail

### 13.1 Build-time object

Suppose a processed source context is:

```text
Pakistan's ambassador to France denied his country was building a nuclear
device but admitted it had been shopping for equipment.
```

The benchmark generator chooses entity `France`, records the source document
and extraction row, and creates:

```text
Pakistan's ambassador to [REDACT:R1 type=LOCATION] denied his country was
building a nuclear device but admitted it had been shopping for equipment.
```

The answer key records `R1=France`, source document, type, and source context.

### 13.2 Semantic rewrite

A valid analyst-style rewrite can be:

```text
The Pakistani ambassador to [REDACT:R1 type=LOCATION] denied nuclear device
development.
```

This removes secondary detail and changes wording, but it preserves the actor,
ambassador-to-location relation, denial, nuclear-device predicate/object, and
literal slot marker.

### 13.3 Query parse

The parser creates:

- slot `R1`, expected type LOCATION;
- left context: Pakistani ambassador to;
- right context: denied nuclear device development;
- actor/entity: Pakistani ambassador or Pakistan;
- predicate frame: deny/develop nuclear device;
- relation expectation: ambassador destination/posting; and
- polarity/stance: denial.

Left and right context are explicit slot fields and semantic features. They are
not stored only as character offsets in the raw source.

### 13.4 Retrieval without `chunks.csv`

The rewritten query expands aliases and predicate concepts. Documents are
ranked over entity/event/claim/relation/pseudo surfaces. A document can rank
because it contains a canonical France entity linked to a Pakistani ambassador
and a claim about denial of device development, even if its source sentence is
phrased differently.

This is the main protection against a trivial verbatim fill task. Raw chunk
vocabulary is not part of Stage 1 value-redaction document ranking.

### 13.5 Candidate construction and gates

Within top documents, candidate LOCATION values are collected from entity and
redaction-context records. For each candidate, the engine asks:

1. Is it the requested type?
2. Does it occupy a compatible role relative to ambassador/Pakistan?
3. Does its evidence refer to the same denial/development event?
4. Are local context and canonical aliases compatible?
5. Is raw/source evidence available to certify the conclusion?
6. Is the winner sufficiently better than the nearest eligible alternative?

The final answer should be judged both by engine PASS/FAIL and answer-key truth
match. A PASS with a different value is not counted as correct in the A/B truth
metrics.

### 13.6 Remaining semantic limit

Value redaction still depends on visible context. Removing all source wording
would make the hidden value underdetermined. The objective is not zero lexical
signal; it is to prevent one exact copied passage from being the only path to
the answer. Alias concepts, predicate families, structured roles, event
identity, and pseudo surfaces make semantically equivalent context usable.

## 14. Baseline And Holdout

Pair mode creates two matcher directories per case:

```text
runs/<case>/baseline/
runs/<case>/holdout/
```

### Baseline

Baseline has the full configured corpus. For benchmark reporting, terminal
PASS/FAIL and target-document YES/NO are based on baseline so users do not see
a wall of expected holdout failures.

### Holdout

Holdout excludes the source document(s) recorded in the answer key. It asks
whether independent evidence supports the same result. It is not a second
ground-truth answer and should not be expected to fail in every case.

A holdout PASS can be legitimate when another document contains equivalent
evidence. This is especially common in FISC, where a small set of documents
describes the same investigation with different redaction levels. A holdout
PASS must still be checked against the expected answer; an alternative resolved
value can pass engine gates but fail truth match.

## 15. PASS, Truth, Retrieval, And Useful Outcomes

Four metrics must not be conflated.

### Engine PASS

The task engine found an eligible winner or evidence set satisfying its gates,
thresholds, and margin policy.

### Target-document retrieval

The expected source document appeared in the relevant ranked/selected evidence
audit. A document can be retrieved but fail task gates. Conversely, a different
document can produce a valid PASS.

### Truth match

The selected value/text/source is equivalent to the benchmark answer key under
task-appropriate normalization. This is the correctness check beyond engine
confidence.

### Useful outcome

The summary heatmap reports:

```text
PASS (+ relevant retrieval but FAIL) / total
```

A relevant-retrieval failure is a **good fail**: the system found the right
evidence neighborhood but refused to overclaim because a gate or margin failed.
A failure with no relevant retrieval or a preflight block is a **bad fail**.

The desired operating profile is:

- high PASS plus truth-match;
- some conservative good fails where evidence is genuinely ambiguous;
- near-zero bad fails; and
- zero avoidable preflight blocks in selected build cases.

## 16. Runtime Artifacts

Each case directory contains:

- `query_input.txt`;
- `query_parse/` intermediary tables;
- parser command/stdout/stderr logs;
- `baseline/semantic_match_output/`;
- `holdout/semantic_match_output/`;
- timing and execution ledger files; and
- a compact pair digest.

The matcher output contains detailed CSV/JSON audits. Important files include:

| File | Use |
| --- | --- |
| `task_evaluation_v048.csv` | final task decision |
| task-specific candidate audit | every scored candidate and failed gates |
| `candidate_route_document_accounting_v0420.csv` | ranked/selected document routes |
| `task_evidence_units_v0413.csv` | selected evidence units |
| `performance_timing_v048.csv` | stage runtimes |
| `decision_display_v0420.csv` | compact decision fields |
| `task_preflight_v048.csv` | runtime executability audit |

These are detailed audit artifacts, not analyst-facing panel variants. The
analyst-facing summary is intentionally consolidated separately.

## 17. Caching And Runtime

### 17.1 Shared cache location

Query-independent source and feature caches live at:

```text
output/matryoshka_workbench/.matryoshka_cache/
```

They are outside individual benchmark experiments, so original and rewritten
queries and future experiments can reuse them.

### 17.2 Cache identity and validation

Lookup identity uses the KG project name, extraction directory name, and task.
Reuse is validated by:

- exact source-table signatures/hashes;
- task;
- feature-code fingerprint;
- normalization version; and
- semantic-resource fingerprint.

Absolute package paths are not part of compatibility. Therefore, an unchanged
package can be moved or renamed and still reuse its shipped cache. If extraction
content or relevant code changes, the fingerprint check invalidates the stale
payload.

### 17.3 Query parse cache

Parsed queries are cached by query text, parser mode, and parser model under:

```text
output/matryoshka_workbench/.matryoshka_cache/query_parse/
```

Original and rewritten queries share source/feature caches but normally have
different query-parse cache entries.

### 17.4 Why a first query can be slow

The first query for a corpus/task may build source tables, document token
postings, proposition records, raw validation indexes, and task features.
Later queries reuse those structures. Persistent cache makes future command
runs reuse them as well.

Value redaction can still be heavier than claims because it constructs and
gates many typed candidate representations. The bounded top-document path and
entity-context Stage C source prevent representations from exploding across the
entire corpus.

### 17.5 Heartbeat

Long cases emit a heartbeat every 300 seconds. A heartbeat is evidence that the
process remains active; it is not a per-stage timing line. Final per-case output
reports total seconds, PASS/FAIL, target-document status, decision reason, and
holdout status.

## 18. Analyst Reports

The summary generator writes one outward-facing directory:

```text
summaries/analyst_report/
```

Each branch contains only:

| File | Purpose |
| --- | --- |
| `case_results.csv` | comprehensive query, truth, retrieval, baseline/holdout, and outcome rows |
| `useful_outcomes.csv` | corpus/task aggregate counts |
| `useful_outcomes_heatmap.png` | one analyst heatmap |
| `failure_reasons.csv` | top diagnostic reasons by corpus/task |
| `README.md` | concise interpretation guide |

Artifact and Streamlit locations in `case_results.csv` are relative to the
package root. They remain valid when the handoff directory is moved or renamed;
internal execution indexes may still use absolute paths while a run is active.

The rewritten branch additionally contains:

| File | Purpose |
| --- | --- |
| `original_vs_rephrased_cases.csv` | original/rephrased query pair, rewrite diagnostics, outcomes, and truth |
| `original_vs_rephrased_summary.csv` | corpus/task A/B aggregates and regressions |

There are no duplicate success-only tables, minimal query catalogs, attempt
tables, or multiple heatmap panel variants. `runs/` retains detailed audits for
debugging; `analyst_report/` is the handoff view.

## 19. Streamlit

Launch from the package root:

```bash
streamlit run scripts/matryoshka_workbench/release_v0421/streamlit_semantic_workbench_v3_19.py
```

Open a case directory such as:

```text
output/matryoshka_workbench/benchmarks/
  five_corpus_pair_20x5_semantic_final__original/
  runs/101_rc00_viet
```

When the case has `baseline/` and `holdout/`, the app renders controlled
side-by-side columns automatically. The UI reads the case audit files; it does
not need pre-rendered duplicate HTML panels.

## 20. Final Command

The final full A/B benchmark defaults to all five corpora and all five tasks.
The wrapper's default is 10 cases per corpus/task. This explicit final command
selects up to 20 cases per corpus/task, uses seed 42 and pair mode,
parses selected runtime queries with `gpt-4o`, creates one-shot semantic
rewrites with `gpt-4o`, and overwrites experiments of the same name:

```bash
MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python \
  scripts/matryoshka_workbench/tools/run_rephrase_ab_benchmark.py \
  --stage all \
  --experiment-name five_corpus_pair_20x5_semantic_final \
  --corpora vietnam,cib,iran,fisc,canberra \
  --tasks reconstruction,claims,value_redaction,sentence_redaction,contradictions \
  --per-task 20 \
  --seed 42 \
  --runner-mode pair \
  --use-openai-parser \
  --openai-model gpt-4o \
  --rephrase-mode semantic \
  --overwrite
```

The runner creates:

```text
five_corpus_pair_20x5_semantic_final__original
five_corpus_pair_20x5_semantic_final__rephrased
```

If build completed but runtime was interrupted, rerun only the execution phase:

```bash
MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python \
  scripts/matryoshka_workbench/tools/run_rephrase_ab_benchmark.py \
  --stage run \
  --experiment-name five_corpus_pair_20x5_semantic_final \
  --corpora vietnam,cib,iran,fisc,canberra \
  --tasks reconstruction,claims,value_redaction,sentence_redaction,contradictions \
  --per-task 20 \
  --seed 42 \
  --runner-mode pair \
  --use-openai-parser \
  --openai-model gpt-4o \
  --rephrase-mode semantic \
  --overwrite
```

`--stage run` uses the existing manifests and rewrites. It does not regenerate
queries. Shared corpus/task caches are reused when fingerprints match.

## 21. Lower-Level Commands

Build only, useful for inspecting query quality before paying runtime API cost:

```bash
MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python \
  scripts/matryoshka_workbench/tools/run_rephrase_ab_benchmark.py \
  --stage build \
  --experiment-name five_corpus_semantic_audit \
  --per-task 10 \
  --seed 42 \
  --openai-model gpt-4o \
  --rephrase-mode semantic \
  --overwrite
```

Note that semantic build includes selected-query rewrite and judge calls after
the local original catalog is built. The original candidate catalog itself does
not use OpenAI.

Summarize a single branch again:

```bash
MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python \
  scripts/matryoshka_workbench/tools/summarize_benchmark_results.py \
  --experiment-root \
  output/matryoshka_workbench/benchmarks/five_corpus_pair_20x5_semantic_final__original
```

Run the lower-level non-rephrased workflow only when needed:

```bash
MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python \
  scripts/matryoshka_workbench/tools/multicorpus_benchmark.py all \
  --corpora vietnam,cib,iran,fisc,canberra \
  --tasks reconstruction,claims,value_redaction,sentence_redaction,contradictions \
  --config scripts/matryoshka_workbench/config/benchmark.default.json \
  --experiment-name original_only_audit \
  --per-task 10 \
  --seed 42 \
  --runner-mode pair \
  --use-openai-parser \
  --parser-model gpt-4o \
  --overwrite
```

## 22. Debugging By Failure Type

### Preflight block

Inspect:

```text
query_parse/query_task_preflight.csv
baseline/semantic_match_output/task_preflight_v048.csv
parser.stderr.log
```

Selected benchmark queries should rarely reach runtime with invalid markers,
missing executable propositions, or task mismatch. Fix a reusable build gate or
parser rule rather than relabeling the result.

### Target document not retrieved

Inspect `candidate_route_document_accounting_v0420.csv`. Determine:

- whether the expected document had any alias-expanded overlap;
- its rank among considered documents;
- which query anchors disappeared after rewriting;
- whether the corpus alias resource loaded successfully; and
- whether a distinctive event/predicate concept normalized as expected.

If original retrieves and rewrite does not, examine rewrite anchor recall and
query parser outputs before changing final task gates.

### Target document retrieved but FAIL

Inspect the task-specific candidate audit and `failed_gates`. This is usually an
event, actor, object, polarity, type, or margin issue, not document retrieval.
Such a failure can be a defensible good fail.

### PASS but wrong answer

Compare selected output with `query_answer_key.csv` and truth fields in
`case_results.csv`. This is more serious than a conservative fail. Tighten a
general type/role/event/answer-identity gate or fix answer normalization. Do not
count engine PASS alone as correctness.

### Original and rewritten results differ despite identical query text

Verify exact query hashes, parser mode/model, decision profile fingerprint,
semantic resource fingerprint, source table signatures, and holdout source set.
The runtime is deterministic after parsing. Different cached parser artifacts
or resource fingerprints explain most apparent nondeterminism.

### Runtime is unexpectedly long

Inspect `performance_timing_v048.csv` and
`coordinated_performance_stages_v0420.csv`. Distinguish:

- cache preparation;
- query parsing/API latency;
- document retrieval;
- candidate representation construction;
- task scoring/gates; and
- raw certification.

Do not infer the bottleneck from total seconds alone.

## 23. Scientific Interpretation

The benchmark is controlled, not a random sample of all possible analyst
questions. Query generators create known-answer cases from corpus artifacts so
retrieval, evidence eligibility, truth, and holdout can be evaluated. Ranking
then selects high-quality, diverse cases.

The catalog ceiling, top-doc frontier, and decision thresholds are operating
parameters. They are defensible when reported with:

- source corpus size;
- raw, admissible, and selected counts;
- seed;
- exact configuration snapshot;
- PASS and truth-match separately;
- target-document retrieval;
- good versus bad failure counts;
- baseline versus holdout; and
- original versus semantic rewrite.

They should not be presented as universal natural constants. A new corpus
should first be audited with build-stage count distributions and selected-query
inspection, then evaluated across multiple seeds before profile overrides are
adopted.

## 24. New Corpus Checklist

To add a sixth logical corpus:

1. Build a canonical KG project through Stage C.
2. Create `config/corpus_profiles/<name>.json` with extraction, engine,
   decision profile, and fallback resource paths.
3. Add `data/manifests/<name>.json` if richer metadata is needed.
4. Confirm adjacent `alias/corpus_entity_resource_v1.json` loads.
5. Run build only with small `--per-task` and inspect
   `final_query_catalog.csv` and `build_stage_counts.csv`.
6. Confirm no selected preflight blocks.
7. Run baseline/holdout without rewrite.
8. Run semantic A/B across at least two seeds.
9. Add corpus-specific sampling overrides only when a documented source-format
   or corpus-structure difference requires them.
10. Never add a case-ID exception to make one benchmark row pass.

## 25. Script Responsibility Map

| Script/module | Responsibility |
| --- | --- |
| `run_rephrase_ab_benchmark.py` | final original vs semantic orchestration |
| `multicorpus_benchmark.py` | multicorpus build/run/snapshot/index |
| `rephrase_benchmark_queries.py` | one-shot rewrite, repair, judge, audit |
| `summarize_benchmark_results.py` | compact analyst report |
| `corpus_benchmark.py` | shared generators, selectors, runner integration |
| `task_registry.py` | task names and collector registration |
| `task_collectors.py` | task collector dispatch |
| `task_build_context.py` | shared build-time data context |
| `task_catalog_pipeline.py` | per-task catalog orchestration |
| `task_case_selection.py` | final rank/diversity selection |
| `task_build_rules.py` | selected-slice task rules |
| `task_backfill_rules.py` | task-specific non-starving backfill |
| `task_expected_payloads.py` | benchmark truth payload construction |
| `task_policies.py` | task generation policy resolution |
| `task_redaction_balance.py` | redaction type/slot balancing |
| `query_catalog.py` | canonical catalog writer |
| `query_semantic_parser_v0_4_15.py` | runtime query parser CLI |
| `external_query_semantic_matcher_v0_4_20.py` | matcher CLI |
| `run_coordinated_v0_4_21.py` | parser/matcher coordination |
| `matryoshka_batch_runner_v1_0_18.py` | resumable pair batch execution/cache |
| `streamlit_semantic_workbench_v3_19.py` | interactive audit UI |
| `matryoshka_v0420/corpus.py` | immutable task corpus index |
| `matryoshka_v0420/retrieval.py` | bounded alias-aware document retrieval |
| `matryoshka_v0420/semantic_resources.py` | alias/event concept expansion |
| `matryoshka_v0420/semantics.py` | normalization, signatures, role comparison |
| `matryoshka_v0420/engines/*.py` | five independent task engines |

This separation is intentional. Task engines own their gates, while retrieval,
alias expansion, corpus loading, and reporting are shared reusable layers.
