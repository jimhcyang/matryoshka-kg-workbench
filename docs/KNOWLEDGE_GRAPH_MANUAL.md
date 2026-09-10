# Knowledge Graph Pipeline Manual

**Canonical handoff manual**  
**Updated: September 1, 2026**

## 1. Purpose And Scope

This manual explains the complete supported knowledge-graph workflow in this
package. It is written for a researcher who has the package but does not have
the original developer available. It covers:

- how raw source files become prepared corpus records;
- how Stage A converts text into chunks, entities, events, claims, relations,
  and entity profiles;
- how alias resolution forms canonical entity clusters without destroying the
  original extracted text;
- how Stage B materializes graph nodes, edges, and provenance;
- how Stage C creates task-specific semantic evidence views;
- where OpenAI is called, what is local, and how interrupted runs resume;
- which files are authoritative at each boundary;
- how to rebuild all seven shipped project variants; and
- how to diagnose common extraction, alias, and materialization failures.

The supported architecture is deliberately linear:

```text
raw source
  -> corpus-specific preparation
  -> common prepared corpus
  -> Stage A extraction
  -> alias resolution and canonicalized extraction
  -> Stage B graph
  -> Stage C task evidence views
  -> Matryoshka workbench
```

The pipeline never treats a pseudo chunk as a replacement for source
provenance. Raw and prepared text, extracted structure, canonical identity, and
task-specific semantic surfaces are retained as separate layers.

## 2. Package Root And Portability

In this manual, **package root** means the directory containing the top-level
`README.md`, `requirements.txt`, `data/`, `scripts/`, and `output/` entries.
The package root may have any name. It does not need to be named `src`, and it
does not need to be nested in the original repository.

Run all commands from the package root:

```bash
cd /path/to/the/unpacked/package
```

The supported scripts derive defaults from their own location. They do not
depend on `/Users/...`, the original repository name, or a parent checkout.
Some JSON manifests record absolute paths as audit information. Those recorded
strings describe where a previous run occurred; they are not runtime path
configuration for a new run.

## 3. Environment

Create a Python 3.11 environment and install the package requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set the API key when running Stage A, pair-corpus OCR, or optional alias head
review:

```bash
export OPENAI_API_KEY="your-key"
```

FISC PDF preparation renders pages with Poppler. Install `pdftoppm` before
running the FISC preparer:

```bash
# macOS
brew install poppler

# Debian or Ubuntu
sudo apt-get install poppler-utils
```

The root `requirements.txt` is authoritative. The main dependencies are
NumPy, pandas, OpenAI, tqdm, Streamlit, Matplotlib, and openpyxl. The KG stages
do not require a graph-database server; the authoritative graph outputs are
portable CSV and JSON files.

## 4. Corpus Inventory

The package supports five logical corpora and seven KG projects.

| Logical corpus | Prepared root | Project variants | Dataset mode |
| --- | --- | --- | --- |
| Vietnam | `data/prepared/vietnam` | `viet_full_unredacted` | `generic` |
| CIB | `data/prepared/cib` | `cib_full_unredacted`, `cib_full_redacted` | `cib` |
| Iran | `data/prepared/iran` | `iran_full_unredacted` | `generic` |
| FISC | `data/prepared/fisc` | `fisc_full_unredacted`, `fisc_full_redacted` | `fisc` |
| Canberra | `data/prepared/canberra` | `canberra_full_unredacted` | `generic` |

CIB and FISC are true paired corpora. Their prepared records contain aligned
redacted and unredacted transcriptions. Vietnam and Iran are single-source text
corpora. Canberra is also single-source, but its preparer writes the same
pair-style directory contract with identical redacted and unredacted text so
the shared Stage A reader can be used unchanged.

There should be one Vietnam project named `viet_full_unredacted`. The logical
profile is called `vietnam`, while the output project uses the shorter `viet`
identifier. A second `vietnam_full_unredacted` project is not part of the
supported layout.

## 5. Directory Contract

### 5.1 Raw data

`data/raw/<corpus>/` contains source material that preparation scripts read.
Treat raw data as immutable input. Do not store generated OCR, alignment, or
extraction files there.

Typical raw inputs are:

- Vietnam: `data/raw/vietnam/source_input.csv`
- Iran: `data/raw/iran/source_input.csv`
- Canberra: `data/raw/canberra/canberra_cables.json`
- CIB: metadata CSV plus `redacted_pdfs/` and `unredacted_pdfs/`
- FISC: `Pair_*-1.pdf`, `Pair_*-2.pdf`, and optional workbook metadata

### 5.2 Intermediate data

Pair-corpus transcription and alignment may create:

```text
data/intermediate/cib/transcription_staging/
data/intermediate/fisc/transcription_staging/
```

These directories are generated working state. They are useful for resuming OCR
or auditing alignment, but they are not Stage A input once
`data/prepared/<corpus>/` has been built. The preparers support
`--cleanup-staging` when disk space matters.

### 5.3 Prepared data

`data/prepared/<corpus>/` is the authoritative input boundary for Stage A. A
prepared document or pair includes normalized text, metadata, and the
pair/alignment artifacts needed to select the requested transcription variant.

The important design property is uniformity: Stage A does not need separate
loaders for a CSV cable, JSON cable, or PDF pair after preparation.

### 5.4 KG project output

Every project under `output/knowledge_graph/<project>/` follows this shape:

```text
<project>/
|-- input_documents.csv
|-- input_build_summary.json
|-- prepared/
|-- extraction/
|-- extraction_run/
|-- alias/
|-- extraction_canonicalized/
|-- graph/
`-- tasks/
```

Each directory is a durable stage boundary. A later stage reads the previous
boundary rather than recomputing it implicitly.

## 6. Corpus Preparation

Preparation is corpus-specific only because the raw formats differ. All five
entry points emit the same Stage A-compatible contract.

### 6.1 Vietnam

```bash
python scripts/knowledge_graph/kg_scripts/prepare_vietnam_corpus.py --overwrite
```

The wrapper reads the source CSV columns `doc_id`, `subject`, `date`, `body`,
and `id`. It creates one prepared document directory per source row and writes
identical redacted/unredacted text because the source is not paired.

Use `--limit N` for a preparation smoke test. Do not use a limited prepared
root for the final KG unless that truncation is intentional.

### 6.2 Iran

```bash
python scripts/knowledge_graph/kg_scripts/prepare_iran_corpus.py --overwrite
```

Iran uses the same single-source materializer and column contract as Vietnam.
The collection name differs, but the downstream shape is identical.

### 6.3 Canberra

```bash
python scripts/knowledge_graph/kg_scripts/prepare_canberra_corpus.py --overwrite
```

The Canberra wrapper reads `canberra_cables.json`. It preserves each cable's
document identifier, date, subject, classification metadata, and body lines.
It writes matching redacted and unredacted transcriptions and empty pair-diff
artifacts. This is a compatibility representation, not a claim that Canberra
contains a hidden redacted counterpart.

### 6.4 CIB

```bash
python scripts/knowledge_graph/kg_scripts/prepare_cib_corpus.py \
  --overwrite \
  --openai-model gpt-4o-mini
```

The CIB preparer performs two operations:

1. It transcribes redacted and unredacted PDFs page by page using OpenAI.
2. It aligns the paired transcriptions and materializes redaction labels and
   the common prepared corpus.

Useful controls:

- `--pair-key`: process one pair for a smoke test.
- `--max-pairs`: cap the number of pairs.
- `--skip-transcription`: reuse existing staging and rerun postprocessing.
- `--cleanup-staging`: remove staging after successful preparation.
- `--overwrite`: discard existing staging and prepared output first.

The transcription prompt requires literal visible-page transcription and
forbids guessing hidden text. Pair alignment then identifies text present in
the unredacted version but absent in the redacted version.

### 6.5 FISC

```bash
python scripts/knowledge_graph/kg_scripts/prepare_fisc_corpus.py \
  --overwrite \
  --openai-model gpt-4o-mini
```

FISC starts from paired PDFs such as `Pair_1-1.pdf` and `Pair_1-2.pdf`.
Preparation renders and transcribes each page, aligns pages and text, and
creates the pair-style prepared root. The optional spreadsheet enriches pair
metadata but is not the text source.

Useful controls:

- `--pair-key Pair_1`
- `--max-pairs N`
- `--skip-postprocess`
- `--cleanup-staging`
- `--stop-on-error`
- `--image-render-root /temporary/path`

If OCR is interrupted, retain the transcription staging directory and rerun
without `--overwrite`; completed work can be reused. Use `--overwrite` only
when the intent is a clean re-transcription.

## 7. The Shared Pipeline Runner

`run_pipeline.py` orchestrates four named steps:

```text
A -> ALIAS -> B -> C
```

Its main arguments are:

- `--corpus-root`: a prepared corpus root.
- `--project-dir`: the output KG project.
- `--dataset-mode`: `generic`, `cib`, `fisc`, or `auto`.
- `--transcription-variant`: `unredacted` or `redacted`.
- `--openai-model`: Stage A extraction model.
- `--use-openai-head-review`: enable Stage 2 LLM alias adjudication.
- `--head-review-model`: model used only for ambiguous alias pairs.
- `--start-from`: `A`, `ALIAS`, `B`, or `C`.
- `--skip-existing`: skip stages whose summary file already exists.
- `--limit`: cap input documents for a smoke run.
- `--max-new-chunks`: cap newly processed Stage A chunks.

`--start-from` and `--skip-existing` solve different problems. `--start-from
ALIAS` deliberately reruns alias and all downstream stages. `--skip-existing`
checks stage summary files and avoids stages that already completed.

## 8. Stage A: Extraction

### 8.1 Input normalization

Stage A reads the selected transcription variant from the prepared corpus and
creates an input-document inventory. It carries document identifiers, pair
keys, dates, titles, and provenance into later tables.

### 8.2 Chunking

Documents are divided into bounded text segments. The extractor uses a maximum
segment size of approximately 3,000 characters and retains a sentence of
overlap between adjacent segments. The overlap protects facts and entity
mentions that cross a boundary; it does not create a new source fact.

Each chunk receives stable document and segment identifiers. For example:

```text
doc_id: 1973BANGKO00002
paragraph_id: seg0001
```

### 8.3 OpenAI extraction

Stage A sends one chunk at a time to the configured OpenAI extraction model.
The response is parsed into structured rows rather than treated as a narrative
summary. The core extracted objects are:

- **entity**: a named or typed participant with surface form, type, and context;
- **event**: an occurrence with participants, location, and time when present;
- **claim**: an assertion with text, speaker/target/stance when available;
- **relation**: a typed edge such as support, opposition, affiliation, request,
  communication, negotiation, agreement, or event participation.

The extraction model is allowed to normalize structure, but each row retains
document, paragraph, source text, and extraction provenance.

### 8.4 Output tables

The Stage A boundary is `extraction/`:

| File | Meaning |
| --- | --- |
| `chunks.csv` | processed source segments, still close to source wording |
| `entities.csv` | extracted entity mentions and local keys |
| `events.csv` | extracted event records and roles |
| `claims.csv` | extracted assertion surfaces and metadata |
| `relations.csv` | typed relationships with provenance |
| `base_entity_profiles.csv` | corpus-level statistics and context per pre-alias entity |
| `summary.json` | counts and completion metadata |

`chunks.csv` is not raw binary source. It is the normalized transcription text
sectioned into extraction segments. It is intentionally much closer to the
source wording than pseudo chunks are.

### 8.5 Checkpoint and resume behavior

Stage A writes incremental state under `extraction_run/`:

```text
extraction_run/chunk_results.jsonl
extraction_run/chunk_errors.jsonl
```

A successfully extracted chunk is checkpointed. A chunk that exhausts API
retries is recorded as an error and is not marked complete. Therefore, a Wi-Fi
outage can produce a visible message such as `Skipping Stage A chunk`, but the
chunk is eligible when Stage A is rerun. Rerunning the same project does not
need to repeat successful chunks; it fills missing work from checkpoint state.

To resume extraction and continue downstream stages:

```bash
python scripts/knowledge_graph/kg_scripts/run_pipeline.py \
  --corpus-root data/prepared/vietnam \
  --project-dir output/knowledge_graph/viet_full_unredacted \
  --dataset-mode generic \
  --transcription-variant unredacted \
  --openai-model gpt-4o-mini
```

Do not delete `extraction_run/` before a resume. To audit remaining failures,
inspect `chunk_errors.jsonl` and compare Stage A summary counts to the prepared
chunk inventory.

## 9. Alias Resolution

Alias resolution maps pre-alias entity profiles into corpus-level canonical
entities. It does not overwrite original mention text. Instead, it creates a
mapping layer and writes canonical companion fields into copied extraction
tables.

### 9.1 Why aliasing is necessary

Without aliasing, these may form separate high-frequency entity clusters:

```text
USSR
SOVIET UNION
UNION OF SOVIET SOCIALIST REPUBLICS
```

That fragmentation weakens graph structure and query retrieval. With aliasing,
all three can resolve to one canonical entity such as `SOVIET UNION`, while the
original surfaces remain available for provenance and display.

Aliasing must also reject tempting but invalid merges. For example, `USDAO
SGN` refers to the United States Defense Attache Office in Saigon; it is not a
synonym for the country `UNITED STATES`.

### 9.2 Base entity profiles

`base_entity_profiles.csv` aggregates evidence for each extracted entity key:

- normalized label and observed surfaces;
- primary type and type distribution;
- mention count and distinct document count;
- representative contexts;
- frequent co-occurring entities;
- relation-neighbor entities; and
- generic/acronym-like flags.

This context is why auto-aliasing can be more reliable than string matching
alone. A label pair is judged using type, lexical form, corpus usage, and
neighborhood evidence together.

### 9.3 Candidate blocking

The resolver does not compare every entity to every other entity blindly. It
creates candidate blocks from normalized tokens, acronym/full-form patterns,
name structure, and compatible type families. This preserves broad recall for
plausible pairs while avoiding quadratic work on obviously unrelated labels.

### 9.4 Stage 1 local score

For each candidate pair, the deterministic local score is bounded to `[0, 1]`:

```text
0.25 * type compatibility
+ 0.20 * IDF-weighted token Jaccard
+ 0.10 * plain token Jaccard
+ 0.10 * maximum directional containment
+ 0.08 * character sequence similarity
+ 0.10 * representative-context similarity
+ 0.07 * co-occurrence-neighborhood similarity
+ 0.05 * relation-neighbor similarity
+ 0.15 if exact acronym initials match
+ 0.18 if a compatible person-name core matches
+ 0.08 if a compatible head noun matches
- 0.35 for a hard conflict
```

The weighted terms are evidence channels, not independent probabilities. The
final thresholds are conservative operating policy:

- hard conflict and score below `0.70`: keep separate;
- supported acronym/full form and score at least `0.35`: merge;
- general score at least `0.86`: merge;
- person-name core and score at least `0.85`: merge;
- score at most `0.48`: keep separate;
- otherwise: uncertain and eligible for later review policy.

Short acronyms require stronger context or neighborhood support because labels
such as `US`, `UN`, or `PR` can be ambiguous. An acronym that plausibly expands
to multiple different corpus entities is not auto-merged.

### 9.5 Stage 2 head review

Stage 2 operates on Stage 1 clusters, not raw mention pairs. It focuses effort
on materially relevant unresolved entities and official name/acronym variants.

When defaults are used, the shortlist frontier is corpus-scaled:

```text
if canonical profile count <= 400:
    examine all profiles
else:
    top_k = min(
        max(50, ceil(1.75 * sqrt(profile_count)), ceil(0.005 * profile_count)),
        400,
        profile_count,
    )
```

For no more than 128 selected profiles, per-anchor comparison is exhaustive.
Above 128 profiles, the automatic per-anchor cap is:

```text
ceil(log2(selected_profile_count + 1) * 2.4), bounded to 12..24
```

The total candidate cap is corpus-scaled and bounded. Explicit official-name
and acronym/full-form rescue rules can admit high-value candidates that would
otherwise sit outside the frequency frontier. The aim is not to merge every
singleton; it is to make important canonical clusters complete without sending
large numbers of unsupported pairs to an LLM.

Each Stage 2 candidate is classified as one of:

- `deterministic_merge`: near-identical, official, sufficiently supported;
- `llm_review`: ambiguous but supported by multiple signals and relevant;
- `reject`: incompatible type, conflict, generic ambiguity, or inadequate
  semantic/context support.

### 9.6 OpenAI head review

OpenAI is called in alias resolution only when both conditions are true:

1. `--use-openai-head-review` was supplied; and
2. a Stage 2 candidate is classified `llm_review`.

The LLM is not asked to discover aliases from the entire entity inventory. It
adjudicates a prefiltered pair with rich corpus context: labels, types,
frequencies, representative contexts, co-occurrences, relation neighbors,
acronym evidence, and deterministic scores. The expected answer is a structured
same/separate decision and rationale. `gpt-4o` is the shipped head-review model.

This boundary is economical and auditable: millions of local candidate scores
may yield only a small number of OpenAI adjudications.

### 9.7 Canonical names

Canonical display labels prefer an informative, non-ambiguous, human-readable
surface when evidence permits. Full official forms are preferred over opaque
abbreviations when they clearly denote the same entity. For example, `UNITED
STATES` is preferable to `US`, and `SOVIET UNION` is preferable to `USSR`.

Frequency is evidence, but not the sole rule. A frequent acronym does not win
when a clear full form is available; a very long or locally specific expansion
does not win merely because it is longer.

### 9.8 Alias outputs

The `alias/` directory contains:

| File | Purpose |
| --- | --- |
| `alias_candidates.csv` | Stage 1 candidate features, scores, and decisions |
| `alias_edges.csv` | accepted same-entity edges |
| `alias_map.csv` | every base entity mapped to canonical ID and label |
| `canonical_entities.csv` | one row per canonical cluster |
| `head_review_profiles.csv` | profiles selected for Stage 2 consideration |
| `head_review_candidates.csv` | Stage 2 decisions before optional LLM result |
| `head_review_decisions.csv` | deterministic and OpenAI adjudications |
| `head_review_shortlist_summary.json` | shortlist/cap accounting |
| `corpus_entity_resource_v1.json` | runtime alias resource for the workbench |
| `summary.json` | profile, edge, and canonical counts |

To manually inspect all accepted aliases, start with `alias_map.csv`, group by
`canonical_entity_id`, and sort by cluster document or mention frequency.
`alias_edges.csv` shows which accepted links caused components to join.

### 9.9 Concrete alias result

The current Vietnam output contains one canonical cluster where:

```text
USSR                                  -> SOVIET UNION
SOVIET UNION                          -> SOVIET UNION
UNION OF SOVIET SOCIALIST REPUBLICS   -> SOVIET UNION
canonical_entity_id                   -> CANONICAL_00031
```

The key point is that the long form was not hard-coded as a benchmark answer.
It survived candidate generation and the shared alias logic. Compound entities
such as `USSR GOVERNMENT` or `USSR AMBASSADOR` remain separate because they
refer to institutions or people, not the country itself.

## 10. Canonicalized Extraction

Alias resolution writes `extraction_canonicalized/`. This directory is a copy
of the Stage A tables augmented with canonical identity fields. It is the
authoritative extraction root consumed by Stage B, Stage C, and the workbench.

The pipeline does **not** destructively replace source phrases with canonical
labels. A row can carry all of the following simultaneously:

- original text: `USSR`;
- normalized local label: `USSR`;
- canonical entity ID: `CANONICAL_00031`;
- canonical display surface: `SOVIET UNION`; and
- alias/concept expansions used for retrieval.

This separation permits human-readable provenance and alias-aware matching at
the same time.

## 11. Stage B: Graph Construction

Stage B reads only `extraction_canonicalized/` and creates:

| File | Meaning |
| --- | --- |
| `graph/nodes.csv` | document, segment, canonical entity, event, and claim nodes |
| `graph/edges.csv` | typed graph relationships |
| `graph/provenance_links.csv` | links from graph objects back to extraction rows |
| `graph/summary.json` | node and edge counts by family |

Typical edge families include mention, claim speaker/target, event participant,
event location/time, support, opposition, affiliation, request, communication,
negotiation, agreement, causal, temporal, and other-related links.

Stage B is deterministic. It does not call OpenAI. If alias outputs change,
rerun Stage B so graph identity and edge endpoints use the new canonical map.

## 12. Stage C: Task Evidence Materialization

Stage C converts canonical extraction and graph-adjacent structure into compact
semantic surfaces for downstream tasks. It is deterministic and does not call
OpenAI.

### 12.1 Task directory

`tasks/` contains the explicit task views:

| File | Purpose |
| --- | --- |
| `pseudo_chunks.csv` | structured segment summaries |
| `pseudo_chunk_components.csv` | provenance rows for every included component |
| `reconstruction_view.csv` | reconstruction-oriented semantic signatures |
| `short_redaction_recovery_view.csv` | true short paired-redaction slots |
| `sentence_redaction_recovery_view.csv` | true paired sentence-redaction slots |
| `claim_support_view.csv` | claim support surfaces |
| `contradiction_tension_view.csv` | contradiction-oriented surfaces |
| `summary.json` | row counts and compatibility output accounting |

### 12.2 Matcher compatibility files

Stage C also writes task files directly into `extraction_canonicalized/` so the
runtime has one extraction root:

```text
kg_pseudo_chunks.csv
kg_pseudo_chunk_components.csv
kg_pseudo_chunks_reconstruction.csv
kg_pseudo_chunks_redaction.csv
kg_pseudo_chunks_claims.csv
kg_pseudo_chunks_contradictions.csv
```

These are runtime evidence sources, not query-generation-only artifacts.

### 12.3 What a pseudo chunk is

A pseudo chunk is a structured semantic surface tied to one source document
and segment. It may combine:

- document ID, paragraph ID, date, and subject;
- canonical entity labels and IDs;
- extracted claim text;
- event names, participants, location, and time;
- typed relation strings;
- a compact signature used for retrieval; and
- component provenance back to exact extraction rows.

It is not a free-form LLM summary and it is not the raw source passage. One
pseudo chunk can contain multiple entities because a segment can describe a
multi-actor event. Canonical entity fields make those participants matchable
even when source and query surfaces differ.

### 12.4 Why component rows outnumber pseudo chunks

`kg_pseudo_chunk_components.csv` is one-to-many provenance. A single pseudo
chunk can contain 15 entities, two claims, one event, and three relations. That
one pseudo chunk therefore produces many component rows. A larger component
count is expected and does not mean source text was duplicated into extra
semantic chunks.

### 12.5 What is in `kg_pseudo_chunks_redaction.csv`

The redaction compatibility file intentionally contains two row families:

1. `redaction_slot` rows derived from actual paired redacted/unredacted
   alignment where a hidden span is known; and
2. `entity_candidate_context` rows derived from canonical entity-bearing
   segment context, used as semantic candidate evidence at runtime.

This explains why Vietnam, Iran, and Canberra have redaction compatibility rows
despite not being paired corpora. They do **not** acquire synthetic historical
redactions. They contribute entity-context evidence that can help answer a
redaction query generated elsewhere. True benchmark redaction targets for pair
corpora still come from the prepared pair span index or paired recovery views.

The Stage C summary distinguishes true short/sentence redaction row counts from
the larger matcher compatibility row count.

## 13. End-To-End Concrete Example

Consider Vietnam document `1973BANGKO00002`, segment `seg0001`.

### 13.1 Source-like chunk

The normalized segment includes a statement that sources inside and outside
Thailand opposed to the U.S. presence would use the coming interregnum period.
`chunks.csv` preserves this source-like wording and document provenance.

### 13.2 Stage A objects

Stage A extracts, among other rows:

```text
entity: UNITED STATES, type Country
entity: THAILAND, type Country
claim: those sources within and outside Thailand opposed to the U.S. presence
       will take opportunities during the interregnum
relation: UNITED STATES --OPPOSITION--> THAILAND
```

The relation is an extracted semantic assertion with a pointer back to the
source segment. It is not evidence that every mention of the United States is
opposition to Thailand.

### 13.3 Alias resolution

The local source surface `U.S.` maps through its base entity and accepted alias
cluster to canonical display label `UNITED STATES`. The source sentence remains
unchanged in provenance fields.

### 13.4 Stage B

Stage B creates canonical entity nodes for the United States and Thailand,
claim/segment/document nodes, and typed edges linking the relation and claim to
their source.

### 13.5 Stage C

The corresponding pseudo chunk contains a readable semantic surface similar to:

```text
Document: 1973BANGKO00002
Paragraph: seg0001
Entities: ...; UNITED STATES; THAILAND
Claims: those sources ... opposed to the U.S. presence ...
Relations: UNITED STATES --OPPOSITION--> THAILAND
```

It also carries canonical keys and a structured component list. A workbench
query using `US`, `United States`, or a mapped equivalent can therefore emit the
same alias concept token during retrieval while still displaying the original
claim and source text as evidence.

## 14. Build All Seven Projects

Run preparation first. Then use these project commands from the package root.
The examples enable OpenAI head review because the shipped final alias design
uses it for the small ambiguous Stage 2 subset.

### 14.1 Vietnam unredacted

```bash
python scripts/knowledge_graph/kg_scripts/run_pipeline.py \
  --corpus-root data/prepared/vietnam \
  --project-dir output/knowledge_graph/viet_full_unredacted \
  --dataset-mode generic \
  --transcription-variant unredacted \
  --openai-model gpt-4o-mini \
  --use-openai-head-review \
  --head-review-model gpt-4o
```

### 14.2 CIB unredacted

```bash
python scripts/knowledge_graph/kg_scripts/run_pipeline.py \
  --corpus-root data/prepared/cib \
  --project-dir output/knowledge_graph/cib_full_unredacted \
  --dataset-mode cib \
  --transcription-variant unredacted \
  --openai-model gpt-4o-mini \
  --use-openai-head-review \
  --head-review-model gpt-4o
```

### 14.3 CIB redacted

```bash
python scripts/knowledge_graph/kg_scripts/run_pipeline.py \
  --corpus-root data/prepared/cib \
  --project-dir output/knowledge_graph/cib_full_redacted \
  --dataset-mode cib \
  --transcription-variant redacted \
  --openai-model gpt-4o-mini \
  --use-openai-head-review \
  --head-review-model gpt-4o
```

### 14.4 Iran unredacted

```bash
python scripts/knowledge_graph/kg_scripts/run_pipeline.py \
  --corpus-root data/prepared/iran \
  --project-dir output/knowledge_graph/iran_full_unredacted \
  --dataset-mode generic \
  --transcription-variant unredacted \
  --openai-model gpt-4o-mini \
  --use-openai-head-review \
  --head-review-model gpt-4o
```

### 14.5 FISC unredacted

```bash
python scripts/knowledge_graph/kg_scripts/run_pipeline.py \
  --corpus-root data/prepared/fisc \
  --project-dir output/knowledge_graph/fisc_full_unredacted \
  --dataset-mode fisc \
  --transcription-variant unredacted \
  --openai-model gpt-4o-mini \
  --use-openai-head-review \
  --head-review-model gpt-4o
```

### 14.6 FISC redacted

```bash
python scripts/knowledge_graph/kg_scripts/run_pipeline.py \
  --corpus-root data/prepared/fisc \
  --project-dir output/knowledge_graph/fisc_full_redacted \
  --dataset-mode fisc \
  --transcription-variant redacted \
  --openai-model gpt-4o-mini \
  --use-openai-head-review \
  --head-review-model gpt-4o
```

### 14.7 Canberra unredacted

```bash
python scripts/knowledge_graph/kg_scripts/run_pipeline.py \
  --corpus-root data/prepared/canberra \
  --project-dir output/knowledge_graph/canberra_full_unredacted \
  --dataset-mode generic \
  --transcription-variant unredacted \
  --openai-model gpt-4o-mini \
  --use-openai-head-review \
  --head-review-model gpt-4o
```

## 15. Rerun From Alias Resolution

If Stage A is complete and only alias policy or downstream materialization has
changed, rerun from `ALIAS`:

```bash
python scripts/knowledge_graph/kg_scripts/run_pipeline.py \
  --corpus-root data/prepared/vietnam \
  --project-dir output/knowledge_graph/viet_full_unredacted \
  --dataset-mode generic \
  --transcription-variant unredacted \
  --start-from ALIAS \
  --use-openai-head-review \
  --head-review-model gpt-4o
```

This runs alias resolution, Stage B, and Stage C. It does not call the Stage A
extraction model. Repeat with the correct corpus root, project, mode, and
variant for all seven projects when alias logic changes globally.

To rerun only Stage C after a materialization-code change:

```bash
python scripts/knowledge_graph/kg_scripts/stage_c.py \
  --project-dir output/knowledge_graph/viet_full_unredacted
```

Stage C must be rerun for redacted CIB and FISC projects as well if those
projects are intended to remain current.

## 16. Validation Checklist

For each KG project, verify:

1. `extraction/summary.json` exists and counts are nonzero.
2. `extraction_run/chunk_errors.jsonl` is absent, empty, or understood.
3. `alias/summary.json` reports the expected OpenAI head-review setting.
4. `alias/alias_map.csv` maps every base entity row.
5. Important aliases such as USSR/Soviet Union are consistent where both occur.
6. Bad compound merges such as country versus embassy/office are absent.
7. `extraction_canonicalized/alias_application_manifest.json` exists.
8. `graph/summary.json` has document, segment, entity, event, and claim nodes.
9. `tasks/summary.json` lists all matcher compatibility files.
10. Every compatibility file has headers, even when a true task view has zero
    rows for a single-source corpus.

Useful checks:

```bash
python -m py_compile scripts/knowledge_graph/kg_scripts/*.py

python scripts/knowledge_graph/kg_scripts/run_pipeline.py --help

python scripts/knowledge_graph/kg_scripts/alias_resolution.py \
  --project-dir output/knowledge_graph/viet_full_unredacted \
  --rebuild-head-review-shortlist-only
```

The shortlist-only command is an advanced diagnostic. It should not replace a
full alias materialization before handoff.

## 17. Troubleshooting

### Stage A appears stalled

Stage A calls OpenAI per chunk. A large or difficult chunk, transport retries,
and provider latency can make progress uneven. Inspect the latest lines of
`extraction_run/chunk_results.jsonl` and `chunk_errors.jsonl`. Do not assume a
long item time means the process is dead.

### Wi-Fi failed and chunks were skipped

Rerun the same Stage A command. Failed chunks are not finalized in the success
checkpoint. Successful chunks are reused.

### Alias candidate scoring takes a long time

Large corpora can generate millions of locally scored candidate pairs. That
phase is CPU work, not millions of OpenAI calls. The terminal separately labels
`Alias head review adjudication`; only that phase calls OpenAI, and only for
`llm_review` rows.

### A known alias did not merge

Inspect, in order:

1. both labels exist in `extraction/base_entity_profiles.csv`;
2. the pair appears in `alias/alias_candidates.csv`;
3. its hard-conflict, type, acronym, lexical, and context fields;
4. whether it reached `head_review_candidates.csv`;
5. the final decision in `head_review_decisions.csv`; and
6. whether both base keys map to one ID in `alias_map.csv`.

If the pair never became a candidate, candidate blocking or acronym/full-form
rescue is the likely issue. If it was rejected, the row explains which evidence
or conflict controlled the decision.

### A bad alias merged

Find the canonical cluster in `alias_map.csv`, then identify the accepted link
in `alias_edges.csv`. Determine whether the source was a local auto-merge,
deterministic head merge, or OpenAI review. Fix a reusable type/conflict or
context rule rather than adding a corpus-row exception.

### Stage C redaction count looks unexpectedly large

Check both `short_redaction_rows`/`sentence_redaction_rows` and
`matcher_compatibility_rows` in `tasks/summary.json`. The compatibility file
also contains entity-context evidence and therefore can be much larger than the
true paired-redaction view.

### Workbench reads stale aliases

The runtime prefers `alias/corpus_entity_resource_v1.json` adjacent to the KG
project. After changing aliases, rerun alias resolution and Stage C. Delete only
the generated workbench normalized-resource cache if a fingerprint audit shows
it is stale; normal runs invalidate it by content fingerprint.

## 18. What To Ship

A functional handoff should retain:

- `data/raw/` if collaborators must rebuild preparation;
- `data/prepared/` to avoid repeating OCR and pair alignment;
- the seven current `output/knowledge_graph/` projects;
- `data/resources/` fallback resources;
- all supported scripts;
- this manual and the workbench manual; and
- root `README.md` and `requirements.txt`.

Do not ship alias backup directories, temporary OCR renderings, `__pycache__`,
`.DS_Store`, or abandoned benchmark experiments. Intermediate transcription
staging is optional: retain it only if resume/audit value outweighs its size.

## 19. Script Responsibility Map

| Script | Responsibility | Called directly? |
| --- | --- | --- |
| `prepare_vietnam_corpus.py` | Vietnam raw CSV to prepared corpus | Yes |
| `prepare_iran_corpus.py` | Iran raw CSV to prepared corpus | Yes |
| `prepare_canberra_corpus.py` | Canberra JSON to prepared corpus | Yes |
| `prepare_cib_corpus.py` | CIB pair OCR/alignment/preparation | Yes |
| `prepare_fisc_corpus.py` | FISC pair OCR/alignment/preparation | Yes |
| `prepare_single_source_corpus.py` | shared Vietnam/Iran materializer | Usually imported |
| `run_pipeline.py` | orchestrates A, alias, B, C | Yes |
| `stage_a.py` | extraction CLI | Via runner or advanced direct use |
| `alias_resolution.py` | Stage 1 and Stage 2 aliasing | Via runner or advanced direct use |
| `stage_b.py` | graph materialization | Via runner or advanced direct use |
| `stage_c.py` | task evidence materialization | Via runner or advanced direct use |
| `_extract_core.py` | extraction implementation | Imported |
| `_context_profiles.py` | base entity context/profile construction | Imported |
| `_prepare_pairs.py` | pair corpus loading | Imported |
| `_prepare_fisc.py` | FISC-specific prepared loading | Imported |
| `_prepare_generic.py` | generic prepared loading | Imported |
| `_common.py` | shared I/O, progress, normalization | Imported |
| `openai_pdf_pipeline.py` | pair PDF transcription/alignment helpers | Imported |
| `postprocess_redaction_labels.py` | pair-diff to prepared redaction labels | Imported |

This is the full supported KG code surface. There is no separate Stage 2 alias
script in the shipped workflow; Stage 1 and Stage 2 are both implemented and
audited inside `alias_resolution.py`.
