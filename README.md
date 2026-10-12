# Knowledge Graph and Matryoshka Analyst Workbench

This directory is a self-contained research handoff package. It can be renamed,
moved into an otherwise empty repository, or archived and unpacked elsewhere.
All supported commands resolve `data/`, `scripts/`, and `output/` relative to
the directory containing this README. No parent repository is required.

The package contains two connected systems:

1. The knowledge-graph pipeline prepares source corpora and builds extracted,
   alias-resolved, canonicalized, graph, and task-specific evidence artifacts.
2. The Matryoshka workbench generates controlled benchmark queries, executes
   baseline and source-document-holdout retrieval, creates one-shot semantic
   analyst-style rewrites, and compares the original and rewritten outcomes.

The canonical manuals are:

- [`docs/KNOWLEDGE_GRAPH_MANUAL.md`](docs/KNOWLEDGE_GRAPH_MANUAL.md)
- [`docs/MATRYOSHKA_WORKBENCH_MANUAL.md`](docs/MATRYOSHKA_WORKBENCH_MANUAL.md)

## Iran Analyst Demo

The repository also ships a browser-only Iran corpus demonstration in
[`site/`](site/). It combines a hierarchical Matryoshka graph explorer with a
three-task evidence workbench for reconstruction, hypothesis assessment, and
person redaction.

After GitHub Pages is enabled with **GitHub Actions** as its source, pushes to
`main` deploy:

<https://jimhcyang.github.io/matryoshka-kg-workbench/>

The hosted build replays nine audited recorded cases and three distinct
precomputed live-retrieval packets. Arbitrary new queries and complete
community expansion require the full local research workspace; the hosted UI
labels those limits rather than simulating execution. See
[`docs/IRAN_SITE_VISIT_DEMO.md`](docs/IRAN_SITE_VISIT_DEMO.md).

### Optional Visitor Registration

The landing page contains an optional blurred email-registration gate. It is
disabled until `site/visitor-config.js` contains a deployed Google Apps Script
`/exec` URL and `enabled: true`. The spreadsheet-bound endpoint and deployment
instructions are in
[`integrations/visitor_registration/`](integrations/visitor_registration/).

This records consented visits in a private Sheet; it is not authentication and
does not make the public GitHub Pages site private. No Sheet credential, Sheet
ID, API key, or GitHub write token is shipped to the browser.

## Requirements

- Python 3.11 is recommended.
- An OpenAI API key is required for Stage A extraction, optional alias head
  review, runtime query parsing when enabled, and semantic query rewriting.
- Poppler's `pdftoppm` executable is required when rebuilding FISC from PDFs.

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

On macOS, install Poppler with `brew install poppler`. On Debian or Ubuntu,
use `sudo apt-get install poppler-utils`.

## Package Layout

```text
.
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/                 immutable source copies
|   |-- prepared/            normalized Stage A inputs
|   |-- resources/           fallback corpus semantic resources
|   `-- manifests/           corpus metadata
|-- docs/                    the two canonical manuals
|-- scripts/
|   |-- knowledge_graph/     corpus preparation and KG stages
|   `-- matryoshka_workbench/ benchmark build, run, UI, and reports
`-- output/
    |-- knowledge_graph/     seven built KG projects
    `-- matryoshka_workbench/ benchmark experiments and caches
```

The shipped KG layout contains these seven project variants:

- `viet_full_unredacted`
- `cib_full_unredacted`
- `cib_full_redacted`
- `iran_full_unredacted`
- `fisc_full_unredacted`
- `fisc_full_redacted`
- `canberra_full_unredacted`

FISC and CIB are genuine redacted/unredacted pair corpora. Vietnam, Iran, and
Canberra are single-source corpora and therefore expose only unredacted KG
projects. Canberra is materialized into the common prepared directory shape,
but its redacted and unredacted prepared texts are identical.

## Rebuild A Corpus

Preparation converts raw source files into the common `data/prepared/<corpus>`
shape. Examples:

```bash
python scripts/knowledge_graph/kg_scripts/prepare_vietnam_corpus.py --overwrite
python scripts/knowledge_graph/kg_scripts/prepare_iran_corpus.py --overwrite
python scripts/knowledge_graph/kg_scripts/prepare_canberra_corpus.py --overwrite
python scripts/knowledge_graph/kg_scripts/prepare_cib_corpus.py --overwrite --openai-model gpt-4o-mini
python scripts/knowledge_graph/kg_scripts/prepare_fisc_corpus.py --overwrite --openai-model gpt-4o-mini
```

Then run Stage A, alias resolution, Stage B, and Stage C. This example builds
the CIB unredacted project and enables strong-model review only for ambiguous,
high-value alias candidates:

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

Use `--transcription-variant redacted` and a distinct redacted project path for
CIB or FISC. Use `--start-from ALIAS` to preserve completed extraction and
rerun alias resolution, graph construction, and Stage C materialization.

## Run The Final A/B Benchmark

The supported final workflow compares original queries with one-shot semantic
analyst-style rewrites. It defaults to all five corpora and all five tasks.
Build-stage selection is local and does not send candidate catalogs to OpenAI.
The `--use-openai-parser` flag enables OpenAI only for the selected queries at
runtime. The one `--openai-model` value is shared by parsing, rewriting, and
rewrite diagnostics. The wrapper default is 10 cases per corpus/task; the
explicit final command below overrides it to 20.

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

This creates sibling experiments:

```text
output/matryoshka_workbench/benchmarks/
|-- five_corpus_pair_20x5_semantic_final__original/
`-- five_corpus_pair_20x5_semantic_final__rephrased/
```

The primary outward-facing files are under each branch's
`summaries/analyst_report/`. The rephrased branch also contains the A/B case
comparison and aggregate comparison. Detailed case audits remain under
`runs/` for reproducibility and Streamlit inspection.

## Open The Workbench

```bash
streamlit run scripts/matryoshka_workbench/release_v0421/streamlit_semantic_workbench_v3_19.py
```

In the app, open a case directory under an experiment's `runs/` directory. A
pair run containing `baseline/` and `holdout/` is rendered side by side.

## Supported Entry Points

The public surface is deliberately small.

Knowledge graph:

- `prepare_vietnam_corpus.py`
- `prepare_iran_corpus.py`
- `prepare_canberra_corpus.py`
- `prepare_cib_corpus.py`
- `prepare_fisc_corpus.py`
- `run_pipeline.py`

Workbench:

- `run_rephrase_ab_benchmark.py`
- `multicorpus_benchmark.py`
- `summarize_benchmark_results.py`
- `streamlit_semantic_workbench_v3_19.py`

The remaining Python files are implementation modules imported by those entry
points. They are not separate alternative workflows.
