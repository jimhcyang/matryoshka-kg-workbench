# Knowledge Graph Scripts

The canonical operating and architecture manual is
[`../../docs/KNOWLEDGE_GRAPH_MANUAL.md`](../../docs/KNOWLEDGE_GRAPH_MANUAL.md).

Supported public entry points are the five `kg_scripts/prepare_*_corpus.py`
wrappers and `kg_scripts/run_pipeline.py`. The underscored modules, stage
scripts, and `upstream_pair_processing/` files are implementation dependencies
of those entry points, not competing workflows.

Run commands from the package root, which is the directory containing the
top-level `README.md`. All default paths are package-relative and continue to
work if that directory is renamed or moved.
