# Matryoshka Workbench Scripts

The canonical operating and architecture manual is
[`../../docs/MATRYOSHKA_WORKBENCH_MANUAL.md`](../../docs/MATRYOSHKA_WORKBENCH_MANUAL.md).

The supported final workflow is
`tools/run_rephrase_ab_benchmark.py`: original queries versus one-shot semantic
analyst-style rewrites, each executed in baseline and source-document-holdout
mode. `tools/multicorpus_benchmark.py` and
`tools/summarize_benchmark_results.py` are lower-level supported entry points.
All other Python files are imported implementation modules.

Run commands from the package root. Paths in config files are resolved against
that root, so the package directory may be renamed or moved.
