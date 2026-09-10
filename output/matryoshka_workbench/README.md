# Matryoshka Workbench Outputs

Final experiments are written under `benchmarks/` and reusable corpus/task
caches under `.matryoshka_cache/`. The supported A/B runner creates sibling
`__original` and `__rephrased` experiments.

Use `summaries/analyst_report/` for outward-facing review. Detailed case-level
parser, baseline, and holdout audits remain under each experiment's `runs/`
directory for reproducibility and Streamlit.

See `docs/MATRYOSHKA_WORKBENCH_MANUAL.md` for the output contract.
