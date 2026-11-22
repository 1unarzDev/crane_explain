"""WS-I integration pipeline: sealed schedule -> FieldLog sessions -> loader -> featurize -> sys-ID gate -> frozen
predictors -> PredTables -> LOAO conformal + explanations -> run_frozen -> results.json -> paper.

Entry points: `python -m capxfer.pipeline.run_all` (the post-collection command), `python -m capxfer.pipeline.freeze`
(hash manifest), `python -m capxfer.pipeline.dryrun` (SIMULATION-labelled pseudo-real collection for the dry run).
Adapters for interface mismatches between workstreams live in `adapters.py` (see docs/coordination/I.md).
"""
