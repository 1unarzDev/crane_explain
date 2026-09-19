# CRANE Explain

Evidence-checked natural-language explanations of autonomous robot navigation decisions and
failures. The research target is TRUSTMORE 2026 (deadline: **October 4, 2026 AoE**). The central
failure mode is fluent but unsupported language—not awkward wording.

The system keeps five stages explicit:

`robot decision/execution → retained evidence → checked propositions → language → final-text check`

The Python core has no ROS, Unity, network, GPU, or LLM dependency. `crane_explain_ros` provides
passive Nav2 capture separately. Evaluator-only truth must never enter model-visible episode files.

## Current status

- **IMPLEMENTED, TESTED:** immutable typed evidence records, validation, checked contrastive and
  recovery-count plans, conservative deterministic realization, strict final-text checking,
  Dock/Slalom regression cases A–E, and the information-parity-aware A/B/C/D/E harness.
- **TESTED:** local Nav2 Jazzy 1.3.12 interfaces and installed `RosTopicLogger` header confirm the
  passive `behavior_tree_log` route.
- **NOT_RUN:** live ROS capture and pilot CRANE episode collection.
- **DEFERRED:** arbitrary-LLM proposition extraction until it can be independently evaluated.

## CPU-only setup and demo

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
crane-explain validate research/explanation_fidelity/fixtures/dock_policy.json
crane-explain explain research/explanation_fidelity/fixtures/dock_policy.json --alternative Slalom
```

The demo emits the checked answer plan, realized text, and final-text verification result as JSON.
The policy in that fixture is synthetic and exists only to test arithmetic; it is not CRANE's
policy.

## Repository layout

- `src/crane_explain/`: evidence, validation, reasoning, realization, verification, CLI
- `tests/`: scientific regression tests
- `research/explanation_fidelity/`: fixtures, later splits/configs/results
- `docs/`: architecture, benchmark, study, experiment log, research notes, decisions
- separate repository `crane_explain_ros`: ROS 2 Jazzy/Nav2 observer

## Related runtimes

The checked core is developed beside:

- CRANE simulator: <https://github.com/1unarzDev/crane_ml>
- ROS environment: <https://github.com/1unarzdev/astro_dock>

CRANE's existing full Nav2 fixture can be run from its repository with:

```bash
Tools/Performance/run_nav2_controller_fixture.sh
```

Do not call a fixture client `goalAttempts` value a recovery count. Do not call replay a
counterfactual execution. Do not claim delivered odometry was internally consumed by Nav2.

## Regenerating results

At this initial checkpoint, the only generated result is the test report (14 tests):

```bash
python -m pytest -q
```

Pilot, benchmark, statistics, figure, and paper commands will be added only when their executable
paths exist. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the run ledger and
[docs/STUDY_DESIGN.md](docs/STUDY_DESIGN.md) for freeze gates.

## Troubleshooting

- `ModuleNotFoundError`: install editable (`pip install -e .`) or run tests from this root.
- Validation rejects a score without a policy expression: retain the policy or omit the score; a
  naked number cannot support a policy-based reason.
- LLM wording is rejected: use the checked template fallback. Unparsed clauses do not pass.
- ROS topic absent: verify `bt_navigator` is active, namespace/remapping, ROS domain, and DDS IPC.
  Topic discovery alone does not prove message delivery across isolated container IPC namespaces.
