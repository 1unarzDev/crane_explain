# CRANE Explanation Benchmark

## Episode artifacts

Each opaque episode ID has two separately rooted artifacts:

- robot-visible: evidence records and parity-matched prose;
- evaluator-only: injection, ground truth, expected propositions, answerability, annotations.

The visible schema contains provenance, completeness, decision-time candidates/policy/consumed
inputs when known, BT/action execution events, unique recovery-attempt IDs, and later outcome. It
never labels simulator truth as robot knowledge.

## Scenario taxonomy

Success; contextual recovery→success; repeated recovery; terminal navigation failure; planning
failure; reproducible controller/progress failure; client cancellation (distinct from task/BT
timeout); incomplete/omitted/contradictory evidence; false premise; decision plus unrelated later
failure; equivalent outcomes through different software mechanisms.

## Question taxonomy

Factual execution, contrastive, failure attribution, temporal, alternative status, misleading
premise, insufficient evidence, and unsupported counterfactual. Human-motivated families may be
mapped from Wachowiak et al.; their question corpus is not treated as a fidelity ground truth.

## Gold proposition format

Each proposition records text-independent predicate/arguments, support status, required/forbidden
evidence IDs, derivation, temporal scope, evidence level, assumptions, material-error category,
and answerability. Counts distinguish `at least N recorded` from `exactly N occurred`.

## Mandatory regression suite

- A: Dock selected with no policy/scores → no highest-utility or causal claim.
- B: explicit **synthetic test-only** policy produces 8.3/6.8/7.9 and supports the checked Dock
  versus Slalom contribution comparison.
- C: duplicate messages with one attempt ID count once; two IDs count twice; incomplete history is
  qualified; the premise “both retries” is rejected.
- D: later outcome changes do not alter decision-time rationale.
- E: infeasible, unevaluated, missing-score, tied, near-tied, contradictory-selection cases.

Current executable coverage is in `tests/test_dock_slalom.py`. Near ties preserve exact recorded
scores and are not described as ties; the practical-equivalence threshold remains a pilot design
choice and must be frozen rather than selected from final outcomes.
