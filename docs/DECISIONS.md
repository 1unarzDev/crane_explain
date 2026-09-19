# Decision Log

## 2026-09-19 — strict template verification as initial trust boundary

- Decision: accept only final sentences exactly licensed by a checked plan; reject arbitrary extra
  clauses and fall back to deterministic realization.
- Evidence: no independently validated proposition extractor exists in the repository yet.
- Alternatives: regex fact checks alone; LLM self-check; permissive semantic similarity.
- RQ impact: provides a conservative D/E implementation and measurable coverage cost for RQ2/RQ4.
- Validity risk: exact matching understates achievable LLM coverage and favors templates.
- Revisit: after independent verifier evaluation on held-out external and CRANE propositions.

## 2026-09-19 — passive Nav2 observer, no native hook

- Decision: use Jazzy `BehaviorTreeLog`, `NavigateToPose`, exact BT XML, and harness events.
- Evidence: local package/interface/header inspection confirms required level-1/2 fields.
- Alternatives: BehaviorTree.CPP/C++ blackboard hook; parsing console logs.
- RQ impact: faster auditable capture with no simulator/navigation behavior changes.
- Validity risk: transient blackboard values and exact internal sensor consumption remain unknown.
- Revisit: only under the five evidence-gap criteria in `ARCHITECTURE.md` after pilot.

## 2026-09-19 — prioritize independent episodes over broad integrations

- Decision: external benchmarks are gated to roughly one day and only after CRANE capture health.
- Evidence: deadline is October 4; primary inference depends on independent scenario instances.
- RQ impact: maximizes power and reduces risk that many paraphrases masquerade as sample size.
- Validity risk: narrower external generalization evidence.

## 2026-09-19 — treat Nav2 lifecycle readiness as capture quality, not robot failure

- Decision: pilot/final runs must distinguish action-server startup rejection from a navigation
  failure and record the startup margin/readiness procedure. The locally validated temporary
  setting is `CRANE_FIXTURE_DELAY=15`; it is not yet a frozen collection rule.
- Evidence: default-delay run timed out after two inactive-server rejections and only ~8.15 s of
  accepted execution; changing only the pre-fixture delay produced terminal success.
- Alternatives: call the timeout a mission failure; increase the action deadline; patch Nav2.
- RQ impact: prevents infrastructure startup from contaminating failure labels and recovery counts.
- Validity risk: a fixed delay can conceal host-load variation; explicit lifecycle readiness is
  preferable before final collection.
