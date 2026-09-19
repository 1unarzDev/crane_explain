# Architecture

## Trust boundary

The core transforms immutable episode evidence into explicitly supported propositions. A language
realizer may only express those propositions and explicit limitations. The final text—not merely
the internal plan—is checked. On failure, one bounded repair may be attempted by a future LLM
adapter; otherwise the deterministic checked answer is returned.

Decision-time evidence and later outcomes are physically separate fields. Changing an outcome
cannot change a decision rationale. Candidate state distinguishes selected, evaluated/rejected,
infeasible, known-not-considered, and unknown. `not_considered` is used only when recorded.

## Evidence flow

1. CRANE/run harness supplies episode, run, tick, sensor/observation identity, and explicit client
   cancellation/deadline events.
2. The ROS observer records Nav2 action goal/feedback/result and BT status transitions, plus the
   exact BT XML/version. It does not infer internal consumption from nearby topic values.
3. The core validates identity, chronology, completeness, candidate, and policy invariants.
4. Reasoning constructs claims with evidence IDs, derivation, temporal scope, assumptions, support
   status, and evidence level.
5. A realizer emits language; the verifier checks final sentences. Unverified language falls back
   to templates.
6. Evaluation compares responses with evaluator-only truth stored outside all model-visible paths.

## Evidence levels

1. recorded sequence;
2. reconstructed software execution mechanism from BT semantics plus transitions;
3. explicit-model causal inference;
4. controlled-intervention result.

The initial submission targets levels 1–2. Temporal succession is not physical causation.

## Boundaries

- **CRANE:** simulation truth, episode identity, observations, action application, later outcomes.
- **Nav2:** planner/controller/behavior/BT execution. BT transitions show software mechanism.
- **ROS capture:** passive serialization only; it must not rewrite history or invent consumption.
- **Core:** schema, validation, checked reasoning, answerability, generation policy, verification.
- **LLM:** optional realization/extraction component, never the trust boundary.
- **Evaluator truth:** fault injection and gold propositions, inaccessible to explanation methods.

## Why no C++ BT hook

The locally installed Jazzy packages (`nav2_msgs` and `nav2_bt_navigator` 1.3.12) expose
`BehaviorTreeLog` and `NavigateToPose` feedback/result. The installed `RosTopicLogger` publishes
node name, UID, previous status, and current status on `behavior_tree_log`; feedback exposes
navigation time, ETA, recovery count, and distance remaining. The static BT XML supplies control
semantics. This is enough to pilot levels 1–2 without changing Nav2.

A narrow hook is reconsidered only after a documented pilot evidence gap blocks a primary RQ,
cannot be observed elsewhere, cannot be handled by narrowing the claim, and is worth more than
additional independent episodes.

