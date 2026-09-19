# Study Design (pre-final-collection draft)

This document must be frozen after pilot/power analysis and before sealed-test evaluation. Items
marked **TBD-PILOT** are not yet frozen.

## Research questions and hypotheses

- RQ1: structured versus information-matched prose evidence.
- RQ2: checked plans/final verification versus direct generation and self-checking.
- RQ3: useful partial answers and correct withholding under insufficient/contradictory evidence.
- RQ4: trustworthiness at useful coverage, including matched-coverage comparisons.

Primary H1: method D reduces response-level material errors versus strong prose baseline A while
maintaining useful substantive coverage. Secondary: B>A representation effect; D>B checking
effect; D versus C native capture effect; D improves insufficient-evidence, contrastive, and
causal-claim handling. E versus D tests whether LLM realization adds enough usefulness to justify
its risk.

## Conditions

- A: well-written free-text record → direct LLM.
- B: structured record with exactly the same facts → same direct LLM.
- C: free text → extraction → checked generation.
- D: native structure → checked plan → generation → output verification/fallback.
- E: structure → deterministic checked template.

Every episode is captured once. A/B information parity is audited proposition-by-proposition;
neither format receives privileged facts. Model, decoding, system prompt, and answer prompt are
identical for A/B. Calls are single-sample and cached with raw/final output, model/version,
parameters, prompt hash, latency, tokens, cost, and verification result.

## Primary outcome

A response has a material error if any substantive assertion is contradicted/unsupported, has an
incorrect comparison/count/status, or makes an unjustified causal, counterfactual, or explanatory
relationship. Primary unit: response. Annotation is blind to condition where formatting permits.

Jointly report substantive answer coverage, full/partial/abstained proportions,
answerable-information coverage, correct abstention, risk–coverage curves, and matched-coverage
comparisons. Claim-level error categories are secondary. Fluency/preference is not correctness.

## Units, split, and inclusion

Independent unit is scenario instance/episode, never a paraphrase. Split by scenario family before
question realization; all paraphrases stay together. Include episodes passing frozen capture
quality checks and questions mapped to frozen families. Exclude corrupt/missing required artifacts
using reasons set before inspecting answers. Deliberate incomplete-evidence episodes remain in
scope. **TBD-PILOT:** minimum evidence per family and target family balance.

## Analysis

Use paired episode-cluster bootstrap confidence intervals for risk and coverage differences and a
mixed-effects logistic model if sample size/convergence supports it (condition fixed effect;
episode/scenario-family random effects). McNemar is a secondary response-pair sanity check, not the
main clustered inference. Report effect sizes and intervals regardless of significance. Adjust
secondary confirmatory comparisons with Holm; exploratory analyses are labeled.

Power planning uses development episodes only: estimate paired discordance/effect, define the
smallest practically meaningful reduction before seeing sealed test, simulate the clustered paired
design, and target ≥80% power (prefer 90%). **TBD-PILOT:** effect, ICC, threshold, episode count.

## Freeze and stopping rules

Before final collection freeze: metric/annotation guide, H1/A-vs-D comparison, prompts, model,
verifier threshold, templates, instrumentation, inclusion rules, split manifest, seeds, and
analysis code. Hash frozen files. Do not inspect sealed labels to tune. Collection stops at the
power-planned independent episode target or the September 28 deadline/compute ceiling, whichever
comes first; report shortfall honestly. Essential experiments do not run October 4.

## Leakage prevention

Robot-visible and evaluator-only artifacts live in physically separate roots with opaque IDs.
Fault names and gold outcomes never appear in model-visible filenames, metadata, prompts, or
retrieval indexes. A release builder performs an allowlist copy and scans for forbidden fields.

