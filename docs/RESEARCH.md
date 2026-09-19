# Research and benchmark source audit

Last primary-source check: 2026-09-19. This note records only claims checked against a paper, an official proceedings/venue page, or an authors' repository. Recommendations for this project are labelled as such; they are not claims made by the cited authors.

## Immediate synthesis for RQ1--RQ4

- The strongest methodological precedent for the proposed pipeline is the separation of symbolic content planning from language realization. Moryossef et al. show that an explicit plan can improve semantic faithfulness without sacrificing judged fluency; their planner is not a verifier, however, and their WebNLG result should not be presented as evidence for robot explanations [Moryossef et al. 2019](https://doi.org/10.18653/v1/N19-1236).
- Provenance captured by design is a closer conceptual match than post-hoc log summarization. Huynh et al. explicitly separate requirements, provenance capture/querying, explanation plans, and realization. Their evidence is a two-scenario software-engineering case study, not an explanation-fidelity experiment [Huynh et al.](https://arxiv.org/abs/2206.06251).
- Generic NLI is useful but cannot be the trust boundary. Dušek and Kasner report useful cross-domain semantic-error detection, while Pramanick and Rossi show that off-the-shelf NLI models perform poorly on their robotics coherence labels until domain fine-tuning. Both motivate independent evaluation and deterministic checks for identifiers, counts, order, arithmetic, and status [Dušek and Kasner 2020](https://doi.org/10.18653/v1/2020.inlg-1.19), [Pramanick and Rossi 2024](https://doi.org/10.1109/IROS58592.2024.10802671).
- Atomic-claim evaluation is directly useful for claim-level precision, but FActScore's Wikipedia-biography estimator and its reported error rate do not transfer to robot evidence. Reuse the decomposition idea, evaluate proposition extraction separately on this domain, and report response-level material errors as the primary outcome [Min et al. 2023](https://doi.org/10.18653/v1/2023.emnlp-main.741).
- REFLECT/RoboFail is relevant external evidence for failure-explanation behavior, but its manipulated-object domain, ground-truth simulation perception, and human-rated “correct and informative” metric do not replace the paired CRANE/Nav2 experiment [Liu et al. 2023](https://arxiv.org/abs/2306.15724).
- Selective prediction formalizes the reason to report risk jointly with coverage. It does not supply an explanation-specific abstention policy, so this project must define answerable-information coverage and calibrate its verifier/abstention operating point on development data [Geifman and El-Yaniv 2019](https://proceedings.mlr.press/v97/geifman19a.html).

## Venue: TRUSTMORE 2026

Official sources: [workshop call](https://trustmoreai.github.io/workshop2026/), a stable [site-source snapshot](https://github.com/trustmoreai/workshop2026/blob/ddf08aca9e0c555a9e3955806ce6f51eb0012809/index.html), and the live [OpenReview invitation API](https://api2.openreview.net/invitations?id=IEEE.org%2FBigData%2F2026%2FWorkshop%2FTRUSTMORE%2F-%2FSubmission).

Verified requirements and dates:

- Full research papers are 8--9 pages including references; short/work-in-progress and system/demo papers are 4--6 pages including references. The IEEE BigData 2026 template is required.
- Work must be original, unpublished, and not simultaneously submitted elsewhere, consistent with IEEE policy.
- Review is double-blind. Remove author names, affiliations, and acknowledgments and refer to the authors' prior work in the third person. Anonymous supplementary code/data is encouraged where appropriate.
- The stated review criteria are technical quality, originality, significance, practical impact, relevance, clarity, and reproducibility; each submission is to receive at least two relevant program-committee reviews.
- The workshop page gives submission **2026-10-04 AoE**, reviews 2026-10-18, notification 2026-10-21, camera-ready only as “End of October,” and workshop date TBD during IEEE BigData, 2026-12-14 through 2026-12-17. Accepted papers are stated to appear in IEEE Computer Society Press workshop proceedings.
- The live OpenReview invitation's `duedate` was `1791183540000` when checked, i.e. **2026-10-05 06:59 UTC** (2026-10-05 01:59 America/Chicago), five hours earlier than the conventional end of October 4 AoE. Treat the portal timestamp as the operative latest cutoff and submit earlier; do not rely on the site's statement that an extension may occur.
- OpenReview currently requires author profiles and exposes title, authors, keywords, abstract, PDF (maximum 50 MB), optional TL;DR, and a CC BY 4.0 agreement. No separate supplement-upload field was visible in the live form during this audit.

Topical fit is direct: the official call names grounding, observability, error detection/failure recovery, auditability, safety/reliability evaluation, uncertainty/selective abstention, benchmark provenance, formal auditing, and reproducibility. Negative results and failure analyses are explicitly encouraged.

Unresolved: the exact camera-ready deadline, workshop day, appendix/supplement policy, registration/presentation obligation, and whether the portal deadline will be reconciled with literal AoE. The generic IEEE template page could not be verified from this environment because of its web-application firewall; do not import page limits or review rules from the main conference.

## Question taxonomies and execution retrieval

### Wachowiak et al., *What Questions Should Robots Be Able to Answer?*

Publication: Lennart Wachowiak, Andrew Coles, Gerard Canal, and Oya Celiktutan, *ACM Transactions on Human-Robot Interaction*, online 2026-07-20, article 3832777, [DOI 10.1145/3832777](https://doi.org/10.1145/3832777), [arXiv:2510.16435](https://arxiv.org/abs/2510.16435), and the authors' [dataset/analysis repository](https://github.com/lwachowiak/xai-questions-dataset).

Verified findings:

- The study collected 2,037 questions, excluded 144, and retained 1,893 questions from 100 participants. Prompts used 15 video and 7 text stimuli depicting household-robot situations. The final taxonomy has 12 main and 70 subcategories.
- The most frequent main categories were execution details (21.4%), abilities (12.6%), and self/task assessment (10.7%); why-questions were 9.8%. Explicit contrasts occurred in 28% of why-questions.
- Participants assigned the highest importance to potential-issue questions (estimated marginal mean 4.01, 95% CI [3.82, 4.20], n=118), compared with why-questions at 3.58 [3.42, 3.75], n=190. This supports including prospective-risk questions even though the evidence system will often have to withhold counterfactual answers.
- A second annotator independently recoded a balanced subset of 96 questions, not the whole dataset: agreement was 79% and Cohen's kappa 0.77, 95% CI [0.68, 0.86].

Project implication: use the taxonomy to select human-motivated question families, audit what the evidence schema can answer (RQ1/RQ3), and report answerable-information coverage across those families (RQ4). It is not an answer-fidelity benchmark: it has no robot execution ground truth or gold answers and provides limited direct evidence for RQ2.

Fit and cost: high ecological-validity value and very low integration cost (the repository exposes a semicolon-delimited CSV and Python 3.12 analysis). A curated navigation-compatible subset and held-out templates should take less than half a day after core schemas stabilize.

Unresolved/limitations: the study itself notes a mostly Western/Anglophone Prolific sample, video/text rather than live interaction, ambiguous intent, no repeated interaction, and few interaction ruptures. Most taxonomy coding was by one annotator. Exact THRI volume/issue/pages were not present in the checked primary metadata and should not be invented.

### Wachowiak et al., *Neurosymbolic Explanation Selection in Robotics*

Publication: Lennart Wachowiak, Andrew I. Coles, Oya Celiktutan, and Gerard Canal, *Companion Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction* (HRI Companion 2026), Edinburgh, 2026-03-16--19, pp. 222--227, [DOI 10.1145/3776734.3794387](https://doi.org/10.1145/3776734.3794387), [accepted manuscript](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/363027894/2026_Neurosymbolic_Explanation_Selection.pdf), and [evaluation repository](https://github.com/lwachowiak/explanation-selection-eval).

Verified findings:

- The pipeline associates JSONL logs and heterogeneous XAI artifacts with PDDL plan steps, selects steps relevant to a natural-language question, and consolidates their evidence into a response.
- The retrieval evaluation contains 180 manually authored questions over six plans in office-robot and crew domains (30 questions per plan; plan length 14--51, mean 31, SD 13), balanced across six question categories.
- Command A operating on raw PDDL achieved F1 0.91 (reported precision and recall both 0.93), versus EmbeddingGemma-300m at F1 0.62 and the PlanVerb syntax/keyword matcher at 0.02. The paper reports that many LLM errors selected too many instances (36.4%) or missed instances (22.7%).
- In a preliminary N=30 study, mean helpfulness ranks were 1.35 for LLM summaries, 2.13 for raw logs, and 2.52 for PlanVerb explanations, with reported pairwise p<0.05. Preference/helpfulness is not correctness.

Project implication: use the public question/plan/gold-step files for an external execution-retrieval/entity-resolution comparison only. It is relevant upstream evidence for RQ1/RQ2 but has no answer-level factuality gold labels. The prompt requests an absent-information response, but the study does not evaluate calibrated abstention (RQ3) or risk--coverage (RQ4).

Fit and cost: low (likely under one day) because the repository supplies six plan/question/gold-ID sets and a notebook under an MIT code license; rerunning Command A requires its provider/model. Defer until the CRANE benchmark is healthy.

Unresolved/limitations: questions are handcrafted from a taxonomy; authors report no tuning and no train/test split; one LLM and one embedding model are evaluated. The full ROS/XAI system is not released in the repository, and raw user-study responses were not located.

## Robot failure explanation and external evaluation data

### Liu, Bahety, and Song, *REFLECT: Summarizing Robot Experiences for FaiLure Explanation and CorrecTion*

Publication: Proceedings of the 7th Conference on Robot Learning (CoRL 2023), PMLR 229:3468--3484; [official proceedings page](https://proceedings.mlr.press/v229/liu23g.html), [arXiv:2306.15724](https://arxiv.org/abs/2306.15724), [official project page](https://robot-reflect.github.io/), and [authors' code repository](https://github.com/real-stanford/reflect). The project page's former `columbia-ai-robotics/reflect` link currently redirects to `real-stanford/reflect`.

Verified findings:

- RoboFail contains 100 manually injected simulation failure scenarios (10 cases for each of 10 AI2THOR tasks) and 30 real-world scenarios across 11 tasks collected by teleoperating a UR5e. Simulation artifacts include RGB-D, 20 sound classes, robot states, and simulator ground truth; real-world artifacts include RGB-D, sound, and proprioception.
- The method builds sensory-input, event-based, and subgoal-based summaries, then progressively checks subgoals before analyzing execution or planning failures. This is a hierarchy of summaries and LLM reasoning, not evidence-constrained proposition verification.
- The paper reports simulation explanation/localization/correction-plan success of 88.4/96.0/79.1% for execution failures and 84.2/80.7/80.7% for planning failures. “Explanation” is the percentage judged correct and informative by human evaluators. These results use GPT-4 and ground-truth object/state detection in simulation.
- The authors identify limitations in scene-graph heuristics, assumed candidate object-state lists, a static-environment assumption, and weak handling of low-level control failures.

Project implication: RoboFail can test whether the proposed verifier rejects unsupported clauses in existing/generated failure explanations (RQ2/RQ3). It cannot support RQ1's same-information structured-versus-prose comparison without constructing parity-controlled presentations, and it cannot substitute for navigation-domain episodes. Keep evaluator metadata separate from model-visible inputs.

Fit and cost: medium relevance, medium adaptation. The official repository provides code and a demo; the project page's dataset endpoint returned HTTP 403 from this environment on 2026-09-19. Confirm dataset acquisition before scheduling work. A narrow text/explanation subset is likely cheaper than reproducing AI2THOR perception and correction execution.

Unresolved: obtain and inventory the released dataset; determine its license independently of the MIT-licensed code; inspect whether all human judgments and evaluator-only failure annotations are included; confirm whether failure explanations can be paired with exactly delimited allowed evidence.

### Pramanick and Rossi, *Multimodal Coherent Explanation Generation of Robot Failures*

Publication: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024), pp. 2487--2493, [DOI 10.1109/IROS58592.2024.10802671](https://doi.org/10.1109/IROS58592.2024.10802671), [arXiv:2410.00659](https://arxiv.org/abs/2410.00659), and [authors' code/data](https://github.com/pradippramanick/coexp-iros24).

Verified findings:

- The work casts cross-modal coherence as three-way classification: contradiction, entailment, or non-entailment between a textual explanation and either a graphical scene representation, plan, or observation.
- The authors manually annotate 260 RoboFail-derived examples and add counterfactual examples to address class imbalance. They use a 70:10:20 stratified split and add held-out examples from unseen task types, producing a final test set of 364 data points.
- On graphical/text pairs, off-the-shelf RoBERTa-large-MNLI and DeBERTa-v3-base-NLI obtain macro-F1 0.29 and 0.23; the NLI-pretrained then robotics-fine-tuned model obtains 0.87. It obtains macro-F1 0.86 for plan/text and observation/text pairs, and 0.59 on graphical/text pairs from task types unseen during training. These are coherence-classification results, not factual-faithfulness rates for complete answers.
- The released repository contains labeled JSON, fixed train/validation/test CSVs, evaluation scripts, and fine-tuning scripts. Its README says counterfactual-generation code is not released and must be requested. The data README warns that supplied semantic-role parses are imperfect.

Project implication: this is a strong external test for the learned proposition/NLI layer and a direct falsification test for any claim that generic NLI suffices (RQ2). Compare generic NLI, their domain-tuned checkpoint or reproducible fine-tuning, and the evidence-specific verifier. Never equate cross-modal coherence with support by allowed robot evidence.

Fit and cost: high relevance, low-to-medium adaptation for classification-only evaluation; defer multimodal refinement generation. The small released data and scripts make a one-day compatibility spike plausible, but GPU/model download requirements must be measured first.

Unresolved: repository does not declare a license in GitHub metadata as checked on 2026-09-19; obtain permission or restrict use accordingly. Verify that the published split can be reproduced exactly with current dependency/model versions and determine whether duplicate or near-duplicate source episodes cross splits.

## Planning, provenance, and faithfulness verification

### Moryossef, Goldberg, and Dagan, *Step-by-Step: Separating Planning from Realization in Neural Data-to-Text Generation*

Publication: NAACL-HLT 2019, pp. 2267--2277, [DOI 10.18653/v1/N19-1236](https://doi.org/10.18653/v1/N19-1236), [arXiv:1904.03396](https://arxiv.org/abs/1904.03396), and [authors' Chimera repository](https://github.com/AmitMY/chimera).

Verified findings:

- The method explicitly separates symbolic ordering/structuring from neural realization and learns a plan-to-text realizer on WebNLG.
- In a manual evaluation of 440 RDF triples from 139 seen-domain input sets, the plan-based system had 6 omissions, 17 wrong lexicalizations, and 3 over-generations versus 41, 39, and 29 for the strong end-to-end neural system: reported reductions of 85%, 56%, and 90%. Pairwise crowd evaluation found fluency on par with that neural system. This is one dataset and the faithfulness annotation was performed blind to system identity by the first author.
- The realization layer still fails: best-plan entity coverage was 98.9% on seen and 66.7% on unseen inputs, although outputs containing all requested entities preserved their plan order in the reported evaluation.

Project implication: directly motivates checked answer plan → language realization → final-text verification (RQ2). The non-perfect plan realization result is particularly important: checking the plan alone is insufficient. Use deterministic templates as condition E because the paper also notes that a grammar system was fully faithful by design and therefore excluded from its plan-versus-neural faithfulness comparison.

Fit and cost: conceptual baseline only; reproducing WebNLG training is low value for the deadline. Cite the released MIT-licensed implementation, but implement a domain-specific typed answer plan rather than importing Chimera/OpenNMT.

Unresolved: whether modern instruction-tuned realization preserves domain plans better at fixed prompts; this must be measured on CRANE outputs, not assumed from WebNLG.

### Huynh et al., *A Methodology and Software Architecture to Support Explainability-by-Design*

Publication status verified here only as an arXiv preprint, [arXiv:2206.06251](https://arxiv.org/abs/2206.06251); [authors' example artifacts](https://github.com/plead-project/EbD-artefacts) and [reference service](https://explain.openprovenance.org/).

Verified findings:

- The methodology has explanation-requirement analysis, explanation technical design, and explanation validation phases. The technical workflow records application decisions as provenance, queries relevant records, and feeds results to explanation plans.
- The two case-study scenarios produced provenance templates, queries, and syntax-tree explanation plans. The school-allocation scenario was estimated at 3.2 person-days and about two engineering hours per explanation sentence after the reusable service existed; the authors describe this as an upper bound in their setting.
- The study explicitly excludes integration effort and service implementation effort, estimates rather than directly measures development time from Git history, involves experienced author-engineers, simulates decision pipelines, and assumes the generated explanations are suitable and separately validated.
- Iteration exposed missing evidence: some explanation-plan needs required query changes and, when the provenance trace had not captured the data, provenance-template changes. This is direct support for documenting evidence gaps before adding instrumentation.

Project implication: motivates immutable decision-time evidence, explicit support links, query/derivation records, and separate answer plans (RQ1/RQ2). It also supports freezing explanation requirements before capture. The project should not import a general PROV stack or graph service: typed JSON records test the same scientific issue at much lower cost.

Fit and cost: high conceptual fit, low direct code reuse. The artifact repository offers concrete provenance/query/plan examples but no robot benchmark.

Unresolved: no peer-reviewed venue/DOI was verified from the cited primary sources; cite the preprint accurately unless a publisher record is found. Its case study does not establish explanation correctness or user trust.

### Min et al., *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*

Publication: EMNLP 2023, pp. 12076--12100, [DOI 10.18653/v1/2023.emnlp-main.741](https://doi.org/10.18653/v1/2023.emnlp-main.741), [arXiv:2305.14251](https://arxiv.org/abs/2305.14251), and [authors' MIT-licensed repository/package](https://github.com/shmsw25/FActScore).

Verified findings:

- FActScore decomposes a generation into atomic facts, labels each against a specified knowledge source, and reports the supported fraction. It is factual precision, not completeness or response-level material-error rate.
- On biography generation, human FActScores were 42% for InstructGPT, 58% for ChatGPT, and 71% for retrieval-augmented PerplexityAI. The paper's automated estimator had less than 2% aggregate error in that experimental setup and used retrieval plus a strong evaluator model.
- The authors explicitly report response ratio and facts per valid response alongside factual precision in later experiments. This supports joint risk/coverage reporting.
- The default implementation is built around a Wikipedia snapshot and either API-backed ChatGPT or LLaMA-7B-based evaluation; the README estimates API cost and supports custom knowledge sources. These dependencies are unnecessary for deterministic checks over a small typed robot record.

Project implication: reuse atomic proposition accounting and distinguish precision from coverage (RQ2/RQ4). Evaluate the proposition extractor against human annotations and the final verifier against adversarial material errors. Do not cite the paper's `<2%` estimator error as expected performance here.

Fit and cost: medium conceptual fit, low value in direct integration before the main benchmark. A custom evidence-ID-aware scorer is simpler, more auditable, and CPU-only.

Unresolved: define atomicity rules for causal/contrastive clauses and qualification scope; determine inter-annotator agreement for those rules on development answers.

### Dušek and Kasner, *Evaluating Semantic Accuracy of Data-to-Text Generation with Natural Language Inference*

Publication: 13th International Conference on Natural Language Generation (INLG 2020), pp. 131--137, [DOI 10.18653/v1/2020.inlg-1.19](https://doi.org/10.18653/v1/2020.inlg-1.19), and [authors' code](https://github.com/ufal/nlgi_eval).

Verified findings:

- The metric templates structured facts into text and applies a pretrained RoBERTa-large-MNLI model in both directions: generated text → individual fact for omissions, and concatenated facts → generated text for hallucinations.
- Against the paper's labels, the default setup obtained 0.775 accuracy/0.784 F1 on WebNLG and 0.933 rough binary accuracy/0.903 F1 on E2E (fine-grained E2E accuracy 0.911). The paper notes limitations of both crowdsourced WebNLG labels and the E2E handcrafted script.
- The authors explicitly limit two-way checking to tasks without implicit content selection, or to cases where the selected facts are supplied. They also call for further evaluation on long texts and content-selection tasks.
- Reusing the released implementation on another domain requires predicate templates and a data loader; its repository does not expose a license in GitHub metadata as checked on 2026-09-19.

Project implication: use as a generic NLI baseline and possibly a bounded safety net for freer wording, not as the verifier oracle (RQ2). Compare its errors with deterministic checks and the robotics-tuned coherence classifier. Feed only selected answer-plan claims when testing omission, or the method will incorrectly penalize intentional partial answers (RQ3).

Fit and cost: low-to-medium adaptation and useful as a secondary baseline after core tests pass.

Unresolved: select and freeze entailment thresholds on development data; measure false acceptance of counts, negation, temporal scope, causal connectives, and completeness qualifiers.

## Selective answering and risk--coverage

### Geifman and El-Yaniv, *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*

Publication: Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97, [official paper page](https://proceedings.mlr.press/v97/geifman19a.html).

Verified findings:

- A selective model comprises a prediction function and a selection function. Coverage is the probability/mass accepted by the selection function; selective risk is loss conditioned on acceptance. SelectiveNet jointly trains prediction and rejection for a target coverage using a coverage-constrained objective.
- The paper reports improved risk--coverage tradeoffs over softmax response and Monte Carlo dropout on several classification/regression datasets. These results concern supervised prediction, not natural-language explanation fidelity.

Project implication: report material-error risk against substantive/answerable-information coverage across verifier thresholds (RQ3/RQ4), at native and matched-coverage operating points. Full, partial, and abstained responses need separate labels because a partially supported answer is not equivalent to classification rejection.

Fit and cost: citation and analysis pattern only; implementing SelectiveNet would not address the current RQs.

Unresolved: choose a practically meaningful coverage definition and pre-freeze how partial answers contribute; calibrate thresholds by episode-level development splits.

## Recent systems and behavior-tree counterfactuals

### HEXAR

Publication status: Tamlin Love et al. (eight authors), *HEXAR: a Hierarchical Explainability Architecture for Robots*, [arXiv:2601.03070](https://arxiv.org/abs/2601.03070), submitted 2026-01-06; the [authors' project page](https://pradippramanick.github.io/hexar/) labels it ICRA 2026, and [code/data/results](https://github.com/fgebelli/HEXAR) are public. Final official proceedings metadata/DOI was not located, so cite the arXiv version unless that changes.

Verified findings:

- HEXAR uses a selector over five specialized explainers and compares it with an end-to-end LLM and an “all components” LLM. Experiments use 20 situations × 3 task variants = 60 physical TIAGo/ROS 2 bag runs; all systems consume the same bags, with 3 question variants per run, yielding 180 responses per system and 540 total.
- Three blinded coauthor annotators labeled root-cause presence and incorrect facts, with reported disagreement rates of 0.93% and 1.30%; majority labels determine results.
- Reported root-cause/incorrect-fact/combined-accuracy/runtime results are: HEXAR 97.22%/7.22%/92.78%/1.73 s; end-to-end 72.78%/27.78%/65.56%/7.86 s; all-components 92.22%/32.22%/67.22%/10.05 s. The selector chose the intended explainer in 179/180 cases. The authors report Cochran's Q p<0.001 and Holm-corrected pairwise McNemar tests.
- The repository's detailed CSV contains the 540 generated texts and per-annotator/majority labels; another CSV contains experiment ground truth. Components declare Apache-2.0 licensing.

Project implication: this is the highest-value low-cost external verifier test found for RQ2. Score the retained 540 outputs offline while keeping experiment ground truth evaluator-only. Its labels cover root cause and incorrect facts, not all of this project's material-error categories, completeness, or answer coverage. Do not call the published combined accuracy our metric.

Fit and cost: high for offline verification and likely under one day. Full reproduction is poor deadline value because it requires ROS 2 Humble, physical-run bags, an Ollama endpoint, model/GPU resources, and several hours.

Unresolved/validity issue: 180 questions are nested within 60 runs and 20 situation families, while the reported response-level tests do not appear to model that clustering. Our analysis should use episode/scenario-clustered inference. The paper does not systematically evaluate abstention/risk--coverage; one selector mistake happened to yield an insufficiency response.

### *Temporal Counterfactual Explanations of Behavior Tree Decisions*

Publication status: Tamlin Love, Antonio Andriella, and Guillem Alenyà, *Temporal Counterfactual Explanations of Behaviour Tree Decisions*, [arXiv:2509.07674v2](https://arxiv.org/abs/2509.07674), 2026-05-20, with [authors' code](https://github.com/tamlinlove/btcm). No peer-reviewed venue/DOI or repository license was verified.

Verified findings:

- The method constructs a causal explanation model from BT structure, injected node/state domain knowledge, and episodic snapshots, then answers formal contrastive questions by counterfactual search. It is not a passive-log-only method and it does not verify generated natural language.
- In a random evaluation of 450 BT/state combinations and 1,800 perturbation comparisons, 687 changed behavior and the method recovered the target in all 687 (target recovery rate 1.0); reported mean explanation time was 0.0266 s and maximum 0.3709 s.
- In a serial-recall example (33 BT nodes/18 leaves, 26 state variables, explanation model 177 nodes/406 edges), 196 of 200 comparisons changed behavior; target recovery was 1.0, with mean 0.5203 s and maximum 1.3029 s.
- On 200 runs per configuration, phi-4 14B and DeepSeek-R1 32B baselines achieved target-recovery rates 0.06/0.20 and 0.45/0.53 for simple/complete prompts; nonexistent-variable explanations appeared in 188/161 and 8/31 runs respectively.
- The method assumes deterministic state and BT execution, is only as faithful as supplied domain knowledge, supports sequence/fallback rather than all BT decorators/parallel forms, can return large explanation sets, and provides no human evaluation.

Project implication: cite as strong precedent for checked formal reasoning and as a boundary for causal/counterfactual claims (RQ2/RQ3). Passive Nav2 traces support reconstructed control flow, not this paper's intervention semantics. Only answer counterfactuals in explicitly modeled synthetic cases; otherwise state that evidence does not establish them.

Fit and cost: high positioning value but low direct integration value. Its custom `py-trees` state API is not stock Nav2 replay; integration likely exceeds one day and still would not verify final language. Defer implementation.

## Benchmark adoption decisions

| Source | Proposed use | Supports | Deadline decision |
|---|---|---|---|
| Human robot-question taxonomy | Select representative question families and report evidence-model taxonomy coverage | RQ3, ecological validity | Adopt after source/data audit; no answer-fidelity claim |
| Explanation-selection evaluation | External execution-retrieval/entity-resolution comparison | Retrieval component, not faithfulness | Adopt only if format mapping is under one day |
| RoboFail | Unsupported-clause and failure-reasoning stress test | RQ2, RQ3 | Use a text subset if dataset access/licensing is resolved |
| CoExp-IROS24 | Generic NLI versus robotics-domain coherence test | RQ2 | High-priority low-cost external test after core benchmark |
| Chimera/WebNLG | Conceptual planning/realization precedent | RQ2 | Cite; do not reproduce |
| FActScore | Atomic-claim and precision/coverage precedent | RQ2, RQ4 | Reimplement domain-specific accounting; do not import stack initially |
| NLGI-Eval | Generic bidirectional NLI baseline | RQ2 | Secondary baseline if core deterministic verifier is healthy |
| SelectiveNet | Risk--coverage formalism | RQ3, RQ4 | Adopt analysis concepts only |
| HEXAR | Offline external incorrect-fact/root-cause labels | RQ2 | High-priority external verifier check after core tests |
| Temporal counterfactual BT | Formal counterfactual precedent and evidence boundary | RQ2, RQ3 | Cite; defer code integration |

## Claims not yet verified or not transferable

- No external paper's metric should be described as equivalent to this project's response-level material-error rate.
- No published generic-NLI, FActScore, RoboFail, or coherence-classification number establishes verifier accuracy on CRANE/Nav2 explanations.
- RoboFail dataset availability and license remain unresolved: its download endpoint was inaccessible from this environment, although code is available under MIT.
- The question-taxonomy repository had no declared top-level license when inspected; use its taxonomy for analysis/citation but resolve redistribution permission before copying the dataset into a release.
- The CoExp and NLGI-Eval repositories had no declared GitHub license at the time of inspection; availability does not imply permission for redistribution.
- The Explainability-by-Design work was verified as an arXiv preprint only; do not invent a journal venue or DOI.
- External datasets must be audited for episode-level duplication and evaluator-truth leakage before use.

## Primary-source index

- TRUSTMORE 2026 official site and portal configuration: <https://trustmoreai.github.io/workshop2026/>, <https://api2.openreview.net/invitations?id=IEEE.org%2FBigData%2F2026%2FWorkshop%2FTRUSTMORE%2F-%2FSubmission>
- Wachowiak et al. question taxonomy: <https://doi.org/10.1145/3832777>, <https://arxiv.org/abs/2510.16435>, <https://github.com/lwachowiak/xai-questions-dataset>
- Wachowiak et al. explanation selection: <https://doi.org/10.1145/3776734.3794387>, <https://github.com/lwachowiak/explanation-selection-eval>
- REFLECT/RoboFail: <https://proceedings.mlr.press/v229/liu23g.html>, <https://arxiv.org/abs/2306.15724>, <https://robot-reflect.github.io/>, <https://github.com/real-stanford/reflect>
- Coherent multimodal failure explanations: <https://doi.org/10.1109/IROS58592.2024.10802671>, <https://github.com/pradippramanick/coexp-iros24>
- Explainability-by-Design: <https://arxiv.org/abs/2206.06251>, <https://github.com/plead-project/EbD-artefacts>
- Step-by-Step planning/realization: <https://doi.org/10.18653/v1/N19-1236>, <https://github.com/AmitMY/chimera>
- FActScore: <https://doi.org/10.18653/v1/2023.emnlp-main.741>, <https://github.com/shmsw25/FActScore>
- SelectiveNet: <https://proceedings.mlr.press/v97/geifman19a.html>
- NLI semantic-accuracy evaluation: <https://doi.org/10.18653/v1/2020.inlg-1.19>, <https://github.com/ufal/nlgi_eval>
- HEXAR: <https://arxiv.org/abs/2601.03070>
- HEXAR project/data: <https://pradippramanick.github.io/hexar/>, <https://github.com/fgebelli/HEXAR>
- Temporal counterfactual BT explanations: <https://arxiv.org/abs/2509.07674>, <https://github.com/tamlinlove/btcm>
