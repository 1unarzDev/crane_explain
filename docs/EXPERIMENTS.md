# Experiment Log

## 2026-09-19 — initial implementation checkpoint

- Git state: two new, initially empty target repositories; CRANE
  `3fefd98904abc83842593be79be8eae133b3bb65`; astro_dock
  `36202373ae186a8fd247a20b7b477312a744de99`.
- Runtime: host Python 3.14; local Docker image `lunarzdev/astro:cuda`, image ID prefix
  `sha256:9c286b78dc`; ROS packages `nav2_msgs`/`nav2_bt_navigator` 1.3.12.
- Command: `python -m pytest -q`.
- Result: 14 passed; Dock/Slalom missing-policy, explicit-policy, recovery de-duplication,
  completeness qualification, outcome isolation, alternative status, final-clause rejection.
- Benchmark harness tests cover A/B shared generator/question, information-parity rejection,
  C extraction into checked reasoning, D no-resampling fallback, and deterministic E.
- Interface probe: inspected installed message/action definitions and `ros_topic_logger.hpp` in the
  container. Passive fields confirmed. This was not a live publication test.
- Data collected: 0 independent robot episodes; 0 benchmark responses; 0 exclusions.
- Model/provider/prompt: none; CPU-only deterministic test.
- Observation: existing CRANE fixture correctly labels odometry as delivered rather than proven
  internally consumed, but its `goalAttempts` is action-server startup/goal submission—not recovery.
- ROS package: isolated Jazzy container `colcon build` **TESTED**; writer test 1 passed. Live Nav2
  publication/capture and CRANE pilot remain **NOT_RUN**.

## 2026-09-19 — CRANE/Nav2 baseline reproduction and cold-start diagnosis

- Build command: `unity build /home/lunarz/crane_ml --editor-version 6000.5.10f1
  --target StandaloneLinux64 --execute-method CranePerformanceBuild.BuildLinuxWorker
  --allow-dirty-build`.
- Build result: **TESTED/PASS**; `CRANE_BUILD_COMPLETE`, Linux worker 551,473,945 bytes;
  build manifest asset-set SHA-256
  `B12ED54E542E2B34A9C4AE762FE4DB66041731830CB9F384FEDFDEC0BED4A228`.
- Default fixture command used `Tools/Performance/run_nav2_controller_fixture.sh` with outputs at
  `research/explanation_fidelity/results/dev/nav2-baseline-20260919/` (ignored, retained locally).
- Default result: **TESTED/INVALID**. Navigation status `timeout`; displacement 0.51986 m;
  `goalAttempts=3`; 65 accepted actions; fresh observations; RTF 1.00012. Controller logs show two
  inactive-server goal rejections, accepted execution at 1789833415.34, and client cancellation at
  1789833423.49. `goalAttempts` is not interpreted as recovery.
- Falsifiable diagnosis: the fixture's 20 s wall deadline included lifecycle/TF startup, leaving
  only about 8.15 s after the accepted goal. Single-variable rerun set
  `CRANE_FIXTURE_DELAY=15`; no code/config/physics changes.
- Delay-15 result: **TESTED/PASS**, `valid=true`; action succeeded in 9.983 s; one goal submission;
  displacement 0.52074 m; 79 accepted, 0 rejected/stale/cross-episode actions; depth 450,
  detections 240, LiDAR 300, 0 failed/stale observations; RTF 1.00053. Artifacts retained at
  `research/explanation_fidelity/results/dev/nav2-baseline-delay15-20260919/`.
- Interpretation: the local full Nav2 loop is reproducible after adequate cold-start margin. This
  is one baseline scenario instance, not an explanation benchmark response and not evidence for
  RQ1–RQ4. Current explanation-benchmark sample size remains 0.
- Remaining threat: live `crane_explain_ros` capture has not yet been co-run with the fixture, and
  the default delay can still create invalid cold-start runs. Freeze an explicit startup-readiness
  rule before final collection; do not silently exclude timeouts after inspecting answers.
