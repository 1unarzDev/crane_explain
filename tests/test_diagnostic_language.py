from crane_explain.diagnostic_language import verify_bounded_diagnostic_text
from crane_explain.diagnostics import (
    CommandMotionObservation,
    CommandMotionWindow,
    GoalTerminationObservation,
    diagnose_terminal_stopping_margin,
    render_diagnostic,
)


def _result():
    return diagnose_terminal_stopping_margin(
        GoalTerminationObservation(
            episode_id="boat-1",
            evidence_ids=("fixture:abc", "odom:return", "task:limit"),
            frame="odom",
            action_status="succeeded",
            configured_goal_tolerance_m=0.4,
            configured_stopped_speed_mps=0.05,
            task_acceptance_tolerance_m=0.4,
            action_return_error_m=0.3738,
            measured_speed_at_return_mps=0.0486,
            post_result_coast_m=0.1876,
            settled_error_m=0.5614,
            action_return_timestamp_s=10.0,
            settled_timestamp_s=18.0,
        )
    )


GOOD_CANDIDATE = """## Diagnosis

The supported mechanism is insufficient terminal stopping margin: only 0.026 m remained.

## Decisive evidence

The action returned at 0.374 m error and 0.0486 m/s. After the action returned, error grew by 0.188 m and settled at 0.561 m, outside the 0.400 m task limit.

## Failure chain

Post-result motion exceeded the available margin after the action returned.

## Limits and next check

The physical source of the motion remains unresolved and is not uniquely established. Repeat with a prospectively fixed smaller tolerance.
"""


def test_bounded_verifier_accepts_supported_paraphrase_after_citation_only_repair():
    verification = verify_bounded_diagnostic_text(_result(), GOOD_CANDIDATE)

    assert verification.accepted, verification.reasons
    assert verification.repair_applied
    assert verification.policy == "bounded-diagnostic-language-v2"
    assert "Evidence IDs: fixture:abc, odom:return, task:limit." in verification.checked_text


def test_bounded_verifier_rejects_unlicensed_wave_cause():
    candidate = GOOD_CANDIDATE.replace(
        "The physical source of the motion remains unresolved and is not uniquely established.",
        "Wave drift caused the post-result motion.",
    )
    verification = verify_bounded_diagnostic_text(_result(), candidate)

    assert not verification.accepted
    assert any("unsupported physical-cause wording: wave" in item for item in verification.reasons)


def test_bounded_verifier_rejects_unlicensed_number():
    candidate = GOOD_CANDIDATE.replace("0.561 m", "9.999 m")
    verification = verify_bounded_diagnostic_text(_result(), candidate)

    assert not verification.accepted
    assert "unlicensed numeric claim: 9.999" in verification.reasons


def test_bounded_verifier_rejects_missing_section():
    candidate = GOOD_CANDIDATE.replace("## Failure chain", "## Other")
    verification = verify_bounded_diagnostic_text(_result(), candidate)

    assert not verification.accepted
    assert any("sections must appear" in item for item in verification.reasons)


def test_bounded_verifier_accepts_existing_checked_template_without_repair():
    verification = verify_bounded_diagnostic_text(_result(), render_diagnostic(_result()))

    assert verification.accepted, verification.reasons
    assert not verification.repair_applied


def _command_motion_result(*, action_status="aborted", measured_after=0.0, recovered=False):
    later = [
        CommandMotionWindow(i, i, i + 1, 10, 40, 0.8, measured_after)
        for i in range(5, 9)
    ]
    if recovered:
        later.extend(
            CommandMotionWindow(i, i, i + 1, 10, 40, 0.8, 0.26)
            for i in range(9, 12)
        )
    windows = tuple(
        [CommandMotionWindow(i, i, i + 1, 10, 40, 0.8, 0.26) for i in range(5)]
        + later
    )
    from crane_explain.diagnostics import diagnose_command_motion_discrepancy

    return diagnose_command_motion_discrepancy(
        CommandMotionObservation(
            episode_id="land-command-motion-1",
            evidence_ids=("events:abc", "bt:def"),
            command_frame="base_link-command-convention",
            measured_frame="odom",
            action_status=action_status,
            windows=windows,
            raw_command_sample_count=90,
            raw_odometry_sample_count=360,
            window_seconds=1.0,
            minimum_command_samples_per_window=5,
            minimum_odometry_samples_per_window=20,
            calibration_window_count=5,
            minimum_commanded_speed_mps=0.4,
            minimum_healthy_measured_speed_mps=0.1,
            maximum_discrepancy_response_ratio=0.2,
            minimum_consecutive_discrepancy_windows=3,
            follow_path_failure_count=2 if action_status == "aborted" else 0,
            follow_path_attempt_count=3 if action_status == "aborted" else 1,
            source_qualified_recovery_count=2 if action_status == "aborted" else 0,
        )
    )


def _command_motion_missing_odometry_result():
    from crane_explain.diagnostics import diagnose_command_motion_discrepancy

    return diagnose_command_motion_discrepancy(
        CommandMotionObservation(
            episode_id="land-command-motion-missing-odometry",
            evidence_ids=("events:abc", "bt:def"),
            command_frame="base_link-command-convention",
            measured_frame="odom",
            action_status="aborted",
            windows=(),
            raw_command_sample_count=376,
            raw_odometry_sample_count=0,
            window_seconds=1.0,
            minimum_command_samples_per_window=5,
            minimum_odometry_samples_per_window=20,
            calibration_window_count=5,
            minimum_commanded_speed_mps=0.4,
            minimum_healthy_measured_speed_mps=0.1,
            maximum_discrepancy_response_ratio=0.2,
            minimum_consecutive_discrepancy_windows=3,
            follow_path_failure_count=2,
            follow_path_attempt_count=3,
            source_qualified_recovery_count=2,
        )
    )


def _compensated_command_motion_result():
    from crane_explain.diagnostics import diagnose_command_motion_discrepancy

    healthy_speed = 0.2597399950027466
    windows = tuple(
        [
            CommandMotionWindow(index, index, index + 1, 10, 40, 0.8, healthy_speed)
            for index in range(5)
        ]
        + [
            CommandMotionWindow(index, index, index + 1, 10, 40, 0.8, 0.0)
            for index in range(8, 18)
        ]
        + [CommandMotionWindow(20, 20, 21, 10, 40, 0.8, healthy_speed)]
    )
    return diagnose_command_motion_discrepancy(
        CommandMotionObservation(
            episode_id="diagnostic-motion-dev-cm-compensated-001",
            evidence_ids=("events:abc", "bt:def"),
            command_frame="base_link-command-convention",
            measured_frame="odom",
            action_status="succeeded",
            windows=windows,
            raw_command_sample_count=485,
            raw_odometry_sample_count=2232,
            window_seconds=1.0,
            minimum_command_samples_per_window=5,
            minimum_odometry_samples_per_window=20,
            calibration_window_count=5,
            minimum_commanded_speed_mps=0.4,
            minimum_healthy_measured_speed_mps=0.1,
            maximum_discrepancy_response_ratio=0.2,
            minimum_consecutive_discrepancy_windows=3,
            follow_path_failure_count=1,
            follow_path_attempt_count=2,
            source_qualified_recovery_count=1,
        )
    )


COMPENSATED_RAW_P_CANDIDATE = """## Diagnosis

Yes. A sustained command-to-motion discrepancy temporarily prevented continued progress: Nav2 commanded 0.800 m/s while independently measured odometry recorded 0.000 m/s for 10.0 s.

## Decisive evidence

Healthy measured planar speed was 0.2597 m/s. During the discrepancy, the response ratio was 0.0. Afterward, measured speed recovered to 0.2597 m/s, a response ratio of 1.0. FollowPath recorded 1 failure, and the action ultimately succeeded.

## Failure chain

The response loss was followed by 1 source-qualified Wait recovery invocation. Before navigation succeeded, a later sufficiently sampled command-active window showed restored measured motion at the calibrated healthy level.

## Limits and next check

The evidence establishes the discrepancy and its execution sequence, but not a unique physical cause. Delivered commands do not prove actuator acceptance, and odometry does not prove Nav2 consumption. The unresolved causes include actuator rejection, mobility constraint, collision or obstruction, slip, or another execution-layer cause. Next, record downstream accepted actuation or actuator feedback with contact, clearance, and wheel-motion evidence during the discrepancy interval.
"""


def test_bounded_verifier_accepts_archived_compensated_raw_candidate():
    verification = verify_bounded_diagnostic_text(
        _compensated_command_motion_result(),
        COMPENSATED_RAW_P_CANDIDATE,
    )

    assert verification.accepted, verification.reasons
    assert verification.repair_applied


def test_bounded_verifier_rejects_compensated_candidate_without_follow_path_failure():
    candidate = COMPENSATED_RAW_P_CANDIDATE.replace(
        "FollowPath recorded 1 failure",
        "The controller recorded 1 failure",
    )

    verification = verify_bounded_diagnostic_text(
        _compensated_command_motion_result(),
        candidate,
    )

    assert not verification.accepted
    assert "missing required proposition: controller failure sequence" in verification.reasons


def test_bounded_verifier_rejects_claim_that_wait_caused_response_recovery():
    candidate = COMPENSATED_RAW_P_CANDIDATE.replace(
        "The response loss was followed by 1 source-qualified Wait recovery invocation. Before navigation succeeded, a later sufficiently sampled command-active window showed restored measured motion at the calibrated healthy level.",
        "One source-qualified Wait recovery invocation caused the measured motion response to recover before navigation succeeded.",
    )

    verification = verify_bounded_diagnostic_text(
        _compensated_command_motion_result(),
        candidate,
    )

    assert not verification.accepted
    assert "unsupported claim that recovery caused measured response" in verification.reasons


def test_bounded_verifier_accepts_command_motion_checked_template():
    result = _command_motion_result()
    verification = verify_bounded_diagnostic_text(result, render_diagnostic(result))

    assert verification.accepted, verification.reasons


def test_bounded_verifier_accepts_command_motion_nominal_template():
    result = _command_motion_result(action_status="succeeded", measured_after=0.26)
    verification = verify_bounded_diagnostic_text(result, render_diagnostic(result))

    assert verification.accepted, verification.reasons


def test_bounded_verifier_accepts_supported_discrepancy_with_recovered_motion_and_success():
    result = _command_motion_result(action_status="succeeded", recovered=True)
    verification = verify_bounded_diagnostic_text(result, render_diagnostic(result))

    assert verification.accepted, verification.reasons


def test_bounded_verifier_does_not_license_cause_by_putting_it_in_next_check():
    result = _command_motion_result()
    candidate = render_diagnostic(result).replace(
        "Next check: Record downstream accepted actuation or actuator feedback together with contact, clearance, and wheel-motion evidence over the discrepancy interval.",
        "Next check: Motor failure caused the discrepancy.",
    )
    verification = verify_bounded_diagnostic_text(result, candidate)

    assert not verification.accepted
    assert "unsupported physical-cause wording: motor" in verification.reasons


def test_bounded_verifier_allows_next_measurement_without_treating_it_as_cause():
    result = _command_motion_result()
    candidate = render_diagnostic(result).replace(
        "Next check: Record downstream accepted actuation or actuator feedback together with contact, clearance, and wheel-motion evidence over the discrepancy interval.",
        "Next, record downstream accepted actuation or actuator feedback with contact evidence.",
    )
    verification = verify_bounded_diagnostic_text(result, candidate)

    assert verification.accepted, verification.reasons


def test_bounded_verifier_does_not_match_wind_inside_fixed_window():
    result = _command_motion_result(action_status="succeeded", measured_after=0.26)
    candidate = render_diagnostic(result).replace(
        "declared thresholds",
        "declared fixed-window thresholds",
    )
    verification = verify_bounded_diagnostic_text(result, candidate)

    assert verification.accepted, verification.reasons


def test_bounded_verifier_accepts_missing_odometry_paraphrase_after_citation_repair():
    result = _command_motion_missing_odometry_result()
    candidate = """## Diagnosis
Insufficient evidence: a command-to-motion discrepancy cannot be assessed because the independently delivered odometry stream is missing. The premise is therefore not established.

## Decisive evidence
There were 376 delivered command samples but 0 independent odometry samples. The action ended aborted, with 2 FollowPath failures and 2 source-qualified Wait recoveries.

## Failure chain
The navigation action aborted after the recorded FollowPath failures and Wait recoveries. Without time-aligned, independently measured motion, this sequence cannot establish that commanded motion failed to produce robot motion or that such a discrepancy prevented continuation.

## Limits and next check
Execution evidence alone does not establish a command-to-motion discrepancy or a unique physical cause. Record synchronized delivered commands and independently measured planar odometry.
"""

    verification = verify_bounded_diagnostic_text(result, candidate)

    assert verification.accepted, verification.reasons
    assert verification.repair_applied
