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


def _command_motion_result(*, action_status="aborted", measured_after=0.0):
    windows = tuple(
        [CommandMotionWindow(i, i, i + 1, 10, 40, 0.8, 0.26) for i in range(5)]
        + [CommandMotionWindow(i, i, i + 1, 10, 40, 0.8, measured_after)
           for i in range(5, 9)]
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


def test_bounded_verifier_accepts_command_motion_checked_template():
    result = _command_motion_result()
    verification = verify_bounded_diagnostic_text(result, render_diagnostic(result))

    assert verification.accepted, verification.reasons


def test_bounded_verifier_accepts_command_motion_nominal_template():
    result = _command_motion_result(action_status="succeeded", measured_after=0.26)
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
