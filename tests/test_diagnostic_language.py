from crane_explain.diagnostic_language import verify_bounded_diagnostic_text
from crane_explain.diagnostics import (
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
