from crane_explain.diagnostics import (
    DiagnosticDisposition,
    GoalTerminationObservation,
    diagnose_terminal_stopping_margin,
    render_diagnostic,
    verify_diagnostic_text,
)


def _observation(**overrides):
    values = {
        "episode_id": "roboboat-gate5-known-dock-1",
        "evidence_ids": ("action-result", "measured-odometry", "task-contract"),
        "frame": "odom",
        "action_status": "succeeded",
        "configured_goal_tolerance_m": 0.40,
        "configured_stopped_speed_mps": 0.05,
        "task_acceptance_tolerance_m": 0.40,
        "action_return_error_m": 0.3738084684842852,
        "measured_speed_at_return_mps": 0.0486084585847379,
        "post_result_coast_m": 0.1875955526646154,
        "settled_error_m": 0.558571457862854,
        "action_return_timestamp_s": 240.59,
        "settled_timestamp_s": 248.64,
        "source_anchor_ids": ("stopped-goal-checker-config",),
    }
    values.update(overrides)
    return GoalTerminationObservation(**values)


def test_goal_checker_margin_diagnosis_uses_measured_motion():
    result = diagnose_terminal_stopping_margin(_observation())

    assert result.disposition == DiagnosticDisposition.SUPPORTED
    assert "only 0.026 m of task margin" in result.diagnosis
    assert "grew by 0.185 m" in result.diagnosis
    assert "not a unique claim" in result.limits
    assert result.source_anchor_ids == ("stopped-goal-checker-config",)
    values = {measurement.id: measurement.value for measurement in result.measurements}
    assert values["task_margin_at_return"] == 0.40 - 0.3738084684842852
    assert values["radial_error_growth_after_return"] == (
        0.558571457862854 - 0.3738084684842852
    )


def test_missing_measured_speed_fails_to_insufficient():
    result = diagnose_terminal_stopping_margin(
        _observation(measured_speed_at_return_mps=None)
    )

    assert result.disposition == DiagnosticDisposition.INSUFFICIENT
    assert "independently measured speed" in result.limits
    assert "cannot be assessed" in result.diagnosis


def test_settling_inside_task_tolerance_does_not_trigger_diagnosis():
    result = diagnose_terminal_stopping_margin(
        _observation(settled_error_m=0.39, post_result_coast_m=0.02)
    )

    assert result.disposition == DiagnosticDisposition.NOT_TRIGGERED


def test_diagnostic_rendering_has_required_sections_and_verifies_exactly():
    result = diagnose_terminal_stopping_margin(_observation())
    text = render_diagnostic(result)

    assert text.startswith("Diagnosis:")
    assert "\nDecisive evidence:" in text
    assert "\nFailure chain:" in text
    assert "\nLimits and next check:" in text
    assert verify_diagnostic_text(result, text).accepted
    assert not verify_diagnostic_text(result, text + " Wave drift caused it.").accepted


def test_invalid_negative_measurement_is_rejected():
    try:
        diagnose_terminal_stopping_margin(_observation(post_result_coast_m=-0.1))
    except ValueError as error:
        assert "post_result_coast_m" in str(error)
    else:
        raise AssertionError("negative distance was accepted")
