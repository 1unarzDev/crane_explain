from crane_explain.diagnostics import (
    CausalLanguageLevel,
    DiagnosticDisposition,
    GeometricRouteObservation,
    GoalTerminationObservation,
    diagnose_geometric_route_restriction,
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
    assert result.mechanism == "post_return_motion_exceeded_position_margin"
    assert "only 0.026 m of positional task margin" in result.diagnosis
    assert "grew by 0.185 m" in result.diagnosis
    assert "Missing return speed" in result.diagnosis
    assert "does not establish" in result.limits


def test_missing_speed_and_settled_error_withholds_positional_chain():
    result = diagnose_terminal_stopping_margin(
        _observation(measured_speed_at_return_mps=None, settled_error_m=None)
    )

    assert result.disposition == DiagnosticDisposition.INSUFFICIENT
    assert result.mechanism == "terminal_stopping_margin"
    assert "cannot be assessed" in result.diagnosis
    assert "independently measured speed" in result.limits
    assert "settled pose error" in result.limits


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


def _route_observation(**overrides):
    values = {
        "episode_id": "land-blockage-global-002",
        "evidence_ids": ("global-costmap", "delivered-odometry", "action-result"),
        "frame": "odom",
        "action_status": "aborted",
        "direct_route_has_lethal_cell": True,
        "direct_route_minimum_clearance_m": 0.0,
        "direct_route_first_lethal_x_m": 8.8,
        "direct_route_first_lethal_y_m": 0.0,
        "grid_connected": True,
        "connectivity_origin": "action-result-pose",
        "maximum_lateral_deviation_m": 2.7046,
        "maximum_forward_progress_m": 6.9489,
        "goal_distance_m": 18.0,
        "successful_plan_count": 69,
        "action_wall_seconds": 70.8602,
        "configured_deadline_seconds": 70.0,
        "configured_robot_radius_m": 0.22,
        "configured_inflation_radius_m": 0.55,
        "costmap_resolution_m": 0.1,
        "costmap_snapshot_sha256": "0" * 64,
        "costmap_snapshot_timestamp_s": 74.039,
        "terminal_transition_observed": False,
        "source_anchor_ids": ("bt-timeout-source", "nav2-costmap-config"),
    }
    values.update(overrides)
    return GeometricRouteObservation(**values)


def test_geometric_route_diagnosis_preserves_connected_detour_distinction():
    result = diagnose_geometric_route_restriction(_route_observation())

    assert result.disposition == DiagnosticDisposition.SUPPORTED
    assert "still contained a traversable connection" in result.diagnosis
    assert "global physical infeasibility" in result.limits
    assert "recovery exhaustion" in result.failure_chain
    assert "retained-grid-connectivity" in result.contradictory_evidence
    rendered = render_diagnostic(result)
    assert "maximum_lateral_deviation" in rendered
    assert "costmap_resolution" not in rendered


def test_geometric_route_diagnosis_fails_to_insufficient_without_connectivity():
    result = diagnose_geometric_route_restriction(
        _route_observation(grid_connected=None)
    )

    assert result.disposition == DiagnosticDisposition.INSUFFICIENT
    assert "retained-grid connectivity" in result.limits
    assert result.mechanism == "deadline_aligned_abort_with_unresolved_geometry"
    assert "abort was aligned within 0.86 s" in result.diagnosis
    assert "does not establish why navigation remained incomplete" in result.limits


def test_geometric_route_diagnosis_does_not_trigger_without_lethal_route_cell():
    result = diagnose_geometric_route_restriction(
        _route_observation(direct_route_has_lethal_cell=False)
    )

    assert result.disposition == DiagnosticDisposition.NOT_TRIGGERED


def test_geometric_route_diagnosis_rejects_false_failure_premise_on_success():
    result = diagnose_geometric_route_restriction(
        _route_observation(
            action_status="succeeded",
            direct_route_has_lethal_cell=False,
            direct_route_minimum_clearance_m=1.94,
            maximum_lateral_deviation_m=0.0,
        )
    )

    assert result.disposition == DiagnosticDisposition.NOT_TRIGGERED
    assert result.mechanism == "no_failure_observed"
    assert result.causal_language_level == CausalLanguageLevel.RECORDED_SEQUENCE
    assert "failure premise is false" in result.diagnosis
    assert "fully covered direct-route audit" in result.diagnosis
    assert "1.940 m minimum clearance" in result.diagnosis
    assert result.contradictory_evidence == ()
    assert "No terminal failure chain" in result.failure_chain
    rendered = render_diagnostic(result)
    assert "action_status=succeeded status" in rendered
    assert "direct_route_minimum_clearance=1.9400 m" in rendered
    assert "maximum_lateral_deviation=0.0000 m" in rendered
    assert "action aborted" not in rendered


def test_geometric_route_success_rejects_false_premise_without_costmap_cells():
    result = diagnose_geometric_route_restriction(
        _route_observation(
            action_status="succeeded",
            direct_route_has_lethal_cell=None,
            direct_route_minimum_clearance_m=None,
            grid_connected=None,
        )
    )

    assert result.disposition == DiagnosticDisposition.NOT_TRIGGERED
    assert result.mechanism == "no_failure_observed"
    assert "failure premise is false" in result.diagnosis
    assert "route classification is unavailable" in result.limits
    assert result.decisive_measurement_ids == ("action_status",)


def test_geometric_route_success_reports_restriction_without_calling_it_a_failure():
    result = diagnose_geometric_route_restriction(
        _route_observation(action_status="succeeded")
    )

    assert result.disposition == DiagnosticDisposition.NOT_TRIGGERED
    assert "failure premise is false" in result.diagnosis
    assert "marked the requested direct route as restricted near x=8.80 m" in result.diagnosis
    assert "did not prevent the recorded successful outcome" in result.diagnosis
    assert "what physical object produced it" in result.limits
    rendered = render_diagnostic(result)
    assert "first_lethal_route_x=8.8000 m" in rendered
    assert "maximum_lateral_deviation=2.7046 m" in rendered
    assert "action aborted" not in rendered
