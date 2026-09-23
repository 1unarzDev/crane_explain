"""Prospective physical-diagnosis records and bounded computations.

This module is deliberately separate from the frozen legacy reasoning path.  It starts with one
auditable mechanism: whether an action's configured return tolerance left enough distance for the
measured platform to settle inside a separately declared task tolerance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class DiagnosticDisposition(str, Enum):
    SUPPORTED = "supported"
    NOT_TRIGGERED = "not_triggered"
    INSUFFICIENT = "insufficient"


class CausalLanguageLevel(str, Enum):
    RECORDED_SEQUENCE = "recorded_sequence"
    EXECUTION_MECHANISM = "execution_mechanism"
    CONTROLLED_INTERVENTION = "controlled_intervention"


@dataclass(frozen=True)
class DiagnosticMeasurement:
    id: str
    value: float | int | str
    unit: str
    frame: str | None
    evidence_ids: tuple[str, ...]
    timestamp_s: float | None = None
    interval_s: tuple[float, float] | None = None


@dataclass(frozen=True)
class GoalTerminationObservation:
    """Robot-visible inputs for a terminal stopping-margin check.

    Pose error and speed must be independently measured; command-derived feedback is not an
    acceptable substitute.  ``task_acceptance_tolerance_m`` is a declared task requirement, not
    hidden evaluator geometry.
    """

    episode_id: str
    evidence_ids: tuple[str, ...]
    frame: str
    action_status: str
    configured_goal_tolerance_m: float
    configured_stopped_speed_mps: float
    task_acceptance_tolerance_m: float
    action_return_error_m: float
    measured_speed_at_return_mps: float | None
    post_result_coast_m: float | None
    settled_error_m: float | None
    action_return_timestamp_s: float | None = None
    settled_timestamp_s: float | None = None
    source_anchor_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeometricRouteObservation:
    """Robot-visible inputs for a bounded route-restriction diagnosis.

    ``direct_route_has_lethal_cell`` and ``grid_connected`` are outputs of an independently
    versioned audit of the retained Nav2 costmap, not evaluator geometry.  ``False`` means the
    complete sampled direct route was covered and no lethal cell was found; partial coverage must
    be represented as ``None``.  Connectivity applies only to the retained grid at its timestamp.
    The trajectory is delivered odometry and does not by itself prove why Nav2 selected a
    particular command.
    """

    episode_id: str
    evidence_ids: tuple[str, ...]
    frame: str
    action_status: str
    direct_route_has_lethal_cell: bool | None
    direct_route_minimum_clearance_m: float | None
    direct_route_first_lethal_x_m: float | None
    direct_route_first_lethal_y_m: float | None
    grid_connected: bool | None
    connectivity_origin: str
    maximum_lateral_deviation_m: float | None
    maximum_forward_progress_m: float | None
    goal_distance_m: float
    successful_plan_count: int | None
    action_wall_seconds: float | None
    configured_deadline_seconds: float | None
    configured_robot_radius_m: float
    configured_inflation_radius_m: float
    costmap_resolution_m: float
    costmap_snapshot_sha256: str
    costmap_snapshot_timestamp_s: float | None
    terminal_transition_observed: bool
    computation_version: str = "geometric-route-restriction-v1"
    source_anchor_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BehaviorTreeTransition:
    """One retained, goal-scoped BehaviorTreeLog status transition."""

    record_id: str
    node_name: str
    previous_status: str
    current_status: str
    goal_id: str


@dataclass(frozen=True)
class RecoveryInvocationRecord:
    """A capture-side recovery invocation with its bounded classifier provenance."""

    invocation_id: str
    node_name: str
    goal_id: str
    start_transition_id: str
    end_transition_id: str | None
    complete: bool
    terminal_status: str | None
    classifier_basis: str
    classifier_rule: str
    policy_sha256: str
    observed_start_transition_id: str


@dataclass(frozen=True)
class RecoveryExecutionObservation:
    """Robot-visible inputs for a bounded Nav2 recovery-mechanism diagnosis.

    The transition sequence supports reconstruction of the software execution mechanism. It does
    not establish the physical condition that made planning fail, and the Nav2 feedback recovery
    count is deliberately retained only as a non-identity observation.
    """

    episode_id: str
    evidence_ids: tuple[str, ...]
    action_status: str
    goal_id: str
    transitions: tuple[BehaviorTreeTransition, ...]
    recovery_invocations: tuple[RecoveryInvocationRecord, ...]
    recovery_policy_sha256: str
    whole_history_complete: bool
    maximum_feedback_recovery_count: int | None
    physical_cause_established: bool
    source_anchor_ids: tuple[str, ...] = ()
    computation_version: str = "recovery-execution-sequence-v1"


@dataclass(frozen=True)
class DiagnosticResult:
    schema_version: str
    diagnostic_id: str
    episode_id: str
    mechanism: str
    disposition: DiagnosticDisposition
    diagnosis: str
    measurements: tuple[DiagnosticMeasurement, ...]
    computation: str
    computation_version: str
    assumptions: tuple[str, ...]
    supporting_evidence: tuple[str, ...]
    contradictory_evidence: tuple[str, ...]
    unresolved_alternatives: tuple[str, ...]
    causal_language_level: CausalLanguageLevel
    failure_chain: str
    limits: str
    next_check: str
    source_anchor_ids: tuple[str, ...]
    decisive_measurement_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticTextVerification:
    accepted: bool
    expected_text: str


def _require_nonnegative(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")


def _measurement(
    measurement_id: str,
    value: float,
    unit: str,
    observation: GoalTerminationObservation,
    *,
    timestamp_s: float | None = None,
    interval_s: tuple[float, float] | None = None,
) -> DiagnosticMeasurement:
    return DiagnosticMeasurement(
        id=measurement_id,
        value=value,
        unit=unit,
        frame=observation.frame,
        evidence_ids=observation.evidence_ids,
        timestamp_s=timestamp_s,
        interval_s=interval_s,
    )


def diagnose_terminal_stopping_margin(
    observation: GoalTerminationObservation,
) -> DiagnosticResult:
    """Check whether return margin was smaller than measured post-result error growth.

    The computation supports a configuration/execution mechanism only when the action returned
    success, independently measured speed satisfied the configured stopped threshold, the result
    was inside Nav2's goal tolerance, and the platform subsequently settled outside the declared
    task tolerance.  It does not identify why the platform coasted.
    """

    numeric = {
        "configured_goal_tolerance_m": observation.configured_goal_tolerance_m,
        "configured_stopped_speed_mps": observation.configured_stopped_speed_mps,
        "task_acceptance_tolerance_m": observation.task_acceptance_tolerance_m,
        "action_return_error_m": observation.action_return_error_m,
        "measured_speed_at_return_mps": observation.measured_speed_at_return_mps,
        "post_result_coast_m": observation.post_result_coast_m,
        "settled_error_m": observation.settled_error_m,
    }
    for name, value in numeric.items():
        _require_nonnegative(name, value)
    if not observation.evidence_ids:
        raise ValueError("at least one robot-visible evidence ID is required")

    measurements = [
        _measurement(
            "configured_goal_tolerance",
            observation.configured_goal_tolerance_m,
            "m",
            observation,
        ),
        _measurement(
            "configured_stopped_speed",
            observation.configured_stopped_speed_mps,
            "m/s",
            observation,
        ),
        _measurement(
            "task_acceptance_tolerance",
            observation.task_acceptance_tolerance_m,
            "m",
            observation,
        ),
        _measurement(
            "action_return_error",
            observation.action_return_error_m,
            "m",
            observation,
            timestamp_s=observation.action_return_timestamp_s,
        ),
    ]
    missing = []
    if observation.measured_speed_at_return_mps is None:
        missing.append("independently measured speed at action return")
    else:
        measurements.append(
            _measurement(
                "measured_speed_at_return",
                observation.measured_speed_at_return_mps,
                "m/s",
                observation,
                timestamp_s=observation.action_return_timestamp_s,
            )
        )
    if observation.post_result_coast_m is None:
        missing.append("post-result displacement")
    else:
        measurements.append(
            _measurement(
                "post_result_displacement",
                observation.post_result_coast_m,
                "m",
                observation,
                interval_s=(
                    observation.action_return_timestamp_s,
                    observation.settled_timestamp_s,
                )
                if observation.action_return_timestamp_s is not None
                and observation.settled_timestamp_s is not None
                else None,
            )
        )
    if observation.settled_error_m is None:
        missing.append("settled pose error")
    else:
        measurements.append(
            _measurement(
                "settled_error",
                observation.settled_error_m,
                "m",
                observation,
                timestamp_s=observation.settled_timestamp_s,
            )
        )

    common = dict(
        schema_version="crane-diagnostic-result-v1",
        diagnostic_id=f"{observation.episode_id}:terminal-stopping-margin",
        episode_id=observation.episode_id,
        mechanism="terminal_stopping_margin",
        measurements=tuple(measurements),
        computation="task_margin_at_return = task_tolerance - return_error; "
        "radial_error_growth = settled_error - return_error",
        computation_version="terminal-stopping-margin-v2",
        assumptions=(
            "Pose errors use the same goal, metric, and coordinate frame.",
            "Speed and pose are independently measured rather than command-derived feedback.",
            "The declared task tolerance is available to the diagnostic method.",
        ),
        causal_language_level=CausalLanguageLevel.EXECUTION_MECHANISM,
        source_anchor_ids=observation.source_anchor_ids,
    )

    positional_chain_available = (
        observation.post_result_coast_m is not None
        and observation.settled_error_m is not None
    )
    if observation.measured_speed_at_return_mps is None and positional_chain_available:
        assert observation.post_result_coast_m is not None
        assert observation.settled_error_m is not None
        margin = observation.task_acceptance_tolerance_m - observation.action_return_error_m
        error_growth = observation.settled_error_m - observation.action_return_error_m
        measurements.extend(
            (
                _measurement("task_margin_at_return", margin, "m", observation),
                _measurement("radial_error_growth_after_return", error_growth, "m", observation),
            )
        )
        common["measurements"] = tuple(measurements)
        positional_chain_supported = (
            observation.action_status.lower() == "succeeded"
            and observation.action_return_error_m <= observation.configured_goal_tolerance_m
            and observation.settled_error_m > observation.task_acceptance_tolerance_m
            and error_growth > margin
        )
        if positional_chain_supported:
            partial_common = {
                **common,
                "mechanism": "post_return_motion_exceeded_position_margin",
            }
            return DiagnosticResult(
                **partial_common,
                disposition=DiagnosticDisposition.INSUFFICIENT,
                diagnosis=(
                    f"The action returned with only {margin:.3f} m of positional task margin, "
                    f"then measured pose error grew by {error_growth:.3f} m and settled outside "
                    "the task tolerance. Missing return speed prevents determining whether the "
                    "configured stopped-speed criterion was physically satisfied."
                ),
                supporting_evidence=observation.evidence_ids,
                contradictory_evidence=(),
                unresolved_alternatives=(
                    "The retained evidence does not establish the measured speed at action return.",
                    "The physical source of the residual motion remains unresolved.",
                ),
                failure_chain=(
                    f"The action reported success at {observation.action_return_error_m:.3f} m "
                    f"error, leaving {margin:.3f} m inside the task limit. After return, the "
                    f"platform moved {observation.post_result_coast_m:.3f} m and settled at "
                    f"{observation.settled_error_m:.3f} m error, outside the "
                    f"{observation.task_acceptance_tolerance_m:.3f} m task limit."
                ),
                limits=(
                    "The positional failure chain is supported, but independently measured speed "
                    "at return is missing. The evidence therefore does not establish that the "
                    "physical platform satisfied the configured stopped-speed threshold or identify "
                    "why residual motion occurred."
                ),
                next_check=(
                    "Record synchronized independently measured speed and controller state at the "
                    "action-success transition through settling."
                ),
                decisive_measurement_ids=(
                    "task_margin_at_return",
                    "post_result_displacement",
                    "settled_error",
                    "radial_error_growth_after_return",
                ),
            )

    if missing:
        return DiagnosticResult(
            **common,
            disposition=DiagnosticDisposition.INSUFFICIENT,
            diagnosis="The terminal stopping margin cannot be assessed from the retained evidence.",
            supporting_evidence=observation.evidence_ids,
            contradictory_evidence=(),
            unresolved_alternatives=(
                "The action may have returned with adequate or inadequate stopping margin.",
            ),
            failure_chain="The terminal action result is recorded, but the post-result motion chain is incomplete.",
            limits=f"Missing decisive evidence: {', '.join(missing)}.",
            next_check="Record synchronized measured speed, pose error at return, and settled pose error.",
        )

    assert observation.measured_speed_at_return_mps is not None
    assert observation.post_result_coast_m is not None
    assert observation.settled_error_m is not None
    margin = observation.task_acceptance_tolerance_m - observation.action_return_error_m
    error_growth = observation.settled_error_m - observation.action_return_error_m
    measurements.extend(
        (
            _measurement("task_margin_at_return", margin, "m", observation),
            _measurement("radial_error_growth_after_return", error_growth, "m", observation),
        )
    )
    common["measurements"] = tuple(measurements)

    supported = (
        observation.action_status.lower() == "succeeded"
        and observation.action_return_error_m <= observation.configured_goal_tolerance_m
        and observation.measured_speed_at_return_mps <= observation.configured_stopped_speed_mps
        and observation.settled_error_m > observation.task_acceptance_tolerance_m
        and error_growth > margin
    )
    if not supported:
        return DiagnosticResult(
            **common,
            disposition=DiagnosticDisposition.NOT_TRIGGERED,
            diagnosis="The retained measurements do not establish an inadequate terminal stopping margin.",
            supporting_evidence=observation.evidence_ids,
            contradictory_evidence=observation.evidence_ids,
            unresolved_alternatives=(
                "A different terminal failure mechanism may apply.",
                "The task may have remained within its acceptance tolerance after settling.",
            ),
            failure_chain="The required sequence for this diagnostic was not observed.",
            limits="This check evaluates only terminal return margin and post-result motion.",
            next_check="Inspect the first failed diagnostic gate and test the corresponding mechanism.",
        )

    diagnosis = (
        f"The action returned with only {margin:.3f} m of task margin, while the measured pose "
        f"error then grew by {error_growth:.3f} m; the terminal criterion did not reserve enough "
        "margin for the observed post-result motion."
    )
    failure_chain = (
        f"The action reported success at {observation.action_return_error_m:.3f} m error and "
        f"{observation.measured_speed_at_return_mps:.4f} m/s, both within its configured "
        f"{observation.configured_goal_tolerance_m:.3f} m and "
        f"{observation.configured_stopped_speed_mps:.3f} m/s thresholds. After return, the "
        f"platform moved {observation.post_result_coast_m:.3f} m and settled at "
        f"{observation.settled_error_m:.3f} m error, outside the task's "
        f"{observation.task_acceptance_tolerance_m:.3f} m limit."
    )
    return DiagnosticResult(
        **common,
        disposition=DiagnosticDisposition.SUPPORTED,
        diagnosis=diagnosis,
        supporting_evidence=observation.evidence_ids,
        contradictory_evidence=(),
        unresolved_alternatives=(
            "The evidence does not uniquely identify the physical source of the residual motion.",
            "Inertia, disturbance, actuation, and model mismatch remain unresolved alternatives.",
        ),
        failure_chain=failure_chain,
        limits=(
            "This supports a terminal configuration/execution mechanism, not a unique claim about "
            "motor, hydrodynamic, wind, current, wave, or collision causation."
        ),
        next_check=(
            "Repeat from the same initial condition with a prospectively fixed smaller return "
            "tolerance and compare independently measured settled error."
        ),
    )


def diagnose_geometric_route_restriction(
    observation: GeometricRouteObservation,
) -> DiagnosticResult:
    """Diagnose a retained planner-grid restriction without claiming hidden geometry.

    A supported result requires a lethal cell on the requested direct route plus observed route
    deviation.  Grid connectivity determines whether the result may say that the retained model
    was disconnected or must instead report an available modeled detour.  Timing near a configured
    deadline can reconstruct a likely execution mechanism, but an unobserved terminal transition is
    explicitly qualified.
    """

    numeric = {
        "direct_route_minimum_clearance_m": observation.direct_route_minimum_clearance_m,
        "maximum_lateral_deviation_m": observation.maximum_lateral_deviation_m,
        "maximum_forward_progress_m": observation.maximum_forward_progress_m,
        "goal_distance_m": observation.goal_distance_m,
        "action_wall_seconds": observation.action_wall_seconds,
        "configured_deadline_seconds": observation.configured_deadline_seconds,
        "configured_robot_radius_m": observation.configured_robot_radius_m,
        "configured_inflation_radius_m": observation.configured_inflation_radius_m,
        "costmap_resolution_m": observation.costmap_resolution_m,
        "costmap_snapshot_timestamp_s": observation.costmap_snapshot_timestamp_s,
    }
    for name, value in numeric.items():
        _require_nonnegative(name, value)
    if not observation.evidence_ids:
        raise ValueError("at least one robot-visible evidence ID is required")
    if not observation.costmap_snapshot_sha256:
        raise ValueError("costmap snapshot SHA-256 is required")
    if observation.successful_plan_count is not None and observation.successful_plan_count < 0:
        raise ValueError("successful_plan_count must be nonnegative")

    measurements: list[DiagnosticMeasurement] = [
        DiagnosticMeasurement(
            id="action_status",
            value=observation.action_status.lower(),
            unit="status",
            frame=None,
            evidence_ids=observation.evidence_ids,
        ),
        DiagnosticMeasurement(
            id="costmap_snapshot_sha256",
            value=observation.costmap_snapshot_sha256,
            unit="sha256",
            frame=observation.frame,
            evidence_ids=observation.evidence_ids,
            timestamp_s=observation.costmap_snapshot_timestamp_s,
        ),
        DiagnosticMeasurement(
            id="configured_robot_radius",
            value=observation.configured_robot_radius_m,
            unit="m",
            frame=observation.frame,
            evidence_ids=observation.evidence_ids,
        ),
        DiagnosticMeasurement(
            id="configured_inflation_radius",
            value=observation.configured_inflation_radius_m,
            unit="m",
            frame=observation.frame,
            evidence_ids=observation.evidence_ids,
        ),
        DiagnosticMeasurement(
            id="costmap_resolution",
            value=observation.costmap_resolution_m,
            unit="m/cell",
            frame=observation.frame,
            evidence_ids=observation.evidence_ids,
        ),
    ]
    optional_measurements = (
        ("direct_route_minimum_clearance", observation.direct_route_minimum_clearance_m, "m"),
        ("maximum_lateral_deviation", observation.maximum_lateral_deviation_m, "m"),
        ("maximum_forward_progress", observation.maximum_forward_progress_m, "m"),
        ("goal_distance", observation.goal_distance_m, "m"),
        ("action_wall_time", observation.action_wall_seconds, "s"),
        ("configured_deadline", observation.configured_deadline_seconds, "s"),
    )
    for measurement_id, value, unit in optional_measurements:
        if value is not None:
            measurements.append(
                DiagnosticMeasurement(
                    id=measurement_id,
                    value=value,
                    unit=unit,
                    frame=observation.frame,
                    evidence_ids=observation.evidence_ids,
                )
            )
    if observation.direct_route_first_lethal_x_m is not None:
        measurements.append(
            DiagnosticMeasurement(
                id="first_lethal_route_x",
                value=observation.direct_route_first_lethal_x_m,
                unit="m",
                frame=observation.frame,
                evidence_ids=observation.evidence_ids,
                timestamp_s=observation.costmap_snapshot_timestamp_s,
            )
        )
    if observation.direct_route_first_lethal_y_m is not None:
        measurements.append(
            DiagnosticMeasurement(
                id="first_lethal_route_y",
                value=observation.direct_route_first_lethal_y_m,
                unit="m",
                frame=observation.frame,
                evidence_ids=observation.evidence_ids,
                timestamp_s=observation.costmap_snapshot_timestamp_s,
            )
        )

    deadline_delta = None
    deadline_aligned = False
    if (
        observation.action_wall_seconds is not None
        and observation.configured_deadline_seconds is not None
    ):
        deadline_delta = abs(
            observation.action_wall_seconds - observation.configured_deadline_seconds
        )
        deadline_aligned = (
            observation.action_status.lower() == "aborted"
            and deadline_delta
            <= max(2.0, observation.configured_deadline_seconds * 0.05)
        )
        measurements.append(
            DiagnosticMeasurement(
                id="absolute_deadline_timing_difference",
                value=deadline_delta,
                unit="s",
                frame=None,
                evidence_ids=observation.evidence_ids,
            )
        )

    common = dict(
        schema_version="crane-diagnostic-result-v1",
        diagnostic_id=f"{observation.episode_id}:geometric-route-restriction",
        episode_id=observation.episode_id,
        mechanism="geometric_route_restriction",
        measurements=tuple(measurements),
        computation=(
            "decode and hash-check retained costmap; sample requested direct route against "
            "cost>=253; test 8-connected traversal below cost 253; summarize delivered odometry"
        ),
        computation_version=observation.computation_version,
        assumptions=(
            "Costmap values at or above 253 are non-traversable for this audit.",
            "The retained cost field already incorporates the configured robot radius and inflation.",
            "Connectivity applies only to the retained grid, frame, and timestamp.",
            "Delivered odometry records motion but is not proof of Nav2's internal consumed state.",
        ),
        supporting_evidence=observation.evidence_ids,
        causal_language_level=CausalLanguageLevel.EXECUTION_MECHANISM,
        source_anchor_ids=observation.source_anchor_ids,
        decisive_measurement_ids=(
            "direct_route_minimum_clearance",
            "maximum_lateral_deviation",
            "action_wall_time",
            "configured_deadline",
        ),
    )

    if observation.action_status.lower() == "succeeded":
        nominal_measurement_ids = ["action_status"]
        nominal_context = ""
        nominal_limit = (
            "The retained route classification is unavailable, so success alone does not "
            "establish that the requested route was clear. "
        )
        if observation.direct_route_has_lethal_cell is False:
            if observation.direct_route_minimum_clearance_m is not None:
                nominal_measurement_ids.append("direct_route_minimum_clearance")
            if observation.maximum_lateral_deviation_m is not None:
                nominal_measurement_ids.append("maximum_lateral_deviation")
            if observation.maximum_forward_progress_m is not None:
                nominal_measurement_ids.append("maximum_forward_progress")
            clearance = (
                f" with {observation.direct_route_minimum_clearance_m:.3f} m minimum clearance "
                "from a lethal cell"
                if observation.direct_route_minimum_clearance_m is not None
                else ""
            )
            deviation = (
                f"; delivered odometry deviated at most "
                f"{observation.maximum_lateral_deviation_m:.3f} m laterally"
                if observation.maximum_lateral_deviation_m is not None
                else ""
            )
            nominal_context = (
                " The fully covered direct-route audit found no costmap cell at or above 253"
                f"{clearance}{deviation}."
            )
            nominal_limit = (
                "The retained snapshot supports only a bounded route observation at its recorded "
                "time; it does not establish universal obstacle freedom or exact planner "
                "consumption. "
            )
        elif observation.direct_route_has_lethal_cell is True:
            if observation.direct_route_first_lethal_x_m is not None:
                nominal_measurement_ids.append("first_lethal_route_x")
            if observation.maximum_lateral_deviation_m is not None:
                nominal_measurement_ids.append("maximum_lateral_deviation")
            location = (
                f" near x={observation.direct_route_first_lethal_x_m:.2f} m"
                if observation.direct_route_first_lethal_x_m is not None
                else ""
            )
            nominal_context = (
                " The retained navigation model marked the requested direct route as restricted"
                f"{location}, but that restriction did not prevent the recorded successful "
                "outcome."
            )
            nominal_limit = (
                "The retained snapshot does not prove when the restriction arose, what physical "
                "object produced it, or whether Nav2 consumed that exact state. "
            )
        nominal_common = {
            **common,
            "mechanism": "no_failure_observed",
            "causal_language_level": CausalLanguageLevel.RECORDED_SEQUENCE,
            "decisive_measurement_ids": tuple(nominal_measurement_ids),
        }
        return DiagnosticResult(
            **nominal_common,
            disposition=DiagnosticDisposition.NOT_TRIGGERED,
            diagnosis=(
                "The action succeeded, so the question's failure premise is false. The retained "
                "measurements do not establish the route-restriction-and-deadline failure "
                f"mechanism in this episode.{nominal_context}"
            ),
            contradictory_evidence=(),
            unresolved_alternatives=(
                "A successful result does not prove that no temporary route constraint or "
                "control difficulty occurred during execution.",
            ),
            failure_chain=(
                "No terminal failure chain is recorded: the navigation action returned success."
            ),
            limits=(
                "This nominal outcome rejects the failure premise for this episode. "
                f"{nominal_limit}"
                "A successful result does not prove that no temporary route constraint or "
                "control difficulty occurred during execution."
            ),
            next_check=(
                "Use this episode as a nominal comparator and inspect synchronized plans and "
                "costmaps only if transient restrictions are the question of interest."
            ),
        )

    missing = []
    if observation.direct_route_has_lethal_cell is None:
        missing.append("direct-route cost classification")
    if observation.grid_connected is None:
        missing.append("retained-grid connectivity")
    if observation.maximum_lateral_deviation_m is None:
        missing.append("measured lateral trajectory extent")
    if observation.action_wall_seconds is None:
        missing.append("action wall time")
    if observation.configured_deadline_seconds is None:
        missing.append("configured task deadline")
    if missing:
        if deadline_aligned:
            assert deadline_delta is not None
            diagnosis = (
                "The retained evidence cannot establish the physical/geometric reason navigation "
                f"remained incomplete, but the action abort was aligned within {deadline_delta:.2f} s "
                f"of the configured {observation.configured_deadline_seconds:.1f} s task deadline."
            )
            failure_chain = (
                "Planning and delivered motion were recorded, but the missing costmap cells leave "
                "the geometry-to-motion link unresolved. The action then aborted at "
                f"{observation.action_wall_seconds:.2f} s, consistent with the source-qualified "
                "task deadline; the terminal deadline tick was not directly observed."
            )
            mechanism = "deadline_aligned_abort_with_unresolved_geometry"
        else:
            diagnosis = (
                "The geometric route restriction cannot be assessed from the retained evidence."
            )
            failure_chain = (
                "A terminal action result is retained, but the geometry-to-motion chain is incomplete."
            )
            mechanism = "geometric_route_restriction"
        return DiagnosticResult(
            **{**common, "mechanism": mechanism},
            disposition=DiagnosticDisposition.INSUFFICIENT,
            diagnosis=diagnosis,
            contradictory_evidence=(),
            unresolved_alternatives=(
                "The direct route may have been clear or restricted in the relevant planner model.",
                (
                    "The missing terminal BT transition prevents direct observation of the "
                    "deadline decorator's final tick."
                    if deadline_aligned and not observation.terminal_transition_observed
                    else "The terminal action mechanism remains unresolved."
                ),
            ),
            failure_chain=failure_chain,
            limits=(
                f"Missing decisive physical evidence: {', '.join(missing)}. "
                "Deadline alignment reconstructs a bounded execution mechanism from source and "
                "timing; it does not establish why navigation remained incomplete."
                if deadline_aligned
                else f"Missing decisive evidence: {', '.join(missing)}."
            ),
            next_check=(
                "Retain a hash-checked global costmap synchronized with planned paths around the "
                "first deviation to diagnose the unresolved physical mechanism."
                if deadline_aligned
                else "Retain a hash-checked global costmap, synchronized trajectory, and exact task deadline."
            ),
        )

    assert observation.maximum_lateral_deviation_m is not None
    assert observation.action_wall_seconds is not None
    assert observation.configured_deadline_seconds is not None
    assert deadline_delta is not None

    supported = (
        observation.direct_route_has_lethal_cell is True
        and observation.maximum_lateral_deviation_m > observation.configured_inflation_radius_m
        and observation.action_status.lower() == "aborted"
        and deadline_delta <= max(2.0, observation.configured_deadline_seconds * 0.05)
    )
    if not supported:
        return DiagnosticResult(
            **common,
            disposition=DiagnosticDisposition.NOT_TRIGGERED,
            diagnosis="The retained measurements do not establish the route-restriction-and-deadline mechanism.",
            contradictory_evidence=observation.evidence_ids,
            unresolved_alternatives=(
                "A different geometric, planning, control, or terminal mechanism may apply.",
            ),
            failure_chain="At least one required geometric, motion, status, or timing condition was not observed.",
            limits="This check does not infer hidden obstacle identity or universal route infeasibility.",
            next_check="Inspect the failed diagnostic gate against the synchronized grid and trajectory.",
        )

    connection = (
        "the retained grid still contained a traversable connection"
        if observation.grid_connected
        else "the retained grid contained no traversable connection between the audited endpoints"
    )
    lethal_location = ""
    if observation.direct_route_first_lethal_x_m is not None:
        lethal_location = f" near x={observation.direct_route_first_lethal_x_m:.2f} m"
    diagnosis = (
        f"The retained navigation model marked the requested direct route as non-traversable"
        f"{lethal_location}, while {connection}. The robot deviated "
        f"{observation.maximum_lateral_deviation_m:.2f} m laterally, and the action aborted "
        f"{deadline_delta:.2f} s from its configured {observation.configured_deadline_seconds:.1f} s deadline."
    )
    plan_text = (
        f"{observation.successful_plan_count} successful planning updates were recorded"
        if observation.successful_plan_count is not None
        else "planning updates were recorded"
    )
    return DiagnosticResult(
        **common,
        disposition=DiagnosticDisposition.SUPPORTED,
        diagnosis=diagnosis,
        contradictory_evidence=(
            "retained-grid-connectivity" if observation.grid_connected else ""
        ,) if observation.grid_connected else (),
        unresolved_alternatives=(
            "The robot-visible evidence does not identify the hidden obstacle semantic ID.",
            "The retained snapshot does not prove the exact grid state consumed by every planning update.",
            "The missing terminal BT transition prevents direct observation of the deadline decorator's final tick."
            if not observation.terminal_transition_observed
            else "No unresolved terminal-transition observation gap remains.",
        ),
        failure_chain=(
            f"The direct route intersected lethal costmap cells, {plan_text}, and delivered odometry "
            f"recorded a substantial detour. The action then failed at {observation.action_wall_seconds:.2f} s, "
            "consistent with the source-qualified task deadline rather than recovery exhaustion."
        ),
        limits=(
            "This establishes a restriction in the retained navigation model and a deadline-aligned abort. "
            "It does not prove global physical infeasibility, a unique obstacle identity, or that the retained "
            "snapshot was the exact state consumed by every planner invocation."
        ),
        next_check=(
            "Retain time-aligned global-costmap snapshots and full planned paths around the first deviation "
            "to distinguish a feasible but too-long detour from changing map state."
        ),
    )


def diagnose_recovery_execution_sequence(
    observation: RecoveryExecutionObservation,
) -> DiagnosticResult:
    """Reconstruct a source-qualified recovery sequence without inventing physical causation.

    Each accepted invocation must be a unique, complete capture-side invocation whose start and
    end transitions match the configured recovery-leaf classifier.  It must also be preceded,
    since the prior retained invocation ended, by the ordered sequence ``ComputePathToPose``
    failure, navigation-pipeline failure, successful planner-recovery eligibility check, and
    system-recovery entry.  Missing context yields an insufficient result rather than silently
    treating any recovery-named transition as an invocation.
    """

    if not observation.evidence_ids:
        raise ValueError("at least one robot-visible evidence ID is required")
    if not observation.goal_id:
        raise ValueError("goal_id is required")
    if not observation.recovery_policy_sha256:
        raise ValueError("recovery_policy_sha256 is required")
    if (
        observation.maximum_feedback_recovery_count is not None
        and observation.maximum_feedback_recovery_count < 0
    ):
        raise ValueError("maximum_feedback_recovery_count must be nonnegative")

    transition_ids = [transition.record_id for transition in observation.transitions]
    if len(transition_ids) != len(set(transition_ids)):
        raise ValueError("transition record IDs must be unique")
    invocation_ids = [item.invocation_id for item in observation.recovery_invocations]
    if len(invocation_ids) != len(set(invocation_ids)):
        raise ValueError("recovery invocation IDs must be unique")

    transitions_by_id = {
        transition.record_id: transition for transition in observation.transitions
    }
    transition_index = {
        transition.record_id: index
        for index, transition in enumerate(observation.transitions)
    }
    ordered_invocations = sorted(
        observation.recovery_invocations,
        key=lambda item: transition_index.get(item.start_transition_id, math.inf),
    )

    qualified: list[RecoveryInvocationRecord] = []
    context_ids: list[str] = []
    rejected_reasons: list[str] = []
    lower_bound = 0

    def nearest_before(
        start_index: int,
        lower_index: int,
        node_name: str,
        current_status: str,
    ) -> int | None:
        for index in range(start_index - 1, lower_index - 1, -1):
            transition = observation.transitions[index]
            if (
                transition.goal_id == observation.goal_id
                and transition.node_name == node_name
                and transition.current_status.upper() == current_status
            ):
                return index
        return None

    for invocation in ordered_invocations:
        start = transitions_by_id.get(invocation.start_transition_id)
        end = (
            transitions_by_id.get(invocation.end_transition_id)
            if invocation.end_transition_id is not None
            else None
        )
        source_qualified = (
            invocation.goal_id == observation.goal_id
            and invocation.policy_sha256 == observation.recovery_policy_sha256
            and invocation.classifier_basis
            == "configured_exact_node_name_allowlist"
            and invocation.classifier_rule == "configured_leaf_idle_to_running"
            and invocation.observed_start_transition_id
            == invocation.start_transition_id
            and start is not None
            and start.goal_id == observation.goal_id
            and start.node_name == invocation.node_name
            and start.previous_status.upper() == "IDLE"
            and start.current_status.upper() == "RUNNING"
        )
        complete = (
            invocation.complete
            and invocation.terminal_status is not None
            and end is not None
            and end.goal_id == observation.goal_id
            and end.node_name == invocation.node_name
            and end.previous_status.upper() == "RUNNING"
            and end.current_status.upper() == invocation.terminal_status.upper()
            and transition_index[end.record_id] > transition_index[start.record_id]
            if start is not None
            else False
        )
        if not source_qualified:
            rejected_reasons.append(
                f"{invocation.invocation_id} lacks matching bounded classifier provenance"
            )
            continue
        if not complete:
            rejected_reasons.append(
                f"{invocation.invocation_id} lacks a matching complete terminal transition"
            )
            continue

        assert start is not None
        start_index = transition_index[start.record_id]
        system_index = nearest_before(
            start_index, lower_bound, "RecoveryFallback", "RUNNING"
        )
        pipeline_index = (
            nearest_before(
                system_index, lower_bound, "NavigateWithReplanning", "FAILURE"
            )
            if system_index is not None
            else None
        )
        planner_index = (
            nearest_before(
                pipeline_index, lower_bound, "ComputePathToPose", "FAILURE"
            )
            if pipeline_index is not None
            else None
        )
        eligibility_index = None
        if pipeline_index is not None and system_index is not None:
            eligibility_index = next(
                (
                    index
                    for index in range(pipeline_index + 1, system_index)
                    if observation.transitions[index].goal_id == observation.goal_id
                    and observation.transitions[index].node_name
                    == "WouldAPlannerRecoveryHelp"
                    and observation.transitions[index].current_status.upper() == "SUCCESS"
                ),
                None,
            )
        if None in (planner_index, pipeline_index, eligibility_index, system_index):
            rejected_reasons.append(
                f"{invocation.invocation_id} lacks the ordered planner-failure and "
                "recovery-eligibility context"
            )
            lower_bound = transition_index[end.record_id] + 1
            continue

        assert planner_index is not None
        assert pipeline_index is not None
        assert eligibility_index is not None
        assert system_index is not None
        qualified.append(invocation)
        context_ids.extend(
            observation.transitions[index].record_id
            for index in (
                planner_index,
                pipeline_index,
                eligibility_index,
                system_index,
                start_index,
                transition_index[end.record_id],
            )
        )
        lower_bound = transition_index[end.record_id] + 1

    count = len(qualified)
    sequence = ", ".join(
        f"{item.node_name}->{item.terminal_status}" for item in qualified
    )
    count_measurement_id = (
        "source_qualified_recovery_invocation_count"
        if observation.whole_history_complete
        else "minimum_source_qualified_recovery_invocation_count"
    )
    measurements: list[DiagnosticMeasurement] = [
        DiagnosticMeasurement(
            id="action_status",
            value=observation.action_status.lower(),
            unit="status",
            frame=None,
            evidence_ids=observation.evidence_ids,
        ),
        DiagnosticMeasurement(
            id=count_measurement_id,
            value=count,
            unit="invocations",
            frame=None,
            evidence_ids=tuple(dict.fromkeys((*observation.evidence_ids, *context_ids))),
        ),
        DiagnosticMeasurement(
            id="qualified_recovery_sequence",
            value=sequence or "none",
            unit="ordered_nodes_and_terminal_statuses",
            frame=None,
            evidence_ids=tuple(dict.fromkeys((*observation.evidence_ids, *context_ids))),
        ),
        DiagnosticMeasurement(
            id="bt_history_completeness",
            value="complete" if observation.whole_history_complete else "not_proven",
            unit="status",
            frame=None,
            evidence_ids=observation.evidence_ids,
        ),
    ]
    if observation.maximum_feedback_recovery_count is not None:
        measurements.append(
            DiagnosticMeasurement(
                id="maximum_nav2_feedback_recovery_count",
                value=observation.maximum_feedback_recovery_count,
                unit="feedback_count_not_invocation_identity",
                frame=None,
                evidence_ids=observation.evidence_ids,
            )
        )

    common = dict(
        schema_version="crane-diagnostic-result-v1",
        diagnostic_id=f"{observation.episode_id}:recovery-execution-sequence",
        episode_id=observation.episode_id,
        measurements=tuple(measurements),
        computation=(
            "validate unique invocation IDs; match policy hash and exact-node classifier; "
            "match IDLE-to-RUNNING start and RUNNING-to-terminal transitions; reconstruct "
            "ordered ComputePathToPose failure, NavigateWithReplanning failure, "
            "WouldAPlannerRecoveryHelp success, and RecoveryFallback entry"
        ),
        computation_version=observation.computation_version,
        assumptions=(
            "The retained BT transitions are ordered as observed for one accepted goal.",
            "The hash-pinned exact-name policy classifies recovery leaves, not every feedback increment.",
            "Software execution ordering does not identify the physical cause of planning failure.",
        ),
        supporting_evidence=tuple(
            dict.fromkeys((*observation.evidence_ids, *context_ids))
        ),
        contradictory_evidence=tuple(rejected_reasons),
        unresolved_alternatives=(
            "The physical condition that caused each planner failure is unresolved.",
            "Costmap delivery would not by itself prove the exact state consumed by the planner.",
        ),
        causal_language_level=CausalLanguageLevel.EXECUTION_MECHANISM,
        source_anchor_ids=observation.source_anchor_ids,
        decisive_measurement_ids=(
            count_measurement_id,
            "qualified_recovery_sequence",
            "action_status",
            "bt_history_completeness",
        ),
    )

    if not observation.recovery_invocations:
        return DiagnosticResult(
            **common,
            mechanism="recovery_execution_sequence",
            disposition=DiagnosticDisposition.NOT_TRIGGERED,
            diagnosis="No retained source-qualified recovery invocation was available to diagnose.",
            failure_chain="The required recovery execution sequence was not observed.",
            limits="This absence is not an exact zero unless whole-history completeness is proven.",
            next_check="Inspect the goal-scoped BT capture and its completeness record.",
        )

    if count != len(observation.recovery_invocations):
        return DiagnosticResult(
            **common,
            mechanism="recovery_execution_sequence_with_missing_context",
            disposition=DiagnosticDisposition.INSUFFICIENT,
            diagnosis=(
                f"Only {count} of {len(observation.recovery_invocations)} retained recovery "
                "records could be linked to complete source-qualified planner-failure and "
                "recovery-eligibility sequences."
            ),
            failure_chain=(
                "At least one retained recovery record lacked the transition context required "
                "to reconstruct why the behavior tree entered that recovery leaf."
            ),
            limits=(
                "The incomplete linkage prevents a complete recovery-mechanism account. "
                "The Nav2 feedback recovery count is not a recovery-invocation identity."
            ),
            next_check=(
                "Retain the goal-scoped planner failure, eligibility check, system-recovery entry, "
                "and leaf terminal transition around every invocation."
            ),
        )

    quantifier = "exactly" if observation.whole_history_complete else "at least"
    history_limit = (
        "Whole-history completeness is proven for the accepted goal."
        if observation.whole_history_complete
        else "Whole-history completeness is not proven, so the retained count is a lower bound."
    )
    physical_limit = (
        "This diagnostic does not independently validate the separately asserted physical cause."
        if observation.physical_cause_established
        else "The robot-visible evidence does not establish the physical cause of the planner failures."
    )
    return DiagnosticResult(
        **common,
        mechanism="planner_failure_eligible_recovery_sequence",
        disposition=DiagnosticDisposition.SUPPORTED,
        diagnosis=(
            f"The retained behavior-tree evidence establishes {quantifier} {count} "
            f"source-qualified recovery invocations ({sequence}). Each followed a planner "
            "branch failure and a successful recovery-eligibility check."
        ),
        failure_chain=(
            f"For each of the {count} retained invocations, ComputePathToPose failed, the "
            "NavigateWithReplanning pipeline failed, WouldAPlannerRecoveryHelp succeeded, and "
            f"the tree entered a recovery leaf. The navigation action eventually "
            f"{observation.action_status.lower()}."
        ),
        limits=(
            f"{history_limit} {physical_limit} The maximum Nav2 feedback recovery count is "
            "retained only as feedback and is not treated as the number of unique BT invocations."
        ),
        next_check=(
            "Time-align retained plans and hash-checked costmap observations around the first "
            "planner failure to test a physical route-restriction mechanism."
        ),
    )


def render_diagnostic(result: DiagnosticResult) -> str:
    """Render the checked result without introducing additional propositions."""

    selected = (
        tuple(item for item in result.measurements if item.id in result.decisive_measurement_ids)
        if result.decisive_measurement_ids
        else result.measurements
    )
    evidence = "; ".join(
        f"{item.id}={item.value:.4f} {item.unit}"
        if isinstance(item.value, float)
        else f"{item.id}={item.value} {item.unit}"
        for item in selected
    )
    alternatives = " ".join(
        alternative
        for alternative in result.unresolved_alternatives
        if alternative.strip() not in result.limits
    )
    limits_and_alternatives = " ".join(
        part for part in (result.limits, alternatives) if part
    )
    return "\n".join(
        (
            f"Diagnosis: {result.diagnosis}",
            f"Decisive evidence: {evidence}. Evidence IDs: {', '.join(result.supporting_evidence)}.",
            f"Failure chain: {result.failure_chain}",
            f"Limits and next check: {limits_and_alternatives} Next check: {result.next_check}",
        )
    )


def verify_diagnostic_text(
    result: DiagnosticResult, text: str
) -> DiagnosticTextVerification:
    """Fail closed unless final text is the deterministic checked rendering."""

    expected = render_diagnostic(result)
    return DiagnosticTextVerification(accepted=text == expected, expected_text=expected)
