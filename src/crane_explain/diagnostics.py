"""Prospective physical-diagnosis records and bounded computations.

This module is deliberately separate from the frozen legacy reasoning path.  It starts with one
auditable mechanism: whether an action's configured return tolerance left enough distance for the
measured platform to settle inside a separately declared task tolerance.
"""

from __future__ import annotations

import math
import statistics
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
class GridDisconnectionObservation:
    """Robot-visible inputs for a bounded retained-grid disconnection diagnosis.

    This contract is for planner failures where a hash-checked navigation grid can be tested
    directly.  It establishes only a disconnection in that retained model at its timestamp.  It
    does not identify a physical obstacle, prove that the planner consumed this exact snapshot,
    or prove that no route existed outside the retained grid.
    """

    episode_id: str
    evidence_ids: tuple[str, ...]
    frame: str
    action_status: str
    start_x_m: float
    start_y_m: float
    goal_x_m: float
    goal_y_m: float
    start_cost: int
    goal_cost: int
    blocked_cost_threshold: int
    grid_connected_below_threshold: bool | None
    planner_failure_message_count: int | None
    planner_error_text: str | None
    action_wall_seconds: float | None
    costmap_resolution_m: float
    costmap_snapshot_sha256: str
    costmap_snapshot_timestamp_s: float | None
    source_anchor_ids: tuple[str, ...] = ()
    computation_version: str = "retained-grid-disconnection-v1"


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
class CommandMotionWindow:
    """One fixed robot-visible command/odometry comparison window."""

    index: int
    start_offset_s: float
    end_offset_s: float
    command_sample_count: int
    odometry_sample_count: int
    median_commanded_planar_speed_mps: float | None
    median_measured_planar_speed_mps: float | None


@dataclass(frozen=True)
class CommandMotionObservation:
    """Robot-visible inputs for a bounded command-to-motion discrepancy check.

    Command delivery does not prove actuator acceptance, and delivered odometry does not prove
    Nav2 consumption.  This contract therefore supports an execution-layer discrepancy without
    assigning it to a motor, collision, slip, or another unique physical cause.
    """

    episode_id: str
    evidence_ids: tuple[str, ...]
    command_frame: str
    measured_frame: str
    action_status: str
    windows: tuple[CommandMotionWindow, ...]
    raw_command_sample_count: int
    raw_odometry_sample_count: int
    window_seconds: float
    minimum_command_samples_per_window: int
    minimum_odometry_samples_per_window: int
    calibration_window_count: int
    minimum_commanded_speed_mps: float
    minimum_healthy_measured_speed_mps: float
    maximum_discrepancy_response_ratio: float
    minimum_consecutive_discrepancy_windows: int
    follow_path_failure_count: int
    follow_path_attempt_count: int
    source_qualified_recovery_count: int
    source_anchor_ids: tuple[str, ...] = ()
    computation_version: str = "command-motion-discrepancy-v2"


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


def diagnose_retained_grid_disconnection(
    observation: GridDisconnectionObservation,
) -> DiagnosticResult:
    """Diagnose a planner-model disconnection while withholding physical overclaiming."""

    numeric = {
        "action_wall_seconds": observation.action_wall_seconds,
        "costmap_resolution_m": observation.costmap_resolution_m,
        "costmap_snapshot_timestamp_s": observation.costmap_snapshot_timestamp_s,
    }
    for name, value in numeric.items():
        _require_nonnegative(name, value)
    if not observation.evidence_ids:
        raise ValueError("at least one robot-visible evidence ID is required")
    if not observation.costmap_snapshot_sha256:
        raise ValueError("costmap snapshot SHA-256 is required")
    if observation.blocked_cost_threshold < 1 or observation.blocked_cost_threshold > 255:
        raise ValueError("blocked_cost_threshold must be in [1, 255]")
    for name, value in (("start_cost", observation.start_cost), ("goal_cost", observation.goal_cost)):
        if value < 0 or value > 255:
            raise ValueError(f"{name} must be in [0, 255]")
    if (
        observation.planner_failure_message_count is not None
        and observation.planner_failure_message_count < 0
    ):
        raise ValueError("planner_failure_message_count must be nonnegative")

    measurements = (
        DiagnosticMeasurement(
            id="action_status",
            value=observation.action_status.lower(),
            unit="status",
            frame=None,
            evidence_ids=observation.evidence_ids,
        ),
        DiagnosticMeasurement(
            id="start_grid_cost",
            value=observation.start_cost,
            unit="cost",
            frame=observation.frame,
            evidence_ids=observation.evidence_ids,
            timestamp_s=observation.costmap_snapshot_timestamp_s,
        ),
        DiagnosticMeasurement(
            id="goal_grid_cost",
            value=observation.goal_cost,
            unit="cost",
            frame=observation.frame,
            evidence_ids=observation.evidence_ids,
            timestamp_s=observation.costmap_snapshot_timestamp_s,
        ),
        DiagnosticMeasurement(
            id="blocked_cost_threshold",
            value=observation.blocked_cost_threshold,
            unit="cost",
            frame=observation.frame,
            evidence_ids=observation.evidence_ids,
        ),
        DiagnosticMeasurement(
            id="retained_grid_connected_below_threshold",
            value=(
                "unknown"
                if observation.grid_connected_below_threshold is None
                else str(observation.grid_connected_below_threshold).lower()
            ),
            unit="boolean",
            frame=observation.frame,
            evidence_ids=observation.evidence_ids,
            timestamp_s=observation.costmap_snapshot_timestamp_s,
        ),
        DiagnosticMeasurement(
            id="planner_failure_messages",
            value=(
                "unknown"
                if observation.planner_failure_message_count is None
                else observation.planner_failure_message_count
            ),
            unit="messages",
            frame=None,
            evidence_ids=observation.evidence_ids,
        ),
        DiagnosticMeasurement(
            id="costmap_resolution",
            value=observation.costmap_resolution_m,
            unit="m/cell",
            frame=observation.frame,
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
    )
    common = dict(
        schema_version="crane-diagnostic-result-v1",
        diagnostic_id=f"{observation.episode_id}:retained-grid-disconnection",
        episode_id=observation.episode_id,
        mechanism="retained_navigation_model_disconnection",
        measurements=measurements,
        computation=(
            "decode and hash-check retained costmap; map result and goal poses to cells; test "
            "8-connected traversal through cells below the declared blocked-cost threshold; "
            "count exact retained planner-failure log messages"
        ),
        computation_version=observation.computation_version,
        assumptions=(
            "The retained grid metadata and pose coordinates use the same frame.",
            "Cells at or above the declared threshold are excluded by this audit.",
            "Connectivity applies only to the retained grid and timestamp.",
            "A retained log message records software output, not a unique physical cause.",
        ),
        supporting_evidence=observation.evidence_ids,
        causal_language_level=CausalLanguageLevel.EXECUTION_MECHANISM,
        source_anchor_ids=observation.source_anchor_ids,
        decisive_measurement_ids=(
            "start_grid_cost",
            "goal_grid_cost",
            "blocked_cost_threshold",
            "retained_grid_connected_below_threshold",
            "planner_failure_messages",
            "action_status",
        ),
    )

    missing = []
    if observation.grid_connected_below_threshold is None:
        missing.append("retained-grid connectivity")
    if observation.planner_failure_message_count is None:
        missing.append("planner-failure log count")
    if not observation.planner_error_text:
        missing.append("exact planner error text")
    if missing:
        return DiagnosticResult(
            **common,
            disposition=DiagnosticDisposition.INSUFFICIENT,
            diagnosis="The retained navigation-model disconnection cannot be established.",
            contradictory_evidence=(),
            unresolved_alternatives=(
                "The planner may have failed because of geometry, transient map state, or another planning condition.",
            ),
            failure_chain="The action result is retained, but the model-to-planner failure chain is incomplete.",
            limits=f"Missing decisive evidence: {', '.join(missing)}.",
            next_check="Retain a hash-checked grid and the exact planner result for the same goal-scoped interval.",
        )

    assert observation.planner_failure_message_count is not None
    assert observation.planner_error_text is not None
    supported = (
        observation.action_status.lower() == "aborted"
        and observation.start_cost < observation.blocked_cost_threshold
        and observation.goal_cost >= observation.blocked_cost_threshold
        and observation.grid_connected_below_threshold is False
        and observation.planner_failure_message_count > 0
    )
    if not supported:
        return DiagnosticResult(
            **common,
            disposition=DiagnosticDisposition.NOT_TRIGGERED,
            diagnosis="The retained measurements do not establish a navigation-model disconnection linked to the abort.",
            contradictory_evidence=observation.evidence_ids,
            unresolved_alternatives=(
                "A different geometric, planning, control, or terminal mechanism may apply.",
            ),
            failure_chain="At least one required grid, planner-message, or action-result condition was not observed.",
            limits="This check does not infer a physical obstacle or a route outside the retained grid.",
            next_check="Inspect the failed diagnostic gate against a synchronized grid and planner result.",
        )

    return DiagnosticResult(
        **common,
        disposition=DiagnosticDisposition.SUPPORTED,
        diagnosis=(
            f"The retained navigation model contained no 8-connected route below cost "
            f"{observation.blocked_cost_threshold} from the recorded result pose to the goal. "
            f"The goal cell cost was {observation.goal_cost}, and "
            f"{observation.planner_failure_message_count} matching planner-failure messages were "
            "retained before the action aborted."
        ),
        contradictory_evidence=(),
        unresolved_alternatives=(
            "The retained evidence does not identify the physical object or configuration that produced the blocked grid cells.",
            "The snapshot does not prove the exact costmap state consumed by every planner invocation.",
            "No connection in the retained grid does not prove that no physical route existed outside that grid or under another configuration.",
        ),
        failure_chain=(
            f"The recorded result pose was in a free cell (cost {observation.start_cost}), but "
            f"the requested goal was in a blocked cell (cost {observation.goal_cost}) and no "
            f"connection below cost {observation.blocked_cost_threshold} existed in the retained "
            f"grid. The planner logged '{observation.planner_error_text}' "
            f"{observation.planner_failure_message_count} times, after which the action aborted."
        ),
        limits=(
            "This establishes a bounded navigation-model restriction and its correspondence with "
            "recorded planning failures. It does not establish a unique physical obstacle, exact "
            "planner consumption of the retained snapshot, or global physical infeasibility."
        ),
        next_check=(
            "Retain time-aligned planner inputs and test a prospectively declared corrected goal "
            "from the same initial state and configuration."
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
        # Keep the user-facing citation list concise. The decisive measurements above retain the
        # complete transition-level derivation for audit and reconstruction.
        supporting_evidence=tuple(
            dict.fromkeys(
                (
                    *observation.evidence_ids,
                    *(item.invocation_id for item in qualified),
                )
            )
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


def diagnose_command_motion_discrepancy(
    observation: CommandMotionObservation,
) -> DiagnosticResult:
    """Identify a sustained loss of measured response to nonzero delivered commands.

    The first sufficiently sampled command-active windows calibrate the observed healthy response.
    Later windows trigger only when their measured-motion/healthy-response ratio stays at or below
    the declared threshold for the declared consecutive-window count.  The computation deliberately
    does not infer why the command-to-motion chain diverged.
    """

    if not observation.evidence_ids:
        raise ValueError("at least one robot-visible evidence ID is required")
    positive = {
        "window_seconds": observation.window_seconds,
        "minimum_command_samples_per_window": observation.minimum_command_samples_per_window,
        "minimum_odometry_samples_per_window": observation.minimum_odometry_samples_per_window,
        "calibration_window_count": observation.calibration_window_count,
        "minimum_commanded_speed_mps": observation.minimum_commanded_speed_mps,
        "minimum_healthy_measured_speed_mps": observation.minimum_healthy_measured_speed_mps,
        "minimum_consecutive_discrepancy_windows": (
            observation.minimum_consecutive_discrepancy_windows
        ),
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    ratio_threshold = observation.maximum_discrepancy_response_ratio
    if not math.isfinite(ratio_threshold) or not 0.0 <= ratio_threshold < 1.0:
        raise ValueError("maximum_discrepancy_response_ratio must be in [0, 1)")
    for name, value in (
        ("raw_command_sample_count", observation.raw_command_sample_count),
        ("raw_odometry_sample_count", observation.raw_odometry_sample_count),
        ("follow_path_failure_count", observation.follow_path_failure_count),
        ("follow_path_attempt_count", observation.follow_path_attempt_count),
        ("source_qualified_recovery_count", observation.source_qualified_recovery_count),
    ):
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")

    previous_index = -1
    for window in observation.windows:
        if window.index <= previous_index:
            raise ValueError("command-motion windows must have strictly increasing indexes")
        previous_index = window.index
        if window.end_offset_s <= window.start_offset_s:
            raise ValueError("command-motion window end must follow its start")
        for value in (
            window.median_commanded_planar_speed_mps,
            window.median_measured_planar_speed_mps,
        ):
            _require_nonnegative("window speed", value)

    common = dict(
        schema_version="crane-diagnostic-result-v1",
        diagnostic_id=f"{observation.episode_id}:command-motion-discrepancy",
        episode_id=observation.episode_id,
        mechanism="command_to_motion_discrepancy",
        computation=(
            "fixed wall-time windows; healthy_response = median(initial sufficiently sampled "
            "command-active window medians); response_ratio = later measured planar speed / "
            "healthy_response; require bounded consecutive low-response windows; after the "
            "earliest discrepancy, report the first sufficiently sampled command-active window "
            "whose measured speed is again above both the low-response boundary and the minimum "
            "healthy measured speed"
        ),
        computation_version=observation.computation_version,
        assumptions=(
            "Command and odometry receipt clocks are comparable within the capture process.",
            "Planar odometry speed is an independent measured-motion signal, not command-derived NavigateToPose feedback.",
            "The initial sufficiently sampled command-active windows represent a healthy response for this run.",
            "Source-qualified recovery counts use the retained behavior-tree policy and transition stream.",
        ),
        causal_language_level=CausalLanguageLevel.EXECUTION_MECHANISM,
        source_anchor_ids=observation.source_anchor_ids,
    )

    missing = []
    if observation.raw_command_sample_count == 0:
        missing.append("delivered Nav2 command stream")
    if observation.raw_odometry_sample_count == 0:
        missing.append("independently delivered odometry stream")
    if missing:
        missing_text = " and ".join(missing)
        measurements = (
            DiagnosticMeasurement(
                id="delivered_command_sample_count",
                value=observation.raw_command_sample_count,
                unit="samples",
                frame=observation.command_frame,
                evidence_ids=observation.evidence_ids,
            ),
            DiagnosticMeasurement(
                id="independent_odometry_sample_count",
                value=observation.raw_odometry_sample_count,
                unit="samples",
                frame=observation.measured_frame,
                evidence_ids=observation.evidence_ids,
            ),
            DiagnosticMeasurement(
                id="action_status",
                value=observation.action_status.lower(),
                unit="status",
                frame=None,
                evidence_ids=observation.evidence_ids,
            ),
            DiagnosticMeasurement(
                id="follow_path_failures",
                value=observation.follow_path_failure_count,
                unit="count",
                frame=None,
                evidence_ids=observation.evidence_ids,
            ),
            DiagnosticMeasurement(
                id="source_qualified_wait_recoveries",
                value=observation.source_qualified_recovery_count,
                unit="count",
                frame=None,
                evidence_ids=observation.evidence_ids,
            ),
        )
        return DiagnosticResult(
            **common,
            disposition=DiagnosticDisposition.INSUFFICIENT,
            diagnosis="The command-to-motion discrepancy cannot be assessed because "
            + missing_text
            + (" is missing." if len(missing) == 1 else " are missing."),
            measurements=measurements,
            supporting_evidence=observation.evidence_ids,
            contradictory_evidence=(),
            unresolved_alternatives=(),
            failure_chain=(
                f"The navigation action {observation.action_status.lower()} after "
                f"{observation.follow_path_failure_count} recorded FollowPath failures and "
                f"{observation.source_qualified_recovery_count} source-qualified Wait recovery "
                f"invocations, but the missing {missing_text} prevents a time-aligned "
                "command-response chain."
            ),
            limits=(
                "Missing robot-visible motion-chain evidence prevents this diagnostic. The "
                "execution sequence alone does not establish a command-to-motion discrepancy or "
                "a unique physical cause."
            ),
            next_check="Record synchronized delivered commands and independently measured planar odometry.",
            decisive_measurement_ids=tuple(item.id for item in measurements),
        )

    eligible = [
        window
        for window in observation.windows
        if window.command_sample_count >= observation.minimum_command_samples_per_window
        and window.odometry_sample_count >= observation.minimum_odometry_samples_per_window
        and window.median_commanded_planar_speed_mps is not None
        and window.median_measured_planar_speed_mps is not None
        and window.median_commanded_planar_speed_mps
        >= observation.minimum_commanded_speed_mps
    ]
    calibration = eligible[: observation.calibration_window_count]
    if len(calibration) < observation.calibration_window_count:
        return DiagnosticResult(
            **common,
            disposition=DiagnosticDisposition.INSUFFICIENT,
            diagnosis="The command-to-motion discrepancy cannot be assessed because too few sufficiently sampled command-active windows were retained for healthy-response calibration.",
            measurements=(),
            supporting_evidence=observation.evidence_ids,
            contradictory_evidence=(),
            unresolved_alternatives=(),
            failure_chain="A calibrated command-response baseline could not be reconstructed.",
            limits="The retained stream does not establish a healthy within-run motion response.",
            next_check="Retain the initial command-active interval with synchronized command and odometry samples.",
        )

    healthy_speed = statistics.median(
        float(window.median_measured_planar_speed_mps) for window in calibration
    )
    healthy_command = statistics.median(
        float(window.median_commanded_planar_speed_mps) for window in calibration
    )
    calibration_end_index = calibration[-1].index
    if healthy_speed < observation.minimum_healthy_measured_speed_mps:
        measurement = DiagnosticMeasurement(
            id="calibrated_healthy_planar_speed",
            value=healthy_speed,
            unit="m/s",
            frame=observation.measured_frame,
            evidence_ids=observation.evidence_ids,
            interval_s=(calibration[0].start_offset_s, calibration[-1].end_offset_s),
        )
        return DiagnosticResult(
            **common,
            disposition=DiagnosticDisposition.INSUFFICIENT,
            diagnosis="The retained initial windows did not establish the minimum healthy measured-motion response required for comparison.",
            measurements=(measurement,),
            supporting_evidence=observation.evidence_ids,
            contradictory_evidence=(),
            unresolved_alternatives=(),
            failure_chain="No calibrated healthy command-to-motion baseline was established.",
            limits="A low response without a healthy reference cannot establish a later response loss.",
            next_check="Capture a nominal command-active interval under the same platform and timing configuration.",
            decisive_measurement_ids=("calibrated_healthy_planar_speed",),
        )

    qualifying = {
        window.index: window
        for window in eligible
        if window.index > calibration_end_index
        and float(window.median_measured_planar_speed_mps) / healthy_speed
        <= ratio_threshold
    }
    runs: list[list[CommandMotionWindow]] = []
    current: list[CommandMotionWindow] = []
    for index in sorted(qualifying):
        window = qualifying[index]
        if current and index != current[-1].index + 1:
            runs.append(current)
            current = []
        current.append(window)
    if current:
        runs.append(current)
    qualifying_runs = [
        run
        for run in runs
        if len(run) >= observation.minimum_consecutive_discrepancy_windows
    ]

    baseline_measurements = (
        DiagnosticMeasurement(
            id="calibrated_healthy_commanded_planar_speed",
            value=healthy_command,
            unit="m/s",
            frame=observation.command_frame,
            evidence_ids=observation.evidence_ids,
            interval_s=(calibration[0].start_offset_s, calibration[-1].end_offset_s),
        ),
        DiagnosticMeasurement(
            id="calibrated_healthy_planar_speed",
            value=healthy_speed,
            unit="m/s",
            frame=observation.measured_frame,
            evidence_ids=observation.evidence_ids,
            interval_s=(calibration[0].start_offset_s, calibration[-1].end_offset_s),
        ),
        DiagnosticMeasurement(
            id="action_status",
            value=observation.action_status.lower(),
            unit="status",
            frame=None,
            evidence_ids=observation.evidence_ids,
        ),
        DiagnosticMeasurement(
            id="follow_path_failures",
            value=observation.follow_path_failure_count,
            unit="count",
            frame=None,
            evidence_ids=observation.evidence_ids,
        ),
        DiagnosticMeasurement(
            id="source_qualified_wait_recoveries",
            value=observation.source_qualified_recovery_count,
            unit="count",
            frame=None,
            evidence_ids=observation.evidence_ids,
        ),
    )
    if not qualifying_runs:
        succeeded = observation.action_status.lower() == "succeeded"
        return DiagnosticResult(
            **common,
            disposition=DiagnosticDisposition.NOT_TRIGGERED,
            diagnosis=(
                (
                    "The navigation action succeeded, so the premise that a command-to-motion "
                    "failure prevented continuation is false. "
                )
                if succeeded
                else ""
            )
            + (
                "The retained streams did not contain the required consecutive low-response "
                "windows after healthy-response calibration."
            ),
            measurements=baseline_measurements,
            supporting_evidence=observation.evidence_ids,
            contradictory_evidence=("required-sustained-low-response-sequence-not-observed",),
            unresolved_alternatives=(),
            failure_chain=(
                "The recorded action succeeded with no observed FollowPath failure or "
                "source-qualified Wait recovery invocation."
                if succeeded
                else "No supported command-to-motion discrepancy failure chain was observed."
            ),
            limits=(
                "This negative check applies only to the retained synchronized interval and "
                "declared thresholds. Success does not prove that every transient execution "
                "difficulty was absent."
                if succeeded
                else "This negative check applies only to the retained synchronized interval and declared thresholds."
            ),
            next_check=(
                "Retain the same synchronized streams if a later run fails; no failure diagnosis "
                "is warranted for this successful action."
                if succeeded
                else "Inspect other physical or execution mechanisms if navigation still failed."
            ),
            decisive_measurement_ids=(
                "action_status",
                "calibrated_healthy_commanded_planar_speed",
                "calibrated_healthy_planar_speed",
                "follow_path_failures",
                "source_qualified_wait_recoveries",
            ),
        )

    # Use the earliest qualifying sequence so the stated failure chain cannot be inverted by a
    # longer post-recovery segment. Later qualifying runs remain represented in the input windows.
    run = sorted(qualifying_runs, key=lambda item: item[0].index)[0]
    discrepancy_command = statistics.median(
        float(window.median_commanded_planar_speed_mps) for window in run
    )
    discrepancy_motion = statistics.median(
        float(window.median_measured_planar_speed_mps) for window in run
    )
    response_ratio = discrepancy_motion / healthy_speed
    duration = run[-1].end_offset_s - run[0].start_offset_s
    recovery_window = next(
        (
            window
            for window in eligible
            if window.index > run[-1].index
            and float(window.median_measured_planar_speed_mps)
            >= observation.minimum_healthy_measured_speed_mps
            and float(window.median_measured_planar_speed_mps) / healthy_speed
            > ratio_threshold
        ),
        None,
    )
    measurements = baseline_measurements + (
        DiagnosticMeasurement(
            id="discrepancy_commanded_planar_speed",
            value=discrepancy_command,
            unit="m/s",
            frame=observation.command_frame,
            evidence_ids=observation.evidence_ids,
            interval_s=(run[0].start_offset_s, run[-1].end_offset_s),
        ),
        DiagnosticMeasurement(
            id="discrepancy_measured_planar_speed",
            value=discrepancy_motion,
            unit="m/s",
            frame=observation.measured_frame,
            evidence_ids=observation.evidence_ids,
            interval_s=(run[0].start_offset_s, run[-1].end_offset_s),
        ),
        DiagnosticMeasurement(
            id="measured_response_ratio",
            value=response_ratio,
            unit="ratio",
            frame=None,
            evidence_ids=observation.evidence_ids,
            interval_s=(run[0].start_offset_s, run[-1].end_offset_s),
        ),
        DiagnosticMeasurement(
            id="sustained_discrepancy_duration",
            value=duration,
            unit="s",
            frame=None,
            evidence_ids=observation.evidence_ids,
            interval_s=(run[0].start_offset_s, run[-1].end_offset_s),
        ),
    )
    recovery_measurement_ids: tuple[str, ...] = ()
    recovery_clause = ""
    if recovery_window is not None:
        recovered_motion = float(recovery_window.median_measured_planar_speed_mps)
        recovered_ratio = recovered_motion / healthy_speed
        recovery_interval = (
            recovery_window.start_offset_s,
            recovery_window.end_offset_s,
        )
        measurements += (
            DiagnosticMeasurement(
                id="recovered_measured_planar_speed",
                value=recovered_motion,
                unit="m/s",
                frame=observation.measured_frame,
                evidence_ids=observation.evidence_ids,
                interval_s=recovery_interval,
            ),
            DiagnosticMeasurement(
                id="recovered_response_ratio",
                value=recovered_ratio,
                unit="ratio",
                frame=None,
                evidence_ids=observation.evidence_ids,
                interval_s=recovery_interval,
            ),
        )
        recovery_measurement_ids = (
            "recovered_measured_planar_speed",
            "recovered_response_ratio",
        )
        recovery_clause = (
            " A later sufficiently sampled command-active window recorded that the measured "
            f"response recovered to {recovered_motion:.3f} m/s "
            f"({recovered_ratio:.3f} of the calibrated healthy response)."
        )
    attempt_clause = (
        f"A third FollowPath attempt was active before the navigation action {observation.action_status.lower()}."
        if observation.follow_path_attempt_count >= 3
        else f"The navigation action {observation.action_status.lower()}."
    )
    return DiagnosticResult(
        **common,
        disposition=DiagnosticDisposition.SUPPORTED,
        diagnosis=(
            "The retained command and odometry streams establish a sustained command-to-motion "
            f"discrepancy: Nav2 continued publishing a median {discrepancy_command:.3f} m/s "
            "planar command while independently delivered odometry recorded a median "
            f"{discrepancy_motion:.3f} m/s planar motion response for {duration:.1f} s."
            + recovery_clause
        ),
        measurements=measurements,
        supporting_evidence=observation.evidence_ids,
        contradictory_evidence=(),
        unresolved_alternatives=(
            "The evidence does not distinguish actuator rejection, a mobility constraint, collision or obstruction, slip, or another execution-layer cause.",
        ),
        failure_chain=(
            f"After the response loss, FollowPath recorded {observation.follow_path_failure_count} "
            f"failures and {observation.source_qualified_recovery_count} source-qualified Wait "
            "recovery invocations ran."
            + (
                " The later measured response recovered before the navigation action "
                f"{observation.action_status.lower()}."
                if recovery_window is not None
                else f" {attempt_clause}"
            )
        ),
        limits=(
            "Delivered Nav2 commands do not prove actuator acceptance, and delivered odometry does "
            "not prove Nav2 consumption. The evidence establishes the discrepancy and its recorded "
            "execution sequence, but not a unique physical cause."
        ),
        next_check=(
            "Record downstream accepted actuation or actuator feedback together with contact, "
            "clearance, and wheel-motion evidence over the discrepancy interval."
        ),
        decisive_measurement_ids=(
            "calibrated_healthy_planar_speed",
            "discrepancy_commanded_planar_speed",
            "discrepancy_measured_planar_speed",
            "measured_response_ratio",
            "sustained_discrepancy_duration",
            "follow_path_failures",
            "source_qualified_wait_recoveries",
        )
        + recovery_measurement_ids,
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
