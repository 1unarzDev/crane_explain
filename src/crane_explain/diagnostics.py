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
    value: float | str
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
        computation_version="terminal-stopping-margin-v1",
        assumptions=(
            "Pose errors use the same goal, metric, and coordinate frame.",
            "Speed and pose are independently measured rather than command-derived feedback.",
            "The declared task tolerance is available to the diagnostic method.",
        ),
        causal_language_level=CausalLanguageLevel.EXECUTION_MECHANISM,
        source_anchor_ids=observation.source_anchor_ids,
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


def render_diagnostic(result: DiagnosticResult) -> str:
    """Render the checked result without introducing additional propositions."""

    evidence = "; ".join(
        f"{item.id}={item.value:.4f} {item.unit}"
        if isinstance(item.value, float)
        else f"{item.id}={item.value} {item.unit}"
        for item in result.measurements
    )
    alternatives = " ".join(result.unresolved_alternatives)
    return "\n".join(
        (
            f"Diagnosis: {result.diagnosis}",
            f"Decisive evidence: {evidence}. Evidence IDs: {', '.join(result.supporting_evidence)}.",
            f"Failure chain: {result.failure_chain}",
            f"Limits and next check: {result.limits} {alternatives} Next check: {result.next_check}",
        )
    )


def verify_diagnostic_text(
    result: DiagnosticResult, text: str
) -> DiagnosticTextVerification:
    """Fail closed unless final text is the deterministic checked rendering."""

    expected = render_diagnostic(result)
    return DiagnosticTextVerification(accepted=text == expected, expected_text=expected)
