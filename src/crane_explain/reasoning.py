"""Checked answer-plan construction for decision and recovery questions."""

from __future__ import annotations

from .models import (
    AnswerPlan,
    Candidate,
    CandidateStatus,
    Claim,
    ClaimClass,
    EpisodeRecord,
    EvidenceLevel,
    SupportStatus,
)
from .provenance import ResolvedProvenance, mechanism_source_anchor_ids


def _candidate(episode: EpisodeRecord, candidate_id: str) -> Candidate | None:
    if not episode.decision:
        return None
    return next((c for c in episode.decision.candidates if c.id == candidate_id), None)


def _join(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def plan_contrast(episode: EpisodeRecord, alternative_id: str) -> AnswerPlan:
    decision = episode.decision
    if not decision:
        return AnswerPlan("contrast", "abstain", (), ("No decision record is available.",))
    selected = _candidate(episode, decision.selected_id)
    alternative = _candidate(episode, alternative_id)
    if alternative is None:
        return AnswerPlan(
            "contrast",
            "partial",
            (),
            (f"The available evidence does not establish that {alternative_id} was considered.",),
        )
    if alternative.status in {
        CandidateStatus.INFEASIBLE,
        CandidateStatus.NOT_CONSIDERED,
        CandidateStatus.UNKNOWN,
    }:
        wording = {
            CandidateStatus.INFEASIBLE: "was recorded as infeasible, not as a scored alternative",
            CandidateStatus.NOT_CONSIDERED: "was recorded as not considered",
            CandidateStatus.UNKNOWN: "has unknown candidate status",
        }[alternative.status]
        return AnswerPlan("contrast", "partial", (), (f"{alternative_id} {wording}.",))
    if (
        not selected
        or selected.score is None
        or alternative.score is None
        or not decision.policy_id
    ):
        return AnswerPlan(
            "contrast",
            "partial",
            (
                Claim(
                    "selected",
                    f"{decision.selected_id} was selected.",
                    SupportStatus.SUPPORTED,
                    decision.evidence_ids,
                    "recorded selected_id",
                    "decision-time",
                    EvidenceLevel.RECORDED_SEQUENCE,
                ),
            ),
            ("The evidence does not establish a scored reason for the selection.",),
        )
    relation = (
        ">"
        if selected.score > alternative.score
        else ("=" if selected.score == alternative.score else "<")
    )
    support = (
        SupportStatus.SUPPORTED
        if selected.score >= alternative.score
        else SupportStatus.CONTRADICTED
    )
    claims = [
        Claim(
            "score-comparison",
            f"Under policy {decision.policy_id}, {selected.id} scored {selected.score:g} and "
            f"{alternative.id} scored {alternative.score:g}.",
            SupportStatus.SUPPORTED,
            selected.evidence_ids + alternative.evidence_ids,
            f"{selected.score:g} {relation} {alternative.score:g}",
            "decision-time",
            EvidenceLevel.SOFTWARE_MECHANISM,
            claim_class=ClaimClass.DERIVED,
        )
    ]
    if support == SupportStatus.CONTRADICTED:
        return AnswerPlan(
            "contrast",
            "partial",
            tuple(claims),
            ("The recorded selection contradicts the recorded score ordering.",),
        )
    if selected.score == alternative.score:
        return AnswerPlan(
            "contrast",
            "partial",
            tuple(claims),
            ("The recorded scores are tied, so they do not establish why one was selected.",),
        )
    shared = [key for key in selected.contributions if key in alternative.contributions]
    deltas = {key: selected.contributions[key] - alternative.contributions[key] for key in shared}
    positive = [key for key, value in deltas.items() if value > 0]
    negative = [key for key, value in deltas.items() if value < 0]
    if positive and negative:
        pos = sum(deltas[key] for key in positive)
        neg = sum(deltas[key] for key in negative)
        if pos + neg > 0:
            labels = {"success": "success-estimate", "points": "points", "distance": "distance"}
            claims.append(
                Claim(
                    "outweighed",
                    f"The {_join([labels.get(x, x) for x in positive])} advantage "
                    f"outweighed the {_join([labels.get(x, x) for x in negative])} disadvantages "
                    f"under that policy.",
                    SupportStatus.SUPPORTED,
                    tuple(dict.fromkeys(selected.evidence_ids + alternative.evidence_ids)),
                    f"positive delta {pos:+g}; negative delta {neg:+g}; net {pos + neg:+g}",
                    "decision-time",
                    EvidenceLevel.SOFTWARE_MECHANISM,
                    claim_class=ClaimClass.DERIVED,
                )
            )
    return AnswerPlan(
        "contrast",
        "full",
        tuple(claims),
        (
            "This does not establish that the selected option was objectively best or would succeed.",
        ),
    )


def plan_recovery_count(episode: EpisodeRecord, premise_count: int | None = None) -> AnswerPlan:
    if not episode.outcome:
        return AnswerPlan("recovery-count", "abstain", (), ("No execution history is available.",))
    attempts = {
        event.attempt_id
        for event in episode.outcome.events
        if event.kind == "recovery_attempt" and event.attempt_id
    }
    recovery_complete = (
        episode.outcome.recovery_history_complete
        if episode.outcome.recovery_history_complete is not None
        else episode.outcome.history_complete
    )
    qualifier = "Exactly" if recovery_complete else "At least"
    verb = (
        "occurred"
        if recovery_complete
        else ("is recorded" if len(attempts) == 1 else "are recorded")
    )
    recorded = [
        event
        for event in episode.outcome.events
        if event.kind == "recovery_attempt" and event.attempt_id in attempts
    ]
    action_names = {event.action_name for event in recorded if event.action_name}
    if len(action_names) == 1 and len(recorded) == len(attempts):
        unit = f"{next(iter(action_names))} recovery action invocation"
    else:
        unit = "recovery attempt"
    proposition = f"{qualifier} {len(attempts)} {unit}{'s' if len(attempts) != 1 else ''} {verb}."
    evidence = tuple(event.id for event in episode.outcome.events if event.attempt_id in attempts)
    claims = [
        Claim(
            "recovery-count",
            proposition,
            SupportStatus.SUPPORTED,
            evidence,
            "count distinct non-null attempt_id values",
            "execution",
            EvidenceLevel.RECORDED_SEQUENCE,
            claim_class=ClaimClass.DERIVED,
        )
    ]
    if (
        attempts
        and len(recorded) == len(attempts)
        and all(event.status == "completed_success" for event in recorded)
    ):
        if len(attempts) == 1:
            success_text = "The recorded recovery attempt returned SUCCESS."
        elif len(attempts) == 2:
            success_text = "Both recorded recovery attempts returned SUCCESS."
        else:
            success_text = f"All {len(attempts)} recorded recovery attempts returned SUCCESS."
        claims.append(
            Claim(
                "recovery-status",
                success_text,
                SupportStatus.SUPPORTED,
                tuple(event.id for event in recorded),
                "all distinct recorded recovery attempts have completed_success status",
                "execution",
                EvidenceLevel.RECORDED_SEQUENCE,
                claim_class=ClaimClass.DERIVED,
            )
        )
    limitations = (
        []
        if recovery_complete
        else ["The incomplete history does not establish the total number of attempts."]
    )
    if premise_count is not None and premise_count != len(attempts):
        limitations.append(
            f"The question's premise of {premise_count} attempts is not supported by the record."
        )
    return AnswerPlan(
        "recovery-count",
        "full" if recovery_complete else "partial",
        tuple(claims),
        tuple(limitations),
    )


def plan_terminal_status(episode: EpisodeRecord) -> AnswerPlan:
    """Report recorded termination mechanics without inventing a physical cause."""
    outcome = episode.outcome
    if not outcome:
        return AnswerPlan("terminal-status", "abstain", (), ("No terminal outcome is available.",))
    status = outcome.terminal_status.lower()
    claims = [
        Claim(
            "terminal-status",
            f"The recorded task outcome was {outcome.terminal_status}.",
            SupportStatus.SUPPORTED,
            outcome.evidence_ids,
            "recorded terminal_status",
            "execution",
            EvidenceLevel.RECORDED_SEQUENCE,
        )
    ]
    deadline_events = tuple(event for event in outcome.events if event.kind == "client_deadline")
    cancel_events = tuple(event for event in outcome.events if event.kind == "client_cancel")
    limitations: list[str] = []
    if deadline_events:
        claims.append(
            Claim(
                "client-deadline",
                "The experiment harness recorded a client deadline.",
                SupportStatus.SUPPORTED,
                tuple(event.id for event in deadline_events),
                "explicit client_deadline event",
                "execution",
                EvidenceLevel.RECORDED_SEQUENCE,
            )
        )
    if cancel_events:
        claims.append(
            Claim(
                "client-cancel",
                "The experiment harness requested cancellation.",
                SupportStatus.SUPPORTED,
                tuple(event.id for event in cancel_events),
                "explicit client_cancel event",
                "execution",
                EvidenceLevel.RECORDED_SEQUENCE,
            )
        )
    if deadline_events or cancel_events:
        limitations.append(
            "These client events do not establish that a Behavior Tree timeout or a physical "
            "navigation failure occurred."
        )
    elif status in {"failed", "failure", "aborted", "timeout"}:
        limitations.append(
            "The terminal status alone does not establish the physical cause of failure."
        )
    elif status in {"success", "succeeded"}:
        limitations.append(
            "The successful outcome does not establish that no intermediate branch failed."
        )
    return AnswerPlan(
        "terminal-status",
        "full" if outcome.history_complete else "partial",
        tuple(claims),
        tuple(limitations),
    )


def plan_recovery_mechanism(
    episode: EpisodeRecord,
    provenance: ResolvedProvenance | None = None,
) -> AnswerPlan:
    """Explain an ordered software transition without promoting it to physical causation."""
    outcome = episode.outcome
    if not outcome:
        return AnswerPlan(
            "recovery-mechanism", "abstain", (), ("No execution history is available.",)
        )
    failures = sorted(
        (event for event in outcome.events if event.kind == "follow_path_failure"),
        key=lambda event: event.timestamp,
    )
    guards = sorted(
        (event for event in outcome.events if event.kind == "controller_recovery_guard_success"),
        key=lambda event: event.timestamp,
    )
    recoveries = sorted(
        (event for event in outcome.events if event.kind == "recovery_attempt"),
        key=lambda event: event.timestamp,
    )
    recovery_complete = (
        outcome.recovery_history_complete
        if outcome.recovery_history_complete is not None
        else outcome.history_complete
    )
    if not recoveries and recovery_complete:
        return AnswerPlan(
            "recovery-mechanism",
            "full",
            (
                Claim(
                    "recovery-count",
                    "Exactly 0 recovery attempts occurred.",
                    SupportStatus.SUPPORTED,
                    outcome.evidence_ids,
                    "complete recovery history contains no distinct recovery attempt IDs",
                    "execution",
                    EvidenceLevel.RECORDED_SEQUENCE,
                    claim_class=ClaimClass.DERIVED,
                ),
            ),
            (
                (
                    "The question's premise that the Behavior Tree entered recovery is contradicted by "
                    "the complete recovery-count evidence."
                ),
            ),
        )
    if not failures or not guards or not recoveries:
        return AnswerPlan(
            "recovery-mechanism",
            "abstain",
            (),
            (
                (
                    "The available record does not contain the transitions required to establish why "
                    "the Behavior Tree entered recovery."
                ),
            ),
        )
    failure = failures[0]
    guard = next((event for event in guards if event.timestamp >= failure.timestamp), None)
    recovery = next(
        (event for event in recoveries if guard is not None and event.timestamp >= guard.timestamp),
        None,
    )
    if guard is None or recovery is None:
        return AnswerPlan(
            "recovery-mechanism",
            "abstain",
            (),
            ("The available transitions do not establish an ordered path into recovery.",),
        )
    runtime_evidence_ids = (failure.id, guard.id, recovery.id)
    source_anchor_ids = (
        mechanism_source_anchor_ids(provenance, set(runtime_evidence_ids)) if provenance else ()
    )
    claim_class = ClaimClass.MECHANISM_SUPPORTED if source_anchor_ids else ClaimClass.DERIVED
    limitations = [
        (
            "This recorded software mechanism does not establish the physical reason that "
            "FollowPath failed."
        ),
    ]
    if not source_anchor_ids:
        limitations.insert(
            0,
            "The runtime record establishes the transition order, but no exact "
            "source/configuration anchor is attached to this claim.",
        )
    claims = [
        Claim(
            "recovery-mechanism",
            "The recorded FollowPath branch returned FAILURE, the controller-recovery "
            "eligibility guard returned SUCCESS, and the Behavior Tree then entered Wait.",
            SupportStatus.SUPPORTED,
            runtime_evidence_ids,
            "ordered Behavior Tree transitions",
            "execution",
            EvidenceLevel.SOFTWARE_MECHANISM,
            claim_class=claim_class,
            source_anchor_ids=source_anchor_ids,
        )
    ]
    if source_anchor_ids:
        joined_anchors = ", ".join(source_anchor_ids)
        claims.append(
            Claim(
                "recovery-source-provenance",
                f"Exact retained source anchor {joined_anchors} governs that recorded control-flow "
                "relationship.",
                SupportStatus.SUPPORTED,
                runtime_evidence_ids,
                "validated runtime-to-source links cover every transition in the mechanism claim",
                "running artifact",
                EvidenceLevel.SOFTWARE_MECHANISM,
                claim_class=ClaimClass.SOURCE_DEFINED,
                source_anchor_ids=source_anchor_ids,
            )
        )
    return AnswerPlan(
        "recovery-mechanism",
        "full",
        tuple(claims),
        tuple(limitations),
    )


def plan_failure_cause(episode: EpisodeRecord) -> AnswerPlan:
    """Answer a physical-cause question only to the level licensed by the trace."""
    outcome = episode.outcome
    if not outcome:
        return AnswerPlan("failure-cause", "abstain", (), ("No terminal outcome is available.",))
    status = outcome.terminal_status.lower()
    recorded_failures = tuple(
        event
        for event in outcome.events
        if event.kind in {"follow_path_failure", "planning_no_valid_path"}
    )
    if status in {"success", "succeeded"} and not recorded_failures:
        limitation = (
            "The question's premise of a navigation failure is contradicted by the available "
            "record, which contains a successful outcome and no recorded execution failure."
        )
    else:
        limitation = (
            "The robot-visible evidence does not establish the physical cause of the recorded "
            "failure."
        )
    claims = [
        Claim(
            "terminal-status",
            f"The recorded task outcome was {outcome.terminal_status}.",
            SupportStatus.SUPPORTED,
            outcome.evidence_ids,
            "recorded terminal_status",
            "execution",
            EvidenceLevel.RECORDED_SEQUENCE,
        )
    ]
    follow_path_failures = tuple(
        event for event in recorded_failures if event.kind == "follow_path_failure"
    )
    if follow_path_failures:
        claims.append(
            Claim(
                "intermediate-follow-path-failure",
                "The record also contains an intermediate FollowPath FAILURE.",
                SupportStatus.SUPPORTED,
                tuple(event.id for event in follow_path_failures),
                "recorded FollowPath failure transition",
                "execution",
                EvidenceLevel.RECORDED_SEQUENCE,
            )
        )
    return AnswerPlan(
        "failure-cause",
        "partial",
        tuple(claims),
        (limitation,),
    )


def plan_planning_failure(episode: EpisodeRecord) -> AnswerPlan:
    """Explain a recorded planner error without inventing its physical cause."""
    outcome = episode.outcome
    if not outcome:
        return AnswerPlan("planning-failure", "abstain", (), ("No terminal outcome is available.",))
    events = tuple(event for event in outcome.events if event.kind == "planning_no_valid_path")
    if events:
        return AnswerPlan(
            "planning-failure",
            "full",
            (
                Claim(
                    "planning-no-valid-path",
                    "The NavigateToPose result recorded planner error NO_VALID_PATH (208) while "
                    "ComputePathToPose was active.",
                    SupportStatus.SUPPORTED,
                    tuple(event.id for event in events),
                    "recorded action error code mapped through installed nav2_msgs plus active BT node",
                    "execution",
                    EvidenceLevel.SOFTWARE_MECHANISM,
                    claim_class=ClaimClass.DERIVED,
                ),
            ),
            (
                (
                    "The robot-visible evidence does not establish the physical reason that the planner "
                    "found no valid path."
                ),
            ),
        )
    if outcome.terminal_status.lower() in {"success", "succeeded"}:
        return AnswerPlan(
            "planning-failure",
            "full",
            (
                Claim(
                    "terminal-status",
                    f"The recorded task outcome was {outcome.terminal_status}.",
                    SupportStatus.SUPPORTED,
                    outcome.evidence_ids,
                    "recorded terminal_status",
                    "execution",
                    EvidenceLevel.RECORDED_SEQUENCE,
                ),
            ),
            (
                (
                    "The question's premise of a planning failure is contradicted by the recorded "
                    "successful outcome."
                ),
            ),
        )
    return AnswerPlan(
        "planning-failure",
        "abstain",
        (),
        ("The available evidence does not establish that planning failed with NO_VALID_PATH.",),
    )


def plan_unsupported_counterfactual(episode: EpisodeRecord) -> AnswerPlan:
    """Reject a hypothetical outcome when no intervention or causal model is recorded."""
    return AnswerPlan(
        "unsupported-counterfactual",
        "abstain",
        (),
        (
            (
                "The available evidence does not establish what would have happened under that "
                "hypothetical change."
            ),
        ),
    )
