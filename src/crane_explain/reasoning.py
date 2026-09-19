"""Checked answer-plan construction for decision and recovery questions."""

from __future__ import annotations

from .models import (
    AnswerPlan, Candidate, CandidateStatus, Claim, EpisodeRecord, EvidenceLevel, SupportStatus,
)


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
            "contrast", "partial", (),
            (f"The available evidence does not establish that {alternative_id} was considered.",),
        )
    if alternative.status in {CandidateStatus.INFEASIBLE, CandidateStatus.NOT_CONSIDERED,
                              CandidateStatus.UNKNOWN}:
        wording = {
            CandidateStatus.INFEASIBLE: "was recorded as infeasible, not as a scored alternative",
            CandidateStatus.NOT_CONSIDERED: "was recorded as not considered",
            CandidateStatus.UNKNOWN: "has unknown candidate status",
        }[alternative.status]
        return AnswerPlan("contrast", "partial", (), (f"{alternative_id} {wording}.",))
    if not selected or selected.score is None or alternative.score is None or not decision.policy_id:
        return AnswerPlan(
            "contrast", "partial",
            (Claim("selected", f"{decision.selected_id} was selected.", SupportStatus.SUPPORTED,
                   decision.evidence_ids, "recorded selected_id", "decision-time",
                   EvidenceLevel.RECORDED_SEQUENCE),),
            ("The evidence does not establish a scored reason for the selection.",),
        )
    relation = ">" if selected.score > alternative.score else ("=" if selected.score == alternative.score else "<")
    support = SupportStatus.SUPPORTED if selected.score >= alternative.score else SupportStatus.CONTRADICTED
    claims = [Claim(
        "score-comparison",
        f"Under policy {decision.policy_id}, {selected.id} scored {selected.score:g} and "
        f"{alternative.id} scored {alternative.score:g}.",
        SupportStatus.SUPPORTED, selected.evidence_ids + alternative.evidence_ids,
        f"{selected.score:g} {relation} {alternative.score:g}", "decision-time",
        EvidenceLevel.SOFTWARE_MECHANISM,
    )]
    if support == SupportStatus.CONTRADICTED:
        return AnswerPlan(
            "contrast", "partial", tuple(claims),
            ("The recorded selection contradicts the recorded score ordering.",),
        )
    if selected.score == alternative.score:
        return AnswerPlan(
            "contrast", "partial", tuple(claims),
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
            claims.append(Claim(
                "outweighed", f"The {_join([labels.get(x, x) for x in positive])} advantage "
                f"outweighed the {_join([labels.get(x, x) for x in negative])} disadvantages "
                f"under that policy.",
                SupportStatus.SUPPORTED,
                tuple(dict.fromkeys(selected.evidence_ids + alternative.evidence_ids)),
                f"positive delta {pos:+g}; negative delta {neg:+g}; net {pos + neg:+g}",
                "decision-time", EvidenceLevel.SOFTWARE_MECHANISM,
            ))
    return AnswerPlan(
        "contrast", "full", tuple(claims),
        ("This does not establish that the selected option was objectively best or would succeed.",),
    )


def plan_recovery_count(episode: EpisodeRecord, premise_count: int | None = None) -> AnswerPlan:
    if not episode.outcome:
        return AnswerPlan("recovery-count", "abstain", (), ("No execution history is available.",))
    attempts = {
        event.attempt_id for event in episode.outcome.events
        if event.kind == "recovery_attempt" and event.attempt_id
    }
    qualifier = "Exactly" if episode.outcome.history_complete else "At least"
    verb = "occurred" if episode.outcome.history_complete else (
        "is recorded" if len(attempts) == 1 else "are recorded")
    proposition = f"{qualifier} {len(attempts)} recovery attempt{'s' if len(attempts) != 1 else ''} {verb}."
    evidence = tuple(event.id for event in episode.outcome.events if event.attempt_id in attempts)
    claims = [Claim(
        "recovery-count", proposition, SupportStatus.SUPPORTED, evidence,
        "count distinct non-null attempt_id values", "execution",
        EvidenceLevel.RECORDED_SEQUENCE,
    )]
    recorded = [
        event for event in episode.outcome.events
        if event.kind == "recovery_attempt" and event.attempt_id in attempts
    ]
    if attempts and len(recorded) == len(attempts) and all(
            event.status == "completed_success" for event in recorded):
        if len(attempts) == 1:
            success_text = "The recorded recovery attempt returned SUCCESS."
        elif len(attempts) == 2:
            success_text = "Both recorded recovery attempts returned SUCCESS."
        else:
            success_text = f"All {len(attempts)} recorded recovery attempts returned SUCCESS."
        claims.append(Claim(
            "recovery-status", success_text, SupportStatus.SUPPORTED,
            tuple(event.id for event in recorded),
            "all distinct recorded recovery attempts have completed_success status",
            "execution", EvidenceLevel.RECORDED_SEQUENCE,
        ))
    limitations = [] if episode.outcome.history_complete else [
        "The incomplete history does not establish the total number of attempts."]
    if premise_count is not None and premise_count != len(attempts):
        limitations.append(
            f"The question's premise of {premise_count} attempts is not supported by the record.")
    return AnswerPlan(
        "recovery-count", "full" if episode.outcome.history_complete else "partial",
        tuple(claims),
        tuple(limitations),
    )


def plan_terminal_status(episode: EpisodeRecord) -> AnswerPlan:
    """Report recorded termination mechanics without inventing a physical cause."""
    outcome = episode.outcome
    if not outcome:
        return AnswerPlan("terminal-status", "abstain", (),
                          ("No terminal outcome is available.",))
    status = outcome.terminal_status.lower()
    claims = [Claim(
        "terminal-status",
        f"The recorded task outcome was {outcome.terminal_status}.",
        SupportStatus.SUPPORTED, outcome.evidence_ids,
        "recorded terminal_status", "execution",
        EvidenceLevel.RECORDED_SEQUENCE,
    )]
    deadline_events = tuple(event for event in outcome.events if event.kind == "client_deadline")
    cancel_events = tuple(event for event in outcome.events if event.kind == "client_cancel")
    limitations: list[str] = []
    if deadline_events:
        claims.append(Claim(
            "client-deadline",
            "The experiment harness recorded a client deadline.",
            SupportStatus.SUPPORTED, tuple(event.id for event in deadline_events),
            "explicit client_deadline event", "execution",
            EvidenceLevel.RECORDED_SEQUENCE,
        ))
    if cancel_events:
        claims.append(Claim(
            "client-cancel",
            "The experiment harness requested cancellation.",
            SupportStatus.SUPPORTED, tuple(event.id for event in cancel_events),
            "explicit client_cancel event", "execution",
            EvidenceLevel.RECORDED_SEQUENCE,
        ))
    if deadline_events or cancel_events:
        limitations.append(
            "These client events do not establish that a Behavior Tree timeout or a physical "
            "navigation failure occurred.")
    elif status in {"failed", "failure", "aborted", "timeout"}:
        limitations.append(
            "The terminal status alone does not establish the physical cause of failure.")
    elif status in {"success", "succeeded"}:
        limitations.append(
            "The successful outcome does not establish that no intermediate branch failed.")
    return AnswerPlan("terminal-status", "full" if outcome.history_complete else "partial",
                      tuple(claims), tuple(limitations))


def plan_recovery_mechanism(episode: EpisodeRecord) -> AnswerPlan:
    """Explain an ordered software transition without promoting it to physical causation."""
    outcome = episode.outcome
    if not outcome:
        return AnswerPlan("recovery-mechanism", "abstain", (),
                          ("No execution history is available.",))
    failures = sorted(
        (event for event in outcome.events if event.kind == "follow_path_failure"),
        key=lambda event: event.timestamp,
    )
    guards = sorted(
        (event for event in outcome.events
         if event.kind == "controller_recovery_guard_success"),
        key=lambda event: event.timestamp,
    )
    recoveries = sorted(
        (event for event in outcome.events if event.kind == "recovery_attempt"),
        key=lambda event: event.timestamp,
    )
    if not failures or not guards or not recoveries:
        return AnswerPlan(
            "recovery-mechanism", "abstain", (),
            ("The available record does not contain the transitions required to establish why "
             "the Behavior Tree entered recovery.",),
        )
    failure = failures[0]
    guard = next((event for event in guards if event.timestamp >= failure.timestamp), None)
    recovery = next(
        (event for event in recoveries
         if guard is not None and event.timestamp >= guard.timestamp), None)
    if guard is None or recovery is None:
        return AnswerPlan(
            "recovery-mechanism", "abstain", (),
            ("The available transitions do not establish an ordered path into recovery.",),
        )
    return AnswerPlan(
        "recovery-mechanism", "full",
        (Claim(
            "recovery-mechanism",
            "The recorded FollowPath branch returned FAILURE, the controller-recovery "
            "eligibility guard returned SUCCESS, and the Behavior Tree then entered Wait.",
            SupportStatus.SUPPORTED,
            (failure.id, guard.id, recovery.id),
            "ordered Behavior Tree transitions",
            "execution",
            EvidenceLevel.SOFTWARE_MECHANISM,
        ),),
        ("This recorded software mechanism does not establish the physical reason that "
         "FollowPath failed.",),
    )


def plan_failure_cause(episode: EpisodeRecord) -> AnswerPlan:
    """Answer a physical-cause question only to the level licensed by the trace."""
    outcome = episode.outcome
    if not outcome:
        return AnswerPlan("failure-cause", "abstain", (),
                          ("No terminal outcome is available.",))
    status = outcome.terminal_status.lower()
    limitation = (
        "The question's premise of a navigation failure is contradicted by the recorded "
        "successful outcome."
        if status in {"success", "succeeded"}
        else "The robot-visible evidence does not establish the physical cause of the failure."
    )
    return AnswerPlan(
        "failure-cause", "partial",
        (Claim(
            "terminal-status",
            f"The recorded task outcome was {outcome.terminal_status}.",
            SupportStatus.SUPPORTED,
            outcome.evidence_ids,
            "recorded terminal_status",
            "execution",
            EvidenceLevel.RECORDED_SEQUENCE,
        ),),
        (limitation,),
    )


def plan_unsupported_counterfactual(episode: EpisodeRecord) -> AnswerPlan:
    """Reject a hypothetical outcome when no intervention or causal model is recorded."""
    return AnswerPlan(
        "unsupported-counterfactual", "abstain", (),
        ("The available evidence does not establish what would have happened under that "
         "hypothetical change.",),
    )
