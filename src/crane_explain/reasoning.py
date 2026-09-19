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
    limitations = [] if episode.outcome.history_complete else [
        "The incomplete history does not establish the total number of attempts."]
    if premise_count is not None and premise_count != len(attempts):
        limitations.append(
            f"The question's premise of {premise_count} attempts is not supported by the record.")
    return AnswerPlan(
        "recovery-count", "full" if episode.outcome.history_complete else "partial",
        (Claim("recovery-count", proposition, SupportStatus.SUPPORTED, evidence,
               "count distinct non-null attempt_id values", "execution",
               EvidenceLevel.RECORDED_SEQUENCE),),
        tuple(limitations),
    )
