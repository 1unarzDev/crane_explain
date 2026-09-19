"""Record invariants that prevent unsupported downstream claims."""

from __future__ import annotations

from .models import CandidateStatus, EpisodeRecord


def validate_episode(episode: EpisodeRecord) -> list[str]:
    errors: list[str] = []
    evidence_ids = [item.id for item in episode.evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("duplicate evidence ID")
    known = set(evidence_ids)
    if episode.decision:
        candidates = episode.decision.candidates
        ids = [candidate.id for candidate in candidates]
        if len(ids) != len(set(ids)):
            errors.append("duplicate candidate ID")
        selected = [c for c in candidates if c.status == CandidateStatus.SELECTED]
        if len(selected) != 1 or selected[0].id != episode.decision.selected_id:
            errors.append("selected candidate/status mismatch")
        for candidate in candidates:
            if candidate.score is not None and episode.decision.policy_expression is None:
                errors.append(f"candidate {candidate.id} has score without policy expression")
            if not set(candidate.evidence_ids) <= known:
                errors.append(f"candidate {candidate.id} references unknown evidence")
    if episode.outcome:
        event_ids = [event.id for event in episode.outcome.events]
        if len(event_ids) != len(set(event_ids)):
            errors.append("duplicate event ID")
        if episode.decision and episode.outcome.timestamp < episode.decision.timestamp:
            errors.append("outcome predates decision")
    return errors

