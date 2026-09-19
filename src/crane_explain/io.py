"""Strict JSON loading for evidence records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    Candidate, CandidateStatus, DecisionRecord, EpisodeRecord, EvidenceItem, ExecutionEvent,
    OutcomeRecord,
)


def episode_from_dict(raw: dict[str, Any]) -> EpisodeRecord:
    evidence = tuple(EvidenceItem(**item) for item in raw.get("evidence", []))
    decision_raw = raw.get("decision")
    decision = None
    if decision_raw:
        candidates = tuple(
            Candidate(
                id=item["id"], status=CandidateStatus(item["status"]),
                features=item.get("features", {}), contributions=item.get("contributions", {}),
                score=item.get("score"), evidence_ids=tuple(item.get("evidence_ids", ())),
            )
            for item in decision_raw.get("candidates", [])
        )
        decision = DecisionRecord(
            id=decision_raw["id"], selected_id=decision_raw["selected_id"],
            timestamp=decision_raw["timestamp"], candidates=candidates,
            policy_id=decision_raw.get("policy_id"),
            policy_expression=decision_raw.get("policy_expression"),
            complete_candidate_set=decision_raw.get("complete_candidate_set", False),
            evidence_ids=tuple(decision_raw.get("evidence_ids", ())),
        )
    outcome_raw = raw.get("outcome")
    outcome = None
    if outcome_raw:
        outcome = OutcomeRecord(
            terminal_status=outcome_raw["terminal_status"], timestamp=outcome_raw["timestamp"],
            events=tuple(ExecutionEvent(**{**e, "evidence_ids": tuple(e.get("evidence_ids", ()))})
                         for e in outcome_raw.get("events", [])),
            history_complete=outcome_raw.get("history_complete", False),
            evidence_ids=tuple(outcome_raw.get("evidence_ids", ())),
        )
    return EpisodeRecord(
        schema_version=raw["schema_version"], episode_id=raw["episode_id"],
        evidence=evidence, decision=decision, outcome=outcome,
    )


def load_episode(path: str | Path) -> EpisodeRecord:
    with Path(path).open(encoding="utf-8") as stream:
        return episode_from_dict(json.load(stream))

