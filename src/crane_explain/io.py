"""Strict JSON loading for evidence records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    Candidate,
    CandidateStatus,
    DecisionRecord,
    EpisodeRecord,
    EvidenceItem,
    ExecutionEvent,
    OutcomeRecord,
    ProvenanceBundle,
    ProvenanceRelationship,
    ProvenanceStrength,
    RuntimeSourceLink,
    SourceAnchor,
    SourceArtifact,
    SourceArtifactKind,
)


def episode_from_dict(raw: dict[str, Any]) -> EpisodeRecord:
    evidence = tuple(EvidenceItem(**item) for item in raw.get("evidence", []))
    decision_raw = raw.get("decision")
    decision = None
    if decision_raw:
        candidates = tuple(
            Candidate(
                id=item["id"],
                status=CandidateStatus(item["status"]),
                features=item.get("features", {}),
                contributions=item.get("contributions", {}),
                score=item.get("score"),
                evidence_ids=tuple(item.get("evidence_ids", ())),
            )
            for item in decision_raw.get("candidates", [])
        )
        decision = DecisionRecord(
            id=decision_raw["id"],
            selected_id=decision_raw["selected_id"],
            timestamp=decision_raw["timestamp"],
            candidates=candidates,
            policy_id=decision_raw.get("policy_id"),
            policy_expression=decision_raw.get("policy_expression"),
            complete_candidate_set=decision_raw.get("complete_candidate_set", False),
            evidence_ids=tuple(decision_raw.get("evidence_ids", ())),
        )
    outcome_raw = raw.get("outcome")
    outcome = None
    if outcome_raw:
        outcome = OutcomeRecord(
            terminal_status=outcome_raw["terminal_status"],
            timestamp=outcome_raw["timestamp"],
            events=tuple(
                ExecutionEvent(**{**e, "evidence_ids": tuple(e.get("evidence_ids", ()))})
                for e in outcome_raw.get("events", [])
            ),
            history_complete=outcome_raw.get("history_complete", False),
            recovery_history_complete=outcome_raw.get("recovery_history_complete"),
            evidence_ids=tuple(outcome_raw.get("evidence_ids", ())),
        )
    return EpisodeRecord(
        schema_version=raw["schema_version"],
        episode_id=raw["episode_id"],
        evidence=evidence,
        decision=decision,
        outcome=outcome,
    )


def load_episode(path: str | Path) -> EpisodeRecord:
    with Path(path).open(encoding="utf-8") as stream:
        return episode_from_dict(json.load(stream))


def provenance_from_dict(raw: dict[str, Any]) -> ProvenanceBundle:
    return ProvenanceBundle(
        schema_version=raw["schema_version"],
        artifacts=tuple(
            SourceArtifact(
                id=item["id"],
                kind=SourceArtifactKind(item["kind"]),
                uri=item["uri"],
                content_sha256=item["content_sha256"],
                repository=item.get("repository"),
                commit=item.get("commit"),
                package=item.get("package"),
                version=item.get("version"),
                path=item.get("path"),
            )
            for item in raw.get("artifacts", [])
        ),
        anchors=tuple(
            SourceAnchor(
                id=item["id"],
                artifact_id=item["artifact_id"],
                locator=item["locator"],
                excerpt=item["excerpt"],
                excerpt_sha256=item["excerpt_sha256"],
                symbol=item.get("symbol"),
                line_start=item.get("line_start"),
                line_end=item.get("line_end"),
                configuration_keys=tuple(item.get("configuration_keys", ())),
            )
            for item in raw.get("anchors", [])
        ),
        links=tuple(
            RuntimeSourceLink(
                id=item["id"],
                runtime_evidence_ids=tuple(item.get("runtime_evidence_ids", ())),
                source_anchor_ids=tuple(item.get("source_anchor_ids", ())),
                relationship=ProvenanceRelationship(item["relationship"]),
                strength=ProvenanceStrength(item["strength"]),
                rationale=item["rationale"],
            )
            for item in raw.get("links", [])
        ),
    )


def load_provenance(path: str | Path) -> ProvenanceBundle:
    with Path(path).open(encoding="utf-8") as stream:
        return provenance_from_dict(json.load(stream))
