"""Dependency-free, JSON-serializable evidence records.

Decision evidence and later execution outcomes are deliberately separate fields. Frozen
dataclasses make accidental post-hoc mutation difficult; retained JSON artifacts are the durable
audit boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CandidateStatus(str, Enum):
    SELECTED = "selected"
    REJECTED = "evaluated_rejected"
    INFEASIBLE = "infeasible"
    NOT_CONSIDERED = "not_considered"
    UNKNOWN = "unknown"


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    NOT_ESTABLISHED = "not_established"
    CONTRADICTED = "contradicted"


class EvidenceLevel(int, Enum):
    RECORDED_SEQUENCE = 1
    SOFTWARE_MECHANISM = 2
    MODEL_CAUSAL = 3
    INTERVENTION = 4


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    kind: str
    value: Any
    timestamp: float | None = None
    source: str = "record"
    consumed: bool | None = None


@dataclass(frozen=True)
class Candidate:
    id: str
    status: CandidateStatus
    features: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    score: float | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionRecord:
    id: str
    selected_id: str
    timestamp: float
    candidates: tuple[Candidate, ...]
    policy_id: str | None = None
    policy_expression: str | None = None
    complete_candidate_set: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionEvent:
    id: str
    kind: str
    timestamp: float
    status: str | None = None
    attempt_id: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutcomeRecord:
    terminal_status: str
    timestamp: float
    events: tuple[ExecutionEvent, ...] = ()
    history_complete: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EpisodeRecord:
    schema_version: str
    episode_id: str
    evidence: tuple[EvidenceItem, ...]
    decision: DecisionRecord | None = None
    outcome: OutcomeRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Claim:
    id: str
    proposition: str
    support: SupportStatus
    evidence_ids: tuple[str, ...]
    derivation: str
    temporal_scope: str
    evidence_level: EvidenceLevel
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerPlan:
    question: str
    disposition: str  # full, partial, abstain
    claims: tuple[Claim, ...]
    not_established: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

