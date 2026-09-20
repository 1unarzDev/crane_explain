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


class ClaimClass(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"
    SOURCE_DEFINED = "source_defined"
    MECHANISM_SUPPORTED = "mechanism_supported"
    HYPOTHESIS = "hypothesis"
    INTERVENTION_SUPPORTED = "intervention_supported"


class SourceArtifactKind(str, Enum):
    SOURCE_FILE = "source_file"
    CONFIGURATION = "configuration"
    BEHAVIOR_TREE_XML = "behavior_tree_xml"
    PACKAGE_METADATA = "package_metadata"


class ProvenanceRelationship(str, Enum):
    GOVERNED_BY = "governed_by"
    IMPLEMENTED_BY = "implemented_by"
    CONFIGURED_BY = "configured_by"
    EMITTED_BY = "emitted_by"
    DEFINED_BY = "defined_by"


class ProvenanceStrength(str, Enum):
    EXACT_ARTIFACT = "exact_artifact"
    VERSIONED_SOURCE = "versioned_source"
    DECLARED_RUNTIME = "declared_runtime"
    PLAUSIBLE_ONLY = "plausible_only"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    kind: str
    value: Any
    timestamp: float | None = None
    source: str = "record"
    consumed: bool | None = None


@dataclass(frozen=True)
class SourceArtifact:
    id: str
    kind: SourceArtifactKind
    uri: str
    content_sha256: str
    repository: str | None = None
    commit: str | None = None
    package: str | None = None
    version: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class SourceAnchor:
    id: str
    artifact_id: str
    locator: str
    excerpt: str
    excerpt_sha256: str
    symbol: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    configuration_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeSourceLink:
    id: str
    runtime_evidence_ids: tuple[str, ...]
    source_anchor_ids: tuple[str, ...]
    relationship: ProvenanceRelationship
    strength: ProvenanceStrength
    rationale: str


@dataclass(frozen=True)
class ProvenanceBundle:
    schema_version: str
    artifacts: tuple[SourceArtifact, ...]
    anchors: tuple[SourceAnchor, ...]
    links: tuple[RuntimeSourceLink, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    action_name: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutcomeRecord:
    terminal_status: str
    timestamp: float
    events: tuple[ExecutionEvent, ...] = ()
    history_complete: bool = False
    recovery_history_complete: bool | None = None
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
    claim_class: ClaimClass = ClaimClass.OBSERVED
    source_anchor_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerPlan:
    question: str
    disposition: str  # full, partial, abstain
    claims: tuple[Claim, ...]
    not_established: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
