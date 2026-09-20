"""Validated, bounded runtime-to-source provenance resolution.

The resolver follows explicit links from runtime evidence to retained source anchors. It never
searches a repository for merely plausible code, and it preserves unresolved evidence IDs rather
than filling gaps by inference.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from .models import (
    ProvenanceBundle,
    ProvenanceRelationship,
    ProvenanceStrength,
    RuntimeSourceLink,
    SourceAnchor,
    SourceArtifact,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceValidationError(ValueError):
    """Raised when provenance cannot safely support downstream explanation claims."""


@dataclass(frozen=True)
class ResolvedProvenance:
    artifacts: tuple[SourceArtifact, ...]
    anchors: tuple[SourceAnchor, ...]
    links: tuple[RuntimeSourceLink, ...]
    unresolved_runtime_evidence_ids: tuple[str, ...]


def _require_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ProvenanceValidationError(f"duplicate {label} ID")


def _validate(bundle: ProvenanceBundle, known_runtime_evidence_ids: AbstractSet[str]) -> None:
    artifact_ids = [artifact.id for artifact in bundle.artifacts]
    anchor_ids = [anchor.id for anchor in bundle.anchors]
    link_ids = [link.id for link in bundle.links]
    _require_unique(artifact_ids, "source artifact")
    _require_unique(anchor_ids, "source anchor")
    _require_unique(link_ids, "runtime-to-source link")

    artifacts = set(artifact_ids)
    anchors = set(anchor_ids)
    for artifact in bundle.artifacts:
        if not _SHA256.fullmatch(artifact.content_sha256):
            raise ProvenanceValidationError(
                f"source artifact {artifact.id} has invalid content SHA-256"
            )
    for anchor in bundle.anchors:
        if anchor.artifact_id not in artifacts:
            raise ProvenanceValidationError(
                f"source anchor {anchor.id} references unknown artifact {anchor.artifact_id}"
            )
        actual = hashlib.sha256(anchor.excerpt.encode("utf-8")).hexdigest()
        if actual != anchor.excerpt_sha256:
            raise ProvenanceValidationError(f"source anchor {anchor.id} excerpt SHA-256 mismatch")
        if anchor.line_start is not None and anchor.line_start < 1:
            raise ProvenanceValidationError(f"source anchor {anchor.id} has invalid line_start")
        if anchor.line_end is not None and (
            anchor.line_start is None or anchor.line_end < anchor.line_start
        ):
            raise ProvenanceValidationError(f"source anchor {anchor.id} has invalid line range")
    for link in bundle.links:
        unknown_runtime = set(link.runtime_evidence_ids) - set(known_runtime_evidence_ids)
        if unknown_runtime:
            raise ProvenanceValidationError(
                f"runtime-to-source link {link.id} references unknown runtime evidence: "
                f"{sorted(unknown_runtime)}"
            )
        unknown_anchors = set(link.source_anchor_ids) - anchors
        if unknown_anchors:
            raise ProvenanceValidationError(
                f"runtime-to-source link {link.id} references unknown source anchors: "
                f"{sorted(unknown_anchors)}"
            )


def resolve_provenance(
    bundle: ProvenanceBundle,
    *,
    known_runtime_evidence_ids: AbstractSet[str],
    requested_runtime_evidence_ids: AbstractSet[str],
) -> ResolvedProvenance:
    """Return only source context explicitly linked to the requested runtime evidence."""

    _validate(bundle, known_runtime_evidence_ids)
    unknown_requests = set(requested_runtime_evidence_ids) - set(known_runtime_evidence_ids)
    if unknown_requests:
        raise ProvenanceValidationError(
            f"requested unknown runtime evidence: {sorted(unknown_requests)}"
        )

    links = tuple(
        link
        for link in bundle.links
        if set(link.runtime_evidence_ids) & set(requested_runtime_evidence_ids)
    )
    linked_runtime = {
        evidence_id
        for link in links
        for evidence_id in link.runtime_evidence_ids
        if evidence_id in requested_runtime_evidence_ids
    }
    source_anchor_ids = {anchor_id for link in links for anchor_id in link.source_anchor_ids}
    anchors = tuple(anchor for anchor in bundle.anchors if anchor.id in source_anchor_ids)
    artifact_ids = {anchor.artifact_id for anchor in anchors}
    artifacts = tuple(artifact for artifact in bundle.artifacts if artifact.id in artifact_ids)
    unresolved = tuple(sorted(set(requested_runtime_evidence_ids) - linked_runtime))
    return ResolvedProvenance(artifacts, anchors, links, unresolved)


def mechanism_source_anchor_ids(
    resolved: ResolvedProvenance,
    required_runtime_evidence_ids: AbstractSet[str],
) -> tuple[str, ...]:
    """Return anchors only when strong links cover every required runtime fact."""

    qualifying_strengths = {
        ProvenanceStrength.EXACT_ARTIFACT,
        ProvenanceStrength.VERSIONED_SOURCE,
    }
    qualifying_relationships = {
        ProvenanceRelationship.GOVERNED_BY,
        ProvenanceRelationship.IMPLEMENTED_BY,
        ProvenanceRelationship.CONFIGURED_BY,
        ProvenanceRelationship.DEFINED_BY,
    }
    qualifying = tuple(
        link
        for link in resolved.links
        if link.strength in qualifying_strengths and link.relationship in qualifying_relationships
    )
    covered = {
        evidence_id
        for link in qualifying
        for evidence_id in link.runtime_evidence_ids
        if evidence_id in required_runtime_evidence_ids
    }
    if covered != set(required_runtime_evidence_ids):
        return ()
    return tuple(
        dict.fromkeys(anchor_id for link in qualifying for anchor_id in link.source_anchor_ids)
    )
