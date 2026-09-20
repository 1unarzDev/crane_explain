import hashlib

import pytest

from crane_explain.io import episode_from_dict, provenance_from_dict
from crane_explain.models import (
    ClaimClass,
    ProvenanceBundle,
    ProvenanceRelationship,
    ProvenanceStrength,
    RuntimeSourceLink,
    SourceAnchor,
    SourceArtifact,
    SourceArtifactKind,
)
from crane_explain.provenance import ProvenanceValidationError, resolve_provenance
from crane_explain.reasoning import plan_recovery_mechanism


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bt_bundle(*, excerpt_sha256: str | None = None) -> ProvenanceBundle:
    excerpt = '<Wait wait_duration="1.0"/>'
    artifact = SourceArtifact(
        id="bt-xml",
        kind=SourceArtifactKind.BEHAVIOR_TREE_XML,
        uri="capture/behavior_tree.xml",
        content_sha256="14939b78c72149b9c71b3806f2d3af63fc5de48c8bd9d07f0d13b55563f48520",
        repository="https://github.com/1unarzDev/crane_ml.git",
        commit="c559932a5ebef00bfa7752511799fd904e5c9dbe",
        package="crane_ml",
        path="Tools/Performance/nav2_land_progress_recovery.xml",
    )
    anchor = SourceAnchor(
        id="bt-wait-node",
        artifact_id=artifact.id,
        locator="/root/BehaviorTree/RecoveryNode/Sequence[2]/Wait",
        excerpt=excerpt,
        excerpt_sha256=excerpt_sha256 or _sha256(excerpt),
        symbol="Wait",
    )
    link = RuntimeSourceLink(
        id="wait-transition-to-bt-node",
        runtime_evidence_ids=("bt-wait-running-1",),
        source_anchor_ids=(anchor.id,),
        relationship=ProvenanceRelationship.GOVERNED_BY,
        strength=ProvenanceStrength.EXACT_ARTIFACT,
        rationale="Captured node UID/name resolves to the retained exact BT XML node.",
    )
    return ProvenanceBundle(
        schema_version="crane-explain-provenance/v1",
        artifacts=(artifact,),
        anchors=(anchor,),
        links=(link,),
    )


def test_resolver_returns_only_context_linked_to_requested_runtime_evidence():
    resolved = resolve_provenance(
        _bt_bundle(),
        known_runtime_evidence_ids={"bt-wait-running-1", "unrelated-feedback"},
        requested_runtime_evidence_ids={"bt-wait-running-1"},
    )

    assert [anchor.id for anchor in resolved.anchors] == ["bt-wait-node"]
    assert [link.id for link in resolved.links] == ["wait-transition-to-bt-node"]
    assert resolved.unresolved_runtime_evidence_ids == ()


def test_resolver_rejects_tampered_source_excerpt():
    with pytest.raises(ProvenanceValidationError, match="excerpt SHA-256 mismatch"):
        resolve_provenance(
            _bt_bundle(excerpt_sha256="0" * 64),
            known_runtime_evidence_ids={"bt-wait-running-1"},
            requested_runtime_evidence_ids={"bt-wait-running-1"},
        )


def test_recovery_claim_requires_runtime_and_exact_source_support_for_promotion():
    raw = {
        "schema_version": "crane-explain-episode/v1",
        "episode_id": "pilot",
        "evidence": [
            {"id": "follow-path-failure", "kind": "bt_transition", "value": "FAILURE"},
            {"id": "recovery-guard", "kind": "bt_transition", "value": "SUCCESS"},
            {"id": "bt-wait-running-1", "kind": "bt_transition", "value": "RUNNING"},
        ],
        "outcome": {
            "terminal_status": "aborted",
            "timestamp": 4.0,
            "events": [
                {"id": "follow-path-failure", "kind": "follow_path_failure", "timestamp": 1.0},
                {
                    "id": "recovery-guard",
                    "kind": "controller_recovery_guard_success",
                    "timestamp": 2.0,
                },
                {
                    "id": "bt-wait-running-1",
                    "kind": "recovery_attempt",
                    "timestamp": 3.0,
                    "attempt_id": "wait-1",
                    "status": "completed_success",
                },
            ],
        },
    }
    episode = episode_from_dict(raw)
    bundle = _bt_bundle()
    original = bundle.links[0]
    bundle = ProvenanceBundle(
        schema_version=bundle.schema_version,
        artifacts=bundle.artifacts,
        anchors=bundle.anchors,
        links=(
            RuntimeSourceLink(
                id=original.id,
                runtime_evidence_ids=("follow-path-failure", "recovery-guard", "bt-wait-running-1"),
                source_anchor_ids=original.source_anchor_ids,
                relationship=original.relationship,
                strength=original.strength,
                rationale=original.rationale,
            ),
        ),
    )
    resolved = resolve_provenance(
        bundle,
        known_runtime_evidence_ids={item.id for item in episode.evidence},
        requested_runtime_evidence_ids={item.id for item in episode.evidence},
    )

    without_source = plan_recovery_mechanism(episode)
    with_source = plan_recovery_mechanism(episode, resolved)

    assert without_source.claims[0].claim_class == ClaimClass.DERIVED
    assert "exact source/configuration anchor" in without_source.not_established[0]
    assert with_source.claims[0].claim_class == ClaimClass.MECHANISM_SUPPORTED
    assert with_source.claims[0].source_anchor_ids == ("bt-wait-node",)
    assert with_source.claims[1].claim_class == ClaimClass.DERIVED
    assert "follow-path-failure at 1.000000 s" in with_source.claims[1].proposition
    assert "Wait invocation wait-1 at 3.000000 s" in with_source.claims[1].proposition
    assert with_source.claims[2].claim_class == ClaimClass.SOURCE_DEFINED
    assert "Tools/Performance/nav2_land_progress_recovery.xml" in (
        with_source.claims[2].proposition
    )
    assert "c559932a5ebef00bfa7752511799fd904e5c9dbe" in (
        with_source.claims[2].proposition
    )
    assert "14939b78c72149b9c71b3806f2d3af63fc5de48c8bd9d07f0d13b55563f48520" in (
        with_source.claims[2].proposition
    )


def test_recovery_sequence_count_is_exact_only_for_complete_recovery_history():
    events = []
    for index, base in ((1, 1.0), (2, 4.0)):
        events.extend(
            (
                {
                    "id": f"failure-{index}",
                    "kind": "follow_path_failure",
                    "timestamp": base,
                },
                {
                    "id": f"guard-{index}",
                    "kind": "controller_recovery_guard_success",
                    "timestamp": base + 1,
                },
                {
                    "id": f"wait-{index}",
                    "kind": "recovery_attempt",
                    "timestamp": base + 2,
                    "attempt_id": f"wait-attempt-{index}",
                },
            )
        )

    def episode(complete: bool):
        return episode_from_dict(
            {
                "schema_version": "crane-explain-episode/v1",
                "episode_id": "count-qualification",
                "evidence": [],
                "outcome": {
                    "terminal_status": "aborted",
                    "timestamp": 10.0,
                    "events": events,
                    "recovery_history_complete": complete,
                },
            }
        )

    incomplete = plan_recovery_mechanism(episode(False)).claims[1].proposition
    complete = plan_recovery_mechanism(episode(True)).claims[1].proposition

    assert incomplete.startswith("2 ordered recovery-entry sequences are recorded")
    assert "available history" in incomplete
    assert complete.startswith("Exactly 2 ordered recovery-entry sequences are recorded")


def test_provenance_bundle_round_trips_through_its_portable_json_shape():
    original = _bt_bundle()

    loaded = provenance_from_dict(original.to_dict())

    assert loaded == original
