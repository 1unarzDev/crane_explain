import hashlib

import pytest

from crane_explain.io import episode_from_dict
from crane_explain.models import SourceArtifact, SourceArtifactKind
from crane_explain.nav2_provenance import build_behavior_tree_provenance
from crane_explain.provenance import ProvenanceValidationError

BT_XML = """<root BTCPP_format="4" main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="2" name="NavigateRecovery">
      <Sequence name="NavigateAttempt">
        <FollowPath path="{path}" controller_id="FollowPath"/>
      </Sequence>
      <Sequence name="ProgressRecovery">
        <WouldAControllerRecoveryHelp error_code="{follow_path_error_code}"/>
        <Wait wait_duration="1.0"/>
      </Sequence>
    </RecoveryNode>
  </BehaviorTree>
</root>
"""


def _episode():
    return episode_from_dict(
        {
            "schema_version": "crane-explain-episode/v1",
            "episode_id": "pilot",
            "evidence": [
                {
                    "id": "failure",
                    "kind": "bt_transition",
                    "value": {"node": "FollowPath", "node_uid": 4, "to": "FAILURE"},
                },
                {
                    "id": "guard",
                    "kind": "bt_transition",
                    "value": {
                        "node": "WouldAControllerRecoveryHelp",
                        "node_uid": 6,
                        "to": "SUCCESS",
                    },
                },
                {
                    "id": "wait",
                    "kind": "bt_transition",
                    "value": {"node": "Wait", "node_uid": 7, "to": "RUNNING"},
                },
                {"id": "feedback", "kind": "navigate_to_pose_feedback", "value": 2},
            ],
        }
    )


def _artifact(content_sha256: str | None = None):
    return SourceArtifact(
        id="captured-bt-xml",
        kind=SourceArtifactKind.BEHAVIOR_TREE_XML,
        uri="capture/behavior_tree.xml",
        content_sha256=content_sha256 or hashlib.sha256(BT_XML.encode()).hexdigest(),
        repository="https://github.com/1unarzDev/crane_ml.git",
        commit="c559932a5ebef00bfa7752511799fd904e5c9dbe",
        package="crane_ml",
        path="Tools/Performance/nav2_land_progress_recovery.xml",
    )


def test_builder_links_runtime_nodes_to_the_smallest_recovery_subtree():
    bundle = build_behavior_tree_provenance(_episode(), BT_XML, _artifact())

    assert len(bundle.anchors) == 1
    assert bundle.anchors[0].symbol == "NavigateRecovery"
    assert "<RecoveryNode" in bundle.anchors[0].excerpt
    assert {runtime_id for link in bundle.links for runtime_id in link.runtime_evidence_ids} == {
        "failure",
        "guard",
        "wait",
    }
    assert "feedback" not in {item for link in bundle.links for item in link.runtime_evidence_ids}


def test_builder_rejects_bt_xml_that_does_not_match_retained_artifact_hash():
    with pytest.raises(ProvenanceValidationError, match="BT XML SHA-256 mismatch"):
        build_behavior_tree_provenance(_episode(), BT_XML, _artifact("0" * 64))
