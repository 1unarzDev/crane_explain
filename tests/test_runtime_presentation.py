import hashlib
import json

import pytest

from crane_explain.io import episode_from_dict
from crane_explain.runtime_presentation import (
    PARITY_AUDIT_SCHEMA,
    PRESENTATION_SCHEMA,
    PresentationError,
    audit_fgh_information_parity,
    build_nav2_runtime_presentation,
    validate_episode_projection,
)


def _capture():
    xml = b"<root main_tree_to_execute='MainTree'><BehaviorTree ID='MainTree'/></root>\n"
    goal_id = "goal-123"
    records = [
        {"type": "capture_started", "wall_time_ns": 1},
        {
            "type": "harness_event",
            "event": {
                "type": "navigate_to_pose_goal",
                "accepted": True,
                "action_name": "/navigate_to_pose",
                "action_mode": "navigate-to-pose",
                "goal_attempt": 1,
                "goal_id": goal_id,
                "goal": {"frame_id": "odom", "position": {"x": 3.0, "y": 0.0}},
                "wall_time_ns": 2,
            },
        },
        {
            "type": "navigate_to_pose_feedback",
            "goal_id": goal_id,
            "navigation_time_s": 1.0,
            "estimated_time_remaining_s": 2.0,
            "number_of_recoveries": 0,
            "distance_remaining": 3.0,
            "current_pose_frame": "odom",
            "current_pose_stamp": {"sec": 1, "nanosec": 0},
        },
        {
            "type": "bt_transition",
            "node_name": "FollowPath",
            "node_uid": 4,
            "previous_status": "RUNNING",
            "current_status": "FAILURE",
            "event_stamp": {"sec": 10, "nanosec": 1},
            "message_stamp": {"sec": 2, "nanosec": 1},
        },
        {
            "type": "bt_transition",
            "node_name": "WouldAControllerRecoveryHelp",
            "node_uid": 6,
            "previous_status": "IDLE",
            "current_status": "SUCCESS",
            "event_stamp": {"sec": 10, "nanosec": 2},
            "message_stamp": {"sec": 2, "nanosec": 1},
        },
        {
            "type": "bt_transition",
            "node_name": "Wait",
            "node_uid": 7,
            "previous_status": "IDLE",
            "current_status": "RUNNING",
            "event_stamp": {"sec": 10, "nanosec": 3},
            "message_stamp": {"sec": 2, "nanosec": 1},
        },
        {
            "type": "navigate_to_pose_feedback",
            "goal_id": goal_id,
            "navigation_time_s": 2.0,
            "estimated_time_remaining_s": 0.0,
            "number_of_recoveries": 1,
            "distance_remaining": 3.0,
            "current_pose_frame": "odom",
            "current_pose_stamp": {"sec": 2, "nanosec": 0},
        },
        {
            "type": "harness_event",
            "event": {
                "type": "navigate_to_pose_result",
                "action_name": "/navigate_to_pose",
                "goal_id": goal_id,
                "status": "aborted",
                "status_code": 6,
                "error_code": 105,
                "error_msg": "",
                "wall_time_ns": 3,
            },
        },
        {"type": "capture_stopped", "wall_time_ns": 4},
    ]
    manifest = {
        "schema": "crane-explain-ros-capture/v1",
        "episode_id": "episode-1",
        "run_id": "run-1",
        "bt_xml_source": "/runtime/tree.xml",
        "bt_xml_sha256": hashlib.sha256(xml).hexdigest(),
        "limitations": ["topic delivery does not prove internal consumption"],
    }
    episode = episode_from_dict(
        {
            "schema_version": "crane-explain-episode/v1",
            "episode_id": "episode-1",
            "evidence": [
                {
                    "id": "terminal-status",
                    "kind": "navigate_to_pose_terminal_status",
                    "value": "aborted",
                },
                {
                    "id": "recovery-count",
                    "kind": "navigate_to_pose_feedback_recovery_count",
                    "value": 1,
                },
                {
                    "id": "physical-cause-status",
                    "kind": "answerability",
                    "value": "not_established",
                },
                {
                    "id": "counterfactual-status",
                    "kind": "answerability",
                    "value": "not_established",
                },
                {
                    "id": "history-completeness",
                    "kind": "capture_boundary_check",
                    "value": False,
                },
                {
                    "id": "recovery-history-completeness",
                    "kind": "capture_boundary_terminal_feedback_bt_cross_check",
                    "value": True,
                },
                {
                    "id": "follow-path-failure-1",
                    "kind": "bt_transition",
                    "value": {
                        "node": "FollowPath",
                        "from": "RUNNING",
                        "to": "FAILURE",
                    },
                    "timestamp": 10.000000001,
                },
                {
                    "id": "recovery-guard-success-1",
                    "kind": "bt_transition",
                    "value": {
                        "node": "WouldAControllerRecoveryHelp",
                        "from": "IDLE",
                        "to": "SUCCESS",
                    },
                    "timestamp": 10.000000002,
                },
                {
                    "id": "wait-recovery-1",
                    "kind": "bt_transition",
                    "value": {
                        "node": "Wait",
                        "from": "IDLE",
                        "to": "RUNNING",
                    },
                    "timestamp": 10.000000003,
                },
            ],
            "outcome": {
                "terminal_status": "aborted",
                "timestamp": 3.0,
                "events": [],
                "history_complete": False,
                "recovery_history_complete": True,
            },
        }
    )
    return records, manifest, xml, episode


def test_runtime_presentation_preserves_exact_shared_runtime_units():
    records, manifest, xml, episode = _capture()
    presentation = build_nav2_runtime_presentation(records, manifest, xml, episode)

    assert presentation["schema"] == PRESENTATION_SCHEMA
    assert presentation["action"]["accepted_goal"]["goal_id"] == "goal-123"
    assert presentation["action"]["terminal"]["error_code"] == 105
    assert presentation["action"]["feedback_summary"]["message_count"] == 2
    assert presentation["action"]["feedback_summary"]["maximum_number_of_recoveries"] == 1
    assert presentation["completeness"]["complete_bt_execution_history"] is False
    assert presentation["completeness"]["recovery_count_history_complete"] is True
    assert presentation["bt_transitions"][0]["derived_evidence_id"] == "follow-path-failure-1"
    assert presentation["field_provenance"]["recovery_entries"] == [
        "capture-record-000004",
        "capture-record-000005",
        "capture-record-000006",
    ]
    assert presentation["recovery_entries"] == [
        {
            "id": "recovery-entry-1",
            "follow_path_failure_transition_id": "bt-transition-000001",
            "guard_success_transition_id": "bt-transition-000002",
            "wait_entry_transition_id": "bt-transition-000003",
            "derived_evidence_ids": [
                "follow-path-failure-1",
                "recovery-guard-success-1",
                "wait-recovery-1",
            ],
        }
    ]


@pytest.mark.parametrize("question_kind", ["recovery-mechanism", "failure-cause"])
def test_fgh_parity_audit_requires_shared_question_units(question_kind):
    records, manifest, xml, episode = _capture()
    presentation = build_nav2_runtime_presentation(records, manifest, xml, episode)
    audit = audit_fgh_information_parity(presentation, question_kind)

    assert audit["schema"] == PARITY_AUDIT_SCHEMA
    assert audit["accepted"] is True
    assert audit["evaluator_truth_available_to_methods"] is False
    assert all(unit["accepted"] for unit in audit["units"])
    assert all(
        unit["condition_access"]
        == {
            "F": "raw_capture_derivable",
            "G": "shared_runtime_presentation",
            "H": "shared_runtime_presentation",
        }
        for unit in audit["units"]
    )
    assert "runtime-to-source links" in audit["excluded_from_shared_runtime_parity"]


def test_presentation_rejects_bt_xml_hash_mismatch():
    records, manifest, xml, episode = _capture()
    manifest["bt_xml_sha256"] = "0" * 64

    with pytest.raises(PresentationError, match="does not match"):
        build_nav2_runtime_presentation(records, manifest, xml, episode)


def test_runtime_manifest_is_hash_checked_and_added_to_parity_units():
    records, manifest, xml, episode = _capture()
    runtime = json.dumps(
        {
            "schema": "crane-runtime-provenance/v1",
            "run_id": "run-1",
            "artifacts": [
                {
                    "role": "nav2_parameter_file",
                    "content_sha256": "1" * 64,
                    "repository_commit": "2" * 40,
                }
            ],
        },
        sort_keys=True,
    ).encode()
    manifest["runtime_manifest_sha256"] = hashlib.sha256(runtime).hexdigest()

    presentation = build_nav2_runtime_presentation(
        records, manifest, xml, episode, runtime
    )
    audit = audit_fgh_information_parity(presentation, "recovery-mechanism")

    assert presentation["runtime_provenance"]["run_id"] == "run-1"
    assert audit["units"][-1]["id"] == "runtime-configuration-identity"
    assert audit["units"][-1]["raw_derivation_sources"] == ["runtime-manifest"]

    with pytest.raises(PresentationError, match="does not match"):
        build_nav2_runtime_presentation(records, manifest, xml, episode, runtime + b" ")


def test_parity_audit_rejects_missing_raw_traceability():
    records, manifest, xml, episode = _capture()
    presentation = build_nav2_runtime_presentation(records, manifest, xml, episode)
    presentation["field_provenance"]["recovery_entries"] = []

    with pytest.raises(PresentationError, match="no raw derivation sources"):
        audit_fgh_information_parity(presentation, "recovery-mechanism")


def test_checked_plan_episode_is_validated_as_shared_presentation_projection():
    records, manifest, xml, episode = _capture()
    presentation = build_nav2_runtime_presentation(records, manifest, xml, episode)

    validate_episode_projection(presentation, episode)

    incompatible = episode_from_dict(episode.to_dict())
    object.__setattr__(incompatible.outcome, "terminal_status", "succeeded")
    with pytest.raises(PresentationError, match="terminal statuses disagree"):
        validate_episode_projection(presentation, incompatible)
