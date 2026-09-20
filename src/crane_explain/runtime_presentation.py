"""Deterministic, evaluator-truth-free presentations of passive Nav2 captures.

The presentation is the shared runtime-information boundary for provenance experiments.  It does
not add source provenance (the G treatment) and it does not infer physical cause.  Every presented
field records which raw capture records support it so that F's raw input and G/H's structured input
can be audited for question-relevant information parity before a model call is made.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .models import EpisodeRecord, EvidenceItem

PRESENTATION_SCHEMA = "crane-explain-runtime-presentation/v1"
PARITY_AUDIT_SCHEMA = "crane-explain-fgh-information-parity/v1"


class PresentationError(ValueError):
    """Raised when a capture cannot support an auditable runtime presentation."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _record_id(index: int) -> str:
    return f"capture-record-{index + 1:06d}"


def _record_selector(record_type: str, record_ids: Sequence[str]) -> str:
    """Compactly identify a deterministic set of raw records without copying thousands of IDs."""

    return (
        f"selector:type={record_type};count={len(record_ids)};"
        f"record_ids_sha256={_canonical_sha256(list(record_ids))}"
    )


def _stamp(record: Mapping[str, Any]) -> dict[str, int]:
    raw = record.get("event_stamp")
    if not isinstance(raw, Mapping):
        raise PresentationError("BT transition lacks an event_stamp")
    return {"sec": int(raw["sec"]), "nanosec": int(raw["nanosec"])}


def _stamp_seconds(stamp: Mapping[str, int]) -> float:
    return int(stamp["sec"]) + int(stamp["nanosec"]) / 1_000_000_000


def _evidence_matches_transition(item: EvidenceItem, transition: Mapping[str, Any]) -> bool:
    if item.kind != "bt_transition" or not isinstance(item.value, Mapping):
        return False
    value = item.value
    recorded_node_uid = value.get("node_uid")
    return (
        value.get("node") == transition["node_name"]
        and value.get("from") == transition["previous_status"]
        and value.get("to") == transition["current_status"]
        and (recorded_node_uid is None or recorded_node_uid == transition["node_uid"])
        and item.timestamp is not None
        and abs(item.timestamp - _stamp_seconds(transition["event_stamp"])) < 1e-6
    )


def _match_derived_evidence_ids(
    episode: EpisodeRecord, transitions: Sequence[dict[str, Any]]
) -> None:
    unused = list(episode.evidence)
    for transition in transitions:
        match = next(
            (item for item in unused if _evidence_matches_transition(item, transition)), None
        )
        if match is not None:
            transition["derived_evidence_id"] = match.id
            unused.remove(match)


def _terminal_status_from_action_status(
    records: Sequence[Mapping[str, Any]], goal_id: str
) -> tuple[str | None, list[str]]:
    names = {4: "succeeded", 5: "canceled", 6: "aborted"}
    candidates: list[tuple[int, str, str]] = []
    for index, record in enumerate(records):
        if record.get("type") != "action_status":
            continue
        for status in record.get("statuses", ()):
            if status.get("goal_id") == goal_id and status.get("status") in names:
                candidates.append(
                    (
                        int(record.get("received_wall_time_ns", 0)),
                        names[int(status["status"])],
                        _record_id(index),
                    )
                )
    if not candidates:
        return None, []
    _, status, record_id = max(candidates)
    return status, [record_id]


def _ordered_recovery_entries(
    transitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures = [
        item
        for item in transitions
        if item["node_name"] == "FollowPath"
        and item["previous_status"] == "RUNNING"
        and item["current_status"] == "FAILURE"
    ]
    guards = [
        item
        for item in transitions
        if item["node_name"] == "WouldAControllerRecoveryHelp"
        and item["previous_status"] == "IDLE"
        and item["current_status"] == "SUCCESS"
    ]
    waits = [
        item
        for item in transitions
        if item["node_name"] == "Wait"
        and item["previous_status"] == "IDLE"
        and item["current_status"] == "RUNNING"
    ]
    used_guards: set[str] = set()
    used_waits: set[str] = set()
    entries: list[dict[str, Any]] = []
    for failure in failures:
        failure_time = _stamp_seconds(failure["event_stamp"])
        guard = next(
            (
                item
                for item in guards
                if item["id"] not in used_guards
                and _stamp_seconds(item["event_stamp"]) >= failure_time
            ),
            None,
        )
        if guard is None:
            continue
        guard_time = _stamp_seconds(guard["event_stamp"])
        wait = next(
            (
                item
                for item in waits
                if item["id"] not in used_waits
                and _stamp_seconds(item["event_stamp"]) >= guard_time
            ),
            None,
        )
        if wait is None:
            continue
        used_guards.add(str(guard["id"]))
        used_waits.add(str(wait["id"]))
        entries.append(
            {
                "id": f"recovery-entry-{len(entries) + 1}",
                "follow_path_failure_transition_id": failure["id"],
                "guard_success_transition_id": guard["id"],
                "wait_entry_transition_id": wait["id"],
                "derived_evidence_ids": [
                    item["derived_evidence_id"]
                    for item in (failure, guard, wait)
                    if "derived_evidence_id" in item
                ],
            }
        )
    return entries


def build_nav2_runtime_presentation(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    behavior_tree_xml: bytes,
    episode: EpisodeRecord,
    runtime_manifest_payload: bytes | None = None,
) -> dict[str, Any]:
    """Build and validate one shared F/G/H runtime presentation.

    F keeps the raw records.  G and H receive this presentation.  ``field_provenance`` is retained
    for the pre-call audit, not as a substitute for source-code provenance.
    """

    if not records:
        raise PresentationError("capture contains no records")
    episode_id = str(manifest.get("episode_id", ""))
    if not episode_id or episode_id != episode.episode_id:
        raise PresentationError("capture manifest and derived episode IDs disagree")
    xml_sha256 = hashlib.sha256(behavior_tree_xml).hexdigest()
    if xml_sha256 != manifest.get("bt_xml_sha256"):
        raise PresentationError("captured BT XML does not match the manifest SHA-256")
    runtime_provenance = None
    if runtime_manifest_payload is not None:
        retained_sha256 = hashlib.sha256(runtime_manifest_payload).hexdigest()
        if retained_sha256 != manifest.get("runtime_manifest_sha256"):
            raise PresentationError("runtime manifest does not match the capture manifest SHA-256")
        runtime_provenance = json.loads(runtime_manifest_payload)
        if runtime_provenance.get("schema") != "crane-runtime-provenance/v1":
            raise PresentationError("unsupported runtime provenance schema")
        if runtime_provenance.get("run_id") != manifest.get("run_id"):
            raise PresentationError("runtime provenance and capture run IDs disagree")
    elif manifest.get("runtime_manifest_sha256"):
        raise PresentationError("capture manifest names a missing runtime manifest")

    indexed = [(_record_id(index), record) for index, record in enumerate(records)]
    harness = [
        (record_id, record["event"])
        for record_id, record in indexed
        if record.get("type") == "harness_event" and isinstance(record.get("event"), Mapping)
    ]
    goals = [
        (record_id, event)
        for record_id, event in harness
        if event.get("type") == "navigate_to_pose_goal" and event.get("accepted") is True
    ]
    if len(goals) != 1:
        raise PresentationError(f"expected one accepted NavigateToPose goal, found {len(goals)}")
    goal_record_id, goal = goals[0]
    goal_id = str(goal["goal_id"])

    results = [
        (record_id, event)
        for record_id, event in harness
        if event.get("type") == "navigate_to_pose_result" and event.get("goal_id") == goal_id
    ]
    if len(results) > 1:
        raise PresentationError("capture has multiple terminal results for the accepted goal")
    if results:
        terminal_record_id, result = results[0]
        terminal = {
            "source": "navigate_to_pose_result",
            "goal_id": goal_id,
            "status": result.get("status"),
            "status_code": result.get("status_code"),
            "error_code": result.get("error_code"),
            "error_msg": result.get("error_msg", ""),
            "wall_time_ns": result.get("wall_time_ns"),
        }
        terminal_sources = [terminal_record_id]
    else:
        status, terminal_sources = _terminal_status_from_action_status(records, goal_id)
        if status is None:
            raise PresentationError("capture has no terminal result or terminal action status")
        terminal = {
            "source": "action_status",
            "goal_id": goal_id,
            "status": status,
            "status_code": None,
            "error_code": None,
            "error_msg": None,
            "wall_time_ns": None,
        }

    feedback = [
        (record_id, record)
        for record_id, record in indexed
        if record.get("type") == "navigate_to_pose_feedback"
    ]
    if not feedback:
        raise PresentationError("capture contains no NavigateToPose feedback")
    feedback_goal_ids = sorted({str(record.get("goal_id")) for _, record in feedback})
    recovery_counts = [int(record["number_of_recoveries"]) for _, record in feedback]
    feedback_summary = {
        "message_count": len(feedback),
        "goal_ids": feedback_goal_ids,
        "first": {
            key: feedback[0][1].get(key)
            for key in (
                "goal_id",
                "navigation_time_s",
                "estimated_time_remaining_s",
                "number_of_recoveries",
                "distance_remaining",
                "current_pose_frame",
                "current_pose_stamp",
            )
        },
        "last": {
            key: feedback[-1][1].get(key)
            for key in (
                "goal_id",
                "navigation_time_s",
                "estimated_time_remaining_s",
                "number_of_recoveries",
                "distance_remaining",
                "current_pose_frame",
                "current_pose_stamp",
            )
        },
        "minimum_number_of_recoveries": min(recovery_counts),
        "maximum_number_of_recoveries": max(recovery_counts),
        "distinct_number_of_recoveries": sorted(set(recovery_counts)),
    }

    transitions: list[dict[str, Any]] = []
    transition_sources: list[str] = []
    for record_id, record in indexed:
        if record.get("type") != "bt_transition":
            continue
        transition_sources.append(record_id)
        transitions.append(
            {
                "id": f"bt-transition-{len(transitions) + 1:06d}",
                "raw_record_id": record_id,
                "node_name": record["node_name"],
                "node_uid": record["node_uid"],
                "previous_status": record["previous_status"],
                "current_status": record["current_status"],
                "event_stamp": _stamp(record),
                "message_stamp": record.get("message_stamp"),
            }
        )
    _match_derived_evidence_ids(episode, transitions)
    recovery_entries = _ordered_recovery_entries(transitions)

    capture_started = [record_id for record_id, item in indexed if item.get("type") == "capture_started"]
    capture_stopped = [record_id for record_id, item in indexed if item.get("type") == "capture_stopped"]
    boundaries_complete = (
        len(capture_started) == 1
        and len(capture_stopped) == 1
        and records[0].get("type") == "capture_started"
        and records[-1].get("type") == "capture_stopped"
    )
    outcome = episode.outcome
    if outcome is None:
        raise PresentationError("derived episode lacks an outcome")
    if terminal["status"] != outcome.terminal_status:
        raise PresentationError("raw and derived terminal statuses disagree")
    recovery_complete = outcome.recovery_history_complete
    whole_history_complete = outcome.history_complete

    limitations = [str(value) for value in manifest.get("limitations", ())]
    limitations.extend(
        (
            "a delivered topic value is not proof that a controller consumed it",
            "BT/action chronology establishes software execution order, not physical causation",
            "the capture contains no evaluator-only intervention or simulator ground truth",
        )
    )
    limitations = list(dict.fromkeys(limitations))

    transition_raw_ids = {item["id"]: item["raw_record_id"] for item in transitions}
    feedback_record_ids = [record_id for record_id, _ in feedback]
    record_type_inventory_hash = _canonical_sha256(
        sorted(str(item.get("type")) for item in records)
    )
    field_provenance = {
        "action.accepted_goal": [goal_record_id],
        "action.terminal": terminal_sources,
        "action.feedback_summary": [
            _record_selector("navigate_to_pose_feedback", feedback_record_ids)
        ],
        "bt_transitions": [_record_selector("bt_transition", transition_sources)],
        "recovery_entries": sorted(
            {
                transition_raw_ids[transition_id]
                for entry in recovery_entries
                for transition_id in (
                    entry["follow_path_failure_transition_id"],
                    entry["guard_success_transition_id"],
                    entry["wait_entry_transition_id"],
                )
            }
        ),
        "completeness.capture_boundaries_complete": capture_started + capture_stopped,
        "completeness.complete_bt_execution_history": [
            "derived-field:bt_transitions",
            *terminal_sources,
        ],
        "completeness.recovery_count_history_complete": [
            "derived-field:action.feedback_summary",
            "derived-field:recovery_entries",
            *terminal_sources,
        ],
        "evidence_scope.limitations": [
            "capture-manifest",
            f"capture-record-type-inventory:{record_type_inventory_hash}",
        ],
    }
    if runtime_provenance is not None:
        field_provenance["runtime_provenance"] = ["runtime-manifest"]
    presentation = {
        "schema": PRESENTATION_SCHEMA,
        "episode_id": episode_id,
        "capture": {
            "schema": manifest.get("schema"),
            "run_id": manifest.get("run_id"),
            "record_count": len(records),
            "behavior_tree": {
                "source_as_recorded": manifest.get("bt_xml_source"),
                "sha256": xml_sha256,
            },
        },
        "action": {
            "accepted_goal": {
                key: goal.get(key)
                for key in (
                    "goal_id",
                    "action_name",
                    "action_mode",
                    "goal_attempt",
                    "goal",
                    "wall_time_ns",
                )
            },
            "terminal": terminal,
            "feedback_summary": feedback_summary,
        },
        "bt_transitions": transitions,
        "recovery_entries": recovery_entries,
        "completeness": {
            "capture_boundaries_complete": boundaries_complete,
            "complete_bt_execution_history": whole_history_complete,
            "recovery_count_history_complete": recovery_complete,
            "scope_note": (
                "Recovery-count completeness and complete BT-execution-history completeness are "
                "separate claims. A terminal BT transition can be absent even when the accepted "
                "goal, terminal action state, final monotonic feedback count, and distinct "
                "recovery entries establish the recovery count."
            ),
        },
        "evidence_scope": {
            "limitations": limitations,
            "physical_cause": "not_established_by_available_capture",
            "counterfactual_outcome": "not_established_by_available_capture",
        },
        "field_provenance": field_provenance,
    }
    if runtime_provenance is not None:
        presentation["runtime_provenance"] = runtime_provenance
    return presentation


_QUESTION_REQUIREMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "recovery-mechanism": (
        ("accepted-goal-identity", "action.accepted_goal"),
        ("terminal-result-identity", "action.terminal"),
        ("feedback-recovery-summary", "action.feedback_summary"),
        ("exact-bt-transitions", "bt_transitions"),
        ("ordered-recovery-entries", "recovery_entries"),
        ("bt-xml-identity", "capture.behavior_tree"),
        ("bt-history-completeness", "completeness.complete_bt_execution_history"),
        ("recovery-history-completeness", "completeness.recovery_count_history_complete"),
        ("physical-cause-boundary", "evidence_scope.physical_cause"),
    ),
    "failure-cause": (
        ("accepted-goal-identity", "action.accepted_goal"),
        ("terminal-result-identity", "action.terminal"),
        ("exact-bt-transitions", "bt_transitions"),
        ("bt-history-completeness", "completeness.complete_bt_execution_history"),
        ("capture-limitations", "evidence_scope.limitations"),
        ("physical-cause-boundary", "evidence_scope.physical_cause"),
    ),
}


def _lookup(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise PresentationError(f"presentation lacks required field {path}")
        value = value[part]
    return value


def _raw_sources_for_path(presentation: Mapping[str, Any], path: str) -> list[str]:
    provenance = presentation.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise PresentationError("presentation lacks field_provenance")
    if path == "capture.behavior_tree":
        return ["capture-manifest", "behavior-tree-xml"]
    if path == "evidence_scope.physical_cause":
        return list(provenance.get("evidence_scope.limitations", ()))
    sources = provenance.get(path)
    if sources is None and path == "evidence_scope.limitations":
        sources = provenance.get("evidence_scope.limitations")
    if not isinstance(sources, list) or not sources:
        raise PresentationError(f"no raw derivation sources recorded for {path}")
    return [str(value) for value in sources]


def audit_fgh_information_parity(
    presentation: Mapping[str, Any], question_kind: str
) -> dict[str, Any]:
    """Audit question-relevant runtime units before any F/G/H model call.

    Source anchors/spans are intentionally excluded: they are G's treatment. F and H instead get
    the exact checkout and may retrieve source without the retained runtime-to-source link.
    """

    if presentation.get("schema") != PRESENTATION_SCHEMA:
        raise PresentationError("unsupported runtime-presentation schema")
    try:
        requirements = list(_QUESTION_REQUIREMENTS[question_kind])
    except KeyError as error:
        raise PresentationError(f"unknown question kind: {question_kind}") from error
    if "runtime_provenance" in presentation:
        requirements.append(("runtime-configuration-identity", "runtime_provenance"))
    units = []
    for unit_id, path in requirements:
        value = _lookup(presentation, path)
        sources = _raw_sources_for_path(presentation, path)
        units.append(
            {
                "id": unit_id,
                "structured_path": path,
                "value_sha256": _canonical_sha256(value),
                "raw_derivation_sources": sources,
                "condition_access": {
                    "F": "raw_capture_derivable",
                    "G": "shared_runtime_presentation",
                    "H": "shared_runtime_presentation",
                },
                "accepted": True,
            }
        )
    return {
        "schema": PARITY_AUDIT_SCHEMA,
        "episode_id": presentation["episode_id"],
        "question_kind": question_kind,
        "accepted": all(unit["accepted"] for unit in units),
        "shared_runtime_presentation_sha256": _canonical_sha256(presentation),
        "units": units,
        "treatment_specific_information": {
            "F": "unrestricted retrieval over the exact repository checkout",
            "G": "retained runtime-to-source links plus bounded source context and checking",
            "H": "unrestricted retrieval over the exact repository checkout",
        },
        "excluded_from_shared_runtime_parity": [
            "runtime-to-source links",
            "bounded source spans",
            "checked answer plan",
            "final-text verification result",
        ],
        "evaluator_truth_available_to_methods": False,
    }


def validate_episode_projection(
    presentation: Mapping[str, Any], episode: EpisodeRecord
) -> None:
    """Reject a checked-plan input that disagrees with the shared runtime presentation.

    The episode is the narrow internal type consumed by current deterministic reasoners.  This
    check makes that narrowing explicit: it may omit irrelevant raw publications, but it may not
    change the terminal state, completeness scopes, recovery count, answerability boundaries, or
    any BT transition it retains.
    """

    if presentation.get("schema") != PRESENTATION_SCHEMA:
        raise PresentationError("unsupported runtime-presentation schema")
    if presentation.get("episode_id") != episode.episode_id:
        raise PresentationError("presentation and checked-plan episode IDs disagree")
    if episode.outcome is None:
        raise PresentationError("checked-plan episode lacks an outcome")
    outcome = episode.outcome
    if _lookup(presentation, "action.terminal")["status"] != outcome.terminal_status:
        raise PresentationError("presentation and checked-plan terminal statuses disagree")
    if (
        _lookup(presentation, "completeness.complete_bt_execution_history")
        != outcome.history_complete
    ):
        raise PresentationError("presentation and checked-plan BT completeness disagree")
    if (
        _lookup(presentation, "completeness.recovery_count_history_complete")
        != outcome.recovery_history_complete
    ):
        raise PresentationError("presentation and checked-plan recovery completeness disagree")

    evidence_by_id = {item.id: item for item in episode.evidence}
    expected_scalars = {
        "terminal-status": _lookup(presentation, "action.terminal")["status"],
        "recovery-count": _lookup(presentation, "action.feedback_summary")[
            "maximum_number_of_recoveries"
        ],
        "physical-cause-status": "not_established",
        "counterfactual-status": "not_established",
        "history-completeness": _lookup(
            presentation, "completeness.complete_bt_execution_history"
        ),
        "recovery-history-completeness": _lookup(
            presentation, "completeness.recovery_count_history_complete"
        ),
    }
    for evidence_id, expected in expected_scalars.items():
        item = evidence_by_id.get(evidence_id)
        if item is None or item.value != expected:
            raise PresentationError(
                f"checked-plan evidence {evidence_id} disagrees with the presentation"
            )

    transitions = {
        item["derived_evidence_id"]: item
        for item in _lookup(presentation, "bt_transitions")
        if "derived_evidence_id" in item
    }
    for evidence_id, item in evidence_by_id.items():
        if item.kind != "bt_transition":
            continue
        transition = transitions.get(evidence_id)
        if transition is None or not _evidence_matches_transition(item, transition):
            raise PresentationError(
                f"checked-plan BT evidence {evidence_id} disagrees with the presentation"
            )


def load_jsonl_records(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Parse capture JSONL without adding a filesystem dependency to the core logic."""

    return [json.loads(line) for line in lines if line.strip()]
