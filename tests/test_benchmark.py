from dataclasses import replace

from test_dock_slalom import decision

from crane_explain.benchmark import (
    BenchmarkCase,
    Condition,
    RepositoryAccess,
    audit_information_parity,
    run_condition,
)
from crane_explain.io import episode_from_dict
from crane_explain.models import ProvenanceRelationship, ProvenanceStrength, RuntimeSourceLink
from crane_explain.provenance import ResolvedProvenance


def case(prose_ids=None):
    facts = frozenset({"selected", "dock-score", "slalom-score", "policy"})
    return BenchmarkCase(
        "c1",
        decision(),
        "Under synthetic-test-v1, Dock was selected with score 8.3; Slalom scored 6.8.",
        "Why Dock rather than Slalom?",
        "contrast",
        "Slalom",
        facts,
        facts if prose_ids is None else frozenset(prose_ids),
    )


def recovery_case(kind="recovery_mechanism", premise_count=None):
    raw = {
        "schema_version": "crane-explain-episode/v1",
        "episode_id": "captured-recovery",
        "evidence": [
            {"id": "failure-1", "kind": "bt_transition", "value": "FAILURE"},
            {"id": "guard-1", "kind": "bt_transition", "value": "SUCCESS"},
            {"id": "wait-1", "kind": "bt_transition", "value": "RUNNING"},
            {"id": "result", "kind": "action_result", "value": "aborted"},
        ],
        "decision": None,
        "outcome": {
            "terminal_status": "aborted",
            "timestamp": 4.0,
            "history_complete": False,
            "evidence_ids": ["result"],
            "events": [
                {"id": "failure-1", "kind": "follow_path_failure", "timestamp": 1.0},
                {"id": "guard-1", "kind": "controller_recovery_guard_success", "timestamp": 2.0},
                {
                    "id": "wait-1",
                    "kind": "recovery_attempt",
                    "timestamp": 3.0,
                    "attempt_id": "wait-1",
                    "status": "completed_success",
                },
            ],
        },
    }
    episode = episode_from_dict(raw)
    facts = frozenset(("failure-1", "guard-1", "wait-1", "result"))
    return BenchmarkCase(
        "recovery", episode, "matched prose", "question", kind, None, facts, facts, premise_count
    )


def test_parity_audit_rejects_privileged_structure():
    try:
        audit_information_parity(case({"selected", "policy"}))
    except ValueError as exc:
        assert "structured_only" in str(exc)
    else:
        raise AssertionError("parity mismatch passed")


def test_a_and_b_use_same_generator_and_question():
    calls = []
    generator = lambda evidence, question: calls.append((evidence, question)) or "answer"
    run_condition(case(), Condition.A_PROSE_DIRECT, direct_generator=generator)
    run_condition(case(), Condition.B_STRUCTURED_DIRECT, direct_generator=generator)
    assert calls[0][1] == calls[1][1]
    assert calls[0][0] != calls[1][0]


def test_c_uses_extracted_record_and_checked_plan():
    output = run_condition(
        case(), Condition.C_PROSE_EXTRACT_CHECKED, extractor=lambda _: decision()
    )
    assert output.verification_accepted
    assert "Dock scored 8.3" in output.text


def test_d_falls_back_without_resampling_on_unverified_language():
    calls = []
    output = run_condition(
        case(),
        Condition.D_NATIVE_CHECKED,
        plan_realizer=lambda _: calls.append(1) or "Dock was obviously best.",
    )
    assert calls == [1]
    assert output.used_template_fallback and not output.verification_accepted
    assert "objectively best" in output.text and "does not establish" in output.text


def test_e_is_deterministic_template():
    first = run_condition(case(), Condition.E_TEMPLATE)
    second = run_condition(case(), Condition.E_TEMPLATE)
    assert first == second


def test_checked_recovery_mechanism_preserves_physical_cause_limit():
    output = run_condition(recovery_case(), Condition.E_TEMPLATE)
    assert "FollowPath branch returned FAILURE" in output.text
    assert "does not establish the physical reason" in output.text
    assert output.disposition == "full"


def test_misleading_recovery_count_premise_is_rejected():
    output = run_condition(recovery_case("recovery_count", premise_count=2), Condition.E_TEMPLATE)
    assert "At least 1 recovery attempt is recorded" in output.text
    assert "The recorded recovery attempt returned SUCCESS" in output.text
    assert "premise of 2 attempts is not supported" in output.text


def test_recovery_count_completeness_is_independent_of_full_bt_history():
    case_value = recovery_case("recovery_count")
    raw = case_value.episode.to_dict()
    raw["outcome"]["history_complete"] = False
    raw["outcome"]["recovery_history_complete"] = True
    exact_case = BenchmarkCase(
        case_value.case_id,
        episode_from_dict(raw),
        case_value.prose,
        case_value.question,
        case_value.question_kind,
        None,
        case_value.structured_fact_ids,
        case_value.prose_fact_ids,
    )
    output = run_condition(exact_case, Condition.E_TEMPLATE)
    assert "Exactly 1 recovery attempt occurred" in output.text
    assert "incomplete history" not in output.text
    assert output.disposition == "full"


def test_physical_cause_and_counterfactual_are_withheld():
    cause = run_condition(recovery_case("failure_cause"), Condition.E_TEMPLATE)
    hypothetical = run_condition(recovery_case("unsupported_counterfactual"), Condition.E_TEMPLATE)
    assert "does not establish the physical cause" in cause.text
    assert cause.disposition == "partial"
    assert "does not establish what would have happened" in hypothetical.text
    assert hypothetical.disposition == "abstain"


def test_successful_outcome_preserves_recorded_intermediate_failure():
    case_value = recovery_case("failure_cause")
    raw = case_value.episode.to_dict()
    raw["outcome"]["terminal_status"] = "succeeded"
    successful = BenchmarkCase(
        case_value.case_id,
        episode_from_dict(raw),
        case_value.prose,
        case_value.question,
        case_value.question_kind,
        None,
        case_value.structured_fact_ids,
        case_value.prose_fact_ids,
    )
    output = run_condition(successful, Condition.E_TEMPLATE)
    assert "intermediate FollowPath FAILURE" in output.text
    assert "does not establish the physical cause" in output.text
    assert "premise of a navigation failure is contradicted" not in output.text


def test_successful_outcome_without_failure_rejects_failure_cause_premise():
    case_value = recovery_case("failure_cause")
    raw = case_value.episode.to_dict()
    raw["outcome"]["terminal_status"] = "succeeded"
    raw["outcome"]["events"] = []
    successful = BenchmarkCase(
        case_value.case_id,
        episode_from_dict(raw),
        case_value.prose,
        case_value.question,
        case_value.question_kind,
        None,
        case_value.structured_fact_ids,
        case_value.prose_fact_ids,
    )
    output = run_condition(successful, Condition.E_TEMPLATE)
    assert "premise of a navigation failure is contradicted" in output.text


def test_complete_zero_recovery_evidence_rejects_recovery_premise():
    case_value = recovery_case("recovery_mechanism")
    raw = case_value.episode.to_dict()
    raw["outcome"]["events"] = []
    raw["outcome"]["history_complete"] = False
    raw["outcome"]["recovery_history_complete"] = True
    exact_zero = BenchmarkCase(
        case_value.case_id,
        episode_from_dict(raw),
        case_value.prose,
        case_value.question,
        case_value.question_kind,
        None,
        case_value.structured_fact_ids,
        case_value.prose_fact_ids,
    )
    output = run_condition(exact_zero, Condition.E_TEMPLATE)
    assert "Exactly 0 recovery attempts occurred" in output.text
    assert "premise that the Behavior Tree entered recovery is contradicted" in output.text
    assert output.disposition == "full"


def test_planning_failure_reports_software_error_not_physical_cause():
    case_value = recovery_case("planning_failure")
    raw = case_value.episode.to_dict()
    raw["outcome"]["events"] = [
        {
            "id": "planning-error-208",
            "kind": "planning_no_valid_path",
            "timestamp": 3.5,
            "evidence_ids": ["result"],
        }
    ]
    planning_case = BenchmarkCase(
        case_value.case_id,
        episode_from_dict(raw),
        case_value.prose,
        case_value.question,
        "planning_failure",
        None,
        case_value.structured_fact_ids,
        case_value.prose_fact_ids,
    )
    output = run_condition(planning_case, Condition.E_TEMPLATE)
    assert "NO_VALID_PATH (208)" in output.text
    assert "does not establish the physical reason" in output.text
    assert output.disposition == "full"


def test_f_and_h_expose_matched_repository_agent_access_without_evaluator_truth():
    requests = []
    base = case()
    repo_case = replace(
        base,
        raw_robot_visible="raw captured events",
        repository="https://github.com/example/robot.git",
        repository_commit="a" * 40,
    )

    def agent(request):
        requests.append(request)
        return "agent answer"

    f_output = run_condition(repo_case, Condition.F_REPOSITORY_AGENT, repository_agent=agent)
    h_output = run_condition(
        repo_case, Condition.H_UNRESTRICTED_REPOSITORY_AGENT, repository_agent=agent
    )

    assert f_output.text == h_output.text == "agent answer"
    assert requests[0].access == RepositoryAccess.READ_ONLY_UNRESTRICTED
    assert requests[0].evidence == "raw captured events"
    assert requests[1].access == RepositoryAccess.READ_ONLY_UNRESTRICTED
    assert requests[1].evidence.startswith("{")
    assert requests[0].repository_commit == requests[1].repository_commit == "a" * 40
    assert not hasattr(requests[0], "evaluator_truth")


def test_g_uses_bounded_provenance_in_checked_plan_and_final_verification():
    base = recovery_case()
    runtime_ids = ("failure-1", "guard-1", "wait-1")
    resolved = ResolvedProvenance(
        artifacts=(),
        anchors=(),
        links=(
            RuntimeSourceLink(
                id="link",
                runtime_evidence_ids=runtime_ids,
                source_anchor_ids=("exact-recovery-subtree",),
                relationship=ProvenanceRelationship.GOVERNED_BY,
                strength=ProvenanceStrength.EXACT_ARTIFACT,
                rationale="test exact link",
            ),
        ),
        unresolved_runtime_evidence_ids=(),
    )
    provenance_case = replace(base, provenance=resolved)

    output = run_condition(provenance_case, Condition.G_PROVENANCE_CHECKED)

    assert output.verification_accepted
    assert "exact-recovery-subtree" in output.text
    assert "does not establish the physical reason" in output.text
