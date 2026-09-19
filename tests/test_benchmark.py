from crane_explain.benchmark import (
    BenchmarkCase, Condition, audit_information_parity, run_condition,
)
from crane_explain.io import episode_from_dict
from test_dock_slalom import decision


def case(prose_ids=None):
    facts = frozenset({"selected", "dock-score", "slalom-score", "policy"})
    return BenchmarkCase(
        "c1", decision(),
        "Under synthetic-test-v1, Dock was selected with score 8.3; Slalom scored 6.8.",
        "Why Dock rather than Slalom?", "contrast", "Slalom", facts,
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
                {"id": "guard-1", "kind": "controller_recovery_guard_success",
                 "timestamp": 2.0},
                {"id": "wait-1", "kind": "recovery_attempt", "timestamp": 3.0,
                 "attempt_id": "wait-1"},
            ],
        },
    }
    episode = episode_from_dict(raw)
    facts = frozenset(("failure-1", "guard-1", "wait-1", "result"))
    return BenchmarkCase("recovery", episode, "matched prose", "question", kind, None,
                         facts, facts, premise_count)


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
    output = run_condition(case(), Condition.C_PROSE_EXTRACT_CHECKED,
                           extractor=lambda _: decision())
    assert output.verification_accepted
    assert "Dock scored 8.3" in output.text


def test_d_falls_back_without_resampling_on_unverified_language():
    calls = []
    output = run_condition(
        case(), Condition.D_NATIVE_CHECKED,
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
    assert output.disposition == "partial"


def test_misleading_recovery_count_premise_is_rejected():
    output = run_condition(
        recovery_case("recovery_count", premise_count=2), Condition.E_TEMPLATE)
    assert "At least 1 recovery attempt is recorded" in output.text
    assert "premise of 2 attempts is not supported" in output.text


def test_physical_cause_and_counterfactual_are_withheld():
    cause = run_condition(recovery_case("failure_cause"), Condition.E_TEMPLATE)
    hypothetical = run_condition(
        recovery_case("unsupported_counterfactual"), Condition.E_TEMPLATE)
    assert "does not establish the physical cause" in cause.text
    assert cause.disposition == "partial"
    assert "does not establish what would have happened" in hypothetical.text
    assert hypothetical.disposition == "abstain"
