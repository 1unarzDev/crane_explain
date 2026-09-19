from crane_explain.io import episode_from_dict
from crane_explain.reasoning import plan_contrast, plan_recovery_count
from crane_explain.realize import render_template
from crane_explain.verification import verify_final_text


def decision(policy=True, outcome=None, candidates=None):
    base = [
        {"id": "Dock", "status": "selected", "features": {"distance": 18, "success": .91, "points": 5}, "evidence_ids": ["table"]},
        {"id": "Slalom", "status": "evaluated_rejected", "features": {"distance": 11, "success": .63, "points": 8}, "evidence_ids": ["table"]},
        {"id": "Speed Gate", "status": "evaluated_rejected", "features": {"distance": 27, "success": .86, "points": 10}, "evidence_ids": ["table"]},
    ]
    if policy:
        scores = {"Dock": 8.3, "Slalom": 6.8, "Speed Gate": 7.9}
        contrib = {
            "Dock": {"success": 9.1, "points": 1.0, "distance": -1.8},
            "Slalom": {"success": 6.3, "points": 1.6, "distance": -1.1},
            "Speed Gate": {"success": 8.6, "points": 2.0, "distance": -2.7},
        }
        for item in base:
            item["score"] = scores[item["id"]]
            item["contributions"] = contrib[item["id"]]
    raw = {
        "schema_version": "crane-explain/v1", "episode_id": "dock",
        "evidence": [{"id": "table", "kind": "candidate_table", "value": "fixture"}],
        "decision": {"id": "D42", "selected_id": "Dock", "timestamp": 1,
                     "candidates": candidates or base, "complete_candidate_set": True,
                     "evidence_ids": ["table"]},
    }
    if policy:
        raw["decision"].update(policy_id="synthetic-test-v1", policy_expression="10*p + .2*points - .1*distance")
    if outcome:
        raw["outcome"] = outcome
    return episode_from_dict(raw)


def test_case_a_missing_objective_does_not_invent_utility():
    text = render_template(plan_contrast(decision(False), "Slalom"))
    assert "does not establish a scored reason" in text
    assert "highest" not in text and "outweighed" not in text


def test_case_b_explicit_policy_checks_contributions():
    plan = plan_contrast(decision(), "Slalom")
    text = render_template(plan)
    assert "8.3" in text and "6.8" in text
    assert "success-estimate advantage outweighed" in text
    assert verify_final_text(plan, text).accepted


def test_case_c_distinct_attempt_ids_and_incomplete_qualification():
    outcome = {"terminal_status": "timeout", "timestamp": 9, "history_complete": False,
               "events": [
                   {"id": "e1", "kind": "recovery_attempt", "timestamp": 2, "attempt_id": "r1"},
                   {"id": "e2", "kind": "recovery_attempt", "timestamp": 2.1, "attempt_id": "r1"},
                   {"id": "e3", "kind": "lost_target", "timestamp": 3},
               ]}
    text = render_template(plan_recovery_count(decision(outcome=outcome)))
    assert "At least 1 recovery attempt is recorded" in text
    assert "Exactly" not in text


def test_case_c_two_unique_attempts():
    outcome = {"terminal_status": "failed", "timestamp": 9, "history_complete": True,
               "events": [
                   {"id": "e1", "kind": "recovery_attempt", "timestamp": 2, "attempt_id": "r1"},
                   {"id": "e2", "kind": "recovery_attempt", "timestamp": 4, "attempt_id": "r2"},
               ]}
    assert "Exactly 2 recovery attempts occurred" in render_template(plan_recovery_count(decision(outcome=outcome)))


def test_case_c_misleading_both_retries_premise_is_rejected():
    outcome = {"terminal_status": "timeout", "timestamp": 9, "history_complete": True,
               "events": [{"id": "e1", "kind": "recovery_attempt", "timestamp": 2,
                           "attempt_id": "r1"}]}
    text = render_template(plan_recovery_count(decision(outcome=outcome), premise_count=2))
    assert "premise of 2 attempts is not supported" in text


def test_case_d_later_outcome_does_not_change_decision_rationale():
    success = {"terminal_status": "success", "timestamp": 9, "history_complete": True, "events": []}
    failure = {"terminal_status": "failed", "timestamp": 9, "history_complete": True, "events": []}
    assert plan_contrast(decision(outcome=success), "Slalom") == plan_contrast(decision(outcome=failure), "Slalom")


def test_case_e_infeasible_and_missing_alternatives_are_not_compared():
    candidates = [
        {"id": "Dock", "status": "selected", "score": 8.3, "evidence_ids": ["table"]},
        {"id": "Slalom", "status": "infeasible", "evidence_ids": ["table"]},
    ]
    episode = decision(candidates=candidates)
    assert "infeasible" in render_template(plan_contrast(episode, "Slalom"))
    assert "does not establish that Unknown was considered" in render_template(plan_contrast(episode, "Unknown"))


def test_case_e_tied_missing_and_contradictory_scores_are_qualified():
    tied = [
        {"id": "Dock", "status": "selected", "score": 8.3, "evidence_ids": ["table"]},
        {"id": "Slalom", "status": "evaluated_rejected", "score": 8.3,
         "evidence_ids": ["table"]},
    ]
    assert "scores are tied" in render_template(plan_contrast(decision(candidates=tied), "Slalom"))
    missing = [
        {"id": "Dock", "status": "selected", "score": 8.3, "evidence_ids": ["table"]},
        {"id": "Slalom", "status": "evaluated_rejected", "evidence_ids": ["table"]},
    ]
    assert "does not establish a scored reason" in render_template(
        plan_contrast(decision(candidates=missing), "Slalom"))
    contradictory = [
        {"id": "Dock", "status": "selected", "score": 8.29, "evidence_ids": ["table"]},
        {"id": "Slalom", "status": "evaluated_rejected", "score": 8.30,
         "evidence_ids": ["table"]},
    ]
    assert "contradicts the recorded score ordering" in render_template(
        plan_contrast(decision(candidates=contradictory), "Slalom"))


def test_unplanned_fluent_clause_is_rejected():
    plan = plan_contrast(decision(), "Slalom")
    text = render_template(plan) + " Dock was the objectively best choice."
    result = verify_final_text(plan, text)
    assert not result.accepted
    assert result.unsupported_sentences == ("Dock was the objectively best choice.",)
