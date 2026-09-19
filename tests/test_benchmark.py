from crane_explain.benchmark import (
    BenchmarkCase, Condition, audit_information_parity, run_condition,
)
from test_dock_slalom import decision


def case(prose_ids=None):
    facts = frozenset({"selected", "dock-score", "slalom-score", "policy"})
    return BenchmarkCase(
        "c1", decision(),
        "Under synthetic-test-v1, Dock was selected with score 8.3; Slalom scored 6.8.",
        "Why Dock rather than Slalom?", "contrast", "Slalom", facts,
        facts if prose_ids is None else frozenset(prose_ids),
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
