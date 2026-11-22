import os, pathlib
os.environ["CAPXFER_STUDY_CONSTANTS"] = str(pathlib.Path(__file__).parent / "fixtures" / "study_constants_e.yaml")
"""WS-E end-to-end on synthetic stubs: healthy, singleton fault, ambiguous fault, OOD payload/low_mu, E0 vs E2."""
import numpy as np
import pytest

from capxfer import explain as X
from capxfer.explain import Evidence, load_contracts
from capxfer.explain._stub import SIDEWAYS_DOCK, case

C = load_contracts()


def run(name, decision=None, B=1):
    c = case(name, B=B)
    cal = X.explain(c["pred"], c["intervals"], c["wheel_sets"], c["ood_flag"], c["asm"], c["rung"], decision=decision)
    pt = X.point_estimate(c["pred"], c["asm"], c["rung"], decision=decision)
    unc = X.uncalibrated(c["pred"], c["asm"], c["rung"], decision=decision)
    return c, cal, pt, unc


def types(e, support="supported"):
    return [(cl["type"], cl["value"]) for cl in e["claims"] if cl["support"] == support]


def evidence(c, i=0):
    return Evidence(lo=c["intervals"][0][i], hi=c["intervals"][1][i], sets=c["wheel_sets"][i], ood=bool(c["ood_flag"][i]),
                    rung=c["rung"], tau=C.tau, keys=C.keys, wheels=tuple(c["asm"].wheel_corner))


@pytest.mark.parametrize("name", ["healthy", "singleton_roller", "ambiguous_e0", "ood_payload", "ood_low_mu"])
@pytest.mark.parametrize("dec", [None, SIDEWAYS_DOCK])
def test_every_explanation_verifies_and_calibrated_claims_are_licensed(name, dec):
    c, cal, pt, unc = run(name, dec)
    for e in cal + pt + unc:
        assert e["verification"]["accepted"], e["verification"]["reasons"]
        assert X.verify(e)["accepted"]
        assert set(e) >= {"text", "claims", "withheld", "disposition"}
    assert X.license_violations(cal[0], evidence(c)) == []


def test_healthy():
    c, cal, pt, _ = run("healthy")
    e = cal[0]
    assert e["disposition"] == "full" and e["withheld"] == []
    assert ("wheel_mode", "ok") in types(e) and not any(t == "cause" for t, _ in types(e))
    s = X.score(cal, c["labels"], c["y_true"])
    assert s["correct_claim_rate"] == 1.0 and s["informativeness"] == 1.0 and s["unsupported_cause_rate"] == 0.0


def test_singleton_fault_names_wheel_and_cause():
    c, cal, _, _ = run("singleton_roller", SIDEWAYS_DOCK)
    e = cal[0]
    assert ("wheel_mode", "roller_locked") in types(e) and ("cause", "roller_locked") in types(e)
    assert e["decision"]["outcome"] == "no_go" and e["next_probe"] is None
    s = X.score(cal, c["labels"], c["y_true"], decision=SIDEWAYS_DOCK)
    assert s["correct_claim_rate"] == 1.0 and s["fault_localization_rate"] == 1.0 and s["unsafe_go_rate"] == 0.0


def test_ambiguous_fault_withholds_cause_and_requests_probe():
    c, cal, pt, unc = run("ambiguous_e0", SIDEWAYS_DOCK)
    e = cal[0]
    assert not any(t in ("cause",) for t, _ in types(e))
    assert not any(t == "wheel_mode" and v != "ok" for t, v in types(e))
    assert any(cl["contract_id"] == "wheel_ambiguous" for cl in e["claims"] if cl["support"] == "withheld")
    assert e["decision"]["outcome"] == "probe" and e["next_probe"] == {"pid": "P2", "key": "P2+"}
    # the point estimate names the wrong wheel (argmax rr) as the cause: unlicensed and contradicted by truth
    assert ("cause", "roller_locked") in types(pt[0])
    assert X.license_violations(pt[0], evidence(c))
    s_cal = X.score(cal, c["labels"], c["y_true"], decision=SIDEWAYS_DOCK)
    s_pt = X.score(pt, c["labels"], c["y_true"], decision=SIDEWAYS_DOCK)
    assert s_cal["unsupported_cause_rate"] == 0.0 and s_pt["unsupported_cause_rate"] == 1.0
    assert s_cal["deferral_rate"] == 1.0 and s_cal["informativeness"] == 0.0


@pytest.mark.parametrize("name", ["ood_payload", "ood_low_mu"])
def test_ood_names_no_wheel(name):
    c, cal, pt, unc = run(name, B=8)
    for e in cal:
        assert types(e) == [("ood", "outside_known_wheel_faults")] and e["disposition"] == "abstain"
    s_cal, s_pt, s_unc = (X.score(x, c["labels"], c["y_true"]) for x in (cal, pt, unc))
    assert s_cal["ood_misattribution"] == 0.0 and s_cal["unsupported_cause_rate"] == 0.0
    assert s_cal["correct_rate_by_type"]["ood"] == 1.0 and s_cal["abstention_rate"] == 1.0
    assert s_pt["ood_misattribution"] == 1.0 and s_pt["unsupported_cause_rate"] == 1.0
    # uncalibrated uncertainty has no OOD test: it never flags the trial and still certifies capability
    assert s_unc["claims_by_type"]["ood"] == 0 and s_unc["informativeness"] > 0.5


def test_e0_vs_e2_specificity():
    c0, cal0, _, _ = run("ambiguous_e0")
    c2, cal2, _, _ = run("roller_e2")
    assert c0["rung"] == "E0" and c2["rung"] == "E2"
    assert not any(t == "cause" for t, _ in types(cal0[0]))
    assert ("cause", "roller_locked") in types(cal2[0])
    s0 = X.score(cal0, c0["labels"], c0["y_true"])
    s2 = X.score(cal2, c2["labels"], c2["y_true"])
    assert s2["informativeness"] > s0["informativeness"]
    assert s0["unsupported_cause_rate"] == s2["unsupported_cause_rate"] == 0.0
    assert s2["fault_localization_rate"] == 1.0 and s0["fault_localization_rate"] == 0.0


def test_torch_inputs_and_blackbox_without_sets():
    torch = pytest.importorskip("torch")
    c = case("singleton_roller", B=3)
    lo, hi = (torch.tensor(x) for x in c["intervals"])
    out = X.explain(c["pred"], (lo, hi), None, torch.tensor(c["ood_flag"]), c["asm"], "E1")
    for e in out:
        assert e["verification"]["accepted"]
        assert not any(cl["type"] in ("cause",) and cl["support"] == "supported" for cl in e["claims"])
        assert any(cl["contract_id"] == "wheel_no_evidence" for cl in e["claims"])


def test_random_evidence_calibrated_always_licensed_and_verified():
    rng = np.random.default_rng(1)
    c = case("healthy")
    B, n = 40, c["asm"].n
    lo = rng.uniform(0, 1, (B, 14))
    hi = lo + rng.uniform(0, 0.5, (B, 14))
    sets = rng.random((B, n, 5)) < 0.3
    sets[..., 0] |= rng.random((B, n)) < 0.6
    ood = rng.random(B) < 0.2
    for rung in ("E0", "E2"):
        out = X.explain(None, (lo, hi), sets, ood, c["asm"], rung, decision=SIDEWAYS_DOCK if rung == "E2" else None)
        for i, e in enumerate(out):
            assert e["verification"]["accepted"], e["verification"]["reasons"]
            ev = Evidence(lo=lo[i], hi=hi[i], sets=sets[i], ood=bool(ood[i]), rung=rung, tau=C.tau, keys=C.keys,
                          wheels=tuple(c["asm"].wheel_corner))
            assert X.license_violations(e, ev) == []
