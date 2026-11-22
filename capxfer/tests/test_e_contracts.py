import os, pathlib
os.environ["CAPXFER_STUDY_CONSTANTS"] = str(pathlib.Path(__file__).parent / "fixtures" / "study_constants_e.yaml")
"""WS-E: the wheel claim-contract catalog is complete, operative and closed."""
import itertools

import numpy as np

from capxfer.explain.contracts import PREDICATES, Evidence, load_contracts

C = load_contracts()


def _ev(sets=None, ood=False, rung="E2", lo=0.8, hi=0.9):
    return Evidence(lo=np.full(14, lo), hi=np.full(14, hi), sets=sets, ood=ood, rung=rung, tau=C.tau,
                    keys=C.keys, wheels=("a", "b", "c", "d"))


def test_catalog_shape_matches_old_schema():
    d = C.data
    for k in ("schema", "catalog_id", "catalog_version", "evidence_requirements", "claim_contracts",
              "non_entailments", "diagnostic_nodes"):
        assert k in d
    assert d["schema"] == "crane-evidence-calibration-ontology/v1"
    claim_ids = set(C.claims)
    for ne in d["non_entailments"]:
        assert set(ne["basis_claim_ids"]) <= claim_ids
        assert set(ne["unsupported_consequent_claim_ids"]) <= claim_ids
        assert set(ne["additional_evidence_requirement_ids"]) <= set(C.requirements)
    for c in d["claim_contracts"]:
        assert set(c.get("non_entailment_ids", [])) <= set(C.non_entailments)
    for node in d["diagnostic_nodes"]:
        assert set(node["claim_ids"]) <= claim_ids
    assert {r["predicate_id"] for r in d["evidence_requirements"]} <= set(PREDICATES)


def test_floor_and_payload_causes_never_licensed():
    rng = np.random.default_rng(0)
    for _ in range(200):
        sets = rng.random((4, 5)) < 0.4
        ev = _ev(sets=sets, ood=bool(rng.random() < 0.5), rung=rng.choice(["E0", "E2"]),
                 lo=rng.uniform(0, 1), hi=rng.uniform(0, 1))
        assert not C.licensed("claim-cause-floor", ev)
        assert not C.licensed("claim-cause-payload", ev)


def test_upper_bound_licenses_deficit_and_lower_bound_licenses_capability():
    assert C.licensed("claim-cap-unreliable", _ev(lo=0.2, hi=C.tau - 0.01), 3)
    assert not C.licensed("claim-cap-unreliable", _ev(lo=0.2, hi=C.tau + 0.01), 3)
    assert C.licensed("claim-cap-reliable", _ev(lo=C.tau, hi=1.0), 3)
    assert not C.licensed("claim-cap-reliable", _ev(lo=C.tau - 0.01, hi=1.0), 3)
    # OOD licenses only the OOD claim
    ev = _ev(lo=0.9, hi=1.0, ood=True, sets=np.eye(5, dtype=bool)[[0, 0, 0, 4]])
    assert C.licensed("claim-ood", ev)
    for cid in ("claim-cap-reliable", "claim-cap-unreliable", "claim-wheel-healthy", "claim-wheel-mode",
                "claim-cause-wheel"):
        assert not C.licensed(cid, ev, 3)


def test_wheel_naming_requires_singleton_and_cause_requires_unique_account():
    one_hot = np.eye(5, dtype=bool)
    sets = one_hot[[0, 0, 0, 4]]
    ev = _ev(sets=sets, lo=0.2, hi=0.5)
    assert C.licensed("claim-wheel-mode", ev, 3)
    assert C.licensed("claim-cause-wheel", ev, 3)
    amb = sets.copy()
    amb[3, 2] = True                      # {coast, roller}: faulted, mode unresolved
    ev2 = _ev(sets=amb, lo=0.2, hi=0.5)
    assert not C.licensed("claim-wheel-mode", ev2, 3)
    assert C.licensed("claim-wheel-faulted-unresolved", ev2, 3)
    assert not C.licensed("claim-cause-wheel", ev2, 3)
    # a capability deficit alone never names a wheel
    ev3 = _ev(sets=None, lo=0.2, hi=0.5)
    assert C.licensed("claim-cap-unreliable", ev3, 3)
    assert not any(C.licensed("claim-cause-wheel", ev3, w) for w in range(4))


def test_e0_non_singleton_sets_never_license_a_cause():
    one_hot = np.eye(5, dtype=bool)
    for extra_wheel, extra_mode in itertools.product(range(3), range(1, 5)):
        sets = one_hot[[0, 0, 0, 4]].copy()
        sets[extra_wheel, extra_mode] = True       # some other wheel is {ok, x}
        ev = _ev(sets=sets, rung="E0", lo=0.2, hi=0.5)
        assert not C.licensed("claim-cause-wheel", ev, 3)
    assert C.licensed("claim-cause-wheel", _ev(sets=one_hot[[0, 0, 0, 4]], rung="E0", lo=0.2, hi=0.5), 3)
