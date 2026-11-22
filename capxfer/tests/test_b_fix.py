"""WS-FIX tests for WS-B: loader trial/fault ids joined from markers and the sealed schedule (M4); synth_session
true-plant injection (M6)."""
import copy

import numpy as np
import pytest
import torch

from capxfer import assembly as A
from capxfer import sim as S
from capxfer.field import loader as L
from capxfer.field import schedule as SCH
from capxfer.field import synth_session as SS

P = A.load_params(L.PARAMS_PATH)
SMALL = dict(counts={"F0": 2, "F1": 2, "F2": 1, "F3": 1, "F5": 2}, operator={"F4": (1, 2), "F6": (1, 2), "F7": (1, 2)},
             auto_block=4)
TRUE = dict(mu=0.55, latency=0.075, izz_scale=1.18)


def test_build_domain():
    rng = np.random.default_rng(0)
    d = SS.build_domain(4, rng, dom=dict(TRUE, kp_scale=[0.8, 0.85, 0.9, 0.95]))
    assert torch.allclose(d["mu"], torch.full((4,), 0.55)) and torch.allclose(d["kp_scale"][-1], torch.tensor(0.95))
    assert set(d) == set(S.sample_domain(4, np.random.default_rng(0)))
    d2 = SS.build_domain(3, np.random.default_rng(1), dr={"mu": (0.2, 0.21)})
    assert (d2["mu"] >= 0.2).all() and (d2["mu"] <= 0.21).all()
    d3 = SS.build_domain(2, np.random.default_rng(1), dom=lambda B, rng: {"mu": np.full(B, 0.4)})
    assert torch.allclose(d3["mu"], torch.full((2,), 0.4))
    with pytest.raises(ValueError):
        SS.build_domain(2, rng, dom={"not_a_key": 1.0})
    with pytest.raises(ValueError):
        SS.build_domain(2, rng, dom={"mu": [0.1, 0.2, 0.3]})


@pytest.fixture(scope="module")
def sess(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthfix")
    s = SCH.generate("34" * 16, assemblies=("pair",), params=P, **SMALL)
    pick = [next(t for _, _, t in SCH.iter_trials(s) if t["fault_id"] == f) for f in ("F1", "F5", "F7")]
    s["sessions"][0]["blocks"] = [{"block": 1, "kind": "auto", "fault_id": None, "condition": {}, "trials": pick}]
    d, truth = SS.make_session(root, "pair", n_trials=3, seed=3, schedule=s, dom=TRUE)
    return d, truth, s, L.verify(d)


def test_synth_session_uses_given_plant(sess):
    _, truth, _, rep = sess
    assert rep["verdict"] == "SIMULATION"
    assert torch.allclose(truth["dom"]["mu"], torch.full((3,), 0.55))
    assert torch.allclose(truth["dom"]["latency"], torch.full((3,), 0.075))


def test_loader_emits_trial_and_fault_ids_from_echoes(sess):
    d, truth, s, rep = sess
    lab = L.load_session(d, allow_unverified=True, verdict=rep)[0]["meta"]["labels"]
    assert lab["schedule_trial"] == [t["trial"] for t in truth["trials"]]
    assert lab["fault_id"] == ["F1", "F5", "F7"] and lab["fault_id_source"] == ["echo"] * 3
    assert lab["n_faulted"] == [1, 2, 0] and lab["module_set"][0] == "+".join(sorted(m for m, *_ in truth["asm"].modules))
    assert lab["assembly_type"] == "pair" and lab["schedule_match"] == [True] * 3


def test_loader_joins_sealed_schedule_and_flags_mismatch(sess, tmp_path):
    d, truth, s, rep = sess
    # sealed file path (hash-checked) -> labels from the schedule, consistent with the echoes
    raw = SCH.canonical(s)
    (tmp_path / "schedule.json").write_bytes(raw)
    (tmp_path / "schedule.sha256").write_text(f"{SCH.sha256_bytes(raw)}  schedule.json\n")
    r = L.load_session(d, allow_unverified=True, verdict=rep, schedule=tmp_path / "schedule.json")[0]
    lab = r["meta"]["labels"]
    assert lab["fault_id"] == ["F1", "F5", "F7"] and lab["fault_id_source"] == ["schedule"] * 3
    assert lab["schedule_match"] == [True] * 3 and lab["ood_per_trial"] == ["none", "none", "low_mu"]
    for b, tt in enumerate(truth["trials"]):
        assert torch.equal(lab["mode"][b], tt["mode"]) and torch.allclose(lab["g"][b], tt["g"], atol=1e-3)
    assert r["meta"]["fault_schedule_sha256"] == SCH.sha256_bytes(raw)
    # an altered schedule file is refused
    (tmp_path / "schedule.json").write_bytes(raw.replace(b'"F1"', b'"F2"', 1))
    with pytest.raises(ValueError):
        L.load_session(d, allow_unverified=True, verdict=rep, schedule=tmp_path / "schedule.json")
    # a schedule (dict) that disagrees with the echoes: flagged (prereg exclusion 7b), or refused on request
    bad = copy.deepcopy(s)
    t0 = bad["sessions"][0]["blocks"][0]["trials"][0]
    t0["faults"][0]["mode"] = "coast"
    t0["fault_id"] = "F2"
    lab = L.load_session(d, allow_unverified=True, verdict=rep, schedule=bad)[0]["meta"]["labels"]
    assert lab["schedule_match"] == [False, True, True] and lab["fault_id"][0] == "F2"
    assert lab["schedule_mismatch"][0] and int(lab["mode_echo"][0].max()) == 1
    with pytest.raises(ValueError):
        L.load_session(d, allow_unverified=True, verdict=rep, schedule=bad, on_mismatch="raise")
    # a trial number missing from the schedule always raises
    gone = copy.deepcopy(s)
    gone["sessions"][0]["blocks"][0]["trials"] = gone["sessions"][0]["blocks"][0]["trials"][1:]
    with pytest.raises(ValueError):
        L.load_session(d, allow_unverified=True, verdict=rep, schedule=gone)


def test_infer_fault_id():
    assert L.infer_fault_id([0, 0], "none") == "F0"
    assert L.infer_fault_id([0, 4], "none") == "F4"
    assert L.infer_fault_id([1, 3], "none") == "F5"
    assert L.infer_fault_id([0, 0], "payload") == "F6"
    assert L.infer_fault_id([2, 0], "low_mu") == "F7+F2"
