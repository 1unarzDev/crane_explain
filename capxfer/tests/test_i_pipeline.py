"""WS-I integration pipeline: freeze manifest, adapters, explanation stage, docking stub (each test < 60 s)."""
import json

import numpy as np
import pytest
import torch

from capxfer import assembly as A
from capxfer import sim as S
from capxfer.field import schedule as SCH
from capxfer.pipeline import adapters as AD
from capxfer.pipeline import dryrun as DR
from capxfer.pipeline import freeze as FR
from capxfer.pipeline import stages as ST

TINY = dict(counts={"F0": 1, "F1": 1, "F2": 0, "F3": 0, "F5": 1}, operator={"F4": (0, 1), "F6": (1, 1), "F7": (0, 1)},
            auto_block=4)


def _schedule(tmp_path, assemblies=("pair",)):
    w = SCH.write(tmp_path / "sched", seed_hex="ab" * 16, seed_dir=tmp_path / "sealed", assemblies=assemblies, **TINY)
    return w, SCH.load(w["schedule"])


def test_freeze_roundtrip(tmp_path):
    md = tmp_path / "models"
    md.mkdir()
    (md / "m1_A.pt").write_bytes(b"model one")
    _schedule(tmp_path)
    p = tmp_path / "freeze.json"
    man = FR.write(p, models_dir=md, schedule_dir=tmp_path / "sched", methods=["m1"], suffix="_A")
    assert man["models"] == {"m1": "m1_A.pt"} and "seed.txt" not in json.dumps(man)
    assert FR.verify(p) == []
    (md / "m1_A.pt").write_bytes(b"model two")
    assert any("changed models/m1_A.pt" in q for q in FR.verify(p))
    man = FR.load(p)
    man["files"] = man["files"][1:]
    assert any("aggregate" in q for q in FR.verify(man, check_new_code=False))


def test_freeze_refuses_unsealed_schedule(tmp_path):
    md = tmp_path / "models"
    md.mkdir()
    (md / "m1.pt").write_bytes(b"x")
    _schedule(tmp_path)
    (tmp_path / "sched" / "schedule.sha256").write_text("0" * 64 + "  schedule.json\n")
    with pytest.raises(ValueError):
        FR.build(md, tmp_path / "sched", methods=["m1"])


def test_perturbed_domain_patch_is_scoped():
    orig = S.sample_domain
    with DR.perturbed_domain():
        d = S.sample_domain(64, np.random.default_rng(0))
        assert abs(float(d["mu"].mean()) - DR.TRUE_DOMAIN["mu"]) < 0.01
        assert set(orig(2, np.random.default_rng(0))) == set(d)
    assert S.sample_domain is orig


def test_session_join_and_subset(tmp_path):
    from capxfer.field import loader as LD
    from capxfer.field import synth_session as SS
    _, sched = _schedule(tmp_path)
    n = sum(len(b["trials"]) for b in sched["sessions"][0]["blocks"])
    with DR.perturbed_domain():
        d, _ = SS.make_session(tmp_path / "sess", "pair", n_trials=n, seed=3, schedule=sched, device="cpu")
    raw = LD.load_session(d, allow_unverified=True)[0]
    assert raw["meta"]["source"] == "SIMULATION"
    meta = AD.trial_meta(raw, AD.schedule_index(sched))
    assert sorted(meta["fault"].tolist()) == sorted(t["fault_id"] for _, _, t in SCH.iter_trials(sched))
    assert len(set(meta["module_set"])) == 1 and "+" in meta["module_set"][0]
    h = [i for i, f in enumerate(meta["fault"]) if f == "F0"]
    sub = AD.subset_raw(raw, h)
    assert sub["duty"].shape[0] == len(h) and sub["meta"]["labels"]["ood"] == "none"
    assert not sub["meta"]["labels"]["mode"].any()
    bad = AD.schedule_index(sched)
    k = next(k for k, t in bad.items() if t["fault_id"] == "F0")
    bad[k] = dict(bad[k], fault_id="F1")
    with pytest.raises(ValueError, match="disagree"):
        AD.trial_meta(raw, bad)


def test_explain_stage_on_synthetic_tables():
    from capxfer.analysis import _synth as SY
    rng = np.random.default_rng(0)
    df = SY._trials(rng, 6)
    modes, mean, y = SY._truth(rng, df)
    t = SY._pred_table(rng, df, modes, mean, y, err=0.03, spread=1.0, sharp=3.0, source="SIMULATION")
    prm = ST.params()
    loaded = {k: ({"meta": {"assembly": A.physical(k, prm)}}, None) for k in SY.TYPES}
    ex, info = ST.explain_stage(t, loaded, dict(alpha=0.1, seed=0))
    assert set(ex.explainer) == {"calibrated", "point_estimate"}
    assert len(ex) == 2 * len(df) and ex.verified.all()
    assert set(info["folds"]) == set(SY.TYPES)
    cal = ex[ex.explainer == "calibrated"].n_unsupported_cause.mean()
    pe = ex[ex.explainer == "point_estimate"].n_unsupported_cause.mean()
    assert cal <= pe


def test_docking_stub(tmp_path):
    rows = [dict(assembly="single", fault="F4x2d", arm=a, run=i, success=(i % 2 == 0), pos_err=0.01)
            for a in ("proposed", "omni") for i in range(4)] + [dict(assembly="pair", fault="F4", arm="omni", run=0,
                                                                         success=True)]
    src = tmp_path / "pilot.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))
    df = ST.docking_stage(src, tmp_path / "dock.csv")
    assert len(df) == 8 and set(df.columns) >= {"arm", "run", "success"} and df.success.sum() == 4


def test_pred_numpy_and_concat():
    p = dict(y_q=torch.zeros(2, 14, 3), mode_prob=None)
    out = AD.pred_numpy(p, (0.05, 0.5, 0.95))
    assert out["q_levels"].tolist() == [0.05, 0.5, 0.95] and out["mode_prob"] is None
    assert AD.trial_num("sess:gnd_capxprobe_01:17") == 17
