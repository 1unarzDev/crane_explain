"""WS-A dataset generation: manifest, sharding, compact round trip, family filters, OOD labels."""
import json

import numpy as np
import torch

from capxfer import assembly, datagen, sensors, sim

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ENG = "graph" if DEV == "cuda" else "eager"


def test_generate_and_load(tmp_path):
    m = datagen.generate("t", 160, {"single": 0.5, "random": 0.5}, seed=0, sizes=(1, 2), trials_per_asm=16,
                         phys_trials_per_asm=32, wheel_budget=256, shard_trials=64, engine=ENG, device=DEV,
                         out_root=tmp_path, log=lambda s: None)
    man = json.loads((tmp_path / "t" / "manifest.json").read_text())
    assert man["complete"] and man["counts"]["trials"] == 160 == sum(s["trials"] for s in man["shards"])
    assert len(man["params_sha256"]) == 64 and len(man["code_sha256"]) == 64
    raws = list(datagen.iter_dataset(tmp_path / "t"))
    assert sum(r["counts"].shape[0] for r in raws) == 160
    r = raws[0]
    assert r["counts"].dtype == torch.int64 and r["duty"].dtype == torch.float32
    assert isinstance(r["meta"]["assembly"], assembly.Assembly)
    assert r["meta"]["labels"]["rho_twist"].shape[1] == 14
    assert r["meta"]["labels"]["family"] is not None and r["meta"]["source"] == "SIMULATION"
    assert r["cmd"].shape == (r["counts"].shape[0], r["t"].shape[0], 3)


def test_compact_roundtrip():
    p = assembly.load_params(datagen.PARAMS)
    asm = assembly.physical("pair", p)
    rng = np.random.default_rng(0)
    dom = sim.sample_domain(4, rng)
    mode, g = sim.sample_faults(4, asm.n, rng)
    R = sim.simulate(asm, p, dom, mode, g, device=DEV)
    raw = sensors.to_raw(R, p, dom, rng)
    back = datagen.expand(datagen.compact(raw))
    assert torch.equal(back["counts"].cpu(), raw["counts"].cpu())
    assert (back["truth_pose"] - raw["truth_pose"].cpu()).abs().max() < 1e-6
    assert (back["gyro_z"] - raw["gyro_z"].cpu()).abs().max() < 2e-3
    assert (back["cmd"] - raw["cmd"].cpu()).abs().max() < 1e-6
    assert back["meta"]["assembly"].n == asm.n and np.allclose(back["meta"]["assembly"].wheel_pos, asm.wheel_pos)


def test_family_filter_and_ood(tmp_path):
    datagen.generate("h", 32, {"random": 1.0}, seed=1, sizes=(2,), trials_per_asm=16, wheel_budget=256,
                     only_families=["k2_bb1x2"], ood="low_mu", engine=ENG, device=DEV, out_root=tmp_path,
                     log=lambda s: None)
    for r in datagen.iter_dataset(tmp_path / "h"):
        lab = r["meta"]["labels"]
        assert lab["family"] == "k2_bb1x2" and lab["ood"] == "low_mu"
        assert (lab["mode"] == 0).all()
        mu_eff = lab["domain"]["mu"] * lab["domain"]["mu_factor"]
        assert mu_eff.min() >= 0.15 - 1e-5 and mu_eff.max() <= 0.35 + 1e-5
