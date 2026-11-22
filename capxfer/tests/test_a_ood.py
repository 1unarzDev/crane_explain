"""WS-A2: per-wheel friction factor, v3 OOD samplers (lifter payload, low-mu patch), paired twin labels."""
import json

import numpy as np
import torch

from capxfer import assembly, datagen, sim

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ENG = "graph" if DEV == "cuda" else "eager"
P = assembly.load_params(datagen.PARAMS)


def _run(asm, B, seed=5, **kw):
    rng = np.random.default_rng(0)
    dom = sim.sample_domain(B, rng)
    mode, g = sim.sample_faults(B, asm.n, rng)
    return sim.simulate(asm, P, dom, mode, g, device=DEV, engine=ENG, seed=seed, **kw)


def test_per_wheel_mu_factor_consistent_with_global():
    asm = assembly.physical("single", P)
    B = 8
    y_none = sim.targets_from_twist(_run(asm, B))[0]
    y_ones = sim.targets_from_twist(_run(asm, B, mu_factor=torch.ones(B, asm.n)))[0]
    assert (y_none - y_ones).abs().max() < 1e-5
    y_glob = sim.targets_from_twist(_run(asm, B, mu_factor=torch.full((B,), 0.3)))[0]
    y_pw = sim.targets_from_twist(_run(asm, B, mu_factor=torch.full((B, asm.n), 0.3)))[0]
    assert (y_glob - y_pw).abs().max() < 1e-5


def test_v3_samplers():
    rng = np.random.default_rng(1)
    asm = assembly.physical("pair", P)
    dom = sim.sample_domain(64, rng)
    e = datagen.ood_args(rng, asm, dom, 64, "payload_lifter", P)
    pay = e["payload"]
    assert pay["mass"].min() >= 2.0 and pay["mass"].max() <= 6.0
    assert pay["dx"].abs().max() <= 0.12 + 1e-6 and pay["dy"].abs().max() <= 0.12 + 1e-6
    lifters = set(datagen.lifter_modules(asm, P))
    assert set(pay["module"].long().tolist()) <= lifters
    e = datagen.ood_args(rng, asm, dom, 64, "low_mu_patch", P)
    mf = e["mu_factor"]
    assert mf.shape == (64, asm.n)
    low = mf < 1
    assert (low.sum(1) == asm.n // 2).all()                        # one side of a symmetric assembly = half
    assert mf[low].min() >= 0.2 - 1e-6 and mf[low].max() <= 0.4 + 1e-6
    single = assembly.physical("single", P)
    left = datagen.patch_wheels(single, "left")
    assert [single.wheel_corner[i].split(":")[1] for i in np.flatnonzero(left)] == ["fl", "rl"]


def test_generate_v3_with_faults_and_twin(tmp_path):
    datagen.generate("o", 64, {"single": 1.0}, seed=2, phys_trials_per_asm=32, wheel_budget=256,
                     ood="low_mu_patch", ood_faults=True, engine=ENG, device=DEV, out_root=tmp_path,
                     log=lambda s: None)
    man = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert man["args"]["ood_twin"] and "ood_consequential" in man["counts"]
    n_faulted, n_cons = 0, 0
    for r in datagen.iter_dataset(tmp_path / "o"):
        lab = r["meta"]["labels"]
        assert lab["ood"] == "low_mu" and lab["ood_kind"] == "low_mu_patch"
        assert lab["domain"]["mu_factor"].shape == lab["mode"].shape
        assert torch.allclose(lab["ood_dy"], lab["rho_twist"] - lab["rho_twist_noood"])
        n_faulted += int((lab["mode"] > 0).any(1).sum())
        n_cons += int(lab["ood_consequential"].sum())
    assert n_faulted > 0 and n_cons == man["counts"]["ood_consequential"]


def test_world_sheet_limits():
    """A sheet covering the whole floor equals a global mu factor; one far away changes nothing."""
    asm = assembly.physical("single", P)
    B = 8
    one = lambda v: torch.full((B,), float(v))
    y_glob = sim.targets_from_twist(_run(asm, B, mu_factor=one(0.3)))[0]
    y_all = sim.targets_from_twist(_run(asm, B, sheet=dict(nx=one(1), ny=one(0), c=one(-50), factor=one(0.3))))[0]
    assert (y_glob - y_all).abs().max() < 1e-4
    y_none = sim.targets_from_twist(_run(asm, B))[0]
    y_far = sim.targets_from_twist(_run(asm, B, sheet=dict(nx=one(1), ny=one(0), c=one(50), factor=one(0.3))))[0]
    assert (y_none - y_far).abs().max() < 1e-4
    rng = np.random.default_rng(0)
    e = datagen.ood_args(rng, asm, sim.sample_domain(16, rng), 16, "low_mu_sheet", P)
    assert e["sheet"]["factor"].min() >= 0.2 and set(e) >= {"sheet", "mu_factor", "patch_side"}
