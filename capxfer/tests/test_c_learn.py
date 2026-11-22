"""WS-C tests: stub data, views, Predictor API for every method (shapes, save/load), decoder morphology-agnosticism.
Each test < 60 s (sim on GPU if available)."""
import numpy as np
import pytest
import torch

from capxfer import assembly as A
from capxfer.learn import _stub as S
from capxfer.learn import data as D
from capxfer.learn import nets as N

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def feats():
    P = S.load_params()
    rng = np.random.default_rng(0)
    out = []
    for name in ("single", "pair"):
        a = A.physical(name, P)
        a.meta["family"] = f"phys_{name}"
        out.append(S.simulate_feat(a, P, 8, rng, device=DEV))
    return out


def test_stub_shapes(feats):
    F = feats[1]
    B, n = 8, 8
    L = sum(2 * v for v in S.SEG_LEN.values())
    assert F["wheel_x"].shape == (B, n, L, 4)
    assert F["body_x"].shape == (B, L, 9)
    assert F["wheel_geom"].shape == (B, n, 8)
    assert F["y"].shape == (B, 14) and torch.isfinite(F["y"]).all()
    assert F["labels"]["mode"].shape == (B, n)


def test_views_zero_unseen(feats):
    v = D.apply_view(feats[1], "E0", ("P1", "P5"))
    assert v["body_x"][..., 3:].abs().sum() == 0              # no IMU / lidar at E0
    assert v["wheel_x"][..., 2].abs().sum() == 0              # no current below E3
    assert v["wheel_x"][:, :, D.SEG["P2"]].abs().sum() == 0   # unseen primitive
    assert v["wheel_x"][:, :, D.SEG["P1"], 0].abs().sum() > 0
    v3 = D.apply_view(feats[1], "E3", D.PIDS)
    assert v3["wheel_x"][..., 2].abs().sum() > 0
    b = D.collate([D.apply_view(feats[0], "E1", ("P1",)), v3])
    assert b["wheel_x"].shape[1] == 8 and (~b["wheel_valid"][:8, 4:]).all()


def test_decoder_any_morphology(feats):
    dec = N.KinematicDecoder()
    for F in feats:
        v = D.apply_view(F, "E0")
        ht = N.health_tokens(v["labels"]["mode"], v["labels"]["g"])[:, None]
        y, s = dec(ht, v["wheel_geom"], v["wheel_valid"])
        assert y.shape == (8, 1, 14) and torch.isfinite(y).all() and (s > 0).all()


def test_decoder_permutation_invariant(feats):
    dec = N.KinematicDecoder()
    v = D.apply_view(feats[1], "E0")
    perm = torch.randperm(8)
    ht = N.health_tokens(v["labels"]["mode"], v["labels"]["g"])[:, None]
    y1, _ = dec(ht, v["wheel_geom"], v["wheel_valid"])
    y2, _ = dec(ht[:, :, perm], v["wheel_geom"][:, perm], v["wheel_valid"][:, perm])
    assert torch.allclose(y1, y2, atol=1e-4)


@pytest.mark.parametrize("meth", ["structured_npe", "blackbox", "nominal_sim"])
def test_predictor_api(feats, meth, tmp_path):
    from capxfer.learn.train import CONFIGS
    klass, cfg, _ = CONFIGS[meth]
    m = klass({**cfg, "steps": 3, "log_every": 1, "val_every": 3, "val_batches": 1, "d": 32, "layers": 1,
               "device": DEV, "mem_frac": 0.1})
    m.fit(D.batch_iter(feats, 8, 4), D.batch_iter(feats, 8, 4, seed=1))
    v = D.apply_view(feats[1], "E1")
    p = m.predict(v, n_samples=16)
    assert p["y_samples"].shape == (8, 16, 14) and p["y_q"].shape == (8, 14, 3) and p["ppc_score"].shape == (8,)
    assert torch.isfinite(p["y_q"]).all() and (p["y_q"].diff(dim=-1) >= -1e-5).all()
    assert p["drift_q"].shape == (8, 14, 3) and p["drift_samples"].shape == (8, 16, 14) and (p["drift_q"] >= 0).all()
    if p["mode_prob"] is not None:
        assert p["mode_prob"].shape == (8, 8, 5) and torch.allclose(p["mode_prob"].sum(-1), torch.ones(8, 8), atol=1e-4)
    m.save(tmp_path / "m.pt")
    m2 = type(m).load(tmp_path / "m.pt")
    p2 = m2.predict(v, n_samples=16)
    assert torch.allclose(p["y_q"], p2["y_q"], atol=1e-4)


def test_ensemble_api(feats):
    from capxfer.learn.ensemble import Ensemble
    m = Ensemble(dict(n_members=2, steps=2, log_every=1, val_every=2, val_batches=1, d=32, layers=1, device=DEV))
    m.fit(D.batch_iter(feats, 8, 4), None)
    p = m.predict(D.apply_view(feats[0], "E0"), n_samples=10)
    assert p["y_samples"].shape[:2] == (8, 10) and p["y_q"].shape == (8, 14, 3) and p["drift_q"].shape == (8, 14, 3)


def test_analytic_bayes(feats):
    from capxfer.learn.analytic_bayes import AnalyticBayes, assembly_from_geom, hypotheses
    assert len(hypotheses(24, cap=300)) == 300
    asm = assembly_from_geom(feats[1]["wheel_geom"][0].numpy(), S.load_params())
    assert asm.n == 8 and len(asm.module_mass) == 2
    ab = AnalyticBayes(dict(cap=64, device=DEV))
    p = ab.predict(D.apply_view(feats[0], "E2"), n_samples=32)
    assert p["mode_prob"].shape == (8, 4, 5) and torch.allclose(p["mode_prob"].sum(-1), torch.ones(8, 4), atol=1e-4)
    assert p["y_samples"].shape == (8, 32, 14) and torch.isfinite(p["ppc_score"]).all()
    assert p["drift_q"].shape == (8, 14, 3) and (p["drift_q"] >= 0).all()


def test_featurize_adapter_matches_views():
    """WS-B featurize output (allowed pids only) maps onto the WS-C layout exactly as apply_view on all-pid features."""
    featurize = pytest.importorskip("capxfer.featurize")
    sensors = pytest.importorskip("capxfer.sensors")
    from capxfer import sim
    P = S.load_params()
    rng = np.random.default_rng(0)
    a = A.physical("pair", P)
    dom = sim.sample_domain(4, rng, device=DEV)
    m, g = sim.sample_faults(4, a.n, rng)
    raw = sensors.to_raw(sim.simulate(a, P, dom, m, g, device=DEV), P, dom, rng)
    full = {k: (x.cpu() if torch.is_tensor(x) else x) for k, x in D.from_featurize(featurize.featurize(raw, "E3", D.PIDS)).items()}
    part = D.ensure_batch_fields(featurize.featurize(raw, "E1", ("P1", "P5")))
    v = D.apply_view(full, "E1", ("P1", "P5"))
    assert torch.allclose(v["wheel_x"], part["wheel_x"].cpu(), atol=1e-5)
    assert torch.allclose(v["body_x"], part["body_x"].cpu(), atol=1e-5)
    assert (v["body_mask"] == part["body_mask"].cpu()).all() and (v["pid_mask"] == part["pid_mask"].cpu()).all()
