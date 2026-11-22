"""WS-B featurizer tests (CPU; stub sensor model)."""
import dataclasses

import numpy as np
import pytest
import torch

from capxfer import assembly as A
from capxfer import featurize as F
from capxfer import sim as S
from capxfer.field._stub_raw import to_raw
from capxfer.probes import PIDS, TARGET_KEYS

P = A.load_params("params/module_measured.yaml")


def _run(asm, B=6, seed=0, current_wired=False):
    rng = np.random.default_rng(seed)
    dom = S.sample_domain(B, rng)
    mode, g = S.sample_faults(B, asm.n, rng)
    mode[0] = 0
    g[0] = 1
    R = S.simulate(asm, P, dom, mode, g, dt=0.005, device="cpu", seed=seed)
    raw = to_raw(R, P, dom, rng, asm=asm, mode=mode, g=g, current_wired=current_wired)
    return R, raw


@pytest.fixture(scope="module")
def pair():
    return _run(A.physical("pair", P), B=8, seed=3, current_wired=True)


@pytest.fixture(scope="module")
def nolidar():
    return _run(A.build("solo_m3", [("M3", 0, 0, 0)], P), B=3, seed=4)


def _yref(R, asm):
    sched = R["meta"]["schedule"]
    Lc = asm.char_length()
    W = np.array([1, 1, Lc ** 2])
    tw, tt = R["twist"].numpy(), R["t"].numpy()
    return np.stack([((tw[:, (tt >= ta) & (tt <= tb)].mean(1)) * c * W).sum(-1) / (c * c * W).sum()
                     for _, _, ta, tb, c in sched.windows()], 1)


def test_shapes_and_rung_masks(pair):
    R, raw = pair
    B, n = 8, 8
    f0 = F.featurize(raw, "E0")
    Lw = 2 * 2 * 40                                               # P1, P5: two legs x 2 s x 20 Hz
    assert f0["wheel_x"].shape == (B, n, Lw, 4)
    assert f0["wheel_geom"].shape == (B, n, 8)
    assert f0["body_x"].shape == (B, Lw, 9)
    assert f0["y"].shape == (B, 14) and f0["drift"].shape == (B, 14)
    assert f0["pid_mask"].tolist() == [1, 0, 0, 0, 1, 0, 0]
    assert f0["n_wheels"].tolist() == [n] * B
    assert f0["body_mask"].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert f0["wheel_mask"][..., 3].sum() == 0 and torch.all(f0["wheel_x"][..., 3] == 0)
    assert torch.all(f0["body_x"][..., 3:] == 0)
    f1 = F.featurize(raw, "E1")
    assert f1["body_mask"].tolist() == [1] * 6 + [0] * 3
    f2 = F.featurize(raw, "E2")
    assert f2["body_mask"].tolist() == [1] * 9
    f3 = F.featurize(raw, "E3")
    assert torch.all(f3["wheel_mask"][..., 3] == 1) and f3["wheel_x"][..., 3].abs().sum() > 0
    for f in (f0, f1, f2, f3):
        for k in ("wheel_x", "body_x", "wheel_geom"):
            assert torch.isfinite(f[k]).all()
    # robot-visible features do not depend on the rung for the shared channels
    assert torch.equal(f0["wheel_x"][..., :3], f3["wheel_x"][..., :3])
    # other primitive subsets: fixed length per primitive, P6 legs are 44 steps
    fa = F.featurize(raw, "E0", pids=PIDS)
    assert fa["wheel_x"].shape[2] == 2 * (5 * 40 + 2 * 44)
    assert fa["seg_pid"].unique().tolist() == list(range(7))


def test_no_lidar_and_unwired_current(nolidar):
    R, raw = nolidar
    assert torch.isnan(raw["lidar_twist"]).all()
    f = F.featurize(raw, "E3")
    assert f["body_mask"][6:].sum() == 0 and torch.all(f["body_x"][..., 6:] == 0)
    assert f["wheel_mask"][..., 3].sum() == 0
    assert torch.isfinite(f["wheel_x"]).all()


def test_ref_speed_matches_assembly_ik(pair):
    R, raw = pair
    asm = raw["meta"]["assembly"]
    xi = np.random.default_rng(0).normal(size=(5, 3))
    ref = F.wheel_ref_speed(torch.tensor(xi, dtype=torch.float64), asm, 0.05).numpy()
    np.testing.assert_allclose(ref, asm.ik(xi, 0.05), rtol=1e-9, atol=1e-12)


def test_encoder_speed_tracks_true_wheel_speed(pair):
    R, raw = pair
    f = F.featurize(raw, "E0")
    seg = [s for s in raw["meta"]["schedule"].segments if s.pid in ("P1", "P5")]
    idx, per, _, _ = F._bin_index(seg, raw["t"])
    om = R["omega"][:, idx].mean(2).permute(0, 2, 1)             # (B,n,L) true speed averaged over each 50 ms bin
    err = (f["wheel_x"][..., 1] - om).abs()
    # count quantisation 2pi/6533/0.05 = 0.019 rad/s; bin-end vs bin-mean alignment adds a little during ramps
    assert err.median() < 0.05 and err.max() < 1.0


def test_target_matches_true_twist(pair):
    R, raw = pair
    asm = raw["meta"]["assembly"]
    f = F.featurize(raw, "E0")
    e = f["y"].numpy() - _yref(R, asm)
    assert np.isfinite(e).all()
    assert np.sqrt((e ** 2).mean()) < 0.02 and np.abs(e).max() < 0.06
    healthy = f["y"][0]
    # WS-MR (2026-11-21): measured robot, confirmed firmware (feed-forward v_ref/0.65 + slow PI): a healthy assembly ends
    # the 1 s holds at ~0.8-0.9 of command, uniformly over directions (capability is relative to this baseline, D17)
    assert healthy.min() > 0.75 and healthy.max() < 1.05
    assert healthy.max() - healthy.min() < 0.12
    assert (f["drift"] >= 0).all()


def test_target_wrong_latency_is_worse(pair):
    R, raw = pair
    yref = _yref(R, raw["meta"]["assembly"])
    good = np.abs(F.featurize(raw, "E0", truth_latency=0.15)["y"].numpy() - yref).mean()
    bad = np.abs(F.featurize(raw, "E0", truth_latency=0.70)["y"].numpy() - yref).mean()
    assert bad > 2 * good


def test_target_robust_to_outliers_and_dropouts(pair):
    R, raw = pair
    base = F.featurize(raw, "E0")["y"]
    r2 = dict(raw)
    tp = raw["truth_pose"].clone()
    T = tp.shape[1]
    rng = np.random.default_rng(1)
    # tag glitches: 3 isolated frames per trial jump by 5 cm, and a 150 ms dropout (NaN)
    for b in range(tp.shape[0]):
        for i in rng.choice(np.arange(50, T - 50), 3, replace=False):
            tp[b, i:i + 3, :2] += 0.05
        j = int(rng.integers(100, T - 100))
        tp[b, j:j + 15] = float("nan")
    r2["truth_pose"] = tp
    y2 = F.featurize(r2, "E0")["y"]
    d = (y2 - base).abs()
    assert torch.nanquantile(d.flatten(), 0.9) < 0.05


def test_batch_invariance_and_dict_assembly(pair):
    R, raw = pair
    f = F.featurize(raw, "E2")
    sub = {k: (v[2:5] if torch.is_tensor(v) and v.dim() > 1 else v) for k, v in raw.items() if k != "meta"}
    sub["t"] = raw["t"]
    meta = dict(raw["meta"])
    meta["assembly"] = dataclasses.asdict(raw["meta"]["assembly"])
    sub["meta"] = meta
    fs = F.featurize(sub, "E2")
    for k in ("wheel_x", "body_x", "wheel_geom"):
        torch.testing.assert_close(fs[k], f[k][2:5])
    torch.testing.assert_close(fs["y"], f["y"][2:5], equal_nan=True)


def test_geometry_frame(pair):
    R, raw = pair
    asm = raw["meta"]["assembly"]
    g = F.featurize(raw, "E0")["wheel_geom"][0].numpy()
    np.testing.assert_allclose(g[:, :2], asm.wheel_pos - asm.cg(), atol=1e-6)
    assert np.isclose(g[:, 6].max(), 1.0)
    assert set(np.round(g[:, 7] * 4).astype(int).tolist()) == {0, 1}


def test_normalizer_roundtrip(pair, tmp_path):
    R, raw = pair
    f = F.featurize(raw, "E1")
    nz = F.Normalizer().fit([f])
    p = tmp_path / "norm.json"
    nz.save(p)
    nz2 = F.Normalizer.load(p)
    g = nz2.apply(f)
    m = f["body_mask"].bool()
    bx = g["body_x"][..., m].reshape(-1, int(m.sum()))
    assert torch.allclose(bx.mean(0), torch.zeros(int(m.sum())), atol=1e-3)
    assert torch.all(g["body_x"][..., ~m] == 0)
    assert len(nz2.y_mean) == len(TARGET_KEYS)


def test_featurize_on_ws_a_sensor_model():
    sensors = pytest.importorskip("capxfer.sensors")
    asm = A.physical("L3", P)
    rng = np.random.default_rng(5)
    dom = S.sample_domain(4, rng)
    mode, g = S.sample_faults(4, asm.n, rng)
    R = S.simulate(asm, P, dom, mode, g, dt=0.005, device="cpu", seed=5)
    if not hasattr(R["meta"].get("assembly"), "wheel_pos"):
        pytest.skip("sim meta has no Assembly object yet")
    raw = sensors.to_raw(R, P, dom, rng)
    f = F.featurize(raw, "E3")
    for k in ("wheel_x", "body_x", "wheel_geom"):
        assert torch.isfinite(f[k]).all()
    e = f["y"].numpy() - _yref(R, asm)
    assert np.sqrt(np.nanmean(e ** 2)) < 0.03
