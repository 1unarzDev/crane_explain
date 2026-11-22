"""WS-A sensor model tests (CONTRACTS C1)."""
import math

import numpy as np
import torch

from capxfer import assembly, sensors, sim

P = assembly.load_params("params/module_measured.yaml")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_CACHE = {}


def _sim(name, B=16, seed=0):
    key = (name, B, seed)
    if key not in _CACHE:
        asm = assembly.physical(name, P)
        rng = np.random.default_rng(seed)
        dom = sim.sample_domain(B, rng)
        mode, g = sim.sample_faults(B, asm.n, rng)
        _CACHE[key] = (sim.simulate(asm, P, dom, mode, g, device=DEV, seed=seed), dom)
    return _CACHE[key]


def test_contract_shapes_dtypes():
    R, dom = _sim("pair")
    raw = sensors.to_raw(R, P, dom, np.random.default_rng(0))
    B, T, n = R["omega"].shape
    spec = dict(cmd=((B, T, 3), torch.float32), counts=((B, T, n), torch.int64), duty=((B, T, n), torch.float32),
                gyro_z=((B, T), torch.float32), acc=((B, T, 2), torch.float32),
                lidar_twist=((B, T, 3), torch.float32), current=((B, T, n), torch.float32),
                truth_pose=((B, T, 3), torch.float32))
    assert raw["t"].shape == (T,)
    for k, (shape, dt) in spec.items():
        assert tuple(raw[k].shape) == shape and raw[k].dtype == dt, k
        assert torch.isfinite(raw[k]).all(), k                         # pair has a lidar module (M1)
    m = raw["meta"]
    assert m["source"] == "SIMULATION" and len(m["trial_ids"]) == B and m["cpr"] == sim.nominal(P)["cpr"]
    lab = m["labels"]
    assert lab["mode"].shape == (B, n) and lab["g"].shape == (B, n) and lab["ood"] == "none"
    for k in ("mu", "latency", "sensor_gyro_noise", "sensor_lidar_latency", "sensor_truth_latency"):
        assert lab["domain"][k].shape == (B,), k


def test_no_lidar_and_no_current_nan():
    R, dom = _sim("single")                                            # M1 alone has lidar; force it off
    raw = sensors.to_raw(R, P, dom, np.random.default_rng(0), lidar=False, current_wired=False)
    assert torch.isnan(raw["lidar_twist"]).all() and torch.isnan(raw["current"]).all()
    asm = assembly.build("nolidar", [("M3", 0, 0, 0)], P)
    assert not asm.has_lidar()


def test_encoder_quantisation():
    R, dom = _sim("single")
    raw = sensors.to_raw(R, P, dom, np.random.default_rng(0))
    cpr = raw["meta"]["cpr"]
    d = raw["meta"]["labels"]["domain"]["sensor_delay_enc"].double()
    th = sensors._delay(R["wheel_angle"].double().cpu(), d, 0.01)
    err = th - raw["counts"].cpu().double() * 2 * math.pi / cpr
    assert err.min() >= -1e-9 and err.max() <= 2 * math.pi / cpr + 1e-9


def test_lidar_hold_latency_and_accuracy():
    R, dom = _sim("single")
    raw = sensors.to_raw(R, P, dom, np.random.default_rng(1))
    lt = raw["lidar_twist"].cpu()
    # sample-and-hold at 10 Hz: changes happen at most ~10 times per second
    changes = (lt[:, 1:, 0] != lt[:, :-1, 0]).sum(1).float()
    assert changes.max() <= 10 * float(raw["t"][-1]) + 2
    # delayed: best alignment with the true twist is a lag ~ latency + half a scan
    tw = R["twist"][..., 0].cpu()
    lags = list(range(0, 30))
    errs = [float((lt[:, l:, 0] - tw[:, :tw.shape[1] - l]).pow(2).mean()) for l in lags]
    best = lags[int(np.argmin(errs))] * 0.01
    lat = float(raw["meta"]["labels"]["domain"]["sensor_lidar_latency"].mean())
    assert abs(best - (lat + 0.05)) < 0.06
    assert min(errs) ** 0.5 < 0.05


def test_truth_pose_and_imu_noise_levels():
    R, dom = _sim("single")
    raw = sensors.to_raw(R, P, dom, np.random.default_rng(2))
    d = raw["meta"]["labels"]["domain"]
    gz = raw["gyro_z"].cpu() - sensors._delay(R["twist"][..., 2].cpu(), d["sensor_delay_imu"], 0.01) - \
        d["sensor_gyro_bias"][:, None]
    assert torch.allclose(gz.std(1), d["sensor_gyro_noise"], rtol=0.15)
    # truth at rest before motion starts (first 0.4 s): pose ~0 within noise
    tp = raw["truth_pose"].cpu()[:, :40]
    assert tp[..., :2].abs().max() < 0.02
    # truth lags the pose by ~latency
    p = R["pose"][..., 0].cpu()
    tpx = raw["truth_pose"][..., 0].cpu()
    errs = [float((tpx[:, l:] - p[:, :p.shape[1] - l]).pow(2).mean()) for l in range(30)]
    best = int(np.argmin(errs)) * 0.01
    assert abs(best - float(d["sensor_truth_latency"].mean()) - 1 / 60) < 0.04


def test_current_model():
    R, dom = _sim("single", B=32)
    raw = sensors.to_raw(R, P, dom, np.random.default_rng(3), current_wired=True)
    cur = raw["current"].cpu()
    mode = R["mode"].cpu()
    assert (cur >= 0).all() and cur.max() < 1.5 * P["motor"]["stall_current_a_12v"]
    passive = (mode == sim.COAST) | (mode == sim.BRAKE)
    if passive.any():
        assert cur.permute(0, 2, 1)[passive].mean() < 0.4
    assert cur.permute(0, 2, 1)[~passive].mean() > 0.2
    # noise-free: CS reads exactly 0 on BRAKE/COAST wheels (G2 CS inactive outside drive)
    clean = sensors.to_raw(R, P, dom, np.random.default_rng(3),
                           sensor_dr=dict(current_noise_a=(0.0, 0.0), current_offset_std=0.0))
    c = clean["current"].cpu().permute(0, 2, 1)
    if passive.any():
        assert c[passive].abs().max() == 0
    assert c[~passive].mean() > 0.1


def test_deterministic_given_rng():
    R, dom = _sim("single")
    a = sensors.to_raw(R, P, dom, np.random.default_rng(7))
    b = sensors.to_raw(R, P, dom, np.random.default_rng(7))
    for k in ("gyro_z", "acc", "lidar_twist", "current", "truth_pose", "counts"):
        assert torch.equal(a[k], b[k]), k


def test_stream_delay():
    x = torch.arange(10.0)[None].repeat(3, 1)
    y = sensors._delay(x, torch.tensor([0.0, 0.005, 0.01]), 0.01)
    assert torch.allclose(y[0], x[0]) and torch.allclose(y[1, 1:], x[1, 1:] - 0.5) and torch.allclose(y[2, 1:], x[2, :-1])
    R, dom = _sim("single")
    lab = sensors.to_raw(R, P, dom, np.random.default_rng(0))["meta"]["labels"]["domain"]
    for k in ("enc", "duty", "imu", "current"):
        v = lab[f"sensor_delay_{k}"]
        assert v.min() >= 0 and v.max() <= 0.010
