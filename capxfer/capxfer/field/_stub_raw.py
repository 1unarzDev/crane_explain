"""WS-B local stub of WS-A's `sensors.to_raw` (CONTRACTS C1). Used only until `capxfer.sensors.to_raw` exists.

Converts `sim.simulate` output to a RawTrial with simple noise. It is deliberately minimal; the real sensor model
(WS-A) replaces it via `get_to_raw()`, which prefers `capxfer.sensors.to_raw` when importable.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _sample_hold(x, t, rate_hz, latency, rng_t, noise_std):
    """x (B,T,C) on grid t (T,). Returns the value seen at t: last frame acquired at k/rate <= t - latency, plus noise
    drawn once per frame (so the held value is constant between frames)."""
    B, T, C = x.shape
    dt = float(t[1] - t[0])
    frame_t = torch.arange(0.0, float(t[-1]) + 1e-9, 1.0 / rate_hz)
    frame_idx = torch.clamp(torch.round(frame_t / dt).long(), 0, T - 1)
    frames = x[:, frame_idx]                                                    # (B,F,C)
    noise = torch.as_tensor(rng_t.normal(0, 1, size=frames.shape), dtype=x.dtype) * torch.as_tensor(noise_std, dtype=x.dtype)
    frames = frames + noise
    k = torch.floor((t - latency) * rate_hz + 1e-6).long()                       # frame index visible at t
    valid = k >= 0
    out = frames[:, torch.clamp(k, 0, len(frame_t) - 1)]
    out[:, ~valid] = frames[:, :1].expand(-1, int((~valid).sum()), -1)           # before the first frame: hold the first
    return out


def to_raw(R, params, dom=None, rng=None, asm=None, mode=None, g=None, current_wired=False, ood="none", trial_ids=None):
    """sim.simulate output -> RawTrial (dict of CPU torch tensors + meta). Noise levels from params['sensors'/'truth']."""
    rng = rng if rng is not None else np.random.default_rng(0)
    sp, tp = params["sensors"], params["truth"]
    t = R["t"].detach().cpu().float()
    B, T, n = R["wheel_angle"].shape
    cpr = float(R["meta"]["cpr"])
    sched = R["meta"]["schedule"]
    nrm = lambda *s: torch.as_tensor(rng.normal(0, 1, size=s), dtype=torch.float32)  # noqa: E731
    ang = R["wheel_angle"].detach().cpu().float()
    counts = torch.floor(ang * cpr / (2 * math.pi)).long()
    duty = (R["duty"].detach().cpu().float() + 0.002 * nrm(B, T, n)).clamp(-1, 1)
    tw = R["twist"].detach().cpu().float()
    bias = 0.002 * nrm(B, 1)
    gyro = tw[..., 2] + bias + float(sp["imu_gyro_noise_radps"]) * nrm(B, T)
    acc = R["accel"].detach().cpu().float() + float(sp["imu_acc_noise_mps2"]) * nrm(B, T, 2)
    if asm is None or asm.has_lidar():
        ls = float(sp["lidar_odom_noise_mps"])
        lidar = _sample_hold(tw, t, float(sp["lidar_odom_rate_hz"]), float(sp["lidar_odom_latency_s"]), rng,
                             torch.tensor([ls, ls, ls * 2]))
    else:
        lidar = torch.full((B, T, 3), float("nan"))
    if current_wired:
        m = params["motor"]
        kt = float(m["stall_torque_nm_12v"]) / float(m["stall_current_a_12v"])
        current = R["torque"].detach().cpu().float() / kt + float(sp["current_noise_a"]) * nrm(B, T, n)
    else:
        current = torch.full((B, T, n), float("nan"))
    pose = R["pose"].detach().cpu().float()
    pn, yn = float(tp["apriltag_pos_noise_m"]), float(tp["apriltag_yaw_noise_rad"])
    truth = _sample_hold(pose, t, float(tp["apriltag_rate_hz"]), float(tp["apriltag_latency_s"]), rng,
                         torch.tensor([pn, pn, yn]))
    cmd = torch.as_tensor(sched.twist(t.numpy()), dtype=torch.float32).expand(B, T, 3).clone()   # as published
    labels = {"ood": ood}
    if mode is not None:
        labels["mode"] = mode.detach().cpu().long()
    if g is not None:
        labels["g"] = g.detach().cpu().float()
    if dom is not None:
        labels["domain"] = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in dom.items()}
    meta = dict(assembly=asm, cpr=cpr, schedule=sched, source="SIMULATION",
                trial_ids=list(trial_ids) if trial_ids is not None else [f"sim{b:05d}" for b in range(B)], labels=labels)
    return dict(t=t, cmd=cmd, counts=counts, duty=duty, gyro_z=gyro, acc=acc, lidar_twist=lidar, current=current,
                truth_pose=truth, meta=meta)


def get_to_raw():
    """Prefer WS-A's real sensor model when it exists; fall back to this stub."""
    try:
        from capxfer.sensors import to_raw as real  # noqa: F401
        return real
    except Exception:
        return to_raw
