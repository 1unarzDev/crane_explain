"""Sensor model: simulator ground-truth streams -> RawTrial (docs/CONTRACTS.md C1).

Every noise/latency parameter is randomised per trial around the parameter file's nominal value and recorded in
labels["domain"] together with the simulator's domain parameters.

Modelling choices (stated, ESTIMATE until the field pilot):
* Encoders: cumulative counts = floor(theta / 2 pi * cpr) (quadrature edges at the output shaft), int64.
* Duty: the PI output before fault scaling, passed through (it is what the firmware publishes).
* IMU at 100 Hz: gyro_z = wz + bias + white noise; acc = body accel + bias + white noise.
* Lidar odometry (rf2o-like): scans at 10 Hz with a random phase; each estimate is the mean body twist over the
  previous scan interval plus noise, published after a latency, sample-and-hold on the 100 Hz grid. Zero before the
  first estimate. NaN when the assembly has no lidar module.
* Motor current (docs/HARDWARE_ESTIMATES.md): G2 CS is unidirectional and active only in the drive state (reads ~0
  in brake/coast, including the sign-magnitude PWM off-time). Driven wheels: I = |I_free sign(omega) + tau_em / Kt|
  (tau_em = electromagnetic torque at the wheel, Kt = tau_stall / I_stall). Per trial the ADC read is either
  PWM-synchronised (reads I) or unsynchronised/averaged (reads |duty_applied| * I), recorded as
  sensor_current_synced. Then gain error, a residual zero-offset error and white noise (~0.25 A at 20 mV/A on the
  Pico ADC), clipped at 0. BRAKE/COAST wheels: offset + noise only. NaN if not wired.
* AprilTag truth pose (evaluator-only): frames at 30 Hz with a random phase, pose interpolated at the frame time,
  plus Gaussian position/yaw noise, available after a latency; sample-and-hold on the 100 Hz grid.
* Receive-time placement (FieldLog puts streams on the 100 Hz grid by receive time): encoder, duty, IMU and current
  streams each get a per-trial delay U(0, 10 ms) (fractional, linear interpolation), recorded as sensor_delay_*.
* cmd: the published command = the probe schedule twist (WITHOUT the actuation latency the simulator applies).
"""
from __future__ import annotations

import math

import numpy as np
import torch

I_FREE_A = 0.25        # ESTIMATE Pololu 37D 12 V free-run current (not yet in params/module_measured.yaml)

# Per-trial sensor randomisation. "*_mult" multiply the nominal noise from params["sensors"]/params["truth"].
SENSOR_DR = dict(
    gyro_noise_mult=(0.5, 2.0), gyro_bias_std=0.01, acc_noise_mult=(0.5, 2.0), acc_bias_std=0.05,
    lidar_noise_mult=(0.5, 2.0), lidar_latency=(0.06, 0.16), lidar_yaw_noise_radps=0.02,
    current_noise_a=(0.15, 0.35), current_gain=(0.85, 1.15), current_offset_std=0.05, p_current_synced=0.5,
    truth_latency=(0.12, 0.18), truth_noise_mult=(0.5, 2.0), stream_delay=(0.0, 0.010),
)


def sample_sensor_domain(B, rng, params, dr=None):
    """Per-trial sensor parameters (numpy float32 arrays of shape (B,))."""
    dr = {**SENSOR_DR, **(dr or {})}
    s, tr = params["sensors"], params["truth"]
    u = lambda k: rng.uniform(*dr[k], size=B).astype(np.float32)
    nrm = lambda sd: (rng.normal(0, sd, size=B)).astype(np.float32)
    d = dict(
        gyro_noise=u("gyro_noise_mult") * float(s["imu_gyro_noise_radps"]),
        gyro_bias=nrm(dr["gyro_bias_std"]),
        acc_noise=u("acc_noise_mult") * float(s["imu_acc_noise_mps2"]),
        acc_bias_x=nrm(dr["acc_bias_std"]), acc_bias_y=nrm(dr["acc_bias_std"]),
        lidar_noise=u("lidar_noise_mult") * float(s["lidar_odom_noise_mps"]),
        lidar_yaw_noise=u("lidar_noise_mult") * float(dr["lidar_yaw_noise_radps"]),
        lidar_latency=u("lidar_latency"),
        lidar_phase=rng.uniform(0, 1.0 / float(s["lidar_odom_rate_hz"]), size=B).astype(np.float32),
        current_noise=u("current_noise_a"),                       # absolute (WS-H: sigma ~0.2-0.3 A per sample)
        current_gain=u("current_gain"),
        current_offset=nrm(dr["current_offset_std"]),
        current_synced=(rng.uniform(size=B) < dr["p_current_synced"]).astype(np.float32),
        truth_latency=u("truth_latency"),
        truth_pos_noise=u("truth_noise_mult") * float(tr["apriltag_pos_noise_m"]),
        truth_yaw_noise=u("truth_noise_mult") * float(tr["apriltag_yaw_noise_rad"]),
        truth_phase=rng.uniform(0, 1.0 / float(tr["apriltag_rate_hz"]), size=B).astype(np.float32),
    )
    # receive-time placement on the 100 Hz grid (CONTRACTS amendment): per-stream delay U(0, 10 ms)
    for k in ("enc", "duty", "imu", "current"):
        d[f"delay_{k}"] = rng.uniform(*dr["stream_delay"], size=B).astype(np.float32)
    return d


def _hold_index(t, phase, period, latency):
    """For each grid time t (T,) and trial (B,): index k of the latest sample with k*period + phase + latency <= t
    (k = -1 if none yet). Returns (B,T) int64 and the sample times (B,T)."""
    k = torch.floor((t[None, :] - phase[:, None] - latency[:, None]) / period).long()
    return k, k.float() * period + phase[:, None]


def _interp(x, t_grid, ts):
    """Linear interpolation of x (B,T,C) on uniform grid t_grid (T,) at per-trial times ts (B,T')."""
    dt = float(t_grid[1] - t_grid[0])
    T = x.shape[1]
    f = ((ts - t_grid[0]) / dt).clamp(0, T - 1)
    i0 = f.floor().long().clamp(max=T - 2)
    w = (f - i0.float())[..., None]
    g0 = torch.gather(x, 1, i0[..., None].expand(-1, -1, x.shape[2]))
    g1 = torch.gather(x, 1, (i0 + 1)[..., None].expand(-1, -1, x.shape[2]))
    return g0 * (1 - w) + g1 * w


def _delay(x, delay, dt):
    """Fractional delay of a (B,T,...) stream on a uniform grid by per-trial `delay` seconds (linear interpolation,
    the first sample is held)."""
    fr = (delay / dt).to(x.dtype)
    idx = torch.floor(fr).long()
    w = fr - idx.to(x.dtype)
    T = x.shape[1]
    ar = torch.arange(T, device=x.device)
    i0 = (ar[None] - idx[:, None]).clamp(0, T - 1)
    i1 = (i0 - 1).clamp(min=0)
    shp = (x.shape[0], T) + (1,) * (x.dim() - 2)
    g0 = torch.gather(x, 1, i0.view(shp).expand_as(x))
    g1 = torch.gather(x, 1, i1.view(shp).expand_as(x))
    w = w.view((-1,) + (1,) * (x.dim() - 1))
    return g0 * (1 - w) + g1 * w


def _gen(rng, device):
    return torch.Generator(device=device).manual_seed(int(rng.integers(2 ** 62)))


@torch.no_grad()
def to_raw(R, params, dom, rng, sensor_dr=None, current_wired=True, lidar=None, trial_ids=None, ood=None,
           source="SIMULATION"):
    """Simulator output R (sim.simulate) -> RawTrial dict (CONTRACTS C1).

    dom: the simulator domain dict used for R (recorded in labels["domain"]).
    rng: numpy Generator (sensor randomisation and noise; deterministic given its state).
    current_wired: bool, or a float probability per call that the current rung exists.
    lidar: override the assembly's lidar presence (default: R["meta"]["assembly"].has_lidar()).
    """
    meta = R["meta"]
    asm = meta["assembly"]
    sched = meta["schedule"]
    dev = R["twist"].device
    B, T, n = R["omega"].shape
    t = R["t"].float()
    f32 = dict(dtype=torch.float32, device=dev)
    gen = _gen(rng, dev)
    randn = lambda *s: torch.randn(*s, generator=gen, **f32)
    sd = sample_sensor_domain(B, rng, params, sensor_dr)
    S = {k: torch.as_tensor(v, device=dev) for k, v in sd.items()}
    cpr = float(meta["cpr"])

    raw = {"t": t.clone()}
    cmd = torch.tensor(sched.twist(t.cpu().numpy()), **f32)
    raw["cmd"] = cmd[None].expand(B, T, 3).contiguous()
    gdt = float(t[1] - t[0])
    raw["counts"] = torch.floor(_delay(R["wheel_angle"].double(), S["delay_enc"].double(), gdt)
                                / (2 * math.pi) * cpr).long()
    raw["duty"] = _delay(R["duty"].float(), S["delay_duty"], gdt)

    tw = R["twist"].float()
    raw["gyro_z"] = _delay(tw[..., 2], S["delay_imu"], gdt) + S["gyro_bias"][:, None] + \
        S["gyro_noise"][:, None] * randn(B, T)
    acc_bias = torch.stack([S["acc_bias_x"], S["acc_bias_y"]], -1)[:, None]
    raw["acc"] = _delay(R["accel"].float(), S["delay_imu"], gdt) + acc_bias + S["acc_noise"][:, None, None] * randn(B, T, 2)

    # lidar odometry: mean twist over each scan interval [k-1, k] periods, held from k*P + phase + latency
    has_lidar = asm.has_lidar() if lidar is None else bool(lidar)
    if has_lidar:
        P = 1.0 / float(params["sensors"]["lidar_odom_rate_hz"])
        k, ts = _hold_index(t, S["lidar_phase"], P, S["lidar_latency"])
        pose = R["pose"].float()
        p1 = _interp(pose, t, ts)
        p0 = _interp(pose, t, ts - P)
        dth = torch.atan2(torch.sin(p1[..., 2] - p0[..., 2]), torch.cos(p1[..., 2] - p0[..., 2]))
        thm = p0[..., 2] + 0.5 * dth
        dx, dy = p1[..., 0] - p0[..., 0], p1[..., 1] - p0[..., 1]
        vx = (torch.cos(thm) * dx + torch.sin(thm) * dy) / P
        vy = (-torch.sin(thm) * dx + torch.cos(thm) * dy) / P
        est = torch.stack([vx, vy, dth / P], -1)
        # noise is per scan (constant while held): draw per scan index, gather by k
        kmax = int(T * 0.01 / P) + 3
        nz = randn(B, kmax, 3) * torch.stack([S["lidar_noise"], S["lidar_noise"], S["lidar_yaw_noise"]], -1)[:, None]
        nzk = torch.gather(nz, 1, k.clamp(0, kmax - 1)[..., None].expand(-1, -1, 3))
        valid = (k >= 1)[..., None]                                    # need a previous scan
        raw["lidar_twist"] = torch.where(valid, est + nzk, torch.zeros_like(est))
    else:
        raw["lidar_twist"] = torch.full((B, T, 3), float("nan"), **f32)

    # motor current
    wired = current_wired if isinstance(current_wired, bool) else bool(rng.uniform() < float(current_wired))
    if wired:
        m = params["motor"]
        kt = float(m["stall_torque_nm_12v"]) / float(m["stall_current_a_12v"])
        mode = R["mode"].to(dev)
        driven = ((mode != 2) & (mode != 3))[:, None, :]
        om = R["omega"].float()
        i_mot = (I_FREE_A * torch.tanh(om / 0.2) + R["tau_motor"].float() / kt).abs()
        duty_applied = (R["duty"].float() * R["g"].to(dev).float()[:, None, :]).abs()
        syn = S["current_synced"][:, None, None]
        i_read = (syn * i_mot + (1 - syn) * duty_applied * i_mot) * S["current_gain"][:, None, None]
        cur = torch.where(driven, i_read, torch.zeros_like(i_read))
        cur = _delay(cur, S["delay_current"], gdt)
        raw["current"] = (cur + S["current_offset"][:, None, None]
                          + S["current_noise"][:, None, None] * randn(B, T, n)).clamp(min=0)
    else:
        raw["current"] = torch.full((B, T, n), float("nan"), **f32)

    # AprilTag truth pose (evaluator-only)
    Pt = 1.0 / float(params["truth"]["apriltag_rate_hz"])
    k, ts = _hold_index(t, S["truth_phase"], Pt, S["truth_latency"])
    ts = ts.clamp(min=0)                                               # before the first frame: pose at t=0
    pz = _interp(R["pose"].float(), t, ts)
    kmax = int(T * 0.01 / Pt) + 3
    nz = randn(B, kmax + 1, 3) * torch.stack([S["truth_pos_noise"], S["truth_pos_noise"], S["truth_yaw_noise"]],
                                             -1)[:, None]
    nzk = torch.gather(nz, 1, (k + 1).clamp(0, kmax)[..., None].expand(-1, -1, 3))
    raw["truth_pose"] = pz + nzk

    # labels and meta
    dom_rec = {}
    for kk, v in dom.items():
        dom_rec[kk] = v.detach().float().cpu() if torch.is_tensor(v) else torch.full((B,), float(v))
    for kk, v in sd.items():
        dom_rec["sensor_" + kk] = torch.as_tensor(v).float()
    dom_rec["current_wired"] = torch.full((B,), float(wired))
    dom_rec["has_lidar"] = torch.full((B,), float(has_lidar))
    labels = dict(mode=R["mode"].long().cpu(), g=R["g"].float().cpu(), ood=ood or meta.get("ood", "none"),
                  domain=dom_rec)
    raw["meta"] = dict(assembly=asm, cpr=cpr, schedule=sched, source=source,
                       trial_ids=list(trial_ids) if trial_ids is not None else [f"sim_{meta.get('seed', 0)}_{b}"
                                                                                 for b in range(B)],
                       labels=labels, n=n, char_length=meta["char_length"], dt=meta.get("dt"))
    raw["labels"] = labels                                             # convenience alias (evaluator-only)
    return raw
